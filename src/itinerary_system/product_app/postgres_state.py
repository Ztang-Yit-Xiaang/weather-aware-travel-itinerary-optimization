"""PostgreSQL authority for the existing atomic, file-based product stores.

Apply deploy/supabase/001_itinerary_state.sql explicitly as the database owner.
Use a trusted owner Supabase UID as namespace, never a browser-supplied value.
The backend connection must own the tables (or have separately reviewed grants
and RLS policies); browser roles intentionally have no access.

Call restore() once on a fresh private ephemeral directory before creating the
runtime. Wrap runtime initialization and every stateful handler in transaction(),
and release its buffered response only after the context exits successfully.
All access to the runtime must be serialized by the caller; initially only one
worker/replica is supported. This synchronous context must not block an async
event loop waiting for another handler on that same loop.

Disk and in-memory runtime objects are disposable caches. Any failure poisons
this instance: discard the runtime and restart with a new store and fresh root.
There is deliberately no automatic retry, schema creation, disk import, or
in-process recovery after an uncertain commit.

The current durable artifacts are JSON/JSONL, including plan/evidence snapshots,
decision ledgers, trip-planning and provider-budget records. Portable ASCII paths
are bounded to 512 bytes and 16 components; lock and temporary files are omitted.
Limits are 4096 files, 8 MiB per file, 64 MiB per namespace and 16384 tree entries.
The private root must have no other filesystem writers.
"""

from __future__ import annotations

import hashlib
import os
import re
import stat
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from uuid import UUID

MAX_FILES = 4096
MAX_FILE_BYTES = 8 * 1024 * 1024
MAX_TOTAL_BYTES = 64 * 1024 * 1024
MAX_PATH_BYTES = 512
MAX_PATH_DEPTH = 16
MAX_TREE_ENTRIES = 16384
_SEGMENT = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}\Z")
_DEVICE = re.compile(r"(?:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\.|$)", re.IGNORECASE)


class PostgresStateError(RuntimeError):
    """Stable, secret-free adapter failure; code is safe for a health response."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class PostgresStateStore:
    def __init__(
        self,
        state_root: Path,
        dsn: str,
        namespace: str,
        connect_factory: Callable[[str], Any] | None = None,
    ) -> None:
        try:
            self.namespace = str(UUID(namespace))
            self.state_root = Path(os.path.abspath(state_root))
            if not isinstance(dsn, str) or not dsn.strip():
                raise ValueError
        except (ValueError, TypeError, AttributeError, OSError):
            raise PostgresStateError("postgres_state_configuration_invalid") from None
        self._dsn = dsn
        self._connect_factory = connect_factory
        self._lock_key = int.from_bytes(
            hashlib.sha256(b"itinerary-state-v1\0" + self.namespace.encode("ascii")).digest()[:8],
            "big",
            signed=True,
        )
        self._guard = threading.RLock()
        self._active = False
        self._poisoned = False
        self._revision: int | None = None
        self._hashes: dict[str, str] = {}

    @property
    def revision(self) -> int | None:
        return self._revision

    @property
    def poisoned(self) -> bool:
        return self._poisoned

    def probe(self) -> None:
        """Check DB revision and cache integrity without calling any provider.

        Return None on success; raise PostgresStateError and poison on failure.
        The revision and files are unchanged. Call under the same async request
        serialization lock used for transactions.
        """
        with self.transaction():
            pass

    def restore(self) -> None:
        """Restore exact bytes from a locked database snapshot, without merging."""
        with self._guard:
            self._require_usable()
            try:
                if self._revision is not None or self._active:
                    raise PostgresStateError("postgres_state_restore_requires_new_store")
                self._check_root()
                if self.state_root.exists() and any(self.state_root.iterdir()):
                    raise PostgresStateError("postgres_state_root_must_be_fresh")
                with self._connect() as connection:
                    revision = self._lock_namespace(connection)
                    sizes = connection.execute(
                        "SELECT count(*), COALESCE(sum(octet_length(content)), 0), "
                        "COALESCE(max(octet_length(content)), 0) "
                        "FROM itinerary_private.state_files WHERE namespace = %s",
                        (self.namespace,),
                    ).fetchone()
                    if sizes is None or sizes[0] > MAX_FILES or sizes[1] > MAX_TOTAL_BYTES or sizes[2] > MAX_FILE_BYTES:
                        raise PostgresStateError("postgres_state_limit_exceeded")
                    files: dict[str, bytes] = {}
                    for path, content, digest in connection.execute(
                        "SELECT path, content, sha256 FROM itinerary_private.state_files "
                        "WHERE namespace = %s ORDER BY path",
                        (self.namespace,),
                    ):
                        self._validate_path(path)
                        if path in files or not isinstance(content, (bytes, memoryview)):
                            raise PostgresStateError("postgres_state_snapshot_invalid")
                        raw = bytes(content)
                        if hashlib.sha256(raw).hexdigest() != digest:
                            raise PostgresStateError("postgres_state_snapshot_invalid")
                        files[path] = raw
                    hashes = self._manifest(files)
                    # Validate the complete snapshot before materializing any of it.
                    self._check_root()
                    self.state_root.mkdir(mode=0o700, parents=True, exist_ok=True)
                    self.state_root.chmod(0o700)
                    if any(self.state_root.iterdir()):
                        raise PostgresStateError("postgres_state_root_must_be_fresh")
                    for path, content in files.items():
                        destination = self.state_root / path
                        destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                        with destination.open("xb") as handle:
                            handle.write(content)
                # Never advertise a revision until the connection has committed.
                self._hashes = hashes
                self._revision = revision
            except BaseException as exc:
                self._fail(exc, "postgres_state_restore_failed_restart_required")

    @contextmanager
    def transaction(self) -> Iterator[PostgresStateStore]:
        """Commit changed files and deletions atomically after a successful body."""
        with self._guard:
            self._require_usable()
            try:
                if self._revision is None:
                    raise PostgresStateError("postgres_state_restore_required")
                if self._active:
                    raise PostgresStateError("postgres_state_nested_transaction")
                self._active = True
                with self._connect() as connection:
                    revision = self._lock_namespace(connection)
                    if revision != self._revision:
                        raise PostgresStateError("postgres_state_stale_revision_restart_required")
                    if self._manifest(self._snapshot()) != self._hashes:
                        raise PostgresStateError("postgres_state_cache_changed_restart_required")
                    yield self
                    self._require_usable()
                    files = self._snapshot()
                    hashes = self._manifest(files)
                    changed = [path for path in hashes if hashes[path] != self._hashes.get(path)]
                    deleted = self._hashes.keys() - hashes.keys()
                    for path in sorted(deleted):
                        connection.execute(
                            "DELETE FROM itinerary_private.state_files WHERE namespace = %s AND path = %s",
                            (self.namespace, path),
                        )
                    for path in changed:
                        connection.execute(
                            "INSERT INTO itinerary_private.state_files (namespace, path, content, sha256) "
                            "VALUES (%s, %s, %s, %s) ON CONFLICT (namespace, path) DO UPDATE "
                            "SET content = EXCLUDED.content, sha256 = EXCLUDED.sha256",
                            (self.namespace, path, files[path], hashes[path]),
                        )
                    if changed or deleted:
                        row = connection.execute(
                            "UPDATE itinerary_private.state_namespaces SET revision = revision + 1 "
                            "WHERE namespace = %s AND revision = %s RETURNING revision",
                            (self.namespace, revision),
                        ).fetchone()
                        if row is None or row[0] != revision + 1:
                            raise PostgresStateError("postgres_state_stale_revision_restart_required")
                        revision = row[0]
                self._hashes = hashes
                self._revision = revision
            except BaseException as exc:
                self._fail(exc, "postgres_state_transaction_failed_restart_required")
            finally:
                self._active = False

    def _require_usable(self) -> None:
        if self._poisoned:
            raise PostgresStateError("postgres_state_poisoned_restart_required")

    def _fail(self, exc: BaseException, code: str) -> None:
        self._poisoned = True
        if isinstance(exc, PostgresStateError):
            raise exc from None
        if not isinstance(exc, Exception):
            raise exc from None
        raise PostgresStateError(code) from None

    def _connect(self) -> Any:
        if self._connect_factory is not None:
            # Injection is for task-owned local tests; it bypasses the TLS policy.
            return self._connect_factory(self._dsn)
        try:
            import psycopg
            from psycopg.conninfo import conninfo_to_dict
        except ImportError:
            raise PostgresStateError("postgres_state_driver_unavailable") from None
        options = conninfo_to_dict(self._dsn)
        sslmode = options.get("sslmode", "require")
        if sslmode not in {"require", "verify-full"}:
            raise PostgresStateError("postgres_state_tls_required")
        hosts = options.get("host") or options.get("hostaddr") or ""
        if not hosts or any(
            not host or host.startswith("@") or "/" in host or "\\" in host for host in hosts.split(",")
        ):
            raise PostgresStateError("postgres_state_tcp_host_required")
        return psycopg.connect(
            self._dsn,
            sslmode=sslmode,
            gssencmode="disable",
            connect_timeout=10,
            autocommit=False,
            prepare_threshold=None,
        )

    def _lock_namespace(self, connection: Any) -> int:
        # READ COMMITTED must be set before the lock query: a waiter must read
        # the revision committed by the previous holder, not an older snapshot.
        connection.execute("SET TRANSACTION ISOLATION LEVEL READ COMMITTED")
        connection.execute("SET LOCAL lock_timeout = '10s'")
        connection.execute("SET LOCAL statement_timeout = '15s'")
        connection.execute("SET LOCAL synchronous_commit = on")
        connection.execute("SELECT pg_advisory_xact_lock(%s)", (self._lock_key,))
        connection.execute(
            "INSERT INTO itinerary_private.state_namespaces (namespace, revision) "
            "VALUES (%s, 0) ON CONFLICT (namespace) DO NOTHING",
            (self.namespace,),
        )
        row = connection.execute(
            "SELECT revision FROM itinerary_private.state_namespaces WHERE namespace = %s FOR UPDATE",
            (self.namespace,),
        ).fetchone()
        if row is None or type(row[0]) is not int or row[0] < 0:
            raise PostgresStateError("postgres_state_snapshot_invalid")
        return row[0]

    @staticmethod
    def _check_entry(metadata: os.stat_result) -> None:
        if stat.S_ISLNK(metadata.st_mode) or getattr(metadata, "st_file_attributes", 0) & getattr(
            stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400
        ):
            raise PostgresStateError("postgres_state_unsafe_path")

    def _check_root(self) -> None:
        for path in (*reversed(self.state_root.parents), self.state_root):
            if path.exists() or path.is_symlink():
                metadata = path.lstat()
                self._check_entry(metadata)
                if not stat.S_ISDIR(metadata.st_mode):
                    raise PostgresStateError("postgres_state_unsafe_path")
        if self.state_root.exists() and os.name != "nt" and self.state_root.stat().st_mode & 0o077:
            raise PostgresStateError("postgres_state_root_must_be_private")

    @staticmethod
    def _excluded(path: str) -> bool:
        parts = path.casefold().split("/")
        name = parts[-1]
        return (
            any(part in {"locks", "temporary"} for part in parts)
            or name.endswith((".lock", ".tmp"))
            or name.startswith(".tmp-")
        )

    @staticmethod
    def _validate_path(path: str, *, directory: bool = False) -> None:
        if not isinstance(path, str) or len(path.encode("utf-8")) > MAX_PATH_BYTES:
            raise PostgresStateError("postgres_state_unsafe_path")
        parts = path.split("/")
        if (
            len(parts) > MAX_PATH_DEPTH
            or any(not _SEGMENT.fullmatch(part) or part.endswith(".") or _DEVICE.match(part) for part in parts)
            or PostgresStateStore._excluded(path)
            or (not directory and not path.endswith((".json", ".jsonl")))
        ):
            raise PostgresStateError("postgres_state_unsafe_path")

    @staticmethod
    def _manifest(files: dict[str, bytes]) -> dict[str, str]:
        if len(files) > MAX_FILES or sum(map(len, files.values())) > MAX_TOTAL_BYTES:
            raise PostgresStateError("postgres_state_limit_exceeded")
        aliases: dict[str, str] = {}
        hashes: dict[str, str] = {}
        for path, content in files.items():
            PostgresStateStore._validate_path(path)
            if len(content) > MAX_FILE_BYTES:
                raise PostgresStateError("postgres_state_limit_exceeded")
            parts = path.split("/")
            for length in range(1, len(parts) + 1):
                prefix = "/".join(parts[:length])
                key = prefix.casefold()
                if key in aliases and aliases[key] != prefix:
                    raise PostgresStateError("postgres_state_unsafe_path")
                if length < len(parts) and prefix in files:
                    raise PostgresStateError("postgres_state_unsafe_path")
                aliases[key] = prefix
                if len(aliases) > MAX_TREE_ENTRIES:
                    raise PostgresStateError("postgres_state_limit_exceeded")
            hashes[path] = hashlib.sha256(content).hexdigest()
        return hashes

    def _snapshot(self) -> dict[str, bytes]:
        self._check_root()
        if not self.state_root.is_dir():
            raise PostgresStateError("postgres_state_unsafe_path")
        files: dict[str, bytes] = {}
        total = entries = 0
        pending = [self.state_root]
        while pending:
            directory = pending.pop()
            with os.scandir(directory) as children:
                for child in children:
                    entries += 1
                    if entries > MAX_TREE_ENTRIES:
                        raise PostgresStateError("postgres_state_limit_exceeded")
                    path = Path(child.path)
                    relative = path.relative_to(self.state_root).as_posix()
                    # DirEntry.stat() omits inode/link identity on Windows.
                    metadata = path.lstat()
                    self._check_entry(metadata)
                    if self._excluded(relative):
                        continue
                    is_directory = stat.S_ISDIR(metadata.st_mode)
                    self._validate_path(relative, directory=is_directory)
                    if is_directory:
                        pending.append(path)
                        continue
                    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                        raise PostgresStateError("postgres_state_unsafe_path")
                    if len(files) >= MAX_FILES or metadata.st_size > min(MAX_FILE_BYTES, MAX_TOTAL_BYTES - total):
                        raise PostgresStateError("postgres_state_limit_exceeded")
                    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0)
                    with os.fdopen(os.open(path, flags), "rb") as handle:
                        opened = os.fstat(handle.fileno())
                        self._check_entry(opened)
                        if (opened.st_dev, opened.st_ino) != (metadata.st_dev, metadata.st_ino):
                            raise PostgresStateError("postgres_state_unsafe_path")
                        content = handle.read(min(MAX_FILE_BYTES, MAX_TOTAL_BYTES - total) + 1)
                        after = os.fstat(handle.fileno())
                    if (opened.st_size, opened.st_mtime_ns, opened.st_nlink) != (after.st_size, after.st_mtime_ns, 1):
                        raise PostgresStateError("postgres_state_cache_changed_restart_required")
                    total += len(content)
                    if len(content) > MAX_FILE_BYTES or total > MAX_TOTAL_BYTES:
                        raise PostgresStateError("postgres_state_limit_exceeded")
                    files[relative] = content
        return files
