"""Versioned local-state layout and atomic product-state persistence.

W1 owns only the storage boundary: directory initialization, a process-safe
layout lock, an atomic writability probe, and detection of legacy state.  It
does not persist sessions, drafts, conversations, plans, or decisions.
"""

from __future__ import annotations

import json
import math
import os
import stat
import threading
import time
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from types import TracebackType
from typing import Any, Literal
from uuid import uuid4

from .models import DraftOperationV1, ProductSession, utc_now

STATE_SCHEMA_VERSION = "product-app-state-v1"
LEGACY_REPORT_SCHEMA_VERSION = "legacy-state-report-v1"
LAYOUT_VERSION = 1
LAYOUT_DIRECTORIES = (
    "locks",
    "plan_repository",
    "workspaces",
    "sessions",
    "conversations",
    "migrations",
    "temporary",
)
SESSION_SNAPSHOT_SCHEMA_VERSION = "product-session-snapshot-v1"
TOKEN_VERIFIER_ALGORITHM = "sha256-salted-v1"


class StateLockTimeoutError(TimeoutError):
    """Raised when the product-state layout lock cannot be acquired in time."""


@dataclass(frozen=True)
class LegacyStateReport:
    """Counts of preserved pre-W1 state; no legacy content is interpreted."""

    workspace_pointer_count: int
    decision_file_count: int
    import_status: str = "deferred_w5"

    def as_dict(self) -> dict[str, int | str]:
        return asdict(self)


@dataclass(frozen=True)
class StateReadiness:
    """Sanitized result returned to the runtime health boundary."""

    status: Literal["ready", "failed"]
    code: str
    layout_version: int = LAYOUT_VERSION
    legacy: LegacyStateReport | None = None

    @property
    def ready(self) -> bool:
        return self.status == "ready"

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "status": self.status,
            "code": self.code,
            "layout_version": self.layout_version,
        }
        if self.legacy is not None:
            payload["legacy"] = self.legacy.as_dict()
        return payload


class StateFileLock(AbstractContextManager["StateFileLock"]):
    """A bounded, cross-process exclusive lock with in-process serialization."""

    _guard = threading.Lock()
    _thread_locks: dict[str, threading.Lock] = {}

    def __init__(self, path: Path, timeout_seconds: float = 2.0) -> None:
        if timeout_seconds <= 0:
            raise ValueError("lock timeout must be positive")
        self.path = Path(path)
        self.timeout_seconds = float(timeout_seconds)
        self._handle: Any = None
        self._thread_lock: threading.Lock | None = None

    @classmethod
    def _local_lock(cls, path: Path) -> threading.Lock:
        key = os.path.normcase(str(path.resolve(strict=False)))
        with cls._guard:
            return cls._thread_locks.setdefault(key, threading.Lock())

    def __enter__(self) -> StateFileLock:
        deadline = time.monotonic() + self.timeout_seconds
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._thread_lock = self._local_lock(self.path)
        if not self._thread_lock.acquire(timeout=self.timeout_seconds):
            raise StateLockTimeoutError("state_lock_timeout")

        try:
            self._handle = self.path.open("a+b")
            self._ensure_lock_byte()
            while True:
                try:
                    self._lock_file()
                    return self
                except OSError:
                    if time.monotonic() >= deadline:
                        raise StateLockTimeoutError("state_lock_timeout") from None
                    time.sleep(min(0.025, max(0.0, deadline - time.monotonic())))
        except BaseException:
            self._close_handle()
            self._thread_lock.release()
            self._thread_lock = None
            raise

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        try:
            if self._handle is not None:
                self._unlock_file()
        finally:
            self._close_handle()
            if self._thread_lock is not None:
                self._thread_lock.release()
                self._thread_lock = None

    def _ensure_lock_byte(self) -> None:
        self._handle.seek(0, os.SEEK_END)
        if self._handle.tell() == 0:
            self._handle.write(b"\0")
            self._handle.flush()
            os.fsync(self._handle.fileno())
        self._handle.seek(0)

    def _lock_file(self) -> None:
        self._handle.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(self._handle.fileno(), msvcrt.LK_NBLCK, 1)
            return
        import fcntl

        fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    def _unlock_file(self) -> None:
        self._handle.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(self._handle.fileno(), msvcrt.LK_UNLCK, 1)
            return
        import fcntl

        fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)

    def _close_handle(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None


class LocalStateLayout:
    """Initialize and probe the non-authoritative W1 local-state boundary."""

    def __init__(self, state_root: Path, lock_timeout_seconds: float = 2.0) -> None:
        self.root = Path(state_root)
        self.lock_timeout_seconds = float(lock_timeout_seconds)
        self.state_path = self.root / "state.json"
        self.lock_path = self.root / "locks" / "layout.lock"
        self.legacy_report_path = self.root / "migrations" / "legacy-v0.json"

    def initialize(self) -> StateReadiness:
        """Create the W1 layout, record legacy counts, and verify atomic writes."""

        try:
            self.root.mkdir(parents=True, exist_ok=True)
            (self.root / "locks").mkdir(parents=True, exist_ok=True)
            with StateFileLock(self.lock_path, self.lock_timeout_seconds):
                for directory in LAYOUT_DIRECTORIES:
                    (self.root / directory).mkdir(parents=True, exist_ok=True)
                self._initialize_metadata()
                legacy = self._detect_legacy_strict()
                self._write_legacy_report(legacy)
                probe_code = self._probe_locked()
                if probe_code != "state_store_ready":
                    return StateReadiness(status="failed", code=probe_code, legacy=legacy)
                return StateReadiness(status="ready", code=probe_code, legacy=legacy)
        except StateLockTimeoutError:
            return StateReadiness(status="failed", code="state_lock_timeout")
        except _StateMetadataError:
            return StateReadiness(status="failed", code="state_metadata_invalid")
        except (OSError, ValueError, TypeError):
            return StateReadiness(status="failed", code="state_root_unavailable")

    def probe(self) -> StateReadiness:
        """Check atomic write/read/replace/delete behavior without retaining data."""

        try:
            with StateFileLock(self.lock_path, self.lock_timeout_seconds):
                legacy = self._detect_legacy_strict()
                code = self._probe_locked()
                status: Literal["ready", "failed"] = (
                    "ready" if code == "state_store_ready" else "failed"
                )
                return StateReadiness(status=status, code=code, legacy=legacy)
        except StateLockTimeoutError:
            return StateReadiness(status="failed", code="state_lock_timeout")
        except (OSError, ValueError, TypeError):
            return StateReadiness(status="failed", code="state_root_unavailable")

    def detect_legacy(self) -> LegacyStateReport:
        """Return only legacy file counts; never parse, move, or rewrite them."""

        try:
            return self._detect_legacy_strict()
        except OSError:
            return LegacyStateReport(workspace_pointer_count=0, decision_file_count=0)

    def _initialize_metadata(self) -> None:
        now = utc_now()
        if self.state_path.exists():
            try:
                payload = json.loads(self.state_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise _StateMetadataError from exc
            if not isinstance(payload, dict):
                raise _StateMetadataError
            if payload.get("schema_version") != STATE_SCHEMA_VERSION:
                raise _StateMetadataError
            if payload.get("layout_version") != LAYOUT_VERSION:
                raise _StateMetadataError
            created_at = payload.get("created_at")
            if not isinstance(created_at, str) or not created_at:
                raise _StateMetadataError
        else:
            created_at = now
        self._atomic_json_write(
            self.state_path,
            {
                "schema_version": STATE_SCHEMA_VERSION,
                "layout_version": LAYOUT_VERSION,
                "created_at": created_at,
                "updated_at": now,
            },
        )

    def _detect_legacy_strict(self) -> LegacyStateReport:
        pointer = self.root / "workspace_pointer.json"
        decision_dir = self.root / "decisions"
        pointer_count = 1 if pointer.is_file() else 0
        decision_count = 0
        if decision_dir.is_dir():
            decision_count = sum(
                1
                for path in decision_dir.iterdir()
                if path.is_file() and path.suffix == ".json"
            )
        return LegacyStateReport(
            workspace_pointer_count=pointer_count,
            decision_file_count=decision_count,
        )

    def _write_legacy_report(self, legacy: LegacyStateReport) -> None:
        self._atomic_json_write(
            self.legacy_report_path,
            {
                "schema_version": LEGACY_REPORT_SCHEMA_VERSION,
                **legacy.as_dict(),
            },
        )

    def _probe_locked(self) -> str:
        temporary_dir = self.root / "temporary"
        try:
            temporary_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            return "state_probe_failed"

        marker = uuid4().hex
        temporary_path = temporary_dir / f".state-probe-{marker}.tmp"
        target_path = temporary_dir / f".state-probe-{marker}.json"
        payload = {"schema_version": "state-probe-v1", "marker": marker}
        result = "state_store_ready"
        cleanup_failed = False
        try:
            self._write_json_file(temporary_path, payload, exclusive=True)
            if self._read_json_file(temporary_path) != payload:
                raise _StateProbeError
            os.replace(temporary_path, target_path)
            self._fsync_directory(temporary_dir)
            if self._read_json_file(target_path) != payload:
                raise _StateProbeError
            target_path.unlink()
            self._fsync_directory(temporary_dir)
        except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError):
            result = "state_probe_failed"
        finally:
            for path in (temporary_path, target_path):
                try:
                    if path.exists():
                        path.unlink()
                except OSError:
                    cleanup_failed = True
        return "state_cleanup_failed" if cleanup_failed else result

    @classmethod
    def _atomic_json_write(cls, path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".tmp-{uuid4().hex}.json")
        try:
            cls._write_json_file(temporary, payload, exclusive=True)
            os.replace(temporary, path)
            cls._fsync_directory(path.parent)
        finally:
            if temporary.exists():
                temporary.unlink()

    @staticmethod
    def _write_json_file(path: Path, payload: dict[str, Any], *, exclusive: bool) -> None:
        mode = "x" if exclusive else "w"
        with path.open(mode, encoding="utf-8", newline="\n") as handle:
            json.dump(
                payload,
                handle,
                allow_nan=False,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())

    @staticmethod
    def _read_json_file(path: Path) -> Any:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        if os.name == "nt":
            return
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


class ConversationFileStore:
    """Bounded JSON files using the product layout's shared process lock."""

    def __init__(
        self,
        state_root: Path,
        *,
        lock_timeout_seconds: float = 2.0,
        max_file_bytes: int = 1024 * 1024,
    ) -> None:
        self.root = Path(state_root) / "conversations"
        self.lock_path = Path(state_root) / "locks" / "layout.lock"
        self.lock_timeout_seconds = float(lock_timeout_seconds)
        self.max_file_bytes = int(max_file_bytes)
        if self.max_file_bytes <= 0:
            raise ValueError("max file size must be positive")

    def locked(self) -> StateFileLock:
        return StateFileLock(self.lock_path, self.lock_timeout_seconds)

    def ensure_directory(self) -> None:
        if self.root.is_symlink():
            raise OSError("conversation directory must not be a symlink")
        self.root.mkdir(parents=True, exist_ok=True)
        if self.root.is_symlink() or not self.root.is_dir():
            raise OSError("conversation directory is unavailable")

    def paths(self) -> tuple[Path, ...]:
        self.ensure_directory()
        paths: list[Path] = []
        for path in self.root.iterdir():
            metadata = path.lstat()
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
                raise OSError("invalid conversation store entry")
            paths.append(path)
        return tuple(sorted(paths, key=lambda item: item.name))

    def read(self, path: Path) -> Any:
        self._require_child(path)
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise OSError("invalid conversation file")
        if metadata.st_size > self.max_file_bytes:
            raise ValueError("conversation file is too large")
        raw = path.read_bytes()
        if len(raw) > self.max_file_bytes:
            raise ValueError("conversation file is too large")
        return json.loads(raw.decode("utf-8"))

    def write(
        self,
        path: Path,
        payload: dict[str, Any],
        *,
        validate: Callable[[Any], Any],
    ) -> None:
        self.ensure_directory()
        self._require_child(path)
        encoded = (
            json.dumps(
                payload,
                allow_nan=False,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
        if len(encoded) > self.max_file_bytes:
            raise ValueError("conversation file is too large")

        temporary = self.root / f".tmp-{uuid4().hex}.json"
        try:
            with temporary.open("xb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            validate(self.read(temporary))
            os.replace(temporary, path)
            LocalStateLayout._fsync_directory(self.root)
            validate(self.read(path))
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass

    def delete(self, path: Path, *, fsync: bool = True) -> None:
        self._require_child(path)
        path.unlink(missing_ok=True)
        if fsync:
            LocalStateLayout._fsync_directory(self.root)

    def fsync_directory(self) -> None:
        LocalStateLayout._fsync_directory(self.root)

    def _require_child(self, path: Path) -> None:
        candidate = Path(path)
        if candidate.parent != self.root or candidate.name in {"", ".", ".."}:
            raise ValueError("invalid conversation path")


class _StateMetadataError(ValueError):
    """Internal marker used to sanitize invalid-state metadata failures."""


class _StateProbeError(ValueError):
    """Internal marker used when a probe read does not match its write."""


class SessionSnapshotError(ValueError):
    """Stable, path-free failure raised for an unusable session snapshot."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class SessionSnapshotStore:
    """Strict, atomic session snapshots protected by the product-state lock."""

    _SESSION_FIELDS = {
        "session_id",
        "trip_id",
        "run_id",
        "revision",
        "accepted_plan_id",
        "selected_day",
        "selected_stop_id",
        "selected_segment_id",
        "selected_candidate_id",
        "selected_alternative_id",
        "draft",
        "proposal",
        "permission_decisions",
        "conversation_id",
        "created_at",
        "last_accessed_at",
    }
    _OPERATION_FIELDS = {
        "operation_id",
        "type",
        "target",
        "parameters",
        "source",
        "evidence_refs",
        "created_at",
    }

    def __init__(self, state_root: Path, lock_timeout_seconds: float = 2.0) -> None:
        self.root = Path(state_root)
        self.sessions_dir = self.root / "sessions"
        self.lock_path = self.root / "locks" / "layout.lock"
        self.lock_timeout_seconds = float(lock_timeout_seconds)

    def locked(self) -> StateFileLock:
        return StateFileLock(self.lock_path, self.lock_timeout_seconds)

    def ensure_directories(self) -> None:
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

    def snapshot_path(self, session_id: str) -> Path:
        if not _valid_session_id(session_id):
            raise SessionSnapshotError("invalid_session_id")
        return self.sessions_dir / f"{session_id}.json"

    def load(self, session_id: str) -> ProductSession:
        path = self.snapshot_path(session_id)
        if not path.is_file():
            raise SessionSnapshotError("unknown_session")
        try:
            payload = LocalStateLayout._read_json_file(path)
            return self._decode(payload, expected_session_id=session_id)
        except SessionSnapshotError:
            raise
        except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError):
            raise SessionSnapshotError("session_snapshot_invalid") from None

    def session_ids(self) -> tuple[str, ...]:
        if not self.sessions_dir.is_dir():
            return ()
        try:
            return tuple(sorted(path.stem for path in self.sessions_dir.glob("session_*.json")))
        except OSError:
            raise SessionSnapshotError("session_store_unavailable") from None

    def save(self, session: ProductSession) -> None:
        path = self.snapshot_path(session.session_id)
        try:
            LocalStateLayout._atomic_json_write(path, self._encode(session))
        except (OSError, TypeError, ValueError):
            raise SessionSnapshotError("session_write_failed") from None

    def delete(self, session_id: str) -> None:
        path = self.snapshot_path(session_id)
        try:
            path.unlink(missing_ok=True)
            LocalStateLayout._fsync_directory(self.sessions_dir)
        except OSError:
            raise SessionSnapshotError("session_delete_failed") from None

    @classmethod
    def _encode(cls, session: ProductSession) -> dict[str, Any]:
        public = session.public_dict()
        public.pop("schema_version", None)
        public.update(
            {
                "created_at": session.created_at,
                "last_accessed_at": session.last_accessed_at,
            }
        )
        return {
            "schema_version": SESSION_SNAPSHOT_SCHEMA_VERSION,
            "token_verifier": {
                "algorithm": TOKEN_VERIFIER_ALGORITHM,
                "salt": session.mutation_token_salt,
                "digest": session.mutation_token_verifier,
            },
            "session": public,
        }

    @classmethod
    def _decode(cls, payload: Any, *, expected_session_id: str) -> ProductSession:
        if not _json_value_is_finite(payload):
            raise SessionSnapshotError("session_snapshot_invalid")
        if not isinstance(payload, dict):
            raise SessionSnapshotError("session_snapshot_invalid")
        if payload.get("schema_version") != SESSION_SNAPSHOT_SCHEMA_VERSION:
            raise SessionSnapshotError("session_snapshot_schema_unsupported")
        if set(payload) != {"schema_version", "token_verifier", "session"}:
            raise SessionSnapshotError("session_snapshot_invalid")
        verifier = payload.get("token_verifier")
        session = payload.get("session")
        if not isinstance(verifier, dict) or set(verifier) != {"algorithm", "salt", "digest"}:
            raise SessionSnapshotError("session_snapshot_invalid")
        if verifier.get("algorithm") != TOKEN_VERIFIER_ALGORITHM:
            raise SessionSnapshotError("session_snapshot_schema_unsupported")
        salt = verifier.get("salt")
        digest = verifier.get("digest")
        if not _valid_hex(salt, 32) or not _valid_hex(digest, 64):
            raise SessionSnapshotError("session_snapshot_invalid")
        if not isinstance(session, dict) or set(session) != cls._SESSION_FIELDS:
            raise SessionSnapshotError("session_snapshot_invalid")
        if session.get("session_id") != expected_session_id:
            raise SessionSnapshotError("session_snapshot_invalid")
        _require_text_fields(session, ("trip_id", "run_id", "accepted_plan_id", "created_at", "last_accessed_at"))
        for timestamp_name in ("created_at", "last_accessed_at"):
            try:
                timestamp = datetime.fromisoformat(session[timestamp_name])
            except ValueError:
                raise SessionSnapshotError("session_snapshot_invalid") from None
            if timestamp.tzinfo is None:
                raise SessionSnapshotError("session_snapshot_invalid")
        revision = session.get("revision")
        selected_day = session.get("selected_day")
        if not isinstance(revision, int) or isinstance(revision, bool) or revision < 0:
            raise SessionSnapshotError("session_snapshot_invalid")
        if selected_day is not None and (
            not isinstance(selected_day, int) or isinstance(selected_day, bool) or selected_day < 1
        ):
            raise SessionSnapshotError("session_snapshot_invalid")
        optional_text_fields = (
            "selected_stop_id",
            "selected_segment_id",
            "selected_candidate_id",
            "selected_alternative_id",
            "conversation_id",
        )
        for field_name in optional_text_fields:
            value = session.get(field_name)
            if value is not None and (not isinstance(value, str) or not value or len(value) > 128):
                raise SessionSnapshotError("session_snapshot_invalid")
        draft = session.get("draft")
        permissions = session.get("permission_decisions")
        proposal = session.get("proposal")
        if not isinstance(draft, list) or len(draft) > 32:
            raise SessionSnapshotError("session_snapshot_invalid")
        if not isinstance(permissions, list) or len(permissions) > 32 or any(
            not isinstance(item, dict) for item in permissions
        ):
            raise SessionSnapshotError("session_snapshot_invalid")
        if proposal is not None and not isinstance(proposal, dict):
            raise SessionSnapshotError("session_snapshot_invalid")
        operations = [cls._decode_operation(item) for item in draft]
        return ProductSession(
            session_id=expected_session_id,
            mutation_token_salt=salt,
            mutation_token_verifier=digest,
            trip_id=session["trip_id"],
            run_id=session["run_id"],
            revision=revision,
            accepted_plan_id=session["accepted_plan_id"],
            selected_day=selected_day,
            selected_stop_id=session["selected_stop_id"],
            selected_segment_id=session["selected_segment_id"],
            selected_candidate_id=session["selected_candidate_id"],
            selected_alternative_id=session["selected_alternative_id"],
            draft=operations,
            proposal=proposal,
            permission_decisions=permissions,
            conversation_id=session["conversation_id"],
            created_at=session["created_at"],
            last_accessed_at=session["last_accessed_at"],
        )

    @classmethod
    def _decode_operation(cls, payload: Any) -> DraftOperationV1:
        if not isinstance(payload, dict) or set(payload) != cls._OPERATION_FIELDS:
            raise SessionSnapshotError("session_snapshot_invalid")
        _require_text_fields(payload, ("operation_id", "type", "target", "source", "created_at"))
        if not isinstance(payload.get("parameters"), dict):
            raise SessionSnapshotError("session_snapshot_invalid")
        refs = payload.get("evidence_refs")
        if not isinstance(refs, (list, tuple)) or len(refs) > 16 or any(
            not isinstance(ref, str) or not ref or len(ref) > 256 for ref in refs
        ):
            raise SessionSnapshotError("session_snapshot_invalid")
        return DraftOperationV1(
            operation_id=payload["operation_id"],
            type=payload["type"],
            target=payload["target"],
            parameters=payload["parameters"],
            source=payload["source"],
            evidence_refs=tuple(refs),
            created_at=payload["created_at"],
        )


def _valid_session_id(value: str) -> bool:
    return (
        isinstance(value, str)
        and value.startswith("session_")
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value[8:])
    )


def _valid_hex(value: Any, length: int) -> bool:
    return isinstance(value, str) and len(value) == length and all(
        character in "0123456789abcdef" for character in value
    )


def _require_text_fields(payload: dict[str, Any], names: tuple[str, ...]) -> None:
    if any(
        not isinstance(payload.get(name), str) or not payload[name] or len(payload[name]) > 256
        for name in names
    ):
        raise SessionSnapshotError("session_snapshot_invalid")


def _json_value_is_finite(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(
            isinstance(key, str) and _json_value_is_finite(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return all(_json_value_is_finite(item) for item in value)
    return value is None or isinstance(value, (str, int, bool))
