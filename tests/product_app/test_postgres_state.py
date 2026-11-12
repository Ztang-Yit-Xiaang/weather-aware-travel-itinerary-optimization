"""Offline failure injection plus opt-in, strictly local PostgreSQL integration.

For the integration cases, explicitly point ITINERARY_TEST_POSTGRES_DSN at a
task-owned disposable local database named itinerary_state_test_*. Install the
optional psycopg extra in the task environment first. No server is provisioned
by these tests, and a remote database DSN is rejected before connecting.
"""

from __future__ import annotations

import hashlib
import os
import stat
import subprocess
import sys
import threading
import types
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

import pytest

from itinerary_system.plans.repository import PlanRepository, PlanRepositoryConflict
from itinerary_system.product_app import postgres_state
from itinerary_system.product_app.persistence import ConversationFileStore, LocalStateLayout
from itinerary_system.product_app.postgres_state import PostgresStateError, PostgresStateStore
from itinerary_system.research_artifacts import PlanArtifactV2

OWNER = "b947697e-5587-4bf4-a726-6b7f98a4d963"
OTHER_OWNER = "7a6ca9bd-cdb9-4e2f-87fe-e1fb1e08cc36"
SECRET = "private-password-and-raw-file-content"
DSN = f"postgresql://owner:{SECRET}@unused.invalid/database"


class Result:
    def __init__(self, rows=()):
        self.rows = list(rows)

    def fetchone(self):
        return self.rows[0] if self.rows else None

    def __iter__(self):
        return iter(self.rows)


class Database:
    """A transactional DB-API double, not evidence of PostgreSQL correctness."""

    def __init__(self):
        self.records = {}
        self.lock = threading.Lock()
        self.calls = []
        self.commit_failure = None
        self.write_failure = False
        self.rollback_failure = False
        self.rollback_count = 0

    def connect(self, _dsn):
        return Connection(self)

    def seed(self, files, *, namespace=OWNER, revision=0):
        self.records[namespace] = {
            "revision": revision,
            "files": {path: (content, hashlib.sha256(content).hexdigest()) for path, content in files.items()},
        }

    def contents(self, namespace=OWNER):
        return {path: record[0] for path, record in self.records[namespace]["files"].items()}


class Connection:
    def __init__(self, database):
        self.database = database
        self.locked = False
        self.records = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, _value, _traceback):
        try:
            if exc_type is not None:
                self.database.rollback_count += 1
                if self.database.rollback_failure:
                    raise OSError(SECRET)
            else:
                failure = self.database.commit_failure
                self.database.commit_failure = None
                if failure == "before":
                    raise OSError(SECRET)
                self.database.records = self.records
                if failure == "after":
                    raise OSError(SECRET)
        finally:
            if self.locked:
                self.database.lock.release()

    def execute(self, query, parameters=()):
        self.database.calls.append((query, parameters))
        if query.startswith("SET "):
            return Result()
        if query.startswith("SELECT pg_advisory_xact_lock"):
            assert self.database.lock.acquire(timeout=5)
            self.locked = True
            self.records = deepcopy(self.database.records)
            return Result([(None,)])
        assert self.locked, "state accessed without a transaction-scoped advisory lock"
        namespace = parameters[0]
        if query.startswith("INSERT INTO itinerary_private.state_namespaces"):
            self.records.setdefault(namespace, {"revision": 0, "files": {}})
            return Result()
        record = self.records[namespace]
        if query.startswith("SELECT revision"):
            return Result([(record["revision"],)])
        if query.startswith("SELECT count(*)"):
            sizes = [len(content) for content, _digest in record["files"].values()]
            return Result([(len(sizes), sum(sizes), max(sizes, default=0))])
        if query.startswith("SELECT path, content, sha256"):
            return Result([(path, memoryview(content), digest) for path, (content, digest) in record["files"].items()])
        if query.startswith("DELETE FROM itinerary_private.state_files"):
            del record["files"][parameters[1]]
            return Result()
        if query.startswith("INSERT INTO itinerary_private.state_files"):
            if self.database.write_failure:
                raise OSError(SECRET)
            record["files"][parameters[1]] = (parameters[2], parameters[3])
            return Result()
        if query.startswith("UPDATE itinerary_private.state_namespaces"):
            if parameters[1] != record["revision"]:
                return Result()
            record["revision"] += 1
            return Result([(record["revision"],)])
        raise AssertionError("unexpected SQL")


@pytest.fixture
def database():
    return Database()


def store_at(tmp_path, database, *, name="state", namespace=OWNER):
    return PostgresStateStore(tmp_path / name, DSN, namespace, database.connect)


def write(root, path, content):
    destination = root / path
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(content)
    return destination


def assert_poisoned(store):
    assert store.poisoned
    for operation in (store.restore, lambda: store.transaction().__enter__()):
        with pytest.raises(PostgresStateError, match="postgres_state_poisoned_restart_required"):
            operation()


def test_exact_bytes_changed_files_deletions_and_restart(tmp_path, database):
    original = {
        "sessions/session_a.json": b'\xef\xbb\xbf{ "saved": true }\r\n',
        "decisions/trip/ledger.jsonl": '{"place":"瑞安"}\r\n{"v":2}'.encode(),
        "acceptance_snapshots/decision_a.json": b'{"immutable": true}\n',
        "trip_plans/preview.json": b'{"plan":1}',
    }
    database.seed(original, revision=7)
    store = store_at(tmp_path, database)
    store.restore()
    assert store.revision == 7
    assert {path: (store.state_root / path).read_bytes() for path in original} == original
    database.calls.clear()

    with store.transaction():
        LocalStateLayout._atomic_json_write(store.state_root / "sessions/session_a.json", {"saved": False})
        (store.state_root / "trip_plans/preview.json").unlink()
        write(store.state_root, "nearby_places/provider.json", b'{"next_attempt_at":123}')
        write(store.state_root, "locks/layout.lock", b"\0")
        write(store.state_root, "temporary/probe.json", b"ignored")
        write(store.state_root, "sessions/.tmp-unused.json", b"ignored")
        write(store.state_root, "sessions/.session.other.tmp", b"ignored")
        write(store.state_root, "sessions/writer.lock", b"ignored")
    assert store.revision == 8
    upserts = [
        args[1] for query, args in database.calls if query.startswith("INSERT INTO itinerary_private.state_files")
    ]
    deletions = [args[1] for query, args in database.calls if query.startswith("DELETE FROM")]
    assert set(upserts) == {"sessions/session_a.json", "nearby_places/provider.json"}
    assert deletions == ["trip_plans/preview.json"]
    assert (
        database.contents()["acceptance_snapshots/decision_a.json"] == original["acceptance_snapshots/decision_a.json"]
    )
    assert database.contents()["decisions/trip/ledger.jsonl"] == original["decisions/trip/ledger.jsonl"]
    restored = store_at(tmp_path, database, name="restart")
    restored.restore()
    assert {
        p.relative_to(restored.state_root).as_posix(): p.read_bytes()
        for p in restored.state_root.rglob("*")
        if p.is_file()
    } == database.contents()


def test_existing_atomic_layout_conversations_and_append_only_plans(tmp_path, database):
    store = store_at(tmp_path, database)
    store.restore()
    plan = PlanArtifactV2("plan_a", "run_a", "request_a", "catalog_a", "context_a", ())
    with store.transaction():
        assert LocalStateLayout(store.state_root).initialize().ready
        repository = PlanRepository(store.state_root / "plan_repository")
        repository.save(plan)
        conversations = ConversationFileStore(store.state_root)
        conversations.write(conversations.root / "conversation_a.json", {"turns": []}, validate=lambda value: value)
    clone = store_at(tmp_path, database, name="restarted")
    clone.restore()
    repository = PlanRepository(clone.state_root / "plan_repository")
    assert repository.load(plan.plan_id).content_hash == plan.content_hash
    with clone.transaction():
        repository.save(plan)
        with pytest.raises(PlanRepositoryConflict):
            repository.save(replace(plan, context_snapshot_id="different"))
    assert clone.revision == 1
    assert not any("locks/" in path or ".tmp" in path for path in database.contents())


def test_unchanged_and_identical_replacement_do_not_bump_revision(tmp_path, database):
    database.seed({"state.json": b"{}"}, revision=12)
    store = store_at(tmp_path, database)
    store.restore()
    database.calls.clear()
    with store.transaction():
        replacement = write(store.state_root, ".tmp-replace.json", b"{}")
        os.replace(replacement, store.state_root / "state.json")
    assert store.revision == 12
    assert not any(
        query.startswith(("UPDATE ", "DELETE ", "INSERT INTO itinerary_private.state_files"))
        for query, _ in database.calls
    )


def test_probe_is_provider_free_preserves_bytes_and_rejects_stale_state(tmp_path, database):
    database.seed({"state.json": b"exact\r\n"}, revision=5)
    first = store_at(tmp_path, database)
    stale = store_at(tmp_path, database, name="stale")
    first.restore()
    stale.restore()
    before = deepcopy(database.records)
    assert first.probe() is None
    assert first.revision == 5
    assert database.records == before
    assert (first.state_root / "state.json").read_bytes() == b"exact\r\n"
    with first.transaction():
        write(first.state_root, "state.json", b"new")
    with pytest.raises(PostgresStateError, match="postgres_state_stale_revision_restart_required"):
        stale.probe()
    assert_poisoned(stale)


def test_stale_writer_is_rejected_before_handler_and_cannot_erase_new_files(tmp_path, database):
    first = store_at(tmp_path, database, name="first")
    stale = store_at(tmp_path, database, name="stale")
    first.restore()
    stale.restore()
    with first.transaction():
        write(first.state_root, "sessions/new.json", b"committed")
    entered = False
    with pytest.raises(PostgresStateError, match="postgres_state_stale_revision_restart_required"):
        with stale.transaction():
            entered = True
    assert not entered
    assert database.contents() == {"sessions/new.json": b"committed"}
    assert_poisoned(stale)


def test_local_threads_serialize_the_entire_handler(tmp_path, database):
    store = store_at(tmp_path, database)
    store.restore()
    entered = threading.Event()
    release = threading.Event()
    second_entered = threading.Event()
    failures = []

    def worker(first):
        try:
            with store.transaction():
                if first:
                    write(store.state_root, "state.json", b"one")
                    entered.set()
                    assert release.wait(5)
                else:
                    second_entered.set()
                    assert (store.state_root / "state.json").read_bytes() == b"one"
                    write(store.state_root, "state.json", b"two")
        except BaseException as exc:
            failures.append(exc)

    first = threading.Thread(target=worker, args=(True,))
    second = threading.Thread(target=worker, args=(False,))
    first.start()
    try:
        assert entered.wait(5)
        second.start()
        assert not second_entered.wait(0.05)
    finally:
        release.set()
        first.join(5)
        if second.ident is not None:
            second.join(5)
    assert not first.is_alive() and not second.is_alive()
    assert not failures
    assert store.revision == 2
    assert database.contents() == {"state.json": b"two"}


@pytest.mark.parametrize("outside_change", ["overwrite", "delete", "add"])
def test_out_of_transaction_cache_mutation_fails_closed(tmp_path, database, outside_change):
    database.seed({"state.json": b"authoritative"}, revision=2)
    store = store_at(tmp_path, database)
    store.restore()
    if outside_change == "delete":
        (store.state_root / "state.json").unlink()
    else:
        write(store.state_root, "state.json" if outside_change == "overwrite" else "new.json", b"uncommitted")
    with pytest.raises(PostgresStateError, match="postgres_state_cache_changed_restart_required"):
        with store.transaction():
            pytest.fail("handler must not run")
    assert database.contents() == {"state.json": b"authoritative"}
    assert_poisoned(store)


@pytest.mark.parametrize("failure", ["handler", "write", "commit_before", "commit_after", "rollback"])
def test_failure_poisoning_sanitization_and_uncertain_commit(tmp_path, database, failure):
    database.seed({"state.json": b"before"}, revision=3)
    store = store_at(tmp_path, database)
    store.restore()
    if failure.startswith("commit_"):
        database.commit_failure = failure.removeprefix("commit_")
    database.write_failure = failure == "write"
    database.rollback_failure = failure == "rollback"
    with pytest.raises(PostgresStateError) as caught:
        with store.transaction():
            write(store.state_root, "state.json", b"after")
            write(store.state_root, "sessions/new.json", b"new")
            if failure in {"handler", "rollback"}:
                raise ValueError(SECRET)
    assert SECRET not in str(caught.value)
    assert caught.value.__suppress_context__
    assert store.revision == 3
    if failure == "commit_after":
        assert database.contents() == {"state.json": b"after", "sessions/new.json": b"new"}
        assert database.records[OWNER]["revision"] == 4
    else:
        assert database.contents() == {"state.json": b"before"}
        assert database.records[OWNER]["revision"] == 3
    assert_poisoned(store)


def test_delete_and_insert_roll_back_as_a_unit(tmp_path, database):
    database.seed({"old.json": b"keep"})
    store = store_at(tmp_path, database)
    store.restore()
    database.write_failure = True
    with pytest.raises(PostgresStateError):
        with store.transaction():
            (store.state_root / "old.json").unlink()
            write(store.state_root, "new.json", b"replacement")
    assert database.contents() == {"old.json": b"keep"}
    assert database.rollback_count == 1


def test_interrupt_still_poisons_and_rolls_back(tmp_path, database):
    store = store_at(tmp_path, database)
    store.restore()
    with pytest.raises(KeyboardInterrupt):
        with store.transaction():
            write(store.state_root, "state.json", b"uncommitted")
            raise KeyboardInterrupt
    assert database.contents() == {}
    assert_poisoned(store)


def test_caught_nested_transaction_failure_cannot_commit_outer_body(tmp_path, database):
    store = store_at(tmp_path, database)
    store.restore()
    with pytest.raises(PostgresStateError, match="postgres_state_poisoned_restart_required"):
        with store.transaction():
            write(store.state_root, "state.json", b"never commit")
            with pytest.raises(PostgresStateError, match="postgres_state_nested_transaction"):
                with store.transaction():
                    pytest.fail("nested body must not run")
    assert database.contents() == {}
    assert_poisoned(store)


def test_restore_is_required_and_one_shot(tmp_path, database):
    store = store_at(tmp_path, database)
    with pytest.raises(PostgresStateError, match="postgres_state_restore_required"):
        with store.transaction():
            pytest.fail("unrestored handler must not run")
    assert_poisoned(store)
    restored = store_at(tmp_path, database, name="restored")
    restored.restore()
    with pytest.raises(PostgresStateError, match="postgres_state_restore_requires_new_store"):
        restored.restore()
    assert_poisoned(restored)


@pytest.mark.parametrize("failure", ["connect", "commit_before", "commit_after"])
def test_restore_connection_and_commit_failures_never_mark_store_ready(tmp_path, database, failure):
    database.seed({"state.json": b"authoritative"}, revision=9)
    store = store_at(tmp_path, database)
    if failure == "connect":

        def unavailable(_dsn):
            raise OSError(SECRET)

        store = PostgresStateStore(store.state_root, DSN, OWNER, unavailable)
    else:
        database.commit_failure = failure.removeprefix("commit_")
    with pytest.raises(PostgresStateError, match="postgres_state_restore_failed_restart_required") as caught:
        store.restore()
    assert SECRET not in str(caught.value)
    assert store.revision is None
    assert_poisoned(store)


def test_existing_root_is_never_merged_or_deleted(tmp_path, database):
    database.seed({"state.json": b"remote"})
    root = tmp_path / "private"
    root.mkdir(mode=0o700)
    existing = write(root, "state.json", b"local")
    store = PostgresStateStore(root, DSN, OWNER, database.connect)
    with pytest.raises(PostgresStateError, match="postgres_state_root_must_be_fresh"):
        store.restore()
    assert existing.read_bytes() == b"local"
    assert not database.calls


def test_precreated_private_empty_root_supports_hosted_startup(tmp_path, database):
    root = tmp_path / "temporary-directory"
    root.mkdir(mode=0o700)
    store = PostgresStateStore(root, DSN, OWNER, database.connect)
    store.restore()
    assert store.revision == 0
    with store.transaction():
        assert LocalStateLayout(root).initialize().ready
    store.probe()
    assert store.revision == 1
    assert not store.poisoned


@pytest.mark.parametrize("namespace", ["", "../owner", "not-a-uuid", None])
def test_owner_namespace_requires_supabase_uid(tmp_path, namespace):
    with pytest.raises(PostgresStateError, match="postgres_state_configuration_invalid"):
        PostgresStateStore(tmp_path / "state", DSN, namespace)


def test_namespace_isolation_and_canonical_lock_identity(tmp_path, database):
    first = store_at(tmp_path, database, namespace=OWNER.upper())
    other = store_at(tmp_path, database, name="other", namespace=OTHER_OWNER)
    first.restore()
    other.restore()
    with first.transaction():
        write(first.state_root, "state.json", b"owner one")
    with other.transaction():
        write(other.state_root, "state.json", b"owner two")
    assert database.contents(OWNER) == {"state.json": b"owner one"}
    assert database.contents(OTHER_OWNER) == {"state.json": b"owner two"}
    assert first._lock_key == store_at(tmp_path, database)._lock_key
    assert first._lock_key != other._lock_key


@pytest.mark.parametrize(
    "path",
    [
        "../escape.json",
        "/absolute.json",
        "a/../../escape.json",
        "a//b.json",
        "a/./b.json",
        "a\\escape.json",
        "C:/escape.json",
        "state.json:stream",
        "a/CON.json",
        "a/nul.json",
        "a/trailing./b.json",
        "a/white space.json",
        "a/line\nbreak.json",
        "a/\0.json",
        "a/.hidden.json",
        "a.json/../b.json",
        "locks/state.json",
        "temporary/state.json",
        "a/.tmp-x.json",
        "secret.txt",
        "x" * 129 + ".json",
        "/".join(["deep"] * 17) + "/state.json",
        "/".join(["x" * 110] * 5) + "/state.json",
    ],
)
def test_untrusted_database_paths_rejected_before_any_file_is_restored(tmp_path, database, path):
    database.seed({"good.json": b"good", path: b"hostile"})
    store = store_at(tmp_path, database)
    with pytest.raises(PostgresStateError, match="postgres_state_unsafe_path"):
        store.restore()
    assert not store.state_root.exists()
    assert not (tmp_path / "escape.json").exists()
    assert_poisoned(store)


@pytest.mark.parametrize("paths", [("a.json", "a.json/b.json"), ("A/a.json", "a/b.json"), ("A.json", "a.json")])
def test_file_directory_and_portable_case_collisions_fail_restore(tmp_path, database, paths):
    database.seed(dict.fromkeys(paths, b"bytes"))
    store = store_at(tmp_path, database)
    with pytest.raises(PostgresStateError, match="postgres_state_unsafe_path"):
        store.restore()
    assert not store.state_root.exists()


def test_hash_corruption_fails_restore(tmp_path, database):
    database.seed({"state.json": b"changed"})
    database.records[OWNER]["files"]["state.json"] = (b"changed", "0" * 64)
    store = store_at(tmp_path, database)
    with pytest.raises(PostgresStateError, match="postgres_state_snapshot_invalid"):
        store.restore()
    assert not store.state_root.exists()


@pytest.mark.parametrize(
    "limit,files",
    [
        ("MAX_FILES", {"a.json": b"1", "b.json": b"2"}),
        ("MAX_FILE_BYTES", {"a.json": b"12"}),
        ("MAX_TOTAL_BYTES", {"a.json": b"1", "b.json": b"2"}),
    ],
)
@pytest.mark.parametrize("restore", [True, False])
def test_file_count_per_file_and_total_byte_limits(tmp_path, database, monkeypatch, limit, files, restore):
    store = store_at(tmp_path, database)
    if restore:
        database.seed(files)
    else:
        store.restore()
    monkeypatch.setattr(postgres_state, limit, 1)
    with pytest.raises(PostgresStateError, match="postgres_state_limit_exceeded"):
        if restore:
            store.restore()
        else:
            with store.transaction():
                for path, content in files.items():
                    write(store.state_root, path, content)
    if restore:
        assert not any(query.startswith("SELECT path, content") for query, _ in database.calls)
    else:
        assert database.contents() == {}
    assert_poisoned(store)


def test_directory_count_is_bounded(tmp_path, database, monkeypatch):
    store = store_at(tmp_path, database)
    store.restore()
    monkeypatch.setattr(postgres_state, "MAX_TREE_ENTRIES", 2)
    with pytest.raises(PostgresStateError, match="postgres_state_limit_exceeded"):
        with store.transaction():
            for name in ("a", "b", "c"):
                (store.state_root / name).mkdir()
    assert database.contents() == {}


def test_restore_directory_count_matches_runtime_snapshot_limit(tmp_path, database, monkeypatch):
    database.seed({"a/b/c.json": b"1"})
    store = store_at(tmp_path, database)
    monkeypatch.setattr(postgres_state, "MAX_TREE_ENTRIES", 2)
    with pytest.raises(PostgresStateError, match="postgres_state_limit_exceeded"):
        store.restore()
    assert not store.state_root.exists()


def directory_link(link, destination):
    try:
        link.symlink_to(destination, target_is_directory=True)
    except OSError:
        if os.name != "nt":
            raise
        # Windows junctions need no developer-mode symlink privilege.
        subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(destination)], check=True, capture_output=True)


@pytest.mark.parametrize("location", ["root", "ancestor", "durable", "excluded"])
def test_directory_symlinks_and_windows_junctions_cannot_escape(tmp_path, database, location):
    outside = tmp_path / "outside"
    outside.mkdir(mode=0o700)
    protected = write(outside, "protected.json", b"unchanged")
    root = tmp_path / "state"
    if location == "root":
        directory_link(root, outside)
    elif location == "ancestor":
        directory_link(tmp_path / "ancestor", outside)
        root = tmp_path / "ancestor" / "fresh"
    store = PostgresStateStore(root, DSN, OWNER, database.connect)
    if location in {"root", "ancestor"}:
        with pytest.raises(PostgresStateError, match="postgres_state_unsafe_path"):
            store.restore()
    else:
        store.restore()
        with pytest.raises(PostgresStateError, match="postgres_state_unsafe_path"):
            with store.transaction():
                directory_link(root / ("sessions" if location == "durable" else "locks"), outside)
        assert database.contents() == {}
    assert protected.read_bytes() == b"unchanged"
    assert_poisoned(store)


@pytest.mark.parametrize("hardlink", [False, True])
def test_file_links_cannot_import_external_bytes(tmp_path, database, hardlink):
    outside = write(tmp_path, "outside.json", b"external secret")
    store = store_at(tmp_path, database)
    store.restore()
    with pytest.raises(PostgresStateError, match="postgres_state_unsafe_path"):
        with store.transaction():
            if hardlink:
                os.link(outside, store.state_root / "state.json")
            else:
                try:
                    (store.state_root / "state.json").symlink_to(outside)
                except OSError:
                    pytest.skip("file symlink creation is unavailable on this host")
    assert outside.read_bytes() == b"external secret"
    assert database.contents() == {}


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits; Windows production is not supported")
def test_root_permissions_are_private_and_rechecked(tmp_path, database):
    store = store_at(tmp_path, database)
    store.restore()
    assert stat.S_IMODE(store.state_root.stat().st_mode) == 0o700
    store.state_root.chmod(0o755)
    with pytest.raises(PostgresStateError, match="postgres_state_root_must_be_private"):
        with store.transaction():
            pytest.fail("unsafe root must not be used")


@pytest.mark.parametrize(
    "sslmode,host,error",
    [
        (None, "database.invalid", None),
        ("require", "database.invalid", None),
        ("verify-full", "database.invalid", None),
        ("disable", "database.invalid", "postgres_state_tls_required"),
        ("prefer", "database.invalid", "postgres_state_tls_required"),
        ("allow", "database.invalid", "postgres_state_tls_required"),
        ("verify-ca", "database.invalid", "postgres_state_tls_required"),
        (None, "", "postgres_state_tcp_host_required"),
        (None, "/tmp", "postgres_state_tcp_host_required"),
        (None, "database.invalid,/tmp", "postgres_state_tcp_host_required"),
        (None, "@socket", "postgres_state_tcp_host_required"),
        (None, "database.invalid,@socket", "postgres_state_tcp_host_required"),
    ],
)
def test_lazy_default_driver_enforces_tls_and_disables_prepared_statements(
    tmp_path, database, monkeypatch, sslmode, host, error
):
    calls = []
    driver = types.ModuleType("psycopg")
    conninfo = types.ModuleType("psycopg.conninfo")
    options = {"host": host}
    if sslmode is not None:
        options["sslmode"] = sslmode
    conninfo.conninfo_to_dict = lambda _dsn: options

    def connect(dsn, **kwargs):
        calls.append(kwargs)
        return database.connect(dsn)

    driver.connect = connect
    monkeypatch.setitem(sys.modules, "psycopg", driver)
    monkeypatch.setitem(sys.modules, "psycopg.conninfo", conninfo)
    store = PostgresStateStore(tmp_path / "state", DSN, OWNER)
    if error is None:
        store.restore()
        assert calls == [
            {
                "sslmode": sslmode or "require",
                "gssencmode": "disable",
                "connect_timeout": 10,
                "autocommit": False,
                "prepare_threshold": None,
            }
        ]
    else:
        with pytest.raises(PostgresStateError, match=error):
            store.restore()
        assert not calls


def test_absent_optional_driver_fails_without_exposing_dsn(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "psycopg", None)
    store = PostgresStateStore(tmp_path / "state", DSN, OWNER)
    with pytest.raises(PostgresStateError, match="postgres_state_driver_unavailable") as caught:
        store.restore()
    assert SECRET not in str(caught.value)
    assert_poisoned(store)


@pytest.fixture
def local_postgres(tmp_path):
    dsn = os.environ.get("ITINERARY_TEST_POSTGRES_DSN")
    if not dsn:
        pytest.skip("no task-owned local PostgreSQL DSN; real PostgreSQL integration not executed")
    psycopg = pytest.importorskip("psycopg")
    from psycopg.conninfo import conninfo_to_dict

    options = conninfo_to_dict(dsn)
    if options.get("host") not in {"127.0.0.1", "localhost", "::1"} or not options.get("dbname", "").startswith(
        "itinerary_state_test_"
    ):
        pytest.fail("integration tests require an explicitly selected disposable local test database")
    if options.get("hostaddr") not in {None, "127.0.0.1", "::1"} or options.get("service"):
        pytest.fail("integration tests prohibit remote host overrides")
    migration = Path(__file__).resolve().parents[2] / "deploy/supabase/001_itinerary_state.sql"
    with psycopg.connect(dsn, autocommit=True) as connection:
        connection.execute(migration.read_text(encoding="utf-8"), prepare=False)
    namespace = str(uuid4())

    def connect(value):
        return psycopg.connect(value, autocommit=False, prepare_threshold=None)

    def new_store(name):
        return PostgresStateStore(tmp_path / name, dsn, namespace, connect)

    try:
        yield new_store, connect, dsn, namespace
    finally:
        with connect(dsn) as connection:
            connection.execute("DELETE FROM itinerary_private.state_namespaces WHERE namespace = %s", (namespace,))


def test_postgres_integration_restore_delta_rollback_and_schema_security(local_postgres):
    new_store, connect, dsn, namespace = local_postgres
    first = new_store("first")
    first.restore()
    exact = '{"place":"瑞安"}\r\n'.encode()
    with first.transaction():
        write(first.state_root, "decisions/trip/ledger.jsonl", exact)
        write(first.state_root, "old.json", b"old")
    with first.transaction():
        (first.state_root / "old.json").unlink()
        write(first.state_root, "new.json", b"new")
    with pytest.raises(PostgresStateError):
        with first.transaction():
            write(first.state_root, "new.json", b"rollback")
            raise ValueError(SECRET)
    restarted = new_store("restart")
    restarted.restore()
    assert restarted.revision == 2
    assert (restarted.state_root / "decisions/trip/ledger.jsonl").read_bytes() == exact
    assert (restarted.state_root / "new.json").read_bytes() == b"new"
    assert not (restarted.state_root / "old.json").exists()
    with connect(dsn) as connection:
        assert connection.execute(
            "SELECT count(*) FROM itinerary_private.state_files WHERE namespace = %s", (namespace,)
        ).fetchone() == (2,)
        assert connection.execute(
            "SELECT count(*) FROM pg_catalog.pg_class c JOIN pg_catalog.pg_namespace n ON c.relnamespace = n.oid "
            "WHERE n.nspname = 'itinerary_private' AND c.relname IN ('state_files', 'state_namespaces') AND c.relrowsecurity"
        ).fetchone() == (2,)
        for role in ("anon", "authenticated", "service_role"):
            if connection.execute("SELECT 1 FROM pg_catalog.pg_roles WHERE rolname = %s", (role,)).fetchone():
                assert connection.execute(
                    "SELECT has_schema_privilege(%s, 'itinerary_private', 'USAGE')", (role,)
                ).fetchone() == (False,)


def test_postgres_integration_lock_waiter_rejects_newly_committed_revision(local_postgres):
    new_store, _connect, _dsn, _namespace = local_postgres
    writer = new_store("writer")
    waiter = new_store("waiter")
    writer.restore()
    waiter.restore()
    started = threading.Event()
    finished = threading.Event()
    failures = []
    entered = []

    def competing_writer():
        started.set()
        try:
            with waiter.transaction():
                entered.append(True)
        except PostgresStateError as exc:
            failures.append(exc.code)
        finally:
            finished.set()

    thread = threading.Thread(target=competing_writer)
    try:
        with writer.transaction():
            write(writer.state_root, "new.json", b"durable")
            thread.start()
            assert started.wait(5)
            assert not finished.wait(0.1)
    finally:
        if thread.ident is not None:
            thread.join(15)
    assert not thread.is_alive()
    assert not entered
    assert failures == ["postgres_state_stale_revision_restart_required"]
    assert_poisoned(waiter)
