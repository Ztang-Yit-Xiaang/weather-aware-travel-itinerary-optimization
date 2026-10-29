"""State-loss regressions with isolated stores, no registered runs or provider calls."""

from __future__ import annotations

import gzip
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import Mock
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from itinerary_system.product_app import workspace as workspace_module
from itinerary_system.product_app.api import create_product_app
from itinerary_system.product_app.conversations import ConversationError, ConversationRepository
from itinerary_system.product_app.copilot import DeterministicCopilotAdapter
from itinerary_system.product_app.draft_compiler import _operation_record
from itinerary_system.product_app.models import CopilotTurnV1
from itinerary_system.product_app.service import ProductService, _draft_content_hash
from itinerary_system.product_app.trip_replanner import _hash, validate_planning_bundle
from itinerary_system.product_app.workspace import WorkspaceError, WorkspaceStore
from itinerary_system.research_artifacts import stable_content_hash


def saved_work(root: Path):
    store = WorkspaceStore(root)
    owner, token = store.create_session("run", "parent", 1, "trip")
    store.add_operation(
        owner,
        {"expected_revision": 0, "type": "keep_stop", "target": "stop", "parameters": {}},
        valid_stop_ids={"stop"},
        day_count=1,
    )
    store.set_proposal(owner, {"state": "synthetic-not-certified", "child_plan_id": "preview"}, owner.revision)
    return store, owner, token


def age_snapshot(store, owner, age):
    owner.created_at = (datetime.now(UTC) - age - timedelta(hours=1)).isoformat()
    owner.last_accessed_at = (datetime.now(UTC) - age).isoformat()
    store._snapshots.save(owner)
    return store._snapshots.snapshot_path(owner.session_id)


@pytest.mark.parametrize("action", ["get", "authenticate", "authenticate_readonly", "mutation"])
def test_expired_work_survives_restart_and_rejected_access(tmp_path, action):
    store, owner, token = saved_work(tmp_path)
    path = age_snapshot(store, owner, timedelta(hours=25))
    before = path.read_bytes()
    restarted = WorkspaceStore(tmp_path)
    assert path.exists(), "Authentication expiry must not erase saved draft/preview data"
    with pytest.raises(WorkspaceError, match="^session_expired$") as error:
        if action == "get":
            restarted.get(owner.session_id)
        elif action == "mutation":
            restarted.select(owner, {"expected_revision": owner.revision, "selected_day": 1})
        else:
            getattr(restarted, action)(owner.session_id, token)
    assert error.value.status_code == 404
    assert path.read_bytes() == before


@pytest.mark.parametrize("action", ["authenticate", "authenticate_readonly"])
def test_wrong_token_against_expired_work_is_403_and_changes_nothing(tmp_path, action):
    store, owner, _ = saved_work(tmp_path)
    path = age_snapshot(store, owner, timedelta(hours=25))
    before = path.read_bytes()
    with pytest.raises(WorkspaceError, match="^invalid_session_token$") as error:
        getattr(store, action)(owner.session_id, "wrong-token")
    assert error.value.status_code == 403
    assert path.read_bytes() == before


def test_retained_expired_work_does_not_consume_active_capacity(tmp_path, monkeypatch):
    store, owner, _ = saved_work(tmp_path)
    path = age_snapshot(store, owner, timedelta(days=29))
    before = path.read_bytes()
    monkeypatch.setattr(WorkspaceStore, "MAX_SESSIONS", 1)
    restarted = WorkspaceStore(tmp_path)
    fresh, _ = restarted.create_session("run", "parent", 1)
    assert path.exists() and path.read_bytes() == before
    assert restarted.get(fresh.session_id).session_id == fresh.session_id
    with pytest.raises(WorkspaceError, match="^session_capacity_reached$"):
        restarted.create_session("run", "parent", 1)


@pytest.mark.parametrize(
    "age,retained",
    [
        (timedelta(days=30), True),
        (timedelta(days=30, seconds=1), False),
    ],
)
def test_snapshot_retention_is_thirty_days_not_authentication_lifetime(tmp_path, monkeypatch, age, retained):
    store, owner, _ = saved_work(tmp_path)
    now = datetime.now(UTC)
    owner.created_at = (now - age - timedelta(hours=1)).isoformat()
    owner.last_accessed_at = (now - age).isoformat()
    store._snapshots.save(owner)
    path = store._snapshots.snapshot_path(owner.session_id)
    before = path.read_bytes()

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return now

    monkeypatch.setattr(workspace_module, "datetime", Clock)
    WorkspaceStore(tmp_path)
    assert path.exists() is retained
    if retained:
        assert path.read_bytes() == before


def test_cleanup_reloads_activity_from_another_store_under_shared_lock(tmp_path, monkeypatch):
    first, owner, token = saved_work(tmp_path)
    second = WorkspaceStore(tmp_path)
    clock_start = datetime.now(UTC)

    class Clock(datetime):
        value = clock_start

        @classmethod
        def now(cls, tz=None):
            return cls.value

    monkeypatch.setattr(workspace_module, "datetime", Clock)
    monkeypatch.setattr(workspace_module, "utc_now", lambda: Clock.value.isoformat())
    # B stays active while A's cached record becomes older than even retention.
    for _ in range(33):
        Clock.value += timedelta(hours=23)
        second.authenticate(owner.session_id, token)
    Clock.value += timedelta(hours=2)
    path = first._snapshots.snapshot_path(owner.session_id)
    before = path.read_bytes()
    assert Clock.value - datetime.fromisoformat(owner.last_accessed_at) > timedelta(days=30)
    first.create_session("unrelated-run", "parent", 1)
    assert path.exists(), "An idle store must not delete another store's active work"
    assert path.read_bytes() == before
    current = second.authenticate(owner.session_id, token)
    assert len(current.draft) == 1 and current.proposal is not None


def test_expired_rejection_does_not_reconcile_or_reactivate_old_preview(tmp_path):
    store, owner, token = saved_work(tmp_path)
    path = age_snapshot(store, owner, timedelta(hours=25))
    before = path.read_bytes()
    calls = []
    store._accepted_plan_resolver = lambda trip: calls.append(trip) or {"accepted_plan_id": "new-child"}
    with pytest.raises(WorkspaceError, match="^session_expired$"):
        store.authenticate(owner.session_id, token)
    assert calls == [] and path.read_bytes() == before


def test_active_auth_still_reconciles_changed_accepted_pointer(tmp_path):
    store, owner, token = saved_work(tmp_path)
    old_revision = owner.revision
    store._accepted_plan_resolver = lambda _: {"accepted_plan_id": "new-child"}
    current = store.authenticate(owner.session_id, token)
    assert current.accepted_plan_id == "new-child" and current.revision == old_revision + 1
    assert current.draft == [] and current.proposal is None


@pytest.mark.parametrize("delete_fails", [False, True])
def test_conversation_delete_and_rollback_exclude_concurrent_read(tmp_path, monkeypatch, delete_fails):
    store, owner, token = saved_work(tmp_path)
    service = ProductService.__new__(ProductService)
    service.workspace = store
    service.conversations = ConversationRepository(tmp_path)
    original = service.read_conversation(owner)
    timestamp = datetime.now(UTC).isoformat()
    service.conversations.append_turn(
        original.conversation_id,
        owner.session_id,
        CopilotTurnV1(
            turn_id="copilot_turn_" + uuid4().hex,
            client_message_id="client_message_" + uuid4().hex,
            context_revision=owner.revision,
            provider="deterministic",
            model=None,
            prompt_version="test-v1",
            prompt_sha256="a" * 64,
            state="completed",
            user_message="Retain history",
            assistant_message="Saved",
            interpretation=None,
            error_code=None,
            created_at=timestamp,
            completed_at=timestamp,
        ),
    )
    transcript = service.conversations.root / (original.conversation_id + ".json")
    original_bytes = transcript.read_bytes()
    detached, release, reader_started, reader_done = (threading.Event() for _ in range(4))
    clear = store.clear_conversation_id

    def pause_after_detach(*args):
        clear(*args)
        detached.set()
        assert release.wait(5)

    def fail_delete(*_args):
        raise ConversationError("conversation_delete_failed")

    def delete():
        try:
            return service.delete_conversation(owner, owner.revision)
        except (ConversationError, WorkspaceError) as exc:
            return exc

    def read():
        reader_started.set()
        session = store.authenticate(owner.session_id, token)
        conversation = service.read_conversation(session)
        reader_done.set()
        return conversation

    monkeypatch.setattr(store, "clear_conversation_id", pause_after_detach)
    if delete_fails:
        monkeypatch.setattr(service.conversations, "delete", fail_delete)
    with ThreadPoolExecutor(max_workers=2) as pool:
        deletion = pool.submit(delete)
        assert detached.wait(5)
        unlocked = store._lock.acquire(blocking=False)
        if unlocked:
            store._lock.release()
        reader = pool.submit(read)
        assert reader_started.wait(5)
        overlapped = reader_done.wait(0.2)
        release.set()
        result, visible = deletion.result(timeout=5), reader.result(timeout=5)
    assert not unlocked and not overlapped, "Reads must wait for deletion or rollback to finish"
    assert len(owner.draft) == 1 and owner.proposal is not None
    if delete_fails:
        assert isinstance(result, ConversationError) and result.code == "conversation_delete_failed"
        assert visible.conversation_id == original.conversation_id and len(visible.turns) == 1
        assert transcript.read_bytes() == original_bytes
    else:
        assert result is True and not transcript.exists()
        assert visible.conversation_id != original.conversation_id and visible.turns == ()
    assert len(list(service.conversations.root.glob("conversation_*.json"))) == 1


@pytest.fixture
def expired_planning_api(tmp_path, monkeypatch):
    """Replay retained route evidence in new state; intercept every refresh fetch."""
    root = Path(__file__).resolve().parents[2]
    fixture = root / "results/repairs/coordinated_acceptance_20261022/real_preview_execution.json.gz"
    with gzip.open(fixture, "rt", encoding="utf-8") as stream:
        bundle = json.load(stream)["planning_bundle"]
    official = Mock(side_effect=WorkspaceError("official_source_unavailable", 409))
    monkeypatch.setattr("itinerary_system.product_app.trip_replanner.read_official_sources", official)
    app = create_product_app(
        repository_root=root,
        state_root=tmp_path / "state",
        additional_allowed_authorities=("testserver",),
        copilot=DeterministicCopilotAdapter(),
    )
    with TestClient(app) as api:
        response = api.post("/api/sessions", json={})
        assert response.status_code == 200, response.text
        created = response.json()
        service = app.state.product_service
        owner = service.workspace.get(created["session"]["session_id"])
        service.workspace.add_operation(
            owner,
            {
                "expected_revision": owner.revision,
                "type": "replan_trip",
                "target": "trip",
                "parameters": bundle["parameters"],
            },
            valid_stop_ids=set(),
            day_count=7,
        )
        compiler = service.compiler_for_session(owner)
        identity = stable_content_hash(
            {
                "parent": compiler.parent_plan["content_hash"],
                "operation": _operation_record(owner.draft[0]),
            }
        )
        bundle["request_id"] = "trip_request_" + identity
        bundle["created_at"] = (datetime.now(UTC) - timedelta(hours=1)).isoformat()
        bundle["content_hash"] = _hash({k: v for k, v in bundle.items() if k != "content_hash"})
        # Require real integrity/route checks to pass before testing the age gate.
        validate_planning_bundle(compiler.parent_plan, bundle, check_age=False)
        with pytest.raises(WorkspaceError, match="^trip_planning_evidence_expired$"):
            validate_planning_bundle(compiler.parent_plan, bundle)
        cache = compiler.planning_state_root / (identity + ".json")
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(bundle), encoding="utf-8")
        service.workspace.set_proposal(
            owner,
            {
                "schema_version": "draft-preview-v1",
                "execution_mode": "deterministic_repair_pipeline",
                "expected_revision": owner.revision,
                "session_revision": owner.revision + 1,
                "draft_content_hash": _draft_content_hash(owner.draft),
                "runtime_execution": {"schema_version": "runtime-repair-execution-v1", "planning_bundle": bundle},
            },
            owner.revision,
        )
        yield api, service, owner, {"X-Session-Token": created["mutation_token"]}, cache, official


@pytest.mark.parametrize("cached_proposal", [False, True])
def test_explicit_trip_preview_endpoint_requests_fresh_expired_evidence(expired_planning_api, cached_proposal):
    api, service, owner, headers, cache, official = expired_planning_api
    if not cached_proposal:
        service.workspace.set_proposal(owner, {}, owner.revision)
    before, proposal, revision = cache.read_bytes(), deepcopy(owner.proposal), owner.revision
    response = api.post(
        f"/api/sessions/{owner.session_id}/preview", headers=headers, json={"expected_revision": revision}
    )
    assert response.status_code == 409, response.text
    assert response.json() == {"detail": "official_source_unavailable"}
    official.assert_called_once()
    assert cache.read_bytes() == before
    assert owner.revision == revision and owner.proposal == proposal


def test_accept_endpoint_still_rejects_expired_trip_preview_without_refresh(expired_planning_api):
    api, service, owner, headers, cache, official = expired_planning_api
    before, proposal, revision = cache.read_bytes(), deepcopy(owner.proposal), owner.revision
    pointer = service.acceptance.load_pointer(owner.trip_id)
    response = api.post(
        f"/api/sessions/{owner.session_id}/accept",
        headers=headers,
        json={
            "expected_revision": revision,
            "confirmation": "accept_evaluated_repair",
            "idempotency_key": "accept_" + uuid4().hex,
        },
    )
    assert response.status_code == 409, response.text
    assert response.json() == {"detail": "trip_planning_evidence_expired"}
    official.assert_not_called()
    assert cache.read_bytes() == before
    assert owner.revision == revision and owner.proposal == proposal
    assert service.acceptance.load_pointer(owner.trip_id) == pointer


def test_restore_and_readonly_trip_compile_do_not_refresh_expired_evidence(expired_planning_api):
    _, service, owner, _, cache, official = expired_planning_api
    before, proposal = cache.read_bytes(), deepcopy(owner.proposal)
    restored = WorkspaceStore(service.workspace.state_root).get(owner.session_id)
    assert restored.proposal == proposal
    compiler = service.compiler_for_session(restored)
    compiler.compile(restored.draft, accepted_plan_id=restored.accepted_plan_id, read_only=True)
    official.assert_not_called()
    assert cache.read_bytes() == before and restored.proposal == proposal
