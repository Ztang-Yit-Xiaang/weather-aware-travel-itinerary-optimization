from __future__ import annotations

import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from itinerary_system.product_app.api import create_product_app
from itinerary_system.product_app.persistence import SessionSnapshotError
from itinerary_system.product_app.workspace import WorkspaceError, WorkspaceStore

ROOT = Path(__file__).absolute().parents[2]
# Frozen local fixtures and normal application imports use this isolated checkout.


def identity():
    return "mutation_" + uuid4().hex


def snapshot(store, session):
    return store._snapshots.snapshot_path(session.session_id).read_bytes()


def seed(root):
    store = WorkspaceStore(root)
    session, token = store.create_session("run", "parent", 1)
    return store, session, token


def append(store, session, target, **extra):
    payload = {"expected_revision": session.revision, "type": "keep_stop", "target": target, "parameters": {}, **extra}
    return store.add_operation(session, payload, valid_stop_ids={"a", "b"}, day_count=7)


def request_for(action, store, session):
    if action == "undo":
        append(store, session, "a")
        append(store, session, "b")
        return {"expected_revision": session.revision, "mutation_id": identity()}
    if action == "selection":
        return {
            "expected_revision": session.revision,
            "selected_day": 4,
            "selected_stop_id": "a",
            "mutation_id": identity(),
        }
    row = {"type": "keep_stop", "target": "a", "parameters": {}}
    if action == "single":
        return {"expected_revision": session.revision, "mutation_id": identity(), **row}
    return {
        "expected_revision": session.revision,
        "mutation_id": identity(),
        "operations": [row, {**row, "target": "b"}],
    }


def apply(action, store, session, body):
    if action == "selection":
        return store.select(session, body)
    if action == "undo":
        return store.undo(session, body["expected_revision"], mutation_id=body["mutation_id"])
    rows = body.get("operations") or [body]
    return store.add_operation(
        session,
        {**rows[0], "expected_revision": body["expected_revision"]},
        additional_operations=tuple(rows[1:]),
        mutation_request=body,
        valid_stop_ids={"a", "b"},
        day_count=7,
    )


@pytest.mark.parametrize("action", ["selection", "single", "batch", "undo"])
@pytest.mark.parametrize("restart", [False, True])
def test_exact_retry_preserves_later_state_and_original_result(tmp_path, action, restart):
    store, session, token = seed(tmp_path / "state")
    body = request_for(action, store, session)
    first = apply(action, store, session, body)
    receipt = json.loads(json.dumps(session.mutation_receipts[0]))
    assert (
        receipt["expected_revision"] == body["expected_revision"]
        and receipt["committed_revision"] == body["expected_revision"] + 1
    )
    assert "mutation_receipts" not in session.public_dict()
    store.select(session, {"expected_revision": session.revision, "selected_day": 5, "selected_stop_id": "b"})
    if restart:
        store = WorkspaceStore(tmp_path / "state")
        session = store.authenticate_readonly(session.session_id, token)
    before = snapshot(store, session)
    visible = session.public_dict()
    replay = apply(action, store, session, body)
    assert snapshot(store, session) == before and session.public_dict() == visible and session.selected_day == 5
    assert session.mutation_receipts == [receipt]
    if action != "selection":
        assert replay.as_dict() == first.as_dict()
    expected_action = "operations" if action in ("single", "batch") else action
    canonical = json.dumps(
        {"action": expected_action, "payload": body}, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    assert receipt["request_sha256"] == hashlib.sha256(canonical).hexdigest()


@pytest.mark.parametrize("action", ["selection", "single", "batch", "undo"])
def test_same_id_different_payload_is_refused_without_write(tmp_path, action):
    store, session, _ = seed(tmp_path / "state")
    body = request_for(action, store, session)
    apply(action, store, session, body)
    before = snapshot(store, session)
    changed = {**body, "expected_revision": session.revision}
    with pytest.raises(WorkspaceError, match="mutation_id_conflict") as failure:
        apply(action, store, session, changed)
    assert failure.value.status_code == 409 and snapshot(store, session) == before


@pytest.mark.parametrize("action", ["selection", "batch", "undo"])
def test_concurrent_duplicate_across_stores_commits_once(tmp_path, action):
    store, session, token = seed(tmp_path / "state")
    body = request_for(action, store, session)
    barrier = threading.Barrier(2)

    def submit(_):
        other = WorkspaceStore(tmp_path / "state")
        current = other.authenticate_readonly(session.session_id, token)
        barrier.wait(timeout=5)
        result = apply(action, other, current, body)
        return result.as_dict() if action != "selection" else None

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(submit, range(2)))
    final = WorkspaceStore(tmp_path / "state").authenticate_readonly(session.session_id, token)
    assert (
        final.revision == body["expected_revision"] + 1
        and len(final.mutation_receipts) == 1
        and results[0] == results[1]
    )
    assert len(final.draft) == ({"selection": 0, "batch": 2, "undo": 1}[action])


@pytest.mark.parametrize("after_commit", [False, True])
def test_atomic_save_failure_leaves_recoverable_truth(tmp_path, monkeypatch, after_commit):
    store, session, token = seed(tmp_path / "state")
    body = request_for("single", store, session)
    before = snapshot(store, session)
    save = store._snapshots.save

    def fail(current):
        if after_commit:
            save(current)
        raise SessionSnapshotError("session_write_failed")

    monkeypatch.setattr(store._snapshots, "save", fail)
    with pytest.raises(WorkspaceError, match="session_write_failed"):
        apply("single", store, session, body)
    assert session.revision == 0 and session.draft == [] and session.mutation_receipts == []
    restarted = WorkspaceStore(tmp_path / "state")
    current = restarted.authenticate_readonly(session.session_id, token)
    assert current.revision == int(after_commit) and len(current.mutation_receipts) == int(after_commit)
    if not after_commit:
        assert snapshot(restarted, current) == before
    apply("single", restarted, current, body)
    assert current.revision == 1 and len(current.draft) == 1 and len(current.mutation_receipts) == 1


@pytest.mark.parametrize(
    "bad", [None, True, 12, "", "../x", "mutation_" + "a" * 31, "mutation_" + "A" * 32, " mutation_" + "a" * 32]
)
def test_invalid_id_has_no_effect(tmp_path, bad):
    store, session, _ = seed(tmp_path / "state")
    before = snapshot(store, session)
    with pytest.raises(WorkspaceError, match="invalid_mutation_id"):
        store.select(session, {"expected_revision": 0, "selected_day": 2, "mutation_id": bad})
    assert snapshot(store, session) == before and session.revision == 0


@pytest.mark.parametrize(
    "defect", ["unknown_field", "duplicate_id", "future_revision", "wrong_result", "bad_digest", "bad_action"]
)
def test_malformed_receipt_snapshot_fails_closed(tmp_path, defect):
    store, session, token = seed(tmp_path / "state")
    body = request_for("single", store, session)
    apply("single", store, session, body)
    payload = json.loads(snapshot(store, session))
    receipt = payload["mutation_receipts"][0]
    if defect == "unknown_field":
        receipt["extra"] = True
    elif defect == "duplicate_id":
        payload["mutation_receipts"].append(dict(receipt))
    elif defect == "future_revision":
        receipt["committed_revision"] = 999
    elif defect == "wrong_result":
        receipt["result"] = {"undone": receipt["result"]["operation"]}
    elif defect == "bad_digest":
        receipt["request_sha256"] = "x" * 64
    elif defect == "bad_action":
        receipt["action"] = "accept"
    store._snapshots.snapshot_path(session.session_id).write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(WorkspaceError, match="session_snapshot_invalid"):
        WorkspaceStore(tmp_path / "state").authenticate_readonly(session.session_id, token)


def test_legacy_snapshot_and_public_session_contract_remain_exact(tmp_path):
    store, session, token = seed(tmp_path / "state")
    before = session.public_dict()
    assert "mutation_receipts" not in json.loads(snapshot(store, session))
    restarted = WorkspaceStore(tmp_path / "state").authenticate_readonly(session.session_id, token)
    assert restarted.public_dict() == before and restarted.mutation_receipts == []


def test_receipt_capacity_refuses_new_id_but_preserves_old_replay(tmp_path):
    store, session, _ = seed(tmp_path / "state")
    first = {"expected_revision": 0, "selected_day": 1, "mutation_id": identity()}
    store.select(session, first)
    for _ in range(255):
        store.select(session, {"expected_revision": session.revision, "selected_day": 1, "mutation_id": identity()})
    before = snapshot(store, session)
    with pytest.raises(WorkspaceError, match="mutation_receipt_limit_reached"):
        store.select(session, {"expected_revision": session.revision, "selected_day": 2, "mutation_id": identity()})
    store.select(session, first)
    assert snapshot(store, session) == before and session.revision == 256 and len(session.mutation_receipts) == 256


class NoProvider:
    provider_name = "disabled"

    async def interpret(self, *args, **kwargs):
        raise AssertionError("No provider call is permitted in recovery tests")


def client(state_root):
    return TestClient(
        create_product_app(
            repository_root=ROOT,
            registry_path=ROOT / "configs/product_app_registry.json",
            state_root=state_root,
            additional_allowed_authorities=("testserver",),
            copilot=NoProvider(),
        )
    )


@pytest.mark.parametrize("adapter", ["deterministic", "openai"])
def test_injected_no_provider_bootstraps_under_configured_adapter(tmp_path, monkeypatch, adapter):
    monkeypatch.setenv("PRODUCT_COPILOT_ADAPTER", adapter)
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-not-usable" if adapter == "openai" else "")
    with client(tmp_path / "state") as api:
        health = api.get("/api/health").json()
        assert health["core_ready"]
        assert health["components"]["openai"]["status"] == ("degraded" if adapter == "openai" else "disabled")
        created = api.post("/api/sessions", json={})
        assert created.status_code == 200, created.text
        assert created.json()["session"]["revision"] == 0


@pytest.mark.parametrize("action", ["selection", "single", "batch", "undo"])
def test_api_lost_reply_restart_status_and_exact_retry(tmp_path, action):
    state_root = tmp_path / "api_state"
    with client(state_root) as api:
        created = api.post("/api/sessions", json={})
        assert created.status_code == 200, created.text
        data = created.json()
        session = data["session"]
        headers = {"X-Session-Token": data["mutation_token"]}
        base = "/api/sessions/" + session["session_id"]
        targets = data["workspace"]["draft_capabilities"]["parent_targets"]
        target = targets[0]
        row = {"type": "keep_stop", "target": target, "parameters": {}}
        if action == "undo":
            seeded = api.post(
                base + "/draft/operations", headers=headers, json={"expected_revision": session["revision"], **row}
            )
            assert seeded.status_code == 200, seeded.text
            session = seeded.json()["session"]
        path = base + (
            "/selection" if action == "selection" else "/draft/undo" if action == "undo" else "/draft/operations"
        )
        body = {"expected_revision": session["revision"], "mutation_id": identity()}
        if action == "selection":
            body["selected_day"] = 2
        elif action == "single":
            body.update(row)
        elif action == "batch":
            body["operations"] = [row, {**row, "target": targets[1]}]
        committed = api.post(path, headers=headers, json=body)
        assert committed.status_code == 200, committed.text
        response = committed.json()
        assert response["mutation_receipt"]["mutation_id"] == body["mutation_id"]
    with client(state_root) as api:
        receipt_path = base + "/mutations/" + body["mutation_id"]
        assert api.get(receipt_path).status_code == 403
        assert api.get(receipt_path, headers={"X-Session-Token": "wrong"}).status_code == 403
        before = (state_root / "sessions" / f"{session['session_id']}.json").read_bytes()
        status = api.get(receipt_path, headers=headers)
        assert status.status_code == 200, status.text
        assert status.json()["state"] == "applied" and status.json()["mutation_receipt"] == response["mutation_receipt"]
        assert (state_root / "sessions" / f"{session['session_id']}.json").read_bytes() == before
        retried = api.post(path, headers=headers, json=body)
        assert retried.status_code == 200, retried.text
        assert retried.json() == response
        assert (state_root / "sessions" / f"{session['session_id']}.json").read_bytes() == before
        changed = api.post(path, headers=headers, json={**body, "expected_revision": response["session"]["revision"]})
        assert changed.status_code == 409 and changed.json()["detail"] == "mutation_id_conflict"
        assert (state_root / "sessions" / f"{session['session_id']}.json").read_bytes() == before
        unseen = api.get(base + "/mutations/" + identity(), headers=headers)
        assert unseen.status_code == 200 and unseen.json()["state"] == "unseen"
        assert (
            unseen.json()["mutation_receipt"] is None
            and unseen.json()["result"] is None
            and "session" not in unseen.json()
        )


@pytest.mark.parametrize("revision", [False, True, -1, 0.0, None, float("nan")])
def test_invalid_keyed_revision_refuses_before_persistence(tmp_path, revision):
    store, session, _ = seed(tmp_path / "state")
    before = snapshot(store, session)
    with pytest.raises(WorkspaceError, match="invalid_mutation_request") as failure:
        store.select(session, {"expected_revision": revision, "selected_day": 2, "mutation_id": identity()})
    assert failure.value.status_code == 422 and snapshot(store, session) == before and session.revision == 0


@pytest.mark.parametrize("action", ["selection", "batch", "undo"])
def test_competing_ids_at_same_revision_cannot_apply_twice(tmp_path, action):
    store, session, token = seed(tmp_path / "state")
    original = request_for(action, store, session)
    barrier = threading.Barrier(2)

    def submit(_):
        other = WorkspaceStore(tmp_path / "state")
        current = other.authenticate_readonly(session.session_id, token)
        body = {**original, "mutation_id": identity()}
        barrier.wait(timeout=5)
        try:
            apply(action, other, current, body)
            return "applied"
        except WorkspaceError as error:
            return error.code

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(submit, range(2)))
    final = WorkspaceStore(tmp_path / "state").authenticate_readonly(session.session_id, token)
    assert sorted(results) == ["applied", "stale_session_revision"]
    assert final.revision == original["expected_revision"] + 1 and len(final.mutation_receipts) == 1


def test_unseen_read_does_not_prove_an_inflight_request_cannot_commit(tmp_path):
    store, session, token = seed(tmp_path / "state")
    body = request_for("undo", store, session)
    release = threading.Event()
    entered = threading.Event()

    def delayed():
        other = WorkspaceStore(tmp_path / "state")
        current = other.authenticate_readonly(session.session_id, token)
        entered.set()
        assert release.wait(timeout=5)
        return apply("undo", other, current, body)

    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(delayed)
        assert entered.wait(timeout=5)
        read = WorkspaceStore(tmp_path / "state").authenticate_readonly(session.session_id, token)
        assert read.revision == body["expected_revision"] and read.mutation_receipts == []
        release.set()
        pending.result(timeout=5)
    read = WorkspaceStore(tmp_path / "state").authenticate_readonly(session.session_id, token)
    assert (
        read.revision == body["expected_revision"] + 1
        and read.mutation_receipts[0]["mutation_id"] == body["mutation_id"]
    )


def test_conflicting_batch_has_no_partial_draft_or_receipt(tmp_path):
    store, session, _ = seed(tmp_path / "state")
    body = request_for("batch", store, session)
    body["operations"][1]["target"] = "a"
    before = snapshot(store, session)
    with pytest.raises(WorkspaceError, match="draft_duplicate_operation"):
        apply("batch", store, session, body)
    assert snapshot(store, session) == before and session.draft == [] and session.mutation_receipts == []


def test_first_receipt_upgrades_private_snapshot_and_public_shape_stays_exact(tmp_path):
    store, session, _ = seed(tmp_path / "state")
    keys = set(session.public_dict())
    assert json.loads(snapshot(store, session))["schema_version"] == "product-session-snapshot-v1"
    apply("single", store, session, request_for("single", store, session))
    saved = json.loads(snapshot(store, session))
    assert saved["schema_version"] == "product-session-snapshot-v2"
    assert (
        set(saved) == {"schema_version", "token_verifier", "session", "mutation_receipts"}
        and set(session.public_dict()) == keys
    )


class DropCommittedReply:
    def __init__(self, app):
        self.app = app
        self.path = None
        self.dropped = []

    async def __call__(self, scope, receive, send):
        status = None

        async def recorded_send(message):
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
            if (
                scope["type"] == "http"
                and scope.get("method") == "POST"
                and scope.get("path") == self.path
                and message["type"] == "http.response.body"
                and message.get("body")
            ):
                self.path = None
                self.dropped.append({"status": status, "body_bytes": len(message.get("body", b""))})
                raise ConnectionError("simulated_committed_reply_body_lost")
            await send(message)

        await self.app(scope, receive, recorded_send)


@pytest.mark.parametrize("action", ["selection", "single", "batch", "undo"])
def test_response_body_loss_after_actual_api_commit_recovers_after_restart(tmp_path, action):
    state_root = tmp_path / "dropped_reply_state"
    application = create_product_app(
        repository_root=ROOT,
        registry_path=ROOT / "configs/product_app_registry.json",
        state_root=state_root,
        additional_allowed_authorities=("testserver",),
        copilot=NoProvider(),
    )
    wire = DropCommittedReply(application)
    with TestClient(wire) as api:
        created = api.post("/api/sessions", json={}).json()
        session = created["session"]
        headers = {"X-Session-Token": created["mutation_token"]}
        base = "/api/sessions/" + session["session_id"]
        targets = created["workspace"]["draft_capabilities"]["parent_targets"]
        row = {"type": "keep_stop", "target": targets[0], "parameters": {}}
        if action == "undo":
            seeded = api.post(
                base + "/draft/operations", headers=headers, json={"expected_revision": session["revision"], **row}
            )
            assert seeded.status_code == 200, seeded.text
            session = seeded.json()["session"]
        path = base + (
            "/selection" if action == "selection" else "/draft/undo" if action == "undo" else "/draft/operations"
        )
        body = {"expected_revision": session["revision"], "mutation_id": identity()}
        if action == "selection":
            body["selected_day"] = 3
        elif action == "single":
            body.update(row)
        elif action == "batch":
            body["operations"] = [row, {**row, "target": targets[1]}]
        wire.path = path
        with pytest.raises(ConnectionError, match="simulated_committed_reply_body_lost"):
            api.post(path, headers=headers, json=body)
        assert len(wire.dropped) == 1 and wire.dropped[0]["status"] == 200 and wire.dropped[0]["body_bytes"] > 0
    with client(state_root) as api:
        status = api.get(base + "/mutations/" + body["mutation_id"], headers=headers)
        assert status.status_code == 200, status.text
        recovered = status.json()
        assert recovered["state"] == "applied"
        assert recovered["mutation_receipt"]["committed_revision"] == body["expected_revision"] + 1
        fresh = api.get(base, headers=headers)
        assert fresh.status_code == 200, fresh.text
        current = fresh.json()["session"]
        assert current["revision"] == body["expected_revision"] + 1 and "session" not in recovered
        draft_count = {"selection": 0, "single": 1, "batch": 2, "undo": 0}[action]
        assert len(current["draft"]) == draft_count
        before = (state_root / "sessions" / f"{session['session_id']}.json").read_bytes()
        retried = api.post(path, headers=headers, json=body)
        assert retried.status_code == 200, retried.text
        assert (
            retried.json()["session"] == current and retried.json()["mutation_receipt"] == recovered["mutation_receipt"]
        )
        assert (state_root / "sessions" / f"{session['session_id']}.json").read_bytes() == before


@pytest.mark.parametrize("action", ["selection", "single", "batch", "undo"])
@pytest.mark.parametrize("already_committed", [False, True])
def test_accepted_pointer_change_requires_refresh_before_mutation_or_replay(tmp_path, action, already_committed):
    store, session, token = seed(tmp_path / "state")
    body = request_for(action, store, session)
    if already_committed:
        apply(action, store, session, body)
    before = snapshot(store, session)
    visible = session.public_dict()
    store._accepted_plan_resolver = lambda _: {"accepted_plan_id": "new-parent"}
    with pytest.raises(WorkspaceError, match="mutation_workspace_changed") as failure:
        apply(action, store, session, body)
    assert failure.value.status_code == 409 and snapshot(store, session) == before and session.public_dict() == visible
    if already_committed:
        store.authenticate(session.session_id, token)
        refreshed = snapshot(store, session)
        apply(action, store, session, body)
        assert (
            snapshot(store, session) == refreshed and session.accepted_plan_id == "new-parent" and session.draft == []
        )


@pytest.mark.parametrize("defect", ["invalid_target", "forged_role_proposal"])
def test_replay_must_not_return_unvalidated_current_workspace(tmp_path, defect):
    state_root = tmp_path / "invalid_current_state"
    with client(state_root) as api:
        created = api.post("/api/sessions", json={}).json()
        session = created["session"]
        headers = {"X-Session-Token": created["mutation_token"]}
        base = "/api/sessions/" + session["session_id"]
        target = created["workspace"]["draft_capabilities"]["parent_targets"][0]
        body = {
            "expected_revision": session["revision"],
            "mutation_id": identity(),
            "type": "keep_stop",
            "target": target,
            "parameters": {},
        }
        result = api.post(base + "/draft/operations", headers=headers, json=body)
        assert result.status_code == 200, result.text
        path = state_root / "sessions" / f"{session['session_id']}.json"
        saved = json.loads(path.read_text(encoding="utf-8"))
        if defect == "invalid_target":
            saved["session"]["draft"][0]["target"] = "not-a-trip-stop"
        else:
            saved["session"]["draft"][0].update(type="set_stop_role", parameters={"role": "scenic_stop"})
            saved["session"]["proposal"] = {"state": "eligible"}
        path.write_text(json.dumps(saved), encoding="utf-8")
        before = path.read_bytes()
        retried = api.post(base + "/draft/operations", headers=headers, json=body)
        assert retried.status_code in (409, 422), retried.text
        assert path.read_bytes() == before
