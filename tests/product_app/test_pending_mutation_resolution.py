import json
import threading
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy

import pytest
from test_mutation_recovery import append, apply, client, identity, request_for, seed, snapshot

from itinerary_system.product_app.persistence import SessionSnapshotError
from itinerary_system.product_app.workspace import WorkspaceError, WorkspaceStore


def action_name(action):
    return "operations" if action in {"single", "batch"} else action


@pytest.mark.parametrize("action", ["selection", "single", "batch", "undo"])
@pytest.mark.parametrize("restart", [False, True])
def test_cancel_unseen_request_fences_delayed_write_and_is_idempotent(tmp_path, action, restart):
    root = tmp_path / "state"
    store, session, token = seed(root)
    body = request_for(action, store, session)
    draft = deepcopy(session.draft)
    selection = session.selected_day
    assert store.resolve_pending_mutation(session, action_name(action), body) is None
    assert session.revision == body["expected_revision"] + 1
    assert session.draft == draft and session.selected_day == selection
    assert session.mutation_receipts == []
    if restart:
        store = WorkspaceStore(root)
        session = store.authenticate_readonly(session.session_id, token)
    before = snapshot(store, session)
    with pytest.raises(WorkspaceError, match="stale_session_revision"):
        apply(action, store, session, body)
    assert snapshot(store, session) == before
    assert store.resolve_pending_mutation(session, action_name(action), body) is None
    assert snapshot(store, session) == before


@pytest.mark.parametrize("action", ["selection", "single", "batch", "undo"])
def test_cancel_committed_request_preserves_effect_and_returns_its_receipt(tmp_path, action):
    store, session, _ = seed(tmp_path / "state")
    body = request_for(action, store, session)
    apply(action, store, session, body)
    before = snapshot(store, session)
    receipt = json.loads(before)["mutation_receipts"][0]
    assert store.resolve_pending_mutation(session, action_name(action), body) == receipt
    assert snapshot(store, session) == before


@pytest.mark.parametrize("action", ["selection", "batch", "undo"])
def test_cancel_and_original_request_race_across_stores_is_atomic(tmp_path, action):
    root = tmp_path / "state"
    store, session, token = seed(root)
    body = request_for(action, store, session)
    initial_draft = deepcopy(session.draft)
    barrier = threading.Barrier(2)

    def run(cancel):
        other = WorkspaceStore(root)
        current = other.authenticate_readonly(session.session_id, token)
        barrier.wait(timeout=5)
        if cancel:
            return ("cancel", other.resolve_pending_mutation(current, action_name(action), body))
        try:
            apply(action, other, current, body)
            return ("original", "applied")
        except WorkspaceError as error:
            assert error.code == "stale_session_revision"
            return ("original", "stale")

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = dict(pool.map(run, [False, True]))
    current = WorkspaceStore(root).authenticate_readonly(session.session_id, token)
    assert current.revision == body["expected_revision"] + 1
    if results["original"] == "applied":
        assert results["cancel"] == current.mutation_receipts[0]
        assert len(current.mutation_receipts) == 1
    else:
        assert results["cancel"] is None and current.mutation_receipts == []
        assert current.draft == initial_draft


@pytest.mark.parametrize("after_commit", [False, True])
def test_cancel_save_failure_survives_restart_without_false_confirmation(tmp_path, monkeypatch, after_commit):
    root = tmp_path / "state"
    store, session, token = seed(root)
    body = request_for("single", store, session)
    save = store._snapshots.save

    def fail(current):
        if after_commit:
            save(current)
        raise SessionSnapshotError("session_write_failed")

    monkeypatch.setattr(store._snapshots, "save", fail)
    with pytest.raises(WorkspaceError, match="session_write_failed"):
        store.resolve_pending_mutation(session, "operations", body)
    assert session.revision == 0 and session.draft == []
    restarted = WorkspaceStore(root)
    current = restarted.authenticate_readonly(session.session_id, token)
    assert current.revision == int(after_commit)
    assert restarted.resolve_pending_mutation(current, "operations", body) is None
    assert current.revision == 1 and current.draft == []
    with pytest.raises(WorkspaceError, match="stale_session_revision"):
        apply("single", restarted, current, body)


@pytest.mark.parametrize("revision", [None, False, True, -1, 0.0, float("nan"), 2])
def test_invalid_or_future_cancel_revision_does_not_write(tmp_path, revision):
    store, session, _ = seed(tmp_path / "state")
    before = snapshot(store, session)
    with pytest.raises(WorkspaceError, match="invalid_mutation_request"):
        store.resolve_pending_mutation(session, "undo", {"expected_revision": revision, "mutation_id": identity()})
    assert snapshot(store, session) == before


@pytest.mark.parametrize("action", ["selection", "single", "batch", "undo"])
def test_cancel_cannot_relabel_an_applied_request(tmp_path, action):
    store, session, _ = seed(tmp_path / "state")
    body = request_for(action, store, session)
    apply(action, store, session, body)
    before = snapshot(store, session)
    with pytest.raises(WorkspaceError, match="mutation_id_conflict"):
        store.resolve_pending_mutation(session, action_name(action), {**body, "expected_revision": session.revision})
    assert snapshot(store, session) == before


def test_cancel_of_unseen_request_after_another_write_does_not_advance_again(tmp_path):
    store, session, _ = seed(tmp_path / "state")
    body = request_for("single", store, session)
    append(store, session, "b")
    before = snapshot(store, session)
    assert store.resolve_pending_mutation(session, "operations", body) is None
    assert snapshot(store, session) == before and len(session.draft) == 1


@pytest.mark.parametrize("action", ["selection", "single", "batch", "undo"])
def test_api_cancel_rejected_request_restores_usable_revision_without_draft_effect(tmp_path, action):
    with client(tmp_path / "state") as api:
        created = api.post("/api/sessions", json={}).json()
        session = created["session"]
        headers = {"X-Session-Token": created["mutation_token"]}
        base = "/api/sessions/" + session["session_id"]
        original = {"expected_revision": session["revision"], "mutation_id": identity()}
        if action == "selection":
            original["selected_day"] = 999
        elif action == "single":
            original.update(type="unsupported", target="missing", parameters={})
        elif action == "batch":
            original["operations"] = [{"type": "unsupported", "target": "missing", "parameters": {}}]
        path = "/selection" if action == "selection" else "/draft/undo" if action == "undo" else "/draft/operations"
        rejected = api.post(base + path, headers=headers, json=original)
        assert rejected.status_code == (400 if action in {"selection", "undo"} else 422), rejected.text
        current = api.get(base, headers=headers).json()["session"]
        assert current["revision"] == session["revision"] and current["draft"] == []
        cancel_path = base + "/mutations/" + original["mutation_id"] + "/cancel"
        resolved = api.post(cancel_path, headers=headers, json={"action": action_name(action), "request": original})
        assert resolved.status_code == 200, resolved.text
        proof = resolved.json()
        assert proof["state"] == "not_applied" and proof["resolved_revision"] == session["revision"] + 1
        assert proof["mutation_receipt"] is None and proof["result"] is None and "session" not in proof
        current = api.get(base, headers=headers).json()["session"]
        assert current["revision"] == proof["resolved_revision"] and current["draft"] == []
        assert current["accepted_plan_id"] == session["accepted_plan_id"]
        retried = api.post(base + path, headers=headers, json=original)
        assert retried.status_code == {"selection": 400, "single": 422, "batch": 422, "undo": 409}[action]
        again = api.post(cancel_path, headers=headers, json={"action": action_name(action), "request": original})
        assert again.json() == proof
        valid = api.post(
            base + "/selection",
            headers=headers,
            json={"expected_revision": current["revision"], "selected_day": 2, "mutation_id": identity()},
        )
        assert valid.status_code == 200, valid.text


def test_api_cancel_authentication_and_closed_envelope_have_no_effect(tmp_path):
    with client(tmp_path / "state") as api:
        created = api.post("/api/sessions", json={}).json()
        session = created["session"]
        base = "/api/sessions/" + session["session_id"]
        body = {"expected_revision": session["revision"], "mutation_id": identity()}
        envelope = {"action": "undo", "request": body}
        path = base + "/mutations/" + body["mutation_id"] + "/cancel"
        headers = {"X-Session-Token": created["mutation_token"]}
        file = tmp_path / "state/sessions" / f"{session['session_id']}.json"
        before = file.read_bytes()
        assert api.post(path, json=envelope).status_code == 403
        assert api.post(path, headers={"X-Session-Token": "wrong"}, json=envelope).status_code == 403
        for bad in [
            {**envelope, "extra": 1},
            {"action": [], "request": body},
            {"action": "undo", "request": None},
            {"action": "undo", "request": {**body, "expected_revision": True}},
            {"action": "undo", "request": {**body, "mutation_id": identity()}},
        ]:
            assert api.post(path, headers=headers, json=bad).status_code == 422
            assert file.read_bytes() == before


def test_cancel_preserves_and_rebinds_valid_evaluated_role_preview(tmp_path):
    with client(tmp_path / "state") as api:
        created = api.post("/api/sessions", json={}).json()
        session = created["session"]
        base = "/api/sessions/" + session["session_id"]
        headers = {"X-Session-Token": created["mutation_token"]}
        added = api.post(
            base + "/draft/operations",
            headers=headers,
            json={
                "expected_revision": session["revision"],
                "type": "set_stop_role",
                "target": "stearns_wharf",
                "parameters": {"role": "meal"},
                "source": "typed_stop_editor",
            },
        )
        assert added.status_code == 200, added.text
        session = added.json()["session"]
        evaluated = api.post(base + "/preview", headers=headers, json={"expected_revision": session["revision"]})
        assert evaluated.status_code == 200, evaluated.text
        session = evaluated.json()["session"]
        proposal = deepcopy(evaluated.json()["proposal"])
        original = {
            "expected_revision": session["revision"],
            "mutation_id": identity(),
            "type": "unsupported",
            "target": "missing",
            "parameters": {},
        }
        assert api.post(base + "/draft/operations", headers=headers, json=original).status_code == 422
        resolved = api.post(
            base + "/mutations/" + original["mutation_id"] + "/cancel",
            headers=headers,
            json={"action": "operations", "request": original},
        )
        assert resolved.status_code == 200, resolved.text
        refreshed = api.get(base, headers=headers)
        assert refreshed.status_code == 200, refreshed.text
        current = refreshed.json()["session"]
        assert current["revision"] == session["revision"] + 1 and current["draft"] == session["draft"]
        assert current["proposal"]["session_revision"] == current["revision"]
        for value in [proposal, current["proposal"]]:
            for key in ["expected_revision", "session_revision", "role_proposal_integrity"]:
                value.pop(key, None)
        assert current["proposal"] == proposal


@pytest.mark.parametrize("action", ["selection", "single", "batch", "undo"])
def test_api_cancel_already_applied_request_returns_receipt_without_another_write(tmp_path, action):
    with client(tmp_path / "state") as api:
        created = api.post("/api/sessions", json={}).json()
        session = created["session"]
        base = "/api/sessions/" + session["session_id"]
        headers = {"X-Session-Token": created["mutation_token"]}
        targets = created["workspace"]["draft_capabilities"]["parent_targets"]
        row = {"type": "keep_stop", "target": targets[0], "parameters": {}}
        if action == "undo":
            staged = api.post(
                base + "/draft/operations", headers=headers, json={"expected_revision": session["revision"], **row}
            )
            assert staged.status_code == 200, staged.text
            session = staged.json()["session"]
        original = {"expected_revision": session["revision"], "mutation_id": identity()}
        if action == "selection":
            original["selected_day"] = 2
        elif action == "single":
            original.update(row)
        elif action == "batch":
            original["operations"] = [row, {**row, "target": targets[1]}]
        path = "/selection" if action == "selection" else "/draft/undo" if action == "undo" else "/draft/operations"
        applied = api.post(base + path, headers=headers, json=original)
        assert applied.status_code == 200, applied.text
        file = tmp_path / "state/sessions" / f"{session['session_id']}.json"
        before = file.read_bytes()
        resolved = api.post(
            base + "/mutations/" + original["mutation_id"] + "/cancel",
            headers=headers,
            json={"action": action_name(action), "request": original},
        )
        assert resolved.status_code == 200, resolved.text
        proof = resolved.json()
        assert proof["state"] == "applied"
        assert proof["mutation_receipt"] == applied.json()["mutation_receipt"]
        assert proof["resolved_revision"] == applied.json()["session"]["revision"]
        assert file.read_bytes() == before


@pytest.mark.parametrize("action", [[], "other"])
def test_invalid_cancel_action_has_no_effect(tmp_path, action):
    store, session, _ = seed(tmp_path / "state")
    before = snapshot(store, session)
    with pytest.raises(WorkspaceError, match="invalid_mutation_request"):
        store.resolve_pending_mutation(session, action, {"expected_revision": 0, "mutation_id": identity()})
    assert snapshot(store, session) == before


def test_cancel_requires_refresh_when_accepted_pointer_changed(tmp_path):
    store, session, token = seed(tmp_path / "state")
    body = request_for("single", store, session)
    before = snapshot(store, session)
    store._accepted_plan_resolver = lambda _trip: {"accepted_plan_id": "new_parent"}
    with pytest.raises(WorkspaceError, match="mutation_workspace_changed"):
        store.resolve_pending_mutation(session, "operations", body)
    assert snapshot(store, session) == before
    refreshed = store.authenticate(session.session_id, token)
    assert refreshed.accepted_plan_id == "new_parent" and refreshed.revision == 1
    before = snapshot(store, refreshed)
    assert store.resolve_pending_mutation(refreshed, "operations", body) is None
    assert snapshot(store, refreshed) == before
