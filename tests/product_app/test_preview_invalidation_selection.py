"""Real local API regression: invalidating a preview restores accepted selection scope."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest
from fastapi.encoders import jsonable_encoder
from fastapi.testclient import TestClient

from itinerary_system.product_app.api import create_product_app
from itinerary_system.product_app.persistence import SessionSnapshotError
from itinerary_system.product_app.service import ProductService
from itinerary_system.product_app.workspace import WorkspaceError, WorkspaceStore

ROOT = Path(__file__).resolve().parents[2]
BIXBY = "bixby_creek_bridge_viewpoint"
PFEIFFER = "pfeiffer_big_sur_state_park"


@pytest.mark.parametrize("mutation", ["undo", "append", "clear"])
@pytest.mark.parametrize("accepted_child", [False, True])
def test_invalidated_preview_selection_uses_actual_accepted_artifact(
    tmp_path: Path, monkeypatch, mutation: str, accepted_child: bool,
) -> None:
    selection_loads = []
    load_selection = ProductService._load_accepted_selection

    def observe_selection(service, trip_id):
        # Workspace restoration happens before the service caches exist. This
        # callback must only run later, at a real preview-to-parent transition.
        assert hasattr(service, "workspace") and hasattr(service, "_geographies")
        selection_loads.append(trip_id)
        return load_selection(service, trip_id)

    monkeypatch.setattr(ProductService, "_load_accepted_selection", observe_selection)
    app = create_product_app(
        repository_root=ROOT,
        registry_path=ROOT / "configs/product_app_registry.json",
        state_root=tmp_path / "state",
        additional_allowed_authorities=("testserver",),
    )
    records: list[dict] = []
    session: dict = {}
    headers: dict[str, str] = {}
    evidence = tmp_path / "exact_selection_io.json"

    with TestClient(app) as client:
        def post(suffix: str, body: dict) -> dict:
            nonlocal session
            path = f"/api/sessions/{session['session_id']}/{suffix}"
            response = client.post(path, headers=headers, json=body)
            records.append({"method": "POST", "path": path, "input": body,
                            "status": response.status_code, "output": response.json()})
            evidence.write_text(json.dumps(records, indent=2), encoding="utf-8")
            assert response.status_code == 200, response.text
            payload = response.json()
            session = payload["session"]
            return payload

        created = client.post("/api/sessions", json={}).json()
        session = created["session"]
        headers = {"X-Session-Token": created["mutation_token"]}
        # Only the authentication token is redacted; complete synthetic bodies remain.
        records.append({"method": "POST", "path": "/api/sessions", "input": {},
                        "output": {**created, "mutation_token": "OMITTED_SECRET"}})
        original_geography = next(row for row in created["workspace"]["geography"]["plans"]
                                  if row["plan_id"] == session["accepted_plan_id"])
        original_day = next(row["properties"]["day"]
                            for row in original_geography["stops"]["features"]
                            if row["properties"]["stop_id"] == BIXBY)
        post("selection", {"expected_revision": session["revision"],
                           "selected_day": original_day, "selected_stop_id": BIXBY})
        post("draft/operations", {"expected_revision": session["revision"],
                                  "type": "replace_nearby", "target": BIXBY,
                                  "parameters": {"candidate_id": PFEIFFER}})
        preview = post("preview", {"expected_revision": session["revision"]})
        assert preview["proposal"]["acceptance_eligible"] is True
        assert session["selected_stop_id"] == PFEIFFER
        if accepted_child:
            accepted = post("accept", {"expected_revision": session["revision"],
                                       "confirmation": "accept_evaluated_repair",
                                       "idempotency_key": "selection-accepted-child"})
            baseline = next(row for row in accepted["workspace"]["geography"]["plans"]
                            if row["plan_id"] == session["accepted_plan_id"])
            post("draft/operations", {"expected_revision": session["revision"],
                                      "type": "keep_stop", "target": PFEIFFER, "parameters": {}})
            preview = post("preview", {"expected_revision": session["revision"]})
            assert preview["proposal"]["acceptance_eligible"] is True
        else:
            baseline = original_geography
        accepted_id = session["accepted_plan_id"]
        assert selection_loads == [], "Ordinary authentication and staging must not load selection geography"
        if mutation == "clear":
            service = app.state.product_runtime.require_service()
            current = service.workspace.authenticate(session["session_id"], headers["X-Session-Token"])
            before = jsonable_encoder(current.public_dict())
            service.workspace.clear_draft(current)
            session = jsonable_encoder(current.public_dict())
            records.append({"method": "WorkspaceStore.clear_draft", "input": before, "output": session})
        elif mutation == "append":
            post("draft/operations", {"expected_revision": session["revision"],
                                      "type": "keep_stop", "target": "golden_gate_bridge",
                                      "parameters": {}})
        else:
            post("draft/undo", {"expected_revision": session["revision"]})
        restored = client.get(f"/api/sessions/{session['session_id']}", headers=headers)
        records.append({"method": "GET", "path": restored.request.url.path,
                        "status": restored.status_code, "output": restored.json()})
        records.append({"selection_resolution_calls": selection_loads})
        evidence.write_text(json.dumps(records, indent=2), encoding="utf-8")
        assert restored.status_code == 200
        assert restored.json()["session"] == session
        assert session["accepted_plan_id"] == accepted_id
        assert session["proposal"] is None
        assert selection_loads == [session["trip_id"]]
        assert len(session["draft"]) == (2 if mutation == "append" else 0)
        assert session["selected_day"] == original_day
        assert session["selected_stop_id"] == (PFEIFFER if accepted_child else None)
        stop_ids = {row["properties"]["stop_id"] for row in baseline["stops"]["features"]}
        assert session["selected_stop_id"] is None or session["selected_stop_id"] in stop_ids


def _selection_store(tmp_path: Path):
    pointer = {
        "accepted_plan_id": "parent", "accepted_plan_content_hash": "parent-hash",
        "selection_geography": {
            "plan_id": "parent", "content_hash": "parent-hash",
            "stops": {"features": [{"properties": {"stop_id": "shared", "day": 2}},
                                    {"properties": {"stop_id": "bixby", "day": 5}}]},
            "validated_legs": {"features": [{"properties": {"route_leg_id": "shared-leg"}}]},
        },
    }
    store = WorkspaceStore(tmp_path / "state", accepted_plan_resolver=lambda trip: {
        key: value for key, value in pointer.items() if key != "selection_geography"
    }, accepted_selection_resolver=lambda trip: deepcopy(pointer))
    session, token = store.create_session("run", "parent", 5)
    store.add_operation(session, {"expected_revision": 0, "type": "keep_stop", "target": "bixby",
                                  "parameters": {}}, valid_stop_ids={"bixby"}, day_count=7)
    store.add_operation(session, {"expected_revision": 1, "type": "keep_stop", "target": "shared",
                                  "parameters": {}}, valid_stop_ids={"bixby", "shared"}, day_count=7)
    store.set_proposal(session, {
        "state": "eligible", "child_plan_id": "child",
        "geography_plan": {"plan_id": "child", "stops": {"features": [
            {"properties": {"stop_id": "shared", "day": 3}},
            {"properties": {"stop_id": "pfeiffer", "day": 5}},
        ]}},
    }, session.revision)
    return store, session, token, pointer


@pytest.mark.parametrize(("stop", "segment", "day", "expected_stop", "expected_segment", "expected_day"), [
    ("pfeiffer", "child-leg", 5, None, None, 5),
    ("shared", "shared-leg", 3, "shared", "shared-leg", 2),
    ("shared", "child-leg", 3, "shared", None, 2),
    (None, "shared-leg", 5, None, "shared-leg", 5),
    (None, "child-leg", 5, None, None, 5),
])
def test_undo_projects_selection_not_remaining_draft(
    tmp_path: Path, stop, segment, day, expected_stop, expected_segment, expected_day,
) -> None:
    store, session, token, _ = _selection_store(tmp_path)
    store.select(session, {"expected_revision": session.revision, "selected_stop_id": stop,
                           "selected_segment_id": segment, "selected_day": day})
    remaining = deepcopy(session.draft[0].as_dict())
    before = jsonable_encoder(session.public_dict())
    store.undo(session, session.revision)
    after = jsonable_encoder(session.public_dict())
    (tmp_path / "exact_selection_io.json").write_text(json.dumps({"input": before, "output": after}), encoding="utf-8")
    assert [row.as_dict() for row in session.draft] == [remaining]
    assert session.proposal is None
    assert (session.selected_stop_id, session.selected_segment_id, session.selected_day) == (
        expected_stop, expected_segment, expected_day,
    )
    assert store.authenticate(session.session_id, token).public_dict() == session.public_dict()


@pytest.mark.parametrize("proposal", [None, {"state": "ineligible"}, {"state": "eligible"}])
def test_ordinary_draft_mutation_preserves_draft_waypoint_selection(tmp_path: Path, proposal) -> None:
    store, session, _, _ = _selection_store(tmp_path)
    store.clear_draft(session)
    store.add_operation(session, {"expected_revision": session.revision, "type": "keep_stop", "target": "bixby",
                                  "parameters": {}}, valid_stop_ids={"bixby"}, day_count=7)
    if proposal is not None:
        store.set_proposal(session, proposal, session.revision)
    store.select(session, {"expected_revision": session.revision, "selected_stop_id": "draft-waypoint",
                           "selected_segment_id": "draft-leg", "selected_day": 5})
    store.add_operation(session, {"expected_revision": session.revision, "type": "keep_stop", "target": "shared",
                                  "parameters": {}}, valid_stop_ids={"bixby", "shared"}, day_count=7)
    assert (session.selected_stop_id, session.selected_segment_id, session.selected_day) == (
        "draft-waypoint", "draft-leg", 5,
    )


@pytest.mark.parametrize("fault", ["pointer", "plan", "hash", "write"])
def test_selection_reconciliation_failure_is_atomic(tmp_path: Path, monkeypatch, fault: str) -> None:
    store, session, _, pointer = _selection_store(tmp_path)
    store.select(session, {"expected_revision": session.revision, "selected_stop_id": "pfeiffer",
                           "selected_segment_id": "child-leg", "selected_day": 5})
    before = deepcopy(session.public_dict())
    snapshot = tmp_path / "state/sessions" / f"{session.session_id}.json"
    raw_before = snapshot.read_bytes()
    if fault == "pointer":
        pointer["accepted_plan_id"] = "foreign"
    elif fault == "plan":
        pointer["selection_geography"]["plan_id"] = "foreign"
    elif fault == "hash":
        pointer["selection_geography"]["content_hash"] = "foreign"
    else:
        def fail_save(_session):
            raise SessionSnapshotError("session_write_failed")
        monkeypatch.setattr(store._snapshots, "save", fail_save)
    code = "session_write_failed" if fault == "write" else "accepted_selection_context_unavailable"
    with pytest.raises(WorkspaceError, match=code):
        store.undo(session, session.revision)
    assert session.public_dict() == before
    assert snapshot.read_bytes() == raw_before


def test_restoration_does_not_resolve_selection_before_service_initialization(tmp_path: Path) -> None:
    store, session, _, _ = _selection_store(tmp_path)

    def unavailable_during_construction(_trip_id):
        raise AssertionError("Selection callback ran during workspace restoration")

    restored = WorkspaceStore(
        store.state_root, accepted_selection_resolver=unavailable_during_construction,
    )
    assert restored.get(session.session_id).public_dict() == session.public_dict()
