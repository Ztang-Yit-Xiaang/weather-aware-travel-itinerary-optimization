"""Read-only interest exploration follows an independently evaluated preview."""

from copy import deepcopy
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from itinerary_system.product_app.api import create_product_app
from itinerary_system.product_app.copilot import DeterministicCopilotAdapter
from itinerary_system.product_app.workspace import WorkspaceError

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def evaluated_preview(tmp_path_factory):
    app = create_product_app(
        repository_root=ROOT, state_root=tmp_path_factory.mktemp("interest-preview"),
        additional_allowed_authorities=("testserver",), copilot=DeterministicCopilotAdapter(),
    )
    with TestClient(app) as client:
        created = client.post("/api/sessions", json={}).json()
        headers = {"X-Session-Token": created["mutation_token"]}
        path = "/api/sessions/" + created["session"]["session_id"]
        staged = client.post(path + "/draft/operations", headers=headers, json={
            "expected_revision": created["session"]["revision"], "source": "map",
            "type": "replace_nearby", "target": "bixby_creek_bridge_viewpoint",
            "parameters": {"candidate_id": "pfeiffer_big_sur_state_park"},
        })
        assert staged.status_code == 200, staged.text
        preview = client.post(path + "/preview", headers=headers, json={
            "expected_revision": staged.json()["session"]["revision"],
        })
        assert preview.status_code == 200, preview.text
        payload = preview.json()
        assert payload["proposal"]["state"] == "eligible"
        yield client, app.state.product_service, payload, path, headers


def test_actual_evaluated_preview_interest_keeps_the_accepted_parent_and_session(evaluated_preview):
    client, service, payload, path, headers = evaluated_preview
    geography = payload["proposal"]["geography_plan"]
    route = next(f["properties"] for f in geography["validated_legs"]["features"] if f["properties"]["day"] == 5)
    session = service.workspace.get(payload["session"]["session_id"])
    before = deepcopy(session.public_dict())
    response = client.get(path + "/map/interest-regions", headers=headers, params={
        "expected_revision": session.revision, "day": 5, "route_leg_id": route["route_leg_id"],
        "plan_id": geography["plan_id"],
    })
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["plan_scope"] == "evaluated_preview"
    assert result["accepted_plan_id"] == before["accepted_plan_id"] != geography["plan_id"]
    assert result["exploration_plan_id"] == geography["plan_id"]
    assert result["exploration_plan_content_hash"] == geography["content_hash"]
    assert result["candidate_universe"]["count"] > 0
    assert result["surface"]["status"] == "ready"
    assert session.public_dict() == before


@pytest.mark.parametrize("tamper", ["missing", "ineligible", "child_hash", "parent_hash", "draft_hash"])
def test_interest_rejects_withdrawn_or_changed_preview(evaluated_preview, tamper):
    _, service, payload, _, _ = evaluated_preview
    session = deepcopy(service.workspace.get(payload["session"]["session_id"]))
    plan_id = session.proposal["child_plan_id"]
    if tamper == "missing":
        session.proposal = None
    elif tamper == "ineligible":
        session.proposal["state"] = "ineligible"
    else:
        field = {"child_hash": "child_plan_content_hash", "parent_hash": "parent_plan_content_hash",
                 "draft_hash": "draft_content_hash"}[tamper]
        session.proposal[field] = "changed"
    with pytest.raises(WorkspaceError, match="interest_preview_unavailable"):
        service.interest_plan_context(session, plan_id)


def test_interest_does_not_accept_an_arbitrary_plan_identifier(evaluated_preview):
    client, _, payload, path, headers = evaluated_preview
    response = client.get(path + "/map/interest-regions", headers=headers, params={
        "expected_revision": payload["session"]["revision"], "day": 5,
        "route_leg_id": "foreign_route", "plan_id": "foreign_plan",
    })
    assert response.status_code == 409
    assert response.json()["detail"] == "interest_preview_unavailable"


def test_runtime_interest_routes_require_full_geometry_and_exact_provenance(evaluated_preview):
    from itinerary_system.product_app.candidate_discovery import _raw_route_cell_is_exact

    _, service, payload, _, _ = evaluated_preview
    session = service.workspace.get(payload["session"]["session_id"])
    _, _, matrix, _ = service.interest_plan_context(session)
    row = deepcopy(next(cell for cell in matrix["cells"] if cell.get("geometry")))
    row.update(provider="runtime_osrm", query_hash="a" * 64, routing_status="osrm_route_validated",
               geometry_source="runtime_osrm_geojson", distance_source="runtime_osrm_route",
               duration_source="runtime_osrm_route")
    context = row["context_snapshot_id"]
    assert not _raw_route_cell_is_exact(row, context)
    assert _raw_route_cell_is_exact(row, context, allow_runtime_routes=True)
    for field, value in [("geometry", []), ("query_hash", "a" * 16), ("fallback_used", True),
                         ("routing_status", "osrm_table_validated"), ("duration_source", "estimated")]:
        assert not _raw_route_cell_is_exact({**row, field: value}, context, allow_runtime_routes=True)


def test_selection_consults_current_accepted_snapshot_routes(evaluated_preview, monkeypatch):
    client, service, _, _, _ = evaluated_preview
    created = client.post("/api/sessions", json={}).json()
    monkeypatch.setattr(service, "valid_segments_for_plan", lambda run_id, plan_id: {"snapshot_only_leg"})
    response = client.post("/api/sessions/" + created["session"]["session_id"] + "/selection", json={
        "expected_revision": created["session"]["revision"], "selected_day": 5,
        "selected_segment_id": "snapshot_only_leg",
    }, headers={"X-Session-Token": created["mutation_token"]})
    assert response.status_code == 200, response.text
    assert response.json()["session"]["selected_segment_id"] == "snapshot_only_leg"


def test_preview_can_still_be_accepted_after_interest_route_selection(evaluated_preview):
    client, _, payload, path, headers = evaluated_preview
    geography = payload["proposal"]["geography_plan"]
    leg = next(f["properties"] for f in geography["validated_legs"]["features"] if f["properties"]["day"] == 5)
    selected = client.post(path + "/selection", headers=headers, json={
        "expected_revision": payload["session"]["revision"], "selected_day": 5,
        "selected_segment_id": leg["route_leg_id"],
    })
    assert selected.status_code == 200, selected.text
    session = selected.json()["session"]
    assert session["proposal"]["session_revision"] == session["revision"]
    accepted = client.post(path + "/accept", headers=headers, json={
        "expected_revision": session["revision"], "confirmation": "accept_evaluated_repair",
        "idempotency_key": "interest-preview-after-selection-20261022",
    })
    assert accepted.status_code == 200, accepted.text
    assert accepted.json()["session"]["accepted_plan_id"] == geography["plan_id"]


def test_aged_real_trip_preview_is_inspectable_but_not_executable(monkeypatch):
    from itinerary_system.product_app import trip_replanner

    directory = os.environ.get("INTEREST_TRIP_REVIEW_ROOT")
    if not directory:
        pytest.skip("Set INTEREST_TRIP_REVIEW_ROOT to the task-owned real trip review fixture")
    directory = Path(directory)
    app = create_product_app(repository_root=ROOT, state_root=directory / "state",
                             additional_allowed_authorities=("testserver",))
    service = app.state.product_service
    from itinerary_system.product_app.persistence import SessionSnapshotStore

    session_id = SessionSnapshotStore(directory / "state").session_ids()[0]
    session = service.workspace.get(session_id)
    assert session.proposal["execution_mode"] == "trip_replan_independent_evaluation"
    before = deepcopy(session.public_dict())
    monkeypatch.setattr(trip_replanner, "MAX_EVIDENCE_AGE_SECONDS", -1)
    plan, geography, matrix, scope = service.interest_plan_context(session, session.proposal["child_plan_id"])
    assert scope == "evaluated_preview"
    assert len(plan["ordered_days"]) == 14
    assert geography["plan_id"] == plan["plan_id"]
    assert matrix["cells"]
    assert session.public_dict() == before
    compiler = service.compiler_for_session(session)
    with pytest.raises(WorkspaceError, match="trip_planning_evidence_expired"):
        compiler.compile(session.draft, accepted_plan_id=session.accepted_plan_id)
