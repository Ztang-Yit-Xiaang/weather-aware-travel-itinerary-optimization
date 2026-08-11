from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from fastapi.testclient import TestClient

from itinerary_system.product_app.api import create_product_app

ROOT = Path(__file__).resolve().parents[2]
STATIC = ROOT / "src" / "itinerary_system" / "product_app" / "static"
APP = STATIC / "js" / "app.js"
CSS = STATIC / "css" / "app.css"
INDEX = STATIC / "index.html"
NODE_TEST = Path(__file__).with_name("typed_edit_frontend_contract.test.mjs")
REGISTRY = ROOT / "configs" / "product_app_registry.json"


def test_typed_capabilities_and_impact_are_exact_fail_closed_contracts() -> None:
    source = APP.read_text(encoding="utf-8")

    assert 'value.schema_version !== "product-typed-edit-capabilities-v1"' in source
    assert "exactKeys(value.operations, TYPED_EDIT_OPERATIONS)" in source
    assert '["must_keep", "booked"]' in source
    assert 'capability.protected_strengths_blocking_code !== "commitment_permission_required"' in source
    assert 'value.schema_version !== "product-draft-impact-preview-v1"' in source
    assert 'value.certification_status !== "not_certified"' in source
    assert "value.is_certified !== false" in source
    assert 'method: "POST"' in source
    assert "body: { expected_revision: state.session.revision }" in source
    assert "normalizeDraftImpactPreview(payload)" in source


def test_stop_editor_exposes_only_exact_server_typed_operations() -> None:
    source = APP.read_text(encoding="utf-8")

    for operation in (
        "set_stop_role",
        "set_stop_day",
        "set_stop_order",
        "set_stop_duration",
        "set_stop_time_window",
        "set_stop_commitment",
        "set_attribute_constraint",
    ):
        assert f'data-typed-operation="{operation}"' in source
    assert "role, timing, duration" not in source.lower()
    assert 'protectedValues: commitmentCapability.protected_strengths' in source
    assert 'protectedValues: attributeCapability.protected_strengths' in source
    assert "Each save appends exactly one server-validated operation" in source
    assert "Draft, not accepted" in source


def test_route_editor_is_human_named_typed_and_never_freehand() -> None:
    source = APP.read_text(encoding="utf-8")

    assert "routeNamePair(properties, routePlan)" in source
    assert 'data-typed-operation="report_route_issue"' in source
    assert "route_leg_id: routeLegId" in source
    assert "The route geometry is not freehand-editable" in source
    assert "No simulated scenic, toll, highway, or mode change is applied" in source
    assert "Change route preference" in source
    assert 'id="inspect-route-edit"' in source
    inspector_binding = source[source.index('$("#inspect-route-edit")'):]
    inspector_binding = inspector_binding[: inspector_binding.index('$("#exploratory-meaning")')]
    assert "addDraft" not in inspector_binding
    assert "openMapEdit" in inspector_binding


def test_typed_editors_require_exact_accepted_plan_identity() -> None:
    source = APP.read_text(encoding="utf-8")

    assert "selectedPlanId && selectedPlanId !== state.session.accepted_plan_id" in source
    assert "selected.plan_id !== state.session?.accepted_plan_id" in source
    assert "selectedAcceptedStopFeature(activeGeography())" in source
    assert "selectedAcceptedRouteFeature(geography)" in source
    assert "Repair-preview route legs cannot be edited in place" in source
    assert "Only an accepted-plan route leg can be edited" in source


def test_stop_inspector_shows_truthful_detail_and_parent_draft_state() -> None:
    source = APP.read_text(encoding="utf-8")

    for label in (
        "Itinerary role",
        "Day and order",
        "Arrival",
        "Departure",
        "Duration rule",
        "Commitment",
        "Attribute constraints",
        "Route access",
        "Source / freshness",
    ):
        assert label in source
    assert "Matches accepted parent" in source
    assert "Draft differs from parent" in source
    assert "Not certified. The accepted parent remains unchanged." in source
    assert "Road validated · no fallback" in source


def test_primary_draft_and_waypoint_content_uses_human_names() -> None:
    source = APP.read_text(encoding="utf-8")

    assert "operationDisplayName(op.type)" in source
    assert "draftTargetLabel(op)" in source
    assert "<li>${escapeHtml(op.type)} · ${escapeHtml(op.target)}</li>" not in source
    assert "stopNameForId(waypoint.insertion?.predecessor_id" in source
    assert "stopNameForId(waypoint.insertion?.successor_id" in source
    assert "routeDurationLabel(properties.duration_s)" in source
    assert "routeDistanceLabel(properties.distance_m)" in source
    assert "stopNameForId(gap.origin_id, displayed)" in source
    assert "stopNameForId(gap.destination_id, displayed)" in source


def test_impact_controls_evaluation_and_refreshes_after_writes() -> None:
    source = APP.read_text(encoding="utf-8")

    assert "await loadDraftImpactPreview({ render: false });" in source
    assert "canRunEvaluatedPreview()" in source
    assert "impactSummary.evaluated_executable_count" in source
    assert "impactSummary.draft_only_count" in source
    assert "if (!canRunEvaluatedPreview())" in source
    assert "draft_contains_non_executable_operation" in source
    assert "#inline-undo" in source


def test_editor_has_touch_keyboard_and_mobile_bottom_sheet_equivalence() -> None:
    css = CSS.read_text(encoding="utf-8")
    markup = INDEX.read_text(encoding="utf-8")

    assert 'id="typed-edit-surface"' in markup
    assert 'id="close-map-edit" type="button"' in markup
    assert "prototype-tools" not in markup
    assert ".typed-operation-form > button { min-height: 44px;" in css
    assert ".typed-editor details > summary" in css
    assert "min-height: 44px" in css
    assert "#map-edit-dialog { width: 100%;" in css
    assert "margin: auto 0 0" in css
    assert "env(safe-area-inset-bottom)" in css


def test_executable_contract_normalizers_and_parameter_reducers() -> None:
    completed = subprocess.run(
        ["node", str(NODE_TEST)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "typed edit frontend adversarial cases passed" in completed.stdout


def test_actual_backend_contracts_are_accepted_by_frontend_normalizers(
    tmp_path: Path,
) -> None:
    app = create_product_app(
        repository_root=ROOT,
        registry_path=REGISTRY,
        state_root=tmp_path / "state",
        additional_allowed_authorities=("testserver",),
    )
    with TestClient(app) as client:
        created = client.post("/api/sessions", json={})
        assert created.status_code == 200
        payload = created.json()
        session = payload["session"]
        headers = {"X-Session-Token": payload["mutation_token"]}
        added = client.post(
            f"/api/sessions/{session['session_id']}/draft/operations",
            headers=headers,
            json={
                "expected_revision": session["revision"],
                "type": "set_stop_duration",
                "target": "griffith_observatory",
                "parameters": {
                    "duration": {
                        "mode": "range",
                        "preferred_minutes": None,
                        "minimum_minutes": 60,
                        "maximum_minutes": 120,
                    }
                },
                "source": "typed_stop_editor",
                "evidence_refs": [],
            },
        )
        assert added.status_code == 200
        session = added.json()["session"]
        impact = client.post(
            f"/api/sessions/{session['session_id']}/draft/impact-preview",
            headers=headers,
            json={"expected_revision": session["revision"]},
        )
        assert impact.status_code == 200

    contract_path = tmp_path / "actual-typed-contract.json"
    contract_path.write_text(
        json.dumps(
            {
                "capabilities": payload["workspace"]["typed_edit_capabilities"],
                "draft_capabilities": payload["workspace"]["draft_capabilities"],
                "map_edit_capabilities": payload["workspace"]["map_edit_capabilities"],
                "draft": session["draft"],
                "impact": impact.json(),
            }
        ),
        encoding="utf-8",
    )
    environment = os.environ.copy()
    environment["ACTUAL_TYPED_CONTRACT_PATH"] = str(contract_path)
    completed = subprocess.run(
        ["node", str(NODE_TEST)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode == 0, completed.stderr
    assert "typed edit frontend adversarial cases passed" in completed.stdout
