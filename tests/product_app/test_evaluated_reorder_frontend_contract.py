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
NODE_TEST = Path(__file__).with_name("evaluated_reorder_frontend_contract.test.mjs")
REGISTRY = ROOT / "configs" / "product_app_registry.json"


def test_reorder_editor_is_same_day_human_position_and_truthful_scope() -> None:
    source = APP.read_text(encoding="utf-8")
    css = CSS.read_text(encoding="utf-8")

    assert '["set_stop_day", "set_stop_order", "set_stop_time_window"].includes(name)' in source
    assert 'capability.supported_scope !== "same_day"' in source
    assert "capability.sequence_index_base !== 0" in source
    assert 'name="sequence_position"' in source
    assert "Positions are shown starting at 1" in source
    assert "sequence_index: Number(form.elements.sequence_position.value) - 1" in source
    assert "This reorders only within the accepted day" in source
    assert 'name="day" type="hidden"' in source
    assert '"duration_mode_evaluation_not_supported"' in source
    assert 'capability.latest_departure_semantics !== "departure_after_visit"' in source
    assert "Evaluated edit evidence" in source
    assert "stop-by-stop schedule accounting is Unavailable" in source
    assert "The accepted parent is unchanged; acceptance remains disabled until W5" in source
    assert "draft_evaluated_operation_combination_unsupported" in source
    assert "Preview blocked:" in source
    assert "blockingCodes.length === 0" in source
    assert ".typed-operation-form > button { min-height: 44px;" in css
    assert ".typed-readonly-value { min-height: 44px;" in css
    assert ".evaluated-edit-evidence > summary { min-height: 44px;" in css


def test_preview_response_is_normalized_before_session_state_changes() -> None:
    source = APP.read_text(encoding="utf-8")
    preview = source[source.index("async function previewDraft()") :]

    assert "await normalizeEvaluatedPreviewResponse(payload, expected)" in preview
    assert "if (!normalized)" in preview
    assert preview.index("await normalizeEvaluatedPreviewResponse(payload, expected)") < preview.index(
        "state.session = normalized.session"
    )
    assert "computedDraftContentHash(expected.draft)" in source
    assert "proposal.diff_identity.parent_plan_id !== proposal.parent_plan_id" in source
    assert "proposal.certificate_identity.plan_id !== proposal.child_plan_id" in source
    assert "proposal.route_validation_identity[key] === routeValidation[key]" in source
    assert "feature.properties.content_hash !== value.content_hash" in source


def test_actual_api_reorder_preview_passes_js_and_forged_variants_fail(
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
        initial = created.json()
        workspace = initial["workspace"]
        session = initial["session"]
        headers = {"X-Session-Token": initial["mutation_token"]}
        day = next(row for row in workspace["timeline"] if len(row["stops"]) >= 2)
        target = day["stops"][0]["id"]
        added = client.post(
            f"/api/sessions/{session['session_id']}/draft/operations",
            headers=headers,
            json={
                "expected_revision": session["revision"],
                "type": "set_stop_order",
                "target": target,
                "parameters": {"day": day["day"], "sequence_index": 1},
                "source": "map",
                "evidence_refs": [],
            },
        )
        assert added.status_code == 200
        before_session = added.json()["session"]
        impact_response = client.post(
            f"/api/sessions/{session['session_id']}/draft/impact-preview",
            headers=headers,
            json={"expected_revision": before_session["revision"]},
        )
        assert impact_response.status_code == 200
        impact = impact_response.json()
        preview_response = client.post(
            f"/api/sessions/{session['session_id']}/preview",
            headers=headers,
            json={"expected_revision": before_session["revision"]},
        )
        assert preview_response.status_code == 200
        legacy_created = client.post("/api/sessions", json={}).json()
        legacy_session = legacy_created["session"]
        legacy_headers = {"X-Session-Token": legacy_created["mutation_token"]}
        legacy_candidate = next(
            row
            for row in legacy_created["workspace"]["draft_capabilities"]["candidate_choices"]
            if row.get("replaces_stop_id")
        )
        legacy_added = client.post(
            f"/api/sessions/{legacy_session['session_id']}/draft/operations",
            headers=legacy_headers,
            json={
                "expected_revision": legacy_session["revision"],
                "type": "replace_nearby",
                "target": legacy_candidate["replaces_stop_id"],
                "parameters": {"candidate_id": legacy_candidate["candidate_id"]},
                "source": "map",
                "evidence_refs": [],
            },
        )
        assert legacy_added.status_code == 200
        legacy_before = legacy_added.json()["session"]
        legacy_impact_response = client.post(
            f"/api/sessions/{legacy_session['session_id']}/draft/impact-preview",
            headers=legacy_headers,
            json={"expected_revision": legacy_before["revision"]},
        )
        assert legacy_impact_response.status_code == 200
        legacy_impact = legacy_impact_response.json()
        legacy_preview_response = client.post(
            f"/api/sessions/{legacy_session['session_id']}/preview",
            headers=legacy_headers,
            json={"expected_revision": legacy_before["revision"]},
        )
        assert legacy_preview_response.status_code == 200
        mixed_created = client.post("/api/sessions", json={}).json()
        mixed_session = mixed_created["session"]
        mixed_headers = {"X-Session-Token": mixed_created["mutation_token"]}
        mixed_order = client.post(
            f"/api/sessions/{mixed_session['session_id']}/draft/operations",
            headers=mixed_headers,
            json={
                "expected_revision": mixed_session["revision"],
                "type": "set_stop_order",
                "target": target,
                "parameters": {"day": day["day"], "sequence_index": 1},
                "source": "map",
                "evidence_refs": [],
            },
        )
        assert mixed_order.status_code == 200
        mixed_session = mixed_order.json()["session"]
        day_one_stop = next(row for row in workspace["timeline"] if row["day"] == 1)[
            "stops"
        ][0]["id"]
        mixed_day = client.post(
            f"/api/sessions/{mixed_session['session_id']}/draft/operations",
            headers=mixed_headers,
            json={
                "expected_revision": mixed_session["revision"],
                "type": "set_stop_day",
                "target": day_one_stop,
                "parameters": {"day": 2},
                "source": "map",
                "evidence_refs": [],
            },
        )
        assert mixed_day.status_code == 200
        mixed_session = mixed_day.json()["session"]
        mixed_impact_response = client.post(
            f"/api/sessions/{mixed_session['session_id']}/draft/impact-preview",
            headers=mixed_headers,
            json={"expected_revision": mixed_session["revision"]},
        )
        assert mixed_impact_response.status_code == 200

    contract_path = tmp_path / "actual-evaluated-reorder-contract.json"
    contract_path.write_text(
        json.dumps(
            {
                "capabilities": workspace["typed_edit_capabilities"],
                "draft_capabilities": workspace["draft_capabilities"],
                "map_edit_capabilities": workspace["map_edit_capabilities"],
                "timeline": workspace["timeline"],
                "before_session": before_session,
                "impact": impact,
                "preview": preview_response.json(),
                "expected": {
                    "session_id": before_session["session_id"],
                    "trip_id": before_session["trip_id"],
                    "run_id": before_session["run_id"],
                    "revision": before_session["revision"],
                    "accepted_plan_id": before_session["accepted_plan_id"],
                    "parent_plan_content_hash": impact["parent_plan_content_hash"],
                    "draft": before_session["draft"],
                    "permission_decisions": before_session["permission_decisions"],
                    "conversation_id": before_session["conversation_id"],
                },
                "legacy": {
                    "preview": legacy_preview_response.json(),
                    "expected": {
                        "session_id": legacy_before["session_id"],
                        "trip_id": legacy_before["trip_id"],
                        "run_id": legacy_before["run_id"],
                        "revision": legacy_before["revision"],
                        "accepted_plan_id": legacy_before["accepted_plan_id"],
                        "parent_plan_content_hash": legacy_impact["parent_plan_content_hash"],
                        "draft": legacy_before["draft"],
                        "permission_decisions": legacy_before["permission_decisions"],
                        "conversation_id": legacy_before["conversation_id"],
                    },
                },
                "mixed": {
                    "session": mixed_session,
                    "impact": mixed_impact_response.json(),
                },
            }
        ),
        encoding="utf-8",
    )
    environment = os.environ.copy()
    environment["ACTUAL_EVALUATED_REORDER_CONTRACT_PATH"] = str(contract_path)
    completed = subprocess.run(
        ["node", str(NODE_TEST)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode == 0, completed.stderr
    assert "evaluated reorder frontend contract and forgeries passed" in completed.stdout


def test_asset_token_is_bumped_for_evaluated_reorder_contract() -> None:
    markup = INDEX.read_text(encoding="utf-8")

    assert "/static/css/app.css?v=20260810-stability5" in markup
    assert "/static/js/app.js?v=20260810-stability5" in markup
