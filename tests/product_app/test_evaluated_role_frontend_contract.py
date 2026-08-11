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
NODE_TEST = Path(__file__).with_name("evaluated_role_frontend_contract.test.mjs")
REGISTRY = ROOT / "configs" / "product_app_registry.json"
TARGET = "stearns_wharf"


def _append_role(
    client: TestClient,
    session: dict,
    headers: dict[str, str],
    *,
    role: str,
    target: str = TARGET,
) -> dict:
    response = client.post(
        f"/api/sessions/{session['session_id']}/draft/operations",
        headers=headers,
        json={
            "expected_revision": session["revision"],
            "type": "set_stop_role",
            "target": target,
            "parameters": {"role": role},
            "source": "typed_stop_editor",
            "evidence_refs": [],
        },
    )
    assert response.status_code == 200
    return response.json()["session"]


def _expected(session: dict, impact: dict) -> dict:
    return {
        "session_id": session["session_id"],
        "trip_id": session["trip_id"],
        "run_id": session["run_id"],
        "revision": session["revision"],
        "accepted_plan_id": session["accepted_plan_id"],
        "parent_plan_content_hash": impact["parent_plan_content_hash"],
        "draft": session["draft"],
        "permission_decisions": session["permission_decisions"],
        "conversation_id": session["conversation_id"],
    }


def test_role_editor_separates_trip_role_from_place_identity_and_categories() -> None:
    source = APP.read_text(encoding="utf-8")
    css = CSS.read_text(encoding="utf-8")

    assert "properties.itinerary_role" in source
    assert "properties.itinerary_role_source" in source
    assert "Unavailable — choose a trip role" in source
    assert "Place categories" in source
    assert "trip-specific user-declared use" in source
    assert "does not change the PlaceEntity identity or categories" in source
    assert "semantic fit and recommendation remain Unavailable" in source
    assert "structural role remains draft only" in source
    assert 'properties?.itinerary_role ?? null' in source
    assert 'optionsHtml(vocab.stop_roles, properties?.role)' not in source
    assert ".typed-operation-form > button { min-height: 44px;" in css


def test_actual_role_api_payload_passes_js_and_forgeries_fail(tmp_path: Path) -> None:
    app = create_product_app(
        repository_root=ROOT,
        registry_path=REGISTRY,
        state_root=tmp_path / "state",
        additional_allowed_authorities=("testserver",),
    )
    with TestClient(app) as client:
        created = client.post("/api/sessions", json={}).json()
        workspace = created["workspace"]
        headers = {"X-Session-Token": created["mutation_token"]}
        before = _append_role(client, created["session"], headers, role="meal")
        impact_response = client.post(
            f"/api/sessions/{before['session_id']}/draft/impact-preview",
            headers=headers,
            json={"expected_revision": before["revision"]},
        )
        preview_response = client.post(
            f"/api/sessions/{before['session_id']}/preview",
            headers=headers,
            json={"expected_revision": before["revision"]},
        )
        assert impact_response.status_code == 200
        assert preview_response.status_code == 200
        acceptance = client.post(
            f"/api/sessions/{before['session_id']}/accept",
            headers=headers,
            json={"expected_revision": before["revision"] + 1},
        )
        assert acceptance.status_code == 409
        assert acceptance.json() == {"detail": "acceptance_not_enabled_until_w5"}

        structural_created = client.post("/api/sessions", json={}).json()
        structural_headers = {
            "X-Session-Token": structural_created["mutation_token"]
        }
        structural_session = _append_role(
            client,
            structural_created["session"],
            structural_headers,
            role="lodging",
        )
        structural_impact = client.post(
            f"/api/sessions/{structural_session['session_id']}/draft/impact-preview",
            headers=structural_headers,
            json={"expected_revision": structural_session["revision"]},
        )
        structural_preview = client.post(
            f"/api/sessions/{structural_session['session_id']}/preview",
            headers=structural_headers,
            json={"expected_revision": structural_session["revision"]},
        )
        assert structural_impact.status_code == 200
        assert structural_preview.status_code == 409
        assert structural_preview.json() == {
            "detail": "stop_role_evaluation_not_supported"
        }

        mixed_created = client.post("/api/sessions", json={}).json()
        mixed_headers = {"X-Session-Token": mixed_created["mutation_token"]}
        mixed_session = _append_role(
            client, mixed_created["session"], mixed_headers, role="meal"
        )
        duration = client.post(
            f"/api/sessions/{mixed_session['session_id']}/draft/operations",
            headers=mixed_headers,
            json={
                "expected_revision": mixed_session["revision"],
                "type": "set_stop_duration",
                "target": "surf_n_wear_s_beach_house",
                "parameters": {
                    "duration": {
                        "mode": "exact",
                        "preferred_minutes": 60,
                        "minimum_minutes": 60,
                        "maximum_minutes": 60,
                    }
                },
                "source": "typed_stop_editor",
                "evidence_refs": [],
            },
        )
        assert duration.status_code == 200
        mixed_session = duration.json()["session"]
        mixed_impact = client.post(
            f"/api/sessions/{mixed_session['session_id']}/draft/impact-preview",
            headers=mixed_headers,
            json={"expected_revision": mixed_session["revision"]},
        )
        assert mixed_impact.status_code == 200

        stale_created = client.post("/api/sessions", json={}).json()
        stale_headers = {"X-Session-Token": stale_created["mutation_token"]}
        stale_session = _append_role(
            client, stale_created["session"], stale_headers, role="activity"
        )
        stale = client.post(
            f"/api/sessions/{stale_session['session_id']}/draft/operations",
            headers=stale_headers,
            json={
                "expected_revision": 0,
                "type": "set_stop_role",
                "target": "surf_n_wear_s_beach_house",
                "parameters": {"role": "scenic_stop"},
                "source": "typed_stop_editor",
                "evidence_refs": [],
            },
        )
        assert stale.status_code == 409
        assert stale.json() == {"detail": "stale_session_revision"}

    impact = impact_response.json()
    contract_path = tmp_path / "actual-evaluated-role-contract.json"
    contract_path.write_text(
        json.dumps(
            {
                "workspace": {
                    "typed_edit_capabilities": workspace["typed_edit_capabilities"],
                    "draft_capabilities": workspace["draft_capabilities"],
                    "map_edit_capabilities": workspace["map_edit_capabilities"],
                    "timeline": workspace["timeline"],
                    "geography": workspace["geography"],
                    "role_constraint_evidence": workspace[
                        "role_constraint_evidence"
                    ],
                },
                "eligible": {
                    "before_session": before,
                    "impact": impact,
                    "preview": preview_response.json(),
                    "expected": _expected(before, impact),
                },
                "structural": {
                    "session": structural_session,
                    "impact": structural_impact.json(),
                },
                "mixed": {"session": mixed_session, "impact": mixed_impact.json()},
            }
        ),
        encoding="utf-8",
    )
    environment = os.environ.copy()
    environment["ACTUAL_EVALUATED_ROLE_CONTRACT_PATH"] = str(contract_path)
    environment["ACTUAL_EVALUATED_ROLE_TOKEN"] = created["mutation_token"]
    completed = subprocess.run(
        ["node", str(NODE_TEST)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "evaluated role frontend contract and forgeries passed" in completed.stdout


def test_asset_token_is_bumped_for_evaluated_role_contract() -> None:
    markup = INDEX.read_text(encoding="utf-8")

    assert "/static/css/app.css?v=20260810-stability5" in markup
    assert "/static/js/app.js?v=20260810-stability5" in markup
