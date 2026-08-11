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
INDEX = STATIC / "index.html"
NODE_TEST = Path(__file__).with_name("evaluated_duration_frontend_contract.test.mjs")
REGISTRY = ROOT / "configs" / "product_app_registry.json"


def _duration(mode: str, minutes: int) -> dict[str, object]:
    return {
        "mode": mode,
        "preferred_minutes": minutes if mode in {"exact", "preferred"} else None,
        "minimum_minutes": minutes if mode in {"exact", "minimum"} else None,
        "maximum_minutes": minutes if mode in {"exact", "maximum"} else None,
    }


def test_duration_editor_is_exact_evaluated_and_other_modes_are_truthfully_draft_only() -> None:
    source = APP.read_text(encoding="utf-8")

    assert 'supported_evaluated_modes, ["exact"]' in source
    assert 'draft_only_modes, ["preferred", "minimum", "maximum", "range"]' in source
    assert "Exact duration is evaluated-preview capable" in source
    assert "This duration mode remains draft only" in source
    assert "duration_mode_evaluation_not_supported" in source
    assert "preferred_minutes: preferred" in source
    assert "minimum_minutes: preferred" in source
    assert "maximum_minutes: preferred" in source
    assert "evaluator-owned modeled accounting" in source
    assert "parking/drop-off, walking transfer, queue wait, and service buffer" in source
    assert "PlanDiff v2 evidence" in source
    assert "Contextual risk delta: ${formatMetric(repair.tradeoffs.weather_risk_delta)}" in source


def test_actual_api_duration_preview_passes_js_and_forged_variants_fail(
    tmp_path: Path,
) -> None:
    app = create_product_app(
        repository_root=ROOT,
        registry_path=REGISTRY,
        state_root=tmp_path / "state",
        additional_allowed_authorities=("testserver",),
    )
    with TestClient(app) as client:
        created = client.post("/api/sessions", json={}).json()
        workspace = created["workspace"]
        session = created["session"]
        headers = {"X-Session-Token": created["mutation_token"]}
        target = "griffith_observatory"
        added = client.post(
            f"/api/sessions/{session['session_id']}/draft/operations",
            headers=headers,
            json={
                "expected_revision": session["revision"],
                "type": "set_stop_duration",
                "target": target,
                "parameters": {"duration": _duration("exact", 60)},
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
        preview_response = client.post(
            f"/api/sessions/{session['session_id']}/preview",
            headers=headers,
            json={"expected_revision": before_session["revision"]},
        )
        assert preview_response.status_code == 200

        preferred_created = client.post("/api/sessions", json={}).json()
        preferred_session = preferred_created["session"]
        preferred_headers = {"X-Session-Token": preferred_created["mutation_token"]}
        preferred_added = client.post(
            f"/api/sessions/{preferred_session['session_id']}/draft/operations",
            headers=preferred_headers,
            json={
                "expected_revision": preferred_session["revision"],
                "type": "set_stop_duration",
                "target": target,
                "parameters": {"duration": _duration("preferred", 60)},
                "source": "map",
                "evidence_refs": [],
            },
        )
        assert preferred_added.status_code == 200
        preferred_session = preferred_added.json()["session"]
        preferred_impact = client.post(
            f"/api/sessions/{preferred_session['session_id']}/draft/impact-preview",
            headers=preferred_headers,
            json={"expected_revision": preferred_session["revision"]},
        )
        assert preferred_impact.status_code == 200

        mixed_created = client.post("/api/sessions", json={}).json()
        mixed_session = mixed_created["session"]
        mixed_headers = {"X-Session-Token": mixed_created["mutation_token"]}
        mixed_duration = client.post(
            f"/api/sessions/{mixed_session['session_id']}/draft/operations",
            headers=mixed_headers,
            json={
                "expected_revision": mixed_session["revision"],
                "type": "set_stop_duration",
                "target": target,
                "parameters": {"duration": _duration("exact", 60)},
                "source": "map",
                "evidence_refs": [],
            },
        )
        assert mixed_duration.status_code == 200
        mixed_session = mixed_duration.json()["session"]
        day = next(row for row in workspace["timeline"] if len(row["stops"]) >= 2)
        mixed_order = client.post(
            f"/api/sessions/{mixed_session['session_id']}/draft/operations",
            headers=mixed_headers,
            json={
                "expected_revision": mixed_session["revision"],
                "type": "set_stop_order",
                "target": day["stops"][0]["id"],
                "parameters": {"day": day["day"], "sequence_index": 1},
                "source": "map",
                "evidence_refs": [],
            },
        )
        assert mixed_order.status_code == 200
        mixed_session = mixed_order.json()["session"]
        mixed_impact = client.post(
            f"/api/sessions/{mixed_session['session_id']}/draft/impact-preview",
            headers=mixed_headers,
            json={"expected_revision": mixed_session["revision"]},
        )
        assert mixed_impact.status_code == 200

        ineligible_created = client.post("/api/sessions", json={}).json()
        ineligible_session = ineligible_created["session"]
        ineligible_headers = {
            "X-Session-Token": ineligible_created["mutation_token"]
        }
        for target in ("stearns_wharf", "surf_n_wear_s_beach_house"):
            ineligible_added = client.post(
                f"/api/sessions/{ineligible_session['session_id']}/draft/operations",
                headers=ineligible_headers,
                json={
                    "expected_revision": ineligible_session["revision"],
                    "type": "set_stop_duration",
                    "target": target,
                    "parameters": {"duration": _duration("exact", 480)},
                    "source": "map",
                    "evidence_refs": [],
                },
            )
            assert ineligible_added.status_code == 200
            ineligible_session = ineligible_added.json()["session"]
        ineligible_impact = client.post(
            f"/api/sessions/{ineligible_session['session_id']}/draft/impact-preview",
            headers=ineligible_headers,
            json={"expected_revision": ineligible_session["revision"]},
        )
        assert ineligible_impact.status_code == 200
        ineligible_preview = client.post(
            f"/api/sessions/{ineligible_session['session_id']}/preview",
            headers=ineligible_headers,
            json={"expected_revision": ineligible_session["revision"]},
        )
        assert ineligible_preview.status_code == 200
        ineligible_accept = client.post(
            f"/api/sessions/{ineligible_session['session_id']}/accept",
            headers=ineligible_headers,
            json={"expected_revision": ineligible_session["revision"] + 1},
        )
        assert ineligible_accept.status_code == 409
        assert ineligible_accept.json() == {
            "detail": "acceptance_not_enabled_until_w5"
        }

    contract_path = tmp_path / "actual-evaluated-duration-contract.json"
    contract_path.write_text(
        json.dumps(
            {
                "workspace": {
                    "typed_edit_capabilities": workspace["typed_edit_capabilities"],
                    "draft_capabilities": workspace["draft_capabilities"],
                    "map_edit_capabilities": workspace["map_edit_capabilities"],
                    "timeline": workspace["timeline"],
                    "geography": workspace["geography"],
                },
                "before_session": before_session,
                "impact": impact_response.json(),
                "preview": preview_response.json(),
                "expected": {
                    "session_id": before_session["session_id"],
                    "trip_id": before_session["trip_id"],
                    "run_id": before_session["run_id"],
                    "revision": before_session["revision"],
                    "accepted_plan_id": before_session["accepted_plan_id"],
                    "parent_plan_content_hash": impact_response.json()["parent_plan_content_hash"],
                    "draft": before_session["draft"],
                    "permission_decisions": before_session["permission_decisions"],
                    "conversation_id": before_session["conversation_id"],
                },
                "preferred": {
                    "session": preferred_session,
                    "impact": preferred_impact.json(),
                },
                "mixed": {"session": mixed_session, "impact": mixed_impact.json()},
                "ineligible": {
                    "before_session": ineligible_session,
                    "impact": ineligible_impact.json(),
                    "preview": ineligible_preview.json(),
                    "expected": {
                        "session_id": ineligible_session["session_id"],
                        "trip_id": ineligible_session["trip_id"],
                        "run_id": ineligible_session["run_id"],
                        "revision": ineligible_session["revision"],
                        "accepted_plan_id": ineligible_session["accepted_plan_id"],
                        "parent_plan_content_hash": ineligible_impact.json()[
                            "parent_plan_content_hash"
                        ],
                        "draft": ineligible_session["draft"],
                        "permission_decisions": ineligible_session[
                            "permission_decisions"
                        ],
                        "conversation_id": ineligible_session["conversation_id"],
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    environment = os.environ.copy()
    environment["ACTUAL_EVALUATED_DURATION_CONTRACT_PATH"] = str(contract_path)
    completed = subprocess.run(
        ["node", str(NODE_TEST)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "evaluated duration frontend contract and forgeries passed" in completed.stdout


def test_asset_token_is_bumped_for_evaluated_duration_contract() -> None:
    markup = INDEX.read_text(encoding="utf-8")

    assert "/static/css/app.css?v=20260810-stability5" in markup
    assert "/static/js/app.js?v=20260810-stability5" in markup
