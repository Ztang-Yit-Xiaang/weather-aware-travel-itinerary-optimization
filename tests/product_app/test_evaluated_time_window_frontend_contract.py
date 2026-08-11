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
NODE_TEST = Path(__file__).with_name("evaluated_time_window_frontend_contract.test.mjs")
REGISTRY = ROOT / "configs" / "product_app_registry.json"


def _append_window(
    client: TestClient,
    session: dict,
    headers: dict[str, str],
    *,
    earliest: str | None,
    latest: str | None,
    target: str = "stearns_wharf",
) -> dict:
    response = client.post(
        f"/api/sessions/{session['session_id']}/draft/operations",
        headers=headers,
        json={
            "expected_revision": session["revision"],
            "type": "set_stop_time_window",
            "target": target,
            "parameters": {
                "earliest_arrival": earliest,
                "latest_departure": latest,
            },
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


def test_time_window_editor_discloses_evaluated_semantics_and_accessibility() -> None:
    source = APP.read_text(encoding="utf-8")
    css = CSS.read_text(encoding="utf-8")

    assert 'constraint_schema_version !== "stop-time-window-constraint-v1"' in source
    assert 'capability.early_arrival_policy !== "wait_until_earliest_arrival"' in source
    assert 'capability.latest_departure_semantics !== "departure_after_visit"' in source
    assert "Earliest service admission" in source
    assert "Latest departure after visit" in source
    assert "Raw road arrival may be earlier" in source
    assert "separate from place opening hours" in source
    assert "not a latest-start rule" in source
    assert "Cross-midnight windows are not supported" in source
    assert "Required-window schedule trace" in source
    assert "Incoming road leg from" in source
    assert "opening-hours evidence" in source
    assert "Raw road arrival is reconstructed from the exact validated incoming route leg" in source
    assert "configured evaluator fallback; not source-observed" in source
    assert ".typed-operation-form > button { min-height: 44px;" in css


def test_actual_time_window_api_payloads_pass_js_and_forgeries_fail(tmp_path: Path) -> None:
    app = create_product_app(
        repository_root=ROOT,
        registry_path=REGISTRY,
        state_root=tmp_path / "state",
        additional_allowed_authorities=("testserver",),
    )
    with TestClient(app) as client:
        eligible_created = client.post("/api/sessions", json={}).json()
        workspace = eligible_created["workspace"]
        eligible_headers = {"X-Session-Token": eligible_created["mutation_token"]}
        eligible_before = _append_window(
            client,
            eligible_created["session"],
            eligible_headers,
            earliest="10:00",
            latest=None,
        )
        eligible_impact_response = client.post(
            f"/api/sessions/{eligible_before['session_id']}/draft/impact-preview",
            headers=eligible_headers,
            json={"expected_revision": eligible_before["revision"]},
        )
        eligible_preview_response = client.post(
            f"/api/sessions/{eligible_before['session_id']}/preview",
            headers=eligible_headers,
            json={"expected_revision": eligible_before["revision"]},
        )
        assert eligible_impact_response.status_code == 200
        assert eligible_preview_response.status_code == 200
        acceptance = client.post(
            f"/api/sessions/{eligible_before['session_id']}/accept",
            headers=eligible_headers,
            json={"expected_revision": eligible_before["revision"] + 1},
        )
        assert acceptance.status_code == 409
        assert acceptance.json() == {"detail": "acceptance_not_enabled_until_w5"}

        latest_created = client.post("/api/sessions", json={}).json()
        latest_headers = {"X-Session-Token": latest_created["mutation_token"]}
        latest_before = _append_window(
            client,
            latest_created["session"],
            latest_headers,
            earliest=None,
            latest="00:01",
        )
        latest_impact_response = client.post(
            f"/api/sessions/{latest_before['session_id']}/draft/impact-preview",
            headers=latest_headers,
            json={"expected_revision": latest_before["revision"]},
        )
        latest_preview_response = client.post(
            f"/api/sessions/{latest_before['session_id']}/preview",
            headers=latest_headers,
            json={"expected_revision": latest_before["revision"]},
        )
        assert latest_impact_response.status_code == 200
        assert latest_preview_response.status_code == 200

        overrun_created = client.post("/api/sessions", json={}).json()
        overrun_headers = {"X-Session-Token": overrun_created["mutation_token"]}
        overrun_before = _append_window(
            client,
            overrun_created["session"],
            overrun_headers,
            earliest="23:59",
            latest=None,
        )
        overrun_impact_response = client.post(
            f"/api/sessions/{overrun_before['session_id']}/draft/impact-preview",
            headers=overrun_headers,
            json={"expected_revision": overrun_before["revision"]},
        )
        overrun_preview_response = client.post(
            f"/api/sessions/{overrun_before['session_id']}/preview",
            headers=overrun_headers,
            json={"expected_revision": overrun_before["revision"]},
        )
        assert overrun_impact_response.status_code == 200
        assert overrun_preview_response.status_code == 200

        mixed_created = client.post("/api/sessions", json={}).json()
        mixed_headers = {"X-Session-Token": mixed_created["mutation_token"]}
        mixed_session = _append_window(
            client,
            mixed_created["session"],
            mixed_headers,
            earliest="10:00",
            latest=None,
        )
        mixed_duration = client.post(
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
        assert mixed_duration.status_code == 200
        mixed_session = mixed_duration.json()["session"]
        mixed_impact_response = client.post(
            f"/api/sessions/{mixed_session['session_id']}/draft/impact-preview",
            headers=mixed_headers,
            json={"expected_revision": mixed_session["revision"]},
        )
        assert mixed_impact_response.status_code == 200

        stale_created = client.post("/api/sessions", json={}).json()
        stale_headers = {"X-Session-Token": stale_created["mutation_token"]}
        stale_session = _append_window(
            client,
            stale_created["session"],
            stale_headers,
            earliest="10:00",
            latest=None,
        )
        stale = client.post(
            f"/api/sessions/{stale_session['session_id']}/draft/operations",
            headers=stale_headers,
            json={
                "expected_revision": 0,
                "type": "set_stop_time_window",
                "target": "surf_n_wear_s_beach_house",
                "parameters": {"earliest_arrival": "11:00", "latest_departure": None},
                "source": "typed_stop_editor",
                "evidence_refs": [],
            },
        )
        stale_snapshot = client.get(
            f"/api/sessions/{stale_session['session_id']}", headers=stale_headers
        ).json()
        assert stale.status_code == 409
        assert stale.json() == {"detail": "stale_session_revision"}
        assert stale_snapshot["session"]["revision"] == 1
        assert len(stale_snapshot["session"]["draft"]) == 1

    eligible_impact = eligible_impact_response.json()
    latest_impact = latest_impact_response.json()
    overrun_impact = overrun_impact_response.json()
    contract_path = tmp_path / "actual-evaluated-time-window-contract.json"
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
                "eligible": {
                    "before_session": eligible_before,
                    "impact": eligible_impact,
                    "preview": eligible_preview_response.json(),
                    "expected": _expected(eligible_before, eligible_impact),
                },
                "latest_ineligible": {
                    "before_session": latest_before,
                    "impact": latest_impact,
                    "preview": latest_preview_response.json(),
                    "expected": _expected(latest_before, latest_impact),
                },
                "day_overrun": {
                    "before_session": overrun_before,
                    "impact": overrun_impact,
                    "preview": overrun_preview_response.json(),
                    "expected": _expected(overrun_before, overrun_impact),
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
    environment["ACTUAL_EVALUATED_TIME_WINDOW_CONTRACT_PATH"] = str(contract_path)
    completed = subprocess.run(
        ["node", str(NODE_TEST)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "evaluated time-window frontend contract and forgeries passed" in completed.stdout


def test_asset_token_is_bumped_for_evaluated_time_window_contract() -> None:
    markup = INDEX.read_text(encoding="utf-8")

    assert "/static/css/app.css?v=20260810-stability5" in markup
    assert "/static/js/app.js?v=20260810-stability5" in markup
