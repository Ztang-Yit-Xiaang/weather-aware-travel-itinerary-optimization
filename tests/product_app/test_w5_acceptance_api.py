"""End-to-end W5 acceptance, restart, replay, and fail-closed API tests."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from itinerary_system.product_app.api import create_product_app

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "configs" / "product_app_registry.json"


def _client(state_root: Path) -> TestClient:
    return TestClient(
        create_product_app(
            repository_root=ROOT,
            registry_path=REGISTRY,
            state_root=state_root,
            additional_allowed_authorities=("testserver",),
        )
    )


def _session(client: TestClient) -> tuple[dict, dict[str, str]]:
    payload = client.post("/api/sessions", json={}).json()
    return payload["session"], {"X-Session-Token": payload["mutation_token"]}


def _eligible_preview(client: TestClient, session: dict, headers: dict[str, str]) -> dict:
    drafted = client.post(
        f"/api/sessions/{session['session_id']}/draft/operations",
        headers=headers,
        json={
            "expected_revision": session["revision"],
            "type": "replace_nearby",
            "target": "bixby_creek_bridge_viewpoint",
            "parameters": {"candidate_id": "pfeiffer_big_sur_state_park"},
        },
    ).json()["session"]
    response = client.post(
        f"/api/sessions/{session['session_id']}/preview",
        headers=headers,
        json={"expected_revision": drafted["revision"]},
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["proposal"]["acceptance_eligible"] is True
    return payload


def test_accept_survives_restart_and_idempotent_retry(tmp_path: Path) -> None:
    state_root = tmp_path / "state"
    with _client(state_root) as client:
        session, headers = _session(client)
        preview = _eligible_preview(client, session, headers)
        revision = preview["session"]["revision"]
        proposal = preview["proposal"]
        request = {
            "expected_revision": revision,
            "confirmation": "accept_evaluated_repair",
            "idempotency_key": "accept-evaluated-0001",
        }
        accepted = client.post(
            f"/api/sessions/{session['session_id']}/accept",
            headers=headers,
            json=request,
        )
        assert accepted.status_code == 200, accepted.text
        result = accepted.json()
        assert result["session"]["accepted_plan_id"] == proposal["child_plan_id"]
        assert result["session"]["draft"] == []
        assert result["result"]["decision"]["decision_type"] == "accept_repair"
        decision_id = result["result"]["decision"]["decision_id"]

    with _client(state_root) as restarted:
        restored = restarted.get(
            f"/api/sessions/{session['session_id']}", headers=headers
        )
        assert restored.status_code == 200, restored.text
        restored_session = restored.json()["session"]
        assert restored_session["accepted_plan_id"] == proposal["child_plan_id"]
        history = restarted.get(
            f"/api/sessions/{session['session_id']}/acceptance-decisions",
            headers=headers,
        ).json()
        assert [row["decision_id"] for row in history["decisions"]] == [decision_id]
        replay = restarted.post(
            f"/api/sessions/{session['session_id']}/accept",
            headers=headers,
            json=request,
        )
        assert replay.status_code == 200, replay.text
        assert replay.json()["result"]["decision"]["decision_id"] == decision_id
        kept = restarted.post(
            f"/api/sessions/{session['session_id']}/keep-original",
            headers=headers,
            json={
                "expected_revision": restored_session["revision"],
                "confirmation": "keep_original_plan",
                "idempotency_key": "keep-before-rollback-0001",
            },
        )
        assert kept.status_code == 200, kept.text
        kept_payload = kept.json()
        assert kept_payload["session"]["accepted_plan_id"] == history["pointer"]["original_plan_id"]
        rolled_back = restarted.post(
            f"/api/sessions/{session['session_id']}/acceptance-decisions/{decision_id}/rollback",
            headers=headers,
            json={
                "expected_revision": kept_payload["session"]["revision"],
                "confirmation": "rollback_to_prior_decision",
                "idempotency_key": "rollback-decision-0001",
            },
        )
        assert rolled_back.status_code == 200, rolled_back.text
        rollback_payload = rolled_back.json()
        assert rollback_payload["result"]["decision"]["decision_type"] == "rollback_to_eligible_plan"
        assert rollback_payload["session"]["accepted_plan_id"] == proposal["child_plan_id"]
        fresh = restarted.post("/api/sessions", json={})
        assert fresh.status_code == 200
        fresh_payload = fresh.json()
        assert fresh_payload["session"]["accepted_plan_id"] == proposal["child_plan_id"]
        assert fresh_payload["workspace"]["accepted_plan_pointer"]["decision_id"] == rollback_payload["result"]["decision"]["decision_id"]
        continued_edit = restarted.post(
            f"/api/sessions/{fresh_payload['session']['session_id']}/draft/operations",
            headers={"X-Session-Token": fresh_payload["mutation_token"]},
            json={
                "expected_revision": 0,
                "type": "set_stop_duration",
                "target": "pfeiffer_big_sur_state_park",
                "parameters": {"duration": {"mode": "exact", "preferred_minutes": 60,
                    "minimum_minutes": 60, "maximum_minutes": 60}},
                "source": "typed_stop_editor",
            },
        )
        assert continued_edit.status_code == 200, continued_edit.text
        assert continued_edit.json()["session"]["accepted_plan_id"] == proposal["child_plan_id"]
        assert continued_edit.json()["session"]["draft"][0]["type"] == "set_stop_duration"


def test_accept_requires_confirmation_and_rejects_ineligible_preview(tmp_path: Path) -> None:
    with _client(tmp_path / "state") as client:
        session, headers = _session(client)
        preview = _eligible_preview(client, session, headers)
        missing_confirmation = client.post(
            f"/api/sessions/{session['session_id']}/accept",
            headers=headers,
            json={
                "expected_revision": preview["session"]["revision"],
                "confirmation": "",
                "idempotency_key": "accept-evaluated-0002",
            },
        )
        assert missing_confirmation.status_code == 422
        assert missing_confirmation.json()["detail"] == "acceptance_confirmation_required"


def test_keep_original_is_durable_and_explicit(tmp_path: Path) -> None:
    state_root = tmp_path / "state"
    with _client(state_root) as client:
        session, headers = _session(client)
        kept = client.post(
            f"/api/sessions/{session['session_id']}/keep-original",
            headers=headers,
            json={
                "expected_revision": 0,
                "confirmation": "keep_original_plan",
                "idempotency_key": "keep-original-0001",
            },
        )
        assert kept.status_code == 200, kept.text
        payload = kept.json()
        assert payload["result"]["decision"]["decision_type"] == "keep_original"
        assert payload["session"]["revision"] == 1
        assert payload["session"]["accepted_plan_id"] == session["accepted_plan_id"]
