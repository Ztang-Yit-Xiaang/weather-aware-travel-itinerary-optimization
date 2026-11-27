from __future__ import annotations

import json
import os
import subprocess
from copy import deepcopy
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from itinerary_system.product_app.api import create_product_app
from itinerary_system.product_app.models import ComponentHealthV1
from itinerary_system.product_app.runtime import ProductRuntime
from itinerary_system.product_app.service import _replay_role_certificate_with_persisted_timestamp
from itinerary_system.product_app.workspace import WorkspaceError
from itinerary_system.research_artifacts import stable_content_hash

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
    assert "properties?.itinerary_role ?? null" in source
    assert "optionsHtml(vocab.stop_roles, properties?.role)" not in source
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
            json={
                "expected_revision": before["revision"] + 1,
                "confirmation": "accept_evaluated_repair",
                "idempotency_key": "role-frontend-0001",
            },
        )
        assert acceptance.status_code == 409
        assert preview_response.json()["proposal"]["state"] == "ineligible"
        assert acceptance.json() == {"detail": "acceptance_proposal_invalid"}

        structural_created = client.post("/api/sessions", json={}).json()
        structural_headers = {"X-Session-Token": structural_created["mutation_token"]}
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
        assert structural_preview.json() == {"detail": "stop_role_evaluation_not_supported"}

        mixed_created = client.post("/api/sessions", json={}).json()
        mixed_headers = {"X-Session-Token": mixed_created["mutation_token"]}
        mixed_session = _append_role(client, mixed_created["session"], mixed_headers, role="meal")
        duration = client.post(
            f"/api/sessions/{mixed_session['session_id']}/draft/operations",
            headers=mixed_headers,
            json={
                "expected_revision": mixed_session["revision"],
                "type": "set_stop_duration",
                "target": "griffith_observatory",
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
        stale_session = _append_role(client, stale_created["session"], stale_headers, role="activity")
        stale = client.post(
            f"/api/sessions/{stale_session['session_id']}/draft/operations",
            headers=stale_headers,
            json={
                "expected_revision": 0,
                "type": "set_stop_role",
                "target": "griffith_observatory",
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
                    "role_constraint_evidence": workspace["role_constraint_evidence"],
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


def test_assets_are_build_bound_for_evaluated_role_contract() -> None:
    markup = INDEX.read_text(encoding="utf-8")

    assert "/static/css/app.css?v=__PRODUCT_BUILD_ID__" in markup
    assert "/static/js/app.js?v=__PRODUCT_BUILD_ID__" in markup
    assert 'name="product-build-id" content="__PRODUCT_BUILD_ID__"' in markup


def test_real_replacement_duration_time_window_role_acceptance_survives_restart(tmp_path, monkeypatch):
    monkeypatch.setenv("PRODUCT_COPILOT_ADAPTER", "deterministic")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setattr(
        ProductRuntime, "_probe_map", lambda self: ComponentHealthV1("map", "ready", False, "maplibre_ready")
    )
    state_root = tmp_path / "s"
    exchanges, snapshots, plan_ids = [], {}, []
    authentication = {}

    def client():
        return TestClient(
            create_product_app(
                repository_root=ROOT,
                registry_path=REGISTRY,
                state_root=state_root,
                additional_allowed_authorities=("testserver",),
            )
        )

    def request(api, method, path, body=None):
        response = api.request(method, path, json=body, headers=authentication)
        output = response.json()
        exchanges.append(
            {
                "method": method,
                "path": path,
                "input": body,
                "status": response.status_code,
                "output": {key: value for key, value in output.items() if key != "mutation_token"},
            }
        )
        (tmp_path / "exact_role_chain_io.json").write_text(json.dumps(exchanges, sort_keys=True), encoding="utf-8")
        assert response.status_code == 200, response.text
        return output

    with client() as api:
        created = request(api, "POST", "/api/sessions", {})
    session = created["session"]
    authentication["X-Session-Token"] = created["mutation_token"]
    path = f"/api/sessions/{session['session_id']}"
    original = session["accepted_plan_id"]
    operations = [
        {
            "type": "replace_nearby",
            "target": "bixby_creek_bridge_viewpoint",
            "parameters": {"candidate_id": "pfeiffer_big_sur_state_park"},
        },
        {
            "type": "set_stop_duration",
            "target": "griffith_observatory",
            "parameters": {
                "duration": {"mode": "exact", "preferred_minutes": 60, "minimum_minutes": 60, "maximum_minutes": 60}
            },
        },
        {
            "type": "set_stop_time_window",
            "target": TARGET,
            "parameters": {"earliest_arrival": "10:00", "latest_departure": None},
        },
        {"type": "set_stop_role", "target": TARGET, "parameters": {"role": "meal"}},
    ]
    parent_id = original
    for index, operation in enumerate(operations):
        with client() as api:
            restored = request(api, "GET", path)
            session = restored["session"]
            assert session["accepted_plan_id"] == parent_id
            staged = request(
                api,
                "POST",
                path + "/draft/operations",
                {"expected_revision": session["revision"], "source": "typed_stop_editor", **operation},
            )
            preview = request(api, "POST", path + "/preview", {"expected_revision": staged["session"]["revision"]})
            proposal = preview["proposal"]
            assert proposal["state"] == "eligible" and proposal["acceptance_eligible"] is True
            assert proposal["parent_plan_id"] == parent_id
            certificate = (
                proposal.get("runtime_execution", {}).get("evaluation_certificate")
                or proposal.get("certificate_schedule_evidence", {}).get("certificate_record")
                or proposal.get("certificate_role_evidence", {}).get("certificate_record")
            )
            assert certificate["plan_id"] == proposal["child_plan_id"]
            assert certificate["plan_content_hash"] == proposal["child_plan_content_hash"]
            assert certificate["content_hash"] == proposal["certificate_content_hash"]
            assert (
                stable_content_hash({key: value for key, value in certificate.items() if key != "content_hash"})
                == certificate["content_hash"]
            )
            if operation["type"] == "set_stop_role":
                service = api.app.state.product_service
                active = service.workspace.get(session["session_id"])
                compiler = service.compiler_for_session(active)
                fresh = compiler.compile(active.draft, accepted_plan_id=active.accepted_plan_id).certificate
                assert any(isinstance(warning["evidence_refs"], tuple) for warning in fresh["warnings"])
                assert _replay_role_certificate_with_persisted_timestamp(fresh, proposal) == certificate
                forged = deepcopy(fresh)
                forged["warnings"][0]["evidence_refs"] = ("forged_stop",)
                with pytest.raises(WorkspaceError, match="draft_role_proposal_invalid"):
                    _replay_role_certificate_with_persisted_timestamp(forged, proposal)
            accepted = request(
                api,
                "POST",
                path + "/accept",
                {
                    "expected_revision": preview["session"]["revision"],
                    "confirmation": "accept_evaluated_repair",
                    "idempotency_key": f"role-chain-accept-{index:04d}",
                },
            )
            decision = accepted["result"]["decision"]
            assert decision["selected_plan_id"] == proposal["child_plan_id"]
            assert decision["certificate_content_hash"] == certificate["content_hash"]
            service = api.app.state.product_service
            snapshot = service.acceptance.load_snapshot(decision["decision_id"])
            assert snapshot["certificate"] == certificate
            assert snapshot["plan"]["parent_plan_id"] == parent_id
            for decision_id, prior_snapshot in snapshots.items():
                assert service.acceptance.load_snapshot(decision_id) == prior_snapshot
            snapshots[decision["decision_id"]] = deepcopy(snapshot)
            parent_id = proposal["child_plan_id"]
            plan_ids.append(parent_id)
        with client() as api:
            restored = request(api, "GET", path)
            assert restored["session"]["accepted_plan_id"] == parent_id
            plan = next(row for row in restored["workspace"]["geography"]["plans"] if row["plan_id"] == parent_id)
            assert plan["content_hash"] == proposal["child_plan_content_hash"]
            assert {row["properties"]["stop_id"] for row in plan["stops"]["features"]} == {
                row["stop_id"] for row in snapshot["plan"]["selected_stops"]
            }
            assert {"golden_gate_bridge", "pfeiffer_big_sur_state_park", TARGET} <= {
                row["properties"]["stop_id"] for row in plan["stops"]["features"]
            }
            if operation["type"] == "set_stop_role":
                target = next(
                    row["properties"] for row in plan["stops"]["features"] if row["properties"]["stop_id"] == TARGET
                )
                assert target["itinerary_role"] == "meal"
                assert [
                    row["selected_plan_id"]
                    for row in api.app.state.product_service.acceptance.history(session["trip_id"])
                ] == plan_ids
    assert len(set([original, *plan_ids])) == 5
