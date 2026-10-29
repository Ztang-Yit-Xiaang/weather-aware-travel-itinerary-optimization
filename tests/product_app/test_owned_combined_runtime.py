"""Owned mixed edits through the real authenticated API and internal stack."""

from __future__ import annotations

import json
import os
import subprocess
from copy import deepcopy
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from owned_product_fixture import ROOT, RUN_ID, build_owned_product_registry

from itinerary_system.product_app.api import create_product_app
from itinerary_system.product_app.draft_compiler import _plan_artifact, _verify_diff
from itinerary_system.product_app.models import ComponentHealthV1
from itinerary_system.product_app.runtime import ProductRuntime
from itinerary_system.product_app.workspace import WorkspaceError
from itinerary_system.research_artifacts import stable_content_hash


@pytest.mark.parametrize("families,with_selection", [
    (("duration", "time_window", "role"), True), (("duration", "time_window"), True),
    (("duration", "role"), True), (("time_window", "role"), True),
    (("duration",), True), (("time_window",), True),
    pytest.param(("duration",), False, id="standalone-duration"),
    pytest.param(("time_window",), False, id="standalone-window"),
])
@pytest.mark.parametrize("protected", [False, True])
def test_owned_combined_api_keeps_complete_child_or_blocks_acceptance(tmp_path: Path, monkeypatch, families, with_selection, protected):
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("PRODUCT_COPILOT_ADAPTER", "deterministic")
    # Only the separate browser gate proves actual map readiness.
    monkeypatch.setattr(ProductRuntime, "_probe_map", lambda self: ComponentHealthV1("map", "ready", False, "maplibre_ready"))
    registry = build_owned_product_registry(tmp_path, protected_attribute=families[0] if protected else None)
    state_root = tmp_path / "state"
    exchanges, authentication = [], {}

    def client():
        return TestClient(create_product_app(repository_root=ROOT, registry_path=registry,
            state_root=state_root, additional_allowed_authorities=("testserver",)))

    def request(api, method, path, body=None, status=200):
        response = api.request(method, path, json=body, headers=authentication)
        result = response.json()
        exchanges.append({"method": method, "path": path, "input": body, "status": response.status_code,
                          "output": {key: value for key, value in result.items() if key != "mutation_token"}})
        (tmp_path / "exact_owned_api_io.json").write_text(json.dumps(exchanges, indent=2), encoding="utf-8")
        assert response.status_code == status, response.text
        return result

    with client() as api:
        created = request(api, "POST", "/api/sessions", {"run_id": RUN_ID})
        session = created["session"]
        authentication["X-Session-Token"] = created["mutation_token"]
        path = f"/api/sessions/{session['session_id']}"
        if not with_selection:
            # Resolve the demo's weather disruption through the real pipeline
            # first; a schedule-only edit must not bypass unrelated infeasibility.
            staged = request(api, "POST", path + "/draft/operations", {
                "expected_revision": session["revision"], "source": "typed_stop_editor",
                "type": "replace_nearby", "target": "bixby_creek_bridge_viewpoint",
                "parameters": {"candidate_id": "pfeiffer_big_sur_state_park"}})
            baseline = request(api, "POST", path + "/preview", {"expected_revision": staged["session"]["revision"]})
            assert baseline["proposal"]["state"] == "eligible"
            request(api, "POST", path + "/accept", {"expected_revision": baseline["session"]["revision"],
                "confirmation": "accept_evaluated_repair", "idempotency_key": "owned-baseline-0001"})
            created = request(api, "GET", path)
            session = created["session"]
        parent_id = session["accepted_plan_id"]
        parent_geography = next(plan for plan in created["workspace"]["geography"]["plans"] if plan["plan_id"] == parent_id)
        parent_stop_ids = {feature["properties"]["stop_id"] for feature in parent_geography["stops"]["features"]}
        operations = [
            {"type": "replace_nearby", "target": "bixby_creek_bridge_viewpoint", "parameters": {"candidate_id": "pfeiffer_big_sur_state_park"}},
            {"type": "set_stop_duration", "target": "griffith_observatory", "parameters": {"duration": {
                "mode": "exact", "preferred_minutes": 90, "minimum_minutes": 90, "maximum_minutes": 90}}},
            {"type": "set_stop_time_window", "target": "golden_gate_bridge", "parameters": {"earliest_arrival": "10:00", "latest_departure": None}},
            {"type": "set_stop_role", "target": "golden_gate_bridge", "parameters": {"role": "scenic_stop"}},
        ]
        operations = [operation for operation in operations if (with_selection and operation["type"] == "replace_nearby")
                      or operation["type"].removeprefix("set_stop_") in families]
        for operation in operations:
            staged = request(api, "POST", path + "/draft/operations", {
                "expected_revision": session["revision"], "source": "typed_stop_editor", **operation})
            session = staged["session"]
        impact = request(api, "POST", path + "/draft/impact-preview", {"expected_revision": session["revision"]})
        preview = request(api, "POST", path + "/preview", {"expected_revision": session["revision"]})
        proposal = preview["proposal"]
        receipt = proposal["runtime_execution"]
        child, diff, certificate = receipt["child_plan"], receipt["plan_diff"], receipt["evaluation_certificate"]
        assert diff["schema_version"] == ("plan-diff-v7" if len(families) > 1 else "plan-diff-v5")
        assert certificate["evaluator_version"] == "verify-001-plan-evaluator-v3"
        assert child["plan_id"] == diff["child_plan_id"] == certificate["plan_id"]
        assert child["content_hash"] == certificate["plan_content_hash"]
        assert diff["attribute_ownership_evidence"]["parent_plan_content_hash"] == proposal["parent_plan_content_hash"]
        assert {row["attribute"] for row in diff["attribute_ownership_evidence"]["constraints"]} == set(families)
        assert set(child["sequence"]) == ((parent_stop_ids - {"bixby_creek_bridge_viewpoint"}) | {"pfeiffer_big_sur_state_park"}
                                         if with_selection else parent_stop_ids)
        assert proposal["state"] == ("ineligible" if protected else "eligible")
        assert preview["session"]["accepted_plan_id"] == parent_id
        compiler = api.app.state.product_service.compiler_for_session(
            api.app.state.product_service.workspace.get(session["session_id"]))
        rejected = []
        for name in ("bare_cost_evidence", "foreign_parent", "wrong_evaluator", "altered_cost", "removed_constraint"):
            altered, cert, child_record = deepcopy(diff), deepcopy(certificate), deepcopy(child)
            parent = compiler._parent_artifact
            if name == "foreign_parent":
                altered["attribute_ownership_evidence"]["parent_plan_content_hash"] = "1111111111111111"
            elif name == "wrong_evaluator":
                cert["evaluator_version"] = "verify-001-plan-evaluator-v1"
            elif name == "altered_cost":
                altered["weighted_edit_cost"] = 0
            elif name == "removed_constraint":
                child_record["owned_constraints"] = []
                child_record["content_hash"] = stable_content_hash({key: value for key, value in child_record.items() if key != "content_hash"})
                cert["plan_content_hash"] = child_record["content_hash"]
                cert["certificate_id"] = "cert_" + stable_content_hash({
                    "plan_id": cert["plan_id"], "plan_content_hash": cert["plan_content_hash"],
                    "source_run_id": cert["source_run_id"],
                    "failures": [row["code"] for row in cert["failures"]],
                    "warnings": [row["code"] for row in cert["warnings"]],
                })
                cert["evaluation_id"] = cert["certificate_id"]
            cert["content_hash"] = stable_content_hash({key: value for key, value in cert.items() if key != "content_hash"})
            inputs = {"diff": altered} if name == "bare_cost_evidence" else {
                "diff": altered, "parent": parent, "child": _plan_artifact(child_record),
                "certificate": cert, "default_visit_minutes": compiler._runtime_inputs()[1].default_visit_minutes}
            with pytest.raises(WorkspaceError, match="draft_pipeline_diff_invalid") as error:
                _verify_diff(**inputs)
            rejected.append({"case": name, "input": {**inputs,
                **({"parent": parent.to_record(), "child": inputs["child"].to_record()} if "parent" in inputs else {})},
                "output": {"code": error.value.code, "status_code": error.value.status_code}})
        (tmp_path / "exact_owned_boundary_io.json").write_text(json.dumps(rejected, indent=2), encoding="utf-8")
        contract_path = tmp_path / "exact_owned_frontend_contract.json"
        contract_path.write_text(json.dumps({
            "workspace": created["workspace"], "before_session": session, "impact": impact, "preview": preview,
            "expected": {key: session[key] for key in ("session_id", "trip_id", "run_id", "revision",
                "accepted_plan_id", "draft", "permission_decisions", "conversation_id")} | {
                "parent_plan_content_hash": proposal["parent_plan_content_hash"]},
        }), encoding="utf-8")
        node = subprocess.run(["node", str(Path(__file__).with_name("combined_preview_frontend_contract.test.mjs"))],
            cwd=ROOT, env={**os.environ, "ACTUAL_COMBINED_PREVIEW_CONTRACT_PATH": str(contract_path)},
            text=True, encoding="utf-8", capture_output=True, check=False)
        (tmp_path / "exact_node_owned_contract_io.json").write_text(json.dumps({
            "command": node.args, "input_file": str(contract_path), "exit_code": node.returncode,
            "stdout": node.stdout, "stderr": node.stderr,
        }, indent=2), encoding="utf-8")
        assert node.returncode == 0, node.stdout + node.stderr
        if protected:
            assert "owned_attribute_constraint_unsatisfied" in {row["code"] for row in certificate["failures"]}
            request(api, "POST", path + "/accept", {"expected_revision": preview["session"]["revision"],
                "confirmation": "accept_evaluated_repair", "idempotency_key": "owned-blocked-0001"}, status=409)
            restored = request(api, "GET", path)
            assert restored["session"]["accepted_plan_id"] == parent_id
            assert len(restored["session"]["draft"]) == len(operations)
        else:
            assert proposal["geography_plan"]["plan_id"] == child["plan_id"]
            accepted = request(api, "POST", path + "/accept", {"expected_revision": preview["session"]["revision"],
                "confirmation": "accept_evaluated_repair", "idempotency_key": "owned-eligible-0001"})
            snapshot = api.app.state.product_service.acceptance.load_snapshot(accepted["result"]["decision"]["decision_id"])
            assert snapshot["plan"] == child and snapshot["diff"] == diff and snapshot["certificate"] == certificate
    with client() as api:
        restored = request(api, "GET", path)
        assert restored["session"]["accepted_plan_id"] == (parent_id if protected else child["plan_id"])
