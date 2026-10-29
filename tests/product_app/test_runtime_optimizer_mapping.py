"""Real compiler, heuristic, evaluator, geography, and persistence connectivity."""

import json
import os
import subprocess
from copy import deepcopy
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from itinerary_system.product_app.api import create_product_app
from itinerary_system.product_app.copilot import DeterministicCopilotAdapter
from itinerary_system.product_app.draft_compiler import FrozenDraftCompiler
from itinerary_system.product_app.product_demo import load_product_demo_package
from itinerary_system.product_app.registry import ProductRunRegistry
from itinerary_system.product_app.service import ProductService
from itinerary_system.product_app.workspace import WorkspaceError
from itinerary_system.research_artifacts import stable_content_hash

ROOT = Path(__file__).resolve().parents[2]
RUN_ID = "california_coast_coherent_demo_v1"
TARGET = "bixby_creek_bridge_viewpoint"


@pytest.mark.parametrize(
    "attribute_families",
    [
        ("duration", "time_window", "role"),
        ("duration", "time_window"),
        ("duration", "role"),
        ("time_window", "role"),
    ],
)
def test_atomic_combined_api_preview_acceptance_and_restart_preserve_complete_child(
    tmp_path, monkeypatch, attribute_families
):
    """Real API/internal boundaries, not proof that the browser enables the draft yet."""
    from itinerary_system.product_app.models import ComponentHealthV1
    from itinerary_system.product_app.runtime import ProductRuntime

    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("PRODUCT_COPILOT_ADAPTER", "deterministic")
    # Map readiness requires a browser only in the separate runtime gate.
    monkeypatch.setattr(
        ProductRuntime, "_probe_map", lambda self: ComponentHealthV1("map", "ready", False, "maplibre_ready")
    )
    state_root = tmp_path / "atomic-state"
    exchanges, authentication = [], {}

    def client():
        return TestClient(
            create_product_app(
                repository_root=ROOT,
                registry_path=ROOT / "configs/product_app_registry.json",
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
        (tmp_path / "exact_atomic_combined_api_io.json").write_text(
            json.dumps(exchanges, sort_keys=True), encoding="utf-8"
        )
        assert response.status_code == 200, response.text
        return output

    with client() as api:
        created = request(api, "POST", "/api/sessions", {"run_id": RUN_ID})
        session = created["session"]
        authentication["X-Session-Token"] = created["mutation_token"]
        path = f"/api/sessions/{session['session_id']}"
        parent_id = session["accepted_plan_id"]
        operations = [
            replacement("pfeiffer_big_sur_state_park"),
            {
                "type": "set_stop_duration",
                "target": "griffith_observatory",
                "parameters": {
                    "duration": {"mode": "exact", "preferred_minutes": 90, "minimum_minutes": 90, "maximum_minutes": 90}
                },
            },
            {
                "type": "set_stop_time_window",
                "target": "golden_gate_bridge",
                "parameters": {"earliest_arrival": "10:00", "latest_departure": None},
            },
            {"type": "set_stop_role", "target": "golden_gate_bridge", "parameters": {"role": "scenic_stop"}},
            {
                "type": "set_stop_order",
                "target": "sequoia_national_park",
                "parameters": {"day": 4, "sequence_index": 1},
            },
        ]
        operations = [
            operation
            for operation in operations
            if not operation["type"].startswith("set_stop_")
            or operation["type"] == "set_stop_order"
            or operation["type"].removeprefix("set_stop_") in attribute_families
        ]
        for operation in operations:
            staged = request(
                api,
                "POST",
                path + "/draft/operations",
                {"expected_revision": session["revision"], "source": "typed_stop_editor", **operation},
            )
            session = staged["session"]
            assert session["accepted_plan_id"] == parent_id
        impact = request(api, "POST", path + "/draft/impact-preview", {"expected_revision": session["revision"]})
        preview = request(api, "POST", path + "/preview", {"expected_revision": session["revision"]})
        proposal = preview["proposal"]
        assert proposal["state"] == "eligible"
        assert proposal["execution_mode"] == "atomic_combined_independent_evaluation"
        assert proposal["provenance"] == "independent_evaluated_combined_edit"
        receipt = proposal["runtime_execution"]
        child, certificate, diff = receipt["child_plan"], receipt["evaluation_certificate"], receipt["plan_diff"]
        assert len(receipt["operations"]) == len(proposal["compiled_request"]["operations"]) == len(operations)
        assert child["parent_plan_id"] == parent_id
        assert (
            child["plan_id"] == certificate["plan_id"] == diff["child_plan_id"] == proposal["geography_plan"]["plan_id"]
        )
        assert child["content_hash"] == certificate["plan_content_hash"] == proposal["geography_plan"]["content_hash"]
        assert (
            receipt["optimizer_execution"]["planner_run"]["method_executed"] == "deterministic_context_aware_heuristic"
        )
        assert diff["schema_version"] == "plan-diff-v6"
        assert all(diff[name + "_changes"] for name in attribute_families)
        assert diff["reorder_changes"]
        assert all(
            name + "_changes" not in diff for name in {"duration", "time_window", "role"} - set(attribute_families)
        )
        assert proposal["geography_plan"]["coverage"]["complete"] is True
        assert preview["session"]["accepted_plan_id"] == parent_id
        contract_path = tmp_path / "exact_combined_frontend_contract.json"
        contract_path.write_text(
            json.dumps(
                {
                    "workspace": created["workspace"],
                    "before_session": session,
                    "impact": impact,
                    "preview": preview,
                    "expected": {
                        "session_id": session["session_id"],
                        "trip_id": session["trip_id"],
                        "run_id": session["run_id"],
                        "revision": session["revision"],
                        "accepted_plan_id": parent_id,
                        "parent_plan_content_hash": proposal["parent_plan_content_hash"],
                        "draft": session["draft"],
                        "permission_decisions": session["permission_decisions"],
                        "conversation_id": session["conversation_id"],
                    },
                }
            ),
            encoding="utf-8",
        )
        environment = {**os.environ, "ACTUAL_COMBINED_PREVIEW_CONTRACT_PATH": str(contract_path)}
        node = subprocess.run(
            ["node", str(Path(__file__).with_name("combined_preview_frontend_contract.test.mjs"))],
            cwd=ROOT,
            env=environment,
            text=True,
            encoding="utf-8",
            capture_output=True,
            check=False,
        )
        (tmp_path / "exact_node_contract_io.json").write_text(
            json.dumps(
                {
                    "command": node.args,
                    "input_file": str(contract_path),
                    "exit_code": node.returncode,
                    "stdout": node.stdout,
                    "stderr": node.stderr,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        assert node.returncode == 0, node.stdout + node.stderr
        service = api.app.state.product_service
        active = service.workspace.get(session["session_id"])
        fresh_certificate = (
            service.compiler_for_session(active)
            .compile(active.draft, accepted_plan_id=active.accepted_plan_id)
            .certificate
        )
        ignored = {"content_hash", "evaluated_at"}
        fresh_semantics = {key: value for key, value in fresh_certificate.items() if key not in ignored}
        saved_semantics = {key: value for key, value in certificate.items() if key not in ignored}
        representation = {
            "fresh_certificate": fresh_certificate,
            "saved_certificate": certificate,
            "raw_semantics_equal": fresh_semantics == saved_semantics,
            "json_normalized_semantics_equal": json.loads(json.dumps(fresh_semantics)) == saved_semantics,
            "tuple_evidence_reference_count": sum(
                isinstance(row.get("evidence_refs"), tuple) for row in fresh_certificate["warnings"]
            ),
        }
        (tmp_path / "exact_certificate_roundtrip_io.json").write_text(
            json.dumps(representation, indent=2), encoding="utf-8"
        )
        assert representation["json_normalized_semantics_equal"] is True
        # Adversarial local-state injection, separate from the HTTP journey:
        # valid JSON and a recomputed hash must not conceal changed evidence.
        original_proposal = deepcopy(active.proposal)
        forged = deepcopy(original_proposal)
        forged_certificate = forged["runtime_execution"]["evaluation_certificate"]
        forged_certificate["warnings"][0]["evidence_refs"] = ["forged_stop"]
        forged_certificate["content_hash"] = stable_content_hash(
            {key: value for key, value in forged_certificate.items() if key != "content_hash"}
        )
        forged["runtime_execution"]["evaluation_content_hash"] = forged_certificate["content_hash"]
        forged["certificate_content_hash"] = forged_certificate["content_hash"]
        forged["certificate_identity"]["content_hash"] = forged_certificate["content_hash"]
        active.proposal = forged
        try:
            with pytest.raises(WorkspaceError, match="acceptance_proposal_invalid") as rejection:
                service.accept_evaluated(
                    active,
                    expected_revision=active.revision,
                    confirmation="accept_evaluated_repair",
                    idempotency_key="atomic-forged-0001",
                    mutation_token=created["mutation_token"],
                )
            pointer_after_rejection = service.acceptance.load_pointer(active.trip_id)
            assert pointer_after_rejection["accepted_plan_id"] == parent_id
            (tmp_path / "exact_forged_certificate_io.json").write_text(
                json.dumps(
                    {
                        "scope": "Adversarial local-state injection into the real acceptance service, not an HTTP request",
                        "input": {
                            "certificate": forged_certificate,
                            "other_proposal_fields": "Same preview except certificate hash bindings",
                        },
                        "output": {
                            "error_code": rejection.value.code,
                            "status_code": rejection.value.status_code,
                            "accepted_plan_id": pointer_after_rejection["accepted_plan_id"],
                        },
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        finally:
            active.proposal = original_proposal
        accepted = request(
            api,
            "POST",
            path + "/accept",
            {
                "expected_revision": preview["session"]["revision"],
                "confirmation": "accept_evaluated_repair",
                "idempotency_key": "atomic-combined-0001",
            },
        )
        decision = accepted["result"]["decision"]
        snapshot = api.app.state.product_service.acceptance.load_snapshot(decision["decision_id"])
        assert snapshot["plan"] == child and snapshot["certificate"] == certificate and snapshot["diff"] == diff
    with client() as api:
        restored = request(api, "GET", path)
        assert restored["session"]["accepted_plan_id"] == child["plan_id"]
        plan = next(plan for plan in restored["workspace"]["geography"]["plans"] if plan["plan_id"] == child["plan_id"])
        assert plan["content_hash"] == child["content_hash"]
        assert {feature["properties"]["stop_id"] for feature in plan["stops"]["features"]} == set(child["sequence"])
        assert next(day["stop_ids"] for day in child["ordered_days"] if day["day"] == 4) == [
            "kings_canyon_national_park",
            "sequoia_national_park",
        ]


def replacement(candidate: str) -> dict:
    return {
        "type": "replace_nearby",
        "target": TARGET,
        "parameters": {"candidate_id": candidate},
        "source": "copilot",
    }


@pytest.fixture(scope="module")
def package():
    return load_product_demo_package(ROOT, ROOT / "runs/california-coast-coherent-demo-v1")


@pytest.mark.parametrize(
    "candidate",
    [
        "pfeiffer_big_sur_state_park",
        "point_lobos_state_natural_reserve",
    ],
)
def test_product_preview_runs_optimizer_and_evaluator_for_actual_request(
    tmp_path: Path,
    candidate: str,
) -> None:
    service = ProductService(ProductRunRegistry(ROOT, ROOT / "configs/product_app_registry.json"), tmp_path / "state")
    session, token, _ = service.create_session(RUN_ID)
    compiler = service._draft_compilers[RUN_ID]
    parent = deepcopy(compiler.parent_plan)
    service.workspace.add_operation(
        session,
        {"expected_revision": 0, **replacement(candidate)},
        valid_stop_ids=service.valid_stops(RUN_ID),
        day_count=7,
    )
    proposal = service.preview(session.session_id, 1)
    receipt = proposal["runtime_execution"]
    child = receipt["child_plan"]
    certificate = receipt["evaluation_certificate"]
    request = receipt["repair_request"]
    assert proposal["state"] == "eligible"
    assert proposal["execution_mode"] == "deterministic_repair_pipeline"
    assert receipt["planner_run"]["execution_status"] == "COMPLETED"
    assert receipt["planner_run"]["method_executed"] == "deterministic_context_aware_heuristic"
    assert receipt["planner_run"]["result_plan_id"] == child["plan_id"]
    assert request["request_id"] == child["planning_request_id"] == receipt["request_id"]
    assert receipt["request_content_hash"] == stable_content_hash(request)
    assert request["confirmed_constraints"]["must_include"] == [candidate]
    assert request["confirmed_constraints"]["must_delete"] == [TARGET]
    assert request["confirmed_constraints"]["affected_days"] == [5]
    assert receipt["evaluation_executed"] is True
    matrix, _, _ = compiler._runtime_inputs()
    assert (TARGET, candidate) not in matrix.cells
    assert certificate["plan_id"] == child["plan_id"] == proposal["geography_plan"]["plan_id"]
    assert certificate["plan_content_hash"] == child["content_hash"]
    assert (
        datetime.fromisoformat(receipt["started_at"])
        <= datetime.fromisoformat(certificate["evaluated_at"])
        <= datetime.fromisoformat(receipt["completed_at"])
    )
    assert child["plan_id"] not in service._evidence_bundles[RUN_ID]
    assert child["parent_plan_id"] == parent["plan_id"]
    assert set(child["sequence"]) == (set(parent["sequence"]) - {TARGET}) | {candidate}
    assert child["lodging_assignments"] == parent["lodging_assignments"]
    assert {row["day"]: row["stop_ids"] for row in child["ordered_days"] if row["day"] != 5} == {
        row["day"]: row["stop_ids"] for row in parent["ordered_days"] if row["day"] != 5
    }
    assert proposal["geography_plan"]["coverage"]["complete"] is True
    assert proposal["geography_plan"]["coverage"]["road_validated_leg_count"] == 15
    assert proposal["geography_plan"]["coverage"]["gap_count"] == 0
    for collection in ("validated_legs", "gaps"):
        for feature in proposal["geography_plan"][collection]["features"]:
            properties = feature["properties"]
            assert properties["requirement_id"] == f"{child['plan_id']}:required-leg:{properties['leg_index']}"
            assert properties["evidence_scope"] == "runtime_optimizer_route_leg"
    selected = [row for row in receipt["planner_trace"]["decision_records"] if row["selected"]]
    assert [row["stop_sequence"] for row in selected] == [[candidate]]
    assert compiler.parent_plan == parent
    assert service.workspace.get(session.session_id).accepted_plan_id == parent["plan_id"]
    (tmp_path / f"{candidate}_exact_preview.json").write_text(json.dumps(proposal, indent=2), encoding="utf-8")
    # Persistence round-trips the actual receipt, then acceptance independently
    # reruns evaluation and retains the exact preview certificate inspected.
    assert service.workspace.get(session.session_id).proposal["runtime_execution"] == json.loads(json.dumps(receipt))
    service.accept_evaluated(
        session,
        expected_revision=2,
        confirmation="accept_evaluated_repair",
        idempotency_key="runtime-repair-test",
        mutation_token=token,
    )
    restored = ProductService(ProductRunRegistry(ROOT, ROOT / "configs/product_app_registry.json"), tmp_path / "state")
    new_session, _, _ = restored.create_session(RUN_ID)
    assert new_session.accepted_plan_id == child["plan_id"]
    accepted = next(
        plan
        for plan in restored.workspace_view_for_session(new_session)["geography"]["plans"]
        if plan["plan_id"] == child["plan_id"]
    )
    for feature in accepted["validated_legs"]["features"]:
        properties = feature["properties"]
        assert properties["requirement_id"] == f"{child['plan_id']}:required-leg:{properties['leg_index']}"
        assert properties["evidence_scope"] == "runtime_optimizer_route_leg"


def test_registered_child_certificate_cannot_supply_runtime_result(package) -> None:
    bundles = {
        key: replace(bundle, certificate={**bundle.certificate, "plan_content_hash": "poisoned_registered_certificate"})
        for key, bundle in package.evidence_bundles.items()
    }
    compiler = FrozenDraftCompiler(package.primary_bundle.parent_plan, bundles, repository_root=ROOT)
    first = compiler.compile([replacement("pfeiffer_big_sur_state_park")], accepted_plan_id=compiler.parent_plan_id)
    second = compiler.compile(
        [replacement("point_lobos_state_natural_reserve")], accepted_plan_id=compiler.parent_plan_id
    )
    assert first.state == second.state == "eligible"
    assert first.source_request_id != second.source_request_id
    assert first.child_plan["sequence"] != second.child_plan["sequence"]
    assert first.certificate["certificate_id"] != second.certificate["certificate_id"]
    assert first.runtime_execution["execution_id"] != second.runtime_execution["execution_id"]


def test_missing_required_route_fails_actual_optimizer_without_fallback(package) -> None:
    bundles = {}
    for key, bundle in package.evidence_bundles.items():
        matrix = deepcopy(bundle.route_matrix)
        matrix["cells"] = [row for row in matrix["cells"] if row["destination_id"] != "pfeiffer_big_sur_state_park"]
        bundles[key] = replace(bundle, route_matrix=matrix)
    compiler = FrozenDraftCompiler(package.primary_bundle.parent_plan, bundles, repository_root=ROOT)
    result = compiler.compile([replacement("pfeiffer_big_sur_state_park")], accepted_plan_id=compiler.parent_plan_id)
    assert result.state == "blocked"
    assert result.reason == "blocked_missing_route_evidence"
    assert result.child_plan is None
    assert result.runtime_execution["evaluation_executed"] is False
    assert result.runtime_execution["planner_run"]["execution_status"] == "FAILED"
    assert any(
        "missing route matrix cell" in value for value in result.runtime_execution["planner_trace"]["failure_reasons"]
    )


@pytest.mark.parametrize(
    "kind,parameters",
    [
        ("mark_flexible", {}),
        ("lock_stop", {}),
        ("move_day", {"day": 6}),
        ("route_feedback", {"preference": "reduce_driving"}),
    ],
)
def test_unmapped_product_operations_are_disabled_and_cannot_trigger_demo_repair(package, kind, parameters):
    compiler = FrozenDraftCompiler(package.primary_bundle.parent_plan, package.evidence_bundles, repository_root=ROOT)
    assert compiler.capabilities()["operations"][kind]["preview_executable"] is False
    with pytest.raises(WorkspaceError, match="draft_optimizer_mapping_unsupported"):
        compiler.compile(
            [
                {
                    "type": kind,
                    "target": "selected_route" if kind == "route_feedback" else TARGET,
                    "parameters": parameters,
                }
            ],
            accepted_plan_id=compiler.parent_plan_id,
        )


def test_accepted_runtime_session_restores_same_identity_selection_and_conversation(tmp_path: Path):
    state_root = tmp_path / "api-state"

    def client_for_state():
        return TestClient(
            create_product_app(
                repository_root=ROOT,
                registry_path=ROOT / "configs/product_app_registry.json",
                state_root=state_root,
                additional_allowed_authorities=("testserver",),
                copilot=DeterministicCopilotAdapter(),
            )
        )

    with client_for_state() as client:
        created = client.post("/api/sessions", json={"run_id": RUN_ID})
        assert created.status_code == 200, created.text
        payload = created.json()
        session = payload["session"]
        session_url = f"/api/sessions/{session['session_id']}"
        headers = {"X-Session-Token": payload["mutation_token"]}
        selected = client.post(
            f"{session_url}/selection",
            headers=headers,
            json={
                "expected_revision": session["revision"],
                "selected_day": 5,
                "selected_stop_id": TARGET,
            },
        )
        assert selected.status_code == 200, selected.text
        session = selected.json()["session"]
        sent = client.post(
            f"{session_url}/copilot/messages",
            headers=headers,
            json={
                "expected_revision": session["revision"],
                "client_message_id": "client_message_1234567890abcdef1234567890abcdef",
                "message": "Can you explain what you can help me with?",
            },
        )
        assert sent.status_code == 200, sent.text
        restored_before = client.get(session_url, headers=headers)
        assert restored_before.status_code == 200, restored_before.text
        session = restored_before.json()["session"]
        conversation_before = client.get(f"{session_url}/conversation", headers=headers).json()["conversation"]
        staged = client.post(
            f"{session_url}/draft/operations",
            headers=headers,
            json={
                "expected_revision": session["revision"],
                **replacement("pfeiffer_big_sur_state_park"),
            },
        )
        assert staged.status_code == 200, staged.text
        preview = client.post(
            f"{session_url}/preview",
            headers=headers,
            json={
                "expected_revision": staged.json()["session"]["revision"],
            },
        )
        assert preview.status_code == 200, preview.text
        preview_session = preview.json()["session"]
        child_id = preview.json()["proposal"]["child_plan_id"]
        assert preview_session["selected_stop_id"] == "pfeiffer_big_sur_state_park"
        accepted = client.post(
            f"{session_url}/accept",
            headers=headers,
            json={
                "expected_revision": preview_session["revision"],
                "confirmation": "accept_evaluated_repair",
                "idempotency_key": "runtime-same-session-restore",
            },
        )
        assert accepted.status_code == 200, accepted.text
        accepted_session = accepted.json()["session"]
        assert accepted_session["draft"] == []
        assert accepted_session["proposal"] is None

    # Authenticate the SAME stored session after a full API restart. Creating a
    # new session that reads the accepted pointer would miss this regression.
    with client_for_state() as restarted:
        unauthorized = restarted.get(session_url, headers={"X-Session-Token": "wrong"})
        assert unauthorized.status_code == 403
        restored = restarted.get(session_url, headers=headers)
        assert restored.status_code == 200, restored.text
        session = restored.json()["session"]
        assert session["session_id"] == accepted_session["session_id"]
        assert session["accepted_plan_id"] == child_id
        assert session["revision"] == accepted_session["revision"]
        assert session["selected_stop_id"] == "pfeiffer_big_sur_state_park"
        assert session["selected_day"] == 5
        assert session["conversation_id"] == conversation_before["conversation_id"]
        impact = restarted.post(
            f"{session_url}/draft/impact-preview",
            headers=headers,
            json={
                "expected_revision": session["revision"],
            },
        )
        assert impact.status_code == 200, impact.text
        assert impact.json()["parent_plan_id"] == child_id
        assert impact.json()["parent_plan_content_hash"] == preview.json()["proposal"]["child_plan_content_hash"]
        assert impact.json()["summary"]["operation_count"] == 0
        assert impact.json()["summary"]["can_run_evaluated_preview"] is False
        conversation_after = restarted.get(f"{session_url}/conversation", headers=headers)
        assert conversation_after.status_code == 200, conversation_after.text
        assert conversation_after.json()["conversation"] == conversation_before
        assert len(list((state_root / "sessions").glob("session_*.json"))) == 1
