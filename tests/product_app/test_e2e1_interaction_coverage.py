"""Synthetic coverage-projection tests; no browser, server, router or provider calls."""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import e2e1_interaction_coverage as coverage  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def _registry():
    return json.loads((ROOT / "configs/e2e1_interaction_coverage.json").read_text(encoding="utf-8"))


def _snapshot(context="accepted_child"):
    return {
        "context": context,
        "session_id": "session_unit",
        "plan_id": "parent_unit",
        "plan_content_hash": "parent_hash_unit",
        "artifact": "unit_fixture.json",
        "capabilities": {
            "typed_edit_capabilities": {
                "operations": {
                    "set_stop_duration": {
                        "enabled": True,
                        "preview_executable": True,
                        "feedback_tier": "conditional",
                        "supported_evaluated_modes": ["exact"],
                        "draft_only_modes": ["preferred", "minimum", "maximum", "range"],
                    }
                }
            }
        },
    }


def _fixture(tmp_path, *, context="accepted_child"):
    registry = _registry()
    mapping = next(row for row in registry["rows"] if row["operation"] == "set_stop_duration")
    registry.update(rows=[mapping], unverified_requirements=[], required_contexts=[context])
    journey = mapping["journeys"][0]
    base = tmp_path / "suites" / journey["browser_mode"] / journey["case_directory"]
    base.mkdir(parents=True)
    for name in journey["evidence"]:
        (base / name).write_text(json.dumps({"assertions_passed": True}), encoding="utf-8")
    operation = {
        "type": "set_stop_duration",
        "target": "griffith_observatory",
        "parameters": {"duration": {"mode": "exact", "preferred_minutes": 60}},
        "source": "map",
    }
    stored = {**operation, "operation_id": "operation_unit"}
    session = {"session_id": "session_unit", "accepted_plan_id": "parent_unit", "draft": [stored]}
    stage = {
        "schema_version": "e2e1-product-http-io-v1",
        "method": "POST",
        "response_status_code": 200,
        "url": "http://127.0.0.1:8127/api/sessions/session_unit/draft/operations",
        "request_body_utf8": json.dumps(operation),
        "response_body_utf8": json.dumps({"operation": stored, "session": session}),
    }
    preview = {
        **stage,
        "url": "http://127.0.0.1:8127/api/sessions/session_unit/preview",
        "request_body_utf8": '{"expected_revision": 1}',
        "response_body_utf8": json.dumps(
            {
                "session": session,
                "proposal": {
                    "state": "eligible",
                    "parent_plan_id": "parent_unit",
                    "parent_plan_content_hash": "parent_hash_unit",
                    "child_plan_id": "child_unit",
                },
            }
        ),
    }
    (base / "http_001.json").write_text(json.dumps(stage), encoding="utf-8")
    (base / "http_002.json").write_text(json.dumps(preview), encoding="utf-8")
    suites = [
        {
            "mode": journey["browser_mode"],
            "status": "passed",
            "test_nodeids": ["explicit_synthetic_unit_fixture::" + journey["test_name"]],
        }
    ]
    return registry, suites, [_snapshot(context)], base, stage, preview


def test_source_inventory_names_all_three_authoritative_operation_groups():
    from itinerary_system.product_app.draft_compiler import LEGACY_CAPABILITY_OPERATIONS
    from itinerary_system.product_app.interaction_contracts import TYPED_EDIT_OPERATION_NAMES
    from itinerary_system.product_app.workspace import WorkspaceStore

    registry = _registry()
    groups = {
        name: {row["operation"] for row in registry["rows"] if row["capability_group"] == name}
        for name in ("draft_capabilities", "typed_edit_capabilities", "map_edit_capabilities")
    }
    assert groups["draft_capabilities"] == LEGACY_CAPABILITY_OPERATIONS
    assert groups["typed_edit_capabilities"] == set(TYPED_EDIT_OPERATION_NAMES)
    assert groups["map_edit_capabilities"] == WorkspaceStore.CONFIRMED_MAP_OPERATIONS
    assert len({row["action_id"] for row in registry["rows"]}) == len(registry["rows"]) == 19
    assert all(all(row.get(field) for field in coverage.MAPPING_FIELDS) for row in registry["rows"])
    assert all(row["contexts"] == ["original", "accepted_child"] for row in registry["rows"])
    assert registry["unverified_requirements"]
    assert registry["status"] == "source_inventory_complete_runtime_coverage_incomplete"


def test_one_exact_mode_does_not_cover_four_draft_only_modes(tmp_path):
    registry, suites, snapshots, *_ = _fixture(tmp_path)
    result = coverage.coverage_report(snapshots, suites, registry, tmp_path)
    assert result["passed"] is False
    variants = {row["id"]: row["covered"] for row in result["enabled_interactions"][0]["variants"]}
    assert variants == {"exact": True, "preferred": False, "minimum": False, "maximum": False, "range": False}
    (proof,) = result["runtime_operation_evidence"]
    assert proof["parent_plan_id"] == "parent_unit"
    assert proof["parent_plan_content_hash"] == "parent_hash_unit"


def test_foreign_parent_operation_cannot_cover_authoritative_snapshot(tmp_path, monkeypatch):
    registry = _registry()
    mapping = next(row for row in registry["rows"] if row["operation"] == "set_stop_duration")
    registry.update(rows=[mapping], unverified_requirements=[], required_contexts=["accepted_child"])
    observation = {
        "action_id": mapping["action_id"],
        "context": "accepted_child",
        "variant": "exact",
        "feedback_tier": "evaluated",
        "parent_plan_id": "transient_parent",
        "parent_plan_content_hash": "transient_hash",
    }
    monkeypatch.setattr(coverage, "collect_verified_operations", lambda *args: [observation])
    result = coverage.coverage_report([_snapshot()], [], registry, tmp_path)
    exact = next(row for row in result["enabled_interactions"][0]["variants"] if row["id"] == "exact")
    assert exact["covered"] is False and exact["runtime_evidence"] == []
    assert result["passed"] is False


def test_cross_suite_parent_binding_ignores_disposable_session_but_not_identity():
    parent = _snapshot()
    other_session = {**parent, "session_id": "independent_browser_session"}
    assert coverage.same_authoritative_parent(parent, other_session)
    for key in ("context", "plan_id", "plan_content_hash"):
        assert not coverage.same_authoritative_parent(parent, {**other_session, key: "foreign"})


@pytest.mark.parametrize(
    "fault",
    [
        "source",
        "accepted_hash",
        "accepted_id",
        "operation_id",
        "returned_parameters",
        "test_name",
        "missing_artifact",
        "wrong_context",
        "failed_suite",
        "path_escape",
        "malformed_http",
    ],
)
def test_plausible_but_unbound_evidence_never_covers_exact_mode(tmp_path, fault):
    registry, suites, snapshots, base, stage, preview = _fixture(tmp_path)
    if fault == "source":
        request = json.loads(stage["request_body_utf8"])
        request["source"] = "copilot"
        stage["request_body_utf8"] = json.dumps(request)
    elif fault in {"accepted_hash", "accepted_id"}:
        response = json.loads(preview["response_body_utf8"])
        response["proposal"]["parent_plan_content_hash" if fault == "accepted_hash" else "parent_plan_id"] = "other"
        preview["response_body_utf8"] = json.dumps(response)
    elif fault == "operation_id":
        response = json.loads(preview["response_body_utf8"])
        response["session"]["draft"][0]["operation_id"] = "other"
        preview["response_body_utf8"] = json.dumps(response)
    elif fault == "returned_parameters":
        response = json.loads(stage["response_body_utf8"])
        response["operation"]["parameters"] = {"duration": {"mode": "preferred"}}
        stage["response_body_utf8"] = json.dumps(response)
    elif fault == "test_name":
        suites[0]["test_nodeids"] = ["unit_fixture::unrelated_green_test"]
    elif fault == "missing_artifact":
        registry["rows"][0]["journeys"][0]["evidence"].append("not_captured.json")
    elif fault == "wrong_context":
        snapshots[0]["context"] = "original"
    elif fault == "failed_suite":
        suites[0]["status"] = "failed"
    elif fault == "path_escape":
        registry["rows"][0]["journeys"][0]["case_directory"] = "../outside"
    elif fault == "malformed_http":
        stage["request_body_utf8"] = "not JSON"
    (base / "http_001.json").write_text(json.dumps(stage), encoding="utf-8")
    (base / "http_002.json").write_text(json.dumps(preview), encoding="utf-8")
    result = coverage.coverage_report(snapshots, suites, registry, tmp_path)
    assert result["passed"] is False
    assert result["runtime_operation_evidence"] == []
    assert not any(row["covered"] for row in result["enabled_interactions"][0]["variants"])
    if fault == "malformed_http":
        assert {"code": "interaction_runtime_evidence_invalid"} in result["findings"]


def test_passed_node_without_actual_http_is_not_coverage(tmp_path):
    registry, suites, snapshots, base, *_ = _fixture(tmp_path)
    registry["rows"][0]["journeys"][0]["case_directory"] = "unrecorded_case"
    result = coverage.coverage_report(snapshots, suites, registry, tmp_path)
    assert result["passed"] is False
    assert result["runtime_operation_evidence"] == []


def test_unsupported_role_and_map_tier_are_not_reported_as_evaluated():
    from itinerary_system.product_app.interaction_contracts import typed_edit_capabilities

    group = typed_edit_capabilities()
    role = coverage.required_variants("set_stop_role", group["operations"]["set_stop_role"], group)
    assert len(role) == 10
    assert {row["id"] for row in role if row["feedback_tier"] == "evaluated"} == {
        "attraction",
        "activity",
        "meal",
        "rest_stop",
        "scenic_stop",
    }
    assert {row["id"] for row in role if row["feedback_tier"] == "draft_only"} == {
        "lodging",
        "transport_hub",
        "route_waypoint",
        "origin",
        "destination",
    }
    result = coverage.required_variants(
        "add_custom_waypoint", {"enabled": True, "feedback_tier": "route_checked", "preview_executable": False}, {}
    )
    assert result == [{"id": "default", "feedback_tier": "route_checked"}]


def test_capability_cannot_remove_conditional_modes_to_bypass_gate():
    capability = deepcopy(_snapshot()["capabilities"]["typed_edit_capabilities"]["operations"]["set_stop_duration"])
    del capability["draft_only_modes"]
    assert coverage.required_variants("set_stop_duration", capability, {}) == [
        {"id": "invalid_conditional_contract", "feedback_tier": "invalid"}
    ]


def test_missing_registry_or_malformed_capability_returns_stable_finding():
    assert coverage.coverage_report([], [], None)["findings"] == [{"code": "interaction_registry_invalid"}]
    snapshot = _snapshot()
    snapshot["capabilities"]["typed_edit_capabilities"]["operations"]["set_stop_duration"] = None
    result = coverage.coverage_report(
        [snapshot], [], {"required_contexts": [], "rows": [], "unverified_requirements": []}
    )
    assert result["findings"] == [{"code": "runtime_capability_operation_invalid", "operation": "set_stop_duration"}]


def test_exact_chat_sse_is_not_misparsed_as_an_operation_response(tmp_path):
    registry, suites, snapshots, base, *_ = _fixture(tmp_path)
    chat = {
        "schema_version": "e2e1-product-http-io-v1",
        "method": "POST",
        "response_status_code": 200,
        "url": "http://127.0.0.1:8127/api/sessions/session_unit/copilot/messages",
        "request_body_utf8": '{"text": "unit fixture"}',
        "response_body_utf8": "event: done\ndata: {}\n\n",
    }
    (base / "http_003.json").write_text(json.dumps(chat), encoding="utf-8")
    result = coverage.coverage_report(snapshots, suites, registry, tmp_path)
    assert len(result["runtime_operation_evidence"]) == 1
    assert not any(row["code"] == "interaction_runtime_evidence_invalid" for row in result["findings"])
