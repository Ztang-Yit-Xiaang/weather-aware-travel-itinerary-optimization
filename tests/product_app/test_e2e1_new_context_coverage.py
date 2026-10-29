"""Synthetic projection checks only; never substitutes for browser verification."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import e2e1_interaction_coverage as coverage  # noqa: E402


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def fixture(root, mode):
    snapshot = {"context": "accepted_child", "plan_id": "parent", "plan_content_hash": "hash"}
    parent = {"active_plan_id": "parent", "active_plan_content_hash": "hash"}
    mapping = {"operation": "add_candidate", "contexts": ["accepted_child"], "action_id": "add"}
    if mode == "candidate-day-matrix":
        test = "test_browser_candidate_day_readiness_in_original_and_accepted_child"
        base = root / "suites/candidate-day-matrix/candidate_day_matrix"
        rows = [{"context": "accepted_child", "accepted_plan_id": "parent", "accepted_plan_content_hash": "hash",
            "assertions_passed": True, "comparison_eligibility": "not_evaluated", "readiness": {"preview_executable": ready},
            "candidate_id": "candidate", "day": day,
            "scope": "real_staging_and_impact_only" if ready else "disabled_pair_not_interacted"}
            for ready, day in ((True, 5), (False, 6))]
        write(base / "candidate_day_cases.json", rows)
        assertions = {"assertions_passed": True, "pair_scope": "readiness_and_staging_not_eligibility"}
    elif mode == "waypoint-contexts":
        test = "test_route_checked_waypoint_contexts_preserve_accepted_plan[accepted_child]"
        base = root / "suites/waypoint-contexts/waypoint_contexts/accepted_child"
        mapping.update(operation="add_route_waypoint")
        rows = [{"context": "accepted_child", "parent": parent, "assertions_passed": True,
            "operation": "add_route_waypoint", "variant": "default", "feedback_tier": "route_checked",
            "evaluated_repair": False, "scope": "route_checked_draft_only"}]
        write(base / "operation_assertions.json", rows)
        assertions = {"assertions_passed": True, "operations": rows}
    else:
        test = "test_keep_and_chat_families_preserve_runtime_truth[accepted_child]"
        base = root / "suites/chat-closure/chat_closure/accepted_child"
        mapping.update(operation="keep_stop")
        operations = [{"type": "keep_stop", "target": "stop", "parameters": {}, "source": "copilot"}]
        rows = [{"case": "keep_only_accepted", "context": "accepted_child", "assertions_passed": True,
            "parent": parent, "comparison_eligibility": "eligible", "operations": operations,
            "interaction_surface": "copilot_chat", "scope": "evaluated_preview_only"}]
        write(base / "keep_only_accepted/preview_response.json", {"session": {"draft": operations}, "proposal": {
            "state": "eligible", "parent_plan_id": "parent", "parent_plan_content_hash": "hash", "child_plan_id": "child"}})
        write(base / "journey_assertions.json", rows)
        assertions = {"assertions_passed": True, "journeys": rows}
    write(base / "browser_assertions.json", assertions)
    suites = [{"mode": mode, "status": "passed", "test_nodeids": ["explicit_unit_fixture::" + test]}]
    return [snapshot], suites, {"rows": [mapping]}, base, assertions


@pytest.mark.parametrize("mode", ["candidate-day-matrix", "waypoint-contexts", "chat-closure"])
def test_new_modes_keep_their_exact_observed_scope(tmp_path, mode):
    snapshots, suites, registry, *_ = fixture(tmp_path, mode)
    observed = coverage.collect_verified_operations(tmp_path, snapshots, suites, registry)
    if mode == "candidate-day-matrix":
        assert [row["feedback_tier"] for row in observed] == ["readiness_and_staging", "unsupported"]
        assert all(row["preview_state"] == "not_evaluated" and row["child_plan_id"] is None for row in observed)
    elif mode == "waypoint-contexts":
        assert len(observed) == 1 and observed[0]["feedback_tier"] == "route_checked"
        assert observed[0]["evaluated_repair_claimed"] is False and observed[0]["child_plan_id"] is None
    else:
        assert len(observed) == 1 and observed[0]["feedback_tier"] == "evaluated"
        assert observed[0]["child_plan_id"] == "child" and observed[0]["actual_openai_connectivity_claimed"] is False


@pytest.mark.parametrize("mode", ["candidate-day-matrix", "waypoint-contexts", "chat-closure"])
@pytest.mark.parametrize("fault", ["failed_suite", "unregistered_node", "wrong_hash", "wrong_context", "false_assertions"])
def test_unbound_or_incomplete_new_mode_evidence_never_receives_credit(tmp_path, mode, fault):
    snapshots, suites, registry, base, assertions = fixture(tmp_path, mode)
    if fault == "failed_suite":
        suites[0]["status"] = "failed"
    elif fault == "unregistered_node":
        suites[0]["test_nodeids"] = ["not_the_registered_browser_test"]
    elif fault == "wrong_hash":
        snapshots[0]["plan_content_hash"] = "other"
    elif fault == "wrong_context":
        snapshots[0]["context"] = "original"
    else:
        assertions["assertions_passed"] = False
        write(base / "browser_assertions.json", assertions)
    assert coverage.collect_verified_operations(tmp_path, snapshots, suites, registry) == []
