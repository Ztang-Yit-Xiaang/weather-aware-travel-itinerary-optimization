"""R02/R04/R05/R12: real compiler boundaries, never a provider call."""

import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

from itinerary_system.product_app.draft_compiler import FrozenDraftCompiler
from itinerary_system.product_app.product_demo import load_product_demo_package
from itinerary_system.product_app.workspace import WorkspaceError

ROOT = Path(__file__).resolve().parents[2]
TARGET = "bixby_creek_bridge_viewpoint"
CANDIDATE = "pfeiffer_big_sur_state_park"


@pytest.fixture(scope="module")
def package():
    return load_product_demo_package(ROOT, ROOT / "runs/california-coast-coherent-demo-v1")


@pytest.fixture
def compiler(package):
    return FrozenDraftCompiler(package.primary_bundle.parent_plan, package.evidence_bundles, repository_root=ROOT)


def operation(kind, target, parameters=None):
    return {"type": kind, "target": target, "parameters": parameters or {}}


def save_result(tmp_path, draft, compiler, result):
    (tmp_path / "exact_compiler_io.json").write_text(
        json.dumps(
            {
                "input": {"operations": draft, "accepted_plan_id": compiler.parent_plan_id},
                "output": {
                    "state": result.state,
                    "reason": result.reason,
                    "blocking_findings": result.blocking_findings,
                    "runtime_execution": result.runtime_execution,
                    "child_plan": result.child_plan,
                    "plan_diff": result.diff,
                    "evaluation_certificate": result.certificate,
                },
                "provider_calls": 0,
                "provider_mode": "not_used_by_compiler",
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def test_keep_only_never_injects_registered_replacement_candidates(compiler, tmp_path):
    draft = [operation("keep_stop", "golden_gate_bridge")]
    request, _ = compiler._compile_request(tuple(draft))
    assert request.candidate_pois == ()
    assert request.parsed_intent.must_keep == ("golden_gate_bridge",)
    result = compiler.compile(draft, accepted_plan_id=compiler.parent_plan_id)
    save_result(tmp_path, draft, compiler, result)
    assert result.runtime_execution["repair_request"]["candidate_pois"] == []
    assert not any(
        "pfeiffer" in reason or "point_lobos" in reason
        for reason in result.runtime_execution["planner_trace"]["failure_reasons"]
    )


def test_keep_plus_replacement_runs_exact_requested_change(compiler, tmp_path):
    original = deepcopy(compiler.parent_plan)
    draft = [
        operation("keep_stop", "golden_gate_bridge"),
        operation("replace_nearby", TARGET, {"candidate_id": CANDIDATE}),
    ]
    result = compiler.compile(draft, accepted_plan_id=compiler.parent_plan_id)
    save_result(tmp_path, draft, compiler, result)
    assert result.state == "eligible"
    assert result.runtime_execution["evaluation_executed"] is True
    request = result.runtime_execution["repair_request"]
    assert [row["stop_id"] for row in request["candidate_pois"]] == [CANDIDATE]
    assert request["confirmed_constraints"]["preserved_stop_ids"] == ["golden_gate_bridge"]
    assert request["confirmed_constraints"]["affected_days"] == [5]
    assert set(result.child_plan["sequence"]) == (set(original["sequence"]) - {TARGET}) | {CANDIDATE}
    assert compiler.parent_plan == original


def test_combined_validation_rejects_invalid_clause_before_optimizer(compiler, monkeypatch, tmp_path):
    attempts = []

    def forbidden_execution(request):
        attempts.append(request)
        pytest.fail("An invalid combined draft must not dispatch selection search")

    monkeypatch.setattr(compiler, "_execute", forbidden_execution)
    draft = [
        operation("replace_nearby", TARGET, {"candidate_id": CANDIDATE}),
        operation(
            "set_stop_duration",
            "griffith_observatory",
            {"duration": {"mode": "exact", "preferred_minutes": 90, "minimum_minutes": 89, "maximum_minutes": 90}},
        ),
    ]
    with pytest.raises(WorkspaceError) as rejected:
        compiler.compile(draft, accepted_plan_id=compiler.parent_plan_id)
    (tmp_path / "exact_invalid_combination_io.json").write_text(
        json.dumps(
            {
                "input": draft,
                "output": {"code": rejected.value.code, "optimizer_attempts": len(attempts)},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    assert rejected.value.code == "invalid_stop_duration"
    assert attempts == []


@pytest.mark.parametrize(
    "kind,parameters",
    [
        (
            "set_stop_duration",
            {"duration": {"mode": "exact", "preferred_minutes": 60, "minimum_minutes": 60, "maximum_minutes": 60}},
        ),
        ("set_stop_time_window", {"earliest_arrival": "10:00", "latest_departure": None}),
        ("set_stop_role", {"role": "scenic_stop"}),
        ("set_stop_order", {"day": 5, "sequence_index": 0}),
    ],
)
def test_combined_edit_of_removed_target_is_never_silently_retargeted(compiler, kind, parameters):
    draft = [operation("replace_nearby", TARGET, {"candidate_id": CANDIDATE}), operation(kind, TARGET, parameters)]
    with pytest.raises(WorkspaceError, match="draft_conflicts_with_replacement"):
        compiler.compile(draft, accepted_plan_id=compiler.parent_plan_id)


def test_combined_ineligible_result_keeps_all_requested_artifacts(compiler, tmp_path):
    original = deepcopy(compiler.parent_plan)
    draft = [
        operation("replace_nearby", TARGET, {"candidate_id": CANDIDATE}),
        *[
            operation(
                "set_stop_duration",
                stop,
                {
                    "duration": {
                        "mode": "exact",
                        "preferred_minutes": 480,
                        "minimum_minutes": 480,
                        "maximum_minutes": 480,
                    }
                },
            )
            for stop in ("sequoia_national_park", "kings_canyon_national_park")
        ],
    ]
    result = compiler.compile(draft, accepted_plan_id=compiler.parent_plan_id)
    save_result(tmp_path, draft, compiler, result)
    assert result.state == "ineligible"
    assert result.certificate["evaluation_status"] == "FAILED"
    assert len(result.diff["duration_changes"]) == 2
    assert [row["stop_id"] for row in result.diff["added_stops"]] == [CANDIDATE]
    assert [row["stop_id"] for row in result.diff["deleted_stops"]] == [TARGET]
    assert len(result.runtime_execution["operations"]) == 3
    assert compiler.parent_plan == original


def test_combined_final_eligibility_is_not_an_intermediate_prefix_decision(compiler, tmp_path):
    baseline = compiler.compile(
        [operation("replace_nearby", TARGET, {"candidate_id": CANDIDATE})], accepted_plan_id=compiler.parent_plan_id
    )
    assert baseline.state == "eligible"
    compiler = compiler.for_parent(baseline.child_plan)
    short_visit = operation(
        "set_stop_duration",
        "golden_gate_bridge",
        {"duration": {"mode": "exact", "preferred_minutes": 15, "minimum_minutes": 15, "maximum_minutes": 15}},
    )
    window = operation(
        "set_stop_time_window", "golden_gate_bridge", {"earliest_arrival": None, "latest_departure": "10:15"}
    )
    prefix = compiler.compile([window], accepted_plan_id=compiler.parent_plan_id)
    # Frozen coast fixture arrives at 09:50. The original 45-minute visit
    # misses 10:15; the requested 15-minute visit departs at 10:05.
    save_result(tmp_path, [window], compiler, prefix)
    assert prefix.state == "ineligible"
    final = compiler.compile([window, short_visit], accepted_plan_id=compiler.parent_plan_id)
    (tmp_path / "exact_final_combination_io.json").write_text(
        json.dumps(
            {
                "input": {"operations": [window, short_visit], "parent_plan": compiler.parent_plan},
                "output": {"state": final.state, "runtime_execution": final.runtime_execution},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    assert final.state == "eligible"
    assert final.runtime_execution["optimizer_execution"] is None
    assert final.certificate["plan_content_hash"] == final.child_plan["content_hash"]
    assert len(final.diff["duration_changes"]) == len(final.diff["time_window_changes"]) == 1


def test_combined_recompile_has_stable_child_and_diff_identity(compiler):
    draft = [
        operation("replace_nearby", TARGET, {"candidate_id": CANDIDATE}),
        operation("set_stop_role", "griffith_observatory", {"role": "activity"}),
    ]
    first = compiler.compile(draft, accepted_plan_id=compiler.parent_plan_id)
    second = compiler.compile(draft, accepted_plan_id=compiler.parent_plan_id)
    assert first.state == second.state == "eligible"
    assert first.child_plan == second.child_plan
    assert first.diff == second.diff
    assert first.runtime_execution["execution_id"] != second.runtime_execution["execution_id"]


def test_combined_missing_route_evidence_never_leaks_intermediate_child(compiler, tmp_path):
    original = deepcopy(compiler.parent_plan)
    matrix, day_config, trip_config = compiler._runtime_inputs()
    compiler._runtime_cache = (
        replace(matrix, cells={pair: cell for pair, cell in matrix.cells.items() if pair[1] != CANDIDATE}),
        day_config,
        trip_config,
    )
    draft = [
        operation("replace_nearby", TARGET, {"candidate_id": CANDIDATE}),
        operation("set_stop_role", "griffith_observatory", {"role": "activity"}),
    ]
    result = compiler.compile(draft, accepted_plan_id=compiler.parent_plan_id)
    save_result(tmp_path, draft, compiler, result)
    assert result.state == "blocked"
    assert result.reason == "blocked_missing_route_evidence"
    assert result.child_plan is result.certificate is result.diff is None
    assert result.runtime_execution["parent_plan_id"] == compiler.parent_plan_id
    assert result.runtime_execution["child_plan_id"] is None
    assert result.runtime_execution["evaluation_executed"] is False
    assert len(result.runtime_execution["operations"]) == 2
    assert compiler.parent_plan == original


def test_combined_conflicting_final_order_positions_fail_closed(compiler):
    draft = [
        operation("set_stop_order", "sequoia_national_park", {"day": 4, "sequence_index": 1}),
        operation("set_stop_order", "kings_canyon_national_park", {"day": 4, "sequence_index": 1}),
        operation("set_stop_role", "griffith_observatory", {"role": "activity"}),
    ]
    with pytest.raises(WorkspaceError, match="draft_no_effect|draft_conflicting_stop_order"):
        compiler.compile(draft, accepted_plan_id=compiler.parent_plan_id)


@pytest.mark.parametrize("extra_count", [0, 1, 2])
@pytest.mark.parametrize("invalid_first", [True, False])
def test_candidate_target_validation_is_independent_of_draft_length(compiler, monkeypatch, extra_count, invalid_first):
    invalid = operation("replace_nearby", "golden_gate_bridge", {"candidate_id": CANDIDATE})
    extras = [operation("keep_stop", stop) for stop in ("stearns_wharf", "griffith_observatory")[:extra_count]]
    draft = [invalid, *extras] if invalid_first else [*extras, invalid]

    def forbidden_execute(_request):
        pytest.fail("invalid candidate/target pair reached the optimizer")

    monkeypatch.setattr(compiler, "_execute", forbidden_execute)
    with pytest.raises(WorkspaceError, match="draft_candidate_target_mismatch"):
        compiler.compile(draft, accepted_plan_id=compiler.parent_plan_id)


def test_absent_route_cell_is_structured_blocked_not_evaluated(package, tmp_path):
    bundles = {}
    for key, bundle in package.evidence_bundles.items():
        matrix = deepcopy(bundle.route_matrix)
        matrix["cells"] = [row for row in matrix["cells"] if row["destination_id"] != CANDIDATE]
        bundles[key] = replace(bundle, route_matrix=matrix)
    compiler = FrozenDraftCompiler(package.primary_bundle.parent_plan, bundles, repository_root=ROOT)
    draft = [operation("replace_nearby", TARGET, {"candidate_id": CANDIDATE})]
    result = compiler.compile(draft, accepted_plan_id=compiler.parent_plan_id)
    save_result(tmp_path, draft, compiler, result)
    assert result.state == "blocked"
    assert result.reason == "blocked_missing_route_evidence"
    assert result.child_plan is result.certificate is result.diff is None
    assert result.runtime_execution["evaluation_executed"] is False
    assert result.runtime_execution["evaluation_id"] is None
    (finding,) = result.blocking_findings
    assert finding["code"] == "missing_route_matrix_cell"
    assert finding["destination_id"] == CANDIDATE
    assert finding["origin_id"]
    assert finding["matrix_id"] == compiler._base_bundle.route_matrix["matrix_id"]
    assert finding["context_snapshot_id"] == compiler._base_bundle.route_matrix["context_snapshot_id"]
    assert finding["corrective_action"]


@pytest.mark.parametrize(
    "failure,reason",
    [
        (TimeoutError("synthetic local budget exhausted"), "repair_execution_timed_out"),
        (RuntimeError("synthetic internal failure"), "repair_execution_failed"),
    ],
)
def test_internal_executor_failure_is_not_evaluated_ineligibility(compiler, monkeypatch, failure, reason, tmp_path):
    # Fault injection is confined to the local day solver. The actual pipeline
    # catches it and creates its normal planner trace; no provider is involved.
    from itinerary_system.repair import baselines

    def fail_day_solver(*args, **kwargs):
        raise failure

    monkeypatch.setattr(baselines, "solve_day_route_subproblem", fail_day_solver)
    draft = [operation("replace_nearby", TARGET, {"candidate_id": CANDIDATE})]
    result = compiler.compile(draft, accepted_plan_id=compiler.parent_plan_id)
    save_result(tmp_path, draft, compiler, result)
    assert result.state == "failed"
    assert result.reason == reason
    assert result.runtime_execution["evaluation_executed"] is False
    assert result.child_plan is result.certificate is None
    assert result.blocking_findings[0]["code"] == reason


def test_local_search_exhaustion_does_not_claim_independent_evaluation(compiler, tmp_path):
    matrix, day_config, trip_config = compiler._runtime_inputs()
    compiler._runtime_cache = (matrix, replace(day_config, max_day_minutes=1.0), trip_config)
    draft = [operation("replace_nearby", TARGET, {"candidate_id": CANDIDATE})]
    result = compiler.compile(draft, accepted_plan_id=compiler.parent_plan_id)
    save_result(tmp_path, draft, compiler, result)
    assert result.state == "not_evaluated"
    assert result.reason == "no_feasible_local_candidate"
    assert result.runtime_execution["evaluation_executed"] is False
    assert result.certificate is None


def test_actual_evaluator_rejection_remains_ineligible_with_certificate(compiler, tmp_path):
    draft = [
        operation(
            "set_stop_duration",
            stop,
            {
                "duration": {
                    "mode": "exact",
                    "preferred_minutes": 480,
                    "minimum_minutes": 480,
                    "maximum_minutes": 480,
                }
            },
        )
        for stop in ("sequoia_national_park", "kings_canyon_national_park")
    ]
    result = compiler.compile(draft, accepted_plan_id=compiler.parent_plan_id)
    save_result(tmp_path, draft, compiler, result)
    assert result.state == "ineligible"
    assert result.reason == "independent_evaluation_failed"
    assert result.child_plan and result.certificate
    assert result.certificate["plan_id"] == result.child_plan["plan_id"]
    assert result.certificate["comparison_eligibility"] == "ineligible"
    assert result.certificate["evaluation_status"] == "FAILED"


def test_non_road_validated_matrix_is_blocked_before_independent_evaluation(compiler, tmp_path):
    matrix, day_config, trip_config = compiler._runtime_inputs()
    changed_cells = {
        pair: replace(cell, road_validated=False) if pair[1] == CANDIDATE else cell
        for pair, cell in matrix.cells.items()
    }
    compiler._runtime_cache = (replace(matrix, cells=changed_cells), day_config, trip_config)
    draft = [operation("replace_nearby", TARGET, {"candidate_id": CANDIDATE})]
    result = compiler.compile(draft, accepted_plan_id=compiler.parent_plan_id)
    save_result(tmp_path, draft, compiler, result)
    assert result.state == "blocked"
    assert result.reason == "blocked_invalid_route_evidence"
    assert result.runtime_execution["evaluation_executed"] is False
    assert result.certificate is None


def test_local_setup_timeout_has_distinct_stable_error(compiler, monkeypatch):
    def timeout():
        raise TimeoutError("synthetic runtime setup timeout")

    monkeypatch.setattr(compiler, "_runtime_inputs", timeout)
    with pytest.raises(WorkspaceError, match="draft_pipeline_execution_timed_out"):
        compiler.compile(
            [operation("replace_nearby", TARGET, {"candidate_id": CANDIDATE})], accepted_plan_id=compiler.parent_plan_id
        )


@pytest.mark.parametrize(
    "extra_types",
    [
        ("set_stop_duration",),
        ("set_stop_duration", "set_stop_time_window"),
        ("set_stop_order", "set_stop_duration", "set_stop_time_window", "set_stop_role"),
    ],
    ids=["replacement-duration", "replacement-duration-window", "all-supported-edit-families"],
)
def test_combined_supported_edits_produce_one_complete_evaluated_child(compiler, tmp_path, extra_types):
    original = deepcopy(compiler.parent_plan)
    edits = {
        "set_stop_duration": operation(
            "set_stop_duration",
            "griffith_observatory",
            {"duration": {"mode": "exact", "preferred_minutes": 90, "minimum_minutes": 90, "maximum_minutes": 90}},
        ),
        "set_stop_time_window": operation(
            "set_stop_time_window", "golden_gate_bridge", {"earliest_arrival": "10:00", "latest_departure": None}
        ),
        "set_stop_order": operation("set_stop_order", "sequoia_national_park", {"day": 4, "sequence_index": 1}),
        "set_stop_role": operation("set_stop_role", "golden_gate_bridge", {"role": "scenic_stop"}),
    }
    draft = [operation("replace_nearby", TARGET, {"candidate_id": CANDIDATE}), *(edits[kind] for kind in extra_types)]
    exact_input = {
        "operations": draft,
        "accepted_plan_id": compiler.parent_plan_id,
        "parent_plan_content_hash": original["content_hash"],
    }
    try:
        result = compiler.compile(draft, accepted_plan_id=compiler.parent_plan_id)
    except WorkspaceError as error:
        (tmp_path / "exact_combined_compiler_io.json").write_text(
            json.dumps(
                {
                    "input": exact_input,
                    "output": {"error_code": error.code, "status_code": error.status_code},
                    "scope": "Real internal compiler; provider transport not instrumented by this unit test",
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        raise
    save_result(tmp_path, draft, compiler, result)
    assert result.state == "eligible"
    child, diff, certificate = result.child_plan, result.diff, result.certificate
    assert child["parent_plan_id"] == original["plan_id"]
    assert child["plan_id"] == diff["child_plan_id"] == certificate["plan_id"]
    assert certificate["plan_content_hash"] == child["content_hash"]
    assert diff["parent_plan_id"] == original["plan_id"]
    assert set(child["sequence"]) == (set(original["sequence"]) - {TARGET}) | {CANDIDATE}
    stops = {row["stop_id"]: row for row in child["selected_stops"]}
    assert stops["griffith_observatory"]["visit_duration_minutes"] == 90
    assert len(diff["duration_changes"]) == 1
    if "set_stop_time_window" in extra_types:
        assert stops["golden_gate_bridge"]["time_window_constraint"]["earliest_arrival"] == "10:00"
        assert len(diff["time_window_changes"]) == 1
    if "set_stop_order" in extra_types:
        assert next(day["stop_ids"] for day in child["ordered_days"] if day["day"] == 4) == [
            "kings_canyon_national_park",
            "sequoia_national_park",
        ]
        assert diff["reorder_changes"]
    if "set_stop_role" in extra_types:
        assert stops["golden_gate_bridge"]["itinerary_role"] == "scenic_stop"
        assert len(diff["role_changes"]) == 1
    assert compiler.parent_plan == original
