from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import pytest

from itinerary_system.benchmark.response_aware import (
    SCENARIOS,
    SyntheticModelEvaluator,
    compare_matched_pool,
    toy_inputs,
)
from itinerary_system.plans import OwnedConstraint, OwnershipPolicy, compute_plan_diff
from itinerary_system.product_app.draft_compiler import _verify_diff
from itinerary_system.product_app.workspace import WorkspaceError
from itinerary_system.repair.lexicographic import planner_runs_from_lexicographic_result, solve_lexicographically
from itinerary_system.repair.master_model import RepairSolution, build_repair_master_model
from itinerary_system.repair.neighborhood import RepairRadius, build_repair_neighborhood
from itinerary_system.research_artifacts import stable_content_hash


def inputs(*, explicit_order=True, multi_day=False, policy=None):
    parent, request, matrix, config, scope = toy_inputs(SCENARIOS[1])
    records = []
    assignments = dict(parent.day_assignments)
    for index, raw in enumerate(parent.selected_stops, 1):
        day = 2 if multi_day and raw["stop_id"] == "o" else 1
        assignments[raw["stop_id"]] = day
        record = {**raw, "day": day}
        if explicit_order:
            record["stop_order"] = 1 if day == 2 else index
        records.append(record)
    parent = replace(
        parent,
        selected_stops=tuple(records),
        day_assignments=assignments,
        lodging_assignments={"1": "hotel", **({"2": "hotel"} if multi_day else {})},
    )
    model = build_repair_master_model(
        parent,
        request,
        build_repair_neighborhood(parent, request, RepairRadius.FULL_REOPTIMIZATION),
        matrix,
        ownership_policy=policy,
    )
    return parent, request, matrix, config, model


def solution(parent, sequence=None, **kwargs):
    sequence = sequence or parent.sequence
    return RepairSolution(
        selected_stop_ids=sequence,
        day_assignments=dict(parent.day_assignments),
        lodging_assignments=dict(parent.lodging_assignments),
        route_ids_by_day=dict(parent.route_ids_by_day),
        metadata={"candidate_id": str(sequence), "day_sequences": {"1": sequence}},
        **kwargs,
    )


def test_extraction_replaces_stale_order_and_preserves_parent():
    parent, _, _, _, model = inputs()
    before = deepcopy(parent.to_record())
    order = ("l", "p", "b", "a", "o")
    child = model.extract_plan(solution(parent, order))
    assert [stop["stop_order"] for stop in child.selected_stops] == [1, 2, 3, 4, 5]
    assert {row.stop_id for row in compute_plan_diff(parent, child).reorder_changes} == {"a", "p", "b"}
    assert child.change_components["weighted_edit_cost"] == 1.5
    assert parent.to_record() == before


def test_order_aliases_follow_actual_sequence():
    parent, request, matrix, _, _ = inputs()
    parent = replace(
        parent,
        selected_stops=tuple(
            {
                **stop,
                "order": stop["stop_order"],
                "route_sequence_index": stop["stop_order"],
                "visit_order": stop["stop_order"],
            }
            for stop in parent.selected_stops
        ),
    )
    model = build_repair_master_model(
        parent, request, build_repair_neighborhood(parent, request, RepairRadius.FULL_REOPTIMIZATION), matrix
    )
    child = model.extract_plan(solution(parent, ("l", "p", "b", "a", "o")))
    for index, stop in enumerate(child.selected_stops, 1):
        assert all(stop[key] == index for key in ("stop_order", "order", "route_sequence_index", "visit_order"))


def test_implicit_order_resets_per_day_and_unchanged_multiday_cost_is_zero():
    parent, _, _, _, model = inputs(explicit_order=False, multi_day=True)
    candidate = replace(solution(parent), metadata={"candidate_id": "same"})
    child = model.extract_plan(candidate)
    assert [stop["stop_order"] for stop in child.selected_stops] == [1, 2, 3, 4, 1]
    assert compute_plan_diff(parent, child).weighted_edit_cost == 0


def test_day_move_has_consistent_records_assignments_and_day_sequences():
    parent, _, _, _, model = inputs(multi_day=True)
    candidate = replace(
        solution(parent),
        day_assignments={**parent.day_assignments, "a": 2},
        metadata={"candidate_id": "move", "day_sequences": {"1": ("l", "p", "b"), "2": ("o", "a")}},
    )
    child = model.extract_plan(candidate)
    assert child.sequence == ("l", "p", "b", "o", "a")
    assert child.ordered_days == ({"day": 1, "stop_ids": ("l", "p", "b")}, {"day": 2, "stop_ids": ("o", "a")})
    assert [(stop["stop_id"], stop["day"], stop["stop_order"]) for stop in child.selected_stops] == [
        ("l", 1, 1),
        ("p", 1, 2),
        ("b", 1, 3),
        ("o", 2, 1),
        ("a", 2, 2),
    ]
    assert [(move.stop_id, move.from_day, move.to_day) for move in compute_plan_diff(parent, child).day_moves] == [
        ("a", 1, 2)
    ]


@pytest.mark.parametrize("mutation", ["raw_day", "raw_order", "alias", "ordered_day", "duplicate_ordered_day"])
def test_independent_evaluator_rejects_conflicting_representations(mutation):
    parent, _, matrix, config, model = inputs()
    child = model.extract_plan(solution(parent))
    stops = [dict(stop) for stop in child.selected_stops]
    if mutation == "raw_day":
        stops[0]["day"] = 2
    elif mutation == "raw_order":
        stops[0]["stop_order"] = 2
    elif mutation == "alias":
        stops[0]["visit_order"] = 2
    if mutation == "ordered_day":
        child = replace(child, ordered_days=({"day": 1, "stop_ids": tuple(reversed(child.sequence))},))
    elif mutation == "duplicate_ordered_day":
        child = replace(child, ordered_days=(*child.ordered_days, *child.ordered_days))
    child = replace(child, selected_stops=tuple(stops))
    result = SyntheticModelEvaluator(parent, matrix, config).evaluate(child)
    assert not result["eligible"]
    assert "plan_representation_conflict" in {row["code"] for row in result["modeled_failures"]}


def test_same_pool_full_diff_selector_removes_proxy_blind_spot():
    parent, _, matrix, config, model = inputs()
    bad = solution(parent, ("l", "p", "b", "a", "o"))
    good = solution(parent)
    pool = (bad, good)
    assert all(
        SyntheticModelEvaluator(parent, matrix, config).evaluate(model.extract_plan(candidate))["eligible"]
        for candidate in pool
    )
    proxy = solve_lexicographically(model, candidate_solutions=pool)
    corrected = solve_lexicographically(model, candidate_solutions=pool, use_plan_diff=True)
    assert proxy.selected_solution == bad
    assert corrected.selected_solution == good
    assert corrected.selected_components["weighted_edit_cost"] == 0
    assert corrected.stage_results[1].objective_value == 0
    assert corrected.preservation_cost_basis == "plan_diff_plus_relaxation"
    assert "plan-diff" in planner_runs_from_lexicographic_result(corrected)[0].planner_specification_id
    # No extra coverage, travel objective or follower is needed for this gain.
    only = solve_lexicographically(model, candidate_solutions=(bad,), use_plan_diff=True)
    assert only.selected_solution == bad and only.selected_components["weighted_edit_cost"] == 1.5


def test_custom_policy_is_used_by_extraction_and_full_diff_selector():
    policy = replace(
        OwnershipPolicy(), change_type_weights={**OwnershipPolicy().change_type_weights, "order_change": 2}
    )
    parent, _, _, _, model = inputs(policy=policy)
    bad = solution(parent, ("l", "p", "b", "a", "o"))
    child = model.extract_plan(bad)
    result = solve_lexicographically(model, candidate_solutions=(bad,), use_plan_diff=True)
    assert (
        child.change_components["weighted_edit_cost"]
        == compute_plan_diff(parent, child, policy).weighted_edit_cost
        == 6
    )
    assert result.selected_components["ownership_weighted_change"] == 6


def test_full_diff_selector_cannot_trade_off_booked_relaxation():
    parent, request, matrix, _, _ = inputs()
    request.confirmed_constraints["allow_booked_relaxation"] = True
    model = build_repair_master_model(
        parent, request, build_repair_neighborhood(parent, request, RepairRadius.FULL_REOPTIMIZATION), matrix
    )
    safe = solution(parent, ("l", "p", "b", "a", "o"))
    relaxed = replace(solution(parent), relaxed_constraint_ids=("protect_b",))
    result = solve_lexicographically(model, candidate_solutions=(relaxed, safe), use_plan_diff=True)
    assert result.selected_solution == safe
    assert result.stage_results[0].objective_value == 0


def test_matched_pool_cost_gain_is_not_extra_candidate_coverage():
    experiment = compare_matched_pool()
    for name, record in experiment["cases"].items():
        proxy, full = (record["outcomes"][key] for key in ("change_variable_proxy", "plan_diff_plus_relaxation"))
        costs = (proxy["diff"]["weighted_edit_cost"], full["diff"]["weighted_edit_cost"])
        assert costs == {"disruptive_first": (7, 6), "preserving_first": (6, 6), "restricted_pool": (7, 7)}[name]
        assert (
            proxy["solver"]["stage_results"][0]["objective_value"]
            == full["solver"]["stage_results"][0]["objective_value"]
            == 0
        )
        assert proxy["stage_runs"][0]["run_id"] != full["stage_runs"][0]["run_id"]
        for row in (proxy, full):
            assert row["evaluation"]["eligible"]
            assert row["evaluation"]["certificate"]["comparison_eligibility"] == "ineligible"


def test_empty_destination_day_honors_explicit_solved_order():
    parent, request, matrix, _, _ = inputs(multi_day=True)
    parent = replace(parent, lodging_assignments={**parent.lodging_assignments, "3": "hotel"})
    model = build_repair_master_model(
        parent, request, build_repair_neighborhood(parent, request, RepairRadius.FULL_REOPTIMIZATION), matrix
    )
    candidate = replace(
        solution(parent),
        day_assignments={**parent.day_assignments, "a": 3, "o": 3},
        metadata={"day_sequences": {"3": ("o", "a")}},
    )
    child = model.extract_plan(candidate)
    assert child.sequence == ("l", "p", "b", "o", "a")
    assert child.ordered_days[-1] == {"day": 3, "stop_ids": ("o", "a")}


@pytest.mark.parametrize("day_sequences", [{"1": ("l", "l")}, {"2": ("l",)}, {"1": "l"}])
def test_extraction_rejects_invalid_solved_day_sequences(day_sequences):
    parent, _, _, _, model = inputs()
    with pytest.raises(ValueError, match="day sequence"):
        model.extract_plan(replace(solution(parent), metadata={"day_sequences": day_sequences}))


def test_full_diff_mode_retains_soft_relaxation_penalty():
    parent, request, matrix, _, _ = inputs()
    soft = OwnedConstraint(
        constraint_id="soft_o",
        origin="user",
        strength="soft",
        scope="stop",
        target_id="o",
        relation="must_keep",
        value=True,
        confirmed=True,
        relaxation_policy="auto_with_penalty",
    )
    parent = replace(parent, owned_constraints=(*parent.owned_constraints, soft.to_record()))
    model = build_repair_master_model(
        parent, request, build_repair_neighborhood(parent, request, RepairRadius.FULL_REOPTIMIZATION), matrix
    )
    keep = replace(solution(parent), metadata={"candidate_id": "keep"})
    relaxed = replace(keep, relaxed_constraint_ids=("soft_o",), metadata={"candidate_id": "relaxed"})
    result = solve_lexicographically(model, candidate_solutions=(relaxed, keep), use_plan_diff=True)
    assert result.selected_solution == keep
    assert result.selected_components["ownership_weighted_change"] == 0
    only = solve_lexicographically(model, candidate_solutions=(relaxed,), use_plan_diff=True)
    assert only.selected_components["weighted_edit_cost"] == 0
    assert only.selected_components["ownership_weighted_change"] == 10


def test_product_diff_verifier_accepts_v3_and_rejects_rehashed_v2_without_demo_files():
    parent, request, matrix, config, _ = inputs()
    soft = OwnedConstraint(
        constraint_id="soft_o",
        origin="user",
        strength="soft",
        scope="stop",
        target_id="o",
        relation="must_keep",
        value=True,
        confirmed=True,
        relaxation_policy="auto_with_penalty",
    )
    parent = replace(parent, owned_constraints=(*parent.owned_constraints, soft.to_record()))
    model = build_repair_master_model(
        parent, request, build_repair_neighborhood(parent, request, RepairRadius.FULL_REOPTIMIZATION), matrix
    )
    candidate = replace(
        solution(parent),
        selected_stop_ids=parent.sequence[:-1],
        day_assignments={sid: 1 for sid in parent.sequence[:-1]},
        metadata={"candidate_id": "drop_o"},
    )
    child = model.extract_plan(candidate)
    diff = compute_plan_diff(parent, child, default_visit_minutes=15).to_record()
    assert diff["schema_version"] == "plan-diff-v5"
    certificate = SyntheticModelEvaluator(parent, matrix, config).evaluate(child)["certificate"]
    assert certificate["evaluator_version"] == "verify-001-plan-evaluator-v3"
    _verify_diff(diff, parent=parent, child=child, certificate=certificate, default_visit_minutes=15)
    old = {**certificate, "evaluator_version": "verify-001-plan-evaluator-v2"}
    old.pop("content_hash")
    old["content_hash"] = stable_content_hash(old)
    with pytest.raises(WorkspaceError, match="draft_pipeline_diff_invalid"):
        _verify_diff(diff, parent=parent, child=child, certificate=old, default_visit_minutes=15)
