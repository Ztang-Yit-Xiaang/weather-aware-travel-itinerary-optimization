from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import pytest

from itinerary_system.benchmark.response_aware import (
    SCENARIOS,
    SyntheticModelEvaluator,
    ToyScenario,
    run_toy_scenario,
    select_response,
    toy_inputs,
)
from itinerary_system.plans import compute_plan_diff
from itinerary_system.repair.induced_disruption import DisruptionScope, measure_induced_disruption
from itinerary_system.repair.master_model import RepairSolution, build_repair_master_model
from itinerary_system.repair.neighborhood import RepairRadius, build_repair_neighborhood


@pytest.fixture(scope="module")
def experiment(tmp_path_factory):
    root = tmp_path_factory.mktemp("response_aware")
    return {scenario.scenario_id: run_toy_scenario(scenario, root / scenario.scenario_id) for scenario in SCENARIOS}


def test_local_improvement_induces_secondary_disruption(experiment):
    before = experiment["travel_before"]["strategies"]["full_reoptimization"]
    after = experiment["travel_after"]["strategies"]["full_reoptimization"]
    assert before["plan"]["sequence"] == ["l", "a", "p", "b", "o"]
    assert after["plan"]["sequence"] == ["l", "p", "b", "a", "o"]
    assert (before["travel_minutes"], after["travel_minutes"]) == (30, 26)
    assert (before["disruption"]["weighted_edit_cost"], after["disruption"]["weighted_edit_cost"]) == (0, 4.5)
    assert after["disruption"]["outside_neighborhood_cost"] == 0.5
    assert after["disruption"]["outside_stop_ids"] == ("p",)
    assert experiment["travel_after"]["strategies"]["progressive_repair"]["disruption"]["weighted_edit_cost"] == 0


def test_unchanged_zero_and_exact_cost_partition():
    parent, _, _, _, scope = toy_inputs(SCENARIOS[1])
    unchanged = measure_induced_disruption(parent, parent, scope)
    assert unchanged.weighted_edit_cost == unchanged.direct_cost == unchanged.secondary_cost == 0
    assert unchanged.secondary_event_count == unchanged.parent_order_propagation_radius == 0
    child = replace(parent, plan_id="child", sequence=("l", "p", "b", "a", "o"))
    metric = measure_induced_disruption(parent, child, scope)
    assert metric.weighted_edit_cost == metric.direct_cost + metric.secondary_cost == 1.5
    assert len({(event["family"], event["target"]) for event in metric.events}) == len(metric.events)
    assert metric.parent_order_propagation_radius == 1
    direct = replace(scope, direct_edits=(("reorder_changes", "a"),))
    assert measure_induced_disruption(parent, child, direct).direct_cost == 0.5
    assert measure_induced_disruption(parent, child, direct).secondary_cost == 1.0


def test_relative_order_is_not_positional_reindexing():
    parent, _, _, _, _ = toy_inputs(SCENARIOS[1])
    sequence = tuple(sid for sid in parent.sequence if sid != "a")
    child = replace(
        parent,
        plan_id="drop_a",
        sequence=sequence,
        selected_stops=tuple(stop for stop in parent.selected_stops if stop["stop_id"] != "a"),
        day_assignments=dict.fromkeys(sequence, 1),
    )
    metric = measure_induced_disruption(parent, child, DisruptionScope(("a",), (1,), (("deleted_stops", "a"),)))
    assert metric.secondary_stop_ids == ("b", "o", "p")
    assert metric.unrelated_order_inversions == 0
    swapped = replace(child, sequence=("l", "b", "p", "o"))
    assert measure_induced_disruption(parent, swapped, DisruptionScope(("a",), (1,))).unrelated_order_inversions == 1


def test_attribute_specific_ownership_and_hotel_day_metrics():
    parent, _, _, _, _ = toy_inputs(SCENARIOS[1])
    # p's strong existence ownership weights deletion, not every order edit.
    child = replace(
        parent,
        plan_id="delete_preferred",
        sequence=("l", "a", "b", "o"),
        selected_stops=tuple(
            {**stop, "day": 2} if stop["stop_id"] == "o" else stop
            for stop in parent.selected_stops
            if stop["stop_id"] != "p"
        ),
        day_assignments={"l": 1, "a": 1, "b": 1, "o": 2},
        lodging_assignments={"1": "new_hotel", "2": "hotel"},
    )
    diff = compute_plan_diff(parent, child)
    assert diff.deleted_stops[0].cost == 100
    metric = measure_induced_disruption(parent, child, DisruptionScope(("a",), (1,)))
    assert metric.changed_assignment_stop_ids == ("o",)
    assert metric.changed_lodging_days == (1, 2)
    assert metric.weighted_edit_cost == metric.direct_cost + metric.secondary_cost
    assert measure_induced_disruption(parent, parent, DisruptionScope((), (1,))).parent_order_propagation_radius is None


def test_counterfactual_weather_and_rerank_ablation(experiment):
    record = experiment["weather_choice"]
    b, c = (record["strategies"][key] for key in ("progressive_repair", "response_aware"))
    assert b["disruption"]["weighted_edit_cost"] == c["disruption"]["weighted_edit_cost"] == 6
    assert (b["travel_minutes"], c["travel_minutes"]) == (30, 28)
    assert c["disruption"] == record["rerank_existing_only"]["disruption"]
    assert c["travel_minutes"] == record["rerank_existing_only"]["travel_minutes"]
    responses = [action["response"] for action in record["downstream_actions"] if action["response"]]
    x_response = next(row for row in responses if "x" in row["plan"]["sequence"])
    assert x_response["travel_minutes"] == 26
    assert x_response["disruption"]["weighted_edit_cost"] == 7
    assert c["disruption"]["secondary_cost"] == 0


def test_expanded_actions_fix_coverage_but_are_not_a_causal_advantage(experiment):
    record = experiment["weather_two_stops"]
    assert record["strategies"]["progressive_repair"] is None
    c = record["strategies"]["response_aware"]
    assert c is not None and c["evaluation"]["eligible"]
    assert not {"a", "o"}.intersection(c["plan"]["sequence"])
    assert c["travel_minutes"] == 58  # Preservation can deliberately cost travel.


def test_locked_booked_and_preferred_preserved_in_all_model_feasible_rows(experiment):
    for record in experiment.values():
        for row in record["strategies"].values():
            if row is None:
                continue
            assert row["evaluation"]["eligible"], row["evaluation"]["modeled_failures"]
            assert {"l", "b", "p"}.issubset(row["plan"]["sequence"])
            booked = next(stop for stop in row["plan"]["selected_stops"] if stop["stop_id"] == "b")
            assert booked["time_window_constraint"]["latest_departure"] == "10:45"
            assert row["plan"]["owned_constraints"] == record["parent"]["owned_constraints"]
    assert all(row is None for row in experiment["booked_infeasible"]["strategies"].values())


def test_master_rejects_booked_deletion_without_permission():
    parent, request, matrix, _, _ = toy_inputs(ToyScenario("booked_master"))
    model = build_repair_master_model(
        parent, request, build_repair_neighborhood(parent, request, RepairRadius.FULL_REOPTIMIZATION), matrix
    )
    sequence = tuple(sid for sid in parent.sequence if sid != "b")
    solution = RepairSolution(
        selected_stop_ids=sequence, day_assignments=dict.fromkeys(sequence, 1), lodging_assignments={"1": "hotel"}
    )
    assert "booked_select_b:b" in model.validate_solution(solution)


def test_lexicographic_preservation_precedes_utility_and_travel(experiment):
    b = deepcopy(experiment["weather_choice"]["strategies"]["progressive_repair"])
    worse = deepcopy(b)
    worse["action_id"] = "tempting_but_disruptive"
    worse["disruption"]["weighted_edit_cost"] += 1
    worse["travel_minutes"], worse["utility"] = 0, 100
    assert select_response([worse, b])["action_id"] == b["action_id"]
    worse["disruption"]["weighted_edit_cost"] = 0
    worse["evaluation"]["eligible"] = False
    assert select_response([worse, b])["action_id"] == b["action_id"]
    assert select_response([]) is None


def test_synthetic_route_certificate_remains_failed(experiment):
    for record in experiment.values():
        for row in record["strategies"].values():
            if row:
                cert = row["evaluation"]["certificate"]
                assert cert["evaluation_status"] == "FAILED"
                assert cert["comparison_eligibility"] == "ineligible"
                assert {finding["code"] for finding in cert["failures"]} == {"route_matrix_not_publication_ready"}
    parent, _, matrix, config, _ = toy_inputs(SCENARIOS[0])
    first_key = next(iter(matrix.cells))
    cells = {**matrix.cells, first_key: replace(matrix.cells[first_key], road_validated=True)}
    with pytest.raises(ValueError, match="unvalidated toy"):
        SyntheticModelEvaluator(parent, replace(matrix, cells=cells), config)


def test_reproduction_and_small_perturbation(experiment, tmp_path):
    repeated = run_toy_scenario(SCENARIOS[1], tmp_path / "repeat")
    for method, row in repeated["strategies"].items():
        original = experiment["travel_after"]["strategies"][method]
        assert (row["plan"]["sequence"], row["disruption"], row["travel_minutes"]) == (
            original["plan"]["sequence"],
            original["disruption"],
            original["travel_minutes"],
        )
    low = run_toy_scenario(ToyScenario("epsilon_low", 4.9), tmp_path / "low")
    high = run_toy_scenario(ToyScenario("epsilon_high", 5.1), tmp_path / "high")
    assert (
        low["strategies"]["full_reoptimization"]["plan"]["sequence"]
        != high["strategies"]["full_reoptimization"]["plan"]["sequence"]
    )
    for method in ("progressive_repair", "response_aware"):
        assert low["strategies"][method]["plan"]["sequence"] == high["strategies"][method]["plan"]["sequence"]
