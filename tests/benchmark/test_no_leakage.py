from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from itinerary_system.benchmark import (
    BenchmarkLeakageError,
    BenchmarkMethodAdapter,
    assert_no_parent_family_leakage,
    benchmark_split_key,
    extract_benchmark_metrics,
    generate_disruption_scenarios,
    run_benchmark_suite,
    split_by_parent_disruption_family,
)
from itinerary_system.research_artifacts import PlanArtifactV2


def parent_plan(plan_id: str = "parent_benchmark") -> PlanArtifactV2:
    return PlanArtifactV2(
        plan_id=plan_id,
        source_run_id=f"run_{plan_id}",
        planning_request_id=f"request_{plan_id}",
        catalog_snapshot_id="catalog_benchmark",
        context_snapshot_id="context_benchmark",
        selected_stops=(
            {
                "stop_id": "poi_museum",
                "name": "Museum",
                "day": 1,
                "stop_order": 1,
                "lodging_id": "hotel_a",
                "weather_sensitivity": 0.1,
            },
            {
                "stop_id": "poi_bridge",
                "name": "Bridge",
                "day": 2,
                "stop_order": 1,
                "lodging_id": "hotel_b",
                "weather_sensitivity": 0.9,
                "weather_risk": 0.4,
                "route_id": "route_2",
            },
            {
                "stop_id": "poi_grove",
                "name": "Grove",
                "day": 2,
                "stop_order": 2,
                "lodging_id": "hotel_b",
                "weather_sensitivity": 0.8,
                "weather_risk": 0.2,
                "route_id": "route_2",
            },
        ),
        day_assignments={"poi_museum": 1, "poi_bridge": 2, "poi_grove": 2},
        sequence=("poi_museum", "poi_bridge", "poi_grove"),
        lodging_assignments={"1": "hotel_a", "2": "hotel_b"},
        ordered_days=({"day": 1, "stop_ids": ("poi_museum",)}, {"day": 2, "stop_ids": ("poi_bridge", "poi_grove")}),
        route_ids_by_day={1: "route_1", 2: "route_2"},
        created_at="2026-07-09T00:00:00+00:00",
    )


def test_split_validation_rejects_same_parent_family_across_splits():
    weather_seed_1 = generate_disruption_scenarios(parent_plan(), seed=1)[0]
    weather_seed_2 = generate_disruption_scenarios(parent_plan(), seed=2)[0]

    with pytest.raises(BenchmarkLeakageError) as excinfo:
        assert_no_parent_family_leakage({"train": (weather_seed_1,), "test": (weather_seed_2,)})

    assert "parent_benchmark" in str(excinfo.value)
    assert "weather_deterioration" in str(excinfo.value)


def test_split_by_parent_disruption_family_keeps_groups_together():
    scenarios = (
        *generate_disruption_scenarios(parent_plan("parent_a"), seed=3),
        *generate_disruption_scenarios(parent_plan("parent_a"), seed=4),
        *generate_disruption_scenarios(parent_plan("parent_b"), seed=3),
    )

    splits = split_by_parent_disruption_family(scenarios, split_names=("train", "development", "test"), seed=19)

    assert set(splits) == {"train", "development", "test"}
    assert sorted(scenario.scenario_id for split in splits.values() for scenario in split) == sorted(
        scenario.scenario_id for scenario in scenarios
    )
    assert_no_parent_family_leakage(splits)

    split_for_key: dict[str, str] = {}
    for split_name, split_scenarios in splits.items():
        for scenario in split_scenarios:
            key = benchmark_split_key(scenario)
            if key in split_for_key:
                assert split_for_key[key] == split_name
            else:
                split_for_key[key] = split_name


def test_run_benchmark_suite_pairs_methods_on_identical_inputs_and_exports_metrics(tmp_path: Path):
    scenarios = generate_disruption_scenarios(parent_plan(), seed=7)[:2]
    seen_inputs: dict[tuple[str, str], dict[str, Any]] = {}

    def runner_for(method_id: str, *, weighted_edit_cost: float, runtime_seconds: float):
        def run(scenario):
            seen_inputs[(method_id, scenario.scenario_id)] = scenario.to_record()
            return {
                "run_id": f"{method_id}_{scenario.scenario_id}",
                "status": "completed",
                "output_plans": ({"plan_id": f"plan_{method_id}_{scenario.scenario_id}"},),
                "diff_records": (
                    {
                        "diff_id": f"diff_{method_id}_{scenario.scenario_id}",
                        "weighted_edit_cost": weighted_edit_cost,
                        "unchanged_days": [1],
                    },
                ),
                "evaluations": (
                    {
                        "certificate_id": f"cert_{method_id}_{scenario.scenario_id}",
                        "comparison_eligibility": "eligible",
                        "evaluation_status": "PASSED",
                        "metrics": {
                            "utility_retained": 0.85,
                            "weather_risk_delta": -0.2,
                        },
                    },
                ),
                "explanation_records": (
                    {
                        "evidence_id": f"explain_{method_id}_{scenario.scenario_id}",
                        "claims": [{"claim_id": "claim_supported"}],
                    },
                ),
                "metrics": {
                    "runtime_seconds": runtime_seconds,
                    "repair_attempt_count": 2,
                    "fallback_used": False,
                },
            }

        return run

    methods = (
        BenchmarkMethodAdapter(
            method_id="full_reoptimization",
            runner=runner_for("full_reoptimization", weighted_edit_cost=8.0, runtime_seconds=0.5),
            method_family="baseline",
            baseline=True,
        ),
        BenchmarkMethodAdapter(
            method_id="progressive_sequential_lexicographic_repair",
            runner=runner_for(
                "progressive_sequential_lexicographic_repair",
                weighted_edit_cost=3.0,
                runtime_seconds=0.2,
            ),
        ),
    )

    result = run_benchmark_suite(
        scenarios=scenarios,
        methods=methods,
        output_dir=tmp_path / "benchmark",
        benchmark_id="bench_002_tdd",
    )

    rows = [json.loads(line) for line in result.metrics_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == len(scenarios) * len(methods)
    assert {row["scenario_id"] for row in rows} == {scenario.scenario_id for scenario in scenarios}
    assert {row["method_id"] for row in rows} == {method.method_id for method in methods}
    assert {row["catalog_snapshot_id"] for row in rows} == {"catalog_benchmark"}
    assert {row["context_snapshot_id"] for row in rows} == {"context_benchmark"}

    first_row = rows[0]
    assert first_row["benchmark_id"] == "bench_002_tdd"
    assert first_row["preservation_weighted_edit_cost"] in {3.0, 8.0}
    assert first_row["preservation_unchanged_day_count"] == 1
    assert first_row["quality_utility_retained"] == 0.85
    assert first_row["quality_weather_risk_delta"] == -0.2
    assert first_row["computation_repair_attempt_count"] == 2
    assert first_row["certificate_comparison_eligibility"] == "eligible"
    assert first_row["certificate_evaluation_status"] == "PASSED"
    assert first_row["explanation_count"] == 1

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["benchmark_id"] == "bench_002_tdd"
    assert manifest["scenario_ids"] == [scenario.scenario_id for scenario in scenarios]
    assert manifest["method_ids"] == [method.method_id for method in methods]
    assert Path(manifest["metrics_path"]).name == "benchmark_metrics.jsonl"

    for scenario in scenarios:
        baseline_input = seen_inputs[("full_reoptimization", scenario.scenario_id)]
        repair_input = seen_inputs[("progressive_sequential_lexicographic_repair", scenario.scenario_id)]
        assert baseline_input["scenario_id"] == repair_input["scenario_id"] == scenario.scenario_id
        assert baseline_input["catalog_snapshot_id"] == repair_input["catalog_snapshot_id"]
        assert baseline_input["context_snapshot_id"] == repair_input["context_snapshot_id"]
        assert baseline_input["parent_plan_id"] == repair_input["parent_plan_id"]
        assert baseline_input["request"]["confirmed_constraints"] == repair_input["request"]["confirmed_constraints"]


def test_quality_metrics_are_owned_only_by_independent_evaluator():
    scenario = generate_disruption_scenarios(parent_plan(), seed=11)[0]
    planner_only = extract_benchmark_metrics(
        {
            "metrics": {
                "utility_retained": 999,
                "utility_regret": -999,
                "weather_risk_delta": -999,
            },
            "evaluations": (
                {
                    "comparison_eligibility": "eligible",
                    "evaluation_status": "PASSED",
                    "route_validation": {"road_validated": True},
                },
            ),
        },
        scenario=scenario,
    )

    assert planner_only["quality_utility_retained"] is None
    assert planner_only["quality_utility_regret"] is None
    assert planner_only["quality_weather_risk_delta"] is None
    assert planner_only["quality_metric_owner"] == "independent_evaluator"
    assert planner_only["quality_metrics_present"] is False
    assert planner_only["benchmark_ranking_eligible"] is False

    malformed = extract_benchmark_metrics(
        {
            "evaluations": (
                {
                    "comparison_eligibility": "eligible",
                    "evaluation_status": "PASSED",
                    "route_validation": {"road_validated": True},
                    "metrics": {"utility_retained": "not-a-number"},
                },
            ),
        },
        scenario=scenario,
    )
    assert malformed["quality_metrics_present"] is False
    assert malformed["benchmark_ranking_eligible"] is False

    evaluated = extract_benchmark_metrics(
        {
            "metrics": {"utility_retained": 999},
            "evaluations": (
                {
                    "comparison_eligibility": "eligible",
                    "evaluation_status": "PASSED",
                    "route_validation": {"road_validated": True},
                    "metrics": {"utility_retained": 0.8, "weather_risk_delta": -0.1},
                },
            ),
        },
        scenario=scenario,
    )

    assert evaluated["quality_utility_retained"] == 0.8
    assert evaluated["quality_weather_risk_delta"] == -0.1
    assert evaluated["quality_route_validated"] is True
    assert evaluated["quality_metrics_present"] is True
    assert evaluated["benchmark_ranking_eligible"] is True


def test_runner_rejects_duplicate_method_ids_before_execution(tmp_path: Path):
    scenario = generate_disruption_scenarios(parent_plan(), seed=12)[0]
    method = BenchmarkMethodAdapter(method_id="duplicate", runner=lambda _: {})

    with pytest.raises(ValueError, match="method IDs must be unique"):
        run_benchmark_suite(
            scenarios=(scenario,),
            methods=(method, method),
            output_dir=tmp_path / "duplicates",
        )


def test_publication_mode_requires_exact_four_methods(tmp_path: Path):
    scenario = generate_disruption_scenarios(parent_plan(), seed=13)[0]
    incomplete = (BenchmarkMethodAdapter(method_id="full_reoptimization", runner=lambda _: {}),)

    with pytest.raises(ValueError, match="missing required methods"):
        run_benchmark_suite(
            scenarios=(scenario,),
            methods=incomplete,
            output_dir=tmp_path / "incomplete",
            publication_mode=True,
        )


def test_publication_manifest_blocks_unvalidated_routes(tmp_path: Path):
    scenario = generate_disruption_scenarios(parent_plan(), seed=14)[0]

    def unvalidated_result(_):
        return {
            "status": "completed",
            "evaluations": (
                {
                    "comparison_eligibility": "eligible",
                    "evaluation_status": "PASSED",
                    "route_validation": {"road_validated": False},
                    "metrics": {"utility_retained": 0.9},
                },
            ),
        }

    method_ids = (
        "context_blind_solver",
        "deterministic_context_aware_heuristic",
        "progressive_sequential_lexicographic_repair",
        "full_reoptimization",
    )
    result = run_benchmark_suite(
        scenarios=(scenario,),
        methods=tuple(BenchmarkMethodAdapter(method_id=method_id, runner=unvalidated_result) for method_id in method_ids),
        output_dir=tmp_path / "publication",
        publication_mode=True,
    )

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    readiness = manifest["publication_readiness"]
    assert manifest["publication_mode"] is True
    assert readiness["complete"] is True
    assert readiness["ranking_eligible_run_count"] == 0
    assert readiness["publication_ready"] is False


def test_publication_manifest_requires_method_specific_planner_provenance(tmp_path: Path):
    scenario = generate_disruption_scenarios(parent_plan(), seed=15)[0]
    method_ids = (
        "context_blind_solver",
        "deterministic_context_aware_heuristic",
        "progressive_sequential_lexicographic_repair",
        "full_reoptimization",
    )

    def result_for(method_id: str):
        def run(_):
            return {
                "status": "completed",
                "planner_runs": (
                    {"method_requested": method_id, "method_executed": method_id},
                ),
                "evaluations": (
                    {
                        "comparison_eligibility": "eligible",
                        "evaluation_status": "PASSED",
                        "route_validation": {"road_validated": True},
                        "metrics": {"utility_retained": 0.9},
                    },
                ),
                "route_records": (
                    {
                        "matrix_id": "frozen_matrix",
                        "source_bundle_id": "frozen_bundle",
                        "source_content_sha256": "a" * 64,
                        "cells": {"a-b": 1},
                    },
                ),
            }

        return run

    result = run_benchmark_suite(
        scenarios=(scenario,),
        methods=tuple(
            BenchmarkMethodAdapter(method_id=method_id, runner=result_for(method_id))
            for method_id in method_ids
        ),
        output_dir=tmp_path / "provenance",
        publication_mode=True,
    )
    rows = [json.loads(line) for line in result.metrics_path.read_text(encoding="utf-8").splitlines()]
    readiness = json.loads(result.manifest_path.read_text(encoding="utf-8"))["publication_readiness"]

    assert all(row["benchmark_method_provenance_valid"] for row in rows)
    assert all(row["benchmark_ranking_eligible"] for row in rows)
    assert readiness["publication_ready"] is True


def test_publication_manifest_rejects_relabelled_method_implementation(tmp_path: Path):
    scenario = generate_disruption_scenarios(parent_plan(), seed=16)[0]
    method_ids = (
        "context_blind_solver",
        "deterministic_context_aware_heuristic",
        "progressive_sequential_lexicographic_repair",
        "full_reoptimization",
    )

    def relabelled_result(_):
        return {
            "status": "completed",
            "planner_runs": (
                {"method_requested": "full_reoptimization", "method_executed": "full_reoptimization"},
            ),
            "evaluations": (
                {
                    "comparison_eligibility": "eligible",
                    "evaluation_status": "PASSED",
                    "route_validation": {"road_validated": True},
                    "metrics": {"utility_retained": 0.9},
                },
            ),
        }

    result = run_benchmark_suite(
        scenarios=(scenario,),
        methods=tuple(BenchmarkMethodAdapter(method_id=method_id, runner=relabelled_result) for method_id in method_ids),
        output_dir=tmp_path / "relabelled",
        publication_mode=True,
    )
    rows = [json.loads(line) for line in result.metrics_path.read_text(encoding="utf-8").splitlines()]
    readiness = json.loads(result.manifest_path.read_text(encoding="utf-8"))["publication_readiness"]

    assert sum(bool(row["benchmark_method_provenance_valid"]) for row in rows) == 1
    assert readiness["publication_ready"] is False


def test_nonfinite_quality_metrics_and_unknown_status_are_not_rankable():
    scenario = generate_disruption_scenarios(parent_plan(), seed=17)[0]

    for value in (float("nan"), float("inf"), float("-inf")):
        metrics = extract_benchmark_metrics(
            result={
                "evaluations": (
                    {
                        "comparison_eligibility": "eligible",
                        "evaluation_status": "PASSED",
                        "route_validation": {"road_validated": True},
                        "metrics": {"utility_retained": value},
                    },
                ),
            },
            scenario=scenario,
            runtime_seconds=0.1,
        )
        assert metrics["quality_utility_retained"] is None
        assert metrics["benchmark_ranking_eligible"] is False

    unknown_status = extract_benchmark_metrics(
        result={
            "evaluations": (
                {
                    "comparison_eligibility": "eligible",
                    "evaluation_status": "UNKNOWN",
                    "route_validation": {"road_validated": True},
                    "metrics": {"utility_retained": 0.9},
                },
            ),
        },
        scenario=scenario,
        runtime_seconds=0.1,
    )
    assert unknown_status["benchmark_ranking_eligible"] is False


def test_preservation_metrics_never_fall_back_to_planner_owned_values():
    scenario = generate_disruption_scenarios(parent_plan(), seed=18)[0]

    no_diff = extract_benchmark_metrics(
        result={"metrics": {"weighted_edit_cost": 99, "unchanged_days": [1, 2]}},
        scenario=scenario,
        runtime_seconds=0.1,
    )
    assert no_diff["preservation_weighted_edit_cost"] is None
    assert no_diff["preservation_unchanged_day_count"] == 0
    assert no_diff["preservation_metric_owner"] is None

    explicit_empty_diff = extract_benchmark_metrics(
        result={"diff_records": ({"unchanged_days": []},)},
        scenario=scenario,
        runtime_seconds=0.1,
    )
    assert explicit_empty_diff["preservation_weighted_edit_cost"] is None
    assert explicit_empty_diff["preservation_unchanged_day_count"] == 0
    assert explicit_empty_diff["preservation_metric_owner"] == "plan_diff"