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
