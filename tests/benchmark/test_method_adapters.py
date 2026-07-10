from __future__ import annotations

import json
from pathlib import Path

from itinerary_system.benchmark import (
    build_pipeline_benchmark_method_adapter,
    generate_disruption_scenarios,
    run_benchmark_suite,
)
from itinerary_system.evaluation import PlanEvaluationCertificate
from itinerary_system.pipeline_runner import PipelineExecutionResult
from itinerary_system.research_artifacts import PlanArtifactV2, PlannerRun


def write_config(path: Path) -> Path:
    path.write_text(
        """
data:
  catalog_snapshot_id: catalog_benchmark
  context_snapshot_id: context_benchmark
enrichment:
  run_live_apis: false
""".strip(),
        encoding="utf-8",
    )
    return path


def parent_plan() -> PlanArtifactV2:
    return PlanArtifactV2(
        plan_id="parent_pipeline_benchmark",
        source_run_id="run_parent_pipeline_benchmark",
        planning_request_id="request_parent_pipeline_benchmark",
        catalog_snapshot_id="catalog_benchmark",
        context_snapshot_id="context_benchmark",
        selected_stops=(
            {"stop_id": "poi_a", "name": "Museum", "day": 1, "stop_order": 1, "lodging_id": "hotel_a"},
            {
                "stop_id": "poi_b",
                "name": "Weather Bridge",
                "day": 2,
                "stop_order": 1,
                "lodging_id": "hotel_b",
                "weather_sensitivity": 0.9,
                "weather_risk": 0.5,
            },
        ),
        day_assignments={"poi_a": 1, "poi_b": 2},
        sequence=("poi_a", "poi_b"),
        lodging_assignments={"1": "hotel_a", "2": "hotel_b"},
        ordered_days=({"day": 1, "stop_ids": ("poi_a",)}, {"day": 2, "stop_ids": ("poi_b",)}),
        route_ids_by_day={1: "route_1", 2: "route_2"},
        created_at="2026-07-09T00:00:00+00:00",
    )


def child_plan(plan_id: str, parent_plan_id: str) -> PlanArtifactV2:
    return PlanArtifactV2(
        plan_id=plan_id,
        parent_plan_id=parent_plan_id,
        source_run_id=f"run_{plan_id}",
        planning_request_id="request_pipeline_child",
        catalog_snapshot_id="catalog_benchmark",
        context_snapshot_id="context_benchmark",
        selected_stops=(
            {"stop_id": "poi_a", "name": "Museum", "day": 1, "stop_order": 1, "lodging_id": "hotel_a"},
            {"stop_id": "poi_c", "name": "Indoor Gallery", "day": 2, "stop_order": 1, "lodging_id": "hotel_b"},
        ),
        day_assignments={"poi_a": 1, "poi_c": 2},
        sequence=("poi_a", "poi_c"),
        lodging_assignments={"1": "hotel_a", "2": "hotel_b"},
        ordered_days=({"day": 1, "stop_ids": ("poi_a",)}, {"day": 2, "stop_ids": ("poi_c",)}),
        route_ids_by_day={1: "route_1", 2: "route_2"},
        created_at="2026-07-09T00:00:00+00:00",
    )


def certificate(plan: PlanArtifactV2) -> PlanEvaluationCertificate:
    return PlanEvaluationCertificate(
        certificate_id=f"cert_{plan.plan_id}",
        plan_id=plan.plan_id,
        source_run_id=plan.source_run_id,
        plan_content_hash=plan.content_hash,
        evaluator_version="benchmark-adapter-test",
        artifact_grounding_status="PASSED",
        hard_feasibility_status="PASSED",
        evaluation_status="PASSED",
        comparison_eligibility="eligible",
        metrics={"utility_retained": 0.91, "weather_risk_delta": -0.3},
    )


def planner_run(plan: PlanArtifactV2) -> PlannerRun:
    return PlannerRun(
        run_id=plan.source_run_id,
        planning_request_id=plan.planning_request_id,
        catalog_snapshot_id=plan.catalog_snapshot_id,
        context_snapshot_id=plan.context_snapshot_id,
        planner_specification_id="benchmark-adapter-test",
        method_requested="progressive_sequential_lexicographic_repair",
        method_executed="progressive_sequential_lexicographic_repair",
        execution_status="COMPLETED",
        solver_certification="FEASIBILITY_CERTIFIED",
        result_plan_id=plan.plan_id,
        runtime_seconds=0.42,
    )


def test_pipeline_benchmark_method_adapter_runs_pipeline_and_exposes_artifacts(tmp_path: Path):
    scenario = generate_disruption_scenarios(parent_plan(), seed=5)[0]
    seen_contexts = []

    def executor_factory(bound_scenario):
        assert bound_scenario.scenario_id == scenario.scenario_id

        def executor(context):
            seen_contexts.append(context)
            plan = child_plan(f"child_{context.run_id}", context.parent_plan_id or scenario.parent_plan_id)
            return PipelineExecutionResult(
                planner_runs=(planner_run(plan),),
                output_plans=(plan,),
                evaluations=(certificate(plan),),
                parent_plan={"plan_id": scenario.parent_plan_id},
                diff_records=(
                    {
                        "diff_id": f"diff_{context.run_id}",
                        "parent_plan_id": scenario.parent_plan_id,
                        "child_plan_id": plan.plan_id,
                        "weighted_edit_cost": 4.0,
                        "unchanged_days": [1],
                    },
                ),
                explanation_records=(
                    {
                        "evidence_id": f"explain_{context.run_id}",
                        "claims": [{"claim_id": "claim_adapter"}],
                    },
                ),
                request_records=(scenario.request.confirmed_constraints,),
                metrics={"repair_attempt_count": 1, "runtime_seconds": 0.42},
            )

        return executor

    method = build_pipeline_benchmark_method_adapter(
        method_id="progressive_sequential_lexicographic_repair",
        config_path=write_config(tmp_path / "config.yaml"),
        output_root=tmp_path / "pipeline_runs",
        executor_factory=executor_factory,
        strict=False,
    )

    result = run_benchmark_suite(
        scenarios=(scenario,),
        methods=(method,),
        output_dir=tmp_path / "benchmark",
        benchmark_id="bench_pipeline_adapter",
    )

    assert len(seen_contexts) == 1
    context = seen_contexts[0]
    assert context.catalog_snapshot_id == scenario.catalog_snapshot_id
    assert context.context_snapshot_id == scenario.context_snapshot_id
    assert context.parent_plan_id == scenario.parent_plan_id
    assert context.repair_request_id == scenario.request.request_id

    row = json.loads(result.metrics_path.read_text(encoding="utf-8").splitlines()[0])
    assert row["method_id"] == "progressive_sequential_lexicographic_repair"
    assert row["status"] == "completed"
    assert row["preservation_weighted_edit_cost"] == 4
    assert row["quality_utility_retained"] == 0.91
    assert row["quality_weather_risk_delta"] == -0.3
    assert row["computation_repair_attempt_count"] == 1
    assert row["certificate_comparison_eligibility"] == "eligible"
    assert row["explanation_count"] == 1

    run_record = result.run_records[0].to_record()
    assert Path(run_record["manifest_path"]).exists()
    assert Path(run_record["metrics_path"]).exists()
