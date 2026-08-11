from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd
import pytest

from itinerary_system.evaluation import PlanEvaluationCertificate
from itinerary_system.explanation import EvidenceRecord, ExplanationClaim, WhyEvidence
from itinerary_system.phase0_exporter import PHASE0_ARTIFACT_FILES
from itinerary_system.pipeline_runner import (
    PipelineExecutionResult,
    PipelineStrictModeError,
    RefreshPolicy,
    RunDirectoryExists,
    build_phase0_generation_executor,
    build_production_generation_executor,
    build_progressive_repair_executor,
    run_research_pipeline,
)
from itinerary_system.repair import DayRouteSolverConfig, RepairRadius
from itinerary_system.research_artifacts import PlanArtifactV2, PlannerRun
from itinerary_system.routing import RouteMatrix, RouteMatrixCell


def write_config(path):
    path.write_text(
        """
data:
  catalog_snapshot_id: catalog_pipe
  context_snapshot_id: context_pipe
enrichment:
  run_live_apis: true
  use_yelp_live_api: true
credentials:
  api_key: sk-test-secret
  nested:
    access_token: token-secret
""".strip(),
        encoding="utf-8",
    )
    return path


def planner_run(run_id: str = "planner_pipe") -> PlannerRun:
    return PlannerRun(
        run_id=run_id,
        planning_request_id="request_pipe",
        catalog_snapshot_id="catalog_pipe",
        context_snapshot_id="context_pipe",
        planner_specification_id="pipeline-test",
        method_requested="generation",
        method_executed="generation",
        execution_status="COMPLETED",
        solver_certification="FEASIBILITY_CERTIFIED",
        result_plan_id="plan_pipe",
    )


def plan(plan_id: str = "plan_pipe", *, parent_plan_id: str | None = None) -> PlanArtifactV2:
    return PlanArtifactV2(
        plan_id=plan_id,
        parent_plan_id=parent_plan_id,
        source_run_id="planner_pipe",
        planning_request_id="request_pipe",
        catalog_snapshot_id="catalog_pipe",
        context_snapshot_id="context_pipe",
        selected_stops=(
            {"stop_id": "poi_a", "name": "Museum", "day": 1, "stop_order": 1},
            {"stop_id": "poi_b", "name": "Park", "day": 1, "stop_order": 2},
        ),
        day_assignments={"poi_a": 1, "poi_b": 1},
        sequence=("poi_a", "poi_b"),
        ordered_days=({"day": 1, "stop_ids": ("poi_a", "poi_b")},),
        created_at="2026-07-09T00:00:00+00:00",
    )


def certificate(plan_id: str = "plan_pipe", *, eligible: bool = True) -> PlanEvaluationCertificate:
    target_plan = plan(plan_id)
    return PlanEvaluationCertificate(
        certificate_id=f"cert_{plan_id}",
        plan_id=plan_id,
        source_run_id=target_plan.source_run_id,
        plan_content_hash=target_plan.content_hash,
        evaluator_version="test-evaluator",
        artifact_grounding_status="PASSED",
        hard_feasibility_status="PASSED" if eligible else "FAILED",
        evaluation_status="PASSED" if eligible else "FAILED",
        comparison_eligibility="eligible" if eligible else "ineligible",
        metrics={"selected_stop_count": 2.0},
    )


def route_matrix() -> RouteMatrix:
    return RouteMatrix(
        matrix_id="matrix_pipe",
        context_snapshot_id="context_pipe",
        entity_ids=(),
        cells={
            ("poi_a", "poi_b"): RouteMatrixCell(
                origin_id="poi_a",
                destination_id="poi_b",
                distance_m=1000.0,
                duration_s=600.0,
                provider="unit",
                road_validated=True,
            )
        },
    )


def phase0_method_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "method": "hierarchical_bandit_gurobi_repair",
                "method_display_name": "Hierarchical + Bandit + Small Gurobi Repair",
                "comparison_label": "Method - Hierarchical + Bandit + Small Gurobi Repair",
                "trip_days": 7,
                "status": "FEASIBLE",
                "local_route_solver": "small_gurobi",
                "objective": 12.5,
                "solve_seconds": 0.2,
                "total_utility": 42.0,
                "total_travel_time": 120.0,
                "total_travel_distance_km": 12.0,
                "selected_attractions": 2,
            }
        ]
    )


def phase0_route_stops_frame() -> pd.DataFrame:
    common = {
        "comparison_type": "method",
        "comparison_label": "Method - Hierarchical + Bandit + Small Gurobi Repair",
        "method": "hierarchical_bandit_gurobi_repair",
        "trip_days": 7,
        "day": 1,
        "route_start_name": "SFO",
        "route_start_latitude": 37.6213,
        "route_start_longitude": -122.3790,
        "route_end_name": "Optimizer Hotel",
        "route_end_latitude": 37.7749,
        "route_end_longitude": -122.4194,
        "drive_time_source": "geodesic_proxy",
        "source_confidence": 0.7,
    }
    return pd.DataFrame(
        [
            {
                **common,
                "stop_order": 1,
                "attraction_name": "Golden Gate Bridge",
                "city": "San Francisco",
                "latitude": 37.8199,
                "longitude": -122.4783,
            },
            {
                **common,
                "stop_order": 2,
                "attraction_name": "Ferry Building",
                "city": "San Francisco",
                "latitude": 37.7955,
                "longitude": -122.3937,
            },
        ]
    )


def progressive_parent_plan() -> PlanArtifactV2:
    return PlanArtifactV2(
        plan_id="parent_progressive_pipe",
        source_run_id="run_parent_pipe",
        planning_request_id="request_parent_pipe",
        catalog_snapshot_id="catalog_pipe",
        context_snapshot_id="context_pipe",
        selected_stops=(
            {"stop_id": "poi_a", "name": "Museum", "day": 1, "stop_order": 1, "lodging_id": "hotel_sf"},
            {
                "stop_id": "poi_b",
                "name": "Bridge",
                "day": 2,
                "stop_order": 1,
                "lodging_id": "hotel_y",
                "visit_duration_minutes": 30,
                "opening_start": "09:00",
                "opening_end": "12:00",
            },
            {
                "stop_id": "poi_c",
                "name": "Closed Grove",
                "day": 2,
                "stop_order": 2,
                "lodging_id": "hotel_y",
                "visit_duration_minutes": 40,
                "opening_start": "09:30",
                "opening_end": "13:00",
                "closed": True,
            },
            {"stop_id": "poi_d", "name": "Coast", "day": 3, "stop_order": 1, "lodging_id": "hotel_m"},
        ),
        day_assignments={"poi_a": 1, "poi_b": 2, "poi_c": 2, "poi_d": 3},
        sequence=("poi_a", "poi_b", "poi_c", "poi_d"),
        lodging_assignments={"1": "hotel_sf", "2": "hotel_y", "3": "hotel_m"},
        route_ids_by_day={1: "route_1", 2: "route_2", 3: "route_3"},
        created_at="2026-07-09T00:00:00+00:00",
    )


def progressive_repair_request() -> SimpleNamespace:
    return SimpleNamespace(
        request_id="repair_progressive_pipe",
        parent_plan_id="parent_progressive_pipe",
        allowed_radii=(RepairRadius.SAME_DAY_REPLACEMENT,),
        confirmed_constraints={"affected_days": (2,)},
        candidate_pois=(
            {
                "stop_id": "poi_e",
                "name": "Indoor Aquarium",
                "day": 2,
                "stop_order": 2,
                "lodging_id": "hotel_y",
                "visit_duration_minutes": 35,
                "opening_start": "09:00",
                "opening_end": "14:00",
            },
        ),
    )


def progressive_day_route_config() -> DayRouteSolverConfig:
    return DayRouteSolverConfig(
        max_day_minutes=240,
        day_start_time="09:00",
        default_visit_minutes=30,
        start_anchor_by_day={2: "hotel_y_start"},
        end_anchor_by_day={2: "hotel_y_end"},
        strict_route_matrix=True,
    )


def progressive_route_matrix() -> RouteMatrix:
    def progressive_cell(origin: str, destination: str, minutes: float) -> RouteMatrixCell:
        return RouteMatrixCell(
            origin_id=origin,
            destination_id=destination,
            distance_m=minutes * 1000.0,
            duration_s=minutes * 60.0,
            provider="unit",
            road_validated=True,
        )

    cells = {
        ("hotel_y_start", "poi_b"): progressive_cell("hotel_y_start", "poi_b", 10),
        ("poi_b", "poi_c"): progressive_cell("poi_b", "poi_c", 15),
        ("poi_c", "hotel_y_end"): progressive_cell("poi_c", "hotel_y_end", 20),
        ("hotel_y_start", "poi_e"): progressive_cell("hotel_y_start", "poi_e", 10),
        ("poi_e", "poi_c"): progressive_cell("poi_e", "poi_c", 14),
        ("poi_b", "poi_e"): progressive_cell("poi_b", "poi_e", 8),
        ("poi_e", "hotel_y_end"): progressive_cell("poi_e", "hotel_y_end", 12),
        ("poi_c", "poi_e"): progressive_cell("poi_c", "poi_e", 9),
    }
    return RouteMatrix(matrix_id="matrix_progressive_pipe", context_snapshot_id="context_pipe", entity_ids=(), cells=cells)


def explanation() -> WhyEvidence:
    record = EvidenceRecord(
        ref_id="evaluation:selected_stop_count",
        source_type="evaluation",
        source_id="cert_plan_pipe",
        field_path="metrics.selected_stop_count",
        payload={"value": 2.0},
    )
    return WhyEvidence(
        evidence_id="why_plan_pipe",
        plan_id="plan_pipe",
        target_id="plan_pipe",
        claims=(
            ExplanationClaim(
                claim_id="claim_stop_count",
                claim_type="numeric",
                text_template="The plan includes {count} selected stops.",
                values={"count": 2},
                evidence_refs=(record.ref_id,),
            ),
        ),
        evidence_records=(record,),
    )


def execution_result(*, eligible: bool = True, repair: bool = False) -> PipelineExecutionResult:
    parent = plan("parent_pipe") if repair else None
    child = plan("plan_pipe", parent_plan_id="parent_pipe") if repair else plan("plan_pipe")
    return PipelineExecutionResult(
        planner_runs=(planner_run(),),
        output_plans=(child,),
        evaluations=(certificate(child.plan_id, eligible=eligible),),
        parent_plan=parent,
        diff_records=({"diff_id": "diff_pipe", "parent_plan_id": "parent_pipe", "child_plan_id": child.plan_id},)
        if repair
        else (),
        route_records=(route_matrix(),),
        explanation_records=(explanation(),),
        request_records=({"request_id": "request_pipe", "kind": "repair" if repair else "generation"},),
        metrics={"output_plan_count": 1.0},
    )


def test_generation_pipeline_creates_run_layout_and_redacts_config(tmp_path):
    seen_contexts = []

    def executor(context):
        seen_contexts.append(context)
        assert context.mode == "generation"
        assert context.refresh_policy == RefreshPolicy.NEVER
        assert context.config.run_live_apis is False
        assert context.config.get("enrichment", "use_yelp_live_api") is False
        return execution_result()

    run = run_research_pipeline(
        config_path=write_config(tmp_path / "config.yaml"),
        catalog_snapshot_id="catalog_pipe",
        context_snapshot_id="context_pipe",
        output_root=tmp_path / "runs",
        run_id="run_pipe",
        executor=executor,
    )

    assert len(seen_contexts) == 1
    assert run.run_id == "run_pipe"
    assert run.output_dir == tmp_path / "runs" / "run_pipe"
    for name in ("requests", "plans", "diffs", "routing", "evaluations", "explanations", "metrics", "dashboard"):
        assert (run.output_dir / name).is_dir()
    assert (run.output_dir / "manifest.json").exists()
    assert (run.output_dir / "planner_runs.jsonl").read_text(encoding="utf-8").count("\n") == 1

    redacted_text = (run.output_dir / "resolved_config.redacted.json").read_text(encoding="utf-8")
    assert "sk-test-secret" not in redacted_text
    assert "token-secret" not in redacted_text
    redacted = json.loads(redacted_text)
    assert redacted["credentials"]["api_key"] == "***REDACTED***"
    assert redacted["credentials"]["nested"]["access_token"] == "***REDACTED***"

    manifest = json.loads((run.output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["run_id"] == "run_pipe"
    assert manifest["mode"] == "generation"
    assert manifest["status"] == "completed"
    assert manifest["artifact_counts"]["plans"] == 1
    assert manifest["artifacts"]["plans"] == ["plans/plan_pipe.json"]


def test_pipeline_refuses_to_overwrite_existing_run(tmp_path):
    kwargs = {
        "config_path": write_config(tmp_path / "config.yaml"),
        "catalog_snapshot_id": "catalog_pipe",
        "context_snapshot_id": "context_pipe",
        "output_root": tmp_path / "runs",
        "run_id": "run_pipe",
        "executor": lambda _context: execution_result(),
    }

    first = run_research_pipeline(**kwargs)

    with pytest.raises(RunDirectoryExists):
        run_research_pipeline(**kwargs)

    assert json.loads((first.output_dir / "manifest.json").read_text(encoding="utf-8"))["status"] == "completed"


def test_production_generation_executor_wraps_optimizer_outputs_as_pipeline_artifacts(tmp_path):
    calls = []

    def production_runner(**kwargs):
        calls.append(kwargs)
        assert kwargs["output_dir"] == tmp_path / "runs" / "production_run" / "production_legacy"
        assert kwargs["config"].run_live_apis is False
        assert kwargs["city_names"] == ["San Francisco"]
        return {
            "production_method_comparison_df": phase0_method_frame(),
            "production_method_route_stops_df": phase0_route_stops_frame(),
        }

    run = run_research_pipeline(
        config_path=write_config(tmp_path / "config.yaml"),
        catalog_snapshot_id="catalog_pipe",
        context_snapshot_id="context_pipe",
        output_root=tmp_path / "runs",
        run_id="production_run",
        executor=build_production_generation_executor(
            all_business_df=pd.DataFrame({"name": ["Golden Gate Bridge"]}),
            hotels_df=pd.DataFrame({"name": ["Optimizer Hotel"]}),
            city_names=("San Francisco",),
            production_runner=production_runner,
        ),
        strict=False,
    )

    assert len(calls) == 1
    assert run.status == "completed_with_warnings"
    assert (run.output_dir / "production_legacy" / "production_phase0_plan_artifacts.jsonl").exists()
    manifest = json.loads((run.output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["artifact_counts"]["plans"] == 1
    assert manifest["artifact_counts"]["evaluations"] == 1
    metrics = json.loads((run.output_dir / "metrics" / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["production_legacy_artifact_dir"] == "production_legacy"
    assert metrics["production_method_row_count"] == 1
    assert metrics["production_route_stop_row_count"] == 2



def test_production_executor_with_relative_output_root_does_not_duplicate_run_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    def production_runner(**_kwargs):
        return {
            "production_method_comparison_df": phase0_method_frame(),
            "production_method_route_stops_df": phase0_route_stops_frame(),
        }

    run = run_research_pipeline(
        config_path=write_config(tmp_path / "config.yaml"),
        catalog_snapshot_id="catalog_pipe",
        context_snapshot_id="context_pipe",
        output_root="runs",
        run_id="relative_production_run",
        executor=build_production_generation_executor(
            all_business_df=pd.DataFrame({"name": ["Golden Gate Bridge"]}),
            hotels_df=pd.DataFrame({"name": ["Optimizer Hotel"]}),
            city_names=("San Francisco",),
            production_runner=production_runner,
        ),
        strict=False,
    )

    physical_run_dir = tmp_path / run.output_dir
    assert (physical_run_dir / "production_legacy" / "production_phase0_plan_artifacts.jsonl").exists()
    assert not (physical_run_dir / run.output_dir / "production_legacy").exists()
    metrics = json.loads((physical_run_dir / "metrics" / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["phase0_legacy_artifact_dir"] == "production_legacy"
    assert metrics["production_legacy_artifact_dir"] == "production_legacy"

def test_repair_pipeline_exports_parent_child_diff_and_explanation(tmp_path):
    def executor(context):
        assert context.mode == "repair"
        assert context.parent_plan_id == "parent_pipe"
        assert context.repair_request_id == "repair_pipe"
        return execution_result(repair=True)

    run = run_research_pipeline(
        config_path=write_config(tmp_path / "config.yaml"),
        catalog_snapshot_id="catalog_pipe",
        context_snapshot_id="context_pipe",
        output_root=tmp_path / "runs",
        run_id="repair_run",
        parent_plan_id="parent_pipe",
        repair_request_id="repair_pipe",
        executor=executor,
    )

    assert run.parent_plan is not None
    assert (run.output_dir / "plans" / "parent_pipe.json").exists()
    assert (run.output_dir / "plans" / "plan_pipe.json").exists()
    assert (run.output_dir / "diffs" / "diff_pipe.json").exists()
    assert (run.output_dir / "explanations" / "why_plan_pipe.json").exists()
    manifest = json.loads((run.output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["mode"] == "repair"
    assert manifest["parent_plan_id"] == "parent_pipe"
    assert manifest["repair_request_id"] == "repair_pipe"


def test_repair_pipeline_marks_missing_output_as_failed_and_bounds_long_artifact_names(tmp_path):
    long_evidence_id = "repair_failure_" + "x" * 80
    failed_planner = PlannerRun(
        **{
            **planner_run("planner_failed").__dict__,
            "execution_status": "FAILED",
            "solver_certification": "NO_CERTIFICATE",
            "result_plan_id": None,
            "error_summary": "bounded failure",
        }
    )

    def executor(_context):
        return PipelineExecutionResult(
            planner_runs=(failed_planner,),
            parent_plan=plan("parent_pipe"),
            explanation_records=({"evidence_id": long_evidence_id, "status": "failed"},),
            metrics={"method_status": "failed"},
        )

    run = run_research_pipeline(
        config_path=write_config(tmp_path / "config.yaml"),
        catalog_snapshot_id="catalog_pipe",
        context_snapshot_id="context_pipe",
        output_root=tmp_path / "runs",
        run_id="failed_repair_run",
        parent_plan_id="parent_pipe",
        repair_request_id="repair_pipe",
        executor=executor,
        strict=True,
    )

    assert run.status == "failed"
    manifest = json.loads((run.output_dir / "manifest.json").read_text(encoding="utf-8"))
    metrics = json.loads((run.output_dir / "metrics" / "metrics.json").read_text(encoding="utf-8"))
    assert manifest["planner_failure_count"] == 1
    assert manifest["repair_output_missing"] is True
    assert metrics["planner_failure_count"] == 1
    assert metrics["repair_output_missing"] is True
    explanation_path = run.output_dir / manifest["artifacts"]["explanations"][0]
    assert explanation_path.exists()
    assert len(explanation_path.stem) <= 40
    explanation_record = json.loads(explanation_path.read_text(encoding="utf-8"))
    assert explanation_record["evidence_id"] == long_evidence_id


def test_strict_mode_writes_diagnostics_then_blocks_ineligible_plan(tmp_path):
    with pytest.raises(PipelineStrictModeError) as exc_info:
        run_research_pipeline(
            config_path=write_config(tmp_path / "config.yaml"),
            catalog_snapshot_id="catalog_pipe",
            context_snapshot_id="context_pipe",
            output_root=tmp_path / "runs",
            run_id="strict_run",
            executor=lambda _context: execution_result(eligible=False),
            strict=True,
        )

    run = exc_info.value.pipeline_run
    assert run is not None
    assert run.status == "failed_strict"
    assert (run.output_dir / "evaluations" / "cert_plan_pipe.json").exists()
    manifest = json.loads((run.output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed_strict"
    assert manifest["strict_failure_count"] == 1


def test_permissive_mode_records_ineligible_plan_without_raising(tmp_path):
    run = run_research_pipeline(
        config_path=write_config(tmp_path / "config.yaml"),
        catalog_snapshot_id="catalog_pipe",
        context_snapshot_id="context_pipe",
        output_root=tmp_path / "runs",
        run_id="permissive_run",
        executor=lambda _context: execution_result(eligible=False),
        strict=False,
    )

    assert run.status == "completed_with_warnings"
    manifest = json.loads((run.output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "completed_with_warnings"
    assert manifest["strict_failure_count"] == 1


def test_phase0_generation_executor_exports_legacy_and_canonical_artifacts(tmp_path):
    run = run_research_pipeline(
        config_path=write_config(tmp_path / "config.yaml"),
        catalog_snapshot_id="catalog_pipe",
        context_snapshot_id="context_pipe",
        output_root=tmp_path / "runs",
        run_id="phase0_run",
        executor=build_phase0_generation_executor(
            method_df=phase0_method_frame(),
            route_stops_df=phase0_route_stops_frame(),
        ),
        strict=False,
    )

    legacy_dir = run.output_dir / "phase0_legacy"
    for filename in PHASE0_ARTIFACT_FILES:
        assert (legacy_dir / filename).exists(), filename
    assert len(run.planner_runs) == 1
    assert len(run.output_plans) == 1
    assert len(run.evaluations) == 1
    assert run.status == "completed_with_warnings"

    manifest = json.loads((run.output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["artifact_counts"]["plans"] == 1
    assert manifest["artifact_counts"]["evaluations"] == 1
    assert manifest["artifact_counts"]["routing"] == 1
    assert len(manifest["artifacts"]["routing"]) == 1

    route_path = run.output_dir / manifest["artifacts"]["routing"][0]
    route_record = json.loads(route_path.read_text(encoding="utf-8"))
    assert route_record["route_id"].startswith("route_plan_")
    assert route_record["schema_version"] == "phase0-route-audit-v1"
    assert route_record["leg_count"] == 3
    assert route_record["road_validated"] is False

    metrics = json.loads((run.output_dir / "metrics" / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["phase0_legacy_artifact_dir"] == "phase0_legacy"
    assert metrics["phase0_plan_artifact_count"] == 1


def test_phase0_generation_executor_strict_mode_blocks_after_diagnostics(tmp_path):
    with pytest.raises(PipelineStrictModeError) as exc_info:
        run_research_pipeline(
            config_path=write_config(tmp_path / "config.yaml"),
            catalog_snapshot_id="catalog_pipe",
            context_snapshot_id="context_pipe",
            output_root=tmp_path / "runs",
            run_id="phase0_strict_run",
            executor=build_phase0_generation_executor(
                method_df=phase0_method_frame(),
                route_stops_df=phase0_route_stops_frame(),
            ),
            strict=True,
        )

    run = exc_info.value.pipeline_run
    assert run is not None
    assert run.status == "failed_strict"
    assert (run.output_dir / "phase0_legacy" / "production_phase0_evaluation_reports.csv").exists()
    assert (run.output_dir / "evaluations").is_dir()
    manifest = json.loads((run.output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["strict_failure_count"] == 1


def test_progressive_repair_failure_retains_canonical_planner_provenance(tmp_path):
    missing_matrix = RouteMatrix(
        matrix_id="matrix_progressive_missing",
        context_snapshot_id="context_pipe",
        entity_ids=(),
        cells={},
    )
    run = run_research_pipeline(
        config_path=write_config(tmp_path / "config.yaml"),
        catalog_snapshot_id="catalog_pipe",
        context_snapshot_id="context_pipe",
        output_root=tmp_path / "runs",
        run_id="progressive_failure_run",
        parent_plan_id="parent_progressive_pipe",
        repair_request_id="repair_progressive_pipe",
        executor=build_progressive_repair_executor(
            parent_plan=progressive_parent_plan(),
            repair_request=progressive_repair_request(),
            route_matrix=missing_matrix,
            day_route_config=progressive_day_route_config(),
            publication_mode=True,
        ),
        strict=True,
    )

    assert run.status == "failed"
    assert len(run.planner_runs) == 1
    planner_record = run.planner_runs[0].to_record()
    assert planner_record["method_requested"] == "progressive_sequential_lexicographic_repair"
    assert planner_record["method_executed"] == "progressive_sequential_lexicographic_repair"
    assert planner_record["execution_status"] == "FAILED"
    assert planner_record["solver_certification"] == "NO_CERTIFICATE"
    assert "missing route matrix cell" in planner_record["error_summary"]


def test_progressive_repair_executor_exports_child_certificate_diff_and_explanation(tmp_path):
    run = run_research_pipeline(
        config_path=write_config(tmp_path / "config.yaml"),
        catalog_snapshot_id="catalog_pipe",
        context_snapshot_id="context_pipe",
        output_root=tmp_path / "runs",
        run_id="progressive_repair_run",
        parent_plan_id="parent_progressive_pipe",
        repair_request_id="repair_progressive_pipe",
        executor=build_progressive_repair_executor(
            parent_plan=progressive_parent_plan(),
            repair_request=progressive_repair_request(),
            route_matrix=progressive_route_matrix(),
            day_route_config=progressive_day_route_config(),
            publication_mode=True,
        ),
        strict=True,
    )

    assert run.status == "completed"
    assert run.parent_plan is not None
    assert len(run.output_plans) == 1
    child_record = run.output_plans[0].to_record() if hasattr(run.output_plans[0], "to_record") else run.output_plans[0]
    assert child_record["parent_plan_id"] == "parent_progressive_pipe"
    assert child_record["sequence"] == ["poi_a", "poi_b", "poi_e", "poi_d"]

    manifest = json.loads((run.output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["mode"] == "repair"
    assert manifest["artifact_counts"]["plans"] == 1
    assert manifest["artifact_counts"]["diffs"] == 1
    assert manifest["artifact_counts"]["evaluations"] == 1
    assert manifest["artifact_counts"]["explanations"] == 1
    assert manifest["strict_failure_count"] == 0

    planner_runs = [
        json.loads(line)
        for line in (run.output_dir / "planner_runs.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    canonical_method = "progressive_sequential_lexicographic_repair"
    assert any(
        record["method_requested"] == canonical_method and record["method_executed"] == canonical_method
        for record in planner_runs
    )

    evaluation_path = run.output_dir / manifest["artifacts"]["evaluations"][0]
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    assert evaluation["schema_version"] == "plan-evaluation-certificate-v1"
    assert evaluation["comparison_eligibility"] == "eligible"
    assert evaluation["plan_id"] == child_record["plan_id"]

    explanation_path = run.output_dir / manifest["artifacts"]["explanations"][0]
    explanation_record = json.loads(explanation_path.read_text(encoding="utf-8"))
    assert explanation_record["evidence_type"] == "contrastive"
    assert explanation_record["child_plan_id"] == child_record["plan_id"]
    assert explanation_record["findings"] == []

    metrics = json.loads((run.output_dir / "metrics" / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["repair_outcome_status"] == "accepted"
    assert metrics["accepted_radius"] == RepairRadius.SAME_DAY_REPLACEMENT.value
    assert metrics["repair_attempt_count"] == 1
