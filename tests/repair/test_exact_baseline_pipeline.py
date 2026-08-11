import json

from itinerary_system.pipeline_runner import (
    build_context_blind_solver_executor,
    build_full_reoptimization_executor,
    run_research_pipeline,
)
from itinerary_system.repair.day_route_solver import DayRouteSolverConfig
from itinerary_system.repair_planner import RepairRequest
from itinerary_system.research_artifacts import PlanArtifactV2
from itinerary_system.routing import RouteMatrix, RouteMatrixCell


def parent_plan() -> PlanArtifactV2:
    return PlanArtifactV2(
        plan_id="parent_exact_pipeline",
        source_run_id="run_parent_exact_pipeline",
        planning_request_id="request_parent_exact_pipeline",
        catalog_snapshot_id="catalog_exact_pipeline",
        context_snapshot_id="context_exact_pipeline",
        selected_stops=(
            {"stop_id": "outdoor", "name": "Outdoor", "day": 1, "lodging_id": "hotel", "utility": 8},
            {"stop_id": "museum", "name": "Museum", "day": 1, "lodging_id": "hotel", "utility": 7},
        ),
        day_assignments={"outdoor": 1, "museum": 1},
        sequence=("outdoor", "museum"),
        lodging_assignments={"1": "hotel"},
        ordered_days=({"day": 1, "stop_ids": ("outdoor", "museum")},),
        route_ids_by_day={1: "route_parent"},
        created_at="2026-07-13T00:00:00+00:00",
    )


def request() -> RepairRequest:
    return RepairRequest(
        request_id="repair_exact_pipeline",
        baseline_route=parent_plan().selected_stops,
        user_intent="repair weather disruption",
        confirmed_constraints={
            "parent_plan_id": "parent_exact_pipeline",
            "affected_days": (1,),
            "weather_risk_overrides": {"outdoor": 0.95},
            "weather_feasible": {"outdoor": False},
        },
        candidate_pois=(
            {"stop_id": "indoor", "name": "Indoor", "day": 1, "lodging_id": "hotel", "utility": 6},
        ),
    )


def matrix() -> RouteMatrix:
    ids = ("start", "outdoor", "museum", "indoor", "end")
    cells = {
        (origin, destination): RouteMatrixCell(
            origin_id=origin,
            destination_id=destination,
            distance_m=5000,
            duration_s=300,
            provider="unit",
            road_validated=True,
        )
        for origin in ids
        for destination in ids
        if origin != destination
    }
    return RouteMatrix(
        matrix_id="matrix_exact_pipeline",
        context_snapshot_id="context_exact_pipeline",
        entity_ids=ids,
        cells=cells,
    )


def route_config() -> DayRouteSolverConfig:
    return DayRouteSolverConfig(
        max_day_minutes=240,
        default_visit_minutes=30,
        start_anchor_by_day={1: "start"},
        end_anchor_by_day={1: "end"},
        strict_route_matrix=True,
    )


def write_config(path):
    path.write_text(
        "data:\n  catalog_snapshot_id: catalog_exact_pipeline\n  context_snapshot_id: context_exact_pipeline\n",
        encoding="utf-8",
    )
    return path


def certificate_record(run):
    manifest = json.loads(run.manifest_path.read_text(encoding="utf-8"))
    path = run.output_dir / manifest["artifacts"]["evaluations"][0]
    return json.loads(path.read_text(encoding="utf-8"))


def test_full_reoptimization_pipeline_is_strictly_eligible(tmp_path):
    run = run_research_pipeline(
        config_path=write_config(tmp_path / "config.yaml"),
        catalog_snapshot_id="catalog_exact_pipeline",
        context_snapshot_id="context_exact_pipeline",
        parent_plan_id="parent_exact_pipeline",
        repair_request_id="repair_exact_pipeline",
        output_root=tmp_path / "runs",
        run_id="full_reoptimization_pipeline",
        executor=build_full_reoptimization_executor(
            parent_plan=parent_plan(),
            repair_request=request(),
            route_matrix=matrix(),
            day_route_config=route_config(),
            publication_mode=True,
        ),
        strict=True,
    )

    certificate = certificate_record(run)
    assert run.status == "completed"
    assert certificate["comparison_eligibility"] == "eligible"
    assert certificate["route_validation"]["publication_ready"] is True
    assert certificate["metrics"]["utility_retained"] == 13 / 15


def test_context_blind_output_is_independently_rejected_when_context_infeasible(tmp_path):
    run = run_research_pipeline(
        config_path=write_config(tmp_path / "config.yaml"),
        catalog_snapshot_id="catalog_exact_pipeline",
        context_snapshot_id="context_exact_pipeline",
        parent_plan_id="parent_exact_pipeline",
        repair_request_id="repair_exact_pipeline",
        output_root=tmp_path / "runs",
        run_id="context_blind_pipeline",
        executor=build_context_blind_solver_executor(
            parent_plan=parent_plan(),
            repair_request=request(),
            route_matrix=matrix(),
            day_route_config=route_config(),
            publication_mode=True,
        ),
        strict=False,
    )

    certificate = certificate_record(run)
    codes = {failure["code"] for failure in certificate["failures"]}
    assert run.status == "completed_with_warnings"
    assert certificate["comparison_eligibility"] == "ineligible"
    assert "context_excluded_stop_selected" in codes
    assert certificate["route_validation"]["publication_ready"] is True
