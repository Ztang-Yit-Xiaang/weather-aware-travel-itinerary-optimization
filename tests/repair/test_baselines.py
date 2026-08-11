import json

from itinerary_system.pipeline_runner import (
    build_deterministic_context_aware_heuristic_executor,
    run_research_pipeline,
)
from itinerary_system.repair.baselines import (
    DETERMINISTIC_CONTEXT_AWARE_HEURISTIC,
    plan_deterministic_context_aware_heuristic,
)
from itinerary_system.repair.day_route_solver import DayRouteSolverConfig
from itinerary_system.repair_planner import RepairRequest
from itinerary_system.research_artifacts import PlanArtifactV2
from itinerary_system.routing import RouteMatrix, RouteMatrixCell


def parent_plan() -> PlanArtifactV2:
    return PlanArtifactV2(
        plan_id="parent_heuristic",
        source_run_id="run_parent_heuristic",
        planning_request_id="request_parent_heuristic",
        catalog_snapshot_id="catalog_heuristic",
        context_snapshot_id="context_heuristic",
        selected_stops=(
            {"stop_id": "outdoor", "name": "Outdoor", "day": 1, "lodging_id": "hotel", "utility": 8},
            {"stop_id": "museum", "name": "Museum", "day": 1, "lodging_id": "hotel", "utility": 7},
        ),
        day_assignments={"outdoor": 1, "museum": 1},
        sequence=("outdoor", "museum"),
        lodging_assignments={"1": "hotel"},
        ordered_days=({"day": 1, "stop_ids": ("outdoor", "museum")},),
        route_ids_by_day={1: "route_parent"},
    )


def request() -> RepairRequest:
    return RepairRequest(
        request_id="repair_heuristic",
        baseline_route=parent_plan().selected_stops,
        user_intent="replace weather-infeasible outdoor stop",
        confirmed_constraints={
            "parent_plan_id": "parent_heuristic",
            "affected_days": (1,),
            "weather_risk_overrides": {"outdoor": 0.95},
            "weather_feasible": {"outdoor": False},
        },
        candidate_pois=(
            {"stop_id": "indoor", "name": "Indoor", "day": 1, "lodging_id": "hotel", "utility": 6},
        ),
    )


def cell(origin: str, destination: str, minutes: float) -> RouteMatrixCell:
    return RouteMatrixCell(
        origin_id=origin,
        destination_id=destination,
        distance_m=minutes * 1000,
        duration_s=minutes * 60,
        provider="unit",
        road_validated=True,
    )


def matrix() -> RouteMatrix:
    ids = ("start", "outdoor", "museum", "indoor", "end")
    cells = {
        (origin, destination): cell(origin, destination, 5 if "indoor" in {origin, destination} else 10)
        for origin in ids
        for destination in ids
        if origin != destination
    }
    return RouteMatrix(matrix_id="matrix_heuristic", context_snapshot_id="context_heuristic", entity_ids=ids, cells=cells)


def config() -> DayRouteSolverConfig:
    return DayRouteSolverConfig(
        max_day_minutes=240,
        default_visit_minutes=30,
        start_anchor_by_day={1: "start"},
        end_anchor_by_day={1: "end"},
        strict_route_matrix=True,
    )


def test_context_aware_heuristic_replaces_infeasible_stop_deterministically():
    frozen_parent = parent_plan()
    first = plan_deterministic_context_aware_heuristic(
        frozen_parent, request(), matrix(), day_route_config=config(), publication_mode=True
    )
    second = plan_deterministic_context_aware_heuristic(
        frozen_parent, request(), matrix(), day_route_config=config(), publication_mode=True
    )

    assert first.status == "completed"
    assert first.method_id == DETERMINISTIC_CONTEXT_AWARE_HEURISTIC
    assert first.planner_run.method_requested == DETERMINISTIC_CONTEXT_AWARE_HEURISTIC
    assert first.child_plan is not None
    assert first.child_plan.sequence == ("indoor", "museum")
    assert first.child_plan.content_hash == second.child_plan.content_hash
    assert sum(record.selected for record in first.decision_records) == 1
    assert all(record.candidate_id for record in first.decision_records)


def test_heuristic_pipeline_emits_independent_certificate_and_provenance(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "data:\n  catalog_snapshot_id: catalog_heuristic\n  context_snapshot_id: context_heuristic\n",
        encoding="utf-8",
    )
    run = run_research_pipeline(
        config_path=config_path,
        catalog_snapshot_id="catalog_heuristic",
        context_snapshot_id="context_heuristic",
        parent_plan_id="parent_heuristic",
        repair_request_id="repair_heuristic",
        output_root=tmp_path / "runs",
        run_id="heuristic_pipeline",
        executor=build_deterministic_context_aware_heuristic_executor(
            parent_plan=parent_plan(),
            repair_request=request(),
            route_matrix=matrix(),
            day_route_config=config(),
            publication_mode=True,
        ),
        strict=True,
    )

    assert run.status == "completed"
    planner_rows = [json.loads(line) for line in (run.output_dir / "planner_runs.jsonl").read_text().splitlines()]
    assert planner_rows[0]["method_requested"] == DETERMINISTIC_CONTEXT_AWARE_HEURISTIC
    manifest = json.loads(run.manifest_path.read_text(encoding="utf-8"))
    certificate_path = run.output_dir / manifest["artifacts"]["evaluations"][0]
    certificate = json.loads(certificate_path.read_text(encoding="utf-8"))
    assert certificate["comparison_eligibility"] == "eligible"
    assert certificate["route_validation"]["publication_ready"] is True
    assert abs(certificate["metrics"]["utility_retained"] - (13 / 15)) < 1e-9
