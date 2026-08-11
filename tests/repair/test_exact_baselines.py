from itinerary_system.repair.day_route_solver import DayRouteSolverConfig
from itinerary_system.repair.exact_baselines import (
    CONTEXT_BLIND_SOLVER,
    FULL_REOPTIMIZATION,
    plan_context_blind_solver,
    plan_full_reoptimization,
)
from itinerary_system.repair_planner import RepairRequest
from itinerary_system.research_artifacts import PlanArtifactV2
from itinerary_system.routing import RouteMatrix, RouteMatrixCell


def parent_plan() -> PlanArtifactV2:
    return PlanArtifactV2(
        plan_id="parent_exact",
        source_run_id="run_parent_exact",
        planning_request_id="request_parent_exact",
        catalog_snapshot_id="catalog_exact",
        context_snapshot_id="context_exact",
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
        request_id="repair_exact",
        baseline_route=parent_plan().selected_stops,
        user_intent="repair weather disruption",
        confirmed_constraints={
            "parent_plan_id": "parent_exact",
            "affected_days": (1,),
            "weather_risk_overrides": {"outdoor": 0.95},
            "weather_feasible": {"outdoor": False},
        },
        candidate_pois=(
            {"stop_id": "indoor", "name": "Indoor", "day": 1, "lodging_id": "hotel", "utility": 6},
        ),
    )


def cell(origin: str, destination: str) -> RouteMatrixCell:
    return RouteMatrixCell(
        origin_id=origin,
        destination_id=destination,
        distance_m=5000,
        duration_s=300,
        provider="unit",
        road_validated=True,
    )


def matrix() -> RouteMatrix:
    ids = ("start", "outdoor", "museum", "indoor", "end")
    cells = {
        (origin, destination): cell(origin, destination)
        for origin in ids
        for destination in ids
        if origin != destination
    }
    return RouteMatrix(matrix_id="matrix_exact", context_snapshot_id="context_exact", entity_ids=ids, cells=cells)


def config() -> DayRouteSolverConfig:
    return DayRouteSolverConfig(
        max_day_minutes=240,
        default_visit_minutes=30,
        start_anchor_by_day={1: "start"},
        end_anchor_by_day={1: "end"},
        strict_route_matrix=True,
    )


def test_exact_baselines_respect_distinct_context_boundaries():
    blind = plan_context_blind_solver(
        parent_plan(), request(), matrix(), day_route_config=config(), publication_mode=True
    )
    full = plan_full_reoptimization(
        parent_plan(), request(), matrix(), day_route_config=config(), publication_mode=True
    )

    assert blind.status == "completed"
    assert blind.method_id == CONTEXT_BLIND_SOLVER
    assert blind.search_complete is True
    assert blind.planner_run.solver_certification == "OPTIMALITY_CERTIFIED"
    assert blind.child_plan is not None
    assert "outdoor" in blind.child_plan.sequence

    assert full.status == "completed"
    assert full.method_id == FULL_REOPTIMIZATION
    assert full.search_complete is True
    assert full.planner_run.solver_certification == "OPTIMALITY_CERTIFIED"
    assert full.child_plan is not None
    assert "outdoor" not in full.child_plan.sequence
    assert set(full.child_plan.sequence) == {"museum", "indoor"}


def test_exact_solver_refuses_optimality_claim_when_enumeration_limit_is_exceeded():
    result = plan_full_reoptimization(
        parent_plan(),
        request(),
        matrix(),
        day_route_config=config(),
        publication_mode=True,
        max_complete_candidates=1,
    )

    assert result.status == "failed"
    assert result.search_complete is False
    assert result.child_plan is None
    assert result.planner_run.solver_certification == "NO_CERTIFICATE"
    assert "complete_candidate_limit_exceeded:1" in result.failure_reasons
    assert result.candidate_space_lower_bound > 1


def test_exact_solver_preflights_large_space_when_parent_has_no_lodging_assignments():
    no_lodging_parent = PlanArtifactV2(
        **{
            **parent_plan().__dict__,
            "lodging_assignments": {},
            "selected_stops": tuple(
                {key: value for key, value in stop.items() if key != "lodging_id"}
                for stop in parent_plan().selected_stops
            ),
        }
    )
    no_lodging_request = RepairRequest(
        request_id="repair_exact_no_lodging",
        baseline_route=no_lodging_parent.selected_stops,
        user_intent="repair without lodging decisions",
        confirmed_constraints={"parent_plan_id": no_lodging_parent.plan_id, "affected_days": (1,)},
    )

    result = plan_full_reoptimization(
        no_lodging_parent,
        no_lodging_request,
        matrix(),
        day_route_config=config(),
        publication_mode=True,
        max_complete_candidates=1,
    )

    assert result.status == "failed"
    assert result.search_complete is False
    assert result.candidate_count == 0
    assert result.candidate_space_lower_bound == 3
    assert "complete_candidate_limit_exceeded:1" in result.failure_reasons
    assert "candidate_space_lower_bound:3" in result.failure_reasons
