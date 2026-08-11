from itinerary_system.repair.day_route_solver import DayRouteSolverConfig
from itinerary_system.repair.exact_baselines import plan_full_reoptimization
from itinerary_system.repair_planner import RepairRequest
from itinerary_system.research_artifacts import PlanArtifactV2
from itinerary_system.routing import RouteMatrix, RouteMatrixCell


def _parent_plan() -> PlanArtifactV2:
    return PlanArtifactV2(
        plan_id="parent_exact_budget_gate",
        source_run_id="run_parent_exact_budget_gate",
        planning_request_id="request_parent_exact_budget_gate",
        catalog_snapshot_id="catalog_exact_budget_gate",
        context_snapshot_id="context_exact_budget_gate",
        selected_stops=(
            {
                "stop_id": "outdoor",
                "name": "Outdoor",
                "day": 1,
                "lodging_id": "hotel",
                "utility": 8,
                "estimated_cost": 10.0,
            },
            {
                "stop_id": "museum",
                "name": "Museum",
                "day": 1,
                "lodging_id": "hotel",
                "utility": 7,
                "estimated_cost": 10.0,
            },
        ),
        day_assignments={"outdoor": 1, "museum": 1},
        sequence=("outdoor", "museum"),
        lodging_assignments={"1": "hotel"},
        ordered_days=({"day": 1, "stop_ids": ("outdoor", "museum")},),
        route_ids_by_day={1: "route_parent"},
        created_at="2026-07-26T00:00:00+00:00",
    )


def _request(parent: PlanArtifactV2) -> RepairRequest:
    return RepairRequest(
        request_id="repair_exact_budget_gate",
        baseline_route=parent.selected_stops,
        user_intent="find any itinerary within an impossible zero budget",
        confirmed_constraints={
            "parent_plan_id": parent.plan_id,
            "affected_days": (1,),
            "budget_limit": 0.0,
        },
        candidate_pois=(
            {
                "stop_id": "indoor",
                "name": "Indoor",
                "day": 1,
                "lodging_id": "hotel",
                "utility": 6,
                "estimated_cost": 10.0,
            },
        ),
    )


def _matrix() -> RouteMatrix:
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
        matrix_id="matrix_exact_budget_gate",
        context_snapshot_id="context_exact_budget_gate",
        entity_ids=ids,
        cells=cells,
    )


def _config() -> DayRouteSolverConfig:
    return DayRouteSolverConfig(
        max_day_minutes=240,
        default_visit_minutes=30,
        start_anchor_by_day={1: "start"},
        end_anchor_by_day={1: "end"},
        strict_route_matrix=True,
    )


def test_budget_pruning_is_exact_but_does_not_relax_raw_space_preflight(monkeypatch):
    parent = _parent_plan()
    request = _request(parent)

    def unexpected_route_evaluation(*args, **kwargs):
        raise AssertionError("budget-infeasible assignments must be pruned before route evaluation")

    monkeypatch.setattr(
        "itinerary_system.repair.exact_baselines.evaluate_route_sequence",
        unexpected_route_evaluation,
    )

    raw_space_blocked = plan_full_reoptimization(
        parent,
        request,
        _matrix(),
        day_route_config=_config(),
        publication_mode=True,
        max_complete_candidates=1,
    )
    completely_pruned = plan_full_reoptimization(
        parent,
        request,
        _matrix(),
        day_route_config=_config(),
        publication_mode=True,
        max_complete_candidates=10,
    )

    assert raw_space_blocked.search_complete is False
    assert raw_space_blocked.candidate_count == 0
    assert raw_space_blocked.candidate_space_lower_bound == 7
    assert "complete_candidate_limit_exceeded:1" in raw_space_blocked.failure_reasons

    assert completely_pruned.status == "failed"
    assert completely_pruned.search_complete is True
    assert completely_pruned.candidate_count == 0
    assert completely_pruned.planner_run.solver_status_raw == "complete"
    assert completely_pruned.planner_run.solver_certification == "NO_CERTIFICATE"
    assert "no_feasible_complete_candidate" in completely_pruned.failure_reasons
