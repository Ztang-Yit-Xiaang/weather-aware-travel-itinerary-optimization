from itinerary_system.repair.context import PlannerContextMode
from itinerary_system.repair.day_route_solver import DayRouteSolverConfig
from itinerary_system.repair.gurobi_exact_v2 import (
    ExactModelDataError,
    build_gurobi_exact_model_data_v2,
    iter_structural_assignments_v2,
)
from itinerary_system.repair.master_model import build_repair_master_model
from itinerary_system.repair.neighborhood import (
    RepairRadius,
    build_repair_neighborhood,
)
from itinerary_system.repair_planner import RepairRequest
from itinerary_system.research_artifacts import PlanArtifactV2
from itinerary_system.routing import RouteMatrix, RouteMatrixCell


def _parent() -> PlanArtifactV2:
    return PlanArtifactV2(
        plan_id="parent_gurobi_v2",
        source_run_id="run_gurobi_v2",
        planning_request_id="request_parent_gurobi_v2",
        catalog_snapshot_id="catalog_gurobi_v2",
        context_snapshot_id="context_gurobi_v2",
        selected_stops=(
            {
                "stop_id": "outdoor",
                "day": 1,
                "lodging_id": "hotel",
                "utility": 8,
                "estimated_cost": 10,
                "visit_duration_minutes": 30,
            },
            {
                "stop_id": "museum",
                "day": 1,
                "lodging_id": "hotel",
                "utility": 7,
                "estimated_cost": 12,
                "visit_duration_minutes": 45,
            },
        ),
        day_assignments={"outdoor": 1, "museum": 1},
        sequence=("outdoor", "museum"),
        lodging_assignments={"1": "hotel"},
        ordered_days=({"day": 1, "stop_ids": ("outdoor", "museum")},),
        route_ids_by_day={1: "parent_route"},
    )


def _request(parent: PlanArtifactV2) -> RepairRequest:
    return RepairRequest(
        request_id="repair_gurobi_v2",
        baseline_route=parent.selected_stops,
        user_intent="repair weather disruption",
        confirmed_constraints={
            "parent_plan_id": parent.plan_id,
            "affected_days": (1,),
            "weather_feasible": {"outdoor": False},
        },
        candidate_pois=(
            {
                "stop_id": "indoor",
                "day": 1,
                "lodging_id": "hotel",
                "utility": 6,
                "visit_duration_minutes": 30,
            },
        ),
    )


def _matrix(*, fallback_pair: tuple[str, str] | None = None) -> RouteMatrix:
    ids = ("start", "outdoor", "museum", "indoor", "end")
    cells = {}
    for origin in ids:
        for destination in ids:
            if origin == destination:
                continue
            fallback = (origin, destination) == fallback_pair
            cells[(origin, destination)] = RouteMatrixCell(
                origin_id=origin,
                destination_id=destination,
                distance_m=5000,
                duration_s=300,
                provider="tiny_fixture",
                road_validated=not fallback,
                fallback_used=fallback,
                fallback_reason="test_fallback" if fallback else None,
            )
    return RouteMatrix(
        matrix_id="matrix_gurobi_v2",
        context_snapshot_id="context_gurobi_v2",
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


def _repair_model():
    parent = _parent()
    request = _request(parent)
    neighborhood = build_repair_neighborhood(
        parent,
        request,
        RepairRadius.FULL_REOPTIMIZATION,
    )
    return build_repair_master_model(
        parent,
        request,
        neighborhood,
        _matrix(),
        publication_mode=True,
        planner_context_mode=PlannerContextMode.AWARE,
    )


def test_canonical_index_matches_tiny_structural_universe_without_permutations():
    data = build_gurobi_exact_model_data_v2(
        _repair_model(),
        _matrix(),
        day_route_config=_config(),
    )

    assignments = tuple(iter_structural_assignments_v2(data))

    assert data.schema_version == "gurobi-exact-model-data-v2"
    assert data.structural_candidate_lower_bound == 3
    assert len(assignments) == 3
    assert {stop.stop_id for stop in data.stops} == {
        "outdoor",
        "museum",
        "indoor",
    }
    assert all(assignment.selected_day_by_stop for assignment in assignments)
    assert all(assignment.lodging_by_day == ((1, "hotel"),) for assignment in assignments)
    assert next(stop for stop in data.stops if stop.stop_id == "indoor").estimated_cost is None


def test_variable_index_is_deterministic_unique_and_arc_complete():
    first = build_gurobi_exact_model_data_v2(
        _repair_model(),
        _matrix(),
        day_route_config=_config(),
    )
    second = build_gurobi_exact_model_data_v2(
        _repair_model(),
        _matrix(),
        day_route_config=_config(),
    )

    assert first.variable_index == second.variable_index
    assert first.route_arcs == second.route_arcs
    assert len(first.variable_index.route_arc) == 12
    assert len(set(first.variable_index.route_arc)) == 12
    assert (1, "start", "outdoor") in first.variable_index.route_arc
    assert (1, "outdoor", "museum") in first.variable_index.route_arc
    assert (1, "indoor", "end") in first.variable_index.route_arc
    assert first.variable_index.variable_count == 34


def test_fallback_route_evidence_is_rejected_not_silently_omitted():
    with __import__("pytest").raises(
        ExactModelDataError,
        match="publication_route_arc_unavailable:start->outdoor",
    ):
        build_gurobi_exact_model_data_v2(
            _repair_model(),
            _matrix(fallback_pair=("start", "outdoor")),
            day_route_config=_config(),
        )


def test_route_matrix_identity_and_context_are_bound():
    wrong_matrix = RouteMatrix(
        matrix_id="wrong_matrix",
        context_snapshot_id="context_gurobi_v2",
        entity_ids=_matrix().entity_ids,
        cells=_matrix().cells,
    )

    with __import__("pytest").raises(ExactModelDataError, match="route_matrix_id_mismatch"):
        build_gurobi_exact_model_data_v2(
            _repair_model(),
            wrong_matrix,
            day_route_config=_config(),
        )
