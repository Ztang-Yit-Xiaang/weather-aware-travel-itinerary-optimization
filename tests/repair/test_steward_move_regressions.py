from types import SimpleNamespace

from itinerary_system.repair.baselines import _select_joint_move_candidates
from itinerary_system.repair.day_route_solver import solve_day_route_subproblem
from itinerary_system.repair.master_model import build_repair_master_model
from itinerary_system.repair.neighborhood import RepairRadius, build_repair_neighborhood
from itinerary_system.repair_planner import RepairRequest
from itinerary_system.research_artifacts import PlanArtifactV2
from itinerary_system.routing import RouteMatrix, RouteMatrixCell


def test_empty_move_candidate_pool_does_not_report_time_limit():
    selected, failures = _select_joint_move_candidates(
        SimpleNamespace(), {1: SimpleNamespace(candidates=())}
    )

    assert selected == {}
    assert failures == ["no_feasible_local_candidate:day:1"]

    candidate = SimpleNamespace(repair_solution=None, violations=("day_time_exceeded:2",))
    model = SimpleNamespace(validate_solution=lambda solution: ())
    _, failures = _select_joint_move_candidates(model, {2: SimpleNamespace(candidates=(candidate,))})

    assert failures == ["no_feasible_local_candidate:day:2", "day_time_exceeded:2"]


def test_context_excluded_move_sequences_do_not_report_time_limit():
    parent = PlanArtifactV2(
        plan_id="parent", source_run_id="run", planning_request_id="request",
        catalog_snapshot_id="catalog", context_snapshot_id="context",
        selected_stops=({"stop_id": "a", "day": 1}, {"stop_id": "b", "day": 2}),
        day_assignments={"a": 1, "b": 2}, sequence=("a", "b"),
        lodging_assignments={"1": "hotel", "2": "hotel"},
        route_ids_by_day={1: "route1", 2: "route2"},
    )
    request = RepairRequest(
        request_id="move", baseline_route=parent.selected_stops, user_intent="move a",
        confirmed_constraints={"affected_days": [1, 2], "move_stops": {"a": 2}, "must_delete": ["b"]},
    )
    matrix = RouteMatrix(
        matrix_id="matrix", context_snapshot_id="context", entity_ids=("a", "b"),
        cells={("a", "b"): RouteMatrixCell(
            origin_id="a", destination_id="b", duration_s=60, distance_m=100,
            provider="unit", road_validated=True,
        )},
    )
    neighborhood = build_repair_neighborhood(parent, request, RepairRadius.SAME_DAY_REPLACEMENT)
    model = build_repair_master_model(parent, request, neighborhood, matrix)
    result = solve_day_route_subproblem(model, matrix, day=2)

    assert model.metadata["move_stops"] == {"a": 2}
    assert result.candidates == ()
    _, failures = _select_joint_move_candidates(model, {2: result})
    assert failures == ["no_feasible_local_candidate:day:2"]


def test_named_move_target_is_stored_as_stop_id():
    parent = PlanArtifactV2(
        plan_id="parent_named_move", source_run_id="run", planning_request_id="request",
        catalog_snapshot_id="catalog", context_snapshot_id="context",
        selected_stops=(
            {"stop_id": "coast", "name": "Big Sur Coast", "day": 1},
            {"stop_id": "bridge", "name": "Golden Gate Bridge", "day": 2},
        ),
        day_assignments={"coast": 1, "bridge": 2}, sequence=("coast", "bridge"),
        lodging_assignments={"1": "hotel", "2": "hotel"},
        route_ids_by_day={1: "route1", 2: "route2"},
    )
    request = RepairRequest(
        request_id="named_move", baseline_route=parent.selected_stops,
        user_intent="move Big Sur Coast to day 2",
        confirmed_constraints={"move_stops": {"Big Sur Coast": 2}},
    )
    matrix = RouteMatrix(matrix_id="matrix", context_snapshot_id="context", entity_ids=())
    neighborhood = build_repair_neighborhood(parent, request, RepairRadius.SAME_DAY_REPLACEMENT)
    model = build_repair_master_model(parent, request, neighborhood, matrix)

    assert model.metadata["move_stops"] == {"coast": 2}
