import unittest
from types import SimpleNamespace

from itinerary_system.plans import (
    ConstraintOrigin,
    ConstraintScope,
    ConstraintStrength,
    OwnedConstraint,
    RelaxationPolicy,
)
from itinerary_system.repair import RepairRadius, build_repair_neighborhood
from itinerary_system.repair.lexicographic import (
    LexicographicRepairSolver,
    ObjectiveTolerances,
    planner_runs_from_lexicographic_result,
    solve_gurobi_candidate_reference,
    solve_lexicographically,
)
from itinerary_system.repair.master_model import RepairSolution, build_repair_master_model
from itinerary_system.research_artifacts import PlanArtifactV2
from itinerary_system.routing import RouteMatrix


def owned_constraint(
    constraint_id: str,
    *,
    target_id: str,
    strength: ConstraintStrength,
    scope: ConstraintScope,
    origin: ConstraintOrigin = ConstraintOrigin.USER,
    relaxation_policy: RelaxationPolicy = RelaxationPolicy.AUTO_WITH_PENALTY,
) -> dict:
    return OwnedConstraint(
        constraint_id=constraint_id,
        origin=origin,
        strength=strength,
        scope=scope,
        target_id=target_id,
        relation="protect",
        value=True,
        confirmed=True,
        relaxation_policy=relaxation_policy,
    ).to_record()


def parent_plan() -> PlanArtifactV2:
    return PlanArtifactV2(
        plan_id="parent_lexicographic",
        source_run_id="run_parent",
        planning_request_id="request_demo",
        catalog_snapshot_id="california_v1",
        context_snapshot_id="context_static_demo_2026_06",
        selected_stops=(
            {"stop_id": "poi_a", "name": "Ferry Building", "day": 1, "stop_order": 1, "lodging_id": "hotel_sf"},
            {"stop_id": "poi_b", "name": "Golden Gate Bridge", "day": 2, "stop_order": 1, "lodging_id": "hotel_y"},
            {"stop_id": "poi_c", "name": "Redwood Grove", "day": 2, "stop_order": 2, "lodging_id": "hotel_y"},
            {"stop_id": "poi_d", "name": "Big Sur Coast", "day": 3, "stop_order": 1, "lodging_id": "hotel_m"},
        ),
        day_assignments={"poi_a": 1, "poi_b": 2, "poi_c": 2, "poi_d": 3},
        sequence=("poi_a", "poi_b", "poi_c", "poi_d"),
        lodging_assignments={"1": "hotel_sf", "2": "hotel_y", "3": "hotel_m"},
        route_ids_by_day={1: "route_1", 2: "route_2", 3: "route_3"},
        owned_constraints=(
            owned_constraint(
                "locked_bridge",
                target_id="poi_b",
                strength=ConstraintStrength.LOCKED,
                scope=ConstraintScope.STOP,
                relaxation_policy=RelaxationPolicy.NEVER,
            ),
            owned_constraint(
                "booked_hotel_y",
                target_id="hotel_y",
                strength=ConstraintStrength.BOOKED,
                scope=ConstraintScope.LODGING,
                origin=ConstraintOrigin.USER_BOOKING,
                relaxation_policy=RelaxationPolicy.EXPLICIT_ONLY,
            ),
            owned_constraint(
                "soft_redwood",
                target_id="poi_c",
                strength=ConstraintStrength.SOFT,
                scope=ConstraintScope.STOP,
            ),
        ),
        created_at="2026-07-07T00:00:00+00:00",
    )


def request(**confirmed_constraints):
    return SimpleNamespace(
        request_id="repair_003",
        confirmed_constraints=confirmed_constraints,
        candidate_pois=(
            {
                "stop_id": "poi_e",
                "name": "Indoor Aquarium",
                "day": 2,
                "stop_order": 2,
                "lodging_id": "hotel_y",
            },
        ),
    )


def model_with_booked_relaxation():
    parent = parent_plan()
    repair_request = request(affected_days=(2,), allow_booked_relaxation=True)
    neighborhood = build_repair_neighborhood(parent, repair_request, RepairRadius.HOTEL_CHANGING_REPAIR)
    matrix = RouteMatrix(matrix_id="matrix_demo", context_snapshot_id="context_static_demo_2026_06", entity_ids=())
    return build_repair_master_model(parent, repair_request, neighborhood, matrix)


def edit_without_booked_relaxation() -> RepairSolution:
    return RepairSolution(
        selected_stop_ids=("poi_a", "poi_b", "poi_e", "poi_d"),
        day_assignments={"poi_a": 1, "poi_b": 2, "poi_e": 2, "poi_d": 3},
        lodging_assignments={"1": "hotel_sf", "2": "hotel_y", "3": "hotel_m"},
        route_ids_by_day={1: "route_1", 2: "route_2", 3: "route_3"},
        metadata={"candidate_id": "safe_edit"},
    )


def no_edit_with_booked_relaxation() -> RepairSolution:
    return RepairSolution(
        selected_stop_ids=("poi_a", "poi_b", "poi_c", "poi_d"),
        day_assignments={"poi_a": 1, "poi_b": 2, "poi_c": 2, "poi_d": 3},
        lodging_assignments={"1": "hotel_sf", "2": "hotel_y", "3": "hotel_m"},
        relaxed_constraint_ids=("booked_hotel_y",),
        route_ids_by_day={1: "route_1", 2: "route_2", 3: "route_3"},
        metadata={"candidate_id": "relax_booking"},
    )


class LexicographicSolverTests(unittest.TestCase):
    def test_later_stage_cannot_degrade_prior_stage_without_tolerance(self):
        model = model_with_booked_relaxation()

        result = solve_lexicographically(
            model,
            ObjectiveTolerances(),
            candidate_solutions=(edit_without_booked_relaxation(), no_edit_with_booked_relaxation()),
        )

        self.assertEqual(result.status, "OPTIMAL")
        self.assertEqual(result.selected_solution.metadata["candidate_id"], "safe_edit")
        self.assertEqual(result.stage_results[0].stage, "hard_and_booked_relaxation")
        self.assertEqual(result.stage_results[0].candidate_count, 1)
        self.assertEqual(result.stage_results[0].objective_value, 0.0)

    def test_stage_tolerance_allows_later_stage_tradeoff(self):
        model = model_with_booked_relaxation()

        result = solve_lexicographically(
            model,
            ObjectiveTolerances(stage_tolerances={"hard_and_booked_relaxation": 500.0}),
            candidate_solutions=(edit_without_booked_relaxation(), no_edit_with_booked_relaxation()),
        )

        self.assertEqual(result.status, "OPTIMAL")
        self.assertEqual(result.selected_solution.metadata["candidate_id"], "relax_booking")
        self.assertEqual(result.stage_results[0].fixed_limit, 500.0)
        self.assertEqual(result.stage_results[1].stage, "ownership_weighted_change")

    def test_stage_statuses_bounds_and_components_are_persisted(self):
        solver = LexicographicRepairSolver(tolerances=ObjectiveTolerances())
        result = solver.solve(model_with_booked_relaxation(), (edit_without_booked_relaxation(),))

        self.assertTrue(result.accepted)
        self.assertEqual([stage.status for stage in result.stage_results], ["OPTIMAL", "OPTIMAL", "OPTIMAL", "OPTIMAL"])
        self.assertTrue(all(stage.best_bound == stage.objective_value for stage in result.stage_results))
        self.assertTrue(all(stage.optimality_gap == 0.0 for stage in result.stage_results))
        self.assertIn("ownership_weighted_change", result.selected_components)
        self.assertEqual(result.to_record()["solver_backend"], "enumeration")

    def test_infeasible_stage_emits_failed_planner_run(self):
        model = model_with_booked_relaxation()
        invalid = RepairSolution(
            selected_stop_ids=("poi_a", "poi_c", "poi_d"),
            day_assignments={"poi_a": 1, "poi_c": 2, "poi_d": 3},
            lodging_assignments={"1": "hotel_sf", "2": "hotel_y", "3": "hotel_m"},
            route_ids_by_day={1: "route_1", 2: "route_2", 3: "route_3"},
            metadata={"candidate_id": "invalid_delete_locked"},
        )

        result = solve_lexicographically(model, ObjectiveTolerances(), candidate_solutions=(invalid,))
        runs = planner_runs_from_lexicographic_result(result)

        self.assertEqual(result.status, "INFEASIBLE")
        self.assertEqual(result.stage_results[0].status, "INFEASIBLE")
        self.assertEqual(runs[0].execution_status, "FAILED")
        self.assertEqual(runs[0].error_class, "lexicographic_stage_infeasible")

    def test_small_instance_matches_gurobi_candidate_reference(self):
        model = model_with_booked_relaxation()
        tolerances = ObjectiveTolerances(stage_tolerances={"hard_and_booked_relaxation": 500.0})
        candidates = (edit_without_booked_relaxation(), no_edit_with_booked_relaxation())

        enumeration = solve_lexicographically(model, tolerances, candidate_solutions=candidates)
        gurobi = solve_gurobi_candidate_reference(model, candidates, tolerances)

        self.assertEqual(enumeration.status, "OPTIMAL")
        self.assertEqual(gurobi.status, "OPTIMAL")
        self.assertEqual(enumeration.selected_solution.metadata["candidate_id"], gurobi.selected_solution.metadata["candidate_id"])
        self.assertEqual(enumeration.selected_components, gurobi.selected_components)


if __name__ == "__main__":
    unittest.main()
