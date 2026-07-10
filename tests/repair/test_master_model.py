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
from itinerary_system.repair.change_variables import ChangeVariableType
from itinerary_system.repair.master_model import RepairSolution, build_repair_master_model
from itinerary_system.research_artifacts import PlanArtifactV2
from itinerary_system.routing import RouteMatrix


def owned_constraint(
    constraint_id: str,
    *,
    target_id: str,
    strength: ConstraintStrength | str,
    scope: ConstraintScope | str,
    origin: ConstraintOrigin | str = ConstraintOrigin.USER,
    relaxation_policy: RelaxationPolicy | str = RelaxationPolicy.AUTO_WITH_PENALTY,
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
        plan_id="parent_repair_002",
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
        request_id="repair_002",
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


def empty_matrix() -> RouteMatrix:
    return RouteMatrix(matrix_id="matrix_demo", context_snapshot_id="context_static_demo_2026_06", entity_ids=())


class RepairMasterModelTests(unittest.TestCase):
    def test_model_declares_selection_day_lodging_and_change_variables(self):
        parent = parent_plan()
        neighborhood = build_repair_neighborhood(parent, request(affected_days=(2,)), RepairRadius.SAME_DAY_REPLACEMENT)

        model = build_repair_master_model(parent, request(affected_days=(2,)), neighborhood, empty_matrix())

        self.assertEqual(model.route_matrix_id, "matrix_demo")
        self.assertEqual(model.variables.decision_variable("select_stop[poi_a]").fixed_value, 1)
        self.assertEqual(model.variables.decision_variable("assign_day[poi_a]").fixed_value, 1)
        self.assertIsNone(model.variables.decision_variable("select_stop[poi_c]").fixed_value)
        self.assertIsNone(model.variables.decision_variable("select_stop[poi_e]").fixed_value)
        self.assertEqual(model.variables.decision_variable("assign_lodging[2]").fixed_value, "hotel_y")
        self.assertEqual(model.required_route_pairs, (("poi_b", "poi_c"),))
        self.assertTrue(any(change.change_type == ChangeVariableType.DELETE_STOP for change in model.variables.change_variables))
        self.assertTrue(any(change.change_type == ChangeVariableType.ADD_STOP for change in model.variables.change_variables))

    def test_locked_poi_cannot_be_deleted_or_relaxed(self):
        parent = parent_plan()
        neighborhood = build_repair_neighborhood(parent, request(affected_days=(2,)), RepairRadius.SAME_DAY_REPLACEMENT)
        model = build_repair_master_model(parent, request(affected_days=(2,)), neighborhood, empty_matrix())

        self.assertEqual(model.variables.decision_variable("select_stop[poi_b]").fixed_value, 1)
        self.assertEqual(model.variables.change_variable("delete_stop[poi_b]").upper_bound, 0.0)
        self.assertEqual(model.variables.decision_variable("relax_constraint[locked_bridge]").fixed_value, 0)
        bad_solution = RepairSolution(
            selected_stop_ids=("poi_a", "poi_c", "poi_d"),
            day_assignments={"poi_a": 1, "poi_c": 2, "poi_d": 3},
            lodging_assignments={"1": "hotel_sf", "2": "hotel_y", "3": "hotel_m"},
            route_ids_by_day={1: "route_1", 2: "route_2", 3: "route_3"},
        )

        violations = model.validate_solution(bad_solution)

        self.assertIn("locked_stop_deleted:poi_b", violations)

    def test_booked_hotel_requires_explicit_permission_before_lodging_change_or_relaxation(self):
        parent = parent_plan()
        request_without_permission = request(affected_days=(2,))
        locked_neighborhood = build_repair_neighborhood(
            parent,
            request_without_permission,
            RepairRadius.HOTEL_CHANGING_REPAIR,
        )
        locked_model = build_repair_master_model(parent, request_without_permission, locked_neighborhood, empty_matrix())

        self.assertEqual(locked_model.variables.decision_variable("assign_lodging[2]").fixed_value, "hotel_y")
        self.assertEqual(locked_model.variables.decision_variable("relax_constraint[booked_hotel_y]").fixed_value, 0)

        request_with_permission = request(affected_days=(2,), allow_booked_relaxation=True)
        editable_neighborhood = build_repair_neighborhood(
            parent,
            request_with_permission,
            RepairRadius.HOTEL_CHANGING_REPAIR,
        )
        editable_model = build_repair_master_model(parent, request_with_permission, editable_neighborhood, empty_matrix())

        self.assertIsNone(editable_model.variables.decision_variable("assign_lodging[2]").fixed_value)
        self.assertIsNone(editable_model.variables.decision_variable("relax_constraint[booked_hotel_y]").fixed_value)

    def test_objective_components_export_values_for_candidate_solution(self):
        parent = parent_plan()
        repair_request = request(affected_days=(2,))
        neighborhood = build_repair_neighborhood(parent, repair_request, RepairRadius.SAME_DAY_REPLACEMENT)
        model = build_repair_master_model(parent, repair_request, neighborhood, empty_matrix())
        solution = RepairSolution(
            selected_stop_ids=("poi_a", "poi_b", "poi_e", "poi_d"),
            day_assignments={"poi_a": 1, "poi_b": 2, "poi_e": 2, "poi_d": 3},
            lodging_assignments={"1": "hotel_sf", "2": "hotel_y", "3": "hotel_m"},
            route_ids_by_day={1: "route_1", 2: "route_2", 3: "route_3"},
        )

        components = model.export_components(solution)

        self.assertEqual(components["deleted_stop_count"], 1.0)
        self.assertEqual(components["added_stop_count"], 1.0)
        self.assertEqual(components["validation_violation_count"], 0.0)
        self.assertGreater(components["ownership_weighted_change"], 0.0)
        self.assertEqual(components["weighted_edit_cost"], components["ownership_weighted_change"])

    def test_extract_plan_preserves_lineage_and_records_diff_cost(self):
        parent = parent_plan()
        repair_request = request(affected_days=(2,))
        neighborhood = build_repair_neighborhood(parent, repair_request, RepairRadius.SAME_DAY_REPLACEMENT)
        model = build_repair_master_model(parent, repair_request, neighborhood, empty_matrix())
        solution = RepairSolution(
            selected_stop_ids=("poi_a", "poi_b", "poi_e", "poi_d"),
            day_assignments={"poi_a": 1, "poi_b": 2, "poi_e": 2, "poi_d": 3},
            lodging_assignments={"1": "hotel_sf", "2": "hotel_y", "3": "hotel_m"},
            route_ids_by_day={1: "route_1", 2: "route_2", 3: "route_3"},
            metadata={"case": "unit"},
        )

        child = model.extract_plan(solution, child_plan_id="child_repair_002")

        self.assertEqual(child.parent_plan_id, parent.plan_id)
        self.assertEqual(child.plan_id, "child_repair_002")
        self.assertIn("poi_e", child.sequence)
        self.assertNotIn("poi_c", child.sequence)
        self.assertGreater(child.change_components["weighted_edit_cost"], 0.0)


if __name__ == "__main__":
    unittest.main()
