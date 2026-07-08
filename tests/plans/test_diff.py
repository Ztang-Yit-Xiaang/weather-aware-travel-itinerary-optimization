import unittest
from dataclasses import replace

from itinerary_system.plans import (
    ConstraintOrigin,
    ConstraintScope,
    ConstraintStrength,
    OwnedConstraint,
    OwnershipPolicy,
    RelaxationPolicy,
    compute_plan_diff,
)
from itinerary_system.research_artifacts import PlanArtifactV2


def parent_plan() -> PlanArtifactV2:
    locked_stop = OwnedConstraint(
        constraint_id="keep_poi_c",
        origin=ConstraintOrigin.USER,
        strength=ConstraintStrength.LOCKED,
        scope=ConstraintScope.STOP,
        target_id="poi_c",
        relation="must_keep",
        value=True,
        confirmed=True,
        relaxation_policy=RelaxationPolicy.NEVER,
    )
    return PlanArtifactV2(
        plan_id="plan_parent",
        source_run_id="run_parent",
        planning_request_id="request_demo",
        catalog_snapshot_id="california_v1",
        context_snapshot_id="context_static_demo_2026_06",
        selected_stops=(
            {"stop_id": "poi_a", "day": 1, "stop_order": 1, "start_time": "09:00", "lodging_id": "hotel_sf"},
            {"stop_id": "poi_b", "day": 1, "stop_order": 2, "start_time": "10:00", "lodging_id": "hotel_sf"},
            {"stop_id": "poi_c", "day": 2, "stop_order": 1, "start_time": "11:00", "lodging_id": "hotel_y"},
        ),
        day_assignments={"poi_a": 1, "poi_b": 1, "poi_c": 2},
        sequence=("poi_a", "poi_b", "poi_c"),
        lodging_assignments={"1": "hotel_sf", "2": "hotel_y"},
        route_ids_by_day={1: "route_1", 2: "route_2"},
        owned_constraints=(locked_stop.to_record(),),
        created_at="2026-07-07T00:00:00+00:00",
    )


class PlanDiffTests(unittest.TestCase):
    def test_identical_content_has_zero_cost_and_preserved_days(self):
        parent = parent_plan()
        child = replace(parent, plan_id="plan_child", parent_plan_id=parent.plan_id)

        diff = compute_plan_diff(parent, child, OwnershipPolicy())

        self.assertEqual(diff.weighted_edit_cost, 0.0)
        self.assertEqual(diff.added_stops, ())
        self.assertEqual(diff.deleted_stops, ())
        self.assertEqual(diff.unchanged_days, (1, 2))

    def test_diff_detects_typed_changes_and_weighted_costs(self):
        parent = parent_plan()
        child = replace(
            parent,
            plan_id="plan_child",
            parent_plan_id=parent.plan_id,
            selected_stops=(
                {"stop_id": "poi_b", "day": 1, "stop_order": 1, "start_time": "10:30", "lodging_id": "hotel_sf2"},
                {"stop_id": "poi_a", "day": 2, "stop_order": 1, "start_time": "09:00", "lodging_id": "hotel_y"},
                {"stop_id": "poi_d", "day": 2, "stop_order": 2, "start_time": "12:00", "lodging_id": "hotel_y"},
            ),
            day_assignments={"poi_b": 1, "poi_a": 2, "poi_d": 2},
            sequence=("poi_b", "poi_a", "poi_d"),
            lodging_assignments={"1": "hotel_sf2", "2": "hotel_y"},
            route_ids_by_day={1: "route_1b"},
        )

        diff = compute_plan_diff(parent, child, OwnershipPolicy())

        self.assertEqual([change.stop_id for change in diff.added_stops], ["poi_d"])
        self.assertEqual([change.stop_id for change in diff.deleted_stops], ["poi_c"])
        self.assertEqual(diff.deleted_stops[0].owner_strength, "locked")
        self.assertEqual(diff.deleted_stops[0].cost, 2000.0)
        self.assertEqual([move.stop_id for move in diff.day_moves], ["poi_a"])
        self.assertEqual([shift.stop_id for shift in diff.time_shifts], ["poi_b"])
        self.assertEqual([change.stop_id for change in diff.reorder_changes], ["poi_b"])
        self.assertEqual([change.day for change in diff.lodging_changes], [1])
        self.assertEqual([change.day for change in diff.road_changes], [1, 2])
        self.assertGreater(diff.weighted_edit_cost, 2000.0)
        self.assertEqual(diff.unchanged_days, ())

    def test_incompatible_catalog_raises(self):
        parent = parent_plan()
        child = replace(parent, plan_id="child", catalog_snapshot_id="different_catalog")

        with self.assertRaises(ValueError):
            compute_plan_diff(parent, child, OwnershipPolicy())


if __name__ == "__main__":
    unittest.main()
