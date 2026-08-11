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
        self.assertEqual(diff.schema_version, "plan-diff-v1")
        self.assertNotIn("duration_changes", diff.to_record())
        self.assertNotIn("time_window_changes", diff.to_record())
        self.assertNotIn("role_changes", diff.to_record())

    def test_duration_change_uses_versioned_typed_record_and_preserves_missingness(self):
        parent = parent_plan()
        stops = [dict(stop) for stop in parent.selected_stops]
        stops[0]["duration_constraint"] = {
            "mode": "exact",
            "preferred_minutes": 90,
            "minimum_minutes": 90,
            "maximum_minutes": 90,
        }
        stops[0]["visit_duration_minutes"] = 90
        child = replace(
            parent,
            plan_id="plan_duration_child",
            parent_plan_id=parent.plan_id,
            selected_stops=tuple(stops),
        )

        diff = compute_plan_diff(
            parent,
            child,
            OwnershipPolicy(),
            default_visit_minutes=45,
        )
        record = diff.to_record()

        self.assertEqual(record["schema_version"], "plan-diff-v2")
        self.assertEqual(len(record["duration_changes"]), 1)
        change = record["duration_changes"][0]
        self.assertEqual(change["stop_id"], "poi_a")
        self.assertIsNone(change["from_constraint"])
        self.assertIsNone(change["from_minutes"])
        self.assertEqual(change["from_source"], "unavailable")
        self.assertEqual(change["accounting_from_minutes"], 45.0)
        self.assertEqual(change["accounting_from_source"], "configured_evaluator_default")
        self.assertEqual(change["to_minutes"], 90.0)
        self.assertEqual(change["delta_minutes"], 45.0)
        self.assertEqual(record["time_shifts"], [])
        self.assertNotIn("role_changes", record)

    def test_time_window_change_uses_v3_without_mutating_v1_or_v2_shapes(self):
        parent = parent_plan()
        stops = [dict(stop) for stop in parent.selected_stops]
        stops[0]["time_window_constraint"] = {
            "schema_version": "stop-time-window-constraint-v1",
            "earliest_arrival": "10:00",
            "latest_departure": "12:00",
            "early_arrival_policy": "wait_until_earliest_arrival",
            "latest_departure_semantics": "departure_after_visit",
        }
        child = replace(
            parent,
            plan_id="plan_window_child",
            parent_plan_id=parent.plan_id,
            selected_stops=tuple(stops),
        )

        record = compute_plan_diff(parent, child, OwnershipPolicy()).to_record()

        self.assertEqual(record["schema_version"], "plan-diff-v3")
        self.assertNotIn("duration_changes", record)
        self.assertNotIn("role_changes", record)
        self.assertEqual(record["time_shifts"], [])
        self.assertEqual(
            record["time_window_changes"],
            [
                {
                    "stop_id": "poi_a",
                    "day": 1,
                    "from_constraint": None,
                    "to_constraint": stops[0]["time_window_constraint"],
                    "from_earliest_arrival": None,
                    "to_earliest_arrival": "10:00",
                    "from_latest_departure": None,
                    "to_latest_departure": "12:00",
                    "owner_strength": "",
                    "cost": 0.25,
                }
            ],
        )

    def test_time_window_diff_rejects_malformed_nonnull_constraint(self):
        parent = parent_plan()
        malformed = [dict(stop) for stop in parent.selected_stops]
        malformed[0]["time_window_constraint"] = {
            "schema_version": "stop-time-window-constraint-v1",
            "earliest_arrival": "10:00",
            "latest_departure": "12:00",
            "early_arrival_policy": "wait_until_earliest_arrival",
            "latest_departure_semantics": "latest_start",
        }
        child = replace(parent, plan_id="malformed_window", selected_stops=tuple(malformed))

        with self.assertRaisesRegex(ValueError, "time_window_constraint"):
            compute_plan_diff(parent, child)

    def test_role_change_uses_v4_and_preserves_missingness_and_attribute_ownership(self):
        parent = parent_plan()
        stops = [dict(stop) for stop in parent.selected_stops]
        stops[0]["itinerary_role"] = "meal"
        stops[0]["itinerary_role_source"] = "user_declared_itinerary_role"
        child = replace(
            parent,
            plan_id="plan_role_child",
            parent_plan_id=parent.plan_id,
            selected_stops=tuple(stops),
        )

        record = compute_plan_diff(parent, child, OwnershipPolicy()).to_record()

        self.assertEqual(record["schema_version"], "plan-diff-v4")
        self.assertNotIn("duration_changes", record)
        self.assertNotIn("time_window_changes", record)
        self.assertEqual(record["time_shifts"], [])
        self.assertEqual(
            record["role_changes"],
            [
                {
                    "stop_id": "poi_a",
                    "day": 1,
                    "from_role": None,
                    "to_role": "meal",
                    "from_source": "unavailable",
                    "to_source": "user_declared_itinerary_role",
                    "owner_strength": "",
                    "cost": 0.25,
                }
            ],
        )
        self.assertEqual(record["weighted_edit_cost"], 0.25)

        soft_role = OwnedConstraint(
            constraint_id="prefer_role_a",
            origin=ConstraintOrigin.USER,
            strength=ConstraintStrength.SOFT,
            scope=ConstraintScope.STOP,
            target_id="poi_a",
            relation="role",
            value="activity",
            confirmed=True,
            relaxation_policy=RelaxationPolicy.ALWAYS,
        )
        role_owned_parent = replace(
            parent,
            owned_constraints=parent.owned_constraints + (soft_role.to_record(),),
        )
        role_owned_child = replace(child, parent_plan_id=role_owned_parent.plan_id)
        owned_record = compute_plan_diff(
            role_owned_parent,
            role_owned_child,
            OwnershipPolicy(),
        ).to_record()
        self.assertEqual(owned_record["role_changes"][0]["owner_strength"], "soft")
        self.assertEqual(owned_record["role_changes"][0]["cost"], 2.5)

    def test_role_diff_rejects_malformed_provenance(self):
        parent = parent_plan()
        malformed = [dict(stop) for stop in parent.selected_stops]
        malformed[0]["itinerary_role"] = "meal"
        malformed[0]["itinerary_role_source"] = "place_category_inference"
        child = replace(parent, plan_id="malformed_role", selected_stops=tuple(malformed))

        with self.assertRaisesRegex(ValueError, "itinerary_role"):
            compute_plan_diff(parent, child)

    def test_duration_diff_rejects_malformed_constraints_and_conflicting_aliases(self):
        parent = parent_plan()
        malformed = [dict(stop) for stop in parent.selected_stops]
        malformed[0]["duration_constraint"] = {
            "mode": "range",
            "preferred_minutes": None,
            "minimum_minutes": 30,
            "maximum_minutes": 60,
        }
        child = replace(parent, plan_id="malformed", selected_stops=tuple(malformed))
        with self.assertRaisesRegex(ValueError, "canonical exact"):
            compute_plan_diff(parent, child, default_visit_minutes=45)

        conflicting = [dict(stop) for stop in parent.selected_stops]
        conflicting[0]["visit_duration_minutes"] = 45
        conflicting[0]["duration_minutes"] = 60
        child = replace(parent, plan_id="conflicting", selected_stops=tuple(conflicting))
        with self.assertRaisesRegex(ValueError, "aliases disagree"):
            compute_plan_diff(parent, child, default_visit_minutes=45)

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
