import unittest

import pandas as pd

from itinerary_system.plans import (
    ConstraintOrigin,
    ConstraintScope,
    ConstraintStrength,
    OwnedConstraint,
    RelaxationPolicy,
    plan_from_phase0_route_stops,
    validate_owned_constraints,
)


class PlanModelTests(unittest.TestCase):
    def test_owned_constraint_round_trip_preserves_enum_values(self):
        constraint = OwnedConstraint(
            constraint_id="keep_golden_gate",
            origin=ConstraintOrigin.USER,
            strength=ConstraintStrength.LOCKED,
            scope=ConstraintScope.STOP,
            target_id="poi_golden_gate",
            relation="must_keep",
            value=True,
            confirmed=True,
            relaxation_policy=RelaxationPolicy.NEVER,
            evidence_refs=("user_profile",),
        )

        record = constraint.to_record()
        restored = OwnedConstraint.from_record(record)

        self.assertEqual(record["origin"], "user")
        self.assertEqual(restored, constraint)
        self.assertTrue(restored.is_active)

    def test_unconfirmed_llm_constraint_is_inactive(self):
        constraint = OwnedConstraint(
            constraint_id="llm_guess",
            origin="llm_interpretation",
            strength="soft",
            scope="stop",
            target_id="poi_a",
            relation="prefer",
            value=True,
            confirmed=False,
            relaxation_policy="auto_with_penalty",
        )

        self.assertFalse(constraint.is_active)
        self.assertFalse(constraint.can_relax())

    def test_locked_and_booked_constraints_require_safe_relaxation_policy(self):
        locked = OwnedConstraint(
            constraint_id="locked_auto",
            origin="user",
            strength="locked",
            scope="stop",
            target_id="poi_a",
            relation="must_keep",
            value=True,
            confirmed=True,
            relaxation_policy="auto_with_penalty",
        )
        booked = OwnedConstraint(
            constraint_id="booked_auto",
            origin="user_booking",
            strength="booked",
            scope="lodging",
            target_id="hotel_a",
            relation="reserved",
            value=True,
            confirmed=True,
            relaxation_policy="always",
        )

        errors = validate_owned_constraints((locked, booked))

        self.assertFalse(locked.can_relax(explicit_permission=True))
        self.assertIn("locked_auto is locked but has automatic relaxation", errors)
        self.assertIn("booked_auto is booked but does not require explicit relaxation permission", errors)

    def test_phase0_route_stops_frame_builds_v2_plan(self):
        frame = pd.DataFrame(
            [
                {
                    "attraction_name": "Ferry Building",
                    "day": 1,
                    "stop_order": 2,
                    "route_end_name": "San Francisco Hotel",
                    "route_id": "route_day_1",
                    "catalog_snapshot_id": "california_v1",
                    "context_snapshot_id": "context_static_demo_2026_06",
                },
                {
                    "attraction_name": "Golden Gate Bridge",
                    "day": 1,
                    "stop_order": 1,
                    "route_end_name": "San Francisco Hotel",
                    "route_id": "route_day_1",
                    "catalog_snapshot_id": "california_v1",
                    "context_snapshot_id": "context_static_demo_2026_06",
                },
            ]
        )

        plan = plan_from_phase0_route_stops(frame, plan_id="plan_phase0_parent")

        self.assertEqual(plan.plan_id, "plan_phase0_parent")
        self.assertEqual(plan.sequence, ("Golden Gate Bridge", "Ferry Building"))
        self.assertEqual(plan.ordered_days, ({"day": 1, "stop_ids": ("Golden Gate Bridge", "Ferry Building")},))
        self.assertEqual(plan.route_ids_by_day, {1: "route_day_1"})


if __name__ == "__main__":
    unittest.main()
