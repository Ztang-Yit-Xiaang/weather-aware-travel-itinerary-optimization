import unittest
from types import SimpleNamespace

from itinerary_system.plans import (
    ConstraintOrigin,
    ConstraintScope,
    ConstraintStrength,
    OwnedConstraint,
    RelaxationPolicy,
)
from itinerary_system.repair import REPAIR_RADIUS_ORDER, RepairRadius, affected_days, build_repair_neighborhood
from itinerary_system.research_artifacts import PlanArtifactV2


def owned_constraint(
    constraint_id: str,
    *,
    target_id: str,
    strength: ConstraintStrength | str,
    scope: ConstraintScope | str,
    origin: ConstraintOrigin | str = ConstraintOrigin.USER,
    confirmed: bool = True,
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
        confirmed=confirmed,
        relaxation_policy=relaxation_policy,
    ).to_record()


def parent_plan() -> PlanArtifactV2:
    return PlanArtifactV2(
        plan_id="parent_demo",
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
                "booked_yosemite_hotel",
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
            owned_constraint(
                "unconfirmed_prompt",
                target_id="poi_d",
                strength=ConstraintStrength.PREFERENCE,
                scope=ConstraintScope.STOP,
                origin=ConstraintOrigin.LLM_INTERPRETATION,
                confirmed=False,
            ),
            owned_constraint(
                "route_day3",
                target_id="route_3",
                strength=ConstraintStrength.SOFT,
                scope=ConstraintScope.ROUTE,
            ),
        ),
        created_at="2026-07-07T00:00:00+00:00",
    )


def request(**confirmed_constraints):
    return SimpleNamespace(confirmed_constraints=confirmed_constraints)


class RepairNeighborhoodTests(unittest.TestCase):
    def test_same_day_replacement_freezes_unaffected_days_and_protected_constraints(self):
        neighborhood = build_repair_neighborhood(
            parent_plan(),
            request(affected_days=(2,)),
            RepairRadius.SAME_DAY_REPLACEMENT,
        )

        self.assertEqual(neighborhood.editable_days, (2,))
        self.assertEqual(neighborhood.editable_stop_ids, ("poi_b", "poi_c"))
        self.assertEqual(neighborhood.frozen_stop_ids, ("poi_a", "poi_d"))
        self.assertEqual(neighborhood.editable_lodging_nights, ())
        self.assertIn("locked_bridge", neighborhood.frozen_constraint_ids)
        self.assertIn("booked_yosemite_hotel", neighborhood.frozen_constraint_ids)
        self.assertNotIn("soft_redwood", neighborhood.frozen_constraint_ids)
        self.assertNotIn("unconfirmed_prompt", neighborhood.frozen_constraint_ids)
        self.assertIn("route_day3", neighborhood.frozen_constraint_ids)

    def test_adjacent_day_move_includes_boundary_route_days_and_lodging_nights(self):
        neighborhood = build_repair_neighborhood(
            parent_plan(),
            request(affected_days=(2,)),
            RepairRadius.ADJACENT_DAY_MOVE,
        )

        self.assertEqual(neighborhood.editable_days, (1, 2, 3))
        self.assertEqual(neighborhood.editable_stop_ids, ("poi_a", "poi_b", "poi_c", "poi_d"))
        self.assertEqual(neighborhood.boundary_route_days, (1, 2, 3))
        self.assertEqual(neighborhood.editable_lodging_nights, (1, 2, 3))
        self.assertIn("locked_bridge", neighborhood.frozen_constraint_ids)
        self.assertIn("booked_yosemite_hotel", neighborhood.frozen_constraint_ids)

    def test_booked_lodging_requires_explicit_relaxation_and_lodging_edit_radius(self):
        preserving = build_repair_neighborhood(
            parent_plan(),
            request(affected_days=(2,), allow_booked_relaxation=True),
            RepairRadius.HOTEL_PRESERVING_REROUTE,
        )
        changing = build_repair_neighborhood(
            parent_plan(),
            request(affected_days=(2,), allow_booked_relaxation=True),
            RepairRadius.HOTEL_CHANGING_REPAIR,
        )

        self.assertEqual(preserving.editable_lodging_nights, ())
        self.assertIn("booked_yosemite_hotel", preserving.frozen_constraint_ids)
        self.assertEqual(changing.editable_lodging_nights, (1, 2, 3))
        self.assertNotIn("booked_yosemite_hotel", changing.frozen_constraint_ids)
        self.assertIn("locked_bridge", changing.frozen_constraint_ids)

    def test_affected_days_include_target_stop_day_and_requested_day(self):
        days = affected_days(
            request(move_stops={"Redwood Grove": 3}),
            parent=parent_plan(),
        )

        self.assertEqual(days, (2, 3))

    def test_full_reoptimization_is_last_and_exposes_all_parent_assignments(self):
        neighborhood = build_repair_neighborhood(
            parent_plan(),
            request(affected_days=(2,)),
            RepairRadius.FULL_REOPTIMIZATION,
        )

        self.assertEqual(REPAIR_RADIUS_ORDER[-1], RepairRadius.FULL_REOPTIMIZATION)
        self.assertEqual(neighborhood.editable_days, (1, 2, 3))
        self.assertEqual(neighborhood.editable_stop_ids, ("poi_a", "poi_b", "poi_c", "poi_d"))
        self.assertEqual(neighborhood.editable_lodging_nights, (1, 2, 3))
        self.assertIn("fallback:full_reoptimization", neighborhood.reason_codes)


if __name__ == "__main__":
    unittest.main()
