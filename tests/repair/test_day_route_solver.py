import unittest
from dataclasses import replace
from types import SimpleNamespace

from itinerary_system.plans import ConstraintScope, ConstraintStrength, OwnedConstraint, RelaxationPolicy
from itinerary_system.repair import RepairRadius, build_repair_neighborhood
from itinerary_system.repair.day_route_solver import (
    DayRouteSolver,
    DayRouteSolverConfig,
    solve_day_route_subproblem,
)
from itinerary_system.repair.master_model import build_repair_master_model
from itinerary_system.research_artifacts import PlanArtifactV2
from itinerary_system.routing import RouteMatrix, RouteMatrixCell, RouteMatrixNotPublicationEligible


def owned_constraint(constraint_id: str, *, target_id: str, strength: ConstraintStrength) -> dict:
    return OwnedConstraint(
        constraint_id=constraint_id,
        origin="user",
        strength=strength,
        scope=ConstraintScope.STOP,
        target_id=target_id,
        relation="protect",
        value=True,
        confirmed=True,
        relaxation_policy=RelaxationPolicy.NEVER if strength == ConstraintStrength.LOCKED else RelaxationPolicy.AUTO_WITH_PENALTY,
    ).to_record()


def parent_plan() -> PlanArtifactV2:
    return PlanArtifactV2(
        plan_id="parent_day_route",
        source_run_id="run_parent",
        planning_request_id="request_demo",
        catalog_snapshot_id="california_v1",
        context_snapshot_id="context_static_demo_2026_06",
        selected_stops=(
            {"stop_id": "poi_a", "name": "Ferry Building", "day": 1, "stop_order": 1, "lodging_id": "hotel_sf"},
            {
                "stop_id": "poi_b",
                "name": "Golden Gate Bridge",
                "day": 2,
                "stop_order": 1,
                "lodging_id": "hotel_y",
                "visit_duration_minutes": 30,
                "opening_start": "09:00",
                "opening_end": "12:00",
            },
            {
                "stop_id": "poi_c",
                "name": "Redwood Grove",
                "day": 2,
                "stop_order": 2,
                "lodging_id": "hotel_y",
                "visit_duration_minutes": 40,
                "opening_start": "09:30",
                "opening_end": "13:00",
            },
            {"stop_id": "poi_d", "name": "Big Sur Coast", "day": 3, "stop_order": 1, "lodging_id": "hotel_m"},
        ),
        day_assignments={"poi_a": 1, "poi_b": 2, "poi_c": 2, "poi_d": 3},
        sequence=("poi_a", "poi_b", "poi_c", "poi_d"),
        lodging_assignments={"1": "hotel_sf", "2": "hotel_y", "3": "hotel_m"},
        route_ids_by_day={1: "route_1", 2: "route_2", 3: "route_3"},
        owned_constraints=(
            owned_constraint("locked_bridge", target_id="poi_b", strength=ConstraintStrength.LOCKED),
            owned_constraint("soft_redwood", target_id="poi_c", strength=ConstraintStrength.SOFT),
        ),
        created_at="2026-07-07T00:00:00+00:00",
    )


def request(candidates: tuple[dict, ...] | None = None):
    return SimpleNamespace(
        request_id="repair_004",
        confirmed_constraints={"affected_days": (2,)},
        candidate_pois=candidates
        or (
            {
                "stop_id": "poi_e",
                "name": "Indoor Aquarium",
                "day": 2,
                "stop_order": 2,
                "lodging_id": "hotel_y",
                "visit_duration_minutes": 35,
                "opening_start": "09:00",
                "opening_end": "14:00",
            },
        ),
    )


def cell(origin: str, destination: str, minutes: float, *, road_validated: bool = True, fallback: bool = False):
    return RouteMatrixCell(
        origin_id=origin,
        destination_id=destination,
        distance_m=minutes * 1000.0,
        duration_s=minutes * 60.0,
        provider="unit",
        road_validated=road_validated,
        fallback_used=fallback,
    )


def route_matrix(*, fallback: bool = False) -> RouteMatrix:
    cells = {
        ("hotel_y_start", "poi_b"): cell("hotel_y_start", "poi_b", 10, road_validated=not fallback, fallback=fallback),
        ("poi_b", "poi_c"): cell("poi_b", "poi_c", 15),
        ("poi_c", "hotel_y_end"): cell("poi_c", "hotel_y_end", 20),
        ("poi_b", "poi_e"): cell("poi_b", "poi_e", 8),
        ("poi_e", "hotel_y_end"): cell("poi_e", "hotel_y_end", 12),
        ("hotel_y_start", "poi_a"): cell("hotel_y_start", "poi_a", 5),
        ("poi_a", "poi_e"): cell("poi_a", "poi_e", 8),
    }
    return RouteMatrix(matrix_id="matrix_day_route", context_snapshot_id="context_static_demo_2026_06", entity_ids=(), cells=cells)


def repair_model(candidates: tuple[dict, ...] | None = None):
    parent = parent_plan()
    repair_request = request(candidates)
    neighborhood = build_repair_neighborhood(parent, repair_request, RepairRadius.SAME_DAY_REPLACEMENT)
    return build_repair_master_model(parent, repair_request, neighborhood, route_matrix())


def config(strict: bool = False) -> DayRouteSolverConfig:
    return DayRouteSolverConfig(
        max_day_minutes=240,
        day_start_time="09:00",
        default_visit_minutes=30,
        strict_route_matrix=strict,
        start_anchor_by_day={2: "hotel_y_start"},
        end_anchor_by_day={2: "hotel_y_end"},
    )


class DayRouteSolverTests(unittest.TestCase):
    def test_empty_parent_lodging_is_not_a_fixed_lodging_violation(self):
        parent = replace(
            parent_plan(),
            selected_stops=tuple(
                {key: value for key, value in stop.items() if key != "lodging_id"}
                for stop in parent_plan().selected_stops
            ),
            lodging_assignments={},
        )
        repair_request = request()
        neighborhood = build_repair_neighborhood(parent, repair_request, RepairRadius.SAME_DAY_REPLACEMENT)
        model = build_repair_master_model(parent, repair_request, neighborhood, route_matrix())

        result = solve_day_route_subproblem(
            model,
            route_matrix(),
            day=2,
            config=config(),
            candidate_sequences=(("poi_b", "poi_c"),),
        )

        self.assertTrue(result.candidates[0].feasible)
        self.assertFalse(
            any("fix_assign_lodging" in violation for violation in result.candidates[0].violations)
        )

    def test_label_keyed_parent_day_assignments_use_canonical_stop_ids(self):
        parent = replace(
            parent_plan(),
            day_assignments={
                "Ferry Building": 1,
                "Golden Gate Bridge": 2,
                "Redwood Grove": 2,
                "Big Sur Coast": 3,
            },
        )
        repair_request = request()
        neighborhood = build_repair_neighborhood(parent, repair_request, RepairRadius.SAME_DAY_REPLACEMENT)
        model = build_repair_master_model(parent, repair_request, neighborhood, route_matrix())

        result = solve_day_route_subproblem(
            model,
            route_matrix(),
            day=2,
            config=config(),
            candidate_sequences=(("poi_b", "poi_c"),),
        )

        candidate = result.feasible_candidates[0]
        self.assertEqual(candidate.repair_solution.day_assignments["poi_b"], 2)
        self.assertEqual(candidate.repair_solution.day_assignments["poi_c"], 2)
        self.assertEqual(
            set(candidate.repair_solution.day_assignments),
            {"poi_a", "poi_b", "poi_c", "poi_d"},
        )

    def test_route_matrix_backed_candidate_uses_matrix_durations_and_extracts_solution(self):
        model = repair_model()

        result = solve_day_route_subproblem(
            model,
            route_matrix(),
            day=2,
            config=config(),
            candidate_sequences=(("poi_b", "poi_e"),),
        )

        candidate = result.feasible_candidates[0]
        self.assertEqual(candidate.route_pairs, (("hotel_y_start", "poi_b"), ("poi_b", "poi_e"), ("poi_e", "hotel_y_end")))
        self.assertEqual(candidate.travel_minutes, 30.0)
        self.assertEqual(candidate.visit_minutes, 65.0)
        self.assertEqual(candidate.total_minutes, 95.0)
        self.assertEqual(candidate.repair_solution.day_assignments["poi_e"], 2)
        child = model.extract_plan(candidate.repair_solution, child_plan_id="child_day_route")
        self.assertEqual(child.sequence, ("poi_a", "poi_b", "poi_e", "poi_d"))

    def test_default_solver_generates_parent_and_same_day_replacement_candidates(self):
        solver = DayRouteSolver(route_matrix(), config=config())

        result = solver.solve(repair_model(), day=2)

        sequences = {candidate.stop_sequence for candidate in result.candidates}
        self.assertIn(("poi_b", "poi_c"), sequences)
        self.assertIn(("poi_b", "poi_e"), sequences)
        self.assertTrue(all(candidate.route_evidence_ids for candidate in result.candidates))

    def test_opening_windows_visit_duration_and_fixed_day_assignment_are_enforced(self):
        late_candidate = {
            "stop_id": "poi_e",
            "name": "Indoor Aquarium",
            "day": 2,
            "lodging_id": "hotel_y",
            "visit_duration_minutes": 35,
            "opening_start": "09:00",
            "opening_end": "09:30",
        }
        model = repair_model((late_candidate,))

        result = solve_day_route_subproblem(
            model,
            route_matrix(),
            day=2,
            config=config(),
            candidate_sequences=(("poi_b", "poi_e"), ("poi_a", "poi_e")),
        )

        by_sequence = {candidate.stop_sequence: candidate for candidate in result.candidates}
        self.assertIn("opening_window_missed:poi_e", by_sequence[("poi_b", "poi_e")].violations)
        self.assertIn("fix_assign_day[poi_a]:poi_a", by_sequence[("poi_a", "poi_e")].violations)
        self.assertFalse(by_sequence[("poi_b", "poi_e")].feasible)
        self.assertFalse(by_sequence[("poi_a", "poi_e")].feasible)

    def test_strict_publication_mode_rejects_fallback_route_cells(self):
        model = repair_model()

        with self.assertRaises(RouteMatrixNotPublicationEligible):
            solve_day_route_subproblem(
                model,
                route_matrix(fallback=True),
                day=2,
                config=config(strict=True),
                candidate_sequences=(("poi_b", "poi_e"),),
            )


if __name__ == "__main__":
    unittest.main()
