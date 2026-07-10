from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from itinerary_system.plans import ConstraintScope, ConstraintStrength, OwnedConstraint, RelaxationPolicy
from itinerary_system.plans.repository import PlanRepository
from itinerary_system.repair import (
    DayRouteSolverConfig,
    ProgressiveRepairController,
    RepairRadius,
)
from itinerary_system.research_artifacts import PlanArtifactV2
from itinerary_system.routing import RouteMatrix, RouteMatrixCell


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
        plan_id="parent_progressive",
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


def repair_request(*, allowed_radii: tuple[RepairRadius, ...]) -> SimpleNamespace:
    return SimpleNamespace(
        request_id="repair_005",
        parent_plan_id="parent_progressive",
        allowed_radii=allowed_radii,
        confirmed_constraints={"affected_days": (2,)},
        candidate_pois=(
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


def cell(origin: str, destination: str, minutes: float) -> RouteMatrixCell:
    return RouteMatrixCell(
        origin_id=origin,
        destination_id=destination,
        distance_m=minutes * 1000.0,
        duration_s=minutes * 60.0,
        provider="unit",
        road_validated=True,
    )


def route_matrix() -> RouteMatrix:
    cells = {
        ("hotel_y_start", "poi_b"): cell("hotel_y_start", "poi_b", 10),
        ("poi_b", "poi_c"): cell("poi_b", "poi_c", 15),
        ("poi_c", "hotel_y_end"): cell("poi_c", "hotel_y_end", 20),
        ("poi_b", "poi_e"): cell("poi_b", "poi_e", 8),
        ("poi_e", "hotel_y_end"): cell("poi_e", "hotel_y_end", 12),
    }
    return RouteMatrix(matrix_id="matrix_progressive", context_snapshot_id="context_static_demo_2026_06", entity_ids=(), cells=cells)


def day_route_config() -> DayRouteSolverConfig:
    return DayRouteSolverConfig(
        max_day_minutes=240,
        day_start_time="09:00",
        default_visit_minutes=30,
        start_anchor_by_day={2: "hotel_y_start"},
        end_anchor_by_day={2: "hotel_y_end"},
    )


class SequenceGateEvaluator:
    def __init__(self, eligible_sequences: set[tuple[str, ...]] | None = None) -> None:
        self.eligible_sequences = eligible_sequences or set()
        self.calls: list[tuple[str, ...]] = []

    def evaluate(self, child_plan: PlanArtifactV2) -> dict:
        sequence = tuple(child_plan.sequence)
        self.calls.append(sequence)
        eligible = sequence in self.eligible_sequences
        return {
            "evaluation_id": f"eval_{len(self.calls)}",
            "eligible": eligible,
            "comparison_eligibility": "eligible" if eligible else "ineligible",
            "evaluation_status": "PASSED" if eligible else "FAILED",
            "reason": "accepted_sequence" if eligible else "not_accepted",
        }


def repository_with_parent(tmp_path: Path) -> PlanRepository:
    repository = PlanRepository(tmp_path / "plans")
    repository.save(parent_plan())
    return repository


def controller(repository: PlanRepository, evaluator: SequenceGateEvaluator) -> ProgressiveRepairController:
    return ProgressiveRepairController(
        plan_repository=repository,
        route_matrix=route_matrix(),
        evaluator=evaluator,
        day_route_config=day_route_config(),
    )


def test_stops_at_smallest_independently_eligible_radius_and_saves_child(tmp_path: Path):
    repository = repository_with_parent(tmp_path)
    evaluator = SequenceGateEvaluator({("poi_a", "poi_b", "poi_e", "poi_d")})

    outcome = controller(repository, evaluator).repair_progressively(
        repair_request(
            allowed_radii=(
                RepairRadius.SAME_DAY_REPLACEMENT,
                RepairRadius.FULL_REOPTIMIZATION,
            )
        )
    )

    assert outcome.status == "accepted"
    assert outcome.accepted_radius == RepairRadius.SAME_DAY_REPLACEMENT
    assert [attempt.radius for attempt in outcome.attempts] == [RepairRadius.SAME_DAY_REPLACEMENT]
    assert outcome.child_plan is not None
    assert outcome.child_plan.sequence == ("poi_a", "poi_b", "poi_e", "poi_d")
    assert repository.exists(outcome.child_plan.plan_id)
    assert [record.eligible for record in outcome.attempts[0].candidate_evaluations][-1] is True


def test_full_reoptimization_is_attempted_only_as_final_configured_fallback(tmp_path: Path):
    repository = repository_with_parent(tmp_path)
    evaluator = SequenceGateEvaluator()

    outcome = controller(repository, evaluator).repair_progressively(
        repair_request(
            allowed_radii=(
                RepairRadius.FULL_REOPTIMIZATION,
                RepairRadius.SAME_DAY_REPLACEMENT,
            )
        )
    )

    assert outcome.status == "infeasible"
    assert [attempt.radius for attempt in outcome.attempts] == [
        RepairRadius.SAME_DAY_REPLACEMENT,
        RepairRadius.FULL_REOPTIMIZATION,
    ]
    assert all(attempt.candidate_evaluations for attempt in outcome.attempts)
    assert outcome.diagnosis is not None
    assert outcome.diagnosis.attempted_radii == tuple(attempt.radius for attempt in outcome.attempts)


def test_no_success_returns_smallest_relaxation_diagnosis_with_component_metrics(tmp_path: Path):
    repository = repository_with_parent(tmp_path)
    evaluator = SequenceGateEvaluator()

    outcome = controller(repository, evaluator).repair_progressively(
        repair_request(allowed_radii=(RepairRadius.SAME_DAY_REPLACEMENT,))
    )

    assert outcome.status == "infeasible"
    assert outcome.diagnosis is not None
    assert outcome.diagnosis.status == "no_eligible_radius"
    assert outcome.diagnosis.minimum_relaxation_radius == RepairRadius.SAME_DAY_REPLACEMENT
    assert outcome.diagnosis.minimum_relaxation_count == 0.0
    assert outcome.diagnosis.best_weighted_edit_cost == 0.0
    assert "evaluation_ineligible" in outcome.diagnosis.failure_reasons
    assert all(record.weighted_edit_cost is not None for record in outcome.attempts[0].candidate_evaluations)
