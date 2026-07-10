from __future__ import annotations

from dataclasses import replace

from itinerary_system.evaluation import PlanEvaluator, PlanEvaluatorConfig
from itinerary_system.plans import ConstraintScope, ConstraintStrength, OwnedConstraint, RelaxationPolicy
from itinerary_system.research_artifacts import PlanArtifactV2, PlannerRun
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


def planner_run() -> PlannerRun:
    return PlannerRun(
        run_id="run_verify_001",
        planning_request_id="request_verify",
        catalog_snapshot_id="catalog_verify",
        context_snapshot_id="context_verify",
        planner_specification_id="repair-005-progressive-v1",
        method_requested="progressive_repair",
        method_executed="progressive_repair",
        execution_status="COMPLETED",
        solver_certification="FEASIBILITY_CERTIFIED",
        result_plan_id="plan_verify",
    )


def plan(*, sequence: tuple[str, ...] = ("poi_a", "poi_b"), certificate_id: str | None = None) -> PlanArtifactV2:
    return PlanArtifactV2(
        plan_id="plan_verify",
        parent_plan_id="parent_verify",
        source_run_id="run_verify_001",
        planning_request_id="request_verify",
        catalog_snapshot_id="catalog_verify",
        context_snapshot_id="context_verify",
        selected_stops=(
            {
                "stop_id": "poi_a",
                "name": "Morning Museum",
                "day": 1,
                "stop_order": 1,
                "lodging_id": "hotel_1",
                "visit_duration_minutes": 30,
                "opening_start": "09:00",
                "opening_end": "12:00",
                "weather_risk": 0.7,
            },
            {
                "stop_id": "poi_b",
                "name": "Afternoon Park",
                "day": 1,
                "stop_order": 2,
                "lodging_id": "hotel_1",
                "visit_duration_minutes": 30,
                "opening_start": "09:00",
                "opening_end": "14:00",
            },
        ),
        day_assignments={"poi_a": 1, "poi_b": 1},
        sequence=sequence,
        lodging_assignments={"1": "hotel_1"},
        ordered_days=({"day": 1, "stop_ids": ("poi_a", "poi_b")},),
        route_ids_by_day={1: "route_1"},
        owned_constraints=(
            owned_constraint("locked_a", target_id="poi_a", strength=ConstraintStrength.LOCKED),
            owned_constraint("soft_b", target_id="poi_b", strength=ConstraintStrength.SOFT),
        ),
        certificate_id=certificate_id,
        created_at="2026-07-08T00:00:00+00:00",
    )


def cell(origin: str, destination: str, minutes: float, *, road_validated: bool = True, fallback: bool = False) -> RouteMatrixCell:
    return RouteMatrixCell(
        origin_id=origin,
        destination_id=destination,
        distance_m=minutes * 1000.0,
        duration_s=minutes * 60.0,
        provider="unit",
        road_validated=road_validated,
        fallback_used=fallback,
    )


def route_matrix(*, road_validated: bool = True) -> RouteMatrix:
    return RouteMatrix(
        matrix_id="matrix_verify",
        context_snapshot_id="context_verify",
        entity_ids=(),
        cells={
            ("hotel_start", "poi_a"): cell("hotel_start", "poi_a", 10, road_validated=road_validated, fallback=not road_validated),
            ("poi_a", "poi_b"): cell("poi_a", "poi_b", 15, road_validated=road_validated, fallback=not road_validated),
            ("poi_b", "hotel_end"): cell("poi_b", "hotel_end", 10, road_validated=road_validated, fallback=not road_validated),
        },
    )


def evaluator(*, matrix: RouteMatrix | None = None) -> PlanEvaluator:
    return PlanEvaluator(
        route_matrix=matrix or route_matrix(),
        planner_runs={"run_verify_001": planner_run()},
        config=PlanEvaluatorConfig(
            start_anchor_by_day={1: "hotel_start"},
            end_anchor_by_day={1: "hotel_end"},
            max_day_minutes=120,
            weather_warning_threshold=0.5,
        ),
    )


def test_valid_road_checked_plan_gets_eligible_certificate_with_warnings_separated():
    certificate = evaluator().evaluate_final_plan(plan())

    assert certificate.comparison_eligibility == "eligible"
    assert certificate.evaluation_status == "PASSED_WITH_WARNINGS"
    assert certificate.failures == ()
    assert [warning.code for warning in certificate.warnings] == ["weather_risk_warning"]
    assert certificate.metrics["route_required_leg_count"] == 3.0
    assert certificate.valid_for_plan(plan())


def test_certificate_content_hash_detects_post_solve_mutation():
    original = plan()
    certificate = evaluator().evaluate_final_plan(original)
    mutated = replace(original, selected_stops=original.selected_stops[:-1], sequence=("poi_a",), certificate_id=certificate.certificate_id)

    assert not certificate.valid_for_plan(mutated)

    reevaluation = evaluator().evaluate_final_plan(mutated, expected_certificate=certificate)

    assert reevaluation.comparison_eligibility == "ineligible"
    assert "certificate_content_hash_mismatch" in {failure.code for failure in reevaluation.failures}


def test_unvalidated_route_matrix_blocks_comparison():
    certificate = evaluator(matrix=route_matrix(road_validated=False)).evaluate_final_plan(plan())

    assert certificate.comparison_eligibility == "ineligible"
    assert certificate.hard_feasibility_status == "FAILED"
    assert "route_matrix_not_publication_ready" in {failure.code for failure in certificate.failures}


def test_locked_stop_violation_is_recomputed_from_plan_artifact():
    child = replace(plan(), selected_stops=plan().selected_stops[1:], sequence=("poi_b",))

    certificate = evaluator().evaluate_final_plan(child)

    assert certificate.comparison_eligibility == "ineligible"
    assert "locked_stop_missing" in {failure.code for failure in certificate.failures}
