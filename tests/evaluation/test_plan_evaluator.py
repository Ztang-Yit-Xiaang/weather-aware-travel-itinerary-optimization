from __future__ import annotations

from copy import deepcopy
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


def role_constraint(
    constraint_id: str,
    *,
    target_id: str,
    role: str,
    strength: ConstraintStrength,
    relaxation_policy: RelaxationPolicy | None = None,
) -> dict:
    return OwnedConstraint(
        constraint_id=constraint_id,
        origin="user",
        strength=strength,
        scope=ConstraintScope.STOP,
        target_id=target_id,
        relation="role",
        value=role,
        confirmed=True,
        relaxation_policy=relaxation_policy
        or (
            RelaxationPolicy.NEVER
            if strength in {
                ConstraintStrength.LOCKED,
                ConstraintStrength.BOOKED,
                ConstraintStrength.HARD,
            }
            else RelaxationPolicy.AUTO_WITH_PENALTY
        ),
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


def test_optional_lodging_policy_accepts_plan_without_lodging_assignments():
    no_lodging_plan = replace(plan(), lodging_assignments={})
    configured = PlanEvaluator(
        route_matrix=route_matrix(),
        planner_runs={"run_verify_001": planner_run()},
        config=PlanEvaluatorConfig(
            start_anchor_by_day={1: "hotel_start"},
            end_anchor_by_day={1: "hotel_end"},
            max_day_minutes=120,
            require_lodging_assignments=False,
        ),
    )

    certificate = configured.evaluate_final_plan(no_lodging_plan)

    assert "lodging_missing" not in {failure.code for failure in certificate.failures}
    assert certificate.comparison_eligibility == "eligible"


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


def test_context_requirements_are_independently_recomputed():
    configured = PlanEvaluator(
        route_matrix=route_matrix(),
        planner_runs={"run_verify_001": planner_run()},
        config=PlanEvaluatorConfig(
            start_anchor_by_day={1: "hotel_start"},
            end_anchor_by_day={1: "hotel_end"},
            max_day_minutes=120,
            required_stop_ids=("poi_required",),
            excluded_stop_ids=("poi_b",),
            closed_route_ids=("route_1",),
            unavailable_lodging_ids=("hotel_1",),
        ),
    )

    certificate = configured.evaluate_final_plan(plan())
    codes = {failure.code for failure in certificate.failures}

    assert "context_required_stop_missing" in codes
    assert "context_excluded_stop_selected" in codes
    assert "context_closed_route_selected" in codes
    assert "context_unavailable_lodging_selected" in codes
    assert certificate.comparison_eligibility == "ineligible"


def test_quality_metrics_are_recomputed_against_frozen_reference_plan():
    reference = replace(
        plan(),
        selected_stops=(
            {**plan().selected_stops[0], "utility": 10.0, "weather_risk": 0.8},
            {**plan().selected_stops[1], "utility": 10.0, "weather_risk": 0.6},
        ),
    )
    child = replace(
        plan(),
        selected_stops=(
            {**plan().selected_stops[0], "utility": 10.0, "weather_risk": 0.2},
            {**plan().selected_stops[1], "utility": 5.0, "weather_risk": 0.1},
        ),
    )
    configured = PlanEvaluator(
        route_matrix=route_matrix(),
        planner_runs={"run_verify_001": planner_run()},
        config=PlanEvaluatorConfig(
            start_anchor_by_day={1: "hotel_start"},
            end_anchor_by_day={1: "hotel_end"},
            max_day_minutes=120,
        ),
        reference_plan=reference,
    )

    certificate = configured.evaluate_final_plan(child)

    assert certificate.metrics["utility_retained"] == 0.75
    assert abs(certificate.metrics["weather_risk_delta"] - 1.1) < 1e-9
    assert certificate.metrics["preservation_rate"] == 1.0


def test_weather_risk_delta_requires_complete_finite_reference_and_child_evidence():
    reference = replace(
        plan(),
        selected_stops=tuple(
            {**stop, "weather_risk": 0.8 - (index * 0.2)}
            for index, stop in enumerate(plan().selected_stops)
        ),
    )
    for invalid_value in (None, float("nan"), float("inf")):
        child_stops = [
            {**stop, "weather_risk": 0.2 - (index * 0.1)}
            for index, stop in enumerate(plan().selected_stops)
        ]
        if invalid_value is None:
            child_stops[1].pop("weather_risk")
        else:
            child_stops[1]["weather_risk"] = invalid_value
        child = replace(plan(), selected_stops=tuple(child_stops))
        configured = PlanEvaluator(
            route_matrix=route_matrix(),
            planner_runs={"run_verify_001": planner_run()},
            config=PlanEvaluatorConfig(
                start_anchor_by_day={1: "hotel_start"},
                end_anchor_by_day={1: "hotel_end"},
                max_day_minutes=120,
            ),
            reference_plan=reference,
        )

        certificate = configured.evaluate_final_plan(child)

        assert "weather_risk_delta" not in certificate.metrics


def test_required_earliest_arrival_waits_before_service_without_mutating_plan():
    original = plan()
    before = deepcopy(original.to_record())
    stops = [dict(stop) for stop in original.selected_stops]
    stops[0]["time_window_constraint"] = {
        "schema_version": "stop-time-window-constraint-v1",
        "earliest_arrival": "10:00",
        "latest_departure": "10:45",
        "early_arrival_policy": "wait_until_earliest_arrival",
        "latest_departure_semantics": "departure_after_visit",
    }
    child = replace(original, selected_stops=tuple(stops))
    configured = PlanEvaluator(
        route_matrix=route_matrix(),
        planner_runs={"run_verify_001": planner_run()},
        config=PlanEvaluatorConfig(
            start_anchor_by_day={1: "hotel_start"},
            end_anchor_by_day={1: "hotel_end"},
            max_day_minutes=180,
            include_schedule_metrics=True,
        ),
    )

    accounting = configured.evaluate_time_window_schedule_accounting(child)
    certificate = configured.evaluate_final_plan(child)
    trace = accounting["stop_traces"][0]

    assert trace["road_arrival_minute"] == 550.0
    assert trace["opening_wait_minutes"] == 0.0
    assert trace["required_window_wait_minutes"] == 50.0
    assert trace["service_start_minute"] == 600.0
    assert trace["visit_minutes"] == 30.0
    assert trace["departure_minute"] == 630.0
    assert trace["latest_departure_status"] == "satisfied"
    assert trace["latest_departure_overrun_minutes"] == 0.0
    assert certificate.comparison_eligibility == "eligible"
    assert original.to_record() == before


def test_itinerary_role_is_optional_but_explicit_role_requires_user_provenance():
    legacy = plan()
    legacy_certificate = evaluator().evaluate_final_plan(legacy)
    assert "itinerary_stop_role_invalid" not in {
        failure.code for failure in legacy_certificate.failures
    }

    explicit_stops = [dict(stop) for stop in legacy.selected_stops]
    explicit_stops[0]["itinerary_role"] = "meal"
    explicit_stops[0]["itinerary_role_source"] = "user_declared_itinerary_role"
    explicit = replace(legacy, selected_stops=tuple(explicit_stops))
    explicit_certificate = evaluator().evaluate_final_plan(explicit)
    assert "itinerary_stop_role_invalid" not in {
        failure.code for failure in explicit_certificate.failures
    }

    malformed_stops = [dict(stop) for stop in legacy.selected_stops]
    malformed_stops[0]["itinerary_role"] = "meal"
    malformed_stops[0]["itinerary_role_source"] = "place_category_inference"
    malformed = replace(legacy, selected_stops=tuple(malformed_stops))
    malformed_certificate = evaluator().evaluate_final_plan(malformed)
    assert "itinerary_stop_role_invalid" in {
        failure.code for failure in malformed_certificate.failures
    }


def test_owned_role_constraints_are_value_aware_and_existence_lock_is_independent():
    base = plan()
    stops = [dict(stop) for stop in base.selected_stops]
    stops[0]["itinerary_role"] = "meal"
    stops[0]["itinerary_role_source"] = "user_declared_itinerary_role"
    explicit = replace(base, selected_stops=tuple(stops))

    existence_only = evaluator().evaluate_final_plan(explicit)
    assert "owned_role_constraint_unsatisfied" not in {
        finding.code
        for finding in (*existence_only.failures, *existence_only.warnings)
    }

    matching = replace(
        explicit,
        owned_constraints=explicit.owned_constraints
        + (
            role_constraint(
                "role_a",
                target_id="poi_a",
                role="meal",
                strength=ConstraintStrength.LOCKED,
            ),
        ),
    )
    matching_certificate = evaluator().evaluate_final_plan(matching)
    assert "owned_role_constraint_unsatisfied" not in {
        finding.code
        for finding in (*matching_certificate.failures, *matching_certificate.warnings)
    }

    hard_mismatch = replace(
        explicit,
        owned_constraints=explicit.owned_constraints
        + (
            role_constraint(
                "role_a",
                target_id="poi_a",
                role="activity",
                strength=ConstraintStrength.LOCKED,
            ),
        ),
    )
    hard_certificate = evaluator().evaluate_final_plan(hard_mismatch)
    assert "owned_role_constraint_unsatisfied" in {
        failure.code for failure in hard_certificate.failures
    }

    soft_mismatch = replace(
        explicit,
        owned_constraints=explicit.owned_constraints
        + (
            role_constraint(
                "role_a",
                target_id="poi_a",
                role="activity",
                strength=ConstraintStrength.SOFT,
            ),
        ),
    )
    soft_certificate = evaluator().evaluate_final_plan(soft_mismatch)
    assert "owned_role_constraint_unsatisfied" in {
        warning.code for warning in soft_certificate.warnings
    }
    assert "owned_role_constraint_unsatisfied" not in {
        failure.code for failure in soft_certificate.failures
    }

    nonrelaxable_mismatch = replace(
        explicit,
        owned_constraints=explicit.owned_constraints
        + (
            role_constraint(
                "role_a",
                target_id="poi_a",
                role="activity",
                strength=ConstraintStrength.STRONG,
                relaxation_policy=RelaxationPolicy.NEVER,
            ),
        ),
    )
    nonrelaxable_certificate = evaluator().evaluate_final_plan(
        nonrelaxable_mismatch
    )
    assert "owned_role_constraint_unsatisfied" in {
        failure.code for failure in nonrelaxable_certificate.failures
    }


def test_required_latest_departure_is_checked_after_visit_with_exact_overrun():
    original = plan()
    stops = [dict(stop) for stop in original.selected_stops]
    stops[0]["time_window_constraint"] = {
        "schema_version": "stop-time-window-constraint-v1",
        "earliest_arrival": None,
        "latest_departure": "09:35",
        "early_arrival_policy": "wait_until_earliest_arrival",
        "latest_departure_semantics": "departure_after_visit",
    }
    child = replace(original, selected_stops=tuple(stops))
    configured = PlanEvaluator(
        route_matrix=route_matrix(),
        planner_runs={"run_verify_001": planner_run()},
        config=PlanEvaluatorConfig(
            start_anchor_by_day={1: "hotel_start"},
            end_anchor_by_day={1: "hotel_end"},
            max_day_minutes=180,
            include_schedule_metrics=True,
        ),
    )

    accounting = configured.evaluate_time_window_schedule_accounting(child)
    certificate = configured.evaluate_final_plan(child)
    trace = accounting["stop_traces"][0]

    assert trace["road_arrival_minute"] == 550.0
    assert trace["service_start_minute"] == 550.0
    assert trace["departure_minute"] == 580.0
    assert trace["latest_departure_status"] == "violated"
    assert trace["latest_departure_overrun_minutes"] == 5.0
    assert trace["failure_codes"] == [
        "stop_time_window_latest_departure_exceeded"
    ]
    assert [finding.code for finding in certificate.failures] == [
        "stop_time_window_latest_departure_exceeded"
    ]


def test_malformed_required_time_window_is_an_artifact_failure():
    original = plan()
    stops = [dict(stop) for stop in original.selected_stops]
    stops[0]["time_window_constraint"] = {
        "schema_version": "stop-time-window-constraint-v1",
        "earliest_arrival": "10:00",
        "latest_departure": "10:45",
        "early_arrival_policy": "wait_until_earliest_arrival",
        "latest_departure_semantics": "latest_start",
    }
    child = replace(original, selected_stops=tuple(stops))

    certificate = evaluator().evaluate_final_plan(child)

    assert certificate.artifact_grounding_status == "FAILED"
    assert certificate.comparison_eligibility == "ineligible"
    assert "time_window_constraint_invalid" in {
        finding.code for finding in certificate.failures
    }


def test_zero_or_nonfinite_reference_utility_is_not_reported_as_perfect_retention():
    for value in (0.0, float("nan"), float("inf")):
        reference = replace(
            plan(),
            selected_stops=tuple({**stop, "utility": value} for stop in plan().selected_stops),
        )
        child = replace(
            plan(),
            selected_stops=tuple({**stop, "utility": 5.0} for stop in plan().selected_stops),
        )
        configured = PlanEvaluator(
            route_matrix=route_matrix(),
            planner_runs={"run_verify_001": planner_run()},
            config=PlanEvaluatorConfig(
                start_anchor_by_day={1: "hotel_start"},
                end_anchor_by_day={1: "hotel_end"},
                max_day_minutes=120,
            ),
            reference_plan=reference,
        )

        certificate = configured.evaluate_final_plan(child)

        assert "utility_retained" not in certificate.metrics


def test_off_sequence_selected_stop_is_ineligible_and_excluded_from_quality_metrics():
    reference = replace(
        plan(),
        selected_stops=tuple({**stop, "utility": 10.0} for stop in plan().selected_stops),
    )
    extra = {
        **plan().selected_stops[0],
        "stop_id": "poi_extra",
        "name": "Off-sequence extra",
        "utility": 1000.0,
    }
    child = replace(reference, selected_stops=reference.selected_stops + (extra,))
    configured = PlanEvaluator(
        route_matrix=route_matrix(),
        planner_runs={"run_verify_001": planner_run()},
        config=PlanEvaluatorConfig(
            start_anchor_by_day={1: "hotel_start"},
            end_anchor_by_day={1: "hotel_end"},
            max_day_minutes=120,
        ),
        reference_plan=reference,
    )

    certificate = configured.evaluate_final_plan(child)

    assert certificate.comparison_eligibility == "ineligible"
    assert "selected_stop_not_in_sequence" in {failure.code for failure in certificate.failures}
    assert certificate.metrics["utility_retained"] == 1.0


def test_duplicate_selected_stop_records_are_ineligible():
    original = plan()
    child = replace(original, selected_stops=original.selected_stops + (original.selected_stops[0],))

    certificate = evaluator().evaluate_final_plan(child)

    assert certificate.comparison_eligibility == "ineligible"
    assert "duplicate_selected_stop_record" in {failure.code for failure in certificate.failures}


def test_unsuccessful_or_invalidated_planner_run_is_ineligible():
    cases = (
        (replace(planner_run(), execution_status="FAILED"), "planner_run_not_successful"),
        (
            replace(planner_run(), solver_certification="INVALIDATED_AFTER_EDIT"),
            "planner_certification_invalidated",
        ),
    )
    for linked_run, expected_code in cases:
        certificate = evaluator().evaluate_final_plan(plan(), planner_run=linked_run)

        assert certificate.comparison_eligibility == "ineligible"
        assert expected_code in {failure.code for failure in certificate.failures}


def test_mixed_utility_units_do_not_produce_a_retention_ratio():
    reference = replace(
        plan(),
        selected_stops=tuple({**stop, "utility": 10.0} for stop in plan().selected_stops),
    )
    child = replace(
        plan(),
        selected_stops=(
            {**plan().selected_stops[0], "utility": 10.0},
            {**plan().selected_stops[1], "score": 10.0},
        ),
    )
    configured = PlanEvaluator(
        route_matrix=route_matrix(),
        planner_runs={"run_verify_001": planner_run()},
        config=PlanEvaluatorConfig(
            start_anchor_by_day={1: "hotel_start"},
            end_anchor_by_day={1: "hotel_end"},
            max_day_minutes=120,
        ),
        reference_plan=reference,
    )

    certificate = configured.evaluate_final_plan(child)

    assert "utility_retained" not in certificate.metrics
