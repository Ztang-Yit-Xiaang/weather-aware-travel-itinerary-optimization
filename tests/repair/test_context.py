from itinerary_system.repair.context import (
    PlannerContextMode,
    context_evaluation_requirements,
    context_selection_requirements,
    contextualize_plan_for_evaluation,
    contextualize_stop_records,
)
from itinerary_system.repair_planner import RepairRequest


def request() -> RepairRequest:
    return RepairRequest(
        request_id="context_test",
        baseline_route=(
            {"stop_id": "outdoor", "name": "Outdoor", "weather_risk": 0.2},
            {"stop_id": "closed", "name": "Closed"},
        ),
        user_intent="repair",
        tolerance_profile={"max_daily_travel_minutes": 240},
        confirmed_constraints={
            "weather_risk_overrides": {"outdoor": 0.95},
            "weather_feasible": {"outdoor": False},
            "closure_status_by_stop": {"closed": "closed"},
            "must_include": ("indoor",),
            "closed_route_ids": ("route_old",),
            "unavailable_lodging_ids": ("hotel_old",),
        },
        candidate_pois=({"stop_id": "indoor", "name": "Indoor", "weather_risk": 0.1},),
    )


def test_contextualization_attaches_frozen_weather_and_closure_evidence():
    records = contextualize_stop_records(request().baseline_route, request())
    by_id = {record["stop_id"]: record for record in records}

    assert by_id["outdoor"]["weather_risk"] == 0.95
    assert by_id["outdoor"]["weather_feasible"] is False
    assert by_id["closed"]["closed"] is True


def test_context_aware_and_blind_planners_have_different_information_boundaries():
    problem = request()
    records = contextualize_stop_records((*problem.baseline_route, *problem.candidate_pois), problem)

    aware = context_selection_requirements(problem, records, mode=PlannerContextMode.AWARE)
    blind = context_selection_requirements(problem, records, mode=PlannerContextMode.BLIND)

    assert set(aware.excluded_stop_ids) == {"closed", "outdoor"}
    assert aware.required_stop_ids == ("indoor",)
    assert blind.required_stop_ids == ()
    assert blind.excluded_stop_ids == ()


def test_independent_evaluation_requirements_always_include_context():
    requirements = context_evaluation_requirements(request())

    assert set(requirements.excluded_stop_ids) == {"closed", "outdoor"}
    assert requirements.required_stop_ids == ("indoor",)
    assert requirements.closed_route_ids == ("route_old",)
    assert requirements.unavailable_lodging_ids == ("hotel_old",)
    assert requirements.max_day_minutes == 240


def test_contextualized_reference_plan_preserves_identity_and_adds_frozen_risk():
    from itinerary_system.research_artifacts import PlanArtifactV2

    plan = PlanArtifactV2(
        plan_id="parent_context",
        source_run_id="run_parent_context",
        planning_request_id="request_parent_context",
        catalog_snapshot_id="catalog_context",
        context_snapshot_id="context_context",
        selected_stops=request().baseline_route,
        sequence=("outdoor", "closed"),
    )

    contextualized = contextualize_plan_for_evaluation(plan, request())

    assert contextualized.plan_id == plan.plan_id
    assert contextualized is not plan
    assert contextualized.selected_stops[0]["weather_risk"] == 0.95
    assert plan.selected_stops[0]["weather_risk"] == 0.2
