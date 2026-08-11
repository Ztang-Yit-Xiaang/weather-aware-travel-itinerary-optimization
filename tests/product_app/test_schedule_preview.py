from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from itinerary_system.product_app.schedule_preview import (
    DayScheduleInputV1,
    DurationConstraintV1,
    MinuteComponentV1,
    SchedulePrecheckError,
    ScheduleStopInputV1,
    ScheduleWindowV1,
    WindowEndSemanticsV1,
    preview_day_schedule,
)


def _minutes(value: float | None, source: str = "fixture", *refs: str) -> MinuteComponentV1:
    return MinuteComponentV1(value, source if value is not None else None, tuple(refs))


def _duration(
    mode: str = "exact",
    *,
    preferred: float | None = 60,
    minimum: float | None = None,
    maximum: float | None = None,
) -> DurationConstraintV1:
    return DurationConstraintV1(
        mode=mode,
        preferred_minutes=preferred,
        minimum_minutes=minimum,
        maximum_minutes=maximum,
        source="user",
        evidence_refs=("duration:user",),
    )


def _stop(
    *,
    stop_id: str = "museum",
    travel: float | None = 30,
    parking_walk: float | None = 10,
    wait: float | None = 0,
    duration: DurationConstraintV1 | None = None,
    buffer: float | None = 5,
    opening: ScheduleWindowV1 | None = None,
    opening_hours_applicable: bool = False,
    required: ScheduleWindowV1 | None = None,
) -> ScheduleStopInputV1:
    return ScheduleStopInputV1(
        stop_id=stop_id,
        travel=_minutes(travel, "route", "route:incoming"),
        parking_walk=_minutes(parking_walk, "place", "access:walk"),
        expected_wait=_minutes(wait, "user", "wait:user"),
        duration=duration or _duration(),
        buffer=_minutes(buffer, "policy", "buffer:policy"),
        opening_window=opening,
        required_window=required,
        opening_hours_applicable=opening_hours_applicable,
    )


def _day(
    *stops: ScheduleStopInputV1,
    limit: float = 480,
    end_travel: float | None = 20,
) -> DayScheduleInputV1:
    return DayScheduleInputV1(
        day=1,
        day_start_minute=9 * 60,
        day_limit_minutes=limit,
        stops=tuple(stops),
        end_travel=_minutes(end_travel, "route", "route:end"),
        day_limit_evidence_refs=("policy:day-limit",),
    )


def test_complete_preview_accounts_for_every_component_and_exact_slack() -> None:
    opening = ScheduleWindowV1(
        start_minute=600,
        end_minute=720,
        source="place_snapshot",
        evidence_refs=("hours:museum",),
    )
    result = preview_day_schedule(
        _day(_stop(opening=opening, opening_hours_applicable=True), limit=240, end_travel=20)
    )

    stop = result.stops[0]
    assert stop.arrival_minute == 570
    assert stop.service_start_minute == 600
    assert stop.departure_minute == 665
    assert stop.components.travel_minutes == 30
    assert stop.components.parking_walk_minutes == 10
    assert stop.components.expected_wait_minutes == 0
    assert stop.components.opening_wait_minutes == 20
    assert stop.components.waiting_minutes == 20
    assert stop.components.visit_minutes == 60
    assert stop.components.buffer_minutes == 5
    assert stop.components.total_minutes == 125
    assert result.total_minutes == 145
    assert result.slack_minutes == 95
    assert result.overrun_minutes == 0
    assert result.projected_end_minute == 685
    assert result.status == "likely_feasible"
    assert result.accounting_complete is True
    assert result.certification_status == "not_certified"
    assert result.is_certified is False
    assert "hours:museum" in result.evidence_refs
    assert "route:end" in result.evidence_refs


def test_day_overrun_is_exact_and_has_stable_blocking_code() -> None:
    result = preview_day_schedule(_day(_stop(), limit=100, end_travel=20))

    assert result.total_minutes == 125
    assert result.slack_minutes == 0
    assert result.overrun_minutes == 25
    assert result.status == "likely_infeasible"
    assert result.blocking_codes == ("schedule_day_limit_exceeded",)
    finding = next(item for item in result.findings if item.code == "schedule_day_limit_exceeded")
    assert finding.evidence_refs == ("policy:day-limit",)


def test_missing_component_remains_none_and_is_never_zero_coerced() -> None:
    result = preview_day_schedule(_day(_stop(parking_walk=None)))
    stop = result.stops[0]

    assert stop.components.parking_walk_minutes is None
    assert stop.components.total_minutes is None
    assert stop.service_start_minute is None
    assert stop.departure_minute is None
    assert result.projected_end_minute is None
    assert result.total_minutes is None
    assert result.slack_minutes is None
    assert result.overrun_minutes is None
    assert result.status == "incomplete_evidence"
    assert result.incomplete_codes == ("schedule_parking_walk_unavailable",)


def test_missing_opening_hours_prevents_timeline_completion_when_applicable() -> None:
    result = preview_day_schedule(_day(_stop(opening_hours_applicable=True, opening=None)))

    assert result.stops[0].components.opening_wait_minutes is None
    assert result.stops[0].components.waiting_minutes is None
    assert result.total_minutes is None
    assert "schedule_opening_window_unavailable" in result.incomplete_codes


@pytest.mark.parametrize(
    ("duration", "minutes", "kind", "incomplete_code"),
    [
        (_duration("exact", preferred=60), 60, "exact", None),
        (_duration("preferred", preferred=55, minimum=40, maximum=70), 55, "preferred", None),
        (_duration("minimum", preferred=None, minimum=30), 30, "lower_bound", "schedule_duration_lower_bound_only"),
        (_duration("maximum", preferred=40, maximum=60), 40, "preferred", None),
        (_duration("maximum", preferred=None, maximum=60), None, "unavailable", "schedule_visit_duration_unavailable"),
        (_duration("range", preferred=45, minimum=30, maximum=60), 45, "preferred", None),
        (_duration("range", preferred=None, minimum=30, maximum=60), None, "unavailable", "schedule_visit_duration_unavailable"),
    ],
)
def test_duration_modes_have_truthful_accounting_semantics(
    duration: DurationConstraintV1,
    minutes: float | None,
    kind: str,
    incomplete_code: str | None,
) -> None:
    result = preview_day_schedule(_day(_stop(duration=duration)))
    stop = result.stops[0]

    assert stop.components.visit_minutes == minutes
    assert stop.duration_accounting_kind == kind
    if incomplete_code:
        assert incomplete_code in result.incomplete_codes
    else:
        assert incomplete_code is None


@pytest.mark.parametrize(
    ("kwargs", "code"),
    [
        ({"mode": "exact", "preferred_minutes": None}, "schedule_duration_exact_value_required"),
        ({"mode": "preferred", "preferred_minutes": None}, "schedule_duration_preferred_value_required"),
        ({"mode": "minimum", "minimum_minutes": None}, "schedule_duration_minimum_value_required"),
        ({"mode": "maximum", "maximum_minutes": None}, "schedule_duration_maximum_value_required"),
        ({"mode": "range", "minimum_minutes": 10}, "schedule_duration_range_bounds_required"),
        (
            {"mode": "range", "minimum_minutes": 60, "maximum_minutes": 30},
            "schedule_duration_bounds_invalid",
        ),
    ],
)
def test_invalid_duration_contracts_fail_with_stable_codes(kwargs: dict[str, object], code: str) -> None:
    with pytest.raises(SchedulePrecheckError, match=code) as error:
        DurationConstraintV1(source="user", **kwargs)

    assert error.value.code == code


def test_opening_window_can_require_departure_before_closing() -> None:
    opening = ScheduleWindowV1(
        start_minute=540,
        end_minute=620,
        end_semantics=WindowEndSemanticsV1.LATEST_DEPARTURE,
        source="place_snapshot",
        evidence_refs=("hours:museum",),
    )
    result = preview_day_schedule(_day(_stop(opening=opening, opening_hours_applicable=True)))

    assert result.stops[0].service_start_minute == 580
    assert result.stops[0].departure_minute == 645
    assert result.blocking_codes == ("schedule_opening_latest_departure_exceeded",)
    assert result.status == "likely_infeasible"


def test_required_window_wait_and_violation_are_distinct_from_opening_hours() -> None:
    required = ScheduleWindowV1(
        start_minute=600,
        end_minute=630,
        end_semantics="latest_start",
        source="booking",
        evidence_refs=("booking:reservation",),
    )
    result = preview_day_schedule(_day(_stop(required=required)))

    assert result.stops[0].service_start_minute == 600
    assert result.stops[0].components.opening_wait_minutes == 20
    assert result.blocking_codes == ()
    assert "booking:reservation" in result.evidence_refs


def test_missing_end_travel_is_explicit_and_not_zero() -> None:
    result = preview_day_schedule(_day(_stop(), end_travel=None))

    assert result.end_travel_minutes is None
    assert result.total_minutes is None
    assert result.incomplete_codes == ("schedule_end_travel_unavailable",)


def test_contracts_are_immutable_and_expose_no_eligibility_or_acceptance_claim() -> None:
    result = preview_day_schedule(_day(_stop()))

    with pytest.raises(FrozenInstanceError):
        result.total_minutes = 0  # type: ignore[misc]
    payload = result.as_dict()
    assert "decision_eligible" not in payload
    assert "ranking_eligible" not in payload
    assert "acceptance_eligible" not in payload
    assert payload["certification_status"] == "not_certified"


def test_component_values_require_a_source_and_evidence_refs_are_bounded() -> None:
    with pytest.raises(SchedulePrecheckError, match="schedule_component_source_required"):
        MinuteComponentV1(5)
    with pytest.raises(SchedulePrecheckError, match="schedule_evidence_refs_too_many"):
        MinuteComponentV1(None, evidence_refs=tuple(f"ref:{index}" for index in range(17)))
