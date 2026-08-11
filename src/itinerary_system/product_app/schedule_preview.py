"""Truthful, non-certifying schedule previews for W4R direct edits.

The preview accounts only for values explicitly supplied by server-owned route,
place, and user-input contracts.  Missing values remain ``None`` and make the
precheck incomplete; they are never replaced with zero.  This module does not
mutate drafts, rank candidates, evaluate plans, issue certificates, or make an
option eligible for acceptance.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from math import isfinite
from typing import Any

from .interaction_contracts import DurationModeV1


class SchedulePrecheckError(ValueError):
    """Invalid preview input with a stable, presentation-safe code."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class WindowEndSemanticsV1(StrEnum):
    LATEST_START = "latest_start"
    LATEST_DEPARTURE = "latest_departure"


class SchedulePrecheckStatusV1(StrEnum):
    LIKELY_FEASIBLE = "likely_feasible"
    LIKELY_INFEASIBLE = "likely_infeasible"
    INCOMPLETE_EVIDENCE = "incomplete_evidence"


@dataclass(frozen=True)
class MinuteComponentV1:
    """One explicitly sourced schedule component; ``None`` means unavailable."""

    minutes: float | None
    source: str | None = None
    evidence_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "minutes", _optional_minutes(self.minutes, "schedule_component_minutes_invalid"))
        object.__setattr__(self, "evidence_refs", _evidence_refs(self.evidence_refs))
        if self.minutes is not None and not _clean_source(self.source):
            raise SchedulePrecheckError("schedule_component_source_required")
        object.__setattr__(self, "source", _clean_source(self.source))


@dataclass(frozen=True)
class DurationResolutionV1:
    minutes: float | None
    accounting_kind: str
    incomplete_code: str | None


@dataclass(frozen=True)
class DurationConstraintV1:
    """Visit-duration semantics used only for fast schedule accounting."""

    mode: DurationModeV1 | str
    preferred_minutes: float | None = None
    minimum_minutes: float | None = None
    maximum_minutes: float | None = None
    source: str | None = None
    evidence_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        try:
            mode = self.mode if isinstance(self.mode, DurationModeV1) else DurationModeV1(self.mode)
        except (TypeError, ValueError):
            raise SchedulePrecheckError("schedule_duration_mode_invalid") from None
        preferred = _optional_minutes(self.preferred_minutes, "schedule_duration_value_invalid")
        minimum = _optional_minutes(self.minimum_minutes, "schedule_duration_value_invalid")
        maximum = _optional_minutes(self.maximum_minutes, "schedule_duration_value_invalid")
        source = _clean_source(self.source)
        if any(value is not None for value in (preferred, minimum, maximum)) and not source:
            raise SchedulePrecheckError("schedule_duration_source_required")
        if minimum is not None and maximum is not None and minimum > maximum:
            raise SchedulePrecheckError("schedule_duration_bounds_invalid")
        if preferred is not None and minimum is not None and preferred < minimum:
            raise SchedulePrecheckError("schedule_duration_preferred_out_of_bounds")
        if preferred is not None and maximum is not None and preferred > maximum:
            raise SchedulePrecheckError("schedule_duration_preferred_out_of_bounds")
        if mode is DurationModeV1.EXACT:
            if preferred is None:
                raise SchedulePrecheckError("schedule_duration_exact_value_required")
            if minimum is not None and minimum != preferred:
                raise SchedulePrecheckError("schedule_duration_exact_bounds_mismatch")
            if maximum is not None and maximum != preferred:
                raise SchedulePrecheckError("schedule_duration_exact_bounds_mismatch")
        elif mode is DurationModeV1.PREFERRED and preferred is None:
            raise SchedulePrecheckError("schedule_duration_preferred_value_required")
        elif mode is DurationModeV1.MINIMUM and minimum is None:
            raise SchedulePrecheckError("schedule_duration_minimum_value_required")
        elif mode is DurationModeV1.MAXIMUM and maximum is None:
            raise SchedulePrecheckError("schedule_duration_maximum_value_required")
        elif mode is DurationModeV1.RANGE and (minimum is None or maximum is None):
            raise SchedulePrecheckError("schedule_duration_range_bounds_required")
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "preferred_minutes", preferred)
        object.__setattr__(self, "minimum_minutes", minimum)
        object.__setattr__(self, "maximum_minutes", maximum)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "evidence_refs", _evidence_refs(self.evidence_refs))

    def resolve_for_precheck(self) -> DurationResolutionV1:
        """Resolve an accounting value without pretending a bound is an exact duration."""

        if self.mode is DurationModeV1.EXACT:
            return DurationResolutionV1(self.preferred_minutes, "exact", None)
        if self.preferred_minutes is not None:
            return DurationResolutionV1(self.preferred_minutes, "preferred", None)
        if self.mode is DurationModeV1.MINIMUM:
            return DurationResolutionV1(
                self.minimum_minutes,
                "lower_bound",
                "schedule_duration_lower_bound_only",
            )
        return DurationResolutionV1(None, "unavailable", "schedule_visit_duration_unavailable")


@dataclass(frozen=True)
class ScheduleWindowV1:
    """Opening or user-owned time window in minutes from local midnight."""

    start_minute: float | None
    end_minute: float | None
    end_semantics: WindowEndSemanticsV1 | str = WindowEndSemanticsV1.LATEST_START
    source: str | None = None
    evidence_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        start = _optional_clock_minute(self.start_minute, "schedule_window_time_invalid")
        end = _optional_clock_minute(self.end_minute, "schedule_window_time_invalid")
        if start is None and end is None:
            raise SchedulePrecheckError("schedule_window_empty")
        if start is not None and end is not None and start > end:
            raise SchedulePrecheckError("schedule_window_order_invalid")
        try:
            semantics = (
                self.end_semantics
                if isinstance(self.end_semantics, WindowEndSemanticsV1)
                else WindowEndSemanticsV1(self.end_semantics)
            )
        except (TypeError, ValueError):
            raise SchedulePrecheckError("schedule_window_end_semantics_invalid") from None
        source = _clean_source(self.source)
        if not source:
            raise SchedulePrecheckError("schedule_window_source_required")
        object.__setattr__(self, "start_minute", start)
        object.__setattr__(self, "end_minute", end)
        object.__setattr__(self, "end_semantics", semantics)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "evidence_refs", _evidence_refs(self.evidence_refs))


@dataclass(frozen=True)
class ScheduleStopInputV1:
    stop_id: str
    travel: MinuteComponentV1
    parking_walk: MinuteComponentV1
    expected_wait: MinuteComponentV1
    duration: DurationConstraintV1
    buffer: MinuteComponentV1
    opening_window: ScheduleWindowV1 | None = None
    required_window: ScheduleWindowV1 | None = None
    opening_hours_applicable: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "stop_id", _identifier(self.stop_id, "schedule_stop_id_invalid"))
        if not isinstance(self.opening_hours_applicable, bool):
            raise SchedulePrecheckError("schedule_opening_applicability_invalid")


@dataclass(frozen=True)
class DayScheduleInputV1:
    day: int
    day_start_minute: float
    day_limit_minutes: float
    stops: tuple[ScheduleStopInputV1, ...]
    end_travel: MinuteComponentV1
    day_limit_evidence_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.day, bool) or not isinstance(self.day, int) or self.day < 1:
            raise SchedulePrecheckError("schedule_day_invalid")
        start = _required_clock_minute(self.day_start_minute, "schedule_day_start_invalid")
        limit = _required_positive_minutes(self.day_limit_minutes, "schedule_day_limit_invalid")
        stops = tuple(self.stops)
        if len({stop.stop_id for stop in stops}) != len(stops):
            raise SchedulePrecheckError("schedule_stop_id_duplicate")
        object.__setattr__(self, "day_start_minute", start)
        object.__setattr__(self, "day_limit_minutes", limit)
        object.__setattr__(self, "stops", stops)
        object.__setattr__(self, "day_limit_evidence_refs", _evidence_refs(self.day_limit_evidence_refs))


@dataclass(frozen=True)
class ScheduleFindingV1:
    code: str
    category: str
    target_id: str
    evidence_refs: tuple[str, ...]


@dataclass(frozen=True)
class ScheduleComponentBreakdownV1:
    travel_minutes: float | None
    parking_walk_minutes: float | None
    expected_wait_minutes: float | None
    opening_wait_minutes: float | None
    waiting_minutes: float | None
    visit_minutes: float | None
    buffer_minutes: float | None
    total_minutes: float | None


@dataclass(frozen=True)
class ScheduleStopPreviewV1:
    stop_id: str
    arrival_minute: float | None
    service_start_minute: float | None
    departure_minute: float | None
    duration_mode: str
    duration_accounting_kind: str
    components: ScheduleComponentBreakdownV1
    findings: tuple[ScheduleFindingV1, ...]
    evidence_refs: tuple[str, ...]


@dataclass(frozen=True)
class DaySchedulePreviewV1:
    day: int
    status: str
    accounting_complete: bool
    certification_status: str
    is_certified: bool
    day_start_minute: float
    projected_end_minute: float | None
    day_limit_minutes: float
    total_minutes: float | None
    slack_minutes: float | None
    overrun_minutes: float | None
    end_travel_minutes: float | None
    stops: tuple[ScheduleStopPreviewV1, ...]
    findings: tuple[ScheduleFindingV1, ...]
    blocking_codes: tuple[str, ...]
    incomplete_codes: tuple[str, ...]
    evidence_refs: tuple[str, ...]
    schema_version: str = "day-schedule-preview-v1"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def preview_day_schedule(schedule: DayScheduleInputV1) -> DaySchedulePreviewV1:
    """Compute a fast accounting preview without producing feasibility evidence."""

    current: float | None = schedule.day_start_minute
    stop_previews: list[ScheduleStopPreviewV1] = []
    findings: list[ScheduleFindingV1] = []

    for stop in schedule.stops:
        preview, current = _preview_stop(stop, current)
        stop_previews.append(preview)
        findings.extend(preview.findings)

    if schedule.end_travel.minutes is None:
        findings.append(
            _finding(
                "schedule_end_travel_unavailable",
                "incomplete",
                f"day:{schedule.day}",
                schedule.end_travel.evidence_refs,
            )
        )
        projected_end = None
    elif current is None:
        projected_end = None
    else:
        projected_end = current + schedule.end_travel.minutes

    total = None if projected_end is None else projected_end - schedule.day_start_minute
    slack = None if total is None else max(0.0, schedule.day_limit_minutes - total)
    overrun = None if total is None else max(0.0, total - schedule.day_limit_minutes)
    if overrun is not None and overrun > 0:
        findings.append(
            _finding(
                "schedule_day_limit_exceeded",
                "blocking",
                f"day:{schedule.day}",
                schedule.day_limit_evidence_refs,
            )
        )

    findings_tuple = _dedupe_findings(findings)
    blocking_codes = _codes(findings_tuple, "blocking")
    incomplete_codes = _codes(findings_tuple, "incomplete")
    if blocking_codes:
        status = SchedulePrecheckStatusV1.LIKELY_INFEASIBLE.value
    elif incomplete_codes:
        status = SchedulePrecheckStatusV1.INCOMPLETE_EVIDENCE.value
    else:
        status = SchedulePrecheckStatusV1.LIKELY_FEASIBLE.value
    evidence_refs = _unique(
        (
            *schedule.day_limit_evidence_refs,
            *schedule.end_travel.evidence_refs,
            *(ref for stop in stop_previews for ref in stop.evidence_refs),
        )
    )
    return DaySchedulePreviewV1(
        day=schedule.day,
        status=status,
        accounting_complete=not incomplete_codes,
        certification_status="not_certified",
        is_certified=False,
        day_start_minute=schedule.day_start_minute,
        projected_end_minute=projected_end,
        day_limit_minutes=schedule.day_limit_minutes,
        total_minutes=total,
        slack_minutes=slack,
        overrun_minutes=overrun,
        end_travel_minutes=schedule.end_travel.minutes,
        stops=tuple(stop_previews),
        findings=findings_tuple,
        blocking_codes=blocking_codes,
        incomplete_codes=incomplete_codes,
        evidence_refs=evidence_refs,
    )


def _preview_stop(stop: ScheduleStopInputV1, current: float | None) -> tuple[ScheduleStopPreviewV1, float | None]:
    local_findings: list[ScheduleFindingV1] = []
    component_inputs = (
        (stop.travel, "schedule_travel_unavailable"),
        (stop.parking_walk, "schedule_parking_walk_unavailable"),
        (stop.expected_wait, "schedule_wait_unavailable"),
        (stop.buffer, "schedule_buffer_unavailable"),
    )
    for component, code in component_inputs:
        if component.minutes is None:
            local_findings.append(_finding(code, "incomplete", stop.stop_id, component.evidence_refs))

    resolution = stop.duration.resolve_for_precheck()
    if resolution.incomplete_code:
        local_findings.append(
            _finding(
                resolution.incomplete_code,
                "incomplete",
                stop.stop_id,
                stop.duration.evidence_refs,
            )
        )
    if stop.opening_hours_applicable and stop.opening_window is None:
        local_findings.append(_finding("schedule_opening_window_unavailable", "incomplete", stop.stop_id, ()))

    arrival = None
    service_start = None
    departure = None
    opening_wait: float | None
    if stop.opening_hours_applicable and stop.opening_window is None:
        opening_wait = None
    else:
        opening_wait = 0.0

    if current is not None and stop.travel.minutes is not None:
        arrival = current + stop.travel.minutes
    access_ready = (
        None
        if arrival is None or stop.parking_walk.minutes is None
        else arrival + stop.parking_walk.minutes
    )
    wait_ready = (
        None
        if access_ready is None or stop.expected_wait.minutes is None
        else access_ready + stop.expected_wait.minutes
    )
    if wait_ready is not None and opening_wait is not None:
        starts = tuple(
            value
            for value in (
                stop.opening_window.start_minute if stop.opening_window else None,
                stop.required_window.start_minute if stop.required_window else None,
            )
            if value is not None
        )
        service_start = max((wait_ready, *starts))
        opening_wait = service_start - wait_ready
    waiting = (
        None
        if stop.expected_wait.minutes is None or opening_wait is None
        else stop.expected_wait.minutes + opening_wait
    )
    if service_start is not None and resolution.minutes is not None and stop.buffer.minutes is not None:
        departure = service_start + resolution.minutes + stop.buffer.minutes

    local_findings.extend(_window_findings(stop.stop_id, "opening", stop.opening_window, service_start, departure))
    local_findings.extend(_window_findings(stop.stop_id, "required", stop.required_window, service_start, departure))

    component_values = (
        stop.travel.minutes,
        stop.parking_walk.minutes,
        waiting,
        resolution.minutes,
        stop.buffer.minutes,
    )
    component_total = sum(component_values) if all(value is not None for value in component_values) else None
    next_current = departure
    evidence_refs = _unique(
        (
            *stop.travel.evidence_refs,
            *stop.parking_walk.evidence_refs,
            *stop.expected_wait.evidence_refs,
            *stop.duration.evidence_refs,
            *stop.buffer.evidence_refs,
            *(stop.opening_window.evidence_refs if stop.opening_window else ()),
            *(stop.required_window.evidence_refs if stop.required_window else ()),
        )
    )
    preview = ScheduleStopPreviewV1(
        stop_id=stop.stop_id,
        arrival_minute=arrival,
        service_start_minute=service_start,
        departure_minute=departure,
        duration_mode=stop.duration.mode.value,
        duration_accounting_kind=resolution.accounting_kind,
        components=ScheduleComponentBreakdownV1(
            travel_minutes=stop.travel.minutes,
            parking_walk_minutes=stop.parking_walk.minutes,
            expected_wait_minutes=stop.expected_wait.minutes,
            opening_wait_minutes=opening_wait,
            waiting_minutes=waiting,
            visit_minutes=resolution.minutes,
            buffer_minutes=stop.buffer.minutes,
            total_minutes=component_total,
        ),
        findings=_dedupe_findings(local_findings),
        evidence_refs=evidence_refs,
    )
    return preview, next_current


def _window_findings(
    target_id: str,
    prefix: str,
    window: ScheduleWindowV1 | None,
    service_start: float | None,
    departure: float | None,
) -> tuple[ScheduleFindingV1, ...]:
    if window is None or window.end_minute is None:
        return ()
    if window.end_semantics is WindowEndSemanticsV1.LATEST_START:
        value = service_start
        suffix = "latest_start_exceeded"
    else:
        value = departure
        suffix = "latest_departure_exceeded"
    if value is not None and value > window.end_minute:
        return (_finding(f"schedule_{prefix}_{suffix}", "blocking", target_id, window.evidence_refs),)
    return ()


def _finding(
    code: str,
    category: str,
    target_id: str,
    evidence_refs: tuple[str, ...],
) -> ScheduleFindingV1:
    return ScheduleFindingV1(code, category, target_id, _evidence_refs(evidence_refs))


def _dedupe_findings(findings: list[ScheduleFindingV1]) -> tuple[ScheduleFindingV1, ...]:
    return tuple(dict.fromkeys(findings))


def _codes(findings: tuple[ScheduleFindingV1, ...], category: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(finding.code for finding in findings if finding.category == category))


def _optional_minutes(value: Any, code: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise SchedulePrecheckError(code)
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise SchedulePrecheckError(code) from None
    if not isfinite(result) or result < 0:
        raise SchedulePrecheckError(code)
    return result


def _required_positive_minutes(value: Any, code: str) -> float:
    result = _optional_minutes(value, code)
    if result is None or result <= 0:
        raise SchedulePrecheckError(code)
    return result


def _optional_clock_minute(value: Any, code: str) -> float | None:
    result = _optional_minutes(value, code)
    if result is not None and result > 1440:
        raise SchedulePrecheckError(code)
    return result


def _required_clock_minute(value: Any, code: str) -> float:
    result = _optional_clock_minute(value, code)
    if result is None:
        raise SchedulePrecheckError(code)
    return result


def _clean_source(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text and len(text) <= 128 else None


def _identifier(value: Any, code: str) -> str:
    text = str(value).strip() if value is not None else ""
    if not text or len(text) > 128:
        raise SchedulePrecheckError(code)
    return text


def _evidence_refs(values: tuple[str, ...]) -> tuple[str, ...]:
    refs = tuple(values)
    if len(refs) > 16:
        raise SchedulePrecheckError("schedule_evidence_refs_too_many")
    for ref in refs:
        if not isinstance(ref, str) or not ref.strip() or len(ref) > 256:
            raise SchedulePrecheckError("schedule_evidence_ref_invalid")
    return _unique(tuple(ref.strip() for ref in refs))


def _unique(values: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))
