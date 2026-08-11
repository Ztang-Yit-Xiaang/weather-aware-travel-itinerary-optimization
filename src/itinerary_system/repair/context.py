"""Typed compilation of frozen repair context for planners and evaluators."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any


class PlannerContextMode(StrEnum):
    """Information boundary used by a benchmark planning method."""

    AWARE = "aware"
    BLIND = "blind"


@dataclass(frozen=True)
class ContextSelectionRequirements:
    """Hard stop-selection requirements compiled from confirmed context."""

    required_stop_ids: tuple[str, ...] = ()
    excluded_stop_ids: tuple[str, ...] = ()
    conflict_stop_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class ContextEvaluationRequirements:
    """Context constraints recomputed by the independent evaluator."""

    required_stop_ids: tuple[str, ...] = ()
    excluded_stop_ids: tuple[str, ...] = ()
    closed_route_ids: tuple[str, ...] = ()
    unavailable_lodging_ids: tuple[str, ...] = ()
    max_day_minutes: float | None = None


def confirmed_constraints(request: Any) -> dict[str, Any]:
    raw = getattr(request, "confirmed_constraints", {})
    return dict(raw) if isinstance(raw, Mapping) else {}


def contextualize_stop_records(
    records: Iterable[Mapping[str, Any]],
    request: Any,
) -> tuple[dict[str, Any], ...]:
    """Attach frozen context evidence to stop records without selecting a method."""

    constraints = confirmed_constraints(request)
    risk_overrides = _mapping(constraints.get("weather_risk_overrides"))
    weather_feasible = _mapping(constraints.get("weather_feasible"))
    closure_status = _mapping(constraints.get("closure_status_by_stop"))
    contextualized: list[dict[str, Any]] = []
    for index, raw in enumerate(records, start=1):
        record = dict(raw)
        stop_id = _stop_id(record, index)
        record.setdefault("stop_id", stop_id)
        if stop_id in risk_overrides:
            value = _float_or_none(risk_overrides[stop_id])
            if value is not None:
                record["weather_risk"] = value
        if stop_id in weather_feasible:
            record["weather_feasible"] = _truthy_or_false(weather_feasible[stop_id])
        if stop_id in closure_status:
            status = str(closure_status[stop_id]).strip().lower()
            record["closure_status"] = status
            if status in {"closed", "cancelled", "canceled"}:
                record["closed"] = True
        contextualized.append(record)
    return tuple(contextualized)


def contextualize_plan_for_evaluation(plan: Any, request: Any) -> Any:
    """Return a frozen parent-plan view with the same context evidence used for child evaluation."""

    return replace(
        plan,
        selected_stops=contextualize_stop_records(getattr(plan, "selected_stops", ()), request),
    )

def context_selection_requirements(
    request: Any,
    stop_records: Iterable[Mapping[str, Any]],
    *,
    mode: PlannerContextMode | str = PlannerContextMode.AWARE,
) -> ContextSelectionRequirements:
    """Compile hard stop decisions visible to the selected planning method."""

    if PlannerContextMode(str(mode)) == PlannerContextMode.BLIND:
        return ContextSelectionRequirements()
    constraints = confirmed_constraints(request)
    records = tuple(dict(record) for record in stop_records)
    required = set(_string_values(constraints.get("must_include")))
    excluded = set(_string_values(constraints.get("must_delete")))
    weather_feasible = _mapping(constraints.get("weather_feasible"))
    closure_status = _mapping(constraints.get("closure_status_by_stop"))
    for index, record in enumerate(records, start=1):
        stop_id = _stop_id(record, index)
        if weather_feasible.get(stop_id) is False or str(weather_feasible.get(stop_id, "")).lower() == "false":
            excluded.add(stop_id)
        status = str(closure_status.get(stop_id, record.get("closure_status", ""))).strip().lower()
        if bool(record.get("closed")) or status in {"closed", "cancelled", "canceled"}:
            excluded.add(stop_id)
    conflicts = required & excluded
    return ContextSelectionRequirements(
        required_stop_ids=tuple(sorted(required)),
        excluded_stop_ids=tuple(sorted(excluded)),
        conflict_stop_ids=tuple(sorted(conflicts)),
    )


def context_evaluation_requirements(request: Any) -> ContextEvaluationRequirements:
    """Return the context constraints that every output method must be checked against."""

    constraints = confirmed_constraints(request)
    stop_records = contextualize_stop_records(
        (*tuple(getattr(request, "baseline_route", ()) or ()), *tuple(getattr(request, "candidate_pois", ()) or ())),
        request,
    )
    selection = context_selection_requirements(request, stop_records, mode=PlannerContextMode.AWARE)
    max_day = _first_float(
        constraints.get("reduced_driving_tolerance_minutes"),
        getattr(request, "tolerance_profile", {}).get("max_daily_travel_minutes")
        if isinstance(getattr(request, "tolerance_profile", {}), Mapping)
        else None,
    )
    return ContextEvaluationRequirements(
        required_stop_ids=selection.required_stop_ids,
        excluded_stop_ids=selection.excluded_stop_ids,
        closed_route_ids=tuple(sorted(_string_values(constraints.get("closed_route_ids")))),
        unavailable_lodging_ids=tuple(sorted(_string_values(constraints.get("unavailable_lodging_ids")))),
        max_day_minutes=max_day,
    )


def context_burden(record: Mapping[str, Any]) -> float:
    """Deterministic contextual burden used by non-progressive baselines."""

    burden = _first_float(record.get("weather_risk"), default=0.0) or 0.0
    if record.get("weather_feasible") is False:
        burden += 1_000_000.0
    status = str(record.get("closure_status") or record.get("status") or "").strip().lower()
    if bool(record.get("closed")) or status in {"closed", "cancelled", "canceled"}:
        burden += 1_000_000.0
    return float(burden)


def stop_utility(record: Mapping[str, Any]) -> float:
    return float(
        _first_float(
            record.get("utility"),
            record.get("final_poi_value"),
            record.get("score"),
            record.get("rating"),
            default=0.0,
        )
        or 0.0
    )


def _mapping(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    return {str(key): item for key, item in value.items()}


def _string_values(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        values = (value,)
    elif isinstance(value, Mapping):
        values = tuple(value)
    else:
        try:
            values = tuple(value)
        except TypeError:
            values = (value,)
    return tuple(str(item).strip() for item in values if str(item).strip())


def _stop_id(record: Mapping[str, Any], index: int) -> str:
    return str(record.get("stop_id") or record.get("poi_id") or record.get("name") or f"stop_{index}").strip()


def _first_float(*values: Any, default: float | None = None) -> float | None:
    for value in values:
        parsed = _float_or_none(value)
        if parsed is not None:
            return parsed
    return default


def _float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _truthy_or_false(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "passed"}
