"""Typed parent/child plan diff utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..research_artifacts import PlanArtifactV2, stable_content_hash
from .models import (
    DayMove,
    LodgingChange,
    OrderChange,
    OwnedConstraint,
    OwnershipPolicy,
    PlanDiff,
    RoadChange,
    StopChange,
    TimeShift,
    active_owned_constraints,
)


@dataclass(frozen=True)
class _PlanStop:
    stop_id: str
    day: int | None
    order: int | None
    start_time: str | None
    end_time: str | None
    lodging_id: str | None
    route_id: str | None


class PlanDiffBuilder:
    """Compute typed plan changes with deterministic ownership-weighted cost."""

    def __init__(self, ownership_policy: OwnershipPolicy | None = None) -> None:
        self.ownership_policy = ownership_policy or OwnershipPolicy()

    def diff(self, parent: PlanArtifactV2, child: PlanArtifactV2) -> PlanDiff:
        return compute_plan_diff(parent, child, self.ownership_policy)

    def weighted_cost(self, diff: PlanDiff) -> float:
        return float(diff.weighted_edit_cost)

    def unchanged_days(self, diff: PlanDiff) -> tuple[int, ...]:
        return diff.unchanged_days


def compute_plan_diff(
    parent: PlanArtifactV2,
    child: PlanArtifactV2,
    policy: OwnershipPolicy | None = None,
) -> PlanDiff:
    """Compute added/deleted stops, moves, ordering, lodging, and road changes."""

    policy = policy or OwnershipPolicy()
    _validate_compatible(parent, child)
    parent_stops = _plan_stop_index(parent)
    child_stops = _plan_stop_index(child)
    parent_strengths = _constraint_strength_by_target(parent)
    child_strengths = _constraint_strength_by_target(child)

    added_stops: list[StopChange] = []
    deleted_stops: list[StopChange] = []
    day_moves: list[DayMove] = []
    time_shifts: list[TimeShift] = []
    reorder_changes: list[OrderChange] = []
    lodging_changes: list[LodgingChange] = []
    road_changes: list[RoadChange] = []
    affected_days: set[int] = set()

    for stop_id in sorted(set(child_stops) - set(parent_stops)):
        stop = child_stops[stop_id]
        owner_strength = _owner_strength(stop_id, child_strengths, policy)
        cost = _change_cost("added_stop", owner_strength, policy)
        added_stops.append(
            StopChange(stop_id=stop.stop_id, day=stop.day, order=stop.order, owner_strength=owner_strength, cost=cost)
        )
        _add_day(affected_days, stop.day)

    for stop_id in sorted(set(parent_stops) - set(child_stops)):
        stop = parent_stops[stop_id]
        owner_strength = _owner_strength(stop_id, parent_strengths, policy)
        cost = _change_cost("deleted_stop", owner_strength, policy)
        deleted_stops.append(
            StopChange(stop_id=stop.stop_id, day=stop.day, order=stop.order, owner_strength=owner_strength, cost=cost)
        )
        _add_day(affected_days, stop.day)

    for stop_id in sorted(set(parent_stops) & set(child_stops)):
        parent_stop = parent_stops[stop_id]
        child_stop = child_stops[stop_id]
        owner_strength = _owner_strength(stop_id, {**parent_strengths, **child_strengths}, policy)

        if parent_stop.day != child_stop.day:
            cost = _change_cost("day_move", owner_strength, policy)
            day_moves.append(
                DayMove(
                    stop_id=stop_id,
                    from_day=parent_stop.day,
                    to_day=child_stop.day,
                    owner_strength=owner_strength,
                    cost=cost,
                )
            )
            _add_day(affected_days, parent_stop.day)
            _add_day(affected_days, child_stop.day)

        if parent_stop.start_time != child_stop.start_time or parent_stop.end_time != child_stop.end_time:
            delta = _time_delta_minutes(parent_stop.start_time, child_stop.start_time)
            cost = _change_cost("time_shift", owner_strength, policy) * max(1.0, abs(delta) / 60.0)
            time_shifts.append(
                TimeShift(
                    stop_id=stop_id,
                    from_start=parent_stop.start_time,
                    to_start=child_stop.start_time,
                    from_end=parent_stop.end_time,
                    to_end=child_stop.end_time,
                    delta_minutes=delta,
                    owner_strength=owner_strength,
                    cost=cost,
                )
            )
            _add_day(affected_days, parent_stop.day)
            _add_day(affected_days, child_stop.day)

        if parent_stop.order != child_stop.order:
            cost = _change_cost("order_change", owner_strength, policy)
            reorder_changes.append(
                OrderChange(
                    stop_id=stop_id,
                    day=child_stop.day if child_stop.day is not None else parent_stop.day,
                    from_order=parent_stop.order,
                    to_order=child_stop.order,
                    owner_strength=owner_strength,
                    cost=cost,
                )
            )
            _add_day(affected_days, parent_stop.day)
            _add_day(affected_days, child_stop.day)

    parent_lodging = _lodging_by_day(parent, parent_stops)
    child_lodging = _lodging_by_day(child, child_stops)
    for day in sorted(set(parent_lodging) | set(child_lodging)):
        if parent_lodging.get(day) == child_lodging.get(day):
            continue
        owner_strength = _owner_strength(
            f"lodging:{day}",
            {**parent_strengths, **child_strengths},
            policy,
            fallback_targets=(str(day), parent_lodging.get(day), child_lodging.get(day)),
        )
        cost = _change_cost("lodging_change", owner_strength, policy)
        lodging_changes.append(
            LodgingChange(
                day=day,
                from_lodging_id=parent_lodging.get(day),
                to_lodging_id=child_lodging.get(day),
                owner_strength=owner_strength,
                cost=cost,
            )
        )
        affected_days.add(day)

    parent_routes = _route_by_day(parent, parent_stops)
    child_routes = _route_by_day(child, child_stops)
    for day in sorted(set(parent_routes) | set(child_routes)):
        if parent_routes.get(day) == child_routes.get(day):
            continue
        owner_strength = _owner_strength(
            f"route:{day}",
            {**parent_strengths, **child_strengths},
            policy,
            fallback_targets=(str(day), parent_routes.get(day), child_routes.get(day)),
        )
        cost = _change_cost("road_change", owner_strength, policy)
        road_changes.append(
            RoadChange(
                day=day,
                from_route_id=parent_routes.get(day),
                to_route_id=child_routes.get(day),
                owner_strength=owner_strength,
                cost=cost,
            )
        )
        affected_days.add(day)

    all_days = _all_days(parent_stops, child_stops, parent_lodging, child_lodging, parent_routes, child_routes)
    unchanged_days = tuple(sorted(day for day in all_days if day not in affected_days))
    weighted_edit_cost = float(
        sum(change.cost for change in added_stops)
        + sum(change.cost for change in deleted_stops)
        + sum(change.cost for change in day_moves)
        + sum(change.cost for change in time_shifts)
        + sum(change.cost for change in reorder_changes)
        + sum(change.cost for change in lodging_changes)
        + sum(change.cost for change in road_changes)
    )
    diff_payload = {
        "parent_plan_id": parent.plan_id,
        "child_plan_id": child.plan_id,
        "added": [change.to_record() for change in added_stops],
        "deleted": [change.to_record() for change in deleted_stops],
        "day_moves": [change.to_record() for change in day_moves],
        "time_shifts": [change.to_record() for change in time_shifts],
        "reorder": [change.to_record() for change in reorder_changes],
        "lodging": [change.to_record() for change in lodging_changes],
        "road": [change.to_record() for change in road_changes],
        "unchanged_days": unchanged_days,
        "weighted_edit_cost": weighted_edit_cost,
    }
    return PlanDiff(
        diff_id=f"diff_{stable_content_hash(diff_payload)}",
        parent_plan_id=parent.plan_id,
        child_plan_id=child.plan_id,
        added_stops=tuple(added_stops),
        deleted_stops=tuple(deleted_stops),
        day_moves=tuple(day_moves),
        time_shifts=tuple(time_shifts),
        reorder_changes=tuple(reorder_changes),
        lodging_changes=tuple(lodging_changes),
        road_changes=tuple(road_changes),
        unchanged_days=unchanged_days,
        weighted_edit_cost=weighted_edit_cost,
    )


def _validate_compatible(parent: PlanArtifactV2, child: PlanArtifactV2) -> None:
    if parent.catalog_snapshot_id != child.catalog_snapshot_id:
        raise ValueError("parent and child plans must share catalog_snapshot_id")
    if parent.context_snapshot_id != child.context_snapshot_id:
        raise ValueError("parent and child plans must share context_snapshot_id")


def _plan_stop_index(plan: PlanArtifactV2) -> dict[str, _PlanStop]:
    sequence_order = {str(stop_id): index for index, stop_id in enumerate(plan.sequence, start=1)}
    day_assignments = {str(stop_id): day for stop_id, day in plan.day_assignments.items()}
    stops: dict[str, _PlanStop] = {}
    for index, raw_stop in enumerate(plan.selected_stops, start=1):
        stop = dict(raw_stop)
        stop_id = _stop_id(stop, index)
        day = _coerce_int(stop.get("day"))
        if day is None and stop_id in day_assignments:
            day = _coerce_int(day_assignments[stop_id])
        order = _coerce_int(
            stop.get("stop_order")
            or stop.get("order")
            or stop.get("route_sequence_index")
            or stop.get("visit_order")
            or sequence_order.get(stop_id)
        )
        stops[stop_id] = _PlanStop(
            stop_id=stop_id,
            day=day,
            order=order,
            start_time=_clean_text(_first_nonempty(stop, "start_time", "arrival_time", "window_start", "scheduled_start")),
            end_time=_clean_text(_first_nonempty(stop, "end_time", "departure_time", "window_end", "scheduled_end")),
            lodging_id=_clean_text(
                _first_nonempty(stop, "lodging_id", "hotel_id", "overnight_lodging_id", "overnight_city")
            ),
            route_id=_clean_text(_first_nonempty(stop, "route_id", "road_route_id")),
        )
    return stops


def _lodging_by_day(plan: PlanArtifactV2, stops: dict[str, _PlanStop]) -> dict[int, str]:
    lodging: dict[int, str] = {}
    for day, lodging_id in plan.lodging_assignments.items():
        coerced_day = _coerce_int(day)
        if coerced_day is not None and str(lodging_id).strip():
            lodging[coerced_day] = str(lodging_id)
    for stop in stops.values():
        if stop.day is not None and stop.lodging_id:
            lodging.setdefault(stop.day, stop.lodging_id)
    return lodging


def _route_by_day(plan: PlanArtifactV2, stops: dict[str, _PlanStop]) -> dict[int, str]:
    routes = {int(day): str(route_id) for day, route_id in plan.route_ids_by_day.items() if str(route_id).strip()}
    for stop in stops.values():
        if stop.day is not None and stop.route_id:
            routes.setdefault(stop.day, stop.route_id)
    return routes


def _constraint_strength_by_target(plan: PlanArtifactV2) -> dict[str, str]:
    strengths: dict[str, str] = {}
    try:
        constraints = active_owned_constraints(tuple(dict(record) for record in plan.owned_constraints))
    except Exception:
        constraints = ()
    for constraint in constraints:
        target = constraint.target_id
        current = strengths.get(target)
        if current is None or _strength_rank(constraint) > _strength_rank(current):
            strengths[target] = constraint.strength.value
    return strengths


def _owner_strength(
    target_id: str,
    strengths_by_target: dict[str, str],
    policy: OwnershipPolicy,
    *,
    fallback_targets: tuple[str | None, ...] = (),
) -> str:
    candidates = (target_id, *tuple(target for target in fallback_targets if target))
    best = ""
    best_weight = 0.0
    for target in candidates:
        strength = strengths_by_target.get(str(target), "")
        weight = policy.weight_for_strength(strength)
        if strength and weight >= best_weight:
            best = strength
            best_weight = weight
    return best


def _change_cost(change_type: str, owner_strength: str, policy: OwnershipPolicy) -> float:
    return float(policy.weight_for_change(change_type) * policy.weight_for_strength(owner_strength))


def _strength_rank(constraint_or_strength: OwnedConstraint | str) -> float:
    policy = OwnershipPolicy()
    if isinstance(constraint_or_strength, OwnedConstraint):
        return policy.weight_for_strength(constraint_or_strength.strength)
    return policy.weight_for_strength(constraint_or_strength)


def _all_days(
    parent_stops: dict[str, _PlanStop],
    child_stops: dict[str, _PlanStop],
    parent_lodging: dict[int, str],
    child_lodging: dict[int, str],
    parent_routes: dict[int, str],
    child_routes: dict[int, str],
) -> set[int]:
    days = {day for day in parent_lodging | child_lodging | parent_routes | child_routes}
    days.update(stop.day for stop in parent_stops.values() if stop.day is not None)
    days.update(stop.day for stop in child_stops.values() if stop.day is not None)
    return days


def _stop_id(stop: dict[str, Any], index: int) -> str:
    value = _first_nonempty(stop, "stop_id", "poi_id", "attraction_id", "attraction_name", "name", "poi", "stop_name")
    return str(value).strip() if value else f"stop_{index}"


def _coerce_int(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except Exception:
        return None


def _first_nonempty(record: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = record.get(key)
        if value is not None and str(value).strip() != "":
            return value
    return None


def _clean_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _add_day(days: set[int], day: int | None) -> None:
    if day is not None:
        days.add(day)


def _time_delta_minutes(parent_time: str | None, child_time: str | None) -> float:
    parent_minutes = _parse_minutes(parent_time)
    child_minutes = _parse_minutes(child_time)
    if parent_minutes is None or child_minutes is None:
        return 0.0
    return float(child_minutes - parent_minutes)


def _parse_minutes(value: str | None) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        if ":" in text:
            hours, minutes = text.split(":", 1)
            return int(hours) * 60 + int(minutes[:2])
        return int(float(text))
    except Exception:
        return None
