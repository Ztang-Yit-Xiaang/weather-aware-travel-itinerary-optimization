"""Typed parent/child plan diff utilities."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

from ..research_artifacts import PlanArtifactV2, stable_content_hash
from .attribute_ownership import (
    build_attribute_ownership_evidence,
    ownership_change_evidence,
    resolve_attribute_ownership,
)
from .models import (
    DayMove,
    DurationChange,
    LodgingChange,
    OrderChange,
    OwnedConstraint,
    OwnershipPolicy,
    PlanDiff,
    RoadChange,
    RoleChange,
    StopChange,
    TimeShift,
    TimeWindowChange,
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
    duration_constraint: dict[str, Any] | None
    duration_minutes: float | None
    duration_source: str
    time_window_constraint: dict[str, Any] | None
    itinerary_role: str | None
    itinerary_role_source: str


_STOP_ROLES = frozenset(
    {
        "attraction",
        "activity",
        "meal",
        "lodging",
        "transport_hub",
        "rest_stop",
        "scenic_stop",
        "route_waypoint",
        "origin",
        "destination",
    }
)


class PlanDiffBuilder:
    """Compute typed plan changes with deterministic ownership-weighted cost."""

    def __init__(
        self,
        ownership_policy: OwnershipPolicy | None = None,
        *,
        default_visit_minutes: float | None = None,
    ) -> None:
        if default_visit_minutes is not None and not _valid_duration_number(default_visit_minutes):
            raise ValueError("default_visit_minutes must be a finite number from 15 to 480")
        self.ownership_policy = ownership_policy or OwnershipPolicy()
        self.default_visit_minutes = (
            float(default_visit_minutes) if default_visit_minutes is not None else None
        )

    def diff(self, parent: PlanArtifactV2, child: PlanArtifactV2) -> PlanDiff:
        return compute_plan_diff(
            parent,
            child,
            self.ownership_policy,
            default_visit_minutes=self.default_visit_minutes,
        )

    def weighted_cost(self, diff: PlanDiff) -> float:
        return float(diff.weighted_edit_cost)

    def unchanged_days(self, diff: PlanDiff) -> tuple[int, ...]:
        return diff.unchanged_days


def compute_plan_diff(
    parent: PlanArtifactV2,
    child: PlanArtifactV2,
    policy: OwnershipPolicy | None = None,
    *,
    default_visit_minutes: float | None = None,
) -> PlanDiff:
    """Compute added/deleted stops, moves, ordering, lodging, and road changes."""

    policy = policy or OwnershipPolicy()
    if default_visit_minutes is not None and not _valid_duration_number(default_visit_minutes):
        raise ValueError("default_visit_minutes must be a finite number from 15 to 480")
    if default_visit_minutes is not None:
        default_visit_minutes = float(default_visit_minutes)
    _validate_compatible(parent, child)
    parent_stops = _plan_stop_index(parent)
    child_stops = _plan_stop_index(child)
    changed_attribute_keys = _changed_attribute_keys(parent_stops, child_stops)
    attribute_ownership = resolve_attribute_ownership(
        parent,
        child,
        changed_attribute_keys,
    )
    legacy_parent_strengths = _legacy_constraint_strength_by_target(parent)
    legacy_child_strengths = _legacy_constraint_strength_by_target(child)
    parent_role_strengths = _role_constraint_strength_by_target(parent)
    child_role_strengths = _role_constraint_strength_by_target(child)

    added_stops: list[StopChange] = []
    deleted_stops: list[StopChange] = []
    day_moves: list[DayMove] = []
    time_shifts: list[TimeShift] = []
    duration_changes: list[DurationChange] = []
    time_window_changes: list[TimeWindowChange] = []
    role_changes: list[RoleChange] = []
    reorder_changes: list[OrderChange] = []
    lodging_changes: list[LodgingChange] = []
    road_changes: list[RoadChange] = []
    affected_days: set[int] = set()

    for stop_id in sorted(set(child_stops) - set(parent_stops)):
        stop = child_stops[stop_id]
        owner_strength = attribute_ownership.strength_for(
            stop_id, ("existence",), policy
        )
        cost = _change_cost("added_stop", owner_strength, policy)
        added_stops.append(
            StopChange(stop_id=stop.stop_id, day=stop.day, order=stop.order, owner_strength=owner_strength, cost=cost)
        )
        _add_day(affected_days, stop.day)

    for stop_id in sorted(set(parent_stops) - set(child_stops)):
        stop = parent_stops[stop_id]
        owner_strength = attribute_ownership.strength_for(
            stop_id, ("existence",), policy
        )
        cost = _change_cost("deleted_stop", owner_strength, policy)
        deleted_stops.append(
            StopChange(stop_id=stop.stop_id, day=stop.day, order=stop.order, owner_strength=owner_strength, cost=cost)
        )
        _add_day(affected_days, stop.day)

    for stop_id in sorted(set(parent_stops) & set(child_stops)):
        parent_stop = parent_stops[stop_id]
        child_stop = child_stops[stop_id]

        if parent_stop.day != child_stop.day:
            owner_strength = attribute_ownership.strength_for(
                stop_id, ("day",), policy
            )
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
            changed_time_attributes = tuple(sorted(
                attribute
                for attribute, changed in (
                    ("start_time", parent_stop.start_time != child_stop.start_time),
                    ("end_time", parent_stop.end_time != child_stop.end_time),
                )
                if changed
            ))
            owner_strength = attribute_ownership.strength_for(
                stop_id, changed_time_attributes, policy
            )
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

        if (
            parent_stop.duration_constraint != child_stop.duration_constraint
            or parent_stop.duration_minutes != child_stop.duration_minutes
            or parent_stop.duration_source != child_stop.duration_source
        ):
            owner_strength = attribute_ownership.strength_for(
                stop_id, ("duration",), policy
            )
            accounting_from, accounting_from_source = _accounting_duration(
                parent_stop,
                default_visit_minutes,
            )
            accounting_to, accounting_to_source = _accounting_duration(
                child_stop,
                default_visit_minutes,
            )
            delta = (
                float(accounting_to - accounting_from)
                if accounting_from is not None and accounting_to is not None
                else None
            )
            cost_scale = 1.0 if delta is None else max(1.0, abs(delta) / 60.0)
            cost = _change_cost("duration_change", owner_strength, policy) * cost_scale
            duration_changes.append(
                DurationChange(
                    stop_id=stop_id,
                    day=child_stop.day if child_stop.day is not None else parent_stop.day,
                    from_constraint=parent_stop.duration_constraint,
                    to_constraint=child_stop.duration_constraint,
                    from_minutes=parent_stop.duration_minutes,
                    to_minutes=child_stop.duration_minutes,
                    from_source=parent_stop.duration_source,
                    to_source=child_stop.duration_source,
                    accounting_from_minutes=accounting_from,
                    accounting_to_minutes=accounting_to,
                    accounting_from_source=accounting_from_source,
                    accounting_to_source=accounting_to_source,
                    delta_minutes=delta,
                    owner_strength=owner_strength,
                    cost=cost,
                )
            )
            _add_day(affected_days, parent_stop.day)
            _add_day(affected_days, child_stop.day)

        if parent_stop.time_window_constraint != child_stop.time_window_constraint:
            owner_strength = attribute_ownership.strength_for(
                stop_id, ("time_window",), policy
            )
            cost = _change_cost("time_window_change", owner_strength, policy)
            time_window_changes.append(
                TimeWindowChange(
                    stop_id=stop_id,
                    day=child_stop.day if child_stop.day is not None else parent_stop.day,
                    from_constraint=parent_stop.time_window_constraint,
                    to_constraint=child_stop.time_window_constraint,
                    from_earliest_arrival=(
                        parent_stop.time_window_constraint.get("earliest_arrival")
                        if parent_stop.time_window_constraint is not None
                        else None
                    ),
                    to_earliest_arrival=(
                        child_stop.time_window_constraint.get("earliest_arrival")
                        if child_stop.time_window_constraint is not None
                        else None
                    ),
                    from_latest_departure=(
                        parent_stop.time_window_constraint.get("latest_departure")
                        if parent_stop.time_window_constraint is not None
                        else None
                    ),
                    to_latest_departure=(
                        child_stop.time_window_constraint.get("latest_departure")
                        if child_stop.time_window_constraint is not None
                        else None
                    ),
                    owner_strength=owner_strength,
                    cost=cost,
                )
            )
            _add_day(affected_days, parent_stop.day)
            _add_day(affected_days, child_stop.day)

        if (
            parent_stop.itinerary_role != child_stop.itinerary_role
            or parent_stop.itinerary_role_source
            != child_stop.itinerary_role_source
        ):
            role_owner_strength = _owner_strength(
                stop_id,
                {**parent_role_strengths, **child_role_strengths},
                policy,
            )
            cost = _change_cost("role_change", role_owner_strength, policy)
            role_changes.append(
                RoleChange(
                    stop_id=stop_id,
                    day=(
                        child_stop.day
                        if child_stop.day is not None
                        else parent_stop.day
                    ),
                    from_role=parent_stop.itinerary_role,
                    to_role=child_stop.itinerary_role,
                    from_source=parent_stop.itinerary_role_source,
                    to_source=child_stop.itinerary_role_source,
                    owner_strength=role_owner_strength,
                    cost=cost,
                )
            )
            _add_day(affected_days, parent_stop.day)
            _add_day(affected_days, child_stop.day)

        if parent_stop.order != child_stop.order:
            owner_strength = attribute_ownership.strength_for(
                stop_id, ("order",), policy
            )
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
            {**legacy_parent_strengths, **legacy_child_strengths},
            policy,
            fallback_targets=(
                str(day),
                parent_lodging.get(day),
                child_lodging.get(day),
            ),
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
            {**legacy_parent_strengths, **legacy_child_strengths},
            policy,
            fallback_targets=(
                str(day),
                parent_routes.get(day),
                child_routes.get(day),
            ),
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
        + sum(change.cost for change in duration_changes)
        + sum(change.cost for change in time_window_changes)
        + sum(change.cost for change in role_changes)
        + sum(change.cost for change in reorder_changes)
        + sum(change.cost for change in lodging_changes)
        + sum(change.cost for change in road_changes)
    )
    attribute_cost_changes: list[dict[str, Any]] = []
    for change in added_stops:
        attribute_cost_changes.append(
            ownership_change_evidence(
                change_kind="added_stop",
                target_stop_id=change.stop_id,
                attributes=("existence",),
                resolution=attribute_ownership,
                policy=policy,
                delta_scale=1.0,
                before_value={"existence": False},
                after_value={"existence": True},
            )
        )
    for change in deleted_stops:
        attribute_cost_changes.append(
            ownership_change_evidence(
                change_kind="deleted_stop",
                target_stop_id=change.stop_id,
                attributes=("existence",),
                resolution=attribute_ownership,
                policy=policy,
                delta_scale=1.0,
                before_value={"existence": True},
                after_value={"existence": False},
            )
        )
    for change in day_moves:
        attribute_cost_changes.append(
            ownership_change_evidence(
                change_kind="day_move",
                target_stop_id=change.stop_id,
                attributes=("day",),
                resolution=attribute_ownership,
                policy=policy,
                delta_scale=1.0,
                before_value={"day": change.from_day},
                after_value={"day": change.to_day},
            )
        )
    for change in time_shifts:
        attributes = tuple(sorted(
            attribute
            for attribute, changed in (
                ("start_time", change.from_start != change.to_start),
                ("end_time", change.from_end != change.to_end),
            )
            if changed
        ))
        attribute_cost_changes.append(
            ownership_change_evidence(
                change_kind="time_shift",
                target_stop_id=change.stop_id,
                attributes=attributes,
                resolution=attribute_ownership,
                policy=policy,
                delta_scale=max(1.0, abs(change.delta_minutes) / 60.0),
                before_value={
                    attribute: (
                        change.from_start
                        if attribute == "start_time"
                        else change.from_end
                    )
                    for attribute in attributes
                },
                after_value={
                    attribute: (
                        change.to_start
                        if attribute == "start_time"
                        else change.to_end
                    )
                    for attribute in attributes
                },
            )
        )
    for change in duration_changes:
        scale = (
            1.0
            if change.delta_minutes is None
            else max(1.0, abs(change.delta_minutes) / 60.0)
        )
        attribute_cost_changes.append(
            ownership_change_evidence(
                change_kind="duration_change",
                target_stop_id=change.stop_id,
                attributes=("duration",),
                resolution=attribute_ownership,
                policy=policy,
                delta_scale=scale,
                before_value={
                    "duration": _ownership_duration_value(
                        change.from_constraint,
                        change.from_minutes,
                    )
                },
                after_value={
                    "duration": _ownership_duration_value(
                        change.to_constraint,
                        change.to_minutes,
                    )
                },
            )
        )
    for change in time_window_changes:
        attribute_cost_changes.append(
            ownership_change_evidence(
                change_kind="time_window_change",
                target_stop_id=change.stop_id,
                attributes=("time_window",),
                resolution=attribute_ownership,
                policy=policy,
                delta_scale=1.0,
                before_value={"time_window": change.from_constraint},
                after_value={"time_window": change.to_constraint},
            )
        )
    for change in reorder_changes:
        attribute_cost_changes.append(
            ownership_change_evidence(
                change_kind="order_change",
                target_stop_id=change.stop_id,
                attributes=("order",),
                resolution=attribute_ownership,
                policy=policy,
                delta_scale=1.0,
                before_value={"order": change.from_order},
                after_value={"order": change.to_order},
            )
        )
    for change in lodging_changes:
        if attribute_ownership.activated and change.owner_strength:
            raise ValueError(
                "relation-specific attribute diff cannot bind legacy lodging ownership"
            )
        attribute_cost_changes.append(
            ownership_change_evidence(
                change_kind="lodging_change",
                target_stop_id=f"lodging:{change.day}",
                attributes=("lodging",),
                resolution=attribute_ownership,
                policy=policy,
                delta_scale=1.0,
                before_value={"lodging": change.from_lodging_id},
                after_value={"lodging": change.to_lodging_id},
            )
        )
    for change in road_changes:
        if attribute_ownership.activated and change.owner_strength:
            raise ValueError(
                "relation-specific attribute diff cannot bind legacy road ownership"
            )
        attribute_cost_changes.append(
            ownership_change_evidence(
                change_kind="road_change",
                target_stop_id=f"route:{change.day}",
                attributes=("road",),
                resolution=attribute_ownership,
                policy=policy,
                delta_scale=1.0,
                before_value={"road": change.from_route_id},
                after_value={"road": change.to_route_id},
            )
        )
    attribute_cost_changes.sort(
        key=lambda row: (
            str(row["change_kind"]),
            str(row["target_stop_id"]),
            tuple(row["attributes"]),
        )
    )
    attribute_ownership_evidence = build_attribute_ownership_evidence(
        parent=parent,
        resolution=attribute_ownership,
        changes=tuple(attribute_cost_changes),
        policy=policy,
    )
    if attribute_ownership_evidence is not None and role_changes:
        raise ValueError(
            "combined role and relation-specific attribute diff is not supported"
        )
    if attribute_ownership_evidence is not None:
        derived_costs = {
            (
                str(row["change_kind"]),
                str(row["target_stop_id"]),
            ): float(row["cost"])
            for row in attribute_cost_changes
        }
        actual_costs = {
            **{("added_stop", row.stop_id): row.cost for row in added_stops},
            **{("deleted_stop", row.stop_id): row.cost for row in deleted_stops},
            **{("day_move", row.stop_id): row.cost for row in day_moves},
            **{("time_shift", row.stop_id): row.cost for row in time_shifts},
            **{
                ("duration_change", row.stop_id): row.cost
                for row in duration_changes
            },
            **{
                ("time_window_change", row.stop_id): row.cost
                for row in time_window_changes
            },
            **{("order_change", row.stop_id): row.cost for row in reorder_changes},
            **{
                ("lodging_change", f"lodging:{row.day}"): row.cost
                for row in lodging_changes
            },
            **{
                ("road_change", f"route:{row.day}"): row.cost
                for row in road_changes
            },
        }
        if derived_costs != actual_costs:
            raise ValueError("attribute ownership cost evidence does not match diff")
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
    if duration_changes:
        diff_payload["duration_changes"] = [change.to_record() for change in duration_changes]
    if time_window_changes:
        diff_payload["time_window_changes"] = [
            change.to_record() for change in time_window_changes
        ]
    if role_changes:
        diff_payload["role_changes"] = [
            change.to_record() for change in role_changes
        ]
    if attribute_ownership_evidence is not None:
        diff_payload["attribute_ownership_evidence"] = (
            attribute_ownership_evidence
        )
    if sum(
        bool(changes)
        for changes in (duration_changes, time_window_changes, role_changes)
    ) > 1:
        raise ValueError("combined typed attribute diff is not supported")
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
        duration_changes=tuple(duration_changes),
        time_window_changes=tuple(time_window_changes),
        role_changes=tuple(role_changes),
        attribute_ownership_evidence=attribute_ownership_evidence,
        schema_version=(
            "plan-diff-v5"
            if attribute_ownership_evidence is not None
            else "plan-diff-v4"
            if role_changes
            else "plan-diff-v3"
            if time_window_changes
            else "plan-diff-v2"
            if duration_changes
            else "plan-diff-v1"
        ),
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
        duration_constraint, duration_minutes, duration_source = _duration_state(stop)
        itinerary_role, itinerary_role_source = _itinerary_role_state(stop)
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
            duration_constraint=duration_constraint,
            duration_minutes=duration_minutes,
            duration_source=duration_source,
            time_window_constraint=_time_window_constraint(
                stop.get("time_window_constraint")
            ),
            itinerary_role=itinerary_role,
            itinerary_role_source=itinerary_role_source,
        )
    return stops


def _duration_constraint(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("duration_constraint must be canonical exact duration or null")
    fields = ("mode", "preferred_minutes", "minimum_minutes", "maximum_minutes")
    if set(value) != set(fields):
        raise ValueError("duration_constraint must be canonical exact duration or null")
    preferred = value.get("preferred_minutes")
    if (
        value.get("mode") != "exact"
        or isinstance(preferred, bool)
        or not isinstance(preferred, int)
        or not 15 <= preferred <= 480
        or value.get("minimum_minutes") != preferred
        or value.get("maximum_minutes") != preferred
    ):
        raise ValueError("duration_constraint must be canonical exact duration or null")
    return {field: value[field] for field in fields}


def _time_window_constraint(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    fields = (
        "schema_version",
        "earliest_arrival",
        "latest_departure",
        "early_arrival_policy",
        "latest_departure_semantics",
    )
    if not isinstance(value, dict) or set(value) != set(fields):
        raise ValueError("time_window_constraint must be canonical or null")
    earliest = value.get("earliest_arrival")
    latest = value.get("latest_departure")
    if (
        value.get("schema_version") != "stop-time-window-constraint-v1"
        or value.get("early_arrival_policy") != "wait_until_earliest_arrival"
        or value.get("latest_departure_semantics") != "departure_after_visit"
        or (earliest is None and latest is None)
        or not _valid_clock(earliest)
        or not _valid_clock(latest)
        or (
            earliest is not None
            and latest is not None
            and _clock_minutes(earliest) > _clock_minutes(latest)
        )
    ):
        raise ValueError("time_window_constraint must be canonical or null")
    return {field: value[field] for field in fields}


def _valid_clock(value: Any) -> bool:
    return value is None or (
        isinstance(value, str)
        and re.fullmatch(r"(?:[01][0-9]|2[0-3]):[0-5][0-9]", value) is not None
    )


def _clock_minutes(value: str) -> int:
    hours, minutes = value.split(":")
    return int(hours) * 60 + int(minutes)


def _duration_state(
    stop: dict[str, Any],
) -> tuple[dict[str, Any] | None, float | None, str]:
    constraint = _duration_constraint(stop.get("duration_constraint"))
    values: list[tuple[str, float]] = []
    for key in ("visit_duration_minutes", "duration_minutes", "service_minutes", "dwell_minutes"):
        value = stop.get(key)
        if value is None:
            continue
        if not _valid_duration_number(value):
            raise ValueError(f"{key} must be a finite number from 15 to 480")
        values.append((key, float(value)))
    if len({minutes for _, minutes in values}) > 1:
        raise ValueError("duration scalar aliases disagree")
    if constraint is not None and (
        not values or values[0][1] != float(constraint["preferred_minutes"])
    ):
        raise ValueError("duration_constraint and scalar duration disagree")
    if not values:
        return constraint, None, "unavailable"
    return constraint, values[0][1], values[0][0]


def _valid_duration_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and 15 <= float(value) <= 480
    )


def _accounting_duration(stop: _PlanStop, default_visit_minutes: float | None) -> tuple[float | None, str]:
    if stop.duration_minutes is not None:
        return stop.duration_minutes, stop.duration_source
    if default_visit_minutes is not None:
        return float(default_visit_minutes), "configured_evaluator_default"
    return None, "unavailable"


def _ownership_duration_value(
    constraint: dict[str, Any] | None,
    minutes: float | None,
) -> dict[str, Any] | None:
    if constraint is not None:
        return dict(constraint)
    if minutes is None:
        return None
    if not float(minutes).is_integer():
        raise ValueError(
            "relation-specific duration ownership requires integer minutes"
        )
    exact_minutes = int(minutes)
    return {
        "mode": "exact",
        "preferred_minutes": exact_minutes,
        "minimum_minutes": exact_minutes,
        "maximum_minutes": exact_minutes,
    }


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


def _legacy_constraint_strength_by_target(plan: PlanArtifactV2) -> dict[str, str]:
    """Preserve pre-v5 lodging/road cost behavior outside migrated relations."""

    strengths: dict[str, str] = {}
    try:
        constraints = active_owned_constraints(
            tuple(dict(record) for record in plan.owned_constraints)
        )
    except Exception:
        constraints = ()
    for constraint in constraints:
        target = constraint.target_id
        current = strengths.get(target)
        if current is None or _strength_rank(constraint) > _strength_rank(current):
            strengths[target] = constraint.strength.value
    return strengths


def _changed_attribute_keys(
    parent_stops: dict[str, _PlanStop],
    child_stops: dict[str, _PlanStop],
) -> frozenset[tuple[str, str]]:
    keys: set[tuple[str, str]] = {
        (stop_id, "existence")
        for stop_id in set(parent_stops) ^ set(child_stops)
    }
    for stop_id in sorted(set(parent_stops) & set(child_stops)):
        parent = parent_stops[stop_id]
        child = child_stops[stop_id]
        if parent.day != child.day:
            keys.add((stop_id, "day"))
        if parent.start_time != child.start_time:
            keys.add((stop_id, "start_time"))
        if parent.end_time != child.end_time:
            keys.add((stop_id, "end_time"))
        if (
            parent.duration_constraint != child.duration_constraint
            or parent.duration_minutes != child.duration_minutes
            or parent.duration_source != child.duration_source
        ):
            keys.add((stop_id, "duration"))
        if parent.time_window_constraint != child.time_window_constraint:
            keys.add((stop_id, "time_window"))
        if parent.order != child.order:
            keys.add((stop_id, "order"))
    return frozenset(keys)


def _role_constraint_strength_by_target(plan: PlanArtifactV2) -> dict[str, str]:
    strengths: dict[str, str] = {}
    try:
        constraints = active_owned_constraints(
            tuple(dict(record) for record in plan.owned_constraints)
        )
    except Exception:
        constraints = ()
    for constraint in constraints:
        if not _is_role_scoped_constraint(constraint):
            continue
        if (
            not isinstance(constraint.value, str)
            or constraint.value not in _STOP_ROLES
        ):
            raise ValueError("role-scoped constraint value is invalid")
        target = constraint.target_id
        current = strengths.get(target)
        if current is None or _strength_rank(constraint) > _strength_rank(current):
            strengths[target] = constraint.strength.value
    return strengths


def _is_role_scoped_constraint(constraint: OwnedConstraint) -> bool:
    if constraint.scope.value not in {"stop", "poi"}:
        return False
    relation = constraint.relation.strip().lower()
    return relation in {
        "role",
        "itinerary_role",
        "role_equals",
        "set_role",
        "attribute:role",
    }


def _itinerary_role_state(stop: dict[str, Any]) -> tuple[str | None, str]:
    value = stop.get("itinerary_role")
    if value is None:
        if stop.get("itinerary_role_source") not in {None, "unavailable"}:
            raise ValueError(
                "missing itinerary_role cannot claim an explicit role source"
            )
        return None, "unavailable"
    if (
        not isinstance(value, str)
        or value not in _STOP_ROLES
        or stop.get("itinerary_role_source") != "user_declared_itinerary_role"
    ):
        raise ValueError(
            "explicit itinerary_role requires the closed vocabulary and user source"
        )
    return value, "user_declared_itinerary_role"


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
