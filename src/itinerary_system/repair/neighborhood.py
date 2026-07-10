"""Progressive repair neighborhoods for immutable parent plans."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any

from ..plans import ConstraintScope, ConstraintStrength, OwnedConstraint, active_owned_constraints
from ..research_artifacts import PlanArtifactV2


class RepairRadius(StrEnum):
    SAME_STOP_TIME_SHIFT = "same_stop_time_shift"
    SAME_DAY_REPLACEMENT = "same_day_replacement"
    ADJACENT_DAY_MOVE = "adjacent_day_move"
    HOTEL_PRESERVING_REROUTE = "hotel_preserving_reroute"
    HOTEL_CHANGING_REPAIR = "hotel_changing_repair"
    FULL_REOPTIMIZATION = "full_reoptimization"


REPAIR_RADIUS_ORDER: tuple[RepairRadius, ...] = (
    RepairRadius.SAME_STOP_TIME_SHIFT,
    RepairRadius.SAME_DAY_REPLACEMENT,
    RepairRadius.ADJACENT_DAY_MOVE,
    RepairRadius.HOTEL_PRESERVING_REROUTE,
    RepairRadius.HOTEL_CHANGING_REPAIR,
    RepairRadius.FULL_REOPTIMIZATION,
)


@dataclass(frozen=True)
class ParentPlanIndex:
    """Compact lookup structure derived from a parent plan artifact."""

    plan_id: str
    days: tuple[int, ...]
    stop_ids: tuple[str, ...]
    stop_day: dict[str, int]
    stops_by_day: dict[int, tuple[str, ...]]
    stop_order: dict[str, int]
    stop_names: dict[str, str]
    stop_id_by_name: dict[str, str]
    lodging_by_day: dict[int, str]
    route_ids_by_day: dict[int, str]
    active_constraints: tuple[OwnedConstraint, ...]

    def stop_ids_for_days(self, days: tuple[int, ...]) -> tuple[str, ...]:
        selected: list[str] = []
        day_set = set(days)
        for stop_id in self.stop_ids:
            if self.stop_day.get(stop_id) in day_set:
                selected.append(stop_id)
        return tuple(selected)


@dataclass(frozen=True)
class RepairNeighborhood:
    radius: RepairRadius
    editable_days: tuple[int, ...]
    editable_stop_ids: tuple[str, ...]
    editable_lodging_nights: tuple[int, ...]
    frozen_constraint_ids: tuple[str, ...]
    boundary_route_days: tuple[int, ...] = ()
    frozen_stop_ids: tuple[str, ...] = ()
    frozen_lodging_nights: tuple[int, ...] = ()
    reason_codes: tuple[str, ...] = ()

    def can_edit_stop(self, stop_id: str) -> bool:
        return str(stop_id) in set(self.editable_stop_ids)

    def can_edit_lodging_night(self, day: int) -> bool:
        return int(day) in set(self.editable_lodging_nights)

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        record["radius"] = self.radius.value
        return record


class RepairNeighborhoodBuilder:
    """Build progressive editable regions around a disruption scope."""

    def build(self, parent: PlanArtifactV2, request: Any, radius: RepairRadius | str) -> RepairNeighborhood:
        return build_repair_neighborhood(parent, request, radius)

    def affected_days(self, parent: PlanArtifactV2, request: Any) -> tuple[int, ...]:
        return affected_days(request, parent=parent)

    def freeze_constraints(
        self,
        parent: PlanArtifactV2,
        neighborhood: RepairNeighborhood,
        *,
        allow_booked_relaxation: bool = False,
    ) -> tuple[OwnedConstraint, ...]:
        return freeze_constraints(parent, neighborhood, allow_booked_relaxation=allow_booked_relaxation)


def build_parent_plan_index(parent: PlanArtifactV2) -> ParentPlanIndex:
    stop_day: dict[str, int] = {}
    stops_by_day: dict[int, list[str]] = {}
    stop_order: dict[str, int] = {}
    stop_names: dict[str, str] = {}
    stop_id_by_name: dict[str, str] = {}
    stop_ids: list[str] = []

    sequence_order = {str(stop_id): index for index, stop_id in enumerate(parent.sequence, start=1)}
    day_assignments = {str(stop_id): int(day) for stop_id, day in parent.day_assignments.items()}
    for index, raw_stop in enumerate(parent.selected_stops, start=1):
        stop = dict(raw_stop)
        stop_id = _stop_id(stop, index)
        day = _coerce_int(stop.get("day"))
        if day is None:
            assigned = day_assignments.get(stop_id)
            if assigned is not None:
                day = assigned
        order = _coerce_int(
            stop.get("stop_order")
            or stop.get("order")
            or stop.get("route_sequence_index")
            or sequence_order.get(stop_id)
            or index
        )
        stop_ids.append(stop_id)
        if day is not None:
            stop_day[stop_id] = day
            stops_by_day.setdefault(day, []).append(stop_id)
        if order is not None:
            stop_order[stop_id] = order
        name = str(stop.get("name") or stop.get("attraction_name") or stop.get("poi") or stop_id).strip()
        stop_names[stop_id] = name
        stop_id_by_name[_normalize_name(name)] = stop_id
        stop_id_by_name[_normalize_name(stop_id)] = stop_id

    lodging_by_day = _lodging_by_day(parent)
    route_ids_by_day = {int(day): str(route_id) for day, route_id in parent.route_ids_by_day.items()}
    days = tuple(sorted(set(stop_day.values()) | set(lodging_by_day) | set(route_ids_by_day)))
    active_constraints = active_owned_constraints(tuple(dict(record) for record in parent.owned_constraints))
    return ParentPlanIndex(
        plan_id=parent.plan_id,
        days=days,
        stop_ids=tuple(stop_ids),
        stop_day=stop_day,
        stops_by_day={day: tuple(ids) for day, ids in sorted(stops_by_day.items())},
        stop_order=stop_order,
        stop_names=stop_names,
        stop_id_by_name=stop_id_by_name,
        lodging_by_day=lodging_by_day,
        route_ids_by_day=route_ids_by_day,
        active_constraints=active_constraints,
    )


def build_repair_neighborhood(
    parent: PlanArtifactV2,
    request: Any,
    radius: RepairRadius | str,
) -> RepairNeighborhood:
    radius = RepairRadius(str(radius))
    index = build_parent_plan_index(parent)
    scoped_days = affected_days(request, parent=parent)
    editable_days = _editable_days_for_radius(index, scoped_days, radius)
    editable_stop_ids = index.stop_ids_for_days(editable_days)
    explicit_targets = _target_stop_ids(request, index)
    if explicit_targets and radius != RepairRadius.FULL_REOPTIMIZATION:
        editable_stop_ids = _ordered_unique((*editable_stop_ids, *explicit_targets), index.stop_ids)
    editable_lodging_nights = _editable_lodging_nights_for_radius(index, editable_days, radius)
    boundary_route_days = _boundary_route_days_for_radius(index, editable_days, radius)
    frozen_stop_ids = tuple(stop_id for stop_id in index.stop_ids if stop_id not in set(editable_stop_ids))
    frozen_lodging_nights = tuple(day for day in index.lodging_by_day if day not in set(editable_lodging_nights))
    preliminary = RepairNeighborhood(
        radius=radius,
        editable_days=editable_days,
        editable_stop_ids=editable_stop_ids,
        editable_lodging_nights=editable_lodging_nights,
        frozen_constraint_ids=(),
        boundary_route_days=boundary_route_days,
        frozen_stop_ids=frozen_stop_ids,
        frozen_lodging_nights=frozen_lodging_nights,
        reason_codes=_reason_codes(radius, scoped_days),
    )
    frozen_constraints = freeze_constraints(
        parent,
        preliminary,
        allow_booked_relaxation=_allow_booked_relaxation(request),
    )
    return RepairNeighborhood(
        radius=preliminary.radius,
        editable_days=preliminary.editable_days,
        editable_stop_ids=preliminary.editable_stop_ids,
        editable_lodging_nights=preliminary.editable_lodging_nights,
        frozen_constraint_ids=tuple(constraint.constraint_id for constraint in frozen_constraints),
        boundary_route_days=preliminary.boundary_route_days,
        frozen_stop_ids=preliminary.frozen_stop_ids,
        frozen_lodging_nights=preliminary.frozen_lodging_nights,
        reason_codes=preliminary.reason_codes,
    )


def affected_days(request: Any, *, parent: PlanArtifactV2 | None = None) -> tuple[int, ...]:
    """Infer disruption days from explicit request fields and confirmed constraints."""

    days: set[int] = set()
    confirmed_constraints = _confirmed_constraints(request)
    for key in ("affected_days", "editable_days", "disrupted_days"):
        days.update(_coerce_int(value) for value in _iter_values(confirmed_constraints.get(key, ())))
    for key in ("target_days", "move_stops", "stop_day_overrides"):
        raw = confirmed_constraints.get(key, {})
        if isinstance(raw, dict):
            days.update(_coerce_int(value) for value in raw.values())
    days.update(_coerce_int(value) for value in _iter_values(getattr(request, "affected_days", ())))
    days.update(_coerce_int(value) for value in _iter_values(getattr(request, "editable_days", ())))

    if parent is not None:
        index = build_parent_plan_index(parent)
        for stop_id in _target_stop_ids(request, index):
            day = index.stop_day.get(stop_id)
            if day is not None:
                days.add(day)
        if not days:
            days.update(index.days)
    return tuple(sorted(day for day in days if day is not None))


def freeze_constraints(
    parent: PlanArtifactV2,
    neighborhood: RepairNeighborhood,
    *,
    allow_booked_relaxation: bool = False,
) -> tuple[OwnedConstraint, ...]:
    """Return active constraints that must remain fixed for this neighborhood."""

    index = build_parent_plan_index(parent)
    frozen: list[OwnedConstraint] = []
    for constraint in index.active_constraints:
        if constraint.strength == ConstraintStrength.LOCKED:
            frozen.append(constraint)
            continue
        if constraint.strength == ConstraintStrength.BOOKED and not allow_booked_relaxation:
            frozen.append(constraint)
            continue
        if not _constraint_target_editable(constraint, neighborhood, index):
            frozen.append(constraint)
    return tuple(frozen)


def _editable_days_for_radius(
    index: ParentPlanIndex,
    affected: tuple[int, ...],
    radius: RepairRadius,
) -> tuple[int, ...]:
    if radius == RepairRadius.FULL_REOPTIMIZATION:
        return index.days
    if radius in {RepairRadius.SAME_STOP_TIME_SHIFT, RepairRadius.SAME_DAY_REPLACEMENT}:
        return tuple(day for day in affected if day in set(index.days))
    if radius in {
        RepairRadius.ADJACENT_DAY_MOVE,
        RepairRadius.HOTEL_PRESERVING_REROUTE,
        RepairRadius.HOTEL_CHANGING_REPAIR,
    }:
        expanded = set(affected)
        for day in affected:
            expanded.add(day - 1)
            expanded.add(day + 1)
        return tuple(day for day in index.days if day in expanded)
    return ()


def _editable_lodging_nights_for_radius(
    index: ParentPlanIndex,
    editable_days: tuple[int, ...],
    radius: RepairRadius,
) -> tuple[int, ...]:
    if radius in {RepairRadius.HOTEL_CHANGING_REPAIR, RepairRadius.FULL_REOPTIMIZATION}:
        return tuple(day for day in index.days if day in set(editable_days))
    if radius == RepairRadius.ADJACENT_DAY_MOVE:
        return tuple(day for day in index.lodging_by_day if day in set(editable_days))
    return ()


def _boundary_route_days_for_radius(
    index: ParentPlanIndex,
    editable_days: tuple[int, ...],
    radius: RepairRadius,
) -> tuple[int, ...]:
    if radius == RepairRadius.FULL_REOPTIMIZATION:
        return index.days
    if radius in {
        RepairRadius.ADJACENT_DAY_MOVE,
        RepairRadius.HOTEL_PRESERVING_REROUTE,
        RepairRadius.HOTEL_CHANGING_REPAIR,
    }:
        return tuple(day for day in index.days if day in set(editable_days))
    return editable_days


def _constraint_target_editable(
    constraint: OwnedConstraint,
    neighborhood: RepairNeighborhood,
    index: ParentPlanIndex,
) -> bool:
    target = constraint.target_id
    scope = constraint.scope
    if neighborhood.radius == RepairRadius.FULL_REOPTIMIZATION and constraint.strength != ConstraintStrength.LOCKED:
        if constraint.strength == ConstraintStrength.BOOKED:
            return target in _editable_lodging_targets(neighborhood, index) or _target_day(target) in set(
                neighborhood.editable_lodging_nights
            )
        return True
    if scope in {ConstraintScope.STOP, ConstraintScope.POI}:
        return target in set(neighborhood.editable_stop_ids)
    if scope in {ConstraintScope.TIME, ConstraintScope.TIME_WINDOW}:
        return target in set(neighborhood.editable_stop_ids) or _target_day(target) in set(neighborhood.editable_days)
    if scope in {ConstraintScope.DAY, ConstraintScope.TRIP, ConstraintScope.PLAN}:
        return _target_day(target) in set(neighborhood.editable_days)
    if scope == ConstraintScope.LODGING:
        return target in _editable_lodging_targets(neighborhood, index) or _target_day(target) in set(
            neighborhood.editable_lodging_nights
        )
    if scope in {ConstraintScope.ROUTE, ConstraintScope.ROAD}:
        target_day = _target_day(target)
        if target_day is not None:
            return target_day in set(neighborhood.boundary_route_days)
        route_days = {day for day, route_id in index.route_ids_by_day.items() if route_id == target}
        return bool(route_days & set(neighborhood.boundary_route_days))
    return False


def _editable_lodging_targets(neighborhood: RepairNeighborhood, index: ParentPlanIndex) -> set[str]:
    return {
        lodging_id
        for day, lodging_id in index.lodging_by_day.items()
        if day in set(neighborhood.editable_lodging_nights)
    }


def _target_stop_ids(request: Any, index: ParentPlanIndex) -> tuple[str, ...]:
    targets: list[str] = []
    confirmed_constraints = _confirmed_constraints(request)
    for key in ("target_stop_ids", "editable_stop_ids", "must_keep", "must_delete", "must_include"):
        targets.extend(_lookup_stop_id(value, index) for value in _iter_values(confirmed_constraints.get(key, ())))
    for key in ("target_days", "move_stops", "stop_day_overrides"):
        raw = confirmed_constraints.get(key, {})
        if isinstance(raw, dict):
            targets.extend(_lookup_stop_id(value, index) for value in raw.keys())
    parsed_intent = getattr(request, "parsed_intent", None)
    if parsed_intent is not None:
        for attr in ("must_keep", "must_delete", "must_include"):
            targets.extend(_lookup_stop_id(value, index) for value in getattr(parsed_intent, attr, ()))
    return _ordered_unique(tuple(target for target in targets if target), index.stop_ids)


def _lookup_stop_id(value: Any, index: ParentPlanIndex) -> str:
    text = str(value).strip()
    if not text:
        return ""
    if text in set(index.stop_ids):
        return text
    return index.stop_id_by_name.get(_normalize_name(text), "")


def _confirmed_constraints(request: Any) -> dict[str, Any]:
    raw = getattr(request, "confirmed_constraints", {})
    return dict(raw) if isinstance(raw, dict) else {}


def _allow_booked_relaxation(request: Any) -> bool:
    confirmed_constraints = _confirmed_constraints(request)
    return bool(
        getattr(request, "allow_booked_relaxation", False)
        or confirmed_constraints.get("allow_booked_relaxation", False)
    )


def _lodging_by_day(parent: PlanArtifactV2) -> dict[int, str]:
    lodging: dict[int, str] = {}
    for raw_day, lodging_id in parent.lodging_assignments.items():
        day = _coerce_int(raw_day)
        if day is not None and str(lodging_id).strip():
            lodging[day] = str(lodging_id)
    for stop in parent.selected_stops:
        day = _coerce_int(stop.get("day"))
        lodging_id = str(stop.get("lodging_id") or stop.get("hotel_id") or stop.get("overnight_city") or "").strip()
        if day is not None and lodging_id:
            lodging.setdefault(day, lodging_id)
    return lodging


def _stop_id(stop: dict[str, Any], index: int) -> str:
    for key in ("stop_id", "poi_id", "attraction_id", "attraction_name", "name", "poi", "stop_name"):
        value = stop.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return f"stop_{index}"


def _target_day(target: str) -> int | None:
    text = str(target).strip().lower()
    if text.startswith("day:"):
        text = text.split(":", 1)[1]
    if text.startswith("lodging:") or text.startswith("route:"):
        text = text.split(":", 1)[1]
    return _coerce_int(text)


def _coerce_int(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except Exception:
        return None


def _iter_values(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, dict):
        return tuple(value.values())
    if isinstance(value, str):
        return (value,)
    try:
        return tuple(value)
    except TypeError:
        return (value,)


def _ordered_unique(values: tuple[str, ...], reference_order: tuple[str, ...]) -> tuple[str, ...]:
    selected = set(values)
    ordered = [value for value in reference_order if value in selected]
    ordered.extend(value for value in values if value not in set(ordered))
    return tuple(dict.fromkeys(ordered))


def _normalize_name(value: str) -> str:
    return " ".join(str(value).strip().lower().split())


def _reason_codes(radius: RepairRadius, affected: tuple[int, ...]) -> tuple[str, ...]:
    codes = [f"radius:{radius.value}"]
    if affected:
        codes.append("scope:explicit_or_inferred_days")
    else:
        codes.append("scope:all_parent_days")
    if radius == RepairRadius.FULL_REOPTIMIZATION:
        codes.append("fallback:full_reoptimization")
    return tuple(codes)
