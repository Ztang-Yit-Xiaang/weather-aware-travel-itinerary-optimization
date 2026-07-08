"""Canonical immutable plan, ownership, and diff model helpers."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any

JsonValue = Any


class ConstraintOrigin(StrEnum):
    USER = "user"
    USER_BOOKING = "user_booking"
    BOOKING = "booking"
    LLM_INTERPRETATION = "llm_interpretation"
    SOLVER = "solver"
    SYSTEM = "system"
    EXTERNAL_DATA = "external_data"
    DATASET = "dataset"
    REPAIR = "repair"
    EXPERIMENT = "experiment"


class ConstraintStrength(StrEnum):
    LOCKED = "locked"
    BOOKED = "booked"
    HARD = "hard"
    STRONG = "strong"
    SOFT = "soft"
    WEAK = "weak"
    PREFERENCE = "preference"
    TEST_ONLY = "test_only"


class ConstraintScope(StrEnum):
    STOP = "stop"
    POI = "poi"
    DAY = "day"
    TIME_WINDOW = "time_window"
    TIME = "time"
    LODGING = "lodging"
    ROUTE = "route"
    ROAD = "road"
    REGION = "region"
    PLAN = "plan"
    TRIP = "trip"
    USER = "user"


class RelaxationPolicy(StrEnum):
    NEVER = "never"
    EXPLICIT_ONLY = "explicit_only"
    AUTO_WITH_PENALTY = "auto_with_penalty"
    ALWAYS = "always"


@dataclass(frozen=True)
class OwnedConstraint:
    """Typed ownership label for a user, booking, data, or system constraint."""

    constraint_id: str
    origin: ConstraintOrigin
    strength: ConstraintStrength
    scope: ConstraintScope
    target_id: str
    relation: str
    value: JsonValue
    confirmed: bool
    relaxation_policy: RelaxationPolicy
    evidence_refs: tuple[str, ...] = ()
    schema_version: str = "owned-constraint-v1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "constraint_id", str(self.constraint_id).strip())
        object.__setattr__(self, "origin", _coerce_enum(ConstraintOrigin, self.origin))
        object.__setattr__(self, "strength", _coerce_enum(ConstraintStrength, self.strength))
        object.__setattr__(self, "scope", _coerce_enum(ConstraintScope, self.scope))
        object.__setattr__(self, "target_id", str(self.target_id).strip())
        object.__setattr__(self, "relation", str(self.relation).strip())
        object.__setattr__(self, "confirmed", bool(self.confirmed))
        object.__setattr__(self, "relaxation_policy", _coerce_enum(RelaxationPolicy, self.relaxation_policy))
        object.__setattr__(self, "evidence_refs", tuple(str(ref) for ref in self.evidence_refs))

    @property
    def is_active(self) -> bool:
        if not self.confirmed:
            return False
        if self.origin == ConstraintOrigin.LLM_INTERPRETATION and not self.confirmed:
            return False
        return self.strength != ConstraintStrength.TEST_ONLY

    def can_relax(self, *, explicit_permission: bool = False, allow_booked_relaxation: bool = False) -> bool:
        if not self.is_active:
            return False
        if self.strength == ConstraintStrength.LOCKED:
            return False
        if self.relaxation_policy == RelaxationPolicy.NEVER:
            return False
        if self.strength == ConstraintStrength.BOOKED:
            return bool(explicit_permission or allow_booked_relaxation)
        if self.relaxation_policy == RelaxationPolicy.EXPLICIT_ONLY:
            return bool(explicit_permission)
        return self.relaxation_policy in {RelaxationPolicy.AUTO_WITH_PENALTY, RelaxationPolicy.ALWAYS}

    def to_record(self) -> dict[str, Any]:
        return {
            "constraint_id": self.constraint_id,
            "origin": self.origin.value,
            "strength": self.strength.value,
            "scope": self.scope.value,
            "target_id": self.target_id,
            "relation": self.relation,
            "value": self.value,
            "confirmed": self.confirmed,
            "relaxation_policy": self.relaxation_policy.value,
            "evidence_refs": list(self.evidence_refs),
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> OwnedConstraint:
        return cls(
            constraint_id=str(record.get("constraint_id", "")),
            origin=record.get("origin", ""),
            strength=record.get("strength", ""),
            scope=record.get("scope", ""),
            target_id=str(record.get("target_id", "")),
            relation=str(record.get("relation", "")),
            value=record.get("value"),
            confirmed=bool(record.get("confirmed", False)),
            relaxation_policy=record.get("relaxation_policy", ""),
            evidence_refs=tuple(record.get("evidence_refs", ())),
            schema_version=str(record.get("schema_version", "owned-constraint-v1")),
        )


@dataclass(frozen=True)
class OwnershipPolicy:
    """Deterministic weights used by plan diffs and later repair objectives."""

    change_type_weights: dict[str, float] = field(
        default_factory=lambda: {
            "added_stop": 1.0,
            "deleted_stop": 2.0,
            "day_move": 2.0,
            "time_shift": 0.25,
            "order_change": 0.5,
            "lodging_change": 5.0,
            "road_change": 3.0,
        }
    )
    constraint_strength_weights: dict[str, float] = field(
        default_factory=lambda: {
            "locked": 1000.0,
            "booked": 500.0,
            "hard": 100.0,
            "strong": 50.0,
            "soft": 10.0,
            "weak": 5.0,
            "preference": 1.0,
            "test_only": 0.0,
        }
    )

    def weight_for_change(self, change_type: str) -> float:
        return float(self.change_type_weights.get(change_type, 1.0))

    def weight_for_strength(self, strength: ConstraintStrength | str | None) -> float:
        if strength is None or strength == "":
            return 1.0
        key = enum_value(strength)
        return float(self.constraint_strength_weights.get(key, 1.0))


@dataclass(frozen=True)
class StopChange:
    stop_id: str
    day: int | None
    order: int | None = None
    owner_strength: str = ""
    cost: float = 0.0

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DayMove:
    stop_id: str
    from_day: int | None
    to_day: int | None
    owner_strength: str = ""
    cost: float = 0.0

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TimeShift:
    stop_id: str
    from_start: str | None
    to_start: str | None
    from_end: str | None = None
    to_end: str | None = None
    delta_minutes: float = 0.0
    owner_strength: str = ""
    cost: float = 0.0

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OrderChange:
    stop_id: str
    day: int | None
    from_order: int | None
    to_order: int | None
    owner_strength: str = ""
    cost: float = 0.0

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LodgingChange:
    day: int
    from_lodging_id: str | None
    to_lodging_id: str | None
    owner_strength: str = ""
    cost: float = 0.0

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RoadChange:
    day: int
    from_route_id: str | None
    to_route_id: str | None
    owner_strength: str = ""
    cost: float = 0.0

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PlanDiff:
    diff_id: str
    parent_plan_id: str
    child_plan_id: str
    added_stops: tuple[StopChange, ...]
    deleted_stops: tuple[StopChange, ...]
    day_moves: tuple[DayMove, ...]
    time_shifts: tuple[TimeShift, ...]
    reorder_changes: tuple[OrderChange, ...]
    lodging_changes: tuple[LodgingChange, ...]
    road_changes: tuple[RoadChange, ...]
    unchanged_days: tuple[int, ...]
    weighted_edit_cost: float
    schema_version: str = "plan-diff-v1"

    def to_record(self) -> dict[str, Any]:
        return {
            "diff_id": self.diff_id,
            "parent_plan_id": self.parent_plan_id,
            "child_plan_id": self.child_plan_id,
            "added_stops": [change.to_record() for change in self.added_stops],
            "deleted_stops": [change.to_record() for change in self.deleted_stops],
            "day_moves": [change.to_record() for change in self.day_moves],
            "time_shifts": [change.to_record() for change in self.time_shifts],
            "reorder_changes": [change.to_record() for change in self.reorder_changes],
            "lodging_changes": [change.to_record() for change in self.lodging_changes],
            "road_changes": [change.to_record() for change in self.road_changes],
            "unchanged_days": list(self.unchanged_days),
            "weighted_edit_cost": self.weighted_edit_cost,
            "schema_version": self.schema_version,
        }


def enum_value(value: StrEnum | str | None) -> str:
    if isinstance(value, StrEnum):
        return value.value
    if value is None:
        return ""
    return str(value)


def validate_owned_constraints(constraints: tuple[OwnedConstraint, ...] | tuple[dict[str, Any], ...]) -> tuple[str, ...]:
    errors: list[str] = []
    seen_ids: set[str] = set()
    for index, raw_constraint in enumerate(constraints):
        try:
            constraint = raw_constraint if isinstance(raw_constraint, OwnedConstraint) else OwnedConstraint.from_record(raw_constraint)
        except Exception as exc:
            errors.append(f"constraint[{index}] has invalid enum or shape: {exc}")
            continue

        if not constraint.constraint_id:
            errors.append(f"constraint[{index}] is missing constraint_id")
        elif constraint.constraint_id in seen_ids:
            errors.append(f"constraint[{index}] duplicates constraint_id {constraint.constraint_id}")
        seen_ids.add(constraint.constraint_id)

        if not constraint.target_id:
            errors.append(f"constraint[{index}] is missing target_id")
        if not constraint.relation:
            errors.append(f"constraint[{index}] is missing relation")
        if (
            constraint.strength == ConstraintStrength.LOCKED
            and constraint.relaxation_policy in {RelaxationPolicy.AUTO_WITH_PENALTY, RelaxationPolicy.ALWAYS}
        ):
            errors.append(f"{constraint.constraint_id} is locked but has automatic relaxation")
        if (
            constraint.strength == ConstraintStrength.BOOKED
            and constraint.relaxation_policy in {RelaxationPolicy.AUTO_WITH_PENALTY, RelaxationPolicy.ALWAYS}
        ):
            errors.append(f"{constraint.constraint_id} is booked but does not require explicit relaxation permission")
    return tuple(errors)


def active_owned_constraints(
    constraints: tuple[OwnedConstraint, ...] | tuple[dict[str, Any], ...],
) -> tuple[OwnedConstraint, ...]:
    active: list[OwnedConstraint] = []
    for raw_constraint in constraints:
        constraint = raw_constraint if isinstance(raw_constraint, OwnedConstraint) else OwnedConstraint.from_record(raw_constraint)
        if constraint.is_active:
            active.append(constraint)
    return tuple(active)


def plan_from_phase0_route_stops(route_stops_df: Any, *, plan_id: str):
    """Create a v2 plan artifact from the Phase 0 route-stops table."""

    from ..research_artifacts import PlanArtifactV2

    if not str(plan_id).strip():
        raise ValueError("plan_id must be nonempty")
    records = [_clean_record(record) for record in route_stops_df.to_dict("records")]
    selected_stops: list[dict[str, Any]] = []
    day_assignments: dict[str, int] = {}
    ordered_by_day: dict[int, list[str]] = {}
    lodging_assignments: dict[str, str] = {}
    route_ids_by_day: dict[int, str] = {}

    for index, record in enumerate(sorted(records, key=_phase0_sort_key), start=1):
        stop_id = _record_stop_id(record, index)
        day = _coerce_int(record.get("day"))
        stop_record = dict(record)
        stop_record["stop_id"] = stop_id
        if day is not None:
            stop_record["day"] = day
            day_assignments[stop_id] = day
            ordered_by_day.setdefault(day, []).append(stop_id)
            lodging_id = _first_nonempty(record, "lodging_id", "hotel_id", "overnight_city", "route_end_name")
            if lodging_id:
                lodging_assignments[str(day)] = str(lodging_id)
            route_id = _first_nonempty(record, "route_id", "road_route_id")
            if route_id:
                route_ids_by_day[day] = str(route_id)
        selected_stops.append(stop_record)

    ordered_days = tuple(
        {"day": day, "stop_ids": tuple(stop_ids)}
        for day, stop_ids in sorted(ordered_by_day.items(), key=lambda item: item[0])
    )
    sequence = tuple(stop["stop_id"] for stop in selected_stops)
    first_record = records[0] if records else {}
    return PlanArtifactV2(
        plan_id=str(plan_id),
        source_run_id=str(_first_nonempty(first_record, "source_run_id", "run_id") or "phase0_route_stops"),
        planning_request_id=str(
            _first_nonempty(first_record, "planning_request_id", "request_id") or "phase0_route_stops"
        ),
        catalog_snapshot_id=str(_first_nonempty(first_record, "catalog_snapshot_id") or "unknown_catalog"),
        context_snapshot_id=str(_first_nonempty(first_record, "context_snapshot_id") or "unknown_context"),
        selected_stops=tuple(selected_stops),
        day_assignments=day_assignments,
        sequence=sequence,
        lodging_assignments=lodging_assignments,
        ordered_days=ordered_days,
        route_ids_by_day=route_ids_by_day,
    )


def _coerce_enum(enum_cls: type[StrEnum], value: StrEnum | str) -> StrEnum:
    if isinstance(value, enum_cls):
        return value
    return enum_cls(str(value))


def _clean_record(record: dict[str, Any]) -> dict[str, Any]:
    clean: dict[str, Any] = {}
    for key, value in record.items():
        if isinstance(value, float) and math.isnan(value):
            clean[key] = None
        else:
            clean[key] = value
    return clean


def _phase0_sort_key(record: dict[str, Any]) -> tuple[int, int, str]:
    day = _coerce_int(record.get("day"))
    order = _coerce_int(record.get("stop_order") or record.get("order") or record.get("route_sequence_index"))
    return (day if day is not None else 10**9, order if order is not None else 10**9, str(record))


def _record_stop_id(record: dict[str, Any], index: int) -> str:
    value = _first_nonempty(record, "stop_id", "poi_id", "attraction_id", "attraction_name", "name", "poi", "stop_name")
    return str(value).strip() if value else f"phase0_stop_{index}"


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
