"""Canonical immutable plan, ownership, and diff model helpers."""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any

from ..research_artifacts import stable_content_hash

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
            "duration_change": 0.25,
            "time_window_change": 0.25,
            "role_change": 0.25,
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
class DurationChange:
    """Typed visit-duration change with raw and evaluator-accounting truth."""

    stop_id: str
    day: int | None
    from_constraint: dict[str, JsonValue] | None
    to_constraint: dict[str, JsonValue] | None
    from_minutes: float | None
    to_minutes: float | None
    from_source: str
    to_source: str
    accounting_from_minutes: float | None
    accounting_to_minutes: float | None
    accounting_from_source: str
    accounting_to_source: str
    delta_minutes: float | None
    owner_strength: str = ""
    cost: float = 0.0

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TimeWindowChange:
    """Trip-specific required time-window change, separate from place hours."""

    stop_id: str
    day: int | None
    from_constraint: dict[str, JsonValue] | None
    to_constraint: dict[str, JsonValue] | None
    from_earliest_arrival: str | None
    to_earliest_arrival: str | None
    from_latest_departure: str | None
    to_latest_departure: str | None
    owner_strength: str = ""
    cost: float = 0.0

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RoleChange:
    """Trip-owned itinerary-role change, separate from place categories."""

    stop_id: str
    day: int | None
    from_role: str | None
    to_role: str | None
    from_source: str
    to_source: str
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
    duration_changes: tuple[DurationChange, ...] = ()
    time_window_changes: tuple[TimeWindowChange, ...] = ()
    role_changes: tuple[RoleChange, ...] = ()
    attribute_ownership_evidence: dict[str, Any] | None = None
    schema_version: str = "plan-diff-v1"

    def to_record(self) -> dict[str, Any]:
        typed_presence = (
            bool(self.duration_changes),
            bool(self.time_window_changes),
            bool(self.role_changes),
        )
        if self.schema_version == "plan-diff-v1" and (
            any(typed_presence) or self.attribute_ownership_evidence is not None
        ):
            raise ValueError("plan-diff-v1 cannot carry typed attribute evidence")
        if self.schema_version == "plan-diff-v2" and (
            not self.duration_changes
            or self.time_window_changes
            or self.role_changes
            or self.attribute_ownership_evidence is not None
        ):
            raise ValueError("plan-diff-v2 must carry only duration changes")
        if self.schema_version == "plan-diff-v3" and (
            self.duration_changes
            or not self.time_window_changes
            or self.role_changes
            or self.attribute_ownership_evidence is not None
        ):
            raise ValueError("plan-diff-v3 must carry only time-window changes")
        if self.schema_version == "plan-diff-v4" and (
            self.duration_changes
            or self.time_window_changes
            or not self.role_changes
            or self.attribute_ownership_evidence is not None
        ):
            raise ValueError("plan-diff-v4 must carry only role changes")
        if self.schema_version == "plan-diff-v5" and (
            not isinstance(self.attribute_ownership_evidence, dict)
            or not self.attribute_ownership_evidence
            or self.role_changes
            or sum((bool(self.duration_changes), bool(self.time_window_changes))) > 1
        ):
            raise ValueError(
                "plan-diff-v5 requires ownership evidence and excludes role changes"
            )
        if self.schema_version == "plan-diff-v5":
            _validate_v5_attribute_ownership_evidence(
                self.attribute_ownership_evidence,
                parent_plan_id=self.parent_plan_id,
                weighted_edit_cost=self.weighted_edit_cost,
                expected_changes=_v5_expected_change_rows(self),
            )
        if self.schema_version not in {
            "plan-diff-v1",
            "plan-diff-v2",
            "plan-diff-v3",
            "plan-diff-v4",
            "plan-diff-v5",
        }:
            raise ValueError("unsupported plan diff schema")
        record = {
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
        if self.schema_version == "plan-diff-v2":
            record["duration_changes"] = [change.to_record() for change in self.duration_changes]
        elif self.schema_version == "plan-diff-v3":
            record["time_window_changes"] = [change.to_record() for change in self.time_window_changes]
        elif self.schema_version == "plan-diff-v4":
            record["role_changes"] = [change.to_record() for change in self.role_changes]
        elif self.schema_version == "plan-diff-v5":
            if self.duration_changes:
                record["duration_changes"] = [
                    change.to_record() for change in self.duration_changes
                ]
            if self.time_window_changes:
                record["time_window_changes"] = [
                    change.to_record() for change in self.time_window_changes
                ]
            record["attribute_ownership_evidence"] = dict(
                self.attribute_ownership_evidence or {}
            )
        return record


def enum_value(value: StrEnum | str | None) -> str:
    if isinstance(value, StrEnum):
        return value.value
    if value is None:
        return ""
    return str(value)


def _v5_expected_change_rows(diff: PlanDiff) -> list[dict[str, Any]]:
    """Derive the exact v5 evidence universe from the typed diff records."""

    rows: list[dict[str, Any]] = []

    def append(
        *,
        change_kind: str,
        target_stop_id: str,
        attributes: tuple[str, ...],
        before_value: dict[str, Any],
        after_value: dict[str, Any],
        owner_strength: str,
        delta_scale: float,
        cost: float,
    ) -> None:
        rows.append(
            {
                "change_kind": change_kind,
                "target_stop_id": target_stop_id,
                "attributes": list(sorted(attributes)),
                "before_value": before_value,
                "after_value": after_value,
                "owner_strength": owner_strength,
                "delta_scale": float(delta_scale),
                "cost": float(cost),
            }
        )

    for change in diff.added_stops:
        append(
            change_kind="added_stop",
            target_stop_id=change.stop_id,
            attributes=("existence",),
            before_value={"existence": False},
            after_value={"existence": True},
            owner_strength=change.owner_strength,
            delta_scale=1.0,
            cost=change.cost,
        )
    for change in diff.deleted_stops:
        append(
            change_kind="deleted_stop",
            target_stop_id=change.stop_id,
            attributes=("existence",),
            before_value={"existence": True},
            after_value={"existence": False},
            owner_strength=change.owner_strength,
            delta_scale=1.0,
            cost=change.cost,
        )
    for change in diff.day_moves:
        append(
            change_kind="day_move",
            target_stop_id=change.stop_id,
            attributes=("day",),
            before_value={"day": change.from_day},
            after_value={"day": change.to_day},
            owner_strength=change.owner_strength,
            delta_scale=1.0,
            cost=change.cost,
        )
    for change in diff.time_shifts:
        attributes = tuple(
            sorted(
                attribute
                for attribute, changed in (
                    ("start_time", change.from_start != change.to_start),
                    ("end_time", change.from_end != change.to_end),
                )
                if changed
            )
        )
        append(
            change_kind="time_shift",
            target_stop_id=change.stop_id,
            attributes=attributes,
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
            owner_strength=change.owner_strength,
            delta_scale=max(1.0, abs(float(change.delta_minutes)) / 60.0),
            cost=change.cost,
        )
    for change in diff.duration_changes:
        delta_scale = (
            1.0
            if change.delta_minutes is None
            else max(1.0, abs(float(change.delta_minutes)) / 60.0)
        )
        append(
            change_kind="duration_change",
            target_stop_id=change.stop_id,
            attributes=("duration",),
            before_value={
                "duration": _v5_duration_observed_value(
                    change.from_constraint,
                    change.from_minutes,
                )
            },
            after_value={
                "duration": _v5_duration_observed_value(
                    change.to_constraint,
                    change.to_minutes,
                )
            },
            owner_strength=change.owner_strength,
            delta_scale=delta_scale,
            cost=change.cost,
        )
    for change in diff.time_window_changes:
        append(
            change_kind="time_window_change",
            target_stop_id=change.stop_id,
            attributes=("time_window",),
            before_value={"time_window": change.from_constraint},
            after_value={"time_window": change.to_constraint},
            owner_strength=change.owner_strength,
            delta_scale=1.0,
            cost=change.cost,
        )
    for change in diff.reorder_changes:
        append(
            change_kind="order_change",
            target_stop_id=change.stop_id,
            attributes=("order",),
            before_value={"order": change.from_order},
            after_value={"order": change.to_order},
            owner_strength=change.owner_strength,
            delta_scale=1.0,
            cost=change.cost,
        )
    for change in diff.lodging_changes:
        append(
            change_kind="lodging_change",
            target_stop_id=f"lodging:{change.day}",
            attributes=("lodging",),
            before_value={"lodging": change.from_lodging_id},
            after_value={"lodging": change.to_lodging_id},
            owner_strength=change.owner_strength,
            delta_scale=1.0,
            cost=change.cost,
        )
    for change in diff.road_changes:
        append(
            change_kind="road_change",
            target_stop_id=f"route:{change.day}",
            attributes=("road",),
            before_value={"road": change.from_route_id},
            after_value={"road": change.to_route_id},
            owner_strength=change.owner_strength,
            delta_scale=1.0,
            cost=change.cost,
        )
    return sorted(
        rows,
        key=lambda row: (
            row["change_kind"],
            row["target_stop_id"],
            tuple(row["attributes"]),
        ),
    )


def _v5_duration_observed_value(
    constraint: dict[str, JsonValue] | None,
    minutes: float | None,
) -> dict[str, JsonValue] | None:
    if constraint is not None:
        return dict(constraint)
    if minutes is None:
        return None
    if not math.isfinite(float(minutes)) or not float(minutes).is_integer():
        raise ValueError("plan-diff-v5 duration evidence is invalid")
    exact_minutes = int(minutes)
    return {
        "mode": "exact",
        "preferred_minutes": exact_minutes,
        "minimum_minutes": exact_minutes,
        "maximum_minutes": exact_minutes,
    }


def _validate_v5_attribute_ownership_evidence(
    value: dict[str, Any] | None,
    *,
    parent_plan_id: str,
    weighted_edit_cost: float,
    expected_changes: list[dict[str, Any]],
) -> None:
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "parent_plan_id",
        "parent_plan_content_hash",
        "semantic_scope",
        "ownership_policy",
        "constraints",
        "changes",
        "content_hash",
    }:
        raise ValueError("plan-diff-v5 ownership evidence is invalid")
    payload = dict(value)
    content_hash = payload.pop("content_hash", None)
    semantic_scope = value.get("semantic_scope")
    constraints = value.get("constraints")
    changes = value.get("changes")
    ownership_policy = value.get("ownership_policy")
    if (
        value.get("schema_version")
        != "plan-diff-attribute-ownership-evidence-v1"
        or value.get("parent_plan_id") != parent_plan_id
        or not isinstance(value.get("parent_plan_content_hash"), str)
        or re.fullmatch(
            r"[0-9a-f]{16}", value["parent_plan_content_hash"]
        )
        is None
        or semantic_scope
        != {
            "cost_semantics": "attribute_sensitivity",
            "permission_claim": "not_evaluated",
            "feasibility_claim": "not_evaluated",
            "cost_coverage": "all_listed_diff_changes",
            "relation_specific_attributes": [
                "day",
                "duration",
                "end_time",
                "existence",
                "order",
                "start_time",
                "time_window",
            ],
            "lodging_and_road_relations": "not_migrated",
        }
        or not isinstance(ownership_policy, dict)
        or set(ownership_policy)
        != {"change_type_weights", "constraint_strength_weights"}
        or not _valid_v5_weight_map(ownership_policy["change_type_weights"])
        or not _valid_v5_weight_map(
            ownership_policy["constraint_strength_weights"]
        )
        or not isinstance(constraints, list)
        or not constraints
        or not isinstance(changes, list)
        or not changes
        or not isinstance(content_hash, str)
        or stable_content_hash(payload) != content_hash
    ):
        raise ValueError("plan-diff-v5 ownership evidence is invalid")
    constraint_fields = {
        "constraint_id",
        "target_stop_id",
        "attribute",
        "relation",
        "canonical_value",
        "strength",
        "scope",
        "origin",
        "relaxation_policy",
    }
    change_fields = {
        "change_kind",
        "target_stop_id",
        "attributes",
        "applicable_constraint_ids",
        "selected_constraint_ids",
        "before_value",
        "after_value",
        "constraint_satisfaction",
        "owner_strength",
        "base_change_weight",
        "strength_multiplier",
        "delta_scale",
        "cost",
    }
    if any(not isinstance(row, dict) or set(row) != constraint_fields for row in constraints):
        raise ValueError("plan-diff-v5 ownership constraint evidence is invalid")
    if any(not isinstance(row, dict) or set(row) != change_fields for row in changes):
        raise ValueError("plan-diff-v5 ownership change evidence is invalid")
    constraint_ids = [row.get("constraint_id") for row in constraints]
    if (
        any(
            not _valid_v5_constraint_row(row)
            for row in constraints
        )
        or constraint_ids != sorted(constraint_ids)
        or len(set(constraint_ids)) != len(constraint_ids)
    ):
        raise ValueError("plan-diff-v5 ownership constraint evidence is invalid")
    constraints_by_id = {
        str(row["constraint_id"]): row for row in constraints
    }
    ordered_changes = sorted(
        changes,
        key=lambda row: (
            str(row.get("change_kind") or ""),
            str(row.get("target_stop_id") or ""),
            tuple(row.get("attributes") or ()),
        ),
    )
    if (
        len(changes) != len(expected_changes)
        or changes != ordered_changes
        or any(
            not _valid_v5_change_row(
                row,
                constraints_by_id=constraints_by_id,
                ownership_policy=ownership_policy,
                expected=expected_changes[index],
            )
            for index, row in enumerate(changes)
        )
    ):
        raise ValueError("plan-diff-v5 ownership change evidence is invalid")
    consumed_constraint_ids = sorted(
        {
            constraint_id
            for row in changes
            for constraint_id in row["applicable_constraint_ids"]
        }
    )
    if consumed_constraint_ids != constraint_ids:
        raise ValueError("plan-diff-v5 ownership constraint evidence is invalid")
    costs = [row.get("cost") for row in changes]
    if (
        not isinstance(weighted_edit_cost, (int, float))
        or isinstance(weighted_edit_cost, bool)
        or not math.isfinite(float(weighted_edit_cost))
        or any(
            not isinstance(cost, (int, float))
            or isinstance(cost, bool)
            or not math.isfinite(float(cost))
            or float(cost) < 0
            for cost in costs
        )
        or not math.isclose(
            sum(float(cost) for cost in costs),
            float(weighted_edit_cost),
            rel_tol=0.0,
            abs_tol=1e-9,
        )
    ):
        raise ValueError("plan-diff-v5 ownership cost evidence is invalid")


def _valid_v5_weight_map(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and all(isinstance(key, str) and key for key in value)
        and all(
            isinstance(weight, (int, float))
            and not isinstance(weight, bool)
            and math.isfinite(float(weight))
            and float(weight) >= 0
            for weight in value.values()
        )
    )


def _valid_v5_constraint_row(value: dict[str, Any]) -> bool:
    attribute = value.get("attribute")
    relation = str(value.get("relation") or "")
    relation_aliases = {
        "existence": {"must_keep", "existence", "exists", "attribute:existence"},
        "day": {"day", "stop_day", "assigned_day", "attribute:day"},
        "start_time": {
            "start_time",
            "arrival_time",
            "stop_start_time",
            "attribute:start_time",
        },
        "end_time": {
            "end_time",
            "departure_time",
            "stop_end_time",
            "attribute:end_time",
        },
        "duration": {
            "duration",
            "visit_duration",
            "duration_constraint",
            "attribute:duration",
        },
        "time_window": {
            "time_window",
            "required_time_window",
            "stop_time_window",
            "attribute:time_window",
        },
        "order": {"stop_order", "attribute:stop_order"},
    }
    return (
        isinstance(value.get("constraint_id"), str)
        and bool(value["constraint_id"])
        and isinstance(value.get("target_stop_id"), str)
        and bool(value["target_stop_id"])
        and attribute in relation_aliases
        and relation in relation_aliases[attribute]
        and _valid_v5_attribute_value(attribute, value.get("canonical_value"))
        and value.get("strength") in {item.value for item in ConstraintStrength}
        and value.get("scope") in {ConstraintScope.STOP.value, ConstraintScope.POI.value}
        and value.get("origin") in {item.value for item in ConstraintOrigin}
        and value.get("relaxation_policy")
        in {item.value for item in RelaxationPolicy}
        and not (relation == "must_keep" and value.get("canonical_value") is not True)
    )


def _valid_v5_attribute_value(attribute: str, value: Any) -> bool:
    clock = re.compile(r"^(?:[01]\d|2[0-3]):[0-5]\d$")
    if attribute == "existence":
        return isinstance(value, bool)
    if attribute == "day":
        return isinstance(value, int) and not isinstance(value, bool) and 1 <= value <= 366
    if attribute == "order":
        return isinstance(value, int) and not isinstance(value, bool) and 1 <= value <= 4096
    if attribute in {"start_time", "end_time"}:
        return isinstance(value, str) and clock.fullmatch(value) is not None
    if attribute == "duration":
        return (
            isinstance(value, dict)
            and set(value)
            == {"mode", "preferred_minutes", "minimum_minutes", "maximum_minutes"}
            and value.get("mode") == "exact"
            and isinstance(value.get("preferred_minutes"), int)
            and not isinstance(value.get("preferred_minutes"), bool)
            and 15 <= value["preferred_minutes"] <= 480
            and value.get("minimum_minutes") == value["preferred_minutes"]
            and value.get("maximum_minutes") == value["preferred_minutes"]
        )
    if attribute == "time_window":
        if not isinstance(value, dict) or set(value) != {
            "schema_version",
            "earliest_arrival",
            "latest_departure",
            "early_arrival_policy",
            "latest_departure_semantics",
        }:
            return False
        earliest = value.get("earliest_arrival")
        latest = value.get("latest_departure")
        return (
            value.get("schema_version") == "stop-time-window-constraint-v1"
            and value.get("early_arrival_policy") == "wait_until_earliest_arrival"
            and value.get("latest_departure_semantics") == "departure_after_visit"
            and not (earliest is None and latest is None)
            and (
                earliest is None
                or isinstance(earliest, str)
                and clock.fullmatch(earliest) is not None
            )
            and (
                latest is None
                or isinstance(latest, str)
                and clock.fullmatch(latest) is not None
            )
            and not (
                isinstance(earliest, str)
                and isinstance(latest, str)
                and earliest > latest
            )
        )
    return False


def _valid_v5_change_row(
    value: dict[str, Any],
    *,
    constraints_by_id: dict[str, dict[str, Any]],
    ownership_policy: dict[str, Any],
    expected: dict[str, Any],
) -> bool:
    change_kind = value.get("change_kind")
    allowed_kinds = {
        "added_stop",
        "deleted_stop",
        "day_move",
        "time_shift",
        "duration_change",
        "time_window_change",
        "order_change",
        "lodging_change",
        "road_change",
    }
    attributes = value.get("attributes")
    before = value.get("before_value")
    after = value.get("after_value")
    applicable = value.get("applicable_constraint_ids")
    selected = value.get("selected_constraint_ids")
    satisfaction = value.get("constraint_satisfaction")
    if (
        change_kind not in allowed_kinds
        or change_kind != expected.get("change_kind")
        or not isinstance(value.get("target_stop_id"), str)
        or not value["target_stop_id"]
        or value["target_stop_id"] != expected.get("target_stop_id")
        or not isinstance(attributes, list)
        or not attributes
        or len(set(attributes)) != len(attributes)
        or attributes != sorted(attributes)
        or not isinstance(before, dict)
        or not isinstance(after, dict)
        or set(before) != set(attributes)
        or set(after) != set(attributes)
        or not isinstance(applicable, list)
        or applicable != sorted(set(applicable))
        or not isinstance(selected, list)
        or selected != sorted(set(selected))
        or not set(selected).issubset(applicable)
        or not isinstance(satisfaction, list)
        or attributes != expected.get("attributes")
        or before != expected.get("before_value")
        or after != expected.get("after_value")
        or value.get("owner_strength") != expected.get("owner_strength")
        or not _same_finite_number(
            value.get("delta_scale"),
            float(expected.get("delta_scale", float("nan"))),
        )
        or not _same_finite_number(
            value.get("cost"),
            float(expected.get("cost", float("nan"))),
        )
    ):
        return False
    supported = {
        "existence",
        "day",
        "start_time",
        "end_time",
        "duration",
        "time_window",
        "order",
        "lodging",
        "road",
    }
    if any(attribute not in supported for attribute in attributes):
        return False
    for attribute in attributes:
        if attribute in {"lodging", "road"}:
            if before[attribute] is not None and (
                not isinstance(before[attribute], str) or not before[attribute]
            ):
                return False
            if after[attribute] is not None and (
                not isinstance(after[attribute], str) or not after[attribute]
            ):
                return False
        else:
            if before[attribute] is not None and not _valid_v5_attribute_value(
                attribute, before[attribute]
            ):
                return False
            if after[attribute] is not None and not _valid_v5_attribute_value(
                attribute, after[attribute]
            ):
                return False
    expected_applicable = sorted(
        constraint_id
        for constraint_id, row in constraints_by_id.items()
        if row["target_stop_id"] == value["target_stop_id"]
        and row["attribute"] in attributes
    )
    if applicable != expected_applicable:
        return False
    strength_weights = ownership_policy["constraint_strength_weights"]
    owner_strength = ""
    if applicable:
        owner_strength = max(
            (str(constraints_by_id[constraint_id]["strength"]) for constraint_id in applicable),
            key=lambda strength: (float(strength_weights.get(strength, 1.0)), strength),
        )
    multiplier = float(strength_weights.get(owner_strength, 1.0))
    expected_selected = sorted(
        constraint_id
        for constraint_id in applicable
        if float(strength_weights.get(constraints_by_id[constraint_id]["strength"], 1.0))
        == multiplier
    )
    change_weights = ownership_policy["change_type_weights"]
    base = float(change_weights.get(change_kind, 1.0))
    delta_scale = value.get("delta_scale")
    cost = value.get("cost")
    if (
        value.get("owner_strength") != owner_strength
        or selected != expected_selected
        or not _same_finite_number(value.get("base_change_weight"), base)
        or not _same_finite_number(value.get("strength_multiplier"), multiplier)
        or not isinstance(delta_scale, (int, float))
        or isinstance(delta_scale, bool)
        or not math.isfinite(float(delta_scale))
        or float(delta_scale) <= 0
        or not _same_finite_number(cost, base * multiplier * float(delta_scale))
    ):
        return False
    expected_satisfaction = []
    for constraint_id in applicable:
        row = constraints_by_id[constraint_id]
        attribute = str(row["attribute"])
        required = row["canonical_value"]
        expected_satisfaction.append(
            {
                "constraint_id": constraint_id,
                "attribute": attribute,
                "canonical_value": required,
                "satisfaction_before": _v5_satisfaction(before[attribute], required),
                "satisfaction_after": _v5_satisfaction(after[attribute], required),
            }
        )
    return satisfaction == expected_satisfaction


def _same_finite_number(value: Any, expected: float) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and math.isclose(float(value), expected, rel_tol=0.0, abs_tol=1e-9)
    )


def _v5_satisfaction(actual: Any, required: Any) -> str:
    if actual is None:
        return "unavailable"
    return "satisfied" if actual == required else "unsatisfied"


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
