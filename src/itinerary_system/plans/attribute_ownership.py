"""Relation-specific ownership evidence for non-role plan attributes."""

from __future__ import annotations

import re
from dataclasses import dataclass
from math import isfinite
from typing import Any

from ..research_artifacts import PlanArtifactV2, stable_content_hash
from .models import OwnedConstraint, OwnershipPolicy

_ATTRIBUTE_RELATIONS = {
    "existence": frozenset(
        {"must_keep", "existence", "exists", "attribute:existence"}
    ),
    "day": frozenset({"day", "stop_day", "assigned_day", "attribute:day"}),
    "start_time": frozenset(
        {"start_time", "arrival_time", "stop_start_time", "attribute:start_time"}
    ),
    "end_time": frozenset(
        {"end_time", "departure_time", "stop_end_time", "attribute:end_time"}
    ),
    "duration": frozenset(
        {"duration", "visit_duration", "duration_constraint", "attribute:duration"}
    ),
    "time_window": frozenset(
        {
            "time_window",
            "required_time_window",
            "stop_time_window",
            "attribute:time_window",
        }
    ),
    # PlanArtifact stop_order is one-based. Deliberately exclude the product
    # editor's zero-based sequence_index vocabulary from this domain contract.
    "order": frozenset({"stop_order", "attribute:stop_order"}),
}
_RELATION_TO_ATTRIBUTE = {
    relation: attribute
    for attribute, relations in _ATTRIBUTE_RELATIONS.items()
    for relation in relations
}
_STOP_SCOPES = frozenset({"stop", "poi"})
_CLOCK_PATTERN = re.compile(r"^(?:[01]\d|2[0-3]):[0-5]\d$")
_OWNED_CONSTRAINT_FIELDS = frozenset(
    {
        "constraint_id",
        "origin",
        "strength",
        "scope",
        "target_id",
        "relation",
        "value",
        "confirmed",
        "relaxation_policy",
        "evidence_refs",
        "schema_version",
    }
)


@dataclass(frozen=True)
class AttributeOwnershipResolution:
    """Accepted-parent relation rows applicable to actual changed attributes."""

    rows_by_key: dict[tuple[str, str], tuple[dict[str, Any], ...]]
    activated: bool

    def strength_for(
        self,
        target_id: str,
        attributes: tuple[str, ...],
        policy: OwnershipPolicy,
    ) -> str:
        rows = self.rows_for(target_id, attributes)
        if not rows:
            return ""
        return max(
            (str(row["strength"]) for row in rows),
            key=lambda strength: (policy.weight_for_strength(strength), strength),
        )

    def rows_for(
        self,
        target_id: str,
        attributes: tuple[str, ...],
    ) -> tuple[dict[str, Any], ...]:
        rows = {
            str(row["constraint_id"]): row
            for attribute in attributes
            for row in self.rows_by_key.get((target_id, attribute), ())
        }
        return tuple(rows[key] for key in sorted(rows))


def resolve_attribute_ownership(
    parent: PlanArtifactV2,
    child: PlanArtifactV2,
    changed_keys: frozenset[tuple[str, str]],
) -> AttributeOwnershipResolution:
    """Resolve only explicit, active, accepted-parent attribute relations."""

    if not changed_keys:
        return AttributeOwnershipResolution(rows_by_key={}, activated=False)
    parent_rows = _applicable_rows(parent.owned_constraints, changed_keys)
    child_rows = _applicable_rows(child.owned_constraints, changed_keys)
    activated = bool(parent_rows or child_rows)
    if not activated:
        return AttributeOwnershipResolution(rows_by_key={}, activated=False)
    if _normalized_constraints(parent.owned_constraints) != _normalized_constraints(
        child.owned_constraints
    ):
        raise ValueError(
            "child owned constraints must exactly match accepted parent constraints"
        )
    if child_rows != parent_rows:
        raise ValueError(
            "child attribute ownership evidence must exactly match accepted parent"
        )
    return AttributeOwnershipResolution(rows_by_key=parent_rows, activated=True)


def build_attribute_ownership_evidence(
    *,
    parent: PlanArtifactV2,
    resolution: AttributeOwnershipResolution,
    changes: tuple[dict[str, Any], ...],
    policy: OwnershipPolicy,
) -> dict[str, Any] | None:
    """Build a deterministic, hash-bound v5 ownership-cost evidence object."""

    if not resolution.activated:
        return None
    for weights in (
        policy.change_type_weights,
        policy.constraint_strength_weights,
    ):
        if any(
            not isinstance(key, str)
            or not key
            or isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not isfinite(float(value))
            or float(value) < 0
            for key, value in weights.items()
        ):
            raise ValueError(
                "attribute ownership policy weights must be finite and nonnegative"
            )
    constraint_rows = {
        str(row["constraint_id"]): row
        for rows in resolution.rows_by_key.values()
        for row in rows
    }
    payload = {
        "schema_version": "plan-diff-attribute-ownership-evidence-v1",
        "parent_plan_id": parent.plan_id,
        "parent_plan_content_hash": parent.content_hash,
        "semantic_scope": {
            "cost_semantics": "attribute_sensitivity",
            "permission_claim": "not_evaluated",
            "feasibility_claim": "not_evaluated",
            "cost_coverage": "all_listed_diff_changes",
            "relation_specific_attributes": sorted(_ATTRIBUTE_RELATIONS),
            "lodging_and_road_relations": "not_migrated",
        },
        "ownership_policy": {
            "change_type_weights": {
                str(key): float(value)
                for key, value in sorted(policy.change_type_weights.items())
            },
            "constraint_strength_weights": {
                str(key): float(value)
                for key, value in sorted(
                    policy.constraint_strength_weights.items()
                )
            },
        },
        "constraints": [constraint_rows[key] for key in sorted(constraint_rows)],
        "changes": list(changes),
    }
    return {**payload, "content_hash": stable_content_hash(payload)}


def ownership_change_evidence(
    *,
    change_kind: str,
    target_stop_id: str,
    attributes: tuple[str, ...],
    resolution: AttributeOwnershipResolution,
    policy: OwnershipPolicy,
    delta_scale: float,
    before_value: dict[str, Any],
    after_value: dict[str, Any],
) -> dict[str, Any]:
    """Return one independently recomputable finite cost derivation."""

    if not isfinite(delta_scale) or delta_scale <= 0:
        raise ValueError("attribute ownership delta scale must be finite and positive")
    if set(before_value) != set(attributes) or set(after_value) != set(attributes):
        raise ValueError("attribute ownership values must match changed attributes")
    normalized_before = {
        attribute: _canonical_observed_value(attribute, before_value[attribute])
        for attribute in attributes
    }
    normalized_after = {
        attribute: _canonical_observed_value(attribute, after_value[attribute])
        for attribute in attributes
    }
    rows = resolution.rows_for(target_stop_id, attributes)
    owner_strength = resolution.strength_for(target_stop_id, attributes, policy)
    strength_multiplier = policy.weight_for_strength(owner_strength)
    base_change_weight = policy.weight_for_change(change_kind)
    cost = float(base_change_weight * strength_multiplier * delta_scale)
    if not all(
        isfinite(value)
        for value in (strength_multiplier, base_change_weight, cost)
    ) or any(
        value < 0 for value in (strength_multiplier, base_change_weight, cost)
    ):
        raise ValueError("attribute ownership cost must be finite and nonnegative")
    selected_ids = sorted(
        str(row["constraint_id"])
        for row in rows
        if policy.weight_for_strength(str(row["strength"])) == strength_multiplier
    )
    satisfaction = []
    for row in rows:
        attribute = str(row["attribute"])
        required = row["canonical_value"]
        satisfaction.append(
            {
                "constraint_id": str(row["constraint_id"]),
                "attribute": attribute,
                "canonical_value": required,
                "satisfaction_before": _satisfaction(
                    normalized_before[attribute], required
                ),
                "satisfaction_after": _satisfaction(
                    normalized_after[attribute], required
                ),
            }
        )
    return {
        "change_kind": change_kind,
        "target_stop_id": target_stop_id,
        "attributes": list(attributes),
        "applicable_constraint_ids": sorted(
            str(row["constraint_id"]) for row in rows
        ),
        "selected_constraint_ids": selected_ids,
        "before_value": normalized_before,
        "after_value": normalized_after,
        "constraint_satisfaction": satisfaction,
        "owner_strength": owner_strength,
        "base_change_weight": float(base_change_weight),
        "strength_multiplier": float(strength_multiplier),
        "delta_scale": float(delta_scale),
        "cost": cost,
    }


def _applicable_rows(
    raw_constraints: tuple[dict[str, Any], ...],
    changed_keys: frozenset[tuple[str, str]],
) -> dict[tuple[str, str], tuple[dict[str, Any], ...]]:
    rows_by_key: dict[tuple[str, str], list[dict[str, Any]]] = {}
    values_by_key: dict[tuple[str, str], Any] = {}
    seen_ids: set[str] = set()
    for raw in raw_constraints:
        if not isinstance(raw, dict):
            continue
        relation = str(raw.get("relation") or "").strip().lower()
        attribute = _RELATION_TO_ATTRIBUTE.get(relation)
        if attribute is None:
            continue
        target_id = str(raw.get("target_id") or "").strip()
        key = (target_id, attribute)
        if key not in changed_keys:
            continue
        confirmed = raw.get("confirmed")
        if not isinstance(confirmed, bool):
            raise ValueError("active attribute constraint confirmation must be boolean")
        if not confirmed:
            continue
        try:
            constraint = OwnedConstraint.from_record(raw)
        except Exception as exc:
            raise ValueError("active attribute constraint is malformed") from exc
        if not constraint.is_active:
            continue
        constraint_id = constraint.constraint_id
        if (
            not constraint_id
            or constraint_id in seen_ids
            or constraint.scope.value not in _STOP_SCOPES
            or not target_id
        ):
            raise ValueError("active attribute constraint identity or scope is invalid")
        canonical_value = _canonical_value(attribute, constraint.value)
        if relation == "must_keep" and canonical_value is not True:
            raise ValueError("must_keep constraint value must be true")
        if key in values_by_key and values_by_key[key] != canonical_value:
            raise ValueError("active attribute constraints are contradictory")
        values_by_key[key] = canonical_value
        seen_ids.add(constraint_id)
        rows_by_key.setdefault(key, []).append(
            {
                "constraint_id": constraint_id,
                "target_stop_id": target_id,
                "attribute": attribute,
                "relation": relation,
                "canonical_value": canonical_value,
                "strength": constraint.strength.value,
                "scope": constraint.scope.value,
                "origin": constraint.origin.value,
                "relaxation_policy": constraint.relaxation_policy.value,
            }
        )
    return {
        key: tuple(sorted(rows, key=lambda row: str(row["constraint_id"])))
        for key, rows in sorted(rows_by_key.items())
    }


def _normalized_constraints(
    raw_constraints: tuple[dict[str, Any], ...],
) -> tuple[dict[str, Any], ...]:
    normalized: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for raw in raw_constraints:
        if not isinstance(raw, dict) or set(raw) != _OWNED_CONSTRAINT_FIELDS:
            raise ValueError("owned constraint record must be an object")
        if not isinstance(raw.get("confirmed"), bool):
            raise ValueError("owned constraint confirmation must be boolean")
        try:
            constraint = OwnedConstraint.from_record(raw)
        except Exception as exc:
            raise ValueError("owned constraint record is malformed") from exc
        if not constraint.constraint_id or constraint.constraint_id in seen_ids:
            raise ValueError("owned constraint IDs must be nonempty and unique")
        seen_ids.add(constraint.constraint_id)
        normalized.append(constraint.to_record())
    return tuple(sorted(normalized, key=lambda row: str(row["constraint_id"])))


def _canonical_value(attribute: str, value: Any) -> Any:
    if attribute == "existence":
        if not isinstance(value, bool):
            raise ValueError("attribute existence value must be boolean")
        return value
    if attribute == "day":
        return _bounded_int(value, minimum=1, maximum=366, label="day")
    if attribute == "order":
        return _bounded_int(value, minimum=1, maximum=4096, label="one-based order")
    if attribute in {"start_time", "end_time"}:
        if not isinstance(value, str) or _CLOCK_PATTERN.fullmatch(value) is None:
            raise ValueError("attribute time value must use exact HH:MM")
        return value
    if attribute == "duration":
        return _canonical_duration(value)
    if attribute == "time_window":
        return _canonical_time_window(value)
    raise ValueError("unsupported attribute ownership relation")


def _canonical_observed_value(attribute: str, value: Any) -> Any:
    if value is None:
        return None
    if attribute in _ATTRIBUTE_RELATIONS:
        return _canonical_value(attribute, value)
    if attribute in {"lodging", "road"}:
        if not isinstance(value, str) or not value:
            raise ValueError("legacy base-cost identity must be nonempty or null")
        return value
    raise ValueError("unsupported changed attribute")


def _bounded_int(value: Any, *, minimum: int, maximum: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(f"attribute {label} value is invalid")
    return value


def _canonical_duration(value: Any) -> dict[str, Any]:
    expected = {"mode", "preferred_minutes", "minimum_minutes", "maximum_minutes"}
    if not isinstance(value, dict) or set(value) != expected or value.get("mode") != "exact":
        raise ValueError("attribute duration value must be a canonical exact object")
    values = tuple(value.get(key) for key in ("preferred_minutes", "minimum_minutes", "maximum_minutes"))
    if any(
        isinstance(item, bool)
        or not isinstance(item, int)
        or not 15 <= item <= 480
        for item in values
    ) or not values[0] == values[1] == values[2]:
        raise ValueError("attribute duration value must be a canonical exact object")
    minutes = values[0]
    return {
        "mode": "exact",
        "preferred_minutes": minutes,
        "minimum_minutes": minutes,
        "maximum_minutes": minutes,
    }


def _canonical_time_window(value: Any) -> dict[str, Any]:
    expected = {
        "schema_version",
        "earliest_arrival",
        "latest_departure",
        "early_arrival_policy",
        "latest_departure_semantics",
    }
    if (
        not isinstance(value, dict)
        or set(value) != expected
        or value.get("schema_version") != "stop-time-window-constraint-v1"
        or value.get("early_arrival_policy") != "wait_until_earliest_arrival"
        or value.get("latest_departure_semantics") != "departure_after_visit"
    ):
        raise ValueError("attribute time-window value must be canonical")
    earliest = value.get("earliest_arrival")
    latest = value.get("latest_departure")
    if (
        (earliest is None and latest is None)
        or (earliest is not None and (
            not isinstance(earliest, str)
            or _CLOCK_PATTERN.fullmatch(earliest) is None
        ))
        or (latest is not None and (
            not isinstance(latest, str)
            or _CLOCK_PATTERN.fullmatch(latest) is None
        ))
        or (
            isinstance(earliest, str)
            and isinstance(latest, str)
            and earliest > latest
        )
    ):
        raise ValueError("attribute time-window value must be canonical")
    return dict(value)


def _satisfaction(actual: Any, required: Any) -> str:
    if actual is None:
        return "unavailable"
    return "satisfied" if actual == required else "unsatisfied"
