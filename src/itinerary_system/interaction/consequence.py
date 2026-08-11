"""Typed, hierarchical consequence analysis for counterfactual probe evidence."""

from __future__ import annotations

from math import isfinite
from typing import Any

from ..plans import ConstraintStrength, OwnedConstraint
from ..research_artifacts import PlanArtifactV2, stable_content_hash
from .models import (
    ConsequenceThresholds,
    ConsequenceVector,
    CounterfactualProbeResult,
    ProbeStatus,
)


def build_consequence_vector(
    *,
    parent: PlanArtifactV2,
    interpretation_id: str,
    probe_result: CounterfactualProbeResult,
) -> ConsequenceVector:
    diff = probe_result.diff_record or {}
    constraints = {
        item.constraint_id: item
        for item in (OwnedConstraint.from_record(dict(record)) for record in parent.owned_constraints)
    }
    changed_constraint_ids = tuple(probe_result.affected_constraint_ids)
    changed_constraints = tuple(constraints[item] for item in changed_constraint_ids if item in constraints)
    locked = sum(item.strength == ConstraintStrength.LOCKED for item in changed_constraints)
    booked = sum(item.strength == ConstraintStrength.BOOKED for item in changed_constraints)
    strong = sum(item.strength == ConstraintStrength.STRONG for item in changed_constraints)
    flexible = sum(
        item.strength in {ConstraintStrength.SOFT, ConstraintStrength.WEAK, ConstraintStrength.PREFERENCE}
        for item in changed_constraints
    )
    core_changes = tuple(
        item.constraint_id
        for item in changed_constraints
        if item.confirmed and item.strength in {ConstraintStrength.LOCKED, ConstraintStrength.STRONG}
    )
    affected_days = _affected_days(diff)
    hypothetical = probe_result.hypothetical_plan_record or {}
    metrics = _metric_deltas(parent, hypothetical)
    evidence_refs = tuple(
        dict.fromkeys(
            str(item).strip()
            for item in (*probe_result.evidence_refs, _diff_evidence(diff))
            if str(item).strip()
        )
    )
    seed = {
        "interpretation_id": interpretation_id,
        "probe_result_id": probe_result.probe_result_id,
        "diff_id": probe_result.diff_id,
    }
    return ConsequenceVector(
        consequence_id=f"consequence_{stable_content_hash(seed)}",
        interpretation_id=interpretation_id,
        probe_result_id=probe_result.probe_result_id,
        hard_feasible=probe_result.status in {ProbeStatus.FEASIBLE_BOUNDED, ProbeStatus.TIME_LIMIT_WITH_INCUMBENT},
        permission_change_count=len(probe_result.permission_constraint_ids),
        locked_change_count=locked,
        booked_change_count=booked,
        strong_change_count=strong,
        flexible_change_count=flexible,
        core_commitment_changes=core_changes,
        weighted_edit_cost=_optional_float(diff.get("weighted_edit_cost")),
        affected_day_count=len(affected_days),
        lodging_change_count=len(diff.get("lodging_changes", ())),
        road_change_count=len(diff.get("road_changes", ())),
        travel_minutes_delta=metrics["travel_minutes_delta"],
        monetary_cost_delta=metrics["monetary_cost_delta"],
        walking_burden_delta=metrics["walking_burden_delta"],
        contextual_risk_delta=metrics["contextual_risk_delta"],
        utility_delta=metrics["utility_delta"],
        accepted_repair_radius=probe_result.accepted_repair_radius,
        evidence_refs=evidence_refs,
    )


def equivalent_typed_repairs(results: tuple[CounterfactualProbeResult, ...]) -> bool:
    if not results or any(result.diff_record is None for result in results):
        return False
    return len({_typed_diff_signature(result.diff_record) for result in results}) == 1


def is_low_consequence(vector: ConsequenceVector, thresholds: ConsequenceThresholds) -> bool:
    if (
        not vector.hard_feasible
        or vector.permission_change_count
        or vector.locked_change_count
        or vector.booked_change_count
    ):
        return False
    if (
        vector.weighted_edit_cost is None
        or not isfinite(vector.weighted_edit_cost)
        or vector.weighted_edit_cost > thresholds.max_low_consequence_edit_cost
    ):
        return False
    if vector.affected_day_count > thresholds.max_low_consequence_affected_days:
        return False
    if (
        vector.travel_minutes_delta is not None
        and abs(vector.travel_minutes_delta) > thresholds.max_low_consequence_travel_minutes
    ):
        return False
    if (
        vector.monetary_cost_delta is not None
        and abs(vector.monetary_cost_delta) > thresholds.max_low_consequence_cost_delta
    ):
        return False
    return True


def consequences_materially_different(
    vectors: tuple[ConsequenceVector, ...],
    thresholds: ConsequenceThresholds,
) -> bool:
    if len(vectors) < 2:
        return False
    feasibility = {vector.hard_feasible for vector in vectors}
    if thresholds.feasibility_difference_is_material and len(feasibility) > 1:
        return True
    permission_shapes = {
        (vector.locked_change_count, vector.booked_change_count, vector.permission_change_count) for vector in vectors
    }
    if thresholds.permission_change_is_material and len(permission_shapes) > 1:
        return True
    if len({vector.core_commitment_changes for vector in vectors}) > 1:
        return True
    if len({vector.accepted_repair_radius for vector in vectors}) > 1:
        return True
    edit_costs = [value for value in (vector.weighted_edit_cost for vector in vectors) if value is not None]
    if edit_costs and max(edit_costs) - min(edit_costs) > thresholds.max_low_consequence_edit_cost:
        return True
    if max(vector.affected_day_count for vector in vectors) - min(vector.affected_day_count for vector in vectors) > 0:
        return True
    return (
        _known_delta_range(vectors, "travel_minutes_delta") > thresholds.max_low_consequence_travel_minutes
        or _known_delta_range(vectors, "monetary_cost_delta") > thresholds.max_low_consequence_cost_delta
    )


def _typed_diff_signature(diff: dict[str, Any] | None) -> tuple[Any, ...]:
    if not diff:
        return ("missing",)
    keys = (
        "added_stops",
        "deleted_stops",
        "day_moves",
        "time_shifts",
        "reorder_changes",
        "lodging_changes",
        "road_changes",
    )
    return tuple((key, _normalized_changes(diff.get(key, ()))) for key in keys)


def _normalized_changes(values: Any) -> tuple[tuple[tuple[str, Any], ...], ...]:
    normalized: list[tuple[tuple[str, Any], ...]] = []
    for value in values or ():
        if isinstance(value, dict):
            normalized.append(
                tuple(sorted((str(key), _hashable(item)) for key, item in value.items() if key != "cost"))
            )
        else:
            normalized.append((("value", _hashable(value)),))
    return tuple(sorted(normalized, key=repr))


def _hashable(value: Any) -> Any:
    if isinstance(value, dict):
        return tuple(sorted((str(key), _hashable(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_hashable(item) for item in value)
    return value


def _affected_days(diff: dict[str, Any]) -> tuple[int, ...]:
    days: set[int] = set()
    for key in ("added_stops", "deleted_stops", "time_shifts", "reorder_changes", "lodging_changes", "road_changes"):
        for change in diff.get(key, ()):
            day = change.get("day") if isinstance(change, dict) else None
            if isinstance(day, int):
                days.add(day)
    for change in diff.get("day_moves", ()):
        if not isinstance(change, dict):
            continue
        for key in ("from_day", "to_day"):
            if isinstance(change.get(key), int):
                days.add(int(change[key]))
    return tuple(sorted(days))


def _metric_deltas(parent: PlanArtifactV2, hypothetical: dict[str, Any]) -> dict[str, float | None]:
    parent_metrics = dict(parent.modeled_metrics)
    child_metrics = dict(hypothetical.get("modeled_metrics", {}))
    keys = {
        "travel_minutes_delta": ("travel_minutes", "total_travel_minutes"),
        "monetary_cost_delta": ("monetary_cost", "total_cost"),
        "walking_burden_delta": ("walking_burden", "walking_minutes"),
        "contextual_risk_delta": ("contextual_risk", "context_burden"),
        "utility_delta": ("utility", "total_utility"),
    }
    return {name: _first_metric_delta(parent_metrics, child_metrics, aliases) for name, aliases in keys.items()}


def _first_metric_delta(parent: dict[str, Any], child: dict[str, Any], aliases: tuple[str, ...]) -> float | None:
    for key in aliases:
        left = _optional_float(parent.get(key))
        right = _optional_float(child.get(key))
        if left is not None and right is not None:
            delta = right - left
            return delta if isfinite(delta) else None
    return None


def _diff_evidence(diff: dict[str, Any]) -> str:
    diff_id = str(diff.get("diff_id", "")).strip()
    return f"plan_diff:{diff_id}" if diff_id else ""


def _optional_float(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        return number if isfinite(number) else None
    return None


def _known_delta_range(vectors: tuple[ConsequenceVector, ...], field: str) -> float:
    values = [value for vector in vectors if (value := getattr(vector, field)) is not None and isfinite(value)]
    return max(values) - min(values) if len(values) > 1 else 0.0
