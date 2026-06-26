"""Counterfactual minimal-change itinerary repair contracts and helpers.

This module implements the research-facing scaffold for the limitation-driven
repair contribution. It does not call an LLM or claim real-world guarantees:
LLM output is represented only as a confirmed ``ParsedRepairIntent`` record,
and every repair is evaluated against frozen route/evidence artifacts.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Literal

RepairOperationName = Literal["ADD", "DELETE", "REPLACE", "MOVE", "RELAX", "KEEP"]
ConflictType = Literal["stale", "contradictory", "low_confidence", "missing"]

REPAIR_OPERATIONS: tuple[RepairOperationName, ...] = ("ADD", "DELETE", "REPLACE", "MOVE", "RELAX", "KEEP")


@dataclass(frozen=True)
class ParsedRepairIntent:
    """Confirmed parser output that may be compiled into repair constraints."""

    raw_text: str
    confirmed: bool
    confidence: float
    operations_allowed: tuple[RepairOperationName, ...] = REPAIR_OPERATIONS  # type: ignore[assignment]
    must_keep: tuple[str, ...] = ()
    must_delete: tuple[str, ...] = ()
    must_include: tuple[str, ...] = ()
    relaxations: dict[str, float] = field(default_factory=dict)
    parser_name: str = "manual_or_confirmed_parser"
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class EvidenceConflict:
    """A source conflict that should be exposed instead of silently resolved."""

    conflict_id: str
    conflict_type: ConflictType
    affected_constraint: str
    confidence: float
    resolution_policy: str
    source_ids: tuple[str, ...] = ()
    details: str = ""


@dataclass(frozen=True)
class RepairOperation:
    """An auditable itinerary edit."""

    operation: RepairOperationName
    target_stop: str | None = None
    replacement_stop: str | None = None
    day: int | None = None
    affected_constraint: str = ""
    reason: str = ""
    metrics_delta: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluationReport:
    """Whole-trip evaluation with hard feasibility separated from soft metrics."""

    route_hash: str
    feasible: bool
    hard_failures: tuple[str, ...]
    metrics: dict[str, float]
    evaluator_version: str = "repair-evaluator-v1"
    eligibility: str = "eligible"


@dataclass(frozen=True)
class CounterfactualExplanation:
    """Actionable what-would-change explanation for repair decisions."""

    target: str
    constraint: str
    minimal_required_change: str
    substitution: str | None = None
    affected_metrics: dict[str, float] = field(default_factory=dict)
    rationale: str = ""


@dataclass(frozen=True)
class RepairRequest:
    """Frozen inputs for a repair run."""

    request_id: str
    baseline_route: tuple[dict[str, Any], ...]
    user_intent: str
    evidence_records: tuple[dict[str, Any], ...] = ()
    tolerance_profile: dict[str, float] = field(default_factory=dict)
    confirmed_constraints: dict[str, Any] = field(default_factory=dict)
    candidate_pois: tuple[dict[str, Any], ...] = ()
    travel_graph: dict[str, Any] = field(default_factory=dict)
    parsed_intent: ParsedRepairIntent | None = None


@dataclass(frozen=True)
class RepairPlan:
    """A repaired route linked to its parent artifact and explanations."""

    plan_id: str
    parent_route_hash: str
    repaired_route: tuple[dict[str, Any], ...]
    operations: tuple[RepairOperation, ...]
    evidence_conflicts: tuple[EvidenceConflict, ...]
    evaluation_report: EvaluationReport
    counterfactual_explanations: tuple[CounterfactualExplanation, ...]
    method: str = "counterfactual_minimal_change_repair"
    metadata: dict[str, Any] = field(default_factory=dict)


def _stop_name(stop: dict[str, Any]) -> str:
    return str(stop.get("name") or stop.get("poi") or stop.get("stop_name") or "").strip()


def _day(stop: dict[str, Any]) -> int | None:
    for key in ("day", "day_index", "route_day"):
        if key in stop:
            try:
                return int(stop[key])
            except Exception:
                return None
    return None


def _float_value(stop: dict[str, Any], *keys: str, default: float = 0.0) -> float:
    for key in keys:
        if key in stop:
            try:
                value = float(stop.get(key) or 0.0)
                if math.isfinite(value):
                    return value
            except Exception:
                continue
    return default


def _route_payload(route: tuple[dict[str, Any], ...] | list[dict[str, Any]]) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for stop in route:
        payload.append(
            {
                "name": _stop_name(stop),
                "day": _day(stop),
                "city": str(stop.get("city", "")),
                "category": str(stop.get("category", "")),
                "overnight_city": str(stop.get("overnight_city", "")),
            }
        )
    return payload


def route_hash(route: tuple[dict[str, Any], ...] | list[dict[str, Any]]) -> str:
    """Return a stable hash for the user-visible route skeleton."""
    raw = json.dumps(_route_payload(route), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def parse_repair_intent(text: str, *, confirmed: bool = False, confidence: float = 0.55) -> ParsedRepairIntent:
    """Create a lightweight parser record for tests and manual workflows.

    This intentionally remains conservative; production LLM parsing should fill
    this dataclass and set ``confirmed=True`` only after validation/user review.
    """
    lowered = str(text).lower()
    must_keep = tuple(_extract_named_items(lowered, ("keep", "preserve", "must keep")))
    must_delete = tuple(_extract_named_items(lowered, ("remove", "delete", "drop", "avoid")))
    must_include = tuple(_extract_named_items(lowered, ("add", "include", "must include")))
    warnings: list[str] = []
    if not confirmed:
        warnings.append("parsed intent is not confirmed and must not control the solver")
    return ParsedRepairIntent(
        raw_text=str(text),
        confirmed=bool(confirmed),
        confidence=max(0.0, min(1.0, float(confidence))),
        must_keep=must_keep,
        must_delete=must_delete,
        must_include=must_include,
        warnings=tuple(warnings),
    )


def _extract_named_items(text: str, verbs: tuple[str, ...]) -> list[str]:
    items: list[str] = []
    for verb in verbs:
        pattern = rf"{re.escape(verb)}\s+([a-z0-9 '&.-]+?)(?:[,.;]|$|\band\b|\bbut\b)"
        for match in re.finditer(pattern, text):
            item = match.group(1).strip(" .,'\"")
            if item and len(item) > 2:
                items.append(item.title())
    return items


def detect_evidence_conflicts(
    evidence_records: tuple[dict[str, Any], ...] | list[dict[str, Any]],
    *,
    stale_after_hours: float = 24.0,
    low_confidence_threshold: float = 0.50,
) -> tuple[EvidenceConflict, ...]:
    """Detect stale, contradictory, low-confidence, and missing evidence."""
    conflicts: list[EvidenceConflict] = []
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in evidence_records:
        constraint = str(record.get("affected_constraint") or record.get("constraint") or "unknown")
        grouped.setdefault(constraint, []).append(record)
        source_id = str(record.get("source_id") or record.get("source") or "unknown")
        confidence = _bounded_confidence(record.get("confidence", 1.0))
        if not record.get("value") and not record.get("status"):
            conflicts.append(
                EvidenceConflict(
                    conflict_id=f"missing:{constraint}:{source_id}",
                    conflict_type="missing",
                    affected_constraint=constraint,
                    confidence=confidence,
                    resolution_policy="manual_review_or_fallback_to_conservative_constraint",
                    source_ids=(source_id,),
                    details="Evidence record lacks both value and status.",
                )
            )
        if confidence < low_confidence_threshold:
            conflicts.append(
                EvidenceConflict(
                    conflict_id=f"low_confidence:{constraint}:{source_id}",
                    conflict_type="low_confidence",
                    affected_constraint=constraint,
                    confidence=confidence,
                    resolution_policy="downweight_or_require_confirmation",
                    source_ids=(source_id,),
                    details=f"Source confidence {confidence:.2f} is below threshold {low_confidence_threshold:.2f}.",
                )
            )
        age_hours = _float_value(record, "age_hours", "freshness_hours", default=0.0)
        if bool(record.get("stale")) or age_hours > stale_after_hours:
            conflicts.append(
                EvidenceConflict(
                    conflict_id=f"stale:{constraint}:{source_id}",
                    conflict_type="stale",
                    affected_constraint=constraint,
                    confidence=confidence,
                    resolution_policy="prefer_fresher_high_confidence_source",
                    source_ids=(source_id,),
                    details=f"Evidence age {age_hours:.1f}h exceeds threshold {stale_after_hours:.1f}h.",
                )
            )

    for constraint, records in grouped.items():
        observed = {
            str(record.get("value", record.get("status", ""))).strip().lower()
            for record in records
            if str(record.get("value", record.get("status", ""))).strip()
        }
        if len(observed) > 1:
            source_ids = tuple(str(record.get("source_id") or record.get("source") or "unknown") for record in records)
            avg_confidence = sum(_bounded_confidence(record.get("confidence", 1.0)) for record in records) / len(records)
            conflicts.append(
                EvidenceConflict(
                    conflict_id=f"contradictory:{constraint}",
                    conflict_type="contradictory",
                    affected_constraint=constraint,
                    confidence=float(avg_confidence),
                    resolution_policy="surface_conditional_alternatives",
                    source_ids=source_ids,
                    details=f"Sources disagree: {sorted(observed)}.",
                )
            )
    return tuple(conflicts)


def _bounded_confidence(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except Exception:
        return 0.0


def evaluate_repair_plan(
    baseline_route: tuple[dict[str, Any], ...] | list[dict[str, Any]],
    repaired_route: tuple[dict[str, Any], ...] | list[dict[str, Any]],
    *,
    max_daily_travel_minutes: float = 480.0,
    max_stop_weather_risk: float = 0.70,
) -> EvaluationReport:
    """Evaluate a repaired route with hard gates before soft metrics."""
    baseline = tuple(baseline_route)
    repaired = tuple(repaired_route)
    baseline_names = [_stop_name(stop) for stop in baseline if _stop_name(stop)]
    repaired_names = [_stop_name(stop) for stop in repaired if _stop_name(stop)]
    preserved = len(set(baseline_names) & set(repaired_names))
    baseline_utility = sum(_float_value(stop, "utility", "final_poi_value", "score", default=0.0) for stop in baseline)
    repaired_utility = sum(_float_value(stop, "utility", "final_poi_value", "score", default=0.0) for stop in repaired)
    baseline_risk = sum(_float_value(stop, "weather_risk", default=0.0) for stop in baseline)
    repaired_risk = sum(_float_value(stop, "weather_risk", default=0.0) for stop in repaired)
    baseline_travel = sum(_float_value(stop, "travel_minutes", "travel_minutes_from_prev", default=0.0) for stop in baseline)
    repaired_travel = sum(_float_value(stop, "travel_minutes", "travel_minutes_from_prev", default=0.0) for stop in repaired)
    edit_distance = _levenshtein(baseline_names, repaired_names)

    hard_failures: list[str] = []
    for stop in repaired:
        name = _stop_name(stop) or "<unnamed stop>"
        if bool(stop.get("closed")) or str(stop.get("status", "")).lower() == "closed":
            hard_failures.append(f"{name} is closed")
        if _float_value(stop, "weather_risk", default=0.0) > max_stop_weather_risk:
            hard_failures.append(f"{name} exceeds weather-risk gate")
    for day, total in _daily_travel_minutes(repaired).items():
        if total > max_daily_travel_minutes:
            hard_failures.append(f"day {day} exceeds travel-time gate")

    metrics = {
        "preserved_stop_count": float(preserved),
        "preservation_rate": float(preserved / max(1, len(set(baseline_names)))),
        "route_edit_distance": float(edit_distance),
        "normalized_route_edit_distance": float(edit_distance / max(1, len(baseline_names))),
        "utility_retained": float(repaired_utility / baseline_utility) if baseline_utility > 0 else 1.0,
        "weather_risk_delta": float(baseline_risk - repaired_risk),
        "baseline_weather_risk": float(baseline_risk),
        "repaired_weather_risk": float(repaired_risk),
        "added_travel_minutes": float(repaired_travel - baseline_travel),
        "lodging_consistency": _lodging_consistency(baseline, repaired),
    }
    feasible = not hard_failures
    return EvaluationReport(
        route_hash=route_hash(repaired),
        feasible=feasible,
        hard_failures=tuple(hard_failures),
        metrics=metrics,
        eligibility="eligible" if feasible else "hard_infeasible",
    )


def _daily_travel_minutes(route: tuple[dict[str, Any], ...]) -> dict[int, float]:
    totals: dict[int, float] = {}
    for stop in route:
        day = _day(stop)
        if day is None:
            continue
        totals[day] = totals.get(day, 0.0) + _float_value(stop, "travel_minutes", "travel_minutes_from_prev", default=0.0)
    return totals


def _lodging_consistency(
    baseline: tuple[dict[str, Any], ...],
    repaired: tuple[dict[str, Any], ...],
) -> float:
    baseline_by_day = {
        _day(stop): str(stop.get("overnight_city", ""))
        for stop in baseline
        if _day(stop) is not None and str(stop.get("overnight_city", ""))
    }
    repaired_by_day = {
        _day(stop): str(stop.get("overnight_city", ""))
        for stop in repaired
        if _day(stop) is not None and str(stop.get("overnight_city", ""))
    }
    if not baseline_by_day:
        return 1.0
    matches = sum(1 for day, city in baseline_by_day.items() if repaired_by_day.get(day) == city)
    return float(matches / len(baseline_by_day))


def _levenshtein(left: list[str], right: list[str]) -> int:
    previous = list(range(len(right) + 1))
    for i, left_item in enumerate(left, start=1):
        current = [i]
        for j, right_item in enumerate(right, start=1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (0 if left_item == right_item else 1),
                )
            )
        previous = current
    return int(previous[-1])


def build_repair_plan(
    request: RepairRequest,
    *,
    max_stop_weather_risk: float | None = None,
    max_daily_travel_minutes: float | None = None,
) -> RepairPlan:
    """Build one conservative minimal-change repair plan."""
    intent = request.parsed_intent or parse_repair_intent(request.user_intent, confirmed=False)
    parent_hash = route_hash(request.baseline_route)
    conflicts = detect_evidence_conflicts(request.evidence_records)
    if not intent.confirmed:
        repaired = tuple(dict(stop) for stop in request.baseline_route)
        evaluation = evaluate_repair_plan(repaired, repaired)
        explanation = CounterfactualExplanation(
            target="repair request",
            constraint="confirmed_intent",
            minimal_required_change="Confirm parsed repair intent before compiling constraints.",
            rationale="The LLM/parser layer is intentionally separated from the solver boundary.",
        )
        return RepairPlan(
            plan_id=f"{request.request_id}:unconfirmed",
            parent_route_hash=parent_hash,
            repaired_route=repaired,
            operations=(),
            evidence_conflicts=conflicts,
            evaluation_report=evaluation,
            counterfactual_explanations=(explanation,),
            metadata={"status": "blocked_unconfirmed_intent"},
        )

    threshold = float(
        max_stop_weather_risk
        if max_stop_weather_risk is not None
        else request.tolerance_profile.get("max_stop_weather_risk", 0.55)
    )
    max_travel = float(
        max_daily_travel_minutes
        if max_daily_travel_minutes is not None
        else request.tolerance_profile.get("max_daily_travel_minutes", 480.0)
    )
    candidate_pool = [dict(candidate) for candidate in request.candidate_pois]
    used_names = {_stop_name(stop) for stop in request.baseline_route}
    repaired: list[dict[str, Any]] = []
    operations: list[RepairOperation] = []
    counterfactuals: list[CounterfactualExplanation] = []
    must_keep = {name.lower() for name in intent.must_keep}
    must_delete = {name.lower() for name in intent.must_delete}
    allowed_operations = set(intent.operations_allowed)
    move_targets = _confirmed_move_targets(request.confirmed_constraints)

    for stop in request.baseline_route:
        stop_copy = dict(stop)
        name = _stop_name(stop_copy)
        lowered = name.lower()
        current_day = _day(stop_copy)
        risk = _float_value(stop_copy, "weather_risk", default=0.0)
        closed = bool(stop_copy.get("closed")) or str(stop_copy.get("status", "")).lower() == "closed"
        should_delete = lowered in must_delete or (closed and lowered not in must_keep)
        should_replace = risk > threshold and lowered not in must_keep

        if should_delete:
            if "DELETE" in allowed_operations:
                operations.append(
                    RepairOperation(
                        operation="DELETE",
                        target_stop=name,
                        day=current_day,
                        affected_constraint="user_intent" if lowered in must_delete else "closure",
                        reason="Stop removed by confirmed intent or hard closure evidence.",
                    )
                )
                counterfactuals.append(
                    CounterfactualExplanation(
                        target=name,
                        constraint="deletion",
                        minimal_required_change="Remove the delete/avoid intent or resolve the closure evidence.",
                        affected_metrics={"preserved_stop_count": -1.0},
                        rationale="The repaired plan deletes only the affected stop and preserves the surrounding route when possible.",
                    )
                )
                used_names.discard(name)
                continue
            counterfactuals.append(
                CounterfactualExplanation(
                    target=name,
                    constraint="operation_scope",
                    minimal_required_change="Allow DELETE operations to remove this infeasible stop.",
                    rationale="The confirmed repair scope does not permit deletion.",
                )
            )

        requested_day = move_targets.get(lowered)
        if requested_day is not None and current_day is not None and requested_day != current_day:
            if "MOVE" in allowed_operations:
                stop_copy["day"] = requested_day
                operations.append(
                    RepairOperation(
                        operation="MOVE",
                        target_stop=name,
                        day=requested_day,
                        affected_constraint="temporal_window",
                        reason="Confirmed constraints move this stop to a different day.",
                        metrics_delta={"day_delta": float(requested_day - current_day)},
                    )
                )
            else:
                counterfactuals.append(
                    CounterfactualExplanation(
                        target=name,
                        constraint="operation_scope",
                        minimal_required_change="Allow MOVE operations to shift this stop to the requested day.",
                        rationale="The confirmed repair scope does not permit temporal edits.",
                    )
                )

        if should_replace:
            relax_reason = "No lower-risk replacement was available; keeping stop requires relaxing weather-risk threshold."
            relax_change = f"Provide a candidate under risk {threshold:.2f} or relax the threshold to {risk:.2f}."
            relax_rationale = "The solver cannot replace this stop without candidate coverage."
            replacement = _best_replacement(stop_copy, candidate_pool, used_names, threshold)
            if replacement is not None:
                replacement_name = _stop_name(replacement)
                replacement["day"] = replacement.get("day", _day(stop_copy))
                replacement.setdefault("overnight_city", stop_copy.get("overnight_city", ""))
                if "REPLACE" in allowed_operations:
                    repaired.append(replacement)
                    used_names.add(replacement_name)
                    operations.append(
                        RepairOperation(
                            operation="REPLACE",
                            target_stop=name,
                            replacement_stop=replacement_name,
                            day=_day(stop_copy),
                            affected_constraint="weather_risk",
                            reason="High-risk stop replaced with lower-risk candidate while preserving route day.",
                            metrics_delta={
                                "weather_risk_delta": risk - _float_value(replacement, "weather_risk", default=0.0),
                                "utility_delta": _float_value(replacement, "utility", "final_poi_value", default=0.0)
                                - _float_value(stop_copy, "utility", "final_poi_value", default=0.0),
                            },
                        )
                    )
                    counterfactuals.append(
                        CounterfactualExplanation(
                            target=name,
                            constraint="weather_risk",
                            minimal_required_change=f"Raise max_stop_weather_risk to at least {risk:.2f} to keep this stop.",
                            substitution=replacement_name,
                            affected_metrics={
                                "weather_risk_delta": risk - _float_value(replacement, "weather_risk", default=0.0)
                            },
                            rationale="This exposes the binding risk threshold behind the replacement.",
                        )
                    )
                    continue
                counterfactuals.append(
                    CounterfactualExplanation(
                        target=name,
                        constraint="operation_scope",
                        minimal_required_change="Allow REPLACE operations to use the lower-risk substitute.",
                        substitution=replacement_name,
                        rationale="A feasible replacement exists but the confirmed repair scope excludes substitution.",
                    )
                )
                relax_reason = (
                    "A lower-risk replacement exists, but confirmed repair scope excludes replacement; keeping stop "
                    "requires relaxing the weather-risk threshold."
                )
                relax_change = f"Allow REPLACE with {replacement_name} or relax the threshold to {risk:.2f}."
                relax_rationale = "The planner preserves the stop because substitution is outside the confirmed scope."
            if "RELAX" in allowed_operations:
                operations.append(
                    RepairOperation(
                        operation="RELAX",
                        target_stop=name,
                        day=_day(stop_copy),
                        affected_constraint="weather_risk",
                        reason=relax_reason,
                    )
                )
                counterfactuals.append(
                    CounterfactualExplanation(
                        target=name,
                        constraint="weather_risk",
                        minimal_required_change=relax_change,
                        rationale=relax_rationale,
                    )
                )

        repaired.append(stop_copy)
        operations.append(
            RepairOperation(
                operation="KEEP",
                target_stop=name,
                day=_day(stop_copy),
                affected_constraint="preservation",
                reason="Stop preserved to minimize itinerary change.",
            )
        )

    for requested_name in intent.must_include:
        if requested_name.lower() in {name.lower() for name in used_names}:
            continue
        addition = _find_candidate_by_name(requested_name, candidate_pool, used_names)
        if "ADD" not in allowed_operations:
            counterfactuals.append(
                CounterfactualExplanation(
                    target=requested_name,
                    constraint="operation_scope",
                    minimal_required_change="Allow ADD operations to insert this requested stop.",
                    rationale="The confirmed repair scope does not permit additions.",
                )
            )
            continue
        if addition is None:
            counterfactuals.append(
                CounterfactualExplanation(
                    target=requested_name,
                    constraint="candidate_coverage",
                    minimal_required_change="Add a grounded candidate record before this POI can be included.",
                    rationale="The repair planner does not hallucinate missing POIs.",
                )
            )
            continue
        repaired.append(addition)
        used_names.add(_stop_name(addition))
        operations.append(
            RepairOperation(
                operation="ADD",
                target_stop=_stop_name(addition),
                day=_day(addition),
                affected_constraint="user_intent",
                reason="Confirmed user intent requested this additional stop.",
            )
        )

    evaluation = evaluate_repair_plan(
        request.baseline_route,
        tuple(repaired),
        max_daily_travel_minutes=max_travel,
        max_stop_weather_risk=max(threshold, request.tolerance_profile.get("hard_weather_gate", threshold)),
    )
    return RepairPlan(
        plan_id=f"{request.request_id}:repair",
        parent_route_hash=parent_hash,
        repaired_route=tuple(repaired),
        operations=tuple(operations),
        evidence_conflicts=conflicts,
        evaluation_report=evaluation,
        counterfactual_explanations=tuple(counterfactuals),
        metadata={
            "max_stop_weather_risk": threshold,
            "max_daily_travel_minutes": max_travel,
            "confirmed_parser": intent.parser_name,
        },
    )


def generate_repair_alternatives(request: RepairRequest) -> tuple[RepairPlan, ...]:
    """Return a small deterministic frontier for evaluation baselines."""
    base_threshold = float(request.tolerance_profile.get("max_stop_weather_risk", 0.55))
    conservative = build_repair_plan(request, max_stop_weather_risk=max(0.05, base_threshold - 0.15))
    balanced = build_repair_plan(request, max_stop_weather_risk=base_threshold)
    preservation_first = build_repair_plan(request, max_stop_weather_risk=min(1.0, base_threshold + 0.15))
    return tuple(_with_plan_id(plan, suffix) for plan, suffix in [
        (conservative, "risk_averse"),
        (balanced, "balanced"),
        (preservation_first, "preservation_first"),
    ])


def _with_plan_id(plan: RepairPlan, suffix: str) -> RepairPlan:
    return RepairPlan(
        plan_id=f"{plan.plan_id}:{suffix}",
        parent_route_hash=plan.parent_route_hash,
        repaired_route=plan.repaired_route,
        operations=plan.operations,
        evidence_conflicts=plan.evidence_conflicts,
        evaluation_report=plan.evaluation_report,
        counterfactual_explanations=plan.counterfactual_explanations,
        method=plan.method,
        metadata={**plan.metadata, "frontier_policy": suffix},
    )


def _confirmed_move_targets(confirmed_constraints: dict[str, Any]) -> dict[str, int]:
    targets: dict[str, int] = {}
    for key in ("move_stops", "stop_day_overrides", "target_days"):
        raw_targets = confirmed_constraints.get(key, {})
        if not isinstance(raw_targets, dict):
            continue
        for stop_name, target_day in raw_targets.items():
            name = str(stop_name).strip().lower()
            if not name:
                continue
            try:
                targets[name] = int(target_day)
            except Exception:
                continue
    return targets


def _best_replacement(
    stop: dict[str, Any],
    candidates: list[dict[str, Any]],
    used_names: set[str],
    max_weather_risk: float,
) -> dict[str, Any] | None:
    stop_category = str(stop.get("category", "")).lower()
    stop_day = _day(stop)
    ranked: list[tuple[float, dict[str, Any]]] = []
    for candidate in candidates:
        name = _stop_name(candidate)
        if not name or name in used_names:
            continue
        if bool(candidate.get("closed")) or str(candidate.get("status", "")).lower() == "closed":
            continue
        risk = _float_value(candidate, "weather_risk", default=0.0)
        if risk > max_weather_risk:
            continue
        category = str(candidate.get("category", "")).lower()
        category_match = 1.0 if stop_category and stop_category in category else 0.0
        if stop_day is not None and _day(candidate) not in (None, stop_day):
            category_match -= 0.25
        score = (
            2.0 * category_match
            + _float_value(candidate, "utility", "final_poi_value", "score", default=0.0)
            - risk
            - 0.002 * _float_value(candidate, "travel_minutes", "travel_minutes_from_prev", default=0.0)
        )
        ranked.append((score, candidate))
    if not ranked:
        return None
    return dict(max(ranked, key=lambda item: item[0])[1])


def _find_candidate_by_name(
    requested_name: str,
    candidates: list[dict[str, Any]],
    used_names: set[str],
) -> dict[str, Any] | None:
    target = requested_name.lower()
    for candidate in candidates:
        name = _stop_name(candidate)
        if name and name.lower() == target and name not in used_names:
            return dict(candidate)
    for candidate in candidates:
        name = _stop_name(candidate)
        if name and target in name.lower() and name not in used_names:
            return dict(candidate)
    return None
