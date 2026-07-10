"""Metric extraction helpers for BENCH-002 benchmark rows."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .disruptions import DisruptionScenario


def extract_benchmark_metrics(
    result: Any,
    *,
    scenario: DisruptionScenario,
    runtime_seconds: float | None = None,
) -> dict[str, Any]:
    """Extract benchmark metric columns from a method result."""

    metrics = _mapping(_value(result, "metrics"))
    diff_records = _records(_value(result, "diff_records"))
    evaluations = _records(_value(result, "evaluations"))
    explanations = _records(_value(result, "explanation_records"))

    diff = diff_records[0] if diff_records else {}
    evaluation = evaluations[0] if evaluations else {}
    evaluation_metrics = _mapping(evaluation.get("metrics"))
    unchanged_days = diff.get("unchanged_days") or metrics.get("unchanged_days") or ()
    route_validation = _mapping(evaluation.get("route_validation"))

    return {
        "preservation_weighted_edit_cost": _number(
            metrics.get("weighted_edit_cost"),
            diff.get("weighted_edit_cost"),
        ),
        "preservation_unchanged_day_count": len(tuple(unchanged_days)),
        "preservation_affected_day_count": len(scenario.affected_days),
        "quality_utility_retained": _number(
            metrics.get("utility_retained"),
            evaluation_metrics.get("utility_retained"),
            metrics.get("utility"),
            evaluation_metrics.get("utility"),
        ),
        "quality_utility_regret": _number(
            metrics.get("utility_regret"),
            evaluation_metrics.get("utility_regret"),
        ),
        "quality_weather_risk_delta": _number(
            metrics.get("weather_risk_delta"),
            evaluation_metrics.get("weather_risk_delta"),
        ),
        "quality_route_validated": _bool_or_none(route_validation.get("road_validated")),
        "computation_runtime_seconds": _number(metrics.get("runtime_seconds"), runtime_seconds),
        "computation_repair_attempt_count": _number(metrics.get("repair_attempt_count")),
        "computation_fallback_used": _bool_or_none(metrics.get("fallback_used")),
        "certificate_id": str(evaluation.get("certificate_id") or evaluation.get("evaluation_id") or ""),
        "certificate_comparison_eligibility": str(evaluation.get("comparison_eligibility") or "unknown"),
        "certificate_evaluation_status": str(evaluation.get("evaluation_status") or "unknown"),
        "explanation_count": len(explanations),
        "explanation_ids": [
            str(record.get("evidence_id") or record.get("explanation_id") or "") for record in explanations
        ],
    }


def _records(value: Any) -> tuple[dict[str, Any], ...]:
    if value is None:
        return ()
    if isinstance(value, Mapping):
        return (_mapping(value),)
    return tuple(_mapping(item) for item in value)


def _mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "to_record"):
        return dict(value.to_record())
    if hasattr(value, "__dict__"):
        return dict(vars(value))
    return {}


def _value(result: Any, key: str) -> Any:
    if isinstance(result, Mapping):
        return result.get(key)
    return getattr(result, key, None)


def _number(*values: Any) -> float | int | None:
    for value in values:
        if value is None or value == "":
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if number.is_integer():
            return int(number)
        return number
    return None


def _bool_or_none(value: Any) -> bool | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "passed"}:
            return True
        if lowered in {"false", "0", "no", "failed"}:
            return False
    return bool(value)
