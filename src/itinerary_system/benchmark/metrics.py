"""Metric extraction helpers for BENCH-002 benchmark rows."""

from __future__ import annotations

from collections.abc import Mapping
from math import isfinite
from typing import Any

from ..research_artifacts import stable_content_hash
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
    route_records = _records(_value(result, "route_records"))
    route_record = route_records[0] if route_records else {}

    diff = diff_records[0] if diff_records else {}
    evaluation = evaluations[0] if evaluations else {}
    evaluation_metrics = _mapping(evaluation.get("metrics"))
    unchanged_days = diff.get("unchanged_days") or ()
    route_validation = _mapping(evaluation.get("route_validation"))
    route_road_validated = _bool_or_none(
        route_validation.get("road_validated")
        if "road_validated" in route_validation
        else route_validation.get("publication_ready")
        if "publication_ready" in route_validation
        else evaluation_metrics.get("route_road_validated")
    )
    comparison_eligibility = str(evaluation.get("comparison_eligibility") or "unknown")
    evaluation_status = str(evaluation.get("evaluation_status") or "unknown")
    quality_utility_retained = _number(evaluation_metrics.get("utility_retained"))
    quality_utility_regret = _number(evaluation_metrics.get("utility_regret"))
    quality_weather_risk_delta = _number(evaluation_metrics.get("weather_risk_delta"))
    quality_metrics_present = any(
        value is not None
        for value in (quality_utility_retained, quality_utility_regret, quality_weather_risk_delta)
    )
    ranking_eligible = (
        comparison_eligibility == "eligible"
        and evaluation_status.upper() in {"PASSED", "PASSED_WITH_WARNINGS"}
        and route_road_validated is True
        and quality_utility_retained is not None
    )

    return {
        "preservation_weighted_edit_cost": (
            _number(diff.get("weighted_edit_cost")) if diff_records else None
        ),
        "preservation_metric_owner": "plan_diff" if diff_records else None,
        "preservation_unchanged_day_count": len(tuple(unchanged_days)),
        "preservation_affected_day_count": len(scenario.affected_days),
        "quality_utility_retained": quality_utility_retained,
        "quality_utility_regret": quality_utility_regret,
        "quality_weather_risk_delta": quality_weather_risk_delta,
        "quality_route_validated": route_road_validated,
        "quality_metric_owner": "independent_evaluator",
        "quality_metrics_present": quality_metrics_present,
        "benchmark_ranking_eligible": ranking_eligible,
        "benchmark_route_matrix_id": str(route_record.get("matrix_id") or ""),
        "benchmark_route_matrix_hash": stable_content_hash(route_record) if route_record else "",
        "benchmark_route_source_bundle_id": str(route_record.get("source_bundle_id") or ""),
        "benchmark_route_source_content_sha256": str(route_record.get("source_content_sha256") or ""),
        "computation_runtime_seconds": _number(metrics.get("runtime_seconds"), runtime_seconds),
        "computation_repair_attempt_count": _number(metrics.get("repair_attempt_count")),
        "computation_fallback_used": _bool_or_none(metrics.get("fallback_used")),
        "certificate_id": str(evaluation.get("certificate_id") or evaluation.get("evaluation_id") or ""),
        "certificate_comparison_eligibility": comparison_eligibility,
        "certificate_evaluation_status": evaluation_status,
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
        if not isfinite(number):
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
