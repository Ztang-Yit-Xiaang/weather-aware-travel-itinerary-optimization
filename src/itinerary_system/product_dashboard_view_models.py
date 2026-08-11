"""Build customer and research product views from validated artifacts."""

from __future__ import annotations

import math
from typing import Any

from .product_dashboard_adapter import _is_exact_incomplete
from .product_dashboard_models import (
    PRODUCT_DASHBOARD_DATA_SCHEMA,
    PRODUCT_DASHBOARD_VERSION,
    TRUTH_STATE_LABELS,
    ProductDashboardSourceBundle,
    assert_finite_json,
)

METHOD_LABELS = {
    "context_blind_solver": "Context-blind exact repair",
    "deterministic_context_aware_heuristic": "Deterministic context-aware repair",
    "progressive_sequential_lexicographic_repair": "Progressive ownership-aware repair",
    "full_reoptimization": "Full exact reoptimization",
    "lexicographic_repair": "Lexicographic repair attempt",
}


def build_product_dashboard_view_model(bundle: ProductDashboardSourceBundle) -> dict[str, Any]:
    """Return one shared, finite read-only model for both presentation modes."""

    parent = bundle.parent_plan
    child = bundle.child_plan
    diff = bundle.diff or {}
    certificate = bundle.certificate or {}
    request = bundle.request
    affected_days = _affected_days(request, diff)
    selected_day = affected_days[0] if affected_days else _plan_days(child or parent)[0]
    timeline = _timeline(parent, child, diff, request)
    comparison = _comparison_metrics(parent, child, diff, certificate, bundle.metrics)
    active_states = list(bundle.truth_states)
    if any(metric["child"]["state"] == "unavailable" for metric in comparison):
        active_states.append("unavailable_metric")
    if any(metric["child"]["value"] is None for metric in comparison):
        active_states.append("null_metric")

    model = {
        "schema_version": PRODUCT_DASHBOARD_DATA_SCHEMA,
        "product_version": PRODUCT_DASHBOARD_VERSION,
        "run": {
            "run_id": str(bundle.run_manifest.get("run_id") or ""),
            "status": str(bundle.run_manifest.get("status") or "unknown"),
            "mode": str(bundle.run_manifest.get("mode") or ""),
            "catalog_snapshot_id": str(bundle.run_manifest.get("catalog_snapshot_id") or ""),
            "context_snapshot_id": str(bundle.run_manifest.get("context_snapshot_id") or ""),
            "interaction_enabled": False,
        },
        "trip": {
            "title": "California itinerary repair review",
            "dates": None,
            "dates_state": "unavailable",
            "day_count": len(timeline),
            "stop_count": len((child or parent).get("selected_stops") or []),
            "selected_day": selected_day,
            "affected_days": affected_days,
        },
        "issue": _issue_summary(request, parent),
        "timeline": timeline,
        "repair": _repair_summary(bundle, affected_days),
        "comparison": comparison,
        "evidence": _evidence_view(bundle.explanations),
        "alternatives": _alternative_rows(bundle),
        "map": _map_view(bundle, selected_day),
        "truth_states": [truth_state_view(state) for state in dict.fromkeys(active_states)],
        "truth_state_catalog": [
            truth_state_view(state) for state in TRUTH_STATE_LABELS
        ],
        "research": _research_view(bundle),
        "map_alternative": _map_alternative_text(timeline, affected_days),
        "interaction": {
            "enabled": False,
            "state": "interaction_mode_disabled",
            "message": (
                "This dashboard is read-only. Repair acceptance, permission, and "
                "preference persistence remain disabled."
            ),
        },
    }
    assert_finite_json(model)
    return model


def truth_state_view(state: str) -> dict[str, str]:
    """Return the stable presentation contract for one truth state."""

    label, tone = TRUTH_STATE_LABELS[state]
    return {"id": state, "label": label, "tone": tone}


def _issue_summary(request: dict[str, Any], parent: dict[str, Any]) -> dict[str, Any]:
    constraints = request.get("confirmed_constraints") or {}
    target_ids = [str(value) for value in constraints.get("target_stop_ids") or []]
    stop_names = _stop_names(parent)
    targets = [stop_names.get(stop_id, _titleize(stop_id)) for stop_id in target_ids]
    family = str(constraints.get("disruption_family") or request.get("kind") or "event")
    observed = bool(constraints.get("observed"))
    return {
        "family": family,
        "label": _titleize(family),
        "summary": str(request.get("user_intent") or f"Review {_titleize(family).lower()}."),
        "affected_days": [int(day) for day in constraints.get("affected_days") or []],
        "targets": targets,
        "source_status": "Observed evidence" if observed else "Synthetic benchmark scenario",
        "synthetic": bool(constraints.get("synthetic", not observed)),
        "evidence_ids": [
            str(record.get("evidence_id") or "")
            for record in request.get("evidence_records") or []
            if record.get("evidence_id")
        ],
    }


def _timeline(
    parent: dict[str, Any],
    child: dict[str, Any] | None,
    diff: dict[str, Any],
    request: dict[str, Any],
) -> list[dict[str, Any]]:
    parent_days = _stops_by_day(parent)
    child_days = _stops_by_day(child or parent)
    affected = set(_affected_days(request, diff))
    changed = set(affected)
    changed.update(int(item.get("day") or 0) for item in _all_diff_changes(diff))
    unchanged = {int(day) for day in diff.get("unchanged_days") or []}
    ownership = _ownership_by_target(child or parent)
    days = sorted(set(parent_days) | set(child_days))
    result = []
    for day in days:
        parent_stops = parent_days.get(day, [])
        child_stops = child_days.get(day, [])
        states = []
        if day in affected:
            states.append("affected")
        if day in changed and day not in unchanged:
            states.append("changed")
        if day in unchanged or parent_stops == child_stops:
            states.append("unchanged")
        stop_views = []
        for stop in child_stops:
            stop_id = _stop_id(stop)
            strength = ownership.get(stop_id, "")
            stop_states = [strength] if strength in {"booked", "locked", "preferred", "flexible"} else []
            stop_views.append(
                {
                    "id": stop_id,
                    "name": _stop_name(stop),
                    "city": str(stop.get("city") or "City unavailable"),
                    "order": int(stop.get("stop_order") or len(stop_views) + 1),
                    "states": stop_states,
                    "changed": stop_id not in {_stop_id(item) for item in parent_stops},
                }
            )
        result.append(
            {
                "day": day,
                "states": list(dict.fromkeys(states)),
                "stops": stop_views,
                "parent_stop_names": [_stop_name(stop) for stop in parent_stops],
                "child_stop_names": [_stop_name(stop) for stop in child_stops],
                "route_id": str((child or parent).get("route_ids_by_day", {}).get(str(day), "")),
                "lodging": (child or parent).get("lodging_assignments", {}).get(str(day)),
            }
        )
    return result


def _repair_summary(
    bundle: ProductDashboardSourceBundle, affected_days: list[int]
) -> dict[str, Any]:
    diff = bundle.diff or {}
    certificate = bundle.certificate or {}
    changes = _all_diff_changes(diff)
    added = [_titleize(str(item.get("stop_id") or "")) for item in diff.get("added_stops") or []]
    removed = [_titleize(str(item.get("stop_id") or "")) for item in diff.get("deleted_stops") or []]
    booked = sum(str(item.get("owner_strength") or "") == "booked" for item in changes)
    locked = sum(str(item.get("owner_strength") or "") == "locked" for item in changes)
    eligible = bool(certificate.get("eligible"))
    return {
        "status": "Eligible repair" if eligible else "Repair not independently eligible",
        "status_state": "eligible_repair" if eligible else "ineligible_repair",
        "result": (
            f"Replaced {', '.join(removed)} with {', '.join(added)}."
            if added or removed
            else "The artifact records no stop replacement."
        ),
        "changed": {
            "added": added,
            "removed": removed,
            "affected_day_count": len(affected_days),
            "road_change_count": len(diff.get("road_changes") or []),
            "lodging_change_count": len(diff.get("lodging_changes") or []),
        },
        "unchanged": {
            "days": [int(day) for day in diff.get("unchanged_days") or []],
            "day_count": len(diff.get("unchanged_days") or []),
        },
        "permissions": {
            "booked_change_count": booked,
            "locked_change_count": locked,
            "permission_required": booked > 0,
            "message": (
                "Permission is required for a booked change."
                if booked
                else "No booked or locked change is recorded in this diff."
            ),
        },
        "tradeoffs": {
            "weighted_edit_cost": diff.get("weighted_edit_cost"),
            "preservation_rate": (certificate.get("metrics") or {}).get("preservation_rate"),
            "utility_retained": (certificate.get("metrics") or {}).get("utility_retained"),
            "weather_risk_delta": (certificate.get("metrics") or {}).get("weather_risk_delta"),
        },
        "accepted_radius": bundle.metrics.get("accepted_radius"),
        "certificate": {
            "id": certificate.get("certificate_id"),
            "eligible": certificate.get("eligible"),
            "evaluation_status": certificate.get("evaluation_status"),
            "failure_count": certificate.get("failure_count"),
            "failures": certificate.get("failures") or [],
        },
        "primary_action": "Review evidence",
    }


def _comparison_metrics(
    parent: dict[str, Any],
    child: dict[str, Any] | None,
    diff: dict[str, Any],
    certificate: dict[str, Any],
    run_metrics: dict[str, Any],
) -> list[dict[str, Any]]:
    cert_metrics = certificate.get("metrics") or {}
    route = certificate.get("route_validation") or {}
    total_days = max(1, len(_plan_days(parent)))
    unchanged_days = len(diff.get("unchanged_days") or [])
    changes = _all_diff_changes(diff)
    booked = sum(str(item.get("owner_strength") or "") == "booked" for item in changes)
    locked = sum(str(item.get("owner_strength") or "") == "locked" for item in changes)
    parent_travel = (parent.get("modeled_metrics") or {}).get("total_travel_time")
    child_exists = child is not None
    return [
        _metric(
            "hard_eligibility",
            "Independent eligibility",
            None,
            certificate.get("comparison_eligibility") if certificate else None,
            direction="required",
            owner="independent_evaluator",
            parent_note="The accepted parent was not reevaluated in this repair run.",
        ),
        _metric(
            "route_validity",
            "Road-validated route legs",
            None,
            (
                f"{route.get('road_validated_leg_count')}/{route.get('required_leg_count')}"
                if route
                else None
            ),
            direction="required",
            owner="independent_evaluator",
        ),
        _metric(
            "contextual_burden",
            "Weather-risk reduction",
            None,
            cert_metrics.get("weather_risk_delta"),
            direction="higher",
            owner="independent_evaluator",
        ),
        _metric(
            "travel_time",
            "Recorded travel time",
            parent_travel,
            (child or {}).get("modeled_metrics", {}).get("total_travel_time") if child_exists else None,
            unit="minutes",
            direction="lower",
            owner="plan_artifact",
            child_note="The child artifact does not record a planner travel-time metric.",
        ),
        _metric(
            "weighted_edit_cost",
            "Ownership-weighted edit cost",
            0.0,
            diff.get("weighted_edit_cost") if diff else None,
            direction="lower",
            owner="plan_diff",
            parent_note="Reference baseline, not an evaluator score.",
        ),
        _metric(
            "affected_days",
            "Affected days",
            0,
            total_days - unchanged_days if diff else None,
            direction="lower",
            owner="plan_diff",
        ),
        _metric(
            "unchanged_ratio",
            "Unchanged-day ratio",
            1.0,
            unchanged_days / total_days if diff else None,
            direction="higher",
            owner="plan_diff",
        ),
        _metric(
            "booked_changes",
            "Booked changes",
            0,
            booked if diff else None,
            direction="lower",
            owner="plan_diff",
        ),
        _metric(
            "locked_changes",
            "Locked changes",
            0,
            locked if diff else None,
            direction="lower",
            owner="plan_diff",
        ),
        _metric(
            "utility_retained",
            "Utility retained",
            1.0,
            cert_metrics.get("utility_retained"),
            direction="higher",
            owner="independent_evaluator",
            parent_note="Reference baseline of the accepted parent.",
        ),
        _metric(
            "repair_radius",
            "Accepted repair radius",
            None,
            run_metrics.get("accepted_radius"),
            direction="descriptive",
            owner="repair_outcome",
        ),
        _metric(
            "method_status",
            "Method status",
            "Accepted parent",
            "Completed" if child_exists else None,
            direction="descriptive",
            owner="planner_run",
        ),
        _metric(
            "failure_status",
            "Failure status",
            "None recorded",
            (
                "None recorded"
                if certificate and not certificate.get("failures")
                else certificate.get("failures")
                if certificate
                else None
            ),
            direction="descriptive",
            owner="independent_evaluator",
        ),
    ]


def _metric(
    metric_id: str,
    label: str,
    parent: Any,
    child: Any,
    *,
    unit: str = "",
    direction: str,
    owner: str,
    parent_note: str = "",
    child_note: str = "",
) -> dict[str, Any]:
    return {
        "id": metric_id,
        "label": label,
        "unit": unit,
        "direction": direction,
        "owner": owner,
        "parent": {
            "value": parent,
            "state": "available" if parent is not None else "unavailable",
            "note": parent_note,
        },
        "child": {
            "value": child,
            "state": "available" if child is not None else "unavailable",
            "note": child_note,
        },
    }


def _evidence_view(explanations: tuple[dict[str, Any], ...]) -> dict[str, Any]:
    claims = []
    records = []
    for explanation in explanations:
        for claim in explanation.get("claims") or []:
            claims.append(
                {
                    "id": str(claim.get("claim_id") or ""),
                    "text": str(claim.get("rendered_text") or ""),
                    "type": str(claim.get("claim_type") or ""),
                    "confidence": str(claim.get("confidence") or ""),
                    "evidence_refs": [str(ref) for ref in claim.get("evidence_refs") or []],
                    "supported": not bool(claim.get("unsupported")),
                }
            )
        for record in explanation.get("evidence_records") or []:
            records.append(
                {
                    "ref_id": str(record.get("ref_id") or ""),
                    "source_type": str(record.get("source_type") or ""),
                    "source_id": str(record.get("source_id") or ""),
                    "field_path": str(record.get("field_path") or ""),
                    "record_hash": str(record.get("record_hash") or ""),
                }
            )
    return {"claims": claims, "records": records}


def _alternative_rows(bundle: ProductDashboardSourceBundle) -> list[dict[str, Any]]:
    scenario_id = str(bundle.request.get("request_id") or bundle.request.get("repair_request_id") or "")
    rows = [row for row in bundle.benchmark_rows if str(row.get("scenario_id") or "") == scenario_id]
    diagnostics = {
        str(run.get("method_executed") or run.get("method_requested") or ""): run
        for run in bundle.benchmark_planner_runs
        if str(run.get("planning_request_id") or "") == scenario_id
    }
    result = []
    for row in rows:
        method_id = str(row.get("method_id") or "")
        status = str(row.get("status") or "unknown")
        diagnostic = diagnostics.get(method_id, {})
        failure = str(
            row.get("error_summary")
            or row.get("error_class")
            or diagnostic.get("error_summary")
            or diagnostic.get("error_class")
            or ""
        )
        exact_incomplete = _is_exact_incomplete({**row, **diagnostic})
        if exact_incomplete:
            display_status = "Exact search incomplete"
        elif status == "failed":
            display_status = "Failed"
        elif bool(row.get("benchmark_ranking_eligible")):
            display_status = "Eligible"
        else:
            display_status = "Completed, not ranking-eligible"
        result.append(
            {
                "method_id": method_id,
                "method_label": METHOD_LABELS.get(method_id, _titleize(method_id)),
                "status": status,
                "display_status": display_status,
                "ranking_eligible": bool(row.get("benchmark_ranking_eligible")),
                "failure_reason": failure or None,
                "exact_search_incomplete": exact_incomplete,
                "requested_method_id": str(row.get("benchmark_expected_method_id") or ""),
                "executed_method_ids": [
                    str(value) for value in row.get("benchmark_planner_method_executed_ids") or []
                ],
                "runtime_seconds": row.get("computation_runtime_seconds"),
                "weighted_edit_cost": row.get("preservation_weighted_edit_cost"),
                "utility_retained": row.get("quality_utility_retained"),
                "route_validated": row.get("quality_route_validated"),
            }
        )
    return result


def _map_view(bundle: ProductDashboardSourceBundle, selected_day: int) -> dict[str, Any]:
    parent = _map_route(bundle.parent_plan, bundle.route_matrix)
    child = _map_route(bundle.child_plan, bundle.route_matrix) if bundle.child_plan else None
    issue = _issue_summary(bundle.request, bundle.parent_plan)
    return {
        "selected_day": selected_day,
        "affected_days": _affected_days(bundle.request, bundle.diff or {}),
        "evidence": {
            "label": issue["label"],
            "targets": issue["targets"],
            "source_status": issue["source_status"],
        },
        "parent": parent,
        "child": child,
        "legend": [
            {"id": "parent", "label": "Original route", "style": "dashed"},
            {"id": "child", "label": "Repaired route", "style": "solid"},
            {"id": "affected", "label": "Affected day", "style": "highlight"},
            {"id": "locked", "label": "Booked or locked stop", "style": "ring"},
        ],
    }


def _map_route(plan: dict[str, Any] | None, route_matrix: dict[str, Any] | None) -> dict[str, Any] | None:
    if plan is None:
        return None
    stops = list(plan.get("selected_stops") or [])
    lookup = {_stop_id(stop): stop for stop in stops}
    ownership = _ownership_by_target(plan)
    sequence = [str(stop_id) for stop_id in plan.get("sequence") or []]
    cells = {
        (str(cell.get("origin_id")), str(cell.get("destination_id"))): cell
        for cell in (route_matrix or {}).get("cells") or []
    }
    segments = []
    missing = []
    for origin_id, destination_id in zip(sequence, sequence[1:], strict=False):
        cell = cells.get((origin_id, destination_id))
        if cell is None:
            missing.append(f"{origin_id}->{destination_id}")
            continue
        geometry = cell.get("geometry") or []
        display_geometry, sampled = _display_geometry(geometry)
        segments.append(
            {
                "origin_id": origin_id,
                "destination_id": destination_id,
                "coordinates": display_geometry,
                "geometry_sampled": sampled,
                "road_validated": bool(cell.get("road_validated")),
                "fallback_used": bool(cell.get("fallback_used")),
                "distance_m": cell.get("distance_m"),
                "duration_s": cell.get("duration_s"),
            }
        )
    return {
        "plan_id": str(plan.get("plan_id") or ""),
        "stops": [
            {
                "id": stop_id,
                "name": _stop_name(lookup.get(stop_id, {"stop_id": stop_id})),
                "city": str(lookup.get(stop_id, {}).get("city") or ""),
                "day": int(lookup.get(stop_id, {}).get("day") or 0),
                "latitude": lookup.get(stop_id, {}).get("latitude"),
                "longitude": lookup.get(stop_id, {}).get("longitude"),
                "ownership_strength": ownership.get(stop_id, ""),
            }
            for stop_id in sequence
        ],
        "segments": segments,
        "missing_segments": missing,
    }


def _display_geometry(geometry: list[Any], *, max_points: int = 320) -> tuple[list[Any], bool]:
    if len(geometry) <= max_points:
        return geometry, False
    step = max(1, math.ceil(len(geometry) / max_points))
    sampled = list(geometry[::step])
    if sampled[-1] != geometry[-1]:
        sampled.append(geometry[-1])
    return sampled, True


def _research_view(bundle: ProductDashboardSourceBundle) -> dict[str, Any]:
    certificate = bundle.certificate or {}
    route = certificate.get("route_validation") or {}
    requested = list(
        dict.fromkeys(str(run.get("method_requested") or "") for run in bundle.planner_runs)
    )
    executed = list(
        dict.fromkeys(str(run.get("method_executed") or "") for run in bundle.planner_runs)
    )
    return {
        "lineage": {
            "run_id": str(bundle.run_manifest.get("run_id") or ""),
            "parent_plan_id": str(bundle.parent_plan.get("plan_id") or ""),
            "parent_content_hash": str(bundle.parent_plan.get("content_hash") or ""),
            "child_plan_id": (bundle.child_plan or {}).get("plan_id"),
            "child_content_hash": (bundle.child_plan or {}).get("content_hash"),
            "diff_id": (bundle.diff or {}).get("diff_id"),
            "certificate_id": certificate.get("certificate_id"),
            "route_matrix_id": (bundle.route_matrix or {}).get("matrix_id"),
            "route_bundle_id": route.get("source_bundle_id"),
            "route_source_sha256": route.get("source_content_sha256"),
        },
        "methods": {
            "requested": requested,
            "executed": executed,
            "planner_runs": [
                {
                    "run_id": run.get("run_id"),
                    "parent_run_id": run.get("parent_run_id"),
                    "method_requested": run.get("method_requested"),
                    "method_executed": run.get("method_executed"),
                    "execution_status": run.get("execution_status"),
                    "solver_certification": run.get("solver_certification"),
                    "fallback_reason": run.get("fallback_reason"),
                    "error_class": run.get("error_class"),
                    "error_summary": run.get("error_summary"),
                    "runtime_seconds": run.get("runtime_seconds"),
                }
                for run in bundle.planner_runs
            ],
            "benchmark_planner_runs": [
                {
                    "run_id": run.get("run_id"),
                    "method_requested": run.get("method_requested"),
                    "method_executed": run.get("method_executed"),
                    "execution_status": run.get("execution_status"),
                    "solver_certification": run.get("solver_certification"),
                    "solver_status_raw": run.get("solver_status_raw"),
                    "error_class": run.get("error_class"),
                    "error_summary": run.get("error_summary"),
                }
                for run in bundle.benchmark_planner_runs
            ],
        },
        "certificate": certificate,
        "diff": bundle.diff,
        "source_artifacts": {
            category: list(paths) for category, paths in sorted(bundle.source_paths.items())
        },
        "source_hashes": dict(sorted(bundle.source_hashes.items())),
    }


def _affected_days(request: dict[str, Any], diff: dict[str, Any]) -> list[int]:
    days = [
        int(day)
        for day in (request.get("confirmed_constraints") or {}).get("affected_days") or []
    ]
    if not days:
        days = sorted(
            {
                int(item.get("day") or 0)
                for item in _all_diff_changes(diff)
                if int(item.get("day") or 0) > 0
            }
        )
    return list(dict.fromkeys(days))


def _all_diff_changes(diff: dict[str, Any]) -> list[dict[str, Any]]:
    keys = (
        "added_stops",
        "deleted_stops",
        "day_moves",
        "time_shifts",
        "reorder_changes",
        "lodging_changes",
        "road_changes",
    )
    return [item for key in keys for item in diff.get(key) or [] if isinstance(item, dict)]


def _plan_days(plan: dict[str, Any]) -> list[int]:
    days = [int(day.get("day") or 0) for day in plan.get("ordered_days") or []]
    return sorted(day for day in days if day > 0) or [1]


def _stops_by_day(plan: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
    result: dict[int, list[dict[str, Any]]] = {}
    for stop in plan.get("selected_stops") or []:
        day = int(stop.get("day") or plan.get("day_assignments", {}).get(_stop_id(stop), 0) or 0)
        result.setdefault(day, []).append(stop)
    for stops in result.values():
        stops.sort(key=lambda item: (int(item.get("stop_order") or 0), _stop_id(item)))
    return result


def _stop_names(plan: dict[str, Any]) -> dict[str, str]:
    return {_stop_id(stop): _stop_name(stop) for stop in plan.get("selected_stops") or []}


def _stop_id(stop: dict[str, Any]) -> str:
    return str(
        stop.get("stop_id")
        or stop.get("poi_id")
        or stop.get("attraction_name")
        or stop.get("name")
        or ""
    )


def _stop_name(stop: dict[str, Any]) -> str:
    return str(
        stop.get("attraction_name")
        or stop.get("name")
        or _titleize(_stop_id(stop))
        or "Unnamed stop"
    )


def _ownership_by_target(plan: dict[str, Any]) -> dict[str, str]:
    return {
        str(item.get("target_id") or ""): str(item.get("strength") or "")
        for item in plan.get("owned_constraints") or []
        if item.get("confirmed")
    }


def _titleize(value: str) -> str:
    return str(value).replace("_", " ").replace("-", " ").strip().title()


def _map_alternative_text(timeline: list[dict[str, Any]], affected_days: list[int]) -> str:
    affected = set(affected_days)
    parts = []
    for day in timeline:
        names = ", ".join(day["child_stop_names"]) or "No recorded stops"
        suffix = " (affected)" if day["day"] in affected else ""
        parts.append(f"Day {day['day']}{suffix}: {names}.")
    return " ".join(parts)
