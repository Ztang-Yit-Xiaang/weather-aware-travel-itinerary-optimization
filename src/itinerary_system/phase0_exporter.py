"""Phase 0 research evidence exports for production itinerary artifacts."""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd
from geopy.distance import geodesic

from .config import TripConfig
from .data.schemas import DatasetValidationReport
from .data.snapshot import load_dataset_bundle, validate_dataset_bundle
from .research_artifacts import PlanArtifact, PlannerRun, evaluate_phase0_plan, stable_content_hash
from .routing import (
    ROAD_ROUTE_CACHE_AUDIT_FILENAME,
    ROAD_ROUTE_CACHE_FILENAME,
    RoadRouteCache,
    RouteLegResult,
    RouteResult,
    load_road_route_cache,
)

PHASE0_ARTIFACT_FILES = [
    "production_phase0_dataset_validation.json",
    "production_phase0_planner_runs.csv",
    "production_phase0_plan_artifacts.jsonl",
    "production_phase0_route_audit.csv",
    "production_phase0_evaluation_reports.csv",
    "production_phase0_evidence_summary.csv",
]


def _safe_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except Exception:
        pass
    text = str(value).strip()
    return text if text else default


def _safe_float(value: Any, default: float = math.nan) -> float:
    try:
        result = float(value)
    except Exception:
        return default
    return result if math.isfinite(result) else default


def _safe_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    text = _safe_str(value).lower()
    if not text:
        return default
    return text in {"1", "true", "yes", "y", "validated", "road_validated"}


def _slug(value: Any, default: str = "item") -> str:
    text = _safe_str(value, default).lower()
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return text or default


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _sha256(path: Path) -> str:
    if not path.exists():
        return ""
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _route_cache_coverage(output_dir: Path, route_cache: RoadRouteCache) -> dict[str, Any]:
    cache_path = route_cache.source_path or (output_dir / ROAD_ROUTE_CACHE_FILENAME)
    audit_path = output_dir / ROAD_ROUTE_CACHE_AUDIT_FILENAME
    audit = _read_csv(audit_path)
    if not audit.empty and "road_validated" in audit.columns:
        requested = int(len(audit))
        validated = int(audit["road_validated"].astype(str).str.lower().isin({"true", "1", "yes"}).sum())
    else:
        requested = int(len(route_cache.frame))
        validated = int(route_cache.frame.get("road_validated_bool", pd.Series(False, index=route_cache.frame.index)).sum())
    missing = max(0, requested - validated)
    coverage_ratio = float(validated / requested) if requested else 0.0
    return {
        "road_route_cache_path": str(cache_path),
        "road_route_cache_hash": _sha256(Path(cache_path)) if cache_path else "",
        "road_route_cache_audit_path": str(audit_path),
        "road_route_cache_audit_hash": _sha256(audit_path),
        "road_route_cache_rows": int(len(route_cache.frame)),
        "road_route_requested_leg_count": requested,
        "road_route_validated_leg_count": validated,
        "road_route_missing_leg_count": missing,
        "road_route_validation_coverage": coverage_ratio,
    }


def _dataset_validation_report(config: TripConfig) -> tuple[DatasetValidationReport, dict[str, Any]]:
    catalog_snapshot_id = _safe_str(config.get("data", "catalog_snapshot_id", "california_v1"), "california_v1")
    context_snapshot_id = _safe_str(
        config.get("data", "context_snapshot_id", "context_static_demo_2026_06"),
        "context_static_demo_2026_06",
    )
    try:
        bundle = load_dataset_bundle(
            catalog_snapshot_id=catalog_snapshot_id,
            context_snapshot_id=context_snapshot_id,
        )
        report = validate_dataset_bundle(bundle)
        payload = {
            **asdict(report),
            "snapshot_dir": str(bundle.snapshot_dir),
            "manifest": bundle.manifest,
            "file_hashes": bundle.file_hashes,
        }
        return report, payload
    except Exception as exc:
        report = DatasetValidationReport(
            catalog_snapshot_id=catalog_snapshot_id,
            context_snapshot_id=context_snapshot_id,
            can_optimize=False,
            final_comparison_eligible=False,
            errors=(f"{type(exc).__name__}: {exc}",),
            warnings=(),
            table_counts={},
        )
        return report, asdict(report)


def _serialize_cell(value: Any) -> Any:
    if isinstance(value, dict | list | tuple):
        return json.dumps(value, sort_keys=True, default=str)
    return value


def _frame_from_records(records: list[dict[str, Any]]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()
    return pd.DataFrame([{key: _serialize_cell(value) for key, value in record.items()} for record in records])


def _method_key(row: pd.Series) -> tuple[str, str, int]:
    return (
        _safe_str(row.get("method"), "unknown_method"),
        _safe_str(row.get("comparison_label"), _safe_str(row.get("method_display_name"), "method")),
        int(_safe_float(row.get("trip_days"), 0.0) or 0),
    )


def _method_route_rows(method_row: pd.Series, route_stops_df: pd.DataFrame) -> pd.DataFrame:
    if route_stops_df.empty:
        return pd.DataFrame()
    method, label, trip_days = _method_key(method_row)
    output = route_stops_df.copy()
    mask = output.get("method", pd.Series("", index=output.index)).astype(str).eq(method)
    if trip_days:
        mask &= pd.to_numeric(output.get("trip_days", 0), errors="coerce").fillna(0).astype(int).eq(trip_days)
    if "comparison_type" in output.columns:
        method_mask = output["comparison_type"].astype(str).eq("method")
        if (mask & method_mask).any():
            mask &= method_mask
    if label and "comparison_label" in output.columns:
        label_mask = output["comparison_label"].astype(str).eq(label)
        if (mask & label_mask).any():
            mask &= label_mask
    return output[mask].copy()


def _sorted_route_rows(route_rows: pd.DataFrame) -> pd.DataFrame:
    if route_rows.empty:
        return route_rows.copy()
    output = route_rows.copy()
    for column in ["day", "stop_order", "route_sequence_index"]:
        if column in output.columns:
            output[column] = pd.to_numeric(output[column], errors="coerce").fillna(0)
    sort_columns = [column for column in ["day", "stop_order", "route_sequence_index"] if column in output.columns]
    return output.sort_values(sort_columns).reset_index(drop=True) if sort_columns else output.reset_index(drop=True)


def _selected_stops(route_rows: pd.DataFrame) -> tuple[dict[str, Any], ...]:
    selected = []
    for index, row in _sorted_route_rows(route_rows).iterrows():
        selected.append(
            {
                "poi_id": _safe_str(row.get("poi_id"), _slug(row.get("attraction_name"), f"stop_{index + 1}")),
                "attraction_name": _safe_str(row.get("attraction_name"), _safe_str(row.get("name"), f"Stop {index + 1}")),
                "city": _safe_str(row.get("city")),
                "day": int(_safe_float(row.get("day"), 0.0) or 0),
                "stop_order": int(_safe_float(row.get("stop_order"), index + 1) or index + 1),
                "latitude": _safe_float(row.get("latitude")),
                "longitude": _safe_float(row.get("longitude")),
                "source_list": _safe_str(row.get("source_list")),
                "source_confidence": _safe_float(row.get("source_confidence")),
            }
        )
    return tuple(selected)


def _point_id(label: str, index: int) -> str:
    return f"{_slug(label, 'point')}_{index}"


def _row_point(row: pd.Series, lat_column: str, lon_column: str, label_column: str, index: int) -> dict[str, Any] | None:
    lat = _safe_float(row.get(lat_column))
    lon = _safe_float(row.get(lon_column))
    if not math.isfinite(lat) or not math.isfinite(lon):
        return None
    label = _safe_str(row.get(label_column), _safe_str(row.get("city"), f"Point {index}"))
    return {"id": _point_id(label, index), "label": label, "latitude": lat, "longitude": lon, "row": row}


def _route_points_for_day(day_rows: pd.DataFrame) -> list[dict[str, Any]]:
    rows = _sorted_route_rows(day_rows)
    if rows.empty:
        return []
    points: list[dict[str, Any]] = []
    first = rows.iloc[0]
    start = _row_point(first, "route_start_latitude", "route_start_longitude", "route_start_name", 0)
    if start is not None:
        points.append(start)
    for index, row in rows.iterrows():
        point = _row_point(row, "latitude", "longitude", "attraction_name", int(index) + 1)
        if point is not None:
            points.append(point)
    end = _row_point(first, "route_end_latitude", "route_end_longitude", "route_end_name", len(points) + 1)
    if end is not None:
        points.append(end)
    return points


def _leg_from_points(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    route_cache: RoadRouteCache | None = None,
) -> RouteLegResult:
    if route_cache is not None:
        cached_leg = route_cache.lookup_leg(left, right)
        if cached_leg is not None:
            return cached_leg
    geometry = ((float(left["latitude"]), float(left["longitude"])), (float(right["latitude"]), float(right["longitude"])))
    distance_m = geodesic(geometry[0], geometry[1]).km * 1000.0
    duration_s = distance_m / 1000.0 * 1.25 / 38.0 * 3600.0
    row = right.get("row") if isinstance(right.get("row"), pd.Series) else pd.Series(dtype=object)
    road_validated = _safe_bool(row.get("road_validated"), False)
    if not road_validated:
        return RouteLegResult.approximate_geodesic(
            origin_id=str(left["id"]),
            destination_id=str(right["id"]),
            geometry=geometry,
            distance_m=distance_m,
            duration_s=duration_s,
            fallback_reason="production_route_stops_without_road_validated_geometry",
        )
    return RouteLegResult(
        origin_id=str(left["id"]),
        destination_id=str(right["id"]),
        geometry=geometry,
        distance_m=distance_m,
        duration_s=duration_s,
        routing_status=_safe_str(row.get("routing_status"), "ok"),
        provider=_safe_str(row.get("routing_provider"), _safe_str(row.get("drive_time_source"), "unknown")),
        geometry_source=_safe_str(row.get("geometry_source"), "road_validated_artifact"),
        distance_source=_safe_str(row.get("distance_source"), "road_validated_artifact"),
        duration_source=_safe_str(row.get("duration_source"), "road_validated_artifact"),
        road_validated=True,
        fallback_used=False,
    )


def _route_result(
    route_id: str,
    route_rows: pd.DataFrame,
    *,
    solver_feasible: bool,
    dataset_valid: bool,
    route_cache: RoadRouteCache | None = None,
) -> RouteResult:
    legs: list[RouteLegResult] = []
    if not route_rows.empty:
        route_groups = route_rows.groupby("day", sort=True, dropna=False) if "day" in route_rows.columns else [(1, route_rows)]
        for _, day_rows in route_groups:
            points = _route_points_for_day(day_rows)
            for left, right in zip(points[:-1], points[1:], strict=False):
                legs.append(_leg_from_points(left, right, route_cache=route_cache))
    sequence_ok = True
    if "sequence_violation_flag" in route_rows.columns:
        sequence_ok = not route_rows["sequence_violation_flag"].astype(str).str.lower().isin({"true", "1", "yes"}).any()
    return RouteResult(
        route_id=route_id,
        legs=tuple(legs),
        solver_feasible=bool(solver_feasible),
        schedule_feasible=bool(sequence_ok and not route_rows.empty),
        dataset_snapshot_valid=bool(dataset_valid),
    )


def _solver_certification(status: str, route_rows: pd.DataFrame) -> str:
    lowered = status.lower()
    if "failed" in lowered or route_rows.empty:
        return "NO_CERTIFICATE"
    if "heuristic" in lowered or "fallback" in lowered:
        return "HEURISTIC_ONLY"
    if "optimal" in lowered:
        return "OPTIMALITY_CERTIFIED"
    return "FEASIBILITY_CERTIFIED"


def _execution_status(status: str, route_rows: pd.DataFrame) -> str:
    lowered = status.lower()
    if "failed" in lowered or route_rows.empty:
        return "FAILED"
    if "heuristic" in lowered or "fallback" in lowered:
        return "FALLBACK_USED"
    return "COMPLETED"


def _planner_and_plan(
    method_row: pd.Series,
    route_rows: pd.DataFrame,
    *,
    config: TripConfig,
    dataset_report: DatasetValidationReport,
) -> tuple[PlannerRun, PlanArtifact]:
    method, label, trip_days = _method_key(method_row)
    scenario = _safe_str(config.get("trip", "scenario", "california_coast"), "california_coast")
    request_payload = {
        "scenario": scenario,
        "method": method,
        "trip_days": trip_days,
        "label": label,
        "catalog_snapshot_id": dataset_report.catalog_snapshot_id,
        "context_snapshot_id": dataset_report.context_snapshot_id,
    }
    request_id = f"request_{stable_content_hash(request_payload)}"
    run_id = f"run_{stable_content_hash({**request_payload, 'status': _safe_str(method_row.get('status'))})}"
    plan_id = f"plan_{stable_content_hash({'run_id': run_id, 'stops': _selected_stops(route_rows)})}"
    status = _safe_str(method_row.get("status"), "FAILED")
    execution_status = _execution_status(status, route_rows)
    planner_run = PlannerRun(
        run_id=run_id,
        planning_request_id=request_id,
        catalog_snapshot_id=dataset_report.catalog_snapshot_id,
        context_snapshot_id=dataset_report.context_snapshot_id,
        planner_specification_id="phase0_production_artifact_adapter_v1",
        method_requested=method,
        method_executed=method,
        execution_status=execution_status,
        solver_certification=_solver_certification(status, route_rows),
        solver_backend=_safe_str(method_row.get("local_route_solver"), _safe_str(method_row.get("allocation_solver"))),
        fallback_reason="heuristic_or_route_fallback" if execution_status == "FALLBACK_USED" else "",
        solver_status_raw=status,
        objective_value=_safe_float(method_row.get("objective"), None),
        optimality_gap=_safe_float(method_row.get("mip_gap"), None),
        runtime_seconds=_safe_float(method_row.get("solve_seconds"), None),
        result_plan_id=plan_id,
    )
    sorted_rows = _sorted_route_rows(route_rows)
    day_assignments = {
        _safe_str(row.get("attraction_name"), f"stop_{index + 1}"): int(_safe_float(row.get("day"), 0.0) or 0)
        for index, row in sorted_rows.iterrows()
    }
    plan = PlanArtifact(
        plan_id=plan_id,
        source_run_id=run_id,
        planning_request_id=request_id,
        catalog_snapshot_id=dataset_report.catalog_snapshot_id,
        context_snapshot_id=dataset_report.context_snapshot_id,
        selected_stops=_selected_stops(sorted_rows),
        day_assignments=day_assignments,
        sequence=tuple(_safe_str(row.get("attraction_name"), f"stop_{index + 1}") for index, row in sorted_rows.iterrows()),
        modeled_metrics={
            "total_utility": _safe_float(method_row.get("total_utility"), 0.0),
            "total_travel_time": _safe_float(method_row.get("total_travel_time"), 0.0),
            "total_travel_distance_km": _safe_float(method_row.get("total_travel_distance_km"), 0.0),
            "selected_attractions": _safe_float(method_row.get("selected_attractions"), float(len(sorted_rows))),
        },
        created_at="",
    )
    return planner_run, plan


def write_phase0_research_artifacts(
    *,
    output_dir: str | Path,
    config: TripConfig,
    method_df: pd.DataFrame | None = None,
    route_stops_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Write Phase 0 evidence tables next to production comparison artifacts."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    if method_df is None:
        method_df = _read_csv(output_path / "production_method_comparison.csv")
    if route_stops_df is None:
        route_stops_df = _read_csv(output_path / "production_method_route_stops.csv")
    method_df = method_df.copy() if method_df is not None else pd.DataFrame()
    route_stops_df = route_stops_df.copy() if route_stops_df is not None else pd.DataFrame()
    route_cache = load_road_route_cache(output_path, config=config)
    route_cache_coverage = _route_cache_coverage(output_path, route_cache)

    dataset_report, dataset_payload = _dataset_validation_report(config)
    dataset_payload["route_cache_coverage"] = route_cache_coverage
    (output_path / "production_phase0_dataset_validation.json").write_text(
        json.dumps(dataset_payload, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )

    planner_records: list[dict[str, Any]] = []
    plan_records: list[dict[str, Any]] = []
    route_records: list[dict[str, Any]] = []
    evaluation_records: list[dict[str, Any]] = []
    summary_records: list[dict[str, Any]] = []

    for _, method_row in method_df.iterrows():
        route_rows = _method_route_rows(method_row, route_stops_df)
        planner_run, plan = _planner_and_plan(
            method_row,
            route_rows,
            config=config,
            dataset_report=dataset_report,
        )
        route = _route_result(
            route_id=f"route_{plan.plan_id}",
            route_rows=route_rows,
            solver_feasible=planner_run.execution_status in {"COMPLETED", "FALLBACK_USED"},
            dataset_valid=dataset_report.can_optimize,
            route_cache=route_cache,
        )
        blocking_warnings = 0 if route.evaluation_eligible else 1
        evaluation = evaluate_phase0_plan(
            plan=plan,
            planner_run=planner_run,
            route_result=route,
            dataset_snapshot_valid=dataset_report.can_optimize,
            blocking_warning_count=blocking_warnings,
            nonblocking_warning_count=len(dataset_report.warnings),
        )
        planner_records.append(planner_run.to_record())
        plan_records.append(plan.to_record())
        for row in route.to_audit_rows():
            route_records.append(
                {
                    "run_id": planner_run.run_id,
                    "plan_id": plan.plan_id,
                    "method": planner_run.method_requested,
                    **row,
                    "route_evaluation_eligible": route.evaluation_eligible,
                }
            )
        evaluation_records.append(evaluation.to_record())
        route_validated_leg_count = sum(1 for leg in route.legs if leg.road_validated)
        route_fallback_leg_count = sum(1 for leg in route.legs if leg.fallback_used)
        route_validation_coverage = float(route_validated_leg_count / len(route.legs)) if route.legs else 0.0
        summary_records.append(
            {
                "method": planner_run.method_requested,
                "run_id": planner_run.run_id,
                "plan_id": plan.plan_id,
                "route_id": route.route_id,
                "selected_stop_count": len(plan.selected_stops),
                "route_leg_count": len(route.legs),
                "route_validated_leg_count": route_validated_leg_count,
                "route_fallback_leg_count": route_fallback_leg_count,
                "route_road_validation_coverage": route_validation_coverage,
                "solver_certification": planner_run.solver_certification,
                "execution_status": planner_run.execution_status,
                "route_road_validated": route.road_validated,
                "route_fallback_used": route.fallback_used,
                "comparison_eligibility": evaluation.comparison_eligibility,
                "dataset_can_optimize": dataset_report.can_optimize,
                "dataset_final_comparison_eligible": dataset_report.final_comparison_eligible,
                **route_cache_coverage,
            }
        )

    _frame_from_records(planner_records).to_csv(output_path / "production_phase0_planner_runs.csv", index=False)
    _frame_from_records(route_records).to_csv(output_path / "production_phase0_route_audit.csv", index=False)
    _frame_from_records(evaluation_records).to_csv(
        output_path / "production_phase0_evaluation_reports.csv",
        index=False,
    )
    _frame_from_records(summary_records).to_csv(output_path / "production_phase0_evidence_summary.csv", index=False)
    with (output_path / "production_phase0_plan_artifacts.jsonl").open("w", encoding="utf-8") as handle:
        for record in plan_records:
            handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")

    return {
        "dataset_validation": dataset_report,
        "planner_runs_df": _frame_from_records(planner_records),
        "route_audit_df": _frame_from_records(route_records),
        "evaluation_reports_df": _frame_from_records(evaluation_records),
        "evidence_summary_df": _frame_from_records(summary_records),
        "route_cache_coverage": route_cache_coverage,
        "plan_artifact_count": len(plan_records),
    }
