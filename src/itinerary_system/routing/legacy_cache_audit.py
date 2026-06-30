"""Audit legacy route geometry caches against Phase 0 route requests."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
from geopy.distance import geodesic

from .road_cache_builder import ROAD_ROUTE_REQUESTS_FILENAME

LEGACY_ROUTE_CACHE_AUDIT_FILENAME = "production_legacy_route_cache_audit.csv"

LEGACY_ROUTE_CACHE_AUDIT_COLUMNS = [
    "method",
    "day",
    "leg_index",
    "origin_label",
    "destination_label",
    "legacy_key",
    "legacy_reverse_key",
    "legacy_match_type",
    "legacy_mode",
    "legacy_path_point_count",
    "legacy_geometry_available",
    "legacy_distance_m",
    "legacy_distance_source",
    "legacy_duration_s",
    "legacy_duration_source",
    "conversion_eligible",
    "blocking_reason",
]


def _safe_float(value: Any, default: float = math.nan) -> float:
    try:
        result = float(value)
    except Exception:
        return default
    return result if math.isfinite(result) else default


def _project_legacy_cache_path(output_dir: str | Path) -> Path:
    output_path = Path(output_dir)
    cache_dir = output_path.parent / "cache" if output_path.name == "outputs" else output_path / "cache"
    return cache_dir / "production_road_route_cache.json"


def _legacy_key(left_lat: Any, left_lon: Any, right_lat: Any, right_lon: Any) -> str:
    return f"{float(left_lat):.5f},{float(left_lon):.5f}|{float(right_lat):.5f},{float(right_lon):.5f}"


def _reverse_key(key: str) -> str:
    return "|".join(reversed(key.split("|")))


def _load_legacy_cache(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _legacy_path_distance_m(path_points: Any) -> float:
    if not isinstance(path_points, list) or len(path_points) < 2:
        return math.nan
    distance_m = 0.0
    for left, right in zip(path_points[:-1], path_points[1:], strict=False):
        if not isinstance(left, list | tuple) or not isinstance(right, list | tuple):
            return math.nan
        if len(left) < 2 or len(right) < 2:
            return math.nan
        distance_m += geodesic((float(left[0]), float(left[1])), (float(right[0]), float(right[1]))).km * 1000.0
    return distance_m


def _first_finite(*values: Any) -> float:
    for value in values:
        result = _safe_float(value)
        if math.isfinite(result):
            return result
    return math.nan


def _audit_row(request: dict[str, Any], legacy_cache: dict[str, Any]) -> dict[str, Any]:
    key = _legacy_key(
        request.get("origin_latitude"),
        request.get("origin_longitude"),
        request.get("destination_latitude"),
        request.get("destination_longitude"),
    )
    reverse_key = _reverse_key(key)
    exact_entry = legacy_cache.get(key)
    reverse_entry = legacy_cache.get(reverse_key)
    match_type = "exact" if isinstance(exact_entry, dict) else ("reverse" if isinstance(reverse_entry, dict) else "missing")
    entry = exact_entry if isinstance(exact_entry, dict) else (reverse_entry if isinstance(reverse_entry, dict) else {})
    path_points = entry.get("path") if isinstance(entry, dict) else None
    path_point_count = len(path_points) if isinstance(path_points, list) else 0
    geometry_available = path_point_count >= 2
    raw_distance_m = _first_finite(entry.get("distance_m"), entry.get("distance"), entry.get("route_distance_m"))
    path_distance_m = _legacy_path_distance_m(path_points)
    distance_m = raw_distance_m if math.isfinite(raw_distance_m) else path_distance_m
    distance_source = ""
    if math.isfinite(raw_distance_m):
        distance_source = "legacy_distance_field"
    elif math.isfinite(path_distance_m):
        distance_source = "computed_from_legacy_path"
    duration_s = _first_finite(entry.get("duration_s"), entry.get("duration"), entry.get("route_duration_s"))
    duration_source = "legacy_duration_field" if math.isfinite(duration_s) else ""
    reasons = []
    if match_type == "missing":
        reasons.append("legacy_cache_entry_missing")
    if not geometry_available:
        reasons.append("legacy_geometry_missing")
    if not math.isfinite(distance_m):
        reasons.append("legacy_distance_missing")
    if not math.isfinite(duration_s):
        reasons.append("legacy_duration_missing")
    conversion_eligible = not reasons and match_type in {"exact", "reverse"}
    return {
        "method": request.get("method", ""),
        "day": request.get("day", ""),
        "leg_index": request.get("leg_index", ""),
        "origin_label": request.get("origin_label", ""),
        "destination_label": request.get("destination_label", ""),
        "legacy_key": key,
        "legacy_reverse_key": reverse_key,
        "legacy_match_type": match_type,
        "legacy_mode": entry.get("mode", "") if isinstance(entry, dict) else "",
        "legacy_path_point_count": path_point_count,
        "legacy_geometry_available": geometry_available,
        "legacy_distance_m": round(float(distance_m), 3) if math.isfinite(distance_m) else "",
        "legacy_distance_source": distance_source,
        "legacy_duration_s": round(float(duration_s), 3) if math.isfinite(duration_s) else "",
        "legacy_duration_source": duration_source,
        "conversion_eligible": conversion_eligible,
        "blocking_reason": "; ".join(reasons) if reasons else "",
    }


def audit_legacy_route_cache(
    *,
    output_dir: str | Path,
    legacy_cache_path: str | Path | None = None,
    request_df: pd.DataFrame | None = None,
    write: bool = True,
) -> dict[str, Any]:
    """Compare a legacy route-path cache with the Phase 0 route request manifest."""

    output_path = Path(output_dir)
    request_path = output_path / ROAD_ROUTE_REQUESTS_FILENAME
    audit_path = output_path / LEGACY_ROUTE_CACHE_AUDIT_FILENAME
    if request_df is None:
        request_df = pd.read_csv(request_path) if request_path.exists() else pd.DataFrame()
    request_df = request_df.copy()
    legacy_path = Path(legacy_cache_path) if legacy_cache_path is not None else _project_legacy_cache_path(output_path)
    legacy_cache = _load_legacy_cache(legacy_path)
    rows = [_audit_row(row, legacy_cache) for row in request_df.to_dict("records")]
    audit_df = pd.DataFrame(rows, columns=LEGACY_ROUTE_CACHE_AUDIT_COLUMNS)
    if write:
        output_path.mkdir(parents=True, exist_ok=True)
        audit_df.to_csv(audit_path, index=False)
    matched = int(audit_df["legacy_match_type"].isin({"exact", "reverse"}).sum()) if not audit_df.empty else 0
    geometry_available = int(audit_df["legacy_geometry_available"].astype(bool).sum()) if not audit_df.empty else 0
    conversion_eligible = int(audit_df["conversion_eligible"].astype(bool).sum()) if not audit_df.empty else 0
    return {
        "audit_df": audit_df,
        "audit_path": audit_path,
        "legacy_cache_path": legacy_path,
        "request_count": int(len(audit_df)),
        "matched_count": matched,
        "geometry_available_count": geometry_available,
        "conversion_eligible_count": conversion_eligible,
    }
