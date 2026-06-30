"""Build validated road-route cache rows from production route-stop artifacts."""

from __future__ import annotations

import hashlib
import json
import math
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd

from .cache import ROAD_ROUTE_CACHE_FILENAME, route_anchor_key

ROAD_ROUTE_CACHE_AUDIT_FILENAME = "production_road_route_cache_audit.csv"
ROAD_ROUTE_REQUESTS_FILENAME = "production_road_route_requests.csv"
OSRM_LOCAL_BASE_URL = "http://127.0.0.1:5000"
OSRM_PUBLIC_BASE_URL = "https://router.project-osrm.org"

ROAD_ROUTE_CACHE_COLUMNS = [
    "origin_id",
    "destination_id",
    "origin_label",
    "destination_label",
    "geometry",
    "distance_m",
    "duration_s",
    "provider",
    "routing_profile",
    "routing_status",
    "geometry_source",
    "distance_source",
    "duration_source",
    "road_validated",
    "fallback_used",
    "fallback_reason",
    "query_hash",
    "retrieved_at",
]

ROAD_ROUTE_CACHE_AUDIT_COLUMNS = [
    "method",
    "day",
    "leg_index",
    "origin_label",
    "destination_label",
    "cache_key",
    "cache_path",
    "status",
    "road_validated",
    "reason",
    "fetch_attempted",
    "fetch_status",
]

ROAD_ROUTE_REQUEST_COLUMNS = [
    "method",
    "day",
    "leg_index",
    "origin_label",
    "destination_label",
    "origin_latitude",
    "origin_longitude",
    "destination_latitude",
    "destination_longitude",
    "cache_key",
    "cache_path",
    "osrm_route_url",
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


def _project_cache_dir(output_dir: str | Path) -> Path:
    output_path = Path(output_dir)
    return output_path.parent / "cache" if output_path.name == "outputs" else output_path / "cache"


def is_public_osrm_base_url(osrm_base_url: str) -> bool:
    """Return whether a base URL targets the public OSRM demo service."""

    parsed = urlparse(str(osrm_base_url).strip())
    host = (parsed.hostname or "").lower()
    return host == "router.project-osrm.org"


def validate_route_fetch_policy(osrm_base_url: str, *, allow_public_osrm: bool = False) -> None:
    """Block accidental public OSRM use unless it was explicitly approved."""

    if is_public_osrm_base_url(osrm_base_url) and not allow_public_osrm:
        raise ValueError(
            "public OSRM fetch requires --allow-public-osrm because route coordinates would be sent "
            "to https://router.project-osrm.org; use a local/pinned OSRM endpoint for reproducible evidence."
        )


def osrm_cache_key(points: list[tuple[float, float]]) -> str:
    """Return the cache key used by the existing OSRM cache convention."""

    normalized = [[float(lat), float(lon)] for lat, lon in points]
    return hashlib.sha1(json.dumps(normalized).encode("utf-8")).hexdigest()[:16]


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _payload_route(payload: dict[str, Any]) -> dict[str, Any]:
    if isinstance(payload.get("raw"), dict):
        raw_routes = payload.get("raw", {}).get("routes", [])
        if raw_routes:
            return raw_routes[0]
    raw_routes = payload.get("routes", [])
    if raw_routes:
        return raw_routes[0]
    return {}


def _latlon_geometry(payload: dict[str, Any]) -> list[list[float]]:
    latlon = payload.get("latlon_geometry")
    if isinstance(latlon, list):
        geometry = []
        for point in latlon:
            if not isinstance(point, list | tuple) or len(point) < 2:
                continue
            lat = _safe_float(point[0])
            lon = _safe_float(point[1])
            if math.isfinite(lat) and math.isfinite(lon):
                geometry.append([lat, lon])
        if len(geometry) >= 2:
            return geometry
    route = _payload_route(payload)
    raw_coords = route.get("geometry", {}).get("coordinates", []) if isinstance(route, dict) else []
    geometry = []
    for point in raw_coords:
        if not isinstance(point, list | tuple) or len(point) < 2:
            continue
        lon = _safe_float(point[0])
        lat = _safe_float(point[1])
        if math.isfinite(lat) and math.isfinite(lon):
            geometry.append([lat, lon])
    return geometry


def _wrap_osrm_payload(payload: dict[str, Any], *, status: str = "osrm_live") -> dict[str, Any]:
    return {
        "status": status,
        "latlon_geometry": _latlon_geometry(payload),
        "raw": payload,
        "retrieved_at": datetime.now(UTC).isoformat(),
    }


def fetch_osrm_payload(
    points: list[tuple[float, float]],
    *,
    osrm_base_url: str = OSRM_LOCAL_BASE_URL,
    timeout_seconds: int = 35,
) -> dict[str, Any]:
    """Fetch an OSRM route response for two or more lat/lon points."""

    if len(points) < 2:
        raise ValueError("OSRM route fetch requires at least two points")
    url = osrm_route_url(points, osrm_base_url=osrm_base_url)
    with urllib.request.urlopen(url, timeout=int(timeout_seconds)) as response:  # nosec B310 - explicit user opt-in.
        raw = response.read().decode("utf-8")
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("OSRM response must be a JSON object")
    return payload


def osrm_route_url(points: list[tuple[float, float]], *, osrm_base_url: str = OSRM_LOCAL_BASE_URL) -> str:
    """Return the OSRM route URL for a list of lat/lon points."""

    if len(points) < 2:
        raise ValueError("OSRM route URL requires at least two points")
    coord_text = ";".join(f"{float(lon)},{float(lat)}" for lat, lon in points)
    query = urllib.parse.urlencode({"overview": "full", "geometries": "geojson"})
    return f"{osrm_base_url.rstrip('/')}/route/v1/driving/{coord_text}?{query}"


def _distance_duration(payload: dict[str, Any]) -> tuple[float, float]:
    route = _payload_route(payload)
    distance = _safe_float(route.get("distance") if isinstance(route, dict) else None)
    duration = _safe_float(route.get("duration") if isinstance(route, dict) else None)
    if not math.isfinite(distance):
        distance = _safe_float(payload.get("distance_m"))
    if not math.isfinite(duration):
        duration = _safe_float(payload.get("duration_s"))
    return distance, duration


def _route_point(row: pd.Series, lat_column: str, lon_column: str, label_column: str) -> dict[str, Any] | None:
    lat = _safe_float(row.get(lat_column))
    lon = _safe_float(row.get(lon_column))
    if not math.isfinite(lat) or not math.isfinite(lon):
        return None
    label = _safe_str(row.get(label_column), _safe_str(row.get("city")))
    if not label:
        return None
    return {"label": label, "latitude": lat, "longitude": lon}


def _sorted_route_rows(route_rows: pd.DataFrame) -> pd.DataFrame:
    if route_rows.empty:
        return route_rows.copy()
    output = route_rows.copy()
    for column in ["day", "stop_order", "route_sequence_index"]:
        if column in output.columns:
            output[column] = pd.to_numeric(output[column], errors="coerce").fillna(0)
    sort_columns = [column for column in ["day", "stop_order", "route_sequence_index"] if column in output.columns]
    return output.sort_values(sort_columns).reset_index(drop=True) if sort_columns else output.reset_index(drop=True)


def _route_points_for_day(day_rows: pd.DataFrame) -> list[dict[str, Any]]:
    rows = _sorted_route_rows(day_rows)
    if rows.empty:
        return []
    points: list[dict[str, Any]] = []
    first = rows.iloc[0]
    start = _route_point(first, "route_start_latitude", "route_start_longitude", "route_start_name")
    if start is not None:
        points.append(start)
    for _, row in rows.iterrows():
        point = _route_point(row, "latitude", "longitude", "attraction_name")
        if point is not None:
            points.append(point)
    end = _route_point(first, "route_end_latitude", "route_end_longitude", "route_end_name")
    if end is not None:
        points.append(end)
    return points


def iter_route_leg_requests(route_stops_df: pd.DataFrame) -> list[dict[str, Any]]:
    """Return ordered leg requests from production method route stops."""

    if route_stops_df.empty:
        return []
    frame = route_stops_df.copy()
    if "comparison_type" in frame.columns:
        method_rows = frame[frame["comparison_type"].astype(str).eq("method")].copy()
        if not method_rows.empty:
            frame = method_rows
    grouping_columns = [column for column in ["method", "comparison_label", "trip_days"] if column in frame.columns]
    grouped = frame.groupby(grouping_columns, sort=True, dropna=False) if grouping_columns else [((), frame)]
    requests: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for group_key, group in grouped:
        method = ""
        if grouping_columns:
            values = group_key if isinstance(group_key, tuple) else (group_key,)
            group_values = dict(zip(grouping_columns, values, strict=False))
            method = _safe_str(group_values.get("method"))
        day_groups = group.groupby("day", sort=True, dropna=False) if "day" in group.columns else [(1, group)]
        for day, day_rows in day_groups:
            points = _route_points_for_day(day_rows)
            for leg_index, (left, right) in enumerate(zip(points[:-1], points[1:], strict=False), start=1):
                point_pair = [
                    (float(left["latitude"]), float(left["longitude"])),
                    (float(right["latitude"]), float(right["longitude"])),
                ]
                cache_key = osrm_cache_key(point_pair)
                dedupe_key = (route_anchor_key(left["label"]), route_anchor_key(right["label"]), cache_key)
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)
                requests.append(
                    {
                        "method": method,
                        "day": day,
                        "leg_index": leg_index,
                        "origin_label": left["label"],
                        "destination_label": right["label"],
                        "origin_latitude": left["latitude"],
                        "origin_longitude": left["longitude"],
                        "destination_latitude": right["latitude"],
                        "destination_longitude": right["longitude"],
                        "points": point_pair,
                        "cache_key": cache_key,
                    }
                )
    return requests


def _cache_row_from_payload(request: dict[str, Any], payload: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
    status = _safe_str(payload.get("status"), "cached_osrm")
    geometry = _latlon_geometry(payload)
    distance_m, duration_s = _distance_duration(payload)
    if not geometry:
        return None, "cached_payload_missing_geometry"
    if not math.isfinite(distance_m) or not math.isfinite(duration_s):
        return None, "cached_payload_missing_distance_or_duration"
    if "error" in status.lower() or "fallback" in status.lower():
        return None, f"cached_payload_status_not_validated:{status}"
    retrieved_at = _safe_str(payload.get("retrieved_at")) or _safe_str(payload.get("timestamp_utc"))
    if not retrieved_at:
        retrieved_at = datetime.now(UTC).isoformat()
    return (
        {
            "origin_id": route_anchor_key(request["origin_label"]),
            "destination_id": route_anchor_key(request["destination_label"]),
            "origin_label": request["origin_label"],
            "destination_label": request["destination_label"],
            "geometry": json.dumps(geometry, separators=(",", ":")),
            "distance_m": float(distance_m),
            "duration_s": float(duration_s),
            "provider": "cached_osrm",
            "routing_profile": "driving",
            "routing_status": status,
            "geometry_source": "cached_osrm_route_geometry",
            "distance_source": "cached_osrm_route_distance",
            "duration_source": "cached_osrm_route_duration",
            "road_validated": True,
            "fallback_used": False,
            "fallback_reason": "",
            "query_hash": request["cache_key"],
            "retrieved_at": retrieved_at,
        },
        "cached_osrm_validated",
    )


@dataclass(frozen=True)
class RoadRouteCacheBuildResult:
    cache_df: pd.DataFrame
    audit_df: pd.DataFrame
    request_df: pd.DataFrame
    cache_path: Path
    audit_path: Path
    request_path: Path

    @property
    def complete(self) -> bool:
        return not self.audit_df.empty and bool(self.audit_df["road_validated"].astype(bool).all())


def build_road_route_cache_from_artifacts(
    *,
    output_dir: str | Path,
    route_stops_df: pd.DataFrame | None = None,
    cache_dir: str | Path | None = None,
    fetch_missing: bool = False,
    osrm_base_url: str = OSRM_LOCAL_BASE_URL,
    request_timeout_seconds: int = 35,
    allow_public_osrm: bool = False,
    osrm_fetcher: Callable[[list[tuple[float, float]], str, int], dict[str, Any]] | None = None,
    write: bool = True,
) -> RoadRouteCacheBuildResult:
    """Build production_road_route_cache.csv from cached OSRM route responses."""

    if fetch_missing:
        validate_route_fetch_policy(osrm_base_url, allow_public_osrm=allow_public_osrm)

    output_path = Path(output_dir)
    if route_stops_df is None:
        route_path = output_path / "production_method_route_stops.csv"
        route_stops_df = pd.read_csv(route_path) if route_path.exists() else pd.DataFrame()
    route_stops_df = route_stops_df.copy()
    cache_path = output_path / ROAD_ROUTE_CACHE_FILENAME
    audit_path = output_path / ROAD_ROUTE_CACHE_AUDIT_FILENAME
    request_path = output_path / ROAD_ROUTE_REQUESTS_FILENAME
    osrm_cache_dir = Path(cache_dir) if cache_dir is not None else _project_cache_dir(output_path)
    rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    request_rows: list[dict[str, Any]] = []

    for request in iter_route_leg_requests(route_stops_df):
        request_cache_path = osrm_cache_dir / f"open_osrm_route_{request['cache_key']}.json"
        request_rows.append(
            {
                "method": request["method"],
                "day": request["day"],
                "leg_index": request["leg_index"],
                "origin_label": request["origin_label"],
                "destination_label": request["destination_label"],
                "origin_latitude": request["origin_latitude"],
                "origin_longitude": request["origin_longitude"],
                "destination_latitude": request["destination_latitude"],
                "destination_longitude": request["destination_longitude"],
                "cache_key": request["cache_key"],
                "cache_path": str(request_cache_path),
                "osrm_route_url": osrm_route_url(request["points"], osrm_base_url=osrm_base_url),
            }
        )
        payload = _read_json(request_cache_path)
        row = None
        status = "missing_osrm_cache"
        reason = "no cached OSRM response for leg"
        fetch_attempted = False
        fetch_status = ""
        if payload is None and fetch_missing:
            fetch_attempted = True
            try:
                fetcher = osrm_fetcher or (
                    lambda points, base_url, timeout: fetch_osrm_payload(
                        points,
                        osrm_base_url=base_url,
                        timeout_seconds=timeout,
                    )
                )
                raw_payload = fetcher(request["points"], osrm_base_url, int(request_timeout_seconds))
                payload = _wrap_osrm_payload(raw_payload)
                if write:
                    _write_json(request_cache_path, payload)
                fetch_status = "fetched_osrm_response"
            except Exception as exc:
                fetch_status = f"osrm_fetch_error:{type(exc).__name__}"
                status = fetch_status
                reason = fetch_status
        if payload is not None:
            row, status = _cache_row_from_payload(request, payload)
            if fetch_attempted and row is not None:
                status = "fetched_osrm_validated"
            elif fetch_attempted and row is None:
                status = f"fetched_osrm_invalid:{status}"
            reason = "" if row is not None else status
        if row is not None:
            rows.append(row)
        audit_rows.append(
            {
                "method": request["method"],
                "day": request["day"],
                "leg_index": request["leg_index"],
                "origin_label": request["origin_label"],
                "destination_label": request["destination_label"],
                "cache_key": request["cache_key"],
                "cache_path": str(request_cache_path),
                "status": status,
                "road_validated": row is not None,
                "reason": reason,
                "fetch_attempted": fetch_attempted,
                "fetch_status": fetch_status,
            }
        )

    cache_df = pd.DataFrame(rows, columns=ROAD_ROUTE_CACHE_COLUMNS)
    audit_df = pd.DataFrame(audit_rows, columns=ROAD_ROUTE_CACHE_AUDIT_COLUMNS)
    request_df = pd.DataFrame(request_rows, columns=ROAD_ROUTE_REQUEST_COLUMNS)
    if write:
        output_path.mkdir(parents=True, exist_ok=True)
        cache_df.to_csv(cache_path, index=False)
        audit_df.to_csv(audit_path, index=False)
        request_df.to_csv(request_path, index=False)
    return RoadRouteCacheBuildResult(
        cache_df=cache_df,
        audit_df=audit_df,
        request_df=request_df,
        cache_path=cache_path,
        audit_path=audit_path,
        request_path=request_path,
    )
