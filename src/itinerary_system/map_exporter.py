"""Lightweight and modular map exports for production itinerary artifacts."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd

from .artifact_metadata import artifact_metadata_matches, read_artifact_metadata
from .config import TripConfig
from .region_scenarios import get_scenario_definition

DEFAULT_DASHBOARD_ROUTE_ID = "selected_route"
DEFAULT_DASHBOARD_METHOD = "hierarchical_bandit_gurobi_repair"
CONTRACT_VERSION = "core-route-index-v1"
PLAYABLE_ROUTE_FAMILIES = {"selected", "route_matrix", "trip_length", "method", "interest_profile"}
MARKER_ONLY_ROUTE_FAMILIES = {"hotel", "nature", "must_go"}
CUSTOMER_ROUTE_FAMILIES = {"selected", "trip_length", "interest_profile"}
CUSTOMER_CONTROL_GROUPS = {
    "selected": "saved_route",
    "trip_length": "trip_days",
    "interest_profile": "interest",
}
PLACEHOLDER_HOTEL_PATTERNS = (
    "lodging candidate pending",
    "city_center_placeholder",
    "base marker",
)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except Exception:
        return default
    return default if pd.isna(result) else result


def _safe_str(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value)


def _safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _is_placeholder_hotel_name(value: Any) -> bool:
    text = _safe_str(value).strip().lower()
    return bool(text) and any(pattern in text for pattern in PLACEHOLDER_HOTEL_PATTERNS)


def _route_name(row: pd.Series) -> str:
    for column in ["attraction_name", "name", "hotel_name"]:
        value = _safe_str(row.get(column, ""))
        if value:
            return value
    return "Stop"


def _selected_rows(route_df: pd.DataFrame, max_routes: int = 1) -> pd.DataFrame:
    if route_df is None or route_df.empty:
        return pd.DataFrame(columns=["name", "latitude", "longitude"])
    frame = route_df.copy()
    if "route_key" in frame.columns and max_routes > 0:
        route_keys = list(frame["route_key"].dropna().astype(str).unique())[:max_routes]
        if route_keys:
            frame = frame[frame["route_key"].astype(str).isin(route_keys)].copy()
    sort_cols = [
        column for column in ["trip_days", "route_sequence_index", "day", "stop_order"] if column in frame.columns
    ]
    return frame.sort_values(sort_cols).reset_index(drop=True) if sort_cols else frame.reset_index(drop=True)


def _sort_route_rows(route_df: pd.DataFrame) -> pd.DataFrame:
    if route_df is None or route_df.empty:
        return pd.DataFrame()
    sort_cols = [
        column
        for column in ["trip_days", "route_sequence_index", "day", "stop_order", "attraction_name"]
        if column in route_df.columns
    ]
    return route_df.sort_values(sort_cols).reset_index(drop=True) if sort_cols else route_df.reset_index(drop=True)


def _read_csv_if_present(path: Path) -> pd.DataFrame:
    try:
        if path.exists() and path.stat().st_size > 0:
            return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
    return pd.DataFrame()


def _default_route_label(route_df: pd.DataFrame, fallback: str = "Selected Route") -> str:
    if route_df is None or route_df.empty:
        return fallback
    for column in ["comparison_label", "method_display_name", "method", "route_key"]:
        if column in route_df.columns:
            values = route_df[column].dropna().astype(str)
            values = values[values.str.strip().ne("")]
            if not values.empty:
                return str(values.iloc[0])
    return fallback


def _default_route_frame(
    route_df: pd.DataFrame,
    *,
    output_dir: Path,
    config: TripConfig,
    max_routes: int = 1,
) -> tuple[pd.DataFrame, str, str]:
    """Resolve the milestone-1 dashboard route without changing optimization output.

    The dashboard prefers the production method-comparison artifact because that is
    what the notebook writes before map export. If that artifact is unavailable,
    the already-materialized day plan passed by map_renderer remains the fallback.
    """
    configured_days = int(config.get("trip", "trip_days", 7))
    method_frame = _read_csv_if_present(output_dir / "production_method_route_stops.csv")
    if not method_frame.empty:
        candidate = method_frame.copy()
        if "method" in candidate.columns:
            preferred = candidate[candidate["method"].astype(str).eq(DEFAULT_DASHBOARD_METHOD)].copy()
            if not preferred.empty:
                candidate = preferred
        if "trip_days" in candidate.columns:
            day_filtered = candidate[
                pd.to_numeric(candidate["trip_days"], errors="coerce").fillna(-1).astype(int).eq(configured_days)
            ].copy()
            if not day_filtered.empty:
                candidate = day_filtered
        candidate = _sort_route_rows(candidate)
        if not candidate.empty:
            return candidate, "production_method_route_stops.csv", _default_route_label(candidate)

    fallback_frame = _sort_route_rows(_selected_rows(route_df, max_routes=max_routes))
    return fallback_frame, "export_map_artifacts input day_plan_df", _default_route_label(fallback_frame)


def _slug(value: Any, fallback: str = "item") -> str:
    text = re.sub(r"[^a-z0-9]+", "_", str(value or "").lower()).strip("_")
    return text or fallback


def _first_nonempty(route_df: pd.DataFrame, columns: list[str], fallback: str = "") -> str:
    if route_df is None or route_df.empty:
        return fallback
    for column in columns:
        if column in route_df.columns:
            values = route_df[column].dropna().astype(str).str.strip()
            values = values[values.ne("")]
            if not values.empty:
                return str(values.iloc[0])
    return fallback


def _route_trip_days(route_df: pd.DataFrame) -> int | None:
    if route_df is None or route_df.empty or "trip_days" not in route_df.columns:
        return None
    values = pd.to_numeric(route_df["trip_days"], errors="coerce").dropna()
    return int(values.iloc[0]) if not values.empty else None


def _quick_groups(route_df: pd.DataFrame, *, family: str, selector_group: str) -> list[str]:
    groups = [family, selector_group]
    trip_days = _route_trip_days(route_df)
    if trip_days:
        groups.append(f"days_{trip_days}")
    for column in ["interest_profile", "profile", "traveler_profile", "method"]:
        value = _first_nonempty(route_df, [column])
        if value:
            groups.append(_slug(value))
    label = _first_nonempty(route_df, ["comparison_label", "method_display_name", "route_key"])
    if "balanced" in label.lower() or any("balanced" in item for item in groups):
        groups.append("balanced")
    return list(dict.fromkeys(group for group in groups if group))


def _route_record(
    route_id: str,
    label: str,
    *,
    default: bool,
    optional: bool,
    family: str,
    selector_group: str,
    route_df: pd.DataFrame | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "id": route_id,
        "label": label or route_id.replace("_", " ").title(),
        "default": bool(default),
        "optional": bool(optional),
        "family": family,
        "selector_group": selector_group,
        "geojson": f"assets/routes/{route_id}.geojson",
        "geojson_js": f"assets/routes/{route_id}.js",
        "pois": f"assets/pois/{route_id}_pois.json",
        "pois_js": f"assets/pois/{route_id}_pois.js",
        "quick_groups": _quick_groups(
            route_df if route_df is not None else pd.DataFrame(), family=family, selector_group=selector_group
        ),
    }
    if route_df is not None and not route_df.empty:
        record.update(
            {
                "trip_days": _route_trip_days(route_df),
                "method": _first_nonempty(route_df, ["method"]),
                "profile": _first_nonempty(route_df, ["profile", "traveler_profile"]),
                "interest_profile": _first_nonempty(route_df, ["interest_profile", "profile"]),
                "comparison_label": _first_nonempty(route_df, ["comparison_label"]),
            }
        )
    if extra:
        record.update({key: value for key, value in extra.items() if value is not None and value != ""})
    record.setdefault("playable", family in PLAYABLE_ROUTE_FAMILIES)
    record.setdefault("marker_only", family in MARKER_ONLY_ROUTE_FAMILIES)
    record.setdefault("customer_visible", family in CUSTOMER_ROUTE_FAMILIES)
    record.setdefault("research_only", family not in CUSTOMER_ROUTE_FAMILIES)
    record.setdefault("customer_control_group", CUSTOMER_CONTROL_GROUPS.get(family, "research"))
    return record


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _point_type(row: pd.Series) -> str:
    for column in ["type", "category", "park_type", "route_type"]:
        value = _safe_str(row.get(column, ""))
        if value:
            return value
    return "point_of_interest"


def _city_or_anchor(row: pd.Series) -> str:
    nature_region = _safe_str(row.get("nature_region", ""))
    category = _safe_str(row.get("category", "")).lower()
    if nature_region and any(token in nature_region.lower() for token in ["national park", "big sur", "bixby"]):
        return nature_region
    if nature_region and any(token in category for token in ["national_park", "park", "viewpoint", "hiking", "nature"]):
        return nature_region
    for column in ["city_or_anchor", "city", "overnight_city", "route_start_city", "route_end_city"]:
        value = _safe_str(row.get(column, ""))
        if value:
            return value
    return nature_region or "Unassigned"


def _source_confidence(row: pd.Series) -> float:
    for column in ["source_confidence", "data_confidence", "confidence", "confidence_score"]:
        value = _safe_float(row.get(column, 0.0))
        if value > 0:
            return round(_clamp(value), 3)
    source = _safe_str(row.get("source_list", row.get("source", ""))).lower()
    if any(token in source for token in ["openstreetmap", "osm", "wikidata", "wikipedia", "yelp"]):
        return 0.75
    if "curated" in source:
        return 0.58
    if source:
        return 0.5
    return 0.4


def _weather_risk(row: pd.Series) -> float:
    for column in ["weather_risk", "risk_score", "weather_score"]:
        value = _safe_float(row.get(column, -1.0), -1.0)
        if value >= 0:
            return round(_clamp(value), 3)
    sensitivity = _safe_float(row.get("weather_sensitivity", 0.0))
    seasonality = _safe_float(row.get("seasonality_risk", 0.0))
    return round(_clamp(0.08 + 0.28 * sensitivity + 0.18 * seasonality), 3)


def _expected_duration_minutes(row: pd.Series) -> int:
    for column in ["expected_duration_minutes", "visit_duration_minutes", "visit_duration_sim", "available_visit_minutes"]:
        value = _safe_float(row.get(column, 0.0))
        if value > 0:
            return int(round(max(30.0, min(240.0, value))))
    category = _safe_str(row.get("category", "")).lower()
    if any(token in category for token in ["national_park", "hiking", "viewpoint", "park", "nature"]):
        return 120
    if "museum" in category or "history" in category:
        return 90
    return 75


def _point_description(row: pd.Series) -> str:
    for column in ["description", "summary", "poi_description", "notes"]:
        value = _safe_str(row.get(column, "")).strip()
        if value and value.lower() not in {"nan", "none", "selected route stop"}:
            return value
    name = _route_name(row)
    anchor = _city_or_anchor(row)
    point_type = _point_type(row).replace("_", " ")
    source = _safe_str(row.get("source_list", row.get("source", "")))
    source_part = f" using {source} signals" if source else ""
    return f"{name} is a {point_type} in {anchor}{source_part}."


def _why_selected(row: pd.Series) -> str:
    for column in ["why_selected", "reason_selected", "selection_reason"]:
        value = _safe_str(row.get(column, "")).strip()
        if value:
            return value
    why_not = _safe_str(row.get("why_not_selected", "")).strip()
    if why_not:
        return f"Not selected for the saved route: {why_not}"
    value = _safe_float(row.get("interest_adjusted_value", row.get("final_poi_value", 0.0)))
    nature = _safe_float(row.get("nature_score", 0.0))
    scenic = _safe_float(row.get("scenic_score", 0.0))
    if nature > 0.35 or scenic > 0.35:
        return "Selected or ranked because it strengthens the nature/scenic objective for the active scenario."
    if value > 0:
        return "Selected or ranked because it contributes positive artifact-backed itinerary value."
    return "Shown for route context or candidate comparison; not counted as a saved optimized stop unless it appears in the selected route layer."


def _point_records(route_df: pd.DataFrame, *, selected_poi: bool = True) -> list[dict[str, Any]]:
    records = []
    for _, row in route_df.iterrows():
        lat = _safe_float(row.get("latitude", row.get("hotel_latitude", 0.0)))
        lon = _safe_float(row.get("longitude", row.get("hotel_longitude", 0.0)))
        if not lat or not lon:
            continue
        final_poi_value = _safe_float(row.get("final_poi_value", 0.0))
        interest_adjusted_value = _safe_float(row.get("interest_adjusted_value", 0.0))
        display_utility = interest_adjusted_value if interest_adjusted_value > 0 else final_poi_value
        optimization_value_source = "interest_adjusted_value" if interest_adjusted_value > 0 else "final_poi_value"
        point_type = _point_type(row)
        city_or_anchor = _city_or_anchor(row)
        source_confidence = _source_confidence(row)
        weather_risk = _weather_risk(row)
        expected_duration = _expected_duration_minutes(row)
        route_sequence_index = int(_safe_float(row.get("route_sequence_index", 0), 0.0))
        display_sequence_index = route_sequence_index if route_sequence_index > 0 else (len(records) + 1 if selected_poi else 0)
        records.append(
            {
                "name": _route_name(row),
                "city": _safe_str(row.get("city", row.get("overnight_city", ""))),
                "city_or_anchor": city_or_anchor,
                "lat": lat,
                "lon": lon,
                "day": int(_safe_float(row.get("day", 0), 0.0)),
                "stop_order": int(_safe_float(row.get("stop_order", 0), 0.0)),
                "trip_days": int(_safe_float(row.get("trip_days", 0), 0.0)),
                "overnight_city": _safe_str(row.get("overnight_city", "")),
                "hotel_name": _safe_str(row.get("hotel_name", "")),
                "hotel_lat": _safe_float(row.get("hotel_latitude", 0.0)),
                "hotel_lon": _safe_float(row.get("hotel_longitude", 0.0)),
                "candidate_rank": int(_safe_float(row.get("candidate_rank", 0), 0.0)),
                "selected": _safe_bool(row.get("selected", False)),
                "hotel_score": _safe_float(row.get("hotel_score", 0.0)),
                "mean_distance_to_selected_stops_km": _safe_float(row.get("mean_distance_to_selected_stops_km", 0.0)),
                "mean_distance_to_must_go_km": _safe_float(row.get("mean_distance_to_must_go_km", 0.0)),
                "selected_hotel_reason": _safe_str(row.get("selected_hotel_reason", "")),
                "category": _safe_str(row.get("category", "")),
                "type": point_type,
                "source_list": _safe_str(row.get("source_list", row.get("source", ""))),
                "source_confidence": source_confidence,
                "nature_region": _safe_str(row.get("nature_region", "")),
                "final_poi_value": final_poi_value,
                "interest_adjusted_value": interest_adjusted_value,
                "display_utility": display_utility,
                "optimization_value_source": optimization_value_source,
                "interest_fit": _safe_float(row.get("interest_fit", 0.0)),
                "interest_delta": _safe_float(row.get("interest_delta", 0.0)),
                "nature_score": _safe_float(row.get("nature_score", 0.0)),
                "scenic_score": _safe_float(row.get("scenic_score", 0.0)),
                "weather_sensitivity": _safe_float(row.get("weather_sensitivity", 0.0)),
                "weather_risk": weather_risk,
                "seasonality_risk": _safe_float(row.get("seasonality_risk", 0.0)),
                "park_type": _safe_str(row.get("park_type", "")),
                "route_type": _safe_str(row.get("route_type", "")),
                "status": _safe_str(row.get("status", "")),
                "notes": _safe_str(row.get("notes", "")),
                "drive_minutes_to_next_base": _safe_float(row.get("drive_minutes_to_next_base", 0.0)),
                "segment_index": int(_safe_float(row.get("segment_index", row.get("day", 0)), 0.0)),
                "route_sequence_index": route_sequence_index,
                "display_sequence_index": display_sequence_index,
                "selected_stop_index": display_sequence_index,
                "node_kind": "selected_poi" if selected_poi else "candidate_poi",
                "is_selected_poi": bool(selected_poi),
                "is_hotel_node": False,
                "is_airport_endpoint": False,
                "allowed_cities": _safe_str(row.get("allowed_cities", "")),
                "route_start_kind": _safe_str(row.get("route_start_kind", "")),
                "route_end_kind": _safe_str(row.get("route_end_kind", "")),
                "route_start_city": _safe_str(row.get("route_start_city", "")),
                "route_end_city": _safe_str(row.get("route_end_city", "")),
                "route_start_airport_code": _safe_str(row.get("route_start_airport_code", "")),
                "route_end_airport_code": _safe_str(row.get("route_end_airport_code", "")),
                "route_start_name": _safe_str(row.get("route_start_name", "")),
                "route_start_lat": _safe_float(row.get("route_start_latitude", 0.0)),
                "route_start_lon": _safe_float(row.get("route_start_longitude", 0.0)),
                "route_end_name": _safe_str(row.get("route_end_name", "")),
                "route_end_lat": _safe_float(row.get("route_end_latitude", 0.0)),
                "route_end_lon": _safe_float(row.get("route_end_longitude", 0.0)),
                "sequence_violation_flag": _safe_bool(row.get("sequence_violation_flag", False)),
                "sequence_violation_reason": _safe_str(row.get("sequence_violation_reason", "")),
                "reason_selected": _safe_str(row.get("reason_selected", "")),
                "why_selected": _why_selected(row),
                "why_not_selected": _safe_str(row.get("why_not_selected", "")),
                "description": _point_description(row),
                "image_url": _safe_str(row.get("image_url", "")),
                "website_url": _safe_str(row.get("website_url", "")),
                "source_url": _safe_str(row.get("source_url", row.get("website_url", ""))),
                "detail_source": _safe_str(row.get("detail_source", "")),
                "expected_duration_minutes": expected_duration,
            }
        )
    return records


def _airport_endpoint_from_point(point: dict[str, Any], *, side: str) -> dict[str, Any] | None:
    lat = _safe_float(point.get(f"route_{side}_lat", 0.0))
    lon = _safe_float(point.get(f"route_{side}_lon", 0.0))
    code = _safe_str(point.get(f"route_{side}_airport_code", ""))
    name = _safe_str(point.get(f"route_{side}_name", ""))
    if not (lat and lon and code):
        return None
    city_lookup = {
        "SFO": "San Francisco",
        "LAX": "Los Angeles",
    }
    return {
        "name": name or code,
        "city": city_lookup.get(code, _safe_str(point.get("city", ""))),
        "city_or_anchor": name or code,
        "lat": lat,
        "lon": lon,
        "day": point.get("day", 0) if side == "start" else point.get("trip_days", point.get("day", 0)),
        "stop_order": 0 if side == "start" else 999,
        "type": "airport_endpoint",
        "category": "airport_endpoint",
        "description": f"{name or code} is the {'starting' if side == 'start' else 'ending'} airport endpoint for this route.",
        "why_selected": "Airport endpoint from the configured trip gateway; not counted as a selected POI.",
        "expected_duration_minutes": 0,
        "weather_risk": 0.0,
        "display_utility": 0.0,
        "final_poi_value": 0.0,
        "interest_adjusted_value": 0.0,
        "source_confidence": 1.0,
        "is_route_endpoint": True,
        "route_endpoint_side": side,
        "airport_code": code,
        "route_sequence_index": 0,
        "display_sequence_index": code,
        "selected_stop_index": 0,
        "node_kind": "airport",
        "is_selected_poi": False,
        "is_hotel_node": False,
        "is_airport_endpoint": True,
    }


def _route_node_key(point: dict[str, Any]) -> tuple[str, str, int, int]:
    return (
        _safe_str(point.get("node_kind", point.get("type", ""))).lower(),
        _safe_str(point.get("name", point.get("hotel_name", ""))).lower(),
        int(round(_safe_float(point.get("lat", 0.0)) * 10000)),
        int(round(_safe_float(point.get("lon", 0.0)) * 10000)),
    )


def _append_unique_route_node(nodes: list[dict[str, Any]], node: dict[str, Any] | None) -> None:
    if not node or not node.get("lat") or not node.get("lon"):
        return
    if nodes and _route_node_key(nodes[-1]) == _route_node_key(node):
        return
    nodes.append(node)


def _hotel_node_from_point(point: dict[str, Any], *, side: str) -> dict[str, Any] | None:
    lat = _safe_float(point.get(f"route_{side}_lat", 0.0))
    lon = _safe_float(point.get(f"route_{side}_lon", 0.0))
    name = _safe_str(point.get(f"route_{side}_name", ""))
    kind = _safe_str(point.get(f"route_{side}_kind", "")).lower()
    if point.get(f"route_{side}_airport_code") or "airport" in kind:
        return None
    if side == "end" and (not name or not lat or not lon):
        name = _safe_str(point.get("hotel_name", ""))
        lat = _safe_float(point.get("hotel_lat", 0.0))
        lon = _safe_float(point.get("hotel_lon", 0.0))
        kind = "hotel"
    if not name or not lat or not lon or _is_placeholder_hotel_name(name):
        return None
    city = _safe_str(point.get(f"route_{side}_city", point.get("overnight_city") or point.get("city")))
    return {
        "name": name,
        "hotel_name": name,
        "city": city,
        "city_or_anchor": city or name,
        "lat": lat,
        "lon": lon,
        "day": point.get("day", 0),
        "stop_order": -1 if side == "start" else 998,
        "category": "selected_hotel",
        "type": "selected_hotel",
        "description": f"{name} is the selected overnight/base lodging node for {city or 'this route segment'}.",
        "why_selected": "Selected overnight/base hotel from the optimizer route segment; not counted as a selected POI.",
        "expected_duration_minutes": 0,
        "weather_risk": _safe_float(point.get("weather_risk", 0.0)),
        "display_utility": 0.0,
        "final_poi_value": 0.0,
        "interest_adjusted_value": 0.0,
        "source_confidence": _safe_float(point.get("source_confidence", 0.5), 0.5),
        "route_sequence_index": 0,
        "display_sequence_index": "H",
        "selected_stop_index": 0,
        "node_kind": "hotel",
        "is_route_endpoint": False,
        "is_selected_poi": False,
        "is_hotel_node": True,
        "is_airport_endpoint": False,
        "route_endpoint_side": side,
    }


def _playback_points_with_endpoints(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not points:
        return []
    playback: list[dict[str, Any]] = []
    ordered_points = list(points)
    start_endpoint = _airport_endpoint_from_point(ordered_points[0], side="start")
    end_endpoint = _airport_endpoint_from_point(ordered_points[-1], side="end")
    _append_unique_route_node(playback, start_endpoint)
    days = sorted({int(_safe_float(point.get("day", 0), 0.0)) for point in ordered_points})
    if not days:
        days = [0]
    for day in days:
        day_points = [
            point for point in ordered_points if int(_safe_float(point.get("day", 0), 0.0)) == day
        ]
        if not day_points:
            continue
        _append_unique_route_node(playback, _hotel_node_from_point(day_points[0], side="start"))
        for point in day_points:
            _append_unique_route_node(playback, point)
        _append_unique_route_node(playback, _hotel_node_from_point(day_points[-1], side="end"))
    _append_unique_route_node(playback, end_endpoint)
    return playback


def _project_cache_dir(output_dir: Path) -> Path:
    return output_dir.parent / "cache" if output_dir.name == "outputs" else output_dir / "cache"


def _cached_route_geometry(
    points: list[dict[str, Any]], output_dir: Path
) -> tuple[list[list[float]], str, bool]:
    """Return cached OSRM geometry in GeoJSON lon/lat order, or straight fallback."""
    coordinates = [
        [float(point["lat"]), float(point["lon"])]
        for point in points
        if point.get("lat") and point.get("lon")
    ]
    straight = [[lon, lat] for lat, lon in coordinates]
    if len(coordinates) < 2:
        return straight, "not_enough_points", False
    cache_key = hashlib.sha1(json.dumps(coordinates).encode("utf-8")).hexdigest()[:16]
    cache_path = _project_cache_dir(output_dir) / f"open_osrm_route_{cache_key}.json"
    if not cache_path.exists():
        scenic = _california_corridor_geometry(points)
        if len(scenic) >= 2:
            return scenic, "scenic_corridor_fallback_no_cached_osrm", False
        return straight, "straight_line_fallback_no_cached_osrm", False
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        latlon = payload.get("latlon_geometry", [])
        routed = [
            [float(lon), float(lat)]
            for lat, lon in latlon
            if lat is not None and lon is not None
        ]
        if len(routed) >= 2:
            return routed, str(payload.get("status", "cached_osrm")), True
    except Exception:
        pass
    scenic = _california_corridor_geometry(points)
    if len(scenic) >= 2:
        return scenic, "scenic_corridor_fallback_invalid_cached_osrm", False
    return straight, "straight_line_fallback_invalid_cached_osrm", False


def _california_corridor_geometry(points: list[dict[str, Any]]) -> list[list[float]]:
    """Road-ish fallback corridors for the California nature demo when OSRM cache is absent."""
    anchors: dict[str, tuple[float, float]] = {
        "san francisco": (37.7749, -122.4194),
        "oakdale": (37.7666, -120.8472),
        "mariposa": (37.4849, -119.9663),
        "yosemite valley": (37.7456, -119.5936),
        "fresno": (36.7378, -119.7871),
        "visalia": (36.3302, -119.2921),
        "sequoia national park": (36.4864, -118.5658),
        "bakersfield": (35.3733, -119.0187),
        "lancaster": (34.6868, -118.1542),
        "palm springs": (33.8303, -116.5453),
        "joshua tree national park": (33.8734, -115.9010),
        "los angeles": (34.0522, -118.2437),
        "santa barbara": (34.4208, -119.6982),
        "san luis obispo": (35.2828, -120.6596),
        "hearst castle": (35.6852, -121.1666),
        "big sur": (36.2704, -121.8081),
        "monterey": (36.6002, -121.8947),
        "soledad": (36.4247, -121.3263),
        "pinnacles national park": (36.4915, -121.1825),
    }
    corridor: dict[tuple[str, str], list[str]] = {
        ("san francisco", "yosemite valley"): ["oakdale", "mariposa"],
        ("yosemite valley", "sequoia national park"): ["fresno", "visalia"],
        ("sequoia national park", "joshua tree national park"): ["bakersfield", "lancaster", "palm springs"],
        ("joshua tree national park", "santa barbara"): ["palm springs", "los angeles"],
        ("santa barbara", "big sur"): ["san luis obispo", "hearst castle"],
        ("big sur", "pinnacles national park"): ["monterey", "soledad"],
        ("san francisco", "pinnacles national park"): ["monterey", "soledad"],
        ("pinnacles national park", "big sur"): ["soledad", "monterey"],
        ("big sur", "santa barbara"): ["hearst castle", "san luis obispo"],
        ("santa barbara", "joshua tree national park"): ["los angeles", "palm springs"],
        ("joshua tree national park", "sequoia national park"): ["palm springs", "lancaster", "bakersfield"],
        ("sequoia national park", "yosemite valley"): ["visalia", "fresno"],
    }

    def key(point: dict[str, Any]) -> str:
        text = f"{point.get('name', '')} {point.get('city', '')} {point.get('nature_region', '')}".lower()
        if "yosemite" in text:
            return "yosemite valley"
        if "sequoia" in text:
            return "sequoia national park"
        if "joshua" in text:
            return "joshua tree national park"
        if "pinnacles" in text:
            return "pinnacles national park"
        if "big sur" in text:
            return "big sur"
        if "santa barbara" in text:
            return "santa barbara"
        if "san francisco" in text:
            return "san francisco"
        return ""

    coords: list[list[float]] = []
    for left, right in zip(points, points[1:], strict=False):
        left_key = key(left)
        right_key = key(right)
        segment = [[float(left["lon"]), float(left["lat"])]]
        for waypoint in corridor.get((left_key, right_key), []):
            lat, lon = anchors[waypoint]
            segment.append([lon, lat])
        segment.append([float(right["lon"]), float(right["lat"])])
        for coord in segment:
            if not coords or coords[-1] != coord:
                coords.append(coord)
    return coords


def _reorder_california_nature_points(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Prefer the demo route order SF -> Yosemite -> Sierra -> desert -> coast when those stops exist."""
    if len(points) < 4:
        return points

    def bucket(point: dict[str, Any]) -> int:
        text = f"{point.get('name', '')} {point.get('city', '')} {point.get('nature_region', '')}".lower()
        if "san francisco" in text:
            return 0
        if "yosemite" in text:
            return 1
        if "sequoia" in text or "kings canyon" in text:
            return 2
        if "joshua" in text or "palm springs" in text:
            return 3
        if "santa barbara" in text or "los angeles" in text:
            return 4
        if "big sur" in text or "san luis obispo" in text or "hearst" in text:
            return 5
        if "pinnacles" in text or "monterey" in text:
            return 6
        return 50

    buckets = {bucket(point) for point in points}
    if not {0, 1}.issubset(buckets):
        return points
    ordered = sorted(enumerate(points), key=lambda item: (bucket(item[1]), item[0]))
    output = []
    for index, (_, point) in enumerate(ordered, start=1):
        updated = dict(point)
        updated["day"] = index
        updated["stop_order"] = index
        output.append(updated)
    return output


def _route_geojson(
    points: list[dict[str, Any]],
    route_label: str = "Selected route",
    *,
    draw_line: bool = True,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    features = []
    for index, point in enumerate(points, start=1):
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [point["lon"], point["lat"]]},
                "properties": {**point, "stop_order": point.get("stop_order") or index},
            }
        )
    if draw_line and len(points) >= 2:
        line_points = _playback_points_with_endpoints(points)
        if output_dir is not None:
            coordinates, geometry_source, road_aligned = _cached_route_geometry(line_points, output_dir)
        else:
            coordinates = [[point["lon"], point["lat"]] for point in line_points]
            geometry_source = "straight_line_fallback_no_output_dir"
            road_aligned = False
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": coordinates},
                "properties": {
                    "name": route_label,
                    "role": "selected_route",
                    "geometry_source": geometry_source,
                    "road_aligned": road_aligned,
                },
            }
        )
    return {"type": "FeatureCollection", "features": features}


def _add_dashboard_record(
    records: list[dict[str, Any]],
    route_geojsons: dict[str, dict[str, Any]],
    route_pois: dict[str, list[dict[str, Any]]],
    record: dict[str, Any],
    points: list[dict[str, Any]],
    output_dir: Path | None = None,
) -> None:
    if not points:
        return
    route_id = str(record["id"])
    if route_id in route_geojsons:
        return
    records.append(record)
    route_geojsons[route_id] = _route_geojson(
        points,
        route_label=str(record.get("label", route_id)),
        draw_line=not bool(record.get("marker_only")),
        output_dir=output_dir,
    )
    route_pois[route_id] = points


def _add_grouped_route_records(
    *,
    frame: pd.DataFrame,
    output_dir: Path,
    group_column: str,
    records: list[dict[str, Any]],
    route_geojsons: dict[str, dict[str, Any]],
    route_pois: dict[str, list[dict[str, Any]]],
    family: str,
    selector_group: str,
    id_prefix: str,
    max_records: int = 16,
    skip_default_method: bool = False,
    configured_days: int | None = None,
) -> None:
    if frame.empty or group_column not in frame.columns:
        return
    grouped = frame.groupby(frame[group_column].fillna("").astype(str), sort=False)
    for group_value, group in list(grouped)[:max_records]:
        group = _sort_route_rows(group)
        if group.empty:
            continue
        method = _first_nonempty(group, ["method"])
        trip_days = _route_trip_days(group)
        if skip_default_method and method == DEFAULT_DASHBOARD_METHOD and trip_days == configured_days:
            continue
        points = _point_records(group)
        if not points:
            continue
        label = _default_route_label(group, fallback=str(group_value))
        route_id = f"{id_prefix}__{_slug(group_value or label)}"
        record = _route_record(
            route_id,
            label,
            default=False,
            optional=True,
            family=family,
            selector_group=selector_group,
            route_df=group,
        )
        _add_dashboard_record(records, route_geojsons, route_pois, record, points, output_dir=output_dir)


def _city_coordinate_lookup(*frames: pd.DataFrame) -> dict[str, tuple[float, float]]:
    lookup: dict[str, tuple[float, float]] = {
        "san francisco": (37.7749, -122.4194),
        "santa cruz": (36.9741, -122.0308),
        "monterey": (36.6002, -121.8947),
        "san luis obispo": (35.2828, -120.6596),
        "santa barbara": (34.4208, -119.6982),
        "los angeles": (34.0522, -118.2437),
        "mariposa": (37.4849, -119.9663),
        "yosemite valley": (37.7456, -119.5936),
    }
    coordinate_sets = [
        ("city", "latitude", "longitude"),
        ("overnight_city", "hotel_latitude", "hotel_longitude"),
        ("from", "from_latitude", "from_longitude"),
        ("to", "to_latitude", "to_longitude"),
    ]
    for frame in frames:
        if frame is None or frame.empty:
            continue
        for name_col, lat_col, lon_col in coordinate_sets:
            if not {name_col, lat_col, lon_col}.issubset(frame.columns):
                continue
            for _, row in frame.iterrows():
                name = str(row.get(name_col, "")).strip().lower()
                lat = _safe_float(row.get(lat_col))
                lon = _safe_float(row.get(lon_col))
                if name and lat and lon:
                    lookup.setdefault(name, (lat, lon))
        if {"city", "latitude", "longitude"}.issubset(frame.columns):
            for city, group in frame.groupby(frame["city"].fillna("").astype(str)):
                city_key = city.strip().lower()
                if not city_key or city_key in lookup:
                    continue
                row = group.iloc[0]
                lat = _safe_float(row.get("latitude"))
                lon = _safe_float(row.get("longitude"))
                if lat and lon:
                    lookup[city_key] = (lat, lon)
    return lookup


def _context_route_records(
    *,
    output_dir: Path,
    lookup: dict[str, tuple[float, float]],
    records: list[dict[str, Any]],
    route_geojsons: dict[str, dict[str, Any]],
    route_pois: dict[str, list[dict[str, Any]]],
) -> None:
    legs = _read_csv_if_present(output_dir / "production_intercity_legs.csv")
    if legs.empty or "route_layer" not in legs.columns:
        return
    for layer_name, group in list(legs.groupby(legs["route_layer"].fillna("context").astype(str), sort=False))[:8]:
        group = group.sort_values("leg_order") if "leg_order" in group.columns else group
        city_sequence: list[str] = []
        for _, row in group.iterrows():
            for column in ["from", "to"]:
                city = str(row.get(column, "")).strip()
                if city and (not city_sequence or city_sequence[-1] != city):
                    city_sequence.append(city)
        points = []
        for city in city_sequence:
            coords = lookup.get(city.lower())
            if not coords:
                continue
            points.append(
                {
                    "name": city,
                    "city": city,
                    "lat": coords[0],
                    "lon": coords[1],
                    "category": "context_route",
                    "nature_region": "",
                    "interest_adjusted_value": 0.0,
                    "interest_fit": 0.0,
                    "interest_delta": 0.0,
                    "nature_score": 0.0,
                    "scenic_score": 0.0,
                    "weather_sensitivity": 0.0,
                    "seasonality_risk": 0.0,
                    "park_type": "",
                    "reason_selected": "context route leg",
                    "why_not_selected": "",
                }
            )
        if len(points) < 2:
            continue
        route_id = f"context__{_slug(layer_name)}"
        record = _route_record(
            route_id,
            f"Context · {layer_name}",
            default=False,
            optional=True,
            family="context",
            selector_group="context",
            extra={"quick_groups": ["context"]},
        )
        _add_dashboard_record(records, route_geojsons, route_pois, record, points, output_dir=output_dir)


def _candidate_record_from_csv(
    *,
    output_dir: Path,
    filename: str,
    route_id: str,
    label: str,
    family: str,
    selector_group: str,
    records: list[dict[str, Any]],
    route_geojsons: dict[str, dict[str, Any]],
    route_pois: dict[str, list[dict[str, Any]]],
    max_points: int = 120,
    nature_only: bool = False,
) -> None:
    frame = _read_csv_if_present(output_dir / filename)
    if frame.empty:
        return
    if "itinerary_eligible" in frame.columns:
        frame = frame[frame["itinerary_eligible"].fillna(True).astype(str).str.lower().isin({"1", "true", "yes"})].copy()
    if nature_only:
        mask = pd.Series(False, index=frame.index)
        for column in ["is_nature", "is_national_park", "is_state_park", "is_protected_area", "is_scenic_viewpoint"]:
            if column in frame.columns:
                mask = mask | frame[column].astype(str).str.lower().isin({"1", "true", "yes"})
        if "nature_score" in frame.columns:
            mask = mask | (pd.to_numeric(frame["nature_score"], errors="coerce").fillna(0.0) >= 0.35)
        frame = frame[mask].copy()
        sort_cols = [
            column for column in ["interest_adjusted_value", "nature_score", "scenic_score"] if column in frame.columns
        ]
        if sort_cols:
            frame = frame.sort_values(sort_cols, ascending=False)
    elif "candidate_rank" in frame.columns:
        frame = frame.sort_values("candidate_rank")
    points = _point_records(frame.head(max_points), selected_poi=False)
    if not points:
        return
    record = _route_record(
        route_id,
        label,
        default=False,
        optional=True,
        family=family,
        selector_group=selector_group,
        extra={"quick_groups": [family, selector_group]},
    )
    _add_dashboard_record(records, route_geojsons, route_pois, record, points, output_dir=output_dir)


def _debug_summary(output_dir: Path) -> dict[str, Any]:
    frame = _read_csv_if_present(output_dir / "production_map_route_debug.csv")
    if frame.empty:
        return {"available": False, "rows": 0, "message": "No production_map_route_debug.csv artifact found."}
    summary: dict[str, Any] = {
        "available": True,
        "rows": int(len(frame)),
        "columns": list(frame.columns[:12]),
    }
    for column in ["route_key", "layer_group", "selector_parent", "method", "trip_days"]:
        if column in frame.columns:
            summary[f"{column}_counts"] = frame[column].fillna("missing").astype(str).value_counts().head(8).to_dict()
    return summary


def _interest_preview(output_dir: Path) -> dict[str, Any]:
    path = output_dir / "production_interest_bar_preview.json"
    if not path.exists() or path.stat().st_size == 0:
        return {"available": False, "message": "No production_interest_bar_preview.json artifact found."}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"available": False, "message": f"Could not read interest preview: {exc}"}
    if isinstance(data, dict):
        data = dict(data)
        data.setdefault("available", True)
        return data
    return {"available": True, "items": data}


def _playback_data(route_records: list[dict[str, Any]], route_pois: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    routes = {}
    for record in route_records:
        if not record.get("playable"):
            continue
        route_id = str(record["id"])
        stops = _playback_points_with_endpoints(route_pois.get(route_id, []))
        playable = [
            {
                "name": point.get("name", "Stop"),
                "city": point.get("city", ""),
                "city_or_anchor": point.get("city_or_anchor", point.get("city", "")),
                "lat": point.get("lat"),
                "lon": point.get("lon"),
                "day": point.get("day", 0),
                "stop_order": point.get("stop_order", index + 1),
                "route_sequence_index": point.get("route_sequence_index", 0),
                "display_sequence_index": point.get("display_sequence_index", point.get("route_sequence_index", "")),
                "selected_stop_index": point.get("selected_stop_index", point.get("route_sequence_index", 0)),
                "node_kind": point.get("node_kind", "selected_poi"),
                "category": point.get("category", ""),
                "type": point.get("type", point.get("category", "")),
                "nature_region": point.get("nature_region", ""),
                "hotel_name": point.get("hotel_name", ""),
                "description": point.get("description", ""),
                "why_selected": point.get("why_selected", point.get("reason_selected", "")),
                "expected_duration_minutes": point.get("expected_duration_minutes", 0),
                "weather_risk": point.get("weather_risk", 0.0),
                "source_confidence": point.get("source_confidence", 0.0),
                "source_list": point.get("source_list", ""),
                "nature_score": point.get("nature_score", 0.0),
                "scenic_score": point.get("scenic_score", 0.0),
                "weather_sensitivity": point.get("weather_sensitivity", 0.0),
                "interest_adjusted_value": point.get("interest_adjusted_value", 0.0),
                "final_poi_value": point.get("final_poi_value", 0.0),
                "display_utility": point.get("display_utility", point.get("final_poi_value", 0.0)),
                "optimization_value_source": point.get("optimization_value_source", "final_poi_value"),
                "image_url": point.get("image_url", ""),
                "website_url": point.get("website_url", ""),
                "source_url": point.get("source_url", ""),
                "detail_source": point.get("detail_source", ""),
                "is_route_endpoint": bool(point.get("is_route_endpoint", False)),
                "is_selected_poi": bool(point.get("is_selected_poi", False)),
                "is_hotel_node": bool(point.get("is_hotel_node", False)),
                "is_airport_endpoint": bool(point.get("is_airport_endpoint", point.get("is_route_endpoint", False))),
                "route_endpoint_side": point.get("route_endpoint_side", ""),
                "airport_code": point.get("airport_code", ""),
            }
            for index, point in enumerate(stops)
            if point.get("lat") and point.get("lon")
        ]
        routes[route_id] = {
            "id": route_id,
            "label": record.get("label", route_id),
            "family": record.get("family", ""),
            "default": bool(record.get("default")),
            "playable": True,
            "stops": playable,
        }
    return {"available": True, "routes": routes}


def _city_details(
    output_dir: Path, route_records: list[dict[str, Any]], route_pois: dict[str, list[dict[str, Any]]]
) -> dict[str, Any]:
    city_map: dict[str, dict[str, Any]] = {}
    hotel_choices = _hotel_choices(output_dir)
    hotel_counts = {
        str(city): len(candidates or [])
        for city, candidates in (hotel_choices.get("cities", {}) if isinstance(hotel_choices, dict) else {}).items()
    }
    for record in route_records:
        for point in route_pois.get(str(record["id"]), []):
            if point.get("is_route_endpoint"):
                continue
            city = _safe_str(point.get("city") or point.get("overnight_city"))
            if not city:
                continue
            entry = city_map.setdefault(
                city,
                {
                    "city": city,
                    "route_ids": set(),
                    "days": set(),
                    "selected_stops": [],
                    "hotels": set(),
                    "hotel_alternative_count": hotel_counts.get(city, 0),
                    "selected_hotel": "",
                    "nature_score": 0.0,
                    "scenic_score": 0.0,
                    "lat": _safe_float(point.get("lat", 0.0)),
                    "lon": _safe_float(point.get("lon", 0.0)),
                },
            )
            entry["route_ids"].add(record["id"])
            if point.get("day"):
                entry["days"].add(int(point.get("day", 0)))
            if point.get("hotel_name"):
                entry["hotels"].add(point["hotel_name"])
                if not entry.get("selected_hotel") and record.get("playable"):
                    entry["selected_hotel"] = point["hotel_name"]
            if record.get("playable") and len(entry["selected_stops"]) < 10:
                entry["selected_stops"].append(
                    {
                        "name": point.get("name", "Stop"),
                        "day": point.get("day", 0),
                        "category": point.get("category", ""),
                        "lat": point.get("lat"),
                        "lon": point.get("lon"),
                        "route_label": record.get("label", ""),
                        "route_family": record.get("family", ""),
                        "utility": point.get("display_utility", point.get("final_poi_value", 0.0)),
                        "optimization_value_source": point.get("optimization_value_source", "final_poi_value"),
                        "source_list": point.get("source_list", ""),
                    }
                )
            entry["nature_score"] = max(float(entry["nature_score"]), _safe_float(point.get("nature_score", 0.0)))
            entry["scenic_score"] = max(float(entry["scenic_score"]), _safe_float(point.get("scenic_score", 0.0)))

    legs_frame = _read_csv_if_present(output_dir / "production_intercity_legs.csv")
    legs = []
    if not legs_frame.empty:
        for _, row in legs_frame.head(80).iterrows():
            legs.append(
                {
                    "route_layer": _safe_str(row.get("route_layer", "")),
                    "leg_order": int(_safe_float(row.get("leg_order", 0), 0.0)),
                    "from": _safe_str(row.get("from", row.get("from_name", ""))),
                    "to": _safe_str(row.get("to", row.get("to_name", ""))),
                    "estimated_drive_minutes": _safe_float(row.get("estimated_drive_minutes", 0.0)),
                    "route_type": _safe_str(row.get("route_type", "")),
                    "geometry_source": _safe_str(row.get("geometry_source", "")),
                }
            )
    cities = []
    for entry in city_map.values():
        cities.append(
            {
                **entry,
                "route_ids": sorted(entry["route_ids"]),
                "days": sorted(entry["days"]),
                "hotels": sorted(entry["hotels"]),
                "hotel_alternative_count": int(entry.get("hotel_alternative_count") or hotel_counts.get(entry["city"], 0)),
            }
        )
    cities.sort(key=lambda item: (min(item["days"]) if item["days"] else 999, item["city"]))
    return {"available": bool(cities), "cities": cities, "intercity_legs": legs}


def _hotel_choices(output_dir: Path) -> dict[str, Any]:
    frame = _read_csv_if_present(output_dir / "production_hotel_selection_debug.csv")
    if frame.empty:
        return {"available": False, "cities": {}, "message": "No production_hotel_selection_debug.csv artifact found."}
    cities: dict[str, list[dict[str, Any]]] = {}
    sort_cols = [column for column in ["city", "candidate_rank"] if column in frame.columns]
    if sort_cols:
        frame = frame.sort_values(sort_cols)
    for _, row in frame.iterrows():
        city = _safe_str(row.get("city", "Unknown city")) or "Unknown city"
        hotel_name = _safe_str(row.get("hotel_name", "Hotel"))
        if _is_placeholder_hotel_name(hotel_name):
            continue
        cities.setdefault(city, []).append(
            {
                "city": city,
                "hotel_name": hotel_name,
                "candidate_rank": int(_safe_float(row.get("candidate_rank", 0), 0.0)),
                "selected": _safe_bool(row.get("selected", False)),
                "selected_for_city": _safe_str(row.get("selected_for_city", "")),
                "lat": _safe_float(row.get("latitude", 0.0)),
                "lon": _safe_float(row.get("longitude", 0.0)),
                "source": _safe_str(row.get("source", "")),
                "hotel_score": _safe_float(row.get("hotel_score", 0.0)),
                "rating_component": _safe_float(row.get("rating_component", 0.0)),
                "mean_distance_to_selected_stops_km": _safe_float(row.get("mean_distance_to_selected_stops_km", 0.0)),
                "mean_distance_to_must_go_km": _safe_float(row.get("mean_distance_to_must_go_km", 0.0)),
                "selected_hotel_reason": _safe_str(row.get("selected_hotel_reason", "")),
            }
        )
    return {"available": bool(cities), "cities": cities}


def _selected_hotels(points: list[dict[str, Any]], hotel_choices: dict[str, Any]) -> dict[str, Any]:
    """Selected hotel markers come from debug-selected real hotels, then route-stop lodging nodes."""
    hotels: dict[tuple[str, str], dict[str, Any]] = {}

    for city, candidates in (hotel_choices.get("cities", {}) if isinstance(hotel_choices, dict) else {}).items():
        for candidate in candidates:
            if not candidate.get("selected"):
                continue
            hotel_name = _safe_str(candidate.get("hotel_name", ""))
            lat = _safe_float(candidate.get("lat", 0.0))
            lon = _safe_float(candidate.get("lon", 0.0))
            if not hotel_name or _is_placeholder_hotel_name(hotel_name) or not lat or not lon:
                continue
            key = (str(city).lower(), hotel_name.lower())
            hotels.setdefault(
                key,
                {
                    **candidate,
                    "city": city,
                    "name": hotel_name,
                    "lat": lat,
                    "lon": lon,
                    "selected": True,
                    "status": "selected_by_optimizer",
                },
            )

    for point in points:
        hotel_name = _safe_str(point.get("hotel_name", ""))
        lat = _safe_float(point.get("hotel_lat", 0.0))
        lon = _safe_float(point.get("hotel_lon", 0.0))
        city = _safe_str(point.get("overnight_city") or point.get("city"))
        if not hotel_name or _is_placeholder_hotel_name(hotel_name) or not lat or not lon:
            continue
        key = (city.lower(), hotel_name.lower())
        hotels.setdefault(
            key,
            {
                "city": city,
                "hotel_name": hotel_name,
                "name": hotel_name,
                "lat": lat,
                "lon": lon,
                "selected": True,
                "candidate_rank": 1,
                "hotel_score": 0.0,
                "source": "optimized_route_stop",
                "selected_hotel_reason": "selected for the optimized route overnight/base city",
                "status": "selected_by_optimizer",
            },
        )

    items = sorted(hotels.values(), key=lambda item: (str(item.get("city", "")), str(item.get("hotel_name", ""))))
    return {"available": bool(items), "items": items}


def _nature_explore(output_dir: Path) -> dict[str, Any]:
    frame = _read_csv_if_present(output_dir / "production_enriched_poi_catalog.csv")
    if frame.empty:
        return {"available": False, "items": [], "regions": {}, "message": "No enriched POI catalog artifact found."}
    mask = pd.Series(False, index=frame.index)
    for column in [
        "is_nature",
        "is_national_park",
        "is_state_park",
        "is_protected_area",
        "is_scenic_viewpoint",
        "is_hiking",
    ]:
        if column in frame.columns:
            mask = mask | frame[column].astype(str).str.lower().isin({"1", "true", "yes"})
    if "nature_score" in frame.columns:
        mask = mask | (pd.to_numeric(frame["nature_score"], errors="coerce").fillna(0.0) >= 0.35)
    frame = frame[mask].copy()
    if frame.empty:
        return {
            "available": False,
            "items": [],
            "regions": {},
            "message": "No nature candidates met the export filter.",
        }
    sort_cols = [
        column for column in ["interest_adjusted_value", "nature_score", "scenic_score"] if column in frame.columns
    ]
    if sort_cols:
        frame = frame.sort_values(sort_cols, ascending=False)
    items = []
    for _, row in frame.head(180).iterrows():
        final_poi_value = _safe_float(row.get("final_poi_value", 0.0))
        interest_adjusted_value = _safe_float(row.get("interest_adjusted_value", 0.0))
        display_utility = interest_adjusted_value if interest_adjusted_value > 0 else final_poi_value
        items.append(
            {
                "name": _safe_str(row.get("name", "Nature stop")),
                "city": _safe_str(row.get("city", "")),
                "lat": _safe_float(row.get("latitude", 0.0)),
                "lon": _safe_float(row.get("longitude", 0.0)),
                "category": _safe_str(row.get("category", "")),
                "source_list": _safe_str(row.get("source_list", "")),
                "is_national_park": _safe_bool(row.get("is_national_park", False)),
                "is_state_park": _safe_bool(row.get("is_state_park", False)),
                "is_protected_area": _safe_bool(row.get("is_protected_area", False)),
                "is_scenic_viewpoint": _safe_bool(row.get("is_scenic_viewpoint", False)),
                "is_hiking": _safe_bool(row.get("is_hiking", False)),
                "nature_score": _safe_float(row.get("nature_score", 0.0)),
                "scenic_score": _safe_float(row.get("scenic_score", 0.0)),
                "hiking_score": _safe_float(row.get("hiking_score", 0.0)),
                "outdoor_intensity": _safe_float(row.get("outdoor_intensity", 0.0)),
                "weather_sensitivity": _safe_float(row.get("weather_sensitivity", 0.0)),
                "seasonality_risk": _safe_float(row.get("seasonality_risk", 0.0)),
                "park_type": _safe_str(row.get("park_type", "")),
                "nature_region": _safe_str(row.get("nature_region", "")),
                "interest_fit": _safe_float(row.get("interest_fit", 0.0)),
                "park_bonus": _safe_float(row.get("park_bonus", 0.0)),
                "final_poi_value": final_poi_value,
                "interest_adjusted_value": interest_adjusted_value,
                "display_utility": display_utility,
                "city_score": _safe_float(row.get("city_score", 0.0)),
                "culture_score": _safe_float(row.get("culture_score", 0.0)),
                "history_score": _safe_float(row.get("history_score", 0.0)),
            }
        )
    regions: dict[str, dict[str, Any]] = {}
    for item in items:
        region_name = item["nature_region"] or item["city"] or "Unassigned"
        region = regions.setdefault(region_name, {"count": 0, "max_nature_score": 0.0, "max_scenic_score": 0.0})
        region["count"] += 1
        region["max_nature_score"] = max(region["max_nature_score"], item["nature_score"])
        region["max_scenic_score"] = max(region["max_scenic_score"], item["scenic_score"])
    return {
        "available": True,
        "items": items,
        "regions": regions,
        "filters": ["national_park", "state_or_protected", "scenic_viewpoint", "hiking"],
    }


def _anchor_audit_payload(output_dir: Path) -> dict[str, Any]:
    path = output_dir / "production_dataset_completeness_audit.csv"
    if not path.exists():
        path = output_dir / "production_route_anchor_audit.csv"
    frame = _read_csv_if_present(path)
    if frame.empty:
        return {
            "available": False,
            "ready": False,
            "readiness": "missing",
            "message": "No route anchor or dataset completeness audit artifact found.",
            "items": [],
            "required_anchor_count": 0,
            "missing_count": 0,
            "warning_count": 1,
        }
    allowed_roles = {
        "selected_stop",
        "gateway",
        "base_city",
        "candidate_only",
        "context_only",
        "missing",
        "infeasible",
    }
    records: list[dict[str, Any]] = []
    missing_count = 0
    warning_count = 0
    for _, row in frame.iterrows():
        role = _safe_str(row.get("role", row.get("status", ""))).strip() or "missing"
        if role not in allowed_roles:
            role = "missing"
        reason = _safe_str(row.get("reason", "")) or "no_reason_recorded"
        if role == "missing":
            missing_count += 1
        if role in {"missing", "context_only"} or reason == "no_reason_recorded":
            warning_count += 1
        records.append(
            {
                "scenario": _safe_str(row.get("scenario", "")),
                "interest": _safe_str(row.get("interest", "")),
                "anchor": _safe_str(row.get("anchor", "")),
                "candidate": _safe_bool(row.get("candidate", False)),
                "selected": _safe_bool(row.get("selected", False)),
                "role": role,
                "reason": reason,
                "candidate_count": int(_safe_float(row.get("candidate_count", 0.0))),
                "selected_count": int(_safe_float(row.get("selected_count", 0.0))),
                "audited_method": _safe_str(row.get("audited_method", "")),
            }
        )
    ready = missing_count == 0 and all(item["reason"] for item in records)
    return {
        "available": True,
        "ready": ready,
        "readiness": "ready" if ready else "needs_attention",
        "path": path.name,
        "items": records,
        "required_anchor_count": len(records),
        "missing_count": missing_count,
        "warning_count": warning_count,
    }


def _build_dashboard_payloads(
    route_df: pd.DataFrame,
    *,
    output_dir: Path,
    config: TripConfig,
    max_routes: int,
) -> tuple[
    list[dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, list[dict[str, Any]]],
    dict[str, Any],
    list[dict[str, Any]],
    str,
]:
    configured_days = int(config.get("trip", "trip_days", 7))
    selected, route_source, route_label = _default_route_frame(
        route_df, output_dir=output_dir, config=config, max_routes=max_routes
    )
    default_points = _point_records(selected)
    records: list[dict[str, Any]] = []
    route_geojsons: dict[str, dict[str, Any]] = {}
    route_pois: dict[str, list[dict[str, Any]]] = {}
    default_record = _route_record(
        DEFAULT_DASHBOARD_ROUTE_ID,
        route_label,
        default=True,
        optional=False,
        family="selected",
        selector_group="default",
        route_df=selected,
        extra={"source": route_source},
    )
    _add_dashboard_record(records, route_geojsons, route_pois, default_record, default_points, output_dir=output_dir)

    matrix = _read_csv_if_present(output_dir / "production_route_matrix_route_stops.csv")
    _add_grouped_route_records(
        frame=matrix,
        output_dir=output_dir,
        group_column="route_key",
        records=records,
        route_geojsons=route_geojsons,
        route_pois=route_pois,
        family="route_matrix",
        selector_group="route_matrix",
        id_prefix="route_matrix",
        max_records=18,
    )

    trip_lengths = _read_csv_if_present(output_dir / "production_trip_length_route_stops.csv")
    _add_grouped_route_records(
        frame=trip_lengths,
        output_dir=output_dir,
        group_column="trip_days",
        records=records,
        route_geojsons=route_geojsons,
        route_pois=route_pois,
        family="trip_length",
        selector_group="trip_length",
        id_prefix="trip_length",
        max_records=8,
    )

    methods = _read_csv_if_present(output_dir / "production_method_route_stops.csv")
    _add_grouped_route_records(
        frame=methods,
        output_dir=output_dir,
        group_column="method",
        records=records,
        route_geojsons=route_geojsons,
        route_pois=route_pois,
        family="method",
        selector_group="method",
        id_prefix="method",
        max_records=8,
        skip_default_method=True,
        configured_days=configured_days,
    )

    interest_routes = _read_csv_if_present(output_dir / "production_interest_route_stops.csv")
    _add_grouped_route_records(
        frame=interest_routes,
        output_dir=output_dir,
        group_column="interest_profile",
        records=records,
        route_geojsons=route_geojsons,
        route_pois=route_pois,
        family="interest_profile",
        selector_group="interest_profiles",
        id_prefix="interest_profile",
        max_records=8,
    )

    lookup = _city_coordinate_lookup(selected, matrix, trip_lengths, methods, interest_routes, route_df)
    _context_route_records(
        output_dir=output_dir,
        lookup=lookup,
        records=records,
        route_geojsons=route_geojsons,
        route_pois=route_pois,
    )
    _candidate_record_from_csv(
        output_dir=output_dir,
        filename="production_hotel_selection_debug.csv",
        route_id="hotel_candidates",
        label="Hotel candidates",
        family="hotel",
        selector_group="candidates",
        records=records,
        route_geojsons=route_geojsons,
        route_pois=route_pois,
        max_points=80,
    )
    _candidate_record_from_csv(
        output_dir=output_dir,
        filename="production_social_must_go_candidates.csv",
        route_id="must_go_candidates",
        label="Must-go candidates",
        family="must_go",
        selector_group="candidates",
        records=records,
        route_geojsons=route_geojsons,
        route_pois=route_pois,
        max_points=80,
    )
    _candidate_record_from_csv(
        output_dir=output_dir,
        filename="production_enriched_poi_catalog.csv",
        route_id="nature_candidates",
        label="National park candidates",
        family="nature",
        selector_group="candidates",
        records=records,
        route_geojsons=route_geojsons,
        route_pois=route_pois,
        max_points=120,
        nature_only=True,
    )

    anchor_audit = _anchor_audit_payload(output_dir)
    artifact_metadata = read_artifact_metadata(output_dir)
    artifact_fresh = artifact_metadata_matches(output_dir, config)
    metrics = _dashboard_metrics(
        default_points,
        config,
        route_label=route_label,
        route_source=route_source,
        route_method=_first_nonempty(selected, ["method"]),
        route_count=len(records),
        optional_route_count=sum(1 for record in records if record.get("optional")),
        layer_families=sorted({str(record.get("family", "")) for record in records if record.get("family")}),
        anchor_audit=anchor_audit,
        artifact_metadata=artifact_metadata,
        artifact_metadata_fresh=artifact_fresh,
    )
    return records, route_geojsons, route_pois, metrics, default_points, route_label


def _dashboard_metrics(
    points: list[dict[str, Any]],
    config: TripConfig,
    route_label: str = "Selected Route",
    route_source: str = "",
    route_method: str = "",
    route_count: int = 1,
    optional_route_count: int = 0,
    layer_families: list[str] | None = None,
    anchor_audit: dict[str, Any] | None = None,
    artifact_metadata: dict[str, Any] | None = None,
    artifact_metadata_fresh: bool = False,
) -> dict[str, Any]:
    display_values = [float(point.get("display_utility", point.get("final_poi_value", 0.0)) or 0.0) for point in points]
    final_values = [float(point.get("final_poi_value", 0.0) or 0.0) for point in points]
    interest_values = [float(point.get("interest_adjusted_value", 0.0) or 0.0) for point in points]
    interest_enabled = bool(config.get("interest", "enabled", False))
    value_column = "interest_adjusted_value" if interest_enabled else "final_poi_value"
    scenario = str(config.get("trip", "scenario", "california_coast"))
    return {
        "trip_days": int(config.get("trip", "trip_days", 7)),
        "scenario": scenario,
        "scenario_label": get_scenario_definition(scenario).label,
        "interest_profile": str(
            config.get("trip", "interest_profile", config.get("interest", "mode", "balanced_interest"))
        ),
        "selected_stop_count": len(points),
        "display_utility": round(float(sum(display_values)), 4),
        "total_final_poi_value": round(float(sum(final_values)), 4),
        "total_interest_adjusted_value": round(float(sum(interest_values)), 4),
        "interest_adjusted_utility": round(float(sum(display_values)), 4),
        "optimization_value_column": value_column,
        "nature_optimization_enabled": bool(config.get("nature", "enabled", False)) and interest_enabled,
        "default_route_label": route_label,
        "default_route_source": route_source,
        "default_route_method": route_method,
        "route_state_label": "Saved optimized route",
        "preview_state_label": "Preview only - rerun pipeline to save",
        "artifact_metadata_fresh": bool(artifact_metadata_fresh),
        "artifact_contract_version": _safe_str((artifact_metadata or {}).get("artifact_contract_version", "")),
        "artifact_timestamp_utc": _safe_str((artifact_metadata or {}).get("timestamp_utc", "")),
        "audit_ready": bool((anchor_audit or {}).get("ready", False)),
        "audit_readiness": _safe_str((anchor_audit or {}).get("readiness", "missing")),
        "audit_missing_count": int((anchor_audit or {}).get("missing_count", 0) or 0),
        "audit_warning_count": int((anchor_audit or {}).get("warning_count", 0) or 0),
        "anchor_audit_path": _safe_str((anchor_audit or {}).get("path", "")),
        "anchor_audit": anchor_audit or {"available": False, "items": []},
        "route_record_count": int(route_count),
        "optional_record_count": int(optional_route_count),
        "layer_families": layer_families or [],
    }


def _json_payload(data: Any) -> str:
    return json.dumps(_json_safe(data), indent=2, allow_nan=False).replace("</", "<\\/")


def _json_safe(data: Any) -> Any:
    if isinstance(data, dict):
        return {str(key): _json_safe(value) for key, value in data.items()}
    if isinstance(data, list | tuple):
        return [_json_safe(value) for value in data]
    if isinstance(data, set):
        return sorted(_json_safe(value) for value in data)
    if isinstance(data, float):
        return data if math.isfinite(data) else None
    try:
        if pd.isna(data):
            return None
    except Exception:
        pass
    return data


def _write_text_asset(path: Path, text: str, written: list[Path]) -> None:
    path.write_text(text, encoding="utf-8")
    written.append(path)


def _write_json_asset(path: Path, data: Any, written: list[Path]) -> None:
    _write_text_asset(path, _json_payload(data), written)


def _global_assignment(global_name: str, data: Any) -> str:
    return f"window.{global_name} = {_json_payload(data)};\n"


def _global_map_assignment(global_name: str, key: str, data: Any) -> str:
    return (
        f"window.{global_name} = window.{global_name} || {{}};\n"
        f"window.{global_name}[{json.dumps(key)}] = {_json_payload(data)};\n"
    )


def _clear_stale_dashboard_assets(assets: Path, routes_dir: Path, pois_dir: Path) -> None:
    """Remove stale generated dashboard assets before writing the current contract."""
    for directory in [routes_dir, pois_dir]:
        if not directory.exists():
            continue
        for pattern in ["*.geojson", "*.json", "*.js"]:
            for path in directory.glob(pattern):
                try:
                    path.unlink(missing_ok=True)
                except PermissionError:
                    # Some Windows sandbox ACLs allow writing but not deleting generated assets.
                    path.write_text("", encoding="utf-8")
    for name in [
        "route_index.json",
        "route_index.js",
        "dashboard_metrics.json",
        "dashboard_metrics.js",
        "debug_summary.json",
        "debug_summary.js",
        "interest_preview.json",
        "interest_preview.js",
        "playback_data.json",
        "playback_data.js",
        "city_details.json",
        "city_details.js",
        "selected_hotels.json",
        "selected_hotels.js",
        "hotel_choices.json",
        "hotel_choices.js",
        "nature_explore.json",
        "nature_explore.js",
        "evaluation_metrics.json",
        "evaluation_metrics.js",
        "data_loader.js",
        "dashboard.js",
        "map_controls.js",
        "style.css",
    ]:
        path = assets / name
        try:
            path.unlink(missing_ok=True)
        except PermissionError:
            path.write_text("", encoding="utf-8")


def _write_lightweight_map(path: Path, points: list[dict[str, Any]], metrics: dict[str, Any]) -> None:
    center = [points[0]["lat"], points[0]["lon"]] if points else [36.7783, -119.4179]
    share_metrics = {
        key: value
        for key, value in metrics.items()
        if key not in {"route_record_count", "optional_record_count", "layer_families"}
    }
    payload = json.dumps({"points": points, "metrics": share_metrics, "center": center}).replace("</", "<\\/")
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Lightweight Share Map</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
  <style>
    html, body, #map {{ height: 100%; margin: 0; font-family: Inter, system-ui, sans-serif; }}
    .share-panel {{ position: absolute; z-index: 900; top: 18px; left: 18px; width: 280px; background: #fff; border: 1px solid #dbe3ea; box-shadow: 0 12px 30px rgba(15,23,42,.18); border-radius: 8px; padding: 12px; }}
    .share-panel h1 {{ font-size: 16px; margin: 0 0 6px; }}
    .share-panel p {{ color: #475569; font-size: 12px; line-height: 1.4; margin: 4px 0; }}
    .share-warning {{ display: none; position: absolute; z-index: 1000; right: 18px; top: 18px; max-width: 360px; padding: 12px; border-radius: 8px; border: 1px solid #f59e0b; background: #fffbeb; color: #78350f; font-size: 13px; box-shadow: 0 12px 30px rgba(15,23,42,.16); }}
  </style>
</head>
<body>
  <div id="map"></div>
  <div id="share-warning" class="share-warning"></div>
  <section class="share-panel">
    <h1>Nature-Aware Route</h1>
    <p><b>Scenario:</b> {metrics.get("scenario", "")}</p>
    <p><b>Interest:</b> {metrics.get("interest_profile", "")}</p>
    <p><b>Stops:</b> {metrics.get("selected_stop_count", 0)}</p>
  </section>
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <script id="share-map-data" type="application/json">{payload}</script>
  <script>
    function showShareWarning(message) {{
      const warning = document.getElementById('share-warning');
      warning.textContent = message;
      warning.style.display = 'block';
      console.error('[share-map-error]', message);
    }}

    function initShareMap() {{
      const payload = JSON.parse(document.getElementById('share-map-data').textContent);
      if (!window.L) {{
        showShareWarning('Leaflet failed to load. Check your network connection or regenerate the map with local Leaflet assets.');
        return;
      }}
      if (!payload.points || payload.points.length === 0) {{
        showShareWarning('No selected route data found. Regenerate route assets first.');
      }}
      const map = L.map('map', {{ preferCanvas: true }}).setView(payload.center, 7);
      L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{ maxZoom: 18, attribution: '&copy; OpenStreetMap contributors' }}).addTo(map);
      const latLngs = [];
      payload.points.forEach((point, index) => {{
        const latLng = [point.lat, point.lon];
        latLngs.push(latLng);
        L.circleMarker(latLng, {{ radius: 7, color: '#166534', fillColor: '#2A9D8F', fillOpacity: 0.9, weight: 2 }})
          .bindPopup(`<b>${{index + 1}}. ${{point.name}}</b><br>${{point.city}}<br>${{point.nature_region || point.category}}`)
          .addTo(map);
      }});
      if (latLngs.length > 1) {{
        L.polyline(latLngs, {{ color: '#0f766e', weight: 4, opacity: 0.82 }}).addTo(map);
        map.fitBounds(latLngs, {{ padding: [40, 40] }});
      }}
    }}

    window.addEventListener('DOMContentLoaded', initShareMap);
  </script>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")


EVALUATION_METHODS = [
    ("hierarchical_gurobi_pipeline", "Hierarchical Gurobi"),
    ("hierarchical_greedy_baseline", "Hierarchical Greedy"),
    ("hierarchical_bandit_gurobi_repair", "Bandit + Repair"),
]


def _evaluation_stop_mask(frame: pd.DataFrame) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=bool)
    mask = pd.Series(False, index=frame.index)
    for column in ["is_nature", "is_national_park", "is_state_park", "is_protected_area", "is_scenic_viewpoint"]:
        if column in frame.columns:
            mask = mask | frame[column].astype(str).str.lower().isin({"1", "true", "yes"})
    if "category" in frame.columns:
        mask = mask | frame["category"].fillna("").astype(str).str.lower().str.contains(
            "national_park|park|viewpoint|scenic|hiking|nature",
            regex=True,
            na=False,
        )
    if "nature_score" in frame.columns:
        mask = mask | (pd.to_numeric(frame["nature_score"], errors="coerce").fillna(0.0) >= 0.35)
    return mask


def _evaluation_metrics(output_dir: Path) -> dict[str, Any]:
    comparison = _read_csv_if_present(output_dir / "production_method_comparison.csv")
    route_stops = _read_csv_if_present(output_dir / "production_method_route_stops.csv")
    methods: list[dict[str, Any]] = []
    for method_id, default_label in EVALUATION_METHODS:
        comparison_rows = (
            comparison[comparison.get("method", pd.Series(dtype=str)).astype(str).eq(method_id)].copy()
            if not comparison.empty and "method" in comparison.columns
            else pd.DataFrame()
        )
        row = comparison_rows.iloc[0] if not comparison_rows.empty else pd.Series(dtype=object)
        stops = (
            route_stops[route_stops.get("method", pd.Series(dtype=str)).astype(str).eq(method_id)].copy()
            if not route_stops.empty and "method" in route_stops.columns
            else pd.DataFrame()
        )
        nature_mask = _evaluation_stop_mask(stops)
        weather_values = pd.to_numeric(
            stops.get("weather_risk", stops.get("weather_sensitivity", pd.Series(dtype=float))),
            errors="coerce",
        ).fillna(0.0)
        method_label = _safe_str(row.get("method_display_name", "")) or default_label
        methods.append(
            {
                "method": method_id,
                "label": method_label,
                "short_label": default_label,
                "status": _safe_str(row.get("status", "missing" if comparison_rows.empty else "")),
                "comparison_score": _safe_float(row.get("comparison_score", 0.0)),
                "objective": _safe_float(row.get("objective", row.get("total_utility", 0.0))),
                "route_distance_km": _safe_float(row.get("total_travel_distance_km", 0.0)),
                "route_time_minutes": _safe_float(row.get("total_travel_time", 0.0)),
                "nature_score": float(
                    pd.to_numeric(stops.get("nature_score", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum()
                ),
                "scenic_score": float(
                    pd.to_numeric(stops.get("scenic_score", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum()
                ),
                "weather_risk": float(weather_values.mean()) if not weather_values.empty else 0.0,
                "selected_nature_stops": int(nature_mask.sum()) if not nature_mask.empty else 0,
                "selected_stop_count": int(len(stops)) if not stops.empty else int(_safe_float(row.get("selected_attractions", 0.0))),
                "runtime_seconds": _safe_float(row.get("solve_seconds", 0.0)),
                "notes": _safe_str(row.get("notes", "")),
            }
        )
    return {
        "available": bool(methods),
        "source_files": [
            "production_method_comparison.csv",
            "production_method_route_stops.csv",
        ],
        "methods": methods,
        "chart_fields": [
            {"key": "comparison_score", "label": "Comparison score", "higher_is_better": True},
            {"key": "route_distance_km", "label": "Route distance (km)", "higher_is_better": False},
            {"key": "route_time_minutes", "label": "Route time (min)", "higher_is_better": False},
            {"key": "nature_score", "label": "Nature score", "higher_is_better": True},
            {"key": "weather_risk", "label": "Weather risk", "higher_is_better": False},
            {"key": "selected_nature_stops", "label": "Selected nature stops", "higher_is_better": True},
            {"key": "runtime_seconds", "label": "Runtime (sec)", "higher_is_better": False},
        ],
        "tradeoff_explanation": [
            "Hierarchical Gurobi optimizes the saved city/base allocation and local route with the strongest exact-solver signal.",
            "Hierarchical Greedy is fastest and useful as a transparent baseline, but can miss globally valuable nature anchors.",
            "Bandit + Repair explores route strategies, then repairs promising candidates with a small Gurobi pass; this is the default saved dashboard route.",
        ],
    }


def _write_evaluation_page(root: Path, assets: Path, metrics: dict[str, Any], written: list[Path]) -> None:
    metrics_json = assets / "evaluation_metrics.json"
    _write_json_asset(metrics_json, metrics, written)
    metrics_js = assets / "evaluation_metrics.js"
    _write_text_asset(metrics_js, _global_assignment("DASHBOARD_EVALUATION_METRICS", metrics), written)
    page = root / "evaluation.html"
    _write_text_asset(
        page,
        """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Itinerary Method Evaluation</title>
  <style>
    :root { --forest:#0a6b53; --teal:#18b8a5; --line:#d8e5e1; --text:#10233d; --muted:#5b6b7d; --panel:#fff; }
    * { box-sizing: border-box; }
    body { margin: 0; color: var(--text); font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #eef6f5; }
    main { width: min(1180px, calc(100% - 32px)); margin: 0 auto; padding: 24px 0 36px; }
    header { display: flex; align-items: end; justify-content: space-between; gap: 16px; margin-bottom: 18px; }
    h1 { margin: 0; font-size: 28px; line-height: 1.05; }
    p { margin: 6px 0 0; color: var(--muted); line-height: 1.45; }
    a { color: var(--forest); font-weight: 800; text-decoration: none; }
    .grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; }
    .panel { border: 1px solid var(--line); border-radius: 8px; background: var(--panel); box-shadow: 0 10px 28px rgba(14,35,58,.10); padding: 16px; }
    .panel h2 { margin: 0 0 12px; font-size: 15px; text-transform: uppercase; }
    .chart-row { display: grid; grid-template-columns: 170px minmax(0, 1fr) 80px; gap: 10px; align-items: center; margin: 9px 0; font-size: 12px; }
    .bar-track { height: 12px; border-radius: 999px; background: #dbe8e7; overflow: hidden; }
    .bar { height: 100%; border-radius: inherit; background: linear-gradient(90deg, var(--forest), var(--teal)); }
    table { width: 100%; border-collapse: collapse; font-size: 12px; }
    th, td { padding: 9px 8px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; }
    th { color: var(--muted); font-size: 11px; text-transform: uppercase; }
    .wide { grid-column: 1 / -1; }
    .note-list { margin: 0; padding-left: 18px; color: var(--muted); line-height: 1.45; }
    @media (max-width: 860px) { header { display: block; } .grid { grid-template-columns: 1fr; } .chart-row { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <h1>Method Evaluation Dashboard</h1>
        <p>Static comparison from saved Python/Gurobi, greedy, and bandit + repair artifacts.</p>
      </div>
      <a href="index.html">Back to map dashboard</a>
    </header>
    <section id="charts" class="grid" aria-label="Method comparison charts"></section>
    <section class="panel wide" aria-label="Method comparison table">
      <h2>Method Comparison</h2>
      <div id="method-table"></div>
    </section>
    <section class="panel wide" aria-label="Solver tradeoff explanation">
      <h2>Solver Tradeoffs</h2>
      <ul id="tradeoff-list" class="note-list"></ul>
    </section>
  </main>
  <script src="assets/evaluation_metrics.js"></script>
  <script>
    const payload = window.DASHBOARD_EVALUATION_METRICS || { methods: [], chart_fields: [], tradeoff_explanation: [] };
    const methods = payload.methods || [];
    const fields = payload.chart_fields || [];
    const charts = document.getElementById('charts');
    const fmt = value => Number.isFinite(Number(value)) ? Number(value).toFixed(Number(value) >= 100 ? 0 : 2) : 'n/a';
    fields.forEach(field => {
      const values = methods.map(method => Number(method[field.key] || 0));
      const max = Math.max(...values, 0.0001);
      const section = document.createElement('section');
      section.className = 'panel';
      section.innerHTML = `<h2>${field.label}</h2>` + methods.map(method => {
        const value = Number(method[field.key] || 0);
        const width = Math.max(3, Math.min(100, (value / max) * 100));
        return `<div class="chart-row"><strong>${method.short_label || method.label}</strong><div class="bar-track"><div class="bar" style="width:${width}%"></div></div><span>${fmt(value)}</span></div>`;
      }).join('');
      charts.appendChild(section);
    });
    document.getElementById('method-table').innerHTML = `
      <table>
        <thead><tr><th>Method</th><th>Score</th><th>Distance</th><th>Time</th><th>Nature</th><th>Weather</th><th>Runtime</th><th>Status</th></tr></thead>
        <tbody>
          ${methods.map(method => `<tr>
            <td><strong>${method.label}</strong><br>${method.notes || ''}</td>
            <td>${fmt(method.comparison_score)}</td>
            <td>${fmt(method.route_distance_km)} km</td>
            <td>${fmt(method.route_time_minutes)} min</td>
            <td>${fmt(method.nature_score)} (${method.selected_nature_stops || 0} stops)</td>
            <td>${fmt(method.weather_risk)}</td>
            <td>${fmt(method.runtime_seconds)} sec</td>
            <td>${method.status || ''}</td>
          </tr>`).join('')}
        </tbody>
      </table>`;
    document.getElementById('tradeoff-list').innerHTML = (payload.tradeoff_explanation || [])
      .map(item => `<li>${item}</li>`)
      .join('');
  </script>
</body>
</html>
""",
        written,
    )


def _dashboard_page_html(mode: str) -> str:
    default_mode = "customer" if mode == "customer" else "research"
    mode_label = "Customer trip planner" if default_mode == "customer" else "Research/Test dashboard"
    switch_label = "Research/Test mode" if default_mode == "customer" else "Customer mode"
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Weather-Aware Itinerary Dashboard</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet/dist/leaflet.css" />
  <link rel="stylesheet" href="assets/style.css" />
</head>
<body class="dashboard-shell" data-dashboard-mode="{default_mode}">
  <div id="map"></div>
  <div id="diagnostic-panel" class="diagnostic-panel" role="status" aria-live="polite"></div>
  <aside class="weather-chip" aria-label="Current weather preview">
    <span class="weather-sun" aria-hidden="true"></span>
    <span>
      <span class="weather-temp">62°F</span>
      <span class="weather-city">Clear · San Francisco</span>
    </span>
  </aside>
  <aside class="map-side-panels" aria-label="Map context controls">
    <section class="map-card" aria-label="Map layers">
      <h2>Map layers</h2>
      <label><input type="checkbox" checked data-map-layer-toggle="terrain" /> Terrain</label>
      <label><input type="checkbox" checked data-map-layer-toggle="roads" /> Roads</label>
      <label><input type="checkbox" checked data-map-layer-toggle="labels" /> Labels</label>
      <label><input type="checkbox" checked data-map-layer-toggle="weather_risk" /> Weather risk</label>
    </section>
    <section class="map-card" aria-label="Weather risk legend">
      <h2>Weather risk (7-day)</h2>
      <div class="risk-row"><span class="risk-dot risk-low"></span>Low (0 - 0.25)</div>
      <div class="risk-row"><span class="risk-dot risk-mid"></span>Moderate (0.25 - 0.5)</div>
      <div class="risk-row"><span class="risk-dot risk-high"></span>High (0.5 - 0.75)</div>
      <div class="risk-row"><span class="risk-dot risk-very"></span>Very high (0.75 - 1)</div>
    </section>
  </aside>
  <aside class="map-legend" aria-label="Route legend">
    <div class="legend-item"><span class="legend-line"></span>Optimized route</div>
    <div class="legend-item"><span class="legend-line alt"></span>Alternative route</div>
    <div class="legend-item"><span class="legend-hotel">H</span>Selected hotel</div>
    <div class="legend-item"><span class="legend-dot risk-low"></span>Nature / park spot</div>
    <div class="legend-item"><span class="risk-dot risk-high"></span>Hotel candidate</div>
    <div class="legend-item"><span class="legend-star">*</span>Must-see stop</div>
    <div class="legend-item"><span class="legend-line preview"></span>Playback trail</div>
    <div class="legend-item"><span class="legend-dot"></span>Route stop</div>
  </aside>
  <section class="dashboard-panel">
    <div id="dashboard-drag-handle" class="dashboard-header">
      <div class="dashboard-mark" aria-hidden="true">
        <svg viewBox="0 0 48 48" width="30" height="30" focusable="false">
          <path d="M5 34 19 10l8 14 5-8 11 18H5Z" fill="currentColor" opacity=".95"/>
          <path d="M5 38c8-5 16-5 24 0 5 3 10 3 14 0" fill="none" stroke="currentColor" stroke-width="4" stroke-linecap="round"/>
        </svg>
      </div>
      <div class="dashboard-title-block">
        <h1>Weather-Aware Itinerary</h1>
        <div id="dashboard-subtitle" class="dashboard-subtitle">Loading scenario</div>
        <div id="dashboard-mode-label" class="dashboard-mode-label">{mode_label}</div>
      </div>
      <button id="dashboard-mode-toggle" type="button">{switch_label}</button>
      <button id="dashboard-collapse" type="button">Collapse</button>
    </div>
    <div id="dashboard-content">
      <details class="dashboard-section" open data-mode-section="customer">
        <summary>Plan your trip</summary>
        <div id="customer-trip-controls"></div>
      </details>
      <details class="dashboard-section" open>
        <summary>
          <span data-mode-section="customer">Trip summary</span>
          <span data-mode-section="research">Dashboard summary</span>
        </summary>
        <div id="metrics"></div>
        <div class="interest-note" data-mode-section="customer">Customer choices load saved route artifacts or clearly labeled browser previews. They do not overwrite optimizer results.</div>
        <div class="interest-note" data-mode-section="research"><a href="evaluation.html">Open evaluation dashboard</a></div>
      </details>
      <details class="dashboard-section" open data-mode-section="research">
        <summary>Route & layers</summary>
        <div class="filter-row">
          <label><input type="checkbox" data-layer-toggle="default_route" checked /> Saved optimized route</label>
          <label><input type="checkbox" data-layer-toggle="selected_hotels" checked /> Selected hotels</label>
          <label><input type="checkbox" data-layer-toggle="hotel_candidates" /> Hotel candidates</label>
          <label><input type="checkbox" data-layer-toggle="nature_candidates" /> National park candidates</label>
          <label><input type="checkbox" data-layer-toggle="live_preview" /> Preview only route</label>
        </div>
        <div id="quick-actions" class="quick-actions"></div>
        <div id="route-selector" class="route-selector"></div>
      </details>
      <details class="dashboard-section" open>
        <summary>Playback</summary>
        <div class="summary-list"><b>Route:</b> <span id="playback-route-label">Loading...</span></div>
        <div class="playback-controls">
          <button id="playback-restart" type="button" aria-label="Restart route playback">|&lt;</button>
          <button id="playback-prev" type="button" aria-label="Previous stop">&lt;&lt;</button>
          <button id="playback-play" type="button" aria-label="Play route animation">Play</button>
          <button id="playback-pause" type="button" aria-label="Pause route animation">Pause</button>
          <button id="playback-next" type="button" aria-label="Next stop">&gt;&gt;</button>
        </div>
        <div class="playback-controls">
          <select id="playback-speed" aria-label="Playback speed">
            <option value="1600">Slow</option>
            <option value="900" selected>Normal</option>
            <option value="420">Fast</option>
          </select>
          <label><input id="playback-follow" type="checkbox" checked /> Follow marker</label>
        </div>
        <div class="playback-progress"><span id="playback-progress-bar"></span></div>
        <div id="playback-current-stop" class="status"></div>
      </details>
      <details class="dashboard-section" open>
        <summary>Active stop details</summary>
        <div id="active-stop-detail"></div>
      </details>
      <details class="dashboard-section" open>
        <summary>Hotel choices</summary>
        <div id="hotel-choices"></div>
      </details>
      <details class="dashboard-section" open>
        <summary>Interest bars</summary>
        <div id="interest-preview"></div>
      </details>
      <details class="dashboard-section" data-mode-section="research">
        <summary>City details</summary>
        <div id="city-details"></div>
      </details>
      <details class="dashboard-section" data-mode-section="research">
        <summary>Nature explore</summary>
        <div id="nature-explore"></div>
      </details>
      <details class="dashboard-section" data-mode-section="research">
        <summary>Debug summary</summary>
        <div id="debug-summary"></div>
      </details>
      <div id="dashboard-status" class="status"></div>
    </div>
  </section>
  <script src="https://unpkg.com/leaflet/dist/leaflet.js"></script>
  <script src="assets/data_loader.js"></script>
  <script src="assets/dashboard.js"></script>
  <script src="assets/map_controls.js"></script>
</body>
</html>
"""


def _write_full_dashboard(
    root: Path,
    route_records: list[dict[str, Any]],
    route_geojsons: dict[str, dict[str, Any]],
    route_pois: dict[str, list[dict[str, Any]]],
    metrics: dict[str, Any],
    debug_summary: dict[str, Any],
    interest_preview: dict[str, Any],
    playback_data: dict[str, Any],
    city_details: dict[str, Any],
    selected_hotels: dict[str, Any],
    hotel_choices: dict[str, Any],
    nature_explore: dict[str, Any],
    evaluation_metrics: dict[str, Any],
) -> list[Path]:
    assets = root / "assets"
    routes_dir = assets / "routes"
    pois_dir = assets / "pois"
    routes_dir.mkdir(parents=True, exist_ok=True)
    pois_dir.mkdir(parents=True, exist_ok=True)
    _clear_stale_dashboard_assets(assets, routes_dir, pois_dir)
    written = []

    style_path = assets / "style.css"
    _write_text_asset(
        style_path,
        """:root {
  --forest: #0a6b53;
  --forest-dark: #064738;
  --teal: #18b8a5;
  --teal-soft: #d9fbf6;
  --coral: #f9735b;
  --amber: #f5b642;
  --sky: #d9edf6;
  --panel: rgba(255, 255, 255, .94);
  --panel-solid: #ffffff;
  --panel-muted: #f6faf9;
  --line: #d8e5e1;
  --line-strong: #b8d3ca;
  --text: #10233d;
  --muted: #5b6b7d;
  --shadow: 0 18px 48px rgba(14, 35, 58, .22);
  --soft-shadow: 0 8px 22px rgba(14, 35, 58, .12);
  --radius: 8px;
  --fast: 160ms ease;
  --medium: 320ms cubic-bezier(.2, .8, .2, 1);
}

html,
body {
  height: 100%;
  margin: 0;
  color: var(--text);
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  background: #dcebf0;
}

* {
  box-sizing: border-box;
}

#map {
  position: fixed;
  inset: 0;
  width: 100%;
  height: 100vh;
  z-index: 0;
  background: #e7efe8;
}

#map::after {
  content: "";
  position: fixed;
  inset: 0;
  pointer-events: none;
  z-index: 430;
  background: linear-gradient(90deg, rgba(255,255,255,.08), transparent 16%, transparent 82%, rgba(255,255,255,.12));
  mix-blend-mode: soft-light;
}

.dashboard-panel {
  position: absolute;
  z-index: 1000;
  top: 14px;
  left: 14px;
  width: 392px;
  max-height: calc(100vh - 28px);
  overflow: auto;
  background: var(--panel);
  border: 1px solid rgba(184, 211, 202, .86);
  border-radius: var(--radius);
  box-shadow: var(--shadow);
  padding: 12px;
  backdrop-filter: blur(18px) saturate(1.15);
  animation: panelEnter var(--medium);
  scrollbar-width: thin;
  scrollbar-color: rgba(10, 107, 83, .45) transparent;
}

.dashboard-panel::-webkit-scrollbar {
  width: 8px;
}

.dashboard-panel::-webkit-scrollbar-thumb {
  background: rgba(10, 107, 83, .34);
  border-radius: 999px;
}

.dashboard-header {
  display: grid;
  grid-template-columns: 42px minmax(0, 1fr) auto auto;
  align-items: center;
  gap: 10px;
  min-height: 48px;
  margin: 0 0 10px;
  cursor: move;
  user-select: none;
}

.dashboard-mark {
  display: grid;
  place-items: center;
  width: 42px;
  height: 42px;
  color: #fff;
  border-radius: 8px;
  background: linear-gradient(145deg, var(--forest), var(--teal));
  box-shadow: 0 8px 18px rgba(10, 107, 83, .24);
}

.dashboard-title-block {
  min-width: 0;
}

.dashboard-panel h1 {
  margin: 0;
  color: var(--text);
  font-size: 20px;
  line-height: 1.05;
  font-weight: 800;
}

.dashboard-subtitle {
  margin-top: 3px;
  color: var(--muted);
  font-size: 12px;
  line-height: 1.2;
}

.dashboard-mode-label {
  display: inline-flex;
  width: fit-content;
  margin-top: 6px;
  border: 1px solid rgba(10, 107, 83, .22);
  border-radius: 999px;
  padding: 4px 8px;
  color: var(--forest);
  background: rgba(217, 251, 246, .72);
  font-size: 11px;
  font-weight: 800;
  line-height: 1;
}

body[data-dashboard-mode="customer"] [data-mode-section="research"],
body[data-dashboard-mode="research"] [data-mode-section="customer"] {
  display: none !important;
}

.dashboard-panel.collapsed {
  max-height: none;
  overflow: visible;
}

.dashboard-panel.collapsed #dashboard-content {
  display: none;
}

.dashboard-section {
  margin-top: 8px;
  border: 1px solid var(--line);
  border-radius: var(--radius);
  background: linear-gradient(180deg, rgba(255,255,255,.90), rgba(246,250,249,.90));
  overflow: hidden;
  box-shadow: 0 1px 0 rgba(255,255,255,.9) inset;
}

.dashboard-section[open] {
  border-color: var(--line-strong);
}

.dashboard-section summary {
  position: relative;
  display: flex;
  align-items: center;
  min-height: 38px;
  padding: 10px 34px 9px 12px;
  cursor: pointer;
  color: var(--forest);
  font-size: 12px;
  font-weight: 800;
  letter-spacing: 0;
  text-transform: uppercase;
  list-style: none;
}

.dashboard-section summary::-webkit-details-marker {
  display: none;
}

.dashboard-section summary::after {
  content: "";
  position: absolute;
  right: 13px;
  width: 8px;
  height: 8px;
  border-right: 2px solid currentColor;
  border-bottom: 2px solid currentColor;
  transform: rotate(45deg);
  transition: transform var(--fast);
}

.dashboard-section[open] summary::after {
  transform: rotate(225deg) translate(-2px, -2px);
}

.dashboard-section > :not(summary) {
  margin-inline: 12px;
}

.dashboard-section > :last-child {
  margin-bottom: 12px;
}

.dashboard-panel button,
.dashboard-panel select {
  min-height: 30px;
  border: 1px solid var(--line-strong);
  background: var(--panel-solid);
  color: var(--text);
  border-radius: 7px;
  padding: 7px 10px;
  font: 700 12px/1 Inter, ui-sans-serif, system-ui, sans-serif;
  cursor: pointer;
  transition: transform var(--fast), border-color var(--fast), box-shadow var(--fast), background var(--fast), color var(--fast);
}

.dashboard-panel a {
  color: var(--forest);
  font-weight: 800;
  text-decoration: none;
}

.dashboard-panel button:hover,
.dashboard-panel select:hover {
  border-color: var(--teal);
  box-shadow: 0 6px 16px rgba(24, 184, 165, .16);
  transform: translateY(-1px);
}

.dashboard-panel button:focus-visible,
.dashboard-panel select:focus-visible,
.dashboard-panel input:focus-visible {
  outline: 3px solid rgba(24, 184, 165, .25);
  outline-offset: 2px;
}

.dashboard-panel button:disabled {
  color: #93a2b1;
  background: #eef4f3;
  cursor: default;
  transform: none;
  box-shadow: none;
}

#dashboard-collapse {
  width: 36px;
  height: 36px;
  padding: 0;
  font-size: 0;
}

#dashboard-mode-toggle {
  min-width: 112px;
  white-space: nowrap;
}

#dashboard-collapse::before {
  content: "‹‹";
  font-size: 18px;
  line-height: 1;
}

.dashboard-panel.collapsed #dashboard-collapse::before {
  content: "››";
}

.metrics-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 8px;
  margin-top: 2px;
}

.metric {
  min-width: 0;
  padding: 9px 8px;
  border: 1px solid var(--line);
  border-radius: 7px;
  background: rgba(255,255,255,.74);
  color: var(--muted);
  font-size: 11px;
  line-height: 1.25;
}

.metric b {
  display: block;
  margin-bottom: 3px;
  color: var(--text);
  font-size: 15px;
  line-height: 1.1;
}

.status {
  min-height: 18px;
  margin-top: 9px;
  color: var(--forest);
  font-size: 12px;
  font-weight: 700;
  line-height: 1.35;
}

.route-selector {
  margin-top: 8px;
  color: #2a3f54;
  font-size: 12px;
}

.route-group-title {
  margin: 10px 0 5px;
  color: var(--forest-dark);
  font-size: 11px;
  font-weight: 800;
  text-transform: uppercase;
}

.route-row {
  display: grid;
  grid-template-columns: 18px minmax(0, 1fr);
  gap: 8px;
  align-items: start;
  padding: 8px 6px;
  border: 1px solid transparent;
  border-radius: 7px;
  transition: background var(--fast), border-color var(--fast), transform var(--fast);
}

.route-row:hover {
  border-color: var(--line);
  background: rgba(217, 251, 246, .56);
  transform: translateX(1px);
}

.route-row label {
  min-width: 0;
  cursor: pointer;
  font-weight: 700;
}

.route-row small {
  display: block;
  margin-top: 3px;
  color: var(--muted);
  font-weight: 500;
  line-height: 1.3;
}

.layer-chip {
  display: inline-block;
  margin-left: 5px;
  border-radius: 999px;
  padding: 2px 7px;
  color: #fff;
  background: linear-gradient(135deg, var(--forest), var(--teal));
  font-size: 10px;
  font-weight: 800;
  vertical-align: middle;
}

.quick-actions,
.playback-controls,
.filter-row {
  display: flex;
  flex-wrap: wrap;
  gap: 7px;
  margin: 8px 0;
}

.quick-actions button:first-child {
  color: #fff;
  border-color: var(--forest);
  background: linear-gradient(135deg, var(--forest), var(--teal));
}

.filter-row label {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  min-height: 28px;
  padding: 5px 8px;
  border: 1px solid var(--line);
  border-radius: 7px;
  background: rgba(255,255,255,.72);
  color: #294056;
  font-size: 12px;
  font-weight: 700;
}

.customer-control-grid {
  display: grid;
  gap: 10px;
  margin-top: 8px;
}

.customer-control-block {
  border: 1px solid var(--line);
  border-radius: 7px;
  padding: 9px;
  background: rgba(255,255,255,.72);
}

.customer-control-block strong {
  display: block;
  margin-bottom: 6px;
  color: var(--forest-dark);
  font-size: 12px;
  text-transform: uppercase;
}

.customer-option-grid {
  display: flex;
  flex-wrap: wrap;
  gap: 7px;
}

.customer-option-grid button.active {
  border-color: var(--forest);
  color: #fff;
  background: var(--forest);
  box-shadow: 0 8px 18px rgba(10, 107, 83, .22);
}

input[type="checkbox"] {
  width: 14px;
  height: 14px;
  accent-color: var(--forest);
}

.interest-axis {
  display: grid;
  grid-template-columns: 58px minmax(0, 1fr) 42px;
  gap: 8px;
  align-items: center;
  margin: 8px 0;
  font-size: 12px;
  font-weight: 700;
}

.interest-axis input[type="range"] {
  width: 100%;
  accent-color: var(--teal);
}

.interest-axis strong {
  color: var(--forest);
  text-align: right;
}

.interest-note {
  color: var(--muted);
  font-size: 11px;
  line-height: 1.35;
  margin-top: 7px;
}

.interest-ranking {
  margin-top: 8px;
}

.summary-list {
  font-size: 12px;
  color: var(--muted);
  line-height: 1.45;
}

.summary-list div {
  margin: 4px 0;
}

.playback-progress {
  height: 7px;
  overflow: hidden;
  border-radius: 999px;
  background: #dbe8e7;
}

.playback-progress span {
  display: block;
  width: 0%;
  height: 100%;
  border-radius: inherit;
  background: linear-gradient(90deg, var(--forest), var(--teal), var(--coral));
  box-shadow: 0 0 12px rgba(24, 184, 165, .42);
  transition: width .18s ease;
}

.detail-card,
.hotel-card,
.nature-card,
.city-card {
  margin: 8px 0;
  padding: 9px;
  border: 1px solid var(--line);
  border-radius: 7px;
  background: rgba(255, 255, 255, .78);
  box-shadow: var(--soft-shadow);
  font-size: 12px;
  color: #2b4258;
  transition: transform var(--fast), border-color var(--fast), box-shadow var(--fast);
}

.detail-card:hover,
.hotel-card:hover,
.nature-card:hover,
.city-card:hover {
  border-color: var(--teal);
  transform: translateY(-1px);
}

.detail-card strong,
.hotel-card strong,
.nature-card strong,
.city-card strong {
  color: var(--text);
  font-size: 13px;
}

.hotel-card.selected,
.hotel-card.preferred {
  border-color: var(--teal);
  background: linear-gradient(180deg, #f6fffd, var(--teal-soft));
}

.city-card-header,
.hotel-actions {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  margin-bottom: 6px;
}

.city-stop-list {
  margin: 8px 0 0;
  padding: 0;
  list-style: none;
}

.city-stop-list li + li {
  margin-top: 5px;
}

.city-stop-list button {
  width: 100%;
  min-height: 26px;
  text-align: left;
  background: #f7fbfa;
}

.nature-card button,
.hotel-card button {
  margin-top: 6px;
}

.playback-pulse {
  position: relative;
  width: 24px;
  height: 24px;
  border: 3px solid #fff;
  border-radius: 999px;
  background: var(--coral);
  box-shadow: 0 0 0 5px rgba(249, 115, 91, .20), 0 8px 18px rgba(15, 23, 42, .26);
}

.playback-pulse::after {
  content: "";
  position: absolute;
  inset: -10px;
  border: 2px solid rgba(249, 115, 91, .42);
  border-radius: inherit;
  animation: pulseRing 1.4s ease-out infinite;
}

.route-line-default {
  filter: drop-shadow(0 0 5px rgba(24, 184, 165, .70));
}

.live-preview-line {
  filter: drop-shadow(0 0 7px rgba(249, 115, 91, .62));
}

.route-marker,
.nature-marker {
  transition: filter var(--fast), opacity var(--fast);
}

.numbered-route-stop {
  display: grid;
  place-items: center;
  width: 30px;
  height: 30px;
  border: 3px solid #fff;
  border-radius: 999px;
  background: var(--teal);
  color: #fff;
  font-size: 13px;
  font-weight: 900;
  line-height: 1;
  box-shadow: 0 2px 0 rgba(10,107,83,.28), 0 0 0 2px rgba(10,143,131,.34), 0 10px 22px rgba(15,23,42,.24);
}

.numbered-route-stop span {
  display: block;
  transform: translateY(-1px);
}

.airport-route-endpoint {
  display: grid;
  place-items: center;
  min-width: 38px;
  height: 24px;
  padding: 0 7px;
  border: 2px solid #fff;
  border-radius: 999px;
  background: #12324a;
  color: #fff;
  font-size: 11px;
  font-weight: 900;
  line-height: 1;
  box-shadow: 0 2px 0 rgba(15,23,42,.22), 0 8px 20px rgba(15,23,42,.22);
}

.map-text-label {
  width: auto !important;
  height: auto !important;
  padding: 2px 7px;
  border: 1px solid rgba(15,23,42,.14);
  border-radius: 5px;
  background: rgba(255,255,255,.86);
  color: #183045;
  font-size: 11px;
  font-weight: 800;
  line-height: 1.1;
  white-space: nowrap;
  box-shadow: 0 3px 9px rgba(15,23,42,.14);
}

.map-text-label.nature-label {
  color: #1f6f2c;
  border-color: rgba(36,122,50,.24);
  background: rgba(249,255,246,.88);
}

.nature-marker {
  filter: drop-shadow(0 0 5px rgba(106, 198, 89, .46));
}

.weather-chip {
  position: absolute;
  z-index: 960;
  top: 16px;
  left: 430px;
  display: grid;
  grid-template-columns: auto 1fr;
  gap: 10px;
  align-items: center;
  min-width: 150px;
  padding: 10px 12px;
  border: 1px solid rgba(184, 211, 202, .86);
  border-radius: var(--radius);
  background: var(--panel);
  box-shadow: var(--soft-shadow);
  backdrop-filter: blur(16px) saturate(1.1);
}

.weather-sun {
  width: 26px;
  height: 26px;
  border-radius: 999px;
  background: #fbbf24;
  box-shadow: 0 0 0 5px rgba(251, 191, 36, .16), 0 0 18px rgba(251, 191, 36, .45);
}

.weather-temp {
  display: block;
  color: var(--text);
  font-size: 20px;
  font-weight: 800;
  line-height: 1;
}

.weather-city {
  display: block;
  margin-top: 3px;
  color: var(--muted);
  font-size: 11px;
  font-weight: 700;
}

.map-side-panels {
  position: absolute;
  z-index: 960;
  top: 16px;
  right: 16px;
  display: grid;
  gap: 12px;
  width: 190px;
}

.map-card,
.map-legend {
  border: 1px solid rgba(184, 211, 202, .86);
  border-radius: var(--radius);
  background: var(--panel);
  box-shadow: var(--soft-shadow);
  backdrop-filter: blur(16px) saturate(1.1);
}

.map-card {
  padding: 12px;
}

.map-card h2 {
  margin: 0 0 10px;
  color: var(--text);
  font-size: 12px;
  font-weight: 800;
  text-transform: uppercase;
}

.map-card label,
.risk-row,
.legend-item {
  display: flex;
  align-items: center;
  gap: 8px;
  color: #26394d;
  font-size: 12px;
  font-weight: 700;
  line-height: 1.25;
}

.map-card label + label,
.risk-row + .risk-row {
  margin-top: 9px;
}

.risk-dot,
.legend-dot {
  flex: 0 0 auto;
  width: 11px;
  height: 11px;
  border-radius: 999px;
  background: var(--teal);
  box-shadow: 0 0 0 2px rgba(255,255,255,.86);
}

.risk-low { background: #4c9f38; }
.risk-mid { background: #eab308; }
.risk-high { background: #f97316; }
.risk-very { background: #dc2626; }

.map-legend {
  position: absolute;
  z-index: 960;
  left: 50%;
  bottom: 22px;
  display: grid;
  grid-template-columns: repeat(4, max-content);
  gap: 12px 20px;
  align-items: center;
  padding: 12px 16px;
  transform: translateX(-38%);
}

.legend-line {
  width: 28px;
  height: 0;
  border-top: 3px solid var(--teal);
  border-radius: 999px;
}

.legend-line.alt {
  border-top-style: dashed;
  border-top-color: #2f6fdd;
}

.legend-line.preview {
  border-top-style: dotted;
  border-top-color: var(--coral);
}

.legend-hotel {
  display: grid;
  place-items: center;
  width: 17px;
  height: 17px;
  border-radius: 4px;
  color: #fff;
  background: #6d4fd7;
  font-size: 11px;
  font-weight: 900;
}

.legend-star {
  color: #d92652;
  font-size: 15px;
  line-height: 1;
}

.stop-visual {
  min-height: 112px;
  margin: -1px -1px 9px;
  border-radius: 7px 7px 0 0;
  background:
    linear-gradient(180deg, rgba(5, 31, 43, .10), rgba(5, 31, 43, .42)),
    url("https://images.unsplash.com/photo-1500530855697-b586d89ba3ee?auto=format&fit=crop&w=600&q=80") center/cover;
}

.stop-visual img {
  display: block;
  width: 100%;
  height: 112px;
  object-fit: cover;
  border-radius: inherit;
}

.detail-card.active-stop-card {
  padding: 0;
  overflow: hidden;
}

.active-stop-body {
  padding: 0 10px 10px;
}

.active-stop-title-row {
  display: flex;
  align-items: start;
  justify-content: space-between;
  gap: 8px;
}

.day-pill {
  flex: 0 0 auto;
  border: 1px solid var(--line-strong);
  border-radius: 999px;
  padding: 3px 8px;
  color: var(--forest);
  background: var(--teal-soft);
  font-size: 11px;
  font-weight: 800;
}

.active-stop-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 6px 10px;
  margin-top: 8px;
}

.active-stop-grid div {
  min-width: 0;
  color: #31485e;
}

.active-stop-grid b {
  display: block;
  color: var(--muted);
  font-size: 10px;
  text-transform: uppercase;
}

.detail-copy {
  margin-top: 8px;
  color: #31485e;
  font-size: 12px;
  line-height: 1.35;
}

.detail-copy b {
  display: block;
  margin-bottom: 2px;
  color: var(--muted);
  font-size: 10px;
  text-transform: uppercase;
}

.leaflet-control-zoom a,
.leaflet-control-layers-toggle {
  color: var(--text) !important;
}

.diagnostic-panel {
  display: none;
  position: absolute;
  z-index: 1100;
  right: 18px;
  top: 18px;
  max-width: 420px;
  border-radius: 8px;
  border: 1px solid var(--amber);
  background: #fffbeb;
  color: #78350f;
  box-shadow: var(--shadow);
  padding: 12px;
  font-size: 13px;
  line-height: 1.45;
}

.diagnostic-panel.error {
  border-color: #ef4444;
  background: #fef2f2;
  color: #7f1d1d;
}

.diagnostic-panel h2 {
  margin: 0 0 6px;
  font-size: 14px;
}

.diagnostic-panel code {
  word-break: break-all;
}

@keyframes panelEnter {
  from { opacity: 0; transform: translateX(-14px); }
  to { opacity: 1; transform: translateX(0); }
}

@keyframes pulseRing {
  from { opacity: .75; transform: scale(.55); }
  to { opacity: 0; transform: scale(1.45); }
}

@media (max-width: 720px) {
  .dashboard-panel {
    top: 10px;
    left: 10px;
    right: 10px;
    width: auto;
    max-height: min(78vh, calc(100vh - 20px));
  }

  .dashboard-header {
    grid-template-columns: 38px minmax(0, 1fr) auto;
  }

  .dashboard-mark {
    width: 38px;
    height: 38px;
  }

  .dashboard-panel h1 {
    font-size: 18px;
  }

  .metrics-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }

  .weather-chip,
  .map-side-panels,
  .map-legend {
    display: none;
  }
}

@media (prefers-reduced-motion: reduce) {
  *,
  *::before,
  *::after {
    scroll-behavior: auto !important;
    animation-duration: .001ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: .001ms !important;
  }
}
""",
        written,
    )

    for record in route_records:
        route_id = str(record["id"])
        route_path = root / str(record["geojson"])
        route_path.parent.mkdir(parents=True, exist_ok=True)
        _write_json_asset(
            route_path, route_geojsons.get(route_id, {"type": "FeatureCollection", "features": []}), written
        )
        route_js_path = root / str(record["geojson_js"])
        _write_text_asset(
            route_js_path,
            _global_map_assignment("DASHBOARD_ROUTES", route_id, route_geojsons.get(route_id, {})),
            written,
        )

        poi_path = root / str(record["pois"])
        poi_path.parent.mkdir(parents=True, exist_ok=True)
        _write_json_asset(poi_path, route_pois.get(route_id, []), written)
        poi_js_path = root / str(record["pois_js"])
        _write_text_asset(
            poi_js_path,
            _global_map_assignment("DASHBOARD_POIS", route_id, route_pois.get(route_id, [])),
            written,
        )

    metrics_path = assets / "dashboard_metrics.json"
    _write_json_asset(metrics_path, metrics, written)

    metrics_js_path = assets / "dashboard_metrics.js"
    _write_text_asset(metrics_js_path, _global_assignment("DASHBOARD_METRICS", metrics), written)

    debug_path = assets / "debug_summary.json"
    _write_json_asset(debug_path, debug_summary, written)

    debug_js_path = assets / "debug_summary.js"
    _write_text_asset(debug_js_path, _global_assignment("DASHBOARD_DEBUG_SUMMARY", debug_summary), written)

    interest_path = assets / "interest_preview.json"
    _write_json_asset(interest_path, interest_preview, written)

    interest_js_path = assets / "interest_preview.js"
    _write_text_asset(interest_js_path, _global_assignment("DASHBOARD_INTEREST_PREVIEW", interest_preview), written)

    playback_path = assets / "playback_data.json"
    _write_json_asset(playback_path, playback_data, written)

    playback_js_path = assets / "playback_data.js"
    _write_text_asset(playback_js_path, _global_assignment("DASHBOARD_PLAYBACK_DATA", playback_data), written)

    city_path = assets / "city_details.json"
    _write_json_asset(city_path, city_details, written)

    city_js_path = assets / "city_details.js"
    _write_text_asset(city_js_path, _global_assignment("DASHBOARD_CITY_DETAILS", city_details), written)

    selected_hotels_path = assets / "selected_hotels.json"
    _write_json_asset(selected_hotels_path, selected_hotels, written)

    selected_hotels_js_path = assets / "selected_hotels.js"
    _write_text_asset(
        selected_hotels_js_path, _global_assignment("DASHBOARD_SELECTED_HOTELS", selected_hotels), written
    )

    hotel_path = assets / "hotel_choices.json"
    _write_json_asset(hotel_path, hotel_choices, written)

    hotel_js_path = assets / "hotel_choices.js"
    _write_text_asset(hotel_js_path, _global_assignment("DASHBOARD_HOTEL_CHOICES", hotel_choices), written)

    nature_path = assets / "nature_explore.json"
    _write_json_asset(nature_path, nature_explore, written)

    nature_js_path = assets / "nature_explore.js"
    _write_text_asset(nature_js_path, _global_assignment("DASHBOARD_NATURE_EXPLORE", nature_explore), written)

    route_index = {
        "contract_version": CONTRACT_VERSION,
        "routes": route_records,
    }

    route_index_path = assets / "route_index.json"
    _write_json_asset(route_index_path, route_index, written)

    route_index_js_path = assets / "route_index.js"
    _write_text_asset(route_index_js_path, _global_assignment("DASHBOARD_ROUTE_INDEX", route_index), written)

    _write_evaluation_page(root, assets, evaluation_metrics, written)

    data_loader = assets / "data_loader.js"
    _write_text_asset(
        data_loader,
        """const DASHBOARD_HINT = 'If fallback assets are missing, run: python scripts/serve_dashboard.py';
let dashboardFileModeNoticeShown = false;

function dashboardIsFileMode() {
  return window.location.protocol === 'file:';
}

function dashboardLogInit(message, extra) {
  console.log('[dashboard-init]', message, extra || '');
}

function dashboardLogLoader(message, extra) {
  console.log('[dashboard-loader]', message, extra || '');
}

function dashboardLogAssets(message, extra) {
  console.log('[dashboard-assets]', message, extra || '');
}

function dashboardLogError(message, extra) {
  console.error('[dashboard-error]', message, extra || '');
}

function showDashboardDiagnostic(level, checkName, assetUrl, detail) {
  if (level !== 'error') {
    dashboardLogLoader(`${checkName}: ${detail || ''}`, assetUrl || '');
    return;
  }
  const panel = document.getElementById('diagnostic-panel');
  if (!panel) return;
  panel.classList.toggle('error', level === 'error');
  panel.style.display = 'block';
  panel.innerHTML = `
    <h2>${level === 'error' ? 'Dashboard Error' : 'Dashboard Notice'}</h2>
    <div><b>Check:</b> ${checkName}</div>
    <div><b>Asset URL:</b> <code>${assetUrl || 'n/a'}</code></div>
    <div><b>Detail:</b> ${detail || 'No detail available.'}</div>
  `;
}

function loadScriptOnce(src) {
  return new Promise((resolve, reject) => {
    if (document.querySelector(`script[data-dashboard-src="${src}"]`)) {
      resolve();
      return;
    }
    const script = document.createElement('script');
    script.src = src;
    script.dataset.dashboardSrc = src;
    script.onload = resolve;
    script.onerror = () => reject(new Error(`Failed to load script: ${src}`));
    document.head.appendChild(script);
  });
}

function readFallbackGlobal(fallback) {
  const value = window[fallback.globalName];
  if (fallback.key !== undefined && value) {
    return value[fallback.key];
  }
  return value;
}

async function loadFallbackAsset(kind, fallback) {
  if (!fallback || !fallback.script || !fallback.globalName) {
    throw new Error(`No JS fallback configured for ${kind}`);
  }
  dashboardLogLoader(`script fallback ${fallback.script}`);
  try {
    await loadScriptOnce(fallback.script);
  } catch (error) {
    const detail = `Both JSON fetch and JS fallback loading failed. Missing file: ${fallback.script}. Regenerate the dashboard export.`;
    showDashboardDiagnostic('error', `${kind} JS fallback loading`, fallback.script, detail);
    throw error;
  }
  const payload = readFallbackGlobal(fallback);
  if (payload === undefined || payload === null) {
    const detail = `JS fallback loaded but did not define ${fallback.globalName}. Regenerate the dashboard export.`;
    showDashboardDiagnostic('error', `${kind} JS fallback global`, fallback.script, detail);
    throw new Error(detail);
  }
  return payload;
}

async function fetchJsonAsset(kind, path) {
  dashboardLogAssets(`fetch ${path}`);
  const response = await fetch(path);
  if (!response.ok) {
    throw new Error(`HTTP ${response.status} ${response.statusText}`);
  }
  return response.json();
}

async function loadDashboardAsset(kind, path, fallbackGlobalName) {
  const fallback = typeof fallbackGlobalName === 'string'
    ? { script: path.replace(/\\.geojson$|\\.json$/, '.js'), globalName: fallbackGlobalName }
    : fallbackGlobalName;

  if (dashboardIsFileMode()) {
    if (!dashboardFileModeNoticeShown) {
      dashboardFileModeNoticeShown = true;
      showDashboardDiagnostic(
        'warning',
        `${kind} file mode`,
        path,
        'This dashboard was opened through file://. JSON fetch is blocked in this mode. Using JS fallback assets instead. If fallback assets are missing, run: python scripts/serve_dashboard.py'
      );
    }
    return loadFallbackAsset(kind, fallback);
  }

  try {
    return await fetchJsonAsset(kind, path);
  } catch (error) {
    dashboardLogError(`${kind} JSON fetch failed`, error);
    return loadFallbackAsset(kind, fallback);
  }
}

async function loadRouteIndex() {
  return loadDashboardAsset('route index', 'assets/route_index.json', {
    script: 'assets/route_index.js',
    globalName: 'DASHBOARD_ROUTE_INDEX'
  });
}

async function loadDashboardMetrics() {
  return loadDashboardAsset('dashboard metrics', 'assets/dashboard_metrics.json', {
    script: 'assets/dashboard_metrics.js',
    globalName: 'DASHBOARD_METRICS'
  });
}

async function loadDebugSummary() {
  return loadDashboardAsset('debug summary', 'assets/debug_summary.json', {
    script: 'assets/debug_summary.js',
    globalName: 'DASHBOARD_DEBUG_SUMMARY'
  });
}

async function loadInterestPreview() {
  return loadDashboardAsset('interest preview', 'assets/interest_preview.json', {
    script: 'assets/interest_preview.js',
    globalName: 'DASHBOARD_INTEREST_PREVIEW'
  });
}

async function loadPlaybackData() {
  return loadDashboardAsset('playback data', 'assets/playback_data.json', {
    script: 'assets/playback_data.js',
    globalName: 'DASHBOARD_PLAYBACK_DATA'
  });
}

async function loadCityDetails() {
  return loadDashboardAsset('city details', 'assets/city_details.json', {
    script: 'assets/city_details.js',
    globalName: 'DASHBOARD_CITY_DETAILS'
  });
}

async function loadSelectedHotels() {
  return loadDashboardAsset('selected hotels', 'assets/selected_hotels.json', {
    script: 'assets/selected_hotels.js',
    globalName: 'DASHBOARD_SELECTED_HOTELS'
  });
}

async function loadHotelChoices() {
  return loadDashboardAsset('hotel choices', 'assets/hotel_choices.json', {
    script: 'assets/hotel_choices.js',
    globalName: 'DASHBOARD_HOTEL_CHOICES'
  });
}

async function loadNatureExplore() {
  return loadDashboardAsset('nature explore', 'assets/nature_explore.json', {
    script: 'assets/nature_explore.js',
    globalName: 'DASHBOARD_NATURE_EXPLORE'
  });
}

async function loadRouteGeoJson(routeRecord) {
  return loadDashboardAsset(`route ${routeRecord.id}`, routeRecord.geojson, {
    script: routeRecord.geojson_js,
    globalName: 'DASHBOARD_ROUTES',
    key: routeRecord.id
  });
}

async function loadPoiJson(routeRecord) {
  if (!routeRecord.pois && !routeRecord.pois_js) return [];
  const key = routeRecord.id;
  return loadDashboardAsset(`POIs ${key}`, routeRecord.pois, {
    script: routeRecord.pois_js,
    globalName: 'DASHBOARD_POIS',
    key
  });
}

window.dashboardIsFileMode = dashboardIsFileMode;
window.loadDashboardAsset = loadDashboardAsset;
window.loadRouteIndex = loadRouteIndex;
window.loadDashboardMetrics = loadDashboardMetrics;
window.loadDebugSummary = loadDebugSummary;
window.loadInterestPreview = loadInterestPreview;
window.loadPlaybackData = loadPlaybackData;
window.loadCityDetails = loadCityDetails;
window.loadSelectedHotels = loadSelectedHotels;
window.loadHotelChoices = loadHotelChoices;
window.loadNatureExplore = loadNatureExplore;
window.loadRouteGeoJson = loadRouteGeoJson;
window.loadPoiJson = loadPoiJson;
window.showDashboardDiagnostic = showDashboardDiagnostic;
window.dashboardLogInit = dashboardLogInit;
window.dashboardLogLoader = dashboardLogLoader;
window.dashboardLogAssets = dashboardLogAssets;
window.dashboardLogError = dashboardLogError;
""",
        written,
    )

    map_controls = assets / "map_controls.js"
    _write_text_asset(
        map_controls,
        """let dashboardMap = null;
let dashboardRouteIndex = null;
let dashboardPlaybackData = null;
let activePlaybackRouteId = null;
let playbackMarker = null;
let playbackTimer = null;
let playbackAnimationFrame = null;
let playbackStopIndex = 0;
let playbackRunning = false;
const loadedRouteLayers = new Map();
const activeRouteIds = new Set();
let selectedHotelLayer = null;
let selectedHotelsVisible = true;
let hotelCandidatesVisible = false;
let livePreviewLayer = null;
let livePreviewPlaybackStops = [];
let livePreviewRouteRecord = null;
let terrainBaseLayer = null;
let roadsOverlayLayer = null;
let labelOverlayLayer = null;
let weatherRiskLayer = null;

const FAMILY_STYLES = {
  selected: { color: '#0a8f83', fill: '#18b8a5', weight: 6, dashArray: null },
  route_matrix: { color: '#2f6fdd', fill: '#7eb6ff', weight: 3, dashArray: '7 7' },
  trip_length: { color: '#7c4dd8', fill: '#bba7ff', weight: 3, dashArray: '4 7' },
  method: { color: '#f9735b', fill: '#ffba8e', weight: 3, dashArray: '2 7' },
  interest_profile: { color: '#0f766e', fill: '#5eead4', weight: 4, dashArray: '10 8' },
  context: { color: '#66788a', fill: '#aab8c4', weight: 2, dashArray: '9 8' },
  hotel: { color: '#6d4fd7', fill: '#d8cffd', weight: 1.5, dashArray: null },
  must_go: { color: '#d92652', fill: '#ff8aa4', weight: 1.5, dashArray: null },
  nature: { color: '#247a32', fill: '#6ac659', weight: 1.5, dashArray: null },
  live_preview: { color: '#f9735b', fill: '#ffb199', weight: 5, dashArray: '3 12' }
};

function setDashboardStatus(message, isError = false) {
  const target = document.getElementById('dashboard-status');
  if (!target) return;
  target.textContent = message;
  target.style.color = isError ? '#b91c1c' : '#0a6b53';
}

function assertMapContainerReady() {
  if (!window.L) {
    showDashboardDiagnostic('error', 'Leaflet loaded', 'https://unpkg.com/leaflet/dist/leaflet.js', 'window.L is missing.');
    throw new Error('Leaflet is not loaded');
  }
  const container = document.getElementById('map');
  if (!container) {
    showDashboardDiagnostic('error', 'map container exists', '#map', 'Missing <div id="map">.');
    throw new Error('Map container missing');
  }
  const rect = container.getBoundingClientRect();
  if (rect.height <= 0 || rect.width <= 0) {
    showDashboardDiagnostic('error', 'map container has nonzero height', '#map', `Measured ${rect.width}x${rect.height}.`);
    throw new Error('Map container has zero size');
  }
}

function defaultRoute(index) {
  const routes = Array.isArray(index?.routes) ? index.routes : [];
  return routes.find(route => route.default) || routes[0] || null;
}

function isPlayableRoute(routeRecord) {
  if (!routeRecord) return false;
  if (routeRecord.id === 'live_preview') return livePreviewPlaybackStops.length > 1;
  if (routeRecord.playable === false || routeRecord.marker_only === true) return false;
  return ['selected', 'route_matrix', 'trip_length', 'method', 'interest_profile'].includes(routeRecord.family);
}

function firstPlayableRouteId() {
  const routes = Array.isArray(dashboardRouteIndex?.routes) ? dashboardRouteIndex.routes : [];
  const current = routeById(activePlaybackRouteId);
  if (isPlayableRoute(current) && activeRouteIds.has(activePlaybackRouteId)) return activePlaybackRouteId;
  if (activePlaybackRouteId === 'live_preview' && livePreviewPlaybackStops.length > 1) return 'live_preview';
  const visible = routes.find(route => activeRouteIds.has(route.id) && isPlayableRoute(route));
  const fallback = visible || routes.find(route => route.default && isPlayableRoute(route)) || routes.find(isPlayableRoute);
  return fallback?.id || null;
}

function playbackStops(routeId = activePlaybackRouteId) {
  if (routeId === 'live_preview') return livePreviewPlaybackStops;
  const record = routeById(routeId);
  if (!isPlayableRoute(record)) return [];
  const fromAsset = dashboardPlaybackData?.routes?.[routeId]?.stops;
  if (Array.isArray(fromAsset) && fromAsset.length) return fromAsset;
  const loaded = loadedRouteLayers.get(routeId);
  return loaded?.pois || [];
}

function playbackSpeedMs() {
  const select = document.getElementById('playback-speed');
  return Number(select?.value || 900);
}

function updatePlaybackProgress() {
  const stops = playbackStops();
  const progress = document.getElementById('playback-progress-bar');
  const label = document.getElementById('playback-current-stop');
  const pct = stops.length > 1 ? (playbackStopIndex / (stops.length - 1)) * 100 : 0;
  if (progress) progress.style.width = `${Math.max(0, Math.min(100, pct))}%`;
  if (label) {
    const stop = stops[playbackStopIndex];
    const ordinal = stop ? routePointOrdinal(stop, playbackStopIndex) : '';
    label.textContent = stop ? `${ordinal}/${stops.length} · ${stop.name || 'Stop'} · ${stop.city || ''}` : 'No playable route loaded.';
  }
}

function ensurePlaybackMarker(stop) {
  if (!stop || !Number.isFinite(Number(stop.lat)) || !Number.isFinite(Number(stop.lon))) return null;
  const latLng = [Number(stop.lat), Number(stop.lon)];
  if (!playbackMarker) {
    playbackMarker = L.marker(latLng, {
      icon: L.divIcon({
        className: 'playback-marker-icon',
        html: '<span class="playback-pulse" aria-hidden="true"></span>',
        iconSize: [24, 24],
        iconAnchor: [12, 12]
      }),
      zIndexOffset: 900
    }).addTo(dashboardMap);
  } else {
    playbackMarker.setLatLng(latLng);
  }
  playbackMarker.bindPopup(`<b>${stop.name || 'Stop'}</b><br>${stop.city || ''}`);
  return playbackMarker;
}

function setPlaybackStop(index, pan = false) {
  const stops = playbackStops();
  if (!stops.length) return;
  playbackStopIndex = Math.max(0, Math.min(stops.length - 1, index));
  const stop = stops[playbackStopIndex];
  ensurePlaybackMarker(stop);
  updatePlaybackProgress();
  if (window.renderActiveStopDetail) {
    window.renderActiveStopDetail(stop, routeById(activePlaybackRouteId), {
      index: playbackStopIndex,
      total: stops.length
    });
  }
  const follow = document.getElementById('playback-follow');
  if (pan && follow?.checked && stop?.lat && stop?.lon) {
    dashboardMap.panTo([Number(stop.lat), Number(stop.lon)]);
  }
}

function interpolateStop(fromStop, toStop, ratio) {
  return {
    lat: Number(fromStop.lat) + (Number(toStop.lat) - Number(fromStop.lat)) * ratio,
    lon: Number(fromStop.lon) + (Number(toStop.lon) - Number(fromStop.lon)) * ratio,
    name: toStop.name,
    city: toStop.city
  };
}

function animateToStop(nextIndex) {
  const stops = playbackStops();
  if (!playbackRunning || nextIndex >= stops.length) {
    playbackRunning = false;
    return;
  }
  const fromStop = stops[playbackStopIndex] || stops[0];
  const toStop = stops[nextIndex];
  const start = performance.now();
  const duration = playbackSpeedMs();
  function frame(now) {
    if (!playbackRunning) return;
    const ratio = Math.min(1, (now - start) / duration);
    ensurePlaybackMarker(interpolateStop(fromStop, toStop, ratio));
    if (ratio < 1) {
      playbackAnimationFrame = requestAnimationFrame(frame);
      return;
    }
    setPlaybackStop(nextIndex, true);
    playbackTimer = window.setTimeout(() => animateToStop(nextIndex + 1), Math.max(120, duration * 0.22));
  }
  playbackAnimationFrame = requestAnimationFrame(frame);
}

function playRouteAnimation() {
  const stops = playbackStops();
  if (stops.length < 2) {
    setDashboardStatus('Playback needs at least two route stops.', true);
    return;
  }
  if (playbackStopIndex >= stops.length - 1) playbackStopIndex = 0;
  playbackRunning = true;
  setPlaybackStop(playbackStopIndex, true);
  animateToStop(playbackStopIndex + 1);
  setDashboardStatus('Route playback running.');
}

function pauseRouteAnimation() {
  playbackRunning = false;
  if (playbackTimer) window.clearTimeout(playbackTimer);
  if (playbackAnimationFrame) window.cancelAnimationFrame(playbackAnimationFrame);
  setDashboardStatus('Route playback paused.');
}

function restartRouteAnimation() {
  pauseRouteAnimation();
  playbackStopIndex = 0;
  setPlaybackStop(0, true);
}

function stepPlayback(delta) {
  pauseRouteAnimation();
  setPlaybackStop(playbackStopIndex + delta, true);
}

function setActivePlaybackRoute(routeId, reset = true) {
  activePlaybackRouteId = routeId;
  if (reset) playbackStopIndex = 0;
  const summary = document.getElementById('playback-route-label');
  const record = routeById(routeId);
  if (summary) summary.textContent = record ? record.label : routeId;
  setPlaybackStop(playbackStopIndex, false);
}

function bindPlaybackControls() {
  document.getElementById('playback-play')?.addEventListener('click', playRouteAnimation);
  document.getElementById('playback-pause')?.addEventListener('click', pauseRouteAnimation);
  document.getElementById('playback-restart')?.addEventListener('click', restartRouteAnimation);
  document.getElementById('playback-prev')?.addEventListener('click', () => stepPlayback(-1));
  document.getElementById('playback-next')?.addEventListener('click', () => stepPlayback(1));
}

function routeLineStyle(routeRecord) {
  const style = FAMILY_STYLES[routeRecord?.family] || FAMILY_STYLES.selected;
  return {
    color: style.color,
    weight: style.weight,
    opacity: routeRecord?.default ? 0.88 : 0.62,
    dashArray: style.dashArray,
    lineCap: 'round',
    lineJoin: 'round',
    className: routeRecord?.default ? 'route-line route-line-default' : 'route-line'
  };
}

function markerPopup(point, index, routeRecord) {
  const candidateFamilies = ['hotel', 'nature', 'must_go', 'context'];
  const isEndpoint = Boolean(point?.is_route_endpoint);
  const isHotelNode = Boolean(point?.is_hotel_node || point?.node_kind === 'hotel');
  const selectionState = isEndpoint
    ? 'Airport endpoint - not counted as a selected POI'
    : (isHotelNode
      ? 'Selected overnight/base hotel - not counted as a selected POI'
    : (routeRecord?.family === 'live_preview'
      ? 'Preview only - not written to optimizer artifacts'
      : (candidateFamilies.includes(routeRecord?.family) ? 'Candidate/context marker - not a selected route stop' : 'Saved optimized route stop')));
  const label = routeRecord?.family === 'hotel' || routeRecord?.family === 'nature'
    ? (point.name || point.hotel_name || routeRecord.label)
    : (isEndpoint ? `${point.airport_code || 'Airport'} · ${point.name || 'Airport'}`
      : (isHotelNode ? `H · ${point.name || point.hotel_name || 'Hotel'}`
        : `${routePointOrdinal(point, index)}. ${point.name || 'Stop'}`));
  const day = point.day ? `Day ${point.day}<br>` : '';
  const displayUtility = Number(point.display_utility ?? point.interest_adjusted_value ?? point.final_poi_value);
  const finalValue = Number(point.final_poi_value);
  const interestValue = Number(point.interest_adjusted_value);
  const utility = Number.isFinite(displayUtility) ? `<br>Optimizer value: ${displayUtility.toFixed(2)}` : '';
  const finalLine = Number.isFinite(finalValue) && finalValue > 0 ? `<br>Final POI value: ${finalValue.toFixed(2)}` : '';
  const interestLine = Number.isFinite(interestValue) && interestValue > 0 && Math.abs(interestValue - finalValue) > 0.0001
    ? `<br>Interest-adjusted value: ${interestValue.toFixed(2)}`
    : '';
  const hotel = point.hotel_name ? `<br>Hotel: ${point.hotel_name}` : '';
  return `<b>${label}</b><br>${selectionState}<br>${day}${point.city || ''}<br>${point.nature_region || point.category || ''}${hotel}${utility}${finalLine}${interestLine}`;
}

function pointToMarker(point, index, routeRecord) {
  const style = FAMILY_STYLES[routeRecord?.family] || FAMILY_STYLES.selected;
  const candidateMarker = ['hotel', 'must_go', 'nature', 'context'].includes(routeRecord?.family);
  if (point?.is_route_endpoint) {
    const marker = L.marker([Number(point.lat), Number(point.lon)], {
      icon: L.divIcon({
        className: 'airport-route-endpoint',
        html: `<span>${point.airport_code || 'AP'}</span>`,
        iconSize: null,
        iconAnchor: [19, 12],
        popupAnchor: [0, -14]
      }),
      zIndexOffset: routeRecord?.default ? 720 : 430
    });
    marker.bindPopup(markerPopup(point, index, routeRecord));
    marker.on('click', () => {
      if (isPlayableRoute(routeRecord)) {
        activePlaybackRouteId = routeRecord?.id || activePlaybackRouteId;
        playbackStopIndex = index;
        setPlaybackStop(index, false);
      }
    });
    return marker;
  }
  if (!candidateMarker) {
    const marker = L.marker([Number(point.lat), Number(point.lon)], {
      icon: L.divIcon({
        className: 'numbered-route-stop',
        html: `<span>${routePointOrdinal(point, index)}</span>`,
        iconSize: [30, 30],
        iconAnchor: [15, 15],
        popupAnchor: [0, -16]
      }),
      zIndexOffset: routeRecord?.default ? 760 : 460
    });
    marker.bindPopup(markerPopup(point, index, routeRecord));
    marker.on('click', () => {
      if (isPlayableRoute(routeRecord)) {
        activePlaybackRouteId = routeRecord?.id || activePlaybackRouteId;
        playbackStopIndex = index;
        setPlaybackStop(index, false);
      }
    });
    return marker;
  }
  const marker = L.circleMarker([Number(point.lat), Number(point.lon)], {
    radius: candidateMarker ? 4.5 : (routeRecord?.optional ? 5 : 8),
    color: style.color,
    fillColor: style.fill,
    fillOpacity: candidateMarker ? 0.5 : (routeRecord?.optional ? 0.72 : 0.9),
    weight: candidateMarker ? 1.5 : (routeRecord?.optional ? 2 : 3),
    className: routeRecord?.family === 'nature' ? 'nature-marker' : 'route-marker'
  });
  marker.bindPopup(routeRecord?.family === 'hotel' ? hotelMarkerPopup(point, Boolean(point.selected)) : markerPopup(point, index, routeRecord));
  marker.on('mouseover', () => marker.setStyle({ radius: candidateMarker ? 6 : (routeRecord?.optional ? 7 : 10), fillOpacity: candidateMarker ? 0.76 : 1 }));
  marker.on('mouseout', () => marker.setStyle({ radius: candidateMarker ? 4.5 : (routeRecord?.optional ? 5 : 8), fillOpacity: candidateMarker ? 0.5 : (routeRecord?.optional ? 0.72 : 0.9) }));
  marker.on('click', () => {
    if (isPlayableRoute(routeRecord)) {
      activePlaybackRouteId = routeRecord?.id || activePlaybackRouteId;
      playbackStopIndex = index;
      setPlaybackStop(index, false);
    } else if (window.renderActiveStopDetail) {
      window.renderActiveStopDetail(point, routeRecord, { index, total: 1 });
    }
  });
  return marker;
}

function drawRouteGeometry(group, route, routeRecord) {
  if (routeRecord?.marker_only) return null;
  if (!route || !Array.isArray(route.features) || route.features.length === 0) return null;
  return L.geoJSON(route, {
    filter: feature => feature.geometry && feature.geometry.type !== 'Point',
    style: () => routeLineStyle(routeRecord)
  }).addTo(group);
}

function drawPoiMarkers(group, pois, routeRecord) {
  const latLngs = [];
  (pois || []).forEach((point, index) => {
    if (!Number.isFinite(Number(point.lat)) || !Number.isFinite(Number(point.lon))) return;
    const latLng = [Number(point.lat), Number(point.lon)];
    latLngs.push(latLng);
    pointToMarker(point, index, routeRecord).addTo(group);
  });
  return latLngs;
}

function hotelMarkerPopup(hotel, selected = false) {
  const status = selected || hotel.selected ? 'Selected by optimizer' : 'Candidate alternative';
  const score = Number(hotel.hotel_score ?? hotel.score ?? 0);
  const rank = hotel.candidate_rank ? `<br>Rank: ${hotel.candidate_rank}` : '';
  const scoreLine = Number.isFinite(score) ? `<br>Score: ${score.toFixed(2)}` : '';
  const sourceValue = hotel.source || hotel.source_list || '';
  const source = sourceValue ? `<br>Source: ${sourceValue}` : '';
  const reason = hotel.selected_hotel_reason ? `<br>Reason: ${hotel.selected_hotel_reason}` : '';
  const distance = Number.isFinite(Number(hotel.mean_distance_to_selected_stops_km))
    ? `<br>Avg distance to selected stops: ${Number(hotel.mean_distance_to_selected_stops_km).toFixed(2)} km`
    : '';
  return `<b>${hotel.hotel_name || hotel.name || 'Hotel'}</b><br>${hotel.city || ''}<br>${status}${rank}${scoreLine}${source}${distance}${reason}`;
}

function drawSelectedHotels(payload) {
  if (selectedHotelLayer && dashboardMap.hasLayer(selectedHotelLayer)) {
    dashboardMap.removeLayer(selectedHotelLayer);
  }
  selectedHotelLayer = L.layerGroup();
  (payload?.items || []).forEach(hotel => {
    if (!Number.isFinite(Number(hotel.lat)) || !Number.isFinite(Number(hotel.lon))) return;
    const marker = L.circleMarker([Number(hotel.lat), Number(hotel.lon)], {
      radius: 8,
      color: '#6d4fd7',
      fillColor: '#d8cffd',
      fillOpacity: 0.95,
      weight: 3
    });
    marker.bindPopup(hotelMarkerPopup(hotel, true));
    marker.addTo(selectedHotelLayer);
  });
  if (selectedHotelsVisible && selectedHotelLayer.getLayers().length) {
    selectedHotelLayer.addTo(dashboardMap);
  }
  updateLayerToggleState();
}

function focusDashboardLocation(lat, lon, label = 'Selected location') {
  if (!dashboardMap || !Number.isFinite(Number(lat)) || !Number.isFinite(Number(lon))) return;
  dashboardMap.flyTo([Number(lat), Number(lon)], Math.max(dashboardMap.getZoom(), 8), { duration: 0.65 });
  L.popup({ closeButton: true, autoClose: true })
    .setLatLng([Number(lat), Number(lon)])
    .setContent(`<b>${label}</b>`)
    .openOn(dashboardMap);
}

function toggleSelectedHotels(forceVisible = null) {
  selectedHotelsVisible = forceVisible === null ? !selectedHotelsVisible : Boolean(forceVisible);
  if (!selectedHotelLayer) {
    setDashboardStatus('No selected hotel markers were exported for this route.');
    return;
  }
  if (selectedHotelsVisible) {
    if (!dashboardMap.hasLayer(selectedHotelLayer)) selectedHotelLayer.addTo(dashboardMap);
    setDashboardStatus('Selected hotel markers shown.');
  } else {
    if (dashboardMap.hasLayer(selectedHotelLayer)) dashboardMap.removeLayer(selectedHotelLayer);
    setDashboardStatus('Selected hotel markers hidden.');
  }
  updateLayerToggleState();
}

function extendBoundsFromLayer(bounds, layer) {
  if (!layer) return bounds;
  try {
    if (typeof layer.getBounds === 'function') {
      const layerBounds = layer.getBounds();
      if (layerBounds?.isValid && layerBounds.isValid()) bounds.extend(layerBounds);
      return bounds;
    }
    if (typeof layer.getLatLng === 'function') {
      bounds.extend(layer.getLatLng());
      return bounds;
    }
    if (typeof layer.eachLayer === 'function') {
      layer.eachLayer(child => extendBoundsFromLayer(bounds, child));
    }
  } catch (error) {
    dashboardLogError('layer bounds skipped', error);
  }
  return bounds;
}

function fitVisibleRoutes() {
  try {
    const groups = [...activeRouteIds].map(id => loadedRouteLayers.get(id)?.group).filter(Boolean);
    if (!groups.length) {
      setDashboardStatus('No visible routes to zoom.');
      return;
    }
    const bounds = L.latLngBounds([]);
    groups.forEach(group => extendBoundsFromLayer(bounds, group));
    if (bounds.isValid()) {
      dashboardMap.fitBounds(bounds, {
        paddingTopLeft: [430, 80],
        paddingBottomRight: [250, 120],
        maxZoom: 7
      });
      return;
    }
  } catch (error) {
    dashboardLogError('fitBounds failed', error);
  }
}

function initializeBaseMapLayers() {
  terrainBaseLayer = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}', {
    maxZoom: 18,
    attribution: 'Tiles &copy; Esri &mdash; Sources: Esri, USGS, NOAA'
  }).addTo(dashboardMap);
  roadsOverlayLayer = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    maxZoom: 18,
    opacity: 0.34,
    zIndex: 220,
    attribution: '&copy; OpenStreetMap contributors'
  }).addTo(dashboardMap);
}

function mapLayerToggleChecked(layerName, fallback = true) {
  const input = document.querySelector(`[data-map-layer-toggle="${layerName}"]`);
  return input ? Boolean(input.checked) : fallback;
}

function setLeafletLayerVisible(layer, visible) {
  if (!dashboardMap || !layer) return;
  if (visible && !dashboardMap.hasLayer(layer)) layer.addTo(dashboardMap);
  if (!visible && dashboardMap.hasLayer(layer)) dashboardMap.removeLayer(layer);
}

function selectedContextStops() {
  const routeId = activePlaybackRouteId || defaultRoute(dashboardRouteIndex)?.id;
  const stops = playbackStops(routeId);
  if (Array.isArray(stops) && stops.length) return stops;
  const routes = dashboardPlaybackData?.routes || {};
  const defaultId = defaultRoute(dashboardRouteIndex)?.id;
  return routes[defaultId]?.stops || [];
}

function loadedCandidatePois(routeId) {
  const loaded = loadedRouteLayers.get(routeId);
  return Array.isArray(loaded?.pois) ? loaded.pois : [];
}

function routePointOrdinal(point, index) {
  if (point?.is_route_endpoint || point?.is_airport_endpoint) return point.airport_code || 'AP';
  if (point?.is_hotel_node || point?.node_kind === 'hotel') return 'H';
  return point?.display_sequence_index
    || point?.route_sequence_index
    || point?.selected_stop_index
    || point?.stop_order
    || index + 1;
}

function labelPointHtml(point, index, selected = false) {
  const ordinal = routePointOrdinal(point, index);
  const prefix = selected && !point?.is_route_endpoint && !point?.is_airport_endpoint && !point?.is_hotel_node
    ? `${ordinal}. `
    : (point?.is_route_endpoint || point?.is_airport_endpoint || point?.is_hotel_node ? `${ordinal} ` : '');
  return `${prefix}${point.name || point.hotel_name || point.nature_region || 'Place'}`;
}

function addTextLabel(group, point, className, html) {
  if (!Number.isFinite(Number(point.lat)) || !Number.isFinite(Number(point.lon))) return;
  L.marker([Number(point.lat), Number(point.lon)], {
    icon: L.divIcon({
      className: `map-text-label ${className || ''}`,
      html,
      iconSize: null,
      iconAnchor: [-10, 8]
    }),
    interactive: false,
    zIndexOffset: className === 'selected-label' ? 520 : 260
  }).addTo(group);
}

function buildLabelLayer() {
  const group = L.layerGroup();
  selectedContextStops().filter(point => point?.is_selected_poi || point?.is_route_endpoint || point?.is_airport_endpoint || point?.is_hotel_node).slice(0, 32).forEach((point, index) => {
    addTextLabel(group, point, 'selected-label', labelPointHtml(point, index, true));
  });
  if (!activeRouteIds.has('nature_candidates')) return group;
  const naturePoints = [
    ...loadedCandidatePois('nature_candidates'),
  ];
  const seen = new Set();
  naturePoints.slice(0, 32).forEach(point => {
    const key = `${point.name || point.nature_region || ''}:${point.lat}:${point.lon}`;
    if (seen.has(key)) return;
    seen.add(key);
    addTextLabel(group, point, 'nature-label', labelPointHtml(point, 0, false));
  });
  return group;
}

function riskColor(value) {
  const risk = Number(value || 0);
  if (risk >= 0.75) return '#dc2626';
  if (risk >= 0.50) return '#f97316';
  if (risk >= 0.25) return '#eab308';
  return '#49a33f';
}

function collectWeatherRiskPoints() {
  const items = [
    ...selectedContextStops(),
    ...loadedCandidatePois('nature_candidates'),
    ...loadedCandidatePois('hotel_candidates'),
    ...(Array.isArray(window.dashboardNatureExplore?.items) ? window.dashboardNatureExplore.items : []),
    ...(Array.isArray(window.dashboardSelectedHotels?.items) ? window.dashboardSelectedHotels.items : [])
  ];
  const seen = new Set();
  return items
    .filter(point => Number.isFinite(Number(point.lat)) && Number.isFinite(Number(point.lon)))
    .filter(point => {
      const key = `${point.name || point.hotel_name || ''}:${point.lat}:${point.lon}`;
      if (seen.has(key)) return false;
      seen.add(key);
      return true;
    });
}

function buildWeatherRiskLayer() {
  const group = L.layerGroup();
  collectWeatherRiskPoints().forEach(point => {
    const risk = Number(point.weather_risk ?? point.weather_sensitivity ?? 0.12);
    const color = riskColor(risk);
    L.circleMarker([Number(point.lat), Number(point.lon)], {
      radius: 10 + Math.min(8, risk * 8),
      color,
      fillColor: color,
      fillOpacity: 0.14,
      opacity: 0.45,
      weight: 2,
      interactive: false
    }).addTo(group);
  });
  return group;
}

function refreshMapContextLayers() {
  if (!dashboardMap) return;
  if (labelOverlayLayer && dashboardMap.hasLayer(labelOverlayLayer)) dashboardMap.removeLayer(labelOverlayLayer);
  if (weatherRiskLayer && dashboardMap.hasLayer(weatherRiskLayer)) dashboardMap.removeLayer(weatherRiskLayer);
  labelOverlayLayer = buildLabelLayer();
  weatherRiskLayer = buildWeatherRiskLayer();
  setLeafletLayerVisible(labelOverlayLayer, mapLayerToggleChecked('labels', true));
  setLeafletLayerVisible(weatherRiskLayer, mapLayerToggleChecked('weather_risk', true));
}

function updateMapLayerToggleState() {
  document.querySelectorAll('[data-map-layer-toggle]').forEach(input => {
    const action = input.dataset.mapLayerToggle;
    if (action === 'terrain') input.checked = Boolean(terrainBaseLayer && dashboardMap?.hasLayer(terrainBaseLayer));
    if (action === 'roads') input.checked = Boolean(roadsOverlayLayer && dashboardMap?.hasLayer(roadsOverlayLayer));
    if (action === 'labels') input.checked = Boolean(labelOverlayLayer && dashboardMap?.hasLayer(labelOverlayLayer));
    if (action === 'weather_risk') input.checked = Boolean(weatherRiskLayer && dashboardMap?.hasLayer(weatherRiskLayer));
  });
}

function bindMapLayerToggles() {
  document.querySelectorAll('[data-map-layer-toggle]').forEach(input => {
    input.addEventListener('change', event => {
      const action = event.target.dataset.mapLayerToggle;
      const checked = event.target.checked;
      if (action === 'terrain') setLeafletLayerVisible(terrainBaseLayer, checked);
      if (action === 'roads') setLeafletLayerVisible(roadsOverlayLayer, checked);
      if (action === 'labels') {
        if (!labelOverlayLayer) labelOverlayLayer = buildLabelLayer();
        setLeafletLayerVisible(labelOverlayLayer, checked);
      }
      if (action === 'weather_risk') {
        if (!weatherRiskLayer) weatherRiskLayer = buildWeatherRiskLayer();
        setLeafletLayerVisible(weatherRiskLayer, checked);
      }
      updateMapLayerToggleState();
    });
  });
}

function routeById(routeId) {
  if (routeId === 'live_preview') return livePreviewRouteRecord;
  const routes = Array.isArray(dashboardRouteIndex?.routes) ? dashboardRouteIndex.routes : [];
  return routes.find(route => route.id === routeId);
}

async function ensureRouteLayer(routeRecord) {
  if (loadedRouteLayers.has(routeRecord.id)) {
    return loadedRouteLayers.get(routeRecord.id);
  }
  const group = L.layerGroup();
  let route = null;
  let pois = [];
  try {
    route = await loadRouteGeoJson(routeRecord);
  } catch (error) {
    dashboardLogError(`route load failed for ${routeRecord.id}`, error);
  }
  try {
    pois = await loadPoiJson(routeRecord);
  } catch (error) {
    dashboardLogError(`POI load failed for ${routeRecord.id}`, error);
  }
  drawRouteGeometry(group, route, routeRecord);
  const markerLatLngs = drawPoiMarkers(group, pois, routeRecord);
  if (group.getLayers().length === 0 && (!markerLatLngs || markerLatLngs.length === 0)) {
    throw new Error(`No drawable assets for ${routeRecord.id}`);
  }
  const payload = { group, route: route || null, pois: pois || [], record: routeRecord };
  loadedRouteLayers.set(routeRecord.id, payload);
  return payload;
}

function updateSelectorState() {
  document.querySelectorAll('[data-route-toggle]').forEach(input => {
    input.checked = activeRouteIds.has(input.value);
  });
  updateLayerToggleState();
}

function updateLayerToggleState() {
  document.querySelectorAll('[data-layer-toggle]').forEach(input => {
    const action = input.dataset.layerToggle;
    if (action === 'selected_hotels') input.checked = selectedHotelsVisible;
    if (action === 'default_route') input.checked = activeRouteIds.has(defaultRoute(dashboardRouteIndex)?.id);
    if (action === 'hotel_candidates') input.checked = activeRouteIds.has('hotel_candidates');
    if (action === 'nature_candidates') input.checked = activeRouteIds.has('nature_candidates');
    if (action === 'live_preview') input.checked = Boolean(livePreviewLayer && dashboardMap?.hasLayer(livePreviewLayer));
  });
}

async function toggleRoute(routeId, visible, fitAfter = true) {
  const routeRecord = routeById(routeId);
  if (!routeRecord) {
    setDashboardStatus(`Route not found: ${routeId}`, true);
    return;
  }
  try {
    const payload = await ensureRouteLayer(routeRecord);
    if (visible) {
      if (!dashboardMap.hasLayer(payload.group)) payload.group.addTo(dashboardMap);
      activeRouteIds.add(routeId);
      if (isPlayableRoute(routeRecord)) setActivePlaybackRoute(routeId);
      else if (!firstPlayableRouteId()) setActivePlaybackRoute(defaultRoute(dashboardRouteIndex)?.id);
      setDashboardStatus(`Loaded ${routeRecord.label}`);
    } else {
      if (dashboardMap.hasLayer(payload.group)) dashboardMap.removeLayer(payload.group);
      activeRouteIds.delete(routeId);
      if (activePlaybackRouteId === routeId) setActivePlaybackRoute(firstPlayableRouteId());
      setDashboardStatus(`Hidden ${routeRecord.label}`);
    }
    if (routeId === 'hotel_candidates') hotelCandidatesVisible = activeRouteIds.has(routeId);
    updateSelectorState();
    refreshMapContextLayers();
    updateMapLayerToggleState();
    if (fitAfter) fitVisibleRoutes();
  } catch (error) {
    setDashboardStatus(`Could not load ${routeRecord.label}: ${error.message}`, true);
    showDashboardDiagnostic('error', `route ${routeId} loads`, routeRecord.geojson, error.message);
  }
}

async function showOnlyRoute(routeRecord) {
  for (const routeId of [...activeRouteIds]) {
    const payload = loadedRouteLayers.get(routeId);
    if (payload && dashboardMap.hasLayer(payload.group)) dashboardMap.removeLayer(payload.group);
    activeRouteIds.delete(routeId);
  }
  await toggleRoute(routeRecord.id, true, true);
}

function clearRoutes() {
  for (const routeId of [...activeRouteIds]) {
    const payload = loadedRouteLayers.get(routeId);
    if (payload && dashboardMap.hasLayer(payload.group)) dashboardMap.removeLayer(payload.group);
    activeRouteIds.delete(routeId);
  }
  updateSelectorState();
  setActivePlaybackRoute(firstPlayableRouteId(), false);
  setDashboardStatus('Cleared visible routes.');
}

function drawLivePreviewRoute(points, weights = {}) {
  clearLivePreviewRoute(false);
  const drawable = (points || []).filter(point => Number.isFinite(Number(point.lat)) && Number.isFinite(Number(point.lon)));
  if (drawable.length === 0) {
    setDashboardStatus('No preview candidates with coordinates were available.', true);
    return;
  }
  let previewStopNumber = 0;
  livePreviewPlaybackStops = drawable.map((point, index) => {
    const isEndpoint = Boolean(point.is_route_endpoint);
    if (!isEndpoint) previewStopNumber += 1;
    const stopNumber = isEndpoint ? (point.stop_order || (point.route_endpoint_side === 'start' ? 0 : 999)) : previewStopNumber;
    return {
      ...point,
      name: point.name || (isEndpoint ? point.airport_code || 'Airport endpoint' : `Preview stop ${stopNumber}`),
      city: point.city || point.nature_region || '',
      day: point.day || Math.max(1, stopNumber || index + 1),
      stop_order: stopNumber,
      type: isEndpoint ? 'airport_endpoint' : (point.type || point.category || point.park_type || (point.nature_region ? 'nature_candidate' : 'preview_candidate')),
      category: isEndpoint ? 'airport_endpoint' : (point.category || point.type || point.park_type || (point.nature_region ? 'nature_candidate' : 'preview_candidate')),
      description: point.description || `${point.name || 'Preview stop'} is a browser-only preview candidate from exported POI data.`,
      why_selected: isEndpoint
        ? 'Preview airport endpoint from the exported segment graph; not counted as a selected POI.'
        : 'Preview only - rerun pipeline to save. This stop is not written back to optimizer artifacts.',
      expected_duration_minutes: point.expected_duration_minutes || 75,
      weather_risk: Number(point.weather_risk || 0.15),
      source_confidence: Number(point.source_confidence || point.data_confidence || 0.5),
      display_utility: Number(point.preview_score || point.display_utility || point.final_poi_value || 0),
      optimization_value_source: isEndpoint ? 'configured_airport_endpoint' : 'browser_preview_score',
      route_sequence_index: isEndpoint ? 0 : stopNumber,
      display_sequence_index: isEndpoint ? (point.airport_code || 'AP') : stopNumber,
      selected_stop_index: isEndpoint ? 0 : stopNumber,
      node_kind: isEndpoint ? 'airport' : 'preview_poi',
      is_airport_endpoint: isEndpoint,
      is_selected_poi: false,
      is_hotel_node: false
    };
  });
  livePreviewRouteRecord = {
    id: 'live_preview',
    label: 'Preview only - rerun pipeline to save',
    family: 'live_preview',
    default: false,
    optional: true,
    playable: true,
    marker_only: false
  };
  livePreviewLayer = L.layerGroup();
  const latLngs = livePreviewPlaybackStops.map(point => [Number(point.lat), Number(point.lon)]);
  if (latLngs.length > 1) {
    L.polyline(latLngs, {
      color: '#f9735b',
      weight: 5,
      opacity: 0.85,
      dashArray: '3 12',
      lineCap: 'round',
      className: 'route-line live-preview-line'
    }).addTo(livePreviewLayer);
  }
  livePreviewPlaybackStops.forEach((point, index) => {
    let marker = null;
    if (point.is_route_endpoint) {
      marker = L.marker([Number(point.lat), Number(point.lon)], {
        icon: L.divIcon({
          className: 'airport-route-endpoint',
          html: `<span>${point.airport_code || 'AP'}</span>`,
          iconSize: null,
          iconAnchor: [19, 12],
          popupAnchor: [0, -14]
        }),
        zIndexOffset: 710
      });
    } else {
      marker = L.circleMarker([Number(point.lat), Number(point.lon)], {
        radius: 6,
        color: '#c94b35',
        fillColor: '#ffb199',
        fillOpacity: 0.88,
        weight: 2
      });
    }
    const score = Number(point.preview_score);
    const label = point.is_route_endpoint ? `${point.airport_code || 'Airport'} · ${point.name || 'Airport'}` : `Preview ${routePointOrdinal(point, index)}. ${point.name || 'Candidate'}`;
    marker.bindPopup(
      `<b>${label}</b><br>${point.city || point.nature_region || ''}<br>Preview score: ${Number.isFinite(score) ? score.toFixed(2) : 'n/a'}<br>Browser preview is approximate; Run All recomputes the real optimized route.`
    );
    marker.on('click', () => {
      playbackStopIndex = index;
      setActivePlaybackRoute('live_preview', false);
      setPlaybackStop(index, false);
    });
    marker.addTo(livePreviewLayer);
  });
  livePreviewLayer.addTo(dashboardMap);
  try {
    const bounds = L.featureGroup(livePreviewLayer.getLayers()).getBounds();
    if (bounds.isValid()) {
      dashboardMap.fitBounds(bounds, {
        paddingTopLeft: [430, 80],
        paddingBottomRight: [250, 120],
        maxZoom: 7
      });
    }
  } catch (error) {
    dashboardLogError('preview route fit failed', error);
  }
  updateLayerToggleState();
  const naturePct = Math.round((Number(weights.nature) || 0) * 100);
  const sequenceStatus = weights.__sequence_feasible === false ? ' Sequence infeasible or missing segment candidates.' : ' Sequence feasible.';
  setActivePlaybackRoute('live_preview', true);
  setDashboardStatus(`Preview only - rerun pipeline to save. Browser-side route rebuilt (${naturePct}% nature).${sequenceStatus}`, weights.__sequence_feasible === false);
}

function clearLivePreviewRoute(updateStatus = true) {
  if (livePreviewLayer && dashboardMap?.hasLayer(livePreviewLayer)) {
    dashboardMap.removeLayer(livePreviewLayer);
  }
  livePreviewLayer = null;
  livePreviewPlaybackStops = [];
  livePreviewRouteRecord = null;
  if (activePlaybackRouteId === 'live_preview') setActivePlaybackRoute(firstPlayableRouteId(), true);
  updateLayerToggleState();
  if (updateStatus) setDashboardStatus('Live preview route cleared.');
}

async function showRouteGroup(groupName) {
  const routes = (dashboardRouteIndex.routes || []).filter(route => {
    const groups = route.quick_groups || [];
    return groups.includes(groupName);
  });
  if (!routes.length) {
    setDashboardStatus(`No route assets found for ${groupName}.`, true);
    return;
  }
  for (const route of routes) {
    await toggleRoute(route.id, true, false);
  }
  fitVisibleRoutes();
  setDashboardStatus(`Loaded ${routes.length} ${groupName.replace('_', '-')} route layer${routes.length === 1 ? '' : 's'}.`);
}

function runQuickAction(action) {
  if (action === 'clear') {
    clearRoutes();
  } else if (action === 'default') {
    const selected = defaultRoute(dashboardRouteIndex);
    if (selected) showOnlyRoute(selected);
  } else if (action === 'zoom_visible') {
    fitVisibleRoutes();
  } else {
    showRouteGroup(action);
  }
}

function toggleHotelCandidates(forceVisible = null) {
  const targetVisible = forceVisible === null ? !activeRouteIds.has('hotel_candidates') : Boolean(forceVisible);
  const routeRecord = routeById('hotel_candidates');
  if (!routeRecord) {
    setDashboardStatus('No hotel candidate assets found. Regenerate dashboard assets.', true);
    return;
  }
  hotelCandidatesVisible = targetVisible;
  toggleRoute('hotel_candidates', targetVisible);
}

function bindLayerToggles() {
  document.querySelectorAll('[data-layer-toggle]').forEach(input => {
    input.addEventListener('change', event => {
      const action = event.target.dataset.layerToggle;
      const checked = event.target.checked;
      if (action === 'default_route') {
        const selected = defaultRoute(dashboardRouteIndex);
        if (!selected) return;
        toggleRoute(selected.id, checked);
      } else if (action === 'selected_hotels') {
        toggleSelectedHotels(checked);
      } else if (action === 'hotel_candidates') {
        toggleHotelCandidates(checked);
      } else if (action === 'nature_candidates') {
        toggleRoute('nature_candidates', checked);
      } else if (action === 'live_preview') {
        if (!checked) clearLivePreviewRoute();
        else if (livePreviewLayer && !dashboardMap.hasLayer(livePreviewLayer)) {
          livePreviewLayer.addTo(dashboardMap);
          setActivePlaybackRoute('live_preview', true);
        } else if (!livePreviewLayer) {
          setDashboardStatus('No preview route is available yet. Drag interest bars or build a preview route first.');
          event.target.checked = false;
        }
      }
    });
  });
  updateLayerToggleState();
}

function renderQuickButtons(index) {
  const target = document.getElementById('quick-actions');
  if (!target) return;
  const routes = Array.isArray(index.routes) ? index.routes : [];
  const availableGroups = new Set(routes.flatMap(route => route.quick_groups || []));
  const buttonSpecs = [
    ['clear', 'Clear'],
    ['default', 'Saved route'],
    ['days_7', '7-day'],
    ['days_9', '9-day'],
    ['days_12', '12-day'],
    ['balanced', 'Balanced'],
    ['zoom_visible', 'Zoom visible']
  ];
  target.innerHTML = '';
  buttonSpecs.forEach(([action, label]) => {
    if (!['clear', 'default', 'zoom_visible'].includes(action) && !availableGroups.has(action)) return;
    const button = document.createElement('button');
    button.type = 'button';
    button.textContent = label;
    button.dataset.quickAction = action;
    button.addEventListener('click', () => runQuickAction(action));
    target.appendChild(button);
  });
}

function renderRouteSelector(index) {
  const target = document.getElementById('route-selector');
  if (!target) return;
  const routes = Array.isArray(index.routes) ? index.routes : [];
  target.innerHTML = '';
  const groups = new Map();
  routes.forEach(route => {
    const groupName = route.selector_group || route.family || 'routes';
    if (!groups.has(groupName)) groups.set(groupName, []);
    groups.get(groupName).push(route);
  });
  groups.forEach((groupRoutes, groupName) => {
    const title = document.createElement('div');
    title.className = 'route-group-title';
    title.textContent = groupName.replaceAll('_', ' ');
    target.appendChild(title);
    groupRoutes.forEach(route => {
      const row = document.createElement('div');
      row.className = 'route-row';
      const input = document.createElement('input');
      input.type = 'checkbox';
      input.value = route.id;
      input.checked = Boolean(route.default);
      input.dataset.routeToggle = 'true';
      input.addEventListener('change', event => toggleRoute(route.id, event.target.checked));
      const label = document.createElement('label');
      label.append(document.createTextNode(route.label || route.id));
      const chip = document.createElement('span');
      chip.className = 'layer-chip';
      chip.textContent = route.family || 'route';
      label.append(chip);
      const meta = document.createElement('small');
      const bits = [route.trip_days ? `${route.trip_days} days` : '', route.method || '', route.profile || ''].filter(Boolean);
      meta.textContent = bits.join(' · ') || (route.optional ? 'Hidden until selected' : 'Visible on load');
      label.append(meta);
      row.append(input, label);
      target.appendChild(row);
    });
  });
}

function loadOptionalLayers() {
  const optionalRoutes = (dashboardRouteIndex?.routes || []).filter(route => route.optional);
  if (!optionalRoutes.length) {
    setDashboardStatus('No optional layer assets found. Regenerate dashboard assets.', true);
    return;
  }
  setDashboardStatus(`${optionalRoutes.length} optional layers are available in the route selector.`);
}

async function drawDefaultRoute(routeRecord) {
  await showOnlyRoute(routeRecord);
  setDashboardStatus(`Loaded default route: ${routeRecord.label || routeRecord.id}`);
}

async function initMap() {
  dashboardLogInit('start');
  try {
    assertMapContainerReady();
    dashboardMap = L.map("map", { preferCanvas: true }).setView([36.5, -119.5], 6);
    window.dashboardMap = dashboardMap;
    initializeBaseMapLayers();

    const index = await loadRouteIndex();
    const metrics = await loadDashboardMetrics();
    const debugSummary = await loadDebugSummary().catch(error => {
      dashboardLogError('debug summary load failed', error);
      return { available: false, message: error.message };
    });
    const interestPreview = await loadInterestPreview().catch(error => {
      dashboardLogError('interest preview load failed', error);
      return { available: false, message: error.message };
    });
    const playbackData = await loadPlaybackData().catch(error => {
      dashboardLogError('playback data load failed', error);
      return { available: false, routes: {}, message: error.message };
    });
    const cityDetails = await loadCityDetails().catch(error => {
      dashboardLogError('city details load failed', error);
      return { available: false, cities: [], message: error.message };
    });
    const selectedHotels = await loadSelectedHotels().catch(error => {
      dashboardLogError('selected hotels load failed', error);
      return { available: false, items: [], message: error.message };
    });
    const hotelChoices = await loadHotelChoices().catch(error => {
      dashboardLogError('hotel choices load failed', error);
      return { available: false, cities: {}, message: error.message };
    });
    const natureExplore = await loadNatureExplore().catch(error => {
      dashboardLogError('nature explore load failed', error);
      return { available: false, items: [], message: error.message };
    });
    window.dashboardMetrics = metrics;
    window.dashboardRouteIndex = index;
    window.dashboardPlaybackData = playbackData;
    window.dashboardNatureExplore = natureExplore;
    window.dashboardSelectedHotels = selectedHotels;
    dashboardRouteIndex = index;
    dashboardPlaybackData = playbackData;
    if (window.bindDashboardShellControls) {
      window.bindDashboardShellControls();
    }
    if (window.renderDashboardMetrics) {
      window.renderDashboardMetrics(metrics);
    }
    if (window.renderDebugSummary) {
      window.renderDebugSummary(debugSummary);
    }
    if (window.renderInterestPreview) {
      window.renderInterestPreview(interestPreview);
    }
    if (window.renderCityDetails) {
      window.renderCityDetails(cityDetails);
    }
    if (window.renderHotelChoices) {
      window.renderHotelChoices(hotelChoices);
    }
    if (window.renderNatureExplore) {
      window.renderNatureExplore(natureExplore);
    }
    drawSelectedHotels(selectedHotels);
    bindPlaybackControls();
    if (window.renderCustomerControls) {
      window.renderCustomerControls(index);
    }
    renderQuickButtons(index);
    renderRouteSelector(index);
    bindLayerToggles();
    bindMapLayerToggles();

    const selected = defaultRoute(index);
    if (!selected) {
      showDashboardDiagnostic('error', 'default route exists', 'assets/route_index.json', 'No default route assets found. Regenerate dashboard assets.');
      throw new Error('No default route found');
    }
    await drawDefaultRoute(selected);
    refreshMapContextLayers();
    updateMapLayerToggleState();
    if (document.body.dataset.dashboardMode === 'customer') {
      setDashboardStatus('Customer planner ready. Changes load saved artifacts or preview-only routes.');
    } else {
      loadOptionalLayers();
    }
    dashboardLogInit('complete');
  } catch (error) {
    dashboardLogError('initialization failed', error);
    if (!document.getElementById('diagnostic-panel')?.textContent) {
      showDashboardDiagnostic('error', 'dashboard initialization', window.location.href, error.message);
    }
  }
}

window.drawDefaultRoute = drawDefaultRoute;
window.renderRouteSelector = renderRouteSelector;
window.activeRouteIds = activeRouteIds;
window.toggleRoute = toggleRoute;
window.toggleSelectedHotels = toggleSelectedHotels;
window.toggleHotelCandidates = toggleHotelCandidates;
window.showOnlyRoute = showOnlyRoute;
window.setActivePlaybackRoute = setActivePlaybackRoute;
window.runQuickAction = runQuickAction;
window.showRouteGroup = showRouteGroup;
window.zoomVisibleRoutes = fitVisibleRoutes;
window.loadOptionalLayers = loadOptionalLayers;
window.drawLivePreviewRoute = drawLivePreviewRoute;
window.clearLivePreviewRoute = clearLivePreviewRoute;
window.focusDashboardLocation = focusDashboardLocation;
window.playRouteAnimation = playRouteAnimation;
window.pauseRouteAnimation = pauseRouteAnimation;
window.restartRouteAnimation = restartRouteAnimation;
window.setPlaybackStop = setPlaybackStop;
window.addEventListener('DOMContentLoaded', initMap);
""",
        written,
    )

    dashboard_js = assets / "dashboard.js"
    _write_text_asset(
        dashboard_js,
"""function renderDashboardMetrics(metrics) {
  const target = document.getElementById('metrics');
  if (!target) return;
  const subtitle = document.getElementById('dashboard-subtitle');
  if (subtitle) subtitle.textContent = metrics.scenario_label || metrics.scenario || 'Itinerary scenario';
  target.innerHTML = `
    <div class="metrics-grid">
      <div class="metric"><b>${metrics.trip_days || '7'}</b>Days</div>
      <div class="metric"><b>${metrics.selected_stop_count}</b>Stops</div>
      <div class="metric"><b>${metrics.display_utility ?? metrics.interest_adjusted_utility}</b>Optimizer value</div>
      <div class="metric"><b>${metrics.nature_optimization_enabled ? 'Active' : 'Off'}</b>Nature mode</div>
    </div>
    <div class="summary-list">
      <div><b>Scenario:</b> ${metrics.scenario}</div>
      <div><b>Interest:</b> ${metrics.interest_profile}</div>
      <div><b>Value column:</b> ${metrics.optimization_value_column || 'final_poi_value'}</div>
      <div><b>Route state:</b> ${metrics.route_state_label || 'Saved optimized route'}</div>
      <div><b>Default route:</b> ${metrics.default_route_label}</div>
      <div><b>Route method:</b> ${metrics.default_route_method || 'selected artifact'}</div>
      <div><b>Artifact freshness:</b> ${metrics.artifact_metadata_fresh ? 'fresh metadata match' : 'metadata missing/stale'}</div>
      <div><b>Anchor audit:</b> ${metrics.audit_readiness || 'missing'}${metrics.audit_missing_count ? ` · ${metrics.audit_missing_count} missing` : ''}</div>
      <div><b>Indexed layers:</b> ${metrics.route_record_count || 1}</div>
    </div>
    <div class="interest-note">Default route is computed by Python optimization using nature-aware value weights.</div>
    <div class="interest-note">${metrics.preview_state_label || 'Preview only - rerun pipeline to save'}.</div>
  `;
}

function renderObjectSummary(targetId, title, payload) {
  const target = document.getElementById(targetId);
  if (!target) return;
  if (!payload || payload.available === false) {
    target.innerHTML = `<div class="summary-list">${payload?.message || `No ${title} artifact available.`}</div>`;
    return;
  }
  const rows = Object.entries(payload)
    .filter(([key]) => !['available', 'items'].includes(key))
    .slice(0, 8)
    .map(([key, value]) => `<div><b>${key.replaceAll('_', ' ')}:</b> ${typeof value === 'object' ? JSON.stringify(value).slice(0, 160) : value}</div>`)
    .join('');
  target.innerHTML = `<div class="summary-list">${rows || `${title} loaded.`}</div>`;
}

function compactValue(value, digits = 2) {
  const number = Number(value);
  return Number.isFinite(number) ? number.toFixed(digits) : '';
}

function escapeHtml(value) {
  return String(value ?? '')
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#39;');
}

function bindDashboardShellControls() {
  const panel = document.querySelector('.dashboard-panel');
  const handle = document.getElementById('dashboard-drag-handle');
  const collapse = document.getElementById('dashboard-collapse');
  const modeToggle = document.getElementById('dashboard-mode-toggle');
  if (!panel || !handle) return;
  collapse?.addEventListener('click', event => {
    event.stopPropagation();
    panel.classList.toggle('collapsed');
    collapse.textContent = panel.classList.contains('collapsed') ? 'Expand' : 'Collapse';
  });
  const applyMode = mode => {
    const nextMode = mode === 'customer' ? 'customer' : 'research';
    document.body.dataset.dashboardMode = nextMode;
    document.body.classList.toggle('customer-dashboard', nextMode === 'customer');
    document.body.classList.toggle('research-dashboard', nextMode === 'research');
    const modeLabel = document.getElementById('dashboard-mode-label');
    if (modeLabel) {
      modeLabel.textContent = nextMode === 'customer' ? 'Customer trip planner' : 'Research/Test dashboard';
    }
    if (modeToggle) {
      modeToggle.textContent = nextMode === 'customer' ? 'Research/Test mode' : 'Customer mode';
      modeToggle.setAttribute('aria-pressed', nextMode === 'customer' ? 'true' : 'false');
    }
    if (window.dashboardMap?.invalidateSize) {
      window.setTimeout(() => window.dashboardMap.invalidateSize(), 80);
    }
  };
  applyMode(document.body.dataset.dashboardMode || 'research');
  modeToggle?.addEventListener('click', event => {
    event.stopPropagation();
    const current = document.body.dataset.dashboardMode === 'customer' ? 'customer' : 'research';
    const next = current === 'customer' ? 'research' : 'customer';
    applyMode(next);
    const status = document.getElementById('dashboard-status');
    if (status) {
      status.textContent = next === 'customer'
        ? 'Customer planner mode: choices load saved artifacts or preview-only routes.'
        : 'Research/Test mode: all comparison, debug, and evaluation controls are visible.';
      status.style.color = '#0a6b53';
    }
  });
  window.applyDashboardMode = applyMode;
  let dragState = null;
  handle.addEventListener('pointerdown', event => {
    if (event.target?.tagName === 'BUTTON') return;
    const rect = panel.getBoundingClientRect();
    dragState = {
      pointerId: event.pointerId,
      offsetX: event.clientX - rect.left,
      offsetY: event.clientY - rect.top
    };
    panel.style.left = `${rect.left}px`;
    panel.style.top = `${rect.top}px`;
    panel.style.right = 'auto';
    handle.setPointerCapture(event.pointerId);
  });
  handle.addEventListener('pointermove', event => {
    if (!dragState || dragState.pointerId !== event.pointerId) return;
    const maxLeft = Math.max(0, window.innerWidth - panel.offsetWidth - 8);
    const maxTop = Math.max(0, window.innerHeight - 52);
    const left = Math.max(8, Math.min(maxLeft, event.clientX - dragState.offsetX));
    const top = Math.max(8, Math.min(maxTop, event.clientY - dragState.offsetY));
    panel.style.left = `${left}px`;
    panel.style.top = `${top}px`;
  });
  handle.addEventListener('pointerup', event => {
    if (dragState?.pointerId === event.pointerId) dragState = null;
  });
  handle.addEventListener('pointercancel', () => {
    dragState = null;
  });
}

function renderActiveStopDetail(stop, routeRecord, playbackMeta = {}) {
  const target = document.getElementById('active-stop-detail');
  if (!target) return;
  if (!stop) {
    target.innerHTML = '<div class="summary-list">Click a marker or start playback to inspect a stop.</div>';
    return;
  }
  const rows = [
    ['Route', routeRecord?.label || ''],
    ['Playback', playbackMeta.total ? `${routePointOrdinal(stop, playbackMeta.index || 0)} of ${playbackMeta.total}` : ''],
    ['Day', stop.day || ''],
    ['City', stop.city || ''],
    ['City/anchor', stop.city_or_anchor || ''],
    ['Overnight hotel', stop.hotel_name || ''],
    ['Type', stop.type || stop.category || ''],
    ['Category', stop.category || ''],
    ['Nature region', stop.nature_region || ''],
    ['Expected duration', stop.expected_duration_minutes ? `${stop.expected_duration_minutes} min` : ''],
    ['Weather risk', compactValue(stop.weather_risk)],
    ['Source confidence', compactValue(stop.source_confidence)],
    ['Optimizer value', compactValue(stop.display_utility ?? stop.interest_adjusted_value ?? stop.final_poi_value)],
    ['Final POI value', compactValue(stop.final_poi_value)],
    ['Interest-adjusted value', Number(stop.interest_adjusted_value) > 0 ? compactValue(stop.interest_adjusted_value) : ''],
    ['Value source', stop.optimization_value_source || ''],
    ['Nature score', compactValue(stop.nature_score)],
    ['Scenic score', compactValue(stop.scenic_score)],
    ['Weather sensitivity', compactValue(stop.weather_sensitivity)],
    ['Source', stop.source_list || '']
  ].filter(([, value]) => value !== '' && value !== undefined && value !== null);
  const keyRows = rows.filter(([key]) => ['City/anchor', 'Type', 'Expected duration', 'Weather risk', 'Source confidence', 'Optimizer value', 'Nature score', 'Scenic score'].includes(key));
  const imageUrl = String(stop.image_url || '').trim();
  const visual = imageUrl
    ? `<div class="stop-visual" role="img" aria-label="${escapeHtml(stop.name || 'Place')} image"><img src="${escapeHtml(imageUrl)}" alt="${escapeHtml(stop.name || 'Place')}"></div>`
    : '<div class="stop-visual" role="img" aria-label="Scenic landscape preview"></div>';
  const sourceLink = stop.source_url || stop.website_url
    ? `<div class="detail-copy"><b>Source</b><a href="${escapeHtml(stop.source_url || stop.website_url)}" target="_blank" rel="noopener">${escapeHtml(stop.detail_source || stop.source_url || stop.website_url)}</a></div>`
    : '';
  target.innerHTML = `
    <div class="detail-card active-stop-card">
      ${visual}
      <div class="active-stop-body">
        <div class="active-stop-title-row">
          <strong>${escapeHtml(stop.name || 'Stop')}</strong>
          <span class="day-pill">Day ${stop.day || playbackMeta.index + 1 || 1}</span>
        </div>
        <div class="active-stop-grid">
          ${keyRows.map(([key, value]) => `<div><b>${escapeHtml(key)}</b>${escapeHtml(value)}</div>`).join('')}
        </div>
        <div class="detail-copy"><b>Description</b>${escapeHtml(stop.description || 'No description exported.')}</div>
        <div class="detail-copy"><b>Why selected</b>${escapeHtml(stop.why_selected || stop.reason_selected || 'Selection reason was not exported.')}</div>
        ${sourceLink}
      </div>
    </div>
  `;
}

function renderCityDetails(payload) {
  const target = document.getElementById('city-details');
  if (!target) return;
  const cities = payload?.cities || [];
  const legs = payload?.intercity_legs || [];
  if (!cities.length) {
    target.innerHTML = `<div class="summary-list">${payload?.message || 'No city detail artifact available.'}</div>`;
    return;
  }
  target.innerHTML = cities.slice(0, 10).map(city => {
    const cityLegs = legs
      .filter(leg => leg.from === city.city || leg.to === city.city)
      .slice(0, 3)
      .map(leg => `${leg.from} → ${leg.to} (${compactValue(leg.estimated_drive_minutes, 0)} min)`);
    const stops = (city.selected_stops || []).slice(0, 5).map(stop => `
      <li>
        <button type="button" data-focus-city-stop="${stop.name || ''}" data-lat="${stop.lat || ''}" data-lon="${stop.lon || ''}">
          ${stop.day ? `D${stop.day}` : 'Stop'} · ${stop.name || 'Stop'}
        </button>
      </li>
    `).join('');
    return `
      <div class="city-card">
        <div class="city-card-header">
          <strong>${city.city}</strong>
          <button type="button" data-focus-city="${city.city}" data-lat="${city.lat || ''}" data-lon="${city.lon || ''}">Focus</button>
        </div>
        <div>Days: ${(city.days || []).join(', ') || 'n/a'}</div>
        <div>Selected hotel: ${city.selected_hotel || (city.hotels || [])[0] || 'n/a'}</div>
        <div>Hotel options: ${city.hotel_alternative_count || 0}</div>
        <div>Travel legs: ${cityLegs.join('; ') || 'n/a'}</div>
        <div>Top nature/scenic: ${compactValue(city.nature_score)} / ${compactValue(city.scenic_score)}</div>
        <ul class="city-stop-list">${stops || '<li>No selected route stops exported for this city.</li>'}</ul>
      </div>
    `;
  }).join('');
  target.querySelectorAll('[data-focus-city], [data-focus-city-stop]').forEach(button => {
    button.addEventListener('click', () => {
      const lat = Number(button.dataset.lat);
      const lon = Number(button.dataset.lon);
      if (window.focusDashboardLocation && Number.isFinite(lat) && Number.isFinite(lon)) {
        window.focusDashboardLocation(lat, lon, button.dataset.focusCity || button.dataset.focusCityStop || 'City detail');
      }
    });
  });
}

function hotelPreferenceKey(city) {
  return `dashboard-preferred-hotel:${city}`;
}

function preferredHotel(city) {
  try {
    return window.localStorage.getItem(hotelPreferenceKey(city));
  } catch {
    return '';
  }
}

function setPreferredHotel(city, hotelName) {
  try {
    window.localStorage.setItem(hotelPreferenceKey(city), hotelName);
  } catch {
    return;
  }
}

function clearPreferredHotel(city) {
  try {
    window.localStorage.removeItem(hotelPreferenceKey(city));
  } catch {
    return;
  }
}

function preferredHotelPayload(payload) {
  const cities = payload?.cities || {};
  const preferred_hotels = {};
  Object.keys(cities).forEach(city => {
    const hotel = preferredHotel(city);
    if (hotel) preferred_hotels[city] = hotel;
  });
  return { hotels: { preferred_hotels, preferred_hotel_bonus: 3.0 } };
}

function downloadPreferredHotelConfig(payload) {
  const blob = new Blob([JSON.stringify(preferredHotelPayload(payload), null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = 'preferred_hotels_config.json';
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

function renderHotelChoices(payload) {
  const target = document.getElementById('hotel-choices');
  if (!target) return;
  const cities = payload?.cities || {};
  const cityNames = Object.keys(cities).sort();
  if (!cityNames.length) {
    target.innerHTML = `<div class="summary-list">${payload?.message || 'No hotel candidates found.'}</div>`;
    return;
  }
  target.innerHTML = `
    <div class="filter-row">
      <button type="button" data-toggle-hotel-candidates="true">Show hotel candidates</button>
      <button type="button" data-export-hotel-preferences="true">Export preferences</button>
    </div>
    <div class="interest-note">Preferred hotels are reversible here and can be exported into config as a soft bonus for the next optimizer rerun.</div>
  ` + cityNames.map(city => {
    const preferred = preferredHotel(city);
    const cityHotels = cities[city] || [];
    const limited = cityHotels.length <= 1 && cityHotels.some(hotel => hotel.source === 'preview');
    const cards = cityHotels.slice(0, 6).map(hotel => {
      const isPreferred = preferred && preferred === hotel.hotel_name;
      const classes = ['hotel-card', hotel.selected ? 'selected' : '', isPreferred ? 'preferred' : ''].join(' ');
      return `
        <div class="${classes}">
          <strong>${hotel.hotel_name}</strong>
          <div>Rank ${hotel.candidate_rank || 'n/a'} · Score ${compactValue(hotel.hotel_score)}</div>
          <div>${hotel.selected ? 'Selected by optimizer' : 'Candidate alternative'} · ${hotel.source || ''}</div>
          <div>${hotel.selected_hotel_reason || ''}</div>
          <div class="hotel-actions">
            <button type="button" data-prefer-hotel="${hotel.hotel_name}" data-hotel-city="${city}">${isPreferred ? 'Preferred' : 'Prefer'}</button>
            ${isPreferred ? `<button type="button" data-clear-hotel-preference="${city}">Clear</button>` : ''}
          </div>
        </div>
      `;
    }).join('');
    const limitedNote = limited ? '<div class="interest-note">Limited hotel data: only preview fallback exported for this city. Rerun with the hotel catalog to show more options.</div>' : '';
    return `<div class="city-card"><strong>${city}</strong>${limitedNote}${cards}</div>`;
  }).join('');
  const candidateButton = target.querySelector('[data-toggle-hotel-candidates]');
  candidateButton?.addEventListener('click', () => {
    if (window.toggleHotelCandidates) {
      const nextVisible = !(window.activeRouteIds?.has && window.activeRouteIds.has('hotel_candidates'));
      window.toggleHotelCandidates(nextVisible);
      candidateButton.textContent = nextVisible ? 'Hide hotel candidates' : 'Show hotel candidates';
    }
  });
  target.querySelectorAll('[data-prefer-hotel]').forEach(button => {
    button.addEventListener('click', () => {
      setPreferredHotel(button.dataset.hotelCity, button.dataset.preferHotel);
      renderHotelChoices(payload);
    });
  });
  target.querySelectorAll('[data-clear-hotel-preference]').forEach(button => {
    button.addEventListener('click', () => {
      clearPreferredHotel(button.dataset.clearHotelPreference);
      renderHotelChoices(payload);
    });
  });
  target.querySelector('[data-export-hotel-preferences]')?.addEventListener('click', () => {
    downloadPreferredHotelConfig(payload);
  });
}

function natureMatchesFilters(item, filters) {
  if (!filters.size) return true;
  if (filters.has('national_park') && item.is_national_park) return true;
  if (filters.has('state_or_protected') && (item.is_state_park || item.is_protected_area)) return true;
  if (filters.has('scenic_viewpoint') && item.is_scenic_viewpoint) return true;
  if (filters.has('hiking') && item.is_hiking) return true;
  return false;
}

function renderNatureExplore(payload) {
  const target = document.getElementById('nature-explore');
  if (!target) return;
  const items = payload?.items || [];
  if (!items.length) {
    target.innerHTML = `<div class="summary-list">${payload?.message || 'No nature candidates found.'}</div>`;
    return;
  }
  target.innerHTML = `
    <div class="filter-row">
      <button type="button" data-show-nature-layer="true">Show nature layer</button>
      <label><input type="checkbox" data-nature-filter="national_park"> National parks</label>
      <label><input type="checkbox" data-nature-filter="state_or_protected"> Protected</label>
      <label><input type="checkbox" data-nature-filter="scenic_viewpoint"> Scenic</label>
      <label><input type="checkbox" data-nature-filter="hiking"> Hiking</label>
    </div>
    <div id="nature-card-list"></div>
  `;
  const renderCards = () => {
    const filters = new Set([...target.querySelectorAll('[data-nature-filter]:checked')].map(input => input.dataset.natureFilter));
    const filtered = items.filter(item => natureMatchesFilters(item, filters)).slice(0, 12);
    target.querySelector('#nature-card-list').innerHTML = filtered.map(item => `
      <div class="nature-card">
        <strong>${item.name}</strong>
        <div>${item.city || item.nature_region || ''} · ${item.park_type || item.category || ''}</div>
        <div>Nature/scenic/hiking: ${compactValue(item.nature_score)} / ${compactValue(item.scenic_score)} / ${compactValue(item.hiking_score)}</div>
        <div>Weather sensitivity: ${compactValue(item.weather_sensitivity)} · Season risk: ${compactValue(item.seasonality_risk)}</div>
        <div>${item.source_list || ''}</div>
      </div>
    `).join('');
  };
  target.querySelectorAll('[data-nature-filter]').forEach(input => input.addEventListener('change', renderCards));
  target.querySelector('[data-show-nature-layer]')?.addEventListener('click', () => {
    if (window.toggleRoute) window.toggleRoute('nature_candidates', true);
  });
  renderCards();
}

function renderDebugSummary(summary) {
  renderObjectSummary('debug-summary', 'debug summary', summary);
}

function interestWeightsFromPreview(preview) {
  const fromWeights = preview?.weights || {};
  const fallback = { nature: 0.25, city: 0.25, culture: 0.25, history: 0.25 };
  return {
    nature: Number(fromWeights.nature ?? fallback.nature),
    city: Number(fromWeights.city ?? fallback.city),
    culture: Number(fromWeights.culture ?? fallback.culture),
    history: Number(fromWeights.history ?? fallback.history)
  };
}

function normalizeInterestBarValues(raw) {
  const axes = ['nature', 'city', 'culture', 'history'];
  const total = axes.reduce((sum, axis) => sum + Math.max(0, Number(raw[axis]) || 0), 0);
  if (total <= 0) return { nature: 0.25, city: 0.25, culture: 0.25, history: 0.25 };
  return Object.fromEntries(axes.map(axis => [axis, Math.max(0, Number(raw[axis]) || 0) / total]));
}

function nearestInterestProfile(preview, weights) {
  const presets = preview?.preset_weights || {};
  const threshold = Number(preview?.saved_profile_match_threshold ?? 0.12);
  let best = { profile: '', distance: Number.POSITIVE_INFINITY, threshold, weights: null };
  Object.entries(presets).forEach(([profile, presetWeights]) => {
    const distance = ['nature', 'city', 'culture', 'history']
      .reduce((sum, axis) => sum + Math.abs(Number(weights[axis] || 0) - Number(presetWeights?.[axis] || 0)), 0);
    if (distance < best.distance) best = { profile, distance, threshold, weights: presetWeights };
  });
  best.matched = Boolean(best.profile) && best.distance <= threshold;
  return best;
}

function savedInterestProfileRoute(profile) {
  const routes = Array.isArray(window.dashboardRouteIndex?.routes) ? window.dashboardRouteIndex.routes : [];
  return routes.find(route =>
    route.family === 'interest_profile'
    && String(route.interest_profile || route.profile || '').toLowerCase() === String(profile || '').toLowerCase()
  );
}

function previewGraphForWeights(preview, weights) {
  const match = nearestInterestProfile(preview, weights);
  const graphs = preview?.preview_route_graphs || {};
  return {
    match,
    graph: graphs[match.profile] || preview?.preview_route_graph || {}
  };
}

function previewItemAxis(item, axis) {
  if (axis === 'city') return Number(item.city_axis ?? item.city ?? item.city_score ?? 0);
  return Number(item[axis] ?? item[`${axis}_score`] ?? 0);
}

function previewAxisFit(item, weights) {
  return ['nature', 'city', 'culture', 'history']
    .reduce((sum, axis) => sum + (weights[axis] || 0) * Math.max(0, previewItemAxis(item, axis) || 0), 0);
}

function dominantPreviewAxis(item) {
  const axes = ['nature', 'city', 'culture', 'history'];
  return axes
    .map(axis => ({ axis, value: Math.max(0, previewItemAxis(item, axis) || 0) }))
    .sort((a, b) => b.value - a.value)[0] || { axis: 'nature', value: 0 };
}

function candidateAllowedByWeights(item, weights) {
  const dominant = dominantPreviewAxis(item);
  const dominantWeight = Number(weights[dominant.axis] || 0);
  const nonNatureFit = ['city', 'culture', 'history']
    .reduce((sum, axis) => sum + Number(weights[axis] || 0) * Math.max(0, previewItemAxis(item, axis) || 0), 0);
  const natureSignal = Math.max(
    Number(item.nature_score || 0),
    Number(item.scenic_score || 0),
    Number(item.park_bonus || 0),
    item.is_national_park ? 1 : 0
  );
  if (Number(weights.nature || 0) <= 0.001 && natureSignal >= 0.35 && nonNatureFit < 0.18) return false;
  if (dominant.value >= 0.35 && dominantWeight <= 0.001) return false;
  return previewAxisFit(item, weights) > 0.02 || dominant.value < 0.2;
}

function previewScore(item, weights, lambdas) {
  const fit = previewAxisFit(item, weights);
  const base = Math.min(Number(item.final_poi_value ?? item.interest_adjusted_value ?? 0) || 0, 120) / 120;
  const park = Number(weights.nature || 0) * Number(item.park_bonus || 0);
  const weather = Number(item.weather_sensitivity || 0) * Number(item.weather_risk || 0.15);
  const season = Number(item.seasonality_risk || 0);
  const detour = Number(item.detour_minutes || 0);
  return base
    + Number(lambdas.lambda_fit ?? 0.65) * fit
    + Number(lambdas.lambda_park ?? 0.35) * park
    - Number(lambdas.lambda_weather ?? 0.30) * weather
    - Number(lambdas.lambda_season ?? 0.20) * season
    - Number(lambdas.lambda_detour ?? 0.006) * detour;
}

function renderInterestPreview(preview) {
  const target = document.getElementById('interest-preview');
  if (!target) return;
  if (!preview || preview.available === false) {
    target.innerHTML = `<div class="summary-list">${preview?.message || 'No interest preview artifact available.'}</div>`;
    return;
  }
  const axes = ['nature', 'city', 'culture', 'history'];
  const weights = interestWeightsFromPreview(preview);
  target.innerHTML = `
    <div id="interest-bar-controls">
      ${axes.map(axis => `
        <label class="interest-axis">
          <span>${axis}</span>
          <input type="range" min="0" max="100" value="${Math.round((weights[axis] || 0) * 100)}" data-interest-axis="${axis}">
          <strong data-interest-percent="${axis}">0%</strong>
        </label>
      `).join('')}
    </div>
    <div class="interest-note">Saved optimized route stays artifact-backed. Dragging bars builds a browser preview only.</div>
    <div class="interest-note">National park candidates stay in candidate/context layers unless the selected route artifact includes them.</div>
    <div class="playback-controls">
      <button type="button" data-build-live-preview="true">Build live preview route</button>
      <button type="button" data-clear-live-preview="true">Clear preview route</button>
    </div>
    <div class="interest-note"><b>Preview only - rerun pipeline to save.</b> Browser preview is approximate; Run All recomputes the real optimized route.</div>
    <div id="interest-route-match" class="summary-list"></div>
    <div id="interest-ranking" class="interest-ranking"></div>
  `;
  let previewDebounce = null;
  let latestPreviewCandidates = [];
  const renderRanking = () => {
    const raw = Object.fromEntries(axes.map(axis => [
      axis,
      Number(target.querySelector(`[data-interest-axis="${axis}"]`)?.value || 0)
    ]));
    const normalized = normalizeInterestBarValues(raw);
    axes.forEach(axis => {
      const node = target.querySelector(`[data-interest-percent="${axis}"]`);
      if (node) node.textContent = `${Math.round(normalized[axis] * 100)}%`;
    });
    const routeMix = preview.route_mix || {};
    const profileMatch = nearestInterestProfile(preview, normalized);
    const savedProfile = savedInterestProfileRoute(profileMatch.profile);
    const routeError = axes.reduce((sum, axis) => {
      const mixValue = Number(routeMix[axis] ?? routeMix[`${axis}_score`] ?? 0);
      return sum + Math.abs((normalized[axis] || 0) - mixValue);
    }, 0);
    const routeScore = Math.max(0, 1 - routeError);
    const matchNode = target.querySelector('#interest-route-match');
    if (matchNode) {
      matchNode.innerHTML = `
        <div><b>Saved route match:</b> ${compactValue(routeScore, 2)}</div>
        <div><b>Closest saved profile:</b> ${profileMatch.profile || 'none'}${profileMatch.matched && savedProfile ? ' (available)' : ''}</div>
        <div><b>Route state:</b> ${profileMatch.matched && savedProfile ? 'Saved optimized profile route available' : 'Preview only - rerun pipeline to save'}</div>
      `;
    }
    const candidates = (preview.preview_candidates || preview.top_boosted_pois || [])
      .map(item => ({ ...item, preview_score: previewScore(item, normalized, preview.lambdas || {}) }))
      .sort((a, b) => b.preview_score - a.preview_score)
    .filter(item => candidateAllowedByWeights(item, normalized))
    .slice(0, 8);
    latestPreviewCandidates = candidates;
    target.querySelector('#interest-ranking').innerHTML = candidates.map(item => `
      <div class="nature-card">
        <strong>${item.name}</strong>
        <div>${item.city || ''} · ${item.nature_region || item.park_type || ''}</div>
        <div>Preview score: ${compactValue(item.preview_score)} · Fit: ${compactValue(previewAxisFit(item, normalized))}</div>
      </div>
    `).join('');
  };
  const currentWeights = () => normalizeInterestBarValues(Object.fromEntries(axes.map(axis => [
    axis,
    Number(target.querySelector(`[data-interest-axis="${axis}"]`)?.value || 0)
  ])));
  const previewId = item => String(item.id || item.poi_id || item.place_id || item.name || '')
    .toLowerCase()
    .trim()
    .replace(/\\s+/g, '_');
  const itemCityText = item => `${item.name || ''} ${item.city || ''} ${item.city_or_anchor || ''} ${item.nature_region || ''} ${item.category || ''}`.toLowerCase();
  const endpointPoint = (endpoint, side) => endpoint && Number.isFinite(Number(endpoint.lat)) && Number.isFinite(Number(endpoint.lon))
    ? {
        name: endpoint.name || endpoint.code || 'Airport endpoint',
        city: endpoint.city || '',
        city_or_anchor: endpoint.name || endpoint.code || '',
        lat: Number(endpoint.lat),
        lon: Number(endpoint.lon),
        day: side === 'start' ? 1 : Number(window.dashboardMetrics?.trip_days || 0) || 1,
        stop_order: side === 'start' ? 0 : 999,
        type: 'airport_endpoint',
        category: 'airport_endpoint',
        is_route_endpoint: true,
        is_airport_endpoint: true,
        is_selected_poi: false,
        is_hotel_node: false,
        node_kind: 'airport',
        display_sequence_index: endpoint.code || 'AP',
        selected_stop_index: 0,
        route_sequence_index: 0,
        route_endpoint_side: side,
        airport_code: endpoint.code || ''
      }
    : null;
  const rankedPreviewCandidates = weights => {
    const previewItems = Array.isArray(preview.preview_candidates)
      ? preview.preview_candidates
      : (Array.isArray(preview.top_boosted_pois) ? preview.top_boosted_pois : []);
    const natureItems = Array.isArray(window.dashboardNatureExplore?.items) ? window.dashboardNatureExplore.items : [];
    const merged = [...previewItems, ...natureItems];
    const seen = new Set();
    return merged
      .filter(item => Number.isFinite(Number(item.lat)) && Number.isFinite(Number(item.lon)))
      .filter(item => {
        const key = `${item.name || ''}:${item.lat}:${item.lon}`;
        if (seen.has(key)) return false;
        seen.add(key);
        return true;
      })
      .filter(item => candidateAllowedByWeights(item, weights))
      .map(item => ({ ...item, preview_score: previewScore(item, weights, preview.lambdas || {}) }))
      .sort((a, b) => b.preview_score - a.preview_score);
  };
  const candidatesForSegment = (ranked, segment, seenKeys) => {
    const candidateIds = new Set((segment.candidate_ids || []).map(value => String(value).toLowerCase()));
    const allowed = (segment.allowed_cities || [])
      .map(value => String(value).toLowerCase())
      .filter(Boolean);
    return ranked.filter(item => {
      const key = `${item.name || ''}:${item.lat}:${item.lon}`.toLowerCase();
      if (seenKeys.has(key)) return false;
      const text = itemCityText(item);
      const id = previewId(item);
      if (candidateIds.has(id)) return true;
      return allowed.some(city => text.includes(city));
    });
  };
  const buildSegmentOrderedPreview = weights => {
    const ranked = rankedPreviewCandidates(weights);
    const graphSelection = previewGraphForWeights(preview, weights);
    const previewGraph = graphSelection.graph || {};
    const segments = Array.isArray(previewGraph.segments) ? previewGraph.segments : [];
    const route = [];
    const missing = [];
    const seen = new Set();
    const start = endpointPoint(previewGraph.start_endpoint, 'start');
    const end = endpointPoint(previewGraph.end_endpoint, 'end');
    if (start) route.push(start);
    if (!segments.length) {
      ranked.slice(0, Math.max(2, Math.min(10, Number(window.dashboardMetrics?.selected_stop_count || 8)))).forEach(item => route.push(item));
      if (end) route.push(end);
      return { route, sequenceFeasible: false, missingSegments: ['no segment graph exported'] };
    }
    segments.forEach(segment => {
      const limit = Math.max(1, Number(segment.stop_limit || 1));
      const picked = candidatesForSegment(ranked, segment, seen).slice(0, limit);
      if (!picked.length) {
        missing.push(`${segment.start_city || '?'} to ${segment.end_city || '?'}`);
      }
      picked.forEach(item => {
        const key = `${item.name || ''}:${item.lat}:${item.lon}`.toLowerCase();
        seen.add(key);
        route.push({
          ...item,
          segment_index: segment.segment_index,
          allowed_cities: (segment.allowed_cities || []).join(', ')
        });
      });
    });
    if (end) route.push(end);
    return { route, sequenceFeasible: missing.length === 0, missingSegments: missing };
  };
  const buildLivePreviewRoute = () => {
    const weights = currentWeights();
    const profileMatch = nearestInterestProfile(preview, weights);
    const savedProfileRoute = profileMatch.matched ? savedInterestProfileRoute(profileMatch.profile) : null;
    if (savedProfileRoute && window.showOnlyRoute) {
      window.showOnlyRoute(savedProfileRoute);
      if (window.clearLivePreviewRoute) window.clearLivePreviewRoute(false);
      const matchNode = target.querySelector('#interest-route-match');
      if (matchNode) {
        matchNode.innerHTML = `
          <div><b>Saved profile route:</b> ${savedProfileRoute.label || profileMatch.profile}</div>
          <div><b>Profile distance:</b> ${compactValue(profileMatch.distance, 2)} (threshold ${compactValue(profileMatch.threshold, 2)})</div>
          <div><b>Route state:</b> Saved optimized route artifact</div>
        `;
      }
      const status = document.getElementById('dashboard-status');
      if (status) {
        status.textContent = `Loaded saved ${profileMatch.profile} route artifact.`;
        status.style.color = '#0a6b53';
      }
      return;
    }
    const previewRoute = buildSegmentOrderedPreview(weights);
    const candidates = previewRoute.route;
    if (!candidates.filter(item => !item.is_route_endpoint).length) {
      const status = document.getElementById('dashboard-status');
      if (status) {
        status.textContent = 'No exported POI/nature candidates with coordinates are available for the live preview route.';
        status.style.color = '#b91c1c';
      }
      return;
    }
    if (window.drawLivePreviewRoute) {
      window.drawLivePreviewRoute(candidates, { ...weights, __sequence_feasible: previewRoute.sequenceFeasible });
    }
    const matchNode = target.querySelector('#interest-route-match');
    const score = candidates.reduce((sum, item) => sum + Number(item.preview_score || 0), 0);
    const natureStops = candidates.filter(item => {
      const text = `${item.name || ''} ${item.category || ''} ${item.nature_region || ''} ${item.park_type || ''}`.toLowerCase();
      return Number(item.nature_score || 0) >= 0.35
        || Number(item.scenic_score || 0) >= 0.35
        || Number(item.park_bonus || 0) > 0
        || item.is_national_park
        || /park|nature|scenic|hiking|viewpoint|yosemite|sequoia|kings canyon|joshua tree|big sur/.test(text);
    }).length;
    if (matchNode) {
      matchNode.innerHTML = `
        <div><b>Preview route score:</b> ${compactValue(score)}</div>
        <div><b>Preview stops:</b> ${candidates.filter(item => !item.is_route_endpoint).length}</div>
        <div><b>Preview nature stops:</b> ${natureStops}</div>
        <div><b>Sequence:</b> ${previewRoute.sequenceFeasible ? 'feasible' : `needs rerun / missing ${previewRoute.missingSegments.join('; ')}`}</div>
        <div><b>Route state:</b> Preview only - rerun pipeline to save</div>
      `;
    }
  };
  target.querySelector('[data-build-live-preview]')?.addEventListener('click', buildLivePreviewRoute);
  target.querySelector('[data-clear-live-preview]')?.addEventListener('click', () => {
    if (window.clearLivePreviewRoute) window.clearLivePreviewRoute();
  });
  target.querySelectorAll('[data-interest-axis]').forEach(input => input.addEventListener('input', () => {
    renderRanking();
    if (previewDebounce) window.clearTimeout(previewDebounce);
    previewDebounce = window.setTimeout(buildLivePreviewRoute, 260);
  }));
  renderRanking();
}

function routeButtonLabel(route) {
  if (route.customer_control_group === 'trip_days') return `${route.trip_days || '?'} days`;
  if (route.customer_control_group === 'interest') {
    return String(route.interest_profile || route.profile || route.label || 'Interest').replaceAll('_', ' ');
  }
  return route.label || 'Saved route';
}

function customerVisibleRoutes(index, controlGroup) {
  const routes = Array.isArray(index?.routes) ? index.routes : [];
  return routes
    .filter(route => route.customer_visible !== false)
    .filter(route => !controlGroup || route.customer_control_group === controlGroup)
    .filter(route => route.playable !== false && route.marker_only !== true);
}

function activateCustomerRoute(route, statusText) {
  if (!route || !window.showOnlyRoute) return;
  if (window.clearLivePreviewRoute) window.clearLivePreviewRoute(false);
  window.showOnlyRoute(route);
  const status = document.getElementById('dashboard-status');
  if (status) {
    status.textContent = statusText || `Loaded saved route artifact: ${route.label || route.id}`;
    status.style.color = '#0a6b53';
  }
  document.querySelectorAll('[data-customer-route-id]').forEach(button => {
    button.classList.toggle('active', button.dataset.customerRouteId === route.id);
  });
}

function renderCustomerControls(index) {
  const target = document.getElementById('customer-trip-controls');
  if (!target) return;
  const defaultRoute = customerVisibleRoutes(index).find(route => route.default);
  const dayRoutes = customerVisibleRoutes(index, 'trip_days')
    .sort((a, b) => Number(a.trip_days || 0) - Number(b.trip_days || 0));
  const interestRoutes = customerVisibleRoutes(index, 'interest')
    .sort((a, b) => String(a.interest_profile || a.profile || a.label).localeCompare(String(b.interest_profile || b.profile || b.label)));
  const dayButtons = dayRoutes.length
    ? dayRoutes.map(route => `<button type="button" data-customer-route-id="${escapeHtml(route.id)}" data-customer-route-kind="days">${escapeHtml(routeButtonLabel(route))}</button>`).join('')
    : '<span class="interest-note">No alternate day-count artifacts were exported. Requires pipeline rerun.</span>';
  const interestButtons = interestRoutes.length
    ? interestRoutes.map(route => `<button type="button" data-customer-route-id="${escapeHtml(route.id)}" data-customer-route-kind="interest">${escapeHtml(routeButtonLabel(route))}</button>`).join('')
    : '<span class="interest-note">No saved interest-profile artifacts were exported. Slider changes are preview-only.</span>';
  target.innerHTML = `
    <div class="customer-control-grid">
      <div class="customer-control-block">
        <strong>Trip length</strong>
        <div class="customer-option-grid">${dayButtons}</div>
        <div class="interest-note">Only exported trip-day artifacts can be selected here; other day counts require a Python pipeline rerun.</div>
      </div>
      <div class="customer-control-block">
        <strong>Interest preset</strong>
        <div class="customer-option-grid">${interestButtons}</div>
        <div class="interest-note">Saved profile routes are optimizer artifacts. Custom slider mixes remain preview-only.</div>
      </div>
      <div class="customer-control-block">
        <strong>Hotels</strong>
        <div class="interest-note">Hotel preferences are saved locally in this browser and can be exported as a soft bonus for the next optimizer rerun.</div>
      </div>
    </div>
  `;
  const byId = new Map(customerVisibleRoutes(index).map(route => [route.id, route]));
  target.querySelectorAll('[data-customer-route-id]').forEach(button => {
    button.addEventListener('click', () => {
      const route = byId.get(button.dataset.customerRouteId);
      const kind = button.dataset.customerRouteKind === 'days' ? 'trip length' : 'interest profile';
      activateCustomerRoute(route, `Loaded saved ${kind} route artifact: ${route?.label || route?.id || ''}`);
    });
  });
  if (defaultRoute) {
    target.querySelector(`[data-customer-route-id="${CSS.escape(defaultRoute.id)}"]`)?.classList.add('active');
  }
}

window.renderDashboardMetrics = renderDashboardMetrics;
window.bindDashboardShellControls = bindDashboardShellControls;
window.renderActiveStopDetail = renderActiveStopDetail;
window.renderCityDetails = renderCityDetails;
window.renderHotelChoices = renderHotelChoices;
window.renderNatureExplore = renderNatureExplore;
window.renderDebugSummary = renderDebugSummary;
window.renderInterestPreview = renderInterestPreview;
window.renderCustomerControls = renderCustomerControls;
""",
        written,
    )

    research_html = _dashboard_page_html("research")
    _write_text_asset(root / "index.html", research_html, written)
    _write_text_asset(root / "research.html", research_html, written)
    _write_text_asset(root / "customer.html", _dashboard_page_html("customer"), written)
    return written


def _artifact_row(
    path: Path, artifact_type: str, layer_count: int, route_count: int, marker_count: int, notes: str
) -> dict[str, Any]:
    size_mb = path.stat().st_size / (1024 * 1024) if path.exists() else 0.0
    return {
        "artifact_path": str(path),
        "artifact_type": artifact_type,
        "file_size_mb": round(size_mb, 4),
        "layer_count": int(layer_count),
        "route_count": int(route_count),
        "marker_count": int(marker_count),
        "notes": notes,
    }


def _feature_count(path: Path) -> int:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return 0
    if isinstance(data, dict) and isinstance(data.get("features"), list):
        return len(data["features"])
    if isinstance(data, list):
        return len(data)
    return 0


def export_map_artifacts(
    route_df: pd.DataFrame,
    *,
    output_dir: str | Path,
    figure_dir: str | Path,
    config: TripConfig,
) -> dict[str, Path]:
    """Export small share and modular dashboard map artifacts."""
    output_dir = Path(output_dir)
    figure_dir = Path(figure_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    max_routes = int(config.get("map_export", "lightweight_max_routes", 1))
    route_records, route_geojsons, route_pois, metrics, points, route_label = _build_dashboard_payloads(
        route_df, output_dir=output_dir, config=config, max_routes=max_routes
    )
    debug_summary = _debug_summary(output_dir)
    interest_preview = _interest_preview(output_dir)
    playback_data = _playback_data(route_records, route_pois)
    city_details = _city_details(output_dir, route_records, route_pois)
    hotel_choices = _hotel_choices(output_dir)
    selected_hotels = _selected_hotels(points, hotel_choices)
    nature_explore = _nature_explore(output_dir)
    evaluation_metrics = _evaluation_metrics(output_dir)
    mode = str(config.get("map_export", "mode", "both")).lower()

    artifacts: dict[str, Path] = {}
    report_rows = []
    if mode in {"both", "lightweight", "share"}:
        share_path = figure_dir / "lightweight_share_map.html"
        _write_lightweight_map(share_path, points, metrics)
        artifacts["lightweight_share_map"] = share_path
        report_rows.append(
            _artifact_row(
                share_path,
                "lightweight_share_map",
                1,
                1 if len(points) > 1 else 0,
                len(points),
                "selected route only; no comparison or debug layers",
            )
        )

    if mode in {"both", "full", "dashboard"}:
        dashboard_root = figure_dir / "full_interactive_dashboard"
        written = _write_full_dashboard(
            dashboard_root,
            route_records,
            route_geojsons,
            route_pois,
            metrics,
            debug_summary,
            interest_preview,
            playback_data,
            city_details,
            selected_hotels,
            hotel_choices,
            nature_explore,
            evaluation_metrics,
        )
        artifacts["full_interactive_dashboard"] = dashboard_root / "index.html"
        artifacts["full_research_dashboard"] = dashboard_root / "research.html"
        artifacts["full_customer_dashboard"] = dashboard_root / "customer.html"
        artifacts["evaluation_dashboard"] = dashboard_root / "evaluation.html"
        route_notes = {}
        for record in route_records:
            route_notes[str(record["geojson"])] = record
            route_notes[str(record["geojson_js"])] = record
            route_notes[str(record["pois"])] = record
            route_notes[str(record["pois_js"])] = record
        for path in written:
            rel_path = path.relative_to(dashboard_root).as_posix() if dashboard_root in path.parents else path.name
            record = route_notes.get(rel_path)
            record_note = ""
            if record:
                record_note = (
                    "selected_default_route"
                    if record.get("default")
                    else f"optional_layer;family={record.get('family')}"
                )
            if path.name == "index.html":
                artifact_type = "full_dashboard_index"
                route_count = len(route_records)
                marker_count = len(points)
                notes = "research_dashboard_entrypoint;compat_index"
            elif path.name == "research.html":
                artifact_type = "full_research_dashboard"
                route_count = len(route_records)
                marker_count = len(points)
                notes = "explicit_research_dashboard_entrypoint"
            elif path.name == "customer.html":
                artifact_type = "full_customer_dashboard"
                route_count = len([record for record in route_records if record.get("customer_visible")])
                marker_count = len(points)
                notes = "customer_dashboard_entrypoint"
            elif path.suffix == ".geojson":
                artifact_type = "route_geojson"
                route_count = 1
                marker_count = max(0, _feature_count(path) - 1)
                notes = f"json_fetch_asset;{record_note}".rstrip(";")
            elif "routes" in path.parts and path.suffix == ".js":
                artifact_type = "route_js_fallback"
                route_count = 1
                marker_count = max(0, _feature_count(path.with_suffix(".geojson")) - 1)
                notes = f"file_mode_js_fallback;{record_note}".rstrip(";")
            elif "pois" in path.parts and path.suffix == ".json":
                artifact_type = "poi_json"
                route_count = 0
                marker_count = _feature_count(path)
                notes = f"json_fetch_asset;{record_note}".rstrip(";")
            elif "pois" in path.parts and path.suffix == ".js":
                artifact_type = "poi_js_fallback"
                route_count = 0
                marker_count = _feature_count(path.with_suffix(".json"))
                notes = f"file_mode_js_fallback;{record_note}".rstrip(";")
            elif path.suffix == ".json":
                artifact_type = "dashboard_json"
                route_count = 0
                marker_count = 0
                notes = "json_fetch_asset"
            elif path.suffix == ".css":
                artifact_type = "dashboard_css"
                route_count = 0
                marker_count = 0
                notes = "dashboard_style"
            elif path.suffix == ".js":
                artifact_type = "dashboard_js"
                route_count = 0
                marker_count = 0
                notes = (
                    "file_mode_js_fallback"
                    if path.name in {"route_index.js", "dashboard_metrics.js"}
                    else "dashboard_behavior"
                )
            else:
                artifact_type = "full_interactive_dashboard_asset"
                route_count = 0
                marker_count = 0
                notes = "externalized modular dashboard asset"
            report_rows.append(_artifact_row(path, artifact_type, 1, route_count, marker_count, notes))

    monolith_path = figure_dir / "production_hierarchical_trip_map.html"
    if monolith_path.exists():
        report_rows.append(
            _artifact_row(
                monolith_path,
                "legacy_monolithic_folium_html",
                0,
                0,
                0,
                "legacy compatibility artifact; not the preferred share export",
            )
        )

    pd.DataFrame(report_rows).to_csv(output_dir / "production_map_artifact_size_report.csv", index=False)
    return artifacts
