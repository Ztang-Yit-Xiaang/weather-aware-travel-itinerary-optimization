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
from .dashboard_assets import dashboard_stylesheet
from .dashboard_data_loader import dashboard_data_loader_script
from .dashboard_evaluation import EVALUATION_METHODS as _EVALUATION_METHODS
from .dashboard_evaluation import build_evaluation_metrics, evaluation_page_html
from .dashboard_map_controls import dashboard_map_controls_script
from .dashboard_ui import dashboard_ui_script
from .region_scenarios import get_scenario_definition

EVALUATION_METHODS = _EVALUATION_METHODS
DEFAULT_DASHBOARD_ROUTE_ID = "selected_route"
DEFAULT_DASHBOARD_METHOD = "hierarchical_bandit_gurobi_repair"
CONTRACT_VERSION = "core-route-index-v1"
PLAYABLE_ROUTE_FAMILIES = {"selected", "route_matrix", "trip_length", "method", "interest_profile"}
MARKER_ONLY_ROUTE_FAMILIES = {"hotel", "nature", "must_go"}
CUSTOMER_ROUTE_FAMILIES = {"selected", "trip_length", "interest_profile", "nature_detail"}
CUSTOMER_CONTROL_GROUPS = {
    "selected": "saved_route",
    "trip_length": "trip_days",
    "interest_profile": "interest",
    "nature_detail": "nature_site_routes",
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
    for column in [
        "expected_duration_minutes",
        "visit_duration_minutes",
        "visit_duration_sim",
        "available_visit_minutes",
    ]:
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
        display_sequence_index = (
            route_sequence_index if route_sequence_index > 0 else (len(records) + 1 if selected_poi else 0)
        )
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
                "internal_route_count": int(_safe_float(row.get("internal_route_count", 0.0))),
                "best_internal_route_score": _safe_float(row.get("best_internal_route_score", 0.0)),
                "best_internal_route_distance_km": _safe_float(row.get("best_internal_route_distance_km", 0.0)),
                "best_internal_route_duration_minutes": _safe_float(
                    row.get("best_internal_route_duration_minutes", 0.0)
                ),
                "internal_route_confidence": _safe_float(row.get("internal_route_confidence", 0.0)),
                "internal_route_source": _safe_str(row.get("internal_route_source", "")),
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
        day_points = [point for point in ordered_points if int(_safe_float(point.get("day", 0), 0.0)) == day]
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


def _cached_route_geometry(points: list[dict[str, Any]], output_dir: Path) -> tuple[list[list[float]], str, bool]:
    """Return cached OSRM geometry in GeoJSON lon/lat order, or straight fallback."""
    coordinates = [
        [float(point["lat"]), float(point["lon"])] for point in points if point.get("lat") and point.get("lon")
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
        routed = [[float(lon), float(lat)] for lat, lon in latlon if lat is not None and lon is not None]
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
        frame = frame[
            frame["itinerary_eligible"].fillna(True).astype(str).str.lower().isin({"1", "true", "yes"})
        ].copy()
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


def _nature_site_route_assets(output_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    routes = _read_csv_if_present(output_dir / "production_nature_site_routes.csv")
    points = _read_csv_if_present(output_dir / "production_nature_site_route_points.csv")
    audit = _read_csv_if_present(output_dir / "production_nature_site_route_audit.csv")
    if routes.empty or points.empty or "route_id" not in routes.columns or "route_id" not in points.columns:
        audit_items = []
        if not audit.empty:
            audit_items = [
                {
                    "site_id": _safe_str(row.get("site_id", "")),
                    "site_name": _safe_str(row.get("site_name", "")),
                    "route_count": int(_safe_float(row.get("route_count", 0))),
                    "source_status": _safe_str(row.get("source_status", "")),
                    "missing_reason": _safe_str(row.get("missing_reason", "")),
                }
                for _, row in audit.iterrows()
            ]
        return (
            {"type": "FeatureCollection", "features": []},
            [],
            {
                "available": False,
                "routes": [],
                "sites": {},
                "audit": audit_items,
                "message": "No production_nature_site_routes.csv artifact found.",
            },
        )

    route_by_id = {str(row.get("route_id", "")): row for _, row in routes.iterrows()}
    features: list[dict[str, Any]] = []
    pois: list[dict[str, Any]] = []
    payload_routes: list[dict[str, Any]] = []
    for route_id, group in points.groupby(points["route_id"].fillna("").astype(str), sort=False):
        if not route_id or route_id not in route_by_id:
            continue
        group = group.sort_values("point_order") if "point_order" in group.columns else group
        coords = [
            [_safe_float(row.get("longitude")), _safe_float(row.get("latitude"))]
            for _, row in group.iterrows()
            if _safe_float(row.get("latitude")) and _safe_float(row.get("longitude"))
        ]
        if len(coords) < 2:
            continue
        route = route_by_id[route_id]
        props = {
            "route_id": route_id,
            "name": _safe_str(route.get("route_name", route_id)),
            "route_name": _safe_str(route.get("route_name", route_id)),
            "site_id": _safe_str(route.get("site_id", "")),
            "site_name": _safe_str(route.get("site_name", "")),
            "city": _safe_str(route.get("city", "")),
            "nature_region": _safe_str(route.get("nature_region", "")),
            "lat": _safe_float(route.get("latitude", 0.0)),
            "lon": _safe_float(route.get("longitude", 0.0)),
            "route_type": _safe_str(route.get("route_type", "")),
            "distance_km": _safe_float(route.get("distance_km", 0.0)),
            "duration_minutes": int(_safe_float(route.get("duration_minutes", 0.0))),
            "difficulty": _safe_str(route.get("difficulty", "")),
            "route_score": _safe_float(route.get("route_score", 0.0)),
            "source_confidence": _safe_float(route.get("source_confidence", 0.0)),
            "source": _safe_str(route.get("source", "")),
            "source_url": _safe_str(route.get("source_url", "")),
            "description": _safe_str(route.get("description", "")),
            "fallback_used": _safe_bool(route.get("fallback_used", False)),
        }
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": coords},
                "properties": {**props, "role": "nature_site_route"},
            }
        )
        first_lon, first_lat = coords[0]
        pois.append(
            {
                "name": props["name"],
                "city": props["city"],
                "city_or_anchor": props["site_name"] or props["nature_region"] or props["city"],
                "lat": first_lat,
                "lon": first_lon,
                "category": "nature_site_route",
                "type": props["route_type"],
                "node_kind": "nature_site_route",
                "is_selected_poi": False,
                "is_hotel_node": False,
                "is_airport_endpoint": False,
                "nature_region": props["nature_region"],
                "site_id": props["site_id"],
                "site_name": props["site_name"],
                "route_id": route_id,
                "route_name": props["name"],
                "route_type": props["route_type"],
                "distance_km": props["distance_km"],
                "duration_minutes": props["duration_minutes"],
                "difficulty": props["difficulty"],
                "route_score": props["route_score"],
                "source_confidence": props["source_confidence"],
                "description": props["description"],
                "why_selected": "Internal nature-site route detail; not counted as a selected itinerary stop.",
                "source_url": props["source_url"],
                "source_list": props["source"],
            }
        )
        payload_routes.append({**props, "route_id": route_id, "point_count": len(coords)})

    sites: dict[str, dict[str, Any]] = {}
    for route in payload_routes:
        site_key = route.get("site_id") or _slug(route.get("site_name", "nature_site"))
        entry = sites.setdefault(
            site_key,
            {
                "site_id": site_key,
                "site_name": route.get("site_name", ""),
                "city": route.get("city", ""),
                "nature_region": route.get("nature_region", ""),
                "routes": [],
            },
        )
        entry["routes"].append(route)
    audit_items = (
        [
            {
                "site_id": _safe_str(row.get("site_id", "")),
                "site_name": _safe_str(row.get("site_name", "")),
                "route_count": int(_safe_float(row.get("route_count", 0))),
                "source_status": _safe_str(row.get("source_status", "")),
                "fallback_used": _safe_bool(row.get("fallback_used", False)),
                "missing_reason": _safe_str(row.get("missing_reason", "")),
            }
            for _, row in audit.iterrows()
        ]
        if not audit.empty
        else []
    )
    return (
        {"type": "FeatureCollection", "features": features},
        pois,
        {
            "available": bool(payload_routes),
            "routes": payload_routes,
            "sites": sites,
            "audit": audit_items,
            "filters": ["scenic_drive", "short_hike", "viewpoint_walk"],
        },
    )


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
                "hotel_alternative_count": int(
                    entry.get("hotel_alternative_count") or hotel_counts.get(entry["city"], 0)
                ),
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
                "internal_route_count": int(_safe_float(row.get("internal_route_count", 0.0))),
                "best_internal_route_score": _safe_float(row.get("best_internal_route_score", 0.0)),
                "best_internal_route_distance_km": _safe_float(row.get("best_internal_route_distance_km", 0.0)),
                "best_internal_route_duration_minutes": _safe_float(
                    row.get("best_internal_route_duration_minutes", 0.0)
                ),
                "internal_route_confidence": _safe_float(row.get("internal_route_confidence", 0.0)),
                "internal_route_source": _safe_str(row.get("internal_route_source", "")),
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
    nature_site_geojson, nature_site_pois, _nature_site_payload = _nature_site_route_assets(output_dir)
    if nature_site_geojson.get("features") and nature_site_pois:
        record = _route_record(
            "nature_site_routes",
            "Nature site routes",
            default=False,
            optional=True,
            family="nature_detail",
            selector_group="nature_details",
            extra={
                "playable": False,
                "marker_only": False,
                "quick_groups": ["nature_detail", "nature_details", "candidates"],
            },
        )
        records.append(record)
        route_geojsons["nature_site_routes"] = nature_site_geojson
        route_pois["nature_site_routes"] = nature_site_pois

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
        "nature_site_routes.json",
        "nature_site_routes.js",
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


def _evaluation_metrics(output_dir: Path) -> dict[str, Any]:
    return build_evaluation_metrics(
        _read_csv_if_present(output_dir / "production_method_comparison.csv"),
        _read_csv_if_present(output_dir / "production_method_route_stops.csv"),
    )


def _write_evaluation_page(root: Path, assets: Path, metrics: dict[str, Any], written: list[Path]) -> None:
    metrics_json = assets / "evaluation_metrics.json"
    _write_json_asset(metrics_json, metrics, written)
    metrics_js = assets / "evaluation_metrics.js"
    _write_text_asset(metrics_js, _global_assignment("DASHBOARD_EVALUATION_METRICS", metrics), written)
    _write_text_asset(root / "evaluation.html", evaluation_page_html(), written)

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
          <label><input type="checkbox" data-layer-toggle="nature_site_routes" /> Nature site routes</label>
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
      <details class="dashboard-section">
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
    nature_site_routes: dict[str, Any],
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
    _write_text_asset(style_path, dashboard_stylesheet(), written)

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

    nature_site_path = assets / "nature_site_routes.json"
    _write_json_asset(nature_site_path, nature_site_routes, written)

    nature_site_js_path = assets / "nature_site_routes.js"
    _write_text_asset(
        nature_site_js_path,
        _global_assignment("DASHBOARD_NATURE_SITE_ROUTES", nature_site_routes),
        written,
    )

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
    _write_text_asset(data_loader, dashboard_data_loader_script(), written)

    map_controls = assets / "map_controls.js"
    _write_text_asset(map_controls, dashboard_map_controls_script(), written)

    dashboard_js = assets / "dashboard.js"
    _write_text_asset(dashboard_js, dashboard_ui_script(), written)

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
    _nature_site_geojson, _nature_site_pois, nature_site_routes = _nature_site_route_assets(output_dir)
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
            nature_site_routes,
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
