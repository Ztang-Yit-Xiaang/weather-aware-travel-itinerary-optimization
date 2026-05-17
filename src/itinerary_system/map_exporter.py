"""Lightweight and modular map exports for production itinerary artifacts."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd

from .config import TripConfig

DEFAULT_DASHBOARD_ROUTE_ID = "selected_route"
DEFAULT_DASHBOARD_METHOD = "hierarchical_bandit_gurobi_repair"
CONTRACT_VERSION = "core-route-index-v1"


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
    sort_cols = [column for column in ["trip_days", "day", "stop_order"] if column in frame.columns]
    return frame.sort_values(sort_cols).reset_index(drop=True) if sort_cols else frame.reset_index(drop=True)


def _sort_route_rows(route_df: pd.DataFrame) -> pd.DataFrame:
    if route_df is None or route_df.empty:
        return pd.DataFrame()
    sort_cols = [
        column for column in ["trip_days", "day", "stop_order", "attraction_name"] if column in route_df.columns
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
    for column in ["profile", "traveler_profile", "interest_profile", "method"]:
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
                "comparison_label": _first_nonempty(route_df, ["comparison_label"]),
            }
        )
    if extra:
        record.update({key: value for key, value in extra.items() if value is not None and value != ""})
    return record


def _point_records(route_df: pd.DataFrame) -> list[dict[str, Any]]:
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
        records.append(
            {
                "name": _route_name(row),
                "city": _safe_str(row.get("city", row.get("overnight_city", ""))),
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
                "source_list": _safe_str(row.get("source_list", row.get("source", ""))),
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
                "seasonality_risk": _safe_float(row.get("seasonality_risk", 0.0)),
                "park_type": _safe_str(row.get("park_type", "")),
                "route_type": _safe_str(row.get("route_type", "")),
                "status": _safe_str(row.get("status", "")),
                "notes": _safe_str(row.get("notes", "")),
                "drive_minutes_to_next_base": _safe_float(row.get("drive_minutes_to_next_base", 0.0)),
                "reason_selected": _safe_str(row.get("reason_selected", "")),
                "why_not_selected": _safe_str(row.get("why_not_selected", "")),
            }
        )
    return records


def _route_geojson(points: list[dict[str, Any]], route_label: str = "Selected route") -> dict[str, Any]:
    features = []
    for index, point in enumerate(points, start=1):
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [point["lon"], point["lat"]]},
                "properties": {**point, "stop_order": index},
            }
        )
    if len(points) >= 2:
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": [[point["lon"], point["lat"]] for point in points]},
                "properties": {"name": route_label, "role": "selected_route"},
            }
        )
    return {"type": "FeatureCollection", "features": features}


def _add_dashboard_record(
    records: list[dict[str, Any]],
    route_geojsons: dict[str, dict[str, Any]],
    route_pois: dict[str, list[dict[str, Any]]],
    record: dict[str, Any],
    points: list[dict[str, Any]],
) -> None:
    if not points:
        return
    route_id = str(record["id"])
    if route_id in route_geojsons:
        return
    records.append(record)
    route_geojsons[route_id] = _route_geojson(points, route_label=str(record.get("label", route_id)))
    route_pois[route_id] = points


def _add_grouped_route_records(
    *,
    frame: pd.DataFrame,
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
        _add_dashboard_record(records, route_geojsons, route_pois, record, points)


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
        _add_dashboard_record(records, route_geojsons, route_pois, record, points)


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
    points = _point_records(frame.head(max_points))
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
    _add_dashboard_record(records, route_geojsons, route_pois, record, points)


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
        route_id = str(record["id"])
        stops = route_pois.get(route_id, [])
        playable = [
            {
                "name": point.get("name", "Stop"),
                "city": point.get("city", ""),
                "lat": point.get("lat"),
                "lon": point.get("lon"),
                "day": point.get("day", 0),
                "stop_order": point.get("stop_order", index + 1),
                "category": point.get("category", ""),
                "nature_region": point.get("nature_region", ""),
                "hotel_name": point.get("hotel_name", ""),
                "interest_adjusted_value": point.get("interest_adjusted_value", 0.0),
                "final_poi_value": point.get("final_poi_value", 0.0),
                "display_utility": point.get("display_utility", point.get("final_poi_value", 0.0)),
                "optimization_value_source": point.get("optimization_value_source", "final_poi_value"),
            }
            for index, point in enumerate(stops)
            if point.get("lat") and point.get("lon")
        ]
        routes[route_id] = {
            "id": route_id,
            "label": record.get("label", route_id),
            "family": record.get("family", ""),
            "default": bool(record.get("default")),
            "stops": playable,
        }
    return {"available": True, "routes": routes}


def _city_details(
    output_dir: Path, route_records: list[dict[str, Any]], route_pois: dict[str, list[dict[str, Any]]]
) -> dict[str, Any]:
    city_map: dict[str, dict[str, Any]] = {}
    for record in route_records:
        for point in route_pois.get(str(record["id"]), []):
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
                    "nature_score": 0.0,
                    "scenic_score": 0.0,
                },
            )
            entry["route_ids"].add(record["id"])
            if point.get("day"):
                entry["days"].add(int(point.get("day", 0)))
            if point.get("hotel_name"):
                entry["hotels"].add(point["hotel_name"])
            if not record.get("optional") or len(entry["selected_stops"]) < 8:
                entry["selected_stops"].append(
                    {
                        "name": point.get("name", "Stop"),
                        "day": point.get("day", 0),
                        "category": point.get("category", ""),
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
        cities.setdefault(city, []).append(
            {
                "city": city,
                "hotel_name": _safe_str(row.get("hotel_name", "Hotel")),
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
    return {"available": True, "cities": cities}


def _selected_hotels(points: list[dict[str, Any]], hotel_choices: dict[str, Any]) -> dict[str, Any]:
    """Selected hotel markers come from optimizer route stops first, then debug-selected candidates."""
    hotels: dict[tuple[str, str], dict[str, Any]] = {}
    for point in points:
        hotel_name = _safe_str(point.get("hotel_name", ""))
        lat = _safe_float(point.get("hotel_lat", 0.0))
        lon = _safe_float(point.get("hotel_lon", 0.0))
        city = _safe_str(point.get("overnight_city") or point.get("city"))
        if not hotel_name or not lat or not lon:
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

    for city, candidates in (hotel_choices.get("cities", {}) if isinstance(hotel_choices, dict) else {}).items():
        for candidate in candidates:
            if not candidate.get("selected"):
                continue
            hotel_name = _safe_str(candidate.get("hotel_name", ""))
            lat = _safe_float(candidate.get("lat", 0.0))
            lon = _safe_float(candidate.get("lon", 0.0))
            if not hotel_name or not lat or not lon:
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
    _add_dashboard_record(records, route_geojsons, route_pois, default_record, default_points)

    matrix = _read_csv_if_present(output_dir / "production_route_matrix_route_stops.csv")
    _add_grouped_route_records(
        frame=matrix,
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

    lookup = _city_coordinate_lookup(selected, matrix, trip_lengths, methods, route_df)
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
        label="Nature candidates",
        family="nature",
        selector_group="candidates",
        records=records,
        route_geojsons=route_geojsons,
        route_pois=route_pois,
        max_points=120,
        nature_only=True,
    )

    metrics = _dashboard_metrics(
        default_points,
        config,
        route_label=route_label,
        route_source=route_source,
        route_count=len(records),
        optional_route_count=sum(1 for record in records if record.get("optional")),
        layer_families=sorted({str(record.get("family", "")) for record in records if record.get("family")}),
    )
    return records, route_geojsons, route_pois, metrics, default_points, route_label


def _dashboard_metrics(
    points: list[dict[str, Any]],
    config: TripConfig,
    route_label: str = "Selected Route",
    route_source: str = "",
    route_count: int = 1,
    optional_route_count: int = 0,
    layer_families: list[str] | None = None,
) -> dict[str, Any]:
    display_values = [float(point.get("display_utility", point.get("final_poi_value", 0.0)) or 0.0) for point in points]
    final_values = [float(point.get("final_poi_value", 0.0) or 0.0) for point in points]
    interest_values = [float(point.get("interest_adjusted_value", 0.0) or 0.0) for point in points]
    interest_enabled = bool(config.get("interest", "enabled", False))
    value_column = "interest_adjusted_value" if interest_enabled else "final_poi_value"
    return {
        "trip_days": int(config.get("trip", "trip_days", 7)),
        "scenario": str(config.get("trip", "scenario", "california_coast")),
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
                path.unlink(missing_ok=True)
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
        "data_loader.js",
        "dashboard.js",
        "map_controls.js",
        "style.css",
    ]:
        (assets / name).unlink(missing_ok=True)


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
        """html,
body {
  height: 100%;
  margin: 0;
  font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

#map {
  position: fixed;
  inset: 0;
  width: 100%;
  height: 100vh;
  z-index: 0;
  background: #e5edf2;
}

.dashboard-panel {
  position: absolute;
  z-index: 1000;
  top: 18px;
  left: 18px;
  width: 360px;
  max-height: calc(100vh - 36px);
  overflow: auto;
  background: #fff;
  border: 1px solid #dbe3ea;
  border-radius: 8px;
  box-shadow: 0 12px 30px rgba(15, 23, 42, .18);
  padding: 12px;
}

.dashboard-panel h1 {
  font-size: 17px;
  margin: 0;
}

.dashboard-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  margin: 0 0 8px;
  cursor: move;
  user-select: none;
}

.dashboard-header button {
  cursor: pointer;
}

.dashboard-panel.collapsed {
  max-height: none;
  overflow: visible;
}

.dashboard-panel.collapsed #dashboard-content {
  display: none;
}

.dashboard-section {
  border-top: 1px solid #e2e8f0;
  padding-top: 8px;
  margin-top: 8px;
}

.dashboard-section summary {
  cursor: pointer;
  font-size: 13px;
  font-weight: 700;
  color: #334155;
}

.dashboard-panel button {
  border: 1px solid #cbd5e1;
  background: #f8fafc;
  border-radius: 6px;
  padding: 6px 8px;
  cursor: pointer;
}

.dashboard-panel button:disabled {
  color: #94a3b8;
  cursor: default;
}

.metric {
  font-size: 12px;
  color: #475569;
  margin: 4px 0;
}

.status {
  min-height: 18px;
  margin-top: 8px;
  color: #0f766e;
  font-size: 12px;
  line-height: 1.35;
}

.route-selector {
  margin-top: 8px;
  color: #334155;
  font-size: 12px;
}

.route-group-title {
  margin: 8px 0 4px;
  color: #0f172a;
  font-size: 11px;
  font-weight: 700;
  text-transform: uppercase;
}

.route-row {
  display: grid;
  grid-template-columns: 18px minmax(0, 1fr);
  gap: 6px;
  align-items: start;
  padding: 5px 0;
  border-bottom: 1px solid #f1f5f9;
}

.route-row label {
  cursor: pointer;
}

.route-row small {
  display: block;
  color: #64748b;
  line-height: 1.3;
}

.layer-chip {
  display: inline-block;
  margin-left: 4px;
  border-radius: 999px;
  padding: 1px 6px;
  color: #fff;
  background: #64748b;
  font-size: 10px;
  vertical-align: middle;
}

.quick-actions {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin-top: 8px;
}

.interest-axis {
  display: grid;
  grid-template-columns: 62px minmax(0, 1fr) 42px;
  gap: 6px;
  align-items: center;
  margin: 7px 0;
  font-size: 12px;
}

.interest-axis input[type="range"] {
  width: 100%;
  accent-color: #0f766e;
}

.interest-note {
  color: #64748b;
  font-size: 11px;
  line-height: 1.35;
  margin-top: 6px;
}

.interest-ranking {
  margin-top: 8px;
}

.summary-list {
  font-size: 12px;
  color: #475569;
  line-height: 1.45;
}

.summary-list div {
  margin: 4px 0;
}

.playback-controls,
.filter-row {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin: 8px 0;
}

.playback-progress {
  height: 6px;
  overflow: hidden;
  border-radius: 999px;
  background: #e2e8f0;
}

.playback-progress span {
  display: block;
  width: 0%;
  height: 100%;
  border-radius: inherit;
  background: #0f766e;
  transition: width .18s ease;
}

.detail-card,
.hotel-card,
.nature-card,
.city-card {
  margin: 8px 0;
  padding: 8px;
  border: 1px solid #e2e8f0;
  border-radius: 7px;
  background: #f8fafc;
  font-size: 12px;
  color: #334155;
}

.detail-card strong,
.hotel-card strong,
.nature-card strong,
.city-card strong {
  color: #0f172a;
}

.hotel-card.selected,
.hotel-card.preferred {
  border-color: #0f766e;
  background: #ecfdf5;
}

.nature-card button,
.hotel-card button {
  margin-top: 6px;
}

.diagnostic-panel {
  display: none;
  position: absolute;
  z-index: 1100;
  right: 18px;
  top: 18px;
  max-width: 420px;
  border-radius: 8px;
  border: 1px solid #f59e0b;
  background: #fffbeb;
  color: #78350f;
  box-shadow: 0 14px 32px rgba(15, 23, 42, .20);
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

const FAMILY_STYLES = {
  selected: { color: '#0f766e', fill: '#2A9D8F', weight: 5, dashArray: null },
  route_matrix: { color: '#2563eb', fill: '#60a5fa', weight: 3, dashArray: '6 6' },
  trip_length: { color: '#9333ea', fill: '#c084fc', weight: 3, dashArray: '4 6' },
  method: { color: '#f97316', fill: '#fdba74', weight: 3, dashArray: '2 6' },
  context: { color: '#64748b', fill: '#94a3b8', weight: 2, dashArray: '8 8' },
  hotel: { color: '#0f172a', fill: '#f8fafc', weight: 2, dashArray: null },
  must_go: { color: '#be123c', fill: '#fb7185', weight: 2, dashArray: null },
  nature: { color: '#166534', fill: '#22c55e', weight: 2, dashArray: null }
};

function setDashboardStatus(message, isError = false) {
  const target = document.getElementById('dashboard-status');
  if (!target) return;
  target.textContent = message;
  target.style.color = isError ? '#b91c1c' : '#0f766e';
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

function playbackStops(routeId = activePlaybackRouteId) {
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
    label.textContent = stop ? `${playbackStopIndex + 1}/${stops.length} · ${stop.name || 'Stop'} · ${stop.city || ''}` : 'No playable route loaded.';
  }
}

function ensurePlaybackMarker(stop) {
  if (!stop || !Number.isFinite(Number(stop.lat)) || !Number.isFinite(Number(stop.lon))) return null;
  const latLng = [Number(stop.lat), Number(stop.lon)];
  if (!playbackMarker) {
    playbackMarker = L.circleMarker(latLng, {
      radius: 10,
      color: '#f59e0b',
      fillColor: '#fbbf24',
      fillOpacity: 0.95,
      weight: 3
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
    dashArray: style.dashArray
  };
}

function markerPopup(point, index, routeRecord) {
  const label = routeRecord?.family === 'hotel' || routeRecord?.family === 'nature'
    ? (point.name || point.hotel_name || routeRecord.label)
    : `${index + 1}. ${point.name || 'Stop'}`;
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
  return `<b>${label}</b><br>${day}${point.city || ''}<br>${point.nature_region || point.category || ''}${hotel}${utility}${finalLine}${interestLine}`;
}

function pointToMarker(point, index, routeRecord) {
  const style = FAMILY_STYLES[routeRecord?.family] || FAMILY_STYLES.selected;
  const marker = L.circleMarker([Number(point.lat), Number(point.lon)], {
    radius: routeRecord?.optional ? 5 : 7,
    color: style.color,
    fillColor: style.fill,
    fillOpacity: routeRecord?.optional ? 0.72 : 0.9,
    weight: 2
  });
  marker.bindPopup(routeRecord?.family === 'hotel' ? hotelMarkerPopup(point, Boolean(point.selected)) : markerPopup(point, index, routeRecord));
  marker.on('click', () => {
    activePlaybackRouteId = routeRecord?.id || activePlaybackRouteId;
    playbackStopIndex = index;
    setPlaybackStop(index, false);
  });
  return marker;
}

function drawRouteGeometry(group, route, routeRecord) {
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
      color: '#6d28d9',
      fillColor: '#ddd6fe',
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

function fitVisibleRoutes() {
  try {
    const groups = [...activeRouteIds].map(id => loadedRouteLayers.get(id)?.group).filter(Boolean);
    if (!groups.length) {
      setDashboardStatus('No visible routes to zoom.');
      return;
    }
    const group = L.featureGroup(groups);
    const bounds = group.getBounds();
    if (bounds.isValid()) {
      dashboardMap.fitBounds(bounds, { padding: [40, 40] });
      return;
    }
  } catch (error) {
    dashboardLogError('fitBounds failed', error);
  }
}

function routeById(routeId) {
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
      setActivePlaybackRoute(routeId);
      setDashboardStatus(`Loaded ${routeRecord.label}`);
    } else {
      if (dashboardMap.hasLayer(payload.group)) dashboardMap.removeLayer(payload.group);
      activeRouteIds.delete(routeId);
      setDashboardStatus(`Hidden ${routeRecord.label}`);
    }
    if (routeId === 'hotel_candidates') hotelCandidatesVisible = activeRouteIds.has(routeId);
    updateSelectorState();
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
  setDashboardStatus('Cleared visible routes.');
}

function drawLivePreviewRoute(points, weights = {}) {
  clearLivePreviewRoute(false);
  const drawable = (points || []).filter(point => Number.isFinite(Number(point.lat)) && Number.isFinite(Number(point.lon)));
  if (drawable.length === 0) {
    setDashboardStatus('No preview candidates with coordinates were available.', true);
    return;
  }
  livePreviewLayer = L.layerGroup();
  const latLngs = drawable.map(point => [Number(point.lat), Number(point.lon)]);
  if (latLngs.length > 1) {
    L.polyline(latLngs, {
      color: '#db2777',
      weight: 4,
      opacity: 0.85,
      dashArray: '10 8'
    }).addTo(livePreviewLayer);
  }
  drawable.forEach((point, index) => {
    const marker = L.circleMarker([Number(point.lat), Number(point.lon)], {
      radius: 6,
      color: '#be185d',
      fillColor: '#f9a8d4',
      fillOpacity: 0.88,
      weight: 2
    });
    const score = Number(point.preview_score);
    marker.bindPopup(
      `<b>Preview ${index + 1}. ${point.name || 'Candidate'}</b><br>${point.city || point.nature_region || ''}<br>Preview score: ${Number.isFinite(score) ? score.toFixed(2) : 'n/a'}<br>Browser preview is approximate; Run All recomputes the real optimized route.`
    );
    marker.addTo(livePreviewLayer);
  });
  livePreviewLayer.addTo(dashboardMap);
  try {
    const bounds = L.featureGroup(livePreviewLayer.getLayers()).getBounds();
    if (bounds.isValid()) dashboardMap.fitBounds(bounds, { padding: [48, 48] });
  } catch (error) {
    dashboardLogError('preview route fit failed', error);
  }
  updateLayerToggleState();
  const naturePct = Math.round((Number(weights.nature) || 0) * 100);
  setDashboardStatus(`Live preview route built from browser-side interest bars (${naturePct}% nature).`);
}

function clearLivePreviewRoute(updateStatus = true) {
  if (livePreviewLayer && dashboardMap?.hasLayer(livePreviewLayer)) {
    dashboardMap.removeLayer(livePreviewLayer);
  }
  livePreviewLayer = null;
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
      } else if (action === 'live_preview' && !checked) {
        clearLivePreviewRoute();
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
    ['default', 'Default route'],
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
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      maxZoom: 18,
      attribution: '&copy; OpenStreetMap contributors'
    }).addTo(dashboardMap);

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
    renderQuickButtons(index);
    renderRouteSelector(index);
    bindLayerToggles();

    const selected = defaultRoute(index);
    if (!selected) {
      showDashboardDiagnostic('error', 'default route exists', 'assets/route_index.json', 'No default route assets found. Regenerate dashboard assets.');
      throw new Error('No default route found');
    }
    await drawDefaultRoute(selected);
    loadOptionalLayers();
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
window.runQuickAction = runQuickAction;
window.showRouteGroup = showRouteGroup;
window.zoomVisibleRoutes = fitVisibleRoutes;
window.loadOptionalLayers = loadOptionalLayers;
window.drawLivePreviewRoute = drawLivePreviewRoute;
window.clearLivePreviewRoute = clearLivePreviewRoute;
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
  target.innerHTML = `
    <div class="metric"><b>Scenario:</b> ${metrics.scenario}</div>
    <div class="metric"><b>Interest:</b> ${metrics.interest_profile}</div>
    <div class="metric"><b>Stops:</b> ${metrics.selected_stop_count}</div>
    <div class="metric"><b>Optimizer value:</b> ${metrics.display_utility ?? metrics.interest_adjusted_utility}</div>
    <div class="metric"><b>Value column:</b> ${metrics.optimization_value_column || 'final_poi_value'}</div>
    <div class="metric"><b>Nature optimization:</b> ${metrics.nature_optimization_enabled ? 'active' : 'inactive'}</div>
    <div class="metric"><b>Default route:</b> ${metrics.default_route_label}</div>
    <div class="metric"><b>Indexed layers:</b> ${metrics.route_record_count || 1}</div>
    <div class="interest-note">Default route is computed by Python optimization using nature-aware value weights.</div>
    <div class="interest-note">Browser hotel choices and interest bars are visual previews until you rerun the notebook.</div>
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

function bindDashboardShellControls() {
  const panel = document.querySelector('.dashboard-panel');
  const handle = document.getElementById('dashboard-drag-handle');
  const collapse = document.getElementById('dashboard-collapse');
  if (!panel || !handle) return;
  collapse?.addEventListener('click', event => {
    event.stopPropagation();
    panel.classList.toggle('collapsed');
    collapse.textContent = panel.classList.contains('collapsed') ? 'Expand' : 'Collapse';
  });
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
    ['Playback', playbackMeta.total ? `${playbackMeta.index + 1} of ${playbackMeta.total}` : ''],
    ['Day', stop.day || ''],
    ['City', stop.city || ''],
    ['Overnight hotel', stop.hotel_name || ''],
    ['Category', stop.category || ''],
    ['Nature region', stop.nature_region || ''],
    ['Optimizer value', compactValue(stop.display_utility ?? stop.interest_adjusted_value ?? stop.final_poi_value)],
    ['Final POI value', compactValue(stop.final_poi_value)],
    ['Interest-adjusted value', Number(stop.interest_adjusted_value) > 0 ? compactValue(stop.interest_adjusted_value) : ''],
    ['Value source', stop.optimization_value_source || ''],
    ['Nature score', compactValue(stop.nature_score)],
    ['Scenic score', compactValue(stop.scenic_score)],
    ['Weather sensitivity', compactValue(stop.weather_sensitivity)],
    ['Source', stop.source_list || '']
  ].filter(([, value]) => value !== '' && value !== undefined && value !== null);
  target.innerHTML = `
    <div class="detail-card">
      <strong>${stop.name || 'Stop'}</strong>
      ${rows.map(([key, value]) => `<div><b>${key}:</b> ${value}</div>`).join('')}
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
    return `
      <div class="city-card">
        <strong>${city.city}</strong>
        <div>Days: ${(city.days || []).join(', ') || 'n/a'}</div>
        <div>Route roles: ${(city.route_ids || []).slice(0, 3).join(', ') || 'n/a'}</div>
        <div>Hotels: ${(city.hotels || []).slice(0, 2).join(', ') || 'n/a'}</div>
        <div>Travel legs: ${cityLegs.join('; ') || 'n/a'}</div>
        <div>Top nature/scenic: ${compactValue(city.nature_score)} / ${compactValue(city.scenic_score)}</div>
        <div>Stops: ${(city.selected_stops || []).slice(0, 3).map(stop => stop.name).join(', ')}</div>
      </div>
    `;
  }).join('');
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
    </div>
    <div class="interest-note">Preferred hotel choices are browser-local only and do not rewrite the optimized route.</div>
  ` + cityNames.map(city => {
    const preferred = preferredHotel(city);
    const cards = (cities[city] || []).slice(0, 6).map(hotel => {
      const isPreferred = preferred && preferred === hotel.hotel_name;
      const classes = ['hotel-card', hotel.selected ? 'selected' : '', isPreferred ? 'preferred' : ''].join(' ');
      return `
        <div class="${classes}">
          <strong>${hotel.hotel_name}</strong>
          <div>Rank ${hotel.candidate_rank || 'n/a'} · Score ${compactValue(hotel.hotel_score)}</div>
          <div>${hotel.selected ? 'Selected by optimizer' : 'Candidate alternative'} · ${hotel.source || ''}</div>
          <div>${hotel.selected_hotel_reason || ''}</div>
          <button type="button" data-prefer-hotel="${hotel.hotel_name}" data-hotel-city="${city}">${isPreferred ? 'Preferred' : 'Mark preferred'}</button>
        </div>
      `;
    }).join('');
    return `<div class="city-card"><strong>${city}</strong>${cards}</div>`;
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

function previewItemAxis(item, axis) {
  if (axis === 'city') return Number(item.city_axis ?? item.city ?? item.city_score ?? 0);
  return Number(item[axis] ?? item[`${axis}_score`] ?? 0);
}

function previewScore(item, weights, lambdas) {
  const fit = ['nature', 'city', 'culture', 'history']
    .reduce((sum, axis) => sum + weights[axis] * previewItemAxis(item, axis), 0);
  const base = Number(item.final_poi_value ?? item.interest_adjusted_value ?? 0);
  const park = Number(item.park_bonus || 0);
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
    <div class="interest-note">Drag the bars to preview route-fit changes. Balanced is equal weights. Full optimized routes use these weights only after rerunning the notebook/config.</div>
    <div class="interest-note">National park candidates are in the Python catalog and enter optimization through nature/park scores when interest mode is enabled.</div>
    <div class="playback-controls">
      <button type="button" data-build-live-preview="true">Build live preview route</button>
      <button type="button" data-clear-live-preview="true">Clear preview route</button>
    </div>
    <div class="interest-note">Browser preview is approximate; Run All recomputes the real optimized route.</div>
    <div id="interest-route-match" class="summary-list"></div>
    <div id="interest-ranking" class="interest-ranking"></div>
  `;
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
    const routeError = axes.reduce((sum, axis) => {
      const mixValue = Number(routeMix[axis] ?? routeMix[`${axis}_score`] ?? 0);
      return sum + Math.abs((normalized[axis] || 0) - mixValue);
    }, 0);
    const routeScore = Math.max(0, 1 - routeError);
    const matchNode = target.querySelector('#interest-route-match');
    if (matchNode) {
      matchNode.innerHTML = `<div><b>Selected-route match preview:</b> ${compactValue(routeScore, 2)}</div>`;
    }
    const candidates = (preview.top_boosted_pois || [])
      .map(item => ({ ...item, preview_score: previewScore(item, normalized, preview.lambdas || {}) }))
      .sort((a, b) => b.preview_score - a.preview_score)
      .slice(0, 8);
    target.querySelector('#interest-ranking').innerHTML = candidates.map(item => `
      <div class="nature-card">
        <strong>${item.name}</strong>
        <div>${item.city || ''} · ${item.nature_region || item.park_type || ''}</div>
        <div>Preview score: ${compactValue(item.preview_score)} · Fit: ${compactValue(
          ['nature', 'city', 'culture', 'history'].reduce((sum, axis) => sum + normalized[axis] * previewItemAxis(item, axis), 0)
        )}</div>
      </div>
    `).join('');
  };
  const currentWeights = () => normalizeInterestBarValues(Object.fromEntries(axes.map(axis => [
    axis,
    Number(target.querySelector(`[data-interest-axis="${axis}"]`)?.value || 0)
  ])));
  const rankedPreviewCandidates = weights => {
    const previewItems = Array.isArray(preview.top_boosted_pois) ? preview.top_boosted_pois : [];
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
      .map(item => ({ ...item, preview_score: previewScore(item, weights, preview.lambdas || {}) }))
      .sort((a, b) => b.preview_score - a.preview_score);
  };
  const buildLivePreviewRoute = () => {
    const weights = currentWeights();
    const candidates = rankedPreviewCandidates(weights).slice(0, Math.max(4, Math.min(10, Number(window.dashboardMetrics?.selected_stop_count || 8))));
    if (!candidates.length) {
      const status = document.getElementById('dashboard-status');
      if (status) {
        status.textContent = 'No exported POI/nature candidates with coordinates are available for the live preview route.';
        status.style.color = '#b91c1c';
      }
      return;
    }
    if (window.drawLivePreviewRoute) window.drawLivePreviewRoute(candidates, weights);
  };
  target.querySelector('[data-build-live-preview]')?.addEventListener('click', buildLivePreviewRoute);
  target.querySelector('[data-clear-live-preview]')?.addEventListener('click', () => {
    if (window.clearLivePreviewRoute) window.clearLivePreviewRoute();
  });
  target.querySelectorAll('[data-interest-axis]').forEach(input => input.addEventListener('input', renderRanking));
  renderRanking();
}

window.renderDashboardMetrics = renderDashboardMetrics;
window.bindDashboardShellControls = bindDashboardShellControls;
window.renderActiveStopDetail = renderActiveStopDetail;
window.renderCityDetails = renderCityDetails;
window.renderHotelChoices = renderHotelChoices;
window.renderNatureExplore = renderNatureExplore;
window.renderDebugSummary = renderDebugSummary;
window.renderInterestPreview = renderInterestPreview;
""",
        written,
    )

    index_path = root / "index.html"
    _write_text_asset(
        index_path,
        """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Full Interactive Dashboard</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet/dist/leaflet.css" />
  <link rel="stylesheet" href="assets/style.css" />
</head>
<body>
  <div id="map"></div>
  <div id="diagnostic-panel" class="diagnostic-panel" role="status" aria-live="polite"></div>
  <section class="dashboard-panel">
    <div id="dashboard-drag-handle" class="dashboard-header">
      <h1>Full Interactive Dashboard</h1>
      <button id="dashboard-collapse" type="button">Collapse</button>
    </div>
    <div id="dashboard-content">
      <details class="dashboard-section" open>
        <summary>Dashboard summary</summary>
        <div id="metrics"></div>
      </details>
      <details class="dashboard-section" open>
        <summary>Route selector</summary>
        <div class="filter-row">
          <label><input type="checkbox" data-layer-toggle="default_route" checked /> Default optimized route</label>
          <label><input type="checkbox" data-layer-toggle="selected_hotels" checked /> Selected hotels</label>
          <label><input type="checkbox" data-layer-toggle="hotel_candidates" /> Hotel candidates</label>
          <label><input type="checkbox" data-layer-toggle="nature_candidates" /> Nature candidates</label>
          <label><input type="checkbox" data-layer-toggle="live_preview" disabled /> Live preview route</label>
        </div>
        <div id="quick-actions" class="quick-actions"></div>
        <div id="route-selector" class="route-selector"></div>
      </details>
      <details class="dashboard-section" open>
        <summary>Playback</summary>
        <div class="summary-list"><b>Route:</b> <span id="playback-route-label">Loading...</span></div>
        <div class="playback-controls">
          <button id="playback-play" type="button">Play</button>
          <button id="playback-pause" type="button">Pause</button>
          <button id="playback-restart" type="button">Restart</button>
          <button id="playback-prev" type="button">Prev</button>
          <button id="playback-next" type="button">Next</button>
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
      <details class="dashboard-section">
        <summary>City details</summary>
        <div id="city-details"></div>
      </details>
      <details class="dashboard-section">
        <summary>Hotel choices</summary>
        <div id="hotel-choices"></div>
      </details>
      <details class="dashboard-section">
        <summary>Nature explore</summary>
        <div id="nature-explore"></div>
      </details>
      <details class="dashboard-section" open>
        <summary>Interest bars</summary>
        <div id="interest-preview"></div>
      </details>
      <details class="dashboard-section">
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
""",
        written,
    )
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
        )
        artifacts["full_interactive_dashboard"] = dashboard_root / "index.html"
        route_notes = {}
        for record in route_records:
            route_notes[str(record["geojson"])] = record
            route_notes[str(record["geojson_js"])] = record
            route_notes[str(record["pois"])] = record
            route_notes[str(record["pois_js"])] = record
        for path in written:
            rel_path = str(path.relative_to(dashboard_root)) if dashboard_root in path.parents else path.name
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
                notes = "modular_dashboard_entrypoint"
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
