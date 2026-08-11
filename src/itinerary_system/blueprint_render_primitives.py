"""Shared geometry, routing, Folium primitives, and debug rows for map rendering."""

from __future__ import annotations

import html
import math

import folium
import numpy as np
import pandas as pd
import requests

from .blueprint_core import (
    CALIFORNIA_ROAD_CORRIDOR,
    _coord_for_place,
    _corridor_segment_points,
    _numeric_catalog_series,
    _point_distance_km,
    _route_distance_km,
)

SCENIC_CA1_COLOR = "#2A9D8F"

FASTEST_ROUTE_COLOR = "#6C757D"

ROUTE_CONTEXT_PANE = "routeContextPane"

ROUTE_CORE_PANE = "routeCorePane"

ROUTE_TOP_PANE = "routeTopPane"

FULL_SCENE_OFFSET_LAT_STEP = 0.009

FULL_SCENE_OFFSET_LON_STEP = 0.016

DAY_COLORS = ["#2563EB", "#4C78A8", "#F4A261", "#7A5195", "#C1121F", "#5FAD56", "#6C757D"]

MATRIX_METHOD_STYLES = {
    "hierarchical_gurobi_pipeline": ("Gurobi", "#0096C7", None, -0.35),
    "hierarchical_greedy_baseline": ("Greedy", "#111827", "7 11", 0.0),
    "hierarchical_bandit_gurobi_repair": ("Bandit + Small Gurobi", "#00A63E", None, 0.35),
}

MATRIX_PROFILE_STYLE_OFFSETS = {
    "relaxed": -0.25,
    "balanced": 0.0,
    "explorer": 0.25,
}

MATRIX_DEFAULT_ROUTE_KEY = "d7__bandit_repair__balanced"

MATRIX_ROUTE_OFFSETS_ENABLED = False



def _escape(value):
    return html.escape(str(value))



def _route_cache_key(points):
    return "|".join(f"{lat:.5f},{lon:.5f}" for lat, lon in points)



def _nearest_corridor_anchor(point, max_distance_km=55.0):
    if point is None or len(point) < 2:
        return None
    best_name = None
    best_distance = float("inf")
    for name, lat, lon in CALIFORNIA_ROAD_CORRIDOR:
        distance = _point_distance_km(float(point[0]), float(point[1]), float(lat), float(lon))
        if distance < best_distance:
            best_name = name
            best_distance = distance
    return best_name if best_name is not None and best_distance <= float(max_distance_km) else None



def _curated_corridor_fallback(left, right):
    """Prefer a reproducible California corridor shape over long straight lines.

    Cached OSRM remains the first choice. This helper only handles long missing
    legs whose endpoints are near known corridor anchors; short local city hops
    still fall through to explicit straight-line fallback warnings.
    """
    leg_distance = _point_distance_km(float(left[0]), float(left[1]), float(right[0]), float(right[1]))
    if leg_distance < 65.0:
        return [], ""
    left_anchor = _nearest_corridor_anchor(left)
    right_anchor = _nearest_corridor_anchor(right)
    if not left_anchor or not right_anchor or left_anchor == right_anchor:
        return [], ""
    corridor_points = _corridor_segment_points(left_anchor, right_anchor)
    if len(corridor_points) < 2:
        return [], ""
    path = [left]
    for point in corridor_points:
        if _point_distance_km(path[-1][0], path[-1][1], point[0], point[1]) > 1.0:
            path.append(point)
    if _point_distance_km(path[-1][0], path[-1][1], right[0], right[1]) > 1.0:
        path.append(right)
    return path, f"curated-corridor:{left_anchor}->{right_anchor}"



def _fetch_osrm_route(points, cache, run_live=False):
    if len(points) < 2:
        return points, "single-point"

    key = _route_cache_key(points)
    if key in cache:
        cached_mode = str(cache[key].get("mode", ""))
        cached_path = cache[key].get("path", points)
        if cached_path and (not run_live or "fallback" not in cached_mode):
            return cached_path, "cached-osrm"

    def stitch_cached_or_fallback(route_points):
        stitched_path = []
        used_cached_leg = False
        used_corridor_leg = False
        used_fallback_leg = False
        for left, right in zip(route_points[:-1], route_points[1:], strict=False):
            leg_key = _route_cache_key([left, right])
            leg_mode = cache.get(leg_key, {}).get("mode", "")
            if leg_key in cache and "fallback" not in str(leg_mode):
                leg_path = cache[leg_key]["path"]
                used_cached_leg = True
            else:
                leg_path, leg_mode = _curated_corridor_fallback(left, right)
                if leg_path:
                    used_corridor_leg = True
                else:
                    leg_path = [left, right]
                    used_fallback_leg = True
            if stitched_path and leg_path:
                stitched_path.extend(leg_path[1:])
            else:
                stitched_path.extend(leg_path)
        if not stitched_path:
            return route_points, "straight-line-fallback"
        if used_fallback_leg and (used_cached_leg or used_corridor_leg):
            return stitched_path, "cached-curated-stitched-with-straight-line-fallback"
        if used_fallback_leg:
            return stitched_path, "straight-line-fallback"
        if used_cached_leg and used_corridor_leg:
            return stitched_path, "cached-curated-corridor-stitched"
        if used_cached_leg:
            return stitched_path, "cached-stitched"
        if used_corridor_leg:
            return stitched_path, "curated-corridor-stitched"
        return route_points, "straight-line-fallback"

    if not run_live:
        return stitch_cached_or_fallback(points)

    def request_route(route_points):
        coord_text = ";".join(f"{lon:.6f},{lat:.6f}" for lat, lon in route_points)
        url = f"https://router.project-osrm.org/route/v1/driving/{coord_text}"
        params = {
            "overview": "full",
            "geometries": "geojson",
            "steps": "false",
            # POIs/hotels are often inside parcels or parks rather than exactly
            # on drivable centerlines. A generous snap radius lets OSRM route to
            # the nearest legal road instead of forcing a straight-line fallback.
            "radiuses": ";".join(["2500"] * len(route_points)),
            "continue_straight": "false",
        }
        response = requests.get(url, params=params, timeout=45)
        response.raise_for_status()
        payload = response.json()
        if payload.get("code") != "Ok" or not payload.get("routes"):
            raise RuntimeError(f"OSRM route failed: {payload.get('code', 'unknown')}")
        coords = payload["routes"][0]["geometry"]["coordinates"]
        return [[lat, lon] for lon, lat in coords]

    try:
        path = request_route(points)
        cache[key] = {"path": path, "mode": "osrm-driving"}
        return path, "osrm-driving"
    except Exception:
        if len(points) <= 2:
            corridor_path, corridor_mode = _curated_corridor_fallback(points[0], points[-1])
            if corridor_path:
                return corridor_path, corridor_mode
            return points, "straight-line-fallback"

    stitched_path = []
    used_live_leg = False
    used_cached_leg = False
    used_fallback = False
    for left, right in zip(points[:-1], points[1:], strict=False):
        leg_key = _route_cache_key([left, right])
        leg_mode = cache.get(leg_key, {}).get("mode", "")
        if leg_key in cache and "fallback" not in str(leg_mode):
            leg_path = cache[leg_key]["path"]
            used_cached_leg = True
        else:
            try:
                leg_path = request_route([left, right])
                cache[leg_key] = {"path": leg_path, "mode": "osrm-driving-leg"}
                used_live_leg = True
            except Exception:
                leg_path, leg_mode = _curated_corridor_fallback(left, right)
                if leg_path:
                    cache[leg_key] = {"path": leg_path, "mode": leg_mode}
                    used_cached_leg = True
                else:
                    leg_path = [left, right]
                    used_fallback = True
        if stitched_path and leg_path:
            stitched_path.extend(leg_path[1:])
        else:
            stitched_path.extend(leg_path)

    if used_fallback:
        mode = "cached-stitched-with-fallback"
    elif used_live_leg:
        mode = "osrm-driving-stitched"
    elif used_cached_leg:
        mode = "cached-stitched"
    else:
        mode = "straight-line-fallback"
    cache[key] = {"path": stitched_path, "mode": mode}
    return stitched_path, mode



def _dedupe_route_points(points):
    output = []
    for point in points:
        if point is None or len(point) < 2:
            continue
        clean = [float(point[0]), float(point[1])]
        if not output or _point_distance_km(output[-1][0], output[-1][1], clean[0], clean[1]) > 0.005:
            output.append(clean)
    return output



def _offset_route_points_for_visibility(points, offset_index):
    """Shift overview routes into visible bands so overlapping plans can be audited."""
    clean_points = _dedupe_route_points(points)
    if not clean_points or not offset_index:
        return clean_points
    lat_shift = float(offset_index) * FULL_SCENE_OFFSET_LAT_STEP
    lon_shift = float(offset_index) * FULL_SCENE_OFFSET_LON_STEP
    return [[lat + lat_shift, lon + lon_shift] for lat, lon in clean_points]



def _fetch_stitched_route(points, cache, run_live=False):
    """Fetch/draw each leg separately so every visible stop is connected.

    This avoids stale multi-waypoint route-cache entries and partial browser
    routing responses that can leave later markers without flow links.
    """
    points = _dedupe_route_points(points)
    if len(points) < 2:
        return points, "single-point"
    stitched_path = []
    modes = []
    for left, right in zip(points[:-1], points[1:], strict=False):
        leg_path, leg_mode = _fetch_osrm_route([left, right], cache, run_live=run_live)
        modes.append(leg_mode)
        if stitched_path and leg_path:
            stitched_path.extend(leg_path[1:])
        else:
            stitched_path.extend(leg_path)
    has_fallback = any("fallback" in mode for mode in modes)
    has_osrm = any("osrm" in mode or "cached" in mode for mode in modes)
    has_curated = any("curated" in mode or "corridor" in mode for mode in modes)
    if has_fallback and has_curated:
        mode = "stitched-leg-curated-corridor-with-local-straight-line-fallback"
    elif has_fallback:
        mode = "stitched-leg-mixed-fallback"
    elif has_osrm:
        mode = "stitched-leg-osrm-driving"
    elif has_curated:
        mode = "stitched-leg-curated-corridor"
    else:
        mode = "stitched-leg-cache"
    return stitched_path, mode



def _route_mode_label(modes):
    if not modes:
        return "single-point"
    if any("fallback" in str(mode) for mode in modes):
        return "stitched-leg-mixed-fallback"
    if any("osrm" in str(mode) for mode in modes):
        return "stitched-leg-osrm-driving"
    return "stitched-leg-cache"



def _add_route_panes(map_object):
    """Keep generated route lines above the basemap and under marker labels."""
    folium.map.CustomPane(ROUTE_CONTEXT_PANE, z_index=620).add_to(map_object)
    folium.map.CustomPane(ROUTE_CORE_PANE, z_index=650).add_to(map_object)
    folium.map.CustomPane(ROUTE_TOP_PANE, z_index=690).add_to(map_object)



def _route_points_for_names(names):
    return [_coord_for_place(name) for name in names]



def _finite_float(value, default=np.nan):
    try:
        value = float(value)
    except Exception:
        return default
    return value if np.isfinite(value) else default



def _add_static_result_line(
    layer,
    points,
    *,
    color,
    dash_array=None,
    tooltip=None,
    popup_html=None,
    weight=6,
    opacity=0.95,
    pane=ROUTE_CORE_PANE,
    geometry_mode_prefix="static-polyline",
):
    clean_points = _dedupe_route_points(points)
    if len(clean_points) < 2:
        return "not_enough_points"

    # White casing so the route is visible on any basemap
    folium.PolyLine(
        clean_points,
        color="#FFFFFF",
        weight=weight + 4,
        opacity=0.86,
        pane=pane,
    ).add_to(layer)

    folium.PolyLine(
        clean_points,
        color=color,
        weight=weight,
        opacity=opacity,
        dash_array=dash_array,
        tooltip=tooltip,
        popup=folium.Popup(popup_html, max_width=320) if popup_html else None,
        pane=pane,
    ).add_to(layer)
    arrow_count = _add_canvas_direction_arrows(
        layer,
        clean_points,
        color=color,
        pane=pane,
        interval_km=220.0,
        max_arrows=2,
        class_name="blueprint-route-arrow blueprint-route-arrow-static",
    )

    return f"{geometry_mode_prefix}; static-polyline; canvas-arrow-markers={arrow_count}"



def _route_bounds(points):
    clean_points = _dedupe_route_points(points)
    if not clean_points:
        return []
    latitudes = [point[0] for point in clean_points]
    longitudes = [point[1] for point in clean_points]
    return [
        [float(min(latitudes)), float(min(longitudes))],
        [float(max(latitudes)), float(max(longitudes))],
    ]



def _route_midpoint(points):
    clean_points = _dedupe_route_points(points)
    if not clean_points:
        return None
    if len(clean_points) == 1:
        return clean_points[0]
    total_distance = _route_distance_km(clean_points)
    if total_distance <= 0:
        return clean_points[len(clean_points) // 2]
    target_distance = total_distance / 2.0
    walked = 0.0
    for left, right in zip(clean_points[:-1], clean_points[1:], strict=False):
        segment_distance = _point_distance_km(left[0], left[1], right[0], right[1])
        if walked + segment_distance >= target_distance:
            ratio = 0.0 if segment_distance <= 0 else (target_distance - walked) / segment_distance
            return [
                float(left[0] + (right[0] - left[0]) * ratio),
                float(left[1] + (right[1] - left[1]) * ratio),
            ]
        walked += segment_distance
    return clean_points[-1]



def _bearing_degrees(left, right):
    lat1 = math.radians(float(left[0]))
    lat2 = math.radians(float(right[0]))
    delta_lon = math.radians(float(right[1]) - float(left[1]))
    y = math.sin(delta_lon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(delta_lon)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0



def _route_arrow_samples(points, interval_km=95.0, max_arrows=9):
    clean_points = _dedupe_route_points(points)
    if len(clean_points) < 2:
        return []
    total_distance = _route_distance_km(clean_points)
    if total_distance <= 0:
        return []
    target_count = max(1, min(int(max_arrows), int(total_distance // float(interval_km)) + 1))
    target_distances = np.linspace(
        total_distance / (target_count + 1), total_distance * target_count / (target_count + 1), target_count
    )
    samples = []
    walked = 0.0
    target_index = 0
    for left, right in zip(clean_points[:-1], clean_points[1:], strict=False):
        segment_distance = _point_distance_km(left[0], left[1], right[0], right[1])
        if segment_distance <= 0:
            continue
        while target_index < len(target_distances) and walked + segment_distance >= target_distances[target_index]:
            ratio = (target_distances[target_index] - walked) / segment_distance
            point = [
                float(left[0] + (right[0] - left[0]) * ratio),
                float(left[1] + (right[1] - left[1]) * ratio),
            ]
            samples.append((point, _bearing_degrees(left, right)))
            target_index += 1
        walked += segment_distance
    return samples



def _add_canvas_direction_arrows(
    layer, points, *, color, pane=ROUTE_TOP_PANE, interval_km=95.0, max_arrows=9, class_name="blueprint-route-arrow"
):
    samples = _route_arrow_samples(points, interval_km=interval_km, max_arrows=max_arrows)
    for index, (point, bearing) in enumerate(samples, start=1):
        folium.Marker(
            location=point,
            icon=folium.DivIcon(
                icon_size=(18, 18),
                icon_anchor=(9, 9),
                html=f"""
                <div class="{class_name}" style="--route-arrow-color:{color}; transform: rotate({bearing:.1f}deg);">
                    <span></span>
                </div>
                """,
            ),
            tooltip=f"Route direction arrow {index}",
            pane=pane,
        ).add_to(layer)
    return len(samples)



def _add_full_scene_route_band(
    layer,
    points,
    *,
    color,
    dash_array=None,
    tooltip=None,
    popup_html=None,
    label="route",
    weight=9,
    opacity=0.96,
    pane=ROUTE_TOP_PANE,
    start_label=None,
    end_label=None,
    arrow_offset=8,
):
    clean_points = _dedupe_route_points(points)
    if len(clean_points) < 2:
        return "not_enough_points"

    folium.PolyLine(
        clean_points,
        color=color,
        weight=weight,
        opacity=opacity,
        dash_array=dash_array,
        tooltip=tooltip,
        popup=folium.Popup(popup_html, max_width=320) if popup_html else None,
        pane=pane,
    ).add_to(layer)
    # Disabled for the same reason as in _add_intercity_route_layer
    # (incompatible with prefer_canvas=True; see comment there).
    # plugins.PolyLineTextPath(
    #     route_line,
    #     "   >   ",
    #     repeat=True,
    #     offset=arrow_offset,
    #     attributes={"fill": color, "font-weight": "800", "font-size": "13"},
    # ).add_to(layer)
    arrow_count = _add_canvas_direction_arrows(
        layer,
        clean_points,
        color=color,
        pane=pane,
        interval_km=260.0,
        max_arrows=3,
        class_name="blueprint-route-arrow blueprint-route-arrow-full",
    )

    endpoint_specs = [
        (clean_points[0], start_label, "S", "blueprint-route-endpoint-start"),
        (clean_points[-1], end_label, "E", "blueprint-route-endpoint-end"),
    ]
    for point, endpoint_label, endpoint_text, class_name in endpoint_specs:
        if not endpoint_label:
            continue
        folium.Marker(
            location=point,
            icon=folium.DivIcon(
                html=f"""
                <div class="blueprint-route-endpoint {class_name}" style="background:rgba(255,255,255,0.95);
                    color:#111827; border:2px solid {color}; border-radius:999px; font-size:10px;
                    font-weight:900; height:19px; line-height:17px; text-align:center; width:19px;
                    box-shadow:0 1px 6px rgba(0,0,0,0.22);">
                    {_escape(endpoint_text)}
                </div>
                """
            ),
            tooltip=endpoint_label,
        ).add_to(layer)

    midpoint = _route_midpoint(clean_points)
    if midpoint and label:
        folium.Marker(
            location=midpoint,
            icon=folium.DivIcon(
                html=f"""
                <div class="blueprint-route-badge" style="background:{color}; color:white;
                    border:2px solid rgba(255,255,255,0.92); border-radius:13px;
                    font-size:10px; font-weight:800; padding:2px 6px;
                    box-shadow:0 2px 8px rgba(0,0,0,0.24); max-width:92px;
                    overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">
                    {_escape(label)}
                </div>
                """
            ),
            tooltip=tooltip or label,
        ).add_to(layer)

    return f"full-route-polyline-with-canvas-arrows; canvas-arrow-markers={arrow_count}"



def _route_debug_value(value, default=""):
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except Exception:
        pass
    return value



def _route_points_from_day_group(group):
    if group is None or group.empty:
        return []
    first = group.iloc[0]
    hotel_lat = _finite_float(first.get("hotel_latitude"))
    hotel_lon = _finite_float(first.get("hotel_longitude"))
    route_start = [
        _finite_float(first.get("route_start_latitude", hotel_lat), hotel_lat),
        _finite_float(first.get("route_start_longitude", hotel_lon), hotel_lon),
    ]
    route_end = [
        _finite_float(first.get("route_end_latitude", hotel_lat), hotel_lat),
        _finite_float(first.get("route_end_longitude", hotel_lon), hotel_lon),
    ]
    stop_points = group[["latitude", "longitude"]].dropna().astype(float).values.tolist()
    return [
        point
        for point in [route_start, *stop_points, route_end]
        if len(point) >= 2 and np.isfinite(float(point[0])) and np.isfinite(float(point[1]))
    ]



def _hotel_point_from_day_group(group):
    if group is None or group.empty:
        return []
    first = group.iloc[0]
    hotel_lat = _finite_float(first.get("hotel_latitude"))
    hotel_lon = _finite_float(first.get("hotel_longitude"))
    if np.isfinite(hotel_lat) and np.isfinite(hotel_lon):
        return [float(hotel_lat), float(hotel_lon)]
    return []



def _hotel_loop_points_from_day_group(group):
    """Draw local city detail as hotel -> local stops -> same hotel."""
    if group is None or group.empty:
        return [], pd.DataFrame()
    first = group.iloc[0]
    overnight_city = str(first.get("overnight_city", first.get("city", "")))
    hotel_point = _hotel_point_from_day_group(group)
    if not hotel_point:
        return [], pd.DataFrame()
    local_rows = group[group["city"].astype(str).eq(overnight_city)].copy()
    if local_rows.empty and not str(first.get("route_type", "")).startswith("relocation"):
        local_rows = group.copy()
    if local_rows.empty:
        return [], local_rows
    local_rows["stop_order"] = _numeric_catalog_series(local_rows, "stop_order", 1).astype(int)
    local_rows = local_rows.sort_values(["stop_order", "attraction_name"]).reset_index(drop=True)
    stop_points = local_rows[["latitude", "longitude"]].dropna().astype(float).values.tolist()
    return [hotel_point, *stop_points, hotel_point], local_rows



def _transition_points_from_day_group(group):
    """Draw relocation separately as previous hotel/base -> pass-through stops -> destination hotel/base."""
    if group is None or group.empty:
        return [], pd.DataFrame()
    first = group.iloc[0]
    route_type = str(first.get("route_type", ""))
    if not route_type.startswith("relocation"):
        return [], pd.DataFrame()
    start_point = [
        _finite_float(first.get("route_start_latitude", first.get("hotel_latitude"))),
        _finite_float(first.get("route_start_longitude", first.get("hotel_longitude"))),
    ]
    end_point = _hotel_point_from_day_group(group)
    if not end_point:
        end_point = [
            _finite_float(first.get("route_end_latitude", first.get("hotel_latitude"))),
            _finite_float(first.get("route_end_longitude", first.get("hotel_longitude"))),
        ]
    if not all(np.isfinite(float(value)) for value in [*start_point, *end_point]):
        return [], pd.DataFrame()
    overnight_city = str(first.get("overnight_city", first.get("city", "")))
    transition_rows = group[~group["city"].astype(str).eq(overnight_city)].copy()
    if not transition_rows.empty:
        transition_rows["stop_order"] = (
            _numeric_catalog_series(transition_rows, "stop_order", 1).astype(int)
        )
        transition_rows = transition_rows.sort_values(["stop_order", "attraction_name"]).reset_index(drop=True)
    stop_points = (
        transition_rows[["latitude", "longitude"]].dropna().astype(float).values.tolist()
        if not transition_rows.empty
        else []
    )
    return [start_point, *stop_points, end_point], transition_rows



def _route_endpoint_city_labels(frame, fallback_start="San Francisco", fallback_end="Los Angeles"):
    if frame is None or frame.empty:
        return str(fallback_start), str(fallback_end)
    sorted_frame = frame.copy()
    if "day" in sorted_frame.columns:
        sorted_frame["day"] = pd.to_numeric(sorted_frame["day"], errors="coerce").fillna(1).astype(int)
    else:
        sorted_frame["day"] = 1
    if "stop_order" in sorted_frame.columns:
        sorted_frame["stop_order"] = pd.to_numeric(sorted_frame["stop_order"], errors="coerce").fillna(1).astype(int)
    else:
        sorted_frame["stop_order"] = range(1, len(sorted_frame) + 1)
    sorted_frame = sorted_frame.sort_values(["day", "stop_order"]).reset_index(drop=True)
    first = sorted_frame.iloc[0]
    last = sorted_frame.iloc[-1]
    start_city = str(
        first.get("route_start_city", first.get("gateway_start", first.get("city", fallback_start))) or fallback_start
    )
    end_city = str(
        last.get("route_end_city", last.get("gateway_end", last.get("overnight_city", last.get("city", fallback_end))))
        or fallback_end
    )
    return start_city, end_city



def _append_route_debug_row(
    route_debug_rows,
    *,
    layer_group,
    layer_name,
    comparison_type,
    points,
    geometry_mode,
    show_by_default=False,
    source_rows=0,
    profile="",
    method="",
    trip_days="",
    day="",
    route_type="",
    route_start_city="",
    route_end_city="",
    pass_through_cities="",
    drive_minutes=np.nan,
    available_visit_minutes=np.nan,
    layer_var="",
    notes="",
):
    if route_debug_rows is None:
        return
    clean_points = _dedupe_route_points(points)
    distance_km = _route_distance_km(clean_points)
    geometry_source = str(geometry_mode).split(";")[0].strip()
    geometry_lower = str(geometry_mode).lower()
    fallback_count = int("fallback" in geometry_lower)
    if "straight-line-fallback" in geometry_lower and "cached" not in geometry_lower and "osrm" not in geometry_lower:
        road_geometry_percent = 0.0
    elif "fallback" in geometry_lower:
        road_geometry_percent = 50.0
    elif any(token in geometry_lower for token in ["osrm", "cached", "corridor"]):
        road_geometry_percent = 100.0
    else:
        road_geometry_percent = np.nan
    issue = ""
    status = "OK"
    if len(clean_points) < 2:
        status = "WARN"
        issue = "single-point marker/loop diagnostic; no route line drawn"
    elif source_rows and distance_km < 0.05:
        status = "FAILED"
        issue = "route collapsed to a single point"
    elif str(route_type).startswith("relocation") and distance_km < 50.0:
        status = "WARN"
        issue = "relocation day has unexpectedly short geometry"
    elif fallback_count and str(comparison_type) in {
        "intercity",
        "traveler_day",
        "city_detail",
        "trip_length",
        "method",
        "traveler_profile",
    }:
        status = "WARN"
        issue = "route contains straight-line fallback geometry"
    elif (
        str(comparison_type) in {"selected_result", "traveler_day", "trip_length", "method"}
        and str(route_start_city) != str(route_end_city)
        and _finite_float(drive_minutes, 0.0) <= 0
    ):
        status = "WARN"
        issue = "route changes base city but has no positive drive-time metadata"

    route_debug_rows.append(
        {
            "layer_group": layer_group,
            "layer_name": layer_name,
            "comparison_type": comparison_type,
            "profile": profile,
            "method": method,
            "trip_days": trip_days,
            "day": day,
            "route_type": route_type,
            "route_start_city": route_start_city,
            "route_end_city": route_end_city,
            "pass_through_cities": pass_through_cities,
            "source_rows": int(source_rows),
            "unique_points": int(len(clean_points)),
            "distance_km": round(float(distance_km), 3),
            "drive_minutes_to_next_base": _finite_float(drive_minutes),
            "available_visit_minutes": _finite_float(available_visit_minutes),
            "geometry_mode": geometry_mode,
            "geometry_source": geometry_source,
            "straight_line_fallback_count": fallback_count,
            "road_geometry_percent": road_geometry_percent,
            "layer_var": layer_var,
            "show_by_default": bool(show_by_default),
            "draw_status": status,
            "issue": issue,
            "notes": notes,
        }
    )



def _matrix_route_color(method, profile, trip_days):
    base_color = MATRIX_METHOD_STYLES.get(str(method), ("Route", "#E91E63", None, 0.0))[1]
    if str(profile) == "balanced":
        return base_color
    if str(profile) == "relaxed":
        return {"#0096C7": "#38BDF8", "#111827": "#64748B", "#00A63E": "#2A9D8F"}.get(base_color, "#00897B")
    return {"#0096C7": "#7C2DFF", "#111827": "#7A5195", "#00A63E": "#E91E63"}.get(base_color, "#A020F0")



def _matrix_route_offset(method, profile, trip_days):
    if not MATRIX_ROUTE_OFFSETS_ENABLED:
        return 0.0
    method_offset = MATRIX_METHOD_STYLES.get(str(method), ("Route", "#E91E63", None, 0.0))[3]
    profile_offset = MATRIX_PROFILE_STYLE_OFFSETS.get(str(profile), 0.0)
    day_offset = {7: -0.18, 9: 0.0, 12: 0.18}.get(int(_finite_float(trip_days, 7)), 0.0)
    return float(method_offset + profile_offset + day_offset)



def _matrix_route_control_label(first):
    if "comparison_label" in first and str(first.get("comparison_label", "")).strip():
        return str(first.get("comparison_label"))
    trip_days = int(_finite_float(first.get("trip_days", 7), 7))
    method_label = MATRIX_METHOD_STYLES.get(str(first.get("method", "")), ("Route", "#E91E63", None, 0.0))[0]
    profile_label = str(first.get("profile_label", str(first.get("profile", "Route")).title()))
    return f"{trip_days}-Day · {method_label} · {profile_label}"



def _full_route_points_from_plan(plan_df):
    if plan_df is None or plan_df.empty:
        return []
    sorted_plan = plan_df.copy()
    sorted_plan["day"] = _numeric_catalog_series(sorted_plan, "day", 1).astype(int)
    sorted_plan["stop_order"] = _numeric_catalog_series(sorted_plan, "stop_order", 1).astype(int)
    sorted_plan = sorted_plan.sort_values(["day", "stop_order", "attraction_name"]).reset_index(drop=True)
    full_points = []
    for _, group in sorted_plan.groupby("day", sort=True):
        day_points = _route_points_from_day_group(group)
        if full_points and day_points:
            full_points.extend(day_points[1:] if full_points[-1] == day_points[0] else day_points)
        else:
            full_points.extend(day_points)
    return _dedupe_route_points(full_points)
