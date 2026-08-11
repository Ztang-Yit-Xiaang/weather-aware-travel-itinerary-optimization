"""Folium layer composition for the production itinerary map."""

from __future__ import annotations

from pathlib import Path

import folium
import numpy as np
import pandas as pd
from folium import plugins

from .blueprint_core import (
    PROFILE_CONFIGS,
    _estimate_drive_minutes,
    _leg_classification,
    _load_csv,
    _numeric_catalog_series,
    _route_distance_km,
    _scenic_leg_waypoints,
)
from .blueprint_render_primitives import (
    DAY_COLORS,
    MATRIX_DEFAULT_ROUTE_KEY,
    MATRIX_METHOD_STYLES,
    ROUTE_CONTEXT_PANE,
    ROUTE_CORE_PANE,
    ROUTE_TOP_PANE,
    _add_canvas_direction_arrows,
    _add_full_scene_route_band,
    _add_static_result_line,
    _append_route_debug_row,
    _dedupe_route_points,
    _escape,
    _fetch_osrm_route,
    _fetch_stitched_route,
    _finite_float,
    _full_route_points_from_plan,
    _hotel_loop_points_from_day_group,
    _hotel_point_from_day_group,
    _matrix_route_color,
    _matrix_route_control_label,
    _matrix_route_offset,
    _offset_route_points_for_visibility,
    _route_bounds,
    _route_debug_value,
    _route_endpoint_city_labels,
    _route_mode_label,
    _route_points_for_names,
    _route_points_from_day_group,
    _transition_points_from_day_group,
)


def _add_flow_route(
    layer,
    points,
    *,
    color,
    route_cache,
    run_live=False,
    tooltip="route",
    popup_html=None,
    weight=5,
    dash_array=None,
    opacity=0.88,
    arrow_offset=7,
    ant_delay=850,
    pane=ROUTE_CORE_PANE,
):
    """Draw every route leg separately so one bad leg cannot hide the rest."""
    clean_points = _dedupe_route_points(points)
    if len(clean_points) < 2:
        return clean_points, "single-point"

    full_path = []
    modes = []
    line_count = 0
    for leg_idx, (left, right) in enumerate(
        zip(clean_points[:-1], clean_points[1:], strict=False),
        start=1,
    ):
        leg_path, leg_mode = _fetch_osrm_route([left, right], route_cache, run_live=run_live)
        if not leg_path or len(leg_path) < 2:
            leg_path = [left, right]
            leg_mode = "straight-line-fallback"
        modes.append(leg_mode)
        if full_path and leg_path:
            full_path.extend(leg_path[1:])
        else:
            full_path.extend(leg_path)

        leg_popup = popup_html if leg_idx == 1 and popup_html else None
        folium.PolyLine(leg_path, color="#FFFFFF", weight=weight + 3, opacity=0.74, pane=pane).add_to(layer)
        folium.PolyLine(
            leg_path,
            color=color,
            weight=weight,
            opacity=opacity,
            dash_array=dash_array,
            tooltip=f"{tooltip} · leg {leg_idx}: {leg_mode}",
            popup=folium.Popup(leg_popup, max_width=320) if leg_popup else None,
            pane=pane,
        ).add_to(layer)
        # PolyLineTextPath inserts a <textPath> child of the polyline's SVG
        # <path>. The map is created with prefer_canvas=True (see
        # build_production_trip_map), so polylines render to canvas and
        # polyline._path is undefined; leaflet.textpath.js then throws
        # "Cannot read properties of undefined (reading 'setAttribute')"
        # which terminates the enclosing <script> block and prevents
        # GroupedLayerControl from initializing. Direction is already
        # conveyed by dash_array and color; switching to
        # prefer_canvas=False is not viable at ~343 polylines.
        # plugins.PolyLineTextPath(
        #     route_line,
        #     "   >   ",
        #     repeat=True,
        #     offset=arrow_offset,
        #     attributes={"fill": color, "font-weight": "700", "font-size": "14"},
        # ).add_to(layer)
        line_count += 1

    mode = _route_mode_label(modes)
    if full_path and len(full_path) >= 2:
        plugins.AntPath(
            full_path,
            color=color,
            pulse_color="#FFFFFF",
            weight=weight,
            opacity=max(0.72, opacity - 0.06),
            delay=ant_delay,
            dash_array=[8, 16] if dash_array else [10, 16],
            pane=pane,
        ).add_to(layer)
        route_distance = _route_distance_km(full_path)
        arrow_count = _add_canvas_direction_arrows(
            layer,
            full_path,
            color=color,
            pane=pane,
            interval_km=90.0 if route_distance < 180 else 240.0,
            max_arrows=1 if route_distance < 60 else 2 if route_distance < 180 else 3,
            class_name="blueprint-route-arrow blueprint-route-arrow-day",
        )
    else:
        arrow_count = 0
    return full_path, f"{mode}; {line_count} connected legs; canvas-arrow-markers={arrow_count}"



def _add_intercity_route_layer(map_object, route_name, sequence, color, route_cache, run_live, scenic=False, show=True):
    layer = folium.FeatureGroup(name=route_name, show=show)
    leg_rows = []
    all_path_points = []

    for leg_order, (left_name, right_name) in enumerate(
        zip(sequence[:-1], sequence[1:], strict=False),
        start=1,
    ):
        classification = _leg_classification(left_name, right_name, scenic)
        waypoints = (
            _scenic_leg_waypoints(left_name, right_name) if scenic else _route_points_for_names([left_name, right_name])
        )
        path, route_mode = _fetch_osrm_route(waypoints, route_cache, run_live=run_live)
        if "fallback" in route_mode and scenic:
            path = waypoints
            route_mode = "curated scenic waypoint fallback"
        if all_path_points and path:
            all_path_points.extend(path[1:])
        else:
            all_path_points.extend(path)

        estimated_minutes = _estimate_drive_minutes(path, classification)
        waypoint_label = f"{left_name} -> {right_name}"
        popup = f"""
        <b>{_escape(route_name)}</b><br/>
        Segment: {_escape(waypoint_label)}<br/>
        Classification: {_escape(classification)}<br/>
        Geometry source: {_escape(route_mode)}<br/>
        Estimated drive: {estimated_minutes:.1f} minutes
        """
        tooltip = f"{waypoint_label}: {classification}"
        opacity = (0.68 if scenic else 0.48) if "fallback" not in route_mode else 0.34
        route_pane = ROUTE_CONTEXT_PANE
        folium.PolyLine(path, color="#FFFFFF", weight=6 if scenic else 5, opacity=0.58, pane=route_pane).add_to(layer)
        folium.PolyLine(
            path,
            color=color,
            weight=4 if scenic else 3,
            opacity=opacity,
            dash_array=None if scenic else "8, 10",
            tooltip=tooltip,
            popup=folium.Popup(popup, max_width=330, min_width=220),
            pane=route_pane,
        ).add_to(layer)
        # Disabled for the same reason as in _add_intercity_route_layer
        # (incompatible with prefer_canvas=True; see comment there).
        # plugins.PolyLineTextPath(
        #     route_line,
        #     "   >   ",
        #     repeat=True,
        #     offset=7,
        #     attributes={"fill": color, "font-weight": "700", "font-size": "13"},
        # ).add_to(layer)

        leg_rows.append(
            {
                "route_layer": route_name,
                "leg_order": leg_order,
                "from": left_name,
                "to": right_name,
                "from_name": left_name,
                "to_name": right_name,
                "route_type": "scenic_ca1_pch_with_stanford_detour" if scenic else "fastest_intercity_route",
                "waypoint_label": waypoint_label,
                "estimated_drive_minutes": estimated_minutes,
                "ca1_connector_classification": classification,
                "geometry_source": route_mode,
            }
        )

    if all_path_points:
        plugins.AntPath(
            all_path_points,
            color=color,
            pulse_color="#FFFFFF",
            weight=4 if scenic else 3,
            opacity=0.52 if scenic else 0.36,
            delay=950 if scenic else 1050,
            dash_array=[14, 18] if scenic else [8, 14],
            pane=route_pane,
        ).add_to(layer)
        _add_canvas_direction_arrows(
            layer,
            all_path_points,
            color=color,
            pane=route_pane,
            interval_km=260.0,
            max_arrows=2,
            class_name="blueprint-route-arrow blueprint-route-arrow-context",
        )

    layer.add_to(map_object)
    return leg_rows, all_path_points, layer



def _add_selected_result_layer(
    map_object, day_plan_df, route_cache=None, run_live=False, route_debug_rows=None, show_by_default=True
):
    if day_plan_df is None or day_plan_df.empty:
        return None

    layer = folium.FeatureGroup(
        name="Selected Result · Default Hierarchical Route",
        show=bool(show_by_default),
    )

    day_frame = day_plan_df.copy()
    day_frame["day"] = _numeric_catalog_series(day_frame, "day", 1).astype(int)
    day_frame["stop_order"] = _numeric_catalog_series(day_frame, "stop_order", 1).astype(int)
    day_frame = day_frame.sort_values(["day", "stop_order", "attraction_name"]).reset_index(drop=True)

    for day, group in day_frame.groupby("day", sort=True):
        color = DAY_COLORS[(int(day) - 1) % len(DAY_COLORS)]
        first = group.iloc[0]
        day_points = _route_points_from_day_group(group)
        draw_points, route_mode = (
            _fetch_stitched_route(day_points, route_cache, run_live=run_live)
            if route_cache is not None
            else (day_points, "static-polyline")
        )

        geometry_mode = _add_static_result_line(
            layer,
            draw_points,
            color=color,
            tooltip=f"Selected result day {int(day)}",
            popup_html=f"<b>Selected result route</b><br/>Day {int(day)}",
            weight=7,
            opacity=0.98,
            geometry_mode_prefix=route_mode,
        )
        _append_route_debug_row(
            route_debug_rows,
            layer_group="Selected Result",
            layer_name=f"Selected Result · Day {int(day)}",
            comparison_type="selected_result",
            profile=str(first.get("profile", "balanced")),
            method=str(first.get("method", "hierarchical_gurobi_pipeline")),
            trip_days=_route_debug_value(first.get("trip_days", "")),
            day=int(day),
            route_type=str(first.get("route_type", "")),
            route_start_city=str(first.get("route_start_city", first.get("city", ""))),
            route_end_city=str(first.get("route_end_city", first.get("overnight_city", ""))),
            pass_through_cities=str(first.get("pass_through_cities", "")),
            drive_minutes=first.get("drive_minutes_to_next_base", np.nan),
            available_visit_minutes=first.get("available_visit_minutes", np.nan),
            points=draw_points,
            geometry_mode=geometry_mode,
            show_by_default=bool(show_by_default),
            source_rows=len(group),
        )

        for row in group.itertuples(index=False):
            point = [float(row.latitude), float(row.longitude)]
            badge_text = f"R{int(row.day)}.{int(row.stop_order)}"
            popup = (
                f"<b>{_escape(row.attraction_name)}</b><br/>"
                f"Selected result route<br/>"
                f"Day {int(row.day)}, stop {int(row.stop_order)}<br/>"
                f"City: {_escape(row.city)}"
            )
            folium.CircleMarker(
                location=point,
                radius=9,
                color=color,
                fill=True,
                fillColor=color,
                fillOpacity=0.92,
                weight=3,
                tooltip=f"Selected result {badge_text}: {row.attraction_name}",
                popup=folium.Popup(popup, max_width=280),
            ).add_to(layer)
            folium.Marker(
                location=point,
                icon=folium.DivIcon(
                    html=f"""
                    <div style='background:{color}; color:white; border:2px solid white;
                        border-radius:14px; font-size:12px; font-weight:800;
                        padding:3px 8px; box-shadow:0 2px 8px rgba(0,0,0,0.35);
                        white-space:nowrap;'>
                        {badge_text}
                    </div>
                    """
                ),
                tooltip=f"Selected result {badge_text}: {row.attraction_name}",
                popup=folium.Popup(popup, max_width=280),
            ).add_to(layer)

    layer.add_to(map_object)
    return layer



def _add_model_comparison_layers(
    map_object, output_dir, route_cache, run_live, show_by_default=False, route_debug_rows=None
):
    output_dir = Path(output_dir)
    grouped_layers = {"Trip Length Comparison": [], "Method Comparison": []}
    # These CSVs are the handoff point from the experiment pipeline into the map.
    route_files = [
        output_dir / "production_trip_length_route_stops.csv",
        output_dir / "production_method_route_stops.csv",
    ]
    frames = [frame for frame in (_load_csv(path) for path in route_files) if not frame.empty]
    if not frames:
        print("[map comparison debug] no comparison route-stop CSVs loaded from", output_dir)
        return {}

    comparison_df = pd.concat(frames, ignore_index=True, sort=False)
    if comparison_df.empty:
        print("[map comparison debug] no comparison route-stop CSVs loaded from", output_dir)
        return {}
    print("[map comparison debug] loaded comparison route stops:", comparison_df.shape)
    debug_cols = [
        col for col in ["comparison_type", "comparison_label", "method", "trip_days"] if col in comparison_df.columns
    ]
    if debug_cols:
        print(comparison_df[debug_cols].drop_duplicates().to_string(index=False))

    def layer_name_for_group(comparison_type, comparison_label, method, trip_days):
        if str(comparison_type) == "trip_length":
            try:
                label_days = int(trip_days)
            except Exception:
                label_days = int(str(comparison_label).split()[0])
            return f"Trip Length · {label_days}-Day Hybrid Bandit + Small Gurobi"
        if str(comparison_label).strip():
            return str(comparison_label)
        return f"Method · {method}"

    def style_for_group(comparison_type, method, trip_days):
        if str(comparison_type) == "trip_length":
            day_color_map = {7: "#2563EB", 9: "#F4A261", 12: "#7A5195"}
            try:
                color = day_color_map.get(int(trip_days), "#2563EB")
            except Exception:
                color = "#2563EB"
            return color, None, f"{int(trip_days)}D"
        if str(method) == "hierarchical_gurobi_pipeline":
            return "#2563EB", None, "H"
        if str(method) == "hierarchical_greedy_baseline":
            return "#6C757D", "8 10", "G"
        if str(method) == "hierarchical_bandit_gurobi_repair":
            return "#5FAD56", None, "B"
        return "#2563EB", None, "M"

    for group_values, layer_df in comparison_df.groupby(
        ["comparison_type", "comparison_label", "method", "trip_days"], dropna=False
    ):
        comparison_type, comparison_label, method, trip_days = group_values
        layer_name = layer_name_for_group(comparison_type, comparison_label, method, trip_days)
        color, dash_array, badge_prefix = style_for_group(comparison_type, method, trip_days)
        show_layer = bool(show_by_default)

        layer = folium.FeatureGroup(
            name=layer_name,
            show=show_layer,
        )
        day_frame = layer_df.copy()
        day_frame["day"] = _numeric_catalog_series(day_frame, "day", 1).astype(int)
        day_frame["stop_order"] = _numeric_catalog_series(day_frame, "stop_order", 1).astype(int)
        day_frame = day_frame.sort_values(["day", "stop_order", "attraction_name"]).reset_index(drop=True)
        for day, group in day_frame.groupby("day", sort=True):
            first = group.iloc[0]
            method_display_name = str(first.get("method_display_name", layer_name.replace("Method · ", "")))
            hotel_lat = _finite_float(first.get("hotel_latitude"))
            hotel_lon = _finite_float(first.get("hotel_longitude"))
            day_points = _route_points_from_day_group(group)
            draw_points, route_mode = _fetch_stitched_route(day_points, route_cache, run_live=run_live)
            popup_html = f"<b>{_escape(layer_name)}</b><br/>Day {int(day)}<br/>Method: {_escape(method_display_name)}"
            geometry_mode = _add_static_result_line(
                layer,
                draw_points,
                color=color,
                dash_array=dash_array,
                tooltip=f"{layer_name} day {int(day)}",
                popup_html=popup_html,
                opacity=0.95,
                geometry_mode_prefix=route_mode,
            )
            _append_route_debug_row(
                route_debug_rows,
                layer_group="Trip Length Comparison" if str(comparison_type) == "trip_length" else "Method Comparison",
                layer_name=f"{layer_name} · Day {int(day)}",
                comparison_type=str(comparison_type),
                profile="balanced",
                method=str(method),
                trip_days=_route_debug_value(trip_days),
                day=int(day),
                route_type=str(first.get("route_type", "")),
                route_start_city=str(first.get("route_start_city", first.get("city", ""))),
                route_end_city=str(first.get("route_end_city", first.get("overnight_city", ""))),
                pass_through_cities=str(first.get("pass_through_cities", "")),
                drive_minutes=first.get("drive_minutes_to_next_base", np.nan),
                available_visit_minutes=first.get("available_visit_minutes", np.nan),
                points=draw_points,
                geometry_mode=geometry_mode,
                show_by_default=show_layer,
                source_rows=len(group),
            )
            if np.isfinite(hotel_lat) and np.isfinite(hotel_lon):
                folium.Marker(
                    location=[hotel_lat, hotel_lon],
                    tooltip=f"{layer_name} hotel/base: {first.get('hotel_name', 'hotel')}",
                    popup=folium.Popup(
                        f"<b>{_escape(str(first.get('hotel_name', 'hotel/base')))}</b><br/>"
                        f"Layer: {_escape(layer_name)}<br/>"
                        f"Overnight city: {_escape(str(first.get('overnight_city', first.get('city', 'unknown'))))}",
                        max_width=260,
                    ),
                    icon=folium.Icon(color="darkred", icon="hotel", prefix="fa"),
                ).add_to(layer)
            for row in group.itertuples(index=False):
                point = [float(row.latitude), float(row.longitude)]
                popup = f"""
                <b>{_escape(row.attraction_name)}</b><br/>
                Layer: {_escape(layer_name)}<br/>
                Method: {_escape(str(getattr(row, "method_display_name", method_display_name)))}<br/>
                Day {int(row.day)}, stop {int(row.stop_order)}<br/>
                City: {_escape(row.city)}<br/>
                Category: {_escape(row.category)}<br/>
                Source list: {_escape(getattr(row, "source_list", "unknown"))}<br/>
                Final POI value: {float(getattr(row, "final_poi_value", 0.0) or 0.0):.3f}<br/>
                Social must-go: {"yes" if bool(getattr(row, "social_must_go", False)) else "no"}<br/>
                Geometry source: {_escape(str(geometry_mode))}<br/>
                Status: {_escape(str(getattr(row, "status", "unknown")))}<br/>
                Notes: {_escape(str(getattr(row, "notes", "")))}
                """
                folium.CircleMarker(
                    location=point,
                    radius=8,
                    color=color,
                    fill=True,
                    fillColor=color,
                    fillOpacity=0.86,
                    weight=3,
                    tooltip=f"{layer_name} stop {int(row.stop_order)}: {row.attraction_name}",
                    popup=folium.Popup(popup, max_width=290),
                ).add_to(layer)
                if str(comparison_type) == "trip_length":
                    badge_text = f"{badge_prefix}{int(row.day)}.{int(row.stop_order)}"
                else:
                    badge_text = f"{badge_prefix}{int(row.day)}.{int(row.stop_order)}"
                folium.Marker(
                    location=point,
                    icon=folium.DivIcon(
                        html=f"""
                        <div style='background:{color}; color:white; border:2px solid white; border-radius:14px;
                            font-size:11px; font-weight:700; padding:2px 7px; box-shadow:0 1px 6px rgba(0,0,0,0.30); white-space:nowrap;'>
                            {badge_text}
                        </div>
                        """
                    ),
                    tooltip=f"{layer_name} stop {int(row.stop_order)}: {row.attraction_name}",
                    popup=folium.Popup(popup, max_width=290),
                ).add_to(layer)
        layer.add_to(map_object)
        group_name = "Trip Length Comparison" if str(comparison_type) == "trip_length" else "Method Comparison"
        grouped_layers[group_name].append(layer)
    return {name: layers for name, layers in grouped_layers.items() if layers}



def _add_traveler_overview_layers(
    map_object, profile_day_plans, route_cache=None, run_live=False, route_debug_rows=None, show_by_default=True
):
    layers = []
    styles = {
        "relaxed": ("#2A9D8F", "8 10"),
        "balanced": ("#2563EB", None),
        "explorer": ("#7A5195", "2 8"),
    }
    for profile_name, plan_df in profile_day_plans.items():
        if plan_df is None or plan_df.empty:
            continue
        config = PROFILE_CONFIGS[profile_name]
        color, dash_array = styles.get(profile_name, ("#2563EB", None))
        layer_name = f"Traveler · {config['label']} Full Route"
        layer = folium.FeatureGroup(name=layer_name, show=bool(show_by_default))

        sorted_plan = plan_df.copy()
        sorted_plan["day"] = _numeric_catalog_series(sorted_plan, "day", 1).astype(int)
        sorted_plan["stop_order"] = (
            _numeric_catalog_series(sorted_plan, "stop_order", 1).astype(int)
        )
        sorted_plan = sorted_plan.sort_values(["day", "stop_order", "attraction_name"]).reset_index(drop=True)

        full_points = []
        for _, group in sorted_plan.groupby("day", sort=True):
            day_points = _route_points_from_day_group(group)
            if full_points and day_points:
                full_points.extend(day_points[1:] if full_points[-1] == day_points[0] else day_points)
            else:
                full_points.extend(day_points)

        draw_points, route_mode = (
            _fetch_stitched_route(full_points, route_cache, run_live=run_live)
            if route_cache is not None
            else (full_points, "static-polyline")
        )
        geometry_mode = _add_static_result_line(
            layer,
            draw_points,
            color=color,
            dash_array=dash_array,
            tooltip=layer_name,
            popup_html=(
                f"<b>{_escape(layer_name)}</b><br/>Stops: {len(sorted_plan)}<br/>Mode: {_escape(config['route_mode'])}"
            ),
            weight=5,
            opacity=0.82,
            geometry_mode_prefix=route_mode,
        )
        _append_route_debug_row(
            route_debug_rows,
            layer_group="Traveler Comparison",
            layer_name=layer_name,
            comparison_type="traveler_profile",
            profile=profile_name,
            method=str(sorted_plan.iloc[0].get("method", "profile_day_plan")),
            trip_days=_route_debug_value(sorted_plan.iloc[0].get("trip_days", sorted_plan["day"].max())),
            day="all",
            route_type="full_profile_route",
            route_start_city=str(sorted_plan.iloc[0].get("route_start_city", sorted_plan.iloc[0].get("city", ""))),
            route_end_city=str(
                sorted_plan.iloc[-1].get("route_end_city", sorted_plan.iloc[-1].get("overnight_city", ""))
            ),
            points=draw_points,
            geometry_mode=geometry_mode,
            show_by_default=bool(show_by_default),
            source_rows=len(sorted_plan),
        )
        layer.add_to(map_object)
        layers.append(layer)
    return layers



def _add_city_detail_layers(
    map_object, day_plan_df, route_cache, run_live, route_debug_rows=None, route_debug_registry=None
):
    if day_plan_df is None or day_plan_df.empty:
        return [], []

    sorted_plan = day_plan_df.copy()
    sorted_plan["day"] = _numeric_catalog_series(sorted_plan, "day", 1).astype(int)
    sorted_plan["stop_order"] = _numeric_catalog_series(sorted_plan, "stop_order", 1).astype(int)
    sorted_plan = sorted_plan.sort_values(["day", "stop_order", "attraction_name"]).reset_index(drop=True)

    city_loop_groups = {}
    transition_groups = []
    for day, group in sorted_plan.groupby("day", sort=True):
        first = group.iloc[0]
        loop_points, local_rows = _hotel_loop_points_from_day_group(group)
        if loop_points and not local_rows.empty:
            city_name = str(first.get("overnight_city", first.get("city", "")))
            city_loop_groups.setdefault(f"City Loop · {city_name}", []).append(
                (int(day), group, local_rows, loop_points)
            )
        transition_points, transition_rows = _transition_points_from_day_group(group)
        if transition_points:
            transition_groups.append((int(day), group, transition_rows, transition_points))

    city_layers = []
    transition_layers = []
    for layer_index, (layer_name, day_groups) in enumerate(city_loop_groups.items()):
        layer = folium.FeatureGroup(name=layer_name, show=False)
        layer_points = []
        for day, group, local_rows, day_points in day_groups:
            first = group.iloc[0]
            color = DAY_COLORS[(int(day) - 1) % len(DAY_COLORS)]
            overnight_city = str(first.get("overnight_city", first["city"]))
            hotel_point = _hotel_point_from_day_group(group)
            popup_html = (
                f"<b>{_escape(layer_name)}</b><br/>"
                f"Day {int(day)}<br/>"
                f"Route type: hotel loop<br/>"
                f"Start/end hotel: {_escape(first.get('hotel_name', 'selected hotel/base'))}<br/>"
                f"Local stops shown: {len(local_rows)}"
            )
            day_path, day_route_mode = _add_flow_route(
                layer,
                day_points,
                color=color,
                route_cache=route_cache,
                run_live=run_live,
                tooltip=f"{layer_name} · Day {int(day)} hotel loop",
                popup_html=popup_html,
                dash_array=None,
                opacity=0.82,
                weight=4,
                ant_delay=760,
                pane=ROUTE_TOP_PANE,
            )
            layer_points.extend(day_path or day_points)
            _append_route_debug_row(
                route_debug_rows,
                layer_group="City Detail",
                layer_name=f"{layer_name} · Day {int(day)}",
                comparison_type="city_detail",
                profile="balanced",
                method=str(first.get("method", "profile_day_plan")),
                trip_days=_route_debug_value(first.get("trip_days", sorted_plan["day"].max())),
                day=int(day),
                route_type="hotel_loop",
                route_start_city=overnight_city,
                route_end_city=overnight_city,
                pass_through_cities=str(first.get("pass_through_cities", "")),
                drive_minutes=0.0,
                available_visit_minutes=first.get("available_visit_minutes", np.nan),
                points=day_points,
                geometry_mode=f"{day_route_mode}; city-detail-arrows",
                show_by_default=False,
                source_rows=len(local_rows),
                layer_var=layer.get_name(),
                notes=(
                    f"hotel_loop_start_end_same=true; hotel_name={_escape(first.get('hotel_name', ''))}; "
                    f"hotel_lat={hotel_point[0]:.6f}; hotel_lon={hotel_point[1]:.6f}"
                ),
            )

            folium.Marker(
                location=hotel_point,
                tooltip=f"{layer_name}: Day {int(day)} start/end hotel",
                popup=folium.Popup(
                    f"<b>{_escape(first.get('hotel_name', 'Selected hotel/base'))}</b><br/>"
                    f"{_escape(layer_name)}<br/>Day {int(day)} loop starts and ends here.",
                    max_width=260,
                ),
                icon=folium.Icon(color="darkred", icon="hotel", prefix="fa"),
            ).add_to(layer)

            for row in local_rows.itertuples(index=False):
                point = [float(row.latitude), float(row.longitude)]
                popup = f"""
                <b>{_escape(row.attraction_name)}</b><br/>
                {_escape(layer_name)}<br/>
                Day {int(row.day)}, stop {int(row.stop_order)}<br/>
                City: {_escape(row.city)}<br/>
                Category: {_escape(row.category)}<br/>
                Source: {_escape(getattr(row, "attraction_source", getattr(row, "source_list", "unknown")))}
                """
                folium.CircleMarker(
                    location=point,
                    radius=8,
                    color=color,
                    fill=True,
                    fillColor=color,
                    fillOpacity=0.88,
                    weight=3,
                    tooltip=f"{layer_name} Day {int(row.day)} stop {int(row.stop_order)}: {row.attraction_name}",
                    popup=folium.Popup(popup, max_width=280),
                ).add_to(layer)
                folium.Marker(
                    location=point,
                    icon=folium.DivIcon(
                        html=f"""
                        <div style='background:{color}; color:white; border:2px solid white; border-radius:14px;
                            font-size:11px; font-weight:700; padding:2px 7px; box-shadow:0 1px 6px rgba(0,0,0,0.30); white-space:nowrap;'>
                            D{int(row.day)}.{int(row.stop_order)}
                        </div>
                        """
                    ),
                    tooltip=f"{layer_name} stop {int(row.stop_order)}: {row.attraction_name}",
                    popup=folium.Popup(popup, max_width=280),
                ).add_to(layer)

        layer.add_to(map_object)
        city_layers.append(layer)
        bounds = _route_bounds(layer_points)
        if route_debug_registry is not None:
            route_debug_registry.append(
                {
                    "label": layer_name.replace("City Loop · ", ""),
                    "control_label": layer_name,
                    "control_id": f"city_detail_{layer_index + 1}",
                    "family": "city_detail",
                    "selector_group": "city_detail",
                    "color": DAY_COLORS[layer_index % len(DAY_COLORS)],
                    "pane": ROUTE_TOP_PANE,
                    "layer_var": layer.get_name(),
                    "offset_index": 0,
                    "default_checked": False,
                    "default_visible": False,
                    "quick_groups": ["city_detail"],
                    "distance_km": round(float(_route_distance_km(_dedupe_route_points(layer_points))), 3),
                    "unique_points": len(_dedupe_route_points(layer_points)),
                    "bounds": bounds,
                }
            )

    for transition_index, (day, group, transition_rows, transition_points) in enumerate(transition_groups, start=1):
        first = group.iloc[0]
        start_city = str(first.get("route_start_city", first.get("city", "")))
        end_city = str(first.get("route_end_city", first.get("overnight_city", "")))
        layer_name = f"Transition Route · Day {int(day)} · {start_city} to {end_city}"
        layer = folium.FeatureGroup(name=layer_name, show=False)
        color = "#64748B"
        popup_html = (
            f"<b>{_escape(layer_name)}</b><br/>"
            f"Previous base: {_escape(first.get('route_start_name', start_city))}<br/>"
            f"Destination base: {_escape(first.get('hotel_name', end_city))}<br/>"
            f"Pass-through stops shown: {len(transition_rows)}"
        )
        transition_path, transition_mode = _add_flow_route(
            layer,
            transition_points,
            color=color,
            route_cache=route_cache,
            run_live=run_live,
            tooltip=layer_name,
            popup_html=popup_html,
            dash_array="8 10",
            opacity=0.76,
            weight=4,
            ant_delay=820,
            pane=ROUTE_TOP_PANE,
        )
        for point, label, icon_name, icon_color in [
            (
                transition_points[0],
                f"Transition start: {first.get('route_start_name', start_city)}",
                "play",
                "cadetblue",
            ),
            (
                transition_points[-1],
                f"Transition end hotel: {first.get('hotel_name', end_city)}",
                "flag-checkered",
                "darkred",
            ),
        ]:
            folium.Marker(
                location=point,
                tooltip=f"{layer_name}: {label}",
                popup=folium.Popup(f"<b>{_escape(label)}</b><br/>{_escape(layer_name)}", max_width=260),
                icon=folium.Icon(color=icon_color, icon=icon_name, prefix="fa"),
            ).add_to(layer)
        for row in transition_rows.itertuples(index=False):
            point = [float(row.latitude), float(row.longitude)]
            folium.CircleMarker(
                location=point,
                radius=7,
                color=color,
                fill=True,
                fillColor=color,
                fillOpacity=0.75,
                weight=2,
                tooltip=f"{layer_name} pass-through stop: {row.attraction_name}",
                popup=folium.Popup(
                    f"<b>{_escape(row.attraction_name)}</b><br/>Transition stop<br/>City: {_escape(row.city)}",
                    max_width=260,
                ),
            ).add_to(layer)
        layer.add_to(map_object)
        transition_layers.append(layer)
        _append_route_debug_row(
            route_debug_rows,
            layer_group="Transition Route",
            layer_name=layer_name,
            comparison_type="transition",
            profile="balanced",
            method=str(first.get("method", "profile_day_plan")),
            trip_days=_route_debug_value(first.get("trip_days", sorted_plan["day"].max())),
            day=int(day),
            route_type=str(first.get("route_type", "relocation")),
            route_start_city=start_city,
            route_end_city=end_city,
            pass_through_cities=str(first.get("pass_through_cities", "")),
            drive_minutes=first.get("drive_minutes_to_next_base", np.nan),
            available_visit_minutes=first.get("available_visit_minutes", np.nan),
            points=transition_points,
            geometry_mode=f"{transition_mode}; transition-route-arrows",
            show_by_default=False,
            source_rows=max(1, len(transition_rows)),
            layer_var=layer.get_name(),
            notes="transition_route_separate_from_city_loop=true",
        )
        if route_debug_registry is not None:
            transition_draw_points = transition_path or transition_points
            route_debug_registry.append(
                {
                    "label": f"Day {int(day)} · {start_city} to {end_city}",
                    "control_label": layer_name,
                    "control_id": f"transition_{transition_index}",
                    "family": "transition",
                    "selector_group": "transition",
                    "color": color,
                    "pane": ROUTE_TOP_PANE,
                    "layer_var": layer.get_name(),
                    "offset_index": 0,
                    "default_checked": False,
                    "default_visible": False,
                    "quick_groups": ["transition"],
                    "distance_km": round(float(_route_distance_km(_dedupe_route_points(transition_draw_points))), 3),
                    "unique_points": len(_dedupe_route_points(transition_draw_points)),
                    "bounds": _route_bounds(transition_draw_points),
                }
            )
    return city_layers, transition_layers



def _add_route_matrix_layers(
    map_object, output_dir, route_cache, run_live, route_debug_rows=None, route_debug_registry=None
):
    matrix_routes = _load_csv(Path(output_dir) / "production_route_matrix_route_stops.csv")
    if matrix_routes.empty or "route_key" not in matrix_routes.columns:
        return [], []

    matrix_routes = matrix_routes.copy()
    matrix_routes["trip_days"] = _numeric_catalog_series(matrix_routes, "trip_days", 7).astype(int)
    matrix_routes["day"] = _numeric_catalog_series(matrix_routes, "day", 1).astype(int)
    matrix_routes["stop_order"] = (
        _numeric_catalog_series(matrix_routes, "stop_order", 1).astype(int)
    )
    matrix_routes = matrix_routes.sort_values(["trip_days", "method", "profile", "day", "stop_order"]).reset_index(
        drop=True
    )

    social_catalog = _load_csv(Path(output_dir) / "production_social_must_go_candidates.csv")
    if social_catalog.empty:
        enriched = _load_csv(Path(output_dir) / "production_enriched_poi_catalog.csv")
        if not enriched.empty:
            social_catalog = enriched[
                enriched.get("social_must_go", pd.Series(False, index=enriched.index)).astype(bool)
                | enriched.get("social_score", pd.Series(0.0, index=enriched.index)).astype(float).ge(0.70)
            ].copy()

    all_layers = []
    hidden_layers = []
    layer_var_updates = {}
    for _route_index, (route_key, group) in enumerate(matrix_routes.groupby("route_key", sort=False), start=1):
        group = group.sort_values(["day", "stop_order", "attraction_name"]).reset_index(drop=True)
        if group.empty:
            continue
        first = group.iloc[0]
        route_key = str(route_key)
        trip_days = int(first.get("trip_days", 7))
        method = str(first.get("method", "route"))
        profile = str(first.get("profile", "balanced"))
        profile_label = str(first.get("profile_label", profile.title()))
        method_label, _base_color, dash_array, _offset = MATRIX_METHOD_STYLES.get(
            method, ("Route", "#E91E63", None, 0.0)
        )
        control_label = _matrix_route_control_label(first)
        map_badge_label = f"{trip_days}d {method_label.split()[0]} {profile_label[:1].upper()}"
        color = _matrix_route_color(method, profile, trip_days)
        offset_index = _matrix_route_offset(method, profile, trip_days)
        default_visible = route_key == MATRIX_DEFAULT_ROUTE_KEY
        selector_parent = str(first.get("selector_parent", f"d{trip_days}__{method}"))
        start_city, end_city = _route_endpoint_city_labels(group)

        full_points = _full_route_points_from_plan(group)
        if len(full_points) >= 2:
            road_points, route_mode = _fetch_stitched_route(full_points, route_cache, run_live=run_live)
            draw_points = _offset_route_points_for_visibility(road_points, offset_index)
            full_layer_name = f"Route Matrix · {control_label}"
            full_layer = folium.FeatureGroup(name=full_layer_name, show=default_visible)
            geometry_mode = _add_full_scene_route_band(
                full_layer,
                draw_points,
                color=color,
                dash_array=dash_array,
                tooltip=f"{control_label} full route",
                popup_html=(
                    f"<b>{_escape(control_label)}</b><br/>"
                    f"Route key: {_escape(route_key)}<br/>"
                    f"Start: {_escape(start_city)}<br/>"
                    f"End: {_escape(end_city)}<br/>"
                    f"Hotels are controlled by Hotel Showcase."
                ),
                label=map_badge_label,
                weight=5 if not default_visible else 7,
                opacity=0.88 if not default_visible else 0.95,
                pane=ROUTE_TOP_PANE,
                start_label=f"Start: {start_city}",
                end_label=f"End: {end_city}",
            )
            full_layer.add_to(map_object)
            all_layers.append(full_layer)
            if not default_visible:
                hidden_layers.append(full_layer)
            layer_var = full_layer.get_name()
            layer_var_updates.setdefault(route_key, {})["route_layer_var"] = layer_var
            if route_debug_registry is not None:
                route_debug_registry.append(
                    {
                        "label": control_label,
                        "control_label": control_label,
                        "control_id": f"route_matrix_full__{route_key}",
                        "family": "route_matrix",
                        "selector_group": "route_matrix",
                        "route_key": route_key,
                        "parent_route_key": route_key,
                        "layer_role": "full_route",
                        "trip_days": trip_days,
                        "method": method,
                        "method_label": method_label,
                        "profile": profile,
                        "profile_label": profile_label,
                        "selector_parent": selector_parent,
                        "color": color,
                        "pane": ROUTE_TOP_PANE,
                        "layer_var": layer_var,
                        "offset_index": offset_index,
                        "default_checked": default_visible,
                        "default_visible": default_visible,
                        "quick_groups": ["route_matrix", f"days_{trip_days}", method, profile],
                        "distance_km": round(float(_route_distance_km(_dedupe_route_points(draw_points))), 3),
                        "unique_points": len(_dedupe_route_points(draw_points)),
                        "bounds": _route_bounds(draw_points),
                        "geometry_source": route_mode,
                        "start_city": start_city,
                        "end_city": end_city,
                    }
                )
            _append_route_debug_row(
                route_debug_rows,
                layer_group="Route Matrix Full Route",
                layer_name=full_layer_name,
                comparison_type="route_matrix",
                profile=profile,
                method=method,
                trip_days=trip_days,
                route_type="matrix_full_route",
                route_start_city=start_city,
                route_end_city=end_city,
                points=draw_points,
                geometry_mode=f"{route_mode}; {geometry_mode}; route_matrix_full_route",
                show_by_default=default_visible,
                source_rows=len(group),
                layer_var=layer_var,
                notes=f"route_key={route_key}; hotel_showcase_controls_hotels=true; geometry_source={route_mode}",
            )

        hotel_layer = folium.FeatureGroup(name=f"Hotels · {control_label}", show=False)
        hotel_rows = (
            group[["overnight_city", "hotel_name", "hotel_latitude", "hotel_longitude"]]
            .dropna()
            .drop_duplicates()
            .reset_index(drop=True)
        )
        for hotel_row in hotel_rows.itertuples(index=False):
            hotel_point = [float(hotel_row.hotel_latitude), float(hotel_row.hotel_longitude)]
            popup = (
                f"<b>{_escape(hotel_row.hotel_name)}</b><br/>"
                f"Route: {_escape(control_label)}<br/>"
                f"Overnight city: {_escape(hotel_row.overnight_city)}<br/>"
                f"Route key: {_escape(route_key)}"
            )
            folium.Marker(
                location=hotel_point,
                icon=folium.Icon(color="darkred", icon="hotel", prefix="fa"),
                tooltip=f"{control_label} hotel/base: {hotel_row.hotel_name}",
                popup=folium.Popup(popup, max_width=280, min_width=190),
            ).add_to(hotel_layer)
            folium.CircleMarker(
                location=hotel_point,
                radius=8,
                color=color,
                fill=True,
                fillColor=color,
                fillOpacity=0.78,
                weight=3,
                tooltip=f"{control_label} hotel/base: {hotel_row.hotel_name}",
                popup=folium.Popup(popup, max_width=280, min_width=190),
            ).add_to(hotel_layer)
        hotel_layer.add_to(map_object)
        all_layers.append(hotel_layer)
        hidden_layers.append(hotel_layer)
        layer_var_updates.setdefault(route_key, {})["hotel_layer_var"] = hotel_layer.get_name()
        if route_debug_registry is not None:
            route_debug_registry.append(
                {
                    "label": f"Hotels · {control_label}",
                    "control_label": "Hotels",
                    "control_id": f"route_matrix_hotels__{route_key}",
                    "family": "hotel",
                    "selector_group": "route_matrix",
                    "route_key": route_key,
                    "parent_route_key": route_key,
                    "layer_role": "selected_hotels",
                    "hotel_showcase_controlled": True,
                    "trip_days": trip_days,
                    "method": method,
                    "method_label": method_label,
                    "profile": profile,
                    "profile_label": profile_label,
                    "selector_parent": selector_parent,
                    "color": color,
                    "pane": "markerPane",
                    "layer_var": hotel_layer.get_name(),
                    "offset_index": 0,
                    "default_checked": False,
                    "default_visible": False,
                    "quick_groups": ["hotel_showcase"],
                    "distance_km": 0.0,
                    "unique_points": len(hotel_rows),
                    "bounds": _route_bounds(
                        hotel_rows[["hotel_latitude", "hotel_longitude"]].astype(float).values.tolist()
                    )
                    if not hotel_rows.empty
                    else [],
                }
            )
        _append_route_debug_row(
            route_debug_rows,
            layer_group="Route Matrix Hotels",
            layer_name=f"Hotels · {control_label}",
            comparison_type="route_matrix_hotels",
            profile=profile,
            method=method,
            trip_days=trip_days,
            route_type="route_specific_selected_hotels",
            points=hotel_rows[["hotel_latitude", "hotel_longitude"]].astype(float).values.tolist()
            if not hotel_rows.empty
            else [],
            geometry_mode="route-specific-hotel-markers",
            show_by_default=False,
            source_rows=len(hotel_rows),
            layer_var=hotel_layer.get_name(),
            notes=f"route_key={route_key}; hotel_showcase_controlled=true",
        )

        city_layer = folium.FeatureGroup(name=f"City loops · {control_label}", show=False)
        city_layer_points = []
        city_loop_days = 0
        for day, day_group in group.groupby("day", sort=True):
            loop_points, local_rows = _hotel_loop_points_from_day_group(day_group)
            if not loop_points or local_rows.empty:
                continue
            city_loop_days += 1
            city_name = str(day_group.iloc[0].get("overnight_city", day_group.iloc[0].get("city", "")))
            day_path, day_route_mode = _add_flow_route(
                city_layer,
                loop_points,
                color=color,
                route_cache=route_cache,
                run_live=run_live,
                tooltip=f"{control_label} · Day {int(day)} · {city_name} hotel loop",
                popup_html=(
                    f"<b>{_escape(control_label)}</b><br/>"
                    f"Day {int(day)} city loop in {_escape(city_name)}<br/>"
                    f"Starts and ends at selected hotel/base."
                ),
                dash_array=None,
                opacity=0.78,
                weight=3,
                ant_delay=820,
                pane=ROUTE_TOP_PANE,
            )
            city_layer_points.extend(day_path or loop_points)
            for row in local_rows.itertuples(index=False):
                point = [float(row.latitude), float(row.longitude)]
                popup = (
                    f"<b>{_escape(row.attraction_name)}</b><br/>"
                    f"{_escape(control_label)}<br/>"
                    f"Day {int(row.day)}, stop {int(row.stop_order)}<br/>"
                    f"City: {_escape(row.city)}"
                )
                folium.CircleMarker(
                    location=point,
                    radius=6,
                    color=color,
                    fill=True,
                    fillColor=color,
                    fillOpacity=0.82,
                    weight=2,
                    tooltip=f"{control_label} Day {int(row.day)} stop {int(row.stop_order)}",
                    popup=folium.Popup(popup, max_width=260),
                ).add_to(city_layer)
            _append_route_debug_row(
                route_debug_rows,
                layer_group="Route Matrix City Loop",
                layer_name=f"{control_label} · Day {int(day)} · {city_name}",
                comparison_type="route_matrix_city_loop",
                profile=profile,
                method=method,
                trip_days=trip_days,
                day=int(day),
                route_type="hotel_loop",
                route_start_city=city_name,
                route_end_city=city_name,
                points=loop_points,
                geometry_mode=f"{day_route_mode}; route-matrix-city-loop-arrows",
                show_by_default=False,
                source_rows=len(local_rows),
                layer_var=city_layer.get_name(),
                notes=f"route_key={route_key}; hotel_loop_start_end_same=true",
            )
        city_layer.add_to(map_object)
        all_layers.append(city_layer)
        hidden_layers.append(city_layer)
        layer_var_updates.setdefault(route_key, {})["city_detail_layer_vars"] = city_layer.get_name()
        if route_debug_registry is not None:
            route_debug_registry.append(
                {
                    "label": f"City loops · {control_label}",
                    "control_label": "City loops",
                    "control_id": f"route_matrix_city_loops__{route_key}",
                    "family": "city_detail",
                    "selector_group": "route_matrix",
                    "route_key": route_key,
                    "parent_route_key": route_key,
                    "layer_role": "city_loop",
                    "trip_days": trip_days,
                    "method": method,
                    "method_label": method_label,
                    "profile": profile,
                    "profile_label": profile_label,
                    "selector_parent": selector_parent,
                    "color": color,
                    "pane": ROUTE_TOP_PANE,
                    "layer_var": city_layer.get_name(),
                    "offset_index": 0,
                    "default_checked": False,
                    "default_visible": False,
                    "quick_groups": ["city_detail", f"days_{trip_days}", method, profile],
                    "distance_km": round(float(_route_distance_km(_dedupe_route_points(city_layer_points))), 3),
                    "unique_points": len(_dedupe_route_points(city_layer_points)),
                    "bounds": _route_bounds(city_layer_points),
                    "child_count": city_loop_days,
                }
            )

        transition_layer = folium.FeatureGroup(name=f"Transitions · {control_label}", show=False)
        transition_points_all = []
        transition_days = 0
        for day, day_group in group.groupby("day", sort=True):
            transition_points, transition_rows = _transition_points_from_day_group(day_group)
            if not transition_points:
                continue
            transition_days += 1
            start_city = str(day_group.iloc[0].get("route_start_city", ""))
            end_city = str(day_group.iloc[0].get("route_end_city", day_group.iloc[0].get("overnight_city", "")))
            transition_path, transition_mode = _add_flow_route(
                transition_layer,
                transition_points,
                color=color,
                route_cache=route_cache,
                run_live=run_live,
                tooltip=f"{control_label} · Day {int(day)} transition",
                popup_html=f"<b>{_escape(control_label)}</b><br/>Day {int(day)} transition: {_escape(start_city)} → {_escape(end_city)}",
                dash_array="8 10",
                opacity=0.72,
                weight=3,
                ant_delay=860,
                pane=ROUTE_TOP_PANE,
            )
            transition_points_all.extend(transition_path or transition_points)
            _append_route_debug_row(
                route_debug_rows,
                layer_group="Route Matrix Transition",
                layer_name=f"{control_label} · Day {int(day)} · {start_city} to {end_city}",
                comparison_type="route_matrix_transition",
                profile=profile,
                method=method,
                trip_days=trip_days,
                day=int(day),
                route_type=str(day_group.iloc[0].get("route_type", "relocation")),
                route_start_city=start_city,
                route_end_city=end_city,
                pass_through_cities=str(day_group.iloc[0].get("pass_through_cities", "")),
                points=transition_points,
                geometry_mode=f"{transition_mode}; route-matrix-transition-arrows",
                show_by_default=False,
                source_rows=max(1, len(transition_rows)),
                layer_var=transition_layer.get_name(),
                notes=f"route_key={route_key}; transition_route_separate_from_city_loop=true",
            )
        transition_layer.add_to(map_object)
        all_layers.append(transition_layer)
        hidden_layers.append(transition_layer)
        layer_var_updates.setdefault(route_key, {})["transition_layer_vars"] = transition_layer.get_name()
        if route_debug_registry is not None:
            route_debug_registry.append(
                {
                    "label": f"Transitions · {control_label}",
                    "control_label": "Transitions",
                    "control_id": f"route_matrix_transitions__{route_key}",
                    "family": "transition",
                    "selector_group": "route_matrix",
                    "route_key": route_key,
                    "parent_route_key": route_key,
                    "layer_role": "transition",
                    "trip_days": trip_days,
                    "method": method,
                    "method_label": method_label,
                    "profile": profile,
                    "profile_label": profile_label,
                    "selector_parent": selector_parent,
                    "color": color,
                    "pane": ROUTE_TOP_PANE,
                    "layer_var": transition_layer.get_name(),
                    "offset_index": 0,
                    "default_checked": False,
                    "default_visible": False,
                    "quick_groups": ["transition", f"days_{trip_days}", method, profile],
                    "distance_km": round(float(_route_distance_km(_dedupe_route_points(transition_points_all))), 3),
                    "unique_points": len(_dedupe_route_points(transition_points_all)),
                    "bounds": _route_bounds(transition_points_all),
                    "child_count": transition_days,
                }
            )

        if not social_catalog.empty:
            must_layer = folium.FeatureGroup(name=f"Must-go · {control_label}", show=False)
            selected_names = set(group.get("attraction_name", pd.Series(dtype=str)).dropna().astype(str).tolist())
            route_cities = set(group.get("city", pd.Series(dtype=str)).dropna().astype(str).tolist()) | set(
                group.get("overnight_city", pd.Series(dtype=str)).dropna().astype(str).tolist()
            )
            route_social = social_catalog[
                social_catalog.get("city", pd.Series(dtype=str)).astype(str).isin(route_cities)
                | social_catalog.get("name", pd.Series(dtype=str)).astype(str).isin(selected_names)
                | social_catalog.get("corridor_fit", pd.Series(0.0, index=social_catalog.index)).astype(float).gt(0.70)
            ].copy()
            for social in route_social.itertuples(index=False):
                point = [float(social.latitude), float(social.longitude)]
                selected = str(social.name) in selected_names
                state = "selected" if selected else "skipped"
                folium.Marker(
                    location=point,
                    tooltip=f"{control_label} must-go {state}: {social.name}",
                    popup=folium.Popup(
                        f"<b>{_escape(social.name)}</b><br/>Route: {_escape(control_label)}<br/>State: {_escape(state)}",
                        max_width=280,
                    ),
                    icon=folium.Icon(color="green" if selected else "orange", icon="star", prefix="fa"),
                ).add_to(must_layer)
            must_layer.add_to(map_object)
            all_layers.append(must_layer)
            hidden_layers.append(must_layer)
            if route_debug_registry is not None:
                route_debug_registry.append(
                    {
                        "label": f"Must-go · {control_label}",
                        "control_label": "Must-go coverage",
                        "control_id": f"route_matrix_must_go__{route_key}",
                        "family": "must_go",
                        "selector_group": "route_matrix",
                        "route_key": route_key,
                        "parent_route_key": route_key,
                        "layer_role": "must_go",
                        "trip_days": trip_days,
                        "method": method,
                        "method_label": method_label,
                        "profile": profile,
                        "profile_label": profile_label,
                        "selector_parent": selector_parent,
                        "color": "#F4A261",
                        "pane": "markerPane",
                        "layer_var": must_layer.get_name(),
                        "offset_index": 0,
                        "default_checked": False,
                        "default_visible": False,
                        "quick_groups": ["must_go", f"days_{trip_days}", method, profile],
                        "distance_km": 0.0,
                        "unique_points": int(len(route_social)),
                        "bounds": _route_bounds(
                            route_social[["latitude", "longitude"]].dropna().astype(float).values.tolist()
                        )
                        if not route_social.empty
                        else [],
                    }
                )

    matrix_summary_path = Path(output_dir) / "production_route_matrix_comparison.csv"
    if layer_var_updates and matrix_summary_path.exists():
        try:
            matrix_summary = _load_csv(matrix_summary_path)
            if not matrix_summary.empty and "route_key" in matrix_summary.columns:
                for column in ["route_layer_var", "hotel_layer_var", "city_detail_layer_vars", "transition_layer_vars"]:
                    if column not in matrix_summary.columns:
                        matrix_summary[column] = ""
                    matrix_summary[column] = matrix_summary[column].fillna("").astype("object")
                for route_key, updates in layer_var_updates.items():
                    mask = matrix_summary["route_key"].astype(str).eq(str(route_key))
                    for column, value in updates.items():
                        matrix_summary.loc[mask, column] = str(value)
                matrix_summary.to_csv(matrix_summary_path, index=False)
        except Exception:
            pass
    return all_layers, hidden_layers



def _add_full_scene_overview_layer(
    map_object,
    *,
    output_dir,
    fastest_path,
    scenic_path,
    profile_day_plans,
    route_cache,
    run_live,
    route_debug_rows=None,
    route_debug_registry=None,
):
    """Create individually selectable full-route overview layers."""
    route_layers = []
    route_specs = []

    comparison_files = [
        Path(output_dir) / "production_trip_length_route_stops.csv",
        Path(output_dir) / "production_method_route_stops.csv",
    ]
    comparison_frames = [frame for frame in (_load_csv(path) for path in comparison_files) if not frame.empty]
    if comparison_frames:
        comparison_df = pd.concat(comparison_frames, ignore_index=True, sort=False)
        for group_values, group in comparison_df.groupby(
            ["comparison_type", "comparison_label", "method", "trip_days"], dropna=False
        ):
            comparison_type, comparison_label, method, trip_days = group_values
            start_city, end_city = _route_endpoint_city_labels(group)
            if str(comparison_type) == "trip_length":
                trip_day_value = int(trip_days)
                color = {7: "#0057FF", 9: "#FF6B00", 12: "#7C2DFF"}.get(trip_day_value, "#0057FF")
                offset_index = {7: -1.2, 9: 0.0, 12: 1.2}.get(trip_day_value, 0.0)
                label = f"{trip_day_value}-day hybrid"
                dash_array = None
                control_label = f"Trip Length · {trip_day_value}-Day Hybrid Bandit + Small Gurobi"
                control_id = f"trip_length_{trip_day_value}"
            elif str(method) == "hierarchical_gurobi_pipeline":
                color = "#0096C7"
                label = "Method: Gurobi"
                offset_index = -0.8
                dash_array = None
                control_label = "Method · Hierarchical Gurobi Pipeline"
                control_id = "method_gurobi"
            elif str(method) == "hierarchical_greedy_baseline":
                color = "#111827"
                label = "Method: Greedy"
                offset_index = 0
                dash_array = "7 11"
                control_label = "Method · Hierarchical Greedy Baseline"
                control_id = "method_greedy"
            else:
                color = "#00A63E"
                label = "Method: Bandit repair"
                offset_index = 0.8
                dash_array = None
                control_label = "Method · Hierarchical + Bandit + Small Gurobi Repair"
                control_id = "method_bandit_repair"
            route_specs.append(
                {
                    "label": label,
                    "control_label": control_label,
                    "control_id": control_id,
                    "comparison_type": str(comparison_type),
                    "points": _full_route_points_from_plan(group),
                    "color": color,
                    "dash_array": dash_array,
                    "weight": 6 if str(comparison_type) == "trip_length" else 5,
                    "opacity": 0.90,
                    "offset_index": offset_index,
                    "pane": ROUTE_TOP_PANE,
                    "default_visible": False,
                    "quick_groups": ["trip_length"] if str(comparison_type) == "trip_length" else ["method"],
                    "start_city": start_city,
                    "end_city": end_city,
                }
            )

    traveler_styles = {
        "relaxed": ("Traveler: Relaxed", "#00897B", "8 12", -0.7, False),
        "balanced": ("Balanced full route", "#E91E63", None, 0.0, True),
        "explorer": ("Traveler: Explorer", "#A020F0", "2 9", 0.7, False),
    }
    for profile_name, plan_df in profile_day_plans.items():
        start_city, end_city = _route_endpoint_city_labels(plan_df)
        label, color, dash_array, offset_index, default_visible = traveler_styles.get(
            profile_name,
            (f"Traveler: {profile_name}", "#2563EB", None, 0.0, False),
        )
        route_specs.append(
            {
                "label": label,
                "control_label": label,
                "control_id": f"traveler_{profile_name}",
                "comparison_type": "traveler_profile",
                "points": _full_route_points_from_plan(plan_df),
                "color": color,
                "dash_array": dash_array,
                "weight": 7 if default_visible else 5,
                "opacity": 0.94 if default_visible else 0.88,
                "offset_index": offset_index,
                "pane": ROUTE_TOP_PANE,
                "default_visible": default_visible,
                "quick_groups": ["traveler"],
                "start_city": start_city,
                "end_city": end_city,
            }
        )

    for spec in route_specs:
        label = spec["label"]
        comparison_type = spec["comparison_type"]
        points = spec["points"]
        if not points or len(points) < 2:
            continue
        road_points, route_mode = _fetch_stitched_route(points, route_cache, run_live=run_live)
        offset_index = float(spec["offset_index"])
        draw_points = _offset_route_points_for_visibility(road_points, offset_index)
        layer_name = f"Full Route · {spec['control_label']}"
        default_visible = bool(spec.get("default_visible", False))
        layer = folium.FeatureGroup(name=layer_name, show=default_visible)
        geometry_mode = _add_full_scene_route_band(
            layer,
            draw_points,
            color=spec["color"],
            dash_array=spec["dash_array"],
            tooltip=f"Full route · {label}",
            popup_html=(
                f"<b>Full route overview</b><br/>"
                f"{_escape(label)}<br/>"
                f"Start: {_escape(spec['start_city'])}<br/>"
                f"End: {_escape(spec['end_city'])}<br/>"
                f"Checkbox default: {_escape('checked' if default_visible else 'unchecked')}<br/>"
                f"Diagnostic offset band: {offset_index:.1f}"
            ),
            label=label,
            weight=spec["weight"],
            opacity=spec["opacity"],
            pane=spec["pane"],
            start_label=f"Start: {spec['start_city']}",
            end_label=f"End: {spec['end_city']}",
        )
        layer.add_to(map_object)
        bounds = _route_bounds(draw_points)
        layer_var = layer.get_name()
        if route_debug_registry is not None:
            route_debug_registry.append(
                {
                    "label": label,
                    "control_label": spec["control_label"],
                    "control_id": spec["control_id"],
                    "family": comparison_type,
                    "selector_group": "core" if spec["control_id"] == "traveler_balanced" else comparison_type,
                    "color": spec["color"],
                    "pane": spec["pane"],
                    "layer_var": layer_var,
                    "offset_index": offset_index,
                    "default_checked": default_visible,
                    "default_visible": default_visible,
                    "quick_groups": spec["quick_groups"],
                    "distance_km": round(float(_route_distance_km(_dedupe_route_points(draw_points))), 3),
                    "unique_points": len(_dedupe_route_points(draw_points)),
                    "bounds": bounds,
                    "geometry_source": route_mode,
                    "start_city": spec["start_city"],
                    "end_city": spec["end_city"],
                }
            )
        _append_route_debug_row(
            route_debug_rows,
            layer_group="Full Route Overview",
            layer_name=layer_name,
            comparison_type=comparison_type,
            points=draw_points,
            geometry_mode=f"{route_mode}; {geometry_mode}; selectable-full-route",
            show_by_default=default_visible,
            source_rows=len(draw_points),
            route_type="selectable_full_route_overview",
            route_start_city=spec["start_city"],
            route_end_city=spec["end_city"],
            layer_var=layer_var,
            notes=f"offset_index={offset_index:.1f}; original_points={len(_dedupe_route_points(points))}; road_points={len(_dedupe_route_points(road_points))}; pane={spec['pane']}; direction_arrows=canvas; no_white_casing=True; geometry_source={route_mode}",
        )
        route_layers.append(layer)

    return route_layers
