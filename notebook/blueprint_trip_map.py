import ast
import html
import json
import math
import sys
from pathlib import Path

import folium
import numpy as np
import pandas as pd
import requests
from folium import plugins

try:
    from itinerary_system.nature_catalog import NATURE_POI_COLUMNS
    from itinerary_system.region_scenarios import all_scenario_coordinates
except Exception:  # pragma: no cover - keeps notebook execution standalone.
    NATURE_POI_COLUMNS = [
        "is_nature",
        "is_national_park",
        "is_state_park",
        "is_protected_area",
        "is_scenic_viewpoint",
        "is_hiking",
        "nature_score",
        "city_score",
        "culture_score",
        "history_score",
        "scenic_score",
        "hiking_score",
        "outdoor_intensity",
        "weather_sensitivity",
        "seasonality_risk",
        "park_type",
        "nature_region",
        "interest_fit",
        "park_bonus",
        "interest_adjusted_value",
        "interest_delta",
        "reason_selected",
    ]

    def all_scenario_coordinates():
        return {}



_PROJECT_SRC = Path(__file__).resolve().parents[1] / "src"
if _PROJECT_SRC.exists() and str(_PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(_PROJECT_SRC))

from itinerary_system.blueprint_core import (  # noqa: E402
    CALIFORNIA_ROAD_CORRIDOR,
    CITY_COORDS,
    CORRIDOR_INDEX,
    OPTIONAL_BOOL_POI_COLUMNS,
    OPTIONAL_POI_COLUMNS,
    OPTIONAL_TEXT_POI_COLUMNS,
    PROFILE_CONFIGS,
    SOCIAL_MUST_GO_PLACES,
    STANFORD_WAYPOINT_NAME,
    _catalog_series,
    _city_hotel_catalog,
    _city_poi_catalog,
    _coerce_days_by_city,
    _coerce_name_list,
    _coerce_optional_poi_columns,
    _coord_for_place,
    _corridor_segment_points,
    _distance_to_city_center,
    _estimate_drive_minutes,
    _insert_stanford_waypoint,
    _leg_classification,
    _load_csv,
    _normalize_catalog_columns,
    _numeric_catalog_series,
    _order_stops_from_hotel,
    _point_distance_km,
    _profile_config,
    _route_distance_km,
    _scenic_leg_waypoints,
    _scenic_route_sequence,
    _select_compact_day_stops,
    _select_hotel_for_city_plan,
    _sequence_between,
    _social_must_go_catalog,
    _stanford_transition_stop,
    _transition_drive_minutes,
    _trip_sequence_with_pass_through,
    _unique_in_order,
)

from itinerary_system.blueprint_day_plans import (  # noqa: E402
    _optional_poi_fields,
    _select_corridor_view_stops,
    _select_pass_through_day_stops,
    build_day_plan,
    build_profile_day_plans,
)
from itinerary_system.blueprint_render_primitives import (  # noqa: E402
    DAY_COLORS,
    FASTEST_ROUTE_COLOR,
    FULL_SCENE_OFFSET_LAT_STEP,
    FULL_SCENE_OFFSET_LON_STEP,
    MATRIX_DEFAULT_ROUTE_KEY,
    MATRIX_METHOD_STYLES,
    MATRIX_PROFILE_STYLE_OFFSETS,
    MATRIX_ROUTE_OFFSETS_ENABLED,
    ROUTE_CONTEXT_PANE,
    ROUTE_CORE_PANE,
    ROUTE_TOP_PANE,
    SCENIC_CA1_COLOR,
    _add_canvas_direction_arrows,
    _add_full_scene_route_band,
    _add_route_panes,
    _add_static_result_line,
    _append_route_debug_row,
    _bearing_degrees,
    _curated_corridor_fallback,
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
    _nearest_corridor_anchor,
    _offset_route_points_for_visibility,
    _route_arrow_samples,
    _route_bounds,
    _route_cache_key,
    _route_debug_value,
    _route_endpoint_city_labels,
    _route_midpoint,
    _route_mode_label,
    _route_points_for_names,
    _route_points_from_day_group,
    _transition_points_from_day_group,
)
from itinerary_system.blueprint_render_layers import (  # noqa: E402
    _add_city_detail_layers,
    _add_flow_route,
    _add_full_scene_overview_layer,
    _add_intercity_route_layer,
    _add_model_comparison_layers,
    _add_route_matrix_layers,
    _add_selected_result_layer,
    _add_traveler_overview_layers,
)
from itinerary_system.blueprint_render_panels import (  # noqa: E402
    _add_blueprint_panel_behavior,
    _add_interest_bar_preview_panel,
    _add_route_debug_controls,
    _build_day_panel,
    _build_hotel_summary_html,
    _build_html_data_source_summary,
    _build_method_comparison,
    _build_method_comparison_html,
    _build_must_go_summary_html,
    _build_profile_comparison,
    _build_route_debug_summary_html,
    _build_trip_length_comparison_html,
    _load_default_hierarchical_gurobi_day_plan,
)
from itinerary_system.blueprint_renderer import build_production_trip_map  # noqa: E402










def _first_existing(context, *keys, default=None):
    for key in keys:
        if key in context:
            return context[key]
    return default


















































def _select_greedy_day_stops(poi_catalog, hotel, used_names, stops_per_day=3):
    if poi_catalog.empty:
        return poi_catalog

    available = poi_catalog[~poi_catalog["name"].astype(str).isin(used_names)].copy()
    if available.empty:
        available = poi_catalog.copy()

    available["latitude"] = pd.to_numeric(available["latitude"], errors="coerce")
    available["longitude"] = pd.to_numeric(available["longitude"], errors="coerce")
    available["final_poi_value"] = pd.to_numeric(
        available.get("final_poi_value", available.get("source_score", 0.0)),
        errors="coerce",
    ).fillna(0.0)
    available["source_score"] = pd.to_numeric(_catalog_series(available, "source_score", 0.0), errors="coerce").fillna(
        0.0
    )
    available["social_score"] = pd.to_numeric(_catalog_series(available, "social_score", 0.0), errors="coerce").fillna(
        0.0
    )
    available["social_must_go"] = available.get("social_must_go", False)
    available["social_must_go"] = available["social_must_go"].fillna(False).astype(bool)
    available["must_go_weight"] = pd.to_numeric(
        _catalog_series(available, "must_go_weight", 0.0), errors="coerce"
    ).fillna(0.0)
    available["corridor_fit"] = pd.to_numeric(
        _catalog_series(available, "corridor_fit", 0.0), errors="coerce"
    ).fillna(0.0)
    available["route_fit"] = pd.to_numeric(_catalog_series(available, "route_fit", 0.0), errors="coerce").fillna(0.0)
    available["route_context_fit"] = np.maximum(available["corridor_fit"], available["route_fit"])
    available["detour_minutes"] = pd.to_numeric(
        _catalog_series(available, "detour_minutes", 0.0), errors="coerce"
    ).fillna(0.0)
    available = available.dropna(subset=["latitude", "longitude"]).reset_index(drop=True)
    if available.empty:
        return poi_catalog.head(stops_per_day)

    hotel_lat = float(hotel.get("latitude"))
    hotel_lon = float(hotel.get("longitude"))
    current_lat = hotel_lat
    current_lon = hotel_lon
    selected_rows = []
    remaining = available.copy()

    while len(selected_rows) < stops_per_day and not remaining.empty:
        remaining["greedy_score"] = remaining.apply(
            lambda row: (
                float(row["final_poi_value"])
                + 0.12 * float(row["social_score"])
                + 0.08 * float(row["route_context_fit"])
                - 0.02 * _point_distance_km(current_lat, current_lon, row["latitude"], row["longitude"])
                - 0.004 * float(row["detour_minutes"])
            ),
            axis=1,
        )
        next_row = remaining.sort_values(
            ["greedy_score", "final_poi_value", "source_score", "name"],
            ascending=[False, False, False, True],
        ).iloc[0]
        selected_rows.append(next_row)
        current_lat = float(next_row["latitude"])
        current_lon = float(next_row["longitude"])
        remaining = remaining[remaining["name"].astype(str) != str(next_row["name"])].reset_index(drop=True)

    selected = pd.DataFrame(selected_rows).reset_index(drop=True) if selected_rows else available.head(stops_per_day)
    output_columns = [
        "city",
        "name",
        "category",
        "latitude",
        "longitude",
        "source",
        "source_score",
        "source_list",
        "social_score",
        "social_must_go",
        "must_go_weight",
        "corridor_fit",
        "detour_minutes",
        "data_confidence",
        "final_poi_value",
        "social_reason",
        *OPTIONAL_POI_COLUMNS,
    ]
    for column in output_columns:
        if column not in selected.columns:
            if column == "social_reason":
                selected[column] = ""
            elif column == "source_list":
                selected[column] = selected.get("source", "unknown")
            else:
                selected[column] = 0.0
    selected = _coerce_optional_poi_columns(selected)
    return selected[output_columns]


def _select_greedy_pass_through_day_stops(context, pass_through_cities, used_names_by_city, max_stops):
    selected_rows = []
    if max_stops <= 0:
        return pd.DataFrame()
    for city in pass_through_cities:
        if len(selected_rows) >= max_stops:
            break
        catalog = _city_poi_catalog(context, city).copy()
        if catalog.empty:
            continue
        used_names = used_names_by_city.setdefault(city, set())
        catalog = catalog[~catalog["name"].astype(str).isin(used_names)].copy()
        if catalog.empty:
            continue
        catalog["final_poi_value"] = _numeric_catalog_series(catalog, "final_poi_value", 0.0)
        catalog["source_score"] = _numeric_catalog_series(catalog, "source_score", 0.0)
        stop = catalog.sort_values(["final_poi_value", "source_score", "name"], ascending=[False, False, True]).iloc[0]
        selected_rows.append(stop)
        used_names.add(str(stop["name"]))
    if not selected_rows:
        return pd.DataFrame()
    return pd.DataFrame(selected_rows).reset_index(drop=True)


def build_greedy_baseline_day_plan(context, stops_per_day=None, profile_name="balanced"):
    profile_config = _profile_config(profile_name, stops_per_day=stops_per_day)
    stops_per_day = int(profile_config["stops_per_day"])
    trip = context["best_hierarchical_trip"]
    sequence = _trip_sequence_with_pass_through(trip)
    days_by_city = _coerce_days_by_city(trip["days_by_city"])
    total_trip_days = int(sum(int(value) for value in days_by_city.values()))
    base_cities = [city for city in sequence if int(days_by_city.get(city, 0)) > 0]

    rows = []
    day_number = 1
    used_names_by_city = {city: set() for city in sequence}
    selected_hotels_by_city = {}

    for base_index, city in enumerate(base_cities):
        allocated_days = int(days_by_city.get(city, 0))
        if allocated_days <= 0:
            continue

        poi_catalog = _city_poi_catalog(context, city).reset_index(drop=True)
        hotel_catalog = _city_hotel_catalog(context, city).reset_index(drop=True)
        used_names = used_names_by_city.setdefault(city, set())
        hotel = _select_hotel_for_city_plan(
            hotel_catalog,
            poi_catalog,
            used_names,
            stops_per_day=stops_per_day,
            profile_config=profile_config,
        )
        selected_hotels_by_city[city] = hotel

        for local_day in range(1, allocated_days + 1):
            is_transition_day = bool(local_day == 1 and base_index > 0)
            previous_city = base_cities[base_index - 1] if is_transition_day else city
            previous_hotel = selected_hotels_by_city.get(previous_city, hotel)
            route_start_name = previous_hotel.get("name", f"{previous_city} hotel")
            route_start_latitude = float(previous_hotel.get("latitude", CITY_COORDS[previous_city][0]))
            route_start_longitude = float(previous_hotel.get("longitude", CITY_COORDS[previous_city][1]))
            route_end_name = hotel.get("name", f"{city} hotel")
            route_end_latitude = float(hotel.get("latitude", CITY_COORDS[city][0]))
            route_end_longitude = float(hotel.get("longitude", CITY_COORDS[city][1]))
            route_type = "base_city_local"
            pass_through_cities = []
            drive_minutes = 0.0
            drive_time_source = "local_day_no_intercity_drive"

            if is_transition_day:
                segment = _sequence_between(sequence, previous_city, city)
                pass_through_cities = [stop_city for stop_city in segment[1:-1] if stop_city != city]
                drive_minutes, drive_time_source = _transition_drive_minutes(
                    context, previous_city, city, pass_through_cities
                )
                route_type = "relocation_pass_through" if pass_through_cities else "relocation_direct"

            available_visit_minutes = max(60.0, 720.0 - float(drive_minutes) - 45.0)
            if is_transition_day:
                transition_stops = _select_greedy_pass_through_day_stops(
                    context,
                    pass_through_cities,
                    used_names_by_city,
                    min(stops_per_day, max(1, int(available_visit_minutes // 30.0))),
                )
                remaining_slots = max(0, stops_per_day - len(transition_stops))
                if remaining_slots > 0:
                    local_fill = _select_greedy_day_stops(poi_catalog, hotel, used_names, stops_per_day=remaining_slots)
                    if not local_fill.empty:
                        used_names.update(local_fill["name"].astype(str).tolist())
                        transition_stops = pd.concat([transition_stops, local_fill], ignore_index=True, sort=False)
                stops = (
                    transition_stops
                    if not transition_stops.empty
                    else _select_greedy_day_stops(
                        poi_catalog,
                        hotel,
                        used_names,
                        stops_per_day=max(1, min(stops_per_day, int(available_visit_minutes // 75.0) or 1)),
                    )
                )
            else:
                stops = _select_greedy_day_stops(poi_catalog, hotel, used_names, stops_per_day=stops_per_day)
            if stops.empty:
                stops = poi_catalog.head(1)
            used_names.update(stops["name"].astype(str).tolist())

            for stop_order, (_, stop) in enumerate(stops.iterrows(), start=1):
                stop_city = str(stop.get("city", city))
                rows.append(
                    {
                        "profile": str(profile_name).lower(),
                        "profile_label": f"{profile_config['label']} Greedy Baseline",
                        "day": day_number,
                        "city": stop_city,
                        "overnight_city": city,
                        "city_day": local_day,
                        "hotel_name": hotel.get("name", f"{city} hotel"),
                        "hotel_latitude": float(hotel.get("latitude", CITY_COORDS[city][0])),
                        "hotel_longitude": float(hotel.get("longitude", CITY_COORDS[city][1])),
                        "hotel_source": hotel.get("source", "unknown"),
                        "overnight_base": bool(stop_city == city),
                        "stop_is_overnight_city": bool(stop_city == city),
                        "hotel_booked": bool(day_number < total_trip_days),
                        "route_type": route_type,
                        "route_start_city": previous_city,
                        "route_start_name": route_start_name,
                        "route_start_latitude": route_start_latitude,
                        "route_start_longitude": route_start_longitude,
                        "route_end_city": city,
                        "route_end_name": route_end_name,
                        "route_end_latitude": route_end_latitude,
                        "route_end_longitude": route_end_longitude,
                        "pass_through_cities": "; ".join(pass_through_cities),
                        "drive_minutes_to_next_base": round(float(drive_minutes), 2),
                        "available_visit_minutes": round(float(available_visit_minutes), 2),
                        "drive_time_source": drive_time_source,
                        "stop_order": stop_order,
                        "attraction_name": stop["name"],
                        "category": stop.get("category", "attraction"),
                        "latitude": float(stop["latitude"]),
                        "longitude": float(stop["longitude"]),
                        "attraction_source": stop.get("source", "unknown"),
                        "source_list": stop.get("source_list", stop.get("source", "unknown")),
                        "social_score": float(stop.get("social_score", 0.0) or 0.0),
                        "social_must_go": bool(stop.get("social_must_go", False)),
                        "must_go_weight": float(stop.get("must_go_weight", 0.0) or 0.0),
                        "corridor_fit": float(stop.get("corridor_fit", 0.0) or 0.0),
                        "detour_minutes": float(stop.get("detour_minutes", 0.0) or 0.0),
                        "data_confidence": float(stop.get("data_confidence", 0.5) or 0.5),
                        "final_poi_value": float(stop.get("final_poi_value", stop.get("source_score", 0.0)) or 0.0),
                        "social_reason": stop.get("social_reason", ""),
                        **_optional_poi_fields(stop),
                    }
                )
            day_number += 1

    return pd.DataFrame(rows)
























def _california_corridor_path(city_sequence):
    city_sequence = [city for city in city_sequence if city in CORRIDOR_INDEX]
    if len(city_sequence) < 2:
        return []

    path = []
    for left_city, right_city in zip(city_sequence[:-1], city_sequence[1:]):
        left_idx = CORRIDOR_INDEX[left_city]
        right_idx = CORRIDOR_INDEX[right_city]
        step = 1 if right_idx >= left_idx else -1
        segment = CALIFORNIA_ROAD_CORRIDOR[left_idx : right_idx + step : step]
        segment_points = [[lat, lon] for _, lat, lon in segment]
        if path and segment_points:
            path.extend(segment_points[1:])
        else:
            path.extend(segment_points)
    return path
































def _parse_name_list(value):
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    for parser in (ast.literal_eval,):
        try:
            parsed = parser(text)
            if isinstance(parsed, list):
                return [str(item) for item in parsed if str(item).strip()]
        except Exception:
            pass
    return [part.strip() for part in text.replace(" | ", "|").split("|") if part.strip()]


def _catalog_lookup_for_names(names, output_dir):
    if not names:
        return pd.DataFrame()
    output_dir = Path(output_dir)
    frames = []
    for path in [
        output_dir / "production_enriched_poi_catalog.csv",
        output_dir / "coastal_attractions.csv",
    ]:
        frame = _load_csv(path)
        if not frame.empty and {"name", "latitude", "longitude"}.issubset(frame.columns):
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    catalog = pd.concat(frames, ignore_index=True, sort=False).drop_duplicates("name")
    catalog["_name_key"] = catalog["name"].astype(str).str.lower()
    rows = []
    for order, name in enumerate(names, start=1):
        match = catalog[catalog["_name_key"].eq(str(name).lower())]
        if match.empty:
            continue
        row = match.iloc[0].copy()
        row["comparison_stop_order"] = order
        rows.append(row)
    return pd.DataFrame(rows).dropna(subset=["latitude", "longitude"]) if rows else pd.DataFrame()


def _add_comparison_route_layer(map_object, *, layer_name, stop_df, color, route_cache, run_live, label_prefix, note):
    layer = folium.FeatureGroup(name=layer_name, show=False)
    if stop_df.empty:
        return layer
    stop_df = stop_df.copy()
    stop_df["comparison_stop_order"] = (
        pd.to_numeric(stop_df.get("comparison_stop_order", range(1, len(stop_df) + 1)), errors="coerce")
        .fillna(1)
        .astype(int)
    )
    stop_df = stop_df.sort_values("comparison_stop_order")
    points = stop_df[["latitude", "longitude"]].astype(float).values.tolist()
    popup_html = f"<b>{_escape(layer_name)}</b><br/>{_escape(note)}"
    path, route_mode = _add_flow_route(
        layer,
        points,
        color=color,
        route_cache=route_cache,
        run_live=run_live,
        tooltip=layer_name,
        popup_html=popup_html,
        dash_array="7 9",
        opacity=0.86,
        ant_delay=900,
    )
    for row in stop_df.itertuples(index=False):
        point = [float(row.latitude), float(row.longitude)]
        name = str(getattr(row, "name", getattr(row, "attraction_name", "route stop")))
        order = int(getattr(row, "comparison_stop_order", 1))
        popup = f"""
        <b>{_escape(name)}</b><br/>
        Layer: {_escape(layer_name)}<br/>
        Stop order: {order}<br/>
        City: {_escape(getattr(row, "city", "unknown"))}<br/>
        Category: {_escape(getattr(row, "category", getattr(row, "categories", "unknown")))}<br/>
        Utility/value: {_escape(str(getattr(row, "final_poi_value", getattr(row, "utility", "n/a"))))}
        """
        folium.CircleMarker(
            location=point,
            radius=7,
            color=color,
            fill=True,
            fillColor=color,
            fillOpacity=0.82,
            weight=2,
            tooltip=f"{layer_name} stop {order}: {name}",
            popup=folium.Popup(popup, max_width=280),
        ).add_to(layer)
        folium.Marker(
            location=point,
            icon=folium.DivIcon(
                html=f"""
                <div style='background:{color}; color:white; border:2px solid white; border-radius:14px;
                    font-size:11px; font-weight:700; padding:2px 7px; box-shadow:0 1px 6px rgba(0,0,0,0.30); white-space:nowrap;'>
                    {label_prefix}{order}
                </div>
                """
            ),
            tooltip=f"{layer_name} stop {order}: {name}",
            popup=folium.Popup(popup, max_width=280),
        ).add_to(layer)
    layer.add_to(map_object)
    return layer






































def _city_detail_layer_label(day, first):
    route_type = str(first.get("route_type", "base_city_local"))
    overnight_city = str(first.get("overnight_city", first.get("city", "")))
    if route_type.startswith("relocation"):
        start_city = str(first.get("route_start_city", first.get("city", "")))
        end_city = str(first.get("route_end_city", overnight_city))
        return f"Transition Route · Day {int(day)} · {start_city} to {end_city}"
    return f"City Loop · {overnight_city}"


























def _add_layer_control_overlap_guard(map_object):
    guard_html = """
    <style>
    .leaflet-top.leaflet-left .leaflet-control-layers-expanded {
        max-height: min(58vh, calc(100vh - 300px));
        overflow-y: auto;
        max-width: 340px;
    }
    #blueprint-map-legend {
        transition: left 160ms ease, opacity 160ms ease, transform 160ms ease;
    }
    body.blueprint-layer-control-open #blueprint-map-legend:not(.blueprint-collapsed) {
        left: 376px;
    }
    @media (max-width: 900px), (max-height: 720px) {
        body.blueprint-layer-control-open #blueprint-map-legend:not(.blueprint-collapsed) {
            opacity: 0;
            pointer-events: none;
            transform: translateY(8px);
        }
    }
    </style>
    """
    guard_js = """
    (function() {
        function protectLayerControlScroll() {
            var control = document.querySelector('.leaflet-control-layers');
            if (!control) {
                return;
            }
            if (window.L && L.DomEvent) {
                L.DomEvent.disableScrollPropagation(control);
                L.DomEvent.disableClickPropagation(control);
            }
            if (!control.dataset.blueprintScrollGuard) {
                control.dataset.blueprintScrollGuard = '1';
                ['wheel', 'mousewheel', 'DOMMouseScroll'].forEach(function(eventName) {
                    control.addEventListener(eventName, function(event) {
                        event.stopPropagation();
                    }, { passive: true });
                });
            }
        }
        function syncBlueprintLayerState() {
            var control = document.querySelector('.leaflet-control-layers');
            var isOpen = !!(control && control.classList.contains('leaflet-control-layers-expanded'));
            document.body.classList.toggle('blueprint-layer-control-open', isOpen);
            protectLayerControlScroll();
        }
        document.addEventListener('click', function() {
            window.setTimeout(syncBlueprintLayerState, 0);
        }, true);
        document.addEventListener('keyup', syncBlueprintLayerState, true);
        document.addEventListener('mouseover', syncBlueprintLayerState, true);
        document.addEventListener('wheel', protectLayerControlScroll, true);
        window.addEventListener('resize', syncBlueprintLayerState);
        window.setTimeout(protectLayerControlScroll, 100);
        window.setTimeout(syncBlueprintLayerState, 250);
        window.setTimeout(syncBlueprintLayerState, 900);
    })();
    """
    map_object.get_root().html.add_child(folium.Element(guard_html))
    map_object.get_root().script.add_child(folium.Element(guard_js))


def _add_full_scene_persistence_guard(map_object, full_scene_layer):
    """Re-add the diagnostic overview after grouped-layer control initialization."""
    if full_scene_layer is None:
        return
    map_var = map_object.get_name()
    layer_var = full_scene_layer.get_name()
    guard_js = f"""
    (function() {{
        function keepFullSceneVisible() {{
            if (typeof {map_var} === "undefined" || typeof {layer_var} === "undefined") {{
                return;
            }}
            if (!{map_var}.hasLayer({layer_var})) {{
                {layer_var}.addTo({map_var});
            }}
            {layer_var}.eachLayer(function(routeLayer) {{
                if (routeLayer && routeLayer.bringToFront) {{
                    routeLayer.bringToFront();
                }}
            }});
        }}
        window.keepFullSceneVisible = keepFullSceneVisible;
        [60, 250, 900, 1600].forEach(function(delay) {{
            window.setTimeout(keepFullSceneVisible, delay);
        }});
        window.setTimeout(function() {{
            if (typeof {map_var} !== "undefined") {{
                {map_var}.on("overlayadd layeradd overlayremove layerremove", keepFullSceneVisible);
            }}
        }}, 1000);
    }})();
    """
    map_object.get_root().script.add_child(folium.Element(guard_js))








def _add_browser_road_routing(map_object, route_specs):
    if not route_specs:
        return

    map_object.get_root().header.add_child(
        folium.Element(
            """
            <link rel="stylesheet" href="https://unpkg.com/leaflet-routing-machine@3.2.12/dist/leaflet-routing-machine.css" />
            <script src="https://unpkg.com/leaflet-routing-machine@3.2.12/dist/leaflet-routing-machine.min.js"></script>
            <style>
            .leaflet-routing-container { display: none; }
            </style>
            """
        )
    )

    safe_specs = []
    for spec in route_specs:
        safe_specs.append(
            {
                "name": str(spec["name"]),
                "color": str(spec["color"]),
                "points": [[float(lat), float(lon)] for lat, lon in spec["points"]],
            }
        )

    route_json = json.dumps(safe_specs)
    control_js = f"""
    (function() {{
        var routeSpecs = {route_json};
        var mapObject = {map_object.get_name()};

        function drawRoute(spec) {{
            if (!window.L || !L.Routing || !mapObject || spec.points.length < 2) {{
                return;
            }}
            var waypoints = spec.points.map(function(point) {{
                return L.latLng(point[0], point[1]);
            }});
            L.Routing.control({{
                waypoints: waypoints,
                addWaypoints: false,
                draggableWaypoints: false,
                fitSelectedRoutes: false,
                show: false,
                createMarker: function() {{ return null; }},
                lineOptions: {{
                    styles: [
                        {{ color: spec.color, opacity: 0.78, weight: 4 }},
                        {{ color: '#FFFFFF', opacity: 0.55, weight: 1.5 }}
                    ]
                }},
                router: L.Routing.osrmv1({{
                    serviceUrl: 'https://router.project-osrm.org/route/v1'
                }})
            }}).addTo(mapObject);
        }}

        function waitForRouting(attempt) {{
            if (window.L && L.Routing) {{
                routeSpecs.forEach(drawRoute);
                return;
            }}
            if (attempt < 40) {{
                window.setTimeout(function() {{ waitForRouting(attempt + 1); }}, 150);
            }}
        }}
        waitForRouting(0);
    }})();
    """
    map_object.get_root().script.add_child(folium.Element(control_js))
