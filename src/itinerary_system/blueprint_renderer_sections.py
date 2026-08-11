"""Request-scoped sections for the production Folium map renderer."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import folium
import numpy as np
import pandas as pd
from folium import plugins

from .blueprint_core import (
    CITY_COORDS,
    PROFILE_CONFIGS,
    STANFORD_WAYPOINT_NAME,
    _city_hotel_catalog,
    _coerce_days_by_city,
    _insert_stanford_waypoint,
    _load_csv,
    _route_distance_km,
    _scenic_route_sequence,
    _social_must_go_catalog,
    _trip_sequence_with_pass_through,
)
from .blueprint_day_plans import build_profile_day_plans
from .blueprint_render_layers import (
    _add_city_detail_layers,
    _add_flow_route,
    _add_full_scene_overview_layer,
    _add_intercity_route_layer,
    _add_model_comparison_layers,
    _add_route_matrix_layers,
    _add_selected_result_layer,
    _add_traveler_overview_layers,
)
from .blueprint_render_panels import (
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
from .blueprint_render_primitives import (
    DAY_COLORS,
    FASTEST_ROUTE_COLOR,
    ROUTE_CONTEXT_PANE,
    SCENIC_CA1_COLOR,
    _add_route_panes,
    _append_route_debug_row,
    _dedupe_route_points,
    _escape,
    _finite_float,
    _route_bounds,
    _route_debug_value,
    _route_points_for_names,
)


@dataclass(slots=True)
class RendererBuildState:
    """Mutable state owned by one production-map render request."""

    context: dict[str, Any]
    output_path: Path
    output_dir: Path
    cache_path: Path
    route_cache: dict[str, Any]
    run_live: bool
    route_debug_rows: list[dict[str, Any]]
    route_debug_registry: list[dict[str, Any]]
    trip: dict[str, Any]
    city_sequence: list[str]
    main_trip_days: int
    profile_day_plans: dict[str, pd.DataFrame]
    day_plan_df: pd.DataFrame
    profile_day_plan_df: pd.DataFrame
    use_route_matrix: bool
    fastest_sequence: list[str]
    scenic_sequence: list[str]
    fit_points: list[list[float]]
    all_points: list[list[float]]
    trip_map: folium.Map
    hide_detail_layers_on_load: bool = False
    fastest_path: list[list[float]] = field(default_factory=list)
    scenic_path: list[list[float]] = field(default_factory=list)
    fastest_layer: Any = None
    scenic_layer: Any = None
    intercity_legs_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    selected_hotel_layer: Any = None
    candidate_hotel_layer: Any = None
    social_layer: Any = None
    nature_layer: Any = None
    enriched_catalog: pd.DataFrame = field(default_factory=pd.DataFrame)
    profile_layer_groups: dict[str, list[Any]] = field(default_factory=dict)
    method_comparison_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    route_matrix_layers: list[Any] = field(default_factory=list)
    route_matrix_hidden_layers: list[Any] = field(default_factory=list)
    comparison_layers: dict[str, list[Any]] = field(default_factory=dict)
    selected_result_layer: Any = None
    traveler_overview_layers: list[Any] = field(default_factory=list)
    full_route_layers: list[Any] = field(default_factory=list)
    city_detail_layers: list[Any] = field(default_factory=list)
    transition_layers: list[Any] = field(default_factory=list)
    bandit_strategy: str = "not available"
    bandit_reward: float = np.nan


def _prepare_renderer_state(context, output_path=None, run_live_routing=None):
    output_dir = Path(context["OUTPUT_DIR"])
    figure_dir = Path(context["FIGURE_DIR"])
    cache_dir = Path(context["PROJECT_ROOT"]) / "results" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    output_path = Path(output_path or figure_dir / "production_hierarchical_trip_map.html")
    cache_path = cache_dir / "production_road_route_cache.json"
    if cache_path.exists():
        with cache_path.open("r", encoding="utf-8") as handle:
            route_cache = json.load(handle)
    else:
        route_cache = {}

    run_live = bool(context.get("RUN_LIVE_APIS", False) if run_live_routing is None else run_live_routing)
    run_live = bool(run_live or context.get("MAP_REFRESH_ROAD_GEOMETRY", False))
    route_debug_rows = []
    route_debug_registry = []

    trip = context["best_hierarchical_trip"]
    city_sequence = _trip_sequence_with_pass_through(trip)
    main_trip_days = int(sum(int(value) for value in _coerce_days_by_city(trip.get("days_by_city", {})).values()))

    profile_day_plans = build_profile_day_plans(context)
    balanced_default_plan = _load_default_hierarchical_gurobi_day_plan(output_dir, profile_day_plans["balanced"])
    profile_day_plans["balanced"] = balanced_default_plan
    day_plan_df = profile_day_plans["balanced"]
    profile_day_plan_df = pd.concat(profile_day_plans.values(), ignore_index=True)
    route_matrix_stops_df = _load_csv(output_dir / "production_route_matrix_route_stops.csv")
    use_route_matrix = bool(
        not route_matrix_stops_df.empty
        and {"route_key", "trip_days", "method", "profile"}.issubset(route_matrix_stops_df.columns)
    )
    day_plan_df.to_csv(output_dir / "production_day_plan.csv", index=False)
    profile_day_plan_df.to_csv(output_dir / "production_day_plan_profiles.csv", index=False)

    fastest_sequence = _insert_stanford_waypoint(city_sequence)
    scenic_sequence = _scenic_route_sequence(city_sequence)
    fastest_points = _route_points_for_names(fastest_sequence)
    scenic_points = _route_points_for_names(scenic_sequence)

    social_catalog_for_bounds = _social_must_go_catalog()
    social_catalog_for_bounds = social_catalog_for_bounds[
        social_catalog_for_bounds["city"].isin(city_sequence)
        | social_catalog_for_bounds["name"].astype(str).eq(STANFORD_WAYPOINT_NAME)
    ]
    fit_points = []
    for left_col, right_col in [
        ("latitude", "longitude"),
        ("hotel_latitude", "hotel_longitude"),
        ("route_start_latitude", "route_start_longitude"),
        ("route_end_latitude", "route_end_longitude"),
    ]:
        if left_col in day_plan_df.columns and right_col in day_plan_df.columns:
            fit_points.extend(day_plan_df[[left_col, right_col]].dropna().astype(float).values.tolist())
    all_points = (
        scenic_points
        + fastest_points
        + profile_day_plan_df[["latitude", "longitude"]].values.tolist()
        + profile_day_plan_df[["hotel_latitude", "hotel_longitude"]].drop_duplicates().values.tolist()
        + social_catalog_for_bounds[["latitude", "longitude"]].values.tolist()
    )
    center_lat = float(np.mean([point[0] for point in all_points]))
    center_lon = float(np.mean([point[1] for point in all_points]))

    trip_map = folium.Map(
        location=[center_lat, center_lon], zoom_start=6, tiles=None, control_scale=True, prefer_canvas=True
    )
    folium.TileLayer("CartoDB Positron", name="Light basemap", control=True, overlay=False, show=True).add_to(trip_map)
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap (alt)", control=True, overlay=False, show=False).add_to(
        trip_map
    )
    _add_route_panes(trip_map)
    plugins.Fullscreen(position="topleft", title="Expand map", title_cancel="Exit full screen").add_to(trip_map)
    return RendererBuildState(
        context=context, output_path=output_path, output_dir=output_dir, cache_path=cache_path,
        route_cache=route_cache, run_live=run_live, route_debug_rows=route_debug_rows,
        route_debug_registry=route_debug_registry, trip=trip, city_sequence=city_sequence,
        main_trip_days=main_trip_days, profile_day_plans=profile_day_plans,
        day_plan_df=day_plan_df, profile_day_plan_df=profile_day_plan_df,
        use_route_matrix=use_route_matrix, fastest_sequence=fastest_sequence,
        scenic_sequence=scenic_sequence,
        fit_points=fit_points, all_points=all_points, trip_map=trip_map,
    )

def _add_context_route_section(state: RendererBuildState):
    context = state.context
    trip_map = state.trip_map
    fastest_sequence = state.fastest_sequence
    scenic_sequence = state.scenic_sequence
    route_cache = state.route_cache
    run_live = state.run_live
    route_debug_rows = state.route_debug_rows
    route_debug_registry = state.route_debug_registry
    output_dir = state.output_dir
    all_points = state.all_points
    route_only_debug_view = bool(context.get("MAP_ROUTE_ONLY_DEBUG_VIEW", False))
    balanced_only_default_view = bool(context.get("MAP_BALANCED_ONLY_DEFAULT_VIEW", True))
    hide_detail_layers_on_load = route_only_debug_view or balanced_only_default_view
    show_context_routes = bool(context.get("SHOW_CONTEXT_ROUTES_BY_DEFAULT", False)) and not hide_detail_layers_on_load
    fastest_rows, fastest_path, fastest_layer = _add_intercity_route_layer(
        trip_map,
        "Fastest inter-city route",
        fastest_sequence,
        FASTEST_ROUTE_COLOR,
        route_cache,
        run_live,
        scenic=False,
        show=show_context_routes,
    )
    scenic_rows, scenic_path, scenic_layer = _add_intercity_route_layer(
        trip_map,
        "Scenic CA-1 / PCH route with Stanford detour",
        scenic_sequence,
        SCENIC_CA1_COLOR,
        route_cache,
        run_live,
        scenic=True,
        show=show_context_routes,
    )
    _append_route_debug_row(
        route_debug_rows,
        layer_group="Routes",
        layer_name="Fastest inter-city route",
        comparison_type="intercity",
        points=fastest_path,
        geometry_mode="intercity-leg-layer",
        show_by_default=show_context_routes,
        source_rows=len(fastest_rows),
        route_type="fastest_intercity_route",
        route_start_city=fastest_sequence[0] if fastest_sequence else "",
        route_end_city=fastest_sequence[-1] if fastest_sequence else "",
    )
    _append_route_debug_row(
        route_debug_rows,
        layer_group="Routes",
        layer_name="Scenic CA-1 / PCH route with Stanford detour",
        comparison_type="intercity",
        points=scenic_path,
        geometry_mode="intercity-leg-layer",
        show_by_default=show_context_routes,
        source_rows=len(scenic_rows),
        route_type="scenic_ca1_pch_with_stanford_detour",
        route_start_city=scenic_sequence[0] if scenic_sequence else "",
        route_end_city=scenic_sequence[-1] if scenic_sequence else "",
    )
    for control_id, label, layer, path, color in [
        ("context_fastest", "Fastest inter-city route", fastest_layer, fastest_path, FASTEST_ROUTE_COLOR),
        ("context_scenic", "Scenic CA-1 / PCH route", scenic_layer, scenic_path, SCENIC_CA1_COLOR),
    ]:
        if layer is not None and path:
            route_debug_registry.append(
                {
                    "label": label,
                    "control_label": label,
                    "control_id": control_id,
                    "family": "context",
                    "selector_group": "context",
                    "color": color,
                    "pane": ROUTE_CONTEXT_PANE,
                    "layer_var": layer.get_name(),
                    "offset_index": 0,
                    "default_checked": False,
                    "default_visible": False,
                    "quick_groups": ["context"],
                    "distance_km": round(float(_route_distance_km(_dedupe_route_points(path))), 3),
                    "unique_points": len(_dedupe_route_points(path)),
                    "bounds": _route_bounds(path),
                }
            )
    intercity_legs_df = pd.DataFrame(fastest_rows + scenic_rows)
    intercity_legs_df.to_csv(output_dir / "production_intercity_legs.csv", index=False)
    all_points.extend(fastest_path + scenic_path)
    state.hide_detail_layers_on_load = hide_detail_layers_on_load
    state.fastest_path = fastest_path
    state.scenic_path = scenic_path
    state.fastest_layer = fastest_layer
    state.scenic_layer = scenic_layer
    state.intercity_legs_df = intercity_legs_df

def _add_hotel_layer_section(state: RendererBuildState):
    context = state.context
    trip_map = state.trip_map
    use_route_matrix = state.use_route_matrix
    profile_day_plan_df = state.profile_day_plan_df
    output_dir = state.output_dir
    city_sequence = state.city_sequence
    route_debug_registry = state.route_debug_registry
    selected_hotel_layer = folium.FeatureGroup(
        name="Selected hotel/base pins",
        show=bool(context.get("SHOW_SELECTED_HOTELS_BY_DEFAULT", True)) and not use_route_matrix,
    )
    candidate_hotel_layer = folium.FeatureGroup(
        name="Hotel candidates",
        show=bool(context.get("SHOW_CANDIDATE_HOTELS_BY_DEFAULT", False)),
    )
    selected_hotels = {
        (str(getattr(row, "overnight_city", row.city)), str(row.hotel_name))
        for row in profile_day_plan_df[["city", "overnight_city", "hotel_name"]]
        .drop_duplicates()
        .itertuples(index=False)
    }
    hotel_debug_df = _load_csv(output_dir / "production_hotel_selection_debug.csv")
    for city in city_sequence:
        if not hotel_debug_df.empty:
            hotel_candidates = hotel_debug_df[hotel_debug_df["city"].astype(str).eq(city)].head(12).copy()
        else:
            hotel_candidates = _city_hotel_catalog(context, city).head(10).copy()
        for rank, hotel in enumerate(hotel_candidates.itertuples(index=False), start=1):
            hotel_name = str(getattr(hotel, "hotel_name", getattr(hotel, "name", f"{city} hotel")))
            hotel_point = [
                float(getattr(hotel, "latitude", CITY_COORDS[city][0])),
                float(getattr(hotel, "longitude", CITY_COORDS[city][1])),
            ]
            is_selected = (
                bool(getattr(hotel, "selected", False)) or (city, hotel_name) in selected_hotels
            ) and not use_route_matrix
            marker_color = "#C1121F" if is_selected else "#6A4C93"
            score = _finite_float(getattr(hotel, "hotel_score", np.nan))
            stop_distance = _finite_float(getattr(hotel, "mean_distance_to_selected_stops_km", np.nan))
            must_go_distance = _finite_float(getattr(hotel, "mean_distance_to_must_go_km", np.nan))
            popup = f"""
            <b>{_escape(hotel_name)}</b><br/>
            City: {_escape(city)}<br/>
            Candidate rank: {int(getattr(hotel, "candidate_rank", rank) or rank)}<br/>
            Selected base: {"yes" if is_selected else "no"}<br/>
            Hotel score: {_escape(f"{score:.3f}" if np.isfinite(score) else "n/a")}<br/>
            Mean distance to selected stops: {_escape(f"{stop_distance:.1f} km" if np.isfinite(stop_distance) else "n/a")}<br/>
            Mean distance to must-go: {_escape(f"{must_go_distance:.1f} km" if np.isfinite(must_go_distance) else "n/a")}<br/>
            Reason: {_escape(getattr(hotel, "selected_hotel_reason", ""))}<br/>
            Source: {_escape(getattr(hotel, "source", "unknown"))}
            """
            target_layer = selected_hotel_layer if is_selected else candidate_hotel_layer
            folium.CircleMarker(
                location=hotel_point,
                radius=9 if is_selected else 5,
                color=marker_color,
                fill=True,
                fillColor=marker_color,
                fillOpacity=0.84 if is_selected else 0.72,
                weight=3 if is_selected else 2,
                popup=folium.Popup(popup, max_width=280, min_width=190),
                tooltip=f"{city} {'selected hotel/base' if is_selected else 'hotel candidate'} {rank}: {hotel_name}",
            ).add_to(target_layer)
            if is_selected:
                folium.Marker(
                    location=hotel_point,
                    icon=folium.Icon(color="darkred", icon="hotel", prefix="fa"),
                    tooltip=f"Selected hotel/base: {hotel_name}",
                    popup=folium.Popup(popup, max_width=280, min_width=190),
                ).add_to(selected_hotel_layer)
    selected_hotel_layer.add_to(trip_map)
    candidate_hotel_layer.add_to(trip_map)
    if not use_route_matrix:
        route_debug_registry.append(
            {
                "label": "Selected hotel/base pins",
                "control_label": "Selected hotel/base pins",
                "control_id": "hotel_selected",
                "family": "hotel",
                "selector_group": "core",
                "color": "#C1121F",
                "pane": "markerPane",
                "layer_var": selected_hotel_layer.get_name(),
                "offset_index": 0,
                "default_checked": True,
                "default_visible": True,
                "quick_groups": ["hotel"],
                "distance_km": 0.0,
                "unique_points": len(selected_hotels),
                "bounds": _route_bounds(
                    profile_day_plan_df[["hotel_latitude", "hotel_longitude"]].dropna().astype(float).values.tolist()
                ),
            }
        )
    route_debug_registry.append(
        {
            "label": "Hotel candidates",
            "control_label": "Hotel candidates",
            "control_id": "hotel_candidates",
            "family": "hotel",
            "selector_group": "hotel",
            "color": "#6A4C93",
            "pane": "markerPane",
            "layer_var": candidate_hotel_layer.get_name(),
            "offset_index": 0,
            "default_checked": False,
            "default_visible": False,
            "quick_groups": ["hotel"],
            "distance_km": 0.0,
            "unique_points": int(len(hotel_debug_df)) if not hotel_debug_df.empty else 0,
            "bounds": _route_bounds(hotel_debug_df[["latitude", "longitude"]].dropna().astype(float).values.tolist())
            if not hotel_debug_df.empty
            else [],
        }
    )
    state.selected_hotel_layer = selected_hotel_layer
    state.candidate_hotel_layer = candidate_hotel_layer

def _add_must_go_layer_section(renderer_state: RendererBuildState):
    context = renderer_state.context
    trip_map = renderer_state.trip_map
    output_dir = renderer_state.output_dir
    city_sequence = renderer_state.city_sequence
    day_plan_df = renderer_state.day_plan_df
    route_debug_registry = renderer_state.route_debug_registry
    social_layer = folium.FeatureGroup(
        name="Must-go candidates",
        show=bool(context.get("SHOW_SOCIAL_CANDIDATES_BY_DEFAULT", False)),
    )
    enriched_catalog = _load_csv(output_dir / "production_enriched_poi_catalog.csv")
    if not enriched_catalog.empty:
        social_catalog = enriched_catalog[
            enriched_catalog.get("social_must_go", pd.Series(False, index=enriched_catalog.index)).astype(bool)
            | enriched_catalog.get("social_score", pd.Series(0.0, index=enriched_catalog.index)).astype(float).ge(0.70)
        ].copy()
    else:
        social_catalog = _social_must_go_catalog()
    social_catalog = social_catalog[
        social_catalog["city"].isin(city_sequence)
        | social_catalog["name"].astype(str).eq(STANFORD_WAYPOINT_NAME)
        | social_catalog.get("corridor_fit", pd.Series(0.0, index=social_catalog.index)).astype(float).gt(0.70)
    ].copy()
    social_catalog.to_csv(output_dir / "production_social_must_go_candidates.csv", index=False)
    selected_must_go_names = set(
        day_plan_df.loc[
            day_plan_df.get("social_must_go", pd.Series(False, index=day_plan_df.index)).astype(bool),
            "attraction_name",
        ]
        .dropna()
        .astype(str)
        .tolist()
    )
    for row in social_catalog.itertuples(index=False):
        final_value = float(getattr(row, "final_poi_value", getattr(row, "source_score", 0.0)) or 0.0)
        corridor_fit = float(getattr(row, "corridor_fit", 0.0) or 0.0)
        detour_minutes = float(getattr(row, "detour_minutes", 0.0) or 0.0)
        is_selected_must_go = str(row.name) in selected_must_go_names
        has_coordinates = np.isfinite(float(row.latitude)) and np.isfinite(float(row.longitude))
        state = "selected" if is_selected_must_go else "skipped" if has_coordinates else "unavailable"
        marker_color = "#16A34A" if state == "selected" else "#F4A261" if state == "skipped" else "#9CA3AF"
        popup = f"""
        <b>{_escape(row.name)}</b><br/>
        City bucket: {_escape(row.city)}<br/>
        Must-go state: {_escape(state)}<br/>
        Category: {_escape(row.category)}<br/>
        Source list: {_escape(getattr(row, "source_list", getattr(row, "source", "unknown")))}<br/>
        Final POI value: {final_value:.3f}<br/>
        Social score: {float(row.social_score):.2f}<br/>
        Must-go weight: {float(getattr(row, "must_go_weight", 0.0) or 0.0):.2f}<br/>
        Corridor fit: {corridor_fit:.2f}<br/>
        Detour estimate: {detour_minutes:.1f} min<br/>
        {_escape(getattr(row, "social_reason", ""))}
        """
        folium.Marker(
            location=[float(row.latitude), float(row.longitude)],
            popup=folium.Popup(popup, max_width=300, min_width=200),
            tooltip=f"Must-go {state}: {row.name}",
            icon=folium.Icon(
                color="green" if state == "selected" else "orange" if state == "skipped" else "gray",
                icon="star",
                prefix="fa",
            ),
        ).add_to(social_layer)
        folium.CircleMarker(
            location=[float(row.latitude), float(row.longitude)],
            radius=9 if state == "selected" else 7,
            color=marker_color,
            fill=True,
            fillColor=marker_color,
            fillOpacity=0.78,
            weight=3 if state == "selected" else 2,
            popup=folium.Popup(popup, max_width=300, min_width=200),
            tooltip=f"Must-go {state}: {row.name}",
        ).add_to(social_layer)
    social_layer.add_to(trip_map)
    route_debug_registry.append(
        {
            "label": "Must-go candidates",
            "control_label": "Must-go candidates",
            "control_id": "must_go_candidates",
            "family": "must_go",
            "selector_group": "must_go",
            "color": "#F4A261",
            "pane": "markerPane",
            "layer_var": social_layer.get_name(),
            "offset_index": 0,
            "default_checked": False,
            "default_visible": False,
            "quick_groups": ["must_go"],
            "distance_km": 0.0,
            "unique_points": int(len(social_catalog)),
            "bounds": _route_bounds(social_catalog[["latitude", "longitude"]].dropna().astype(float).values.tolist())
            if not social_catalog.empty
            else [],
        }
    )
    renderer_state.social_layer = social_layer
    renderer_state.enriched_catalog = enriched_catalog

def _add_nature_layer_section(state: RendererBuildState):
    context = state.context
    trip_map = state.trip_map
    enriched_catalog = state.enriched_catalog
    day_plan_df = state.day_plan_df
    route_debug_registry = state.route_debug_registry
    nature_layer = folium.FeatureGroup(
        name="Nature / park candidates",
        show=bool(context.get("SHOW_NATURE_CANDIDATES_BY_DEFAULT", False)),
    )
    if not enriched_catalog.empty:
        nature_catalog = enriched_catalog[
            enriched_catalog.get("is_nature", pd.Series(False, index=enriched_catalog.index)).astype(bool)
            | enriched_catalog.get("is_national_park", pd.Series(False, index=enriched_catalog.index)).astype(bool)
            | enriched_catalog.get("is_protected_area", pd.Series(False, index=enriched_catalog.index)).astype(bool)
            | pd.to_numeric(
                enriched_catalog.get("nature_score", pd.Series(0.0, index=enriched_catalog.index)), errors="coerce"
            )
            .fillna(0.0)
            .ge(0.45)
        ].copy()
    else:
        nature_catalog = pd.DataFrame()
    selected_stop_names = set(day_plan_df.get("attraction_name", pd.Series(dtype=str)).dropna().astype(str).tolist())
    for row in nature_catalog.head(120).itertuples(index=False):
        try:
            point = [float(row.latitude), float(row.longitude)]
        except Exception:
            continue
        is_np = bool(getattr(row, "is_national_park", False))
        is_state_or_protected = bool(getattr(row, "is_state_park", False)) or bool(
            getattr(row, "is_protected_area", False)
        )
        is_viewpoint = bool(getattr(row, "is_scenic_viewpoint", False))
        marker_color = (
            "#15803D" if is_np else "#22C55E" if is_state_or_protected else "#0F766E" if is_viewpoint else "#65A30D"
        )
        icon_name = "star" if is_np else "tree" if is_state_or_protected else "binoculars" if is_viewpoint else "leaf"
        selected_state = "selected" if str(row.name) in selected_stop_names else "candidate"
        popup = f"""
        <b>{_escape(row.name)}</b><br/>
        State: {_escape(selected_state)}<br/>
        City/region: {_escape(getattr(row, "city", ""))} / {_escape(getattr(row, "nature_region", ""))}<br/>
        Park type: {_escape(getattr(row, "park_type", ""))}<br/>
        Nature score: {float(getattr(row, "nature_score", 0.0) or 0.0):.2f}<br/>
        Scenic score: {float(getattr(row, "scenic_score", 0.0) or 0.0):.2f}<br/>
        Interest fit: {float(getattr(row, "interest_fit", 0.0) or 0.0):.2f}<br/>
        Interest adjusted value: {float(getattr(row, "interest_adjusted_value", getattr(row, "final_poi_value", 0.0)) or 0.0):.3f}<br/>
        Interest delta: {float(getattr(row, "interest_delta", 0.0) or 0.0):.3f}<br/>
        Weather sensitivity: {float(getattr(row, "weather_sensitivity", 0.0) or 0.0):.2f}<br/>
        Seasonality risk: {float(getattr(row, "seasonality_risk", 0.0) or 0.0):.2f}<br/>
        Source list: {_escape(getattr(row, "source_list", getattr(row, "source", "unknown")))}<br/>
        Reason: {_escape(getattr(row, "reason_selected", ""))}
        """
        folium.Marker(
            location=point,
            popup=folium.Popup(popup, max_width=320, min_width=210),
            tooltip=f"Nature {selected_state}: {row.name}",
            icon=folium.Icon(color="green", icon=icon_name, prefix="fa"),
        ).add_to(nature_layer)
        folium.CircleMarker(
            location=point,
            radius=9 if is_np else 7,
            color=marker_color,
            fill=True,
            fillColor=marker_color,
            fillOpacity=0.72,
            weight=3 if is_np else 2,
            popup=folium.Popup(popup, max_width=320, min_width=210),
            tooltip=f"Nature {selected_state}: {row.name}",
        ).add_to(nature_layer)
    nature_layer.add_to(trip_map)
    route_debug_registry.append(
        {
            "label": "Nature / park candidates",
            "control_label": "Nature / park candidates",
            "control_id": "nature_candidates",
            "family": "nature",
            "selector_group": "nature",
            "color": "#15803D",
            "pane": "markerPane",
            "layer_var": nature_layer.get_name(),
            "offset_index": 0,
            "default_checked": False,
            "default_visible": False,
            "quick_groups": ["nature"],
            "distance_km": 0.0,
            "unique_points": int(len(nature_catalog)) if not nature_catalog.empty else 0,
            "bounds": _route_bounds(nature_catalog[["latitude", "longitude"]].dropna().astype(float).values.tolist())
            if not nature_catalog.empty
            else [],
        }
    )
    state.nature_layer = nature_layer

def _add_profile_route_section(state: RendererBuildState):
    trip_map = state.trip_map
    profile_day_plans = state.profile_day_plans
    hide_detail_layers_on_load = state.hide_detail_layers_on_load
    route_cache = state.route_cache
    run_live = state.run_live
    route_debug_rows = state.route_debug_rows
    main_trip_days = state.main_trip_days
    profile_layer_groups = {PROFILE_CONFIGS[profile_name]["label"]: [] for profile_name in PROFILE_CONFIGS}
    for profile_name, plan_df in profile_day_plans.items():
        config = PROFILE_CONFIGS[profile_name]
        profile_prefix = config["label"][0].upper()
        for day, group in plan_df.groupby("day", sort=True):
            color = DAY_COLORS[(int(day) - 1) % len(DAY_COLORS)]
            first = group.iloc[0]
            overnight_city = str(first.get("overnight_city", first["city"]))
            route_type = str(first.get("route_type", "base_city_local"))
            if route_type.startswith("relocation"):
                raw_pass_through = first.get("pass_through_cities", "")
                if raw_pass_through is None or pd.isna(raw_pass_through):
                    raw_pass_through = ""
                pass_through_label = str(raw_pass_through).replace("; ", " -> ")
                day_label_city = f"{first.get('route_start_city', first['city'])} -> {pass_through_label + ' -> ' if pass_through_label else ''}{overnight_city}"
            else:
                day_label_city = overnight_city
            layer_name = f"{config['label']} · Day {int(day)} · {day_label_city}"
            show_profile_day_layer = bool(config["show"]) and not hide_detail_layers_on_load
            day_layer = folium.FeatureGroup(name=layer_name, show=show_profile_day_layer)
            hotel_point = [float(first["hotel_latitude"]), float(first["hotel_longitude"])]
            route_start_point = [
                float(first.get("route_start_latitude", first["hotel_latitude"])),
                float(first.get("route_start_longitude", first["hotel_longitude"])),
            ]
            route_end_point = [
                float(first.get("route_end_latitude", first["hotel_latitude"])),
                float(first.get("route_end_longitude", first["hotel_longitude"])),
            ]
            stop_points = group[["latitude", "longitude"]].values.tolist()
            day_points = [route_start_point, *stop_points, route_end_point]
            popup_html = (
                f"<b>{_escape(config['label'])} Day {int(day)} route</b><br/>"
                f"Day path: {_escape(day_label_city)}<br/>"
                f"Route type: {_escape(route_type)}<br/>"
                f"Estimated inter-city drive on this day: {float(first.get('drive_minutes_to_next_base', 0.0) or 0.0):.1f} min<br/>"
                f"Visit-time budget after drive: {float(first.get('available_visit_minutes', 0.0) or 0.0):.1f} min<br/>"
                f"Drive-time source: {_escape(first.get('drive_time_source', 'unknown'))}"
            )
            day_path, day_route_mode = _add_flow_route(
                day_layer,
                day_points,
                color=color,
                route_cache=route_cache,
                run_live=run_live,
                tooltip=f"{config['label']} Day {int(day)} route",
                popup_html=popup_html,
                dash_array="8 10" if not run_live else None,
                opacity=0.86,
                ant_delay=850,
            )
            _append_route_debug_row(
                route_debug_rows,
                layer_group=f"{config['label']} Traveler",
                layer_name=layer_name,
                comparison_type="traveler_day",
                profile=profile_name,
                method=str(first.get("method", "profile_day_plan")),
                trip_days=_route_debug_value(first.get("trip_days", main_trip_days)),
                day=int(day),
                route_type=route_type,
                route_start_city=str(first.get("route_start_city", first.get("city", ""))),
                route_end_city=str(first.get("route_end_city", overnight_city)),
                pass_through_cities=str(first.get("pass_through_cities", "")),
                drive_minutes=first.get("drive_minutes_to_next_base", np.nan),
                available_visit_minutes=first.get("available_visit_minutes", np.nan),
                points=day_points,
                geometry_mode=day_route_mode,
                show_by_default=show_profile_day_layer,
                source_rows=len(group),
            )

            if route_type.startswith("relocation") and route_start_point != route_end_point:
                folium.Marker(
                    location=route_start_point,
                    popup=folium.Popup(
                        f"<b>Day {int(day)} route start</b><br/>"
                        f"{_escape(first.get('route_start_name', 'Previous base'))}<br/>"
                        f"Travel day toward {_escape(overnight_city)}",
                        max_width=260,
                    ),
                    tooltip=f"{config['label']} Day {int(day)} route start",
                    icon=folium.Icon(color="cadetblue", icon="play", prefix="fa"),
                ).add_to(day_layer)

            folium.Marker(
                location=hotel_point,
                popup=folium.Popup(
                    f"<b>{_escape(first['hotel_name'])}</b><br/>"
                    f"Profile: {_escape(config['label'])}<br/>"
                    f"Day {int(day)} overnight base in {_escape(overnight_city)}<br/>"
                    f"Route type: {_escape(route_type)}<br/>"
                    f"Overnight booked: {_escape('yes' if bool(first.get('hotel_booked', True)) else 'no, final-day base only')}<br/>"
                    f"Source: {_escape(first['hotel_source'])}",
                    max_width=280,
                ),
                tooltip=f"{config['label']} Day {int(day)} hotel/base: {first['hotel_name']}",
                icon=folium.Icon(color="darkred", icon="hotel", prefix="fa"),
            ).add_to(day_layer)

            for row in group.itertuples(index=False):
                point = [float(row.latitude), float(row.longitude)]
                popup = f"""
                <b>{_escape(row.attraction_name)}</b><br/>
                Profile: {_escape(config["label"])}<br/>
                Day {int(row.day)}, stop {int(row.stop_order)}<br/>
                City: {_escape(row.city)}<br/>
                Category: {_escape(row.category)}<br/>
                Source: {_escape(row.attraction_source)}<br/>
                Source list: {_escape(getattr(row, "source_list", getattr(row, "attraction_source", "unknown")))}<br/>
                Final POI value: {float(getattr(row, "final_poi_value", 0.0) or 0.0):.3f}<br/>
                Interest fit: {float(getattr(row, "interest_fit", 0.0) or 0.0):.2f}<br/>
                Interest adjusted value: {float(getattr(row, "interest_adjusted_value", getattr(row, "final_poi_value", 0.0)) or 0.0):.3f}<br/>
                Interest delta: {float(getattr(row, "interest_delta", 0.0) or 0.0):.3f}<br/>
                Nature/city/culture/history: {float(getattr(row, "nature_score", 0.0) or 0.0):.2f} / {float(getattr(row, "city_score", 0.0) or 0.0):.2f} / {float(getattr(row, "culture_score", 0.0) or 0.0):.2f} / {float(getattr(row, "history_score", 0.0) or 0.0):.2f}<br/>
                Park type: {_escape(getattr(row, "park_type", ""))}<br/>
                Nature region: {_escape(getattr(row, "nature_region", ""))}<br/>
                Weather sensitivity: {float(getattr(row, "weather_sensitivity", 0.0) or 0.0):.2f}<br/>
                Seasonality risk: {float(getattr(row, "seasonality_risk", 0.0) or 0.0):.2f}<br/>
                Social must-go: {"yes" if bool(getattr(row, "social_must_go", False)) else "no"}<br/>
                Social score: {float(getattr(row, "social_score", 0.0)):.2f}<br/>
                Must-go weight: {float(getattr(row, "must_go_weight", 0.0) or 0.0):.2f}<br/>
                Corridor fit: {float(getattr(row, "corridor_fit", 0.0) or 0.0):.2f}<br/>
                Detour estimate: {float(getattr(row, "detour_minutes", 0.0) or 0.0):.1f} min<br/>
                Data confidence: {float(getattr(row, "data_confidence", 0.0) or 0.0):.2f}<br/>
                Reason selected: {_escape(getattr(row, "reason_selected", ""))}<br/>
                {_escape(getattr(row, "social_reason", ""))}
                """
                marker_radius = 9 if bool(getattr(row, "social_must_go", False)) else 7
                marker_weight = 4 if bool(getattr(row, "social_must_go", False)) else 2
                folium.CircleMarker(
                    location=point,
                    radius=marker_radius,
                    color=color,
                    fill=True,
                    fillColor=color,
                    fillOpacity=0.86,
                    weight=marker_weight,
                    popup=folium.Popup(popup, max_width=280, min_width=190),
                    tooltip=f"{config['label']} Day {int(row.day)} stop {int(row.stop_order)}: {row.attraction_name}",
                ).add_to(day_layer)
                badge_text = (
                    f"D{int(row.day)}.{int(row.stop_order)}"
                    if profile_name == "balanced"
                    else f"{profile_prefix}{int(row.day)}.{int(row.stop_order)}"
                )
                badge = f"""
                <div style='background:{color}; color:white; border:2px solid white; border-radius:14px;
                    font-size:11px; font-weight:700; padding:2px 7px; box-shadow:0 1px 6px rgba(0,0,0,0.30); white-space:nowrap;'>
                    {badge_text}
                </div>
                """
                folium.Marker(
                    location=point,
                    icon=folium.DivIcon(html=badge),
                    popup=folium.Popup(popup, max_width=280, min_width=190),
                    tooltip=f"{config['label']} Day {int(row.day)} stop {int(row.stop_order)}: {row.attraction_name}",
                ).add_to(day_layer)

            day_layer.add_to(trip_map)
            profile_layer_groups[config["label"]].append(day_layer)
    state.profile_layer_groups = profile_layer_groups

def _add_comparison_layer_section(state: RendererBuildState):
    context = state.context
    output_dir = state.output_dir
    trip_map = state.trip_map
    route_cache = state.route_cache
    run_live = state.run_live
    route_debug_rows = state.route_debug_rows
    route_debug_registry = state.route_debug_registry
    use_route_matrix = state.use_route_matrix
    hide_detail_layers_on_load = state.hide_detail_layers_on_load
    day_plan_df = state.day_plan_df
    profile_day_plans = state.profile_day_plans
    fastest_path = state.fastest_path
    scenic_path = state.scenic_path
    fit_points = state.fit_points
    best_bandit = context.get("hybrid_bandit_summary_df", pd.DataFrame())
    bandit_strategy = "not available"
    bandit_reward = np.nan
    if isinstance(best_bandit, pd.DataFrame) and not best_bandit.empty:
        bandit_strategy = str(best_bandit.iloc[0].get("route_search_strategy", bandit_strategy))
        bandit_reward = float(
            best_bandit.iloc[0].get("posterior_mean_reward", best_bandit.iloc[0].get("posterior_mean", np.nan))
        )

    method_comparison_df = _build_method_comparison(output_dir)
    route_matrix_layers = []
    route_matrix_hidden_layers = []
    if use_route_matrix:
        route_matrix_layers, route_matrix_hidden_layers = _add_route_matrix_layers(
            trip_map,
            output_dir,
            route_cache,
            run_live,
            route_debug_rows=route_debug_rows,
            route_debug_registry=route_debug_registry,
        )
        comparison_layers = {}
        selected_result_layer = None
        traveler_overview_layers = []
        full_route_layers = []
        city_detail_layers = []
        transition_layers = []
    else:
        # Comparison layers are loaded from exported route-stop CSVs so the HTML map
        # can toggle real method/trip-length routes instead of summary-only tables.
        comparison_layers = _add_model_comparison_layers(
            trip_map,
            output_dir,
            route_cache,
            run_live,
            show_by_default=bool(context.get("SHOW_COMPARISON_LAYERS_BY_DEFAULT", False))
            and not hide_detail_layers_on_load,
            route_debug_rows=route_debug_rows,
        )
        selected_result_layer = _add_selected_result_layer(
            trip_map,
            day_plan_df,
            route_cache=route_cache,
            run_live=run_live,
            route_debug_rows=route_debug_rows,
            show_by_default=bool(context.get("SHOW_SELECTED_RESULT_BY_DEFAULT", False))
            and not hide_detail_layers_on_load,
        )
        traveler_overview_layers = _add_traveler_overview_layers(
            trip_map,
            profile_day_plans,
            route_cache=route_cache,
            run_live=run_live,
            route_debug_rows=route_debug_rows,
            show_by_default=bool(context.get("SHOW_TRAVELER_OVERVIEWS_BY_DEFAULT", False))
            and not hide_detail_layers_on_load,
        )
        full_route_layers = _add_full_scene_overview_layer(
            trip_map,
            output_dir=output_dir,
            fastest_path=fastest_path,
            scenic_path=scenic_path,
            profile_day_plans=profile_day_plans,
            route_cache=route_cache,
            run_live=run_live,
            route_debug_rows=route_debug_rows,
            route_debug_registry=route_debug_registry,
        )
        city_detail_layers, transition_layers = _add_city_detail_layers(
            trip_map,
            day_plan_df,
            route_cache,
            run_live,
            route_debug_rows=route_debug_rows,
            route_debug_registry=route_debug_registry,
        )
    for comparison_path in [
        output_dir / "production_trip_length_route_stops.csv",
        output_dir / "production_method_route_stops.csv",
    ]:
        comparison_df = _load_csv(comparison_path)
        if not fit_points and not comparison_df.empty and {"latitude", "longitude"}.issubset(comparison_df.columns):
            fit_points.extend(comparison_df[["latitude", "longitude"]].dropna().astype(float).values.tolist())
        if (
            not fit_points
            and not comparison_df.empty
            and {"hotel_latitude", "hotel_longitude"}.issubset(comparison_df.columns)
        ):
            fit_points.extend(
                comparison_df[["hotel_latitude", "hotel_longitude"]].dropna().astype(float).values.tolist()
            )
    state.method_comparison_df = method_comparison_df
    state.route_matrix_layers = route_matrix_layers
    state.route_matrix_hidden_layers = route_matrix_hidden_layers
    state.comparison_layers = comparison_layers
    state.selected_result_layer = selected_result_layer
    state.traveler_overview_layers = traveler_overview_layers
    state.full_route_layers = full_route_layers
    state.city_detail_layers = city_detail_layers
    state.transition_layers = transition_layers
    state.bandit_strategy = bandit_strategy
    state.bandit_reward = bandit_reward

def _add_dashboard_section(state: RendererBuildState):
    output_dir = state.output_dir
    intercity_legs_df = state.intercity_legs_df
    route_debug_rows = state.route_debug_rows
    trip_map = state.trip_map
    main_trip_days = state.main_trip_days
    trip = state.trip
    profile_day_plans = state.profile_day_plans
    day_plan_df = state.day_plan_df
    method_comparison_df = state.method_comparison_df
    bandit_strategy = state.bandit_strategy
    bandit_reward = state.bandit_reward
    fastest_drive_hours = (
        float(
            intercity_legs_df[intercity_legs_df["route_layer"].eq("Fastest inter-city route")][
                "estimated_drive_minutes"
            ].sum()
        )
        / 60.0
    )
    scenic_drive_hours = (
        float(
            intercity_legs_df[intercity_legs_df["route_layer"].eq("Scenic CA-1 / PCH route with Stanford detour")][
                "estimated_drive_minutes"
            ].sum()
        )
        / 60.0
    )
    route_debug_df = pd.DataFrame(route_debug_rows)
    route_debug_df.to_csv(output_dir / "production_map_route_debug.csv", index=False)
    route_debug_summary_html = _build_route_debug_summary_html(route_debug_df)

    panel_html = f"""
    <style>
    .blueprint-floating-panel {{
        position: fixed;
        background: rgba(255,255,255,0.97); border: 1px solid #CFCFCF; border-radius: 8px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.16); font-size: 11px; line-height: 1.38;
        overflow: hidden;
    }}
    .blueprint-panel-header {{
        align-items: center;
        background: rgba(248,250,252,0.98);
        border: 0;
        border-bottom: 1px solid #E5E7EB;
        color: #2E2E2E;
        cursor: pointer;
        display: flex;
        font: inherit;
        font-size: 13px;
        font-weight: 800;
        justify-content: space-between;
        padding: 9px 12px;
        text-align: left;
        width: 100%;
    }}
    .blueprint-panel-header:hover {{
        background: #F1F5F9;
    }}
    .blueprint-panel-chevron {{
        border: 1px solid #CBD5E1;
        border-radius: 999px;
        display: inline-flex;
        height: 19px;
        align-items: center;
        justify-content: center;
        margin-left: 10px;
        width: 19px;
    }}
    .blueprint-panel-body {{
        max-height: calc(100vh - 126px);
        overflow-y: auto;
        padding: 12px 14px;
    }}
    .blueprint-collapsed {{
        max-height: 42px;
    }}
    .blueprint-collapsed .blueprint-panel-body {{
        display: none;
    }}
    .blueprint-draggable .blueprint-panel-header {{
        cursor: grab;
    }}
    .blueprint-dragging .blueprint-panel-header {{
        cursor: grabbing;
    }}
    .blueprint-route-arrow {{
        align-items: center;
        background: rgba(255,255,255,0.68);
        border: 1px solid var(--route-arrow-color);
        border-radius: 999px;
        box-shadow: 0 1px 5px rgba(0,0,0,0.25);
        display: flex;
        height: 12px;
        justify-content: center;
        opacity: 0.78;
        width: 12px;
    }}
    .blueprint-route-arrow span {{
        border-bottom: 3px solid transparent;
        border-left: 5px solid var(--route-arrow-color);
        border-top: 3px solid transparent;
        display: block;
        height: 0;
        margin-left: 1px;
        width: 0;
    }}
    #blueprint-result-panel {{
        top: 82px; right: 18px; width: 520px; z-index: 9998;
    }}
    #blueprint-result-panel.blueprint-collapsed {{
        width: 265px;
    }}
    #blueprint-map-legend {{
        bottom: 46px;
        left: 14px;
        width: 310px;
        z-index: 9997;
    }}
    #blueprint-map-legend.blueprint-collapsed {{
        width: 230px;
    }}
    #blueprint-result-panel .panel-title {{ font-weight: 700; font-size: 14px; margin-bottom: 3px; }}
    #blueprint-result-panel .panel-subtitle {{ color:#5c5c5c; margin-bottom: 10px; }}
    #blueprint-result-panel .summary-line {{ margin-top: 5px; }}
    #blueprint-result-panel .strategy-pill {{ display:inline-block; background:#E7F4F1; color:#1E655D; border-radius:12px; padding:2px 8px; font-weight:700; }}
    #blueprint-result-panel .panel-section-title {{ border-top:1px solid #ECECEC; padding-top:8px; margin-top:10px; font-weight:700; }}
    #blueprint-result-panel .muted-note {{ color:#666; font-size:11px; }}
    #blueprint-result-panel table {{ width:100%; border-collapse:collapse; margin-top:8px; table-layout:fixed; }}
    #blueprint-result-panel th, #blueprint-result-panel td {{ border-top:1px solid #ECECEC; padding:5px 4px; vertical-align:top; overflow-wrap:anywhere; }}
    #blueprint-result-panel th {{ text-align:left; font-weight:700; color:#333; }}
    #blueprint-result-panel .day-card {{ border-top: 1px solid #ECECEC; padding-top: 8px; margin-top: 8px; }}
    #blueprint-result-panel .day-card-title {{ font-weight:700; }}
    #blueprint-result-panel .hotel-line {{ color:#555; margin-top: 2px; }}
    #blueprint-result-panel .muted-city {{ color:#777; font-size: 11px; }}
    #blueprint-result-panel ol {{ margin: 6px 0 0 18px; padding: 0; }}
    #blueprint-result-panel li {{ margin: 2px 0; overflow-wrap: anywhere; }}
    </style>
    <div id="blueprint-result-panel" class="blueprint-floating-panel blueprint-draggable blueprint-collapsed" data-panel-id="dashboard">
        <button class="blueprint-panel-header" type="button" aria-expanded="false">
            <span>California Itinerary Dashboard</span>
            <span class="blueprint-panel-chevron">+</span>
        </button>
        <div class="blueprint-panel-body">
        <div class="panel-subtitle">Hierarchical day allocation + hybrid bandit/optimization route search</div>
        <div class="summary-line"><b>Default displayed plan:</b> {main_trip_days}-day Balanced route</div>
        <div class="summary-line"><b>Gateway:</b> {_escape(trip["gateway_start"])} → {_escape(trip["gateway_end"])}</div>
        <div class="summary-line"><b>Default profile:</b> <span class="strategy-pill">Balanced</span></div>
        <div class="summary-line"><b>Fastest route estimate:</b> {fastest_drive_hours:.2f} hours</div>
        <div class="summary-line"><b>Scenic route estimate:</b> {scenic_drive_hours:.2f} hours</div>
        <div class="summary-line"><b>CA-1 note:</b> San Francisco → Stanford → Santa Cruz is a Silicon Valley detour / connector, not CA-1. Coastal legs after Santa Cruz use CA-1 / PCH where available, with US-101/local connectors where necessary.</div>
        <div class="summary-line"><b>Bandit strategy:</b> <span class="strategy-pill">{_escape(bandit_strategy)}</span></div>
        <div class="summary-line"><b>Posterior reward:</b> {_escape(f"{bandit_reward:.3f}" if np.isfinite(bandit_reward) else "n/a")}</div>
        <div class="summary-line"><b>Route selector:</b> use the draggable checkbox panel to show 7/9/12-day routes, hierarchical/greedy/bandit routes, traveler profiles, detailed city/day routes, and context layers only when needed.</div>
        {route_debug_summary_html}
        {_build_must_go_summary_html(output_dir)}
        {_build_hotel_summary_html(output_dir)}
        {_build_html_data_source_summary(output_dir)}
        {_build_method_comparison_html(method_comparison_df)}
        {_build_trip_length_comparison_html(output_dir)}
        <table>
            <thead>
                <tr><th>Profile</th><th>Stops</th><th>Must-go</th><th>Mode</th><th>Hotel/base</th></tr>
            </thead>
            <tbody>{_build_profile_comparison(profile_day_plans)}</tbody>
        </table>
        {_build_day_panel(day_plan_df)}
        </div>
    </div>
    """
    trip_map.get_root().html.add_child(folium.Element(panel_html))
    _add_interest_bar_preview_panel(trip_map, output_dir)

def _finalize_renderer(state: RendererBuildState):
    trip_map = state.trip_map
    route_matrix_layers = state.route_matrix_layers
    full_route_layers = state.full_route_layers
    city_detail_layers = state.city_detail_layers
    transition_layers = state.transition_layers
    selected_result_layer = state.selected_result_layer
    scenic_layer = state.scenic_layer
    fastest_layer = state.fastest_layer
    social_layer = state.social_layer
    nature_layer = state.nature_layer
    selected_hotel_layer = state.selected_hotel_layer
    candidate_hotel_layer = state.candidate_hotel_layer
    traveler_overview_layers = state.traveler_overview_layers
    profile_layer_groups = state.profile_layer_groups
    comparison_layers = state.comparison_layers
    route_matrix_hidden_layers = state.route_matrix_hidden_layers
    route_debug_registry = state.route_debug_registry
    fit_points = state.fit_points
    all_points = state.all_points
    run_live = state.run_live
    cache_path = state.cache_path
    route_cache = state.route_cache
    output_path = state.output_path
    day_plan_df = state.day_plan_df
    grouped_layers = {
        "Route Matrix": route_matrix_layers,
        "Full Routes": full_route_layers,
        "City Details": city_detail_layers,
        "Transitions": transition_layers,
        "Selected Result": [selected_result_layer] if selected_result_layer is not None else [],
        "Routes": [scenic_layer, fastest_layer],
        "Map Context": [social_layer, nature_layer, selected_hotel_layer, candidate_hotel_layer],
        "Traveler Comparison": traveler_overview_layers,
        "Relaxed Traveler": profile_layer_groups["Relaxed"],
        "Balanced Traveler": profile_layer_groups["Balanced"],
        "Explorer Traveler": profile_layer_groups["Explorer"],
    }
    grouped_layers.update(comparison_layers)
    grouped_layers = {name: layers for name, layers in grouped_layers.items() if layers}
    unchecked_default_layers = [
        fastest_layer,
        scenic_layer,
        selected_result_layer,
        social_layer,
        nature_layer,
        candidate_hotel_layer,
        *route_matrix_hidden_layers,
        *traveler_overview_layers,
        *city_detail_layers,
        *transition_layers,
    ]
    unchecked_default_layers.extend(
        [layer for layer in full_route_layers if layer is not None and not getattr(layer, "show", False)]
    )
    for layers in comparison_layers.values():
        unchecked_default_layers.extend(layers)
    for layers in profile_layer_groups.values():
        unchecked_default_layers.extend(layers)
    # The custom Route Selector is the only user-facing route/layer control.
    # Folium's grouped control is intentionally omitted to avoid duplicate,
    # conflicting checkbox state and oversized layer panels.
    _add_route_debug_controls(trip_map, route_debug_registry, unchecked_default_layers)
    _add_blueprint_panel_behavior(trip_map)
    bounds_points = fit_points or all_points
    if bounds_points:
        trip_map.fit_bounds(bounds_points, padding=(40, 40))

    if run_live:
        try:
            with cache_path.open("w", encoding="utf-8") as handle:
                json.dump(route_cache, handle)
        except PermissionError as exc:
            print(f"Road route cache write skipped: {exc}")

    trip_map.save(str(output_path))
    return trip_map, day_plan_df, output_path
