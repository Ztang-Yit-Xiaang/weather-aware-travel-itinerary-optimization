"""Package-owned profile day-plan assembly for the production Folium renderer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .blueprint_core import (
    CITY_COORDS,
    OPTIONAL_BOOL_POI_COLUMNS,
    OPTIONAL_POI_COLUMNS,
    OPTIONAL_TEXT_POI_COLUMNS,
    PROFILE_CONFIGS,
    _catalog_series,
    _city_hotel_catalog,
    _city_poi_catalog,
    _coerce_days_by_city,
    _load_csv,
    _normalize_catalog_columns,
    _profile_config,
    _select_compact_day_stops,
    _select_hotel_for_city_plan,
    _sequence_between,
    _stanford_transition_stop,
    _transition_drive_minutes,
    _trip_sequence_with_pass_through,
)


def _optional_poi_fields(stop):
    fields = {}
    for column in OPTIONAL_POI_COLUMNS:
        value = stop.get(column, False if column in OPTIONAL_BOOL_POI_COLUMNS else "")
        try:
            missing = pd.isna(value)
        except Exception:
            missing = False
        if column in OPTIONAL_BOOL_POI_COLUMNS:
            fields[column] = False if missing else bool(value)
        elif column in OPTIONAL_TEXT_POI_COLUMNS:
            fields[column] = "" if missing else str(value)
        else:
            try:
                fields[column] = 0.0 if missing else float(value)
            except Exception:
                fields[column] = 0.0
    return fields



def _select_pass_through_day_stops(context, pass_through_cities, used_names_by_city, max_stops, profile_config):
    selected_rows = []
    if max_stops <= 0:
        return pd.DataFrame()

    for city in pass_through_cities:
        if len(selected_rows) >= max_stops:
            break
        catalog = _city_poi_catalog(context, city).copy()
        if catalog.empty:
            continue
        catalog = catalog[catalog["city"].astype(str).str.lower().eq(str(city).lower())].copy()
        if catalog.empty:
            continue

        used_names = used_names_by_city.setdefault(city, set())
        catalog = catalog[~catalog["name"].astype(str).isin(used_names)].copy()
        if catalog.empty:
            continue

        catalog["final_poi_value"] = pd.to_numeric(
            _catalog_series(catalog, "final_poi_value", 0), errors="coerce"
        ).fillna(0)
        catalog["social_score"] = pd.to_numeric(_catalog_series(catalog, "social_score", 0), errors="coerce").fillna(
            0
        )
        catalog["corridor_fit"] = pd.to_numeric(_catalog_series(catalog, "corridor_fit", 0), errors="coerce").fillna(0)
        catalog["route_fit"] = pd.to_numeric(_catalog_series(catalog, "route_fit", 0), errors="coerce").fillna(0)
        catalog["route_context_fit"] = np.maximum(catalog["corridor_fit"], catalog["route_fit"])
        catalog["detour_minutes"] = pd.to_numeric(
            _catalog_series(catalog, "detour_minutes", 0), errors="coerce"
        ).fillna(0)
        if "social_must_go" in catalog.columns:
            catalog["social_must_go"] = catalog["social_must_go"].fillna(False).astype(bool)
        else:
            catalog["social_must_go"] = False
        catalog["transition_stop_score"] = (
            catalog["final_poi_value"] * float(profile_config.get("value_weight", 1.0))
            + catalog["social_score"] * float(profile_config.get("social_weight", 0.6))
            + catalog["route_context_fit"] * 0.35
            + catalog["social_must_go"].astype(float) * 0.45
            + catalog["category"]
            .astype(str)
            .str.contains("view|bridge|beach|waterfront|landmark|scenic|observatory", case=False, na=False)
            .astype(float)
            * 0.35
            - catalog["detour_minutes"] * 0.006
        )

        stop = catalog.sort_values(
            ["transition_stop_score", "final_poi_value", "name"], ascending=[False, False, True]
        ).iloc[0]
        selected_rows.append(stop)
        used_names.add(str(stop["name"]))

    if not selected_rows:
        return pd.DataFrame()
    return pd.DataFrame(selected_rows).reset_index(drop=True)



def _select_corridor_view_stops(context, segment_cities, used_names_by_city, max_stops, profile_config):
    if max_stops <= 0:
        return pd.DataFrame()
    output_dir = Path(context["OUTPUT_DIR"])
    enriched = _load_csv(output_dir / "production_enriched_poi_catalog.csv")
    if enriched.empty:
        return pd.DataFrame()
    route_context_fit = np.maximum(
        pd.to_numeric(_catalog_series(enriched, "corridor_fit", 0.0), errors="coerce").fillna(0.0),
        pd.to_numeric(_catalog_series(enriched, "route_fit", 0.0), errors="coerce").fillna(0.0),
    )
    pool = enriched[enriched["city"].astype(str).isin(segment_cities) | route_context_fit.ge(0.75)].copy()
    if pool.empty:
        return pd.DataFrame()
    used_names = set()
    for names in used_names_by_city.values():
        used_names.update(str(name) for name in names)
    pool = pool[~pool["name"].astype(str).isin(used_names)].copy()
    if pool.empty:
        return pd.DataFrame()
    pool["final_poi_value"] = pd.to_numeric(_catalog_series(pool, "final_poi_value", 0.0), errors="coerce").fillna(
        0.0
    )
    pool["social_score"] = pd.to_numeric(_catalog_series(pool, "social_score", 0.0), errors="coerce").fillna(0.0)
    pool["corridor_fit"] = pd.to_numeric(_catalog_series(pool, "corridor_fit", 0.0), errors="coerce").fillna(0.0)
    pool["route_fit"] = pd.to_numeric(_catalog_series(pool, "route_fit", 0.0), errors="coerce").fillna(0.0)
    pool["route_context_fit"] = np.maximum(pool["corridor_fit"], pool["route_fit"])
    pool["detour_minutes"] = pd.to_numeric(_catalog_series(pool, "detour_minutes", 0.0), errors="coerce").fillna(0.0)
    pool["view_signal"] = (
        pool["category"]
        .astype(str)
        .str.contains(
            "view|bridge|beach|waterfront|landmark|scenic|observatory|campus",
            case=False,
            na=False,
        )
        .astype(float)
    )
    pool["cross_city_view_score"] = (
        pool["final_poi_value"] * float(profile_config.get("value_weight", 1.0))
        + pool["social_score"] * float(profile_config.get("social_weight", 0.6))
        + pool["route_context_fit"] * 0.35
        + pool["view_signal"] * 0.55
        - pool["detour_minutes"] * 0.006
    )
    selected = pool.sort_values(["cross_city_view_score", "final_poi_value"], ascending=False).head(max_stops).copy()
    for row in selected.itertuples(index=False):
        used_names_by_city.setdefault(str(row.city), set()).add(str(row.name))
    return _normalize_catalog_columns(selected, "corridor", "multi_city_high_utility_view")



def build_day_plan(context, stops_per_day=None, profile_name="balanced"):
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
                stanford_on_route = bool(previous_city == "San Francisco" and "Santa Cruz" in pass_through_cities)
                transition_stop_minutes = 25.0
                transition_stop_capacity = max(stops_per_day, len(pass_through_cities) + int(stanford_on_route))
                max_transition_stops = min(
                    transition_stop_capacity, max(1, int(available_visit_minutes // transition_stop_minutes))
                )
                total_transition_stop_capacity = max_transition_stops
                transition_stop_frames = []
                if stanford_on_route and max_transition_stops > 0:
                    stanford_stop = _stanford_transition_stop(context, used_names_by_city)
                    if not stanford_stop.empty:
                        transition_stop_frames.append(stanford_stop.head(1))
                        max_transition_stops -= 1
                stops = _select_pass_through_day_stops(
                    context,
                    pass_through_cities,
                    used_names_by_city,
                    max_transition_stops,
                    profile_config,
                )
                if transition_stop_frames:
                    if not stops.empty:
                        transition_stop_frames.append(stops)
                    stops = pd.concat(transition_stop_frames, ignore_index=True, sort=False)
                remaining_transition_slots = max(0, total_transition_stop_capacity - len(stops))
                if remaining_transition_slots > 0:
                    high_value_views = _select_corridor_view_stops(
                        context,
                        segment,
                        used_names_by_city,
                        remaining_transition_slots,
                        profile_config,
                    )
                    if not high_value_views.empty:
                        stops = pd.concat([stops, high_value_views], ignore_index=True, sort=False)
                remaining_slots = max(0, stops_per_day - len(stops))
                if remaining_slots > 0 and available_visit_minutes >= 75.0 * (len(stops) + 1):
                    local_fill = _select_compact_day_stops(
                        poi_catalog,
                        hotel,
                        used_names,
                        stops_per_day=remaining_slots,
                        profile_config=profile_config,
                    )
                    if not local_fill.empty:
                        stops = pd.concat([stops, local_fill], ignore_index=True, sort=False)
                        used_names.update(local_fill["name"].astype(str).tolist())
                if stops.empty:
                    stops = poi_catalog.head(1)
                    used_names.update(stops["name"].astype(str).tolist())
            else:
                stops = _select_compact_day_stops(
                    poi_catalog,
                    hotel,
                    used_names,
                    stops_per_day=stops_per_day,
                    profile_config=profile_config,
                )
                if stops.empty:
                    stops = poi_catalog.head(1)
                used_names.update(stops["name"].astype(str).tolist())

            for stop_order, (_, stop) in enumerate(stops.iterrows(), start=1):
                stop_city = str(stop.get("city", city))
                rows.append(
                    {
                        "profile": str(profile_name).lower(),
                        "profile_label": profile_config["label"],
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



def build_profile_day_plans(context):
    return {profile_name: build_day_plan(context, profile_name=profile_name) for profile_name in PROFILE_CONFIGS}
