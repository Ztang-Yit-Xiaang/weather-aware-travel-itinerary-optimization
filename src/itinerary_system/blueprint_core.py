"""Package-owned core helpers migrated from the legacy blueprint module.

The underscore-prefixed names are preserved during E3.C2 so existing package
and notebook callers can be switched without changing behavior.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd

from .nature_catalog import NATURE_POI_COLUMNS
from .region_scenarios import all_scenario_coordinates

CITY_COORDS = {
    "San Diego": (32.7157, -117.1611),
    "Los Angeles": (34.0522, -118.2437),
    "Santa Barbara": (34.4208, -119.6982),
    "San Luis Obispo": (35.2828, -120.6596),
    "Monterey": (36.6002, -121.8947),
    "Santa Cruz": (36.9741, -122.0308),
    "San Francisco": (37.7749, -122.4194),
}
CITY_COORDS.update(all_scenario_coordinates())

OPTIONAL_POI_COLUMNS = ["route_fit", *NATURE_POI_COLUMNS]
OPTIONAL_BOOL_POI_COLUMNS = {
    "is_nature",
    "is_national_park",
    "is_state_park",
    "is_protected_area",
    "is_scenic_viewpoint",
    "is_hiking",
}
OPTIONAL_TEXT_POI_COLUMNS = {"park_type", "nature_region", "reason_selected"}

STANFORD_WAYPOINT_NAME = "Stanford University Main Quad"

PROFILE_CONFIGS = {
    "relaxed": {
        "label": "Relaxed",
        "stops_per_day": 2,
        "show": False,
        "social_quota": 1,
        "social_weight": 0.80,
        "must_go_bonus": 0.45,
        "anchor_distance_weight": 0.24,
        "hotel_distance_weight": 0.18,
        "hotel_proximity_weight": 0.58,
        "hotel_rating_weight": 0.40,
        "route_mode": "compact hotel-proximity route",
    },
    "balanced": {
        "label": "Balanced",
        "stops_per_day": 3,
        "show": True,
        "social_quota": 2,
        "social_weight": 1.15,
        "must_go_bonus": 0.65,
        "anchor_distance_weight": 0.18,
        "hotel_distance_weight": 0.10,
        "hotel_proximity_weight": 0.40,
        "hotel_rating_weight": 0.45,
        "route_mode": "balanced utility/social/compactness route",
    },
    "explorer": {
        "label": "Explorer",
        "stops_per_day": 4,
        "show": False,
        "social_quota": 3,
        "social_weight": 1.75,
        "must_go_bonus": 0.90,
        "anchor_distance_weight": 0.12,
        "hotel_distance_weight": 0.06,
        "hotel_proximity_weight": 0.24,
        "hotel_rating_weight": 0.35,
        "route_mode": "social must-go and high-utility route",
    },
}

SOCIAL_MUST_GO_PLACES = [
    {
        "city": "San Francisco",
        "name": "Golden Gate Bridge",
        "category": "social_must_go:landmark",
        "latitude": 37.8199,
        "longitude": -122.4783,
        "source": "curated_social_must_go",
        "source_score": 9.8,
        "social_score": 0.98,
        "social_must_go": True,
        "must_go_weight": 0.95,
        "social_reason": "High-frequency Bay Area must-go landmark in travel itineraries.",
    },
    {
        "city": "San Francisco",
        "name": "Stanford University Main Quad",
        "category": "social_must_go:campus",
        "latitude": 37.4275,
        "longitude": -122.1697,
        "source": "curated_social_must_go",
        "source_score": 9.3,
        "social_score": 0.93,
        "social_must_go": True,
        "must_go_weight": 0.98,
        "social_reason": "Bay Area / Silicon Valley must-go waypoint; considered as a social landmark candidate.",
    },
    {
        "city": "Los Angeles",
        "name": "Hollywood Walk of Fame",
        "category": "social_must_go:landmark",
        "latitude": 34.1016,
        "longitude": -118.3269,
        "source": "curated_social_must_go",
        "source_score": 9.9,
        "social_score": 0.99,
        "social_must_go": True,
        "must_go_weight": 0.95,
        "social_reason": "Common social-media must-go for first-time Los Angeles trips.",
    },
    {
        "city": "Los Angeles",
        "name": "TCL Chinese Theatre",
        "category": "social_must_go:landmark",
        "latitude": 34.1020,
        "longitude": -118.3409,
        "source": "curated_social_must_go",
        "source_score": 9.6,
        "social_score": 0.95,
        "social_must_go": True,
        "must_go_weight": 0.88,
        "social_reason": "Hollywood cluster candidate often paired with the Walk of Fame.",
    },
    {
        "city": "Los Angeles",
        "name": "Griffith Observatory",
        "category": "social_must_go:viewpoint",
        "latitude": 34.1184,
        "longitude": -118.3004,
        "source": "curated_social_must_go",
        "source_score": 9.4,
        "social_score": 0.94,
        "social_must_go": True,
        "must_go_weight": 0.90,
        "social_reason": "Iconic LA viewpoint and Hollywood-area social landmark.",
    },
    {
        "city": "Monterey",
        "name": "Bixby Creek Bridge Viewpoint",
        "category": "social_must_go:viewpoint",
        "latitude": 36.3715,
        "longitude": -121.9018,
        "source": "curated_social_must_go",
        "source_score": 9.2,
        "social_score": 0.92,
        "social_must_go": True,
        "must_go_weight": 0.86,
        "social_reason": "Big Sur / CA-1 scenic waypoint with strong photo value.",
    },
    {
        "city": "Santa Barbara",
        "name": "Stearns Wharf",
        "category": "social_must_go:waterfront",
        "latitude": 34.4100,
        "longitude": -119.6856,
        "source": "curated_social_must_go",
        "source_score": 8.8,
        "social_score": 0.88,
        "social_must_go": True,
        "must_go_weight": 0.78,
        "social_reason": "Common Santa Barbara waterfront must-go waypoint.",
    },
    {
        "city": "Santa Barbara",
        "name": "Santa Barbara County Courthouse",
        "category": "social_must_go:landmark",
        "latitude": 34.4240,
        "longitude": -119.7024,
        "source": "curated_social_must_go",
        "source_score": 8.7,
        "social_score": 0.87,
        "social_must_go": True,
        "must_go_weight": 0.76,
        "social_reason": "Highly recommended Santa Barbara architecture/photo stop.",
    },
]

CALIFORNIA_ROAD_CORRIDOR = [
    ("San Francisco", 37.7749, -122.4194),
    ("Half Moon Bay", 37.4636, -122.4286),
    ("Santa Cruz", 36.9741, -122.0308),
    ("Watsonville", 36.9102, -121.7569),
    ("Monterey", 36.6002, -121.8947),
    ("Carmel-by-the-Sea", 36.5552, -121.9233),
    ("Big Sur", 36.2704, -121.8081),
    ("San Simeon", 35.6433, -121.1890),
    ("Morro Bay", 35.3658, -120.8499),
    ("San Luis Obispo", 35.2828, -120.6596),
    ("Pismo Beach", 35.1428, -120.6413),
    ("Santa Maria", 34.9530, -120.4357),
    ("Gaviota", 34.4708, -120.2290),
    ("Santa Barbara", 34.4208, -119.6982),
    ("Ventura", 34.2746, -119.2290),
    ("Oxnard", 34.1975, -119.1771),
    ("Malibu", 34.0259, -118.7798),
    ("Santa Monica", 34.0195, -118.4912),
    ("Los Angeles", 34.0522, -118.2437),
]

CORRIDOR_INDEX = {name: idx for idx, (name, _, _) in enumerate(CALIFORNIA_ROAD_CORRIDOR)}

def _load_csv(path):
    path = Path(path)
    if path.exists() and path.stat().st_size > 0:
        try:
            return pd.read_csv(path)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()
    return pd.DataFrame()

def _catalog_series(output, column, default):
    if column in output.columns:
        return output[column]
    return pd.Series(default, index=output.index)

def _numeric_catalog_series(output, column, default):
    return pd.to_numeric(_catalog_series(output, column, default), errors="coerce").fillna(default)

def _coerce_optional_poi_columns(output):
    for column in OPTIONAL_POI_COLUMNS:
        if column not in output.columns:
            if column in OPTIONAL_BOOL_POI_COLUMNS:
                output[column] = False
            elif column in OPTIONAL_TEXT_POI_COLUMNS:
                output[column] = ""
            else:
                output[column] = 0.0
        if column in OPTIONAL_BOOL_POI_COLUMNS:
            output[column] = output[column].fillna(False).astype(bool)
        elif column in OPTIONAL_TEXT_POI_COLUMNS:
            output[column] = output[column].fillna("").astype(str)
        else:
            output[column] = pd.to_numeric(output[column], errors="coerce").fillna(0.0)
    return output

def _normalize_catalog_columns(df, city, source_name):
    if df.empty:
        return pd.DataFrame(
            columns=[
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
        )

    output = df.copy()
    output["city"] = output.get("city", city)
    output["source"] = output.get("source", source_name)
    output["source_list"] = output.get("source_list", output["source"])
    output["category"] = output.get("category", output.get("categories", "attraction"))
    output["source_score"] = pd.to_numeric(
        _catalog_series(
            output, "source_score", _catalog_series(output, "utility", _catalog_series(output, "review_count", 1.0))
        ),
        errors="coerce",
    ).fillna(1.0)
    if "social_score" in output.columns:
        output["social_score"] = pd.to_numeric(output["social_score"], errors="coerce").fillna(0.0)
    else:
        output["social_score"] = 0.0
    if "social_must_go" in output.columns:
        output["social_must_go"] = output["social_must_go"].fillna(False).astype(bool)
    else:
        output["social_must_go"] = False
    if "social_reason" not in output.columns:
        output["social_reason"] = ""
    output["must_go_weight"] = pd.to_numeric(
        _catalog_series(output, "must_go_weight", output["social_must_go"].astype(float)), errors="coerce"
    ).fillna(0.0)
    output["corridor_fit"] = pd.to_numeric(_catalog_series(output, "corridor_fit", 0.0), errors="coerce").fillna(0.0)
    output["detour_minutes"] = pd.to_numeric(_catalog_series(output, "detour_minutes", 0.0), errors="coerce").fillna(
        0.0
    )
    output["data_confidence"] = pd.to_numeric(_catalog_series(output, "data_confidence", 0.5), errors="coerce").fillna(
        0.5
    )
    output["final_poi_value"] = pd.to_numeric(
        _catalog_series(output, "final_poi_value", output["source_score"] + output["social_score"] * 3.0),
        errors="coerce",
    ).fillna(0.0)
    output["latitude"] = pd.to_numeric(output["latitude"], errors="coerce")
    output["longitude"] = pd.to_numeric(output["longitude"], errors="coerce")
    output = _coerce_optional_poi_columns(output)
    output = output.dropna(subset=["name", "latitude", "longitude"])
    base_columns = [
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
    ]
    return output[base_columns + OPTIONAL_POI_COLUMNS]

def _social_must_go_catalog(city=None):
    frame = pd.DataFrame(SOCIAL_MUST_GO_PLACES)
    if city is not None:
        frame = frame[frame["city"].astype(str).str.lower().eq(str(city).lower())]
    return frame.reset_index(drop=True)

def _distance_to_city_center(city, latitudes, longitudes):
    city_lat, city_lon = CITY_COORDS[city]
    latitudes = pd.to_numeric(latitudes, errors="coerce")
    longitudes = pd.to_numeric(longitudes, errors="coerce")
    return np.sqrt((latitudes - city_lat) ** 2 + (longitudes - city_lon) ** 2) * 111.0

def _point_distance_km(left_lat, left_lon, right_lat, right_lon):
    lat_scale = 111.0
    lon_scale = 111.0 * np.cos(np.deg2rad((float(left_lat) + float(right_lat)) / 2.0))
    return float(
        np.sqrt(
            ((float(left_lat) - float(right_lat)) * lat_scale) ** 2
            + ((float(left_lon) - float(right_lon)) * lon_scale) ** 2
        )
    )

def _order_stops_from_hotel(stops, hotel_lat, hotel_lon):
    if stops.empty:
        return stops

    remaining = stops.copy().reset_index(drop=True)
    ordered_rows = []
    current_lat = float(hotel_lat)
    current_lon = float(hotel_lon)

    while not remaining.empty:
        distances = remaining.apply(
            lambda row, current_lat=current_lat, current_lon=current_lon: _point_distance_km(
                current_lat, current_lon, row["latitude"], row["longitude"]
            ),
            axis=1,
        )
        next_idx = distances.idxmin()
        next_row = remaining.loc[next_idx]
        ordered_rows.append(next_row)
        current_lat = float(next_row["latitude"])
        current_lon = float(next_row["longitude"])
        remaining = remaining.drop(index=next_idx).reset_index(drop=True)

    return pd.DataFrame(ordered_rows).reset_index(drop=True)

def _profile_config(profile_name="balanced", stops_per_day=None):
    config = PROFILE_CONFIGS.get(str(profile_name).lower(), PROFILE_CONFIGS["balanced"]).copy()
    if stops_per_day is not None:
        config["stops_per_day"] = int(stops_per_day)
    return config

def _select_compact_day_stops(poi_catalog, hotel, used_names, stops_per_day=3, profile_config=None):
    profile_config = profile_config or _profile_config("balanced", stops_per_day=stops_per_day)
    if poi_catalog.empty:
        return poi_catalog

    available = poi_catalog[~poi_catalog["name"].isin(used_names)].copy()
    if available.empty:
        available = poi_catalog.copy()

    available["latitude"] = pd.to_numeric(available["latitude"], errors="coerce")
    available["longitude"] = pd.to_numeric(available["longitude"], errors="coerce")
    available["source_score"] = pd.to_numeric(_catalog_series(available, "source_score", 1.0), errors="coerce").fillna(
        1.0
    )
    available["social_score"] = pd.to_numeric(_catalog_series(available, "social_score", 0.0), errors="coerce").fillna(
        0.0
    )
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
    available["final_poi_value"] = pd.to_numeric(
        _catalog_series(available, "final_poi_value", available["source_score"] + available["social_score"] * 3.0),
        errors="coerce",
    ).fillna(0.0)
    available["social_must_go"] = _catalog_series(available, "social_must_go", False)
    available["social_must_go"] = available["social_must_go"].fillna(False).astype(bool)
    available = available.dropna(subset=["latitude", "longitude"])
    if available.empty:
        return poi_catalog.head(stops_per_day)

    hotel_lat = float(hotel.get("latitude"))
    hotel_lon = float(hotel.get("longitude"))
    score_span = max(float(available["final_poi_value"].max() - available["final_poi_value"].min()), 1e-6)
    available["source_score_norm"] = (available["final_poi_value"] - available["final_poi_value"].min()) / score_span
    available["distance_to_hotel_km"] = available.apply(
        lambda row: _point_distance_km(hotel_lat, hotel_lon, row["latitude"], row["longitude"]),
        axis=1,
    )

    top_source_pool = available.sort_values(
        ["source_score", "distance_to_hotel_km", "name"],
        ascending=[False, True, True],
    ).head(max(25, stops_per_day * 5))
    social_pool = available[available["social_must_go"]].sort_values(
        ["social_score", "source_score", "distance_to_hotel_km"],
        ascending=[False, False, True],
    )
    nearby_pool = available.sort_values(["distance_to_hotel_km", "source_score"], ascending=[True, False]).head(15)
    candidate_pool = pd.concat([top_source_pool, social_pool, nearby_pool], ignore_index=True)
    candidate_pool = candidate_pool.drop_duplicates("name").reset_index(drop=True)

    best_cluster = None
    best_score = -np.inf
    for _, anchor in candidate_pool.iterrows():
        pool = candidate_pool.copy()
        pool["distance_to_anchor_km"] = pool.apply(
            lambda row, anchor=anchor: _point_distance_km(
                anchor["latitude"], anchor["longitude"], row["latitude"], row["longitude"]
            ),
            axis=1,
        )
        pool["compact_rank_score"] = (
            pool["source_score_norm"] * 1.05
            + pool["social_score"] * float(profile_config.get("social_weight", 1.15))
            + pool["must_go_weight"] * pool["social_score"] * float(profile_config.get("must_go_bonus", 0.65))
            + pool["route_context_fit"] * 0.18
            - pool["distance_to_anchor_km"] * float(profile_config.get("anchor_distance_weight", 0.18))
            - pool["distance_to_hotel_km"] * float(profile_config.get("hotel_distance_weight", 0.10))
            - pool["detour_minutes"] * 0.004
        )
        cluster = pool.sort_values(["compact_rank_score", "source_score"], ascending=[False, False]).head(stops_per_day)
        if cluster.empty:
            continue

        pairwise_distances = []
        points = cluster[["latitude", "longitude"]].values.tolist()
        for i, left in enumerate(points):
            for right in points[i + 1 :]:
                pairwise_distances.append(_point_distance_km(left[0], left[1], right[0], right[1]))
        mean_pairwise = float(np.mean(pairwise_distances)) if pairwise_distances else 0.0
        cluster_score = (
            float(cluster["source_score_norm"].mean()) * 1.6
            + float(cluster["social_score"].mean()) * (float(profile_config.get("social_weight", 1.15)) + 0.20)
            + float((cluster["must_go_weight"] * cluster["social_score"]).mean())
            * (float(profile_config.get("must_go_bonus", 0.65)) + 0.10)
            + float(cluster["route_context_fit"].mean()) * 0.18
            - mean_pairwise * 0.16
            - float(cluster["distance_to_hotel_km"].mean()) * float(profile_config.get("hotel_distance_weight", 0.10))
            - float(cluster["detour_minutes"].mean()) * 0.004
        )
        if cluster_score > best_score:
            best_score = cluster_score
            best_cluster = cluster

    if best_cluster is None:
        best_cluster = candidate_pool.head(stops_per_day)

    ordered = _order_stops_from_hotel(best_cluster, hotel_lat, hotel_lon)
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
        if column not in ordered.columns:
            if column == "social_reason":
                ordered[column] = ""
            elif column == "source_list":
                ordered[column] = ordered.get("source", "unknown")
            else:
                ordered[column] = 0.0
    ordered = _coerce_optional_poi_columns(ordered)
    return ordered[output_columns]

def _select_hotel_for_city_plan(hotel_catalog, poi_catalog, used_names, stops_per_day=3, profile_config=None):
    profile_config = profile_config or _profile_config("balanced", stops_per_day=stops_per_day)
    if hotel_catalog.empty:
        return hotel_catalog.iloc[0]

    hotels = hotel_catalog.copy().head(14).reset_index(drop=True)
    hotels["rating_score"] = _numeric_catalog_series(hotels, "rating_score", 0.0)
    hotels["nightly_price"] = pd.to_numeric(hotels.get("nightly_price", np.nan), errors="coerce")
    rating_span = max(float(hotels["rating_score"].max() - hotels["rating_score"].min()), 1e-6)
    price_filled = hotels["nightly_price"].fillna(
        hotels["nightly_price"].median() if hotels["nightly_price"].notna().any() else 0.0
    )
    price_span = max(float(price_filled.max() - price_filled.min()), 1e-6)
    hotels["rating_norm"] = (hotels["rating_score"] - hotels["rating_score"].min()) / rating_span
    hotels["price_norm"] = (price_filled - price_filled.min()) / price_span

    best_idx = 0
    best_score = -np.inf
    for idx, hotel in hotels.iterrows():
        stops = _select_compact_day_stops(
            poi_catalog,
            hotel,
            used_names,
            stops_per_day=stops_per_day,
            profile_config=profile_config,
        )
        if stops.empty:
            continue
        distances = stops.apply(
            lambda row, hotel=hotel: _point_distance_km(
                hotel["latitude"], hotel["longitude"], row["latitude"], row["longitude"]
            ),
            axis=1,
        )
        score = (
            float(stops["source_score"].mean()) * 0.16
            + float(stops.get("social_score", pd.Series(0.0, index=stops.index)).mean())
            * (1.5 + float(profile_config.get("social_weight", 1.15)))
            + float(stops.get("social_must_go", pd.Series(False, index=stops.index)).astype(float).mean())
            * (0.6 + float(profile_config.get("must_go_bonus", 0.65)))
            + float(hotel["rating_norm"]) * float(profile_config.get("hotel_rating_weight", 0.45))
            - float(hotel["price_norm"]) * 0.20
            - float(distances.mean()) * float(profile_config.get("hotel_proximity_weight", 0.40))
        )
        if score > best_score:
            best_score = score
            best_idx = idx

    return hotels.loc[best_idx]

def _city_poi_catalog(context, city):
    output_dir = Path(context["OUTPUT_DIR"])
    cached_enriched = _load_csv(output_dir / "production_enriched_poi_catalog.csv")
    cached_osm = _load_csv(output_dir / "production_city_poi_catalog.csv")

    city_frames = []
    primary_city = context.get("CITY", "Santa Barbara")
    top100 = context.get("top100_with_waiting_time", context.get("top100", pd.DataFrame()))

    if not cached_enriched.empty:
        include_corridor_social = city in {"San Francisco", "Monterey", "Los Angeles"}
        route_context_fit = np.maximum(
            pd.to_numeric(_catalog_series(cached_enriched, "corridor_fit", 0.0), errors="coerce").fillna(0.0),
            pd.to_numeric(_catalog_series(cached_enriched, "route_fit", 0.0), errors="coerce").fillna(0.0),
        )
        corridor_social_mask = (
            route_context_fit.gt(0.70)
            & cached_enriched.get("social_must_go", pd.Series(False, index=cached_enriched.index)).astype(bool)
            & include_corridor_social
        )
        city_rows = cached_enriched[
            cached_enriched["city"].astype(str).str.lower().eq(city.lower()) | corridor_social_mask
        ]
        if not city_rows.empty:
            city_frames.append(_normalize_catalog_columns(city_rows, city, "multi_source_enriched_catalog"))

    has_enriched_for_city = bool(city_frames)

    if not has_enriched_for_city and city == primary_city and isinstance(top100, pd.DataFrame) and not top100.empty:
        city_rows = top100.copy()
        if "city" in city_rows.columns:
            city_rows = city_rows[city_rows["city"].astype(str).str.lower().eq(city.lower())]
        city_frames.append(_normalize_catalog_columns(city_rows, city, "yelp_business_catalog"))

    if not has_enriched_for_city and not cached_osm.empty:
        city_rows = cached_osm[cached_osm["city"].astype(str).str.lower().eq(city.lower())]
        if not city_rows.empty:
            city_frames.append(_normalize_catalog_columns(city_rows, city, "openstreetmap_overpass_cache"))

    social_rows = _social_must_go_catalog(city)
    if cached_enriched.empty and not social_rows.empty:
        city_frames.append(_normalize_catalog_columns(social_rows, city, "curated_social_must_go"))

    city_frame = pd.concat(city_frames, ignore_index=True) if city_frames else pd.DataFrame()

    if city_frame.empty:
        lat, lon = CITY_COORDS[city]
        city_frame = pd.DataFrame(
            [
                {
                    "city": city,
                    "name": f"{city} attraction catalog pending",
                    "category": "data_ingestion_needed",
                    "latitude": lat,
                    "longitude": lon,
                    "source": "city_center_placeholder",
                    "source_score": 0.0,
                    "social_score": 0.0,
                    "social_must_go": False,
                    "social_reason": "",
                }
            ]
        )

    city_frame["social_score"] = _numeric_catalog_series(city_frame, "social_score", 0.0)
    city_frame["source_score"] = _numeric_catalog_series(city_frame, "source_score", 0.0)
    city_frame["final_poi_value"] = pd.to_numeric(
        city_frame.get("final_poi_value", city_frame["source_score"] + 3.0 * city_frame["social_score"]),
        errors="coerce",
    ).fillna(0.0)
    interest_value = _numeric_catalog_series(city_frame, "interest_adjusted_value", 0.0)
    city_frame["ranking_score"] = np.where(interest_value.gt(0.0), interest_value, city_frame["final_poi_value"])
    return city_frame.sort_values(
        ["ranking_score", "source_score", "name"], ascending=[False, False, True]
    ).drop_duplicates("name")

def _city_hotel_catalog(context, city):
    output_dir = Path(context["OUTPUT_DIR"])
    cached_osm = _load_csv(output_dir / "production_city_hotel_catalog.csv")

    hotels_df = context.get("hotels_df", pd.DataFrame())
    primary_city = context.get("CITY", "Santa Barbara")
    hotel_frames = []
    if city == primary_city and isinstance(hotels_df, pd.DataFrame) and not hotels_df.empty:
        local_hotels = hotels_df.copy()
        local_hotels["city"] = city
        local_hotels["source"] = local_hotels.get("source", "project_hotel_table")
        local_hotels["rating_score"] = pd.to_numeric(
            local_hotels.get("rating_score", local_hotels.get("stars", 0)), errors="coerce"
        ).fillna(0)
        local_hotels["nightly_price"] = _numeric_catalog_series(local_hotels, "nightly_price", 0)
        local_hotels["type_priority"] = (
            local_hotels.get("type", "hotel")
            .map({"hotel": 0, "motel": 1, "guest_house": 2, "hostel": 3, "apartment": 4})
            .fillna(5)
        )
        local_hotels["distance_to_center_km"] = _distance_to_city_center(
            city, local_hotels["latitude"], local_hotels["longitude"]
        )
        hotel_frames.append(local_hotels)

    if not cached_osm.empty:
        city_rows = cached_osm[cached_osm["city"].astype(str).str.lower().eq(city.lower())].copy()
        if not city_rows.empty:
            rating_fallback = _numeric_catalog_series(city_rows, "stars", 0)
            city_rows["rating_score"] = _numeric_catalog_series(city_rows, "rating_score", rating_fallback)
            city_rows["nightly_price"] = pd.to_numeric(city_rows.get("nightly_price", np.nan), errors="coerce")
            city_rows["source"] = city_rows.get("source", "openstreetmap_overpass_cache")
            type_priority = {"hotel": 0, "motel": 1, "guest_house": 2, "hostel": 3, "apartment": 4}
            city_rows["type_priority"] = city_rows.get("type", "hotel").map(type_priority).fillna(5)
            city_rows["distance_to_center_km"] = _distance_to_city_center(
                city, city_rows["latitude"], city_rows["longitude"]
            )
            hotel_frames.append(city_rows)

    if hotel_frames:
        hotels = pd.concat(hotel_frames, ignore_index=True, sort=False)
        hotels["rating_score"] = _numeric_catalog_series(hotels, "rating_score", 0)
        hotels["nightly_price"] = pd.to_numeric(hotels.get("nightly_price", np.nan), errors="coerce")
        hotels["type_priority"] = _numeric_catalog_series(hotels, "type_priority", 5)
        hotels["distance_to_center_km"] = pd.to_numeric(
            hotels.get(
                "distance_to_center_km", _distance_to_city_center(city, hotels["latitude"], hotels["longitude"])
            ),
            errors="coerce",
        ).fillna(999)
        return (
            hotels.sort_values(
                ["rating_score", "type_priority", "distance_to_center_km", "nightly_price", "name"],
                ascending=[False, True, True, True, True],
            )
            .drop_duplicates("name")
            .reset_index(drop=True)
        )

    lat, lon = CITY_COORDS[city]
    return pd.DataFrame(
        [
            {
                "city": city,
                "name": f"{city} lodging candidate pending",
                "type": "hotel",
                "latitude": lat,
                "longitude": lon,
                "rating_score": np.nan,
                "nightly_price": np.nan,
                "source": "city_center_placeholder",
            }
        ]
    )

def _unique_in_order(values):
    seen = set()
    output = []
    for value in values:
        if value not in seen:
            output.append(value)
            seen.add(value)
    return output

def _coerce_name_list(value):
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if str(item).strip()]
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple)):
            return [str(item) for item in parsed if str(item).strip()]
    except Exception:
        pass
    for separator in [" -> ", ";", "|", ","]:
        if separator in text:
            return [part.strip().strip("'\"") for part in text.split(separator) if part.strip().strip("'\"")]
    return [text.strip("'\"")]

def _coerce_days_by_city(value):
    if isinstance(value, dict):
        return {str(key): int(val) for key, val in value.items()}
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return {}
    try:
        parsed = ast.literal_eval(str(value))
        if isinstance(parsed, dict):
            return {str(key): int(val) for key, val in parsed.items()}
    except Exception:
        pass
    return {}

def _trip_sequence_with_pass_through(trip):
    sequence = _coerce_name_list(trip.get("city_sequence", []))
    overnight_bases = _coerce_name_list(trip.get("overnight_bases", []))
    pass_through = _coerce_name_list(trip.get("pass_through_cities", []))
    if len(sequence) >= 2:
        return _unique_in_order(sequence)
    if len(overnight_bases) >= 2:
        if pass_through and len(overnight_bases) >= 3:
            return _unique_in_order([overnight_bases[0], *pass_through, *overnight_bases[1:]])
        if pass_through:
            return _unique_in_order([overnight_bases[0], *pass_through, overnight_bases[-1]])
        return _unique_in_order(overnight_bases)
    days_by_city = _coerce_days_by_city(trip.get("days_by_city", {}))
    return _unique_in_order(list(days_by_city.keys()))

def _sequence_between(sequence, origin_city, destination_city):
    try:
        origin_idx = sequence.index(origin_city)
        destination_idx = sequence.index(destination_city)
    except ValueError:
        return [origin_city, destination_city]
    if origin_idx <= destination_idx:
        return sequence[origin_idx : destination_idx + 1]
    return list(reversed(sequence[destination_idx : origin_idx + 1]))

def _transition_drive_minutes(context, origin_city, destination_city, through_cities):
    output_dir = Path(context.get("OUTPUT_DIR", "results/outputs"))
    legs_path = output_dir / "production_intercity_legs.csv"
    segment_cities = [origin_city, *through_cities, destination_city]

    if legs_path.exists():
        legs_df = _load_csv(legs_path)
        if not legs_df.empty and "estimated_drive_minutes" in legs_df.columns:
            scenic_rows = legs_df[
                legs_df.get("route_layer", "").astype(str).str.contains("Scenic", case=False, na=False)
            ]
            total = 0.0
            matched = 0
            for left, right in zip(segment_cities[:-1], segment_cities[1:], strict=False):
                leg_match = scenic_rows[
                    scenic_rows.get("from", "").astype(str).eq(str(left))
                    & scenic_rows.get("to", "").astype(str).eq(str(right))
                ]
                if not leg_match.empty:
                    total += float(leg_match.iloc[0]["estimated_drive_minutes"])
                    matched += 1
            if matched == len(segment_cities) - 1 and total > 0:
                return total, "cached_osrm_or_scenic_intercity_legs"

    scenic_sequence = _scenic_route_sequence(segment_cities)
    total = 0.0
    for left, right in zip(scenic_sequence[:-1], scenic_sequence[1:], strict=False):
        waypoints = _scenic_leg_waypoints(left, right)
        classification = _leg_classification(left, right, scenic=True)
        total += _estimate_drive_minutes(waypoints, classification)
    return total, "curated_waypoint_speed_limit_proxy"

def _stanford_transition_stop(context, used_names_by_city):
    used_names = used_names_by_city.setdefault("Palo Alto / Stanford", set())
    if STANFORD_WAYPOINT_NAME in used_names:
        return pd.DataFrame()
    san_francisco_catalog = _city_poi_catalog(context, "San Francisco")
    stanford_rows = san_francisco_catalog[san_francisco_catalog["name"].astype(str).eq(STANFORD_WAYPOINT_NAME)].copy()
    if stanford_rows.empty:
        social_catalog = _social_must_go_catalog()
        stanford_rows = social_catalog[social_catalog["name"].astype(str).eq(STANFORD_WAYPOINT_NAME)].copy()
        if not stanford_rows.empty:
            source_score = _numeric_catalog_series(stanford_rows, "source_score", 0.0)
            social_score = _numeric_catalog_series(stanford_rows, "social_score", 0.0)
            stanford_rows["final_poi_value"] = source_score * 0.15 + social_score * 0.90
    if stanford_rows.empty:
        return pd.DataFrame()
    stanford_rows["city"] = "Palo Alto / Stanford"
    used_names.add(STANFORD_WAYPOINT_NAME)
    return _normalize_catalog_columns(stanford_rows, "Palo Alto / Stanford", "curated_social_must_go")

def _coord_for_place(name):
    if name in CITY_COORDS:
        lat, lon = CITY_COORDS[name]
        return [float(lat), float(lon)]

    social_catalog = _social_must_go_catalog()
    match = social_catalog[social_catalog["name"].astype(str).str.lower().eq(str(name).lower())]
    if not match.empty:
        first = match.iloc[0]
        return [float(first["latitude"]), float(first["longitude"])]

    if name in CORRIDOR_INDEX:
        _, lat, lon = CALIFORNIA_ROAD_CORRIDOR[CORRIDOR_INDEX[name]]
        return [float(lat), float(lon)]

    raise KeyError(f"No coordinate available for route waypoint: {name}")

def _insert_stanford_waypoint(sequence):
    sequence = _unique_in_order(sequence)
    if STANFORD_WAYPOINT_NAME in sequence or "San Francisco" not in sequence:
        return sequence

    output = list(sequence)
    sf_idx = output.index("San Francisco")
    if "Santa Cruz" in output and output.index("Santa Cruz") < sf_idx:
        output.insert(sf_idx, STANFORD_WAYPOINT_NAME)
    else:
        output.insert(sf_idx + 1, STANFORD_WAYPOINT_NAME)
    return output

def _scenic_route_sequence(city_sequence):
    sequence = _unique_in_order(city_sequence)
    canonical_southbound = [
        "San Francisco",
        STANFORD_WAYPOINT_NAME,
        "Santa Cruz",
        "Monterey",
        "San Luis Obispo",
        "Santa Barbara",
        "Los Angeles",
    ]
    canonical_northbound = list(reversed(canonical_southbound))
    if "San Francisco" in sequence and "Los Angeles" in sequence:
        if sequence.index("San Francisco") <= sequence.index("Los Angeles"):
            return [name for name in canonical_southbound if name == STANFORD_WAYPOINT_NAME or name in sequence]
        return [name for name in canonical_northbound if name == STANFORD_WAYPOINT_NAME or name in sequence]
    return _insert_stanford_waypoint(sequence)

def _corridor_segment_points(left_name, right_name):
    if left_name not in CORRIDOR_INDEX or right_name not in CORRIDOR_INDEX:
        return [_coord_for_place(left_name), _coord_for_place(right_name)]
    left_idx = CORRIDOR_INDEX[left_name]
    right_idx = CORRIDOR_INDEX[right_name]
    step = 1 if right_idx >= left_idx else -1
    segment = CALIFORNIA_ROAD_CORRIDOR[left_idx : right_idx + step : step]
    return [[float(lat), float(lon)] for _, lat, lon in segment]

def _scenic_leg_waypoints(left_name, right_name):
    if STANFORD_WAYPOINT_NAME in {left_name, right_name}:
        return [_coord_for_place(left_name), _coord_for_place(right_name)]
    return _corridor_segment_points(left_name, right_name)

def _route_distance_km(points):
    if len(points) < 2:
        return 0.0
    return sum(
        _point_distance_km(left[0], left[1], right[0], right[1])
        for left, right in zip(points[:-1], points[1:], strict=False)
    )

def _estimate_drive_minutes(points, classification):
    distance_km = _route_distance_km(points)
    if "CA-1" in classification or "PCH" in classification:
        speed_kmh = 56.0
        road_factor = 1.05
    elif "Stanford" in classification or "connector" in classification:
        speed_kmh = 64.0
        road_factor = 1.12
    else:
        speed_kmh = 78.0
        road_factor = 1.08
    return round((distance_km * road_factor / speed_kmh) * 60.0, 1)

def _leg_classification(left_name, right_name, scenic):
    if not scenic:
        return "fastest OSRM driving route through selected gateway/city waypoints"
    if STANFORD_WAYPOINT_NAME in {left_name, right_name}:
        return "Silicon Valley detour / connector, not CA-1"
    return "CA-1 / PCH where available, US-101/local connector where necessary"
