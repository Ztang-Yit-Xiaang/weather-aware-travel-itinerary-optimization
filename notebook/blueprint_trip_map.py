import ast
import html
import json
import math
from pathlib import Path

import folium
import numpy as np
import pandas as pd
import requests
from folium import plugins


CITY_COORDS = {
    "San Diego": (32.7157, -117.1611),
    "Los Angeles": (34.0522, -118.2437),
    "Santa Barbara": (34.4208, -119.6982),
    "San Luis Obispo": (35.2828, -120.6596),
    "Monterey": (36.6002, -121.8947),
    "Santa Cruz": (36.9741, -122.0308),
    "San Francisco": (37.7749, -122.4194),
}

SCENIC_CA1_COLOR = "#2A9D8F"
FASTEST_ROUTE_COLOR = "#6C757D"
STANFORD_WAYPOINT_NAME = "Stanford University Main Quad"
ROUTE_CONTEXT_PANE = "routeContextPane"
ROUTE_CORE_PANE = "routeCorePane"
ROUTE_TOP_PANE = "routeTopPane"
FULL_SCENE_OFFSET_LAT_STEP = 0.009
FULL_SCENE_OFFSET_LON_STEP = 0.016

DAY_COLORS = ["#2563EB", "#4C78A8", "#F4A261", "#7A5195", "#C1121F", "#5FAD56", "#6C757D"]

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


def _escape(value):
    return html.escape(str(value))


def _load_csv(path):
    path = Path(path)
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def _first_existing(context, *keys, default=None):
    for key in keys:
        if key in context:
            return context[key]
    return default


def _catalog_series(output, column, default):
    if column in output.columns:
        return output[column]
    return pd.Series(default, index=output.index)


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
            ]
        )

    output = df.copy()
    output["city"] = output.get("city", city)
    output["source"] = output.get("source", source_name)
    output["source_list"] = output.get("source_list", output["source"])
    output["category"] = output.get("category", output.get("categories", "attraction"))
    output["source_score"] = pd.to_numeric(
        _catalog_series(output, "source_score", _catalog_series(output, "utility", _catalog_series(output, "review_count", 1.0))),
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
    output["must_go_weight"] = pd.to_numeric(_catalog_series(output, "must_go_weight", output["social_must_go"].astype(float)), errors="coerce").fillna(0.0)
    output["corridor_fit"] = pd.to_numeric(_catalog_series(output, "corridor_fit", 0.0), errors="coerce").fillna(0.0)
    output["detour_minutes"] = pd.to_numeric(_catalog_series(output, "detour_minutes", 0.0), errors="coerce").fillna(0.0)
    output["data_confidence"] = pd.to_numeric(_catalog_series(output, "data_confidence", 0.5), errors="coerce").fillna(0.5)
    output["final_poi_value"] = pd.to_numeric(
        _catalog_series(output, "final_poi_value", output["source_score"] + output["social_score"] * 3.0),
        errors="coerce",
    ).fillna(0.0)
    output["latitude"] = pd.to_numeric(output["latitude"], errors="coerce")
    output["longitude"] = pd.to_numeric(output["longitude"], errors="coerce")
    output = output.dropna(subset=["name", "latitude", "longitude"])
    return output[
        [
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
    ]


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
    return float(np.sqrt(((float(left_lat) - float(right_lat)) * lat_scale) ** 2 + ((float(left_lon) - float(right_lon)) * lon_scale) ** 2))


def _order_stops_from_hotel(stops, hotel_lat, hotel_lon):
    if stops.empty:
        return stops

    remaining = stops.copy().reset_index(drop=True)
    ordered_rows = []
    current_lat = float(hotel_lat)
    current_lon = float(hotel_lon)

    while not remaining.empty:
        distances = remaining.apply(
            lambda row: _point_distance_km(current_lat, current_lon, row["latitude"], row["longitude"]),
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
    available["source_score"] = pd.to_numeric(available.get("source_score", 1.0), errors="coerce").fillna(1.0)
    available["social_score"] = pd.to_numeric(available.get("social_score", 0.0), errors="coerce").fillna(0.0)
    available["must_go_weight"] = pd.to_numeric(available.get("must_go_weight", 0.0), errors="coerce").fillna(0.0)
    available["corridor_fit"] = pd.to_numeric(available.get("corridor_fit", 0.0), errors="coerce").fillna(0.0)
    available["detour_minutes"] = pd.to_numeric(available.get("detour_minutes", 0.0), errors="coerce").fillna(0.0)
    available["final_poi_value"] = pd.to_numeric(
        available.get("final_poi_value", available["source_score"] + available["social_score"] * 3.0),
        errors="coerce",
    ).fillna(0.0)
    available["social_must_go"] = available.get("social_must_go", False)
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
            lambda row: _point_distance_km(anchor["latitude"], anchor["longitude"], row["latitude"], row["longitude"]),
            axis=1,
        )
        pool["compact_rank_score"] = (
            pool["source_score_norm"] * 1.05
            + pool["social_score"] * float(profile_config.get("social_weight", 1.15))
            + pool["must_go_weight"] * pool["social_score"] * float(profile_config.get("must_go_bonus", 0.65))
            + pool["corridor_fit"] * 0.18
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
            + float((cluster["must_go_weight"] * cluster["social_score"]).mean()) * (float(profile_config.get("must_go_bonus", 0.65)) + 0.10)
            + float(cluster["corridor_fit"].mean()) * 0.18
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
    ]
    for column in output_columns:
        if column not in ordered.columns:
            if column == "social_reason":
                ordered[column] = ""
            elif column == "source_list":
                ordered[column] = ordered.get("source", "unknown")
            else:
                ordered[column] = 0.0
    return ordered[output_columns]


def _select_hotel_for_city_plan(hotel_catalog, poi_catalog, used_names, stops_per_day=3, profile_config=None):
    profile_config = profile_config or _profile_config("balanced", stops_per_day=stops_per_day)
    if hotel_catalog.empty:
        return hotel_catalog.iloc[0]

    hotels = hotel_catalog.copy().head(14).reset_index(drop=True)
    hotels["rating_score"] = pd.to_numeric(hotels.get("rating_score", 0.0), errors="coerce").fillna(0.0)
    hotels["nightly_price"] = pd.to_numeric(hotels.get("nightly_price", np.nan), errors="coerce")
    rating_span = max(float(hotels["rating_score"].max() - hotels["rating_score"].min()), 1e-6)
    price_filled = hotels["nightly_price"].fillna(hotels["nightly_price"].median() if hotels["nightly_price"].notna().any() else 0.0)
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
            lambda row: _point_distance_km(hotel["latitude"], hotel["longitude"], row["latitude"], row["longitude"]),
            axis=1,
        )
        score = (
            float(stops["source_score"].mean()) * 0.16
            + float(stops.get("social_score", pd.Series(0.0, index=stops.index)).mean()) * (1.5 + float(profile_config.get("social_weight", 1.15)))
            + float(stops.get("social_must_go", pd.Series(False, index=stops.index)).astype(float).mean()) * (0.6 + float(profile_config.get("must_go_bonus", 0.65)))
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
        corridor_social_mask = (
            cached_enriched.get("corridor_fit", pd.Series(0.0, index=cached_enriched.index)).astype(float).gt(0.70)
            & cached_enriched.get("social_must_go", pd.Series(False, index=cached_enriched.index)).astype(bool)
            & include_corridor_social
        )
        city_rows = cached_enriched[
            cached_enriched["city"].astype(str).str.lower().eq(city.lower())
            | corridor_social_mask
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

    city_frame["social_score"] = pd.to_numeric(city_frame.get("social_score", 0.0), errors="coerce").fillna(0.0)
    city_frame["source_score"] = pd.to_numeric(city_frame.get("source_score", 0.0), errors="coerce").fillna(0.0)
    city_frame["final_poi_value"] = pd.to_numeric(
        city_frame.get("final_poi_value", city_frame["source_score"] + 3.0 * city_frame["social_score"]),
        errors="coerce",
    ).fillna(0.0)
    city_frame["ranking_score"] = city_frame["final_poi_value"]
    return city_frame.sort_values(["ranking_score", "source_score", "name"], ascending=[False, False, True]).drop_duplicates("name")


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
        local_hotels["rating_score"] = pd.to_numeric(local_hotels.get("rating_score", local_hotels.get("stars", 0)), errors="coerce").fillna(0)
        local_hotels["nightly_price"] = pd.to_numeric(local_hotels.get("nightly_price", 0), errors="coerce").fillna(0)
        local_hotels["type_priority"] = local_hotels.get("type", "hotel").map({"hotel": 0, "motel": 1, "guest_house": 2, "hostel": 3, "apartment": 4}).fillna(5)
        local_hotels["distance_to_center_km"] = _distance_to_city_center(city, local_hotels["latitude"], local_hotels["longitude"])
        hotel_frames.append(local_hotels)

    if not cached_osm.empty:
        city_rows = cached_osm[cached_osm["city"].astype(str).str.lower().eq(city.lower())].copy()
        if not city_rows.empty:
            city_rows["rating_score"] = pd.to_numeric(city_rows.get("stars", 0), errors="coerce").fillna(0)
            city_rows["nightly_price"] = np.nan
            city_rows["source"] = city_rows.get("source", "openstreetmap_overpass_cache")
            type_priority = {"hotel": 0, "motel": 1, "guest_house": 2, "hostel": 3, "apartment": 4}
            city_rows["type_priority"] = city_rows.get("type", "hotel").map(type_priority).fillna(5)
            city_rows["distance_to_center_km"] = _distance_to_city_center(city, city_rows["latitude"], city_rows["longitude"])
            hotel_frames.append(city_rows)

    if hotel_frames:
        hotels = pd.concat(hotel_frames, ignore_index=True, sort=False)
        hotels["rating_score"] = pd.to_numeric(hotels.get("rating_score", 0), errors="coerce").fillna(0)
        hotels["nightly_price"] = pd.to_numeric(hotels.get("nightly_price", np.nan), errors="coerce")
        hotels["type_priority"] = pd.to_numeric(hotels.get("type_priority", 5), errors="coerce").fillna(5)
        hotels["distance_to_center_km"] = pd.to_numeric(
            hotels.get("distance_to_center_km", _distance_to_city_center(city, hotels["latitude"], hotels["longitude"])),
            errors="coerce",
        ).fillna(999)
        return hotels.sort_values(
            ["rating_score", "type_priority", "distance_to_center_km", "nightly_price", "name"],
            ascending=[False, True, True, True, True],
        ).drop_duplicates("name").reset_index(drop=True)

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
            scenic_rows = legs_df[legs_df.get("route_layer", "").astype(str).str.contains("Scenic", case=False, na=False)]
            total = 0.0
            matched = 0
            for left, right in zip(segment_cities[:-1], segment_cities[1:]):
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
    for left, right in zip(scenic_sequence[:-1], scenic_sequence[1:]):
        waypoints = _scenic_leg_waypoints(left, right)
        classification = _leg_classification(left, right, scenic=True)
        total += _estimate_drive_minutes(waypoints, classification)
    return total, "curated_waypoint_speed_limit_proxy"


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

        catalog["final_poi_value"] = pd.to_numeric(catalog.get("final_poi_value", 0), errors="coerce").fillna(0)
        catalog["social_score"] = pd.to_numeric(catalog.get("social_score", 0), errors="coerce").fillna(0)
        catalog["corridor_fit"] = pd.to_numeric(catalog.get("corridor_fit", 0), errors="coerce").fillna(0)
        catalog["detour_minutes"] = pd.to_numeric(catalog.get("detour_minutes", 0), errors="coerce").fillna(0)
        if "social_must_go" in catalog.columns:
            catalog["social_must_go"] = catalog["social_must_go"].fillna(False).astype(bool)
        else:
            catalog["social_must_go"] = False
        catalog["transition_stop_score"] = (
            catalog["final_poi_value"] * float(profile_config.get("value_weight", 1.0))
            + catalog["social_score"] * float(profile_config.get("social_weight", 0.6))
            + catalog["corridor_fit"] * 0.35
            + catalog["social_must_go"].astype(float) * 0.45
            + catalog["category"].astype(str).str.contains("view|bridge|beach|waterfront|landmark|scenic|observatory", case=False, na=False).astype(float) * 0.35
            - catalog["detour_minutes"] * 0.006
        )

        stop = catalog.sort_values(["transition_stop_score", "final_poi_value", "name"], ascending=[False, False, True]).iloc[0]
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
    pool = enriched[
        enriched["city"].astype(str).isin(segment_cities)
        | pd.to_numeric(enriched.get("corridor_fit", 0.0), errors="coerce").fillna(0.0).ge(0.75)
    ].copy()
    if pool.empty:
        return pd.DataFrame()
    used_names = set()
    for names in used_names_by_city.values():
        used_names.update(str(name) for name in names)
    pool = pool[~pool["name"].astype(str).isin(used_names)].copy()
    if pool.empty:
        return pd.DataFrame()
    pool["final_poi_value"] = pd.to_numeric(pool.get("final_poi_value", 0.0), errors="coerce").fillna(0.0)
    pool["social_score"] = pd.to_numeric(pool.get("social_score", 0.0), errors="coerce").fillna(0.0)
    pool["corridor_fit"] = pd.to_numeric(pool.get("corridor_fit", 0.0), errors="coerce").fillna(0.0)
    pool["detour_minutes"] = pd.to_numeric(pool.get("detour_minutes", 0.0), errors="coerce").fillna(0.0)
    pool["view_signal"] = pool["category"].astype(str).str.contains(
        "view|bridge|beach|waterfront|landmark|scenic|observatory|campus",
        case=False,
        na=False,
    ).astype(float)
    pool["cross_city_view_score"] = (
        pool["final_poi_value"] * float(profile_config.get("value_weight", 1.0))
        + pool["social_score"] * float(profile_config.get("social_weight", 0.6))
        + pool["corridor_fit"] * 0.35
        + pool["view_signal"] * 0.55
        - pool["detour_minutes"] * 0.006
    )
    selected = pool.sort_values(["cross_city_view_score", "final_poi_value"], ascending=False).head(max_stops).copy()
    for row in selected.itertuples(index=False):
        used_names_by_city.setdefault(str(row.city), set()).add(str(row.name))
    return _normalize_catalog_columns(selected, "corridor", "multi_city_high_utility_view")


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
            source_score = pd.to_numeric(stanford_rows.get("source_score", 0.0), errors="coerce").fillna(0.0)
            social_score = pd.to_numeric(stanford_rows.get("social_score", 0.0), errors="coerce").fillna(0.0)
            stanford_rows["final_poi_value"] = source_score * 0.15 + social_score * 0.90
    if stanford_rows.empty:
        return pd.DataFrame()
    stanford_rows["city"] = "Palo Alto / Stanford"
    used_names.add(STANFORD_WAYPOINT_NAME)
    return _normalize_catalog_columns(stanford_rows, "Palo Alto / Stanford", "curated_social_must_go")


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
                drive_minutes, drive_time_source = _transition_drive_minutes(context, previous_city, city, pass_through_cities)
                route_type = "relocation_pass_through" if pass_through_cities else "relocation_direct"

            available_visit_minutes = max(60.0, 720.0 - float(drive_minutes) - 45.0)
            if is_transition_day:
                stanford_on_route = bool(previous_city == "San Francisco" and "Santa Cruz" in pass_through_cities)
                transition_stop_minutes = 25.0
                transition_stop_capacity = max(stops_per_day, len(pass_through_cities) + int(stanford_on_route))
                max_transition_stops = min(transition_stop_capacity, max(1, int(available_visit_minutes // transition_stop_minutes)))
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
                    }
                )
            day_number += 1

    return pd.DataFrame(rows)


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
    available["source_score"] = pd.to_numeric(available.get("source_score", 0.0), errors="coerce").fillna(0.0)
    available["social_score"] = pd.to_numeric(available.get("social_score", 0.0), errors="coerce").fillna(0.0)
    available["social_must_go"] = available.get("social_must_go", False)
    available["social_must_go"] = available["social_must_go"].fillna(False).astype(bool)
    available["must_go_weight"] = pd.to_numeric(available.get("must_go_weight", 0.0), errors="coerce").fillna(0.0)
    available["corridor_fit"] = pd.to_numeric(available.get("corridor_fit", 0.0), errors="coerce").fillna(0.0)
    available["detour_minutes"] = pd.to_numeric(available.get("detour_minutes", 0.0), errors="coerce").fillna(0.0)
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
            lambda row: float(row["final_poi_value"])
            + 0.12 * float(row["social_score"])
            + 0.08 * float(row["corridor_fit"])
            - 0.02 * _point_distance_km(current_lat, current_lon, row["latitude"], row["longitude"])
            - 0.004 * float(row["detour_minutes"]),
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
    ]
    for column in output_columns:
        if column not in selected.columns:
            if column == "social_reason":
                selected[column] = ""
            elif column == "source_list":
                selected[column] = selected.get("source", "unknown")
            else:
                selected[column] = 0.0
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
        catalog["final_poi_value"] = pd.to_numeric(catalog.get("final_poi_value", 0.0), errors="coerce").fillna(0.0)
        catalog["source_score"] = pd.to_numeric(catalog.get("source_score", 0.0), errors="coerce").fillna(0.0)
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
                drive_minutes, drive_time_source = _transition_drive_minutes(context, previous_city, city, pass_through_cities)
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
                stops = transition_stops if not transition_stops.empty else _select_greedy_day_stops(
                    poi_catalog,
                    hotel,
                    used_names,
                    stops_per_day=max(1, min(stops_per_day, int(available_visit_minutes // 75.0) or 1)),
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
                    }
                )
            day_number += 1

    return pd.DataFrame(rows)


def build_profile_day_plans(context):
    return {
        profile_name: build_day_plan(context, profile_name=profile_name)
        for profile_name in PROFILE_CONFIGS
    }


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
        for left, right in zip(route_points[:-1], route_points[1:]):
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
    for left, right in zip(points[:-1], points[1:]):
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
    for left, right in zip(points[:-1], points[1:]):
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
    for leg_idx, (left, right) in enumerate(zip(clean_points[:-1], clean_points[1:]), start=1):
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
        route_line = folium.PolyLine(
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


def _route_points_for_names(names):
    return [_coord_for_place(name) for name in names]


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
        for left, right in zip(points[:-1], points[1:])
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


def _add_intercity_route_layer(map_object, route_name, sequence, color, route_cache, run_live, scenic=False, show=True):
    layer = folium.FeatureGroup(name=route_name, show=show)
    leg_rows = []
    all_path_points = []

    for leg_order, (left_name, right_name) in enumerate(zip(sequence[:-1], sequence[1:]), start=1):
        classification = _leg_classification(left_name, right_name, scenic)
        waypoints = _scenic_leg_waypoints(left_name, right_name) if scenic else _route_points_for_names([left_name, right_name])
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
        route_line = folium.PolyLine(
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


def _build_profile_comparison(profile_day_plans):
    rows = []
    for profile_name, plan_df in profile_day_plans.items():
        config = PROFILE_CONFIGS[profile_name]
        if plan_df.empty:
            hotel_summary = "n/a"
            stop_count = 0
            must_go_count = 0
        else:
            city_column = "overnight_city" if "overnight_city" in plan_df.columns else "city"
            bases = plan_df[["day", city_column, "hotel_name"]].drop_duplicates()
            hotel_summary = ", ".join(
                f"D{int(row.day)} {getattr(row, city_column)}: {row.hotel_name}" for row in bases.itertuples(index=False)
            )
            stop_count = int(len(plan_df))
            must_go_count = int(plan_df["social_must_go"].astype(bool).sum())
        rows.append(
            f"""
            <tr>
                <td>{_escape(config["label"])}</td>
                <td>{stop_count}</td>
                <td>{must_go_count}</td>
                <td>{_escape(config["route_mode"])}</td>
                <td>{_escape(hotel_summary)}</td>
            </tr>
            """
        )
    return "".join(rows)


def _finite_float(value, default=np.nan):
    try:
        value = float(value)
    except Exception:
        return default
    return value if np.isfinite(value) else default


def _build_method_comparison(output_dir):
    output_dir = Path(output_dir)
    method_df = _load_csv(output_dir / "production_method_comparison.csv")
    if not method_df.empty and "method" in method_df.columns:
        return method_df
    return pd.DataFrame()


def _build_method_comparison_html(method_comparison_df):
    if method_comparison_df.empty:
        return "<div class=\"panel-section-title\">Method Comparison</div><div class=\"summary-line\">Method comparison not generated yet.</div>"

    if "method" in method_comparison_df.columns and "total_utility" in method_comparison_df.columns:
        rows = []
        for row in method_comparison_df.itertuples(index=False):
            utility = _escape(f"{float(getattr(row, 'total_utility', np.nan)):.2f}" if np.isfinite(_finite_float(getattr(row, "total_utility", np.nan))) else "n/a")
            travel_minutes = _finite_float(getattr(row, "total_travel_time", np.nan))
            travel = _escape(f"{travel_minutes / 60.0:.1f}h" if np.isfinite(travel_minutes) else "n/a")
            budget = _escape(f"${float(getattr(row, 'budget_used', getattr(row, 'total_cost', np.nan))):,.0f}" if np.isfinite(_finite_float(getattr(row, "budget_used", getattr(row, "total_cost", np.nan)))) else "n/a")
            stops = _escape(f"{int(getattr(row, 'selected_attractions', 0))}" if np.isfinite(_finite_float(getattr(row, "selected_attractions", np.nan))) else "n/a")
            status = _escape(str(getattr(row, "status", "n/a")))
            score_value = _finite_float(getattr(row, "comparison_score", np.nan))
            score = _escape(f"{score_value:.3f}" if np.isfinite(score_value) else "n/a")
            solvers = _escape(
                f"{getattr(row, 'allocation_solver', 'n/a')} / {getattr(row, 'local_route_solver', 'n/a')}"
            )
            display_name = _escape(str(getattr(row, "method_display_name", getattr(row, "method", "unknown"))))
            rows.append(
                f"""
                <tr>
                    <td>{display_name}</td>
                    <td>{score}</td>
                    <td>{utility}</td>
                    <td>{stops}</td>
                    <td>{travel}</td>
                    <td>{budget}</td>
                    <td>{solvers}</td>
                    <td>{status}</td>
                </tr>
                """
            )
        return f"""
        <div class="panel-section-title">Method Comparison</div>
        <table class="method-table">
            <thead><tr><th>Method</th><th>Score</th><th>Utility</th><th>Stops</th><th>Travel</th><th>Cost</th><th>Solvers</th><th>Status</th></tr></thead>
            <tbody>{''.join(rows)}</tbody>
        </table>
        <div class="summary-line muted-note"><b>Metric math:</b> U=sum selected POI values; T=sum route travel minutes; C=hotel+transport+POI cost; M=count of must-go stops; H=normalized Shannon category diversity. Score=0.40U' + 0.15M' + 0.15H + 0.15T' + 0.10C' + 0.05W', with lower travel/cost/wait inverted.</div>
        <div class="summary-line muted-note">Full comparison saved to production_method_comparison.csv with matching route stops in production_method_route_stops.csv.</div>
        """
    return "<div class=\"panel-section-title\">Method Comparison</div><div class=\"summary-line\">Method comparison not generated yet.</div>"


def _build_trip_length_comparison_html(output_dir):
    comparison_df = _load_csv(Path(output_dir) / "production_trip_length_comparison.csv")
    if comparison_df.empty:
        return "<div class=\"panel-section-title\">Trip Length Comparison</div><div class=\"summary-line\">Trip-length comparison not generated yet.</div>"
    rows = []
    for row in comparison_df.itertuples(index=False):
        drive_hours = _finite_float(getattr(row, "intercity_drive_hours", np.nan))
        budget = _finite_float(getattr(row, "estimated_budget", np.nan))
        status = _escape(str(getattr(row, "status", getattr(row, "solver_status", "n/a"))))
        drive_text = f"{drive_hours:.1f}h" if np.isfinite(drive_hours) else "n/a"
        budget_text = f"${budget:,.0f}" if np.isfinite(budget) else "n/a"
        rows.append(
            f"""
            <tr>
                <td>{int(row.trip_days)}</td>
                <td>{_escape(row.gateway_start)} → {_escape(row.gateway_end)}</td>
                <td>{_escape(row.days_by_city)}</td>
                <td>{_escape(drive_text)}</td>
                <td>{_escape(budget_text)}</td>
                <td>{status}</td>
            </tr>
            """
        )
    return f"""
    <div class="panel-section-title">Trip Length Comparison</div>
    <table class="method-table">
        <thead><tr><th>Days</th><th>Gateway</th><th>Cities/Bases</th><th>Drive</th><th>Budget</th><th>Status</th></tr></thead>
        <tbody>{''.join(rows)}</tbody>
    </table>
    <div class="summary-line muted-note">Saved to production_trip_length_comparison.csv with map-ready stops in production_trip_length_route_stops.csv.</div>
    """


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
    stop_df["comparison_stop_order"] = pd.to_numeric(stop_df.get("comparison_stop_order", range(1, len(stop_df) + 1)), errors="coerce").fillna(1).astype(int)
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
    for left, right in zip(clean_points[:-1], clean_points[1:]):
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
    target_distances = np.linspace(total_distance / (target_count + 1), total_distance * target_count / (target_count + 1), target_count)
    samples = []
    walked = 0.0
    target_index = 0
    for left, right in zip(clean_points[:-1], clean_points[1:]):
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


def _add_canvas_direction_arrows(layer, points, *, color, pane=ROUTE_TOP_PANE, interval_km=95.0, max_arrows=9, class_name="blueprint-route-arrow"):
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

    route_line = folium.PolyLine(
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
    local_rows["stop_order"] = pd.to_numeric(local_rows.get("stop_order", 1), errors="coerce").fillna(1).astype(int)
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
        transition_rows["stop_order"] = pd.to_numeric(transition_rows.get("stop_order", 1), errors="coerce").fillna(1).astype(int)
        transition_rows = transition_rows.sort_values(["stop_order", "attraction_name"]).reset_index(drop=True)
    stop_points = transition_rows[["latitude", "longitude"]].dropna().astype(float).values.tolist() if not transition_rows.empty else []
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
    start_city = str(first.get("route_start_city", first.get("gateway_start", first.get("city", fallback_start))) or fallback_start)
    end_city = str(last.get("route_end_city", last.get("gateway_end", last.get("overnight_city", last.get("city", fallback_end)))) or fallback_end)
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
        status = "FAILED"
        issue = "not enough unique finite coordinates to draw a route"
    elif source_rows and distance_km < 0.05:
        status = "FAILED"
        issue = "route collapsed to a single point"
    elif str(route_type).startswith("relocation") and distance_km < 50.0:
        status = "WARN"
        issue = "relocation day has unexpectedly short geometry"
    elif fallback_count and str(comparison_type) in {"intercity", "traveler_day", "city_detail", "trip_length", "method", "traveler_profile"}:
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


def _build_route_debug_summary_html(route_debug_df):
    if route_debug_df is None or route_debug_df.empty:
        return """
        <div class="panel-section-title">Map Route Validation</div>
        <div class="summary-line">No route validation rows were generated.</div>
        """
    status_counts = route_debug_df["draw_status"].astype(str).value_counts().to_dict()
    ok_count = int(status_counts.get("OK", 0))
    warn_count = int(status_counts.get("WARN", 0))
    failed_count = int(status_counts.get("FAILED", 0))
    visible_count = int(route_debug_df["show_by_default"].astype(bool).sum())
    geometry_counts = (
        route_debug_df.get("geometry_source", pd.Series(dtype=str))
        .fillna("unknown")
        .astype(str)
        .value_counts()
        .head(5)
        .to_dict()
    )
    fallback_count = int(pd.to_numeric(route_debug_df.get("straight_line_fallback_count", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
    road_percent = pd.to_numeric(route_debug_df.get("road_geometry_percent", pd.Series(dtype=float)), errors="coerce")
    road_percent_text = f"{float(road_percent.dropna().mean()):.1f}%" if not road_percent.dropna().empty else "n/a"
    groups = ", ".join(
        f"{group}: {count}"
        for group, count in route_debug_df["layer_group"].astype(str).value_counts().sort_index().items()
    )
    problem_df = route_debug_df[route_debug_df["draw_status"].astype(str).ne("OK")].head(4)
    if problem_df.empty:
        problem_html = "<div class=\"summary-line muted-note\">All audited route layers have drawable geometry.</div>"
    else:
        problem_items = "".join(
            f"<li>{_escape(row.layer_name)}: {_escape(row.issue)}</li>"
            for row in problem_df.itertuples(index=False)
        )
        problem_html = f"<ol>{problem_items}</ol>"
    table_rows = []
    for group, group_df in route_debug_df.groupby("layer_group", sort=True):
        visible = int(group_df["show_by_default"].astype(bool).sum())
        max_distance = _finite_float(pd.to_numeric(group_df["distance_km"], errors="coerce").max(), 0.0)
        max_points = int(pd.to_numeric(group_df["unique_points"], errors="coerce").fillna(0).max())
        layer_vars = []
        if "layer_var" in group_df.columns:
            layer_vars = [
                str(value)
                for value in group_df["layer_var"].dropna().astype(str).unique().tolist()
                if value and value.lower() != "nan"
            ]
        layer_var_text = ", ".join(layer_vars[:2]) if layer_vars else "HTML audit"
        html_state = "added; remove checked by validator" if visible else "toggle-only"
        table_rows.append(
            f"""
            <tr>
                <td>{_escape(group)}</td>
                <td>{len(group_df)}</td>
                <td>{visible}</td>
                <td>{max_distance:.1f} km</td>
                <td>{max_points}</td>
                <td>{_escape(layer_var_text)}</td>
                <td>{_escape(html_state)}</td>
            </tr>
            """
        )
    family_table_html = f"""
    <table class="method-table">
        <thead><tr><th>Route family</th><th>Rows</th><th>Visible</th><th>Distance</th><th>Pts</th><th>Layer var</th><th>HTML state</th></tr></thead>
        <tbody>{''.join(table_rows)}</tbody>
    </table>
    """
    return f"""
    <div class="panel-section-title">Map Route Validation</div>
    <div class="summary-line"><b>Audited routes:</b> {len(route_debug_df)} ({ok_count} OK, {warn_count} warnings, {failed_count} failed)</div>
    <div class="summary-line"><b>Visible by default:</b> {visible_count} route layers</div>
    <div class="summary-line"><b>Road geometry:</b> avg road/cached coverage {road_percent_text}; fallback flags {fallback_count}; sources {_escape(', '.join(f'{k}: {v}' for k, v in geometry_counts.items()))}</div>
    <div class="summary-line muted-note">Debug rows saved to production_map_route_debug.csv. Layer groups: {_escape(groups)}</div>
    {family_table_html}
    {problem_html}
    """


def _add_selected_result_layer(map_object, day_plan_df, route_cache=None, run_live=False, route_debug_rows=None, show_by_default=True):
    if day_plan_df is None or day_plan_df.empty:
        return None

    layer = folium.FeatureGroup(
        name="Selected Result · Default Hierarchical Route",
        show=bool(show_by_default),
    )

    day_frame = day_plan_df.copy()
    day_frame["day"] = pd.to_numeric(day_frame.get("day", 1), errors="coerce").fillna(1).astype(int)
    day_frame["stop_order"] = pd.to_numeric(day_frame.get("stop_order", 1), errors="coerce").fillna(1).astype(int)
    day_frame = day_frame.sort_values(["day", "stop_order", "attraction_name"]).reset_index(drop=True)

    for day, group in day_frame.groupby("day", sort=True):
        color = DAY_COLORS[(int(day) - 1) % len(DAY_COLORS)]
        first = group.iloc[0]
        day_points = _route_points_from_day_group(group)
        draw_points, route_mode = _fetch_stitched_route(day_points, route_cache, run_live=run_live) if route_cache is not None else (day_points, "static-polyline")

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


def _add_model_comparison_layers(map_object, output_dir, route_cache, run_live, show_by_default=False, route_debug_rows=None):
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
        col
        for col in ["comparison_type", "comparison_label", "method", "trip_days"]
        if col in comparison_df.columns
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

    for group_values, layer_df in comparison_df.groupby(["comparison_type", "comparison_label", "method", "trip_days"], dropna=False):
        comparison_type, comparison_label, method, trip_days = group_values
        layer_name = layer_name_for_group(comparison_type, comparison_label, method, trip_days)
        color, dash_array, badge_prefix = style_for_group(comparison_type, method, trip_days)
        show_layer = bool(show_by_default)

        layer = folium.FeatureGroup(
            name=layer_name,
            show=show_layer,
        )
        day_frame = layer_df.copy()
        day_frame["day"] = pd.to_numeric(day_frame.get("day", 1), errors="coerce").fillna(1).astype(int)
        day_frame["stop_order"] = pd.to_numeric(day_frame.get("stop_order", 1), errors="coerce").fillna(1).astype(int)
        day_frame = day_frame.sort_values(["day", "stop_order", "attraction_name"]).reset_index(drop=True)
        for day, group in day_frame.groupby("day", sort=True):
            first = group.iloc[0]
            method_display_name = str(first.get("method_display_name", layer_name.replace("Method · ", "")))
            hotel_lat = _finite_float(first.get("hotel_latitude"))
            hotel_lon = _finite_float(first.get("hotel_longitude"))
            day_points = _route_points_from_day_group(group)
            draw_points, route_mode = _fetch_stitched_route(day_points, route_cache, run_live=run_live)
            popup_html = (
                f"<b>{_escape(layer_name)}</b><br/>"
                f"Day {int(day)}<br/>"
                f"Method: {_escape(method_display_name)}"
            )
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


def _add_traveler_overview_layers(map_object, profile_day_plans, route_cache=None, run_live=False, route_debug_rows=None, show_by_default=True):
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
        sorted_plan["day"] = pd.to_numeric(sorted_plan.get("day", 1), errors="coerce").fillna(1).astype(int)
        sorted_plan["stop_order"] = pd.to_numeric(sorted_plan.get("stop_order", 1), errors="coerce").fillna(1).astype(int)
        sorted_plan = sorted_plan.sort_values(["day", "stop_order", "attraction_name"]).reset_index(drop=True)

        full_points = []
        for _, group in sorted_plan.groupby("day", sort=True):
            day_points = _route_points_from_day_group(group)
            if full_points and day_points:
                full_points.extend(day_points[1:] if full_points[-1] == day_points[0] else day_points)
            else:
                full_points.extend(day_points)

        draw_points, route_mode = _fetch_stitched_route(full_points, route_cache, run_live=run_live) if route_cache is not None else (full_points, "static-polyline")
        geometry_mode = _add_static_result_line(
            layer,
            draw_points,
            color=color,
            dash_array=dash_array,
            tooltip=layer_name,
            popup_html=(
                f"<b>{_escape(layer_name)}</b><br/>"
                f"Stops: {len(sorted_plan)}<br/>"
                f"Mode: {_escape(config['route_mode'])}"
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
            route_end_city=str(sorted_plan.iloc[-1].get("route_end_city", sorted_plan.iloc[-1].get("overnight_city", ""))),
            points=draw_points,
            geometry_mode=geometry_mode,
            show_by_default=bool(show_by_default),
            source_rows=len(sorted_plan),
        )
        layer.add_to(map_object)
        layers.append(layer)
    return layers


def _city_detail_layer_label(day, first):
    route_type = str(first.get("route_type", "base_city_local"))
    overnight_city = str(first.get("overnight_city", first.get("city", "")))
    if route_type.startswith("relocation"):
        start_city = str(first.get("route_start_city", first.get("city", "")))
        end_city = str(first.get("route_end_city", overnight_city))
        return f"Transition Route · Day {int(day)} · {start_city} to {end_city}"
    return f"City Loop · {overnight_city}"


def _add_city_detail_layers(map_object, day_plan_df, route_cache, run_live, route_debug_rows=None, route_debug_registry=None):
    if day_plan_df is None or day_plan_df.empty:
        return [], []

    sorted_plan = day_plan_df.copy()
    sorted_plan["day"] = pd.to_numeric(sorted_plan.get("day", 1), errors="coerce").fillna(1).astype(int)
    sorted_plan["stop_order"] = pd.to_numeric(sorted_plan.get("stop_order", 1), errors="coerce").fillna(1).astype(int)
    sorted_plan = sorted_plan.sort_values(["day", "stop_order", "attraction_name"]).reset_index(drop=True)

    city_loop_groups = {}
    transition_groups = []
    for day, group in sorted_plan.groupby("day", sort=True):
        first = group.iloc[0]
        loop_points, local_rows = _hotel_loop_points_from_day_group(group)
        if loop_points and not local_rows.empty:
            city_name = str(first.get("overnight_city", first.get("city", "")))
            city_loop_groups.setdefault(f"City Loop · {city_name}", []).append((int(day), group, local_rows, loop_points))
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
                { _escape(layer_name) }<br/>
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
            (transition_points[0], f"Transition start: {first.get('route_start_name', start_city)}", "play", "cadetblue"),
            (transition_points[-1], f"Transition end hotel: {first.get('hotel_name', end_city)}", "flag-checkered", "darkred"),
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


def _add_route_matrix_layers(map_object, output_dir, route_cache, run_live, route_debug_rows=None, route_debug_registry=None):
    matrix_routes = _load_csv(Path(output_dir) / "production_route_matrix_route_stops.csv")
    if matrix_routes.empty or "route_key" not in matrix_routes.columns:
        return [], []

    matrix_routes = matrix_routes.copy()
    matrix_routes["trip_days"] = pd.to_numeric(matrix_routes.get("trip_days", 7), errors="coerce").fillna(7).astype(int)
    matrix_routes["day"] = pd.to_numeric(matrix_routes.get("day", 1), errors="coerce").fillna(1).astype(int)
    matrix_routes["stop_order"] = pd.to_numeric(matrix_routes.get("stop_order", 1), errors="coerce").fillna(1).astype(int)
    matrix_routes = matrix_routes.sort_values(["trip_days", "method", "profile", "day", "stop_order"]).reset_index(drop=True)

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
    for route_index, (route_key, group) in enumerate(matrix_routes.groupby("route_key", sort=False), start=1):
        group = group.sort_values(["day", "stop_order", "attraction_name"]).reset_index(drop=True)
        if group.empty:
            continue
        first = group.iloc[0]
        route_key = str(route_key)
        trip_days = int(first.get("trip_days", 7))
        method = str(first.get("method", "route"))
        profile = str(first.get("profile", "balanced"))
        profile_label = str(first.get("profile_label", profile.title()))
        method_label, _base_color, dash_array, _offset = MATRIX_METHOD_STYLES.get(method, ("Route", "#E91E63", None, 0.0))
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
                    "bounds": _route_bounds(hotel_rows[["hotel_latitude", "hotel_longitude"]].astype(float).values.tolist()) if not hotel_rows.empty else [],
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
            points=hotel_rows[["hotel_latitude", "hotel_longitude"]].astype(float).values.tolist() if not hotel_rows.empty else [],
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
                        "bounds": _route_bounds(route_social[["latitude", "longitude"]].dropna().astype(float).values.tolist()) if not route_social.empty else [],
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


def _full_route_points_from_plan(plan_df):
    if plan_df is None or plan_df.empty:
        return []
    sorted_plan = plan_df.copy()
    sorted_plan["day"] = pd.to_numeric(sorted_plan.get("day", 1), errors="coerce").fillna(1).astype(int)
    sorted_plan["stop_order"] = pd.to_numeric(sorted_plan.get("stop_order", 1), errors="coerce").fillna(1).astype(int)
    sorted_plan = sorted_plan.sort_values(["day", "stop_order", "attraction_name"]).reset_index(drop=True)
    full_points = []
    for _, group in sorted_plan.groupby("day", sort=True):
        day_points = _route_points_from_day_group(group)
        if full_points and day_points:
            full_points.extend(day_points[1:] if full_points[-1] == day_points[0] else day_points)
        else:
            full_points.extend(day_points)
    return _dedupe_route_points(full_points)


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
        for group_values, group in comparison_df.groupby(["comparison_type", "comparison_label", "method", "trip_days"], dropna=False):
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


def _build_html_data_source_summary(output_dir):
    output_dir = Path(output_dir)
    enriched = _load_csv(output_dir / "production_enriched_poi_catalog.csv")
    hotels = _load_csv(output_dir / "production_city_hotel_catalog.csv")
    day_plan = _load_csv(output_dir / "production_day_plan.csv")
    routes = _load_csv(output_dir / "production_intercity_legs.csv")
    source_counts = {}
    if not enriched.empty and "source_list" in enriched.columns:
        source_counts = (
            enriched["source_list"]
            .astype(str)
            .str.replace("|", ";", regex=False)
            .str.split(";")
            .explode()
            .str.strip()
            .value_counts()
            .head(4)
            .to_dict()
        )
    source_text = ", ".join(f"{key}: {value}" for key, value in source_counts.items()) if source_counts else "open/local catalog"
    route_source = "not generated"
    if not routes.empty and "geometry_source" in routes.columns:
        route_source = ", ".join(routes["geometry_source"].astype(str).drop_duplicates().head(3).tolist())
    return f"""
    <div class="panel-section-title">Data Used By This HTML</div>
    <div class="summary-line"><b>POIs:</b> {len(enriched)} enriched rows ({_escape(source_text)})</div>
    <div class="summary-line"><b>Hotels:</b> {len(hotels)} open/project hotel candidates; selected bases shown as red hotel pins.</div>
    <div class="summary-line"><b>Day plan:</b> {len(day_plan)} displayed stops from production_day_plan.csv.</div>
    <div class="summary-line"><b>Roads:</b> production_intercity_legs.csv + OSRM/cache geometry ({_escape(route_source)}).</div>
    <div class="summary-line"><b>Bandit/Gurobi:</b> production_hybrid_bandit_optimization_summary.csv and production_bandit_stress_summary.csv.</div>
    """


def _build_must_go_summary_html(output_dir):
    coverage = _load_csv(Path(output_dir) / "production_must_go_coverage.csv")
    candidates = _load_csv(Path(output_dir) / "production_social_must_go_candidates.csv")
    if coverage.empty and candidates.empty:
        return """
        <div class="panel-section-title">Must-Go Coverage</div>
        <div class="summary-line muted-note">Must-go coverage artifact not generated yet.</div>
        """
    if not coverage.empty:
        balanced = coverage[coverage["method"].astype(str).eq("hierarchical_gurobi_pipeline")].copy()
        if balanced.empty:
            balanced = coverage.copy()
        total = int(balanced["must_go_name"].nunique())
        selected = int(balanced[balanced["selected"].astype(bool)]["must_go_name"].nunique())
        skipped_names = sorted(balanced.loc[~balanced["selected"].astype(bool), "must_go_name"].dropna().astype(str).unique().tolist())
        policy = str(balanced.get("must_go_policy", pd.Series(["soft_reward_not_mandatory"])).dropna().iloc[0])
    else:
        total = int(candidates["name"].nunique())
        selected = 0
        skipped_names = sorted(candidates["name"].dropna().astype(str).unique().tolist())
        policy = "soft_reward_not_mandatory"
    skipped_text = "; ".join(skipped_names[:6]) if skipped_names else "none"
    if len(skipped_names) > 6:
        skipped_text += f"; +{len(skipped_names) - 6} more"
    return f"""
    <div class="panel-section-title">Must-Go Coverage</div>
    <div class="summary-line"><b>Policy:</b> {_escape(policy)}</div>
    <div class="summary-line"><b>Selected:</b> {selected} of {total}</div>
    <div class="summary-line"><b>Skipped:</b> {_escape(skipped_text)}</div>
    """


def _build_hotel_summary_html(output_dir):
    hotel_debug = _load_csv(Path(output_dir) / "production_hotel_selection_debug.csv")
    if hotel_debug.empty:
        return """
        <div class="panel-section-title">Hotel Choices</div>
        <div class="summary-line muted-note">Hotel selection debug artifact not generated yet.</div>
        """
    selected = hotel_debug[hotel_debug["selected"].astype(bool)].copy()
    rows = []
    for row in selected.head(8).itertuples(index=False):
        score = _finite_float(getattr(row, "hotel_score", np.nan))
        rows.append(
            f"""
            <tr>
                <td>{_escape(getattr(row, "city", ""))}</td>
                <td>{_escape(getattr(row, "hotel_name", ""))}</td>
                <td>{_escape(f"{score:.2f}" if np.isfinite(score) else "n/a")}</td>
            </tr>
            """
        )
    body = "".join(rows) if rows else "<tr><td colspan=\"3\">No selected hotel rows found.</td></tr>"
    return f"""
    <div class="panel-section-title">Hotel Choices</div>
    <table>
        <thead><tr><th>City</th><th>Selected hotel/base</th><th>Score</th></tr></thead>
        <tbody>{body}</tbody>
    </table>
    """


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


def _add_route_debug_controls(map_object, route_registry, hidden_default_layers=None):
    if not route_registry:
        return

    map_var = map_object.get_name()
    hidden_default_layers = hidden_default_layers or []
    hidden_vars = [layer.get_name() for layer in hidden_default_layers if layer is not None and hasattr(layer, "get_name")]
    bounds_points = []
    for route in route_registry:
        bounds = route.get("bounds") or []
        if len(bounds) == 2:
            bounds_points.extend(bounds)
    registry_json = json.dumps(route_registry).replace("</", "<\\/")
    hidden_json = json.dumps(hidden_vars)
    bounds_json = json.dumps(bounds_points)
    families = sorted({str(route.get("family", "")) for route in route_registry})
    checked_count = sum(1 for route in route_registry if bool(route.get("default_checked", route.get("default_visible", False))))
    matrix_full_routes = [
        route for route in route_registry
        if str(route.get("selector_group")) == "route_matrix" and str(route.get("layer_role")) == "full_route"
    ]

    def _route_checkbox_html(route, *, compact=False):
        color = _escape(str(route.get("color", "#2563EB")))
        control_id = _escape(str(route.get("control_id", route.get("label", "route"))))
        control_label = _escape(str(route.get("control_label", route.get("label", "route"))))
        family = _escape(str(route.get("family", "route")))
        checked = " checked" if bool(route.get("default_checked", route.get("default_visible", False))) else ""
        distance = _finite_float(route.get("distance_km"), 0.0)
        unique_points = int(route.get("unique_points", 0) or 0)
        meta_text = f"{distance:.0f} km" if np.isfinite(distance) and distance > 0 else f"{unique_points} pts" if unique_points else ""
        compact_class = " blueprint-route-check-compact" if compact else ""
        return f"""
            <label class="blueprint-route-check{compact_class}" data-route-family="{family}">
                <input type="checkbox" data-route-checkbox="{control_id}" data-route-family="{family}"{checked}>
                <span class="blueprint-route-swatch" style="background:{color};"></span>
                <span class="blueprint-route-check-text">{control_label}</span>
                <span class="blueprint-route-meta">{_escape(meta_text)}</span>
            </label>
            """

    if matrix_full_routes:
        route_by_key = {str(route.get("route_key")): route for route in matrix_full_routes}
        child_routes = {}
        for route in route_registry:
            role = str(route.get("layer_role", ""))
            if str(route.get("selector_group")) != "route_matrix" or role in {"full_route", "selected_hotels"}:
                continue
            parent_key = str(route.get("parent_route_key", route.get("route_key", "")))
            child_routes.setdefault(parent_key, []).append(route)
        day_values = sorted({int(_finite_float(route.get("trip_days"), 0)) for route in matrix_full_routes if int(_finite_float(route.get("trip_days"), 0)) > 0})
        method_order = ["hierarchical_gurobi_pipeline", "hierarchical_greedy_baseline", "hierarchical_bandit_gurobi_repair"]
        profile_order = ["relaxed", "balanced", "explorer"]
        matrix_html = []
        for day_value in day_values:
            day_routes = [route for route in matrix_full_routes if int(_finite_float(route.get("trip_days"), 0)) == day_value]
            method_blocks = []
            for method in method_order:
                method_routes = [route for route in day_routes if str(route.get("method")) == method]
                if not method_routes:
                    continue
                method_label = str(method_routes[0].get("method_label", method_routes[0].get("method", "Method")))
                method_routes = sorted(method_routes, key=lambda route: profile_order.index(str(route.get("profile"))) if str(route.get("profile")) in profile_order else 99)
                route_rows = []
                for route in method_routes:
                    route_key = str(route.get("route_key"))
                    children = sorted(child_routes.get(route_key, []), key=lambda child: ["city_loop", "transition", "must_go"].index(str(child.get("layer_role"))) if str(child.get("layer_role")) in ["city_loop", "transition", "must_go"] else 99)
                    child_rows = "".join(_route_checkbox_html(child, compact=True) for child in children)
                    route_rows.append(
                        f"""
                        <details class="blueprint-route-leaf" data-route-key="{_escape(route_key)}">
                            <summary>{_route_checkbox_html(route)}<span class="blueprint-route-subhint">details</span></summary>
                            <div class="blueprint-route-subsection">
                                <button type="button" class="blueprint-route-hotel-note" data-route-action="hotel_showcase">
                                    Hotel Showcase controls selected hotels for checked routes.
                                </button>
                                {child_rows}
                            </div>
                        </details>
                        """
                    )
                method_blocks.append(
                    f"""
                    <details class="blueprint-route-method" data-method="{_escape(method)}">
                        <summary>{_escape(method_label)} <span>{len(method_routes)}</span></summary>
                        {''.join(route_rows)}
                    </details>
                    """
                )
            matrix_html.append(
                f"""
                <details class="blueprint-route-group blueprint-route-matrix-day" data-selector-group="route_matrix" open>
                    <summary>{int(day_value)}-Day Routes <span>{len(day_routes)}</span></summary>
                    {''.join(method_blocks)}
                </details>
                """
            )
        auxiliary_rows = []
        for route in route_registry:
            if str(route.get("selector_group")) == "route_matrix":
                continue
            if str(route.get("layer_role")) == "selected_hotels":
                continue
            auxiliary_rows.append(_route_checkbox_html(route))
        if auxiliary_rows:
            matrix_html.append(
                f"""
                <details class="blueprint-route-group" data-selector-group="context">
                    <summary>Context / Candidates <span>{len(auxiliary_rows)}</span></summary>
                    {''.join(auxiliary_rows)}
                </details>
                """
            )
        checkbox_html = "".join(matrix_html)
    else:
        group_order = [
            ("core", "Core"),
            ("trip_length", "Trip Length"),
            ("method", "Methods"),
            ("traveler_profile", "Travelers"),
            ("city_detail", "City Details"),
            ("transition", "Transitions"),
            ("must_go", "Must-Go"),
            ("hotel", "Hotels"),
            ("context", "Context Routes"),
        ]
        grouped_routes = {key: [] for key, _ in group_order}
        for route in route_registry:
            selector_group = str(route.get("selector_group", route.get("family", "context")))
            if selector_group not in grouped_routes:
                grouped_routes[selector_group] = []
            grouped_routes[selector_group].append(_route_checkbox_html(route))
        group_html = []
        for group_key, group_label in group_order:
            rows = grouped_routes.get(group_key, [])
            if not rows:
                continue
            open_attr = " open" if group_key in {"core", "trip_length", "method"} else ""
            group_html.append(
                f"""
                <details class="blueprint-route-group" data-selector-group="{_escape(group_key)}"{open_attr}>
                    <summary>{_escape(group_label)} <span>{len(rows)}</span></summary>
                    {''.join(rows)}
                </details>
                """
            )
        checkbox_html = "".join(group_html)

    controls_html = f"""
    <style>
    #blueprint-route-selector {{
        position: fixed;
        left: 74px;
        top: 82px;
        z-index: 9996;
        width: 430px;
    }}
    #blueprint-route-selector.blueprint-collapsed {{
        width: 230px;
    }}
    #blueprint-route-selector .blueprint-panel-body {{
        max-height: 420px;
    }}
    .blueprint-route-actions {{
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 6px;
        margin-bottom: 10px;
    }}
    .blueprint-route-action {{
        background: #F8FAFC;
        border: 1px solid #CBD5E1;
        border-radius: 6px;
        color: #1F2937;
        cursor: pointer;
        font: inherit;
        font-weight: 800;
        padding: 5px 7px;
    }}
    .blueprint-route-action:hover {{
        background: #EEF2F7;
    }}
    .blueprint-route-check {{
        align-items: center;
        border-top: 1px solid #EDF0F3;
        cursor: pointer;
        display: grid;
        column-gap: 8px;
        grid-template-columns: 17px 14px minmax(0, 1fr) minmax(48px, auto);
        padding: 6px 0;
        min-width: 0;
    }}
    .blueprint-route-check input {{
        margin: 0;
    }}
    .blueprint-route-swatch {{
        border: 1px solid rgba(17,24,39,0.24);
        border-radius: 999px;
        display: inline-block;
        height: 12px;
        width: 12px;
    }}
    .blueprint-route-check-text {{
        line-height: 1.18;
        min-width: 0;
        overflow-wrap: anywhere;
        padding-right: 4px;
    }}
    .blueprint-route-meta {{
        color: #64748B;
        font-size: 10px;
        font-weight: 700;
        justify-self: end;
        min-width: 48px;
        text-align: right;
        white-space: nowrap;
    }}
    .blueprint-route-check-compact {{
        margin-left: 18px;
        padding: 5px 0;
    }}
    .blueprint-route-leaf {{
        border-top: 1px solid #EDF0F3;
    }}
    .blueprint-route-leaf > summary {{
        align-items: center;
        display: grid;
        grid-template-columns: minmax(0, 1fr) auto;
        gap: 8px;
        list-style: none;
    }}
    .blueprint-route-leaf > summary::-webkit-details-marker {{
        display: none;
    }}
    .blueprint-route-leaf > summary .blueprint-route-check {{
        border-top: 0;
    }}
    .blueprint-route-method {{
        border-top: 1px solid #E5E7EB;
        margin-left: 6px;
    }}
    .blueprint-route-method > summary {{
        cursor: pointer;
        font-weight: 800;
        list-style: none;
        padding: 5px 0;
    }}
    .blueprint-route-method > summary::-webkit-details-marker {{
        display: none;
    }}
    .blueprint-route-method > summary span {{
        color: #64748B;
        float: right;
        font-size: 10px;
    }}
    .blueprint-route-subsection {{
        background: #F8FAFC;
        border-left: 2px solid #CBD5E1;
        margin: 0 0 5px 20px;
        padding: 5px 8px;
    }}
    .blueprint-route-subhint {{
        background: #EEF2F7;
        border: 1px solid #CBD5E1;
        border-radius: 999px;
        color: #64748B;
        font-size: 10px;
        font-weight: 700;
        line-height: 1;
        padding: 4px 7px;
        white-space: nowrap;
    }}
    .blueprint-route-hotel-note {{
        background: #FFFFFF;
        border: 1px solid #CBD5E1;
        border-radius: 6px;
        color: #475569;
        cursor: pointer;
        display: block;
        font-size: 10px;
        font-weight: 700;
        margin: 2px 0 6px 18px;
        padding: 5px 7px;
        text-align: left;
        width: calc(100% - 18px);
    }}
    .blueprint-route-hotel-note:hover {{
        background: #EEF2F7;
    }}
    .blueprint-route-check input:disabled + .blueprint-route-swatch,
    .blueprint-route-check input:disabled ~ .blueprint-route-check-text,
    .blueprint-route-check input:disabled ~ .blueprint-route-meta {{
        opacity: 0.48;
    }}
    #blueprint-hotel-showcase-toggle.blueprint-active {{
        background: #0F172A;
        border-color: #0F172A;
        color: #FFFFFF;
    }}
    .blueprint-route-group {{
        border-top: 1px solid #E5E7EB;
        padding: 4px 0;
    }}
    .blueprint-route-group summary {{
        color: #111827;
        cursor: pointer;
        font-weight: 900;
        list-style: none;
        padding: 6px 0;
    }}
    .blueprint-route-group summary::-webkit-details-marker {{
        display: none;
    }}
    .blueprint-route-group summary span {{
        color: #64748B;
        float: right;
        font-size: 10px;
        padding-top: 2px;
    }}
    #blueprint-route-debug-tab {{
        align-items: center;
        background: #0F172A;
        border: 1px solid rgba(255,255,255,0.55);
        border-left: 0;
        border-radius: 0 8px 8px 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.22);
        color: #FFFFFF;
        cursor: pointer;
        display: flex;
        font-size: 11px;
        font-weight: 900;
        height: 88px;
        justify-content: center;
        left: 0;
        letter-spacing: 0;
        padding: 6px 4px;
        position: fixed;
        text-orientation: mixed;
        top: 44%;
        transform: translateY(-50%);
        writing-mode: vertical-rl;
        width: 21px;
        z-index: 10001;
    }}
    #blueprint-route-debug-tab:hover {{
        background: #1E293B;
    }}
    #blueprint-route-debug-panel {{
        left: -326px;
        top: 250px;
        width: 310px;
        z-index: 10000;
        transition: left 180ms ease, opacity 180ms ease;
        opacity: 0.0;
        pointer-events: none;
    }}
    #blueprint-route-debug-panel.blueprint-debug-open {{
        left: 24px;
        opacity: 1.0;
        pointer-events: auto;
    }}
    #blueprint-route-debug-panel .blueprint-panel-body {{
        max-height: 300px;
    }}
    </style>
    <script id="blueprint-route-debug-registry" type="application/json">{registry_json}</script>
    <div id="blueprint-route-selector" class="blueprint-floating-panel blueprint-draggable" data-panel-id="route-selector">
        <button class="blueprint-panel-header" type="button" aria-expanded="true">
            <span>Route Selector</span>
            <span class="blueprint-panel-chevron">-</span>
        </button>
        <div class="blueprint-panel-body">
            <div class="blueprint-route-actions">
                <button type="button" class="blueprint-route-action" data-route-action="clear">Clear</button>
                <button type="button" class="blueprint-route-action" data-route-action="balanced">Balanced default</button>
                <button type="button" class="blueprint-route-action" data-route-action="day_7">Show 7-day methods</button>
                <button type="button" class="blueprint-route-action" data-route-action="day_9">Show 9-day methods</button>
                <button type="button" class="blueprint-route-action" data-route-action="day_12">Show 12-day methods</button>
                <button type="button" class="blueprint-route-action" data-route-action="all_balanced">Show all balanced</button>
                <button type="button" class="blueprint-route-action" data-route-action="city_detail">Show city loops</button>
                <button type="button" class="blueprint-route-action" data-route-action="transition">Show transitions</button>
            </div>
            <button type="button" id="blueprint-hotel-showcase-toggle" class="blueprint-route-action" data-route-action="hotel_showcase" aria-pressed="false" style="width:100%; margin-bottom:6px;">Hotel Showcase: Off</button>
            <button type="button" class="blueprint-route-action" data-route-action="zoom" style="width:100%; margin-bottom:8px;">Zoom to checked routes</button>
            <div class="blueprint-route-checkboxes">{checkbox_html}</div>
        </div>
    </div>
    <button id="blueprint-route-debug-tab" type="button" aria-controls="blueprint-route-debug-panel" aria-expanded="false">Debug</button>
    <div id="blueprint-route-debug-panel" class="blueprint-floating-panel blueprint-collapsed" data-panel-id="route-debug">
        <button class="blueprint-panel-header" type="button" aria-expanded="false">
            <span>Route Debug</span>
            <span class="blueprint-panel-chevron">+</span>
        </button>
        <div class="blueprint-panel-body">
            <div class="summary-line"><b>Default checked routes:</b> {checked_count}</div>
            <div class="summary-line"><b>Registered checkbox routes:</b> {len(route_registry)}</div>
            <div class="summary-line"><b>Unchecked on load:</b> {len(hidden_vars)}</div>
            <div class="summary-line"><b>Families:</b> {_escape(', '.join(families))}</div>
            <div class="summary-line muted-note">Browser registry: blueprint-route-debug-registry. The Route Selector owns route visibility on first open.</div>
        </div>
    </div>
    """

    controls_js = f"""
    (function() {{
        var registry = {registry_json};
        var hiddenLayerVars = {hidden_json};
        var routeBounds = {bounds_json};
        var mapVarName = "{map_var}";
        var mapObject = null;
        var listenersAttached = false;
        var hotelShowcaseEnabled = false;
        var lastDiagnostics = {{
            map_ready: false,
            registry_count: registry.length,
            missing_layers: [],
            checked_routes: [],
            visible_checked_routes: [],
            unchecked_routes: [],
            last_sync: null
        }};

        function getMapObject() {{
            if (mapObject && mapObject.fitBounds && mapObject.hasLayer) {{
                return mapObject;
            }}
            mapObject = window[mapVarName] || (typeof globalThis !== "undefined" ? globalThis[mapVarName] : null) || null;
            return mapObject;
        }}

        function getLayerByName(name) {{
            if (!name) {{
                return null;
            }}
            return window[name] || (typeof globalThis !== "undefined" ? globalThis[name] : null) || null;
        }}

        function addLayer(layer) {{
            var map = getMapObject();
            if (map && layer && !map.hasLayer(layer)) {{
                layer.addTo(map);
            }}
        }}

        function removeLayer(layer) {{
            var map = getMapObject();
            if (map && layer && map.hasLayer(layer)) {{
                map.removeLayer(layer);
            }}
        }}

        function routeById(controlId) {{
            return registry.find(function(route) {{
                return String(route.control_id) === String(controlId);
            }});
        }}

        function checkboxForRoute(route) {{
            return document.querySelector('[data-route-checkbox="' + route.control_id + '"]');
        }}

        function fullRouteForKey(routeKey) {{
            return registry.find(function(route) {{
                return String(route.route_key) === String(routeKey) && String(route.layer_role) === "full_route";
            }});
        }}

        function parentRouteChecked(route) {{
            var parentKey = route.parent_route_key || route.route_key;
            var parent = fullRouteForKey(parentKey);
            var parentCheckbox = parent ? checkboxForRoute(parent) : null;
            return Boolean(parentCheckbox && parentCheckbox.checked);
        }}

        function isRouteChecked(route) {{
            if (route.hotel_showcase_controlled || String(route.layer_role) === "selected_hotels") {{
                return Boolean(hotelShowcaseEnabled && parentRouteChecked(route));
            }}
            if (["city_loop", "transition", "must_go"].indexOf(String(route.layer_role)) !== -1) {{
                var childCheckbox = checkboxForRoute(route);
                return Boolean(childCheckbox && childCheckbox.checked && parentRouteChecked(route));
            }}
            var checkbox = checkboxForRoute(route);
            return Boolean(checkbox && checkbox.checked);
        }}

        function checkedRoutes() {{
            return registry.filter(function(route) {{
                return isRouteChecked(route);
            }});
        }}

        function syncCheckboxState() {{
            var map = getMapObject();
            var missingLayers = [];
            var checkedIds = [];
            var visibleCheckedIds = [];
            var uncheckedIds = [];
            registry.forEach(function(route) {{
                var layer = getLayerByName(route.layer_var);
                var checked = isRouteChecked(route);
                route.runtime_checked = checked;
                route.runtime_layer_found = Boolean(layer);
                route.runtime_hotel_showcase = Boolean(hotelShowcaseEnabled);
                if (!layer) {{
                    missingLayers.push(route.layer_var || route.control_id);
                }}
                if (checked) {{
                    checkedIds.push(route.control_id);
                    addLayer(layer);
                    if (layer && layer.bringToFront) {{
                        layer.bringToFront();
                    }}
                    if (map && layer && map.hasLayer(layer)) {{
                        visibleCheckedIds.push(route.control_id);
                        route.runtime_visible = true;
                    }} else {{
                        route.runtime_visible = false;
                    }}
                }} else {{
                    uncheckedIds.push(route.control_id);
                    removeLayer(layer);
                    route.runtime_visible = false;
                }}
            }});
            lastDiagnostics = {{
                map_ready: Boolean(map),
                registry_count: registry.length,
                missing_layers: missingLayers,
                checked_routes: checkedIds,
                visible_checked_routes: visibleCheckedIds,
                unchecked_routes: uncheckedIds,
                last_sync: new Date().toISOString()
            }};
            window.blueprintCheckedRoutes = checkedIds;
            window.blueprintHotelShowcaseEnabled = hotelShowcaseEnabled;
            window.blueprintRouteRuntimeDiagnostics = lastDiagnostics;
            return Boolean(map) && missingLayers.length === 0;
        }}

        function setAllChecked(value) {{
            registry.forEach(function(route) {{
                var checkbox = checkboxForRoute(route);
                if (checkbox) {{
                    checkbox.checked = Boolean(value);
                }}
            }});
        }}

        function setOnly(predicate) {{
            registry.forEach(function(route) {{
                var checkbox = checkboxForRoute(route);
                if (checkbox) {{
                    checkbox.checked = Boolean(predicate(route));
                }}
            }});
        }}

        function checkedFullRouteKeys() {{
            var keys = [];
            registry.forEach(function(route) {{
                if (!routeIsFullRoute(route)) {{
                    return;
                }}
                var checkbox = checkboxForRoute(route);
                if (checkbox && checkbox.checked && route.route_key) {{
                    keys.push(String(route.route_key));
                }}
            }});
            return keys;
        }}

        function setChildRoleForCheckedRoutes(roleName) {{
            var activeKeys = checkedFullRouteKeys();
            var activeSet = new Set(activeKeys);
            var eligibleChildren = registry.filter(function(route) {{
                var parentKey = String(route.parent_route_key || route.route_key || "");
                return String(route.layer_role) === String(roleName) && activeSet.has(parentKey);
            }});
            var shouldEnable = eligibleChildren.some(function(route) {{
                var checkbox = checkboxForRoute(route);
                return checkbox && !checkbox.checked;
            }});
            registry.forEach(function(route) {{
                if (String(route.layer_role) !== String(roleName)) {{
                    return;
                }}
                var checkbox = checkboxForRoute(route);
                if (!checkbox) {{
                    return;
                }}
                var parentKey = String(route.parent_route_key || route.route_key || "");
                checkbox.checked = Boolean(shouldEnable && activeSet.has(parentKey));
            }});
        }}

        function updateChildAvailability() {{
            registry.forEach(function(route) {{
                var role = String(route.layer_role || "");
                if (["city_loop", "transition", "must_go"].indexOf(role) === -1) {{
                    return;
                }}
                var checkbox = checkboxForRoute(route);
                if (!checkbox) {{
                    return;
                }}
                var disabled = !parentRouteChecked(route);
                checkbox.disabled = disabled;
                var row = checkbox.closest(".blueprint-route-check");
                if (row) {{
                    row.classList.toggle("blueprint-route-disabled", disabled);
                }}
            }});
        }}

        function routeHasQuickGroup(route, groupName) {{
            return Array.isArray(route.quick_groups) && route.quick_groups.indexOf(groupName) !== -1;
        }}

        function routeIsFullRoute(route) {{
            return String(route.layer_role || "") === "full_route" || String(route.family || "") === "traveler_profile";
        }}

        function updateHotelShowcaseButton() {{
            var button = document.getElementById("blueprint-hotel-showcase-toggle");
            if (!button) {{
                return;
            }}
            button.classList.toggle("blueprint-active", Boolean(hotelShowcaseEnabled));
            button.setAttribute("aria-pressed", hotelShowcaseEnabled ? "true" : "false");
            button.textContent = hotelShowcaseEnabled ? "Hotel Showcase: On" : "Hotel Showcase: Off";
        }}

        function zoomToCheckedRoutes() {{
            var checked = checkedRoutes();
            var points = [];
            checked.forEach(function(route) {{
                if (Array.isArray(route.bounds) && route.bounds.length === 2) {{
                    points.push(route.bounds[0]);
                    points.push(route.bounds[1]);
                }}
            }});
            if (!points.length) {{
                points = routeBounds || [];
            }}
            var map = getMapObject();
            if (map && points && points.length >= 2) {{
                map.fitBounds(points, {{ padding: [70, 70] }});
            }}
        }}

        function attachRouteControlListeners() {{
            if (listenersAttached) {{
                return;
            }}
            listenersAttached = true;
            var selector = document.getElementById("blueprint-route-selector");
            var debugPanel = document.getElementById("blueprint-route-debug-panel");
            var debugTab = document.getElementById("blueprint-route-debug-tab");
            [selector, debugPanel, debugTab].forEach(function(node) {{
                if (node && window.L && L.DomEvent) {{
                    L.DomEvent.disableClickPropagation(node);
                    L.DomEvent.disableScrollPropagation(node);
                }}
            }});
            function setDebugDrawerOpen(open) {{
                if (!debugPanel || !debugTab) {{
                    return;
                }}
                debugPanel.classList.toggle("blueprint-debug-open", Boolean(open));
                debugTab.setAttribute("aria-expanded", open ? "true" : "false");
                if (open) {{
                    debugPanel.classList.remove("blueprint-collapsed");
                    var debugHeader = debugPanel.querySelector(".blueprint-panel-header");
                    var chevron = debugPanel.querySelector(".blueprint-panel-chevron");
                    if (debugHeader) {{
                        debugHeader.setAttribute("aria-expanded", "true");
                    }}
                    if (chevron) {{
                        chevron.textContent = "-";
                    }}
                }}
            }}
            window.blueprintSetDebugDrawerOpen = setDebugDrawerOpen;
            if (debugTab) {{
                debugTab.addEventListener("click", function(event) {{
                    event.preventDefault();
                    setDebugDrawerOpen(!debugPanel.classList.contains("blueprint-debug-open"));
                }});
            }}
            if (debugPanel) {{
                var debugHeader = debugPanel.querySelector(".blueprint-panel-header");
                if (debugHeader) {{
                    debugHeader.addEventListener("click", function() {{
                        window.setTimeout(function() {{
                            if (debugPanel.classList.contains("blueprint-collapsed")) {{
                                setDebugDrawerOpen(false);
                            }}
                        }}, 0);
                    }});
                }}
            }}
            document.querySelectorAll("[data-route-checkbox]").forEach(function(checkbox) {{
                checkbox.addEventListener("click", function(event) {{
                    event.stopPropagation();
                }});
                checkbox.addEventListener("change", function() {{
                    updateChildAvailability();
                    syncCheckboxState();
                }});
            }});
            document.querySelectorAll("[data-route-action]").forEach(function(button) {{
                button.addEventListener("click", function(event) {{
                    var action = event.currentTarget.getAttribute("data-route-action");
                    if (action === "clear") {{
                        setAllChecked(false);
                        hotelShowcaseEnabled = false;
                        updateHotelShowcaseButton();
                    }} else if (action === "balanced") {{
                        setOnly(function(route) {{
                            return route.control_id === "route_matrix_full__d7__bandit_repair__balanced" || route.control_id === "traveler_balanced";
                        }});
                    }} else if (action === "day_7") {{
                        setOnly(function(route) {{ return routeIsFullRoute(route) && Number(route.trip_days) === 7; }});
                    }} else if (action === "day_9") {{
                        setOnly(function(route) {{ return routeIsFullRoute(route) && Number(route.trip_days) === 9; }});
                    }} else if (action === "day_12") {{
                        setOnly(function(route) {{ return routeIsFullRoute(route) && Number(route.trip_days) === 12; }});
                    }} else if (action === "all_balanced") {{
                        setOnly(function(route) {{ return routeIsFullRoute(route) && String(route.profile) === "balanced"; }});
                    }} else if (action === "traveler") {{
                        setOnly(function(route) {{ return routeHasQuickGroup(route, "traveler"); }});
                    }} else if (action === "city_detail") {{
                        setChildRoleForCheckedRoutes("city_loop");
                    }} else if (action === "transition") {{
                        setChildRoleForCheckedRoutes("transition");
                    }} else if (action === "must_go") {{
                        setChildRoleForCheckedRoutes("must_go");
                    }} else if (action === "hotel") {{
                        setOnly(function(route) {{
                            return routeHasQuickGroup(route, "hotel");
                        }});
                    }} else if (action === "hotel_showcase") {{
                        hotelShowcaseEnabled = !hotelShowcaseEnabled;
                        updateHotelShowcaseButton();
                    }}
                    updateChildAvailability();
                    syncCheckboxState();
                    if (action === "zoom" || action !== "clear") {{
                        zoomToCheckedRoutes();
                    }}
                }});
            }});
            updateHotelShowcaseButton();
            updateChildAvailability();
        }}

        function bootstrapRouteSelector(attempt) {{
            attachRouteControlListeners();
            hiddenLayerVars.forEach(function(layerVar) {{
                removeLayer(getLayerByName(layerVar));
            }});
            updateChildAvailability();
            var ready = syncCheckboxState();
            if (ready) {{
                zoomToCheckedRoutes();
                return;
            }}
            if (attempt < 60) {{
                window.setTimeout(function() {{
                    bootstrapRouteSelector(attempt + 1);
                }}, 120);
            }}
        }}

        window.blueprintMapDebug = {{
            registry: registry,
            hiddenLayerVars: hiddenLayerVars,
            syncCheckboxState: syncCheckboxState,
            zoomToCheckedRoutes: zoomToCheckedRoutes,
            routeById: routeById
        }};

        if (document.readyState === "loading") {{
            document.addEventListener("DOMContentLoaded", function() {{
                bootstrapRouteSelector(0);
            }});
        }} else {{
            bootstrapRouteSelector(0);
        }}
        window.addEventListener("load", function() {{
            bootstrapRouteSelector(0);
        }});
    }})();
    """
    map_object.get_root().html.add_child(folium.Element(controls_html))
    map_object.get_root().script.add_child(folium.Element(controls_js))


def _add_blueprint_panel_behavior(map_object):
    behavior_js = """
    (function() {
        function clamp(value, minValue, maxValue) {
            return Math.max(minValue, Math.min(maxValue, value));
        }

        function setCollapsed(panel, collapsed) {
            var header = panel.querySelector('.blueprint-panel-header');
            var chevron = panel.querySelector('.blueprint-panel-chevron');
            panel.classList.toggle('blueprint-collapsed', collapsed);
            if (header) {
                header.setAttribute('aria-expanded', collapsed ? 'false' : 'true');
            }
            if (chevron) {
                chevron.textContent = collapsed ? '+' : '-';
            }
        }

        function initPanel(panel) {
            if (!panel || panel.dataset.blueprintPanelReady) {
                return;
            }
            panel.dataset.blueprintPanelReady = '1';
            var header = panel.querySelector('.blueprint-panel-header');
            if (!header) {
                return;
            }
            if (window.L && L.DomEvent) {
                L.DomEvent.disableClickPropagation(panel);
                L.DomEvent.disableScrollPropagation(panel);
            }
            header.addEventListener('click', function(event) {
                if (panel.dataset.dragSuppressClick === '1') {
                    panel.dataset.dragSuppressClick = '0';
                    event.preventDefault();
                    return;
                }
                setCollapsed(panel, !panel.classList.contains('blueprint-collapsed'));
            });
            if (!panel.classList.contains('blueprint-draggable')) {
                return;
            }
            var dragging = false;
            var moved = false;
            var startX = 0;
            var startY = 0;
            var startLeft = 0;
            var startTop = 0;
            header.addEventListener('pointerdown', function(event) {
                if (event.button !== 0) {
                    return;
                }
                dragging = true;
                moved = false;
                startX = event.clientX;
                startY = event.clientY;
                var rect = panel.getBoundingClientRect();
                startLeft = rect.left;
                startTop = rect.top;
                panel.classList.add('blueprint-dragging');
                header.setPointerCapture(event.pointerId);
            });
            header.addEventListener('pointermove', function(event) {
                if (!dragging) {
                    return;
                }
                var dx = event.clientX - startX;
                var dy = event.clientY - startY;
                if (Math.abs(dx) + Math.abs(dy) > 4) {
                    moved = true;
                }
                var maxLeft = Math.max(8, window.innerWidth - panel.offsetWidth - 8);
                var maxTop = Math.max(8, window.innerHeight - panel.offsetHeight - 8);
                panel.style.left = clamp(startLeft + dx, 8, maxLeft) + 'px';
                panel.style.top = clamp(startTop + dy, 8, maxTop) + 'px';
                panel.style.right = 'auto';
                panel.style.bottom = 'auto';
            });
            function stopDrag(event) {
                if (!dragging) {
                    return;
                }
                dragging = false;
                panel.classList.remove('blueprint-dragging');
                panel.dataset.dragSuppressClick = moved ? '1' : '0';
                try {
                    header.releasePointerCapture(event.pointerId);
                } catch (err) {}
            }
            header.addEventListener('pointerup', stopDrag);
            header.addEventListener('pointercancel', stopDrag);
        }

        function initBlueprintPanels() {
            document.querySelectorAll('.blueprint-floating-panel').forEach(initPanel);
        }

        window.initBlueprintPanels = initBlueprintPanels;
        window.setTimeout(initBlueprintPanels, 50);
        window.setTimeout(initBlueprintPanels, 500);
        window.addEventListener('resize', function() {
            document.querySelectorAll('.blueprint-floating-panel').forEach(function(panel) {
                var rect = panel.getBoundingClientRect();
                if (rect.right > window.innerWidth || rect.bottom > window.innerHeight) {
                    panel.style.left = clamp(rect.left, 8, Math.max(8, window.innerWidth - panel.offsetWidth - 8)) + 'px';
                    panel.style.top = clamp(rect.top, 8, Math.max(8, window.innerHeight - panel.offsetHeight - 8)) + 'px';
                    panel.style.right = 'auto';
                    panel.style.bottom = 'auto';
                }
            });
        });
    })();
    """
    map_object.get_root().script.add_child(folium.Element(behavior_js))


def _build_day_panel(day_plan_df):
    cards = []
    for day, group in day_plan_df.groupby("day", sort=True):
        first = group.iloc[0]
        route_type = str(getattr(first, "route_type", "base_city_local"))
        overnight_city = str(getattr(first, "overnight_city", getattr(first, "city", "")))
        if route_type.startswith("relocation"):
            visited_cities = _unique_in_order(group["city"].astype(str).tolist() + [overnight_city])
            title_city = " -> ".join(visited_cities)
            drive_line = (
                f"<div class=\"hotel-line\">Drive: {float(getattr(first, 'drive_minutes_to_next_base', 0.0) or 0.0):.0f} min; "
                f"visit window after drive: {float(getattr(first, 'available_visit_minutes', 0.0) or 0.0):.0f} min</div>"
            )
        else:
            title_city = overnight_city
            drive_line = ""
        stops = "".join(
            f"<li><span>{int(row.stop_order)}.</span> {_escape(row.attraction_name)} <span class=\"muted-city\">({_escape(row.city)})</span></li>"
            for row in group.itertuples(index=False)
        )
        cards.append(
            f"""
            <div class="day-card">
                <div class="day-card-title">Day {int(day)} · {_escape(title_city)}</div>
                <div class="hotel-line">Overnight base: {_escape(overnight_city)} · {_escape(first.hotel_name)}</div>
                {drive_line}
                <ol>{stops}</ol>
            </div>
            """
        )
    return "".join(cards)


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


def _load_default_hierarchical_gurobi_day_plan(output_dir, fallback_df):
    route_stops = _load_csv(Path(output_dir) / "production_method_route_stops.csv")
    if route_stops.empty or "method" not in route_stops.columns:
        return fallback_df
    route_stops = route_stops[route_stops["method"].astype(str).eq("hierarchical_gurobi_pipeline")].copy()
    if route_stops.empty:
        return fallback_df

    route_stops["day"] = pd.to_numeric(route_stops.get("day", 1), errors="coerce").fillna(1).astype(int)
    route_stops["stop_order"] = pd.to_numeric(route_stops.get("stop_order", 1), errors="coerce").fillna(1).astype(int)
    route_stops = route_stops.sort_values(["day", "stop_order", "attraction_name"]).reset_index(drop=True)
    total_days = int(route_stops["day"].max()) if not route_stops.empty else 0
    output = route_stops.copy()
    scalar_series = lambda default: pd.Series(default, index=output.index)
    output["profile"] = "balanced"
    output["profile_label"] = "Balanced"
    output["hotel_source"] = output.get("hotel_source", "production_method_route_stops")
    output["overnight_base"] = output["city"].astype(str).eq(output["overnight_city"].astype(str))
    output["stop_is_overnight_city"] = output["overnight_base"]
    output["hotel_booked"] = output["day"].astype(int) < total_days
    output["route_start_city"] = output.get("route_start_city", output["city"])
    output["route_end_city"] = output.get("route_end_city", output["overnight_city"])
    output["pass_through_cities"] = output.get("pass_through_cities", "")
    output["drive_minutes_to_next_base"] = pd.to_numeric(
        output.get("drive_minutes_to_next_base", scalar_series(0.0)),
        errors="coerce",
    ).fillna(0.0)
    output["available_visit_minutes"] = pd.to_numeric(
        output.get("available_visit_minutes", scalar_series(0.0)),
        errors="coerce",
    ).fillna(0.0)
    output["drive_time_source"] = output.get("drive_time_source", "production_method_route_stops")
    output["attraction_source"] = output.get("attraction_source", output.get("source_list", "production_method_route_stops"))
    output["must_go_weight"] = pd.to_numeric(output.get("must_go_weight", scalar_series(0.0)), errors="coerce").fillna(0.0)
    output["corridor_fit"] = pd.to_numeric(output.get("corridor_fit", scalar_series(0.0)), errors="coerce").fillna(0.0)
    output["detour_minutes"] = pd.to_numeric(output.get("detour_minutes", scalar_series(0.0)), errors="coerce").fillna(0.0)
    output["data_confidence"] = pd.to_numeric(output.get("data_confidence", scalar_series(0.5)), errors="coerce").fillna(0.5)
    output["social_reason"] = output.get("social_reason", "")
    output["pass_through_cities"] = output.get("pass_through_cities", "")
    return output


def build_production_trip_map(context, output_path=None, run_live_routing=None):
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
            fit_points.extend(
                day_plan_df[[left_col, right_col]].dropna().astype(float).values.tolist()
            )
    all_points = (
        scenic_points
        + fastest_points
        + profile_day_plan_df[["latitude", "longitude"]].values.tolist()
        + profile_day_plan_df[["hotel_latitude", "hotel_longitude"]].drop_duplicates().values.tolist()
        + social_catalog_for_bounds[["latitude", "longitude"]].values.tolist()
    )
    center_lat = float(np.mean([point[0] for point in all_points]))
    center_lon = float(np.mean([point[1] for point in all_points]))

    trip_map = folium.Map(location=[center_lat, center_lon], zoom_start=6, tiles=None, control_scale=True, prefer_canvas=True)
    folium.TileLayer("CartoDB Positron", name="Light basemap", control=True, overlay=False, show=True).add_to(trip_map)
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap (alt)", control=True, overlay=False, show=False).add_to(trip_map)
    _add_route_panes(trip_map)
    plugins.Fullscreen(position="topleft", title="Expand map", title_cancel="Exit full screen").add_to(trip_map)

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
        for row in profile_day_plan_df[["city", "overnight_city", "hotel_name"]].drop_duplicates().itertuples(index=False)
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
            is_selected = (bool(getattr(hotel, "selected", False)) or (city, hotel_name) in selected_hotels) and not use_route_matrix
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
                "bounds": _route_bounds(profile_day_plan_df[["hotel_latitude", "hotel_longitude"]].dropna().astype(float).values.tolist()),
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
            "bounds": _route_bounds(hotel_debug_df[["latitude", "longitude"]].dropna().astype(float).values.tolist()) if not hotel_debug_df.empty else [],
        }
    )

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
            icon=folium.Icon(color="green" if state == "selected" else "orange" if state == "skipped" else "gray", icon="star", prefix="fa"),
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
            "bounds": _route_bounds(social_catalog[["latitude", "longitude"]].dropna().astype(float).values.tolist()) if not social_catalog.empty else [],
        }
    )

    profile_layer_groups = {
        PROFILE_CONFIGS[profile_name]["label"]: []
        for profile_name in PROFILE_CONFIGS
    }
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
                Social must-go: {"yes" if bool(getattr(row, "social_must_go", False)) else "no"}<br/>
                Social score: {float(getattr(row, "social_score", 0.0)):.2f}<br/>
                Must-go weight: {float(getattr(row, "must_go_weight", 0.0) or 0.0):.2f}<br/>
                Corridor fit: {float(getattr(row, "corridor_fit", 0.0) or 0.0):.2f}<br/>
                Detour estimate: {float(getattr(row, "detour_minutes", 0.0) or 0.0):.1f} min<br/>
                Data confidence: {float(getattr(row, "data_confidence", 0.0) or 0.0):.2f}<br/>
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
                badge_text = f"D{int(row.day)}.{int(row.stop_order)}" if profile_name == "balanced" else f"{profile_prefix}{int(row.day)}.{int(row.stop_order)}"
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

    best_bandit = context.get("hybrid_bandit_summary_df", pd.DataFrame())
    bandit_strategy = "not available"
    bandit_reward = np.nan
    if isinstance(best_bandit, pd.DataFrame) and not best_bandit.empty:
        bandit_strategy = str(best_bandit.iloc[0].get("route_search_strategy", bandit_strategy))
        bandit_reward = float(best_bandit.iloc[0].get("posterior_mean_reward", best_bandit.iloc[0].get("posterior_mean", np.nan)))

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
            show_by_default=bool(context.get("SHOW_COMPARISON_LAYERS_BY_DEFAULT", False)) and not hide_detail_layers_on_load,
            route_debug_rows=route_debug_rows,
        )
        selected_result_layer = _add_selected_result_layer(
            trip_map,
            day_plan_df,
            route_cache=route_cache,
            run_live=run_live,
            route_debug_rows=route_debug_rows,
            show_by_default=bool(context.get("SHOW_SELECTED_RESULT_BY_DEFAULT", False)) and not hide_detail_layers_on_load,
        )
        traveler_overview_layers = _add_traveler_overview_layers(
            trip_map,
            profile_day_plans,
            route_cache=route_cache,
            run_live=run_live,
            route_debug_rows=route_debug_rows,
            show_by_default=bool(context.get("SHOW_TRAVELER_OVERVIEWS_BY_DEFAULT", False)) and not hide_detail_layers_on_load,
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
        if not fit_points and not comparison_df.empty and {"hotel_latitude", "hotel_longitude"}.issubset(comparison_df.columns):
            fit_points.extend(comparison_df[["hotel_latitude", "hotel_longitude"]].dropna().astype(float).values.tolist())

    fastest_drive_hours = float(intercity_legs_df[intercity_legs_df["route_layer"].eq("Fastest inter-city route")]["estimated_drive_minutes"].sum()) / 60.0
    scenic_drive_hours = float(intercity_legs_df[intercity_legs_df["route_layer"].eq("Scenic CA-1 / PCH route with Stanford detour")]["estimated_drive_minutes"].sum()) / 60.0
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
        <div class="summary-line"><b>Gateway:</b> {_escape(trip['gateway_start'])} → {_escape(trip['gateway_end'])}</div>
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

    grouped_layers = {
        "Route Matrix": route_matrix_layers,
        "Full Routes": full_route_layers,
        "City Details": city_detail_layers,
        "Transitions": transition_layers,
        "Selected Result": [selected_result_layer] if selected_result_layer is not None else [],
        "Routes": [scenic_layer, fastest_layer],
        "Map Context": [social_layer, selected_hotel_layer, candidate_hotel_layer],
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
        candidate_hotel_layer,
        *route_matrix_hidden_layers,
        *traveler_overview_layers,
        *city_detail_layers,
        *transition_layers,
    ]
    unchecked_default_layers.extend([layer for layer in full_route_layers if layer is not None and not getattr(layer, "show", False)])
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
        with cache_path.open("w", encoding="utf-8") as handle:
            json.dump(route_cache, handle)

    trip_map.save(str(output_path))
    return trip_map, day_plan_df, output_path
