"""Production blueprint enrichment helpers.

The notebook uses these functions to keep the math readable while still writing
GitHub-friendly artifacts for city scoring, budget estimation, and hybrid
bandit route search.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from geopy.distance import geodesic

MULTI_CITY_CATEGORY_PATTERN = (
    "Museum|Museums|Park|Parks|Garden|Aquarium|Zoo|"
    "Landmark|Historic|Monument|Observatory|Tour|Attraction|"
    "Beach|Botanical|Theatre|Gallery|Viewpoint|Campus"
)

CALIFORNIA_CITY_COORDS = {
    "San Diego": (32.7157, -117.1611),
    "Los Angeles": (34.0522, -118.2437),
    "Santa Barbara": (34.4208, -119.6982),
    "San Luis Obispo": (35.2828, -120.6596),
    "Monterey": (36.6002, -121.8947),
    "Santa Cruz": (36.9741, -122.0308),
    "San Francisco": (37.7749, -122.4194),
}

CITY_POPULARITY_PRIOR = {
    "Los Angeles": 0.98,
    "San Francisco": 0.96,
    "San Diego": 0.86,
    "Santa Barbara": 0.72,
    "Monterey": 0.70,
    "Santa Cruz": 0.66,
    "San Luis Obispo": 0.58,
}

CITY_ROUTE_IMPORTANCE_PRIOR = {
    "Los Angeles": 1.00,
    "San Francisco": 1.00,
    "Santa Barbara": 0.82,
    "San Luis Obispo": 0.72,
    "Monterey": 0.78,
    "Santa Cruz": 0.70,
    "San Diego": 0.52,
}

CITY_HOTEL_PRICE_PRIOR = {
    "San Francisco": 190,
    "Los Angeles": 170,
    "San Diego": 155,
    "Santa Barbara": 160,
    "Monterey": 150,
    "Santa Cruz": 145,
    "San Luis Obispo": 135,
}

CURATED_SOCIAL_MUST_GO = [
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
        "source_score": 9.4,
        "social_score": 0.96,
        "social_must_go": True,
        "must_go_weight": 0.98,
        "social_reason": "Silicon Valley connector / high-social-value corridor stop, not a pure CA-1 segment.",
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

CALIFORNIA_CORRIDOR_WAYPOINTS = [
    ("San Francisco", 37.7749, -122.4194),
    ("Stanford University Main Quad", 37.4275, -122.1697),
    ("Santa Cruz", 36.9741, -122.0308),
    ("Monterey", 36.6002, -121.8947),
    ("Bixby Creek Bridge Viewpoint", 36.3715, -121.9018),
    ("Big Sur", 36.2704, -121.8081),
    ("San Luis Obispo", 35.2828, -120.6596),
    ("Santa Barbara", 34.4208, -119.6982),
    ("Los Angeles", 34.0522, -118.2437),
]

CANONICAL_POI_COLUMNS = [
    "name",
    "city",
    "latitude",
    "longitude",
    "category",
    "source_list",
    "yelp_rating",
    "yelp_review_count",
    "osm_tags",
    "wikidata_id",
    "wikipedia_title",
    "wikipedia_pageview_score",
    "social_score",
    "social_must_go",
    "must_go_weight",
    "corridor_fit",
    "detour_minutes",
    "data_confidence",
    "final_poi_value",
    "social_reason",
]


def minmax_series(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").fillna(0.0).astype(float)
    if values.empty or np.isclose(values.max(), values.min()):
        return pd.Series(0.5, index=values.index)
    return (values - values.min()) / (values.max() - values.min())


def safe_float(value, default=0.0) -> float:
    try:
        if pd.isna(value):
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def point_distance_km(lat_a, lon_a, lat_b, lon_b) -> float:
    return float(geodesic((float(lat_a), float(lon_a)), (float(lat_b), float(lon_b))).km)


def corridor_metrics(lat, lon, threshold_minutes=45.0) -> tuple[float, float]:
    if pd.isna(lat) or pd.isna(lon):
        return 0.0, float(threshold_minutes)
    nearest_km = min(point_distance_km(lat, lon, wp_lat, wp_lon) for _, wp_lat, wp_lon in CALIFORNIA_CORRIDOR_WAYPOINTS)
    detour_minutes = nearest_km * 2.0 / 48.0 * 60.0
    corridor_fit = max(0.0, 1.0 - detour_minutes / float(threshold_minutes))
    return round(float(corridor_fit), 4), round(float(detour_minutes), 2)


def yelp_city_catalog(city: str, business_df: pd.DataFrame, min_rows=25) -> tuple[pd.DataFrame, str]:
    if business_df.empty or "city" not in business_df.columns or "categories" not in business_df.columns:
        return pd.DataFrame(), "yelp_missing"
    city_rows = business_df[
        business_df["city"].astype(str).str.lower().eq(city.lower())
        & business_df["categories"].notna()
        & business_df["categories"].astype(str).str.contains(MULTI_CITY_CATEGORY_PATTERN, case=False, regex=True)
    ].copy()
    stars = pd.to_numeric(city_rows.get("stars", 0), errors="coerce").fillna(0)
    reviews = pd.to_numeric(city_rows.get("review_count", 0), errors="coerce").fillna(0)
    city_rows["utility"] = stars * np.log1p(reviews)
    city_rows = city_rows.sort_values("utility", ascending=False)
    status = "yelp_sufficient" if len(city_rows) >= min_rows else "yelp_sparse"
    return city_rows.head(100), status


def normalize_yelp_rows(city: str, catalog: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in catalog.iterrows():
        lat = safe_float(row.get("latitude"), np.nan)
        lon = safe_float(row.get("longitude"), np.nan)
        if pd.isna(lat) or pd.isna(lon) or not str(row.get("name", "")).strip():
            continue
        rows.append(
            {
                "name": str(row.get("name")),
                "city": city,
                "latitude": lat,
                "longitude": lon,
                "category": str(row.get("categories", "yelp_attraction")),
                "source_list": "yelp",
                "yelp_rating": safe_float(row.get("stars"), 0.0),
                "yelp_review_count": safe_float(row.get("review_count"), 0.0),
                "osm_tags": "",
                "wikidata_id": "",
                "wikipedia_title": "",
                "wikipedia_pageview_score": 0.0,
                "social_score": 0.0,
                "social_must_go": False,
                "must_go_weight": 0.0,
                "social_reason": "",
                "source_score": safe_float(row.get("utility"), 0.0),
            }
        )
    return pd.DataFrame(rows)


def normalize_osm_rows(city: str, osm_df: pd.DataFrame) -> pd.DataFrame:
    if osm_df.empty or "city" not in osm_df.columns:
        return pd.DataFrame()
    rows = []
    city_rows = osm_df[osm_df["city"].astype(str).str.lower().eq(city.lower())].copy()
    for _, row in city_rows.iterrows():
        lat = safe_float(row.get("latitude"), np.nan)
        lon = safe_float(row.get("longitude"), np.nan)
        if pd.isna(lat) or pd.isna(lon) or not str(row.get("name", "")).strip():
            continue
        osm_tags = {
            key: row.get(key)
            for key in ["tourism", "historic", "leisure", "amenity", "natural", "type"]
            if key in row and pd.notna(row.get(key))
        }
        rows.append(
            {
                "name": str(row.get("name")),
                "city": city,
                "latitude": lat,
                "longitude": lon,
                "category": str(row.get("category", row.get("type", "osm_attraction"))),
                "source_list": str(row.get("source", "osm")),
                "yelp_rating": 0.0,
                "yelp_review_count": 0.0,
                "osm_tags": json.dumps(osm_tags, sort_keys=True),
                "wikidata_id": str(row.get("wikidata", "")) if pd.notna(row.get("wikidata", "")) else "",
                "wikipedia_title": str(row.get("wikipedia", "")) if pd.notna(row.get("wikipedia", "")) else "",
                "wikipedia_pageview_score": 0.0,
                "social_score": 0.0,
                "social_must_go": False,
                "must_go_weight": 0.0,
                "social_reason": "",
                "source_score": safe_float(row.get("source_score"), 1.0),
            }
        )
    return pd.DataFrame(rows)


def normalize_curated_social_rows() -> pd.DataFrame:
    rows = []
    for row in CURATED_SOCIAL_MUST_GO:
        enriched = dict(row)
        enriched.update(
            {
                "source_list": "curated_social_must_go",
                "yelp_rating": 0.0,
                "yelp_review_count": 0.0,
                "osm_tags": "",
                "wikidata_id": "",
                "wikipedia_title": "",
                "wikipedia_pageview_score": 0.0,
            }
        )
        rows.append(enriched)
    return pd.DataFrame(rows)


def fetch_wikidata_wikipedia_stub(row: pd.Series, run_live=False) -> dict:
    """Optional live hook; offline returns an audit-friendly placeholder."""
    if not run_live:
        return {
            "wikidata_id": "",
            "wikipedia_title": "",
            "wikipedia_pageview_score": 0.0,
            "enrichment_status": "offline_placeholder",
        }
    try:
        params = {
            "action": "wbsearchentities",
            "format": "json",
            "language": "en",
            "search": str(row["name"]),
            "limit": 1,
        }
        response = requests.get("https://www.wikidata.org/w/api.php", params=params, timeout=15)
        response.raise_for_status()
        matches = response.json().get("search", [])
        if not matches:
            return {
                "wikidata_id": "",
                "wikipedia_title": "",
                "wikipedia_pageview_score": 0.0,
                "enrichment_status": "wikidata_no_match",
            }
        wikidata_id = matches[0].get("id", "")
        wikipedia_title = matches[0].get("label", str(row["name"]))
        pageview_score = 0.55 if wikidata_id else 0.0
        return {
            "wikidata_id": wikidata_id,
            "wikipedia_title": wikipedia_title,
            "wikipedia_pageview_score": pageview_score,
            "enrichment_status": "wikidata_live_match",
        }
    except Exception as exc:
        return {
            "wikidata_id": "",
            "wikipedia_title": "",
            "wikipedia_pageview_score": 0.0,
            "enrichment_status": f"live_error:{type(exc).__name__}",
        }


def deduplicate_pois(
    raw_df: pd.DataFrame, name_threshold=0.82, distance_threshold_km=0.30
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if raw_df.empty:
        return raw_df.copy(), pd.DataFrame()
    df = raw_df.copy().reset_index(drop=True)
    df["name_key"] = df["name"].astype(str).str.lower().str.replace(r"[^a-z0-9]+", " ", regex=True).str.strip()
    used = set()
    merged_rows = []
    audit_rows = []
    for idx, row in df.iterrows():
        if idx in used:
            continue
        candidates = [idx]
        for other_idx, other in df.iloc[idx + 1 :].iterrows():
            if other_idx in used:
                continue
            similarity = SequenceMatcher(None, row["name_key"], other["name_key"]).ratio()
            distance = point_distance_km(row["latitude"], row["longitude"], other["latitude"], other["longitude"])
            if similarity >= name_threshold and distance <= distance_threshold_km:
                candidates.append(other_idx)
                audit_rows.append(
                    {
                        "kept_name": row["name"],
                        "merged_name": other["name"],
                        "name_similarity": round(float(similarity), 3),
                        "distance_km": round(float(distance), 3),
                        "merge_reason": "name_similarity_and_distance",
                    }
                )
        used.update(candidates)
        group = df.loc[candidates].copy()
        best_coord_row = group.sort_values(
            by="source_list",
            key=lambda s: s.astype(str).map(
                lambda value: (
                    0
                    if "wikidata" in value or "wikipedia" in value
                    else 1
                    if "osm" in value
                    else 2
                    if "yelp" in value
                    else 3
                )
            ),
        ).iloc[0]
        source_list = sorted(set("|".join(group["source_list"].astype(str)).replace(",", "|").split("|")) - {"", "nan"})
        social_score = float(pd.to_numeric(group["social_score"], errors="coerce").fillna(0).max())
        social_must_go = bool(group["social_must_go"].fillna(False).astype(bool).any())
        must_go_weight = float(pd.to_numeric(group.get("must_go_weight", 0), errors="coerce").fillna(0).max())
        merged_rows.append(
            {
                "name": best_coord_row["name"],
                "city": best_coord_row["city"],
                "latitude": float(best_coord_row["latitude"]),
                "longitude": float(best_coord_row["longitude"]),
                "category": best_coord_row.get("category", "attraction"),
                "source_list": "|".join(source_list),
                "yelp_rating": float(pd.to_numeric(group["yelp_rating"], errors="coerce").fillna(0).max()),
                "yelp_review_count": float(pd.to_numeric(group["yelp_review_count"], errors="coerce").fillna(0).max()),
                "osm_tags": next((str(value) for value in group["osm_tags"] if str(value).strip()), ""),
                "wikidata_id": next((str(value) for value in group["wikidata_id"] if str(value).strip()), ""),
                "wikipedia_title": next((str(value) for value in group["wikipedia_title"] if str(value).strip()), ""),
                "wikipedia_pageview_score": float(
                    pd.to_numeric(group["wikipedia_pageview_score"], errors="coerce").fillna(0).max()
                ),
                "social_score": social_score,
                "social_must_go": social_must_go,
                "must_go_weight": must_go_weight,
                "social_reason": next(
                    (
                        str(value)
                        for value in group.get("social_reason", pd.Series(dtype=str)).fillna("")
                        if str(value).strip()
                    ),
                    "",
                ),
                "source_score": float(pd.to_numeric(group["source_score"], errors="coerce").fillna(0).max()),
                "merged_source_count": int(len(group)),
            }
        )
    return pd.DataFrame(merged_rows), pd.DataFrame(audit_rows)


def build_enrichment_outputs(
    *,
    all_business_df: pd.DataFrame,
    city_names: list[str],
    output_dir: Path | str,
    run_live_apis=False,
) -> dict:
    output_dir = Path(output_dir)
    osm_poi_path = output_dir / "production_city_poi_catalog.csv"
    osm_hotel_path = output_dir / "production_city_hotel_catalog.csv"
    osm_city_poi_catalog_df = pd.read_csv(osm_poi_path) if osm_poi_path.exists() else pd.DataFrame()
    osm_city_hotel_catalog_df = pd.read_csv(osm_hotel_path) if osm_hotel_path.exists() else pd.DataFrame()

    raw_catalog_frames = []
    city_source_rows = []
    for city in city_names:
        yelp_catalog, yelp_status = yelp_city_catalog(city, all_business_df)
        yelp_rows = normalize_yelp_rows(city, yelp_catalog)
        osm_rows = normalize_osm_rows(city, osm_city_poi_catalog_df)
        raw_catalog_frames.extend([yelp_rows, osm_rows])
        city_source_rows.append(
            {"city": city, "yelp_rows": int(len(yelp_rows)), "osm_rows": int(len(osm_rows)), "yelp_status": yelp_status}
        )
    raw_catalog_frames.append(normalize_curated_social_rows())
    non_empty = [frame for frame in raw_catalog_frames if not frame.empty]
    raw_poi_catalog_df = pd.concat(non_empty, ignore_index=True, sort=False) if non_empty else pd.DataFrame()

    if raw_poi_catalog_df.empty:
        raw_poi_catalog_df = pd.DataFrame(
            [
                {
                    "name": f"{city} enrichment pending",
                    "city": city,
                    "latitude": CALIFORNIA_CITY_COORDS[city][0],
                    "longitude": CALIFORNIA_CITY_COORDS[city][1],
                    "category": "data_ingestion_needed",
                    "source_list": "city_center_placeholder",
                    "yelp_rating": 0.0,
                    "yelp_review_count": 0.0,
                    "osm_tags": "",
                    "wikidata_id": "",
                    "wikipedia_title": "",
                    "wikipedia_pageview_score": 0.0,
                    "social_score": 0.0,
                    "social_must_go": False,
                    "must_go_weight": 0.0,
                    "social_reason": "",
                    "source_score": 0.0,
                }
                for city in city_names
            ]
        )

    optional_enrichment_rows = []
    if run_live_apis:
        for idx, row in raw_poi_catalog_df.head(40).iterrows():
            enrichment = fetch_wikidata_wikipedia_stub(row, run_live=True)
            for key in ["wikidata_id", "wikipedia_title", "wikipedia_pageview_score"]:
                if enrichment.get(key):
                    raw_poi_catalog_df.at[idx, key] = enrichment[key]
            optional_enrichment_rows.append({"name": row["name"], "city": row["city"], **enrichment})
    else:
        optional_enrichment_rows.append(
            {
                "name": "all",
                "city": "all",
                "enrichment_status": "RUN_LIVE_APIS=0; Wikidata/Wikipedia fields kept schema-compatible",
            }
        )

    enriched_df, dedupe_audit_df = deduplicate_pois(raw_poi_catalog_df)
    for idx, row in enriched_df.iterrows():
        corridor_fit, detour_minutes = corridor_metrics(row["latitude"], row["longitude"])
        enriched_df.at[idx, "corridor_fit"] = corridor_fit
        enriched_df.at[idx, "detour_minutes"] = detour_minutes

    enriched_df["base_score_norm"] = minmax_series(enriched_df["source_score"])
    enriched_df["yelp_signal_norm"] = minmax_series(
        enriched_df["yelp_rating"] * np.log1p(enriched_df["yelp_review_count"])
    )
    enriched_df["data_confidence"] = (
        0.35
        * enriched_df["source_list"].astype(str).str.contains("osm|openstreetmap", case=False, na=False).astype(float)
        + 0.25 * enriched_df["source_list"].astype(str).str.contains("yelp", case=False, na=False).astype(float)
        + 0.20 * enriched_df["source_list"].astype(str).str.contains("curated", case=False, na=False).astype(float)
        + 0.10 * enriched_df["wikidata_id"].astype(str).str.len().gt(0).astype(float)
        + 0.10 * enriched_df["wikipedia_title"].astype(str).str.len().gt(0).astype(float)
    ).clip(0.15, 1.0)
    enriched_df["final_poi_value"] = (
        0.42 * enriched_df["base_score_norm"]
        + 0.20 * enriched_df["yelp_signal_norm"]
        + 1.10 * enriched_df["social_score"]
        + 0.85 * enriched_df["must_go_weight"] * enriched_df["social_score"]
        + 0.30 * enriched_df["corridor_fit"]
        - 0.010 * enriched_df["detour_minutes"]
    ).clip(lower=0.0)
    enriched_df = enriched_df.sort_values(
        ["final_poi_value", "social_score", "source_score"], ascending=[False, False, False]
    ).reset_index(drop=True)
    enriched_df[CANONICAL_POI_COLUMNS].to_csv(output_dir / "production_enriched_poi_catalog.csv", index=False)

    source_audit_df = pd.DataFrame(city_source_rows)
    optional_audit_df = pd.DataFrame(optional_enrichment_rows)
    audit_df = pd.concat(
        [
            source_audit_df.assign(audit_type="source_coverage"),
            dedupe_audit_df.assign(audit_type="deduplication") if not dedupe_audit_df.empty else pd.DataFrame(),
            optional_audit_df.assign(audit_type="wikidata_wikipedia"),
        ],
        ignore_index=True,
        sort=False,
    )
    audit_df.to_csv(output_dir / "production_enrichment_audit.csv", index=False)

    city_summary_df = build_city_catalog_summary(city_names, all_business_df, enriched_df)
    city_summary_df.to_csv(output_dir / "production_city_catalog_summary.csv", index=False)

    social_signal_snapshots_df = city_summary_df.copy()
    social_signal_snapshots_df["source_name"] = "aggregate_multi_source_social_proxy"
    social_signal_snapshots_df["social_popularity_score"] = social_signal_snapshots_df["popularity_score"]
    social_signal_snapshots_df["social_momentum_score"] = social_signal_snapshots_df[
        ["social_signal_score", "external_poi_score"]
    ].mean(axis=1)
    social_signal_snapshots_df["crowding_risk_score"] = (
        0.55 * social_signal_snapshots_df["social_popularity_score"]
        + 0.30 * social_signal_snapshots_df["social_signal_score"]
        + 0.15 * social_signal_snapshots_df["route_importance_score"]
    ).clip(0, 1)
    social_signal_snapshots_df["raw_posts_stored"] = False
    social_signal_snapshots_df.to_csv(output_dir / "social_signal_snapshots.csv", index=False)

    return {
        "osm_city_poi_catalog_df": osm_city_poi_catalog_df,
        "osm_city_hotel_catalog_df": osm_city_hotel_catalog_df,
        "production_enriched_poi_catalog_df": enriched_df,
        "production_enrichment_audit_df": audit_df,
        "city_catalog_summary_df": city_summary_df,
        "social_signal_snapshots_df": social_signal_snapshots_df,
    }


def build_city_catalog_summary(
    city_names: list[str], all_business_df: pd.DataFrame, enriched_df: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    for city in city_names:
        yelp_catalog, yelp_status = yelp_city_catalog(city, all_business_df)
        city_pois = enriched_df[enriched_df["city"].astype(str).str.lower().eq(city.lower())].copy()
        yelp_review_volume = int(
            pd.to_numeric(yelp_catalog.get("review_count", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()
        )
        candidate_attractions = int(len(city_pois))
        popularity_score = max(
            float(CITY_POPULARITY_PRIOR.get(city, 0.50)), min(np.log1p(yelp_review_volume) / np.log1p(25000), 1.0)
        )
        social_signal_score = float(city_pois["social_score"].max()) if not city_pois.empty else 0.0
        external_poi_score = min(candidate_attractions / 60.0, 1.0)
        yelp_signal_score = min(np.log1p(yelp_review_volume) / np.log1p(5000), 1.0) if yelp_review_volume else 0.0
        route_importance_score = float(CITY_ROUTE_IMPORTANCE_PRIOR.get(city, 0.50))
        data_confidence = float(city_pois["data_confidence"].mean()) if not city_pois.empty else 0.15
        city_value_score = (
            0.30 * popularity_score
            + 0.25 * social_signal_score
            + 0.20 * external_poi_score
            + 0.15 * yelp_signal_score
            + 0.10 * route_importance_score
        )
        rows.append(
            {
                "city": city,
                "candidate_attractions": candidate_attractions,
                "mean_utility": float(city_pois["final_poi_value"].mean()) if not city_pois.empty else 0.0,
                "total_reviews": yelp_review_volume,
                "data_status": "multi_source_enriched"
                if candidate_attractions
                else "needs_osm_wikidata_wikipedia_ingestion",
                "fallback_source": "multi_source_enriched_catalog"
                if candidate_attractions
                else "osm_wikidata_wikipedia_ready",
                "popularity_score": round(popularity_score, 3),
                "external_poi_score": round(external_poi_score, 3),
                "social_signal_score": round(social_signal_score, 3),
                "route_importance_score": round(route_importance_score, 3),
                "yelp_signal_score": round(yelp_signal_score, 3),
                "city_value_score": round(float(city_value_score), 3),
                "data_confidence": round(data_confidence, 3),
                "data_uncertainty": round(1.0 - data_confidence, 3),
                "data_quality_score": round(data_confidence, 3),
                "yelp_status": yelp_status,
            }
        )
    return pd.DataFrame(rows)


def hotel_price_proxy_for_city(
    city: str, hotels_df: pd.DataFrame, osm_city_hotel_catalog_df: pd.DataFrame, primary_city="Santa Barbara"
) -> float:
    frames = []
    if city == primary_city and isinstance(hotels_df, pd.DataFrame) and not hotels_df.empty:
        local_hotels = hotels_df.copy()
        local_hotels["city"] = city
        frames.append(local_hotels)
    if (
        isinstance(osm_city_hotel_catalog_df, pd.DataFrame)
        and not osm_city_hotel_catalog_df.empty
        and "city" in osm_city_hotel_catalog_df.columns
    ):
        frames.append(
            osm_city_hotel_catalog_df[osm_city_hotel_catalog_df["city"].astype(str).str.lower().eq(city.lower())]
        )
    if frames:
        merged = pd.concat(frames, ignore_index=True, sort=False)
        price = pd.to_numeric(merged.get("nightly_price", pd.Series(dtype=float)), errors="coerce").dropna()
        if not price.empty and float(price.median()) > 0:
            return float(price.median())
    return float(CITY_HOTEL_PRICE_PRIOR.get(city, 150))


def expand_day_city_sequence(days_by_city: dict) -> list[str]:
    sequence = []
    for city, days in days_by_city.items():
        sequence.extend([city] * int(days))
    return sequence


def estimate_trip_budget(
    trip: dict,
    hotels_df: pd.DataFrame,
    osm_city_hotel_catalog_df: pd.DataFrame,
    output_dir: Path | str,
    trip_days=7,
    user_budget=None,
    primary_city="Santa Barbara",
) -> pd.DataFrame:
    output_dir = Path(output_dir)
    day_base_sequence = expand_day_city_sequence(trip["days_by_city"])
    overnight_bases = day_base_sequence[: max(0, int(trip_days) - 1)]
    expected_hotel_cost = sum(
        hotel_price_proxy_for_city(city, hotels_df, osm_city_hotel_catalog_df, primary_city=primary_city)
        for city in overnight_bases
    )
    low_hotel_cost = expected_hotel_cost * 0.78
    high_hotel_cost = expected_hotel_cost * 1.35

    expected_attraction_cost = 28.0 * trip_days
    low_attraction_cost = 10.0 * trip_days
    high_attraction_cost = 55.0 * trip_days

    expected_food_cost = 72.0 * trip_days
    low_food_cost = 48.0 * trip_days
    high_food_cost = 115.0 * trip_days

    drive_hours = float(trip["intercity_drive_minutes"]) / 60.0
    approximate_drive_miles = drive_hours * 48.0
    expected_drive_cost = approximate_drive_miles * 0.24 + 18.0 * trip_days
    low_drive_cost = approximate_drive_miles * 0.16 + 8.0 * trip_days
    high_drive_cost = approximate_drive_miles * 0.36 + 32.0 * trip_days

    low_before_buffer = low_hotel_cost + low_attraction_cost + low_food_cost + low_drive_cost
    expected_before_buffer = expected_hotel_cost + expected_attraction_cost + expected_food_cost + expected_drive_cost
    high_before_buffer = high_hotel_cost + high_attraction_cost + high_food_cost + high_drive_cost

    low_total = low_before_buffer * 1.08
    expected_total = expected_before_buffer * 1.15
    high_total = high_before_buffer * 1.22

    user_budget = safe_float(user_budget, np.nan)
    soft_budget = float(user_budget) if np.isfinite(user_budget) and user_budget > 0 else float(expected_total)
    hard_budget = max(float(high_total), soft_budget * 1.20)
    budget_df = pd.DataFrame(
        [
            {"component": "hotel", "low": low_hotel_cost, "expected": expected_hotel_cost, "high": high_hotel_cost},
            {
                "component": "attractions",
                "low": low_attraction_cost,
                "expected": expected_attraction_cost,
                "high": high_attraction_cost,
            },
            {"component": "food", "low": low_food_cost, "expected": expected_food_cost, "high": high_food_cost},
            {
                "component": "fuel_parking_local_transport",
                "low": low_drive_cost,
                "expected": expected_drive_cost,
                "high": high_drive_cost,
            },
            {"component": "buffered_total", "low": low_total, "expected": expected_total, "high": high_total},
            {"component": "soft_budget", "low": soft_budget, "expected": soft_budget, "high": soft_budget},
            {"component": "hard_budget", "low": hard_budget, "expected": hard_budget, "high": hard_budget},
        ]
    ).round(2)
    budget_df.to_csv(output_dir / "production_budget_estimate.csv", index=False)
    return budget_df


ROUTE_SEARCH_STRATEGIES = {
    "scenic_social": {
        "reward_bias": 0.08,
        "runtime_minutes_per_city": 2.8,
        "poi_value_weight": 0.34,
        "social_weight": 0.34,
        "must_go_weight": 0.28,
        "corridor_weight": 0.18,
        "detour_penalty": 0.010,
        "cost_multiplier": 1.06,
        "optimizer_backend": "fast_local_route_oracle",
    },
    "fastest_low_cost": {
        "reward_bias": 0.03,
        "runtime_minutes_per_city": 1.4,
        "poi_value_weight": 0.18,
        "social_weight": 0.12,
        "must_go_weight": 0.10,
        "corridor_weight": 0.05,
        "detour_penalty": 0.018,
        "cost_multiplier": 0.88,
        "optimizer_backend": "fast_local_route_oracle",
    },
    "balanced": {
        "reward_bias": 0.06,
        "runtime_minutes_per_city": 2.1,
        "poi_value_weight": 0.26,
        "social_weight": 0.22,
        "must_go_weight": 0.20,
        "corridor_weight": 0.12,
        "detour_penalty": 0.012,
        "cost_multiplier": 1.00,
        "optimizer_backend": "fast_local_route_oracle",
    },
    "high_must_go": {
        "reward_bias": 0.07,
        "runtime_minutes_per_city": 2.5,
        "poi_value_weight": 0.24,
        "social_weight": 0.30,
        "must_go_weight": 0.38,
        "corridor_weight": 0.10,
        "detour_penalty": 0.009,
        "cost_multiplier": 1.08,
        "optimizer_backend": "fast_local_route_oracle_with_optional_gurobi_repair",
    },
    "low_detour": {
        "reward_bias": 0.04,
        "runtime_minutes_per_city": 1.7,
        "poi_value_weight": 0.22,
        "social_weight": 0.12,
        "must_go_weight": 0.12,
        "corridor_weight": 0.22,
        "detour_penalty": 0.022,
        "cost_multiplier": 0.94,
        "optimizer_backend": "fast_local_route_oracle",
    },
}


@dataclass(frozen=True)
class BudgetBounds:
    soft_budget: float
    hard_budget: float
    expected_total: float


def budget_bounds_from_df(budget_df: pd.DataFrame) -> BudgetBounds:
    return BudgetBounds(
        soft_budget=float(budget_df.loc[budget_df["component"].eq("soft_budget"), "expected"].iloc[0]),
        hard_budget=float(budget_df.loc[budget_df["component"].eq("hard_budget"), "expected"].iloc[0]),
        expected_total=float(budget_df.loc[budget_df["component"].eq("buffered_total"), "expected"].iloc[0]),
    )


def plan_enriched_signal(plan: dict, enriched_df: pd.DataFrame) -> dict:
    stay_cities = set(plan["days_by_city"].keys())
    if not isinstance(enriched_df, pd.DataFrame) or enriched_df.empty:
        return {
            "poi_value_signal": 0.0,
            "social_signal": 0.0,
            "must_go_signal": 0.0,
            "corridor_signal": 0.0,
            "detour_minutes_signal": 0.0,
            "candidate_bundle_size": 0,
        }
    poi_pool = enriched_df[
        enriched_df["city"].isin(stay_cities)
        | enriched_df["corridor_fit"].gt(0.55)
        | enriched_df["social_must_go"].astype(bool)
    ].copy()
    if poi_pool.empty:
        return {
            "poi_value_signal": 0.0,
            "social_signal": 0.0,
            "must_go_signal": 0.0,
            "corridor_signal": 0.0,
            "detour_minutes_signal": 0.0,
            "candidate_bundle_size": 0,
        }
    top_pool = poi_pool.sort_values("final_poi_value", ascending=False).head(25)
    return {
        "poi_value_signal": float(top_pool["final_poi_value"].mean()),
        "social_signal": float(top_pool["social_score"].mean()),
        "must_go_signal": float((top_pool["must_go_weight"] * top_pool["social_score"]).mean()),
        "corridor_signal": float(top_pool["corridor_fit"].mean()),
        "detour_minutes_signal": float(top_pool["detour_minutes"].mean()),
        "candidate_bundle_size": int(len(top_pool)),
    }


def build_hybrid_bandit_arms(
    hierarchical_candidates: list[dict],
    enriched_df: pd.DataFrame,
    budget_df: pd.DataFrame,
    route_strategies=None,
) -> pd.DataFrame:
    route_strategies = route_strategies or ROUTE_SEARCH_STRATEGIES
    budget = budget_bounds_from_df(budget_df)
    rows = []
    for plan in hierarchical_candidates:
        stay_cities = list(plan["days_by_city"].keys())
        enriched_signal = plan_enriched_signal(plan, enriched_df)
        for strategy_name, strategy_cfg in route_strategies.items():
            estimated_solver_minutes = strategy_cfg["runtime_minutes_per_city"] * len(stay_cities)
            estimated_total_cost = budget.expected_total * float(strategy_cfg.get("cost_multiplier", 1.0))
            soft_budget_penalty = max(0.0, estimated_total_cost - budget.soft_budget) / max(1.0, budget.soft_budget)
            hard_budget_feasible = estimated_total_cost <= budget.hard_budget
            enriched_reward = (
                strategy_cfg["poi_value_weight"] * enriched_signal["poi_value_signal"]
                + strategy_cfg["social_weight"] * enriched_signal["social_signal"]
                + strategy_cfg["must_go_weight"] * enriched_signal["must_go_signal"]
                + strategy_cfg["corridor_weight"] * enriched_signal["corridor_signal"]
                - strategy_cfg["detour_penalty"] * enriched_signal["detour_minutes_signal"]
            )
            rows.append(
                {
                    "arm_id": f"{plan['gateway_start']}->{plan['gateway_end']}|{strategy_name}",
                    "gateway_start": plan["gateway_start"],
                    "gateway_end": plan["gateway_end"],
                    "city_sequence": " -> ".join(plan["city_sequence"]),
                    "days_by_city": plan["days_by_city"],
                    "route_search_strategy": strategy_name,
                    "candidate_bundle": strategy_name,
                    "bandit_role": "route_finding_candidate_search",
                    "optimizer_role": strategy_cfg["optimizer_backend"],
                    "small_optimizer_calls": len(stay_cities),
                    "hierarchical_objective": plan["objective"],
                    "estimated_solver_minutes": estimated_solver_minutes,
                    "estimated_total_cost": estimated_total_cost,
                    "soft_budget": budget.soft_budget,
                    "hard_budget": budget.hard_budget,
                    "hard_budget_feasible": bool(hard_budget_feasible),
                    "soft_budget_penalty": soft_budget_penalty,
                    **enriched_signal,
                    "reward_prior": (
                        plan["objective"]
                        + strategy_cfg["reward_bias"]
                        + enriched_reward
                        - 0.012 * estimated_solver_minutes
                        - 0.45 * soft_budget_penalty
                        - (0.40 if not hard_budget_feasible else 0.0)
                    ),
                }
            )
    return pd.DataFrame(rows)


def route_optimizer_oracle_reward(arm_row: pd.Series, rng: np.random.Generator) -> float:
    """Proxy for fast route construction, with optional Gurobi repair outside the default path."""
    feasibility_bonus = 0.08 if bool(arm_row["hard_budget_feasible"]) else -0.25
    bundle_quality_bonus = 0.04 * min(float(arm_row["candidate_bundle_size"]) / 20.0, 1.0)
    runtime_penalty = 0.010 * float(arm_row["estimated_solver_minutes"])
    budget_penalty = 0.35 * float(arm_row["soft_budget_penalty"])
    return (
        float(arm_row["reward_prior"])
        + feasibility_bonus
        + bundle_quality_bonus
        - runtime_penalty
        - budget_penalty
        + rng.normal(0.0, 0.04)
    )


def run_hybrid_bandit_optimization_search(
    arms_df: pd.DataFrame, episodes=40, seed=42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    local_rng = np.random.default_rng(seed)
    arm_ids = arms_df["arm_id"].tolist()
    pulls = {arm_id: 0 for arm_id in arm_ids}
    posterior_means = {arm_id: 0.0 for arm_id in arm_ids}
    logs = []

    for episode in range(1, episodes + 1):
        sampled_values = {
            arm_id: posterior_means[arm_id] + local_rng.normal(0.0, 1.0 / math.sqrt(pulls[arm_id] + 1))
            for arm_id in arm_ids
        }
        chosen_arm_id = max(sampled_values, key=sampled_values.get)
        arm_row = arms_df.loc[arms_df["arm_id"].eq(chosen_arm_id)].iloc[0]
        reward = route_optimizer_oracle_reward(arm_row, local_rng)

        pulls[chosen_arm_id] += 1
        posterior_means[chosen_arm_id] += (reward - posterior_means[chosen_arm_id]) / pulls[chosen_arm_id]
        logs.append(
            {
                "episode": episode,
                "chosen_arm": chosen_arm_id,
                "route_search_strategy": arm_row["route_search_strategy"],
                "gateway": f"{arm_row['gateway_start']} -> {arm_row['gateway_end']}",
                "candidate_bundle": arm_row["candidate_bundle"],
                "small_optimizer_calls": int(arm_row["small_optimizer_calls"]),
                "estimated_solver_minutes": float(arm_row["estimated_solver_minutes"]),
                "estimated_total_cost": float(arm_row["estimated_total_cost"]),
                "soft_budget_penalty": float(arm_row["soft_budget_penalty"]),
                "reward": reward,
                "posterior_mean": posterior_means[chosen_arm_id],
            }
        )

    summary = arms_df.assign(
        pulls=arms_df["arm_id"].map(pulls),
        posterior_mean_reward=arms_df["arm_id"].map(posterior_means),
    ).sort_values(["posterior_mean_reward", "pulls"], ascending=False)
    return pd.DataFrame(logs), summary
