"""Open-data enrichment entrypoints.

Yelp is treated as the local academic dataset only. Live enrichment uses free
open sources and writes cache/audit artifacts so the notebook stays
reproducible when network access is off.
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from geopy.distance import geodesic

from ._legacy import import_legacy_module
from .config import TripConfig
from .nature_catalog import (
    NATURE_POI_COLUMNS,
    compute_interest_adjusted_values,
    ensure_nature_columns,
    load_nps_or_curated_nature_pois,
    mark_itinerary_eligible,
    write_interest_catalog_artifacts,
)
from .nature_site_routes import (
    NATURE_SITE_ROUTE_METRIC_COLUMNS,
    apply_nature_site_route_score_bonus,
    build_nature_site_route_artifacts,
)
from .region_scenarios import all_scenario_coordinates, get_route_options, scenario_enrichment_city_universe
from .utility_model import apply_utility_models, learning_to_rank_audit

USER_AGENT = "IE5533-WeatherAwareTravelPlanner/1.0 (academic project; open data cache)"

OPEN_ENRICHED_POI_COLUMNS = [
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
    "route_fit",
    "detour_minutes",
    "source_coverage_score",
    "identity_coverage",
    "location_coverage",
    "semantic_coverage",
    "popularity_coverage",
    "temporal_coverage",
    "routing_coverage",
    "model_uncertainty",
    "annotation_uncertainty",
    "entity_match_confidence",
    "data_confidence",
    "data_uncertainty",
    "final_poi_value",
    "social_reason",
    "description",
    "image_url",
    "website_url",
    "source_url",
    "source_confidence",
    "detail_source",
    "itinerary_eligible",
    "itinerary_exclusion_reason",
    *NATURE_SITE_ROUTE_METRIC_COLUMNS,
    *NATURE_POI_COLUMNS,
]

CURATED_PLACE_DETAIL_FALLBACKS = {
    "stanford university main quad": {
        "description": "Historic Stanford campus core with sandstone arcades, courtyards, and strong Bay Area cultural value.",
        "image_url": "https://commons.wikimedia.org/wiki/Special:FilePath/Stanford_University_Main_Quad_May_2011_001.jpg",
        "source_url": "https://www.stanford.edu/",
    },
    "golden gate bridge": {
        "description": "Iconic San Francisco suspension bridge and viewpoint with high scenic and history value.",
        "image_url": "https://commons.wikimedia.org/wiki/Special:FilePath/Golden_Gate_Bridge_as_seen_from_Battery_East.jpg",
        "source_url": "https://www.goldengate.org/bridge/",
    },
    "bixby creek bridge viewpoint": {
        "description": "Big Sur coastal viewpoint centered on the Bixby Creek Bridge and Pacific Coast Highway scenery.",
        "image_url": "https://commons.wikimedia.org/wiki/Special:FilePath/Bixby_Bridge_Big_Sur_California.jpg",
        "source_url": "https://www.parks.ca.gov/?page_id=570",
    },
    "yosemite national park": {
        "description": "Sierra Nevada national park known for Yosemite Valley, granite cliffs, waterfalls, and landmark hiking scenery.",
        "image_url": "https://commons.wikimedia.org/wiki/Special:FilePath/Yosemite_Valley_from_Tunnel_View,_widescreen.jpg",
        "source_url": "https://www.nps.gov/yose/index.htm",
    },
    "sequoia national park": {
        "description": "National park protecting giant sequoia groves and Sierra Nevada mountain landscapes.",
        "image_url": "https://commons.wikimedia.org/wiki/Special:FilePath/General_Sherman_Tree_in_Sequoia_National_Park.jpg",
        "source_url": "https://www.nps.gov/seki/index.htm",
    },
    "kings canyon national park": {
        "description": "Sierra Nevada national park with deep granite canyon scenery, forests, and high-country wilderness access.",
        "image_url": "https://commons.wikimedia.org/wiki/Special:FilePath/Kings_Canyon_National_Park_-_Zumwalt_Meadow.jpg",
        "source_url": "https://www.nps.gov/seki/index.htm",
    },
    "griffith observatory": {
        "description": "Los Angeles observatory and hilltop viewpoint combining city views, science exhibits, and Griffith Park access.",
        "image_url": "https://commons.wikimedia.org/wiki/Special:FilePath/Griffith_Observatory_2015.jpg",
        "source_url": "https://griffithobservatory.org/",
    },
    "hollywood walk of fame": {
        "description": "Hollywood boulevard landmark with entertainment-history value and dense Los Angeles visitor context.",
        "image_url": "https://commons.wikimedia.org/wiki/Special:FilePath/Hollywood_Walk_of_Fame,_Los_Angeles,_California.jpg",
        "source_url": "https://walkoffame.com/",
    },
    "santa barbara county courthouse": {
        "description": "Spanish Colonial Revival civic landmark in Santa Barbara with architecture, history, and viewpoint value.",
        "image_url": "https://commons.wikimedia.org/wiki/Special:FilePath/Santa_Barbara_County_Courthouse,_Santa_Barbara,_California.jpg",
        "source_url": "https://www.countyofsb.org/1065/Courthouse",
    },
}

HOTEL_TYPE_PRICE_PRIOR = {
    "hotel": 150.0,
    "motel": 95.0,
    "hostel": 60.0,
    "guest_house": 125.0,
    "apartment": 135.0,
}

HOTEL_TYPE_RATING_PRIOR = {
    "hotel": 4.0,
    "motel": 3.3,
    "hostel": 3.5,
    "guest_house": 4.2,
    "apartment": 4.1,
}

CURATED_FALLBACK_HOTELS = [
    {
        "city": "San Francisco",
        "name": "Hotel Zephyr San Francisco",
        "latitude": 37.8070,
        "longitude": -122.4141,
        "type": "hotel",
        "rating_score": 4.1,
        "website_url": "https://www.hotelzephyrsf.com/",
    },
    {
        "city": "Mariposa",
        "name": "Best Western Plus Yosemite Way Station Motel",
        "latitude": 37.4895,
        "longitude": -119.9662,
        "type": "motel",
        "rating_score": 3.8,
        "website_url": "https://www.bestwestern.com/",
    },
    {
        "city": "Oakhurst",
        "name": "Yosemite Southgate Hotel & Suites",
        "latitude": 37.3285,
        "longitude": -119.6499,
        "type": "hotel",
        "rating_score": 3.9,
        "website_url": "https://www.yosemitesouthgate.com/",
    },
    {
        "city": "Fresno",
        "name": "DoubleTree by Hilton Fresno Convention Center",
        "latitude": 36.7338,
        "longitude": -119.7840,
        "type": "hotel",
        "rating_score": 3.9,
        "website_url": "https://www.hilton.com/",
    },
    {
        "city": "Three Rivers",
        "name": "Comfort Inn & Suites Sequoia Kings Canyon",
        "latitude": 36.4488,
        "longitude": -118.9018,
        "type": "hotel",
        "rating_score": 3.7,
        "website_url": "https://www.choicehotels.com/",
    },
    {
        "city": "Los Angeles",
        "name": "The LINE LA",
        "latitude": 34.0628,
        "longitude": -118.3009,
        "type": "hotel",
        "rating_score": 4.0,
        "website_url": "https://www.thelinehotel.com/los-angeles/",
    },
    {
        "city": "Monterey",
        "name": "Portola Hotel & Spa at Monterey Bay",
        "latitude": 36.6022,
        "longitude": -121.8947,
        "type": "hotel",
        "rating_score": 4.1,
        "website_url": "https://www.portolahotel.com/",
    },
    {
        "city": "Santa Barbara",
        "name": "Hotel Milo Santa Barbara",
        "latitude": 34.4106,
        "longitude": -119.6918,
        "type": "hotel",
        "rating_score": 3.9,
        "website_url": "https://www.hotelmilosantabarbara.com/",
    },
]


def _project_cache_dir(output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    if output_dir.name == "outputs":
        cache_dir = output_dir.parent / "cache"
    else:
        cache_dir = output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _read_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _cache_fresh(path: Path, ttl_days: int) -> bool:
    if not path.exists():
        return False
    modified = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
    return datetime.now(UTC) - modified <= timedelta(days=int(ttl_days))


def _safe_name(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in str(value)).strip("_")


def _stable_hash(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:16]


def _request_json(url: str, *, params: dict | None = None, timeout: int = 30) -> dict:
    response = requests.get(url, params=params, headers={"User-Agent": USER_AGENT}, timeout=timeout)
    response.raise_for_status()
    return response.json()


def _city_coordinates(city: str) -> tuple[float, float]:
    scenario_coords = all_scenario_coordinates()
    if city in scenario_coords:
        return float(scenario_coords[city][0]), float(scenario_coords[city][1])
    production_enrichment = import_legacy_module("production_enrichment")
    coords = production_enrichment.CALIFORNIA_CITY_COORDS
    if city not in coords:
        raise KeyError(f"No coordinate fallback for city: {city}")
    return float(coords[city][0]), float(coords[city][1])


def _hotel_price_proxy(city: str, lodging_type: str, config: TripConfig) -> float:
    production_enrichment = import_legacy_module("production_enrichment")
    city_prior = production_enrichment.CITY_HOTEL_PRICE_PRIOR.get(
        city,
        float(config.get("hotels", "hotel_price_fallback", 150)),
    )
    type_prior = HOTEL_TYPE_PRICE_PRIOR.get(str(lodging_type), HOTEL_TYPE_PRICE_PRIOR["hotel"])
    return round(0.65 * float(city_prior) + 0.35 * type_prior, 2)


def _curated_hotel_fallback_rows(city_names: list[str], existing: pd.DataFrame, config: TripConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    existing = existing.copy() if isinstance(existing, pd.DataFrame) else pd.DataFrame()
    existing_cities = (
        set(existing.get("city", pd.Series(dtype=str)).dropna().astype(str).str.lower())
        if not existing.empty and "city" in existing.columns
        else set()
    )
    target_cities = {str(city).lower() for city in city_names}
    rows = []
    audit_rows = []
    for hotel in CURATED_FALLBACK_HOTELS:
        city = str(hotel["city"])
        city_key = city.lower()
        should_use = city_key in target_cities and city_key not in existing_cities
        audit_rows.append(
            {
                "city": city,
                "hotel_name": hotel["name"],
                "fallback_used": bool(should_use),
                "fallback_reason": "no_open_hotel_rows_for_city" if should_use else "open_or_existing_hotel_rows_present",
                "source": "curated_open_web_hotel_fallback",
                "source_confidence": 0.55,
            }
        )
        if not should_use:
            continue
        lodging_type = str(hotel.get("type", "hotel"))
        nightly_price = _hotel_price_proxy(city, lodging_type, config)
        rows.append(
            {
                "name": hotel["name"],
                "city": city,
                "latitude": float(hotel["latitude"]),
                "longitude": float(hotel["longitude"]),
                "type": lodging_type,
                "source": "curated_open_web_hotel_fallback",
                "source_list": "curated_open_web_hotel_fallback",
                "source_confidence": 0.55,
                "fallback_reason": "no_open_hotel_rows_for_city",
                "rating_score": float(hotel.get("rating_score", HOTEL_TYPE_RATING_PRIOR.get(lodging_type, 4.0))),
                "review_count": 0,
                "nightly_price": nightly_price,
                "price_source": "city_type_prior_curated_fallback",
                "price_estimated": True,
                "experience_score": float(hotel.get("rating_score", HOTEL_TYPE_RATING_PRIOR.get(lodging_type, 4.0))),
                "value_score": round(
                    float(hotel.get("rating_score", HOTEL_TYPE_RATING_PRIOR.get(lodging_type, 4.0)))
                    / max(nightly_price, 1.0)
                    * 100.0,
                    3,
                ),
                "website_url": hotel.get("website_url", ""),
                "source_url": hotel.get("website_url", ""),
                "osm_tags": "{}",
            }
        )
    fallback_df = pd.DataFrame(rows)
    audit_df = pd.DataFrame(audit_rows)
    if existing.empty:
        combined = fallback_df
    elif fallback_df.empty:
        combined = existing
    else:
        combined = pd.concat([existing, fallback_df], ignore_index=True, sort=False)
    if not combined.empty:
        combined = combined.drop_duplicates(subset=["city", "name"], keep="first").reset_index(drop=True)
    return combined, audit_df


def _source_score_from_tags(tags: dict[str, Any]) -> float:
    score = 1.0
    if tags.get("wikidata") or tags.get("wikipedia"):
        score += 2.0
    if tags.get("heritage") or tags.get("historic"):
        score += 1.5
    if tags.get("tourism") in {"museum", "attraction", "viewpoint", "zoo", "aquarium"}:
        score += 1.0
    if tags.get("name"):
        score += 0.5
    return float(score)


def _overpass_query(city: str, radius_m: int, kind: str) -> str:
    lat, lon = _city_coordinates(city)
    if kind == "hotels":
        body = f"""
        node["tourism"~"hotel|motel|hostel|guest_house|apartment"](around:{radius_m},{lat},{lon});
        way["tourism"~"hotel|motel|hostel|guest_house|apartment"](around:{radius_m},{lat},{lon});
        relation["tourism"~"hotel|motel|hostel|guest_house|apartment"](around:{radius_m},{lat},{lon});
        """
    else:
        body = f"""
        node["tourism"~"attraction|museum|gallery|zoo|aquarium|viewpoint|theme_park|camp_site|picnic_site|wilderness_hut"](around:{radius_m},{lat},{lon});
        way["tourism"~"attraction|museum|gallery|zoo|aquarium|viewpoint|theme_park|camp_site|picnic_site|wilderness_hut"](around:{radius_m},{lat},{lon});
        relation["tourism"~"attraction|museum|gallery|zoo|aquarium|viewpoint|theme_park|camp_site|picnic_site|wilderness_hut"](around:{radius_m},{lat},{lon});
        node["historic"](around:{radius_m},{lat},{lon});
        way["historic"](around:{radius_m},{lat},{lon});
        node["leisure"~"park|garden|nature_reserve"](around:{radius_m},{lat},{lon});
        way["leisure"~"park|garden|nature_reserve"](around:{radius_m},{lat},{lon});
        relation["leisure"~"park|garden|nature_reserve"](around:{radius_m},{lat},{lon});
        node["boundary"~"national_park|protected_area"](around:{radius_m},{lat},{lon});
        way["boundary"~"national_park|protected_area"](around:{radius_m},{lat},{lon});
        relation["boundary"~"national_park|protected_area"](around:{radius_m},{lat},{lon});
        node["natural"~"viewpoint|beach|peak|waterfall|wood|scrub|cliff"](around:{radius_m},{lat},{lon});
        way["natural"~"viewpoint|beach|peak|waterfall|wood|scrub|cliff"](around:{radius_m},{lat},{lon});
        node["highway"~"trailhead|path"](around:{radius_m},{lat},{lon});
        way["highway"~"path|footway|track"](around:{radius_m},{lat},{lon});
        relation["route"~"hiking|foot"](around:{radius_m},{lat},{lon});
        node["amenity"~"theatre|arts_centre"](around:{radius_m},{lat},{lon});
        way["amenity"~"theatre|arts_centre"](around:{radius_m},{lat},{lon});
        """
    return f"[out:json][timeout:35];({body});out center tags;"


def _load_or_fetch_overpass(city: str, *, kind: str, cache_dir: Path, config: TripConfig) -> tuple[list[dict], str]:
    radius = int(config.get("enrichment", "overpass_radius_meters", 18000))
    ttl_days = int(config.get("enrichment", "cache_ttl_days", 30))
    cache_path = cache_dir / f"open_overpass_{kind}_{_safe_name(city)}_{radius}.json"
    cached = _read_json(cache_path)
    if cached and (not config.run_live_apis or _cache_fresh(cache_path, ttl_days)):
        return list(cached.get("elements", [])), "overpass_cache"
    if not (config.run_live_apis and bool(config.get("enrichment", "use_osm", True))):
        return list(
            cached.get("elements", [])
        ) if cached else [], "overpass_offline_missing" if not cached else "overpass_cache_stale_offline"
    try:
        payload = _request_json(
            "https://overpass-api.de/api/interpreter",
            params={"data": _overpass_query(city, radius, kind)},
            timeout=45,
        )
        payload["_fetched_at"] = datetime.now(UTC).isoformat()
        payload["_source"] = "overpass_live"
        _write_json(cache_path, payload)
        time.sleep(1.0)
        return list(payload.get("elements", [])), "overpass_live"
    except Exception as exc:
        if cached:
            return list(cached.get("elements", [])), f"overpass_cache_after_live_error:{type(exc).__name__}"
        return [], f"overpass_live_error:{type(exc).__name__}"


def _overpass_elements_to_pois(city: str, elements: list[dict], source_status: str) -> pd.DataFrame:
    rows = []
    for element in elements:
        tags = element.get("tags", {}) or {}
        name = str(tags.get("name", "")).strip()
        lat = element.get("lat", element.get("center", {}).get("lat"))
        lon = element.get("lon", element.get("center", {}).get("lon"))
        if not name or lat is None or lon is None:
            continue
        category = (
            tags.get("tourism")
            or tags.get("historic")
            or tags.get("leisure")
            or tags.get("amenity")
            or tags.get("natural")
            or tags.get("boundary")
            or tags.get("route")
            or tags.get("highway")
            or "osm_attraction"
        )
        rows.append(
            {
                "name": name,
                "city": city,
                "latitude": float(lat),
                "longitude": float(lon),
                "category": str(category),
                "source": source_status,
                "source_list": source_status,
                "tourism": tags.get("tourism"),
                "historic": tags.get("historic"),
                "leisure": tags.get("leisure"),
                "amenity": tags.get("amenity"),
                "natural": tags.get("natural"),
                "boundary": tags.get("boundary"),
                "route": tags.get("route"),
                "highway": tags.get("highway"),
                "wikidata": tags.get("wikidata", ""),
                "wikipedia": tags.get("wikipedia", ""),
                "website_url": tags.get("website") or tags.get("url") or "",
                "source_url": tags.get("website") or tags.get("url") or "",
                "image_url": tags.get("image") or tags.get("wikimedia_commons") or "",
                "description": tags.get("description") or tags.get("inscription") or "",
                "detail_source": "openstreetmap_tags",
                "osm_tags": json.dumps(tags, sort_keys=True),
                "source_score": _source_score_from_tags(tags),
            }
        )
    return pd.DataFrame(rows).drop_duplicates(subset=["name", "latitude", "longitude"]) if rows else pd.DataFrame()


def _overpass_elements_to_hotels(
    city: str, elements: list[dict], source_status: str, config: TripConfig
) -> pd.DataFrame:
    rows = []
    for element in elements:
        tags = element.get("tags", {}) or {}
        name = str(tags.get("name", "")).strip()
        lat = element.get("lat", element.get("center", {}).get("lat"))
        lon = element.get("lon", element.get("center", {}).get("lon"))
        if not name or lat is None or lon is None:
            continue
        lodging_type = str(tags.get("tourism", "hotel"))
        nightly_price = _hotel_price_proxy(city, lodging_type, config)
        rating_score = HOTEL_TYPE_RATING_PRIOR.get(lodging_type, HOTEL_TYPE_RATING_PRIOR["hotel"])
        rows.append(
            {
                "name": name,
                "city": city,
                "latitude": float(lat),
                "longitude": float(lon),
                "type": lodging_type,
                "source": source_status,
                "source_list": source_status,
                "rating_score": rating_score,
                "review_count": 0,
                "nightly_price": nightly_price,
                "price_source": "osm_type_city_estimate",
                "price_estimated": True,
                "experience_score": rating_score,
                "value_score": round(rating_score / max(nightly_price, 1.0) * 100.0, 3),
                "osm_tags": json.dumps(tags, sort_keys=True),
            }
        )
    return pd.DataFrame(rows).drop_duplicates(subset=["name", "latitude", "longitude"]) if rows else pd.DataFrame()


def build_open_osm_catalogs(
    city_names: list[str], output_dir: str | Path, config: TripConfig
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fetch or load OSM POI/hotel catalogs before legacy scoring."""
    output_dir = Path(output_dir)
    cache_dir = _project_cache_dir(output_dir)
    poi_frames = []
    hotel_frames = []
    audit_rows = []
    existing_poi = (
        pd.read_csv(output_dir / "production_city_poi_catalog.csv")
        if (output_dir / "production_city_poi_catalog.csv").exists()
        else pd.DataFrame()
    )
    existing_hotel = (
        pd.read_csv(output_dir / "production_city_hotel_catalog.csv")
        if (output_dir / "production_city_hotel_catalog.csv").exists()
        else pd.DataFrame()
    )

    if not bool(config.get("enrichment", "use_osm", True)):
        if bool(config.get("hotels", "use_curated_fallback", True)):
            existing_hotel, hotel_fallback_audit = _curated_hotel_fallback_rows(city_names, existing_hotel, config)
            hotel_fallback_audit.to_csv(output_dir / "production_hotel_fallback_audit.csv", index=False)
            if not existing_hotel.empty:
                existing_hotel.to_csv(output_dir / "production_city_hotel_catalog.csv", index=False)
        return (
            existing_poi,
            existing_hotel,
            pd.DataFrame([{"audit_type": "osm", "source_status": "disabled_by_config"}]),
        )

    for city in city_names:
        poi_elements, poi_status = _load_or_fetch_overpass(city, kind="pois", cache_dir=cache_dir, config=config)
        hotel_elements, hotel_status = _load_or_fetch_overpass(city, kind="hotels", cache_dir=cache_dir, config=config)
        poi_frame = _overpass_elements_to_pois(city, poi_elements, poi_status)
        hotel_frame = _overpass_elements_to_hotels(city, hotel_elements, hotel_status, config)
        if poi_frame.empty and not existing_poi.empty:
            poi_frame = existing_poi[existing_poi["city"].astype(str).str.lower().eq(city.lower())].copy()
        if hotel_frame.empty and not existing_hotel.empty:
            hotel_frame = existing_hotel[existing_hotel["city"].astype(str).str.lower().eq(city.lower())].copy()
        poi_frames.append(poi_frame)
        hotel_frames.append(hotel_frame)
        audit_rows.append(
            {
                "audit_type": "osm",
                "city": city,
                "poi_source_status": poi_status,
                "hotel_source_status": hotel_status,
                "poi_rows": int(len(poi_frame)),
                "hotel_rows": int(len(hotel_frame)),
            }
        )

    poi_catalog = (
        pd.concat([frame for frame in poi_frames if not frame.empty], ignore_index=True, sort=False)
        if any(not frame.empty for frame in poi_frames)
        else existing_poi
    )
    hotel_catalog = (
        pd.concat([frame for frame in hotel_frames if not frame.empty], ignore_index=True, sort=False)
        if any(not frame.empty for frame in hotel_frames)
        else existing_hotel
    )
    if bool(config.get("hotels", "use_curated_fallback", True)):
        hotel_catalog, hotel_fallback_audit = _curated_hotel_fallback_rows(city_names, hotel_catalog, config)
    else:
        hotel_fallback_audit = pd.DataFrame(
            [{"fallback_used": False, "fallback_reason": "disabled_by_config", "source": "curated_open_web_hotel_fallback"}]
        )
    if not poi_catalog.empty:
        poi_catalog.to_csv(output_dir / "production_city_poi_catalog.csv", index=False)
    if not hotel_catalog.empty:
        hotel_catalog.to_csv(output_dir / "production_city_hotel_catalog.csv", index=False)
    hotel_fallback_audit.to_csv(output_dir / "production_hotel_fallback_audit.csv", index=False)
    return poi_catalog, hotel_catalog, pd.DataFrame(audit_rows)


def _wikidata_search(name: str, cache_dir: Path, config: TripConfig) -> tuple[dict, str]:
    if not bool(config.get("enrichment", "use_wikidata", True)):
        return {}, "wikidata_disabled"
    ttl_days = int(config.get("enrichment", "cache_ttl_days", 30))
    cache_path = cache_dir / f"open_wikidata_{_stable_hash(name)}.json"
    cached = _read_json(cache_path)
    if cached and (not config.run_live_apis or _cache_fresh(cache_path, ttl_days)):
        return cached, "wikidata_cache"
    if not config.run_live_apis:
        return cached or {}, "wikidata_offline_missing" if not cached else "wikidata_cache_stale_offline"
    try:
        payload = _request_json(
            "https://www.wikidata.org/w/api.php",
            params={"action": "wbsearchentities", "format": "json", "language": "en", "search": name, "limit": 1},
            timeout=20,
        )
        _write_json(cache_path, payload)
        return payload, "wikidata_live"
    except Exception as exc:
        return cached or {}, f"wikidata_live_error:{type(exc).__name__}"


def _wikipedia_summary(title: str, cache_dir: Path, config: TripConfig) -> tuple[dict, str]:
    if not (title and bool(config.get("enrichment", "use_wikipedia", True))):
        return {}, "wikipedia_disabled_or_missing_title"
    ttl_days = int(config.get("enrichment", "cache_ttl_days", 30))
    cache_path = cache_dir / f"open_wikipedia_{_stable_hash(title)}.json"
    cached = _read_json(cache_path)
    if cached and (not config.run_live_apis or _cache_fresh(cache_path, ttl_days)):
        return cached, "wikipedia_cache"
    if not config.run_live_apis:
        return cached or {}, "wikipedia_offline_missing" if not cached else "wikipedia_cache_stale_offline"
    try:
        payload = _request_json(
            f"https://en.wikipedia.org/api/rest_v1/page/summary/{requests.utils.quote(title)}", timeout=20
        )
        _write_json(cache_path, payload)
        return payload, "wikipedia_live"
    except Exception as exc:
        return cached or {}, f"wikipedia_live_error:{type(exc).__name__}"


def enrich_with_wikidata_wikipedia(
    enriched_df: pd.DataFrame, output_dir: str | Path, config: TripConfig, max_rows: int = 80
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cache_dir = _project_cache_dir(output_dir)
    output = enriched_df.copy()
    audit_rows = []
    if output.empty:
        return output, pd.DataFrame()
    candidate_index = output.sort_values(["final_poi_value", "social_score"], ascending=False).head(max_rows).index
    for idx in candidate_index:
        row = output.loc[idx]
        wikidata_payload, wikidata_status = _wikidata_search(str(row["name"]), cache_dir, config)
        matches = wikidata_payload.get("search", []) if isinstance(wikidata_payload, dict) else []
        wikidata_id = str(matches[0].get("id", "")) if matches else str(row.get("wikidata_id", "") or "")
        title = str(matches[0].get("label", "")) if matches else str(row.get("wikipedia_title", "") or "")
        if wikidata_id and not output.at[idx, "wikidata_id"]:
            output.at[idx, "wikidata_id"] = wikidata_id
        if title and not output.at[idx, "wikipedia_title"]:
            output.at[idx, "wikipedia_title"] = title
        wiki_payload, wiki_status = _wikipedia_summary(title or str(row["name"]), cache_dir, config)
        if isinstance(wiki_payload, dict):
            page_title = wiki_payload.get("title") or title
            if page_title and not output.at[idx, "wikipedia_title"]:
                output.at[idx, "wikipedia_title"] = page_title
            extract = str(wiki_payload.get("extract", "") or "").strip()
            if extract and not str(output.at[idx, "description"] or "").strip():
                output.at[idx, "description"] = extract
            thumbnail = wiki_payload.get("thumbnail", {}) if isinstance(wiki_payload.get("thumbnail", {}), dict) else {}
            image_url = str(thumbnail.get("source", "") or "").strip()
            if image_url and not str(output.at[idx, "image_url"] or "").strip():
                output.at[idx, "image_url"] = image_url
            content_urls = wiki_payload.get("content_urls", {}) if isinstance(wiki_payload.get("content_urls", {}), dict) else {}
            desktop = content_urls.get("desktop", {}) if isinstance(content_urls.get("desktop", {}), dict) else {}
            page_url = str(desktop.get("page", "") or "").strip()
            if page_url:
                if not str(output.at[idx, "source_url"] or "").strip():
                    output.at[idx, "source_url"] = page_url
                if not str(output.at[idx, "website_url"] or "").strip():
                    output.at[idx, "website_url"] = page_url
                output.at[idx, "detail_source"] = wiki_status
            # Free, deterministic popularity proxy when live pageviews are not used.
            if wiki_payload.get("pageid") or page_title:
                output.at[idx, "wikipedia_pageview_score"] = max(
                    float(output.at[idx, "wikipedia_pageview_score"] or 0.0), 0.55
                )
        audit_rows.append(
            {
                "audit_type": "wikidata_wikipedia",
                "name": row["name"],
                "city": row["city"],
                "wikidata_status": wikidata_status,
                "wikipedia_status": wiki_status,
                "wikidata_id": output.at[idx, "wikidata_id"],
                "wikipedia_title": output.at[idx, "wikipedia_title"],
            }
        )
    return output, pd.DataFrame(audit_rows)


def build_open_meteo_context(city_names: list[str], output_dir: str | Path, config: TripConfig) -> pd.DataFrame:
    cache_dir = _project_cache_dir(output_dir)
    rows = []
    if not bool(config.get("enrichment", "use_open_meteo", True)):
        rows.append({"city": "all", "weather_status": "open_meteo_disabled"})
        return pd.DataFrame(rows)
    for city in city_names:
        lat, lon = _city_coordinates(city)
        cache_path = cache_dir / f"open_meteo_{_safe_name(city)}.json"
        cached = _read_json(cache_path)
        status = "open_meteo_cache" if cached else "open_meteo_offline_missing"
        payload = cached or {}
        if config.run_live_apis:
            try:
                payload = _request_json(
                    "https://api.open-meteo.com/v1/forecast",
                    params={
                        "latitude": lat,
                        "longitude": lon,
                        "current": "temperature_2m,precipitation,wind_speed_10m",
                    },
                    timeout=20,
                )
                _write_json(cache_path, payload)
                status = "open_meteo_live"
            except Exception as exc:
                status = (
                    f"open_meteo_live_error:{type(exc).__name__}"
                    if not cached
                    else f"open_meteo_cache_after_live_error:{type(exc).__name__}"
                )
        current = payload.get("current", {}) if isinstance(payload, dict) else {}
        rows.append(
            {
                "city": city,
                "weather_status": status,
                "temperature_2m": current.get("temperature_2m", np.nan),
                "precipitation": current.get("precipitation", np.nan),
                "wind_speed_10m": current.get("wind_speed_10m", np.nan),
            }
        )
    weather_df = pd.DataFrame(rows)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    weather_df.to_csv(Path(output_dir) / "production_open_meteo_context.csv", index=False)
    return weather_df


def fetch_osrm_route_geometry(
    points: list[tuple[float, float]], output_dir: str | Path, config: TripConfig
) -> tuple[list[list[float]], str]:
    """Fetch or load OSRM road geometry for lat/lon points.

    Returned points are [[lat, lon], ...] so Folium can draw them directly.
    """
    if len(points) < 2:
        return [[float(lat), float(lon)] for lat, lon in points], "osrm_not_enough_points"
    cache_dir = _project_cache_dir(output_dir)
    key = _stable_hash(json.dumps(points))
    cache_path = cache_dir / f"open_osrm_route_{key}.json"
    cached = _read_json(cache_path)
    if cached:
        coords = cached.get("latlon_geometry", [])
        return coords, cached.get("status", "osrm_cache")
    if not (config.run_live_apis and bool(config.get("enrichment", "use_osrm", True))):
        return [[float(lat), float(lon)] for lat, lon in points], "osrm_offline_straight_fallback"
    try:
        coord_text = ";".join(f"{lon},{lat}" for lat, lon in points)
        payload = _request_json(
            f"https://router.project-osrm.org/route/v1/driving/{coord_text}",
            params={"overview": "full", "geometries": "geojson"},
            timeout=35,
        )
        route_coords = payload.get("routes", [{}])[0].get("geometry", {}).get("coordinates", [])
        latlon = [[float(lat), float(lon)] for lon, lat in route_coords]
        _write_json(cache_path, {"status": "osrm_live", "latlon_geometry": latlon, "raw": payload})
        return latlon, "osrm_live"
    except Exception as exc:
        return [[float(lat), float(lon)] for lat, lon in points], f"osrm_live_error:{type(exc).__name__}"


def _ensure_open_columns(enriched_df: pd.DataFrame) -> pd.DataFrame:
    output = enriched_df.copy()
    if "source_coverage_score" not in output.columns and "data_confidence" in output.columns:
        output["source_coverage_score"] = pd.to_numeric(output["data_confidence"], errors="coerce").fillna(0.15)
    if "source_coverage_score" not in output.columns:
        output["source_coverage_score"] = 0.15
    if "model_uncertainty" not in output.columns:
        output["model_uncertainty"] = 0.0
    if "annotation_uncertainty" not in output.columns:
        output["annotation_uncertainty"] = 0.0
    if "data_uncertainty" not in output.columns:
        output["data_uncertainty"] = pd.to_numeric(output["model_uncertainty"], errors="coerce").fillna(0.0)
    for column in OPEN_ENRICHED_POI_COLUMNS:
        if column not in output.columns:
            output[column] = (
                ""
                if column
                in {
                    "name",
                    "city",
                    "category",
                    "source_list",
                    "osm_tags",
                    "wikidata_id",
                    "wikipedia_title",
                    "social_reason",
                    "description",
                    "image_url",
                    "website_url",
                    "source_url",
                    "detail_source",
                    "internal_route_source",
                    "park_type",
                    "nature_region",
                    "reason_selected",
                    "itinerary_exclusion_reason",
                }
                else 0.0
            )
            if column == "itinerary_eligible":
                output[column] = True
    return ensure_nature_columns(output)


def _apply_place_detail_fallbacks(enriched_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    output = _ensure_open_columns(enriched_df)
    rows = []
    if output.empty:
        return output, pd.DataFrame()
    for idx, row in output.iterrows():
        name_key = _safe_name(str(row.get("name", ""))).replace("_", " ").strip()
        fallback = CURATED_PLACE_DETAIL_FALLBACKS.get(name_key)
        source = str(row.get("detail_source", "") or "").strip()
        missing_description = not str(row.get("description", "") or "").strip()
        missing_image = not str(row.get("image_url", "") or "").strip()
        missing_source = not str(row.get("source_url", row.get("website_url", "")) or "").strip()
        if fallback:
            if missing_description:
                output.at[idx, "description"] = fallback["description"]
            if missing_image:
                output.at[idx, "image_url"] = fallback["image_url"]
            if missing_source:
                output.at[idx, "source_url"] = fallback["source_url"]
                output.at[idx, "website_url"] = fallback["source_url"]
            if not source or source in {"0", "0.0"}:
                output.at[idx, "detail_source"] = "curated_open_web_fallback"
        elif missing_description:
            output.at[idx, "description"] = (
                f"{row.get('name', 'This stop')} is included from the enriched candidate catalog for "
                f"{row.get('city', 'the active route')}."
            )
            output.at[idx, "detail_source"] = source or "deterministic_catalog_fallback"
        confidence = pd.to_numeric(pd.Series([row.get("data_confidence", 0.2)]), errors="coerce").fillna(0.2).iloc[0]
        output.at[idx, "source_confidence"] = max(float(confidence), 0.45 if fallback else 0.25)
        rows.append(
            {
                "name": row.get("name", ""),
                "city": row.get("city", ""),
                "description_present": bool(str(output.at[idx, "description"]).strip()),
                "image_present": bool(str(output.at[idx, "image_url"]).strip()),
                "source_url_present": bool(str(output.at[idx, "source_url"]).strip()),
                "detail_source": output.at[idx, "detail_source"],
                "used_curated_fallback": bool(fallback and (missing_description or missing_image or missing_source)),
            }
        )
    return output, pd.DataFrame(rows)


def _numeric_series(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column in frame.columns:
        return pd.to_numeric(frame[column], errors="coerce").fillna(default)
    return pd.Series(default, index=frame.index, dtype=float)


def _configured_city_names(city_names: list[str], config: TripConfig) -> list[str]:
    if str(config.get("trip", "scenario_city_universe", "auto")).lower() != "auto":
        return list(city_names)
    return scenario_enrichment_city_universe(
        str(config.get("trip", "scenario", "california_coast")),
        seed_city_names=list(city_names),
        start_city_options=list(config.get("trip", "start_city_options", [])),
        end_city_options=list(config.get("trip", "end_city_options", [])),
    )


def _scenario_route_waypoints(config: TripConfig) -> list[tuple[float, float]]:
    scenario_id = str(config.get("trip", "scenario", "california_coast"))
    starts = {str(city) for city in config.get("trip", "start_city_options", [])}
    ends = {str(city) for city in config.get("trip", "end_city_options", [])}
    coords = all_scenario_coordinates()
    waypoints: list[tuple[float, float]] = []
    options = get_route_options(scenario_id)
    filtered = [
        option
        for option in options
        if (not starts or option.get("gateway_start") in starts)
        and (not ends or option.get("gateway_end") in ends)
    ]
    for option in filtered or options:
        for name in list(option.get("sequence", [])) + list(option.get("nature_regions", [])):
            if name in coords:
                waypoints.append((float(coords[name][0]), float(coords[name][1])))
    if not waypoints:
        waypoints = [(float(lat), float(lon)) for lat, lon in coords.values()]
    deduped: list[tuple[float, float]] = []
    seen = set()
    for lat, lon in waypoints:
        key = (round(lat, 5), round(lon, 5))
        if key not in seen:
            seen.add(key)
            deduped.append((lat, lon))
    return deduped


def _scenario_route_metrics(
    lat: float,
    lon: float,
    config: TripConfig,
    *,
    threshold_minutes: float,
    waypoints: list[tuple[float, float]] | None = None,
) -> tuple[float, float]:
    if pd.isna(lat) or pd.isna(lon):
        return 0.0, float(threshold_minutes)
    points = waypoints if waypoints is not None else _scenario_route_waypoints(config)
    if not points:
        return 0.0, float(threshold_minutes)
    nearest_km = min(geodesic((float(lat), float(lon)), point).km for point in points)
    detour_minutes = nearest_km * 2.0 / 48.0 * 60.0
    route_fit = max(0.0, 1.0 - detour_minutes / float(threshold_minutes))
    return round(float(route_fit), 4), round(float(detour_minutes), 2)


def _recompute_value_columns(enriched_df: pd.DataFrame, config: TripConfig) -> pd.DataFrame:
    production_enrichment = import_legacy_module("production_enrichment")
    output = enriched_df.copy()
    max_detour = float(config.get("time", "max_detour_minutes", 45))
    scenario_id = str(config.get("trip", "scenario", "california_coast"))
    route_waypoints = _scenario_route_waypoints(config)
    for idx, row in output.iterrows():
        corridor_fit, detour_minutes = production_enrichment.corridor_metrics(
            row["latitude"],
            row["longitude"],
            threshold_minutes=max_detour,
        )
        route_fit, route_detour_minutes = _scenario_route_metrics(
            row["latitude"],
            row["longitude"],
            config,
            threshold_minutes=max_detour,
            waypoints=route_waypoints,
        )
        if scenario_id != "california_coast":
            corridor_fit = max(float(corridor_fit), float(route_fit))
            detour_minutes = min(float(detour_minutes), float(route_detour_minutes))
        output.at[idx, "corridor_fit"] = corridor_fit
        output.at[idx, "route_fit"] = route_fit
        output.at[idx, "detour_minutes"] = detour_minutes
    social_weight = float(config.get("social", "social_score_weight", 1.10))
    must_go_weight = float(config.get("social", "must_go_bonus_weight", 0.85))
    corridor_weight = float(config.get("social", "corridor_fit_weight", 0.30))
    detour_penalty = float(config.get("social", "detour_penalty_weight", 0.01))
    if not bool(config.get("enrichment", "use_social_signals", True)):
        output["social_score"] = 0.0
        output["social_must_go"] = False
        output["must_go_weight"] = 0.0
    output["source_coverage_score"] = (
        0.35
        * output["source_list"]
        .astype(str)
        .str.contains("osm|overpass|openstreetmap", case=False, na=False)
        .astype(float)
        + 0.25 * output["source_list"].astype(str).str.contains("yelp", case=False, na=False).astype(float)
        + 0.20 * output["source_list"].astype(str).str.contains("curated", case=False, na=False).astype(float)
        + 0.10 * output["wikidata_id"].astype(str).str.len().gt(0).astype(float)
        + 0.10 * output["wikipedia_title"].astype(str).str.len().gt(0).astype(float)
    ).clip(0.15, 1.0)
    output["identity_coverage"] = output["source_list"].astype(str).str.len().gt(0).astype(float)
    output["location_coverage"] = (
        pd.to_numeric(output.get("latitude", pd.Series(dtype=float)), errors="coerce").notna()
        & pd.to_numeric(output.get("longitude", pd.Series(dtype=float)), errors="coerce").notna()
    ).astype(float)
    output["semantic_coverage"] = output["category"].astype(str).str.len().gt(0).astype(float)
    output["popularity_coverage"] = (
        output["source_list"].astype(str).str.contains("yelp", case=False, na=False)
        | pd.to_numeric(output.get("wikipedia_pageview_score", pd.Series(dtype=float)), errors="coerce").fillna(0).gt(0)
    ).astype(float)
    output["temporal_coverage"] = pd.to_numeric(output.get("weather_risk", pd.Series(dtype=float)), errors="coerce").notna().astype(float)
    output["routing_coverage"] = pd.to_numeric(output.get("route_fit", pd.Series(dtype=float)), errors="coerce").notna().astype(float)
    output["model_uncertainty"] = pd.to_numeric(output.get("model_uncertainty", 0.0), errors="coerce").fillna(0.0)
    output["annotation_uncertainty"] = pd.to_numeric(output.get("annotation_uncertainty", 0.0), errors="coerce").fillna(0.0)
    output["entity_match_confidence"] = pd.to_numeric(output.get("entity_match_confidence", 0.85), errors="coerce").fillna(0.85)
    output["data_confidence"] = output["source_coverage_score"]
    output["data_uncertainty"] = output["model_uncertainty"]
    output["final_poi_value"] = (
        0.42 * _numeric_series(output, "base_score_norm")
        + 0.20 * _numeric_series(output, "yelp_signal_norm")
        + social_weight * _numeric_series(output, "social_score")
        + must_go_weight * _numeric_series(output, "must_go_weight") * _numeric_series(output, "social_score")
        + corridor_weight
        * np.maximum(_numeric_series(output, "corridor_fit"), _numeric_series(output, "route_fit"))
        - detour_penalty * _numeric_series(output, "detour_minutes")
        + 0.10 * _numeric_series(output, "wikipedia_pageview_score")
    ).clip(lower=0.0)
    return output.sort_values(["final_poi_value", "social_score"], ascending=[False, False]).reset_index(drop=True)


def _write_candidate_audit(enriched_df: pd.DataFrame, output_dir: Path, config: TripConfig) -> pd.DataFrame:
    frame = _ensure_open_columns(enriched_df)
    if frame.empty:
        audit = pd.DataFrame()
        audit.to_csv(output_dir / "production_candidate_audit_by_city_category_source.csv", index=False)
        return audit
    frame = frame.copy()
    frame["candidate_anchor"] = frame["nature_region"].where(
        frame["nature_region"].astype(str).str.strip().ne(""),
        frame["city"].astype(str),
    )
    frame["source_group"] = frame["source_list"].astype(str).str.split("|").str[0].fillna("unknown")
    audit = (
        frame.groupby(["city", "candidate_anchor", "category", "source_group"], dropna=False)
        .agg(
            candidate_count=("name", "nunique"),
            nature_candidate_count=("is_nature", "sum"),
            national_park_candidate_count=("is_national_park", "sum"),
            avg_data_confidence=("data_confidence", "mean"),
            avg_final_poi_value=("final_poi_value", "mean"),
            avg_route_fit=("route_fit", "mean"),
        )
        .reset_index()
    )
    audit.insert(0, "scenario", str(config.get("trip", "scenario", "california_coast")))
    audit.insert(1, "interest", str(config.get("interest", "mode", config.get("trip", "interest_profile", ""))))
    audit.to_csv(output_dir / "production_candidate_audit_by_city_category_source.csv", index=False)
    return audit


def _write_excluded_poi_category_audit(enriched_df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    columns = ["itinerary_exclusion_reason", "category", "source_list", "excluded_count", "example_names"]
    if enriched_df.empty or "itinerary_eligible" not in enriched_df.columns:
        audit = pd.DataFrame(columns=columns)
        audit.to_csv(output_dir / "production_excluded_poi_category_audit.csv", index=False)
        return audit
    excluded = enriched_df[~enriched_df["itinerary_eligible"].fillna(True).astype(bool)].copy()
    if excluded.empty:
        audit = pd.DataFrame(columns=columns)
        audit.to_csv(output_dir / "production_excluded_poi_category_audit.csv", index=False)
        return audit
    audit = (
        excluded.groupby(["itinerary_exclusion_reason", "category", "source_list"], dropna=False)
        .agg(
            excluded_count=("name", "count"),
            example_names=("name", lambda values: " | ".join(values.dropna().astype(str).head(5))),
        )
        .reset_index()
    )
    audit.to_csv(output_dir / "production_excluded_poi_category_audit.csv", index=False)
    return audit


def scenario_route_fit_metrics(lat: float, lon: float, config: TripConfig) -> tuple[float, float]:
    return _scenario_route_metrics(
        lat,
        lon,
        config,
        threshold_minutes=float(config.get("time", "max_detour_minutes", 45)),
    )


def build_enriched_catalog(
    *,
    all_business_df: pd.DataFrame,
    city_names: list[str],
    output_dir: str | Path,
    config: TripConfig,
) -> dict:
    """Build the canonical open-enriched catalog before any optimization runs."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    production_enrichment = import_legacy_module("production_enrichment")
    city_names = _configured_city_names(city_names, config)

    osm_poi_catalog, osm_hotel_catalog, osm_audit_df = build_open_osm_catalogs(city_names, output_dir, config)
    local_yelp_df = (
        all_business_df if bool(config.get("enrichment", "use_local_yelp_dataset", True)) else pd.DataFrame()
    )
    outputs = production_enrichment.build_enrichment_outputs(
        all_business_df=local_yelp_df,
        city_names=city_names,
        output_dir=output_dir,
        run_live_apis=False,
    )

    enriched_df = _ensure_open_columns(outputs["production_enriched_poi_catalog_df"])
    nature_seed_df, nps_audit_df = load_nps_or_curated_nature_pois(output_dir, config)
    if bool(config.get("nature", "enabled", True)) and not nature_seed_df.empty:
        enriched_df = pd.concat([enriched_df, _ensure_open_columns(nature_seed_df)], ignore_index=True, sort=False)
        enriched_df = enriched_df.drop_duplicates(subset=["name", "latitude", "longitude"]).reset_index(drop=True)
    enriched_df, wiki_audit_df = enrich_with_wikidata_wikipedia(enriched_df, output_dir, config)
    enriched_df, place_detail_audit_df = _apply_place_detail_fallbacks(enriched_df)
    place_detail_audit_df.to_csv(output_dir / "production_place_detail_audit.csv", index=False)
    enriched_df = _ensure_open_columns(_recompute_value_columns(enriched_df, config))
    enriched_df, nature_site_routes_df, nature_site_route_points_df, nature_site_route_audit_df = (
        build_nature_site_route_artifacts(enriched_df, output_dir, config)
    )
    enriched_df, utility_scores_df, utility_audit_df = apply_utility_models(enriched_df, output_dir, config)
    enriched_df = apply_nature_site_route_score_bonus(enriched_df, config)
    enriched_df = compute_interest_adjusted_values(enriched_df, config)
    enriched_df = mark_itinerary_eligible(enriched_df)
    enriched_df, place_detail_audit_df = _apply_place_detail_fallbacks(enriched_df)
    place_detail_audit_df.to_csv(output_dir / "production_place_detail_audit.csv", index=False)
    excluded_poi_audit_df = _write_excluded_poi_category_audit(enriched_df, output_dir)
    ltr_audit_df = learning_to_rank_audit(enriched_df, config)
    enriched_df[OPEN_ENRICHED_POI_COLUMNS].to_csv(output_dir / "production_enriched_poi_catalog.csv", index=False)
    write_interest_catalog_artifacts(enriched_df, output_dir, config)
    candidate_audit_df = _write_candidate_audit(enriched_df, output_dir, config)

    city_summary_df = production_enrichment.build_city_catalog_summary(city_names, local_yelp_df, enriched_df)
    if not city_summary_df.empty and "city" in city_summary_df.columns:
        must_go_frame = enriched_df.copy()
        if "social_must_go" not in must_go_frame.columns:
            must_go_frame["social_must_go"] = False
        must_go_frame["social_must_go"] = must_go_frame["social_must_go"].fillna(False).astype(bool)
        must_go_frame["social_score"] = _numeric_series(must_go_frame, "social_score")
        must_go_frame["must_go_weight"] = _numeric_series(must_go_frame, "must_go_weight")
        must_go_frame["must_go_signal"] = (
            must_go_frame["social_must_go"].astype(float)
            * must_go_frame["social_score"]
            * must_go_frame["must_go_weight"].clip(lower=0.0)
        )
        must_go_by_city = (
            must_go_frame[must_go_frame["social_must_go"]]
            .groupby("city", dropna=False)
            .agg(
                city_must_go_count=("name", "nunique"),
                city_must_go_score=("must_go_signal", "sum"),
                city_social_score=("social_score", "max"),
            )
            .reset_index()
        )
        city_summary_df = city_summary_df.merge(must_go_by_city, on="city", how="left")
        for column in ["city_must_go_count", "city_must_go_score", "city_social_score"]:
            raw_column = (
                city_summary_df[column]
                if column in city_summary_df.columns
                else pd.Series(0.0, index=city_summary_df.index)
            )
            city_summary_df[column] = pd.to_numeric(raw_column, errors="coerce").fillna(0.0)
    city_summary_df.to_csv(output_dir / "production_city_catalog_summary.csv", index=False)
    weather_df = build_open_meteo_context(city_names, output_dir, config)

    audit_parts = [
        outputs.get("production_enrichment_audit_df", pd.DataFrame()).assign(
            audit_source="legacy_local_yelp_osm_social"
        ),
        osm_audit_df.assign(audit_source="open_osm"),
        nps_audit_df.assign(audit_source="open_nps_or_curated_nature") if not nps_audit_df.empty else pd.DataFrame(),
        wiki_audit_df.assign(audit_source="open_wikidata_wikipedia") if not wiki_audit_df.empty else pd.DataFrame(),
        place_detail_audit_df.assign(audit_type="place_details", audit_source="open_place_details")
        if not place_detail_audit_df.empty
        else pd.DataFrame(),
        weather_df.assign(audit_type="open_meteo", audit_source="open_meteo"),
        utility_audit_df.assign(audit_source="utility_model") if not utility_audit_df.empty else pd.DataFrame(),
        nature_site_route_audit_df.assign(audit_type="nature_site_routes", audit_source="open_osm_nps_curated")
        if not nature_site_route_audit_df.empty
        else pd.DataFrame(),
        excluded_poi_audit_df.assign(audit_type="poi_eligibility", audit_source="itinerary_eligibility")
        if not excluded_poi_audit_df.empty
        else pd.DataFrame(),
        ltr_audit_df.assign(audit_source="learning_to_rank"),
    ]
    audit_df = pd.concat([part for part in audit_parts if not part.empty], ignore_index=True, sort=False)
    audit_df.to_csv(output_dir / "production_enrichment_audit.csv", index=False)

    social_signal_snapshots_df = city_summary_df.copy()
    social_signal_snapshots_df["source_name"] = "aggregate_open_enriched_social_proxy"
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

    outputs.update(
        {
            "osm_city_poi_catalog_df": osm_poi_catalog,
            "osm_city_hotel_catalog_df": osm_hotel_catalog,
            "production_enriched_poi_catalog_df": enriched_df,
            "production_enrichment_audit_df": audit_df,
            "city_catalog_summary_df": city_summary_df,
            "production_utility_scores_df": utility_scores_df,
            "production_signal_matrix_df": pd.read_csv(output_dir / "production_signal_matrix.csv")
            if (output_dir / "production_signal_matrix.csv").exists()
            else pd.DataFrame(),
            "production_utility_model_audit_df": utility_audit_df,
            "learning_to_rank_audit_df": ltr_audit_df,
            "production_candidate_audit_by_city_category_source_df": candidate_audit_df,
            "production_excluded_poi_category_audit_df": excluded_poi_audit_df,
            "production_place_detail_audit_df": place_detail_audit_df,
            "production_nature_site_routes_df": nature_site_routes_df,
            "production_nature_site_route_points_df": nature_site_route_points_df,
            "production_nature_site_route_audit_df": nature_site_route_audit_df,
            "social_signal_snapshots_df": social_signal_snapshots_df,
            "open_meteo_context_df": weather_df,
        }
    )
    return outputs


def canonical_poi_columns() -> list[str]:
    return list(OPEN_ENRICHED_POI_COLUMNS)
