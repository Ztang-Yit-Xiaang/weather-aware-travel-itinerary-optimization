"""Internal route details for nature sites.

This module keeps nature-site detail routes separate from the main itinerary
optimizer. It fetches/cache-loads OSM line geometry when available, falls back
to curated open route sketches, and writes artifacts that the static dashboard
can render as candidate/detail layers.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from .config import TripConfig

USER_AGENT = "IE5533-WeatherAwareTravelPlanner/1.0 (academic project; nature site routes)"

NATURE_SITE_ROUTE_METRIC_COLUMNS = [
    "internal_route_count",
    "best_internal_route_score",
    "best_internal_route_distance_km",
    "best_internal_route_duration_minutes",
    "internal_route_confidence",
    "internal_route_source",
    "internal_route_value_bonus",
]

ROUTE_COLUMNS = [
    "site_id",
    "site_name",
    "city",
    "nature_region",
    "latitude",
    "longitude",
    "route_id",
    "route_name",
    "route_type",
    "distance_km",
    "duration_minutes",
    "difficulty",
    "route_score",
    "source_confidence",
    "source",
    "source_url",
    "description",
    "geometry_source",
    "fallback_used",
    "point_count",
]

POINT_COLUMNS = [
    "route_id",
    "point_order",
    "latitude",
    "longitude",
    "name",
    "route_name",
    "site_name",
    "nature_region",
    "route_type",
    "source_confidence",
]

AUDIT_COLUMNS = [
    "site_id",
    "site_name",
    "city",
    "nature_region",
    "candidate",
    "selected",
    "route_count",
    "source_status",
    "fallback_used",
    "missing_reason",
]


CURATED_INTERNAL_ROUTES: dict[str, list[dict[str, Any]]] = {
    "yosemite": [
        {
            "route_name": "Yosemite Valley scenic loop",
            "route_type": "scenic_drive",
            "description": "A practical valley-floor scenic loop linking Tunnel View, Yosemite Valley, and waterfall viewpoints.",
            "source_url": "https://www.nps.gov/yose/planyourvisit/yosemitevalley.htm",
            "coordinates": [
                (37.7156, -119.6770),
                (37.7439, -119.6043),
                (37.7485, -119.5886),
                (37.7385, -119.5702),
                (37.7156, -119.6770),
            ],
        },
        {
            "route_name": "Lower Yosemite Fall viewpoint walk",
            "route_type": "viewpoint_walk",
            "description": "Short Yosemite Valley walk focused on Lower Yosemite Fall and surrounding valley views.",
            "source_url": "https://www.nps.gov/yose/planyourvisit/lowerfalltrail.htm",
            "coordinates": [
                (37.7487, -119.5967),
                (37.7525, -119.5969),
                (37.7554, -119.5963),
            ],
        },
    ],
    "sequoia": [
        {
            "route_name": "General Sherman and Giant Forest walk",
            "route_type": "short_hike",
            "description": "Short Giant Forest route around the General Sherman area and nearby sequoia grove scenery.",
            "source_url": "https://www.nps.gov/seki/planyourvisit/gensherman.htm",
            "coordinates": [
                (36.5819, -118.7514),
                (36.5786, -118.7517),
                (36.5756, -118.7489),
                (36.5742, -118.7460),
            ],
        },
        {
            "route_name": "Generals Highway sequoia scenic segment",
            "route_type": "scenic_drive",
            "description": "Scenic in-park driving segment connecting the Giant Forest area with core Sequoia viewpoints.",
            "source_url": "https://www.nps.gov/seki/planyourvisit/driving.htm",
            "coordinates": [
                (36.4488, -118.9018),
                (36.5265, -118.7653),
                (36.5819, -118.7514),
            ],
        },
    ],
    "kings_canyon": [
        {
            "route_name": "Kings Canyon scenic byway segment",
            "route_type": "scenic_drive",
            "description": "Scenic canyon road segment toward Grant Grove and Kings Canyon viewpoints.",
            "source_url": "https://www.nps.gov/seki/planyourvisit/kings-canyon-scenic-byway.htm",
            "coordinates": [
                (36.7378, -119.7871),
                (36.7468, -118.9647),
                (36.8879, -118.5551),
            ],
        }
    ],
    "joshua_tree": [
        {
            "route_name": "Hidden Valley nature trail",
            "route_type": "short_hike",
            "description": "Short loop trail through boulder formations and desert ecology near Hidden Valley.",
            "source_url": "https://www.nps.gov/jotr/planyourvisit/hvtrail.htm",
            "coordinates": [
                (34.0122, -116.1686),
                (34.0137, -116.1661),
                (34.0112, -116.1635),
                (34.0122, -116.1686),
            ],
        },
        {
            "route_name": "Keys View scenic drive",
            "route_type": "scenic_drive",
            "description": "In-park scenic drive ending at Keys View, one of Joshua Tree's most recognizable overlooks.",
            "source_url": "https://www.nps.gov/jotr/planyourvisit/keysview.htm",
            "coordinates": [
                (33.8734, -115.9010),
                (33.9277, -116.1878),
                (33.9270, -116.1870),
            ],
        },
    ],
    "big_sur": [
        {
            "route_name": "Bixby Creek and Pfeiffer Big Sur scenic route",
            "route_type": "scenic_drive",
            "description": "Coastal scenic route from Bixby Creek Bridge toward Pfeiffer Big Sur State Park.",
            "source_url": "https://www.parks.ca.gov/?page_id=570",
            "coordinates": [
                (36.3712, -121.9018),
                (36.2704, -121.8081),
                (36.2524, -121.7879),
            ],
        }
    ],
}


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


def _slug(value: str) -> str:
    return "_".join(part for part in "".join(ch.lower() if ch.isalnum() else " " for ch in value).split() if part)


def _canonical_site_id(*parts: Any) -> str:
    text = " ".join(_safe_str(part).lower() for part in parts if _safe_str(part))
    if "yosemite" in text:
        return "yosemite"
    if "sequoia" in text:
        return "sequoia"
    if "kings canyon" in text:
        return "kings_canyon"
    if "joshua" in text:
        return "joshua_tree"
    if "big sur" in text or "bixby" in text or "pfeiffer" in text:
        return "big_sur"
    return _slug(text)[:80] or "nature_site"


def _project_cache_dir(output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    cache_dir = output_dir.parent / "cache" if output_dir.name == "outputs" else output_dir / "cache"
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


def _distance_km(coords: list[tuple[float, float]]) -> float:
    if len(coords) < 2:
        return 0.0
    total = 0.0
    for (lat1, lon1), (lat2, lon2) in zip(coords, coords[1:], strict=False):
        radius = 6371.0
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        d_phi = math.radians(lat2 - lat1)
        d_lam = math.radians(lon2 - lon1)
        a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lam / 2) ** 2
        total += 2 * radius * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return round(float(total), 3)


def _duration_minutes(route_type: str, distance_km: float) -> int:
    if route_type == "scenic_drive":
        return int(max(25, round(distance_km * 1.5)))
    if route_type == "viewpoint_walk":
        return int(max(25, round(distance_km * 22)))
    return int(max(45, round(distance_km * 18)))


def _difficulty(route_type: str, distance_km: float) -> str:
    if route_type == "scenic_drive":
        return "easy" if distance_km <= 35 else "moderate"
    if distance_km <= 2.5:
        return "easy"
    if distance_km <= 8.0:
        return "moderate"
    return "strenuous"


def _classify_route(tags: dict[str, Any], name: str) -> str:
    text = f"{name} {' '.join(str(value) for value in tags.values())}".lower()
    highway = str(tags.get("highway", "")).lower()
    route = str(tags.get("route", "")).lower()
    tourism = str(tags.get("tourism", "")).lower()
    if "view" in text or "overlook" in text or "vista" in text or tourism == "viewpoint":
        return "viewpoint_walk"
    if route in {"hiking", "foot"} or highway in {"path", "footway", "track", "trailhead"} or "trail" in text:
        return "short_hike"
    return "scenic_drive"


def _route_score(route_type: str, distance_km: float, confidence: float) -> float:
    if route_type == "scenic_drive":
        distance_fit = max(0.0, 1.0 - max(0.0, distance_km - 45.0) / 75.0)
        type_fit = 0.78
    elif route_type == "viewpoint_walk":
        distance_fit = max(0.0, 1.0 - max(0.0, distance_km - 4.0) / 10.0)
        type_fit = 0.86
    else:
        distance_fit = max(0.0, 1.0 - max(0.0, distance_km - 8.0) / 14.0)
        type_fit = 0.90
    return round(min(1.0, max(0.0, 0.45 * confidence + 0.30 * distance_fit + 0.25 * type_fit)), 4)


def _nature_mask(frame: pd.DataFrame) -> pd.Series:
    mask = pd.Series(False, index=frame.index)
    for column in ["is_nature", "is_national_park", "is_state_park", "is_protected_area", "is_scenic_viewpoint", "is_hiking"]:
        if column in frame.columns:
            mask = mask | frame[column].astype(str).str.lower().isin({"1", "true", "yes"})
    if "nature_score" in frame.columns:
        mask = mask | (pd.to_numeric(frame["nature_score"], errors="coerce").fillna(0.0) >= 0.35)
    return mask


def _site_rows(enriched_df: pd.DataFrame, selected_route_df: pd.DataFrame | None, config: TripConfig) -> pd.DataFrame:
    if enriched_df.empty:
        return pd.DataFrame()
    frame = enriched_df[_nature_mask(enriched_df)].copy()
    if frame.empty:
        return pd.DataFrame()
    frame["_site_id"] = frame.apply(
        lambda row: _canonical_site_id(row.get("name"), row.get("nature_region"), row.get("city")), axis=1
    )
    frame["_selected"] = False
    if selected_route_df is not None and not selected_route_df.empty:
        selected_ids = {
            _canonical_site_id(row.get("attraction_name", row.get("name", "")), row.get("nature_region", ""), row.get("city", ""))
            for _, row in selected_route_df.iterrows()
        }
        frame["_selected"] = frame["_site_id"].isin(selected_ids)
    sort_cols = [column for column in ["_selected", "interest_adjusted_value", "final_poi_value", "nature_score"] if column in frame.columns]
    if sort_cols:
        frame = frame.sort_values(sort_cols, ascending=False)
    return frame.drop_duplicates("_site_id", keep="first").reset_index(drop=True)


def _overpass_query(lat: float, lon: float, radius_m: int) -> str:
    return f"""
[out:json][timeout:35];
(
  way["name"]["route"~"hiking|foot"](around:{radius_m},{lat},{lon});
  relation["name"]["route"~"hiking|foot"](around:{radius_m},{lat},{lon});
  way["name"]["highway"~"path|footway|track"](around:{radius_m},{lat},{lon});
  way["name"]["tourism"="viewpoint"](around:{radius_m},{lat},{lon});
  way["name"]["scenic"="yes"](around:{radius_m},{lat},{lon});
  way["name"]["highway"~"secondary|tertiary|unclassified|service"](around:{radius_m},{lat},{lon});
);
out geom center tags;
"""


def _load_or_fetch_osm_routes(site_id: str, lat: float, lon: float, output_dir: Path, config: TripConfig) -> tuple[list[dict[str, Any]], str]:
    if not bool(config.get("nature_detail_routes", "use_osm", True)):
        return [], "osm_disabled"
    radius = int(config.get("nature_detail_routes", "overpass_radius_meters", 25000))
    ttl_days = int(config.get("enrichment", "cache_ttl_days", 30))
    cache_dir = _project_cache_dir(output_dir)
    cache_path = cache_dir / f"open_overpass_nature_site_routes_{site_id}_{radius}.json"
    cached = _read_json(cache_path)
    if cached and (not config.run_live_apis or _cache_fresh(cache_path, ttl_days)):
        return list(cached.get("elements", [])), "overpass_route_cache"
    if not config.run_live_apis:
        return list(cached.get("elements", [])) if cached else [], "overpass_route_offline_missing"
    try:
        response = requests.get(
            "https://overpass-api.de/api/interpreter",
            params={"data": _overpass_query(lat, lon, radius)},
            headers={"User-Agent": USER_AGENT},
            timeout=50,
        )
        response.raise_for_status()
        payload = response.json()
        payload["_fetched_at"] = datetime.now(UTC).isoformat()
        payload["_source"] = "overpass_route_live"
        _write_json(cache_path, payload)
        time.sleep(1.0)
        return list(payload.get("elements", [])), "overpass_route_live"
    except Exception as exc:
        if cached:
            return list(cached.get("elements", [])), f"overpass_route_cache_after_live_error:{type(exc).__name__}"
        return [], f"overpass_route_live_error:{type(exc).__name__}"


def _geometry_from_element(element: dict[str, Any]) -> list[tuple[float, float]]:
    coords: list[tuple[float, float]] = []
    for item in element.get("geometry", []) or []:
        lat = item.get("lat")
        lon = item.get("lon")
        if lat is not None and lon is not None:
            coords.append((float(lat), float(lon)))
    if coords:
        return coords
    for member in element.get("members", []) or []:
        for item in member.get("geometry", []) or []:
            lat = item.get("lat")
            lon = item.get("lon")
            if lat is not None and lon is not None:
                coord = (float(lat), float(lon))
                if not coords or coords[-1] != coord:
                    coords.append(coord)
    return coords


def _route_from_geometry(
    *,
    site: pd.Series,
    route_name: str,
    route_type: str,
    coords: list[tuple[float, float]],
    source: str,
    source_url: str,
    description: str,
    confidence: float,
    fallback_used: bool,
    index: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    site_name = _safe_str(site.get("name") or site.get("nature_region") or "Nature site")
    site_id = _safe_str(site.get("_site_id")) or _canonical_site_id(site_name)
    distance = _distance_km(coords)
    duration = _duration_minutes(route_type, distance)
    route_id = f"nature_site__{site_id}__{_slug(route_name)}__{index}"
    route = {
        "site_id": site_id,
        "site_name": site_name,
        "city": _safe_str(site.get("city", "")),
        "nature_region": _safe_str(site.get("nature_region", "")),
        "latitude": _safe_float(site.get("latitude")),
        "longitude": _safe_float(site.get("longitude")),
        "route_id": route_id,
        "route_name": route_name,
        "route_type": route_type,
        "distance_km": distance,
        "duration_minutes": duration,
        "difficulty": _difficulty(route_type, distance),
        "route_score": _route_score(route_type, distance, confidence),
        "source_confidence": confidence,
        "source": source,
        "source_url": source_url,
        "description": description,
        "geometry_source": "curated_geometry" if fallback_used else "osm_geometry",
        "fallback_used": bool(fallback_used),
        "point_count": len(coords),
    }
    points = [
        {
            "route_id": route_id,
            "point_order": order,
            "latitude": lat,
            "longitude": lon,
            "name": route_name,
            "route_name": route_name,
            "site_name": site_name,
            "nature_region": _safe_str(site.get("nature_region", "")),
            "route_type": route_type,
            "source_confidence": confidence,
        }
        for order, (lat, lon) in enumerate(coords, start=1)
    ]
    return route, points


def parse_overpass_nature_site_routes(
    elements: list[dict[str, Any]],
    site: pd.Series,
    source_status: str,
    config: TripConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    allowed_types = set(config.get("nature_detail_routes", "route_types", ["scenic_drive", "short_hike", "viewpoint_walk"]))
    max_short = float(config.get("nature_detail_routes", "max_short_hike_km", 12))
    max_drive = float(config.get("nature_detail_routes", "max_scenic_drive_km", 90))
    route_rows: list[dict[str, Any]] = []
    point_rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for element in elements:
        tags = element.get("tags", {}) or {}
        route_name = _safe_str(tags.get("name", "")).strip()
        if not route_name:
            continue
        coords = _geometry_from_element(element)
        if len(coords) < 2:
            continue
        route_type = _classify_route(tags, route_name)
        if route_type not in allowed_types:
            continue
        distance = _distance_km(coords)
        if route_type == "scenic_drive" and distance > max_drive:
            continue
        if route_type != "scenic_drive" and distance > max_short:
            continue
        key = f"{route_name.lower()}:{round(distance, 1)}:{route_type}"
        if key in seen:
            continue
        seen.add(key)
        source_url = _safe_str(tags.get("website") or tags.get("url") or tags.get("wikipedia") or "")
        description = _safe_str(tags.get("description", "")) or f"{route_name} is an OSM-derived internal nature-site route."
        route, points = _route_from_geometry(
            site=site,
            route_name=route_name,
            route_type=route_type,
            coords=coords,
            source=source_status,
            source_url=source_url,
            description=description,
            confidence=0.72 if "live" in source_status or "cache" in source_status else 0.62,
            fallback_used=False,
            index=len(route_rows) + 1,
        )
        route_rows.append(route)
        point_rows.extend(points)
    route_rows.sort(key=lambda row: row["route_score"], reverse=True)
    keep_ids = {row["route_id"] for row in route_rows[: int(config.get("nature_detail_routes", "max_routes_per_site", 3))]}
    return route_rows[: int(config.get("nature_detail_routes", "max_routes_per_site", 3))], [
        row for row in point_rows if row["route_id"] in keep_ids
    ]


def _curated_routes_for_site(site: pd.Series, config: TripConfig) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not bool(config.get("nature_detail_routes", "use_curated_fallback", True)):
        return [], []
    site_id = _safe_str(site.get("_site_id")) or _canonical_site_id(site.get("name"), site.get("nature_region"))
    specs = CURATED_INTERNAL_ROUTES.get(site_id, [])
    route_rows: list[dict[str, Any]] = []
    point_rows: list[dict[str, Any]] = []
    for spec in specs[: int(config.get("nature_detail_routes", "max_routes_per_site", 3))]:
        route, points = _route_from_geometry(
            site=site,
            route_name=str(spec["route_name"]),
            route_type=str(spec["route_type"]),
            coords=[(float(lat), float(lon)) for lat, lon in spec["coordinates"]],
            source="curated_internal_route_fallback",
            source_url=str(spec.get("source_url", "")),
            description=str(spec.get("description", "")),
            confidence=0.58,
            fallback_used=True,
            index=len(route_rows) + 1,
        )
        route_rows.append(route)
        point_rows.extend(points)
    return route_rows, point_rows


def _empty_outputs(output_dir: Path, enriched_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    for column in NATURE_SITE_ROUTE_METRIC_COLUMNS:
        if column not in enriched_df.columns:
            enriched_df[column] = "" if column == "internal_route_source" else 0.0
    routes = pd.DataFrame(columns=ROUTE_COLUMNS)
    points = pd.DataFrame(columns=POINT_COLUMNS)
    audit = pd.DataFrame(columns=AUDIT_COLUMNS)
    routes.to_csv(output_dir / "production_nature_site_routes.csv", index=False)
    points.to_csv(output_dir / "production_nature_site_route_points.csv", index=False)
    audit.to_csv(output_dir / "production_nature_site_route_audit.csv", index=False)
    return enriched_df, routes, points, audit


def build_nature_site_route_artifacts(
    enriched_df: pd.DataFrame,
    output_dir: str | Path,
    config: TripConfig,
    selected_route_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Write internal nature-site route artifacts and attach route metrics to POIs."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    enriched = enriched_df.copy()
    if not bool(config.get("nature_detail_routes", "enabled", True)):
        return _empty_outputs(output_dir, enriched)
    sites = _site_rows(enriched, selected_route_df, config)
    if sites.empty:
        return _empty_outputs(output_dir, enriched)

    all_routes: list[dict[str, Any]] = []
    all_points: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    for _, site in sites.iterrows():
        site_id = _safe_str(site.get("_site_id")) or _canonical_site_id(site.get("name"), site.get("nature_region"))
        lat = _safe_float(site.get("latitude"))
        lon = _safe_float(site.get("longitude"))
        routes: list[dict[str, Any]] = []
        points: list[dict[str, Any]] = []
        source_status = "missing_coordinates"
        if lat and lon:
            elements, source_status = _load_or_fetch_osm_routes(site_id, lat, lon, output_dir, config)
            routes, points = parse_overpass_nature_site_routes(elements, site, source_status, config)
        fallback_used = False
        if not routes:
            routes, points = _curated_routes_for_site(site, config)
            fallback_used = bool(routes)
            if fallback_used:
                source_status = "curated_internal_route_fallback"
        all_routes.extend(routes)
        all_points.extend(points)
        audit_rows.append(
            {
                "site_id": site_id,
                "site_name": _safe_str(site.get("name")),
                "city": _safe_str(site.get("city")),
                "nature_region": _safe_str(site.get("nature_region")),
                "candidate": True,
                "selected": bool(site.get("_selected", False)),
                "route_count": len(routes),
                "source_status": source_status,
                "fallback_used": fallback_used,
                "missing_reason": "" if routes else f"no_internal_routes_from_{source_status}",
            }
        )

    routes_df = pd.DataFrame(all_routes, columns=ROUTE_COLUMNS)
    points_df = pd.DataFrame(all_points, columns=POINT_COLUMNS)
    audit_df = pd.DataFrame(audit_rows, columns=AUDIT_COLUMNS)
    routes_df.to_csv(output_dir / "production_nature_site_routes.csv", index=False)
    points_df.to_csv(output_dir / "production_nature_site_route_points.csv", index=False)
    audit_df.to_csv(output_dir / "production_nature_site_route_audit.csv", index=False)

    metrics = (
        routes_df.sort_values("route_score", ascending=False)
        .groupby("site_id", dropna=False)
        .agg(
            internal_route_count=("route_id", "nunique"),
            best_internal_route_score=("route_score", "max"),
            best_internal_route_distance_km=("distance_km", "first"),
            best_internal_route_duration_minutes=("duration_minutes", "first"),
            internal_route_confidence=("source_confidence", "max"),
            internal_route_source=("source", "first"),
        )
        .reset_index()
        .rename(columns={"site_id": "_site_id"})
        if not routes_df.empty
        else pd.DataFrame()
    )
    enriched["_site_id"] = enriched.apply(
        lambda row: _canonical_site_id(row.get("name"), row.get("nature_region"), row.get("city")), axis=1
    )
    if not metrics.empty:
        enriched = enriched.merge(metrics, on="_site_id", how="left", suffixes=("", "_internal_route"))
    for column in NATURE_SITE_ROUTE_METRIC_COLUMNS:
        if column not in enriched.columns:
            enriched[column] = "" if column == "internal_route_source" else 0.0
    for column in [
        "internal_route_count",
        "best_internal_route_score",
        "best_internal_route_distance_km",
        "best_internal_route_duration_minutes",
        "internal_route_confidence",
    ]:
        enriched[column] = pd.to_numeric(enriched[column], errors="coerce").fillna(0.0)
    enriched["internal_route_source"] = enriched["internal_route_source"].fillna("").astype(str)
    enriched = enriched.drop(columns=["_site_id"], errors="ignore")
    return enriched, routes_df, points_df, audit_df


def apply_nature_site_route_score_bonus(enriched_df: pd.DataFrame, config: TripConfig) -> pd.DataFrame:
    """Softly reward nature POIs with credible internal route details."""
    output = enriched_df.copy()
    for column in NATURE_SITE_ROUTE_METRIC_COLUMNS:
        if column not in output.columns:
            output[column] = "" if column == "internal_route_source" else 0.0
    weight = float(config.get("nature_detail_routes", "soft_score_weight", 0.06))
    interest_weights = config.get("interest", "weights", {}) or {}
    nature_weight = float(interest_weights.get("nature", 0.25)) if isinstance(interest_weights, dict) else 0.25
    mode = str(config.get("interest", "mode", "")).lower()
    if "nature" in mode:
        nature_weight = max(nature_weight, 0.55)
    route_score = pd.to_numeric(output["best_internal_route_score"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    duration = pd.to_numeric(output["best_internal_route_duration_minutes"], errors="coerce").fillna(0.0)
    long_penalty = ((duration - 180.0).clip(lower=0.0) / 600.0).clip(0.0, 0.04)
    bonus = (weight * nature_weight * route_score - long_penalty).clip(lower=-0.02, upper=weight * nature_weight)
    output["internal_route_value_bonus"] = bonus.round(6)
    base = pd.to_numeric(output.get("final_poi_value", 0.0), errors="coerce").fillna(0.0)
    output["final_poi_value"] = (base + bonus).clip(lower=0.0)
    return output
