"""Read-only, revision-bound interest exploration; never grants itinerary eligibility."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from ..interest_model import FEATURE_NAMES, FEATURE_SCHEMA, BayesianInterestModel, interest_features
from ..interest_surface import build_interest_surface, load_land_support
from ..plans.attribute_ownership import attribute_ownership_rows
from .candidate_discovery import (
    POICandidateDiscoveryError,
    _catalog,
    discover_poi_candidates,
)
from .nearby_places import NearbyPlaceCache


def required_interest_points(parent_plan: dict, geography: dict, day: int, catalog=None) -> list[dict]:
    """Use confirmed existence ownership, never map focus or a hard attribute alone."""
    constraints = attribute_ownership_rows(tuple(parent_plan.get("owned_constraints") or ()))
    plan = next((p for p in geography["plans"] if p["plan_id"] == parent_plan["plan_id"]), None)
    if plan is None:
        raise POICandidateDiscoveryError("interest_parent_geography_missing")
    stops = {f["properties"]["stop_id"]: f for f in plan["stops"]["features"]}
    result = []
    for (target, attribute), rows in constraints.items():
        owned = [
            r
            for r in rows
            if attribute == "existence"
            and r["canonical_value"] is True
            and r["strength"] in {"hard", "strong", "locked", "booked"}
        ]
        if not owned:
            continue
        stop = stops.get(target)
        if stop is not None and stop["properties"].get("day") != day:
            continue
        place = catalog.get_place(target) if catalog is not None else None
        coordinate = (
            list(stop["geometry"]["coordinates"])
            if stop
            else [place.display_coordinate.longitude, place.display_coordinate.latitude]
            if place
            else None
        )
        result.append(
            {
                "place_id": target,
                "stop_id": target,
                "name": stop["properties"].get("name", target) if stop else place.name if place else target,
                "coordinate": coordinate,
                "constraint_ids": [r["constraint_id"] for r in owned],
                "conflict_status": "satisfied" if stop else "required_stop_missing",
            }
        )
    return result


def _model(repository_root: Path) -> BayesianInterestModel:
    path = repository_root / "configs" / "interest_regions" / "feedback.json"
    if not path.exists():
        return BayesianInterestModel.prior()
    if path.is_symlink() or path.stat().st_size > 4_000_000:
        raise ValueError("interest_feedback_invalid")
    data = json.loads(path.read_text(encoding="utf-8"))
    if (
        data.get("schema_version") != "interest-feedback-v1"
        or data.get("feature_schema_version") != FEATURE_SCHEMA
        or data.get("evidence_mode") != "real_explicit_feedback"
    ):
        raise ValueError("interest_feedback_invalid")
    rows = data.get("rows")
    if not isinstance(rows, list) or not 1 <= len(rows) <= 10_000:
        raise ValueError("interest_feedback_invalid")
    identifiers = set()
    for row in rows:
        if (
            not isinstance(row, dict)
            or row.get("label_source") != "explicit_interest"
            or not isinstance(row.get("interested"), bool)
            or not row.get("exposure_id")
            or not isinstance(row.get("feedback_id"), str)
            or not row["feedback_id"]
            or row["feedback_id"] in identifiers
        ):
            raise ValueError("interest_feedback_invalid")
        identifiers.add(row["feedback_id"])
    return BayesianInterestModel.fit(
        np.array([row["features"] for row in rows]), np.array([row["interested"] for row in rows])
    )


def _distance_from_focus(coordinate: dict, focus: dict) -> float:
    longitude, latitude = np.radians([coordinate["longitude"], coordinate["latitude"]])
    center_lon, center_lat = np.radians([focus["longitude"], focus["latitude"]])
    haversine = np.sin((latitude - center_lat) / 2) ** 2
    haversine += np.cos(latitude) * np.cos(center_lat) * np.sin((longitude - center_lon) / 2) ** 2
    return float(2 * 6371.0088 * np.arcsin(np.sqrt(np.clip(haversine, 0, 1))))


def _interest_search_context(
    parent_plan,
    geography,
    day,
    route_leg_id,
    focus_stop_id,
    radius_km,
    repository_root=None,
    search_area="visitor_area",
):
    if (
        isinstance(radius_km, bool)
        or not isinstance(radius_km, (int, float))
        or not np.isfinite(radius_km)
        or not 1 <= radius_km <= 25
    ):
        raise POICandidateDiscoveryError("interest_radius_invalid")
    plan = next(p for p in geography["plans"] if p["plan_id"] == parent_plan["plan_id"])
    stops = [f for f in plan["stops"]["features"] if f["properties"]["day"] == day]
    leg = next(f for f in plan["validated_legs"]["features"] if f["properties"]["route_leg_id"] == route_leg_id)
    identifier = focus_stop_id or leg["properties"].get("destination_id")
    stop = next((f for f in stops if f["properties"]["stop_id"] == identifier), None)
    if focus_stop_id is not None and stop is None or not stops:
        raise POICandidateDiscoveryError("interest_focus_stop_invalid")
    stop = stop or stops[0]
    longitude, latitude = stop["geometry"]["coordinates"]
    visitor_area = None
    if search_area not in {"visitor_area", "itinerary_pin"}:
        raise POICandidateDiscoveryError("interest_search_area_invalid")
    if repository_root is not None:
        manifest = json.loads(
            (repository_root / "configs/interest_regions/exploration_manifest.json").read_text(encoding="utf-8")
        )
        visitor_area = next(
            (area for area in manifest.get("visitor_areas", []) if area["stop_id"] == stop["properties"]["stop_id"]),
            None,
        )
    name = stop["properties"]["name"]
    if visitor_area and search_area == "visitor_area":
        longitude, latitude = (visitor_area["coordinate"][key] for key in ("longitude", "latitude"))
        name += " · " + visitor_area["name"]
    return {
        "stop_id": stop["properties"]["stop_id"],
        "name": name,
        "area_mode": search_area,
        "visitor_area": visitor_area,
        "coordinate": {"longitude": longitude, "latitude": latitude},
        "radius_km": float(radius_km),
        "distance_kind": "straight_line",
    }


def _exploration_places(
    repository_root: Path,
    discovered: dict,
    parent_plan: dict,
    geography: dict,
    role: str | None,
    search_context: dict | None = None,
    place_cache: NearbyPlaceCache | None = None,
    refresh_places: bool = False,
) -> tuple[list, dict]:
    directory = repository_root / "configs" / "interest_regions"
    manifest = json.loads((directory / "exploration_manifest.json").read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != "interest-exploration-manifest-v1"
        or manifest.get("file") != "exploration_catalog.json"
    ):
        raise ValueError("interest_exploration_catalog_invalid")
    path = directory / "exploration_catalog.json"
    if path.is_symlink() or path.stat().st_size > 2_000_000:
        raise ValueError("interest_exploration_catalog_invalid")
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != manifest.get("sha256"):
        raise ValueError("interest_exploration_catalog_hash_mismatch")
    data = json.loads(payload)
    rows = data.get("places")
    if data.get("schema_version") != "interest-exploration-catalog-v1" or not isinstance(rows, list) or len(rows) > 500:
        raise ValueError("interest_exploration_catalog_invalid")
    if search_context is None:
        search_context = _interest_search_context(
            parent_plan, geography, discovered["context"]["day"], discovered["context"]["route_leg_id"], None, 5
        )
    place_data = {"status": "bundled_fallback", "fetched_at": None, "error_code": None, "cache_persisted": False}
    if place_cache is not None:
        online, place_data = place_cache.search(search_context, refresh=refresh_places)
        if place_data["status"] != "bundled_fallback":
            rows = [row for row in rows if not row["place_id"].startswith("osm_")]
            for point in online:
                duplicate = next(
                    (
                        row
                        for row in rows
                        if row["place_id"] == point["place_id"]
                        or row["name"].casefold() == point["name"].casefold()
                        and abs(row["display_coordinate"]["latitude"] - point["display_coordinate"]["latitude"]) < 0.002
                        and abs(row["display_coordinate"]["longitude"] - point["display_coordinate"]["longitude"])
                        < 0.002
                    ),
                    None,
                )
                if duplicate:
                    duplicate["source_refs"] = list(dict.fromkeys(duplicate["source_refs"] + point["source_refs"]))
                else:
                    rows.append(point)
    if place_data["status"] == "bundled_fallback":
        snapshot = next(
            (source for source in manifest.get("local_sources", []) if source["file"] == "itinerary_places.json"), {}
        )
        place_data["bundled_snapshot_at"] = snapshot.get("retrieved_at")
    manifest = {**manifest, "place_data": place_data}
    selected = parent_plan.get("selected_stops") or []
    result = []
    for row in rows:
        coordinate = row["display_coordinate"]
        if role is not None and role not in row["place_categories"]:
            continue
        if _distance_from_focus(coordinate, search_context["coordinate"]) > search_context["radius_km"]:
            continue
        if any(
            str(stop.get("attraction_name", stop.get("name"))).casefold().removesuffix(" viewpoint")
            == row["name"].casefold().removesuffix(" viewpoint")
            and (
                row.get("location_kind") == "park_entry"
                or abs(float(stop.get("latitude", 99)) - coordinate["latitude"]) < 0.01
                and abs(float(stop.get("longitude", 199)) - coordinate["longitude"]) < 0.01
            )
            for stop in selected
        ):
            continue
        result.append(row)
    return result, manifest


def discover_interest_regions(
    *,
    repository_root: Path,
    parent_plan: dict,
    geography: dict,
    nature_preference: float = 0.0,
    focus_stop_id: str | None = None,
    radius_km: float = 5.0,
    search_area: str = "visitor_area",
    place_cache: NearbyPlaceCache | None = None,
    refresh_places: bool = False,
    **context,
) -> dict:
    discovered = discover_poi_candidates(
        repository_root=repository_root,
        parent_plan=parent_plan,
        geography=geography,
        _complete_universe=True,
        **context,
    )
    catalog = _catalog(repository_root, context["trip_id"])
    search_context = _interest_search_context(
        parent_plan,
        geography,
        context["day"],
        context["route_leg_id"],
        focus_stop_id,
        radius_km,
        repository_root,
        search_area,
    )
    try:
        required = required_interest_points(parent_plan, geography, context["day"], catalog)
        model = _model(repository_root)
        for point in required:
            source = catalog.get_place(point["place_id"])
            point["interest_probability_mean"] = None
            point["interest_probability_interval"] = None
            if source is not None:
                estimated = model.predict(
                    np.array(
                        [
                            interest_features(
                                source.place_categories,
                                nature_preference=nature_preference,
                                weather_suitability=source.weather_suitability,
                                visit_minutes=source.recommended_visit_minutes,
                                static_utility=source.static_utility,
                            )
                        ]
                    )
                )
                point["interest_probability_mean"] = float(estimated["mean"][0])
                point["interest_probability_interval"] = estimated["interval"][0].tolist()
    except (ValueError, KeyError, TypeError) as exc:
        raise POICandidateDiscoveryError("interest_model_or_constraints_invalid") from exc
    candidates = discovered["candidates"]
    required_ids = {point["place_id"] for point in required}
    try:
        exploration, exploration_manifest = _exploration_places(
            repository_root,
            discovered,
            parent_plan,
            geography,
            context.get("role"),
            search_context,
            place_cache,
            refresh_places,
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise POICandidateDiscoveryError("interest_exploration_catalog_unavailable") from exc
    optional = []
    matched = set()
    for place in exploration:
        exact = next(
            (
                c
                for c in candidates
                if c["place"]["name"].casefold() == place["name"].casefold()
                and abs(c["place"]["display_coordinate"]["longitude"] - place["display_coordinate"]["longitude"])
                < 0.001
                and abs(c["place"]["display_coordinate"]["latitude"] - place["display_coordinate"]["latitude"]) < 0.001
            ),
            None,
        )
        if exact:
            matched.add(exact["candidate_id"])
        optional.append({"place": place, "candidate": exact})
    for row in candidates:
        if row["candidate_id"] not in required_ids and row["candidate_id"] not in matched:
            optional.append(
                {
                    "place": {**row["place"], "static_utility": catalog.get_place(row["candidate_id"]).static_utility},
                    "candidate": row,
                }
            )
    optional = [row for row in optional if not row["candidate"] or row["candidate"]["candidate_id"] not in required_ids]
    optional = [
        row
        for row in optional
        if _distance_from_focus(row["place"]["display_coordinate"], search_context["coordinate"]) <= radius_km
    ]
    if len(optional) > 500:
        raise POICandidateDiscoveryError("interest_candidate_universe_limit_exceeded")
    features = np.array(
        [
            interest_features(
                row["place"]["place_categories"],
                nature_preference=nature_preference,
                weather_suitability=row["place"].get("weather_suitability"),
                visit_minutes=row["place"].get("recommended_visit_minutes"),
                detour_minutes=row["candidate"]["burden"].get("marginal_travel_minutes") if row["candidate"] else None,
                static_utility=row["place"].get("static_utility"),
            )
            for row in optional
        ]
    ).reshape((-1, len(FEATURE_NAMES)))
    prediction = model.predict(features)
    predictions = []
    for index, row in enumerate(optional):
        predictions.append(
            {
                "place_id": row["candidate"]["candidate_id"] if row["candidate"] else row["place"]["place_id"],
                "name": row["place"]["name"],
                "display_coordinate": row["place"]["display_coordinate"],
                "distance_from_focus_km": _distance_from_focus(
                    row["place"]["display_coordinate"], search_context["coordinate"]
                ),
                "interest_probability_mean": float(prediction["mean"][index]),
                "interest_probability_interval": prediction["interval"][index].tolist(),
                "interval_level": prediction["interval_level"],
                "feasibility_status": row["candidate"]["burden"]["candidate_state"]
                if row["candidate"]
                else "not_evaluated",
                "source_refs": row["place"]["source_refs"],
                "description": row["place"].get("description"),
                "candidate": row["candidate"],
                "place_categories": row["place"]["place_categories"],
            }
        )
    surface = {
        "status": "unavailable",
        "unavailable_reason": "no_optional_candidates",
        "density_geojson": {"type": "FeatureCollection", "features": []},
        "regions_geojson": {"type": "FeatureCollection", "features": []},
    }
    if optional:
        try:
            geometry, manifest = load_land_support(repository_root)
            coordinates = np.array(
                [
                    [row["place"]["display_coordinate"]["longitude"], row["place"]["display_coordinate"]["latitude"]]
                    for row in optional
                ]
            )
            surface = build_interest_surface(
                coordinates,
                prediction["draws"],
                geometry=geometry,
                manifest=manifest,
                search_center=np.array(list(search_context["coordinate"][key] for key in ("longitude", "latitude"))),
                search_radius_m=radius_km * 1000,
            )
            for point, supported in zip(predictions, surface.pop("candidate_support"), strict=True):
                point["region_support_status"] = "supported" if supported else "outside_cartographic_land_support"
        except OSError:
            surface["unavailable_reason"] = "interest_land_support_unavailable"
        except (ValueError, KeyError, TypeError) as exc:
            code = str(exc)
            surface["unavailable_reason"] = (
                code if code.startswith("interest_") and len(code) < 100 else "interest_surface_invalid"
            )
    return {
        "schema_version": "product-interest-regions-v1",
        "session_id": context["session_id"],
        "session_revision": context["session_revision"],
        "accepted_plan_id": parent_plan["plan_id"],
        "accepted_plan_content_hash": parent_plan["content_hash"],
        "context": discovered["context"],
        "search_context": search_context,
        "place_data": exploration_manifest["place_data"],
        "catalog": discovered["catalog"],
        "routing": discovered["routing"],
        "candidate_universe": {
            "count": len(optional),
            "catalog_match_count": exploration_manifest["count"],
            "truncated": False,
            "sha256": hashlib.sha256(json.dumps(optional, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
            "exploration_catalog_sha256": exploration_manifest["sha256"],
            "route_checked_candidate_count": sum(row["candidate"] is not None for row in optional),
            "scope": "stop_radius_online_or_saved_places_plus_local_exact_route_candidates",
            "route_eligibility_note": "Interest exploration does not establish insertion feasibility.",
        },
        "nature_preference": nature_preference,
        "model": model.metadata(),
        "poi_predictions": predictions,
        "required_points": required,
        "surface": surface,
    }
