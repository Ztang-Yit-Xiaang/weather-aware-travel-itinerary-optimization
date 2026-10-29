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


def _exploration_places(
    repository_root: Path, discovered: dict, parent_plan: dict, geography: dict, role: str | None
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
    plan = next(p for p in geography["plans"] if p["plan_id"] == parent_plan["plan_id"])
    leg = next(
        f
        for f in plan["validated_legs"]["features"]
        if f["properties"]["route_leg_id"] == discovered["context"]["route_leg_id"]
    )
    focus = tuple(tuple(point) for point in leg["geometry"]["coordinates"])
    focus_radians = np.radians(np.asarray(focus))
    focus_cosines = np.cos(focus_radians[:, 1])
    selected = parent_plan.get("selected_stops") or []
    result = []
    for row in rows:
        coordinate = row["display_coordinate"]
        if role is not None and role not in row["place_categories"]:
            continue
        longitude, latitude = np.radians([coordinate["longitude"], coordinate["latitude"]])
        haversines = np.sin((latitude - focus_radians[:, 1]) / 2) ** 2
        haversines += np.cos(latitude) * focus_cosines * np.sin((longitude - focus_radians[:, 0]) / 2) ** 2
        nearest = np.clip(haversines.min(), 0, 1)
        distance_m = 2 * 6_371_008.8 * np.arctan2(np.sqrt(nearest), np.sqrt(1 - nearest))
        if distance_m > 50_000:
            continue
        if any(
            stop.get("attraction_name", stop.get("name")) == row["name"]
            and (
                row.get("location_kind") == "park_entry"
                or abs(float(stop.get("latitude", 99)) - coordinate["latitude"]) < 0.001
                and abs(float(stop.get("longitude", 199)) - coordinate["longitude"]) < 0.001
            )
            for stop in selected
        ):
            continue
        result.append(row)
    return result, manifest


def discover_interest_regions(
    *, repository_root: Path, parent_plan: dict, geography: dict, nature_preference: float = 0.0, **context
) -> dict:
    discovered = discover_poi_candidates(
        repository_root=repository_root,
        parent_plan=parent_plan,
        geography=geography,
        _complete_universe=True,
        **context,
    )
    catalog = _catalog(repository_root, context["trip_id"])
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
            repository_root, discovered, parent_plan, geography, context.get("role")
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
            surface = build_interest_surface(coordinates, prediction["draws"], geometry=geometry, manifest=manifest)
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
        "catalog": discovered["catalog"],
        "routing": discovered["routing"],
        "candidate_universe": {
            "count": len(optional),
            "catalog_match_count": exploration_manifest["count"],
            "truncated": False,
            "sha256": hashlib.sha256(json.dumps(optional, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
            "exploration_catalog_sha256": exploration_manifest["sha256"],
            "route_checked_candidate_count": len(candidates),
            "scope": "complete_50km_catalog_neighborhood_plus_exact_route_candidates",
            "route_eligibility_note": "Interest exploration does not establish insertion feasibility.",
        },
        "nature_preference": nature_preference,
        "model": model.metadata(),
        "poi_predictions": predictions,
        "required_points": required,
        "surface": surface,
    }
