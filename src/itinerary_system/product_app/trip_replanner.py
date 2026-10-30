"""Road-backed whole-trip planning with immutable evidence and independent evaluation."""

from __future__ import annotations

import hashlib
import json
import math
import os
import struct
from copy import deepcopy
from dataclasses import asdict, replace
from datetime import UTC, datetime

from ..blueprint_core import SOCIAL_MUST_GO_PLACES
from ..evaluation import PlanEvaluator, PlanEvaluatorConfig
from ..plans.diff import PlanDiffBuilder
from ..research_artifacts import PlanArtifactV2, PlannerRun, stable_content_hash
from ..routing import RouteLegRequest, RouteMatrix, RouteMatrixCell, RouteMatrixRequest
from .island_excursion import CHANNEL_ISLANDS, CHANNEL_URLS, EXCURSION, NOTICE, PORT
from .official_travel import OFFICIAL_URLS, REDWOOD_URLS, read_official_sources
from .persistence import LocalStateLayout
from .routing_runtime import RuntimeRoutingService
from .trip_planning import (
    CITY_ANCHORS,
    IndividualizedTripParameters,
    resolve_trip_profile,
    trip_day_count,
    validate_trip_parameters,
)
from .trip_solver import solve_trip
from .workspace import WorkspaceError
from .yosemite_extension import NOTICE as YOSEMITE_NOTICE
from .yosemite_extension import YOSEMITE_POIS, YOSEMITE_URLS, extend_yosemite_solution, validate_yosemite_extension

INLAND = {
    "yosemite_national_park",
    "sequoia_national_park",
    "kings_canyon_national_park",
    "best_western_yosemite_way_station",
    "yosemite_southgate_hotel",
    "comfort_inn_sequoia_kings_canyon",
}
MAX_EVIDENCE_AGE_SECONDS = 1800


def _hash(value):
    # Raw planning dictionaries mix integers and floats. JSON loses that distinction
    # in JavaScript, so hash each numeric value by its exact binary64 representation.
    def canonical(item):
        if item is None or isinstance(item, (bool, str)):
            return json.dumps(item, ensure_ascii=False, allow_nan=False)
        if isinstance(item, (int, float)):
            if not math.isfinite(item):
                raise ValueError("non_finite_planning_hash_number")
            return "#" + struct.pack(">d", float(item)).hex()
        if isinstance(item, (list, tuple)):
            return "[" + ",".join(canonical(part) for part in item) + "]"
        return (
            "{"
            + ",".join(json.dumps(key, ensure_ascii=False) + ":" + canonical(item[key]) for key in sorted(item))
            + "}"
        )

    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def _crosses_big_sur(geometry):
    """Intersect road segments with a declared conservative corridor, not a fire perimeter."""

    def inequalities(point):
        lat, lon = point
        return lat - 35.7, 36.45 - lat, lon + 122.1, -lon - 121.2 - 0.8 * (lat - 35.7)

    for a, b in zip(geometry, geometry[1:], strict=False):
        low, high = 0.0, 1.0
        for first, last in zip(inequalities(a), inequalities(b), strict=True):
            delta = last - first
            if abs(delta) < 1e-12:
                if first < 0:
                    high = -1
                    break
            elif delta > 0:
                low = max(low, -first / delta)
            else:
                high = min(high, -first / delta)
        if low <= high:
            return True
    return False


def _catalog(compiler, parameters):
    scenario = compiler._base_bundle.run_manifest["scenario"]
    points = {p["entity_id"]: (p["latitude"], p["longitude"]) for p in scenario["place_sources"]}
    names = {p["entity_id"]: p["name"] for p in scenario["place_sources"]}
    # Existing road endpoints outrank a park/landmark display centroid.
    for cell in compiler._base_bundle.route_matrix["cells"]:
        if len(cell.get("geometry", [])) >= 2 and cell.get("road_validated") and not cell.get("fallback_used"):
            points[cell["origin_id"]] = tuple(cell["geometry"][0])
            points[cell["destination_id"]] = tuple(cell["geometry"][-1])
    original_catalog = next(iter(compiler._bundles_by_plan.values())).parent_plan
    stops = {str(p.get("poi_id") or p.get("stop_id")): deepcopy(p) for p in original_catalog["selected_stops"]}
    stops.update(
        {str(p.get("poi_id") or p.get("stop_id")): deepcopy(p) for p in compiler.parent_plan["selected_stops"]}
    )
    for item in compiler._build_candidates(compiler._bundles_by_plan).values():
        stops.setdefault(
            item.candidate_id,
            {
                **deepcopy(item.record),
                "poi_id": item.candidate_id,
                "attraction_name": item.label,
                "visit_duration_minutes": 45,
            },
        )
    for row in SOCIAL_MUST_GO_PLACES:
        key = row["name"].lower().replace(" ", "_")
        if (
            key in stops
            or row["city"] not in {"Los Angeles", "Santa Barbara"}
            and not isinstance(parameters, IndividualizedTripParameters)
        ):
            continue
        points[key] = row["latitude"], row["longitude"]
        names[key] = row["name"]
        stops[key] = {
            "poi_id": key,
            "attraction_name": row["name"],
            "city": row["city"],
            "latitude": row["latitude"],
            "longitude": row["longitude"],
            "visit_duration_minutes": 45,
            "source_list": "repository_curated_social_places",
        }
    points["hearst_castle"] = (35.65082726, -121.18603425666667)
    names["hearst_castle"] = "Hearst Castle · Visitor Center"
    stops["hearst_castle"] = {
        "poi_id": "hearst_castle",
        "attraction_name": names["hearst_castle"],
        "city": "San Simeon",
        "latitude": points["hearst_castle"][0],
        "longitude": points["hearst_castle"][1],
        "visit_duration_minutes": 180,
        "source_list": "OSM way 90955598; official Hearst Castle visitor-center access",
        "official_url": "https://hearstcastle.org/plan/faqs/",
        "tour_reservation_status": "not_verified",
        "access_note": "Road destination is the visitor center. Castle access requires the official shuttle; no tour ticket is reserved.",
    }
    points[CHANNEL_ISLANDS] = (PORT[1], PORT[0])
    names[CHANNEL_ISLANDS] = "Channel Islands day trip · Ventura Harbor ferry"
    stops[CHANNEL_ISLANDS] = {
        "poi_id": CHANNEL_ISLANDS,
        "attraction_name": names[CHANNEL_ISLANDS],
        "city": "Ventura",
        "latitude": PORT[1],
        "longitude": PORT[0],
        "visit_duration_minutes": EXCURSION["planning_minutes"],
        "source_list": "NPS and Island Packers; US Census terminal address; OSM Scorpion pier",
        "official_url": CHANNEL_URLS["channel_islands_nps"],
        "excursion": deepcopy(EXCURSION),
        "access_note": NOTICE,
    }
    # NPS REDW trailhead feature 30580, not the park's discovery/display centroid.
    points["redwood_national_park"] = (41.304183, -124.018552)
    names["redwood_national_park"] = "Redwood National Park · Lady Bird Johnson Grove"
    stops["redwood_national_park"] = {
        "poi_id": "redwood_national_park",
        "attraction_name": names["redwood_national_park"],
        "city": "Orick",
        "latitude": 41.304183,
        "longitude": -124.018552,
        "visit_duration_minutes": 90,
        "source_list": "NPS REDW trailhead feature 30580; Lady Bird Johnson Grove walking guide",
        "official_url": REDWOOD_URLS["redwood_walk"],
        "access_note": "Drive to the Bald Hills Road trailhead parking lot, then walk the forest loop. Allow 90 minutes including the NPS 45–60 minute walk. Narrow road; no RVs or trailers, no roadside parking. Road damage and trip-date conditions need rechecking.",
    }
    original = set(compiler.parent_plan["sequence"])
    excluded = set(parameters.excluded_place_ids) | {
        p for p, row in stops.items() if row.get("weather_feasible") is False
    }
    if parameters.coastal:
        excluded |= INLAND | {"hollywood_walk_of_fame", "tcl_chinese_theatre"}
    if parameters.avoid_regions:
        excluded |= {p for p, point in points.items() if _crosses_big_sur((point, point))}
    hotels = set(compiler.parent_plan["lodging_assignments"].values()) - excluded
    required = (original - excluded) | set(parameters.required_place_ids)
    if "redwood_national_park" in required:
        catalog = json.loads(
            (compiler.repository_root / "configs/product_poi_catalogs/california_statewide/catalog.json").read_text(
                encoding="utf-8"
            )
        )
        gateway = next(p for p in catalog["places"] if p["place_id"] == "ca_gateway_arcata")
        coordinate = gateway["display_coordinate"]
        points["arcata_overnight_area"] = (coordinate["latitude"], coordinate["longitude"])
        names["arcata_overnight_area"] = "Arcata overnight area · hotel not selected"
        hotels.add("arcata_overnight_area")
    stops = {p: row for p, row in stops.items() if p not in excluded}
    if not required <= stops.keys() or set(parameters.required_place_ids) & excluded:
        raise WorkspaceError("trip_planning_required_place_unavailable", 409)
    if not (set(parameters.required_place_ids) | set(parameters.excluded_place_ids)) <= set(points):
        raise WorkspaceError("trip_planning_place_not_in_catalog", 422)
    # A start/closure/required-place request reoptimizes the requested itinerary;
    # only an explicit coastal replan opens the additional coastal candidate pool.
    if not parameters.coastal and not isinstance(parameters, IndividualizedTripParameters):
        stops = {p: row for p, row in stops.items() if p in required}
    if isinstance(parameters, IndividualizedTripParameters):
        for special in (CHANNEL_ISLANDS, "redwood_national_park"):
            if special not in required:
                stops.pop(special, None)
        # Reuse the hash-pinned discovery catalog, but require a fresh near-road
        # access check before a discovery location may enter the route matrix.
        directory = compiler.repository_root / "configs/interest_regions"
        manifest = json.loads((directory / "exploration_manifest.json").read_text())
        payload = (directory / "exploration_catalog.json").read_bytes()
        if hashlib.sha256(payload).hexdigest() != manifest["sha256"]:
            raise WorkspaceError("trip_planning_catalog_evidence_mismatch", 409)
        rows = json.loads(payload)["places"]
        existing_names = {str(r.get("attraction_name", "")).casefold() for r in stops.values()}
        blocked_names = {names[p].casefold() for p in excluded if p in names}
        pool = [
            r
            for r in rows
            if r["name"].casefold() not in existing_names | blocked_names
            and r["place_id"] not in excluded
            and r["place_categories"]
            and "national_park" not in r.get("category_source_text", "")
            and 33.5 <= r["display_coordinate"]["latitude"] <= 38.5
        ]
        interest_set = set(parameters.interests)

        def rank(row):
            point = row["display_coordinate"]
            distance = min(
                (point["latitude"] - points[p][0]) ** 2 + (point["longitude"] - points[p][1]) ** 2 for p in original
            )
            return (-len(interest_set & set(row["place_categories"])), distance, row["place_id"])

        router = getattr(compiler, "planning_router", None) or RuntimeRoutingService(
            base_url=compiler.planning_routing_url
            or os.environ.get("PRODUCT_ROUTING_BASE_URL", "http://127.0.0.1:5000")
        )
        target = max(len(required), parameters.total_days) + 3
        for row in sorted(pool, key=rank):
            if len(stops) >= target:
                break
            point = row["display_coordinate"]
            raw = (point["latitude"], point["longitude"])
            if parameters.coastal and raw[1] > -120 and raw[0] > 35:
                continue
            if parameters.avoid_regions and _crosses_big_sur((raw, raw)):
                continue
            snap = router.nearest(row["place_id"], raw)
            if snap.snap_distance_m > 50 or snap.confirmation_required or snap.validation_state == "rejected":
                continue
            key = row["place_id"]
            points[key] = snap.snapped_point
            names[key] = row["name"]
            stops[key] = {
                "poi_id": key,
                "attraction_name": row["name"],
                "latitude": snap.snapped_point[0],
                "longitude": snap.snapped_point[1],
                "visit_duration_minutes": row.get("recommended_visit_minutes") or 60,
                "source_list": row["source_refs"],
                "place_categories": row["place_categories"],
                "description": row.get("description"),
                "access_note": "Fresh road snap within 50 m; entrance, opening hours and tickets remain unverified.",
                "road_access": {
                    "snap_distance_m": snap.snap_distance_m,
                    "raw_point": list(raw),
                    "status": "road_snap_only",
                },
            }
        for key, row in stops.items():
            categories = set(row.get("place_categories", []))
            label = names.get(key, key).lower()
            if any(word in label for word in ("park", "forest", "trail", "grove", "meadow")):
                categories.add("nature")
            if any(word in label for word in ("museum", "courthouse", "theatre", "campus", "castle")):
                categories.add("culture")
            if any(word in label for word in ("beach", "coast", "wharf", "pier", "lobos", "bixby")):
                categories.add("coast")
            row["preference_penalty_minutes"] = 120 if interest_set and not interest_set & categories else 0
    first = scenario["days"][0]["start_anchor_id"]
    start = CITY_ANCHORS.get(parameters.start_city, first)
    end = scenario["days"][-1]["end_anchor_id"]
    entities = set(stops) | hotels | {start, end}
    if getattr(parameters, "additional_yosemite_days", 0):
        if max(s.get("day", 0) for s in compiler.parent_plan["selected_stops"]) != 7:
            raise WorkspaceError("yosemite_extension_requires_seven_day_parent", 409)
        for key, row in YOSEMITE_POIS.items():
            stops[key] = deepcopy(row)
            points[key] = (row["latitude"], row["longitude"])
            names[key] = row["attraction_name"]
        required |= set(YOSEMITE_POIS)
        entities |= set(YOSEMITE_POIS)
    if not entities <= points.keys():
        raise WorkspaceError("trip_planning_catalog_coordinate_missing", 409)
    # Route only usable candidates and original-plan endpoints needed for comparison.
    # Discovery-only and unused optional points do not belong in the OSRM table.
    comparison = (
        set(original_catalog["sequence"])
        | set(compiler.parent_plan["sequence"])
        | set(compiler.parent_plan["lodging_assignments"].values())
    )
    routing_points = entities | comparison
    return stops, hotels, {p: points[p] for p in sorted(routing_points)}, required, excluded, start, end, names


def _road_leg(routing, origin, destination, points, avoid_big_sur):
    def route(u, v, locations):
        return routing.route(RouteLegRequest(u, v, locations[u], locations[v]))

    leg = route(origin, destination, points)
    if not avoid_big_sur or not _crosses_big_sur(leg.geometry):
        return leg
    via = [("salinas_detour", (36.6777, -121.6555)), ("paso_robles_detour", (35.6369, -120.6545))]
    if points[origin][0] < points[destination][0]:
        via.reverse()
    locations = {**points, **dict(via)}
    ids = [origin, *[name for name, _ in via], destination]
    legs = [route(u, v, locations) for u, v in zip(ids, ids[1:], strict=False)]
    geometry = tuple(point for index, item in enumerate(legs) for point in item.geometry[1 if index else 0 :])
    if _crosses_big_sur(geometry):
        raise WorkspaceError("trip_planning_avoidance_route_unavailable", 409)
    return replace(
        leg,
        geometry=geometry,
        duration_s=sum(p.duration_s for p in legs),
        distance_m=sum(p.distance_m for p in legs),
        query_hash=stable_content_hash([p.query_hash for p in legs]),
        routing_status="ors_waypoint_detour_validated"
        if leg.provider == "openrouteservice"
        else "osrm_waypoint_detour_validated",
        geometry_source="runtime_ors_geojson_via_highway_101"
        if leg.provider == "openrouteservice"
        else "runtime_osrm_geojson_via_highway_101",
    )


def _matrix_record(matrix):
    cells = [asdict(value) for _, value in sorted(matrix.cells.items())]
    digest = _hash(cells)
    record = {
        "matrix_id": "route_matrix_" + digest[:16],
        "context_snapshot_id": matrix.context_snapshot_id,
        "entity_ids": list(matrix.entity_ids),
        "cells": cells,
        "source_bundle_id": "route_bundle_" + digest[:16],
        "source_content_sha256": digest,
    }
    return json.loads(json.dumps(record, allow_nan=False))


def validate_planning_bundle(parent, bundle, *, check_age=True):
    """Recompute artifact, road, user-requirement and schedule checks on every reuse."""
    from .draft_compiler import _plan_artifact, _route_matrix

    payload = {k: v for k, v in bundle.items() if k != "content_hash"}
    if (
        bundle.get("content_hash") != _hash(payload)
        or bundle["parent_plan_id"] != parent["plan_id"]
        or bundle["parent_content_hash"] != parent["content_hash"]
    ):
        raise WorkspaceError("trip_planning_evidence_mismatch", 409)
    requested = validate_trip_parameters(bundle["parameters"])
    parameters = validate_trip_parameters(bundle.get("effective_parameters", bundle["parameters"]))
    if isinstance(requested, IndividualizedTripParameters):
        if not isinstance(parameters, IndividualizedTripParameters) or parameters.total_days != requested.total_days:
            raise WorkspaceError("trip_planning_request_requirements_mismatch", 409)
        if any(getattr(parameters, key) != getattr(requested, key) for key in requested.profile_updates):
            raise WorkspaceError("trip_planning_request_requirements_mismatch", 409)
        if any(
            getattr(parameters, key) != getattr(requested, key)
            for key in ("start_city", "required_place_ids", "avoid_regions", "verify_official", "profile_updates")
        ) or not set(requested.excluded_place_ids) <= set(parameters.excluded_place_ids):
            raise WorkspaceError("trip_planning_request_requirements_mismatch", 409)
    else:
        if (
            any(
                getattr(parameters, key) != getattr(requested, key)
                for key in ("start_city", "required_place_ids", "avoid_regions", "verify_official")
            )
            or not set(requested.excluded_place_ids) <= set(parameters.excluded_place_ids)
            or requested.coastal
            and not parameters.coastal
            or requested.max_daily_drive_minutes != 720
            and parameters.max_daily_drive_minutes != requested.max_daily_drive_minutes
            or isinstance(parameters, IndividualizedTripParameters)
            and parameters.total_days != trip_day_count(requested, parent)
        ):
            raise WorkspaceError("trip_planning_request_requirements_mismatch", 409)
    sources = bundle["official_sources"]
    expected_sources = {"big_sur_park", "highway_1", "hearst_castle"}
    if CHANNEL_ISLANDS in bundle["plan"]["sequence"]:
        expected_sources |= set(CHANNEL_URLS)
    if "redwood_national_park" in bundle["plan"]["sequence"]:
        expected_sources |= set(REDWOOD_URLS)
    if getattr(parameters, "additional_yosemite_days", 0):
        expected_sources |= set(YOSEMITE_URLS)
    if (
        len(sources) != len(expected_sources)
        or {row["source_id"] for row in sources} != expected_sources
        or any(OFFICIAL_URLS[row["source_id"]] != row["url"] or row["status"] != "retrieved" for row in sources)
    ):
        raise WorkspaceError("trip_planning_official_evidence_mismatch", 409)
    user_avoid = bundle["user_avoid_regions"]
    if (
        not set(parameters.avoid_regions) <= set(user_avoid)
        or not set(user_avoid) <= {"big_sur"}
        or bundle["enforced_avoid_big_sur"]
        != (bool(user_avoid) or any(row["big_sur_highway_closure"] for row in sources))
    ):
        raise WorkspaceError("trip_planning_request_requirements_mismatch", 409)
    now = datetime.now(UTC)
    for timestamp in [bundle["created_at"], *[row["retrieved_at"] for row in bundle["official_sources"]]]:
        instant = datetime.fromisoformat(timestamp)
        if instant.tzinfo is None:
            raise WorkspaceError("trip_planning_evidence_mismatch", 409)
        age = (now - instant).total_seconds()
        if age < 0 or check_age and age > MAX_EVIDENCE_AGE_SECONDS:
            raise WorkspaceError("trip_planning_evidence_expired", 409)
    plan = _plan_artifact(bundle["plan"])
    if (
        plan.parent_plan_id != parent["plan_id"]
        or not set(parameters.required_place_ids) <= set(bundle["required_stop_ids"])
        or not set(parameters.excluded_place_ids) <= set(bundle["excluded_stop_ids"])
        or parameters.coastal
        and set(plan.sequence) & INLAND
    ):
        raise WorkspaceError("trip_planning_request_requirements_mismatch", 409)
    island_stops = [s for s in plan.selected_stops if (s.get("poi_id") or s.get("stop_id")) == CHANNEL_ISLANDS]
    if any(
        s.get("excursion") != EXCURSION
        or s.get("visit_duration_minutes") != EXCURSION["planning_minutes"]
        or (s.get("longitude"), s.get("latitude")) != PORT
        for s in island_stops
    ) or any(s.get("excursion") for s in plan.selected_stops if s not in island_stops):
        raise WorkspaceError("trip_planning_island_excursion_mismatch", 409)
    redwood_stops = [s for s in plan.selected_stops if (s.get("poi_id") or s.get("stop_id")) == "redwood_national_park"]
    if redwood_stops and (
        len(redwood_stops) != 1
        or redwood_stops[0].get("visit_duration_minutes") != 90
        or (redwood_stops[0].get("latitude"), redwood_stops[0].get("longitude")) != (41.304183, -124.018552)
        or next(row["park_access"] for row in sources if row["source_id"] == "redwood_conditions")
        not in {"open", "open_with_hazards"}
    ):
        raise WorkspaceError("trip_planning_redwood_access_mismatch", 409)
    validate_yosemite_extension(plan, bundle["days"], bool(getattr(parameters, "additional_yosemite_days", 0)))
    if (
        getattr(parameters, "additional_yosemite_days", 0)
        and next(r["park_access"] for r in sources if r["source_id"] == "yosemite_conditions") != "open_with_hazards"
    ):
        raise WorkspaceError("trip_planning_official_access_unresolved", 409)
    matrix = _route_matrix(bundle["route_matrix"])
    if _hash(bundle["route_matrix"]["cells"]) != matrix.source_content_sha256:
        raise WorkspaceError("trip_planning_route_evidence_mismatch", 409)
    config = PlanEvaluatorConfig(
        strict_routes=True,
        max_day_minutes=720.0,
        day_start_time="09:00",
        default_visit_minutes=45.0,
        enforce_opening_windows=True,
        start_anchor_by_day={r["day"]: r["start_anchor_id"] for r in bundle["days"]},
        end_anchor_by_day={r["day"]: r["end_anchor_id"] for r in bundle["days"]},
        required_stop_ids=tuple(bundle["required_stop_ids"]),
        excluded_stop_ids=tuple(bundle["excluded_stop_ids"]),
        require_lodging_assignments=True,
        include_schedule_metrics=True,
    )
    expected_days = trip_day_count(parameters, parent)
    if [d["day"] for d in bundle["days"]] != list(range(1, expected_days + 1)) or config.start_anchor_by_day[
        1
    ] != bundle["start_anchor_id"]:
        raise WorkspaceError("trip_planning_route_requirements_mismatch", 409)
    if isinstance(parameters, IndividualizedTripParameters):
        limit = {"relaxed": 2, "balanced": 3, "active": 4}[parameters.pace]
        if any(len(day["stop_ids"]) > limit for day in bundle["days"]):
            raise WorkspaceError("trip_planning_pace_not_satisfied", 409)
    if parameters.start_city and config.start_anchor_by_day[1] != CITY_ANCHORS[parameters.start_city]:
        raise WorkspaceError("trip_planning_start_not_satisfied", 409)
    previous = None
    for day in bundle["days"]:
        if previous and day["start_anchor_id"] != previous:
            raise WorkspaceError("trip_planning_overnight_discontinuity", 409)
        previous = day["end_anchor_id"]
        ids = [day["start_anchor_id"], *day["stop_ids"], day["end_anchor_id"]]
        travel = 0.0
        for u, v in zip(ids, ids[1:], strict=False):
            cell = matrix.cell(u, v)
            cell.require_publication_eligible()
            if len(cell.geometry) < 2 or bundle["enforced_avoid_big_sur"] and _crosses_big_sur(cell.geometry):
                raise WorkspaceError("trip_planning_route_requirements_mismatch", 409)
            travel += cell.require_duration_s() / 60
        if travel > parameters.max_daily_drive_minutes + 1e-6:
            raise WorkspaceError("trip_planning_driving_limit_exceeded", 409)
    certificate = (
        PlanEvaluator(route_matrix=matrix, config=config, reference_plan=_plan_artifact(parent))
        .evaluate_final_plan(plan, planner_run=bundle["planner_run"])
        .to_record()
    )
    return plan, matrix, config, certificate


def compile_trip_replan(compiler, operations, *, read_only=False, refresh_expired=False):
    from .draft_compiler import CompiledDraftPreview, _plan_artifact
    from .evaluated_stop_edits import _route_legs

    if len(operations) != 1 or operations[0]["target"] != "trip" or compiler.planning_state_root is None:
        raise WorkspaceError("trip_planning_requires_separate_draft", 409)
    if read_only and refresh_expired:
        raise WorkspaceError("trip_planning_refresh_read_only", 409)
    requested = validate_trip_parameters(operations[0]["parameters"])
    parameters = resolve_trip_profile(requested, compiler._planning_baseline_bundle or {}, parent=compiler.parent_plan)
    identity = stable_content_hash({"parent": compiler.parent_plan["content_hash"], "operation": operations[0]})
    cache = compiler.planning_state_root / (identity + ".json")
    with compiler._runtime_cache_lock:
        bundle = None
        if cache.is_file():
            bundle = json.loads(cache.read_text(encoding="utf-8"))
            if (
                bundle.get("parameters") != requested.model_dump()
                or bundle.get("request_id") != "trip_request_" + identity
            ):
                raise WorkspaceError("trip_planning_request_identity_mismatch", 409)
            if refresh_expired:
                try:
                    validate_planning_bundle(compiler.parent_plan, bundle)
                except WorkspaceError as exc:
                    if exc.code != "trip_planning_evidence_expired":
                        raise
                    # Expiry may precede later integrity/road checks. Verify those
                    # too before replacing evidence; retain the old cache if fetching fails.
                    validate_planning_bundle(compiler.parent_plan, bundle, check_age=False)
                    bundle = None
        if bundle is None:
            if read_only:
                raise WorkspaceError("interest_preview_unavailable", 409)
            stops, hotels, points, required, excluded, start, end, names = _catalog(compiler, parameters)
            official = read_official_sources(
                big_sur=True,
                hearst=True,
                channel_islands=CHANNEL_ISLANDS in required,
                redwood="redwood_national_park" in required,
                yosemite=bool(getattr(parameters, "additional_yosemite_days", 0)),
            )
            if (
                getattr(parameters, "additional_yosemite_days", 0)
                and next(r["park_access"] for r in official if r["source_id"] == "yosemite_conditions")
                != "open_with_hazards"
            ):
                raise WorkspaceError("trip_planning_official_access_unresolved", 409)
            if "redwood_national_park" in required and next(
                row["park_access"] for row in official if row["source_id"] == "redwood_conditions"
            ) not in {"open", "open_with_hazards"}:
                raise WorkspaceError("trip_planning_official_access_unresolved", 409)
            user_avoid = sorted(
                set(parameters.avoid_regions)
                | set((compiler._planning_baseline_bundle or {}).get("user_avoid_regions", []))
            )
            avoid_big_sur = bool(user_avoid) or any(row["big_sur_highway_closure"] for row in official)
            park_status = next(row["park_access"] for row in official if row["source_id"] == "big_sur_park")
            if park_status != "open" or avoid_big_sur:
                if "pfeiffer_big_sur_state_park" in parameters.required_place_ids:
                    raise WorkspaceError("trip_planning_official_access_unresolved", 409)
                stops.pop("pfeiffer_big_sur_state_park", None)
                required.discard("pfeiffer_big_sur_state_park")
                excluded.add("pfeiffer_big_sur_state_park")
            context = "context_trip_" + stable_content_hash(
                {
                    "parent": compiler.parent_plan["content_hash"],
                    "parameters": parameters.model_dump(),
                    "official": official,
                }
            )
            routing = getattr(compiler, "planning_router", None) or RuntimeRoutingService(
                base_url=compiler.planning_routing_url
                or os.environ.get("PRODUCT_ROUTING_BASE_URL", "http://127.0.0.1:5000")
            )
            if len(points) <= 25:
                matrix = routing.table(RouteMatrixRequest(context, points))
            else:
                # Keep each router request within its 25-point bound while
                # retaining a complete directed matrix for the longer trip.
                ids = sorted(points)
                groups = [ids[i : i + 12] for i in range(0, len(ids), 12)]
                cells = {}
                for i, group in enumerate(groups):
                    for other in groups[i:]:
                        subset = {key: points[key] for key in set(group + other)}
                        cells.update(routing.table(RouteMatrixRequest(context, subset)).cells)
                matrix = RouteMatrix("route_matrix_" + stable_content_hash(points), context, tuple(ids), cells)
            cells = dict(matrix.cells)
            # Retain verified original geometry for comparison; new child legs below are fetched afresh.
            for raw in compiler._base_bundle.route_matrix["cells"]:
                pair = (raw["origin_id"], raw["destination_id"])
                if pair in cells and len(raw.get("geometry", [])) >= 2:
                    cells[pair] = replace(cells[pair], geometry=tuple(tuple(p) for p in raw["geometry"]))
            validated = set()
            for _attempt in range(3):
                matrix = replace(matrix, cells=cells)
                solution = solve_trip(
                    stops={
                        k: v
                        for k, v in stops.items()
                        if not getattr(parameters, "additional_yosemite_days", 0) or k not in YOSEMITE_POIS
                    },
                    hotels=hotels,
                    matrix=matrix,
                    required=required - set(YOSEMITE_POIS)
                    if getattr(parameters, "additional_yosemite_days", 0)
                    else required,
                    start=start,
                    end=end,
                    days=7
                    if getattr(parameters, "additional_yosemite_days", 0)
                    else trip_day_count(parameters, compiler.parent_plan),
                    max_stops_per_day={"relaxed": 2, "balanced": 3, "active": 4}[parameters.pace]
                    if isinstance(parameters, IndividualizedTripParameters)
                    else None,
                    max_drive=parameters.max_daily_drive_minutes,
                    time_limit=8,
                )
                if getattr(parameters, "additional_yosemite_days", 0):
                    solution = extend_yosemite_solution(solution, matrix, parameters.max_daily_drive_minutes)
                changed = False
                for day in solution["days"]:
                    ids = [day["start_anchor_id"], *day["stop_ids"], day["end_anchor_id"]]
                    for u, v in zip(ids, ids[1:], strict=False):
                        if (u, v) in validated:
                            continue
                        leg = _road_leg(routing, u, v, points, avoid_big_sur)
                        old = cells[u, v].require_duration_s()
                        values = {k: v for k, v in asdict(leg).items() if k in RouteMatrixCell.__dataclass_fields__}
                        cells[u, v] = RouteMatrixCell(**values, context_snapshot_id=context)
                        validated.add((u, v))
                        changed |= abs(old - leg.duration_s) > 1e-5
                if not changed:
                    break
            if changed or getattr(parameters, "additional_yosemite_days", 0):
                solution["status"] = "FEASIBLE"
                solution["optimality_gap"] = None
                solution["best_bound"] = None
                solution["objective_value"] = sum(
                    cells[u, v].require_duration_s() / 60
                    for day in solution["days"]
                    for ids in [[day["start_anchor_id"], *day["stop_ids"], day["end_anchor_id"]]]
                    for u, v in zip(ids, ids[1:], strict=False)
                ) + sum(
                    15 + stops[p].get("preference_penalty_minutes", 0)
                    for day in solution["days"]
                    for p in day["stop_ids"]
                    if p not in required
                )
            solution["candidate_stop_ids"] = sorted(stops)
            matrix = replace(matrix, cells=cells)
            matrix_record = _matrix_record(matrix)
            request_id = "trip_request_" + identity
            generation = stable_content_hash({"request": identity, "context": context})
            run_id = "trip_run_" + generation
            days = solution["days"]
            total_days = len(days)
            selected = []
            end_lodging = compiler.parent_plan["lodging_assignments"][
                str(max(int(d) for d in compiler.parent_plan["lodging_assignments"]))
            ]
            for day in days:
                for index, stop_id in enumerate(day["stop_ids"], 1):
                    selected.append(
                        {
                            **stops[stop_id],
                            "poi_id": stop_id,
                            "day": day["day"],
                            "stop_order": index,
                            "lodging_id": day["end_anchor_id"] if day["day"] < total_days else end_lodging,
                        }
                    )
            plan = PlanArtifactV2(
                plan_id="plan_" + generation,
                source_run_id=run_id,
                planning_request_id=request_id,
                catalog_snapshot_id="catalog_trip_" + stable_content_hash(points),
                context_snapshot_id=context,
                selected_stops=tuple(selected),
                parent_plan_id=compiler.parent_plan_id,
                day_assignments={s["poi_id"]: s["day"] for s in selected},
                sequence=tuple(s["poi_id"] for s in selected),
                lodging_assignments={
                    str(d["day"]): d["end_anchor_id"] if d["day"] < total_days else end_lodging for d in days
                },
                ordered_days=tuple({"day": d["day"], "stop_ids": d["stop_ids"]} for d in days),
                route_ids_by_day={d["day"]: "trip_route_" + stable_content_hash(d) for d in days},
                owned_constraints=_plan_artifact(compiler.parent_plan).owned_constraints,
                created_at="",
            )
            run = PlannerRun(
                run_id=run_id,
                parent_run_id=compiler.parent_plan["source_run_id"],
                planning_request_id=request_id,
                catalog_snapshot_id=plan.catalog_snapshot_id,
                context_snapshot_id=context,
                planner_specification_id="trip_milp_v1",
                method_requested="trip_milp_v1",
                method_executed="trip_milp_v1",
                execution_status="COMPLETED",
                result_plan_id=plan.plan_id,
                solver_backend="scipy_highs_milp",
                objective_value=solution["objective_value"],
                best_bound=solution["best_bound"],
                optimality_gap=solution["optimality_gap"],
                runtime_seconds=solution["runtime_seconds"],
            )
            bundle = {
                "schema_version": "trip-planning-bundle-v1",
                "created_at": datetime.now(UTC).isoformat(),
                "parent_plan_id": compiler.parent_plan_id,
                "parent_content_hash": compiler.parent_plan["content_hash"],
                "request_id": request_id,
                "parameters": requested.model_dump(),
                "effective_parameters": parameters.model_dump(),
                "plan": plan.to_record(),
                "planner_run": run.to_record(),
                "route_matrix": matrix_record,
                "days": days,
                "start_anchor_id": start,
                "end_anchor_id": end,
                "required_stop_ids": sorted(required),
                "excluded_stop_ids": sorted(excluded),
                "official_sources": official,
                "enforced_avoid_big_sur": avoid_big_sur,
                "user_avoid_regions": user_avoid,
                "solver": solution,
                "place_sources": [
                    {
                        "entity_id": p,
                        "latitude": v[0],
                        "longitude": v[1],
                        "name": names.get(p, p),
                        "source_ref": "california_statewide_gateway_coordinates_v1; city-center planning base, hotel not selected"
                        if p == "arcata_overnight_area"
                        else "trip_planning_catalog_v1",
                    }
                    for p, v in points.items()
                ],
                "notices": [
                    "Road times exclude live traffic. Hotel availability is not booked or verified.",
                    *(
                        [
                            f"Personalized profile: {parameters.total_days} days, {parameters.pace} pace, {parameters.max_daily_drive_minutes} driving minutes/day maximum; interests: {', '.join(parameters.interests) or 'existing trip'}. Each day has at least one distinct visit; free time is not filled with duplicate visits."
                        ]
                        if isinstance(parameters, IndividualizedTripParameters)
                        else []
                    ),
                    *([YOSEMITE_NOTICE] if getattr(parameters, "additional_yosemite_days", 0) else []),
                    *([NOTICE] if CHANNEL_ISLANDS in plan.sequence else []),
                    *(
                        [
                            stops["redwood_national_park"]["access_note"],
                            "Arcata overnight area is a city-center planning base, not a selected or booked hotel. Final accommodation may change travel time.",
                        ]
                        if "redwood_national_park" in plan.sequence
                        else []
                    ),
                    "Pfeiffer Big Sur access is excluded when official notices conflict, report closure, or cannot establish park access.",
                    "Official bulletins apply at retrieval time; recheck for the actual travel dates.",
                    *(
                        [
                            "Hearst Castle requires a shuttle and a tour ticket. A 180-minute planning allowance is not a reserved tour slot."
                        ]
                        if "hearst_castle" in plan.sequence
                        else []
                    ),
                ],
            }
            bundle["content_hash"] = _hash(bundle)
            validate_planning_bundle(compiler.parent_plan, bundle)
            cache.parent.mkdir(parents=True, exist_ok=True)
            LocalStateLayout._atomic_json_write(cache, bundle)
    if bundle["parameters"] != requested.model_dump() or bundle["request_id"] != "trip_request_" + identity:
        raise WorkspaceError("trip_planning_request_identity_mismatch", 409)
    plan, matrix, config, certificate = validate_planning_bundle(compiler.parent_plan, bundle, check_age=not read_only)
    diff = (
        PlanDiffBuilder(default_visit_minutes=45.0, allow_snapshot_transition=True)
        .diff(_plan_artifact(compiler.parent_plan), plan)
        .to_record()
    )
    scenario = {
        **compiler._base_bundle.run_manifest["scenario"],
        "days": bundle["days"],
        "place_sources": bundle["place_sources"],
    }
    source = replace(
        compiler._base_bundle,
        route_matrix=bundle["route_matrix"],
        run_manifest={**compiler._base_bundle.run_manifest, "scenario": scenario},
    )
    old_matrix, old_config, _ = compiler._runtime_inputs()
    receipt = {
        "schema_version": "trip-planning-execution-v1",
        "planning_bundle": bundle,
        "evaluation_certificate": certificate,
        "repair_request": {
            "request_id": bundle["request_id"],
            "user_message": "Whole-trip constrained replan",
            "parameters": bundle["parameters"],
        },
        "plan_diff": diff,
        "solver": bundle["solver"],
        "official_sources": bundle["official_sources"],
        "notices": bundle["notices"],
        "snapshot_transition": {
            "parent_catalog": compiler.parent_plan["catalog_snapshot_id"],
            "child_catalog": plan.catalog_snapshot_id,
            "parent_context": compiler.parent_plan["context_snapshot_id"],
            "child_context": plan.context_snapshot_id,
        },
    }
    eligible = certificate["comparison_eligibility"] == "eligible"
    return CompiledDraftPreview(
        state="eligible" if eligible else "ineligible",
        reason=None if eligible else "independent_evaluation_failed",
        source_request_id=bundle["request_id"],
        operations=operations,
        source_bundle=source,
        child_plan=plan.to_record(),
        diff=diff,
        certificate=certificate,
        planner_run=bundle["planner_run"],
        parent_route_legs=_route_legs(
            _plan_artifact(compiler.parent_plan),
            old_matrix,
            old_config.start_anchor_by_day,
            old_config.end_anchor_by_day,
        ),
        route_legs=_route_legs(plan, matrix, config.start_anchor_by_day, config.end_anchor_by_day),
        execution_mode="trip_replan_independent_evaluation",
        runtime_execution=receipt,
    )
