"""Build the approved, coherent seven-day California product demo package.

The builder uses only repository-curated place coordinates and the configured
loopback OSRM runtime. It never falls back to straight-line geometry. The
result is an immutable direct product package with independently evaluated
repair alternatives.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from itinerary_system.evaluation import PlanEvaluator, PlanEvaluatorConfig  # noqa: E402
from itinerary_system.plans import compute_plan_diff  # noqa: E402
from itinerary_system.product_app.product_demo_coherence import (  # noqa: E402
    validate_product_demo_coherence,
)
from itinerary_system.research_artifacts import (  # noqa: E402
    PlanArtifactV2,
    PlannerRun,
    stable_content_hash,
)
from itinerary_system.routing import RouteMatrix, RouteMatrixCell  # noqa: E402

PACKAGE_SCHEMA = "product-demo-package-v2"
RUN_ID = "california_coast_coherent_demo_v1"
SCENARIO_ID = "california_coast_approved_seven_day"
CONTEXT_ID = "context_coherent_california_demo_2026_08"
CATALOG_ID = "california_coherent_product_catalog_v1"
REQUEST_ID = "weather_repair_bixby_day_5_v1"
CREATED_AT = "2026-08-11T00:00:00+00:00"

PLACES: dict[str, dict[str, Any]] = {
    "san_francisco_international_airport": {
        "name": "San Francisco International Airport",
        "city": "San Francisco",
        "latitude": 37.6213,
        "longitude": -122.3790,
        "source_ref": "existing_product_route_anchor:san_francisco_international_airport",
    },
    "golden_gate_bridge": {
        "name": "Golden Gate Bridge",
        "city": "San Francisco",
        "latitude": 37.8199,
        "longitude": -122.4783,
        "source_ref": "src/itinerary_system/data_enrichment.py:CURATED_PLACE_DETAIL_FALLBACKS",
    },
    "hotel_zephyr_san_francisco": {
        "name": "Hotel Zephyr San Francisco",
        "city": "San Francisco",
        "latitude": 37.8070,
        "longitude": -122.4141,
        "source_ref": "src/itinerary_system/data_enrichment.py:CURATED_FALLBACK_HOTELS",
    },
    "stanford_university_main_quad": {
        "name": "Stanford University Main Quad",
        "city": "Stanford",
        "latitude": 37.4275,
        "longitude": -122.1697,
        "source_ref": "src/itinerary_system/data_enrichment.py:CURATED_PLACE_DETAIL_FALLBACKS",
    },
    "best_western_yosemite_way_station": {
        "name": "Best Western Plus Yosemite Way Station Motel",
        "city": "Mariposa",
        "latitude": 37.4895,
        "longitude": -119.9662,
        "source_ref": "src/itinerary_system/data_enrichment.py:CURATED_FALLBACK_HOTELS",
    },
    "yosemite_national_park": {
        "name": "Yosemite National Park",
        "city": "Yosemite Valley",
        "latitude": 37.8651,
        "longitude": -119.5383,
        "source_ref": "src/itinerary_system/region_scenarios.py:CALIFORNIA_STATEWIDE_NATURE_REGIONS",
    },
    "yosemite_southgate_hotel": {
        "name": "Yosemite Southgate Hotel & Suites",
        "city": "Oakhurst",
        "latitude": 37.3285,
        "longitude": -119.6499,
        "source_ref": "src/itinerary_system/data_enrichment.py:CURATED_FALLBACK_HOTELS",
    },
    "sequoia_national_park": {
        "name": "Sequoia National Park",
        "city": "Three Rivers",
        "latitude": 36.4864,
        "longitude": -118.5658,
        "source_ref": "src/itinerary_system/region_scenarios.py:CALIFORNIA_STATEWIDE_NATURE_REGIONS",
    },
    "kings_canyon_national_park": {
        "name": "Kings Canyon National Park",
        "city": "Three Rivers",
        "latitude": 36.8879,
        "longitude": -118.5551,
        "source_ref": "src/itinerary_system/region_scenarios.py:CALIFORNIA_STATEWIDE_NATURE_REGIONS",
    },
    "comfort_inn_sequoia_kings_canyon": {
        "name": "Comfort Inn & Suites Sequoia Kings Canyon",
        "city": "Three Rivers",
        "latitude": 36.4488,
        "longitude": -118.9018,
        "source_ref": "src/itinerary_system/data_enrichment.py:CURATED_FALLBACK_HOTELS",
    },
    "bixby_creek_bridge_viewpoint": {
        "name": "Bixby Creek Bridge Viewpoint",
        "city": "Big Sur",
        "latitude": 36.371479,
        "longitude": -121.901894,
        "source_ref": "configs/product_poi_catalogs/california_coast_demo/catalog.json",
    },
    "pfeiffer_big_sur_state_park": {
        "name": "Pfeiffer Big Sur State Park",
        "city": "Big Sur",
        "latitude": 36.2524,
        "longitude": -121.7879,
        "source_ref": "src/itinerary_system/region_scenarios.py:california_coast.curated_seed_pois",
    },
    "point_lobos_state_natural_reserve": {
        "name": "Point Lobos State Natural Reserve",
        "city": "Monterey",
        "latitude": 36.5159,
        "longitude": -121.9425,
        "source_ref": "src/itinerary_system/region_scenarios.py:california_coast.curated_seed_pois",
    },
    "portola_hotel_monterey": {
        "name": "Portola Hotel & Spa at Monterey Bay",
        "city": "Monterey",
        "latitude": 36.6022,
        "longitude": -121.8947,
        "source_ref": "src/itinerary_system/data_enrichment.py:CURATED_FALLBACK_HOTELS",
    },
    "stearns_wharf": {
        "name": "Stearns Wharf",
        "city": "Santa Barbara",
        "latitude": 34.4100137734,
        "longitude": -119.6855497819,
        "source_ref": "runs/e3ux-weather-repair-demo-v6/plans/plan_e1c4f803691e3188.json",
    },
    "hotel_milo_santa_barbara": {
        "name": "Hotel Milo Santa Barbara",
        "city": "Santa Barbara",
        "latitude": 34.4106,
        "longitude": -119.6918,
        "source_ref": "src/itinerary_system/data_enrichment.py:CURATED_FALLBACK_HOTELS",
    },
    "griffith_observatory": {
        "name": "Griffith Observatory",
        "city": "Los Angeles",
        "latitude": 34.1184,
        "longitude": -118.3004,
        "source_ref": "src/itinerary_system/data_enrichment.py:CURATED_PLACE_DETAIL_FALLBACKS",
    },
    "the_line_la": {
        "name": "The LINE LA",
        "city": "Los Angeles",
        "latitude": 34.0628,
        "longitude": -118.3009,
        "source_ref": "src/itinerary_system/data_enrichment.py:CURATED_FALLBACK_HOTELS",
    },
    "los_angeles_international_airport": {
        "name": "Los Angeles International Airport",
        "city": "Los Angeles",
        "latitude": 33.9416,
        "longitude": -118.4085,
        "source_ref": "existing_product_route_anchor:los_angeles_international_airport",
    },
}

DAYS = (
    {
        "day": 1,
        "label": "San Francisco",
        "start_anchor_id": "san_francisco_international_airport",
        "end_anchor_id": "hotel_zephyr_san_francisco",
        "parent_stop_ids": ["golden_gate_bridge"],
    },
    {
        "day": 2,
        "label": "Stanford & Silicon Valley",
        "start_anchor_id": "hotel_zephyr_san_francisco",
        "end_anchor_id": "best_western_yosemite_way_station",
        "parent_stop_ids": ["stanford_university_main_quad"],
    },
    {
        "day": 3,
        "label": "Yosemite National Park",
        "start_anchor_id": "best_western_yosemite_way_station",
        "end_anchor_id": "yosemite_southgate_hotel",
        "parent_stop_ids": ["yosemite_national_park"],
    },
    {
        "day": 4,
        "label": "Sequoia & Kings Canyon",
        "start_anchor_id": "yosemite_southgate_hotel",
        "end_anchor_id": "comfort_inn_sequoia_kings_canyon",
        "parent_stop_ids": ["sequoia_national_park", "kings_canyon_national_park"],
    },
    {
        "day": 5,
        "label": "Central Coast",
        "start_anchor_id": "comfort_inn_sequoia_kings_canyon",
        "end_anchor_id": "portola_hotel_monterey",
        "parent_stop_ids": ["bixby_creek_bridge_viewpoint"],
    },
    {
        "day": 6,
        "label": "Santa Barbara",
        "start_anchor_id": "portola_hotel_monterey",
        "end_anchor_id": "hotel_milo_santa_barbara",
        "parent_stop_ids": ["stearns_wharf"],
    },
    {
        "day": 7,
        "label": "Los Angeles",
        "start_anchor_id": "hotel_milo_santa_barbara",
        "end_anchor_id": "los_angeles_international_airport",
        "parent_stop_ids": ["griffith_observatory"],
    },
)

REPLACEMENTS = {
    "recommended": "pfeiffer_big_sur_state_park",
    "low_driving": "point_lobos_state_natural_reserve",
}

LODGING_BY_DAY = {
    "1": "hotel_zephyr_san_francisco",
    "2": "best_western_yosemite_way_station",
    "3": "yosemite_southgate_hotel",
    "4": "comfort_inn_sequoia_kings_canyon",
    "5": "portola_hotel_monterey",
    "6": "hotel_milo_santa_barbara",
    "7": "the_line_la",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY_ROOT / "runs" / "california-coast-coherent-demo-v1",
    )
    parser.add_argument("--routing-base-url", default="http://127.0.0.1:5000")
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def build_package(output_dir: Path, *, routing_base_url: str) -> Path:
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"coherent product demo output already exists: {output_dir}")
    _require_loopback(routing_base_url)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
    created = False
    try:
        parent = _build_plan("parent")
        children = {role: _build_plan(role, parent=parent) for role in REPLACEMENTS}
        sequences = {
            "parent": _plan_route_sequences(parent),
            **{role: _plan_route_sequences(plan) for role, plan in children.items()},
        }
        matrix, matrix_record = _build_route_matrix(sequences, routing_base_url=routing_base_url)
        planner_runs = {role: _planner_run(role, plan) for role, plan in children.items()}
        evaluator_config = PlanEvaluatorConfig(
            strict_routes=True,
            max_day_minutes=720.0,
            default_visit_minutes=45.0,
            enforce_opening_windows=False,
            start_anchor_by_day={day["day"]: day["start_anchor_id"] for day in DAYS},
            end_anchor_by_day={day["day"]: day["end_anchor_id"] for day in DAYS},
            weather_warning_threshold=0.7,
            require_lodging_assignments=True,
        )
        diffs: dict[str, dict[str, Any]] = {}
        certificates: dict[str, dict[str, Any]] = {}
        explanations: dict[str, dict[str, Any]] = {}
        for role, child in children.items():
            diff = compute_plan_diff(parent, child).to_record()
            certificate = PlanEvaluator(
                route_matrix=matrix,
                planner_runs={planner_runs[role].run_id: planner_runs[role]},
                config=evaluator_config,
                reference_plan=parent,
            ).evaluate(child)
            certificate = replace(certificate, evaluated_at=CREATED_AT)
            if certificate.comparison_eligibility != "eligible":
                raise ValueError(
                    f"coherent demo {role} alternative was rejected: "
                    f"{[(finding.code, finding.evidence_refs) for finding in certificate.failures]}"
                )
            diffs[role] = diff
            certificates[role] = certificate.to_record()
            explanations[role] = _explanation(role, parent, child, diff, certificate.content_hash)

        request = _request()
        paths = _write_artifacts(
            temporary,
            parent=parent,
            children=children,
            diffs=diffs,
            certificates=certificates,
            explanations=explanations,
            matrix_record=matrix_record,
            request=request,
            planner_runs=planner_runs,
        )
        matrix_path = temporary / paths["route_matrix"]
        manifest = _manifest(
            parent=parent,
            children=children,
            diffs=diffs,
            certificates=certificates,
            explanations=explanations,
            matrix_record=matrix_record,
            matrix_file_sha256=_file_sha256(matrix_path),
            paths=paths,
        )
        validate_product_demo_coherence(
            manifest=manifest,
            parent_plan=parent.to_record(),
            alternatives={role: (children[role].to_record(), diffs[role], explanations[role]) for role in REPLACEMENTS},
            route_matrix=matrix_record,
        )
        manifest["artifacts_sha256"] = {
            path.relative_to(temporary).as_posix(): _file_sha256(path)
            for path in sorted(temporary.rglob("*"))
            if path.is_file()
        }
        (temporary / "manifest.json").write_text(_stable_json(manifest), encoding="utf-8")
        temporary.replace(output_dir)
        created = True
        return output_dir / "manifest.json"
    finally:
        if not created:
            shutil.rmtree(temporary, ignore_errors=True)


def check_package(package_dir: Path) -> None:
    manifest = _json(package_dir / "manifest.json")
    declared = manifest.get("artifacts_sha256")
    if not isinstance(declared, dict) or not declared:
        raise ValueError("coherent product demo inventory is missing")
    actual = {
        path.relative_to(package_dir).as_posix()
        for path in package_dir.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    if set(declared) != actual:
        raise ValueError("coherent product demo inventory mismatch")
    for relative, expected in declared.items():
        if _file_sha256(package_dir / relative) != expected:
            raise ValueError(f"coherent product demo artifact hash mismatch: {relative}")
    parent = _json(package_dir / manifest["parent"]["plan_relative_path"])
    alternatives = {
        row["role"]: (
            _json(package_dir / row["plan_relative_path"]),
            _json(package_dir / row["diff_relative_path"]),
            _json(package_dir / row["explanation_relative_path"]),
        )
        for row in manifest["alternatives"]
    }
    route_matrix = _json(package_dir / manifest["route_evidence"]["relative_path"])
    validate_product_demo_coherence(
        manifest=manifest,
        parent_plan=parent,
        alternatives=alternatives,
        route_matrix=route_matrix,
    )


def _build_plan(role: str, *, parent: PlanArtifactV2 | None = None) -> PlanArtifactV2:
    replacement = REPLACEMENTS.get(role)
    selected_stops: list[dict[str, Any]] = []
    ordered_days: list[dict[str, Any]] = []
    sequence: list[str] = []
    for day in DAYS:
        stop_ids = list(day["parent_stop_ids"])
        if day["day"] == 5 and replacement:
            stop_ids = [replacement]
        ordered_days.append({"day": day["day"], "stop_ids": stop_ids})
        for order, stop_id in enumerate(stop_ids, start=1):
            place = PLACES[stop_id]
            stop = {
                "poi_id": stop_id,
                "attraction_name": place["name"],
                "city": place["city"],
                "day": day["day"],
                "stop_order": order,
                "latitude": place["latitude"],
                "longitude": place["longitude"],
                "visit_duration_minutes": 45,
                "lodging_id": LODGING_BY_DAY[str(day["day"])],
                "source_list": "repository_curated_product_fixture",
                "source_confidence": 1.0,
                "weather_feasible": stop_id != "bixby_creek_bridge_viewpoint",
                "weather_risk": (
                    0.86
                    if stop_id == "bixby_creek_bridge_viewpoint"
                    else 0.18
                    if stop_id == "pfeiffer_big_sur_state_park"
                    else 0.12
                    if stop_id == "point_lobos_state_natural_reserve"
                    else 0.2
                ),
                "itinerary_role": None,
                "itinerary_role_source": "unavailable",
            }
            selected_stops.append(stop)
            sequence.append(stop_id)
    plan_seed = {
        "scenario_id": SCENARIO_ID,
        "role": role,
        "parent_plan_id": parent.plan_id if parent else None,
        "sequence": sequence,
    }
    plan_id = f"plan_{stable_content_hash(plan_seed)}"
    source_run_id = f"{RUN_ID}_{role}"
    route_ids = {
        day[
            "day"
        ]: f"route_{stable_content_hash({'day': day['day'], 'sequence': [day['start_anchor_id'], *ordered_days[day['day'] - 1]['stop_ids'], day['end_anchor_id']]})}"
        for day in DAYS
    }
    return PlanArtifactV2(
        plan_id=plan_id,
        parent_plan_id=parent.plan_id if parent else None,
        source_run_id=source_run_id,
        planning_request_id=REQUEST_ID,
        catalog_snapshot_id=CATALOG_ID,
        context_snapshot_id=CONTEXT_ID,
        selected_stops=tuple(selected_stops),
        day_assignments={stop["poi_id"]: int(stop["day"]) for stop in selected_stops},
        sequence=tuple(sequence),
        lodging_assignments=dict(LODGING_BY_DAY),
        ordered_days=tuple(ordered_days),
        route_ids_by_day=route_ids,
        modeled_metrics={"selected_attractions": float(len(selected_stops))},
        context_exposure_components={},
        change_components={},
        created_at=CREATED_AT,
    )


def _planner_run(role: str, plan: PlanArtifactV2) -> PlannerRun:
    return PlannerRun(
        run_id=plan.source_run_id,
        planning_request_id=REQUEST_ID,
        catalog_snapshot_id=CATALOG_ID,
        context_snapshot_id=CONTEXT_ID,
        planner_specification_id="coherent_product_fixture_repair_v1",
        method_requested="registered_context_aware_repair",
        method_executed="registered_context_aware_repair",
        execution_status="COMPLETED",
        solver_certification="FEASIBILITY_CERTIFIED",
        solver_backend="deterministic_registered_fixture",
        solver_version="1",
        result_plan_id=plan.plan_id,
        started_at=CREATED_AT,
        completed_at=CREATED_AT,
    )


def _plan_route_sequences(plan: PlanArtifactV2) -> tuple[tuple[str, ...], ...]:
    by_day = {int(row["day"]): tuple(row["stop_ids"]) for row in plan.ordered_days}
    return tuple((day["start_anchor_id"], *by_day[day["day"]], day["end_anchor_id"]) for day in DAYS)


def _build_route_matrix(
    sequences: dict[str, tuple[tuple[str, ...], ...]],
    *,
    routing_base_url: str,
) -> tuple[RouteMatrix, dict[str, Any]]:
    pairs: list[tuple[str, str]] = []
    for plan_sequences in sequences.values():
        for sequence in plan_sequences:
            for pair in zip(sequence[:-1], sequence[1:], strict=True):
                if pair not in pairs:
                    pairs.append(pair)
    for day in DAYS:
        stop_ids = tuple(day["parent_stop_ids"])
        if len(stop_ids) < 2:
            continue
        reversed_sequence = (
            day["start_anchor_id"],
            *reversed(stop_ids),
            day["end_anchor_id"],
        )
        for pair in zip(reversed_sequence[:-1], reversed_sequence[1:], strict=True):
            if pair not in pairs:
                pairs.append(pair)
    records = [_route_cell(origin, destination, routing_base_url=routing_base_url) for origin, destination in pairs]
    source_seed = {
        "provider": "cached_osrm",
        "profile": "driving",
        "context_snapshot_id": CONTEXT_ID,
        "query_hashes": [record["query_hash"] for record in records],
    }
    source_content_sha256 = hashlib.sha256(_compact_json(source_seed).encode("utf-8")).hexdigest()
    entity_ids = sorted(
        {entity_id for record in records for entity_id in (record["origin_id"], record["destination_id"])}
    )
    matrix_seed = {
        "context_snapshot_id": CONTEXT_ID,
        "entity_ids": entity_ids,
        "cells": records,
        "source_content_sha256": source_content_sha256,
    }
    matrix_id = f"route_matrix_{stable_content_hash(matrix_seed)}"
    source_bundle_id = f"route_bundle_{stable_content_hash(source_seed)}"
    matrix_record = {
        "schema_version": "route-matrix-v1",
        "matrix_id": matrix_id,
        "context_snapshot_id": CONTEXT_ID,
        "entity_ids": entity_ids,
        "cells": records,
        "source_bundle_id": source_bundle_id,
        "source_content_sha256": source_content_sha256,
    }
    cells = {
        (record["origin_id"], record["destination_id"]): RouteMatrixCell(
            **{**record, "geometry": tuple(tuple(point) for point in record["geometry"])}
        )
        for record in records
    }
    return (
        RouteMatrix(
            matrix_id=matrix_id,
            context_snapshot_id=CONTEXT_ID,
            entity_ids=tuple(entity_ids),
            cells=cells,
            source_bundle_id=source_bundle_id,
            source_content_sha256=source_content_sha256,
        ),
        matrix_record,
    )


def _route_cell(origin_id: str, destination_id: str, *, routing_base_url: str) -> dict[str, Any]:
    origin = PLACES[origin_id]
    destination = PLACES[destination_id]
    coordinates = f"{origin['longitude']},{origin['latitude']};{destination['longitude']},{destination['latitude']}"
    url = f"{routing_base_url.rstrip('/')}/route/v1/driving/{coordinates}"
    params = {"overview": "full", "geometries": "geojson", "steps": "false"}
    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()
    payload = response.json()
    routes = payload.get("routes") if isinstance(payload, dict) else None
    if payload.get("code") != "Ok" or not isinstance(routes, list) or len(routes) != 1:
        raise ValueError(f"local OSRM returned invalid route evidence: {origin_id}->{destination_id}")
    route = routes[0]
    coordinates = (route.get("geometry") or {}).get("coordinates")
    distance = route.get("distance")
    duration = route.get("duration")
    if (
        not isinstance(coordinates, list)
        or len(coordinates) < 2
        or not isinstance(distance, (int, float))
        or isinstance(distance, bool)
        or distance <= 0
        or not isinstance(duration, (int, float))
        or isinstance(duration, bool)
        or duration <= 0
    ):
        raise ValueError(f"local OSRM route evidence is incomplete: {origin_id}->{destination_id}")
    geometry = [[round(float(point[1]), 6), round(float(point[0]), 6)] for point in coordinates]
    query_payload = {
        "origin_id": origin_id,
        "destination_id": destination_id,
        "origin": [origin["latitude"], origin["longitude"]],
        "destination": [destination["latitude"], destination["longitude"]],
        "profile": "driving",
        "distance_m": round(float(distance), 3),
        "duration_s": round(float(duration), 3),
        "geometry": geometry,
    }
    return {
        "origin_id": origin_id,
        "destination_id": destination_id,
        "distance_m": query_payload["distance_m"],
        "duration_s": query_payload["duration_s"],
        "route_leg_id": f"leg_{origin_id}_{destination_id}",
        "road_validated": True,
        "fallback_used": False,
        "provider": "cached_osrm",
        "query_hash": hashlib.sha256(_compact_json(query_payload).encode("utf-8")).hexdigest()[:16],
        "context_snapshot_id": CONTEXT_ID,
        "geometry": geometry,
        "routing_profile": "driving",
        "routing_status": "osrm_live",
        "geometry_source": "cached_osrm_route_geometry",
        "distance_source": "cached_osrm_route_distance",
        "duration_source": "cached_osrm_route_duration",
        "fallback_reason": None,
    }


def _request(role: str | None = None) -> dict[str, Any]:
    candidate_stop_ids = tuple(REPLACEMENTS.values()) if role is None else (REPLACEMENTS[role],)
    return {
        "schema_version": "repair-request-v1",
        "request_id": REQUEST_ID,
        "user_intent": "Replace the weather-infeasible Day 5 Bixby stop while preserving the approved seven-day San Francisco-to-Los Angeles trip.",
        "baseline_route": [
            {"day": day["day"], "label": day["label"], "stop_ids": list(day["parent_stop_ids"])} for day in DAYS
        ],
        "confirmed_constraints": {
            "affected_stop_id": "bixby_creek_bridge_viewpoint",
            "affected_day": 5,
            "preserve_day_assignments": [1, 2, 3, 4, 6, 7],
            "target_stop_ids": ["bixby_creek_bridge_viewpoint"],
            "require_road_validated_routes": True,
        },
        "candidate_pois": [{"stop_id": stop_id, **PLACES[stop_id]} for stop_id in candidate_stop_ids],
        "evidence_records": [
            {
                "ref_id": "weather_day_5_bixby",
                "source_type": "registered_product_fixture",
                "source_id": REQUEST_ID,
                "field_path": "confirmed_constraints.affected_stop_id",
            }
        ],
        "tolerance_profile": {"route_fallback_allowed": False},
    }


def _explanation(
    role: str,
    parent: PlanArtifactV2,
    child: PlanArtifactV2,
    diff: dict[str, Any],
    certificate_content_hash: str,
) -> dict[str, Any]:
    replacement_id = REPLACEMENTS[role]
    replacement = PLACES[replacement_id]
    evidence = [
        {
            "ref_id": "weather_day_5_bixby",
            "source_type": "registered_product_fixture",
            "source_id": REQUEST_ID,
            "field_path": "confirmed_constraints.affected_stop_id",
            "record_hash": stable_content_hash({"affected_stop_id": "bixby_creek_bridge_viewpoint", "day": 5}),
        },
        {
            "ref_id": f"route_{role}_day_5",
            "source_type": "route_matrix",
            "source_id": child.route_ids_by_day[5],
            "field_path": "route_validation.day_5",
            "record_hash": stable_content_hash({"route_id": child.route_ids_by_day[5], "role": role}),
        },
    ]
    return {
        "schema_version": "product-repair-explanation-v1",
        "explanation_id": f"explanation_{stable_content_hash({'child': child.plan_id, 'role': role})}",
        "parent_plan_id": parent.plan_id,
        "child_plan_id": child.plan_id,
        "affected_stop_id": "bixby_creek_bridge_viewpoint",
        "replacement_stop_id": replacement_id,
        "summary": f"Replace the weather-infeasible Day 5 Bixby stop with {replacement['name']} while preserving Days 1-4 and 6-7.",
        "diff_id": diff["diff_id"],
        "certificate_content_hash": certificate_content_hash,
        "claims": [
            {
                "claim_id": f"claim_{role}_replacement",
                "claim_type": "repair_change",
                "rendered_text": f"Day 5 replaces Bixby Creek Bridge Viewpoint with {replacement['name']}.",
                "confidence": "artifact_backed",
                "evidence_refs": ["weather_day_5_bixby", f"route_{role}_day_5"],
                "unsupported": False,
            },
            {
                "claim_id": f"claim_{role}_preservation",
                "claim_type": "preservation",
                "rendered_text": "The San Francisco, Stanford, Yosemite, Sequoia/Kings Canyon, Santa Barbara, and Los Angeles days remain unchanged.",
                "confidence": "artifact_backed",
                "evidence_refs": [f"route_{role}_day_5"],
                "unsupported": False,
            },
        ],
        "evidence_records": evidence,
    }


def _write_artifacts(
    root: Path,
    *,
    parent: PlanArtifactV2,
    children: dict[str, PlanArtifactV2],
    diffs: dict[str, dict[str, Any]],
    certificates: dict[str, dict[str, Any]],
    explanations: dict[str, dict[str, Any]],
    matrix_record: dict[str, Any],
    request: dict[str, Any],
    planner_runs: dict[str, PlannerRun],
) -> dict[str, Any]:
    paths: dict[str, Any] = {
        "parent_plan": f"plans/{parent.plan_id}.json",
        "request": f"requests/{REQUEST_ID}.json",
        "route_matrix": f"routing/{matrix_record['matrix_id']}.json",
        "planner_runs": "planner_runs.jsonl",
        "alternatives": {},
    }
    _write_json(root / paths["parent_plan"], parent.to_record())
    _write_json(root / paths["request"], request)
    _write_json(root / paths["route_matrix"], matrix_record)
    planner_text = "".join(
        json.dumps(planner_runs[role].to_record(), sort_keys=True, ensure_ascii=False) + "\n" for role in REPLACEMENTS
    )
    (root / paths["planner_runs"]).write_text(planner_text, encoding="utf-8")
    for role, child in children.items():
        role_paths = {
            "plan": f"plans/{child.plan_id}.json",
            "diff": f"diffs/{diffs[role]['diff_id']}.json",
            "request": f"requests/{REQUEST_ID}_{role}.json",
            "certificate": f"evaluations/{certificates[role]['certificate_id']}.json",
            "explanation": f"explanations/{explanations[role]['explanation_id']}.json",
            "metrics": f"metrics/{role}.json",
        }
        paths["alternatives"][role] = role_paths
        _write_json(root / role_paths["plan"], child.to_record())
        _write_json(root / role_paths["diff"], diffs[role])
        _write_json(root / role_paths["request"], _request(role))
        _write_json(root / role_paths["certificate"], certificates[role])
        _write_json(root / role_paths["explanation"], explanations[role])
        _write_json(
            root / role_paths["metrics"],
            {
                "schema_version": "product-demo-metrics-v1",
                "role": role,
                "weighted_edit_cost": diffs[role]["weighted_edit_cost"],
                "decision_eligible": True,
                "ranking_eligible": True,
            },
        )
    return paths


def _manifest(
    *,
    parent: PlanArtifactV2,
    children: dict[str, PlanArtifactV2],
    diffs: dict[str, dict[str, Any]],
    certificates: dict[str, dict[str, Any]],
    explanations: dict[str, dict[str, Any]],
    matrix_record: dict[str, Any],
    matrix_file_sha256: str,
    paths: dict[str, Any],
) -> dict[str, Any]:
    parent_sequences = _plan_route_sequences(parent)
    alternatives = []
    for role, child in children.items():
        route_legs = _route_legs(_plan_route_sequences(child))
        route_total = _route_total_minutes(route_legs, matrix_record)
        role_paths = paths["alternatives"][role]
        alternatives.append(
            {
                "role": role,
                "request_relative_path": role_paths["request"],
                "replacement_stop_id": REPLACEMENTS[role],
                "plan_relative_path": role_paths["plan"],
                "diff_relative_path": role_paths["diff"],
                "certificate_relative_path": role_paths["certificate"],
                "explanation_relative_path": role_paths["explanation"],
                "metrics_relative_path": role_paths["metrics"],
                "plan_id": child.plan_id,
                "plan_content_hash": child.content_hash,
                "certificate_id": certificates[role]["certificate_id"],
                "diff_id": diffs[role]["diff_id"],
                "route_total_minutes": route_total,
                "route_legs": route_legs,
            }
        )
    if alternatives[1]["route_total_minutes"] >= alternatives[0]["route_total_minutes"]:
        raise ValueError("the low-driving coherent alternative does not reduce route minutes")
    return {
        "schema_version": PACKAGE_SCHEMA,
        "run_id": RUN_ID,
        "package_id": RUN_ID,
        "created_at": CREATED_AT,
        "scenario": {
            "schema_version": "product-demo-scenario-v1",
            "scenario_id": SCENARIO_ID,
            "direction": "san_francisco_to_los_angeles",
            "affected_stop_id": "bixby_creek_bridge_viewpoint",
            "days": [dict(day) for day in DAYS],
            "place_sources": [
                {
                    "entity_id": entity_id,
                    "name": place["name"],
                    "latitude": place["latitude"],
                    "longitude": place["longitude"],
                    "source_ref": place["source_ref"],
                }
                for entity_id, place in sorted(PLACES.items())
            ],
        },
        "parent": {
            "plan_relative_path": paths["parent_plan"],
            "plan_id": parent.plan_id,
            "content_hash": parent.content_hash,
            "route_legs": _route_legs(parent_sequences, scope="road_validated_daily_route_leg"),
        },
        "request_relative_path": paths["request"],
        "planner_runs_relative_path": paths["planner_runs"],
        "route_evidence": {
            "relative_path": paths["route_matrix"],
            "matrix_id": matrix_record["matrix_id"],
            "source_bundle_id": matrix_record["source_bundle_id"],
            "source_content_sha256": matrix_record["source_content_sha256"],
            "matrix_file_sha256": matrix_file_sha256,
        },
        "alternatives": alternatives,
    }


def _route_legs(
    sequences: tuple[tuple[str, ...], ...],
    *,
    scope: str = "certified_daily_route_leg",
) -> list[dict[str, Any]]:
    return [
        {
            "day": day,
            "origin_id": origin,
            "destination_id": destination,
            "evidence_scope": scope,
        }
        for day, sequence in enumerate(sequences, start=1)
        for origin, destination in zip(sequence[:-1], sequence[1:], strict=True)
    ]


def _route_total_minutes(route_legs: list[dict[str, Any]], matrix: dict[str, Any]) -> float:
    cells = {(cell["origin_id"], cell["destination_id"]): cell for cell in matrix["cells"]}
    return round(
        sum(cells[(leg["origin_id"], leg["destination_id"])]["duration_s"] for leg in route_legs) / 60.0,
        6,
    )


def _require_loopback(base_url: str) -> None:
    parsed = urlparse(base_url)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost"} or not parsed.port:
        raise ValueError("routing base URL must be an explicit loopback HTTP authority")
    if parsed.path not in {"", "/"} or parsed.params or parsed.query or parsed.fragment:
        raise ValueError("routing base URL must not include a path, query, or fragment")


def _write_json(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_stable_json(record), encoding="utf-8")


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _stable_json(record: Any) -> str:
    return json.dumps(record, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"


def _compact_json(record: Any) -> str:
    return json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    args = parse_args()
    if args.check:
        check_package(args.output.resolve())
        print(args.output.resolve() / "manifest.json")
        return 0
    print(build_package(args.output, routing_base_url=args.routing_base_url))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
