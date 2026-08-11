"""Artifact-derived geographic payloads for the local product map."""

from __future__ import annotations

import math
from typing import Any


class GeographyError(ValueError):
    """Raised when declared plans cannot be represented with validated routes."""


def build_geographic_workspace(
    bundle: Any,
    *,
    additional_plans: tuple[tuple[dict[str, Any], str], ...] = (),
    route_legs_by_plan: dict[str, tuple[dict[str, Any], ...]] | None = None,
) -> dict[str, Any]:
    """Build GeoJSON from validated plans and the declared road-route matrix."""

    route_matrix = bundle.route_matrix
    if not isinstance(route_matrix, dict):
        return _unavailable("route_matrix_missing")
    cells = route_matrix.get("cells")
    if not isinstance(cells, list):
        return _unavailable("route_matrix_invalid")
    cell_index = {
        (str(cell.get("origin_id") or ""), str(cell.get("destination_id") or "")): cell
        for cell in cells
        if isinstance(cell, dict)
    }
    declared = [
        (bundle.parent_plan, "original", "Original route"),
        (bundle.child_plan, "registered_repair", "Registered repair"),
        *((plan, "alternative", label) for plan, label in additional_plans),
    ]
    plans: list[dict[str, Any]] = []
    try:
        for plan, role, label in declared:
            if plan is not None:
                plans.append(
                    _plan_geojson(
                        plan,
                        role,
                        label,
                        cell_index,
                        route_legs=(route_legs_by_plan or {}).get(str(plan.get("plan_id") or "")),
                    )
                )
    except GeographyError as exc:
        return _unavailable(str(exc))
    if len(plans) < 2:
        return _unavailable("registered_repair_missing")
    coordinates = [
        feature["geometry"]["coordinates"]
        for plan in plans
        for collection in (plan["stops"], plan["routes"])
        for feature in collection["features"]
    ]
    flattened = [point for value in coordinates for point in _points(value)]
    if not flattened:
        return _unavailable("geographic_coordinates_missing")
    longitudes = [point[0] for point in flattened]
    latitudes = [point[1] for point in flattened]
    return {
        "schema_version": "product-geography-v1",
        "status": "ready",
        "code": "artifact_geography_ready",
        "bounds": [min(longitudes), min(latitudes), max(longitudes), max(latitudes)],
        "route_matrix_id": str(route_matrix.get("matrix_id") or ""),
        "route_source_sha256": str(route_matrix.get("source_content_sha256") or ""),
        "plans": plans,
        "attribution": {
            "label": "© OpenStreetMap contributors",
            "url": "https://www.openstreetmap.org/copyright",
        },
    }


def _plan_geojson(
    plan: dict[str, Any],
    role: str,
    label: str,
    cell_index: dict[tuple[str, str], dict[str, Any]],
    route_legs: tuple[dict[str, Any], ...] | None = None,
) -> dict[str, Any]:
    plan_id = str(plan.get("plan_id") or "")
    content_hash = str(plan.get("content_hash") or "")
    sequence = [str(value) for value in plan.get("sequence") or [] if str(value)]
    stop_lookup = {
        str(stop.get("stop_id") or stop.get("poi_id") or ""): stop
        for stop in plan.get("selected_stops") or []
        if isinstance(stop, dict)
    }
    if not plan_id or not content_hash or len(sequence) < 2:
        raise GeographyError("plan_geography_invalid")
    ownership = _ownership(plan)
    stop_features = []
    for sequence_index, stop_id in enumerate(sequence, start=1):
        stop = stop_lookup.get(stop_id)
        if stop is None:
            raise GeographyError("plan_stop_missing")
        longitude = _coordinate(stop.get("longitude"), longitude=True)
        latitude = _coordinate(stop.get("latitude"), longitude=False)
        stop_features.append(
            {
                "type": "Feature",
                "id": f"{plan_id}:{stop_id}",
                "geometry": {"type": "Point", "coordinates": [longitude, latitude]},
                "properties": {
                    "plan_id": plan_id,
                    "content_hash": content_hash,
                    "role": role,
                    "stop_id": stop_id,
                    "name": str(stop.get("attraction_name") or stop.get("name") or stop_id),
                    "city": str(stop.get("city") or ""),
                    "day": int(stop.get("day") or 0),
                    "stop_order": int(stop.get("stop_order") or sequence_index),
                    "sequence_index": sequence_index,
                    "ownership_strength": ownership.get(stop_id, ""),
                },
            }
        )
    route_features = []
    route_specs = route_legs or tuple(
        {
            "day": None,
            "origin_id": origin_id,
            "destination_id": destination_id,
            "evidence_scope": "global_plan_sequence_overview",
        }
        for origin_id, destination_id in zip(sequence, sequence[1:], strict=False)
    )
    for leg_index, spec in enumerate(route_specs, start=1):
        origin_id = str(spec.get("origin_id") or "")
        destination_id = str(spec.get("destination_id") or "")
        cell = cell_index.get((origin_id, destination_id))
        if cell is None:
            raise GeographyError("route_leg_missing")
        if not bool(cell.get("road_validated")) or bool(cell.get("fallback_used")):
            raise GeographyError("route_leg_not_road_validated")
        geometry = cell.get("geometry")
        if not isinstance(geometry, list) or len(geometry) < 2:
            raise GeographyError("route_geometry_missing")
        coordinates = [_latlon_to_geojson(point) for point in geometry]
        route_features.append(
            {
                "type": "Feature",
                "id": f"{plan_id}:leg:{leg_index}",
                "geometry": {"type": "LineString", "coordinates": coordinates},
                "properties": {
                    "plan_id": plan_id,
                    "content_hash": content_hash,
                    "role": role,
                    "leg_index": leg_index,
                    "day": spec.get("day"),
                    "evidence_scope": str(spec.get("evidence_scope") or ""),
                    "origin_id": origin_id,
                    "destination_id": destination_id,
                    "road_validated": True,
                    "fallback_used": False,
                    "geometry_source": str(cell.get("geometry_source") or ""),
                    "route_leg_id": str(cell.get("route_leg_id") or ""),
                    "distance_m": cell.get("distance_m"),
                    "duration_s": cell.get("duration_s"),
                },
            }
        )
    return {
        "plan_id": plan_id,
        "content_hash": content_hash,
        "role": role,
        "label": label,
        "stops": {"type": "FeatureCollection", "features": stop_features},
        "routes": {"type": "FeatureCollection", "features": route_features},
    }


def _ownership(plan: dict[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for record in plan.get("owned_constraints") or []:
        if not isinstance(record, dict):
            continue
        target = str(record.get("target_id") or record.get("stop_id") or "")
        strength = str(record.get("strength") or record.get("owner_strength") or "")
        if target and strength:
            result[target] = strength
    return result


def _coordinate(value: Any, *, longitude: bool) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GeographyError("coordinate_invalid")
    result = float(value)
    limit = 180.0 if longitude else 90.0
    if not math.isfinite(result) or not -limit <= result <= limit:
        raise GeographyError("coordinate_invalid")
    return result


def _latlon_to_geojson(value: Any) -> list[float]:
    if not isinstance(value, list) or len(value) != 2:
        raise GeographyError("route_coordinate_invalid")
    latitude = _coordinate(value[0], longitude=False)
    longitude = _coordinate(value[1], longitude=True)
    return [longitude, latitude]


def _points(value: Any) -> list[list[float]]:
    if isinstance(value, list) and len(value) == 2 and all(
        isinstance(item, (int, float)) and not isinstance(item, bool) for item in value
    ):
        return [value]
    if isinstance(value, list):
        return [point for item in value for point in _points(item)]
    return []


def _unavailable(code: str) -> dict[str, Any]:
    return {
        "schema_version": "product-geography-v1",
        "status": "unavailable",
        "code": code,
        "bounds": None,
        "route_matrix_id": None,
        "route_source_sha256": None,
        "plans": [],
        "attribution": {
            "label": "© OpenStreetMap contributors",
            "url": "https://www.openstreetmap.org/copyright",
        },
    }
