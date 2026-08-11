"""Truthful product geography with complete route-path accounting.

Unlike the legacy product-geography adapter, this version keeps an otherwise
usable plan visible when one or more required route legs lack road-valid
evidence.  Validated router geometry and explicit null-geometry gaps are
separate collections, so clients cannot mistake a visual connector for route
evidence.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any

from .route_coverage import RouteCoverageReportV1, audit_route_coverage

_ITINERARY_ROLES = frozenset(
    {
        "attraction",
        "activity",
        "meal",
        "lodging",
        "transport_hub",
        "rest_stop",
        "scenic_stop",
        "route_waypoint",
        "origin",
        "destination",
    }
)


class GeographyV2Error(ValueError):
    """Raised when immutable plan geography is structurally invalid."""


def build_geographic_workspace_v2(
    bundle: Any,
    *,
    additional_plans: tuple[tuple[dict[str, Any], str], ...] = (),
    route_legs_by_plan: Mapping[str, Iterable[Mapping[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Build a plan-complete geographic payload from immutable artifacts.

    A missing or invalid route-matrix cell becomes an explicit gap. Structural
    plan errors still fail closed because the selected itinerary itself cannot
    be represented truthfully in that case.
    """

    route_matrix = bundle.route_matrix
    if not isinstance(route_matrix, Mapping):
        return _unavailable("route_matrix_missing")
    cells = route_matrix.get("cells")
    if not isinstance(cells, list):
        return _unavailable("route_matrix_invalid")
    cell_index = {
        (str(cell.get("origin_id") or ""), str(cell.get("destination_id") or "")): cell
        for cell in cells
        if isinstance(cell, Mapping)
    }
    declared = [
        (bundle.parent_plan, "original", "Original route"),
        (bundle.child_plan, "registered_repair", "Registered repair"),
        *((plan, "alternative", label) for plan, label in additional_plans),
    ]
    plans: list[dict[str, Any]] = []
    try:
        for plan, role, label in declared:
            if plan is None:
                continue
            plan_id = str(plan.get("plan_id") or "")
            route_specs = tuple((route_legs_by_plan or {}).get(plan_id) or ())
            plans.append(
                _plan_geography(
                    plan,
                    role=role,
                    label=label,
                    route_specs=route_specs,
                    cell_index=cell_index,
                )
            )
    except GeographyV2Error as exc:
        return _unavailable(str(exc))
    if len(plans) < 2:
        return _unavailable("registered_repair_missing")

    points = [
        point
        for plan in plans
        for point in _plan_points(plan)
    ]
    if not points:
        return _unavailable("geographic_coordinates_missing")
    total_gaps = sum(plan["coverage"]["gap_count"] for plan in plans)
    total_required = sum(plan["coverage"]["required_leg_count"] for plan in plans)
    total_validated = sum(
        plan["coverage"]["road_validated_leg_count"] for plan in plans
    )
    total_nodes = sum(plan["coverage"]["route_path_node_count"] for plan in plans)
    status = "ready" if total_gaps == 0 else "ready_with_gaps"
    return {
        "schema_version": "product-geography-v2",
        "status": status,
        "code": (
            "artifact_geography_complete"
            if status == "ready"
            else "artifact_geography_has_route_gaps"
        ),
        "bounds": _bounds(points),
        "route_matrix_id": str(route_matrix.get("matrix_id") or ""),
        "route_source_sha256": str(route_matrix.get("source_content_sha256") or ""),
        "coverage": {
            "status": "complete" if total_gaps == 0 else "gaps_present",
            "plan_count": len(plans),
            "route_path_node_count": total_nodes,
            "required_leg_count": total_required,
            "road_validated_leg_count": total_validated,
            "gap_count": total_gaps,
            "all_itinerary_sequences_accounted": all(
                plan["coverage"]["itinerary_sequence_accounted"] for plan in plans
            ),
        },
        "plans": plans,
        "attribution": {
            "label": "© OpenStreetMap contributors",
            "url": "https://www.openstreetmap.org/copyright",
        },
    }


def _plan_geography(
    plan: Mapping[str, Any],
    *,
    role: str,
    label: str,
    route_specs: tuple[Mapping[str, Any], ...],
    cell_index: Mapping[tuple[str, str], Mapping[str, Any]],
) -> dict[str, Any]:
    plan_id = str(plan.get("plan_id") or "")
    content_hash = str(plan.get("content_hash") or "")
    sequence = tuple(str(value) for value in plan.get("sequence") or () if str(value))
    if not plan_id or not content_hash or len(sequence) < 2:
        raise GeographyV2Error("plan_geography_invalid")
    stop_lookup = {
        str(stop.get("stop_id") or stop.get("poi_id") or ""): stop
        for stop in plan.get("selected_stops") or ()
        if isinstance(stop, Mapping)
    }
    if any(stop_id not in stop_lookup for stop_id in sequence):
        raise GeographyV2Error("plan_stop_missing")
    if not route_specs:
        route_specs = _sequence_route_specs(sequence)

    report = audit_route_coverage(plan, route_specs, cell_index)
    ownership = _ownership(plan)
    stops = _stop_features(
        plan_id,
        content_hash,
        role,
        sequence,
        stop_lookup,
        ownership,
    )
    route_path = _route_path_features(
        plan_id,
        content_hash,
        role,
        route_specs,
        sequence,
        stop_lookup,
        cell_index,
    )
    validated_legs, gaps = _leg_features(
        plan_id,
        content_hash,
        role,
        route_specs,
        report,
        cell_index,
    )
    plan_status = "ready" if report.complete else "ready_with_gaps"
    return {
        "plan_id": plan_id,
        "content_hash": content_hash,
        "role": role,
        "label": label,
        "status": plan_status,
        "coverage": {
            "schema_version": report.schema_version,
            "status": "complete" if report.complete else "gaps_present",
            "route_path_node_count": len(route_path["features"]),
            "required_leg_count": report.required_leg_count,
            "road_validated_leg_count": report.road_validated_leg_count,
            "gap_count": report.gap_count,
            "itinerary_sequence_accounted": report.itinerary_sequence_accounted,
            "complete": report.complete,
        },
        "route_path": route_path,
        "stops": {"type": "FeatureCollection", "features": stops},
        "validated_legs": {
            "type": "FeatureCollection",
            "features": validated_legs,
        },
        "gaps": {"type": "FeatureCollection", "features": gaps},
    }


def _stop_features(
    plan_id: str,
    content_hash: str,
    role: str,
    sequence: tuple[str, ...],
    stop_lookup: Mapping[str, Mapping[str, Any]],
    ownership: Mapping[str, str],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for sequence_index, stop_id in enumerate(sequence, start=1):
        stop = stop_lookup[stop_id]
        coordinate = _stop_coordinate(stop)
        itinerary_role, itinerary_role_source = _itinerary_role_state(stop)
        result.append(
            {
                "type": "Feature",
                "id": f"{plan_id}:stop:{stop_id}",
                "geometry": {"type": "Point", "coordinates": coordinate},
                "properties": {
                    "plan_id": plan_id,
                    "content_hash": content_hash,
                    "role": role,
                    "itinerary_role": itinerary_role,
                    "itinerary_role_source": itinerary_role_source,
                    "stop_id": stop_id,
                    "name": str(
                        stop.get("attraction_name") or stop.get("name") or stop_id
                    ),
                    "city": str(stop.get("city") or ""),
                    "day": _optional_day(stop.get("day")),
                    "stop_order": _optional_int(stop.get("stop_order")) or sequence_index,
                    "sequence_index": sequence_index,
                    "ownership_strength": ownership.get(stop_id, ""),
                },
            }
        )
    return result


def _route_path_features(
    plan_id: str,
    content_hash: str,
    role: str,
    route_specs: tuple[Mapping[str, Any], ...],
    sequence: tuple[str, ...],
    stop_lookup: Mapping[str, Mapping[str, Any]],
    cell_index: Mapping[tuple[str, str], Mapping[str, Any]],
) -> dict[str, Any]:
    node_ids = [str(route_specs[0].get("origin_id") or "")]
    node_ids.extend(str(spec.get("destination_id") or "") for spec in route_specs)
    selected_indices = {stop_id: index for index, stop_id in enumerate(sequence, start=1)}
    features: list[dict[str, Any]] = []
    for occurrence_index, node_id in enumerate(node_ids):
        previous_spec = route_specs[occurrence_index - 1] if occurrence_index else None
        next_spec = route_specs[occurrence_index] if occurrence_index < len(route_specs) else None
        coordinate, source = _path_node_coordinate(
            node_id,
            previous_spec,
            next_spec,
            stop_lookup,
            cell_index,
        )
        arrival_day = _optional_day(previous_spec.get("day")) if previous_spec else None
        departure_day = _optional_day(next_spec.get("day")) if next_spec else None
        features.append(
            {
                "type": "Feature",
                "id": f"{plan_id}:route-path-node:{occurrence_index}",
                "geometry": (
                    {"type": "Point", "coordinates": coordinate}
                    if coordinate is not None
                    else None
                ),
                "properties": {
                    "plan_id": plan_id,
                    "content_hash": content_hash,
                    "role": role,
                    "occurrence_index": occurrence_index,
                    "node_id": node_id,
                    "selected_stop": node_id in selected_indices,
                    "selected_sequence_index": selected_indices.get(node_id),
                    "route_anchor": node_id not in selected_indices,
                    "arrival_day": arrival_day,
                    "departure_day": departure_day,
                    "coordinate_source": source,
                },
            }
        )
    return {"type": "FeatureCollection", "features": features}


def _leg_features(
    plan_id: str,
    content_hash: str,
    role: str,
    route_specs: tuple[Mapping[str, Any], ...],
    report: RouteCoverageReportV1,
    cell_index: Mapping[tuple[str, str], Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    validated: list[dict[str, Any]] = []
    gaps: list[dict[str, Any]] = []
    for leg_index, (spec, coverage) in enumerate(
        zip(route_specs, report.legs, strict=True),
        start=1,
    ):
        shared = {
            "plan_id": plan_id,
            "content_hash": content_hash,
            "role": role,
            "leg_index": leg_index,
            "requirement_id": coverage.requirement_id,
            "day": spec.get("day"),
            "from_day": coverage.from_day,
            "to_day": coverage.to_day,
            "cross_day": coverage.cross_day,
            "travel_mode": coverage.travel_mode,
            "evidence_scope": coverage.evidence_scope,
            "origin_id": coverage.origin_id,
            "destination_id": coverage.destination_id,
            "validation_status": coverage.validation_status,
            "route_leg_id": coverage.route_leg_id,
        }
        if coverage.validation_status != "road_validated":
            gaps.append(
                {
                    "type": "Feature",
                    "id": f"{plan_id}:gap:{leg_index}",
                    "geometry": None,
                    "properties": {**shared, "failure_code": coverage.failure_code},
                }
            )
            continue
        cell = cell_index[(coverage.origin_id, coverage.destination_id)]
        validated.append(
            {
                "type": "Feature",
                "id": f"{plan_id}:leg:{leg_index}",
                "geometry": {
                    "type": "LineString",
                    "coordinates": [_latlon_to_geojson(point) for point in cell["geometry"]],
                },
                "properties": {
                    **shared,
                    "road_validated": True,
                    "fallback_used": False,
                    "geometry_source": str(cell.get("geometry_source") or ""),
                    "distance_m": cell.get("distance_m"),
                    "duration_s": cell.get("duration_s"),
                },
            }
        )
    if not report.itinerary_sequence_accounted:
        gaps.append(
            {
                "type": "Feature",
                "id": f"{plan_id}:gap:selected-stop-sequence",
                "geometry": None,
                "properties": {
                    "plan_id": plan_id,
                    "content_hash": content_hash,
                    "role": role,
                    "validation_status": "unvalidated_gap",
                    "failure_code": "selected_stop_sequence_unaccounted",
                },
            }
        )
    return validated, gaps


def _path_node_coordinate(
    node_id: str,
    previous_spec: Mapping[str, Any] | None,
    next_spec: Mapping[str, Any] | None,
    stop_lookup: Mapping[str, Mapping[str, Any]],
    cell_index: Mapping[tuple[str, str], Mapping[str, Any]],
) -> tuple[list[float] | None, str]:
    if previous_spec is not None:
        cell = _cell_for_spec(previous_spec, cell_index)
        geometry = cell.get("geometry") if cell else None
        if _valid_route_geometry(geometry):
            return _latlon_to_geojson(geometry[-1]), "preceding_route_leg"
    if next_spec is not None:
        cell = _cell_for_spec(next_spec, cell_index)
        geometry = cell.get("geometry") if cell else None
        if _valid_route_geometry(geometry):
            return _latlon_to_geojson(geometry[0]), "following_route_leg"
    stop = stop_lookup.get(node_id)
    if stop is not None:
        return _stop_coordinate(stop), "selected_stop"
    return None, "unavailable"


def _cell_for_spec(
    spec: Mapping[str, Any],
    cell_index: Mapping[tuple[str, str], Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    return cell_index.get(
        (str(spec.get("origin_id") or ""), str(spec.get("destination_id") or ""))
    )


def _sequence_route_specs(
    sequence: tuple[str, ...],
) -> tuple[Mapping[str, Any], ...]:
    return tuple(
        {
            "day": None,
            "origin_id": origin_id,
            "destination_id": destination_id,
            "evidence_scope": "global_plan_sequence_overview",
        }
        for origin_id, destination_id in zip(sequence, sequence[1:], strict=False)
    )


def _ownership(plan: Mapping[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for record in plan.get("owned_constraints") or ():
        if not isinstance(record, Mapping):
            continue
        target = str(record.get("target_id") or record.get("stop_id") or "")
        strength = str(record.get("strength") or record.get("owner_strength") or "")
        if target and strength:
            result[target] = strength
    return result


def _stop_coordinate(stop: Mapping[str, Any]) -> list[float]:
    return [
        _coordinate(stop.get("longitude"), longitude=True),
        _coordinate(stop.get("latitude"), longitude=False),
    ]


def _latlon_to_geojson(value: Any) -> list[float]:
    if not isinstance(value, list) or len(value) != 2:
        raise GeographyV2Error("route_coordinate_invalid")
    return [
        _coordinate(value[1], longitude=True),
        _coordinate(value[0], longitude=False),
    ]


def _coordinate(value: Any, *, longitude: bool) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GeographyV2Error("coordinate_invalid")
    result = float(value)
    limit = 180.0 if longitude else 90.0
    if not math.isfinite(result) or not -limit <= result <= limit:
        raise GeographyV2Error("coordinate_invalid")
    return result


def _valid_route_geometry(value: Any) -> bool:
    return isinstance(value, list) and len(value) >= 2


def _optional_day(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


def _optional_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _itinerary_role_state(stop: Mapping[str, Any]) -> tuple[str | None, str]:
    role = stop.get("itinerary_role")
    source = stop.get("itinerary_role_source")
    if role is None:
        if source not in {None, "unavailable"}:
            raise GeographyV2Error("itinerary_role_invalid")
        return None, "unavailable"
    if (
        not isinstance(role, str)
        or role not in _ITINERARY_ROLES
        or source != "user_declared_itinerary_role"
    ):
        raise GeographyV2Error("itinerary_role_invalid")
    return role, source


def _plan_points(plan: Mapping[str, Any]) -> list[list[float]]:
    result: list[list[float]] = []
    for collection_name in ("route_path", "stops", "validated_legs"):
        collection = plan.get(collection_name)
        if not isinstance(collection, Mapping):
            continue
        for feature in collection.get("features") or ():
            if not isinstance(feature, Mapping):
                continue
            geometry = feature.get("geometry")
            if isinstance(geometry, Mapping):
                result.extend(_points(geometry.get("coordinates")))
    return result


def _points(value: Any) -> list[list[float]]:
    if (
        isinstance(value, list)
        and len(value) == 2
        and all(
            isinstance(item, (int, float)) and not isinstance(item, bool)
            for item in value
        )
    ):
        return [[float(value[0]), float(value[1])]]
    if isinstance(value, list):
        return [point for item in value for point in _points(item)]
    return []


def _bounds(points: list[list[float]]) -> list[float]:
    return [
        min(point[0] for point in points),
        min(point[1] for point in points),
        max(point[0] for point in points),
        max(point[1] for point in points),
    ]


def _unavailable(code: str) -> dict[str, Any]:
    return {
        "schema_version": "product-geography-v2",
        "status": "unavailable",
        "code": code,
        "bounds": None,
        "route_matrix_id": None,
        "route_source_sha256": None,
        "coverage": {
            "status": "unavailable",
            "plan_count": 0,
            "route_path_node_count": 0,
            "required_leg_count": 0,
            "road_validated_leg_count": 0,
            "gap_count": 0,
            "all_itinerary_sequences_accounted": False,
        },
        "plans": [],
        "attribution": {
            "label": "© OpenStreetMap contributors",
            "url": "https://www.openstreetmap.org/copyright",
        },
    }
