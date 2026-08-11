"""Server-owned, route-aware POI discovery for one accepted-plan slot.

Catalog inclusion is not a recommendation.  A candidate is exposed only when
its exact directed route cells and the accepted baseline cells are all
road-validated and non-fallback.  Independent evaluator evidence may establish
evaluated feasibility, but ranking remains unavailable unless an exact ranking
artifact exists.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ..routing.matrix import RouteMatrix, RouteMatrixCell, RouteMatrixError
from .candidate_burden import (
    CandidateInsertionBurdenV1,
    EvaluatorCandidateEvidenceV1,
    FastFeasibilityPrecheckV1,
    assess_candidate_insertion,
    assess_candidate_replacement,
    select_bounded_candidate_top_k,
)
from .poi_catalog import (
    PlaceEntityV1,
    POICatalogError,
    POISourceV1,
    ProductPOICatalogV1,
    RouteAccessPointV1,
    load_product_poi_catalog,
)

SCHEMA_VERSION = "product-poi-candidates-v1"
DEFAULT_LIMIT = 5
MAX_LIMIT = 10
DEFAULT_MAX_DETOUR_MINUTES = 60.0
MAX_MAX_DETOUR_MINUTES = 480.0
NEARBY_RADIUS_M = 50_000.0

ALLOWED_ROLES = frozenset(
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

_TRIP_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_QUERY_HASH = re.compile(r"^[0-9a-f]{16}$")
_MATRIX_ID = re.compile(r"^route_matrix_[0-9a-f]{16}$")
_ROUTE_BUNDLE_ID = re.compile(r"^route_bundle_[0-9a-f]{16}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CONTEXT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_ROUTE_MATRIX_FIELDS = frozenset(
    {
        "schema_version",
        "matrix_id",
        "context_snapshot_id",
        "entity_ids",
        "cells",
        "source_bundle_id",
        "source_content_sha256",
    }
)
_ROUTE_CELL_FIELDS = frozenset(
    {
        "origin_id",
        "destination_id",
        "distance_m",
        "duration_s",
        "route_leg_id",
        "road_validated",
        "fallback_used",
        "provider",
        "query_hash",
        "context_snapshot_id",
        "geometry",
        "routing_profile",
        "routing_status",
        "geometry_source",
        "distance_source",
        "duration_source",
        "fallback_reason",
    }
)
_ACCESS_CONFIDENCE_ORDER = {
    "verified_entrance": 0,
    "provider_access_point": 1,
    "road_snap_only": 2,
    "uncertain": 3,
}


class POICandidateDiscoveryError(ValueError):
    """Stable, path-free candidate discovery failure."""

    def __init__(self, code: str, status_code: int = 409) -> None:
        super().__init__(code)
        self.code = code
        self.status_code = status_code


@dataclass(frozen=True, slots=True)
class CandidateRouteContextV1:
    kind: str
    day: int
    route_leg_id: str
    replacement_target_id: str | None
    predecessor_id: str
    successor_id: str
    baseline_route_leg_ids: tuple[str, ...]
    baseline_travel_minutes: float
    baseline_travel_distance_m: float
    focus_coordinates: tuple[tuple[float, float], ...]


@dataclass(frozen=True, slots=True)
class _RegisteredReplacementMatch:
    public_mapping: Mapping[str, str]
    evaluator_evidence_refs: tuple[str, ...]


def discover_poi_candidates(
    *,
    repository_root: Path,
    session_id: str,
    session_revision: int,
    trip_id: str,
    accepted_plan_id: str,
    day: int,
    route_leg_id: str,
    geography: Mapping[str, Any],
    route_matrix_record: Mapping[str, Any],
    parent_plan: Mapping[str, Any],
    registered_bundles: tuple[Any, ...],
    replacement_target_id: str | None = None,
    role: str | None = None,
    maximum_detour_minutes: float = DEFAULT_MAX_DETOUR_MINUTES,
    limit: int = DEFAULT_LIMIT,
) -> dict[str, Any]:
    """Return a bounded read-only discovery response for one exact route slot."""

    if not _TRIP_ID.fullmatch(trip_id):
        raise POICandidateDiscoveryError("poi_catalog_unavailable")
    if role is not None and role not in ALLOWED_ROLES:
        raise POICandidateDiscoveryError("poi_candidate_role_invalid", 422)
    if (
        isinstance(maximum_detour_minutes, bool)
        or not isinstance(maximum_detour_minutes, (int, float))
        or not math.isfinite(float(maximum_detour_minutes))
        or not 0 <= float(maximum_detour_minutes) <= MAX_MAX_DETOUR_MINUTES
    ):
        raise POICandidateDiscoveryError("poi_candidate_max_detour_invalid", 422)
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= MAX_LIMIT:
        raise POICandidateDiscoveryError("poi_candidate_limit_invalid", 422)

    matrix = _route_matrix(route_matrix_record)
    plan = _accepted_geography_plan(geography, accepted_plan_id)
    context = _route_context(
        plan=plan,
        matrix=matrix,
        day=day,
        route_leg_id=route_leg_id,
        replacement_target_id=replacement_target_id,
    )
    catalog = _catalog(repository_root, trip_id)
    sources = {source.source_id: source for source in catalog.sources}
    precheck = FastFeasibilityPrecheckV1(status="unavailable")

    burden_by_id: dict[str, CandidateInsertionBurdenV1] = {}
    records_by_id: dict[str, dict[str, Any]] = {}
    for place in catalog.search(
        categories=(role,) if role else (),
        require_road_access=True,
        limit=50,
    ):
        if place.place_id in {
            context.predecessor_id,
            context.successor_id,
            context.replacement_target_id,
        }:
            continue
        access = _selected_access_point(place)
        if access is None or not _candidate_cells_are_exact(
            matrix, context, place.place_id, access
        ):
            continue
        candidate_id = place.place_id
        registered = _registered_replacement(
            context=context,
            place=place,
            parent_plan=parent_plan,
            registered_bundles=registered_bundles,
        )
        evaluator = _registered_evaluator(registered)
        assessment_args = {
            "candidate_id": candidate_id,
            "place_id": place.place_id,
            "predecessor_id": context.predecessor_id,
            "successor_id": context.successor_id,
            "route_matrix": matrix,
            "geographic_distance_m": _minimum_geographic_distance(
                place.display_coordinate.longitude,
                place.display_coordinate.latitude,
                context.focus_coordinates,
            ),
            "visit_minutes": (
                float(place.recommended_visit_minutes)
                if place.recommended_visit_minutes is not None
                else None
            ),
            "parking_minutes": None,
            "walking_minutes": None,
            "waiting_minutes": None,
            "nearby_radius_m": NEARBY_RADIUS_M,
            "maximum_detour_minutes": float(maximum_detour_minutes),
            "precheck": precheck,
            "evaluator": evaluator,
        }
        if context.kind == "replacement":
            burden = assess_candidate_replacement(
                replacement_target_id=str(context.replacement_target_id),
                **assessment_args,
            )
        else:
            burden = assess_candidate_insertion(**assessment_args)
        # max_detour_minutes is a discovery filter, not an evaluator rank.
        if not burden.route_near:
            continue
        burden_by_id[candidate_id] = burden
        records_by_id[candidate_id] = _candidate_record(
            place=place,
            access=access,
            sources=sources,
            burden=burden,
            matrix=matrix,
            context=context,
            registered=registered,
        )

    selected = select_bounded_candidate_top_k(tuple(burden_by_id.values()), limit=limit)
    return {
        "schema_version": SCHEMA_VERSION,
        "session_id": session_id,
        "session_revision": session_revision,
        "context": {
            "kind": context.kind,
            "day": context.day,
            "route_leg_id": context.route_leg_id,
            "replacement_target_id": context.replacement_target_id,
            "predecessor_id": context.predecessor_id,
            "successor_id": context.successor_id,
            "baseline_route_leg_ids": list(context.baseline_route_leg_ids),
            "baseline_travel_minutes": context.baseline_travel_minutes,
            "baseline_travel_distance_m": context.baseline_travel_distance_m,
        },
        "catalog": {
            "catalog_id": catalog.catalog_id,
            "catalog_sha256": catalog.manifest_sha256,
            "generated_at": catalog.generated_at,
        },
        "routing": {
            "matrix_id": matrix.matrix_id,
            "context_snapshot_id": matrix.context_snapshot_id,
            "source_bundle_id": matrix.source_bundle_id,
            "source_content_sha256": matrix.source_content_sha256,
            "road_validated_only": True,
            "fallback_allowed": False,
        },
        "candidates": [records_by_id[row.candidate_id] for row in selected],
    }


def _catalog(repository_root: Path, trip_id: str) -> ProductPOICatalogV1:
    manifest = repository_root / "configs" / "product_poi_catalogs" / trip_id / "manifest.json"
    try:
        return load_product_poi_catalog(manifest)
    except POICatalogError as exc:
        if exc.code in {"manifest_file_unreadable", "catalog_file_unreadable"}:
            code = "poi_catalog_unavailable"
        elif exc.code == "catalog_hash_mismatch":
            code = "poi_catalog_hash_mismatch"
        else:
            code = "poi_catalog_invalid"
        raise POICandidateDiscoveryError(code) from exc


def _route_matrix(record: Mapping[str, Any]) -> RouteMatrix:
    matrix_id = record.get("matrix_id")
    context_snapshot_id = record.get("context_snapshot_id")
    source_bundle_id = record.get("source_bundle_id")
    source_content_sha256 = record.get("source_content_sha256")
    if (
        set(record) != _ROUTE_MATRIX_FIELDS
        or record.get("schema_version") != "route-matrix-v1"
        or not isinstance(matrix_id, str)
        or not _MATRIX_ID.fullmatch(matrix_id)
        or not isinstance(context_snapshot_id, str)
        or not _CONTEXT_ID.fullmatch(context_snapshot_id)
        or not isinstance(source_bundle_id, str)
        or not _ROUTE_BUNDLE_ID.fullmatch(source_bundle_id)
        or not isinstance(source_content_sha256, str)
        or not _SHA256.fullmatch(source_content_sha256)
    ):
        raise POICandidateDiscoveryError("poi_route_matrix_invalid")
    raw_cells = record.get("cells")
    entity_ids = record.get("entity_ids")
    if (
        not isinstance(raw_cells, list)
        or not raw_cells
        or len(raw_cells) > 100_000
        or not isinstance(entity_ids, list)
        or not entity_ids
        or len(entity_ids) > 10_000
        or any(not isinstance(value, str) or not _CONTEXT_ID.fullmatch(value) for value in entity_ids)
        or len(set(entity_ids)) != len(entity_ids)
    ):
        raise POICandidateDiscoveryError("poi_route_matrix_invalid")
    try:
        cells: dict[tuple[str, str], RouteMatrixCell] = {}
        seen_pairs: set[tuple[str, str]] = set()
        referenced_entities: set[str] = set()
        for raw in raw_cells:
            if not isinstance(raw, Mapping):
                raise ValueError("route cell must be an object")
            origin_id = raw.get("origin_id")
            destination_id = raw.get("destination_id")
            if not isinstance(origin_id, str) or not isinstance(destination_id, str):
                raise ValueError("route cell identifiers invalid")
            if not _CONTEXT_ID.fullmatch(origin_id) or not _CONTEXT_ID.fullmatch(destination_id):
                raise ValueError("route cell identifiers invalid")
            pair = (origin_id, destination_id)
            if pair in seen_pairs:
                raise ValueError("duplicate route cell")
            seen_pairs.add(pair)
            referenced_entities.update(pair)
            # Malformed individual cells are omitted.  If one belongs to the
            # accepted baseline, route-context resolution fails the endpoint;
            # if it belongs only to a candidate, that candidate is excluded.
            if not _raw_route_cell_is_exact(raw, context_snapshot_id):
                continue
            item = dict(raw)
            item["geometry"] = tuple(tuple(point) for point in item.get("geometry") or ())
            cell = RouteMatrixCell(**item)
            cells[(cell.origin_id, cell.destination_id)] = cell
        if referenced_entities != set(entity_ids):
            raise ValueError("route matrix entity index mismatch")
        matrix = RouteMatrix(
            matrix_id=matrix_id,
            context_snapshot_id=context_snapshot_id,
            entity_ids=tuple(entity_ids),
            cells=cells,
            source_bundle_id=source_bundle_id,
            source_content_sha256=source_content_sha256,
        )
    except (TypeError, ValueError) as exc:
        raise POICandidateDiscoveryError("poi_route_matrix_invalid") from exc
    if not matrix.matrix_id or not matrix.context_snapshot_id or not matrix.cells:
        raise POICandidateDiscoveryError("poi_route_matrix_unavailable")
    return matrix


def _raw_route_cell_is_exact(raw: Mapping[str, Any], context_snapshot_id: str) -> bool:
    distance_m = raw.get("distance_m")
    duration_s = raw.get("duration_s")
    query_hash = raw.get("query_hash")
    geometry = raw.get("geometry")
    if (
        set(raw) != _ROUTE_CELL_FIELDS
        or isinstance(distance_m, bool)
        or not isinstance(distance_m, (int, float))
        or not math.isfinite(float(distance_m))
        or float(distance_m) <= 0
        or isinstance(duration_s, bool)
        or not isinstance(duration_s, (int, float))
        or not math.isfinite(float(duration_s))
        or float(duration_s) <= 0
        or not isinstance(query_hash, str)
        or not _QUERY_HASH.fullmatch(query_hash)
        or raw.get("provider") != "cached_osrm"
        or raw.get("routing_profile") != "driving"
        or raw.get("routing_status") != "osrm_live"
        or raw.get("geometry_source") != "cached_osrm_route_geometry"
        or raw.get("distance_source") != "cached_osrm_route_distance"
        or raw.get("duration_source") != "cached_osrm_route_duration"
        or raw.get("context_snapshot_id") != context_snapshot_id
        or raw.get("road_validated") is not True
        or raw.get("fallback_used") is not False
        or raw.get("fallback_reason") not in {None, ""}
        or not isinstance(raw.get("route_leg_id"), str)
        or not str(raw.get("route_leg_id")).strip()
        or not _valid_route_geometry(geometry)
    ):
        return False
    return True


def _valid_route_geometry(value: Any) -> bool:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return False
    for point in value:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            return False
        latitude, longitude = point
        if (
            isinstance(latitude, bool)
            or not isinstance(latitude, (int, float))
            or not math.isfinite(float(latitude))
            or not -90 <= float(latitude) <= 90
            or isinstance(longitude, bool)
            or not isinstance(longitude, (int, float))
            or not math.isfinite(float(longitude))
            or not -180 <= float(longitude) <= 180
        ):
            return False
    return True


def _accepted_geography_plan(
    geography: Mapping[str, Any], accepted_plan_id: str
) -> Mapping[str, Any]:
    matches = [
        plan
        for plan in geography.get("plans") or ()
        if isinstance(plan, Mapping) and plan.get("plan_id") == accepted_plan_id
    ]
    if len(matches) != 1:
        raise POICandidateDiscoveryError("accepted_route_context_unavailable")
    return matches[0]


def _route_context(
    *,
    plan: Mapping[str, Any],
    matrix: RouteMatrix,
    day: int,
    route_leg_id: str,
    replacement_target_id: str | None,
) -> CandidateRouteContextV1:
    legs = tuple(
        feature
        for feature in ((plan.get("validated_legs") or {}).get("features") or ())
        if isinstance(feature, Mapping)
        and isinstance(feature.get("properties"), Mapping)
        and feature["properties"].get("validation_status") == "road_validated"
    )
    selected = [row for row in legs if row["properties"].get("route_leg_id") == route_leg_id]
    if len(selected) != 1:
        raise POICandidateDiscoveryError("selected_route_segment_not_found", 422)
    selected_properties = selected[0]["properties"]
    if selected_properties.get("day") != day:
        raise POICandidateDiscoveryError("selected_route_segment_day_mismatch", 422)

    if replacement_target_id is None:
        predecessor_id = str(selected_properties.get("origin_id") or "")
        successor_id = str(selected_properties.get("destination_id") or "")
        baseline = (_exact_feature_cell(matrix, selected[0]),)
        focus = _line_coordinates(selected[0])
        return _context_record(
            kind="insertion",
            day=day,
            route_leg_id=route_leg_id,
            replacement_target_id=None,
            predecessor_id=predecessor_id,
            successor_id=successor_id,
            baseline=baseline,
            focus_coordinates=focus,
        )

    path_nodes = tuple(
        feature
        for feature in ((plan.get("route_path") or {}).get("features") or ())
        if isinstance(feature, Mapping)
        and isinstance(feature.get("properties"), Mapping)
        and feature["properties"].get("node_id") == replacement_target_id
        and feature["properties"].get("selected_stop") is True
        and feature["properties"].get("arrival_day") == day
        and feature["properties"].get("departure_day") == day
    )
    if len(path_nodes) != 1:
        raise POICandidateDiscoveryError("replacement_target_not_found", 422)
    incoming = [
        row
        for row in legs
        if row["properties"].get("destination_id") == replacement_target_id
        and row["properties"].get("day") == day
    ]
    outgoing = [
        row
        for row in legs
        if row["properties"].get("origin_id") == replacement_target_id
        and row["properties"].get("day") == day
    ]
    if len(incoming) != 1 or len(outgoing) != 1:
        raise POICandidateDiscoveryError("replacement_target_route_context_unavailable")
    baseline_features = (incoming[0], outgoing[0])
    baseline_ids = {row["properties"].get("route_leg_id") for row in baseline_features}
    if route_leg_id not in baseline_ids:
        raise POICandidateDiscoveryError("replacement_route_segment_mismatch", 422)
    baseline = tuple(_exact_feature_cell(matrix, row) for row in baseline_features)
    target_geometry = path_nodes[0].get("geometry") or {}
    target_coordinate = target_geometry.get("coordinates")
    focus = (
        ((float(target_coordinate[0]), float(target_coordinate[1])),)
        if isinstance(target_coordinate, list)
        and len(target_coordinate) == 2
        and all(isinstance(value, (int, float)) for value in target_coordinate)
        else _line_coordinates(selected[0])
    )
    return _context_record(
        kind="replacement",
        day=day,
        route_leg_id=route_leg_id,
        replacement_target_id=replacement_target_id,
        predecessor_id=str(incoming[0]["properties"].get("origin_id") or ""),
        successor_id=str(outgoing[0]["properties"].get("destination_id") or ""),
        baseline=baseline,
        focus_coordinates=focus,
    )


def _context_record(
    *,
    kind: str,
    day: int,
    route_leg_id: str,
    replacement_target_id: str | None,
    predecessor_id: str,
    successor_id: str,
    baseline: tuple[RouteMatrixCell, ...],
    focus_coordinates: tuple[tuple[float, float], ...],
) -> CandidateRouteContextV1:
    if not predecessor_id or not successor_id or not focus_coordinates:
        raise POICandidateDiscoveryError("selected_route_segment_invalid")
    return CandidateRouteContextV1(
        kind=kind,
        day=day,
        route_leg_id=route_leg_id,
        replacement_target_id=replacement_target_id,
        predecessor_id=predecessor_id,
        successor_id=successor_id,
        baseline_route_leg_ids=tuple(cell.route_leg_id for cell in baseline),
        baseline_travel_minutes=sum(cell.require_duration_s() for cell in baseline) / 60.0,
        baseline_travel_distance_m=sum(cell.require_distance_m() for cell in baseline),
        focus_coordinates=focus_coordinates,
    )


def _exact_feature_cell(matrix: RouteMatrix, feature: Mapping[str, Any]) -> RouteMatrixCell:
    properties = feature["properties"]
    try:
        cell = matrix.cell(
            str(properties.get("origin_id") or ""),
            str(properties.get("destination_id") or ""),
        )
        cell.require_publication_eligible()
    except (RouteMatrixError, ValueError) as exc:
        raise POICandidateDiscoveryError("accepted_route_baseline_unavailable") from exc
    if cell.route_leg_id != properties.get("route_leg_id"):
        raise POICandidateDiscoveryError("accepted_route_evidence_mismatch")
    return cell


def _candidate_cells_are_exact(
    matrix: RouteMatrix,
    context: CandidateRouteContextV1,
    place_id: str,
    access: RouteAccessPointV1,
) -> bool:
    pairs = (
        (context.predecessor_id, place_id),
        (place_id, context.successor_id),
    )
    try:
        cells: list[RouteMatrixCell] = []
        for origin_id, destination_id in pairs:
            cell = matrix.cell(origin_id, destination_id)
            cell.require_publication_eligible()
            if (
                not _QUERY_HASH.fullmatch(cell.query_hash)
                or cell.provider != "cached_osrm"
                or cell.routing_status != "osrm_live"
                or cell.geometry_source != "cached_osrm_route_geometry"
                or cell.distance_source != "cached_osrm_route_distance"
                or cell.duration_source != "cached_osrm_route_duration"
                or cell.context_snapshot_id != matrix.context_snapshot_id
                or len(cell.geometry) < 2
            ):
                return False
            cells.append(cell)
    except (RouteMatrixError, ValueError):
        return False
    return (
        access.source_ref == matrix.matrix_id
        and len(access.evidence_refs) == 2
        and set(access.evidence_refs) == {cell.query_hash for cell in cells}
    )


def _selected_access_point(place: PlaceEntityV1) -> RouteAccessPointV1 | None:
    eligible = [point for point in place.access_points if point.road_validated]
    return min(
        eligible,
        key=lambda point: (
            _ACCESS_CONFIDENCE_ORDER.get(point.access_confidence, 99),
            point.access_point_id.casefold(),
        ),
        default=None,
    )


def _registered_replacement(
    *,
    context: CandidateRouteContextV1,
    place: PlaceEntityV1,
    parent_plan: Mapping[str, Any],
    registered_bundles: tuple[Any, ...],
) -> _RegisteredReplacementMatch | None:
    if context.kind != "replacement":
        return None
    matches: list[_RegisteredReplacementMatch] = []
    for bundle in registered_bundles:
        child = getattr(bundle, "child_plan", None)
        diff = getattr(bundle, "diff", None)
        certificate = getattr(bundle, "certificate", None)
        if not isinstance(child, Mapping) or not isinstance(diff, Mapping) or not isinstance(
            certificate, Mapping
        ):
            continue
        added = diff.get("added_stops") or ()
        deleted = diff.get("deleted_stops") or ()
        if len(added) != 1 or len(deleted) != 1:
            continue
        added_stop, deleted_stop = added[0], deleted[0]
        if (
            not isinstance(added_stop, Mapping)
            or not isinstance(deleted_stop, Mapping)
            or deleted_stop.get("stop_id") != context.replacement_target_id
            or deleted_stop.get("day") != context.day
            or added_stop.get("stop_id") != place.place_id
            or added_stop.get("day") != context.day
            or diff.get("parent_plan_id") != parent_plan.get("plan_id")
            or diff.get("child_plan_id") != child.get("plan_id")
            or certificate.get("plan_id") != child.get("plan_id")
            or certificate.get("plan_content_hash") != child.get("content_hash")
            or certificate.get("eligible") is not True
            or certificate.get("comparison_eligibility") != "eligible"
            or certificate.get("hard_feasibility_status") != "PASSED"
            or certificate.get("evaluation_status")
            not in {"PASSED", "PASSED_WITH_WARNINGS"}
        ):
            continue
        refs = tuple(
            str(value)
            for value in (
                child.get("plan_id"),
                diff.get("diff_id"),
                certificate.get("certificate_id"),
            )
            if value
        )
        matches.append(
            _RegisteredReplacementMatch(
                public_mapping={
                    "draft_type": "replace_nearby",
                    "target_stop_id": str(context.replacement_target_id),
                    "candidate_id": place.place_id,
                },
                evaluator_evidence_refs=refs,
            )
        )
    if len(matches) > 1:
        raise POICandidateDiscoveryError("registered_replacement_ambiguous")
    return matches[0] if matches else None


def _registered_evaluator(
    registered: _RegisteredReplacementMatch | None,
) -> EvaluatorCandidateEvidenceV1 | None:
    if registered is None:
        return None
    return EvaluatorCandidateEvidenceV1(
        owner="independent_evaluator",
        decision_eligible=True,
        ranking_eligible=False,
        evaluator_rank=None,
        recommended=False,
        evidence_refs=registered.evaluator_evidence_refs,
    )


def _candidate_record(
    *,
    place: PlaceEntityV1,
    access: RouteAccessPointV1,
    sources: Mapping[str, POISourceV1],
    burden: CandidateInsertionBurdenV1,
    matrix: RouteMatrix,
    context: CandidateRouteContextV1,
    registered: _RegisteredReplacementMatch | None,
) -> dict[str, Any]:
    route_pairs = [
        ("predecessor_candidate", context.predecessor_id, place.place_id),
        ("candidate_successor", place.place_id, context.successor_id),
    ]
    if context.kind == "insertion":
        route_pairs.append(
            ("predecessor_successor", context.predecessor_id, context.successor_id)
        )
    else:
        route_pairs.extend(
            (
                ("predecessor_target", context.predecessor_id, str(context.replacement_target_id)),
                ("target_successor", str(context.replacement_target_id), context.successor_id),
            )
        )
    expected_roles = (
        {
            "predecessor_candidate",
            "candidate_successor",
            "predecessor_successor",
        }
        if context.kind == "insertion"
        else {
            "predecessor_candidate",
            "candidate_successor",
            "predecessor_target",
            "target_successor",
        }
    )
    roles = [label for label, _, _ in route_pairs]
    if len(roles) != len(expected_roles) or set(roles) != expected_roles:
        raise POICandidateDiscoveryError("candidate_route_evidence_invalid")
    return {
        "candidate_id": burden.candidate_id,
        "place": {
            "place_id": place.place_id,
            "name": place.name,
            "place_categories": list(place.place_categories),
            "display_coordinate": asdict(place.display_coordinate),
            "description": place.description,
            "official_url": place.official_url,
            "informational_urls": list(place.informational_urls),
            "source_refs": list(place.source_refs),
            "source_freshness": place.source_freshness,
            "opening_hours_evidence_ref": place.opening_hours_evidence_ref,
            "recommended_visit_minutes": place.recommended_visit_minutes,
            "weather_suitability": place.weather_suitability,
        },
        "selected_access_point": {
            "access_point_id": access.access_point_id,
            "access_type": access.access_type,
            "coordinate": asdict(access.coordinate),
            "source_ref": access.source_ref,
            "road_validated": access.road_validated,
            "access_confidence": access.access_confidence,
            "evidence_refs": list(access.evidence_refs),
        },
        "sources": [
            {
                "source_id": source.source_id,
                "source_type": source.source_type,
                "source_url": source.source_url,
                "retrieved_at": source.retrieved_at,
            }
            for source_id in place.source_refs
            if (source := sources.get(source_id)) is not None
        ],
        "burden": asdict(burden),
        "precheck": {
            "predicted_arrival": None,
            "open_at_arrival": None,
            "status": "unavailable",
            "evidence_refs": [],
        },
        "route_evidence_refs": [
            {
                "role": label,
                "route_leg_id": matrix.cell(origin, destination).route_leg_id,
                "query_hash": matrix.cell(origin, destination).query_hash,
            }
            for label, origin, destination in route_pairs
        ],
        "registered_replacement": (
            dict(registered.public_mapping) if registered is not None else None
        ),
    }


def _line_coordinates(feature: Mapping[str, Any]) -> tuple[tuple[float, float], ...]:
    geometry = feature.get("geometry") or {}
    coordinates = geometry.get("coordinates")
    if not isinstance(coordinates, list):
        return ()
    return tuple(
        (float(point[0]), float(point[1]))
        for point in coordinates
        if isinstance(point, list)
        and len(point) == 2
        and all(isinstance(value, (int, float)) for value in point)
    )


def _minimum_geographic_distance(
    longitude: float,
    latitude: float,
    focus_coordinates: tuple[tuple[float, float], ...],
) -> float | None:
    if not focus_coordinates:
        return None
    return min(
        _haversine_m(latitude, longitude, focus_latitude, focus_longitude)
        for focus_longitude, focus_latitude in focus_coordinates
    )


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_m = 6_371_008.8
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = (
        math.sin(delta_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    )
    return radius_m * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


__all__ = [
    "ALLOWED_ROLES",
    "DEFAULT_LIMIT",
    "DEFAULT_MAX_DETOUR_MINUTES",
    "MAX_LIMIT",
    "MAX_MAX_DETOUR_MINUTES",
    "POICandidateDiscoveryError",
    "SCHEMA_VERSION",
    "discover_poi_candidates",
]
