"""Session-local route overlay with explicit immutable/runtime/gap provenance."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from typing import Any

from itinerary_system.routing.models import RouteLegResult

from .route_coverage import audit_route_coverage


@dataclass(frozen=True)
class SessionRouteLegV1:
    requirement_id: str
    origin_id: str
    destination_id: str
    day: int | None
    evidence_scope: str
    validation_status: str
    evidence_source: str
    route_leg_id: str | None
    query_hash: str | None
    distance_m: float | None
    duration_s: float | None
    geometry: tuple[tuple[float, float], ...] | None
    failure_code: str | None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SessionRouteOverlayV1:
    plan_id: str
    context_snapshot_id: str
    required_leg_count: int
    road_validated_leg_count: int
    gap_count: int
    itinerary_sequence_accounted: bool
    complete: bool
    acceptance_eligible: bool
    failure_codes: tuple[str, ...]
    legs: tuple[SessionRouteLegV1, ...]
    schema_version: str = "session-route-overlay-v1"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_session_route_overlay(
    plan: Mapping[str, Any],
    route_specs: Iterable[Mapping[str, Any]],
    base_cells: Mapping[tuple[str, str], Mapping[str, Any]],
    *,
    runtime_legs: Mapping[tuple[str, str], RouteLegResult] | None = None,
    context_snapshot_id: str,
) -> SessionRouteOverlayV1:
    """Merge immutable base evidence with validated session-local replacements."""

    specs = tuple(route_specs)
    runtime = runtime_legs or {}
    coverage_cells = dict(base_cells)
    for key, leg in runtime.items():
        coverage_cells[key] = _runtime_cell(leg)
    coverage = audit_route_coverage(plan, specs, coverage_cells)
    legs: list[SessionRouteLegV1] = []
    failure_codes: list[str] = []

    for index, (spec, status) in enumerate(zip(specs, coverage.legs, strict=True), start=1):
        key = (status.origin_id, status.destination_id)
        runtime_leg = runtime.get(key)
        if runtime_leg is not None:
            record = _runtime_overlay_leg(index, spec, runtime_leg, status.failure_code)
        else:
            record = _base_overlay_leg(index, spec, base_cells.get(key), status.failure_code)
        legs.append(record)
        if record.failure_code:
            failure_codes.append(record.failure_code)

    if not coverage.itinerary_sequence_accounted:
        failure_codes.append("itinerary_sequence_not_accounted")
    unique_failures = tuple(dict.fromkeys(failure_codes))
    return SessionRouteOverlayV1(
        plan_id=str(plan.get("plan_id") or ""),
        context_snapshot_id=str(context_snapshot_id),
        required_leg_count=coverage.required_leg_count,
        road_validated_leg_count=coverage.road_validated_leg_count,
        gap_count=coverage.gap_count,
        itinerary_sequence_accounted=coverage.itinerary_sequence_accounted,
        complete=coverage.complete,
        acceptance_eligible=coverage.complete,
        failure_codes=unique_failures,
        legs=tuple(legs),
    )


def _runtime_cell(leg: RouteLegResult) -> dict[str, Any]:
    return {
        "route_leg_id": f"runtime_{leg.query_hash}",
        "road_validated": leg.road_validated,
        "fallback_used": leg.fallback_used,
        "geometry": [list(point) for point in leg.geometry],
    }


def _runtime_overlay_leg(
    index: int,
    spec: Mapping[str, Any],
    leg: RouteLegResult,
    failure_code: str | None,
) -> SessionRouteLegV1:
    return SessionRouteLegV1(
        requirement_id=f"required-leg:{index}",
        origin_id=str(spec.get("origin_id") or ""),
        destination_id=str(spec.get("destination_id") or ""),
        day=_day(spec.get("day")),
        evidence_scope=str(spec.get("evidence_scope") or ""),
        validation_status="road_validated" if failure_code is None else "unvalidated_gap",
        evidence_source="session_runtime",
        route_leg_id=f"runtime_{leg.query_hash}" if failure_code is None else None,
        query_hash=leg.query_hash if failure_code is None else None,
        distance_m=float(leg.distance_m) if failure_code is None else None,
        duration_s=float(leg.duration_s) if failure_code is None else None,
        geometry=tuple(leg.geometry) if failure_code is None else None,
        failure_code=failure_code,
    )


def _base_overlay_leg(
    index: int,
    spec: Mapping[str, Any],
    cell: Mapping[str, Any] | None,
    failure_code: str | None,
) -> SessionRouteLegV1:
    geometry = None
    if failure_code is None and cell is not None:
        geometry = tuple(tuple(float(value) for value in point) for point in cell["geometry"])
    return SessionRouteLegV1(
        requirement_id=f"required-leg:{index}",
        origin_id=str(spec.get("origin_id") or ""),
        destination_id=str(spec.get("destination_id") or ""),
        day=_day(spec.get("day")),
        evidence_scope=str(spec.get("evidence_scope") or ""),
        validation_status="road_validated" if failure_code is None else "unvalidated_gap",
        evidence_source="immutable_base" if failure_code is None else "gap",
        route_leg_id=(str(cell.get("route_leg_id") or "") or None)
        if failure_code is None and cell is not None
        else None,
        query_hash=(str(cell.get("query_hash") or "") or None)
        if failure_code is None and cell is not None
        else None,
        distance_m=_number(cell.get("distance_m")) if failure_code is None and cell else None,
        duration_s=_number(cell.get("duration_s")) if failure_code is None and cell else None,
        geometry=geometry,
        failure_code=failure_code,
    )


def _day(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


def _number(value: Any) -> float | None:
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else None
