"""Route-leg coverage accounting for product geography.

The audit is deliberately independent of map rendering. It answers whether
every declared directed leg is backed by road-valid evidence and whether the
declared leg chain accounts for the itinerary stop order. Missing evidence is
reported explicitly and is never replaced with straight-line geometry.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class RouteLegCoverageV1:
    """Evidence status for one required directed itinerary connection."""

    requirement_id: str
    origin_id: str
    destination_id: str
    from_day: int | None
    to_day: int | None
    cross_day: bool
    travel_mode: str
    validation_status: str
    evidence_scope: str
    route_leg_id: str | None
    failure_code: str | None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RouteCoverageReportV1:
    """Plan-level accounting for required route legs and itinerary order."""

    plan_id: str
    required_leg_count: int
    road_validated_leg_count: int
    gap_count: int
    itinerary_sequence_accounted: bool
    complete: bool
    legs: tuple[RouteLegCoverageV1, ...]
    schema_version: str = "route-coverage-v1"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def audit_route_coverage(
    plan: Mapping[str, Any],
    route_specs: Iterable[Mapping[str, Any]] | None,
    cell_index: Mapping[tuple[str, str], Mapping[str, Any]],
) -> RouteCoverageReportV1:
    """Return deterministic coverage for a plan and its required route legs."""

    plan_id = str(plan.get("plan_id") or "")
    sequence = tuple(str(value) for value in plan.get("sequence") or () if str(value))
    stop_days = {
        str(stop.get("stop_id") or stop.get("poi_id") or ""): _optional_day(stop.get("day"))
        for stop in plan.get("selected_stops") or ()
        if isinstance(stop, Mapping)
    }
    declared = tuple(route_specs or _sequence_specs(sequence))
    legs: list[RouteLegCoverageV1] = []
    path_nodes: list[str] = []

    for index, spec in enumerate(declared, start=1):
        origin_id = str(spec.get("origin_id") or "")
        destination_id = str(spec.get("destination_id") or "")
        day = _optional_day(spec.get("day"))
        from_day = stop_days.get(origin_id, day)
        to_day = stop_days.get(destination_id, day)
        failure_code: str | None = None
        cell = cell_index.get((origin_id, destination_id))
        if not origin_id or not destination_id:
            failure_code = "route_requirement_invalid"
        elif cell is None:
            failure_code = "route_leg_missing"
        elif not bool(cell.get("road_validated")) or bool(cell.get("fallback_used")):
            failure_code = "route_leg_not_road_validated"
        elif not _valid_geometry(cell.get("geometry")):
            failure_code = "route_geometry_missing"

        route_leg_id = None if cell is None else str(cell.get("route_leg_id") or "") or None
        legs.append(
            RouteLegCoverageV1(
                requirement_id=f"{plan_id}:required-leg:{index}",
                origin_id=origin_id,
                destination_id=destination_id,
                from_day=from_day,
                to_day=to_day,
                cross_day=from_day is not None and to_day is not None and from_day != to_day,
                travel_mode=str(spec.get("travel_mode") or "driving"),
                validation_status="road_validated" if failure_code is None else "unvalidated_gap",
                evidence_scope=str(spec.get("evidence_scope") or ""),
                route_leg_id=route_leg_id,
                failure_code=failure_code,
            )
        )
        if index == 1:
            path_nodes.append(origin_id)
        path_nodes.append(destination_id)

    accounted = _itinerary_sequence_accounted(sequence, tuple(path_nodes))
    gap_count = sum(leg.validation_status != "road_validated" for leg in legs)
    if not accounted and sequence:
        gap_count += 1
    validated_count = sum(leg.validation_status == "road_validated" for leg in legs)
    return RouteCoverageReportV1(
        plan_id=plan_id,
        required_leg_count=len(legs),
        road_validated_leg_count=validated_count,
        gap_count=gap_count,
        itinerary_sequence_accounted=accounted,
        complete=bool(legs) and gap_count == 0,
        legs=tuple(legs),
    )


def _sequence_specs(sequence: tuple[str, ...]) -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "day": None,
            "origin_id": origin_id,
            "destination_id": destination_id,
            "evidence_scope": "global_plan_sequence_overview",
        }
        for origin_id, destination_id in zip(sequence, sequence[1:], strict=False)
    )


def _itinerary_sequence_accounted(
    itinerary_sequence: tuple[str, ...],
    path_nodes: tuple[str, ...],
) -> bool:
    if not itinerary_sequence:
        return False
    itinerary_ids = set(itinerary_sequence)
    encountered = tuple(node for node in path_nodes if node in itinerary_ids)
    return encountered == itinerary_sequence


def _optional_day(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


def _valid_geometry(value: Any) -> bool:
    return isinstance(value, list) and len(value) >= 2
