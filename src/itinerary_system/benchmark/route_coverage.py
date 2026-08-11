"""Preflight route coverage for publication benchmark search universes."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from ..research_artifacts import PlanArtifactV2
from ..routing import RouteMatrix, RouteMatrixCellMissing, route_anchor_key


@dataclass(frozen=True)
class BenchmarkRouteCoverageReport:
    """Coverage evidence for every directed pair reachable by the benchmark."""

    matrix_id: str
    scenario_count: int
    entity_count: int
    required_pair_count: int
    present_pair_count: int
    road_validated_pair_count: int
    missing_pairs: tuple[tuple[str, str], ...] = ()
    ineligible_pairs: tuple[tuple[str, str, str], ...] = ()
    unlocated_entity_ids: tuple[str, ...] = ()
    missing_anchor_days: tuple[int, ...] = ()
    scenario_required_pair_counts: tuple[tuple[str, int], ...] = ()
    publication_ready: bool = False
    schema_version: str = "benchmark-route-coverage-report-v1"

    def to_record(self) -> dict[str, Any]:
        return {
            "matrix_id": self.matrix_id,
            "scenario_count": self.scenario_count,
            "entity_count": self.entity_count,
            "required_pair_count": self.required_pair_count,
            "present_pair_count": self.present_pair_count,
            "road_validated_pair_count": self.road_validated_pair_count,
            "missing_pair_count": len(self.missing_pairs),
            "ineligible_pair_count": len(self.ineligible_pairs),
            "missing_pairs": [
                {"origin_id": origin, "destination_id": destination}
                for origin, destination in self.missing_pairs
            ],
            "ineligible_pairs": [
                {"origin_id": origin, "destination_id": destination, "reason": reason}
                for origin, destination, reason in self.ineligible_pairs
            ],
            "unlocated_entity_ids": list(self.unlocated_entity_ids),
            "missing_anchor_days": list(self.missing_anchor_days),
            "scenario_required_pair_counts": {
                scenario_id: count for scenario_id, count in self.scenario_required_pair_counts
            },
            "publication_ready": self.publication_ready,
            "schema_version": self.schema_version,
        }


def build_benchmark_route_coverage(
    *,
    parent_plan: PlanArtifactV2,
    scenarios: Iterable[Any],
    route_matrix: RouteMatrix,
    start_anchor_by_day: Mapping[int, str],
    end_anchor_by_day: Mapping[int, str],
    entity_coordinates: Mapping[str, tuple[float, float]] | None = None,
) -> BenchmarkRouteCoverageReport:
    """Audit the conservative full-reoptimization route universe before execution."""

    scenario_tuple = tuple(scenarios)
    locations = _locations(parent_plan, scenario_tuple, entity_coordinates or {})
    parent_ids = _record_ids(parent_plan.selected_stops)
    anchor_days = tuple(sorted(set(start_anchor_by_day) | set(end_anchor_by_day)))
    missing_anchor_days = tuple(
        day
        for day in anchor_days
        if not route_anchor_key(start_anchor_by_day.get(day)) or not route_anchor_key(end_anchor_by_day.get(day))
    )
    required_pairs: set[tuple[str, str]] = set()
    scenario_counts: list[tuple[str, int]] = []
    all_entities = set(parent_ids)
    for index, scenario in enumerate(scenario_tuple, start=1):
        scenario_id = str(getattr(scenario, "scenario_id", "") or f"scenario_{index}")
        request = getattr(scenario, "request", None)
        candidate_records = tuple(getattr(request, "candidate_pois", ()) or ())
        entity_ids = tuple(dict.fromkeys((*parent_ids, *_record_ids(candidate_records))))
        all_entities.update(entity_ids)
        scenario_pairs: set[tuple[str, str]] = set()
        for day in anchor_days:
            start = route_anchor_key(start_anchor_by_day.get(day))
            end = route_anchor_key(end_anchor_by_day.get(day))
            if not start or not end:
                continue
            _add_pair(scenario_pairs, start, end)
            for entity_id in entity_ids:
                _add_pair(scenario_pairs, start, entity_id)
                _add_pair(scenario_pairs, entity_id, end)
            for origin_id in entity_ids:
                for destination_id in entity_ids:
                    _add_pair(scenario_pairs, origin_id, destination_id)
        required_pairs.update(scenario_pairs)
        scenario_counts.append((scenario_id, len(scenario_pairs)))

    missing: list[tuple[str, str]] = []
    ineligible: list[tuple[str, str, str]] = []
    road_validated = 0
    for origin_id, destination_id in sorted(required_pairs):
        try:
            cell = route_matrix.cell(origin_id, destination_id)
        except RouteMatrixCellMissing:
            missing.append((origin_id, destination_id))
            continue
        try:
            cell.require_publication_eligible()
        except Exception as exc:
            ineligible.append((origin_id, destination_id, f"{type(exc).__name__}:{exc}"))
            continue
        road_validated += 1

    missing_entities = {entity_id for pair in missing for entity_id in pair}
    unlocated = tuple(sorted(entity_id for entity_id in missing_entities if entity_id not in locations))
    publication_ready = not missing and not ineligible and not unlocated and not missing_anchor_days
    return BenchmarkRouteCoverageReport(
        matrix_id=route_matrix.matrix_id,
        scenario_count=len(scenario_tuple),
        entity_count=len(all_entities),
        required_pair_count=len(required_pairs),
        present_pair_count=len(required_pairs) - len(missing),
        road_validated_pair_count=road_validated,
        missing_pairs=tuple(missing),
        ineligible_pairs=tuple(ineligible),
        unlocated_entity_ids=unlocated,
        missing_anchor_days=missing_anchor_days,
        scenario_required_pair_counts=tuple(scenario_counts),
        publication_ready=publication_ready,
    )


def _add_pair(pairs: set[tuple[str, str]], origin_id: str, destination_id: str) -> None:
    origin = route_anchor_key(origin_id)
    destination = route_anchor_key(destination_id)
    if origin and destination and origin != destination:
        pairs.add((origin, destination))


def _record_ids(records: Iterable[Mapping[str, Any]]) -> tuple[str, ...]:
    ids: list[str] = []
    for record in records:
        entity_id = route_anchor_key(
            record.get("stop_id")
            or record.get("poi_id")
            or record.get("attraction_id")
            or record.get("attraction_name")
            or record.get("name")
        )
        if entity_id:
            ids.append(entity_id)
    return tuple(dict.fromkeys(ids))


def _locations(
    parent_plan: PlanArtifactV2,
    scenarios: tuple[Any, ...],
    supplied: Mapping[str, tuple[float, float]],
) -> dict[str, tuple[float, float]]:
    locations: dict[str, tuple[float, float]] = {}
    for entity_id, coordinates in supplied.items():
        normalized = route_anchor_key(entity_id)
        if normalized and _valid_coordinates(coordinates):
            locations[normalized] = (float(coordinates[0]), float(coordinates[1]))
    records: list[Mapping[str, Any]] = list(parent_plan.selected_stops)
    for scenario in scenarios:
        request = getattr(scenario, "request", None)
        records.extend(tuple(getattr(request, "candidate_pois", ()) or ()))
    for record in records:
        entity_ids = _record_ids((record,))
        coordinates = (record.get("latitude"), record.get("longitude"))
        if entity_ids and _valid_coordinates(coordinates):
            locations[entity_ids[0]] = (float(coordinates[0]), float(coordinates[1]))
    return locations


def _valid_coordinates(value: Any) -> bool:
    try:
        latitude, longitude = value
        return math.isfinite(float(latitude)) and math.isfinite(float(longitude))
    except (TypeError, ValueError):
        return False
