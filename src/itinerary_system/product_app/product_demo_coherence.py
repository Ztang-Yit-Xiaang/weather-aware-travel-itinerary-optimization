"""Fail-closed semantic validation for the coherent California product demo."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


class ProductDemoCoherenceError(ValueError):
    """Raised when product-facing trip artifacts disagree semantically."""


APPROVED_DAY_LABELS = (
    "San Francisco",
    "Stanford & Silicon Valley",
    "Yosemite National Park",
    "Sequoia & Kings Canyon",
    "Central Coast",
    "Santa Barbara",
    "Los Angeles",
)


def validate_product_demo_coherence(
    *,
    manifest: Mapping[str, Any],
    parent_plan: Mapping[str, Any],
    alternatives: Mapping[str, tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]],
    route_matrix: Mapping[str, Any],
) -> None:
    """Validate day, route, repair, metric, and explanation identity end to end."""

    scenario = manifest.get("scenario")
    if not isinstance(scenario, Mapping) or set(scenario) != {
        "schema_version",
        "scenario_id",
        "direction",
        "affected_stop_id",
        "days",
        "place_sources",
    }:
        _fail("product_demo_scenario_invalid")
    if (
        scenario.get("schema_version") != "product-demo-scenario-v1"
        or scenario.get("scenario_id") != "california_coast_approved_seven_day"
        or scenario.get("direction") != "san_francisco_to_los_angeles"
        or scenario.get("affected_stop_id") != "bixby_creek_bridge_viewpoint"
    ):
        _fail("product_demo_scenario_invalid")

    days = scenario.get("days")
    if not isinstance(days, list) or len(days) != len(APPROVED_DAY_LABELS):
        _fail("product_demo_days_invalid")
    expected_day_keys = {
        "day",
        "label",
        "start_anchor_id",
        "end_anchor_id",
        "parent_stop_ids",
    }
    previous_end: str | None = None
    for day_number, (day, label) in enumerate(zip(days, APPROVED_DAY_LABELS, strict=True), start=1):
        if not isinstance(day, Mapping) or set(day) != expected_day_keys:
            _fail("product_demo_days_invalid")
        start = _identifier(day.get("start_anchor_id"))
        end = _identifier(day.get("end_anchor_id"))
        stop_ids = _identifier_list(day.get("parent_stop_ids"))
        if (
            day.get("day") != day_number
            or day.get("label") != label
            or not start
            or not end
            or not stop_ids
            or (previous_end is not None and start != previous_end)
        ):
            _fail("product_demo_days_invalid")
        previous_end = end

    place_sources = scenario.get("place_sources")
    if not isinstance(place_sources, list) or not place_sources:
        _fail("product_demo_place_sources_invalid")
    expected_place_keys = {
        "entity_id",
        "name",
        "latitude",
        "longitude",
        "source_ref",
    }
    place_ids: set[str] = set()
    for place in place_sources:
        if not isinstance(place, Mapping) or set(place) != expected_place_keys:
            _fail("product_demo_place_sources_invalid")
        entity_id = _identifier(place.get("entity_id"))
        latitude = place.get("latitude")
        longitude = place.get("longitude")
        if (
            not entity_id
            or entity_id in place_ids
            or not str(place.get("name") or "").strip()
            or not _coordinate(latitude, longitude)
            or not str(place.get("source_ref") or "").strip()
        ):
            _fail("product_demo_place_sources_invalid")
        place_ids.add(entity_id)
    required_entities = {
        value
        for day in days
        for value in (
            day["start_anchor_id"],
            *day["parent_stop_ids"],
            day["end_anchor_id"],
        )
    }
    for record in manifest.get("alternatives") or []:
        if isinstance(record, Mapping):
            replacement = record.get("replacement_stop_id")
            if isinstance(replacement, str):
                required_entities.add(replacement)
    if not required_entities <= place_ids:
        _fail("product_demo_place_sources_incomplete")

    parent_sequences = _validate_plan(parent_plan, days=days, role="parent")
    if parent_sequences[4] != ("bixby_creek_bridge_viewpoint",):
        _fail("product_demo_affected_stop_mismatch")

    matrix_cells = _matrix_cells(route_matrix)
    manifest_alternatives = manifest.get("alternatives")
    if not isinstance(manifest_alternatives, list) or [row.get("role") for row in manifest_alternatives] != [
        "recommended",
        "low_driving",
    ]:
        _fail("product_demo_alternatives_invalid")

    roles_seen: set[str] = set()
    for record in manifest_alternatives:
        role = str(record.get("role") or "")
        if role in roles_seen or role not in alternatives:
            _fail("product_demo_alternatives_invalid")
        roles_seen.add(role)
        child, diff, explanation = alternatives[role]
        sequences = _validate_plan(child, days=days, role=role)
        replacement_stop_id = _identifier(record.get("replacement_stop_id"))
        if (
            not replacement_stop_id
            or sequences[4] != (replacement_stop_id,)
            or any(sequences[index] != parent_sequences[index] for index in range(7) if index != 4)
        ):
            _fail("product_demo_repair_scope_invalid")
        _validate_diff(
            diff,
            parent_plan=parent_plan,
            child_plan=child,
            affected_stop_id="bixby_creek_bridge_viewpoint",
            replacement_stop_id=replacement_stop_id,
        )
        _validate_explanation(
            explanation,
            parent_plan=parent_plan,
            child_plan=child,
            affected_stop_id="bixby_creek_bridge_viewpoint",
            replacement_stop_id=replacement_stop_id,
        )
        declared_legs = record.get("route_legs")
        expected_legs = _expected_route_legs(days, sequences)
        if declared_legs != expected_legs:
            _fail("product_demo_route_sequence_mismatch")
        total_minutes = _validate_route_legs(expected_legs, matrix_cells)
        declared_minutes = record.get("route_total_minutes")
        if not _finite_number(declared_minutes) or abs(float(declared_minutes) - total_minutes) > 1e-6:
            _fail("product_demo_route_total_mismatch")

    parent_record = manifest.get("parent")
    if not isinstance(parent_record, Mapping):
        _fail("product_demo_parent_invalid")
    parent_legs = _expected_route_legs(days, parent_sequences, scope="road_validated_daily_route_leg")
    if parent_record.get("route_legs") != parent_legs:
        _fail("product_demo_route_sequence_mismatch")
    _validate_route_legs(parent_legs, matrix_cells)


def _validate_plan(
    plan: Mapping[str, Any],
    *,
    days: Sequence[Mapping[str, Any]],
    role: str,
) -> tuple[tuple[str, ...], ...]:
    selected_stops = plan.get("selected_stops")
    ordered_days = plan.get("ordered_days")
    sequence = plan.get("sequence")
    day_assignments = plan.get("day_assignments")
    if (
        not isinstance(selected_stops, list)
        or not isinstance(ordered_days, list)
        or not isinstance(sequence, list)
        or not isinstance(day_assignments, Mapping)
        or len(ordered_days) != 7
    ):
        _fail("product_demo_plan_shape_invalid")
    stop_index: dict[str, Mapping[str, Any]] = {}
    for stop in selected_stops:
        if not isinstance(stop, Mapping):
            _fail("product_demo_plan_shape_invalid")
        stop_id = _identifier(stop.get("poi_id"))
        day = stop.get("day")
        order = stop.get("stop_order")
        if (
            not stop_id
            or stop_id in stop_index
            or not isinstance(day, int)
            or isinstance(day, bool)
            or day < 1
            or day > 7
            or not isinstance(order, int)
            or isinstance(order, bool)
            or order < 1
            or not _coordinate(stop.get("latitude"), stop.get("longitude"))
            or day_assignments.get(stop_id) != day
        ):
            _fail("product_demo_plan_shape_invalid")
        stop_index[stop_id] = stop

    flattened: list[str] = []
    result: list[tuple[str, ...]] = []
    for day_number, record in enumerate(ordered_days, start=1):
        if not isinstance(record, Mapping) or set(record) != {"day", "stop_ids"}:
            _fail("product_demo_plan_shape_invalid")
        stop_ids = _identifier_list(record.get("stop_ids"))
        if record.get("day") != day_number or not stop_ids:
            _fail("product_demo_plan_shape_invalid")
        expected_order = tuple(
            stop_id
            for stop_id, stop in sorted(
                stop_index.items(), key=lambda item: (int(item[1]["day"]), int(item[1]["stop_order"]))
            )
            if stop["day"] == day_number
        )
        if stop_ids != expected_order:
            _fail("product_demo_plan_order_invalid")
        if role == "parent" and stop_ids != tuple(days[day_number - 1]["parent_stop_ids"]):
            _fail("product_demo_parent_day_mismatch")
        result.append(stop_ids)
        flattened.extend(stop_ids)
    if sequence != flattened or set(flattened) != set(stop_index):
        _fail("product_demo_plan_order_invalid")
    return tuple(result)


def _validate_diff(
    diff: Mapping[str, Any],
    *,
    parent_plan: Mapping[str, Any],
    child_plan: Mapping[str, Any],
    affected_stop_id: str,
    replacement_stop_id: str,
) -> None:
    if (
        diff.get("parent_plan_id") != parent_plan.get("plan_id")
        or diff.get("child_plan_id") != child_plan.get("plan_id")
    ):
        _fail("product_demo_diff_lineage_invalid")
    deleted = diff.get("deleted_stops")
    added = diff.get("added_stops")
    if (
        not isinstance(deleted, list)
        or not isinstance(added, list)
        or [row.get("stop_id") for row in deleted if isinstance(row, Mapping)] != [affected_stop_id]
        or [row.get("stop_id") for row in added if isinstance(row, Mapping)] != [replacement_stop_id]
    ):
        _fail("product_demo_diff_scope_invalid")
    road_changes = diff.get("road_changes")
    if not isinstance(road_changes, list) or not any(
        isinstance(row, Mapping) and row.get("day") == 5 for row in road_changes
    ):
        _fail("product_demo_diff_scope_invalid")


def _validate_explanation(
    explanation: Mapping[str, Any],
    *,
    parent_plan: Mapping[str, Any],
    child_plan: Mapping[str, Any],
    affected_stop_id: str,
    replacement_stop_id: str,
) -> None:
    if (
        explanation.get("parent_plan_id") != parent_plan.get("plan_id")
        or explanation.get("child_plan_id") != child_plan.get("plan_id")
        or explanation.get("affected_stop_id") != affected_stop_id
        or explanation.get("replacement_stop_id") != replacement_stop_id
    ):
        _fail("product_demo_explanation_scope_invalid")
    claims = explanation.get("claims")
    if not isinstance(claims, list) or not claims or any(
        not isinstance(claim, Mapping)
        or claim.get("unsupported") is True
        or not claim.get("evidence_refs")
        for claim in claims
    ):
        _fail("product_demo_explanation_claims_invalid")


def _expected_route_legs(
    days: Sequence[Mapping[str, Any]],
    day_sequences: Sequence[Sequence[str]],
    *,
    scope: str = "certified_daily_route_leg",
) -> list[dict[str, Any]]:
    legs: list[dict[str, Any]] = []
    for day, stop_ids in zip(days, day_sequences, strict=True):
        route_sequence = (
            str(day["start_anchor_id"]),
            *stop_ids,
            str(day["end_anchor_id"]),
        )
        legs.extend(
            {
                "day": day["day"],
                "origin_id": origin,
                "destination_id": destination,
                "evidence_scope": scope,
            }
            for origin, destination in zip(route_sequence[:-1], route_sequence[1:], strict=True)
        )
    return legs


def _matrix_cells(route_matrix: Mapping[str, Any]) -> dict[tuple[str, str], Mapping[str, Any]]:
    if route_matrix.get("schema_version") != "route-matrix-v1":
        _fail("product_demo_route_matrix_invalid")
    raw_cells = route_matrix.get("cells")
    if not isinstance(raw_cells, list) or not raw_cells:
        _fail("product_demo_route_matrix_invalid")
    cells: dict[tuple[str, str], Mapping[str, Any]] = {}
    for cell in raw_cells:
        if not isinstance(cell, Mapping):
            _fail("product_demo_route_matrix_invalid")
        pair = (_identifier(cell.get("origin_id")), _identifier(cell.get("destination_id")))
        if not all(pair) or pair in cells:
            _fail("product_demo_route_matrix_invalid")
        cells[pair] = cell
    return cells


def _validate_route_legs(
    route_legs: Sequence[Mapping[str, Any]],
    cells: Mapping[tuple[str, str], Mapping[str, Any]],
) -> float:
    total_seconds = 0.0
    previous_destination: str | None = None
    for leg in route_legs:
        origin = _identifier(leg.get("origin_id"))
        destination = _identifier(leg.get("destination_id"))
        if previous_destination is not None and previous_destination != origin:
            _fail("product_demo_route_discontinuous")
        cell = cells.get((origin, destination))
        geometry = cell.get("geometry") if isinstance(cell, Mapping) else None
        if (
            not isinstance(cell, Mapping)
            or cell.get("road_validated") is not True
            or cell.get("fallback_used") is not False
            or cell.get("provider") != "cached_osrm"
            or cell.get("routing_status") != "osrm_live"
            or not _finite_number(cell.get("distance_m"))
            or float(cell["distance_m"]) <= 0
            or not _finite_number(cell.get("duration_s"))
            or float(cell["duration_s"]) <= 0
            or not isinstance(geometry, list)
            or len(geometry) < 2
            or any(
                not isinstance(point, list)
                or len(point) != 2
                or not _coordinate(point[0], point[1])
                for point in geometry
            )
        ):
            _fail("product_demo_route_leg_evidence_invalid")
        total_seconds += float(cell["duration_s"])
        previous_destination = destination
    return round(total_seconds / 60.0, 6)


def _identifier(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    normalized = value.strip()
    if not normalized or len(normalized) > 160:
        return ""
    return normalized


def _identifier_list(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        _fail("product_demo_identifier_list_invalid")
    result = tuple(_identifier(item) for item in value)
    if not result or any(not item for item in result) or len(set(result)) != len(result):
        _fail("product_demo_identifier_list_invalid")
    return result


def _coordinate(latitude: Any, longitude: Any) -> bool:
    return (
        _finite_number(latitude)
        and _finite_number(longitude)
        and -90 <= float(latitude) <= 90
        and -180 <= float(longitude) <= 180
    )


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _fail(code: str) -> None:
    raise ProductDemoCoherenceError(code)
