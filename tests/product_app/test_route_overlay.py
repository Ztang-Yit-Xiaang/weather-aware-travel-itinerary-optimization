from __future__ import annotations

from datetime import UTC, datetime

from itinerary_system.product_app.route_overlay import build_session_route_overlay
from itinerary_system.routing.models import RouteLegResult


def _plan() -> dict[str, object]:
    return {
        "plan_id": "plan_test",
        "sequence": ["a", "b", "c"],
        "selected_stops": [
            {"stop_id": "a", "day": 1},
            {"stop_id": "b", "day": 1},
            {"stop_id": "c", "day": 2},
        ],
    }


def _specs() -> tuple[dict[str, object], ...]:
    return (
        {"day": 1, "origin_id": "a", "destination_id": "b", "evidence_scope": "base"},
        {"day": 2, "origin_id": "b", "destination_id": "c", "evidence_scope": "base"},
    )


def _cell(origin: str, destination: str) -> dict[str, object]:
    return {
        "route_leg_id": f"base_{origin}_{destination}",
        "road_validated": True,
        "fallback_used": False,
        "geometry": [[44.0, -93.0], [44.1, -93.1]],
        "distance_m": 1000.0,
        "duration_s": 120.0,
        "query_hash": f"hash_{origin}_{destination}",
    }


def _runtime_leg() -> RouteLegResult:
    return RouteLegResult(
        origin_id="b",
        destination_id="c",
        geometry=((44.1, -93.1), (44.2, -93.2)),
        distance_m=900.0,
        duration_s=100.0,
        routing_status="osrm_route_validated",
        provider="runtime_osrm",
        routing_profile="driving",
        geometry_source="runtime_osrm_geojson",
        distance_source="runtime_osrm_route",
        duration_source="runtime_osrm_route",
        road_validated=True,
        fallback_used=False,
        query_hash="a" * 64,
        retrieved_at=datetime.now(UTC),
        snap_distance_origin_m=5.0,
        snap_distance_destination_m=7.0,
    )


def test_overlay_preserves_base_legs_and_replaces_only_affected_leg() -> None:
    base = {("a", "b"): _cell("a", "b"), ("b", "c"): _cell("b", "c")}

    overlay = build_session_route_overlay(
        _plan(),
        _specs(),
        base,
        runtime_legs={("b", "c"): _runtime_leg()},
        context_snapshot_id="context_v1",
    )

    assert overlay.schema_version == "session-route-overlay-v1"
    assert overlay.required_leg_count == 2
    assert overlay.road_validated_leg_count == 2
    assert overlay.gap_count == 0
    assert overlay.complete is True
    assert overlay.acceptance_eligible is True
    assert overlay.legs[0].evidence_source == "immutable_base"
    assert overlay.legs[1].evidence_source == "session_runtime"
    assert overlay.legs[1].query_hash == "a" * 64
    assert overlay.legs[1].duration_s == 100.0


def test_overlay_exposes_missing_leg_as_gap_without_geometry() -> None:
    overlay = build_session_route_overlay(
        _plan(),
        _specs(),
        {("a", "b"): _cell("a", "b")},
        context_snapshot_id="context_v1",
    )

    assert overlay.road_validated_leg_count == 1
    assert overlay.gap_count == 1
    assert overlay.complete is False
    assert overlay.acceptance_eligible is False
    assert overlay.failure_codes == ("route_leg_missing",)
    assert overlay.legs[1].evidence_source == "gap"
    assert overlay.legs[1].geometry is None
    assert overlay.legs[1].failure_code == "route_leg_missing"


def test_overlay_rejects_continuous_path_that_omits_selected_stop() -> None:
    specs = (
        {"day": 1, "origin_id": "a", "destination_id": "c", "evidence_scope": "base"},
    )
    overlay = build_session_route_overlay(
        _plan(),
        specs,
        {("a", "c"): _cell("a", "c")},
        context_snapshot_id="context_v1",
    )

    assert overlay.itinerary_sequence_accounted is False
    assert overlay.complete is False
    assert overlay.acceptance_eligible is False
    assert "itinerary_sequence_not_accounted" in overlay.failure_codes
