from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta, tzinfo

import pytest

from itinerary_system.product_app.map_interactions import (
    MapInteractionError,
    MapInteractionService,
)
from itinerary_system.product_app.routing_runtime import RuntimeRoutingError, RuntimeSnapResult
from itinerary_system.routing.models import RouteLegResult
from itinerary_system.routing.provider import RouteLegRequest


class FakeRouter:
    def __init__(self, snap: RuntimeSnapResult) -> None:
        self.snap = snap
        self.route_requests: list[RouteLegRequest] = []

    def nearest(self, entity_id: str, point: tuple[float, float]) -> RuntimeSnapResult:
        assert entity_id == self.snap.entity_id
        assert point == self.snap.raw_point
        return self.snap

    def route(self, request: RouteLegRequest) -> RouteLegResult:
        self.route_requests.append(request)
        return RouteLegResult(
            origin_id=request.origin_id,
            destination_id=request.destination_id,
            geometry=(request.origin_point, request.destination_point),
            distance_m=1000.0,
            duration_s=120.0,
            routing_status="osrm_route_validated",
            provider="runtime_osrm",
            routing_profile=request.routing_profile,
            geometry_source="runtime_osrm_geojson",
            distance_source="runtime_osrm_route",
            duration_source="runtime_osrm_route",
            road_validated=True,
            fallback_used=False,
            query_hash=("a" if len(self.route_requests) == 1 else "b") * 64,
            retrieved_at=datetime.now(UTC),
            snap_distance_origin_m=5.0,
            snap_distance_destination_m=5.0,
        )


class NullOffsetTimezone(tzinfo):
    def utcoffset(self, dt: datetime | None) -> None:
        return None

    def dst(self, dt: datetime | None) -> timedelta:
        return timedelta(0)

    def tzname(self, dt: datetime | None) -> str:
        return "null-offset"


def _snap(distance: float, *, allowed: bool = True, confirmation: bool = False) -> RuntimeSnapResult:
    state = "warning" if confirmation else "normal"
    code = "snap_confirmation_required" if confirmation else "snap_within_normal_threshold"
    if not allowed:
        state, code = "rejected", "map_snap_too_far"
    return RuntimeSnapResult(
        entity_id="custom_a",
        snapped_point=(44.01, -93.01),
        snap_distance_m=distance,
        provider="runtime_osrm",
        status=state,
        raw_point=(44.0, -93.0),
        validation_state=state,
        code=code,
        confirmation_required=confirmation,
        draft_append_allowed=allowed,
    )


def test_snap_preview_routes_predecessor_and_successor_without_mutating_a_draft() -> None:
    router = FakeRouter(_snap(25.0))
    preview = MapInteractionService(router).preview(
        entity_id="custom_a",
        raw_coordinate=(44.0, -93.0),
        operation_intent="add_custom_waypoint",
        predecessor=("before", (43.9, -93.1)),
        successor=("after", (44.1, -92.9)),
    )

    assert preview.schema_version == "map-snap-preview-v1"
    assert preview.raw_coordinate == (44.0, -93.0)
    assert preview.snapped_coordinate == (44.01, -93.01)
    assert preview.validation_state == "route_checked"
    assert preview.draft_append_allowed is True
    assert preview.confirmation_required is False
    assert len(preview.affected_route_legs) == 2
    assert preview.selected_access_point is not None
    assert preview.selected_access_point.access_confidence == "road_snap_only"
    assert preview.selected_access_point.evidence_refs == (
        f"route_query:{'a' * 64}",
        f"route_query:{'b' * 64}",
    )
    assert [request.origin_id for request in router.route_requests] == ["before", "custom_a"]
    public = preview.as_dict()
    assert public["raw_coordinate"] == {"latitude": 44.0, "longitude": -93.0}
    assert public["selected_access_point"]["coordinate"] == {
        "latitude": 44.01,
        "longitude": -93.01,
    }
    assert public["affected_route_legs"][0]["geometry"] == {
        "type": "LineString",
        "coordinates": [[-93.1, 43.9], [-93.01, 44.01]],
    }
    assert public["affected_route_legs"][0]["evidence_refs"] == [f"route_query:{'a' * 64}"]


def test_warning_snap_requires_confirmation_but_keeps_route_preview() -> None:
    router = FakeRouter(_snap(250.0, confirmation=True))
    preview = MapInteractionService(router).preview(
        entity_id="custom_a",
        raw_coordinate=(44.0, -93.0),
        operation_intent="relocate_custom_waypoint",
        predecessor=("before", (43.9, -93.1)),
    )

    assert preview.validation_state == "route_checked"
    assert preview.confirmation_required is True
    assert preview.draft_append_allowed is True
    assert len(preview.affected_route_legs) == 1


def test_far_snap_is_non_executable_and_routes_nothing() -> None:
    router = FakeRouter(_snap(501.0, allowed=False))
    preview = MapInteractionService(router).preview(
        entity_id="custom_a",
        raw_coordinate=(44.0, -93.0),
        operation_intent="replace_stop_near_location",
        predecessor=("before", (43.9, -93.1)),
        successor=("after", (44.1, -92.9)),
    )

    assert preview.validation_state == "rejected"
    assert preview.code == "map_snap_too_far"
    assert preview.draft_append_allowed is False
    assert preview.selected_access_point is None
    assert preview.affected_route_legs == ()
    assert router.route_requests == []


def test_snap_without_insertion_context_remains_non_executable() -> None:
    router = FakeRouter(_snap(25.0))
    preview = MapInteractionService(router).preview(
        entity_id="custom_a",
        raw_coordinate=(44.0, -93.0),
        operation_intent="add_custom_waypoint",
    )

    assert preview.validation_state == "snap_only"
    assert preview.code == "snap_ready_route_context_required"
    assert preview.draft_append_allowed is False
    assert router.route_requests == []


def test_exploratory_snap_never_routes_or_becomes_appendable() -> None:
    router = FakeRouter(_snap(25.0))
    preview = MapInteractionService(router).preview(
        entity_id="custom_a",
        raw_coordinate=(44.0, -93.0),
        operation_intent="explore_only",
        predecessor=("before", (43.9, -93.1)),
        successor=("after", (44.1, -92.9)),
    )

    assert preview.validation_state == "snap_only"
    assert preview.code == "exploratory_snap_ready"
    assert preview.draft_append_allowed is False
    assert preview.affected_route_legs == ()
    assert router.route_requests == []


def test_unsupported_intent_mode_and_router_failure_fail_closed() -> None:
    service = MapInteractionService(FakeRouter(_snap(25.0)))
    with pytest.raises(MapInteractionError, match="unsupported_map_operation_intent"):
        service.preview(
            entity_id="custom_a",
            raw_coordinate=(44.0, -93.0),
            operation_intent="move_real_place",
        )
    with pytest.raises(MapInteractionError, match="route_mode_not_enabled"):
        service.preview(
            entity_id="custom_a",
            raw_coordinate=(44.0, -93.0),
            operation_intent="add_route_waypoint",
            travel_mode="walking",
        )

    class BrokenRouter(FakeRouter):
        def nearest(self, entity_id: str, point: tuple[float, float]) -> RuntimeSnapResult:
            raise RuntimeRoutingError("routing_unavailable")

    with pytest.raises(MapInteractionError, match="routing_unavailable"):
        MapInteractionService(BrokenRouter(_snap(25.0))).preview(
            entity_id="custom_a",
            raw_coordinate=(44.0, -93.0),
            operation_intent="add_custom_waypoint",
        )

    class UnvalidatedRouter(FakeRouter):
        def route(self, request: RouteLegRequest) -> RouteLegResult:
            return RouteLegResult(
                origin_id=request.origin_id,
                destination_id=request.destination_id,
                geometry=(request.origin_point, request.destination_point),
                distance_m=1000.0,
                duration_s=120.0,
                routing_status="fallback_geodesic_proxy",
                provider="geodesic_proxy",
                road_validated=False,
                fallback_used=True,
            )

    with pytest.raises(MapInteractionError, match="route_preview_not_road_validated"):
        MapInteractionService(UnvalidatedRouter(_snap(25.0))).preview(
            entity_id="custom_a",
            raw_coordinate=(44.0, -93.0),
            operation_intent="add_custom_waypoint",
            predecessor=("before", (43.9, -93.1)),
        )


@pytest.mark.parametrize(
    "overrides",
    [
        {"entity_id": "different_entity"},
        {"provider": "untrusted_provider"},
        {"raw_point": (44.1, -93.0)},
        {"snapped_point": None},
        {"snapped_point": (float("nan"), -93.0)},
        {"snapped_point": (44.0, 999.0)},
        {"snap_distance_m": float("nan")},
        {"snap_distance_m": -1.0},
        {"status": "warning"},
        {"validation_state": "warning"},
        {"code": "snap_confirmation_required"},
        {"confirmation_required": True},
        {"draft_append_allowed": False},
        {"snap_distance_m": 250.0},
        {"snap_distance_m": 501.0},
    ],
)
def test_claimed_snap_with_invalid_provenance_or_diagnostics_fails_closed(
    overrides: dict[str, object],
) -> None:
    class AdversarialSnapRouter(FakeRouter):
        def nearest(self, entity_id: str, point: tuple[float, float]) -> RuntimeSnapResult:
            return replace(super().nearest(entity_id, point), **overrides)

    with pytest.raises(MapInteractionError, match="snap_preview_not_road_validated"):
        MapInteractionService(AdversarialSnapRouter(_snap(25.0))).preview(
            entity_id="custom_a",
            raw_coordinate=(44.0, -93.0),
            operation_intent="explore_only",
        )


@pytest.mark.parametrize(
    "overrides",
    [
        {"provider": "untrusted_provider"},
        {"routing_status": "claimed_valid"},
        {"routing_profile": "walking"},
        {"geometry_source": "untrusted_geometry"},
        {"distance_source": "untrusted_distance"},
        {"duration_source": "untrusted_duration"},
        {"query_hash": ""},
        {"query_hash": "g" * 64},
        {"geometry": ((float("nan"), -93.1), (44.0, -93.0))},
        {"geometry": ((91.0, -93.1), (44.0, -93.0))},
        {"geometry": ((44.0, -181.0), (44.0, -93.0))},
        {"distance_m": float("nan")},
        {"duration_s": float("inf")},
        {"snap_distance_origin_m": -1.0},
        {"snap_distance_destination_m": float("nan")},
        {"retrieved_at": None},
        {"retrieved_at": datetime.now()},
        {"retrieved_at": datetime(2026, 8, 8, tzinfo=NullOffsetTimezone())},
    ],
)
def test_claimed_route_evidence_with_invalid_provenance_fails_closed(
    overrides: dict[str, object],
) -> None:
    class AdversarialRouter(FakeRouter):
        def route(self, request: RouteLegRequest) -> RouteLegResult:
            valid = super().route(request)
            return replace(valid, **overrides)

    with pytest.raises(MapInteractionError, match="route_preview_not_road_validated"):
        MapInteractionService(AdversarialRouter(_snap(25.0))).preview(
            entity_id="custom_a",
            raw_coordinate=(44.0, -93.0),
            operation_intent="add_route_waypoint",
            predecessor=("before", (43.9, -93.1)),
        )
