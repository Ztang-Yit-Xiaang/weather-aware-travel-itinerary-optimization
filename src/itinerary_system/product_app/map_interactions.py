"""Server-owned snap and affected-leg previews for direct map intents."""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol
from uuid import uuid4

from itinerary_system.routing.models import RouteLegResult
from itinerary_system.routing.provider import RouteLegRequest

from .routing_runtime import RuntimeRoutingError, RuntimeSnapResult

SNAP_PREVIEW_TTL = timedelta(minutes=10)
_NORMAL_SNAP_MAX_M = 100.0
_WARNING_SNAP_MAX_M = 500.0
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_RUNTIME_ROUTE_PROVENANCE = {
    "provider": "runtime_osrm",
    "routing_status": "osrm_route_validated",
    "routing_profile": "driving",
    "geometry_source": "runtime_osrm_geojson",
    "distance_source": "runtime_osrm_route",
    "duration_source": "runtime_osrm_route",
}
ALLOWED_SNAP_INTENTS = frozenset(
    {
        "explore_only",
        "add_custom_waypoint",
        "relocate_custom_waypoint",
        "replace_stop_near_location",
        "add_route_waypoint",
    }
)


class MapInteractionError(ValueError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class InteractiveRouter(Protocol):
    def nearest(self, entity_id: str, point: tuple[float, float]) -> RuntimeSnapResult: ...

    def route(self, request: RouteLegRequest) -> RouteLegResult: ...


@dataclass(frozen=True)
class RouteAccessPointPreviewV1:
    access_point_id: str
    access_type: str
    coordinate: tuple[float, float]
    source: str
    road_validated: bool
    access_confidence: str
    evidence_refs: tuple[str, ...]


@dataclass(frozen=True)
class SnapPreviewV1:
    snap_preview_id: str
    entity_id: str
    operation_intent: str
    raw_coordinate: tuple[float, float]
    snapped_coordinate: tuple[float, float] | None
    selected_access_point: RouteAccessPointPreviewV1 | None
    snap_distance_m: float | None
    validation_state: str
    code: str
    confirmation_required: bool
    draft_append_allowed: bool
    affected_route_legs: tuple[RouteLegResult, ...]
    created_at: str
    expires_at: str
    schema_version: str = "map-snap-preview-v1"

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["raw_coordinate"] = _coordinate_dict(self.raw_coordinate)
        payload["snapped_coordinate"] = (
            _coordinate_dict(self.snapped_coordinate) if self.snapped_coordinate is not None else None
        )
        payload["selected_access_point"] = (
            {
                **asdict(self.selected_access_point),
                "coordinate": _coordinate_dict(self.selected_access_point.coordinate),
            }
            if self.selected_access_point is not None
            else None
        )
        payload["affected_route_legs"] = [_route_leg_dict(leg) for leg in self.affected_route_legs]
        return payload


class MapInteractionService:
    """Produce non-mutating snap/route previews for a raw map intent."""

    def __init__(self, router: InteractiveRouter) -> None:
        self.router = router

    def preview(
        self,
        *,
        entity_id: str,
        raw_coordinate: tuple[float, float],
        operation_intent: str,
        predecessor: tuple[str, tuple[float, float]] | None = None,
        successor: tuple[str, tuple[float, float]] | None = None,
        travel_mode: str = "driving",
    ) -> SnapPreviewV1:
        if operation_intent not in ALLOWED_SNAP_INTENTS:
            raise MapInteractionError("unsupported_map_operation_intent")
        if travel_mode != "driving":
            raise MapInteractionError("route_mode_not_enabled")
        try:
            snap = self.router.nearest(entity_id, raw_coordinate)
        except RuntimeRoutingError as exc:
            raise MapInteractionError(exc.code) from None
        _validate_runtime_snap(snap, entity_id=entity_id, raw_coordinate=raw_coordinate)

        created = datetime.now(UTC)
        if not snap.draft_append_allowed:
            return SnapPreviewV1(
                snap_preview_id=f"snap_{uuid4().hex}",
                entity_id=entity_id,
                operation_intent=operation_intent,
                raw_coordinate=snap.raw_point,
                snapped_coordinate=snap.snapped_point,
                selected_access_point=None,
                snap_distance_m=snap.snap_distance_m,
                validation_state=snap.validation_state,
                code=snap.code,
                confirmation_required=snap.confirmation_required,
                draft_append_allowed=False,
                affected_route_legs=(),
                created_at=created.isoformat(),
                expires_at=(created + SNAP_PREVIEW_TTL).isoformat(),
            )

        access_point_id = f"access_{uuid4().hex}"
        access_point = RouteAccessPointPreviewV1(
            access_point_id=access_point_id,
            access_type="road_snap",
            coordinate=snap.snapped_point,
            source=snap.provider,
            road_validated=True,
            access_confidence="road_snap_only",
            evidence_refs=(),
        )
        affected: list[RouteLegResult] = []
        if operation_intent == "explore_only":
            return SnapPreviewV1(
                snap_preview_id=f"snap_{uuid4().hex}",
                entity_id=entity_id,
                operation_intent=operation_intent,
                raw_coordinate=snap.raw_point,
                snapped_coordinate=snap.snapped_point,
                selected_access_point=access_point,
                snap_distance_m=snap.snap_distance_m,
                validation_state="snap_only",
                code="exploratory_snap_ready",
                confirmation_required=snap.confirmation_required,
                draft_append_allowed=False,
                affected_route_legs=(),
                created_at=created.isoformat(),
                expires_at=(created + SNAP_PREVIEW_TTL).isoformat(),
            )
        try:
            if predecessor is not None:
                affected.append(
                    _validated_route_leg(
                        self.router.route(
                            RouteLegRequest(
                                origin_id=predecessor[0],
                                destination_id=entity_id,
                                origin_point=predecessor[1],
                                destination_point=snap.snapped_point,
                                routing_profile=travel_mode,
                            )
                        ),
                        origin_id=predecessor[0],
                        destination_id=entity_id,
                    )
                )
            if successor is not None:
                affected.append(
                    _validated_route_leg(
                        self.router.route(
                            RouteLegRequest(
                                origin_id=entity_id,
                                destination_id=successor[0],
                                origin_point=snap.snapped_point,
                                destination_point=successor[1],
                                routing_profile=travel_mode,
                            )
                        ),
                        origin_id=entity_id,
                        destination_id=successor[0],
                    )
                )
        except RuntimeRoutingError as exc:
            raise MapInteractionError(exc.code) from None

        if predecessor is None and successor is None:
            code = "snap_ready_route_context_required"
            append_allowed = False
        else:
            code = snap.code
            append_allowed = True
        evidence_refs = tuple(f"route_query:{leg.query_hash}" for leg in affected)
        access_point = RouteAccessPointPreviewV1(
            access_point_id=access_point.access_point_id,
            access_type=access_point.access_type,
            coordinate=access_point.coordinate,
            source=access_point.source,
            road_validated=access_point.road_validated,
            access_confidence=access_point.access_confidence,
            evidence_refs=evidence_refs,
        )
        return SnapPreviewV1(
            snap_preview_id=f"snap_{uuid4().hex}",
            entity_id=entity_id,
            operation_intent=operation_intent,
            raw_coordinate=snap.raw_point,
            snapped_coordinate=snap.snapped_point,
            selected_access_point=access_point,
            snap_distance_m=snap.snap_distance_m,
            validation_state="route_checked" if append_allowed else "snap_only",
            code=code,
            confirmation_required=snap.confirmation_required,
            draft_append_allowed=append_allowed,
            affected_route_legs=tuple(affected),
            created_at=created.isoformat(),
            expires_at=(created + SNAP_PREVIEW_TTL).isoformat(),
        )


def _validated_route_leg(
    leg: RouteLegResult,
    *,
    origin_id: str,
    destination_id: str,
) -> RouteLegResult:
    if (
        leg.origin_id != origin_id
        or leg.destination_id != destination_id
        or not leg.road_validated
        or leg.fallback_used
        or leg.fallback_reason is not None
        or len(leg.geometry) < 2
        or any(not _valid_route_point(point) for point in leg.geometry)
        or not _positive_finite(leg.distance_m)
        or not _positive_finite(leg.duration_s)
        or any(getattr(leg, field) != expected for field, expected in _RUNTIME_ROUTE_PROVENANCE.items())
        or _SHA256_PATTERN.fullmatch(leg.query_hash) is None
        or not _aware_datetime(leg.retrieved_at)
        or not _nonnegative_finite(leg.snap_distance_origin_m)
        or not _nonnegative_finite(leg.snap_distance_destination_m)
    ):
        raise MapInteractionError("route_preview_not_road_validated")
    return leg


def _valid_route_point(point: tuple[float, float]) -> bool:
    if not isinstance(point, (tuple, list)) or len(point) != 2:
        return False
    latitude, longitude = point
    return bool(
        _finite_number(latitude)
        and _finite_number(longitude)
        and -90 <= float(latitude) <= 90
        and -180 <= float(longitude) <= 180
    )


def _validate_runtime_snap(
    snap: RuntimeSnapResult,
    *,
    entity_id: str,
    raw_coordinate: tuple[float, float],
) -> None:
    if (
        snap.entity_id != entity_id
        or snap.provider != "runtime_osrm"
        or not _valid_route_point(snap.raw_point)
        or tuple(snap.raw_point) != tuple(raw_coordinate)
        or snap.snapped_point is None
        or not _valid_route_point(snap.snapped_point)
        or not _nonnegative_finite(snap.snap_distance_m)
    ):
        raise MapInteractionError("snap_preview_not_road_validated")
    distance = float(snap.snap_distance_m)
    if distance <= _NORMAL_SNAP_MAX_M:
        expected = ("normal", "snap_within_normal_threshold", False, True)
    elif distance <= _WARNING_SNAP_MAX_M:
        expected = ("warning", "snap_confirmation_required", True, True)
    else:
        expected = ("rejected", "map_snap_too_far", False, False)
    actual = (
        snap.validation_state,
        snap.code,
        snap.confirmation_required,
        snap.draft_append_allowed,
    )
    if snap.status != snap.validation_state or actual != expected:
        raise MapInteractionError("snap_preview_not_road_validated")


def _positive_finite(value: Any) -> bool:
    return bool(_finite_number(value) and float(value) > 0)


def _nonnegative_finite(value: Any) -> bool:
    return bool(_finite_number(value) and float(value) >= 0)


def _finite_number(value: Any) -> bool:
    return bool(not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value))


def _aware_datetime(value: Any) -> bool:
    if not isinstance(value, datetime) or value.tzinfo is None:
        return False
    try:
        return value.utcoffset() is not None
    except Exception:
        return False


def _coordinate_dict(point: tuple[float, float]) -> dict[str, float]:
    return {"latitude": point[0], "longitude": point[1]}


def _route_leg_dict(leg: RouteLegResult) -> dict[str, Any]:
    return {
        "route_leg_id": f"preview_leg_{leg.query_hash[:16]}",
        "origin_id": leg.origin_id,
        "destination_id": leg.destination_id,
        "travel_mode": leg.routing_profile,
        "validation_status": "road_validated",
        "geometry": {
            "type": "LineString",
            "coordinates": [[point[1], point[0]] for point in leg.geometry],
        },
        "distance_m": leg.distance_m,
        "duration_s": leg.duration_s,
        "provider": leg.provider,
        "routing_status": leg.routing_status,
        "geometry_source": leg.geometry_source,
        "distance_source": leg.distance_source,
        "duration_source": leg.duration_source,
        "road_validated": leg.road_validated,
        "fallback_used": leg.fallback_used,
        "query_hash": leg.query_hash,
        "evidence_refs": [f"route_query:{leg.query_hash}"],
        "retrieved_at": leg.retrieved_at.isoformat() if leg.retrieved_at else None,
        "snap_distance_origin_m": leg.snap_distance_origin_m,
        "snap_distance_destination_m": leg.snap_distance_destination_m,
    }
