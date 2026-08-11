"""Loopback-only OSRM boundary for interactive product routing.

The product uses ``(latitude, longitude)`` tuples internally.  This module is
the only place that translates those coordinates into OSRM's
``longitude,latitude`` URL representation.  Provider failures are reduced to
stable codes so raw URLs and response bodies never escape this boundary.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal, Protocol

from itinerary_system.routing.matrix import RouteMatrix, RouteMatrixCell
from itinerary_system.routing.models import RouteLegResult
from itinerary_system.routing.provider import RouteLegRequest, RouteMatrixRequest, SnapResult

DEFAULT_OSRM_BASE_URL = "http://127.0.0.1:5000"
DEFAULT_TIMEOUT_SECONDS = 3.0
DEFAULT_MAX_RESPONSE_BYTES = 1_000_000
DEFAULT_MAX_TABLE_POINTS = 25

_IDENTIFIER_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}")


class RuntimeRoutingError(RuntimeError):
    """Routing failure carrying only a stable, user-safe code."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class SnapThresholds:
    """Distance thresholds for turning a road snap into a draft candidate."""

    normal_max_m: float = 100.0
    warning_max_m: float = 500.0

    def __post_init__(self) -> None:
        normal = _positive_finite(self.normal_max_m, "invalid_snap_thresholds")
        warning = _positive_finite(self.warning_max_m, "invalid_snap_thresholds")
        if warning <= normal:
            raise RuntimeRoutingError("invalid_snap_thresholds")
        object.__setattr__(self, "normal_max_m", normal)
        object.__setattr__(self, "warning_max_m", warning)


@dataclass(frozen=True)
class RuntimeSnapResult(SnapResult):
    """Nearest-road result with explicit confirmation and rejection semantics."""

    raw_point: tuple[float, float]
    validation_state: Literal["normal", "warning", "rejected"]
    code: str
    confirmation_required: bool
    draft_append_allowed: bool


class RoutingJsonTransport(Protocol):
    """Injectable bounded JSON transport used by deterministic tests."""

    def fetch(
        self,
        url: str,
        *,
        timeout_seconds: float,
        max_response_bytes: int,
        expected_origin: str,
    ) -> Mapping[str, Any]:
        """Fetch one JSON object or raise ``RuntimeRoutingError``."""


class _RejectRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Mapping[str, str],
        newurl: str,
    ) -> None:
        raise RuntimeRoutingError("routing_redirect_not_allowed")


class _UrlLibRoutingTransport:
    """Standard-library transport with redirect and response-size controls."""

    def __init__(self) -> None:
        self._opener = urllib.request.build_opener(_RejectRedirects)

    def fetch(
        self,
        url: str,
        *,
        timeout_seconds: float,
        max_response_bytes: int,
        expected_origin: str,
    ) -> Mapping[str, Any]:
        request = urllib.request.Request(url, headers={"Accept": "application/json"})
        try:
            with self._opener.open(request, timeout=timeout_seconds) as response:
                if _origin(response.geturl()) != expected_origin:
                    raise RuntimeRoutingError("routing_redirect_not_allowed")
                status = int(getattr(response, "status", 200))
                if status != 200:
                    raise RuntimeRoutingError("routing_http_error")
                content_length = response.headers.get("Content-Length")
                if content_length is not None:
                    try:
                        declared_length = int(content_length)
                        if declared_length < 0:
                            raise RuntimeRoutingError("routing_response_invalid")
                        if declared_length > max_response_bytes:
                            raise RuntimeRoutingError("routing_response_too_large")
                    except ValueError as exc:
                        raise RuntimeRoutingError("routing_response_invalid") from exc
                raw = response.read(max_response_bytes + 1)
        except RuntimeRoutingError:
            raise
        except urllib.error.HTTPError as exc:
            if 300 <= int(exc.code) < 400:
                raise RuntimeRoutingError("routing_redirect_not_allowed") from None
            raise RuntimeRoutingError("routing_http_error") from None
        except TimeoutError as exc:
            raise RuntimeRoutingError("routing_timeout") from exc
        except (OSError, urllib.error.URLError) as exc:
            raise RuntimeRoutingError("routing_unavailable") from exc
        if len(raw) > max_response_bytes:
            raise RuntimeRoutingError("routing_response_too_large")
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeRoutingError("routing_response_invalid") from exc
        if not isinstance(payload, dict):
            raise RuntimeRoutingError("routing_response_invalid")
        return payload


class RuntimeRoutingService:
    """Server-authoritative, loopback-only OSRM interaction service."""

    provider_id = "runtime_osrm"

    def __init__(
        self,
        *,
        base_url: str = DEFAULT_OSRM_BASE_URL,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES,
        max_table_points: int = DEFAULT_MAX_TABLE_POINTS,
        thresholds: SnapThresholds | None = None,
        allowed_profiles: tuple[str, ...] = ("driving",),
        transport: RoutingJsonTransport | None = None,
    ) -> None:
        self.base_url = _validated_loopback_base_url(base_url)
        self.origin = _origin(self.base_url)
        self.timeout_seconds = _positive_finite(timeout_seconds, "invalid_routing_timeout")
        if (
            isinstance(max_response_bytes, bool)
            or not isinstance(max_response_bytes, int)
            or not 1 <= max_response_bytes <= 10_000_000
        ):
            raise RuntimeRoutingError("invalid_routing_response_limit")
        if (
            isinstance(max_table_points, bool)
            or not isinstance(max_table_points, int)
            or not 2 <= max_table_points <= 100
        ):
            raise RuntimeRoutingError("invalid_routing_table_limit")
        normalized_profiles = tuple(str(value).strip().lower() for value in allowed_profiles)
        if (
            not normalized_profiles
            or len(set(normalized_profiles)) != len(normalized_profiles)
            or any(re.fullmatch(r"[a-z][a-z0-9_-]{0,31}", value) is None for value in normalized_profiles)
        ):
            raise RuntimeRoutingError("invalid_routing_profiles")
        self.max_response_bytes = max_response_bytes
        self.max_table_points = max_table_points
        self.thresholds = thresholds or SnapThresholds()
        self.allowed_profiles = frozenset(normalized_profiles)
        self._transport = transport or _UrlLibRoutingTransport()

    def nearest(self, entity_id: str, point: tuple[float, float]) -> RuntimeSnapResult:
        """Return a classified nearest-road preview without mutating a draft."""

        entity = _identifier(entity_id)
        raw_point = _point(point)
        path = f"/nearest/v1/driving/{_osrm_coordinate(raw_point)}"
        payload = self._fetch(path, {"number": "1"})
        _require_ok(payload)
        waypoint = _first_record(payload, "waypoints")
        snapped_point = _location(waypoint.get("location"))
        distance = _nonnegative_finite(waypoint.get("distance"), "routing_response_invalid")
        state, code, confirmation_required, append_allowed = self._classify_snap(distance)
        return RuntimeSnapResult(
            entity_id=entity,
            snapped_point=snapped_point,
            snap_distance_m=distance,
            provider=self.provider_id,
            status=state,
            raw_point=raw_point,
            validation_state=state,
            code=code,
            confirmation_required=confirmation_required,
            draft_append_allowed=append_allowed,
        )

    def route(self, request: RouteLegRequest) -> RouteLegResult:
        """Return one road-valid OSRM leg or fail closed with a stable code."""

        origin_id = _identifier(request.origin_id)
        destination_id = _identifier(request.destination_id)
        origin = _point(request.origin_point)
        destination = _point(request.destination_point)
        profile = self._profile(request.routing_profile)
        coordinates = f"{_osrm_coordinate(origin)};{_osrm_coordinate(destination)}"
        query = {"overview": "full", "geometries": "geojson", "steps": "false"}
        payload = self._fetch(f"/route/v1/{profile}/{coordinates}", query)
        _require_ok(payload)
        route = _first_record(payload, "routes")
        distance = _positive_finite(route.get("distance"), "routing_response_invalid")
        duration = _positive_finite(route.get("duration"), "routing_response_invalid")
        geometry = _geometry(route.get("geometry"))
        waypoints = _records(payload, "waypoints", exact_length=2)
        origin_snap = _nonnegative_finite(waypoints[0].get("distance"), "routing_response_invalid")
        destination_snap = _nonnegative_finite(
            waypoints[1].get("distance"), "routing_response_invalid"
        )
        if max(origin_snap, destination_snap) > self.thresholds.warning_max_m:
            raise RuntimeRoutingError("route_endpoint_snap_too_far")
        query_hash = _query_hash(
            {
                "operation": "route",
                "profile": profile,
                "origin": origin,
                "destination": destination,
            }
        )
        return RouteLegResult(
            origin_id=origin_id,
            destination_id=destination_id,
            geometry=geometry,
            distance_m=distance,
            duration_s=duration,
            routing_status="osrm_route_validated",
            provider=self.provider_id,
            routing_profile=profile,
            geometry_source="runtime_osrm_geojson",
            distance_source="runtime_osrm_route",
            duration_source="runtime_osrm_route",
            road_validated=True,
            fallback_used=False,
            query_hash=query_hash,
            retrieved_at=datetime.now(UTC),
            snap_distance_origin_m=origin_snap,
            snap_distance_destination_m=destination_snap,
        )

    def table(self, request: RouteMatrixRequest) -> RouteMatrix:
        """Return a directed OSRM table for a bounded set of named points."""

        profile = self._profile(request.routing_profile)
        if not isinstance(request.entity_points, dict):
            raise RuntimeRoutingError("invalid_routing_points")
        if not 2 <= len(request.entity_points) <= self.max_table_points:
            raise RuntimeRoutingError("routing_table_point_limit")
        items = sorted(
            ((_identifier(entity_id), _point(point)) for entity_id, point in request.entity_points.items()),
            key=lambda item: item[0],
        )
        if len({entity_id for entity_id, _ in items}) != len(items):
            raise RuntimeRoutingError("invalid_routing_points")
        coordinates = ";".join(_osrm_coordinate(point) for _, point in items)
        payload = self._fetch(
            f"/table/v1/{profile}/{coordinates}",
            {"annotations": "duration,distance"},
        )
        _require_ok(payload)
        size = len(items)
        durations = _square_matrix(payload.get("durations"), size)
        distances = _square_matrix(payload.get("distances"), size)
        sources = _records(payload, "sources", exact_length=size)
        destinations = _records(payload, "destinations", exact_length=size)
        snap_distances = [
            _nonnegative_finite(record.get("distance"), "routing_response_invalid")
            for record in (*sources, *destinations)
        ]
        if max(snap_distances) > self.thresholds.warning_max_m:
            raise RuntimeRoutingError("table_endpoint_snap_too_far")
        entity_ids = tuple(entity_id for entity_id, _ in items)
        query_hash = _query_hash(
            {
                "operation": "table",
                "profile": profile,
                "points": items,
                "context_snapshot_id": request.context_snapshot_id,
            }
        )
        cells: dict[tuple[str, str], RouteMatrixCell] = {}
        for origin_index, origin_id in enumerate(entity_ids):
            for destination_index, destination_id in enumerate(entity_ids):
                if origin_id == destination_id:
                    continue
                duration = _optional_positive(durations[origin_index][destination_index])
                distance = _optional_positive(distances[origin_index][destination_index])
                available = duration is not None and distance is not None
                cells[(origin_id, destination_id)] = RouteMatrixCell(
                    origin_id=origin_id,
                    destination_id=destination_id,
                    distance_m=distance,
                    duration_s=duration,
                    route_leg_id=f"runtime_{query_hash}_{origin_index}_{destination_index}",
                    road_validated=available,
                    fallback_used=False,
                    provider=self.provider_id,
                    query_hash=query_hash,
                    context_snapshot_id=str(request.context_snapshot_id),
                    routing_profile=profile,
                    routing_status="osrm_table_validated" if available else "osrm_table_unavailable",
                    geometry_source="not_requested_for_table",
                    distance_source="runtime_osrm_table" if available else "unavailable",
                    duration_source="runtime_osrm_table" if available else "unavailable",
                )
        return RouteMatrix(
            matrix_id=f"runtime_osrm_{query_hash}",
            context_snapshot_id=str(request.context_snapshot_id),
            entity_ids=entity_ids,
            cells=cells,
        )

    def _profile(self, value: str) -> str:
        profile = str(value).strip().lower()
        if profile not in self.allowed_profiles:
            raise RuntimeRoutingError("routing_profile_not_allowed")
        return profile

    def _classify_snap(self, distance_m: float) -> tuple[str, str, bool, bool]:
        if distance_m <= self.thresholds.normal_max_m:
            return "normal", "snap_within_normal_threshold", False, True
        if distance_m <= self.thresholds.warning_max_m:
            return "warning", "snap_confirmation_required", True, True
        return "rejected", "map_snap_too_far", False, False

    def _fetch(self, path: str, query: Mapping[str, str]) -> Mapping[str, Any]:
        encoded_query = urllib.parse.urlencode(query, safe=",")
        url = f"{self.base_url}{path}"
        if encoded_query:
            url = f"{url}?{encoded_query}"
        try:
            payload = self._transport.fetch(
                url,
                timeout_seconds=self.timeout_seconds,
                max_response_bytes=self.max_response_bytes,
                expected_origin=self.origin,
            )
        except RuntimeRoutingError:
            raise
        except Exception as exc:
            raise RuntimeRoutingError("routing_transport_failure") from exc
        if not isinstance(payload, Mapping):
            raise RuntimeRoutingError("routing_response_invalid")
        return payload


def _validated_loopback_base_url(value: str) -> str:
    try:
        parsed = urllib.parse.urlparse(str(value).strip())
        port = parsed.port
    except ValueError as exc:
        raise RuntimeRoutingError("routing_base_url_not_loopback") from exc
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"127.0.0.1", "localhost"}
        or parsed.username is not None
        or parsed.password is not None
        or port is None
        or not 1 <= port <= 65_535
        or parsed.path not in {"", "/"}
        or parsed.params
        or parsed.query
        or parsed.fragment
    ):
        raise RuntimeRoutingError("routing_base_url_not_loopback")
    return f"http://{parsed.hostname}:{port}"


def _origin(value: str) -> str:
    parsed = urllib.parse.urlparse(value)
    return f"{parsed.scheme}://{parsed.hostname}:{parsed.port}"


def _identifier(value: Any) -> str:
    if not isinstance(value, str) or _IDENTIFIER_PATTERN.fullmatch(value.strip()) is None:
        raise RuntimeRoutingError("invalid_routing_identifier")
    return value.strip()


def _point(value: Any) -> tuple[float, float]:
    if not isinstance(value, tuple | list) or len(value) != 2:
        raise RuntimeRoutingError("invalid_routing_coordinate")
    if isinstance(value[0], bool) or isinstance(value[1], bool):
        raise RuntimeRoutingError("invalid_routing_coordinate")
    try:
        latitude, longitude = float(value[0]), float(value[1])
    except (TypeError, ValueError) as exc:
        raise RuntimeRoutingError("invalid_routing_coordinate") from exc
    if (
        not math.isfinite(latitude)
        or not math.isfinite(longitude)
        or not -90.0 <= latitude <= 90.0
        or not -180.0 <= longitude <= 180.0
    ):
        raise RuntimeRoutingError("invalid_routing_coordinate")
    return latitude, longitude


def _osrm_coordinate(point: tuple[float, float]) -> str:
    latitude, longitude = point
    return f"{_coordinate_text(longitude)},{_coordinate_text(latitude)}"


def _coordinate_text(value: float) -> str:
    text = f"{value:.8f}".rstrip("0").rstrip(".")
    return "0" if text in {"-0", ""} else text


def _positive_finite(value: Any, code: str) -> float:
    if isinstance(value, bool):
        raise RuntimeRoutingError(code)
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeRoutingError(code) from exc
    if not math.isfinite(result) or result <= 0:
        raise RuntimeRoutingError(code)
    return result


def _nonnegative_finite(value: Any, code: str) -> float:
    if isinstance(value, bool):
        raise RuntimeRoutingError(code)
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeRoutingError(code) from exc
    if not math.isfinite(result) or result < 0:
        raise RuntimeRoutingError(code)
    return result


def _optional_positive(value: Any) -> float | None:
    if value is None:
        return None
    return _positive_finite(value, "routing_response_invalid")


def _require_ok(payload: Mapping[str, Any]) -> None:
    if payload.get("code") != "Ok":
        raise RuntimeRoutingError("routing_provider_rejected_request")


def _records(
    payload: Mapping[str, Any], key: str, *, exact_length: int | None = None
) -> tuple[Mapping[str, Any], ...]:
    value = payload.get(key)
    if not isinstance(value, list) or (exact_length is not None and len(value) != exact_length):
        raise RuntimeRoutingError("routing_response_invalid")
    records = tuple(item for item in value if isinstance(item, Mapping))
    if len(records) != len(value) or not records:
        raise RuntimeRoutingError("routing_response_invalid")
    return records


def _first_record(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    return _records(payload, key)[0]


def _location(value: Any) -> tuple[float, float]:
    if not isinstance(value, list | tuple) or len(value) != 2:
        raise RuntimeRoutingError("routing_response_invalid")
    try:
        longitude, latitude = float(value[0]), float(value[1])
    except (TypeError, ValueError) as exc:
        raise RuntimeRoutingError("routing_response_invalid") from exc
    return _point((latitude, longitude))


def _geometry(value: Any) -> tuple[tuple[float, float], ...]:
    if not isinstance(value, Mapping) or value.get("type") != "LineString":
        raise RuntimeRoutingError("routing_response_invalid")
    coordinates = value.get("coordinates")
    if not isinstance(coordinates, list) or len(coordinates) < 2:
        raise RuntimeRoutingError("routing_response_invalid")
    return tuple(_location(point) for point in coordinates)


def _square_matrix(value: Any, size: int) -> tuple[tuple[Any, ...], ...]:
    if not isinstance(value, list) or len(value) != size:
        raise RuntimeRoutingError("routing_response_invalid")
    rows: list[tuple[Any, ...]] = []
    for row in value:
        if not isinstance(row, list) or len(row) != size:
            raise RuntimeRoutingError("routing_response_invalid")
        rows.append(tuple(row))
    return tuple(rows)


def _query_hash(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()
