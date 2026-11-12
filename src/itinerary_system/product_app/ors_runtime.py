"""Bounded, server-side OpenRouteService routing; construction never probes ORS.

API shapes: https://giscience.github.io/openrouteservice/api-reference/endpoints/
Migration: https://ask.openrouteservice.org/t/7912
Plans: https://account.heigit.org/info/plans
The only permitted API base is https://api.heigit.org/openrouteservice. There
is no legacy fallback. Defaults of 30 requests/minute and 500/rolling 24 hours
are combined app limits across all operations, not service quota guarantees.
They stay below/equal the Standard endpoint limits verified on 2026-10-30;
account quotas and other applications using the same key still apply.

Persist the entire cache_root, including ors-budget.json, before serving after
a restart. StateFileLock coordinates workers sharing a filesystem; separate
replicas need an equivalent shared/transactional gate in the hosting adapter.
Transport injection replaces the HTTP boundary; injected transports must honor
the same no-redirect, size and timeout contract as the default transport.
"""

from __future__ import annotations

import http.client
import json
import math
import re
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from geopy.distance import geodesic

from itinerary_system.routing.matrix import RouteMatrix, RouteMatrixCell
from itinerary_system.routing.models import RouteLegResult
from itinerary_system.routing.provider import RouteLegRequest, RouteMatrixRequest

from .persistence import LocalStateLayout, StateFileLock, StateLockTimeoutError
from .routing_runtime import (
    RuntimeRoutingError,
    RuntimeSnapResult,
    SnapThresholds,
    _geometry,
    _identifier,
    _nonnegative_finite,
    _point,
    _positive_finite,
    _query_hash,
    _records,
    _square_matrix,
)

ORS_ORIGIN = "https://api.heigit.org"
ORS_BASE_URL = ORS_ORIGIN + "/openrouteservice"
_PATHS = frozenset({"/v2/snap/driving-car", "/v2/directions/driving-car/geojson", "/v2/matrix/driving-car"})
_SAFE_TRANSPORT_CODES = frozenset(
    {
        "routing_redirect_not_allowed",
        "routing_http_error",
        "routing_unauthorized",
        "routing_forbidden_or_quota_exceeded",
        "routing_rate_limited",
        "routing_timeout",
        "routing_unavailable",
        "routing_response_invalid",
        "routing_response_too_large",
    }
)


class ORSJsonTransport(Protocol):
    def fetch(
        self,
        url: str,
        *,
        json_body: Mapping[str, Any],
        headers: Mapping[str, str],
        timeout_seconds: float,
        max_response_bytes: int,
        expected_origin: str,
    ) -> Mapping[str, Any]:
        """POST JSON without redirects or exposing request/response details."""


def _http_error(status: int) -> RuntimeRoutingError:
    if 300 <= status < 400:
        code = "routing_redirect_not_allowed"
    else:
        code = {
            401: "routing_unauthorized",
            403: "routing_forbidden_or_quota_exceeded",
            429: "routing_rate_limited",
        }.get(status, "routing_http_error")
    return RuntimeRoutingError(code)


class _RejectORSRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise RuntimeRoutingError("routing_redirect_not_allowed")


class _UrlLibORSTransport:
    def __init__(self) -> None:
        self._opener = urllib.request.build_opener(_RejectORSRedirects)

    def fetch(
        self,
        url: str,
        *,
        json_body: Mapping[str, Any],
        headers: Mapping[str, str],
        timeout_seconds: float,
        max_response_bytes: int,
        expected_origin: str,
    ) -> Mapping[str, Any]:
        if expected_origin != ORS_ORIGIN or url not in {ORS_BASE_URL + path for path in _PATHS}:
            raise RuntimeRoutingError("routing_redirect_not_allowed")
        request = urllib.request.Request(
            url, data=json.dumps(json_body, allow_nan=False).encode("utf-8"), headers=dict(headers), method="POST"
        )
        try:
            with self._opener.open(request, timeout=timeout_seconds) as response:
                if response.geturl() != url:
                    raise RuntimeRoutingError("routing_redirect_not_allowed")
                status = int(response.status)
                if status != 200:
                    raise _http_error(status)
                length = response.headers.get("Content-Length")
                if length is not None:
                    try:
                        declared = int(length)
                    except (TypeError, ValueError):
                        raise RuntimeRoutingError("routing_response_invalid") from None
                    if declared < 0:
                        raise RuntimeRoutingError("routing_response_invalid")
                    if declared > max_response_bytes:
                        raise RuntimeRoutingError("routing_response_too_large")
                if response.headers.get("Content-Encoding", "identity").lower() != "identity":
                    raise RuntimeRoutingError("routing_response_invalid")
                raw = response.read(max_response_bytes + 1)
        except urllib.error.HTTPError as exc:
            code = _http_error(int(exc.code)).code
            exc.close()
            raise RuntimeRoutingError(code) from None
        except TimeoutError:
            raise RuntimeRoutingError("routing_timeout") from None
        except urllib.error.URLError as exc:
            code = "routing_timeout" if isinstance(exc.reason, TimeoutError) else "routing_unavailable"
            raise RuntimeRoutingError(code) from None
        except (OSError, http.client.HTTPException):
            raise RuntimeRoutingError("routing_unavailable") from None
        if len(raw) > max_response_bytes:
            raise RuntimeRoutingError("routing_response_too_large")
        try:
            payload = json.loads(raw)
        except (ValueError, UnicodeError, RecursionError):
            raise RuntimeRoutingError("routing_response_invalid") from None
        if not isinstance(payload, dict):
            raise RuntimeRoutingError("routing_response_invalid")
        return payload


def _integer(value: Any, minimum: int, maximum: int, code: str) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise RuntimeRoutingError(code)
    return value


def _number(value: Any, *, positive: bool = False) -> float:
    if type(value) not in {int, float}:
        raise RuntimeRoutingError("routing_response_invalid")
    validator = _positive_finite if positive else _nonnegative_finite
    try:
        return validator(value, "routing_response_invalid")
    except OverflowError:
        raise RuntimeRoutingError("routing_response_invalid") from None


def _location(value: Any) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2 or any(type(v) not in {int, float} for v in value):
        raise RuntimeRoutingError("routing_response_invalid")
    try:
        return _point((value[1], value[0]))
    except (RuntimeRoutingError, OverflowError):
        raise RuntimeRoutingError("routing_response_invalid") from None


def _snap_distance(raw: tuple[float, float], snapped: tuple[float, float]) -> float:
    # Only snap diagnostics use geodesic distance; never synthesize route costs.
    return float(geodesic(raw, snapped).meters)


class ORSRuntimeRoutingService:
    """Duck-compatible runtime routing with credential-free durable JSON evidence.

    Only driving is supported. Table cells contain costs, not geometry: callers
    must still fetch selected directions and re-solve if their durations differ.
    No health/probe method issues provider traffic; the parent owns readiness.
    """

    provider_id = "openrouteservice"

    def __init__(
        self,
        api_key: str,
        cache_root: str | Path,
        *,
        transport: ORSJsonTransport | None = None,
        timeout_seconds: float = 5.0,
        max_response_bytes: int = 1_000_000,
        max_table_points: int = 25,
        thresholds: SnapThresholds | None = None,
        requests_per_minute: int = 30,
        requests_per_day: int = 500,
        cache_ttl_seconds: float = 1800.0,
        max_cache_entries: int = 128,
        max_cache_bytes: int = 16_000_000,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if not isinstance(api_key, str) or re.fullmatch(r"[A-Za-z0-9._~+/=-]{1,8192}", api_key) is None:
            raise RuntimeRoutingError("invalid_ors_api_key")
        self._api_key = api_key
        self.cache_root = Path(cache_root)
        self.timeout_seconds = _positive_finite(timeout_seconds, "invalid_routing_timeout")
        if self.timeout_seconds > 30:
            raise RuntimeRoutingError("invalid_routing_timeout")
        self.cache_ttl_seconds = _positive_finite(cache_ttl_seconds, "invalid_routing_cache_ttl")
        if self.cache_ttl_seconds > 86400:
            raise RuntimeRoutingError("invalid_routing_cache_ttl")
        self.max_response_bytes = _integer(max_response_bytes, 1, 10_000_000, "invalid_routing_response_limit")
        self.max_table_points = _integer(max_table_points, 2, 25, "invalid_routing_table_limit")
        self.requests_per_minute = _integer(requests_per_minute, 0, 30, "invalid_routing_budget")
        self.requests_per_day = _integer(requests_per_day, 0, 500, "invalid_routing_budget")
        self.max_cache_entries = _integer(max_cache_entries, 1, 1024, "invalid_routing_cache_limit")
        self.max_cache_bytes = _integer(max_cache_bytes, 1, 128_000_000, "invalid_routing_cache_limit")
        self.thresholds = thresholds or SnapThresholds()
        self._transport = transport or _UrlLibORSTransport()
        self._clock = clock

    def nearest(self, entity_id: str, point: tuple[float, float]) -> RuntimeSnapResult:
        entity, raw = _identifier(entity_id), _point(point)

        def parse(payload):
            locations = payload.get("locations")
            if not isinstance(locations, list) or len(locations) != 1:
                raise RuntimeRoutingError("routing_response_invalid")
            record = locations[0]
            if record is None:
                return None, None
            if not isinstance(record, Mapping):
                raise RuntimeRoutingError("routing_response_invalid")
            snapped = _location(record.get("location"))
            distance = max(_number(record.get("snapped_distance")), _snap_distance(raw, snapped))
            return snapped, distance

        (snapped, distance), _, _ = self._fetch(
            "/v2/snap/driving-car",
            {"locations": [[raw[1], raw[0]]], "radius": self.thresholds.warning_max_m},
            parse,
        )
        if distance is None or distance > self.thresholds.warning_max_m:
            state, code, confirmation, allowed = "rejected", "map_snap_too_far", False, False
        elif distance > self.thresholds.normal_max_m:
            state, code, confirmation, allowed = "warning", "snap_confirmation_required", True, True
        else:
            state, code, confirmation, allowed = "normal", "snap_within_normal_threshold", False, True
        return RuntimeSnapResult(
            entity_id=entity,
            snapped_point=snapped,
            snap_distance_m=distance,
            provider=self.provider_id,
            status=state,
            raw_point=raw,
            validation_state=state,
            code=code,
            confirmation_required=confirmation,
            draft_append_allowed=allowed,
        )

    def route(self, request: RouteLegRequest) -> RouteLegResult:
        origin_id, destination_id = _identifier(request.origin_id), _identifier(request.destination_id)
        origin, destination = _point(request.origin_point), _point(request.destination_point)
        self._profile(request.routing_profile)

        def parse(payload):
            if payload.get("type") != "FeatureCollection":
                raise RuntimeRoutingError("routing_response_invalid")
            feature = _records(payload, "features", exact_length=1)[0]
            properties = feature.get("properties")
            if feature.get("type") != "Feature" or not isinstance(properties, Mapping):
                raise RuntimeRoutingError("routing_response_invalid")
            if properties.get("warnings"):
                raise RuntimeRoutingError("routing_provider_rejected_request")
            summary = properties.get("summary")
            if not isinstance(summary, Mapping):
                raise RuntimeRoutingError("routing_response_invalid")
            distance, duration = (
                _number(summary.get("distance"), positive=True),
                _number(summary.get("duration"), positive=True),
            )
            shape = feature.get("geometry")
            # The shared geometry helper predates strict boolean/type validation.
            if isinstance(shape, Mapping) and isinstance(shape.get("coordinates"), list):
                for coordinate in shape["coordinates"]:
                    _location(coordinate)
            geometry = _geometry(shape)
            indices = properties.get("way_points")
            if (
                not isinstance(indices, list)
                or any(type(index) is not int for index in indices)
                or indices != [0, len(geometry) - 1]
            ):
                raise RuntimeRoutingError("routing_response_invalid")
            snaps = (_snap_distance(origin, geometry[0]), _snap_distance(destination, geometry[-1]))
            if max(snaps) > self.thresholds.warning_max_m:
                raise RuntimeRoutingError("route_endpoint_snap_too_far")
            return distance, duration, geometry, snaps

        (distance, duration, geometry, snaps), fetched, query_hash = self._fetch(
            "/v2/directions/driving-car/geojson",
            {
                "coordinates": [[origin[1], origin[0]], [destination[1], destination[0]]],
                "instructions": False,
                "elevation": False,
                "units": "m",
                "radiuses": [min(350.0, self.thresholds.warning_max_m)] * 2,
            },
            parse,
        )
        return RouteLegResult(
            origin_id=origin_id,
            destination_id=destination_id,
            geometry=geometry,
            distance_m=distance,
            duration_s=duration,
            routing_status="ors_route_validated",
            provider=self.provider_id,
            routing_profile="driving",
            geometry_source="runtime_ors_geojson",
            distance_source="runtime_ors_route",
            duration_source="runtime_ors_route",
            road_validated=True,
            fallback_used=False,
            query_hash=query_hash,
            retrieved_at=fetched,
            cache_created_at=fetched,
            snap_distance_origin_m=snaps[0],
            snap_distance_destination_m=snaps[-1],
        )

    def table(self, request: RouteMatrixRequest) -> RouteMatrix:
        self._profile(request.routing_profile)
        if not isinstance(request.entity_points, dict):
            raise RuntimeRoutingError("invalid_routing_points")
        if not 2 <= len(request.entity_points) <= self.max_table_points:
            raise RuntimeRoutingError("routing_table_point_limit")
        items = sorted((_identifier(key), _point(point)) for key, point in request.entity_points.items())
        if len({key for key, _ in items}) != len(items):
            raise RuntimeRoutingError("invalid_routing_points")
        size = len(items)

        def parse(payload):
            durations = _square_matrix(payload.get("durations"), size)
            distances = _square_matrix(payload.get("distances"), size)
            endpoints = []
            for name in ("sources", "destinations"):
                records = payload.get(name)
                if not isinstance(records, list) or len(records) != size:
                    raise RuntimeRoutingError("routing_response_invalid")
                present = []
                for record, (_, point) in zip(records, items, strict=True):
                    present.append(record is not None)
                    if record is None:
                        continue
                    if not isinstance(record, Mapping):
                        raise RuntimeRoutingError("routing_response_invalid")
                    snap = max(
                        _number(record.get("snapped_distance")),
                        _snap_distance(point, _location(record.get("location"))),
                    )
                    if snap > self.thresholds.warning_max_m:
                        raise RuntimeRoutingError("table_endpoint_snap_too_far")
                endpoints.append(present)
            values = {}
            for i in range(size):
                for j in range(size):
                    duration = None if durations[i][j] is None else _number(durations[i][j], positive=i != j)
                    distance = None if distances[i][j] is None else _number(distances[i][j], positive=i != j)
                    if i == j:
                        if duration not in {None, 0.0} or distance not in {None, 0.0}:
                            raise RuntimeRoutingError("routing_response_invalid")
                        continue
                    if duration is None or distance is None:
                        continue
                    if not endpoints[0][i] or not endpoints[1][j]:
                        raise RuntimeRoutingError("routing_response_invalid")
                    values[i, j] = (distance, duration)
            return values

        values, _, query_hash = self._fetch(
            "/v2/matrix/driving-car",
            {
                "locations": [[point[1], point[0]] for _, point in items],
                "metrics": ["distance", "duration"],
                "units": "m",
            },
            parse,
        )
        context = str(request.context_snapshot_id)
        cells = {}
        for (i, j), (distance, duration) in values.items():
            origin_id, destination_id = items[i][0], items[j][0]
            cells[origin_id, destination_id] = RouteMatrixCell(
                origin_id=origin_id,
                destination_id=destination_id,
                distance_m=distance,
                duration_s=duration,
                route_leg_id=f"ors_{query_hash}_{i}_{j}",
                road_validated=True,
                provider=self.provider_id,
                query_hash=query_hash,
                context_snapshot_id=context,
                routing_profile="driving",
                routing_status="ors_table_validated",
                geometry_source="not_requested_for_table",
                distance_source="runtime_ors_table",
                duration_source="runtime_ors_table",
            )
        return RouteMatrix(f"runtime_ors_{query_hash}", context, tuple(key for key, _ in items), cells)

    @staticmethod
    def _profile(profile: str) -> None:
        if str(profile).strip().lower() != "driving":
            raise RuntimeRoutingError("routing_profile_not_allowed")

    def _fetch(self, path: str, body: Mapping[str, Any], parse: Callable) -> tuple[Any, datetime, str]:
        endpoint = ORS_BASE_URL + path
        query_hash = _query_hash({"provider": self.provider_id, "endpoint": endpoint, "body": body})
        cache_path = self.cache_root / f"ors-{query_hash}.json"
        try:
            with StateFileLock(self.cache_root / "ors-provider.lock", timeout_seconds=2.0):
                now = _nonnegative_finite(self._clock(), "routing_clock_invalid")
                if cache_path.exists():
                    record = self._read(cache_path, self.max_cache_bytes)
                    if (
                        record.get("schema_version") != "ors-response-v1"
                        or record.get("query_hash") != query_hash
                        or record.get("endpoint") != endpoint
                    ):
                        raise RuntimeRoutingError("routing_cache_invalid")
                    fetched = _number(record.get("fetched_at"))
                    if fetched > now:
                        raise RuntimeRoutingError("routing_cache_invalid")
                    if now - fetched <= self.cache_ttl_seconds:
                        payload = record.get("payload")
                        self._checked_payload(payload)
                        return parse(payload), datetime.fromtimestamp(fetched, UTC), query_hash
                self._reserve(now)
                try:
                    payload = self._transport.fetch(
                        endpoint,
                        json_body=body,
                        headers={
                            "Authorization": self._api_key,
                            "Content-Type": "application/json",
                            "Accept": "application/json",
                        },
                        timeout_seconds=self.timeout_seconds,
                        max_response_bytes=self.max_response_bytes,
                        expected_origin=ORS_ORIGIN,
                    )
                except RuntimeRoutingError as exc:
                    code = exc.code if exc.code in _SAFE_TRANSPORT_CODES else "routing_transport_failure"
                    raise RuntimeRoutingError(code) from None
                except Exception:
                    raise RuntimeRoutingError("routing_transport_failure") from None
                self._checked_payload(payload)
                parsed = parse(payload)
                fetched = _nonnegative_finite(self._clock(), "routing_clock_invalid")
                if fetched < now:
                    raise RuntimeRoutingError("routing_clock_invalid")
                record = {
                    "schema_version": "ors-response-v1",
                    "query_hash": query_hash,
                    "endpoint": endpoint,
                    "fetched_at": fetched,
                    "payload": payload,
                }
                self._make_room(cache_path, record)
                LocalStateLayout._atomic_json_write(cache_path, record)
                return parsed, datetime.fromtimestamp(fetched, UTC), query_hash
        except StateLockTimeoutError:
            raise RuntimeRoutingError("routing_cache_busy") from None
        except OSError:
            raise RuntimeRoutingError("routing_cache_unavailable") from None

    def _checked_payload(self, payload: Any) -> None:
        if not isinstance(payload, dict):
            raise RuntimeRoutingError("routing_response_invalid")
        try:
            encoded = json.dumps(payload, allow_nan=False, ensure_ascii=True).encode("utf-8")
        except (TypeError, ValueError, RecursionError):
            raise RuntimeRoutingError("routing_response_invalid") from None
        if len(encoded) > self.max_response_bytes:
            raise RuntimeRoutingError("routing_response_too_large")
        if self._api_key.encode() in encoded:
            raise RuntimeRoutingError("routing_response_invalid")
        if "error" in payload or "errors" in payload:
            raise RuntimeRoutingError("routing_provider_rejected_request")

    @staticmethod
    def _read(path: Path, limit: int) -> dict:
        if path.is_symlink():
            raise RuntimeRoutingError("routing_cache_invalid")
        with path.open("rb") as handle:
            raw = handle.read(limit + 1)
        if len(raw) > limit:
            raise RuntimeRoutingError("routing_cache_invalid")
        try:
            record = json.loads(raw)
        except (ValueError, UnicodeError, RecursionError):
            raise RuntimeRoutingError("routing_cache_invalid") from None
        if not isinstance(record, dict):
            raise RuntimeRoutingError("routing_cache_invalid")
        return record

    def _reserve(self, now: float) -> None:
        path = self.cache_root / "ors-budget.json"
        stamps = []
        if path.exists():
            record = self._read(path, 32_000)
            stamps = record.get("requests")
            if record.get("schema_version") != "ors-budget-v1" or not isinstance(stamps, list) or len(stamps) > 500:
                raise RuntimeRoutingError("routing_budget_state_invalid")
            if any(type(t) not in {int, float} or t < 0 or t > now or not math.isfinite(t) for t in stamps):
                raise RuntimeRoutingError("routing_budget_state_invalid")
            stamps = [t for t in stamps if t > now - 86400]
        if len(stamps) >= self.requests_per_day:
            raise RuntimeRoutingError("routing_daily_budget_exceeded")
        if sum(t > now - 60 for t in stamps) >= self.requests_per_minute:
            raise RuntimeRoutingError("routing_minute_budget_exceeded")
        LocalStateLayout._atomic_json_write(path, {"schema_version": "ors-budget-v1", "requests": [*stamps, now]})

    def _make_room(self, target: Path, record: dict) -> None:
        # Match the existing atomic writer's representation, including its newline.
        size = (
            len(json.dumps(record, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True).encode("utf-8")) + 1
        )
        if size > self.max_cache_bytes:
            raise RuntimeRoutingError("routing_cache_limit_exceeded")
        entries = []
        for path in self.cache_root.glob("ors-*.json"):
            if re.fullmatch(r"ors-[a-f0-9]{64}\.json", path.name) is None or path == target:
                continue
            if path.is_symlink():
                raise RuntimeRoutingError("routing_cache_invalid")
            stat = path.stat()
            entries.append((stat.st_mtime_ns, path, stat.st_size))
        total = sum(entry[2] for entry in entries) + size
        entries.sort()
        while entries and (len(entries) >= self.max_cache_entries or total > self.max_cache_bytes):
            _, path, old_size = entries.pop(0)
            path.unlink()
            total -= old_size
