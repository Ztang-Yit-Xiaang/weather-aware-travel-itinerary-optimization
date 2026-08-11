from __future__ import annotations

import io
import json
from collections.abc import Callable, Mapping
from typing import Any

import pytest

from itinerary_system.product_app.routing_runtime import (
    RuntimeRoutingError,
    RuntimeRoutingService,
    SnapThresholds,
    _UrlLibRoutingTransport,
)
from itinerary_system.routing.provider import RouteLegRequest, RouteMatrixRequest


class FakeTransport:
    def __init__(self, response: Mapping[str, Any] | Callable[[str], Mapping[str, Any]]) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    def fetch(
        self,
        url: str,
        *,
        timeout_seconds: float,
        max_response_bytes: int,
        expected_origin: str,
    ) -> Mapping[str, Any]:
        self.calls.append(
            {
                "url": url,
                "timeout_seconds": timeout_seconds,
                "max_response_bytes": max_response_bytes,
                "expected_origin": expected_origin,
            }
        )
        return self.response(url) if callable(self.response) else self.response


def nearest_payload(distance: float) -> dict[str, Any]:
    return {
        "code": "Ok",
        "waypoints": [
            {
                "distance": distance,
                "location": [-118.2495, 34.0505],
            }
        ],
    }


def route_payload(*, endpoint_snap_distance: float = 12.0) -> dict[str, Any]:
    return {
        "code": "Ok",
        "routes": [
            {
                "distance": 1_295.7,
                "duration": 168.5,
                "geometry": {
                    "type": "LineString",
                    "coordinates": [
                        [-118.2500, 34.0500],
                        [-118.2450, 34.0525],
                        [-118.2400, 34.0550],
                    ],
                },
            }
        ],
        "waypoints": [
            {"distance": endpoint_snap_distance, "location": [-118.2500, 34.0500]},
            {"distance": 20.0, "location": [-118.2400, 34.0550]},
        ],
    }


def table_payload() -> dict[str, Any]:
    return {
        "code": "Ok",
        "durations": [[0.0, 300.0, None], [310.0, 0.0, 420.0], [None, 415.0, 0.0]],
        "distances": [[0.0, 2_000.0, None], [2_100.0, 0.0, 3_100.0], [None, 3_000.0, 0.0]],
        "sources": [
            {"distance": 10.0, "location": [-118.25, 34.05]},
            {"distance": 20.0, "location": [-118.20, 34.10]},
            {"distance": 30.0, "location": [-118.15, 34.15]},
        ],
        "destinations": [
            {"distance": 11.0, "location": [-118.25, 34.05]},
            {"distance": 21.0, "location": [-118.20, 34.10]},
            {"distance": 31.0, "location": [-118.15, 34.15]},
        ],
    }


@pytest.mark.parametrize(
    "base_url",
    [
        "https://127.0.0.1:5000",
        "http://router.project-osrm.org:5000",
        "http://user@127.0.0.1:5000",
        "http://127.0.0.1",
        "http://127.0.0.1:5000/path",
        "http://127.0.0.1:5000?query=true",
        "http://127.0.0.1:5000#fragment",
        "http://127.0.0.1:99999",
    ],
)
def test_runtime_router_rejects_every_non_loopback_or_ambiguous_base_url(base_url: str) -> None:
    with pytest.raises(RuntimeRoutingError, match="routing_base_url_not_loopback") as error:
        RuntimeRoutingService(base_url=base_url)
    assert error.value.code == "routing_base_url_not_loopback"
    assert base_url not in str(error.value)


def test_runtime_router_accepts_only_explicit_bounded_configuration() -> None:
    service = RuntimeRoutingService(
        base_url="http://localhost:5000/",
        timeout_seconds=1.5,
        max_response_bytes=2_048,
        max_table_points=3,
    )
    assert service.base_url == "http://localhost:5000"
    assert service.origin == "http://localhost:5000"

    for kwargs, code in (
        ({"timeout_seconds": 0}, "invalid_routing_timeout"),
        ({"max_response_bytes": 0}, "invalid_routing_response_limit"),
        ({"max_response_bytes": True}, "invalid_routing_response_limit"),
        ({"max_table_points": 1}, "invalid_routing_table_limit"),
        ({"allowed_profiles": ("driving", "driving")}, "invalid_routing_profiles"),
        ({"allowed_profiles": ("../driving",)}, "invalid_routing_profiles"),
        (
            {"thresholds": SnapThresholds(normal_max_m=100, warning_max_m=500)},
            None,
        ),
    ):
        if code is None:
            RuntimeRoutingService(**kwargs)
            continue
        with pytest.raises(RuntimeRoutingError, match=code):
            RuntimeRoutingService(**kwargs)


@pytest.mark.parametrize(
    ("distance", "state", "code", "confirmation", "append_allowed"),
    [
        (0.0, "normal", "snap_within_normal_threshold", False, True),
        (100.0, "normal", "snap_within_normal_threshold", False, True),
        (100.001, "warning", "snap_confirmation_required", True, True),
        (500.0, "warning", "snap_confirmation_required", True, True),
        (500.001, "rejected", "map_snap_too_far", False, False),
    ],
)
def test_nearest_classifies_snap_thresholds_without_changing_coordinates(
    distance: float,
    state: str,
    code: str,
    confirmation: bool,
    append_allowed: bool,
) -> None:
    transport = FakeTransport(nearest_payload(distance))
    result = RuntimeRoutingService(transport=transport).nearest("stop_a", (34.05, -118.25))

    assert result.entity_id == "stop_a"
    assert result.raw_point == (34.05, -118.25)
    assert result.snapped_point == (34.0505, -118.2495)
    assert result.snap_distance_m == distance
    assert result.validation_state == state
    assert result.code == code
    assert result.confirmation_required is confirmation
    assert result.draft_append_allowed is append_allowed
    assert transport.calls == [
        {
            "url": "http://127.0.0.1:5000/nearest/v1/driving/-118.25,34.05?number=1",
            "timeout_seconds": 3.0,
            "max_response_bytes": 1_000_000,
            "expected_origin": "http://127.0.0.1:5000",
        }
    ]


@pytest.mark.parametrize(
    "point",
    [
        (91.0, 0.0),
        (0.0, 181.0),
        (float("nan"), 0.0),
        (True, 0.0),
        (0.0,),
        "34,-118",
    ],
)
def test_nearest_rejects_invalid_coordinates_before_transport(point: Any) -> None:
    transport = FakeTransport(nearest_payload(1.0))
    with pytest.raises(RuntimeRoutingError, match="invalid_routing_coordinate"):
        RuntimeRoutingService(transport=transport).nearest("stop_a", point)
    assert transport.calls == []


def test_route_returns_road_validated_geometry_and_provenance() -> None:
    transport = FakeTransport(route_payload())
    service = RuntimeRoutingService(
        transport=transport,
        timeout_seconds=1.25,
        max_response_bytes=4_096,
    )
    result = service.route(
        RouteLegRequest(
            origin_id="stop_a",
            destination_id="stop_b",
            origin_point=(34.05, -118.25),
            destination_point=(34.055, -118.24),
        )
    )

    assert result.origin_id == "stop_a"
    assert result.destination_id == "stop_b"
    assert result.geometry == (
        (34.05, -118.25),
        (34.0525, -118.245),
        (34.055, -118.24),
    )
    assert result.distance_m == 1_295.7
    assert result.duration_s == 168.5
    assert result.road_validated is True
    assert result.fallback_used is False
    assert result.provider == "runtime_osrm"
    assert result.snap_distance_origin_m == 12.0
    assert result.snap_distance_destination_m == 20.0
    assert len(result.query_hash) == 64
    assert transport.calls[0] == {
        "url": (
            "http://127.0.0.1:5000/route/v1/driving/"
            "-118.25,34.05;-118.24,34.055?overview=full&geometries=geojson&steps=false"
        ),
        "timeout_seconds": 1.25,
        "max_response_bytes": 4_096,
        "expected_origin": "http://127.0.0.1:5000",
    }


def test_route_rejects_far_endpoint_and_unsupported_profile_without_provider_leakage() -> None:
    far = RuntimeRoutingService(transport=FakeTransport(route_payload(endpoint_snap_distance=500.01)))
    request = RouteLegRequest("a", "b", (34.0, -118.0), (34.1, -118.1))
    with pytest.raises(RuntimeRoutingError, match="route_endpoint_snap_too_far"):
        far.route(request)

    transport = FakeTransport(route_payload())
    with pytest.raises(RuntimeRoutingError, match="routing_profile_not_allowed"):
        RuntimeRoutingService(transport=transport).route(
            RouteLegRequest("a", "b", (34.0, -118.0), (34.1, -118.1), "walking")
        )
    assert transport.calls == []


def test_table_builds_directed_matrix_and_preserves_unavailable_cells() -> None:
    transport = FakeTransport(table_payload())
    service = RuntimeRoutingService(transport=transport, max_table_points=3)
    matrix = service.table(
        RouteMatrixRequest(
            context_snapshot_id="context_v1",
            entity_points={
                "c": (34.15, -118.15),
                "a": (34.05, -118.25),
                "b": (34.10, -118.20),
            },
        )
    )

    assert matrix.entity_ids == ("a", "b", "c")
    assert matrix.context_snapshot_id == "context_v1"
    assert matrix.cell("a", "b").duration_s == 300.0
    assert matrix.cell("a", "b").distance_m == 2_000.0
    assert matrix.cell("a", "b").road_validated is True
    assert matrix.cell("a", "c").duration_s is None
    assert matrix.cell("a", "c").distance_m is None
    assert matrix.cell("a", "c").road_validated is False
    assert matrix.cell("a", "c").routing_status == "osrm_table_unavailable"
    assert transport.calls[0]["url"] == (
        "http://127.0.0.1:5000/table/v1/driving/"
        "-118.25,34.05;-118.2,34.1;-118.15,34.15?annotations=duration,distance"
    )


def test_table_enforces_point_and_endpoint_snap_limits() -> None:
    transport = FakeTransport(table_payload())
    service = RuntimeRoutingService(transport=transport, max_table_points=3)
    with pytest.raises(RuntimeRoutingError, match="routing_table_point_limit"):
        service.table(RouteMatrixRequest("context", {"a": (34.0, -118.0)}))
    assert transport.calls == []

    payload = table_payload()
    payload["sources"][1]["distance"] = 500.01
    with pytest.raises(RuntimeRoutingError, match="table_endpoint_snap_too_far"):
        RuntimeRoutingService(transport=FakeTransport(payload), max_table_points=3).table(
            RouteMatrixRequest(
                "context",
                {"a": (34.0, -118.0), "b": (34.1, -118.1), "c": (34.2, -118.2)},
            )
        )


def test_table_rejects_zero_off_diagonal_values_with_a_stable_error() -> None:
    payload = table_payload()
    payload["durations"][0][1] = 0.0
    with pytest.raises(RuntimeRoutingError, match="routing_response_invalid") as error:
        RuntimeRoutingService(transport=FakeTransport(payload), max_table_points=3).table(
            RouteMatrixRequest(
                "context",
                {"a": (34.0, -118.0), "b": (34.1, -118.1), "c": (34.2, -118.2)},
            )
        )
    assert error.value.code == "routing_response_invalid"


@pytest.mark.parametrize(
    ("payload", "code"),
    [
        ({"code": "NoRoute", "message": "sensitive provider detail"}, "routing_provider_rejected_request"),
        ({"code": "Ok", "routes": []}, "routing_response_invalid"),
        (
            {
                "code": "Ok",
                "routes": [{"distance": 1, "duration": 1, "geometry": {"type": "Point"}}],
                "waypoints": [{"distance": 1}, {"distance": 1}],
            },
            "routing_response_invalid",
        ),
    ],
)
def test_provider_failures_return_only_stable_codes(payload: dict[str, Any], code: str) -> None:
    with pytest.raises(RuntimeRoutingError, match=code) as error:
        RuntimeRoutingService(transport=FakeTransport(payload)).route(
            RouteLegRequest("a", "b", (34.0, -118.0), (34.1, -118.1))
        )
    assert error.value.code == code
    assert "sensitive" not in str(error.value)


def test_unexpected_injected_transport_failure_is_sanitized() -> None:
    class BrokenTransport:
        def fetch(self, *args: Any, **kwargs: Any) -> Mapping[str, Any]:
            raise RuntimeError("raw URL, path, and credentials must not escape")

    with pytest.raises(RuntimeRoutingError, match="routing_transport_failure") as error:
        RuntimeRoutingService(transport=BrokenTransport()).nearest("a", (34.0, -118.0))
    assert str(error.value) == "routing_transport_failure"


class FakeResponse(io.BytesIO):
    def __init__(
        self,
        body: bytes,
        *,
        url: str = "http://127.0.0.1:5000/route",
        status: int = 200,
        content_length: str | None = None,
    ) -> None:
        super().__init__(body)
        self.status = status
        self.headers = {} if content_length is None else {"Content-Length": content_length}
        self._url = url

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def geturl(self) -> str:
        return self._url


class FakeOpener:
    def __init__(self, response: FakeResponse | BaseException) -> None:
        self.response = response
        self.requests: list[tuple[Any, float]] = []

    def open(self, request: Any, timeout: float) -> FakeResponse:
        self.requests.append((request, timeout))
        if isinstance(self.response, BaseException):
            raise self.response
        return self.response


def test_default_transport_enforces_redirect_body_limit_and_timeout() -> None:
    payload = json.dumps({"code": "Ok"}).encode()

    redirect_transport = _UrlLibRoutingTransport()
    redirect_transport._opener = FakeOpener(
        FakeResponse(payload, url="http://127.0.0.1:5001/redirected")
    )
    with pytest.raises(RuntimeRoutingError, match="routing_redirect_not_allowed"):
        redirect_transport.fetch(
            "http://127.0.0.1:5000/route",
            timeout_seconds=1,
            max_response_bytes=100,
            expected_origin="http://127.0.0.1:5000",
        )

    large_transport = _UrlLibRoutingTransport()
    large_transport._opener = FakeOpener(FakeResponse(b"x" * 101))
    with pytest.raises(RuntimeRoutingError, match="routing_response_too_large"):
        large_transport.fetch(
            "http://127.0.0.1:5000/route",
            timeout_seconds=1,
            max_response_bytes=100,
            expected_origin="http://127.0.0.1:5000",
        )

    declared_large = _UrlLibRoutingTransport()
    declared_large._opener = FakeOpener(FakeResponse(b"{}", content_length="101"))
    with pytest.raises(RuntimeRoutingError, match="routing_response_too_large"):
        declared_large.fetch(
            "http://127.0.0.1:5000/route",
            timeout_seconds=1,
            max_response_bytes=100,
            expected_origin="http://127.0.0.1:5000",
        )

    negative_length = _UrlLibRoutingTransport()
    negative_length._opener = FakeOpener(FakeResponse(b"{}", content_length="-1"))
    with pytest.raises(RuntimeRoutingError, match="routing_response_invalid"):
        negative_length.fetch(
            "http://127.0.0.1:5000/route",
            timeout_seconds=1,
            max_response_bytes=100,
            expected_origin="http://127.0.0.1:5000",
        )

    timed_out = _UrlLibRoutingTransport()
    timed_out._opener = FakeOpener(TimeoutError("raw timeout detail"))
    with pytest.raises(RuntimeRoutingError, match="routing_timeout") as error:
        timed_out.fetch(
            "http://127.0.0.1:5000/route",
            timeout_seconds=1,
            max_response_bytes=100,
            expected_origin="http://127.0.0.1:5000",
        )
    assert str(error.value) == "routing_timeout"
