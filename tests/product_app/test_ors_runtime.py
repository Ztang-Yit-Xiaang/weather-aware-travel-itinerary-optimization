from __future__ import annotations

import io
import json
import traceback
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import UTC, datetime

import pytest

from itinerary_system.product_app.ors_runtime import (
    ORS_BASE_URL,
    ORS_ORIGIN,
    ORSRuntimeRoutingService,
    _RejectORSRedirects,
    _UrlLibORSTransport,
)
from itinerary_system.product_app.persistence import LocalStateLayout, StateFileLock
from itinerary_system.product_app.routing_runtime import RuntimeRoutingError, SnapThresholds, _query_hash
from itinerary_system.routing.provider import RouteLegRequest, RouteMatrixRequest

KEY = "ors-test-key-keep-private"
A = (34.05, -118.25)
B = (34.055, -118.24)
C = (34.06, -118.23)
LEG = RouteLegRequest("a", "b", A, B)


class FakeTransport:
    def __init__(self, payload=None, error=None):
        self.payload = payload if payload is not None else snap_payload()
        self.error = error
        self.calls = []

    def fetch(self, url, **kwargs):
        self.calls.append({"url": url, **kwargs})
        if self.error:
            raise self.error
        return deepcopy(self.payload)


def snap_payload(distance=10):
    return {"locations": [{"location": [A[1], A[0]], "snapped_distance": distance}]}


def route_payload():
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": [[A[1], A[0]], [B[1], B[0]]]},
                "properties": {"summary": {"distance": 1295.7, "duration": 168.5}, "way_points": [0, 1]},
            }
        ],
    }


def matrix_payload():
    locations = [{"location": [point[1], point[0]], "snapped_distance": 10} for point in (A, B, C)]
    return {
        "durations": [[0, 300, None], [310, 0, 420], [None, 415, 0]],
        "distances": [[0, 2000, None], [2100, 0, 3100], [None, 3000, 0]],
        "sources": deepcopy(locations),
        "destinations": deepcopy(locations),
    }


@pytest.fixture
def clock():
    return [1_800_000_000.0]


def service(root, transport, clock, **kwargs):
    return ORSRuntimeRoutingService(KEY, root, transport=transport, clock=lambda: clock[0], **kwargs)


@pytest.mark.parametrize(
    "distance,state,code,confirmation,allowed",
    [
        (0, "normal", "snap_within_normal_threshold", False, True),
        (100, "normal", "snap_within_normal_threshold", False, True),
        (100.001, "warning", "snap_confirmation_required", True, True),
        (500, "warning", "snap_confirmation_required", True, True),
        (500.001, "rejected", "map_snap_too_far", False, False),
    ],
)
def test_snap_semantics_and_post_coordinate_order(tmp_path, clock, distance, state, code, confirmation, allowed):
    transport = FakeTransport(snap_payload(distance))
    result = service(tmp_path, transport, clock).nearest("stop", A)
    assert result.raw_point == result.snapped_point == A
    assert result.snap_distance_m == distance
    assert result.provider == "openrouteservice"
    assert result.status == result.validation_state == state
    assert result.code == code
    assert result.confirmation_required is confirmation
    assert result.draft_append_allowed is allowed
    call = transport.calls[0]
    assert call["url"] == "https://api.heigit.org/openrouteservice/v2/snap/driving-car"
    assert call["json_body"] == {"locations": [[-118.25, 34.05]], "radius": 500}
    assert call["headers"]["Authorization"] == KEY
    assert call["expected_origin"] == ORS_ORIGIN
    assert call["timeout_seconds"] == 5
    assert call["max_response_bytes"] == 1_000_000
    assert KEY not in call["url"] + json.dumps(call["json_body"])


def test_null_snap_is_rejected_without_inventing_a_point_or_distance(tmp_path, clock):
    result = service(tmp_path, FakeTransport({"locations": [None]}), clock).nearest("stop", A)
    assert result.validation_state == "rejected"
    assert result.snapped_point is result.snap_distance_m is None
    assert not result.draft_append_allowed and not result.confirmation_required


def test_snap_cannot_hide_large_coordinate_displacement(tmp_path, clock):
    transport = FakeTransport({"locations": [{"location": [-120, 35], "snapped_distance": 0}]})
    result = service(tmp_path, transport, clock).nearest("stop", A)
    assert result.validation_state == "rejected"
    assert result.snap_distance_m > 500


def test_route_preserves_real_costs_geometry_and_ors_provenance(tmp_path, clock):
    transport = FakeTransport(route_payload())
    result = service(tmp_path, transport, clock).route(LEG)
    assert result.geometry == (A, B)
    assert (result.distance_m, result.duration_s) == (1295.7, 168.5)
    assert result.snap_distance_origin_m == result.snap_distance_destination_m == 0
    assert result.road_validated and not result.fallback_used
    assert result.provider == "openrouteservice"
    assert result.routing_status == "ors_route_validated"
    assert result.geometry_source == "runtime_ors_geojson"
    assert result.distance_source == result.duration_source == "runtime_ors_route"
    assert "osrm" not in repr(result)
    assert result.routing_profile == "driving"
    call = transport.calls[0]
    assert call["url"] == "https://api.heigit.org/openrouteservice/v2/directions/driving-car/geojson"
    assert call["json_body"]["coordinates"] == [[-118.25, 34.05], [-118.24, 34.055]]
    assert call["json_body"]["radiuses"] == [350, 350]
    assert call["json_body"]["instructions"] is False
    assert result.retrieved_at == datetime.fromtimestamp(clock[0], UTC)


def test_hosted_keyword_interface_exposes_and_applies_custom_thresholds(tmp_path, clock):
    payload = route_payload()
    payload["features"][0]["geometry"]["coordinates"][0][1] += 0.001
    thresholds = SnapThresholds(25, 100)
    transport = FakeTransport(payload)
    routing = ORSRuntimeRoutingService(
        api_key=KEY, cache_root=tmp_path, transport=transport, clock=lambda: clock[0], thresholds=thresholds
    )
    assert routing.thresholds is thresholds
    with pytest.raises(RuntimeRoutingError, match="route_endpoint_snap_too_far"):
        routing.route(LEG)
    assert transport.calls[0]["json_body"]["radiuses"] == [100] * 2


@pytest.mark.parametrize("endpoint", [0, 1])
def test_route_rejects_far_endpoints(tmp_path, clock, endpoint):
    payload = route_payload()
    payload["features"][0]["geometry"]["coordinates"][endpoint] = [-120, 35]
    with pytest.raises(RuntimeRoutingError, match="route_endpoint_snap_too_far"):
        service(tmp_path, FakeTransport(payload), clock).route(LEG)
    assert list(tmp_path.glob("ors-" + "?" * 64 + ".json")) == []


@pytest.mark.parametrize(
    "mutation",
    [
        lambda p: p.update(type="LineString"),
        lambda p: p.update(features=[]),
        lambda p: p["features"].append(deepcopy(p["features"][0])),
        lambda p: p["features"][0]["properties"]["summary"].update(duration=-1),
        lambda p: p["features"][0]["properties"]["summary"].update(distance=True),
        lambda p: p["features"][0]["properties"]["summary"].update(duration="123"),
        lambda p: p["features"][0]["properties"].update(way_points=[1, 0]),
        lambda p: p["features"][0]["properties"].update(way_points=[False, 1]),
        lambda p: p["features"][0]["geometry"].update(type="MultiLineString"),
        lambda p: p["features"][0]["geometry"].update(coordinates=[[True, 34], [-118, 34]]),
        lambda p: p["features"][0]["geometry"].update(coordinates=[[-118, float("nan")], [-118, 34]]),
    ],
)
def test_malformed_directions_fail_closed(tmp_path, clock, mutation):
    payload = route_payload()
    mutation(payload)
    with pytest.raises(RuntimeRoutingError, match="routing_response_invalid"):
        service(tmp_path, FakeTransport(payload), clock).route(LEG)


def test_directions_with_skipped_or_approximate_segment_warning_are_not_evidence(tmp_path, clock):
    payload = route_payload()
    payload["features"][0]["properties"]["warnings"] = [{"code": 3, "message": "skipped segment"}]
    with pytest.raises(RuntimeRoutingError, match="routing_provider_rejected_request"):
        service(tmp_path, FakeTransport(payload), clock).route(LEG)


def test_matrix_is_sorted_directed_and_has_no_geometry_or_unreachable_costs(tmp_path, clock):
    transport = FakeTransport(matrix_payload())
    request = RouteMatrixRequest("context", {"c": C, "b": B, "a": A})
    result = service(tmp_path, transport, clock).table(request)
    assert result.entity_ids == ("a", "b", "c")
    assert result.cells["a", "b"].duration_s == 300
    assert result.cells["b", "a"].duration_s == 310
    assert ("a", "c") not in result.cells and ("c", "a") not in result.cells
    assert len(result.cells) == 4
    for cell in result.cells.values():
        assert cell.geometry == ()
        assert cell.road_validated and not cell.fallback_used
        assert cell.provider == "openrouteservice"
        assert cell.routing_status == "ors_table_validated"
        assert cell.geometry_source == "not_requested_for_table"
        assert cell.distance_source == cell.duration_source == "runtime_ors_table"
    call = transport.calls[0]
    assert call["url"] == "https://api.heigit.org/openrouteservice/v2/matrix/driving-car"
    assert call["json_body"] == {
        "locations": [[-118.25, 34.05], [-118.24, 34.055], [-118.23, 34.06]],
        "metrics": ["distance", "duration"],
        "units": "m",
    }


@pytest.mark.parametrize(
    "mutation,code",
    [
        (lambda p: p.update(durations=[[0, 1], [2, 0]]), "routing_response_invalid"),
        (lambda p: p["distances"][0].append(1), "routing_response_invalid"),
        (lambda p: p["sources"].pop(), "routing_response_invalid"),
        (lambda p: p["durations"][0].__setitem__(0, 1), "routing_response_invalid"),
        (lambda p: p["durations"][0].__setitem__(1, 0), "routing_response_invalid"),
        (lambda p: p["durations"][0].__setitem__(2, -1), "routing_response_invalid"),
        (lambda p: p["distances"][0].__setitem__(1, True), "routing_response_invalid"),
        (lambda p: p["sources"][0].update(snapped_distance=501), "table_endpoint_snap_too_far"),
        (lambda p: p["destinations"][1].update(location=[-120, 35]), "table_endpoint_snap_too_far"),
        (lambda p: p["sources"].__setitem__(0, None), "routing_response_invalid"),
    ],
)
def test_matrix_shape_numeric_and_snap_validation(tmp_path, clock, mutation, code):
    payload = matrix_payload()
    mutation(payload)
    with pytest.raises(RuntimeRoutingError, match=code):
        service(tmp_path, FakeTransport(payload), clock).table(RouteMatrixRequest("ctx", {"a": A, "b": B, "c": C}))


def test_partial_null_cell_and_unsnapped_row_remain_unavailable(tmp_path, clock):
    payload = matrix_payload()
    payload["durations"][0][1] = None
    payload["sources"][0] = None
    result = service(tmp_path, FakeTransport(payload), clock).table(RouteMatrixRequest("ctx", {"a": A, "b": B, "c": C}))
    assert ("a", "b") not in result.cells
    assert result.cells["b", "a"].duration_s == 310


def test_cache_preserves_original_fetch_time_after_restart_and_expiry_is_real_fetch(tmp_path, clock):
    transport = FakeTransport(route_payload())
    initial = service(tmp_path, transport, clock).route(LEG)
    files = list(tmp_path.glob("ors-" + "?" * 64 + ".json"))
    assert len(files) == 1
    record = json.loads(files[0].read_text())
    assert record["payload"] == route_payload()
    assert record["fetched_at"] == clock[0]
    assert record["endpoint"] == "https://api.heigit.org/openrouteservice/v2/directions/driving-car/geojson"
    clock[0] += 61
    cache_only = FakeTransport(error=AssertionError("must not fetch"))
    cached = service(tmp_path, cache_only, clock, requests_per_day=0).route(LEG)
    assert cached.retrieved_at == initial.retrieved_at == cached.cache_created_at
    assert cached.query_hash == initial.query_hash
    assert cache_only.calls == []
    clock[0] += 1800
    refreshed = service(tmp_path, transport, clock).route(LEG)
    assert refreshed.retrieved_at > initial.retrieved_at
    assert len(transport.calls) == 2


def test_current_endpoint_has_distinct_cache_identity_and_preserves_existing_budget(tmp_path, clock):
    body = {"locations": [[A[1], A[0]]], "radius": 500.0}
    legacy_hash = _query_hash({"provider": "openrouteservice", "path": "/v2/snap/driving-car", "body": body})
    legacy_path = tmp_path / f"ors-{legacy_hash}.json"
    legacy_record = {
        "schema_version": "ors-response-v1",
        "query_hash": legacy_hash,
        "fetched_at": clock[0] - 1,
        "payload": snap_payload(),
    }
    legacy_path.write_text(json.dumps(legacy_record))
    (tmp_path / "ors-budget.json").write_text(
        json.dumps({"schema_version": "ors-budget-v1", "requests": [clock[0] - 1]})
    )
    transport = FakeTransport()
    routing = service(tmp_path, transport, clock)
    routing.nearest("a", A)
    endpoint = "https://api.heigit.org/openrouteservice/v2/snap/driving-car"
    assert [call["url"] for call in transport.calls] == [endpoint]
    current_hash = _query_hash({"provider": "openrouteservice", "endpoint": endpoint, "body": body})
    assert current_hash != legacy_hash
    record = json.loads((tmp_path / f"ors-{current_hash}.json").read_text())
    assert record["endpoint"] == endpoint
    assert json.loads(legacy_path.read_text()) == legacy_record
    assert len(json.loads((tmp_path / "ors-budget.json").read_text())["requests"]) == 2
    routing.nearest("a", A)
    assert len(transport.calls) == 1


def test_cache_does_not_relabel_a_new_context_or_entity_and_bounds_entry_count(tmp_path, clock):
    transport = FakeTransport(matrix_payload())
    routing = service(tmp_path, transport, clock)
    first = routing.table(RouteMatrixRequest("ctx1", {"a": A, "b": B, "c": C}))
    second = routing.table(RouteMatrixRequest("ctx2", {"x": A, "y": B, "z": C}))
    assert first.cells["a", "b"].duration_s == second.cells["x", "y"].duration_s
    assert second.cells["x", "y"].context_snapshot_id == "ctx2"
    assert len(transport.calls) == 1
    snap = FakeTransport()
    service(tmp_path, snap, clock, max_cache_entries=1).nearest("a", A)
    assert len(list(tmp_path.glob("ors-" + "?" * 64 + ".json"))) == 1
    assert (tmp_path / "ors-budget.json").is_file()


def test_cache_storage_limit_is_enforced_without_silent_loss_of_budget(tmp_path, clock):
    routing = service(tmp_path, FakeTransport(), clock, max_cache_bytes=10)
    with pytest.raises(RuntimeRoutingError, match="routing_cache_limit_exceeded"):
        routing.nearest("a", A)
    assert len(json.loads((tmp_path / "ors-budget.json").read_text())["requests"]) == 1
    assert not list(tmp_path.glob("ors-" + "?" * 64 + ".json"))


def test_cache_byte_budget_evicts_old_responses_but_keeps_budget(tmp_path, clock):
    transport = FakeTransport()
    service(tmp_path, transport, clock).nearest("a", A)
    size = sum(path.stat().st_size for path in tmp_path.glob("ors-" + "?" * 64 + ".json"))
    service(tmp_path, transport, clock, max_cache_bytes=size + 10).nearest("b", B)
    assert sum(path.stat().st_size for path in tmp_path.glob("ors-" + "?" * 64 + ".json")) <= size + 10
    assert len(json.loads((tmp_path / "ors-budget.json").read_text())["requests"]) == 2


@pytest.mark.parametrize(
    "mutation",
    [
        lambda r: r.update(query_hash="other"),
        lambda r: r.update(endpoint="https://api.openrouteservice.org/v2/snap/driving-car"),
        lambda r: r.update(fetched_at=1_900_000_000),
        lambda r: r.update(payload={"locations": [None], "echo": KEY}),
    ],
)
def test_cache_hits_are_revalidated_without_spending_more_quota(tmp_path, clock, mutation):
    transport = FakeTransport()
    service(tmp_path, transport, clock).nearest("a", A)
    path = next(tmp_path.glob("ors-" + "?" * 64 + ".json"))
    record = json.loads(path.read_text())
    mutation(record)
    path.write_text(json.dumps(record))
    with pytest.raises(RuntimeRoutingError):
        service(tmp_path, transport, clock).nearest("a", A)
    assert len(transport.calls) == 1


def test_expired_cache_is_not_silently_served_when_provider_fails(tmp_path, clock):
    transport = FakeTransport(route_payload())
    service(tmp_path, transport, clock).route(LEG)
    path = next(tmp_path.glob("ors-" + "?" * 64 + ".json"))
    original = path.read_bytes()
    clock[0] += 1801
    transport.error = TimeoutError(KEY)
    with pytest.raises(RuntimeRoutingError, match="routing_transport_failure"):
        service(tmp_path, transport, clock).route(LEG)
    assert path.read_bytes() == original


def test_budget_persistence_failure_prevents_the_provider_call(tmp_path, clock, monkeypatch):
    def fail(*args, **kwargs):
        raise OSError(KEY)

    monkeypatch.setattr(LocalStateLayout, "_atomic_json_write", fail)
    transport = FakeTransport()
    with pytest.raises(RuntimeRoutingError, match="routing_cache_unavailable") as error:
        service(tmp_path, transport, clock).nearest("a", A)
    assert transport.calls == []
    assert KEY not in "".join(traceback.format_exception(error.value))


def test_lock_contention_fails_closed_without_provider_traffic(tmp_path, clock):
    transport = FakeTransport()
    with StateFileLock(tmp_path / "ors-provider.lock"):
        with pytest.raises(RuntimeRoutingError, match="routing_cache_busy"):
            service(tmp_path, transport, clock).nearest("a", A)
    assert transport.calls == []


@pytest.mark.parametrize("field", ["distance", "duration"])
def test_unrepresentable_provider_numbers_are_sanitized(tmp_path, clock, field):
    payload = route_payload()
    payload["features"][0]["properties"]["summary"][field] = 10**400
    with pytest.raises(RuntimeRoutingError, match="routing_response_invalid"):
        service(tmp_path, FakeTransport(payload), clock).route(LEG)


@pytest.mark.parametrize(
    "points,code",
    [
        ({"a": A}, "routing_table_point_limit"),
        ({str(i): A for i in range(26)}, "routing_table_point_limit"),
        ({"a": A, " a ": B}, "invalid_routing_points"),
        ({"a/b": A, "b": B}, "invalid_routing_identifier"),
    ],
)
def test_invalid_matrix_requests_do_not_spend_quota(tmp_path, clock, points, code):
    transport = FakeTransport()
    with pytest.raises(RuntimeRoutingError, match=code):
        service(tmp_path, transport, clock).table(RouteMatrixRequest("ctx", points))
    assert transport.calls == []


def test_rolling_minute_and_daily_budgets_survive_restart_and_failures(tmp_path, clock):
    transport = FakeTransport(error=OSError(KEY))

    def restart():
        return service(tmp_path, transport, clock, requests_per_minute=2, requests_per_day=3)

    for _ in range(2):
        with pytest.raises(RuntimeRoutingError, match="routing_transport_failure"):
            restart().nearest("a", A)
    with pytest.raises(RuntimeRoutingError, match="routing_minute_budget_exceeded"):
        restart().nearest("a", A)
    assert len(transport.calls) == 2
    clock[0] += 60
    with pytest.raises(RuntimeRoutingError, match="routing_transport_failure"):
        restart().nearest("a", A)
    clock[0] += 61
    with pytest.raises(RuntimeRoutingError, match="routing_daily_budget_exceeded"):
        restart().nearest("a", A)
    assert len(transport.calls) == 3
    clock[0] += 86400
    with pytest.raises(RuntimeRoutingError, match="routing_transport_failure"):
        restart().nearest("a", A)
    assert len(transport.calls) == 4


def test_defaults_are_combined_app_limits_and_apply_to_failed_attempts(tmp_path, clock):
    transport = FakeTransport(error=OSError("offline"))
    routing = service(tmp_path, transport, clock)
    assert routing.requests_per_minute == 30
    assert routing.requests_per_day == 500
    for _ in range(30):
        with pytest.raises(RuntimeRoutingError, match="routing_transport_failure"):
            routing.nearest("a", A)
    with pytest.raises(RuntimeRoutingError, match="routing_minute_budget_exceeded"):
        service(tmp_path, transport, clock).nearest("a", A)
    assert len(transport.calls) == 30


def test_all_operations_share_the_same_persisted_request_budget(tmp_path, clock):
    transport = FakeTransport()
    routing = service(tmp_path, transport, clock, requests_per_minute=3, requests_per_day=4)
    routing.nearest("a", A)
    transport.payload = route_payload()
    routing.route(LEG)
    transport.payload = matrix_payload()
    routing.table(RouteMatrixRequest("ctx", {"a": A, "b": B, "c": C}))
    with pytest.raises(RuntimeRoutingError, match="routing_minute_budget_exceeded"):
        routing.nearest("b", B)
    clock[0] += 60
    transport.payload = snap_payload()
    routing.nearest("b", B)
    clock[0] += 60
    with pytest.raises(RuntimeRoutingError, match="routing_daily_budget_exceeded"):
        routing.nearest("c", C)
    assert len(transport.calls) == 4


def test_shared_root_serializes_budget_reservations_across_service_instances(tmp_path, clock):
    transport = FakeTransport(error=OSError("offline"))

    def attempt(_):
        try:
            service(tmp_path, transport, clock, requests_per_day=1).nearest("a", A)
        except RuntimeRoutingError as error:
            return error.code

    with ThreadPoolExecutor(max_workers=4) as pool:
        codes = list(pool.map(attempt, range(4)))
    assert codes.count("routing_transport_failure") == 1
    assert codes.count("routing_daily_budget_exceeded") == 3
    assert len(transport.calls) == 1


@pytest.mark.parametrize(
    "record",
    [
        {},
        {"schema_version": "ors-budget-v1", "requests": [float("nan")]},
        {"schema_version": "ors-budget-v1", "requests": [1_900_000_000]},
    ],
)
def test_corrupt_budget_fails_closed(tmp_path, clock, record):
    (tmp_path / "ors-budget.json").write_text(json.dumps(record))
    transport = FakeTransport()
    with pytest.raises(RuntimeRoutingError, match="routing_budget_state_invalid"):
        service(tmp_path, transport, clock).nearest("a", A)
    assert transport.calls == []


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {"locations": []},
        {"locations": [False]},
        {"locations": [{"location": [True, 34], "snapped_distance": 0}]},
        {"locations": [{"location": [-118, 34], "snapped_distance": -1}]},
        {"error": {"message": KEY}},
        {"locations": [None], "echo": KEY},
    ],
)
def test_invalid_or_secret_echo_response_never_enters_cache(tmp_path, clock, payload):
    with pytest.raises(RuntimeRoutingError) as error:
        service(tmp_path, FakeTransport(payload), clock).nearest("a", A)
    assert KEY not in "".join(traceback.format_exception(error.value))
    for path in tmp_path.glob("*.json"):
        assert KEY not in path.read_text()
    assert not list(tmp_path.glob("ors-" + "?" * 64 + ".json"))


@pytest.mark.parametrize("error", [ValueError(KEY), RuntimeRoutingError(KEY), urllib.error.URLError(KEY)])
def test_transport_exception_text_and_chains_are_redacted(tmp_path, clock, error, caplog):
    with pytest.raises(RuntimeRoutingError, match="routing_transport_failure") as captured:
        service(tmp_path, FakeTransport(error=error), clock).nearest("a", A)
    assert captured.value.__cause__ is None
    assert KEY not in "".join(traceback.format_exception(captured.value)) + caplog.text
    assert KEY not in (tmp_path / "ors-budget.json").read_text()


def test_injected_transport_response_size_is_also_bounded(tmp_path, clock):
    with pytest.raises(RuntimeRoutingError, match="routing_response_too_large"):
        service(
            tmp_path, FakeTransport({"locations": [None], "padding": "x" * 1000}), clock, max_response_bytes=128
        ).nearest("a", A)


@pytest.mark.parametrize("point", [(91, 0), (0, 181), (float("inf"), 0), (True, 0), "1,2", (1,)])
def test_invalid_request_coordinates_do_not_spend_quota(tmp_path, clock, point):
    transport = FakeTransport()
    with pytest.raises(RuntimeRoutingError, match="invalid_routing_coordinate"):
        service(tmp_path, transport, clock).nearest("a", point)
    assert transport.calls == []
    assert not (tmp_path / "ors-budget.json").exists()


@pytest.mark.parametrize("profile", ["walking", "driving-car", "../driving", "cycling"])
def test_only_internal_driving_profile_is_allowed(tmp_path, clock, profile):
    transport = FakeTransport()
    routing = service(tmp_path, transport, clock)
    with pytest.raises(RuntimeRoutingError, match="routing_profile_not_allowed"):
        routing.route(RouteLegRequest("a", "b", A, B, profile))
    with pytest.raises(RuntimeRoutingError, match="routing_profile_not_allowed"):
        routing.table(RouteMatrixRequest("ctx", {"a": A, "b": B}, profile))
    assert transport.calls == []


@pytest.mark.parametrize(
    "kwargs",
    [
        {"timeout_seconds": 0},
        {"timeout_seconds": 31},
        {"max_response_bytes": True},
        {"max_response_bytes": 10_000_001},
        {"max_table_points": 26},
        {"requests_per_minute": 31},
        {"requests_per_day": 501},
        {"requests_per_day": -1},
        {"cache_ttl_seconds": 86401},
        {"max_cache_entries": 0},
        {"max_cache_bytes": 0},
    ],
)
def test_configuration_is_bounded(tmp_path, kwargs):
    with pytest.raises(RuntimeRoutingError):
        ORSRuntimeRoutingService(KEY, tmp_path, **kwargs)


@pytest.mark.parametrize("key", ["", "key\r\nInjected: x", " key", "é", None])
def test_invalid_keys_are_rejected_without_echo(tmp_path, key):
    with pytest.raises(RuntimeRoutingError, match="invalid_ors_api_key"):
        ORSRuntimeRoutingService(key, tmp_path)


def test_constructor_never_opens_transport_or_creates_state(tmp_path):
    transport = FakeTransport(error=AssertionError("no probing"))
    ORSRuntimeRoutingService(KEY, tmp_path / "unused", transport=transport, thresholds=SnapThresholds(50, 250))
    assert transport.calls == []
    assert not (tmp_path / "unused").exists()


class HTTPResponse(io.BytesIO):
    def __init__(self, body=b"{}", status=200, url=None, headers=None):
        super().__init__(body)
        self.status = status
        self.url = url or ORS_BASE_URL + "/v2/snap/driving-car"
        self.headers = headers or {}
        self.read_limits = []

    def geturl(self):
        return self.url

    def read(self, size=-1):
        self.read_limits.append(size)
        return super().read(size)


class Opener:
    def __init__(self, response=None, error=None):
        self.response, self.error = response, error
        self.calls = []

    def open(self, request, *, timeout):
        self.calls.append((request, timeout))
        if self.error:
            raise self.error
        return self.response


def http_fetch(opener, *, url=ORS_BASE_URL + "/v2/snap/driving-car", limit=128):
    transport = _UrlLibORSTransport()
    transport._opener = opener
    return transport.fetch(
        url,
        json_body={"locations": [[-118.25, 34.05]], "radius": 500},
        headers={"Authorization": KEY, "Content-Type": "application/json"},
        timeout_seconds=1.5,
        max_response_bytes=limit,
        expected_origin=ORS_ORIGIN,
    )


def test_real_transport_uses_bounded_post_with_header_only_authentication():
    response = HTTPResponse(b'{"locations": [null]}')
    opener = Opener(response)
    assert http_fetch(opener) == {"locations": [None]}
    request, timeout = opener.calls[0]
    assert request.full_url == "https://api.heigit.org/openrouteservice/v2/snap/driving-car"
    assert request.method == "POST"
    assert request.get_header("Authorization") == KEY
    assert KEY not in request.full_url + request.data.decode()
    assert json.loads(request.data)["locations"] == [[-118.25, 34.05]]
    assert timeout == 1.5
    assert response.read_limits == [129]


@pytest.mark.parametrize("status", [301, 401, 403, 429, 503])
def test_current_endpoint_failure_never_retries_or_falls_back_to_legacy(tmp_path, clock, status):
    transport = _UrlLibORSTransport()
    opener = Opener(HTTPResponse(KEY.encode(), status=status))
    transport._opener = opener
    with pytest.raises(RuntimeRoutingError):
        service(tmp_path, transport, clock).nearest("a", A)
    assert [request.full_url for request, _ in opener.calls] == [
        "https://api.heigit.org/openrouteservice/v2/snap/driving-car"
    ]


@pytest.mark.parametrize(
    "status,code",
    [
        (301, "routing_redirect_not_allowed"),
        (307, "routing_redirect_not_allowed"),
        (401, "routing_unauthorized"),
        (403, "routing_forbidden_or_quota_exceeded"),
        (429, "routing_rate_limited"),
        (500, "routing_http_error"),
    ],
)
@pytest.mark.parametrize("as_exception", [False, True])
def test_http_status_errors_are_sanitized(status, code, as_exception):
    opener = (
        Opener(error=urllib.error.HTTPError(ORS_ORIGIN + "/?key=" + KEY, status, KEY, {}, io.BytesIO(KEY.encode())))
        if as_exception
        else Opener(HTTPResponse(KEY.encode(), status=status))
    )
    with pytest.raises(RuntimeRoutingError, match=code) as error:
        http_fetch(opener)
    assert KEY not in "".join(traceback.format_exception(error.value))


@pytest.mark.parametrize(
    "response,code",
    [
        (HTTPResponse(b"x" * 129), "routing_response_too_large"),
        (HTTPResponse(headers={"Content-Length": "129"}), "routing_response_too_large"),
        (HTTPResponse(headers={"Content-Length": "-1"}), "routing_response_invalid"),
        (HTTPResponse(headers={"Content-Length": "invalid"}), "routing_response_invalid"),
        (HTTPResponse(headers={"Content-Encoding": "gzip"}), "routing_response_invalid"),
        (HTTPResponse(b"not json"), "routing_response_invalid"),
        (HTTPResponse(b"\xff"), "routing_response_invalid"),
        (HTTPResponse(b"[]"), "routing_response_invalid"),
        (HTTPResponse(url="https://example.com/"), "routing_redirect_not_allowed"),
        (HTTPResponse(url=ORS_ORIGIN + "/elsewhere"), "routing_redirect_not_allowed"),
    ],
)
def test_http_invalid_oversize_and_redirect_responses(response, code):
    with pytest.raises(RuntimeRoutingError, match=code):
        http_fetch(Opener(response))


@pytest.mark.parametrize(
    "error,code",
    [
        (TimeoutError(KEY), "routing_timeout"),
        (urllib.error.URLError(TimeoutError(KEY)), "routing_timeout"),
        (urllib.error.URLError(KEY), "routing_unavailable"),
        (OSError(KEY), "routing_unavailable"),
    ],
)
def test_http_network_errors_are_sanitized(error, code):
    with pytest.raises(RuntimeRoutingError, match=code) as captured:
        http_fetch(Opener(error=error))
    assert KEY not in "".join(traceback.format_exception(captured.value))


@pytest.mark.parametrize(
    "url",
    [
        "http://api.heigit.org/openrouteservice/v2/snap/driving-car",
        "https://api.openrouteservice.org/v2/snap/driving-car",
        "https://api.openrouteservice.org/v2/directions/driving-car/geojson",
        "https://api.openrouteservice.org/v2/matrix/driving-car",
        "https://api.heigit.org/v2/snap/driving-car",
        "https://api.heigit.org.evil.example/openrouteservice/v2/snap/driving-car",
        "https://api.heigit.org:443/openrouteservice/v2/snap/driving-car",
        "https://user@api.heigit.org/openrouteservice/v2/snap/driving-car",
        ORS_BASE_URL + "/v2/snap/driving-car?api_key=" + KEY,
    ],
)
def test_only_exact_allowlisted_https_urls_reach_opener(url):
    opener = Opener(HTTPResponse())
    with pytest.raises(RuntimeRoutingError, match="routing_redirect_not_allowed"):
        http_fetch(opener, url=url)
    assert opener.calls == []


def test_redirect_handler_refuses_same_and_cross_origin_redirects():
    for destination in (
        ORS_BASE_URL + "/elsewhere",
        "https://example.com/",
        "https://api.openrouteservice.org/v2/snap/driving-car",
    ):
        with pytest.raises(RuntimeRoutingError, match="routing_redirect_not_allowed"):
            _RejectORSRedirects().redirect_request(urllib.request.Request(ORS_ORIGIN), None, 302, KEY, {}, destination)
