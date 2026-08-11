from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from itinerary_system.product_app.api import create_product_app
from itinerary_system.product_app.models import ComponentHealthV1
from itinerary_system.product_app.routing_runtime import RuntimeRoutingError, RuntimeSnapResult
from itinerary_system.product_app.runtime import ProductRuntime
from itinerary_system.routing.models import RouteLegResult
from itinerary_system.routing.provider import RouteLegRequest

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "configs" / "product_app_registry.json"


class FakeRouter:
    def __init__(
        self,
        result: RuntimeSnapResult | RuntimeRoutingError,
        *,
        route_error: RuntimeRoutingError | None = None,
    ) -> None:
        self.result = result
        self.route_error = route_error
        self.calls: list[tuple[str, tuple[float, float]]] = []
        self.route_requests: list[RouteLegRequest] = []

    def nearest(self, entity_id: str, point: tuple[float, float]) -> RuntimeSnapResult:
        self.calls.append((entity_id, point))
        if isinstance(self.result, RuntimeRoutingError):
            raise self.result
        return replace(self.result, entity_id=entity_id, raw_point=point)

    def route(self, request: RouteLegRequest) -> RouteLegResult:
        self.route_requests.append(request)
        if self.route_error is not None:
            raise self.route_error
        query_hash = ("a" if len(self.route_requests) == 1 else "b") * 64
        return RouteLegResult(
            origin_id=request.origin_id,
            destination_id=request.destination_id,
            geometry=(request.origin_point, request.destination_point),
            distance_m=1200.0,
            duration_s=180.0,
            routing_status="osrm_route_validated",
            provider="runtime_osrm",
            routing_profile=request.routing_profile,
            geometry_source="runtime_osrm_geojson",
            distance_source="runtime_osrm_route",
            duration_source="runtime_osrm_route",
            road_validated=True,
            fallback_used=False,
            query_hash=query_hash,
            retrieved_at=datetime.now(UTC),
            snap_distance_origin_m=3.0,
            snap_distance_destination_m=4.0,
        )


@pytest.fixture(autouse=True)
def ready_external_components(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PRODUCT_ROUTING_BASE_URL", raising=False)
    monkeypatch.setattr(
        ProductRuntime,
        "_probe_map",
        lambda self: ComponentHealthV1("map", "ready", False, "maplibre_ready"),
    )
    monkeypatch.setattr(
        ProductRuntime,
        "_probe_routing",
        lambda self: ComponentHealthV1("routing", "ready", False, "runtime_osrm_ready"),
    )


@pytest.fixture
def app_client(tmp_path: Path) -> tuple[TestClient, ProductRuntime]:
    app = create_product_app(
        repository_root=ROOT,
        registry_path=REGISTRY,
        state_root=tmp_path / "state",
        additional_allowed_authorities=("testserver",),
    )
    return TestClient(app), app.state.product_runtime


def create_session(client: TestClient) -> tuple[dict[str, Any], dict[str, str]]:
    response = client.post("/api/sessions", json={})
    assert response.status_code == 200
    payload = response.json()
    return payload["session"], {"X-Session-Token": payload["mutation_token"]}


def request_body(**overrides: Any) -> dict[str, Any]:
    body = {
        "expected_revision": 0,
        "longitude": -118.2437,
        "latitude": 34.0522,
        "operation_intent": "add_custom_waypoint",
        "selected_day": 3,
        "selected_route_segment_id": None,
        "travel_mode": "driving",
    }
    body.update(overrides)
    return body


def snap_result(*, distance: float = 42.0) -> RuntimeSnapResult:
    if distance <= 100:
        state, code, confirmation, allowed = (
            "normal",
            "snap_within_normal_threshold",
            False,
            True,
        )
    elif distance <= 500:
        state, code, confirmation, allowed = (
            "warning",
            "snap_confirmation_required",
            True,
            True,
        )
    else:
        state, code, confirmation, allowed = "rejected", "map_snap_too_far", False, False
    return RuntimeSnapResult(
        entity_id="snap_fixture",
        snapped_point=(34.0524, -118.2435),
        snap_distance_m=distance,
        provider="runtime_osrm",
        status=state,
        raw_point=(34.0522, -118.2437),
        validation_state=state,
        code=code,
        confirmation_required=confirmation,
        draft_append_allowed=allowed,
    )


@pytest.mark.parametrize(
    ("distance", "state", "confirmation_required", "append_allowed"),
    [(42.0, "normal", False, True), (240.0, "warning", True, True), (700.0, "rejected", False, False)],
)
def test_snap_preview_is_classified_and_never_mutates_the_draft(
    app_client: tuple[TestClient, ProductRuntime],
    distance: float,
    state: str,
    confirmation_required: bool,
    append_allowed: bool,
) -> None:
    client, runtime = app_client
    with client:
        session, headers = create_session(client)
        router = FakeRouter(snap_result(distance=distance))
        runtime.routing = router  # type: ignore[assignment]
        response = client.post(
            f"/api/sessions/{session['session_id']}/map/snap-preview",
            headers=headers,
            json=request_body(),
        )
        restored = client.get(f"/api/sessions/{session['session_id']}", headers=headers)

    assert response.status_code == 200
    payload = response.json()
    assert payload["schema_version"] == "map-snap-preview-v1"
    assert payload["session_revision"] == 0
    assert payload["operation_intent"] == "add_custom_waypoint"
    assert payload["selected_day"] == 3
    assert payload["selected_route_segment_id"] is None
    assert payload["travel_mode"] == "driving"
    assert payload["raw_coordinate"] == {"latitude": 34.0522, "longitude": -118.2437}
    assert payload["snapped_coordinate"] == {
        "latitude": 34.0524,
        "longitude": -118.2435,
    }
    assert payload["snap_distance_m"] == distance
    assert payload["persisted"] is False
    assert payload["affected_route_legs"] == []
    if state == "rejected":
        assert payload["validation_state"] == "rejected"
        assert payload["draft_append_allowed"] is False
        assert payload["selected_access_point"] is None
    else:
        assert payload["validation_state"] == "snap_only"
        assert payload["code"] == "snap_ready_route_context_required"
        assert payload["confirmation_required"] is confirmation_required
        assert payload["draft_append_allowed"] is False
        assert payload["selected_access_point"]["road_validated"] is True
    assert restored.status_code == 200
    assert restored.json()["session"]["revision"] == 0
    assert restored.json()["session"]["draft"] == []
    assert router.calls[0][1] == (34.0522, -118.2437)


def _accepted_segment(
    client: TestClient,
    session: dict[str, Any],
    headers: dict[str, str],
) -> dict[str, Any]:
    restored = client.get(f"/api/sessions/{session['session_id']}", headers=headers)
    assert restored.status_code == 200
    geography = restored.json()["workspace"]["geography"]
    plan = next(row for row in geography["plans"] if row["plan_id"] == session["accepted_plan_id"])
    return plan["validated_legs"]["features"][0]


def test_selected_segment_produces_two_road_valid_affected_legs_without_mutation(
    app_client: tuple[TestClient, ProductRuntime],
) -> None:
    client, runtime = app_client
    with client:
        session, headers = create_session(client)
        segment = _accepted_segment(client, session, headers)
        properties = segment["properties"]
        router = FakeRouter(snap_result())
        runtime.routing = router  # type: ignore[assignment]
        response = client.post(
            f"/api/sessions/{session['session_id']}/map/snap-preview",
            headers=headers,
            json=request_body(
                selected_day=properties["day"],
                selected_route_segment_id=properties["route_leg_id"],
            ),
        )
        restored = client.get(f"/api/sessions/{session['session_id']}", headers=headers)

    assert response.status_code == 200
    payload = response.json()
    assert payload["validation_state"] == "route_checked"
    assert payload["draft_append_allowed"] is True
    assert payload["persisted"] is False
    assert payload["session_revision"] == session["revision"] == 0
    assert payload["snap_preview_id"].startswith("snap_")
    assert payload["created_at"] < payload["expires_at"]
    assert payload["selected_access_point"]["road_validated"] is True
    assert len(payload["affected_route_legs"]) == 2
    first, second = payload["affected_route_legs"]
    assert (first["origin_id"], first["destination_id"]) == (
        properties["origin_id"],
        payload["entity_id"],
    )
    assert (second["origin_id"], second["destination_id"]) == (
        payload["entity_id"],
        properties["destination_id"],
    )
    for leg in payload["affected_route_legs"]:
        assert leg["road_validated"] is True
        assert leg["fallback_used"] is False
        assert leg["geometry"]["type"] == "LineString"
        assert len(leg["geometry"]["coordinates"]) == 2
        assert leg["distance_m"] == 1200.0
        assert leg["duration_s"] == 180.0
        assert leg["evidence_refs"] == [f"route_query:{leg['query_hash']}"]
    assert restored.json()["session"]["revision"] == 0
    assert restored.json()["session"]["draft"] == []


def test_selected_segment_missing_ambiguous_or_provider_failed_fails_closed(
    app_client: tuple[TestClient, ProductRuntime],
) -> None:
    client, runtime = app_client
    with client:
        session, headers = create_session(client)
        segment = _accepted_segment(client, session, headers)
        properties = segment["properties"]
        url = f"/api/sessions/{session['session_id']}/map/snap-preview"
        runtime.routing = FakeRouter(snap_result())  # type: ignore[assignment]
        missing = client.post(
            url,
            headers=headers,
            json=request_body(selected_route_segment_id="missing_leg"),
        )
        segment_days = {
            value
            for value in (
                properties.get("day"),
                properties.get("from_day"),
                properties.get("to_day"),
            )
            if isinstance(value, int) and not isinstance(value, bool)
        }
        mismatched_day = next(day for day in range(1, 8) if day not in segment_days)
        day_mismatch = client.post(
            url,
            headers=headers,
            json=request_body(
                selected_day=mismatched_day,
                selected_route_segment_id=properties["route_leg_id"],
            ),
        )

        service = runtime.require_service()
        plan = next(
            row
            for row in service._geographies[session["run_id"]]["plans"]
            if row["plan_id"] == session["accepted_plan_id"]
        )
        plan["validated_legs"]["features"].append(segment)
        ambiguous = client.post(
            url,
            headers=headers,
            json=request_body(
                selected_day=properties["day"],
                selected_route_segment_id=properties["route_leg_id"],
            ),
        )
        plan["validated_legs"]["features"].pop()

        router = FakeRouter(snap_result(), route_error=RuntimeRoutingError("routing_timeout"))
        runtime.routing = router  # type: ignore[assignment]
        failed = client.post(
            url,
            headers=headers,
            json=request_body(
                selected_day=properties["day"],
                selected_route_segment_id=properties["route_leg_id"],
            ),
        )
        restored = client.get(f"/api/sessions/{session['session_id']}", headers=headers)

    assert missing.status_code == 422
    assert missing.json() == {"detail": "selected_route_segment_not_found"}
    assert day_mismatch.status_code == 422
    assert day_mismatch.json() == {"detail": "selected_route_segment_day_mismatch"}
    assert ambiguous.status_code == 409
    assert ambiguous.json() == {"detail": "selected_route_segment_ambiguous"}
    assert failed.status_code == 503
    assert failed.json() == {"detail": "routing_timeout"}
    assert restored.json()["session"]["revision"] == 0
    assert restored.json()["session"]["draft"] == []


def test_snap_preview_requires_authentication_and_current_revision(
    app_client: tuple[TestClient, ProductRuntime],
) -> None:
    client, runtime = app_client
    runtime.routing = FakeRouter(snap_result())  # type: ignore[assignment]
    with client:
        session, headers = create_session(client)
        url = f"/api/sessions/{session['session_id']}/map/snap-preview"
        assert client.post(url, json=request_body()).status_code == 403
        stale = client.post(url, headers=headers, json=request_body(expected_revision=1))

    assert stale.status_code == 409
    assert stale.json() == {"detail": "stale_session_revision"}


@pytest.mark.parametrize(
    "overrides",
    [
        {"longitude": 181},
        {"latitude": True},
        {"operation_intent": "move_real_place"},
        {"selected_day": 0},
        {"travel_mode": "walking"},
        {"unexpected": "field"},
    ],
)
def test_snap_preview_rejects_invalid_or_unknown_input(
    app_client: tuple[TestClient, ProductRuntime], overrides: dict[str, Any]
) -> None:
    client, runtime = app_client
    runtime.routing = FakeRouter(snap_result())  # type: ignore[assignment]
    with client:
        session, headers = create_session(client)
        response = client.post(
            f"/api/sessions/{session['session_id']}/map/snap-preview",
            headers=headers,
            json=request_body(**overrides),
        )

    assert response.status_code == 422
    assert response.json() == {"detail": "request_validation_failed"}


def test_snap_preview_sanitizes_provider_failure(
    app_client: tuple[TestClient, ProductRuntime],
) -> None:
    client, runtime = app_client
    runtime.routing = FakeRouter(RuntimeRoutingError("routing_timeout"))  # type: ignore[assignment]
    with client:
        session, headers = create_session(client)
        response = client.post(
            f"/api/sessions/{session['session_id']}/map/snap-preview",
            headers=headers,
            json=request_body(),
        )

    assert response.status_code == 503
    assert response.json() == {"detail": "routing_timeout"}
