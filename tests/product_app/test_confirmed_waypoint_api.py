from __future__ import annotations

import json
import subprocess
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from itinerary_system.product_app.api import create_product_app
from itinerary_system.product_app.models import ComponentHealthV1
from itinerary_system.product_app.routing_runtime import RuntimeSnapResult
from itinerary_system.product_app.runtime import ProductRuntime
from itinerary_system.research_artifacts import stable_content_hash
from itinerary_system.routing.models import RouteLegResult
from itinerary_system.routing.provider import RouteLegRequest

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "configs" / "product_app_registry.json"


class FakeRouter:
    def __init__(self, *, distance: float = 42.0) -> None:
        self.distance = distance
        self.calls = 0

    def nearest(self, entity_id: str, point: tuple[float, float]) -> RuntimeSnapResult:
        if self.distance <= 100:
            state, code, confirmation, allowed = "normal", "snap_within_normal_threshold", False, True
        elif self.distance <= 500:
            state, code, confirmation, allowed = "warning", "snap_confirmation_required", True, True
        else:
            state, code, confirmation, allowed = "rejected", "map_snap_too_far", False, False
        return RuntimeSnapResult(
            entity_id=entity_id,
            snapped_point=(point[0] + 0.0001, point[1] + 0.0001),
            snap_distance_m=self.distance,
            provider="runtime_osrm",
            status=state,
            raw_point=point,
            validation_state=state,
            code=code,
            confirmation_required=confirmation,
            draft_append_allowed=allowed,
        )

    def route(self, request: RouteLegRequest) -> RouteLegResult:
        self.calls += 1
        return RouteLegResult(
            origin_id=request.origin_id,
            destination_id=request.destination_id,
            geometry=(request.origin_point, request.destination_point),
            distance_m=1200.0,
            duration_s=180.0,
            routing_status="osrm_route_validated",
            provider="runtime_osrm",
            routing_profile="driving",
            geometry_source="runtime_osrm_geojson",
            distance_source="runtime_osrm_route",
            duration_source="runtime_osrm_route",
            road_validated=True,
            fallback_used=False,
            query_hash=("a" if self.calls % 2 else "b") * 64,
            retrieved_at=datetime.now(UTC),
            snap_distance_origin_m=3.0,
            snap_distance_destination_m=4.0,
        )


@pytest.fixture(autouse=True)
def ready_external_components(monkeypatch: pytest.MonkeyPatch) -> None:
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
def app_client(tmp_path: Path) -> tuple[TestClient, ProductRuntime, Path]:
    state_root = tmp_path / "state"
    app = create_product_app(
        repository_root=ROOT,
        registry_path=REGISTRY,
        state_root=state_root,
        additional_allowed_authorities=("testserver",),
    )
    return TestClient(app), app.state.product_runtime, state_root


def _session(client: TestClient) -> tuple[dict[str, Any], dict[str, str], str]:
    payload = client.post("/api/sessions", json={}).json()
    return (
        payload["session"],
        {"X-Session-Token": payload["mutation_token"]},
        payload["mutation_token"],
    )


def _segment(client: TestClient, session: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
    workspace = client.get(f"/api/sessions/{session['session_id']}", headers=headers).json()["workspace"]
    plan = next(row for row in workspace["geography"]["plans"] if row["plan_id"] == session["accepted_plan_id"])
    return plan["validated_legs"]["features"][0]["properties"]


def _preview(
    client: TestClient,
    runtime: ProductRuntime,
    session: dict[str, Any],
    headers: dict[str, str],
    *,
    distance: float = 42.0,
    revision: int | None = None,
    longitude: float = -118.2437,
    intent: str = "add_custom_waypoint",
    target: str | None = None,
) -> dict[str, Any]:
    runtime.routing = FakeRouter(distance=distance)  # type: ignore[assignment]
    segment = _segment(client, session, headers)
    body: dict[str, Any] = {
        "expected_revision": session["revision"] if revision is None else revision,
        "longitude": longitude,
        "latitude": 34.0522,
        "operation_intent": intent,
        "selected_day": segment["day"],
        "selected_route_segment_id": segment["route_leg_id"],
        "travel_mode": "driving",
    }
    if target is not None:
        body["target_waypoint_id"] = target
    response = client.post(
        f"/api/sessions/{session['session_id']}/map/snap-preview",
        headers=headers,
        json=body,
    )
    assert response.status_code == 200, response.text
    return response.json()


def _duration(minutes: int = 90) -> dict[str, Any]:
    return {
        "mode": "exact",
        "preferred_minutes": minutes,
        "minimum_minutes": minutes,
        "maximum_minutes": minutes,
    }


def _confirm(
    client: TestClient,
    session: dict[str, Any],
    headers: dict[str, str],
    preview: dict[str, Any],
    **overrides: Any,
):
    body = {
        "expected_revision": session["revision"],
        "name": "Local scenic stop",
        "role": "scenic_stop",
        "duration": _duration(),
        "warning_acknowledged": False,
    }
    body.update(overrides)
    return client.post(
        f"/api/sessions/{session['session_id']}/map/snap-previews/{preview['snap_preview_id']}/confirm",
        headers=headers,
        json=body,
    )


def test_confirmed_waypoint_appends_once_persists_and_exposes_truthful_capability(
    app_client: tuple[TestClient, ProductRuntime, Path],
) -> None:
    client, runtime, state_root = app_client
    with client:
        session, headers, token = _session(client)
        service = runtime.require_service()
        parent_before = stable_content_hash(service.load(session["run_id"])[0].parent_plan)
        preview = _preview(client, runtime, session, headers)
        response = _confirm(client, session, headers, preview)
        assert response.status_code == 200, response.text
        result = response.json()
        operation = result["operation"]
        assert result["feedback_tier"] == "route_checked"
        assert result["evaluated_repair"] is False
        assert operation["type"] == "add_custom_waypoint"
        assert operation["target"].startswith("waypoint_")
        assert operation["parameters"]["waypoint_id"] == operation["target"]
        assert operation["parameters"]["snap_preview_id"] == preview["snap_preview_id"]
        assert len(operation["parameters"]["affected_route_legs"]) == 2
        assert tuple(operation["parameters"]["selected_access_point"]["evidence_refs"]) == tuple(
            operation["evidence_refs"]
        )
        assert len(operation["evidence_refs"]) == 2
        assert operation["source"] == "confirmed_map_interaction"
        assert result["session"]["revision"] == 1
        assert len(result["session"]["draft"]) == 1
        assert stable_content_hash(service.load(session["run_id"])[0].parent_plan) == parent_before

        restored = client.get(f"/api/sessions/{session['session_id']}", headers=headers).json()
        assert restored["session"]["draft"] == result["session"]["draft"]
        capabilities = restored["workspace"]["map_edit_capabilities"]
        assert capabilities["schema_version"] == "map-edit-capabilities-v1"
        assert capabilities["operations"]["add_custom_waypoint"] == {
            "enabled": True,
            "feedback_tier": "route_checked",
            "preview_executable": False,
            "evaluated_repair": False,
        }
        reducer = subprocess.run(
            ["node", str(ROOT / "tests" / "product_app" / "confirmed_waypoint_reducer_stdin.mjs")],
            input=json.dumps(
                {
                    "operations": [operation],
                    "day_count": len(restored["workspace"]["timeline"]),
                }
            ),
            text=True,
            capture_output=True,
            check=False,
            timeout=10,
        )
        assert reducer.returncode == 0, reducer.stderr
        reduced = json.loads(reducer.stdout)
        assert reduced["waypoint_count"] == 1
        assert reduced["route_leg_count"] == 2
        assert reduced["waypoint_id"] == operation["target"]

    restarted = create_product_app(
        repository_root=ROOT,
        registry_path=REGISTRY,
        state_root=state_root,
        additional_allowed_authorities=("testserver",),
    )
    with TestClient(restarted) as second:
        restored = second.get(
            f"/api/sessions/{session['session_id']}", headers={"X-Session-Token": token}
        )
    assert restored.status_code == 200
    assert restored.json()["session"]["draft"][0]["target"] == operation["target"]


def test_warning_requires_acknowledgement_then_consumes_preview(
    app_client: tuple[TestClient, ProductRuntime, Path],
) -> None:
    client, runtime, _ = app_client
    with client:
        session, headers, _ = _session(client)
        preview = _preview(client, runtime, session, headers, distance=240.0)
        denied = _confirm(client, session, headers, preview)
        accepted = _confirm(client, session, headers, preview, warning_acknowledged=True)
        replay = _confirm(client, session, headers, preview, warning_acknowledged=True)
    assert denied.status_code == 409
    assert denied.json() == {"detail": "snap_warning_acknowledgement_required"}
    assert accepted.status_code == 200
    assert replay.status_code == 409
    assert replay.json() == {"detail": "snap_preview_already_consumed"}


def test_far_and_no_context_previews_cannot_append(
    app_client: tuple[TestClient, ProductRuntime, Path],
) -> None:
    client, runtime, _ = app_client
    with client:
        session, headers, _ = _session(client)
        far = _preview(client, runtime, session, headers, distance=700.0)
        far_confirm = _confirm(client, session, headers, far)
        runtime.routing = FakeRouter()  # type: ignore[assignment]
        no_context = client.post(
            f"/api/sessions/{session['session_id']}/map/snap-preview",
            headers=headers,
            json={
                "expected_revision": 0,
                "longitude": -118.2,
                "latitude": 34.0,
                "operation_intent": "add_custom_waypoint",
                "selected_day": 3,
                "travel_mode": "driving",
            },
        ).json()
        no_context_confirm = _confirm(client, session, headers, no_context)
        restored = client.get(f"/api/sessions/{session['session_id']}", headers=headers).json()["session"]
    assert far_confirm.status_code == 404
    assert no_context_confirm.status_code == 404
    assert restored["revision"] == 0
    assert restored["draft"] == []


def test_preview_is_bound_to_session_revision_and_expiry(
    app_client: tuple[TestClient, ProductRuntime, Path],
) -> None:
    client, runtime, _ = app_client
    with client:
        first, first_headers, _ = _session(client)
        second, second_headers, _ = _session(client)
        preview = _preview(client, runtime, first, first_headers)
        cross = client.post(
            f"/api/sessions/{second['session_id']}/map/snap-previews/{preview['snap_preview_id']}/confirm",
            headers=second_headers,
            json={"expected_revision": 0, "warning_acknowledged": False},
        )
        service = runtime.require_service()
        service._map_snap_previews[preview["snap_preview_id"]].expires_at = datetime.now(UTC) - timedelta(seconds=1)
        expired = _confirm(client, first, first_headers, preview)
        restored = client.get(f"/api/sessions/{first['session_id']}", headers=first_headers).json()["session"]
    assert cross.status_code == 403
    assert cross.json() == {"detail": "snap_preview_session_mismatch"}
    assert expired.status_code == 410
    assert expired.json() == {"detail": "snap_preview_expired"}
    assert restored["draft"] == []


def test_generic_draft_endpoint_cannot_bypass_map_confirmation(
    app_client: tuple[TestClient, ProductRuntime, Path],
) -> None:
    client, _, _ = app_client
    with client:
        session, headers, _ = _session(client)
        response = client.post(
            f"/api/sessions/{session['session_id']}/draft/operations",
            headers=headers,
            json={
                "expected_revision": 0,
                "type": "add_custom_waypoint",
                "target": "waypoint_" + "a" * 32,
                "parameters": {},
            },
        )
    assert response.status_code == 409
    assert response.json() == {"detail": "map_operation_confirmation_required"}


def test_preview_becomes_stale_after_an_unrelated_session_mutation(
    app_client: tuple[TestClient, ProductRuntime, Path],
) -> None:
    client, runtime, _ = app_client
    with client:
        session, headers, _ = _session(client)
        preview = _preview(client, runtime, session, headers)
        bundle, _ = runtime.require_service().load(session["run_id"])
        stop_id = str(bundle.parent_plan["selected_stops"][0]["poi_id"])
        mutation = client.post(
            f"/api/sessions/{session['session_id']}/draft/operations",
            headers=headers,
            json={
                "expected_revision": 0,
                "type": "keep_stop",
                "target": stop_id,
                "parameters": {},
            },
        )
        assert mutation.status_code == 200
        stale = _confirm(client, session, headers, preview)
    assert stale.status_code == 409
    assert stale.json() == {"detail": "stale_session_revision"}


def test_route_waypoint_requires_null_duration_and_uses_stable_preview_entity(
    app_client: tuple[TestClient, ProductRuntime, Path],
) -> None:
    client, runtime, _ = app_client
    with client:
        session, headers, _ = _session(client)
        preview = _preview(client, runtime, session, headers, intent="add_route_waypoint")
        assert preview["entity_id"].startswith("waypoint_")
        rejected = _confirm(
            client,
            session,
            headers,
            preview,
            name="Route shaping point",
            role="route_waypoint",
            duration=_duration(),
        )
        accepted = _confirm(
            client,
            session,
            headers,
            preview,
            name="Route shaping point",
            role="route_waypoint",
            duration=None,
        )
    assert rejected.status_code == 422
    assert accepted.status_code == 200
    assert accepted.json()["operation"]["target"] == preview["entity_id"]
    assert accepted.json()["operation"]["parameters"]["duration"] is None


def test_relocation_preserves_metadata_and_undo_restores_added_waypoint(
    app_client: tuple[TestClient, ProductRuntime, Path],
) -> None:
    client, runtime, _ = app_client
    with client:
        session, headers, _ = _session(client)
        added_preview = _preview(client, runtime, session, headers)
        added = _confirm(client, session, headers, added_preview).json()
        waypoint_id = added["operation"]["target"]
        current = added["session"]
        relocate_preview = _preview(
            client,
            runtime,
            current,
            headers,
            revision=1,
            longitude=-118.22,
            intent="relocate_custom_waypoint",
            target=waypoint_id,
        )
        relocated = client.post(
            f"/api/sessions/{session['session_id']}/map/snap-previews/{relocate_preview['snap_preview_id']}/confirm",
            headers=headers,
            json={"expected_revision": 1, "warning_acknowledged": False},
        )
        assert relocated.status_code == 200, relocated.text
        relocated_payload = relocated.json()
        relocation = relocated_payload["operation"]
        assert relocation["type"] == "relocate_custom_waypoint"
        assert relocation["target"] == waypoint_id
        for field in ("name", "role", "duration", "day", "insertion"):
            assert relocation["parameters"][field] == added["operation"]["parameters"][field]
        assert relocation["parameters"]["raw_coordinate"] != added["operation"]["parameters"]["raw_coordinate"]

        undone = client.post(
            f"/api/sessions/{session['session_id']}/draft/undo",
            headers=headers,
            json={"expected_revision": 2},
        )
        assert undone.status_code == 200
        assert undone.json()["undone"]["type"] == "relocate_custom_waypoint"
        assert undone.json()["session"]["revision"] == 3
        assert [row["type"] for row in undone.json()["session"]["draft"]] == ["add_custom_waypoint"]


@pytest.mark.parametrize(
    "duration",
    [
        {"mode": "preferred", "preferred_minutes": 90, "minimum_minutes": 15, "maximum_minutes": None},
        {"mode": "range", "preferred_minutes": None, "minimum_minutes": 120, "maximum_minutes": 90},
        {"mode": "exact", "preferred_minutes": 10, "minimum_minutes": 10, "maximum_minutes": 10},
    ],
)
def test_invalid_duration_appends_nothing(
    app_client: tuple[TestClient, ProductRuntime, Path], duration: dict[str, Any]
) -> None:
    client, runtime, _ = app_client
    with client:
        session, headers, _ = _session(client)
        preview = _preview(client, runtime, session, headers)
        response = _confirm(client, session, headers, preview, duration=duration)
        restored = client.get(f"/api/sessions/{session['session_id']}", headers=headers).json()["session"]
    assert response.status_code == 422
    assert restored["revision"] == 0
    assert restored["draft"] == []


def test_confirmed_operation_contains_only_bounded_route_evidence(
    app_client: tuple[TestClient, ProductRuntime, Path],
) -> None:
    client, runtime, _ = app_client
    with client:
        session, headers, token = _session(client)
        preview = _preview(client, runtime, session, headers)
        operation = _confirm(client, session, headers, preview).json()["operation"]
    serialized = str(operation)
    assert token not in serialized
    assert "http://" not in serialized
    assert "https://" not in serialized
    assert str(ROOT) not in serialized
    assert operation["evidence_refs"] == ["route_query:" + "a" * 64, "route_query:" + "b" * 64]


def test_evaluated_preview_rejects_confirmed_map_operation_truthfully(
    app_client: tuple[TestClient, ProductRuntime, Path],
) -> None:
    client, runtime, _ = app_client
    with client:
        session, headers, _ = _session(client)
        confirmed = _confirm(client, session, headers, _preview(client, runtime, session, headers)).json()
        response = client.post(
            f"/api/sessions/{session['session_id']}/preview",
            headers=headers,
            json={"expected_revision": confirmed["session"]["revision"]},
        )
    assert response.status_code == 409
    assert response.json() == {"detail": "draft_operation_unsupported"}


def test_tampered_confirmed_snapshot_cannot_be_restored_or_relocated(
    app_client: tuple[TestClient, ProductRuntime, Path],
) -> None:
    client, runtime, state_root = app_client
    with client:
        session, headers, token = _session(client)
        confirmed = _confirm(client, session, headers, _preview(client, runtime, session, headers)).json()
        waypoint_id = confirmed["operation"]["target"]
        segment = _segment(client, confirmed["session"], headers)

    snapshot_path = state_root / "sessions" / f"{session['session_id']}.json"
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    snapshot["session"]["draft"][0]["parameters"] = {
        "schema_version": "confirmed-map-operation-v1",
        "waypoint_id": waypoint_id,
    }
    snapshot_path.write_text(json.dumps(snapshot), encoding="utf-8")
    tampered_bytes = snapshot_path.read_bytes()

    restarted = create_product_app(
        repository_root=ROOT,
        registry_path=REGISTRY,
        state_root=state_root,
        additional_allowed_authorities=("testserver",),
    )
    with TestClient(restarted) as second:
        restore = second.get(
            f"/api/sessions/{session['session_id']}",
            headers={"X-Session-Token": token},
        )
        relocate = second.post(
            f"/api/sessions/{session['session_id']}/map/snap-preview",
            headers={"X-Session-Token": token},
            json={
                "expected_revision": 1,
                "longitude": -118.22,
                "latitude": 34.05,
                "operation_intent": "relocate_custom_waypoint",
                "selected_day": segment["day"],
                "selected_route_segment_id": segment["route_leg_id"],
                "travel_mode": "driving",
                "target_waypoint_id": waypoint_id,
            },
        )
    assert restore.status_code == 409
    assert restore.json() == {"detail": "confirmed_map_draft_invalid"}
    assert relocate.status_code == 409
    assert relocate.json() == {"detail": "confirmed_map_draft_invalid"}
    assert snapshot_path.read_bytes() == tampered_bytes
    assert len(json.loads(snapshot_path.read_text(encoding="utf-8"))["session"]["draft"]) == 1


@pytest.mark.parametrize("tampered_field", ["name", "insertion"])
def test_full_looking_relocation_cannot_change_preserved_waypoint_state(
    app_client: tuple[TestClient, ProductRuntime, Path],
    tampered_field: str,
) -> None:
    client, runtime, state_root = app_client
    with client:
        session, headers, token = _session(client)
        _confirm(client, session, headers, _preview(client, runtime, session, headers)).raise_for_status()

    snapshot_path = state_root / "sessions" / f"{session['session_id']}.json"
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    forged = deepcopy(snapshot["session"]["draft"][0])
    forged["operation_id"] = "operation_" + "f" * 32
    forged["type"] = "relocate_custom_waypoint"
    forged["parameters"]["snap_preview_id"] = "snap_" + "e" * 32
    if tampered_field == "name":
        forged["parameters"]["name"] = "Injected replacement name"
    else:
        forged["parameters"]["insertion"]["route_leg_id"] = "unrelated_route_leg"
    snapshot["session"]["draft"].append(forged)
    snapshot["session"]["revision"] = 2
    snapshot_path.write_text(json.dumps(snapshot), encoding="utf-8")
    tampered_bytes = snapshot_path.read_bytes()

    restarted = create_product_app(
        repository_root=ROOT,
        registry_path=REGISTRY,
        state_root=state_root,
        additional_allowed_authorities=("testserver",),
    )
    with TestClient(restarted) as second:
        restore = second.get(
            f"/api/sessions/{session['session_id']}",
            headers={"X-Session-Token": token},
        )
    assert restore.status_code == 409
    assert restore.json() == {"detail": "confirmed_map_draft_invalid"}
    assert snapshot_path.read_bytes() == tampered_bytes
