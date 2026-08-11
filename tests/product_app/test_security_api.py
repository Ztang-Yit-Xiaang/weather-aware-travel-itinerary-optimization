from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from starlette.requests import Request

from itinerary_system.product_app.api import PRODUCT_ID, create_product_app
from itinerary_system.product_app.config import ProductRuntimeConfig
from itinerary_system.product_app.models import ComponentHealthV1
from itinerary_system.product_app.runtime import ProductRuntime
from itinerary_system.product_app.security import ProductSecurityBoundary
from itinerary_system.product_app.workspace import WorkspaceError, WorkspaceStore

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "configs" / "product_app_registry.json"
MAP_COMPOSE = ROOT / "docker" / "maplibre" / "docker-compose.yml"
MAP_NGINX = ROOT / "docker" / "maplibre" / "nginx.conf.template"


@pytest.fixture
def api_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setenv("PRODUCT_COPILOT_ADAPTER", "deterministic")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("MAPBOX_ATLAS_LICENSE", raising=False)
    monkeypatch.setattr(
        ProductRuntime,
        "_probe_map",
        lambda self: ComponentHealthV1("map", "ready", False, "maplibre_ready"),
    )
    app = create_product_app(
        repository_root=ROOT,
        registry_path=REGISTRY,
        state_root=tmp_path / "state",
        additional_allowed_authorities=("testserver",),
    )
    with TestClient(app) as client:
        yield client


def create_session(client: TestClient) -> tuple[dict[str, Any], dict[str, str]]:
    response = client.post("/api/sessions", json={})
    assert response.status_code == 200
    payload = response.json()
    return payload["session"], {"X-Session-Token": payload["mutation_token"]}


def test_health_is_versioned_truthful_and_secret_free(api_client: TestClient) -> None:
    response = api_client.get("/api/health")
    payload = response.json()

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert payload["schema_version"] == "product-health-v2"
    assert payload["product_id"] == PRODUCT_ID
    assert len(payload["build_id"]) == 16
    assert payload["status"] == "ready"
    assert payload["core_ready"] is True
    assert payload["ready"] is True
    assert set(payload["components"]) == {
        "registry",
        "default_workspace",
        "state_store",
        "map",
        "routing",
        "openai",
    }
    assert payload["components"]["openai"]["code"] == "deterministic_adapter_selected"
    assert "mutation_token" not in json.dumps(payload)


def test_unexpected_api_failure_is_sanitized_and_keeps_security_headers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PRODUCT_COPILOT_ADAPTER", "deterministic")
    monkeypatch.setattr(
        ProductRuntime,
        "_probe_map",
        lambda self: ComponentHealthV1("map", "ready", False, "maplibre_ready"),
    )
    app = create_product_app(
        repository_root=ROOT,
        registry_path=REGISTRY,
        state_root=tmp_path / "state",
        additional_allowed_authorities=("testserver",),
    )

    def fail_health() -> None:
        raise RuntimeError("sentinel exception must not escape")

    with TestClient(app, raise_server_exceptions=False) as client:
        monkeypatch.setattr(app.state.product_runtime, "health", fail_health)
        response = client.get("/api/health")

    assert response.status_code == 500
    assert response.json() == {"detail": "internal_server_error"}
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-frame-options"] == "DENY"
    assert "default-src 'self'" in response.headers["content-security-policy"]
    assert "sentinel exception" not in response.text


def test_openai_selection_reports_configured_without_exposing_key_or_calling_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PRODUCT_COPILOT_ADAPTER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "configured-but-never-returned")
    monkeypatch.setattr(
        ProductRuntime,
        "_probe_map",
        lambda self: ComponentHealthV1("map", "ready", False, "maplibre_ready"),
    )
    app = create_product_app(
        repository_root=ROOT,
        registry_path=REGISTRY,
        state_root=tmp_path / "state",
        additional_allowed_authorities=("testserver",),
    )
    with TestClient(app) as client:
        health = client.get("/api/health")
        session, headers = create_session(client)
        response = client.post(
            f"/api/sessions/{session['session_id']}/copilot/messages",
            headers=headers,
            json={"expected_revision": session["revision"], "message": "repair my trip"},
        )

    openai = health.json()["components"]["openai"]
    assert openai["status"] == "ready"
    assert openai["code"] == "openai_configured"
    assert response.status_code == 422
    assert response.json() == {"detail": "request_validation_failed"}
    assert "configured-but-never-returned" not in health.text


def test_corrupt_conversation_store_becomes_sanitized_failed_health(tmp_path: Path) -> None:
    state_root = tmp_path / "state"
    conversations = state_root / "conversations"
    conversations.mkdir(parents=True)
    (conversations / "unexpected.json").write_text("{}", encoding="utf-8")

    app = create_product_app(
        repository_root=ROOT,
        registry_path=REGISTRY,
        state_root=state_root,
        additional_allowed_authorities=("testserver",),
    )
    with TestClient(app) as client:
        health = client.get("/api/health")
        runs = client.get("/api/runs")

    payload = health.json()
    assert health.status_code == 200
    assert payload["status"] == "failed"
    assert payload["core_ready"] is False
    assert payload["components"]["state_store"]["code"] == "conversation_store_invalid"
    assert payload["components"]["default_workspace"]["code"] == "product_service_unavailable"
    assert "unexpected.json" not in health.text
    assert runs.status_code == 503
    assert runs.json() == {"detail": "product_core_not_ready"}


def test_failed_core_health_stays_http_200_and_core_apis_fail_503(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PRODUCT_COPILOT_ADAPTER", "deterministic")
    monkeypatch.setattr(
        ProductRuntime,
        "_probe_map",
        lambda self: ComponentHealthV1("map", "degraded", False, "maplibre_unavailable"),
    )
    app = create_product_app(
        repository_root=tmp_path,
        registry_path=tmp_path / "missing-registry.json",
        state_root=tmp_path / "state",
        additional_allowed_authorities=("testserver",),
    )
    with TestClient(app) as client:
        health = client.get("/api/health")
        runs = client.get("/api/runs")

    assert health.status_code == 200
    assert health.json()["status"] == "failed"
    assert health.json()["core_ready"] is False
    assert health.json()["components"]["registry"]["code"] == "registry_unavailable"
    assert runs.status_code == 503
    assert runs.json() == {"detail": "product_core_not_ready"}


def test_host_origin_csp_and_cache_boundaries(api_client: TestClient) -> None:
    blocked_host = api_client.get("/api/health", headers={"Host": "localhost.evil:8127"})
    assert blocked_host.status_code == 400
    assert blocked_host.json() == {"detail": "host_not_allowed"}
    assert blocked_host.headers["cache-control"] == "no-store"

    for origin in (
        "https://127.0.0.1:8127",
        "http://127.0.0.1:9999",
        "http://example.com",
        "null",
    ):
        blocked_origin = api_client.post("/api/sessions", headers={"Origin": origin}, json={})
        assert blocked_origin.status_code == 403
        assert blocked_origin.json() == {"detail": "origin_not_allowed"}
        assert blocked_origin.headers["cache-control"] == "no-store"

    allowed = api_client.post(
        "/api/sessions", headers={"Origin": "http://localhost:8127"}, json={}
    )
    assert allowed.status_code == 200
    assert api_client.get("/api/health", headers={"Origin": "https://example.com"}).status_code == 200

    shell = api_client.get("/app")
    assert shell.headers["cache-control"] == "no-cache"
    static_script = api_client.get("/static/js/app.js")
    assert static_script.headers["cache-control"] == "no-cache"
    assert shell.headers["x-content-type-options"] == "nosniff"
    assert shell.headers["x-frame-options"] == "DENY"
    assert shell.headers["referrer-policy"] == "no-referrer"
    csp = shell.headers["content-security-policy"]
    assert "default-src 'self'" in csp
    assert "http://127.0.0.1:8080" in csp
    assert "frame-ancestors 'none'" in csp
    assert "https://" not in csp


def test_map_config_is_no_store_maplibre_primary_and_contains_no_mapbox_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MAPBOX_ATLAS_TOKEN", "installer-secret-must-not-appear")
    monkeypatch.setenv("MAPBOX_PUBLIC_TOKEN", "public-token-must-not-appear")
    monkeypatch.setenv("MAPBOX_ATLAS_LICENSE", "runtime-license-must-not-appear")
    monkeypatch.setattr(
        ProductRuntime,
        "_probe_map",
        lambda self: ComponentHealthV1("map", "ready", False, "maplibre_ready"),
    )
    app = create_product_app(
        repository_root=ROOT,
        registry_path=REGISTRY,
        state_root=tmp_path / "state",
        additional_allowed_authorities=("testserver",),
    )
    with TestClient(app) as client:
        response = client.get("/api/map/config")
    payload = response.json()

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert payload["schema_version"] == "product-map-configuration-v2"
    assert payload["provider"] == "maplibre_pmtiles"
    assert payload["base_url"] == "http://127.0.0.1:8080"
    assert payload["protocol_script_url"].endswith("/pmtiles/pmtiles.js")
    assert payload["runtime_license"] is None
    assert payload["attribution"] == "Protomaps | © OpenStreetMap contributors"
    assert payload["attribution_url"] == "https://www.openstreetmap.org/copyright"
    assert payload["range_requests_required"] is True
    assert "installer-secret-must-not-appear" not in response.text
    assert "public-token-must-not-appear" not in response.text
    assert "runtime-license-must-not-appear" not in response.text


def test_maplibre_container_validates_origin_before_nginx_template_expansion() -> None:
    compose = MAP_COMPOSE.read_text(encoding="utf-8")
    nginx = MAP_NGINX.read_text(encoding="utf-8")

    assert "origin=\"$${PRODUCT_APP_ORIGIN}\"" in compose
    assert "*[!0-9]*" in compose
    assert '"$${port}" -gt 65535' in compose
    assert "exec /docker-entrypoint.sh" in compose
    assert 'Access-Control-Allow-Origin "${PRODUCT_APP_ORIGIN}"' in nginx
    assert "Access-Control-Allow-Origin \"*\"" not in nginx
    assert "$request_uri ~*" in nginx
    for encoded_segment in ("%2e", "%2f", "%5c"):
        assert encoded_segment in nginx
    assert "$request_method !~ ^(GET|HEAD|OPTIONS)$" in nginx


@pytest.mark.parametrize(
    ("content", "expected_status", "detail"),
    [
        (b"{", 400, "invalid_json"),
        (b"[]", 400, "object_body_required"),
        (b'{"unknown":true}', 422, "request_validation_failed"),
        (b"{" + b'"x":"' + b"a" * 20_001 + b'"}', 413, "request_too_large"),
    ],
)
def test_request_body_and_schema_failures_are_stable_and_no_store(
    api_client: TestClient,
    content: bytes,
    expected_status: int,
    detail: str,
) -> None:
    response = api_client.post(
        "/api/sessions",
        content=content,
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == expected_status
    assert response.json() == {"detail": detail}
    assert response.headers["cache-control"] == "no-store"


def test_streamed_body_limit_applies_without_content_length(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = ProductRuntimeConfig.from_environment(
        repository_root=tmp_path,
        registry_path=tmp_path / "registry.json",
        state_root=tmp_path / "state",
    )
    boundary = ProductSecurityBoundary(config)
    sent = False

    async def receive() -> dict[str, Any]:
        nonlocal sent
        if sent:
            return {"type": "http.request", "body": b"", "more_body": False}
        sent = True
        return {"type": "http.request", "body": b"x" * 20_001, "more_body": False}

    request = Request(
        {
            "type": "http",
            "method": "POST",
            "scheme": "http",
            "path": "/api/sessions",
            "raw_path": b"/api/sessions",
            "query_string": b"",
            "headers": [(b"host", b"127.0.0.1:8127")],
            "server": ("127.0.0.1", 8127),
            "client": ("127.0.0.1", 1),
        },
        receive,
    )
    with pytest.raises(HTTPException) as error:
        asyncio.run(boundary.read_bounded_json(request))
    assert error.value.status_code == 413
    assert error.value.detail == "request_too_large"


def test_alternatives_require_session_token(api_client: TestClient) -> None:
    session, headers = create_session(api_client)
    path = f"/api/sessions/{session['session_id']}/alternatives"

    missing = api_client.get(path)
    invalid = api_client.get(path, headers={"X-Session-Token": "wrong"})
    allowed = api_client.get(path, headers=headers)

    assert missing.status_code == 403
    assert invalid.status_code == 403
    assert allowed.status_code == 200
    assert allowed.headers["cache-control"] == "no-store"
    assert allowed.json()["baseline"] == {
        "id": "keep_original",
        "label": "Keep original",
        "ranking_eligible": False,
        "status": "accepted_parent",
    }


def test_session_routes_reject_noncanonical_ids_before_storage(api_client: TestClient) -> None:
    response = api_client.get(
        "/api/sessions/not-a-session",
        headers={"X-Session-Token": "not-a-token"},
    )

    assert response.status_code == 422
    assert response.json() == {"detail": "invalid_session_id"}
    assert response.headers["cache-control"] == "no-store"


def test_selection_rejects_unregistered_candidate_without_mutating_session(
    api_client: TestClient,
) -> None:
    session, headers = create_session(api_client)
    path = f"/api/sessions/{session['session_id']}"

    path_like = api_client.post(
        f"{path}/selection",
        headers=headers,
        json={"expected_revision": 0, "selected_candidate_id": "../forged-candidate"},
    )
    unknown = api_client.post(
        f"{path}/selection",
        headers=headers,
        json={"expected_revision": 0, "selected_candidate_id": "unknown_candidate"},
    )
    restored = api_client.get(path, headers=headers)

    assert path_like.status_code == 422
    assert path_like.json() == {"detail": "request_validation_failed"}
    assert unknown.status_code == 422
    assert unknown.json() == {"detail": "invalid_selected_candidate"}
    assert restored.status_code == 200
    assert restored.json()["session"]["revision"] == 0
    assert restored.json()["session"]["selected_candidate_id"] is None


def test_invalid_candidate_and_evidence_reference_budget_fail_closed(
    api_client: TestClient,
) -> None:
    session, headers = create_session(api_client)
    path = f"/api/sessions/{session['session_id']}/draft/operations"
    invalid_candidate = api_client.post(
        path,
        headers=headers,
        json={
            "expected_revision": 0,
            "type": "add_candidate",
            "target": "registered_candidate",
            "parameters": {},
        },
    )
    too_many_refs = api_client.post(
        path,
        headers=headers,
        json={
            "expected_revision": 0,
            "type": "route_feedback",
            "target": "selected_route",
            "parameters": {},
            "evidence_refs": [f"ref-{index}" for index in range(17)],
        },
    )

    assert invalid_candidate.status_code == 422
    assert invalid_candidate.json() == {"detail": "invalid_draft_candidate"}
    assert too_many_refs.status_code == 422
    assert too_many_refs.json() == {"detail": "request_validation_failed"}


def test_operation_parameters_reject_nonfinite_and_excessive_nesting(
    api_client: TestClient,
) -> None:
    session, headers = create_session(api_client)
    path = f"/api/sessions/{session['session_id']}/draft/operations"
    base = {
        "expected_revision": 0,
        "type": "route_feedback",
        "target": "selected_route",
    }

    nonfinite = api_client.post(
        path,
        headers={**headers, "Content-Type": "application/json"},
        content=(
            '{"expected_revision":0,"type":"route_feedback",'
            '"target":"selected_route","parameters":{"weight":NaN}}'
        ),
    )
    nested: object = "value"
    for _ in range(10):
        nested = {"child": nested}
    too_deep = api_client.post(path, headers=headers, json={**base, "parameters": nested})

    assert nonfinite.status_code == 422
    assert nonfinite.json() == {"detail": "request_validation_failed"}
    assert too_deep.status_code == 422
    assert too_deep.json() == {"detail": "request_validation_failed"}


def test_workspace_session_draft_and_permission_limits(tmp_path: Path) -> None:
    store = WorkspaceStore(tmp_path / "state")
    for _ in range(store.MAX_SESSIONS):
        store.create_session("run", "parent", 1)
    with pytest.raises(WorkspaceError, match="session_capacity_reached") as capacity:
        store.create_session("run", "parent", 1)
    assert capacity.value.status_code == 429

    limited = WorkspaceStore(tmp_path / "limited-state")
    session, _ = limited.create_session("run", "parent", 1)
    for revision in range(limited.MAX_DRAFT_OPERATIONS):
        limited.add_operation(
            session,
            {
                "expected_revision": revision,
                "type": "route_feedback",
                "target": "selected_route",
                "parameters": {
                    "preference": "reduce_contextual_risk",
                    "weight": revision / limited.MAX_DRAFT_OPERATIONS,
                },
            },
            valid_stop_ids=set(),
            day_count=1,
        )
    with pytest.raises(WorkspaceError, match="draft_operation_limit_reached"):
        limited.add_operation(
            session,
            {
                "expected_revision": session.revision,
                "type": "route_feedback",
                "target": "selected_route",
                "parameters": {"preference": "reduce_contextual_risk"},
            },
            valid_stop_ids=set(),
            day_count=1,
        )

    permission_session, _ = limited.create_session("run", "parent", 1)
    for revision in range(limited.MAX_PERMISSION_DECISIONS):
        limited.append_permission(
            permission_session,
            {
                "expected_revision": revision,
                "permission": "booking_change",
                "decision": "denied",
                "proposal_id": "proposal",
                "scope": "trip",
            },
        )
    with pytest.raises(WorkspaceError, match="permission_decision_limit_reached"):
        limited.append_permission(
            permission_session,
            {
                "expected_revision": permission_session.revision,
                "permission": "booking_change",
                "decision": "denied",
                "proposal_id": "proposal",
                "scope": "trip",
            },
        )
