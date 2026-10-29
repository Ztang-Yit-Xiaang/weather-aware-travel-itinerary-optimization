from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from itinerary_system.product_app import runtime as runtime_module
from itinerary_system.product_app.api import create_product_app
from itinerary_system.product_app.models import ComponentHealthV1

ROOT = Path(__file__).resolve().parents[2]
HEADER = "X-Product-Build-Id"


def test_build_fingerprint_binds_product_registry_route_registration_and_descriptor(tmp_path):
    configs = tmp_path / "configs"
    configs.mkdir()
    registry = configs / "product_app_registry.json"
    registry.write_text('{"runs": []}', encoding="utf-8")
    descriptor = tmp_path / "results" / "delta" / "descriptor.json"
    descriptor.parent.mkdir(parents=True)
    descriptor.write_text('{"evidence_hash": "first"}', encoding="utf-8")
    registration = configs / "product_route_delta.json"
    registration.write_text(json.dumps({"descriptor_path": "results/delta/descriptor.json"}), encoding="utf-8")
    first = runtime_module.product_build_id(tmp_path)
    registry.write_text('{"runs": ["changed"]}', encoding="utf-8")
    second = runtime_module.product_build_id(tmp_path)
    assert second != first
    descriptor.write_text('{"evidence_hash": "second"}', encoding="utf-8")
    third = runtime_module.product_build_id(tmp_path)
    assert third != second
    registration.write_text(
        json.dumps({"descriptor_path": "results/delta/descriptor.json", "version": 2}), encoding="utf-8"
    )
    assert runtime_module.product_build_id(tmp_path) != third


@pytest.fixture
def coherent_app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("PRODUCT_COPILOT_ADAPTER", "deterministic")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setattr(
        runtime_module.ProductRuntime,
        "_probe_map",
        lambda self: ComponentHealthV1("map", "ready", False, "maplibre_ready"),
    )
    # Mutate an isolated source tree, never the repository under a running test.
    source = tmp_path / "source" / "src" / "itinerary_system" / "product_app" / "api.py"
    source.parent.mkdir(parents=True)
    source.write_text("version = 1\n", encoding="utf-8")
    fingerprint = runtime_module.product_build_id
    monkeypatch.setattr(runtime_module, "product_build_id", lambda _: fingerprint(tmp_path / "source"))

    def new_app():
        return create_product_app(
            repository_root=ROOT, state_root=tmp_path / "state", additional_allowed_authorities=("testserver",)
        )

    app = new_app()
    with TestClient(app) as client:
        yield client, app.state.product_runtime, source, new_app


def test_current_html_assets_and_health_bind_the_same_startup_build(coherent_app):
    client, runtime, _, _ = coherent_app
    html = client.get("/app/itinerary")
    assert html.status_code == 200
    assert f'<meta name="product-build-id" content="{runtime.build_id}">' in html.text
    assert "__PRODUCT_BUILD_ID__" not in html.text
    assets = re.findall(r'(?:href|src)="(/static/[^\"]+)"', html.text)
    assert len(assets) == 2
    for asset in assets:
        assert asset.endswith(f"?v={runtime.build_id}")
        response = client.get(asset)
        assert response.status_code == 200
        assert response.headers["cache-control"] == "no-store"
    assert html.headers["cache-control"] == "no-store"
    headers = {"Sec-Fetch-Mode": "cors", HEADER: runtime.build_id}
    assert client.get("/api/health", headers=headers).json()["build_id"] == runtime.build_id
    assert "__PRODUCT_BUILD_ID__" not in client.get("/static/index.html").text


def test_stale_or_missing_browser_build_never_enters_mutation_or_chat(coherent_app, monkeypatch):
    client, runtime, _, _ = coherent_app
    dispatched = []

    def no_service_dispatch():
        dispatched.append("entered_service")
        raise AssertionError("A stale browser entered the application service")

    monkeypatch.setattr(runtime, "require_service", no_service_dispatch)
    paths = [
        "/api/sessions",
        "/api/sessions/session_demo/draft/operations",
        "/api/sessions/session_demo/preview",
        "/api/sessions/session_demo/accept",
        "/api/sessions/session_demo/copilot/messages",
    ]
    for headers in (
        {"Sec-Fetch-Mode": "cors"},
        {"Sec-Fetch-Site": "same-origin"},
        {HEADER: "old-build"},
        {"Sec-Fetch-Mode": "cors", HEADER: ""},
    ):
        for path in paths:
            response = client.post(path, headers=headers, json={})
            assert response.status_code == 409, (path, response.text)
            assert response.json() == {"detail": "product_client_build_mismatch"}
            assert response.headers["cache-control"] == "no-store"
    assert client.get("/api/health", headers={HEADER: "old-build"}).status_code == 409
    assert dispatched == []


def test_current_build_preserves_origin_token_and_revision_boundaries(coherent_app):
    client, runtime, _, _ = coherent_app
    browser = {"Sec-Fetch-Mode": "cors", HEADER: runtime.build_id}
    rejected = client.post("/api/sessions", headers={**browser, "Origin": "https://evil.example"}, json={})
    assert rejected.status_code == 403
    assert rejected.json()["detail"] == "origin_not_allowed"
    # Origin alone remains a documented programmatic client, not browser identity.
    created = client.post("/api/sessions", headers={"Origin": "http://127.0.0.1:8127"}, json={})
    assert created.status_code == 200
    payload = created.json()
    session = payload["session"]
    path = f"/api/sessions/{session['session_id']}/draft/undo"
    assert client.post(path, headers=browser, json={"expected_revision": session["revision"]}).status_code == 403
    stale = client.post(
        path,
        headers={**browser, "X-Session-Token": payload["mutation_token"]},
        json={"expected_revision": session["revision"] + 1},
    )
    assert stale.status_code == 409
    assert stale.json()["detail"] == "stale_session_revision"


def test_source_drift_blocks_before_dispatch_and_restart_rebinds(coherent_app, monkeypatch):
    client, runtime, source, new_app = coherent_app
    old_build = runtime.build_id
    original = source.read_bytes()
    source.write_text("version = 2\n", encoding="utf-8")
    dispatched = []
    monkeypatch.setattr(runtime, "require_service", lambda: dispatched.append(True))
    for path in ("/api/health", "/app/itinerary", "/static/js/app.js"):
        response = client.get(path)
        assert response.status_code == 503, (path, response.text)
        assert response.json() == {"detail": "product_server_build_changed"}
    blocked = client.post("/api/sessions", headers={HEADER: old_build}, json={})
    assert blocked.status_code == 503
    assert dispatched == []
    source.write_bytes(original)
    assert client.get("/api/health").status_code == 503, "Detected drift requires restart even if files are restored"
    source.write_text("version = 2\n", encoding="utf-8")
    with TestClient(new_app()) as restarted:
        new_build = restarted.get("/api/health").json()["build_id"]
        assert new_build != old_build
        assert restarted.post("/api/sessions", headers={HEADER: old_build}, json={}).status_code == 409
        assert restarted.post("/api/sessions", headers={HEADER: new_build}, json={}).status_code == 200


def test_build_change_during_dispatch_does_not_hide_a_completed_mutation(coherent_app, monkeypatch):
    client, runtime, source, _ = coherent_app
    original_create = runtime.service.create_session

    def mutate_during_dispatch(run_id):
        result = original_create(run_id)
        source.write_text("version = 2\n", encoding="utf-8")
        return result

    monkeypatch.setattr(runtime.service, "create_session", mutate_during_dispatch)
    completed = client.post("/api/sessions", json={})
    assert completed.status_code == 200
    assert completed.json()["session"]["session_id"]
    assert client.get("/api/health").status_code == 503
