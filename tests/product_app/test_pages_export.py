from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from scripts.export_product_pages import export_pages

ROOT = Path(__file__).resolve().parents[2]


class HealthResponse(io.BytesIO):
    status = 200


def export(tmp_path, monkeypatch, *, health=None, key="sb_publishable_test", api="https://api.example.test"):
    monkeypatch.setattr("scripts.export_product_pages.product_build_id", lambda _: "build-123")
    payload = (
        health
        if health is not None
        else {
            "product_id": "itinerary-repair-copilot",
            "build_id": "build-123",
            "status": "ready",
        }
    )
    requests = []

    def opener(request, **kwargs):
        requests.append((request.full_url, kwargs))
        return HealthResponse(json.dumps(payload).encode())

    result = export_pages(
        root=ROOT,
        output=tmp_path / "pages",
        api_origin=api,
        supabase_url="https://project.supabase.co",
        publishable_key=key,
        opener=opener,
    )
    return result, requests


def test_pages_export_preserves_subpath_routes_and_public_build_contract(tmp_path, monkeypatch):
    receipt, requests = export(tmp_path, monkeypatch)
    page = (tmp_path / "pages/index.html").read_text(encoding="utf-8")
    assert 'href="./static/css/app.css?v=build-123"' in page
    assert 'src="./static/js/deployment.js?v=build-123"' in page
    assert "__PRODUCT_BUILD_ID__" not in page
    assert 'name="referrer" content="strict-origin-when-cross-origin"' in page
    assert "connect-src 'self' https://api.example.test https://project.supabase.co" in page
    assert requests == [("https://api.example.test/healthz", {"timeout": 20})]
    assert receipt["build_id"] == "build-123"
    assert (tmp_path / "pages/.nojekyll").exists()
    config = (tmp_path / "pages/static/js/deployment.js").read_text(encoding="utf-8")
    assert json.loads(config.removeprefix("window.PRODUCT_DEPLOYMENT = ").strip().removesuffix(";")) == {
        "api_origin": "https://api.example.test",
        "supabase_url": "https://project.supabase.co",
        "supabase_publishable_key": "sb_publishable_test",
        "static_frontend": True,
    }
    assert not (tmp_path / "pages/static/index.html").exists()


@pytest.mark.parametrize(
    "health",
    [
        {},
        {"product_id": "wrong", "status": "ready", "build_id": "build-123"},
        {"product_id": "itinerary-repair-copilot", "status": "ready", "build_id": "old"},
        {"product_id": "itinerary-repair-copilot", "status": "unavailable", "build_id": "build-123"},
    ],
)
def test_missing_or_stale_backend_never_creates_export(tmp_path, monkeypatch, health):
    with pytest.raises(ValueError, match="matching_backend_build_required"):
        export(tmp_path, monkeypatch, health=health)
    assert not (tmp_path / "pages").exists()


@pytest.mark.parametrize("key", ["sb_secret_never_publish", "service_role", "invalid"])
def test_secret_or_invalid_key_never_creates_export(tmp_path, monkeypatch, key):
    with pytest.raises(ValueError, match="supabase_publishable_key_invalid"):
        export(tmp_path, monkeypatch, key=key)
    assert not (tmp_path / "pages").exists()


def test_existing_export_is_preserved(tmp_path, monkeypatch):
    old = tmp_path / "pages/index.html"
    old.parent.mkdir()
    old.write_text("old site", encoding="utf-8")
    with pytest.raises(ValueError, match="export_directory_must_be_empty"):
        export(tmp_path, monkeypatch)
    assert old.read_text(encoding="utf-8") == "old site"
