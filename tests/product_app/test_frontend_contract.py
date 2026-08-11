from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STATIC = ROOT / "src" / "itinerary_system" / "product_app" / "static"
APP_JS = STATIC / "js" / "app.js"
COPILOT_JS = STATIC / "js" / "copilot.js"
INDEX = STATIC / "index.html"
REGISTRY = ROOT / "configs" / "product_app_registry.json"


def test_frontend_checks_health_before_map_configuration_and_session_creation() -> None:
    source = APP_JS.read_text(encoding="utf-8")

    health = source.index('api("/api/health")')
    core_gate = source.index("if (!health.core_ready)")
    map_config = source.index('api("/api/map/config")')
    session = source.index('api("/api/sessions"')
    assert health < core_gate < map_config < session
    assert "renderCoreRecovery(health)" in source
    assert "No session was created and all mutation controls remain unavailable." in source


def test_frontend_exposes_degraded_runtime_and_text_route_fallback() -> None:
    source = APP_JS.read_text(encoding="utf-8")
    markup = INDEX.read_text(encoding="utf-8")

    assert 'id="runtime-banner"' in markup
    assert 'role="status"' in markup
    assert "Limited local runtime" in source
    assert "The textual route remains available." in source
    assert 'class="textual-map-alternative"' in source
    assert "Text route description" in source
    assert "state.mapConfig?.status !== \"ready\"" in source
    assert "#layers-button\").disabled = !mapReady" in source


def test_frontend_provider_boundary_disables_copilot_without_silent_fallback() -> None:
    source = COPILOT_JS.read_text(encoding="utf-8")
    app_source = APP_JS.read_text(encoding="utf-8")

    assert 'interaction().provider === "openai"' in source
    assert '"OpenAI Copilot" : "Deterministic demo"' in source
    assert "return interaction().enabled === true" in source
    assert "control.disabled = !available()" in source
    assert "visible trip context, your message, and a bounded recent conversation window are sent to OpenAI" in source
    assert "requests stay on this computer" in source
    assert "copilot.update()" in app_source


def test_acceptance_controls_are_truthfully_deferred_to_w5() -> None:
    source = APP_JS.read_text(encoding="utf-8")

    assert 'id="accept-repair" disabled' in source
    assert 'id="keep-original" disabled' in source
    assert "Acceptance and Keep original remain disabled until the W5" in source
    assert "/accept`" not in source
    assert "/keep-original`" not in source
    assert "Legacy decision files are preserved but not trusted or extended." in source


def test_frontend_uses_workspace_context_v1_operation_name_and_no_mapbox_initialization() -> None:
    source = APP_JS.read_text(encoding="utf-8")

    assert 'addDraft("lock_stop"' in source
    assert "mark_locked" not in source
    assert "mapboxgl.Map" not in source
    assert "maplibregl.Map" not in source
    assert "addProtocol" not in source
    assert "/api/map/config" in source


def test_changed_product_text_is_strict_utf8_without_mojibake() -> None:
    forbidden = ("\ufffd", "\u9225", "\u923b", "\u9241", "\u8113", "\u9451")
    for path in (APP_JS, INDEX, REGISTRY):
        text = path.read_bytes().decode("utf-8", errors="strict")
        assert all(token not in text for token in forbidden), path


def test_direct_file_opening_explains_the_supported_launcher() -> None:
    markup = INDEX.read_text(encoding="utf-8")
    stylesheet = (STATIC / "css" / "app.css").read_text(encoding="utf-8")

    assert "This HTML file is not the application launcher." in markup
    assert "OPEN_ITINERARY_COPILOT.cmd" in markup
    assert 'href="http://127.0.0.1:8127/app"' in markup
    assert ".direct-file-help { display: none; }" in stylesheet
