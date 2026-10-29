"""Real original/accepted journeys with labeled map-config fault injection.

Run only against a fresh disposable deterministic app. Optimizer, preview,
acceptance, session storage, route geometry, and map rendering are not mocked.
The parameterized test modifies only GET /api/map/config after plan setup. The last
explicitly hides renderable base tiles to reproduce aggregate-ready/blank-map
disagreement. No chat is submitted, and no API response is mocked in test two.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit

import pytest

pytest.importorskip("playwright.sync_api")

from test_e2e1_live_copilot_chat import (
    CHROME,
    _capture_ready,
    _http_record,
    _independent_geometry_hash,
    _map_ack,
    _preserve_browser_evidence,
    _write_json,
    expect,
    sync_playwright,
)

BASE_URL = os.environ.get("PRODUCT_E2E_BASE_URL")
DISPOSABLE = os.environ.get("PRODUCT_E2E_MAP_RECOVERY_DISPOSABLE") == "1"


@pytest.mark.skipif(
    not BASE_URL or not DISPOSABLE,
    reason="Set PRODUCT_E2E_BASE_URL and PRODUCT_E2E_MAP_RECOVERY_DISPOSABLE=1 for a fresh isolated app",
)
@pytest.mark.parametrize("plan_context", ["original", "accepted_child"])
def test_active_map_reload_recovers_without_replacing_session_or_plan(tmp_path: Path, plan_context: str) -> None:
    evidence = Path(os.environ.get("PRODUCT_E2E_MAP_RECOVERY_EVIDENCE_DIR", tmp_path)) / plan_context
    evidence.mkdir(parents=True, exist_ok=True)
    console_errors: list[str] = []
    request_failures: list[str] = []
    http_errors: list[str] = []
    request_log: list[dict[str, object]] = []
    response_log: list[dict[str, object]] = []
    injection_log: list[dict[str, object]] = []
    checkpoints: list[dict[str, object]] = []
    phase = "initial_parent"
    fault = {"mode": "off", "injected": 0, "requests": 0}

    with sync_playwright() as runtime:
        launch: dict[str, object] = {"headless": True}
        if CHROME.is_file():
            launch["executable_path"] = str(CHROME)
        browser = runtime.chromium.launch(**launch)
        context = browser.new_context(viewport={"width": 1440, "height": 900})
        context.add_init_script("""document.addEventListener('click', event => {
          if (!event.target.closest?.('#retry-map-runtime')) return;
          (window.__mapRetryInputs ||= []).push({selector:'#retry-map-runtime', trusted:event.isTrusted,
            clicked_at:new Date().toISOString(), map_stage:{...document.querySelector('.map-stage')?.dataset},
            map_ready:document.querySelector('#geographic-map')?.dataset.mapReady,
            status:document.querySelector('#map-render-status')?.textContent});
        }, true);""")
        page = context.new_page()
        page.on("console", lambda message: console_errors.append(message.text) if message.type == "error" else None)
        page.on("pageerror", lambda error: console_errors.append(str(error)))
        page.on(
            "requestfailed",
            lambda request: request_failures.append(f"{request.method} {request.url}: {request.failure}"),
        )

        def record_request(request) -> None:
            if urlsplit(request.url).path.startswith("/api/"):
                request_log.append(
                    {
                        "phase": phase,
                        "method": request.method,
                        "url": request.url,
                        "request_body_utf8": request.post_data or "",
                    }
                )

        def record_response(response) -> None:
            if not urlsplit(response.url).path.startswith("/api/"):
                return
            if response.status >= 400:
                http_errors.append(f"{response.status} {response.url}")
            row = {
                "phase": phase,
                "observed_at": datetime.now(UTC).isoformat(),
                "fault_injection_scope": "GET /api/map/config only after active-plan setup; removed before actual Retry",
                **_http_record(response),
            }
            filename = f"http_{len(response_log) + 1:03d}.json"
            _write_json(evidence / filename, row)
            response_log.append({"file": filename, "phase": phase, "method": row["method"], "url": response.url})

        page.on("request", record_request)
        page.on("response", record_response)

        def diagnostics(label: str) -> dict[str, object]:
            observed = page.evaluate("""() => ({
              observed_at: new Date().toISOString(),
              geographic_map: {...document.querySelector('#geographic-map')?.dataset},
              map_stage: {...document.querySelector('.map-stage')?.dataset},
              map_status: document.querySelector('#map-render-status')?.textContent,
              map_status_code: document.querySelector('#map-render-status')?.dataset.mapStatusCode,
              runtime_banner: document.querySelector('#runtime-banner')?.textContent,
              retry_visible: !!document.querySelector('#retry-map-runtime'),
              inspector_title: document.querySelector('#map-context-title')?.textContent,
              viewport: {width: innerWidth, height: innerHeight},
            })""")
            row = {"label": label, "fault_mode": fault["mode"], **observed}
            checkpoints.append(row)
            _write_json(evidence / f"{label}_diagnostics.json", row)
            return row

        def inject_config(route) -> None:
            assert route.request.method == "GET"
            fault["requests"] += 1
            if fault["mode"] == "off" or (fault["mode"] == "once" and fault["injected"]):
                route.continue_()
                return
            # Fetch the actual endpoint first and preserve both exact bodies.
            upstream = route.fetch()
            original = upstream.text()
            body = json.loads(original)
            assert upstream.status == 200
            assert body["schema_version"] == "product-map-configuration-v2"
            assert body["status"] == "ready", "fault injection requires a genuinely healthy underlying service"
            body["status"] = "failed"
            injected = json.dumps(body, separators=(",", ":"))
            fault["injected"] += 1
            injection_log.append(
                {
                    "phase": phase,
                    "classification": "explicit_deterministic_http_fault_injection",
                    "method": route.request.method,
                    "url": route.request.url,
                    "request_body_utf8": route.request.post_data or "",
                    "upstream_response_status": upstream.status,
                    "upstream_response_body_utf8": original,
                    "injected_response_status": 200,
                    "injected_response_body_utf8": injected,
                    "reason": "simulate a transient or persistent non-ready local map health/config result",
                    "headers_persisted": False,
                }
            )
            _write_json(evidence / "fault_injections.json", injection_log)
            route.fulfill(response=upstream, body=injected)

        with _preserve_browser_evidence(
            context,
            page,
            evidence,
            {
                "console_errors": console_errors,
                "request_failures": request_failures,
                "http_errors": http_errors,
            },
        ):
            try:
                health_response = context.request.get(f"{BASE_URL}/api/health")
                _write_json(
                    evidence / "preflight_health.json",
                    {
                        "method": "GET",
                        "url": health_response.url,
                        "response_status": health_response.status,
                        "response_body_utf8": health_response.text(),
                    },
                )
                assert health_response.status == 200
                health = health_response.json()
                assert health["components"]["openai"]["status"] == "disabled"
                assert health["components"]["openai"]["code"] == "deterministic_adapter_selected"
                assert health["components"]["map"]["status"] == "ready"
                assert health["components"]["routing"]["status"] == "ready"

                with page.expect_response(
                    lambda response: response.url.endswith("/api/sessions") and response.request.method == "POST"
                ) as created_info:
                    page.goto(f"{BASE_URL}/app/itinerary", wait_until="domcontentloaded")
                assert created_info.value.status == 200
                created = json.loads(_http_record(created_info.value)["response_body_utf8"])
                _write_json(evidence / "initial_response.json", created)
                expect(page.locator("#app-shell")).to_be_visible(timeout=30_000)
                initial = _map_ack(page)
                _capture_ready(page, evidence / "initial_parent.png")
                page.wait_for_load_state("networkidle", timeout=20_000)
                day_five = page.locator('.day-card[data-day="5"]')
                expect(day_five).to_contain_text("Bixby Creek Bridge")
                with page.expect_response(
                    lambda response: response.url.endswith("/selection") and response.request.method == "POST"
                ) as selected_info:
                    day_five.click()
                assert selected_info.value.status == 200
                active_response = selected_info.value.json()
                proposal = None
                geography_source = created
                if plan_context == "accepted_child":
                    page.locator("#map-edit-button").click()
                    page.locator("#inspect-edit-stop").click()
                    page.locator(".legacy-edit-controls summary").click()
                    page.locator("#registered-candidate").select_option("pfeiffer_big_sur_state_park")
                    with page.expect_response(
                        lambda response: (
                            response.url.endswith("/draft/operations") and response.request.method == "POST"
                        )
                    ) as draft_info:
                        page.locator("#draft-replace").click()
                    assert draft_info.value.status == 200
                    page.locator("#close-map-edit").click()
                    staged = _map_ack(page)
                    assert staged["rendered_map_plan_id"] == initial["rendered_map_plan_id"]

                    phase = "real_optimizer_preview"
                    expect(page.locator("#inline-preview")).to_be_enabled(timeout=15_000)
                    with page.expect_response(
                        lambda response: response.url.endswith("/preview") and response.request.method == "POST",
                        timeout=60_000,
                    ) as preview_info:
                        page.locator("#inline-preview").click()
                    assert preview_info.value.status == 200
                    proposal = preview_info.value.json()["session"]["proposal"]
                    _write_json(evidence / "preview_response.json", preview_info.value.json())
                    assert proposal["state"] == "eligible"
                    assert proposal["acceptance_eligible"] is True
                    assert proposal["runtime_execution"], "preview must contain the actual optimizer execution receipt"
                    expect(page.locator(".map-stage")).to_have_attribute(
                        "data-rendered-map-plan-id",
                        proposal["child_plan_id"],
                        timeout=30_000,
                    )
                    expect(page.locator("#accept-repair")).to_be_enabled(timeout=30_000)
                    preview_ack = _map_ack(page)
                    assert preview_ack["rendered_map_plan_id"] == proposal["child_plan_id"]
                    assert preview_ack["active_plan_content_hash"] == proposal["child_plan_content_hash"]

                    phase = "explicit_acceptance"
                    dialogs: list[str] = []

                    with page.expect_response(
                        lambda response: response.url.endswith("/accept") and response.request.method == "POST",
                        timeout=30_000,
                    ) as accept_info:
                        page.locator("#accept-repair").click()
                        expect(page.get_by_role("dialog", name="Confirm trip decision", exact=True)).to_be_visible()
                        dialogs.append(page.locator("#acceptance-message").inner_text())
                        page.get_by_role("dialog", name="Confirm trip decision", exact=True).get_by_role(
                            "button", name="Accept repair", exact=True
                        ).click()
                    assert accept_info.value.status == 200
                    accepted = accept_info.value.json()
                    _write_json(evidence / "accept_response.json", accepted)
                    active_response = geography_source = accepted
                    assert dialogs == ["Accept this independently evaluated repair as the durable trip plan?"]
                    assert accepted["session"]["accepted_plan_id"] == proposal["child_plan_id"]
                session = active_response["session"]
                session_id = session["session_id"]
                _write_json(evidence / "active_session_response.json", active_response)
                accepted_ack = _map_ack(page)
                assert accepted_ack["rendered_map_plan_id"] == session["accepted_plan_id"]
                if proposal:
                    assert accepted_ack["rendered_geometry_hash"] == preview_ack["rendered_geometry_hash"]
                else:
                    assert accepted_ack["rendered_map_plan_id"] == initial["rendered_map_plan_id"]
                geography = next(
                    plan
                    for plan in geography_source["workspace"]["geography"]["plans"]
                    if plan["plan_id"] == session["accepted_plan_id"]
                )
                assert geography["role"] == ("accepted" if proposal else "original")
                assert _independent_geometry_hash(page, geography) == accepted_ack["rendered_geometry_hash"]
                expected_markers = sorted(
                    feature["properties"]["stop_id"] for feature in geography["stops"]["features"]
                )
                assert ("pfeiffer_big_sur_state_park" in expected_markers) is bool(proposal)
                assert ("bixby_creek_bridge_viewpoint" in expected_markers) is not bool(proposal)
                _capture_ready(page, evidence / "active_before_reload.png")

                def assert_same_plan(label: str) -> dict[str, object]:
                    ack = _map_ack(page)
                    assert ack["rendered_map_plan_id"] == session["accepted_plan_id"]
                    assert ack["active_plan_content_hash"] == geography["content_hash"]
                    assert ack["rendered_geometry_hash"] == accepted_ack["rendered_geometry_hash"]
                    assert sorted(ack["rendered_marker_ids"]) == expected_markers
                    expected_name = "Pfeiffer Big Sur State Park" if proposal else "Bixby Creek Bridge"
                    expect(page.locator("#map-context-title")).to_contain_text(expected_name)
                    expect(page.locator('.day-card[data-day="5"]')).to_contain_text(expected_name)
                    diagnostics(label)
                    _capture_ready(page, evidence / f"{label}.png")
                    return ack

                def reload_same_session() -> dict[str, object]:
                    with page.expect_response(
                        lambda response: (
                            response.url == f"{BASE_URL}/api/sessions/{session_id}" and response.request.method == "GET"
                        ),
                        timeout=45_000,
                    ) as reload_info:
                        page.reload(wait_until="domcontentloaded")
                    assert reload_info.value.status == 200
                    restored_payload = reload_info.value.json()
                    restored = restored_payload["session"]
                    for field in (
                        "session_id",
                        "accepted_plan_id",
                        "conversation_id",
                        "selected_stop_id",
                        "selected_day",
                    ):
                        assert restored[field] == session[field], field
                    restored_plan = [
                        plan
                        for plan in restored_payload["workspace"]["geography"]["plans"]
                        if plan["plan_id"] == session["accepted_plan_id"]
                    ]
                    assert restored_plan == [geography]
                    expect(page.locator("#loading-screen")).to_be_hidden(timeout=45_000)
                    return restored

                # These reloads use the actual endpoint with no interception.
                for index in range(1, 4):
                    phase = f"normal_reload_{index}"
                    reload_same_session()
                    assert_same_plan(phase)

                phase = "fault_injected_once"
                fault.update(mode="once", injected=0, requests=0)
                page.route("**/api/map/config", inject_config)
                reload_same_session()
                assert_same_plan("recovered_after_one_nonready_config")
                assert fault["injected"] == 1
                assert fault["requests"] == 2, "one non-ready config must trigger exactly one recovery attempt"
                expect(page.locator("#runtime-banner")).to_be_hidden()

                phase = "fault_injected_persistent"
                fault.update(mode="persistent", injected=0, requests=0)
                reload_same_session()
                expect(page.locator("#retry-map-runtime")).to_be_visible()
                expect(page.locator("#retry-map-runtime")).to_be_enabled()
                expect(page.locator("#layers-button")).to_be_disabled()
                assert fault["requests"] == fault["injected"] == 3, "recovery is bounded to three attempts"
                blocked = diagnostics("persistent_failure_no_ready_claim")
                assert blocked["map_stage"].get("renderedMapPlanId") is None
                assert blocked["geographic_map"].get("mapReady") != "true"
                assert "could not be verified" in blocked["map_status"]
                # Intentionally no screenshot: the map-ready condition is false.
                before_retry_posts = [row for row in request_log if row["method"] == "POST"]

                phase = "explicit_retry_after_fault_removed"
                fault.update(mode="off", injected=0, requests=0)
                page.unroute("**/api/map/config", inject_config)
                with page.expect_response(
                    lambda response: response.url.endswith("/api/map/config") and response.request.method == "GET"
                ) as retry_info:
                    page.locator("#retry-map-runtime").click()
                assert retry_info.value.status == 200 and retry_info.value.json()["status"] == "ready"
                retry_inputs = page.evaluate("window.__mapRetryInputs")
                _write_json(evidence / "retry_control_input.json", retry_inputs)
                _write_json(evidence / "retry_response.json", _http_record(retry_info.value))
                assert len(retry_inputs) == 1 and retry_inputs[0]["trusted"] is True
                assert retry_inputs[0]["map_stage"].get("renderedMapPlanId") is None
                assert retry_inputs[0]["map_ready"] != "true"
                final_ack = assert_same_plan("recovered_after_explicit_retry")
                expect(page.locator("#runtime-banner")).to_be_hidden()
                expect(page.locator("#layers-button")).to_be_enabled()
                assert [row for row in request_log if row["method"] == "POST"] == before_retry_posts
                assert (
                    len(
                        [
                            row
                            for row in request_log
                            if row["method"] == "POST" and urlsplit(row["url"]).path == "/api/sessions"
                        ]
                    )
                    == 1
                )
                assert not any("/copilot/messages" in row["url"] for row in request_log)
                assert console_errors == []
                assert request_failures == []
                assert http_errors == []
                _write_json(
                    evidence / "trace.json",
                    {
                        "schema_version": "map-config-recovery-browser-v2",
                        "context": plan_context,
                        "classification": "real_local_backend_with_explicit_map_configuration_fault_injection",
                        "session_id": session_id,
                        "parent_plan_id": initial["active_plan_id"],
                        "active_plan_id": geography["plan_id"],
                        "active_plan_content_hash": geography["content_hash"],
                        "child_plan_id": proposal["child_plan_id"] if proposal else None,
                        "child_plan_content_hash": proposal["child_plan_content_hash"] if proposal else None,
                        "certificate_id": proposal["certificate_id"] if proposal else None,
                        "runtime_execution": proposal["runtime_execution"] if proposal else None,
                        "rendered_map_plan_id": final_ack["rendered_map_plan_id"],
                        "rendered_geometry_hash": final_ack["rendered_geometry_hash"],
                        "rendered_marker_ids": final_ack["rendered_marker_ids"],
                        "normal_unmodified_reloads": 3,
                        "injected_config_responses": len(injection_log),
                        "maximum_attempts_per_cycle": 3,
                        "new_sessions_after_initial": 0,
                        "chat_messages_submitted": 0,
                        "provider_measurement": "Use the enclosing application-process isolation receipt; this browser test submits no chat.",
                        "screenshots_require_stable_map_ready": True,
                        "console_errors": console_errors,
                        "request_failures": request_failures,
                        "http_errors": http_errors,
                        "assertions_passed": True,
                    },
                )
                _write_json(
                    evidence / "llm_io.json",
                    {
                        "provider_call_count": 0,
                        "llm_input": None,
                        "llm_output": None,
                        "reason": "No chat was submitted; measured zero requires the enclosing finalized provider-isolation receipt.",
                    },
                )
            finally:
                _write_json(evidence / "browser_http_index.json", response_log)
                _write_json(evidence / "browser_requests.json", request_log)
                _write_json(evidence / "fault_injections.json", injection_log)
                _write_json(evidence / "checkpoints.json", checkpoints)
        browser.close()


@pytest.mark.skipif(not BASE_URL or not DISPOSABLE, reason="Requires the guarded disposable local application")
def test_loaded_source_cannot_acknowledge_a_viewport_without_renderable_tiles(tmp_path: Path) -> None:
    evidence = Path(os.environ.get("PRODUCT_E2E_MAP_RECOVERY_EVIDENCE_DIR", tmp_path)) / "tile_coverage"
    evidence.mkdir(parents=True, exist_ok=True)
    instrumentation = """(() => {
      window.__coverageMaps = [];
      const constructors = new WeakMap();
      let api;
      Object.defineProperty(window, 'maplibregl', {configurable: true,
        get: () => api,
        set: value => { api = new Proxy(value, {get(target, key, receiver) {
          const original = Reflect.get(target, key, receiver);
          if (key !== 'Map' || typeof original !== 'function') return original;
          if (!constructors.has(original)) constructors.set(original, new Proxy(original, {
            construct(constructor, args, newTarget) {
              const map = Reflect.construct(constructor, args, newTarget);
              window.__coverageMaps.push(map); return map;
            }
          }));
          return constructors.get(original);
        }}); }
      });
    })();"""
    (evidence / "browser_instrumentation.js").write_text(instrumentation, encoding="utf-8")
    errors: list[str] = []
    failures: list[str] = []
    posts: list[str] = []
    with sync_playwright() as runtime:
        browser = runtime.chromium.launch(
            headless=True, **({"executable_path": str(CHROME)} if CHROME.is_file() else {})
        )
        context = browser.new_context(viewport={"width": 1440, "height": 900})
        context.add_init_script(instrumentation)
        page = context.new_page()
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.on("console", lambda message: errors.append(message.text) if message.type == "error" else None)
        page.on("requestfailed", lambda request: failures.append(f"{request.url}: {request.failure}"))
        page.on("request", lambda request: posts.append(request.url) if request.method == "POST" else None)
        with _preserve_browser_evidence(
            context, page, evidence, {"console_errors": errors, "request_failures": failures}
        ):
            page.goto(f"{BASE_URL}/app/itinerary", wait_until="domcontentloaded")
            before = _map_ack(page)
            _capture_ready(page, evidence / "before_tile_fault.png")
            initial_posts = list(posts)
            fault = page.evaluate("""() => {
              const map = window.__coverageMaps.findLast(item => item.getContainer().isConnected
                && item.getContainer().id === 'geographic-map');
              if (!map) throw new Error('instrumented_runtime_map_missing');
              const sourceId = Object.keys(map.getStyle().sources).find(id => map.getStyle().sources[id].type === 'vector');
              const manager = map.style.tileManagers[sourceId];
              const ids = manager.getVisibleCoordinates();
              if (!ids.length) throw new Error('real_loaded_tiles_missing');
              window.__coverageFault = {map, manager, original:manager.getVisibleCoordinates};
              manager.getVisibleCoordinates = () => [];
              map.triggerRepaint();
              return {classification:'explicit_renderer_tile_visibility_fault', source_id:sourceId,
                tile_ids:ids.map(id => ({canonical:id.canonical,wrap:id.wrap})),
                aggregate_tiles_loaded:map.areTilesLoaded(), observed_at:new Date().toISOString()};
            }""")
            _write_json(evidence / "fault_input.json", fault)
            expect(page.locator("#geographic-map")).to_have_attribute("data-map-ready", "false", timeout=20_000)
            expect(page.locator("#map-render-status")).to_have_attribute(
                "data-map-status-code", "map_tile_coverage_incomplete"
            )
            blocked = page.evaluate("""async () => {
              const script = document.querySelector('script[type="module"]').src;
              const module = await import(new URL('map-controller-v2.js', script).href);
              return {readiness:module.mapTileReadiness(window.__coverageFault.map),
                map:{...document.querySelector('#geographic-map').dataset}, observed_at:new Date().toISOString()};
            }""")
            _write_json(evidence / "blocked_output_no_screenshot.json", blocked)
            assert blocked["readiness"]["tileCoverageRatio"] == 0
            assert blocked["readiness"]["tilesLoaded"] is False
            assert not blocked["map"].get("renderedMapPlanId")
            page.evaluate("""() => {const fault=window.__coverageFault;
              fault.manager.getVisibleCoordinates=fault.original; fault.map.triggerRepaint();}""")
            after = _map_ack(page)
            _capture_ready(page, evidence / "after_tile_recovery.png")
            assert posts == initial_posts
            assert before["rendered_map_plan_id"] == after["rendered_map_plan_id"]
            assert before["rendered_geometry_hash"] == after["rendered_geometry_hash"]
            assert before["rendered_marker_ids"] == after["rendered_marker_ids"]
            assert errors == failures == []
            _write_json(
                evidence / "trace.json",
                {
                    "before": before,
                    "after": after,
                    "blocked_readiness": blocked["readiness"],
                    "backend_mutations_after_load": 0,
                    "console_errors": errors,
                    "request_failures": failures,
                    "assertions_passed": True,
                },
            )
        browser.close()
