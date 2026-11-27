from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit

import pytest

playwright = pytest.importorskip("playwright.sync_api")
expect = playwright.expect
sync_playwright = playwright.sync_playwright

BASE_URL = os.environ.get("PRODUCT_E2E_BASE_URL")
EVIDENCE_DIR = os.environ.get("PRODUCT_E2E_LIVE_COPILOT_EVIDENCE_DIR")
APPROVED_PROMPT = os.environ.get("PRODUCT_E2E_LIVE_COPILOT_PROMPT")
CHROME = Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe")
TARGET_STOP_ID = "bixby_creek_bridge_viewpoint"
CANDIDATE_ID = "pfeiffer_big_sur_state_park"
COMBINED_PROMPT = (
    "Replace Bixby Creek Bridge with Pfeiffer Big Sur State Park; set Griffith Observatory duration to 90 minutes."
)


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _runtime_receipt(page, directory: Path) -> dict:
    repository = Path(__file__).resolve().parents[3]
    if str(repository / "scripts") not in sys.path:
        sys.path.insert(0, str(repository / "scripts"))
    from e2e1_runtime_environment import browser_identity, capture_runtime_environment, validate_runtime_environment

    run_id = os.environ["PRODUCT_E2E_RUN_ID"]
    node_id = os.environ["PYTEST_CURRENT_TEST"].removesuffix(" (call)")
    filename = f"runtime_environment_{_hash_text(node_id)[:16]}.json"
    path = directory / filename
    browser = browser_identity(
        page.context.browser, launch_choice="installed_chrome" if CHROME.is_file() else "playwright_chromium"
    )
    if path.is_file():
        record = json.loads(path.read_text(encoding="utf-8"))
        validate_runtime_environment(record, run_id=run_id, root=repository)
        assert record["browser"] == browser, "Actual browser differs from the recorded journey browser"
    else:
        registered = json.loads((repository / "configs/product_route_delta.json").read_text(encoding="utf-8"))
        descriptor = (repository / registered["descriptor_path"]).resolve()
        assert descriptor.is_relative_to((repository / "results/route_deltas").resolve())
        candidates = subprocess.run(
            ["docker", "ps", "--filter", "publish=5000", "--filter", "status=running", "--format", "{{.ID}}"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.splitlines()
        assert len(candidates) == 1 and re.fullmatch(r"[0-9a-f]{12,64}", candidates[0]), (
            "Expected one observed loopback OSRM container"
        )
        scripts = page.locator('script[src*="maplibre-gl.js"]').evaluate_all(
            "elements => elements.map(element => element.src)"
        )
        assert len(scripts) == 1, "Expected the actual page's one MapLibre script"
        map_url = urlsplit(scripts[0])
        record = capture_runtime_environment(
            root=repository,
            run_id=run_id,
            actual_browser=browser,
            osrm_container_id=candidates[0],
            osrm_provenance_path=descriptor.parent / "source-provenance.json",
            app_origin=os.environ["PRODUCT_E2E_BASE_URL"],
            map_origin=f"{map_url.scheme}://{map_url.netloc}",
        )
        _write_json(path, record)
    return {
        "artifact": filename,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "test_nodeid": node_id,
        "browser": browser,
    }


def _capture_ready(page, path: Path) -> None:
    snapshot = """expected => {
      const elements = [...document.querySelectorAll('#geographic-map, [data-compare-map-canvas]')]
        .filter(element => {const rect = element.getBoundingClientRect(); return rect.width > 0 && rect.height > 0;});
      const element = elements[0];
      const canvas = element?.querySelector('canvas');
      if (!canvas || element.dataset.mapReady !== 'true') return false;
      const bounds = element.getBoundingClientRect();
      const surface = canvas.getBoundingClientRect();
      const coverage = Number(element.dataset.mapTileCoverage);
      if (!Number.isFinite(coverage) || coverage < 0.95 || coverage > 1
          || element.dataset.mapCoverageMethod !== 'loaded_tile_projected_union_area_v1'
          || innerWidth !== expected.width || innerHeight !== expected.height
          || bounds.width <= 0 || bounds.height <= 0
          || Math.abs(surface.width - bounds.width) > 1
          || Math.abs(surface.height - bounds.height) > 1
          || !element.dataset.mapReadyAt) return false;
      const visibleMaps = elements.map(item => {
        const rect = item.getBoundingClientRect();
        const canvasRect = item.querySelector('canvas')?.getBoundingClientRect();
        const ratio = Number(item.dataset.mapTileCoverage);
        if (item.dataset.mapReady !== 'true' || !item.dataset.mapReadyAt
            || item.dataset.mapCoverageMethod !== 'loaded_tile_projected_union_area_v1'
            || !Number.isFinite(ratio) || ratio < .95 || ratio > 1
            || !canvasRect || Math.abs(canvasRect.width-rect.width)>1
            || Math.abs(canvasRect.height-rect.height)>1) return null;
        return {plan_id:item.dataset.renderedMapPlanId, plan_content_hash:item.dataset.renderedPlanContentHash,
          geometry_hash:item.dataset.renderedGeometryHash, marker_ids:JSON.parse(item.dataset.renderedMarkerIds || '[]'),
          map_ready_at:item.dataset.mapReadyAt, visible_tile_area_coverage_ratio:ratio,
          container:{width:rect.width,height:rect.height},
          pane:item.closest('[data-compare-map-pane]')?.dataset.compareMapPane || 'trip'};
      });
      if (!visibleMaps.length || visibleMaps.some(item => !item)) return false;
      return {
        map_ready_at: element.dataset.mapReadyAt,
        visible_tile_area_coverage_ratio: coverage,
        coverage_method: element.dataset.mapCoverageMethod,
        container: {width: bounds.width, height: bounds.height},
        canvas: {width: surface.width, height: surface.height},
        rendered_plan_id: element.dataset.renderedMapPlanId,
        rendered_geometry_hash: element.dataset.renderedGeometryHash,
        visible_maps: visibleMaps,
        viewport: {width: innerWidth, height: innerHeight},
      };
    }"""
    expected_viewport = page.viewport_size
    deadline = time.monotonic() + 30
    for attempt in range(4):
        remaining_ms = max(1, int((deadline - time.monotonic()) * 1000))
        ready_handle = page.wait_for_function(snapshot, arg=expected_viewport, timeout=remaining_ms)
        before = ready_handle.json_value()
        ready_handle.dispose()
        runtime_receipt = _runtime_receipt(page, path.parent)
        before = page.evaluate(snapshot, expected_viewport)
        if not before:
            continue
        started_at = page.evaluate("new Date().toISOString()")
        screenshot = page.screenshot(full_page=True)
        after = page.evaluate(snapshot, expected_viewport)
        if after == before:
            readiness = {
                **before,
                "screenshot_started_at": started_at,
                "screenshot_completed_at": page.evaluate("new Date().toISOString()"),
                "stable_readiness_before_and_after": True,
                "capture_attempt": attempt + 1,
                "runtime_environment": runtime_receipt,
            }
            assert readiness["map_ready_at"] <= started_at
            path.write_bytes(screenshot)
            _write_json(path.with_suffix(".readiness.json"), readiness)
            return
        if time.monotonic() >= deadline:
            break
    raise AssertionError("Map readiness changed during screenshot capture; no unstable screenshot was saved.")


def _map_ack(page) -> dict[str, object]:
    handle = page.wait_for_function(
        """() => {
      const stage = document.querySelector('.map-stage');
      const map = document.querySelector('#geographic-map');
      if (!stage || !map || map.dataset.mapReady !== 'true'
          || !/^plan_/.test(stage.dataset.activePlanId || '')
          || !/^[0-9a-f]{16}$/.test(stage.dataset.activePlanContentHash || '')
          || !/^plan_/.test(stage.dataset.renderedMapPlanId || '')
          || !/^[0-9a-f]{64}$/.test(stage.dataset.renderedGeometryHash || '')
          || !(stage.dataset.renderedMarkerIds || '').startsWith('[')) return false;
      try {
        const markerIds = JSON.parse(stage.dataset.renderedMarkerIds);
        if (!Array.isArray(markerIds)) return false;
        return {
          preview_state: stage.dataset.previewState,
          active_plan_id: stage.dataset.activePlanId,
          active_plan_content_hash: stage.dataset.activePlanContentHash,
          rendered_map_plan_id: stage.dataset.renderedMapPlanId,
          rendered_geometry_hash: stage.dataset.renderedGeometryHash,
          rendered_marker_ids: markerIds,
        };
      } catch (_) {
        return false;
      }
    }""",
        timeout=30_000,
    )
    return handle.json_value()


def _credential_safe_body(raw: str) -> tuple[str, list[str]]:
    """Keep exact bytes unless JSON contains credentials; disclose any redaction."""
    try:
        value = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw, []
    removed: list[str] = []
    sensitive = {
        "mutation_token",
        "session_token",
        "token_verifier",
        "x-session-token",
        "authorization",
        "api_key",
        "openai_api_key",
    }

    def redact(item, path: str) -> None:
        if isinstance(item, dict):
            for key, child in item.items():
                field = f"{path}/{key}"
                if key.lower() in sensitive:
                    item[key] = "OMITTED_SECRET"
                    removed.append(field)
                else:
                    redact(child, field)
        elif isinstance(item, list):
            for index, child in enumerate(item):
                redact(child, f"{path}/{index}")

    redact(value, "")
    return (json.dumps(value, ensure_ascii=False, separators=(",", ":")) if removed else raw), removed


def _http_record(response, *, response_body: str | None = None) -> dict[str, object]:
    request_body = response.request.post_data or ""
    if response_body is None:
        response_body = response.body().decode("utf-8")
    original_request_hash = _hash_text(request_body)
    original_response_hash = _hash_text(response_body)
    request_body, request_redactions = _credential_safe_body(request_body)
    response_body, response_redactions = _credential_safe_body(response_body)
    return {
        "schema_version": "e2e1-product-http-io-v1",
        "method": response.request.method,
        "url": response.url,
        "request_body_utf8": request_body,
        "request_body_sha256": _hash_text(request_body),
        "response_status_code": response.status,
        "response_body_utf8": response_body,
        "response_body_sha256": _hash_text(response_body),
        "request_original_body_sha256": original_request_hash,
        "response_original_body_sha256": original_response_hash,
        "credential_redactions": {"request": request_redactions, "response": response_redactions},
        "body_capture": "credential_redacted" if request_redactions or response_redactions else "exact_utf8",
        "request_headers_persisted": False,
        "response_headers_persisted": False,
        "authorization_persisted": False,
        "session_token_persisted": False,
    }


def _sse_events(raw: str) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    for frame in raw.replace("\r\n", "\n").split("\n\n"):
        if not frame.strip():
            continue
        data_lines = [line.removeprefix("data:").lstrip() for line in frame.splitlines() if line.startswith("data:")]
        event = json.loads("\n".join(data_lines))
        events.append(event)
    return events


@contextmanager
def _preserve_browser_evidence(context, page, evidence_dir, errors):
    """Always finalize the real recording and preserve the observed failure."""
    try:
        yield
    except Exception as error:
        try:
            runtime_state = page.evaluate("""() => ({
              map: {...document.querySelector('#geographic-map')?.dataset},
              stage: {...document.querySelector('.map-stage')?.dataset},
              status: document.querySelector('#map-render-status')?.textContent,
              status_attributes: {...document.querySelector('#map-render-status')?.dataset},
              compare_maps: [...document.querySelectorAll('[data-compare-map-pane]')].map(pane => {
                const map = pane.querySelector('[data-compare-map-canvas]');
                const rect = map?.getBoundingClientRect();
                const canvas = map?.querySelector('canvas')?.getBoundingClientRect();
                const status = pane.querySelector('[data-compare-map-status]');
                return {pane:pane.dataset.compareMapPane,hidden:pane.hidden,
                  map:{...map?.dataset},status:status?.textContent,status_attributes:{...status?.dataset},
                  container:rect ? {x:rect.x,y:rect.y,width:rect.width,height:rect.height} : null,
                  canvas:canvas ? {width:canvas.width,height:canvas.height} : null};
              }),
              viewport: {width:innerWidth,height:innerHeight}, url:location.href,
            })""")
        except Exception as state_error:
            runtime_state = {"read_failure": type(state_error).__name__}
        _write_json(
            evidence_dir / "browser_failure.json",
            {
                "schema_version": "e2e1-browser-failure-v1",
                "assertions_passed": False,
                "error_type": type(error).__name__,
                "message": str(error),
                "observed_at": datetime.now(UTC).isoformat(),
                "runtime_state": runtime_state,
                **errors,
            },
        )
        raise
    finally:
        video = page.video
        context.close()
        if video is not None:
            video.save_as(str(evidence_dir / "website_journey.webm"))
            video.delete()


def _independent_geometry_hash(page, geography):
    # Work from the actual response, independently of the renderer's helper.
    return page.evaluate(
        """async geography => {
      const legs = structuredClone(geography.validated_legs);
      for (const feature of legs.features) delete feature.properties.role;
      const encode = value => Array.isArray(value)
        ? '[' + value.map(encode).join(',') + ']'
        : value !== null && typeof value === 'object'
          ? '{' + Object.keys(value).sort().map(key => JSON.stringify(key) + ':' + encode(value[key])).join(',') + '}'
          : JSON.stringify(value);
      const digest = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(encode(legs)));
      return [...new Uint8Array(digest)].map(byte => byte.toString(16).padStart(2, '0')).join('');
    }""",
        geography,
    )


@pytest.mark.skipif(
    not BASE_URL or not EVIDENCE_DIR or not APPROVED_PROMPT,
    reason="Set the live E2E-1 Copilot environment",
)
def test_live_website_copilot_stages_and_renders_exact_evaluated_child() -> None:
    combined = APPROVED_PROMPT == COMBINED_PROMPT
    evidence_dir = Path(EVIDENCE_DIR)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    console_errors: list[str] = []
    request_failures: list[str] = []
    http_errors: list[str] = []
    mutation_ledger: list[dict[str, object]] = []
    mutation_requests = {}

    def observe_mutation(request):
        if request.method not in {"POST", "PUT", "PATCH", "DELETE"} or not request.url.startswith(f"{BASE_URL}/api/"):
            return
        raw = request.post_data or ""
        body, redactions = _credential_safe_body(raw)
        row = {
            "sequence": len(mutation_ledger) + 1,
            "method": request.method,
            "url": request.url,
            "observed_at": datetime.now(UTC).isoformat(),
            "request_body_utf8": body,
            "request_body_sha256": _hash_text(body),
            "request_original_body_sha256": _hash_text(raw),
            "credential_redactions": redactions,
            "response_status_code": None,
            "response_observed_at": None,
        }
        mutation_ledger.append(row)
        mutation_requests[request] = row
        _write_json(evidence_dir / "product_mutation_ledger.json", mutation_ledger)

    def observe_mutation_response(response):
        row = mutation_requests.get(response.request)
        if row is not None:
            row.update(response_status_code=response.status, response_observed_at=datetime.now(UTC).isoformat())
            _write_json(evidence_dir / "product_mutation_ledger.json", mutation_ledger)

    _write_json(evidence_dir / "product_mutation_ledger.json", mutation_ledger)

    with sync_playwright() as runtime:
        launch: dict[str, object] = {
            "headless": os.environ.get("PRODUCT_E2E_HEADED") != "1",
            "slow_mo": int(os.environ.get("PRODUCT_E2E_SLOW_MO_MS", "0")),
        }
        if CHROME.is_file():
            launch["executable_path"] = str(CHROME)
        browser = runtime.chromium.launch(**launch)
        context_options: dict[str, object] = {"viewport": {"width": 1440, "height": 900}}
        if os.environ.get("PRODUCT_E2E_RECORD_VIDEO") == "1":
            context_options.update(
                record_video_dir=str(evidence_dir),
                record_video_size={"width": 1440, "height": 900},
            )
        context = browser.new_context(**context_options)
        page = context.new_page()
        page.on("console", lambda message: console_errors.append(message.text) if message.type == "error" else None)
        page.on("pageerror", lambda error: console_errors.append(str(error)))
        page.on(
            "requestfailed",
            lambda request: request_failures.append(f"{request.method} {request.url}: {request.failure}"),
        )
        page.on(
            "response",
            lambda response: (
                http_errors.append(f"{response.status} {response.url}") if response.status >= 400 else None
            ),
        )
        page.on("dialog", lambda dialog: dialog.accept())
        page.on("request", observe_mutation)
        page.on("response", observe_mutation_response)

        with _preserve_browser_evidence(
            context,
            page,
            evidence_dir,
            {
                "console_errors": console_errors,
                "request_failures": request_failures,
                "http_errors": http_errors,
            },
        ):
            page.goto(f"{BASE_URL}/app/itinerary", wait_until="domcontentloaded")
            expect(page.locator("#app-shell")).to_be_visible(timeout=15_000)
            stage = page.locator(".map-stage")
            expect(stage).to_have_attribute("data-preview-state", "PARENT_RENDERED", timeout=30_000)
            initial = _map_ack(page)
            day_five = page.locator('.day-card[data-day="5"]')
            expect(day_five).to_contain_text("Bixby Creek Bridge")
            with page.expect_response(
                lambda response: response.url.endswith("/selection") and response.request.method == "POST",
                timeout=15_000,
            ) as selection_info:
                day_five.click()
            selection_http = _http_record(selection_info.value)
            _write_json(evidence_dir / "product_initial_selection_http_io.json", selection_http)
            assert selection_info.value.status == 200
            initial_session = json.loads(selection_http["response_body_utf8"])["session"]
            assert initial_session["accepted_plan_id"] == initial["active_plan_id"]
            expect(page.locator("#map-context-title")).to_contain_text("Bixby Creek Bridge", timeout=15_000)
            _capture_ready(page, evidence_dir / "screenshot_before_submission.png")

            page.locator("#open-copilot").click()
            expect(page.locator("#copilot-dock")).to_be_visible()
            context_chips = page.locator("#context-chips").all_text_contents()
            page.evaluate(
                """() => {
                  const originalFetch = window.fetch.bind(window);
                  window.__e2eCopilotStreamRaw = null;
                  window.__e2eCopilotStreamComplete = false;
                  window.fetch = async (input, init = {}) => {
                    const response = await originalFetch(input, init);
                    const url = typeof input === "string" ? input : input.url;
                    if (
                      url.endsWith("/copilot/messages")
                      && init.headers?.Accept === "text/event-stream"
                    ) {
                      const reader = response.clone().body.getReader();
                      const decoder = new TextDecoder();
                      window.__e2eCopilotStreamRaw = '';
                      (async () => {
                        while (true) {
                          const {value, done} = await reader.read();
                          if (done) break;
                          window.__e2eCopilotStreamRaw += decoder.decode(value, {stream:true});
                        }
                        window.__e2eCopilotStreamRaw += decoder.decode();
                        window.__e2eCopilotStreamComplete = true;
                      })().catch(() => {});
                    }
                    return response;
                  };
                }"""
            )
            page.locator("#copilot-input").fill(APPROVED_PROMPT)
            if combined:
                page.evaluate("""() => {
                  window.__e2eCombinedClicks = [];
                  document.addEventListener('click', event => {
                    const button = event.target.closest('button');
                    if (button && ['Add all to draft', 'Preview combined repair'].includes(button.textContent.trim()))
                      window.__e2eCombinedClicks.push({text:button.textContent.trim(), isTrusted:event.isTrusted});
                  });
                }""")
            with page.expect_response(
                lambda response: response.url.endswith("/copilot/messages") and response.request.method == "POST",
                timeout=90_000,
            ) as response_info:
                page.locator("#copilot-form button[type='submit']").click()
            response = response_info.value
            try:
                page.wait_for_function("() => window.__e2eCopilotStreamComplete === true", timeout=90_000)
            finally:
                capture = page.evaluate(
                    "() => ({raw:window.__e2eCopilotStreamRaw, complete:window.__e2eCopilotStreamComplete})"
                )
                copilot_stream_raw = capture["raw"] if isinstance(capture["raw"], str) else ""
                copilot_http = _http_record(response, response_body=copilot_stream_raw)
                if capture["complete"] is not True:
                    copilot_http["body_capture"] = "incomplete_utf8"
                _write_json(evidence_dir / "product_copilot_http_io.json", copilot_http)
            assert response.status == 200, copilot_stream_raw
            copilot_events = _sse_events(copilot_stream_raw)
            _write_json(evidence_dir / "copilot_stream_events.json", copilot_events)
            assert [event["type"] for event in copilot_events] == [
                "message.accepted",
                "assistant.status",
                "turn.completed",
            ]
            assert [event["sequence"] for event in copilot_events] == [1, 2, 3]
            assert len({event["request_id"] for event in copilot_events}) == 1
            copilot_request = json.loads(copilot_http["request_body_utf8"])
            copilot_payload = copilot_events[-1]["payload"]
            session_id = copilot_payload["session"]["session_id"]
            assert response.url.endswith(f"/api/sessions/{session_id}/copilot/messages")
            turn = copilot_payload["turn"]
            interpretation = turn["interpretation"]
            _write_json(evidence_dir / "copilot_turn.json", turn)
            _write_json(evidence_dir / "typed_interpretation.json", interpretation)
            assert copilot_request["message"] == turn["user_message"] == APPROVED_PROMPT
            assert interpretation["state"] == "proposal_ready"
            assert len(interpretation["intents"]) == (2 if combined else 1)
            intent = interpretation["intents"][0]
            assert interpretation["schema_version"] in {"copilot-interpretation-v1", "copilot-interpretation-v2"}
            expected_intent = {
                "type": "replace_nearby",
                "target_stop_id": TARGET_STOP_ID,
                "destination_day": None,
                "candidate_id": CANDIDATE_ID,
                "preference": None,
                "clarification_question": None,
            }
            if interpretation["schema_version"] == "copilot-interpretation-v2":
                expected_intent["parameters"] = None
            assert intent == expected_intent
            if combined:
                assert interpretation["intents"][1] == {
                    "type": "set_stop_duration",
                    "target_stop_id": "griffith_observatory",
                    "destination_day": None,
                    "candidate_id": None,
                    "preference": None,
                    "clarification_question": None,
                    "parameters": {
                        "duration": {
                            "mode": "exact",
                            "preferred_minutes": 90,
                            "minimum_minutes": 90,
                            "maximum_minutes": 90,
                        }
                    },
                }
            expect(page.locator("#copilot-lifecycle")).to_have_attribute("data-state", "proposal_ready")
            actions = page.locator(".copilot-proposal-actions").last
            add_label = "Add all to draft" if combined else "Add to draft"
            expect(actions.get_by_role("button", name=add_label, exact=True)).to_be_enabled()
            _write_json(
                evidence_dir / "browser_context.json",
                {
                    "approved_prompt": APPROVED_PROMPT,
                    "session_id": session_id,
                    "context_chips": context_chips,
                    "selected_day": 5,
                    "selected_stop_id": TARGET_STOP_ID,
                },
            )
            _write_json(evidence_dir / "product_copilot_http_io.json", copilot_http)
            _write_json(evidence_dir / "copilot_stream_events.json", copilot_events)
            _write_json(evidence_dir / "copilot_turn.json", turn)
            _write_json(evidence_dir / "typed_interpretation.json", interpretation)
            _capture_ready(page, evidence_dir / "screenshot_proposal_ready.png")

            with page.expect_response(
                lambda item: item.url.endswith("/draft/operations") and item.request.method == "POST"
            ) as draft_info:
                actions.get_by_role("button", name=add_label, exact=True).click()
            draft_response = draft_info.value
            draft_http = _http_record(draft_response)
            _write_json(evidence_dir / "product_draft_http_io.json", draft_http)
            assert draft_response.status == 200, draft_http["response_body_utf8"]
            draft_request = json.loads(draft_http["request_body_utf8"])
            draft_payload = json.loads(draft_http["response_body_utf8"])
            assert draft_response.url.endswith(f"/api/sessions/{session_id}/draft/operations")
            assert draft_payload["session"]["session_id"] == session_id
            operations = draft_payload["operations"] if combined else [draft_payload["operation"]]
            operation = operations[0]
            requested_operations = draft_request["operations"] if combined else [draft_request]
            requested = requested_operations[0]
            assert requested["type"] == operation["type"] == "replace_nearby"
            assert requested["target"] == operation["target"] == TARGET_STOP_ID
            assert requested["parameters"] == operation["parameters"] == {"candidate_id": CANDIDATE_ID}
            assert requested["source"] == operation["source"] == "copilot"
            assert len(draft_payload["session"]["draft"]) == (2 if combined else 1)
            if combined:
                assert draft_payload["session"]["revision"] == draft_request["expected_revision"] + 1
                assert operations[1]["type"] == "set_stop_duration"
                assert operations[1]["target"] == "griffith_observatory"
                assert operations[1]["parameters"] == interpretation["intents"][1]["parameters"]
            expect(page.locator("#toast-region")).to_contain_text("Copilot suggestion staged", timeout=15_000)
            expect(page.locator("#copilot-lifecycle")).to_contain_text("Draft staged", timeout=15_000)
            staged = _map_ack(page)
            assert staged == initial
            _write_json(evidence_dir / "product_draft_http_io.json", draft_http)
            _write_json(evidence_dir / "draft_operation.json", operation)
            _capture_ready(page, evidence_dir / "screenshot_draft_staged.png")

            actions = page.locator(".copilot-proposal-actions").last
            preview_button = actions.get_by_role(
                "button", name="Preview combined repair" if combined else "Preview this repair", exact=True
            )
            expect(preview_button).to_be_enabled(timeout=15_000)
            preview_started_at = datetime.now(UTC).isoformat()
            with page.expect_response(
                lambda item: item.url.endswith("/preview") and item.request.method == "POST",
                timeout=90_000,
            ) as preview_info:
                preview_button.click()
            preview_response = preview_info.value
            preview_received_at = datetime.now(UTC).isoformat()
            preview_http = _http_record(preview_response)
            _write_json(evidence_dir / "product_preview_http_io.json", preview_http)
            assert preview_response.status == 200, preview_http["response_body_utf8"]
            preview_payload = json.loads(preview_http["response_body_utf8"])
            _write_json(evidence_dir / "product_preview_http_io.json", preview_http)
            _write_json(evidence_dir / "preview_response.json", preview_payload)
            assert preview_response.url.endswith(f"/api/sessions/{session_id}/preview")
            assert preview_payload["session"]["session_id"] == session_id
            proposal = preview_payload["session"]["proposal"]
            runtime_execution = proposal.get("runtime_execution")
            _write_json(evidence_dir / "runtime_execution.json", runtime_execution)
            assert proposal["execution_mode"] == (
                "atomic_combined_independent_evaluation" if combined else "deterministic_repair_pipeline"
            )
            assert runtime_execution["schema_version"] == (
                "atomic-combined-repair-execution-v1" if combined else "runtime-repair-execution-v1"
            )
            assert runtime_execution["evaluation_executed"] is True
            assert runtime_execution["evaluator"] == "PlanEvaluator.evaluate_final_plan"
            assert (
                preview_started_at
                <= runtime_execution["started_at"]
                <= runtime_execution["completed_at"]
                <= preview_received_at
            )
            assert runtime_execution["request_id"] == proposal["compiled_request"]["source_request_id"]
            assert runtime_execution["child_plan_id"] == proposal["child_plan_id"]
            assert runtime_execution["evaluation_id"] == proposal["certificate_id"]
            optimizer_execution = runtime_execution["optimizer_execution"] if combined else runtime_execution
            constraints = optimizer_execution["repair_request"]["confirmed_constraints"]
            assert TARGET_STOP_ID in constraints["must_delete"]
            assert CANDIDATE_ID in constraints["must_include"]
            assert proposal["state"] == "eligible"
            canonical_operations = [
                {key: value for key, value in row.items() if key != "created_at"} for row in operations
            ]
            assert proposal["compiled_request"]["operations"] == canonical_operations
            if combined:
                griffith = next(
                    stop
                    for stop in runtime_execution["child_plan"]["selected_stops"]
                    if stop["stop_id"] == "griffith_observatory"
                )
                assert griffith["visit_duration_minutes"] == 90
                assert griffith["duration_constraint"] == operations[1]["parameters"]["duration"]
                controls = page.evaluate("window.__e2eCombinedClicks")
                assert controls == [
                    {"text": "Add all to draft", "isTrusted": True},
                    {"text": "Preview combined repair", "isTrusted": True},
                ]
                _write_json(
                    evidence_dir / "combined_staging_observation.json",
                    {
                        "initial_map": initial,
                        "staged_map": staged,
                        "control_clicks": controls,
                    },
                )
            expect(stage).to_have_attribute("data-preview-state", "PREVIEW_RENDERED", timeout=30_000)
            rendered = _map_ack(page)
            assert rendered["rendered_map_plan_id"] == rendered["active_plan_id"] == proposal["child_plan_id"]
            assert rendered["active_plan_content_hash"] == proposal["child_plan_content_hash"]
            assert CANDIDATE_ID in rendered["rendered_marker_ids"]
            assert TARGET_STOP_ID not in rendered["rendered_marker_ids"]
            assert rendered["rendered_geometry_hash"] != initial["rendered_geometry_hash"]
            expected_geometry_hash = _independent_geometry_hash(page, proposal["geography_plan"])
            assert rendered["rendered_geometry_hash"] == expected_geometry_hash
            assert proposal["geography_plan"]["plan_id"] == proposal["child_plan_id"]
            assert preview_payload["session"]["selected_stop_id"] == CANDIDATE_ID
            expect(day_five).to_contain_text("Pfeiffer Big Sur State Park")
            assert "Bixby Creek Bridge" not in day_five.inner_text()
            metric_strip = page.locator("#metric-strip")
            expect(metric_strip).to_have_attribute("data-plan-id", proposal["child_plan_id"])
            expect(metric_strip).to_have_attribute("data-plan-content-hash", proposal["child_plan_content_hash"])
            expect(metric_strip).to_have_attribute("data-evaluation-id", proposal["certificate_id"])
            visible_metrics = metric_strip.evaluate("""element => ({
              scope: {...element.dataset},
              metrics: [...element.querySelectorAll('.metric')].map(item => ({
                scope: {...item.dataset}, label: item.querySelector('span').textContent,
                value: item.querySelector('strong').textContent,
              })),
            })""")
            _write_json(evidence_dir / "visible_metrics.json", visible_metrics)
            from verify_e2e1_live_copilot_chat import _verify_visible_metrics

            _verify_visible_metrics(visible_metrics, proposal)
            _write_json(evidence_dir / "product_preview_http_io.json", preview_http)
            _write_json(evidence_dir / "preview_response.json", preview_payload)
            plan_diff = runtime_execution["plan_diff"]
            _write_json(evidence_dir / "plan_diff.json", plan_diff)
            _write_json(
                evidence_dir / "evaluation_summary.json",
                {
                    "plan_id": proposal["child_plan_id"],
                    "plan_content_hash": proposal["child_plan_content_hash"],
                    "diff_id": proposal["diff_id"],
                    "certificate_id": proposal["certificate_id"],
                    "evaluation_status": proposal["evaluation_status"],
                    "state": proposal["state"],
                    "route_validation": proposal["route_validation"],
                },
            )
            _write_json(evidence_dir / "map_render_ack.json", rendered)
            _write_json(
                evidence_dir / "runtime_browser_binding.json",
                {
                    "preview_started_at": preview_started_at,
                    "preview_received_at": preview_received_at,
                    "independent_geometry_hash": expected_geometry_hash,
                },
            )

            page.locator("#close-copilot").click()
            expect(page.locator("#map-context-title")).to_contain_text("Pfeiffer Big Sur State Park")
            page.locator("#open-copilot").click()
            expect(page.locator("#copilot-dock")).to_be_visible()
            checkpoints = {"after_copilot_restore": _map_ack(page)}
            page.locator("#close-copilot").click()

            page.locator('.nav-item[data-route="/app/compare"]').click()
            compare_card = page.locator(f'[data-option-card="{proposal["child_plan_id"]}"]')
            expect(compare_card).to_be_visible(timeout=30_000)
            expect(compare_card).to_contain_text("Evaluated draft repair")
            option_canvas = page.locator('[data-compare-map-pane="option"] [data-compare-map-canvas]')
            expect(option_canvas).to_have_attribute("data-map-ready", "true", timeout=30_000)
            expect(option_canvas).to_have_attribute("data-rendered-map-plan-id", proposal["child_plan_id"])
            expect(option_canvas).to_have_attribute(
                "data-rendered-plan-content-hash", proposal["child_plan_content_hash"]
            )
            expect(option_canvas).to_have_attribute("data-rendered-geometry-hash", expected_geometry_hash)
            compare_ack = option_canvas.evaluate("""element => ({
              rendered_map_plan_id: element.dataset.renderedMapPlanId,
              rendered_plan_content_hash: element.dataset.renderedPlanContentHash,
              rendered_geometry_hash: element.dataset.renderedGeometryHash,
              rendered_marker_ids: JSON.parse(element.dataset.renderedMarkerIds),
              map_ready_at: element.dataset.mapReadyAt,
            })""")
            assert sorted(compare_ack["rendered_marker_ids"]) == sorted(rendered["rendered_marker_ids"])
            _write_json(evidence_dir / "compare_map_render_ack.json", compare_ack)
            expect(page.locator('[data-compare-map-pane="option"] [data-compare-map-status]')).to_contain_text(
                "Map ready with road-validated route evidence."
            )
            page.locator('.nav-item[data-route="/app/evidence"]').click()
            expect(page.locator("#workspace-content")).to_contain_text("Evaluated draft repair", timeout=15_000)
            expect(page.locator("#workspace-content")).to_contain_text(proposal["child_plan_id"])
            expect(page.locator("#workspace-content")).to_contain_text(proposal["certificate_id"])
            page.locator('.nav-item[data-route="/app/itinerary"]').click()
            expect(stage).to_have_attribute("data-rendered-map-plan-id", proposal["child_plan_id"], timeout=30_000)
            _capture_ready(page, evidence_dir / "screenshot_preview_rendered.png")

            checkpoints["after_module_switching"] = _map_ack(page)
            expect(page.locator("#map-context-title")).to_contain_text("Pfeiffer Big Sur State Park")
            with page.expect_response(
                lambda response: (
                    response.url == f"{BASE_URL}/api/sessions/{session_id}" and response.request.method == "GET"
                ),
                timeout=30_000,
            ) as reload_info:
                page.reload(wait_until="domcontentloaded")
            restored_http = _http_record(reload_info.value)
            _write_json(evidence_dir / "product_reload_http_io.json", restored_http)
            restored_payload = json.loads(restored_http["response_body_utf8"])
            assert restored_payload["session"]["selected_stop_id"] == CANDIDATE_ID
            assert restored_payload["session"]["proposal"]["runtime_execution"] == runtime_execution
            expect(stage).to_have_attribute("data-preview-state", "PREVIEW_RENDERED", timeout=30_000)
            expect(page.locator("#map-context-title")).to_contain_text("Pfeiffer Big Sur State Park")
            checkpoints["after_reload"] = _map_ack(page)
            assert (
                _independent_geometry_hash(page, restored_payload["session"]["proposal"]["geography_plan"])
                == expected_geometry_hash
            )
            _capture_ready(page, evidence_dir / "screenshot_after_reload.png")
            if os.environ.get("PRODUCT_E2E_VIEWPORT_CHECKS") == "1":
                for width, height in ((1280, 800), (1024, 768), (390, 844), (1440, 900)):
                    page.set_viewport_size({"width": width, "height": height})
                    expect(stage).to_have_attribute(
                        "data-rendered-map-plan-id", proposal["child_plan_id"], timeout=30_000
                    )
                    _capture_ready(page, evidence_dir / f"screenshot_preview_{width}x{height}.png")
                    checkpoints[f"{width}x{height}"] = _map_ack(page)
            consistency_failures = {
                name: acknowledgement for name, acknowledgement in checkpoints.items() if acknowledgement != rendered
            }
            _write_json(
                evidence_dir / "navigation_and_viewport_assertions.json",
                {
                    "expected": rendered,
                    "independent_geometry_hash": expected_geometry_hash,
                    "checkpoints": checkpoints,
                    "failures": consistency_failures,
                    "assertions_passed": not consistency_failures,
                },
            )
            accepted_checkpoints = {
                "before_chat": initial_session["accepted_plan_id"],
                "after_chat": copilot_payload["session"]["accepted_plan_id"],
                "after_stage": draft_payload["session"]["accepted_plan_id"],
                "after_preview": preview_payload["session"]["accepted_plan_id"],
                "after_reload": restored_payload["session"]["accepted_plan_id"],
            }
            acceptance_mutations = [
                row
                for row in mutation_ledger
                if re.fullmatch(
                    r"/api/sessions/[^/]+/(accept|keep-original|acceptance-decisions/[^/]+/rollback)",
                    urlsplit(row["url"]).path,
                )
            ]
            acceptance = {
                "schema_version": "e2e1-live-acceptance-observation-v1",
                "session_id": session_id,
                "initial_plan_id": initial["active_plan_id"],
                "initial_content_hash": initial["active_plan_content_hash"],
                "preview_parent_id": proposal["parent_plan_id"],
                "preview_parent_content_hash": proposal["parent_plan_content_hash"],
                "accepted_plan_checkpoints": accepted_checkpoints,
                "acceptance_mutations": acceptance_mutations,
                "accept_or_keep_invoked": bool(acceptance_mutations),
            }
            _write_json(evidence_dir / "acceptance_observation.json", acceptance)
            assert not acceptance_mutations
            assert set(accepted_checkpoints.values()) == {initial["active_plan_id"]} == {proposal["parent_plan_id"]}
            assert proposal["parent_plan_content_hash"] == initial["active_plan_content_hash"]

            assertions = {
                "schema_version": "e2e1-live-copilot-browser-assertions-v1",
                "session_id": session_id,
                "client_message_id": copilot_request["client_message_id"],
                "copilot_stream_request_id": copilot_events[-1]["request_id"],
                "context_revision": turn["context_revision"],
                "conversation_turn_id": turn["turn_id"],
                "provider": turn["provider"],
                "model": turn["model"],
                "intent": intent,
                "draft_operation": operation,
                **({"intents": interpretation["intents"], "draft_operations": operations} if combined else {}),
                "parent_plan_id": proposal["parent_plan_id"],
                "child_plan_id": proposal["child_plan_id"],
                "child_plan_content_hash": proposal["child_plan_content_hash"],
                "diff_id": proposal["diff_id"],
                "certificate_id": proposal["certificate_id"],
                "rendered_map_plan_id": rendered["rendered_map_plan_id"],
                "rendered_geometry_hash": rendered["rendered_geometry_hash"],
                "rendered_marker_ids": rendered["rendered_marker_ids"],
                "console_errors": console_errors,
                "request_failures": request_failures,
                "http_errors": http_errors,
                "accept_or_keep_invoked": acceptance["accept_or_keep_invoked"],
                "assertions_passed": not consistency_failures,
            }
            assert console_errors == []
            assert request_failures == []
            assert http_errors == []
            _write_json(evidence_dir / "browser_assertions.json", assertions)
            assert not consistency_failures, consistency_failures
        browser.close()
