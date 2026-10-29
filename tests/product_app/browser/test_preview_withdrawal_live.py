"""Real website replacement/Undo gate; attach only with explicit live settings."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from test_e2e1_live_copilot_chat import (
    APPROVED_PROMPT,
    BASE_URL,
    CANDIDATE_ID,
    CHROME,
    COMBINED_PROMPT,
    EVIDENCE_DIR,
    TARGET_STOP_ID,
    _capture_ready,
    _http_record,
    _independent_geometry_hash,
    _map_ack,
    _preserve_browser_evidence,
    _sse_events,
    _write_json,
    expect,
    sync_playwright,
)
from waypoint_ui_fixture import CURRENT_MAP, OBSERVE_REAL_MAPS

pytestmark = pytest.mark.skipif(
    not BASE_URL or not EVIDENCE_DIR or APPROVED_PROMPT != COMBINED_PROMPT,
    reason="Requires explicitly authorized live combined-prompt browser settings",
)


def test_replacement_and_undo_keep_copilot_map_and_itinerary_consistent():
    evidence = Path(EVIDENCE_DIR) / "preview_withdrawal"
    evidence.mkdir(parents=True, exist_ok=True)
    errors = {"console_errors": [], "request_failures": [], "http_errors": []}
    checkpoints, mutations = [], []
    with sync_playwright() as runtime:
        launch = {"headless": True}
        if CHROME.is_file():
            launch["executable_path"] = str(CHROME)
        browser = runtime.chromium.launch(**launch)
        context = browser.new_context(viewport={"width": 1440, "height": 900},
            record_video_dir=str(evidence), record_video_size={"width": 1440, "height": 900})
        page = context.new_page()
        page.add_init_script(OBSERVE_REAL_MAPS)
        def drawn_scope():
            return page.evaluate("""() => {
              const map = """ + CURRENT_MAP + """;
              const stops = map.getStyle().sources['v2-stops'].data.features;
              const drawn = map.queryRenderedFeatures({layers:['v2-stops-active','v2-stops-original','v2-stops-selected-day']});
              const highlights = map.queryRenderedFeatures({layers:['v2-route-selected-day']});
              return {source_stop_ids:stops.map(row=>row.properties.stop_id).sort(),
                source_plan_ids:[...new Set(stops.map(row=>row.properties.plan_id))],
                drawn_stop_plan_ids:[...new Set(drawn.map(row=>row.properties.plan_id))],
                highlighted_route_plan_ids:[...new Set(highlights.map(row=>row.properties.plan_id))]};
            }""")
        page.on("pageerror", lambda error: errors["console_errors"].append(str(error)))
        page.on("console", lambda message: errors["console_errors"].append(message.text) if message.type == "error" else None)
        page.on("requestfailed", lambda request: errors["request_failures"].append(f"{request.method} {request.url}: {request.failure}"))
        page.on("response", lambda response: errors["http_errors"].append(f"{response.status} {response.url}") if response.status >= 400 else None)
        page.on("request", lambda request: mutations.append({"method": request.method, "url": request.url})
            if request.method == "POST" and request.url.startswith(f"{BASE_URL}/api/") else None)
        with _preserve_browser_evidence(context, page, evidence, errors):
            page.goto(f"{BASE_URL}/app/itinerary", wait_until="domcontentloaded")
            expect(page.locator("#app-shell")).to_be_visible(timeout=30_000)
            initial = _map_ack(page)
            stage = page.locator(".map-stage")
            day = page.locator('.day-card[data-day="5"]')
            expect(day).to_contain_text("Bixby Creek Bridge")
            page.locator("#open-copilot").click()
            _capture_ready(page, evidence / "before.png")
            # Observe the real SSE stream; Chrome does not retain its CDP body.
            page.evaluate("""() => {
              const original = window.fetch.bind(window);
              window.__withdrawalChat = {raw:'', complete:false, error:null};
              window.fetch = async (...args) => {
                const response = await original(...args);
                const url = typeof args[0] === 'string' ? args[0] : args[0].url;
                if (url.endsWith('/copilot/messages')) {
                  const reader = response.clone().body.getReader(), decoder = new TextDecoder();
                  (async () => {
                    while (true) {
                      const {value, done} = await reader.read();
                      if (done) break;
                      window.__withdrawalChat.raw += decoder.decode(value, {stream:true});
                    }
                    window.__withdrawalChat.raw += decoder.decode();
                    window.__withdrawalChat.complete = true;
                  })().catch(error => {window.__withdrawalChat.error = String(error);});
                }
                return response;
              };
            }""")
            page.locator("#copilot-input").fill(APPROVED_PROMPT)
            with page.expect_response(lambda response: response.request.method == "POST"
                    and response.url.endswith("/copilot/messages"), timeout=90_000) as sent:
                page.locator("#copilot-form button[type='submit']").click()
            page.wait_for_function("() => window.__withdrawalChat.complete || window.__withdrawalChat.error", timeout=90_000)
            captured = page.evaluate("window.__withdrawalChat")
            assert captured["complete"] is True and captured["error"] is None
            chat_io = _http_record(sent.value, response_body=captured["raw"])
            _write_json(evidence / "chat_http_io.json", chat_io)
            assert sent.value.status == 200
            turn = _sse_events(chat_io["response_body_utf8"])[-1]["payload"]["turn"]
            _write_json(evidence / "copilot_turn.json", turn)
            assert turn["user_message"] == APPROVED_PROMPT and turn["provider"] == "openai"
            assert [item["type"] for item in turn["interpretation"]["intents"]] == ["replace_nearby", "set_stop_duration"]
            assert _map_ack(page) == initial, "Chat must not silently execute a repair"
            with page.expect_response(lambda response: response.request.method == "POST"
                    and response.url.endswith("/draft/operations")) as added:
                page.locator(".copilot-proposal-actions").last.get_by_role("button", name="Add all to draft", exact=True).click()
            _write_json(evidence / "stage_http_io.json", _http_record(added.value))
            assert added.value.status == 200
            expect(page.locator("#copilot-lifecycle")).to_contain_text("Draft staged")
            assert _map_ack(page) == initial, "Staging must preserve the accepted parent"
            with page.expect_response(lambda response: response.request.method == "POST"
                    and response.url.endswith("/preview"), timeout=90_000) as previewed:
                page.locator(".copilot-proposal-actions").last.get_by_role("button", name="Preview combined repair", exact=True).click()
            preview_io = _http_record(previewed.value)
            _write_json(evidence / "preview_http_io.json", preview_io)
            assert previewed.value.status == 200
            proposal = json.loads(preview_io["response_body_utf8"])["session"]["proposal"]
            assert proposal["state"] == "eligible"
            execution = proposal["runtime_execution"]
            assert execution["evaluation_executed"] is True
            assert execution["child_plan_id"] == proposal["child_plan_id"]
            assert execution["evaluation_id"] == proposal["certificate_id"]
            assert next(stop for stop in execution["child_plan"]["selected_stops"]
                if stop["stop_id"] == "griffith_observatory")["visit_duration_minutes"] == 90
            expect(stage).to_have_attribute("data-preview-state", "PREVIEW_RENDERED", timeout=30_000)
            rendered = _map_ack(page)
            assert rendered["active_plan_id"] == rendered["rendered_map_plan_id"] == proposal["child_plan_id"]
            assert rendered["active_plan_content_hash"] == proposal["child_plan_content_hash"]
            assert rendered["rendered_geometry_hash"] == _independent_geometry_hash(page, proposal["geography_plan"])
            assert rendered["rendered_geometry_hash"] != initial["rendered_geometry_hash"]
            assert CANDIDATE_ID in rendered["rendered_marker_ids"] and TARGET_STOP_ID not in rendered["rendered_marker_ids"]
            expect(day).to_contain_text("Pfeiffer Big Sur State Park")
            expect(page.locator("#metric-strip")).to_have_attribute("data-plan-id", proposal["child_plan_id"])
            expect(page.locator("#metric-strip")).to_have_attribute("data-evaluation-id", proposal["certificate_id"])
            expect(page.locator("#copilot-lifecycle")).to_have_attribute("data-state", "preview_rendered")
            _capture_ready(page, evidence / "preview.png")
            drawn = drawn_scope()
            _write_json(evidence / "renderer_preview.json", drawn)
            assert drawn["source_stop_ids"] == sorted(rendered["rendered_marker_ids"])
            assert drawn["source_plan_ids"] == drawn["drawn_stop_plan_ids"] == drawn["highlighted_route_plan_ids"] == [proposal["child_plan_id"]]
            checkpoints.append({"step": "preview", "map": rendered, "copilot": page.locator("#copilot-lifecycle").inner_text()})
            for remaining in (1, 0):
                with page.expect_response(lambda response: response.request.method == "POST"
                        and response.url.endswith("/draft/undo")) as undone:
                    page.locator("#inline-undo").click()
                undo_io = _http_record(undone.value)
                _write_json(evidence / f"undo_{remaining}_http_io.json", undo_io)
                assert undone.value.status == 200
                session = json.loads(undo_io["response_body_utf8"])["session"]
                assert len(session["draft"]) == remaining and session["proposal"] is None
                assert session["accepted_plan_id"] == initial["active_plan_id"]
                expect(stage).to_have_attribute("data-preview-state", "PARENT_RENDERED", timeout=30_000)
                assert _map_ack(page) == initial
                expect(day).to_contain_text("Bixby Creek Bridge")
                expect(page.locator("#copilot-lifecycle")).to_contain_text(
                    "Draft changed. Run Preview repair" if remaining else "No evaluated preview is active")
                expect(page.locator(".copilot-preview-result")).to_have_count(0)
                expect(page.locator("#metric-strip")).to_have_attribute("data-plan-id", initial["active_plan_id"])
                checkpoints.append({"step": f"undo_{remaining}", "map": _map_ack(page),
                    "copilot": page.locator("#copilot-lifecycle").inner_text(), "remaining_drafts": remaining})
                _capture_ready(page, evidence / f"undo_{remaining}.png")
                drawn = drawn_scope()
                _write_json(evidence / f"renderer_undo_{remaining}.json", drawn)
                assert drawn["source_stop_ids"] == sorted(initial["rendered_marker_ids"])
                assert drawn["source_plan_ids"] == drawn["drawn_stop_plan_ids"] == drawn["highlighted_route_plan_ids"] == [initial["active_plan_id"]]
            with page.expect_response(lambda response: response.request.method == "GET"
                    and response.url.endswith(f"/api/sessions/{session['session_id']}")) as reloaded:
                page.reload(wait_until="domcontentloaded")
            _write_json(evidence / "reload_http_io.json", _http_record(reloaded.value))
            assert reloaded.value.status == 200 and _map_ack(page) == initial
            page.locator("#open-copilot").click()
            expect(page.locator(".message-user")).to_have_count(1)
            expect(page.locator("#copilot-lifecycle")).not_to_contain_text("Preview ready")
            _capture_ready(page, evidence / "after_reload.png")
            assert not [item for item in mutations if re.search(r"/(accept|keep-original)(?:/|$)", item["url"])]
            assert not any(errors.values()), errors
            _write_json(evidence / "trace.json", {"schema_version": "copilot-map-withdrawal-browser-v1",
                "assertions_passed": True, "request_id": execution["request_id"],
                "session_id": session["session_id"], "parent_plan_id": initial["active_plan_id"],
                "child_plan_id": proposal["child_plan_id"], "child_plan_content_hash": proposal["child_plan_content_hash"],
                "evaluation_id": proposal["certificate_id"], "geometry_hash": rendered["rendered_geometry_hash"],
                "rendered_map_plan_id": rendered["rendered_map_plan_id"], "rendered_marker_ids": rendered["rendered_marker_ids"],
                "checkpoints": checkpoints, "mutation_requests": mutations, "errors": errors,
                "provider": turn["provider"], "model": turn["model"], "chat_submission_count": 1,
                "raw_provider_transport_io_captured": False, "accepted": False})
        browser.close()
