"""Real touch chat -> draft -> evaluated map -> explicit acceptance, with negatives."""

from __future__ import annotations

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
    _sse_events,
    _write_json,
    expect,
    sync_playwright,
)

BASE_URL = os.environ.get("PRODUCT_E2E_BASE_URL")


@pytest.mark.skipif(
    not BASE_URL or os.environ.get("PRODUCT_E2E_MOBILE_CHAT_DISPOSABLE") != "1",
    reason="Requires fresh isolated deterministic mobile-chat app",
)
def test_mobile_chat_preserves_semantics_and_accepts_exact_rendered_child(tmp_path: Path):
    evidence = Path(os.environ.get("PRODUCT_E2E_MOBILE_CHAT_EVIDENCE_DIR", tmp_path))
    evidence.mkdir(parents=True, exist_ok=True)
    errors, failures, http, turns = [], [], [], []
    keyboard = {
        "schema_version": "mobile-keyboard-browser-v1",
        "typing": [],
        "focus": {},
        "os_virtual_keyboard_tested": False,
    }
    message_requests = []
    with sync_playwright() as runtime:
        browser = runtime.chromium.launch(
            headless=True, **({"executable_path": str(CHROME)} if CHROME.is_file() else {})
        )
        context = browser.new_context(viewport={"width": 390, "height": 844}, is_mobile=True, has_touch=True)
        page = context.new_page()
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.on("console", lambda message: errors.append(message.text) if message.type == "error" else None)
        page.on("requestfailed", lambda request: failures.append({"url": request.url, "failure": request.failure}))
        page.on(
            "request",
            lambda request: (
                message_requests.append(request.post_data_json)
                if request.method == "POST" and request.url.endswith("/copilot/messages")
                else None
            ),
        )

        def record(response, body=None):
            if urlsplit(response.url).path.startswith("/api/"):
                if response.url.endswith("/copilot/messages") and body is None:
                    return  # Chrome cannot read a consumed SSE body through Network.getResponseBody.
                row = _http_record(response, response_body=body)
                name = f"http_{len(http) + 1:03d}.json"
                _write_json(evidence / name, row)
                http.append(
                    {"file": name, "url": row["url"], "method": row["method"], "status": row["response_status_code"]}
                )

        page.on("response", record)
        # Passive tee of the real response; the original request and response are untouched.
        page.add_init_script("""(() => {
          window.__mobileKeyboardEvents = [];
          document.addEventListener('keydown', event => {
            if (event.target.id === 'copilot-input') window.__mobileKeyboardEvents.push({
              key:event.key, shift:event.shiftKey, trusted:event.isTrusted,
              target:event.target.id, value:event.target.value
            });
          }, true);
          const originalFetch = window.fetch.bind(window);
          window.__mobileChatRaw = null;
          window.fetch = async (...args) => {
            const response = await originalFetch(...args);
            const input = args[0];
            const url = typeof input === 'string' ? input : input.url;
            if (url.endsWith('/copilot/messages')) {
              response.clone().text().then(raw => { window.__mobileChatRaw = raw; });
            }
            return response;
          };
        })();""")

        def touch(locator):
            locator.scroll_into_view_if_needed()
            expect(locator).to_be_in_viewport()
            locator.tap()

        def send(message, *, enter=False):
            page.evaluate("window.__mobileChatRaw = null")
            composer = page.locator("#copilot-input")
            touch(composer)
            expect(composer).to_be_empty()
            event_start = page.evaluate("window.__mobileKeyboardEvents.length")
            composer.press_sequentially(message)
            expect(composer).to_have_value(message)
            typing = {
                "input": message,
                "events": page.evaluate("start => window.__mobileKeyboardEvents.slice(start)", event_start),
                "value_before_submit": composer.input_value(),
                "focused_before_submit": page.evaluate("document.activeElement.id"),
                "submit_method": "enter" if enter else "touch",
                "requests_before": len(message_requests),
            }
            with page.expect_response(
                lambda response: response.url.endswith("/copilot/messages"), timeout=45_000
            ) as received:
                if enter:
                    composer.press("Enter")
                else:
                    touch(page.locator("#copilot-submit"))
            response = received.value
            assert response.status == 200, response.text()
            page.wait_for_function("() => typeof window.__mobileChatRaw === 'string'", timeout=45_000)
            raw = page.evaluate("window.__mobileChatRaw")
            record(response, raw)
            events = _sse_events(raw)
            assert events[-1]["type"] == "turn.completed", events
            payload = events[-1]["payload"]
            assert payload["turn"]["user_message"] == message
            expect(page.locator("#copilot-lifecycle")).to_have_attribute(
                "data-state", payload["turn"]["interpretation"]["state"]
            )
            user = page.locator(".message-user").last
            assistant = page.locator(".message-assistant").last
            expect(user.locator("p").first).to_have_text(message)
            expect(assistant.locator("p").first).to_have_text(payload["turn"]["assistant_message"])
            assistant.scroll_into_view_if_needed()
            expect(assistant).to_be_in_viewport()
            typing["events_through_submit"] = page.evaluate(
                "start => window.__mobileKeyboardEvents.slice(start)", event_start
            )
            typing["requests_after"] = len(message_requests)
            keyboard["typing"].append(typing)
            turns.append({"input": response.request.post_data_json, "output": payload, "sse_events": events})
            _write_json(evidence / "exact_chat_io.json", turns)
            return payload

        with _preserve_browser_evidence(
            context, page, evidence, {"console_errors": errors, "request_failures": failures}
        ):
            try:
                page.goto(f"{BASE_URL}/app/itinerary", wait_until="domcontentloaded")
                initial = _map_ack(page)
                assert "bixby_creek_bridge_viewpoint" in initial["rendered_marker_ids"]
                _capture_ready(page, evidence / "initial_mobile.png")
                touch(page.locator("#open-copilot"))
                composer = page.locator("#copilot-input")
                expect(composer).to_be_focused()
                keyboard["focus"]["opened"] = page.evaluate("document.activeElement.id")
                composer.press("Shift+Enter")
                expect(composer).to_have_value("\n")
                keyboard["newline"] = {
                    "value": composer.input_value(),
                    "request_count": len(message_requests),
                    "events": page.evaluate("window.__mobileKeyboardEvents"),
                }
                assert not message_requests
                composer.press("Backspace")
                composer.press("Tab")
                expect(page.locator("#copilot-submit")).to_be_focused()
                keyboard["focus"]["tab"] = page.evaluate("document.activeElement.id")
                page.keyboard.press("Shift+Tab")
                expect(composer).to_be_focused()
                keyboard["focus"]["shift_tab"] = page.evaluate("document.activeElement.id")
                composer.press("Escape")
                expect(page.locator("#copilot-dock")).to_be_hidden()
                expect(page.locator("#open-copilot")).to_be_focused()
                keyboard["focus"]["escape"] = page.evaluate("document.activeElement.id")
                touch(page.locator("#open-copilot"))
                preserve = send("Keep Stearns Wharf.")
                assert [
                    (intent["type"], intent["target_stop_id"])
                    for intent in preserve["turn"]["interpretation"]["intents"]
                ] == [("keep_stop", "stearns_wharf")]
                assert preserve["session"]["draft"] == []
                mixed = send("Replace Bixby Creek Bridge with Pfeiffer Big Sur State Park and remove Stearns Wharf.")
                assert mixed["turn"]["interpretation"]["state"] == "unsupported"
                assert mixed["session"]["draft"] == []
                assert not any(row["url"].endswith("/draft/operations") for row in http)
                assert _map_ack(page) == initial
                touch(composer)
                page.set_viewport_size({"width": 390, "height": 480})
                expect(composer).to_be_focused()
                good = send("Replace Bixby Creek Bridge with Pfeiffer Big Sur State Park.", enter=True)
                assert good["turn"]["interpretation"]["state"] == "proposal_ready"
                keyboard["reduced_viewport"] = page.evaluate("""() => ({
                  width:innerWidth,height:innerHeight,
                  controls:['#copilot-input','#copilot-submit','#copilot-form'].map(selector => {
                    const element = document.querySelector(selector), box = element.getBoundingClientRect();
                    const hit = document.elementFromPoint(box.x + box.width / 2,box.y + box.height / 2);
                    return {selector,x:box.x,y:box.y,width:box.width,height:box.height,
                            center_unoccluded:!!hit && (hit === element || element.contains(hit))};
                  })
                })""")
                for box in keyboard["reduced_viewport"]["controls"]:
                    assert box["width"] > 0 and box["height"] > 0 and box["x"] >= 0 and box["y"] >= 0
                    assert box["x"] + box["width"] <= 391 and box["y"] + box["height"] <= 481
                    assert box["center_unoccluded"], box
                conversation = page.locator("#conversation")
                before_scroll = conversation.evaluate(
                    "e => ({top:e.scrollTop,height:e.clientHeight,content_height:e.scrollHeight,page_x:scrollX,page_y:scrollY})"
                )
                assert before_scroll["content_height"] > before_scroll["height"] > 0
                box = conversation.bounding_box()
                assert box is not None
                page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
                page.mouse.wheel(0, -200)
                page.wait_for_function(
                    "top => document.querySelector('#conversation').scrollTop < top", arg=before_scroll["top"]
                )
                after_scroll = conversation.evaluate(
                    "e => ({top:e.scrollTop,height:e.clientHeight,content_height:e.scrollHeight,page_x:scrollX,page_y:scrollY})"
                )
                assert (after_scroll["page_x"], after_scroll["page_y"]) == (
                    before_scroll["page_x"],
                    before_scroll["page_y"],
                )
                keyboard["conversation_scroll"] = {
                    "input_method": "browser_wheel",
                    "before": before_scroll,
                    "after": after_scroll,
                }
                page.mouse.wheel(0, 2000)
                page.wait_for_function(
                    "() => { const e=document.querySelector('#conversation'); return e.scrollTop >= e.scrollHeight-e.clientHeight-1; }"
                )
                boundary_before = conversation.evaluate(
                    "e => ({top:e.scrollTop,height:e.clientHeight,content_height:e.scrollHeight,page_x:scrollX,page_y:scrollY})"
                )
                page.mouse.wheel(0, 200)
                page.wait_for_timeout(250)  # Observe scroll chaining after a real boundary wheel event.
                boundary_after = conversation.evaluate(
                    "e => ({top:e.scrollTop,height:e.clientHeight,content_height:e.scrollHeight,page_x:scrollX,page_y:scrollY})"
                )
                keyboard["boundary_scroll"] = {"before": boundary_before, "after": boundary_after}
                assert boundary_after == boundary_before, keyboard["boundary_scroll"]
                keyboard["message_request_count"] = len(message_requests)
                assert len(message_requests) == 3
                page.set_viewport_size({"width": 390, "height": 844})
                _capture_ready(page, evidence / "chat_proposal_mobile.png")
                actions = page.locator(".copilot-proposal-actions").last
                with page.expect_response(
                    lambda response: response.url.endswith("/draft/operations"), timeout=30_000
                ) as staged:
                    touch(actions.get_by_role("button", name="Add to draft", exact=False))
                assert staged.value.status == 200, staged.value.text()
                operation = staged.value.request.post_data_json
                assert {key: operation[key] for key in ("type", "target", "parameters")} == {
                    "type": "replace_nearby",
                    "target": "bixby_creek_bridge_viewpoint",
                    "parameters": {"candidate_id": "pfeiffer_big_sur_state_park"},
                }
                assert _map_ack(page) == initial
                with page.expect_response(
                    lambda response: response.url.endswith("/preview"), timeout=90_000
                ) as preview:
                    touch(actions.get_by_role("button", name="Preview this repair", exact=False))
                assert preview.value.status == 200, preview.value.text()
                proposal = preview.value.json()["proposal"]
                _write_json(evidence / "preview_response.json", preview.value.json())
                assert proposal["state"] == "eligible"
                expect(page.locator(".map-stage")).to_have_attribute(
                    "data-preview-state", "PREVIEW_RENDERED", timeout=40_000
                )
                rendered = _map_ack(page)
                assert rendered["rendered_map_plan_id"] == proposal["child_plan_id"]
                assert rendered["active_plan_content_hash"] == proposal["child_plan_content_hash"]
                assert rendered["rendered_geometry_hash"] == _independent_geometry_hash(
                    page, proposal["geography_plan"]
                )
                assert "pfeiffer_big_sur_state_park" in rendered["rendered_marker_ids"]
                assert "bixby_creek_bridge_viewpoint" not in rendered["rendered_marker_ids"]
                touch(page.locator("#close-copilot"))
                metrics = page.locator("#metric-strip")
                expect(metrics).to_have_attribute("data-plan-id", proposal["child_plan_id"])
                expect(metrics).to_have_attribute("data-plan-content-hash", proposal["child_plan_content_hash"])
                expect(metrics).to_have_attribute("data-evaluation-id", proposal["certificate_id"])
                expect(page.locator("#day-list")).to_contain_text("Pfeiffer Big Sur State Park")
                expect(page.locator("#day-list")).not_to_contain_text("Bixby Creek Bridge")
                _write_json(
                    evidence / "preview_surfaces.json",
                    {
                        "metrics": metrics.evaluate("e=>({...e.dataset})"),
                        "timeline": page.locator("#day-list").inner_text(),
                    },
                )
                _capture_ready(page, evidence / "preview_mobile.png")
                touch(page.locator(".textual-map-alternative > summary"))
                route_text = page.locator(".text-route-summary")
                expect(route_text).to_be_visible()
                expect(route_text).to_have_attribute("data-plan-id", proposal["child_plan_id"])
                expect(route_text).to_have_attribute("data-plan-content-hash", proposal["child_plan_content_hash"])
                _write_json(
                    evidence / "mobile_text_route.json",
                    {
                        "attributes": route_text.evaluate("e=>({...e.dataset})"),
                        "text": route_text.inner_text(),
                        "visible": route_text.is_visible(),
                    },
                )
                _capture_ready(page, evidence / "text_route_mobile.png")
                touch(page.locator(".textual-map-alternative > summary"))
                expect(page.locator("#accept-repair")).to_be_enabled()
                with page.expect_response(
                    lambda response: response.url.endswith("/accept"), timeout=60_000
                ) as accepted:
                    touch(page.locator("#accept-repair"))
                    expect(page.get_by_role("dialog", name="Confirm trip decision", exact=True)).to_be_visible()
                    expect(page.locator("#acceptance-message")).to_have_text(
                        "Accept this independently evaluated repair as the durable trip plan?"
                    )
                    touch(
                        page.get_by_role("dialog", name="Confirm trip decision", exact=True).get_by_role(
                            "button", name="Accept repair", exact=True
                        )
                    )
                assert accepted.value.status == 200, accepted.value.text()
                decision = accepted.value.json()["result"]["decision"]
                assert decision["selected_plan_id"] == proposal["child_plan_id"]
                assert decision["certificate_content_hash"] == proposal["certificate_content_hash"]
                _write_json(evidence / "acceptance_response.json", accepted.value.json())
                page.reload(wait_until="domcontentloaded")
                reloaded = _map_ack(page)
                assert reloaded["preview_state"] == "PARENT_RENDERED"
                assert {key: value for key, value in reloaded.items() if key != "preview_state"} == {
                    key: value for key, value in rendered.items() if key != "preview_state"
                }
                _capture_ready(page, evidence / "accepted_reload_mobile.png")
                touch(page.locator('.mobile-nav [data-route="/app/evidence"]'))
                panel = page.locator("[data-evidence-plan-id]")
                expect(panel).to_have_attribute("data-evidence-scope", "accepted")
                expect(panel).to_have_attribute("data-evidence-plan-id", proposal["child_plan_id"])
                expect(panel).to_have_attribute("data-plan-content-hash", proposal["child_plan_content_hash"])
                expect(panel).to_have_attribute("data-evaluation-id", proposal["certificate_id"])
                _write_json(
                    evidence / "accepted_evidence.json",
                    {"scope": panel.evaluate("e=>({...e.dataset})"), "text": panel.inner_text()},
                )
                assert page.evaluate("document.documentElement.scrollWidth <= innerWidth + 1")
                assert not errors and not failures
                assert not [row for row in http if row["status"] >= 400]
                assert len({row["file"] for row in http}) == len(http)
                _write_json(
                    evidence / "trace.json",
                    {
                        "schema_version": "mobile-chat-browser-v1",
                        "initial": initial,
                        "rendered": rendered,
                        "reloaded": reloaded,
                        "operation": operation,
                        "decision": decision,
                        "chat_turn_count": len(turns),
                        "touch_enabled": True,
                        "assertions_passed": True,
                        "os_virtual_keyboard_tested": False,
                        "completed_at": datetime.now(UTC).isoformat(),
                    },
                )
                _write_json(
                    evidence / "llm_io.json",
                    {
                        "provider_call_count": 0,
                        "llm_input": None,
                        "llm_output": None,
                        "reason": "Real mobile chat HTTP/SSE uses deterministic adapter; exact chat input/output is recorded separately. No billed LLM claim.",
                    },
                )
            finally:
                _write_json(evidence / "mobile_keyboard_assertions.json", keyboard)
                _write_json(evidence / "http_index.json", http)
                _write_json(evidence / "console_errors.json", errors)
                _write_json(evidence / "network_failures.json", failures)
        browser.close()
