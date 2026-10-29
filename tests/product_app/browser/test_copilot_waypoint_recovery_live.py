from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from test_e2e1_live_copilot_chat import _sse_events

playwright = pytest.importorskip("playwright.sync_api")
expect = playwright.expect
sync_playwright = playwright.sync_playwright


BASE_URL = os.environ.get("PRODUCT_E2E_BASE_URL")
CHROME = Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe")
SESSION_STORAGE_KEY = "itinerary-repair-copilot.session.v1"
DAY_FIVE_LEG = "leg_comfort_inn_sequoia_kings_canyon_bixby_creek_bridge_viewpoint"


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _session_fetch(page, path: str, *, method: str = "GET", body: dict | None = None) -> dict:
    result = page.evaluate(
        """async ({ storageKey, path, method, body }) => {
          const credentials = JSON.parse(sessionStorage.getItem(storageKey) || "null");
          if (!credentials?.session_id || !credentials?.mutation_token) {
            throw new Error("missing test session credentials");
          }
          const response = await fetch(path.replace("{session_id}", credentials.session_id), {
            method,
            headers: {
              "Content-Type": "application/json",
              "X-Session-Token": credentials.mutation_token,
              "X-Product-Build-Id": document.querySelector('meta[name="product-build-id"]')?.content || "",
            },
            body: body === null ? undefined : JSON.stringify(body),
          });
          const text = await response.text();
          return {
            status: response.status,
            body: text ? JSON.parse(text) : null,
          };
        }""",
        {
            "storageKey": SESSION_STORAGE_KEY,
            "path": path,
            "method": method,
            "body": body,
        },
    )
    assert isinstance(result, dict)
    return result


def _send_copilot(page, message: str) -> tuple[dict, dict]:
    stream_index = page.evaluate("window.__waypointRecoveryStreams.length")
    page.locator("#copilot-input").fill(message)
    with page.expect_response(
        lambda response: response.url.endswith("/copilot/messages") and response.request.method == "POST",
        timeout=30_000,
    ) as response_info:
        page.locator("#copilot-form button[type='submit']").click()
    response = response_info.value
    request = {
        "method": response.request.method,
        "url": response.request.url,
        "headers": {
            "Content-Type": response.request.headers.get("content-type", ""),
            "X-Session-Token": "OMITTED_SECRET",
        },
        "body_utf8": response.request.post_data or "",
    }
    assert response.status == 200
    page.wait_for_function(
        "i => window.__waypointRecoveryStreams[i]?.done === true",
        arg=stream_index,
        timeout=30_000,
    )
    stream = page.evaluate("i => window.__waypointRecoveryStreams[i]", stream_index)
    assert stream["error"] is None, stream
    events = _sse_events(stream["raw"])
    assert events and events[-1]["type"] == "turn.completed", events
    payload = events[-1]["payload"]
    assert payload["turn"]["user_message"] == message
    return request, payload


@pytest.mark.skipif(not BASE_URL, reason="Set PRODUCT_E2E_BASE_URL to the live loopback app")
def test_latest_map_waypoint_can_be_removed_and_short_stop_name_is_canonicalized(
    tmp_path: Path,
) -> None:
    evidence_dir = Path(os.environ.get("PRODUCT_E2E_COPILOT_RECOVERY_DIR", tmp_path))
    evidence_dir.mkdir(parents=True, exist_ok=True)
    console_errors: list[str] = []
    request_failures: list[str] = []

    with sync_playwright() as runtime:
        launch: dict[str, object] = {"headless": True}
        if CHROME.is_file():
            launch["executable_path"] = str(CHROME)
        browser = runtime.chromium.launch(**launch)
        context = browser.new_context(viewport={"width": 1440, "height": 900})
        page = context.new_page()
        # Observe a clone of the real stream; never replace a product response.
        page.add_init_script("""(() => {
          const originalFetch = window.fetch.bind(window);
          window.__waypointRecoveryStreams = [];
          window.fetch = async (...args) => {
            const response = await originalFetch(...args);
            const input = args[0], url = typeof input === 'string' ? input : input.url;
            if (url.endsWith('/copilot/messages')) {
              const row = {raw:'',done:false,error:null};
              window.__waypointRecoveryStreams.push(row);
              (async () => {
                const reader = response.clone().body.getReader(), decoder = new TextDecoder();
                try {
                  for (;;) {const part = await reader.read(); if(part.done) break;
                    row.raw += decoder.decode(part.value,{stream:true});}
                  row.raw += decoder.decode();
                } catch(error) {row.error = error.name;}
                finally {row.done = true;}
              })();
            }
            return response;
          };
        })();""")
        page.on(
            "console",
            lambda message: console_errors.append(message.text) if message.type == "error" else None,
        )
        page.on("requestfailed", lambda request: request_failures.append(request.url))
        page.goto(f"{BASE_URL}/app/itinerary", wait_until="domcontentloaded")
        expect(page.locator("#app-shell")).to_be_visible(timeout=15_000)
        expect(page.locator("#geographic-map")).to_have_attribute("data-map-ready", "true", timeout=30_000)

        session_response = _session_fetch(page, "/api/sessions/{session_id}")
        assert session_response["status"] == 200
        session = session_response["body"]["session"]
        accepted_plan_id = session["accepted_plan_id"]
        before_map_plan_id = page.locator(".map-stage").get_attribute("data-rendered-map-plan-id")
        assert before_map_plan_id == accepted_plan_id

        snap_request = {
            "expected_revision": session["revision"],
            "longitude": -121.941786,
            "latitude": 36.517094,
            "operation_intent": "add_custom_waypoint",
            "selected_day": 5,
            "selected_route_segment_id": DAY_FIVE_LEG,
            "travel_mode": "driving",
        }
        snap_response = _session_fetch(
            page,
            "/api/sessions/{session_id}/map/snap-preview",
            method="POST",
            body=snap_request,
        )
        assert snap_response["status"] == 200, snap_response
        confirm_request = {
            "expected_revision": session["revision"],
            "name": "Temporary Point Lobos draft waypoint",
            "role": "scenic_stop",
            "duration": {
                "mode": "exact",
                "preferred_minutes": 45,
                "minimum_minutes": 45,
                "maximum_minutes": 45,
            },
            "warning_acknowledged": False,
        }
        snap_id = snap_response["body"]["snap_preview_id"]
        confirm_response = _session_fetch(
            page,
            f"/api/sessions/{{session_id}}/map/snap-previews/{snap_id}/confirm",
            method="POST",
            body=confirm_request,
        )
        assert confirm_response["status"] == 200, confirm_response
        operation = confirm_response["body"]["operation"]
        waypoint_id = operation["target"]
        assert operation["type"] == "add_custom_waypoint"
        assert len(confirm_response["body"]["session"]["draft"]) == 1

        page.reload(wait_until="domcontentloaded")
        expect(page.locator("#app-shell")).to_be_visible(timeout=15_000)
        expect(page.locator("#geographic-map")).to_have_attribute("data-map-ready", "true", timeout=30_000)
        page.locator(".textual-map-alternative summary").click()
        waypoint = page.locator(f'[data-draft-waypoint="{waypoint_id}"]')
        expect(waypoint).to_be_visible(timeout=30_000)
        expect(page.locator(".draft-bar strong")).to_have_text("1 draft change")
        expect(page.locator(".map-stage")).to_have_attribute(
            "data-rendered-map-plan-id", accepted_plan_id, timeout=30_000
        )

        page.locator("#open-copilot").click()
        removal_message = "remove the added point"
        removal_request, removal_response = _send_copilot(page, removal_message)
        removal_interpretation = removal_response["turn"]["interpretation"]
        assert removal_interpretation["state"] == "unsupported"
        assert [intent["type"] for intent in removal_interpretation["intents"]] == ["unsupported_request"]
        assert len(removal_response["session"]["draft"]) == 1
        expect(page.locator(".message-assistant").last).to_contain_text("staged draft")
        expect(page.locator(".message-assistant").last).to_contain_text("Remove this draft waypoint")
        page.locator("#close-copilot").click()

        if page.locator(".textual-map-alternative").get_attribute("open") is None:
            page.locator(".textual-map-alternative summary").click()
        expect(waypoint).to_be_visible()
        waypoint.click()
        remove_button = page.locator("#remove-custom-waypoint")
        expect(remove_button).to_be_enabled()
        page.once("dialog", lambda dialog: dialog.accept())
        with page.expect_response(
            lambda response: response.url.endswith("/draft/undo") and response.request.method == "POST"
        ) as undo_info:
            remove_button.click()
        undo_response = undo_info.value
        assert undo_response.status == 200, undo_response.text()
        undo_payload = undo_response.json()
        assert undo_payload["undone"]["operation_id"] == operation["operation_id"]
        assert undo_payload["session"]["draft"] == []
        expect(page.locator(".draft-bar strong")).to_have_text("0 draft changes")
        expect(page.locator(f'[data-draft-waypoint="{waypoint_id}"]')).to_have_count(0)
        expect(page.locator(".map-stage")).to_have_attribute(
            "data-rendered-map-plan-id", accepted_plan_id, timeout=30_000
        )

        page.locator("#open-copilot").click()
        bixby_message = "I want to go to Bixby Creek Bridge"
        bixby_request, bixby_response = _send_copilot(page, bixby_message)
        bixby_interpretation = bixby_response["turn"]["interpretation"]
        assert bixby_interpretation["state"] == "proposal_ready"
        assert len(bixby_interpretation["intents"]) == 1
        bixby_intent = bixby_interpretation["intents"][0]
        assert bixby_intent["type"] == "keep_stop"
        assert bixby_intent["target_stop_id"] == "bixby_creek_bridge_viewpoint"
        assert bixby_interpretation["highlights"]["stop_ids"] == ["bixby_creek_bridge_viewpoint"]
        expect(page.locator(".message-assistant").last).to_contain_text("already in the accepted itinerary")
        expect(page.locator(".copilot-proposal-actions").last).to_contain_text("Add to draft")
        assert bixby_response["session"]["draft"] == []
        expect(page.locator("#geographic-map")).to_have_attribute("data-map-ready", "true", timeout=30_000)
        expect(page.locator(".map-stage")).to_have_attribute(
            "data-rendered-map-plan-id", accepted_plan_id, timeout=30_000
        )

        _write_json(evidence_dir / "snap_request.json", snap_request)
        _write_json(evidence_dir / "snap_response.json", snap_response["body"])
        _write_json(evidence_dir / "confirm_request.json", confirm_request)
        _write_json(evidence_dir / "confirm_response.json", confirm_response["body"])
        _write_json(evidence_dir / "undo_response.json", undo_payload)
        _write_json(evidence_dir / "copilot_remove_request.json", removal_request)
        _write_json(evidence_dir / "copilot_remove_response.json", removal_response)
        _write_json(evidence_dir / "copilot_bixby_request.json", bixby_request)
        _write_json(evidence_dir / "copilot_bixby_response.json", bixby_response)
        _write_json(
            evidence_dir / "copilot_exact_streams.json",
            page.evaluate("window.__waypointRecoveryStreams"),
        )
        _write_json(
            evidence_dir / "llm_io.json",
            {
                "provider_call_count": 0,
                "llm_input": None,
                "llm_output": None,
                "reason": (
                    "Both prompts are handled by deterministic server guards before the OpenAI provider boundary."
                ),
                "verification_tests": [
                    "test_draft_point_removal_guidance_runs_before_provider_call",
                    "test_shortened_accepted_stop_name_is_canonicalized_before_provider_call",
                ],
            },
        )
        _write_json(
            evidence_dir / "browser_assertions.json",
            {
                "accepted_plan_id": accepted_plan_id,
                "rendered_map_plan_id": page.locator(".map-stage").get_attribute("data-rendered-map-plan-id"),
                "removed_waypoint_id": waypoint_id,
                "draft_count_after_undo": 0,
                "bixby_intent": bixby_intent,
                "console_errors": console_errors,
                "request_failures": request_failures,
                "assertions_passed": not console_errors and not request_failures,
            },
        )
        page.screenshot(path=str(evidence_dir / "after_recovery.png"), full_page=True)
        assert console_errors == []
        assert request_failures == []
        context.close()
        browser.close()
