"""R19: hold an actual completed HTTP response, never the optimizer or request."""

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
from waypoint_ui_fixture import (
    CURRENT_MAP,
    OBSERVE_REAL_MAPS,
    browser_network_record,
    classify_browser_network,
    map_cancellation_policy,
)

BASE_URL = os.environ.get("PRODUCT_E2E_BASE_URL")


@pytest.mark.skipif(not BASE_URL, reason="Requires the guarded disposable local application")
def test_completed_older_child_response_cannot_replace_distinct_newer_child(tmp_path: Path):
    evidence = Path(os.environ.get("PRODUCT_E2E_COMPLETED_RACE_EVIDENCE_DIR", tmp_path))
    evidence.mkdir(parents=True, exist_ok=True)
    console_errors, request_failures, records = [], [], []
    phase = "completed_response_race"
    map_configuration = None
    map_configuration_http_file = None
    hold_response = """(() => {
      const originalFetch = window.fetch.bind(window);
      const hold = window.__completedPreviewRace = {claimed:false, completed:false, released:false};
      window.fetch = async (...args) => {
        const url = typeof args[0] === 'string' ? args[0] : args[0].url;
        const first = url.endsWith('/preview') && !hold.claimed;
        if (first) hold.claimed = true;
        const response = await originalFetch(...args);
        if (first) {
          hold.requestBody = args[1].body;
          hold.status = response.status;
          hold.responseBody = await response.clone().text();
          hold.completedAt = new Date().toISOString();
          hold.completed = true;
          await new Promise(resolve => {hold.release = resolve;});
          hold.releasedAt = new Date().toISOString();
          hold.released = true;
        }
        return response;
      };
    })();"""

    with sync_playwright() as runtime:
        launch = {"headless": True}
        if CHROME.is_file():
            launch["executable_path"] = str(CHROME)
        browser = runtime.chromium.launch(**launch)
        context = browser.new_context(viewport={"width": 1440, "height": 900})
        context.add_init_script(OBSERVE_REAL_MAPS)
        context.add_init_script(hold_response)
        page = context.new_page()
        page.on("console", lambda message: console_errors.append(message.text) if message.type == "error" else None)
        page.on("pageerror", lambda error: console_errors.append(str(error)))
        page.on("requestfailed", lambda request: request_failures.append(browser_network_record(request, phase)))

        def record_response(response):
            nonlocal map_configuration, map_configuration_http_file
            if not urlsplit(response.url).path.startswith("/api/"):
                return
            record = {"observed_at": datetime.now(UTC).isoformat(), **_http_record(response)}
            name = f"http_{len(records) + 1:03d}.json"
            _write_json(evidence / name, record)
            records.append({"file": name, **record})
            if response.url == f"{BASE_URL}/api/map/config" and response.status == 200:
                payload = json.loads(record["response_body_utf8"])
                assert map_configuration is None or payload == map_configuration
                map_configuration = payload
                map_configuration_http_file = map_configuration_http_file or name

        page.on("response", record_response)
        with _preserve_browser_evidence(
            context,
            page,
            evidence,
            {
                "console_errors": console_errors,
                "request_failures": request_failures,
            },
        ):
            page.goto(f"{BASE_URL}/app/itinerary", wait_until="domcontentloaded")
            expect(page.locator("#app-shell")).to_be_visible(timeout=20_000)
            initial = _map_ack(page)
            _capture_ready(page, evidence / "screenshot_before.png")
            rendered_style = page.evaluate("() => (" + CURRENT_MAP + ").getStyle()")
            network_policy = map_cancellation_policy(map_configuration, rendered_style)

            def stage_replacement(candidate):
                page.locator('.day-card[data-day="5"]').click()
                page.locator("#map-edit-button").click()
                page.locator("#inspect-edit-stop").click()
                expect(page.locator("#map-edit-dialog")).to_be_visible()
                if not page.locator("#registered-candidate").is_visible():
                    page.locator(".legacy-edit-controls summary").click()
                page.locator("#registered-candidate").select_option(candidate)
                with page.expect_response(lambda response: response.url.endswith("/draft/operations")) as staged:
                    page.locator("#draft-replace").click()
                assert staged.value.status == 200, staged.value.text()
                page.locator("#close-map-edit").click()
                expect(page.locator("#inline-preview")).to_be_enabled(timeout=15_000)

            stage_replacement("pfeiffer_big_sur_state_park")
            page.locator("#inline-preview").click()
            page.wait_for_function("() => window.__completedPreviewRace.completed === true", timeout=60_000)
            held = page.evaluate("({ ...window.__completedPreviewRace, release:undefined })")
            assert held["status"] == 200
            a = json.loads(held["responseBody"])
            assert a["proposal"]["state"] == "eligible"
            assert a["session"]["revision"] == json.loads(held["requestBody"])["expected_revision"] + 1
            _write_json(evidence / "preview_a_response.json", a)
            _write_json(evidence / "preview_a_request.json", json.loads(held["requestBody"]))
            assert _map_ack(page)["rendered_map_plan_id"] == initial["rendered_map_plan_id"]
            expect(page.locator(".map-stage")).to_have_attribute("data-preview-state", "PREVIEW_RUNNING")

            with page.expect_response(
                lambda response: response.url.endswith("/draft/undo") and response.status == 200
            ) as undone:
                page.locator("#inline-undo").click()
            assert undone.value.json()["session"]["draft"] == []
            undo_records = [row for row in records if urlsplit(row["url"]).path.endswith("/draft/undo")]
            assert [row["response_status_code"] for row in undo_records] == [409, 200]
            assert json.loads(undo_records[0]["response_body_utf8"])["detail"] == "stale_session_revision"
            expect(page.locator(".map-stage")).to_have_attribute("data-preview-state", "PARENT_RENDERED")
            stage_replacement("point_lobos_state_natural_reserve")
            with page.expect_response(lambda response: response.url.endswith("/preview")) as second:
                page.locator("#inline-preview").click()
            assert second.value.status == 200, second.value.text()
            b = second.value.json()
            proposal = b["proposal"]
            assert proposal["state"] == "eligible"
            assert proposal["child_plan_id"] != a["proposal"]["child_plan_id"]
            expect(page.locator(".map-stage")).to_have_attribute(
                "data-preview-state", "PREVIEW_RENDERED", timeout=40_000
            )
            before_release = _map_ack(page)
            expected_geometry = _independent_geometry_hash(page, proposal["geography_plan"])
            a_geometry = _independent_geometry_hash(page, a["proposal"]["geography_plan"])
            assert before_release["rendered_geometry_hash"] == expected_geometry != a_geometry
            assert before_release["rendered_map_plan_id"] == proposal["child_plan_id"]
            assert "point_lobos_state_natural_reserve" in before_release["rendered_marker_ids"]
            assert "pfeiffer_big_sur_state_park" not in before_release["rendered_marker_ids"]
            assert "bixby_creek_bridge_viewpoint" not in before_release["rendered_marker_ids"]
            _capture_ready(page, evidence / "screenshot_b_before_release.png")
            b_rendered_at = page.evaluate("new Date().toISOString()")

            def surfaces():
                metrics = page.locator(".metric[data-plan-id]").evaluate_all(
                    "elements => elements.map(e => ({plan_id:e.dataset.planId,content_hash:e.dataset.planContentHash,text:e.innerText}))"
                )
                assert metrics and all(
                    row["plan_id"] == proposal["child_plan_id"]
                    and row["content_hash"] == proposal["child_plan_content_hash"]
                    for row in metrics
                )
                expect(page.locator('.day-card[data-day="5"]')).to_contain_text("Point Lobos")
                page.locator('[data-route="/app/evidence"]').first.click()
                view = page.locator("[data-evidence-plan-id]")
                expect(view).to_have_attribute("data-evidence-plan-id", proposal["child_plan_id"])
                expect(view).to_have_attribute("data-plan-content-hash", proposal["child_plan_content_hash"])
                expect(view).to_have_attribute("data-evaluation-id", proposal["certificate_id"])
                expect(view).to_contain_text(proposal["diff_id"])
                result = {
                    "metrics": metrics,
                    "evidence": view.evaluate("e => ({...e.dataset})"),
                    "diff_id": proposal["diff_id"],
                }
                page.locator("#evidence-back").click()
                expect(page.locator(".map-stage")).to_have_attribute(
                    "data-preview-state", "PREVIEW_RENDERED", timeout=40_000
                )
                return result

            before_surfaces = surfaces()
            page.evaluate("window.__completedPreviewRace.release()")
            expect(page.locator("#toast-region")).to_contain_text(
                "An older preview response was ignored", timeout=15_000
            )
            after_release = _map_ack(page)
            assert after_release == before_release
            after_surfaces = surfaces()
            assert before_surfaces == after_surfaces
            _capture_ready(page, evidence / "screenshot_after.png")
            held = page.evaluate("({ ...window.__completedPreviewRace, release:undefined })")
            assert held["completedAt"] <= b_rendered_at <= held["releasedAt"]
            network_classification = classify_browser_network(request_failures, network_policy)
            assert network_classification["blocking_failure_indexes"] == []
            unexpected_console = [message for message in console_errors if "409 (Conflict)" not in message]
            assert unexpected_console == [], unexpected_console
            _write_json(evidence / "preview_b_response.json", b)
            _write_json(evidence / "preview_b_request.json", second.value.request.post_data_json)
            _write_json(evidence / "map_render_ack.json", after_release)
            _write_json(
                evidence / "trace.json",
                {
                    "schema_version": "completed-preview-response-race-v1",
                    "assertions_passed": True,
                    "fault_injection": "delivery of first unchanged real 200 response held after backend completion",
                    "session_id": b["session"]["session_id"],
                    "parent_plan_id": proposal["parent_plan_id"],
                    "a_child_plan_id": a["proposal"]["child_plan_id"],
                    "a_geometry_hash": a_geometry,
                    "a_completed_at": held["completedAt"],
                    "b_rendered_at": b_rendered_at,
                    "a_released_at": held["releasedAt"],
                    "b_child_plan_id": proposal["child_plan_id"],
                    "b_child_plan_content_hash": proposal["child_plan_content_hash"],
                    "b_evaluation_id": proposal["certificate_id"],
                    "b_geometry_hash": expected_geometry,
                    "before_release": before_release,
                    "after_release": after_release,
                    "before_surfaces": before_surfaces,
                    "after_surfaces": after_surfaces,
                    "undo_statuses": [409, 200],
                    "preview_response_statuses": [200, 200],
                    "older_successful_response_ignored": True,
                    "console_errors": console_errors,
                    "unexpected_console_errors": unexpected_console,
                    "request_failures": request_failures,
                    "blocking_request_failure_indexes": network_classification["blocking_failure_indexes"],
                    "http_records": [row["file"] for row in records],
                },
            )
            _write_json(evidence / "network_failures.json", request_failures)
            _write_json(evidence / "rendered_map_style.json", rendered_style)
            _write_json(
                evidence / "network_classification.json",
                {
                    "map_configuration_http_file": map_configuration_http_file,
                    "rendered_map_style_file": "rendered_map_style.json",
                    **network_classification,
                },
            )
            _write_json(
                evidence / "llm_io.json",
                {
                    "provider_call_count": 0,
                    "llm_input": None,
                    "llm_output": None,
                    "reason": "Two real direct-edit optimizer previews; only HTTP response delivery timing is injected, with no chat or provider calls.",
                },
            )
        browser.close()
