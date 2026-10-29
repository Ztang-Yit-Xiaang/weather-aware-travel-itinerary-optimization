from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest

playwright = pytest.importorskip("playwright.sync_api")
expect = playwright.expect
sync_playwright = playwright.sync_playwright
from test_e2e1_live_copilot_chat import _capture_ready, _http_record, _write_json
from waypoint_ui_fixture import CURRENT_MAP, OBSERVE_REAL_MAPS, wait_map

BASE_URL = os.environ.get("PRODUCT_E2E_BASE_URL")
CHROME = Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe")


def _map_ack(page) -> dict[str, object]:
    stage = page.locator(".map-stage")
    expect(stage).to_have_attribute("data-rendered-map-plan-id", re.compile(r"^plan_"), timeout=30_000)
    expect(stage).to_have_attribute("data-rendered-marker-ids", re.compile(r"^\["), timeout=30_000)
    expect(stage).to_have_attribute("data-rendered-geometry-hash", re.compile(r"^[0-9a-f]{64}$"), timeout=30_000)
    return stage.evaluate(
        """element => ({
          preview_state: element.dataset.previewState,
          active_plan_id: element.dataset.activePlanId,
          active_plan_content_hash: element.dataset.activePlanContentHash,
          rendered_map_plan_id: element.dataset.renderedMapPlanId,
          rendered_geometry_hash: element.dataset.renderedGeometryHash,
          rendered_marker_ids: JSON.parse(element.dataset.renderedMarkerIds),
        })"""
    )


def _write_proposal_artifacts(evidence_dir: Path, proposal: dict[str, object]) -> None:
    repair = proposal.get("repair") if isinstance(proposal.get("repair"), dict) else {}
    plan_diff = proposal.get("plan_diff")
    if not isinstance(plan_diff, dict):
        plan_diff = {
            "schema_version": "preview-plan-diff-summary-v1",
            "diff_id": proposal.get("diff_id"),
            "diff_content_hash": proposal.get("diff_content_hash"),
            "diff_identity": proposal.get("diff_identity"),
            "parent_plan_id": proposal.get("parent_plan_id"),
            "child_plan_id": proposal.get("child_plan_id"),
            "changed": repair.get("changed"),
            "unchanged": repair.get("unchanged"),
        }
    evaluation = {
        "schema_version": "preview-evaluation-summary-v1",
        "plan_id": proposal.get("child_plan_id"),
        "plan_content_hash": proposal.get("child_plan_content_hash"),
        "state": proposal.get("state"),
        "evaluation_status": proposal.get("evaluation_status"),
        "decision_eligible": proposal.get("decision_eligible"),
        "ranking_eligible": proposal.get("ranking_eligible"),
        "acceptance_eligible": proposal.get("acceptance_eligible"),
        "certificate_id": proposal.get("certificate_id"),
        "certificate_content_hash": proposal.get("certificate_content_hash"),
        "certificate_identity": proposal.get("certificate_identity"),
        "certificate": repair.get("certificate"),
        "route_validation": proposal.get("route_validation"),
    }
    (evidence_dir / "plan_diff.json").write_text(
        json.dumps(plan_diff, indent=2, sort_keys=True), encoding="utf-8"
    )
    (evidence_dir / "evaluation_summary.json").write_text(
        json.dumps(evaluation, indent=2, sort_keys=True), encoding="utf-8"
    )


def _write_no_llm_io(evidence_dir: Path, reason: str) -> None:
    (evidence_dir / "llm_io.json").write_text(
        json.dumps(
            {
                "provider_call_count": 0,
                "llm_input": None,
                "llm_output": None,
                "reason": reason,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _sse_events(raw: str) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    for frame in raw.replace("\r\n", "\n").split("\n\n"):
        if not frame.strip():
            continue
        event_name: str | None = None
        data_lines: list[str] = []
        for line in frame.splitlines():
            if line.startswith("event:"):
                event_name = line.removeprefix("event:").strip()
            elif line.startswith("data:"):
                data_lines.append(line.removeprefix("data:").lstrip())
        payload = json.loads("\n".join(data_lines))
        if event_name is not None:
            assert payload["type"] == event_name
        events.append(payload)
    return events


@pytest.mark.skipif(not BASE_URL, reason="Set PRODUCT_E2E_BASE_URL to the live loopback app")
def test_registered_replacement_renders_the_exact_evaluated_child(tmp_path: Path) -> None:
    evidence_dir = Path(os.environ.get("PRODUCT_E2E_EVIDENCE_DIR", tmp_path))
    evidence_dir.mkdir(parents=True, exist_ok=True)
    console_errors: list[str] = []
    request_failures: list[str] = []
    post_requests: list[str] = []

    with sync_playwright() as runtime:
        launch: dict[str, object] = {"headless": True}
        if CHROME.is_file():
            launch["executable_path"] = str(CHROME)
        browser = runtime.chromium.launch(**launch)
        context = browser.new_context(viewport={"width": 1440, "height": 900})
        page = context.new_page()
        page.on(
            "console",
            lambda message: console_errors.append(message.text)
            if message.type == "error"
            else None,
        )
        page.on("pageerror", lambda error: console_errors.append(str(error)))
        page.on(
            "requestfailed",
            lambda request: request_failures.append(
                f"{request.method} {request.url}: {request.failure}"
            ),
        )
        page.on(
            "request",
            lambda request: post_requests.append(request.url)
            if request.method == "POST"
            else None,
        )

        page.goto(f"{BASE_URL}/app/itinerary", wait_until="domcontentloaded")
        expect(page.locator("#app-shell")).to_be_visible(timeout=15_000)
        stage = page.locator(".map-stage")
        expect(stage).to_have_attribute(
            "data-preview-state", "PARENT_RENDERED", timeout=30_000
        )
        initial = _map_ack(page)
        day_five = page.locator('.day-card[data-day="5"]')
        expect(day_five).to_contain_text("Bixby Creek Bridge")
        assert "Pfeiffer" not in day_five.inner_text()
        _capture_ready(page, evidence_dir / "screenshot_before.png")

        with page.expect_response(
            lambda response: response.url.endswith("/selection")
            and response.request.method == "POST",
            timeout=15_000,
        ) as selection_info:
            day_five.click()
        assert selection_info.value.status == 200
        page.locator("#open-copilot").click()
        page.evaluate(
            """() => {
              const originalFetch = window.fetch.bind(window);
              window.__e2eCopilotStreamRaw = null;
              window.fetch = async (input, init = {}) => {
                const response = await originalFetch(input, init);
                const url = typeof input === "string" ? input : input.url;
                if (
                  url.endsWith("/copilot/messages")
                  && init.headers?.Accept === "text/event-stream"
                ) {
                  response.clone().text().then((raw) => {
                    window.__e2eCopilotStreamRaw = raw;
                  });
                }
                return response;
              };
            }"""
        )
        prompt = "Replace Bixby Creek Bridge with Pfeiffer Big Sur State Park"
        page.locator("#copilot-input").fill(prompt)
        with page.expect_response(
            lambda response: response.url.endswith("/copilot/messages")
            and response.request.method == "POST",
            timeout=30_000,
        ) as copilot_info:
            page.locator("#copilot-submit").click()
        copilot_response = copilot_info.value
        assert copilot_response.status == 200, copilot_response.text()
        page.wait_for_function(
            "() => typeof window.__e2eCopilotStreamRaw === 'string'",
            timeout=30_000,
        )
        copilot_stream_raw = page.evaluate("window.__e2eCopilotStreamRaw")
        copilot_events = _sse_events(copilot_stream_raw)
        assert [event["type"] for event in copilot_events] == [
            "message.accepted",
            "assistant.status",
            "turn.completed",
        ]
        assert [event["sequence"] for event in copilot_events] == [1, 2, 3]
        assert len({event["request_id"] for event in copilot_events}) == 1
        copilot_payload = copilot_events[-1]["payload"]
        assert copilot_payload["turn"]["user_message"] == prompt
        assert copilot_payload["turn"]["interpretation"]["state"] == "proposal_ready"
        expect(page.locator("#copilot-lifecycle")).to_contain_text(
            "Proposal ready", timeout=15_000
        )
        card = page.locator(
            '.copilot-proposal-card[data-copilot-intent-type="replace_nearby"]'
        ).last
        expect(card).to_contain_text("Bixby Creek Bridge Viewpoint")
        expect(card).to_contain_text("Pfeiffer Big Sur State Park")
        actions = page.locator(".copilot-proposal-actions").last
        expect(actions.get_by_role("button", name="Preview this repair")).to_be_enabled()
        with page.expect_response(
            lambda response: response.url.endswith("/draft/operations")
            and response.request.method == "POST",
            timeout=30_000,
        ) as draft_info:
            actions.get_by_role("button", name="Add to draft").click()
        draft_response = draft_info.value
        assert draft_response.status == 200, draft_response.text()
        draft_payload = draft_response.json()
        draft_request_body = json.loads(draft_response.request.post_data or "{}")
        assert {
            field: draft_request_body[field]
            for field in ("type", "target", "parameters")
        } == {
            "type": "replace_nearby",
            "target": "bixby_creek_bridge_viewpoint",
            "parameters": {"candidate_id": "pfeiffer_big_sur_state_park"},
        }
        assert draft_request_body["source"] == "copilot"
        assert isinstance(draft_request_body["expected_revision"], int)
        expect(page.locator("#copilot-lifecycle")).to_contain_text(
            "Draft staged", timeout=15_000
        )
        expect(page.locator("#draft-preview")).to_be_enabled(timeout=15_000)
        expect(stage).to_have_attribute(
            "data-preview-state", "PARENT_RENDERED", timeout=30_000
        )
        staged = _map_ack(page)
        assert staged["rendered_map_plan_id"] == initial["rendered_map_plan_id"]
        assert staged["rendered_marker_ids"] == initial["rendered_marker_ids"]
        assert staged["rendered_geometry_hash"] == initial["rendered_geometry_hash"]

        preview_action = page.get_by_role("button", name="Preview this repair").last
        expect(preview_action).to_be_enabled(timeout=15_000)
        with page.expect_response(
            lambda response: response.url.endswith("/preview")
            and response.request.method == "POST",
            timeout=30_000,
        ) as response_info:
            preview_action.click()
        response = response_info.value
        assert response.status == 200, response.text()
        payload = response.json()
        preview_request_body = json.loads(response.request.post_data or "{}")
        proposal = payload["session"]["proposal"]
        assert proposal["state"] == "eligible"

        expect(stage).to_have_attribute(
            "data-preview-state", "PREVIEW_RENDERED", timeout=30_000
        )
        rendered = _map_ack(page)
        expect(page.locator("#copilot-lifecycle")).to_contain_text(
            "Preview ready", timeout=15_000
        )
        preview_card = page.locator(".copilot-preview-result").last
        expect(preview_card).to_have_attribute(
            "data-preview-plan-id", proposal["child_plan_id"]
        )
        expect(preview_card).to_have_attribute(
            "data-rendered-map-plan-id", proposal["child_plan_id"]
        )
        expect(day_five).to_contain_text("Pfeiffer Big Sur State Park")
        assert "Bixby" not in day_five.inner_text()
        assert rendered["active_plan_id"] == proposal["child_plan_id"]
        assert rendered["rendered_map_plan_id"] == proposal["child_plan_id"]
        assert (
            rendered["active_plan_content_hash"]
            == proposal["child_plan_content_hash"]
        )
        assert "pfeiffer_big_sur_state_park" in rendered["rendered_marker_ids"]
        assert "bixby_creek_bridge_viewpoint" not in rendered["rendered_marker_ids"]
        assert initial["active_plan_id"] != proposal["child_plan_id"]
        assert rendered["rendered_geometry_hash"] != initial["rendered_geometry_hash"]
        assert sum(url.endswith("/copilot/messages") for url in post_requests) == 1
        assert sum(url.endswith("/draft/operations") for url in post_requests) == 1
        assert sum(url.endswith("/preview") for url in post_requests) == 1
        assert not [url for url in post_requests if url.endswith("/accept")]
        assert console_errors == []
        assert request_failures == []

        _capture_ready(page, evidence_dir / "screenshot_after.png")
        trace = {
            "request_id": proposal["compiled_request"]["source_request_id"],
            "parent_plan_id": proposal["parent_plan_id"],
            "parent_plan_content_hash": proposal["parent_plan_content_hash"],
            "child_plan_id": proposal["child_plan_id"],
            "child_plan_content_hash": proposal["child_plan_content_hash"],
            "diff_id": proposal["diff_id"],
            "certificate_id": proposal["certificate_id"],
            "comparison_eligibility": proposal["eligibility"],
            "geometry_hash": rendered["rendered_geometry_hash"],
            "rendered_map_plan_id": rendered["rendered_map_plan_id"],
            "rendered_marker_ids": rendered["rendered_marker_ids"],
            "copilot_client_message_id": copilot_events[-1]["client_message_id"],
            "copilot_stream_request_id": copilot_events[-1]["request_id"],
            "assertions_passed": True,
        }
        (evidence_dir / "trace.json").write_text(
            json.dumps(trace, indent=2, sort_keys=True), encoding="utf-8"
        )
        (evidence_dir / "map_render_ack.json").write_text(
            json.dumps(rendered, indent=2, sort_keys=True), encoding="utf-8"
        )
        (evidence_dir / "preview_response.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )
        (evidence_dir / "browser_assertions.json").write_text(
            json.dumps(
                {
                    "initial": initial,
                    "staged": staged,
                    "rendered": rendered,
                    "post_requests": post_requests,
                    "console_errors": console_errors,
                    "request_failures": request_failures,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        (evidence_dir / "preview_request.json").write_text(
            json.dumps(preview_request_body, indent=2, sort_keys=True), encoding="utf-8"
        )
        (evidence_dir / "copilot_message_request.json").write_text(
            json.dumps(
                {
                    "method": copilot_response.request.method,
                    "url": copilot_response.request.url,
                    "headers": {
                        "Accept": copilot_response.request.headers.get("accept"),
                        "Content-Type": copilot_response.request.headers.get("content-type"),
                        "X-Session-Token": "OMITTED_SECRET",
                    },
                    "body_utf8": copilot_response.request.post_data or "",
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        (evidence_dir / "copilot_stream_events.json").write_text(
            json.dumps(copilot_events, indent=2, sort_keys=True), encoding="utf-8"
        )
        (evidence_dir / "copilot_stream_raw.txt").write_text(
            copilot_stream_raw, encoding="utf-8"
        )
        (evidence_dir / "draft_request.json").write_text(
            json.dumps(
                {
                    "method": draft_response.request.method,
                    "url": draft_response.request.url,
                    "headers": {
                        "Content-Type": draft_response.request.headers.get("content-type"),
                        "X-Session-Token": "OMITTED_SECRET",
                    },
                    "body_utf8": draft_response.request.post_data or "",
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        (evidence_dir / "draft_response.json").write_text(
            json.dumps(draft_payload, indent=2, sort_keys=True), encoding="utf-8"
        )
        _write_proposal_artifacts(evidence_dir, proposal)
        _write_no_llm_io(
            evidence_dir,
            "Website chatbox used the deterministic Copilot adapter; no external provider call.",
        )
        context.close()
        browser.close()


@pytest.mark.skipif(not BASE_URL, reason="Set PRODUCT_E2E_BASE_URL to the live loopback app")
def test_copilot_one_click_preview_from_empty_draft_renders_exact_child(
    tmp_path: Path,
) -> None:
    evidence_dir = Path(os.environ.get("PRODUCT_E2E_EVIDENCE_DIR", tmp_path))
    evidence_dir.mkdir(parents=True, exist_ok=True)
    console_errors: list[str] = []
    request_failures: list[str] = []
    repair_post_sequence: list[str] = []

    def capture_repair_post(request) -> None:
        if request.method != "POST":
            return
        if request.url.endswith("/draft/operations"):
            repair_post_sequence.append("draft")
        elif request.url.endswith("/preview"):
            repair_post_sequence.append("preview")
        elif request.url.endswith("/accept"):
            repair_post_sequence.append("accept")
        elif request.url.endswith("/keep-original"):
            repair_post_sequence.append("keep-original")

    with sync_playwright() as runtime:
        launch: dict[str, object] = {"headless": True}
        if CHROME.is_file():
            launch["executable_path"] = str(CHROME)
        browser = runtime.chromium.launch(**launch)
        context = browser.new_context(viewport={"width": 1440, "height": 900})
        page = context.new_page()
        page.on(
            "console",
            lambda message: console_errors.append(message.text)
            if message.type == "error"
            else None,
        )
        page.on("pageerror", lambda error: console_errors.append(str(error)))
        page.on(
            "requestfailed",
            lambda request: request_failures.append(
                f"{request.method} {request.url}: {request.failure}"
            ),
        )
        page.on("request", capture_repair_post)

        page.goto(f"{BASE_URL}/app/itinerary", wait_until="domcontentloaded")
        expect(page.locator("#app-shell")).to_be_visible(timeout=15_000)
        stage = page.locator(".map-stage")
        expect(stage).to_have_attribute(
            "data-preview-state", "PARENT_RENDERED", timeout=30_000
        )
        initial = _map_ack(page)
        day_five = page.locator('.day-card[data-day="5"]')
        expect(day_five).to_contain_text("Bixby Creek Bridge")
        assert "Pfeiffer" not in day_five.inner_text()
        _capture_ready(page, evidence_dir / "one_click_screenshot_before.png")

        with page.expect_response(
            lambda response: response.url.endswith("/selection")
            and response.request.method == "POST",
            timeout=15_000,
        ) as selection_info:
            day_five.click()
        assert selection_info.value.status == 200
        page.locator("#open-copilot").click()
        prompt = "Replace Bixby Creek Bridge with Pfeiffer Big Sur State Park"
        page.locator("#copilot-input").fill(prompt)
        page.evaluate(
            """() => {
              const originalFetch = window.fetch.bind(window);
              window.__e2eOneClickCopilotStreamRaw = null;
              window.fetch = async (...args) => {
                const raw = args[0];
                const url = typeof raw === "string" ? raw : raw?.url || "";
                const response = await originalFetch(...args);
                if (url.endsWith("/copilot/messages")) {
                  window.__e2eOneClickCopilotStreamRaw = await response.clone().text();
                }
                return response;
              };
            }"""
        )
        with page.expect_response(
            lambda response: response.url.endswith("/copilot/messages")
            and response.request.method == "POST",
            timeout=30_000,
        ) as copilot_info:
            page.locator("#copilot-submit").click()
        copilot_response = copilot_info.value
        assert copilot_response.status == 200, copilot_response.text()
        page.wait_for_function(
            "() => typeof window.__e2eOneClickCopilotStreamRaw === 'string'",
            timeout=30_000,
        )
        copilot_stream_raw = page.evaluate("window.__e2eOneClickCopilotStreamRaw")
        copilot_events = _sse_events(copilot_stream_raw)
        copilot_turn = copilot_events[-1]["payload"]["turn"]
        assert copilot_turn["user_message"] == prompt
        assert copilot_turn["interpretation"]["state"] == "proposal_ready"
        assert copilot_turn["interpretation"]["intents"] == [
            {
                "type": "replace_nearby",
                "target_stop_id": "bixby_creek_bridge_viewpoint",
                "destination_day": None,
                "candidate_id": "pfeiffer_big_sur_state_park",
                "preference": None,
                "clarification_question": None,
            }
        ]
        expect(page.locator("#copilot-lifecycle")).to_have_attribute(
            "data-state", "proposal_ready", timeout=15_000
        )
        expect(page.locator(".draft-bar")).to_contain_text("0 draft changes")
        expect(page.locator("#inline-preview")).to_be_disabled()

        page.evaluate(
            """() => {
              const originalFetch = window.fetch.bind(window);
              window.__e2eOneClickPreview = {
                held: false,
                release: null,
                requestBody: null,
              };
              window.fetch = async (...args) => {
                const raw = args[0];
                const url = typeof raw === "string" ? raw : raw?.url || "";
                if (url.endsWith("/preview") && !window.__e2eOneClickPreview.held) {
                  window.__e2eOneClickPreview.held = true;
                  window.__e2eOneClickPreview.requestBody = JSON.parse(
                    args[1]?.body || "{}",
                  );
                  await new Promise((resolve) => {
                    window.__e2eOneClickPreview.release = resolve;
                  });
                }
                return originalFetch(...args);
              };
            }"""
        )
        row = page.locator(
            '.copilot-proposal-action-row[data-copilot-intent-type="replace_nearby"]'
        ).last
        preview_action = row.get_by_role(
            "button",
            name=re.compile(
                r"^Preview this repair: Replace Bixby Creek Bridge Viewpoint "
                r"with Pfeiffer Big Sur State Park\."
            ),
        )
        expect(preview_action).to_be_enabled()
        with page.expect_response(
            lambda response: response.url.endswith("/draft/operations")
            and response.request.method == "POST",
            timeout=30_000,
        ) as draft_info:
            preview_action.click()
        draft_response = draft_info.value
        assert draft_response.status == 200, draft_response.text()
        draft_payload = draft_response.json()
        draft_request_body = json.loads(draft_response.request.post_data or "{}")
        operation = draft_payload["operation"]
        assert draft_payload["session"]["draft"] == [operation]
        assert {
            field: draft_request_body[field]
            for field in ("type", "target", "parameters", "source")
        } == {
            "type": "replace_nearby",
            "target": "bixby_creek_bridge_viewpoint",
            "parameters": {"candidate_id": "pfeiffer_big_sur_state_park"},
            "source": "copilot",
        }

        page.wait_for_function(
            "() => window.__e2eOneClickPreview.held === true", timeout=30_000
        )
        expect(stage).to_have_attribute(
            "data-preview-state", "PREVIEW_RUNNING", timeout=15_000
        )
        parent_while_preview_runs = _map_ack(page)
        assert parent_while_preview_runs["rendered_map_plan_id"] == initial["rendered_map_plan_id"]
        assert parent_while_preview_runs["active_plan_content_hash"] == initial["active_plan_content_hash"]
        assert parent_while_preview_runs["rendered_geometry_hash"] == initial["rendered_geometry_hash"]
        assert parent_while_preview_runs["rendered_marker_ids"] == initial["rendered_marker_ids"]
        expect(day_five).to_contain_text("Bixby Creek Bridge")
        assert "Pfeiffer" not in day_five.inner_text()
        _capture_ready(page, evidence_dir / "one_click_screenshot_parent_while_running.png")

        with page.expect_response(
            lambda response: response.url.endswith("/preview")
            and response.request.method == "POST",
            timeout=30_000,
        ) as preview_info:
            page.evaluate("window.__e2eOneClickPreview.release()")
        preview_response = preview_info.value
        assert preview_response.status == 200, preview_response.text()
        preview_payload = preview_response.json()
        preview_request_body = json.loads(preview_response.request.post_data or "{}")
        proposal = preview_payload["session"]["proposal"]
        assert proposal["state"] == "eligible"
        canonical_operation = {
            key: value for key, value in operation.items() if key != "created_at"
        }
        assert proposal["compiled_request"]["operations"] == [canonical_operation]

        expect(stage).to_have_attribute(
            "data-preview-state", "PREVIEW_RENDERED", timeout=30_000
        )
        rendered = _map_ack(page)
        expect(page.locator("#copilot-lifecycle")).to_contain_text(
            "Preview ready", timeout=15_000
        )
        expect(day_five).to_contain_text("Pfeiffer Big Sur State Park")
        assert "Bixby" not in day_five.inner_text()
        assert rendered["active_plan_id"] == proposal["child_plan_id"]
        assert rendered["rendered_map_plan_id"] == proposal["child_plan_id"]
        assert rendered["active_plan_content_hash"] == proposal["child_plan_content_hash"]
        assert rendered["rendered_geometry_hash"] != initial["rendered_geometry_hash"]
        assert "pfeiffer_big_sur_state_park" in rendered["rendered_marker_ids"]
        assert "bixby_creek_bridge_viewpoint" not in rendered["rendered_marker_ids"]
        assert repair_post_sequence == ["draft", "preview"]
        assert console_errors == []
        assert request_failures == []

        trace = {
            "interaction": "copilot_one_click_preview",
            "copilot_client_message_id": copilot_turn["client_message_id"],
            "parent_plan_id": proposal["parent_plan_id"],
            "child_plan_id": proposal["child_plan_id"],
            "child_plan_content_hash": proposal["child_plan_content_hash"],
            "diff_id": proposal["diff_id"],
            "certificate_id": proposal["certificate_id"],
            "rendered_map_plan_id": rendered["rendered_map_plan_id"],
            "rendered_geometry_hash": rendered["rendered_geometry_hash"],
            "rendered_marker_ids": rendered["rendered_marker_ids"],
            "repair_post_sequence": repair_post_sequence,
            "accept_or_keep_invoked": False,
            "assertions_passed": True,
        }
        artifacts: dict[str, object] = {
            "one_click_copilot_request.json": {
                "method": copilot_response.request.method,
                "url": copilot_response.request.url,
                "headers": {
                    "Accept": copilot_response.request.headers.get("accept"),
                    "Content-Type": copilot_response.request.headers.get("content-type"),
                    "X-Session-Token": "OMITTED_SECRET",
                },
                "body_utf8": copilot_response.request.post_data or "",
            },
            "one_click_copilot_stream_events.json": copilot_events,
            "one_click_draft_request.json": {
                "method": draft_response.request.method,
                "url": draft_response.request.url,
                "headers": {
                    "Content-Type": draft_response.request.headers.get("content-type"),
                    "X-Session-Token": "OMITTED_SECRET",
                },
                "body_utf8": draft_response.request.post_data or "",
            },
            "one_click_draft_response.json": draft_payload,
            "one_click_preview_request.json": {
                "method": preview_response.request.method,
                "url": preview_response.request.url,
                "headers": {
                    "Content-Type": preview_response.request.headers.get("content-type"),
                    "X-Session-Token": "OMITTED_SECRET",
                },
                "body_utf8": preview_response.request.post_data or "",
            },
            "one_click_preview_response.json": preview_payload,
            "one_click_map_render_ack.json": rendered,
            "one_click_trace.json": trace,
            "one_click_browser_assertions.json": {
                "initial": initial,
                "parent_while_preview_runs": parent_while_preview_runs,
                "rendered": rendered,
                "draft_request_body": draft_request_body,
                "preview_request_body": preview_request_body,
                "repair_post_sequence": repair_post_sequence,
                "console_errors": console_errors,
                "request_failures": request_failures,
            },
            "one_click_plan_diff.json": proposal.get("plan_diff") or {
                "diff_id": proposal["diff_id"],
                "diff_content_hash": proposal["diff_content_hash"],
                "parent_plan_id": proposal["parent_plan_id"],
                "child_plan_id": proposal["child_plan_id"],
                "changed": proposal["repair"]["changed"],
                "unchanged": proposal["repair"]["unchanged"],
            },
            "one_click_evaluation_summary.json": {
                "plan_id": proposal["child_plan_id"],
                "plan_content_hash": proposal["child_plan_content_hash"],
                "state": proposal["state"],
                "evaluation_status": proposal["evaluation_status"],
                "certificate_id": proposal["certificate_id"],
                "route_validation": proposal["route_validation"],
            },
            "one_click_llm_io.json": {
                "provider_call_count": 0,
                "llm_input": None,
                "llm_output": None,
                "reason": "Website chatbox used the deterministic Copilot adapter; no external provider call.",
            },
        }
        for name, value in artifacts.items():
            (evidence_dir / name).write_text(
                json.dumps(value, indent=2, sort_keys=True), encoding="utf-8"
            )
        (evidence_dir / "one_click_copilot_stream_raw.txt").write_text(
            copilot_stream_raw, encoding="utf-8"
        )
        _capture_ready(page, evidence_dir / "one_click_screenshot_after.png")
        context.close()
        browser.close()


@pytest.mark.skipif(not BASE_URL, reason="Set PRODUCT_E2E_BASE_URL to the live loopback app")
def test_route_aware_candidate_discovery_stages_and_renders_exact_child(
    tmp_path: Path,
) -> None:
    evidence_dir = Path(os.environ.get("PRODUCT_E2E_POI_EVIDENCE_DIR", tmp_path))
    evidence_dir.mkdir(parents=True, exist_ok=True)
    console_errors: list[str] = []
    request_failures: list[str] = []
    http_errors: list[str] = []

    with sync_playwright() as runtime:
        launch: dict[str, object] = {"headless": True}
        if CHROME.is_file():
            launch["executable_path"] = str(CHROME)
        browser = runtime.chromium.launch(**launch)
        context = browser.new_context(viewport={"width": 1440, "height": 900})
        context.add_init_script(OBSERVE_REAL_MAPS)
        context.add_init_script("""window.__poiControlEvents=[];
          for(const type of ['click','change','submit']) document.addEventListener(type,event=>{
            const e=event.target.closest?.('button,input,form');
            if(e) window.__poiControlEvents.push({type,trusted:event.isTrusted,id:e.id,
              form:e.closest('form')?.id || null,stop:e.dataset.geoStop||null,
              leg:e.dataset.geoRouteLeg||null,candidate:e.dataset.poiCandidate||null,at:Date.now()});
          },true);""")
        page = context.new_page()
        control_requests, poi_controls = [], []
        page.on("request",lambda request:control_requests.append({"method":request.method,"url":request.url}))

        def layers():
            wait_map(page)
            return page.evaluate("""() => {
              const map=""" + CURRENT_MAP + """;
              return Object.fromEntries(['v2-poi-candidates','v2-poi-access-points'].map(id=>[id,map.getStyle().sources[id].data]));
            }""")

        def remember_control(selector, before, *, response=None, fields=None):
            events=page.evaluate("window.__poiControlEvents")
            poi_controls.append({"selector":selector,"visible":True,"disabled":False,
                "events":events,"before_layers":before,"after_layers":layers(),"map":_map_ack(page),
                "http":_http_record(response) if response is not None else None,"fields":fields,
                "drawer_visible":page.locator('#poi-candidate-heading').is_visible(),
                "assertions_passed":True})
            _write_json(evidence_dir / "poi_control_journeys.json",poi_controls)
        page.on(
            "response",
            lambda response: http_errors.append(
                f"{response.status} {response.url}"
            ) if response.status >= 400 else None,
        )
        page.on(
            "console",
            lambda message: console_errors.append(message.text)
            if message.type == "error"
            else None,
        )
        page.on("pageerror", lambda error: console_errors.append(str(error)))
        page.on(
            "requestfailed",
            lambda request: request_failures.append(
                f"{request.method} {request.url}: {request.failure}"
            ),
        )

        page.goto(f"{BASE_URL}/app/itinerary", wait_until="domcontentloaded")
        expect(page.locator("#app-shell")).to_be_visible(timeout=15_000)
        stage = page.locator(".map-stage")
        expect(stage).to_have_attribute(
            "data-preview-state", "PARENT_RENDERED", timeout=30_000
        )
        initial = _map_ack(page)

        day_five = page.locator('.day-card[data-day="5"]')
        day_five.click()
        expect(page.locator("#map-context-title")).to_contain_text(
            "Bixby Creek Bridge", timeout=15_000
        )
        page.locator("#map-edit-button").click()
        page.locator("#prepare-replacement-candidates").click()
        expect(page.locator("#toast-region")).to_contain_text(
            "select an exact route leg", timeout=15_000
        )

        route_leg_id = (
            "leg_comfort_inn_sequoia_kings_canyon_"
            "bixby_creek_bridge_viewpoint"
        )
        page.locator("details.textual-map-alternative > summary").click()
        page.locator(f'[data-geo-route-leg="{route_leg_id}"]').click()
        expect(page.locator("#find-route-candidates")).to_be_enabled(timeout=15_000)
        with page.expect_response(
            lambda response: "/map/poi-candidates?" in response.url
            and response.request.method == "GET"
        ) as candidate_response_info:
            page.locator("#find-route-candidates").click()
        candidate_response = candidate_response_info.value
        candidate_request = candidate_response.request
        (evidence_dir / "browser_poi_candidate_request.json").write_text(
            json.dumps(
                {
                    "method": candidate_request.method,
                    "url": candidate_request.url,
                    "headers": {"X-Session-Token": "OMITTED_SECRET"},
                    "body_utf8": candidate_request.post_data or "",
                    "llm_call_made": False,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        assert candidate_response.status == 200
        candidate_payload = candidate_response.json()
        (evidence_dir / "browser_poi_candidate_response.json" ).write_bytes(
            candidate_response.body()
        )
        assert {
            row["candidate_id"] for row in candidate_payload["candidates"]
        } == {
            "pfeiffer_big_sur_state_park",
            "point_lobos_state_natural_reserve",
        }

        expect(page.locator("#poi-candidate-heading")).to_be_visible(timeout=15_000)
        expect(page.locator(".poi-candidate-list")).to_contain_text(
            "Pfeiffer Big Sur State Park"
        )
        expect(page.locator(".poi-candidate-list")).to_contain_text(
            "Point Lobos State Natural Reserve"
        )
        control_request_start=len(control_requests)
        for fields in ({"role":"meal","max_detour_minutes":"120","limit":"3"},
                       {"role":"","max_detour_minutes":"60","limit":"5"}):
            before_layers=layers()
            page.locator('#poi-role-filter').select_option(fields['role'])
            page.locator('#poi-detour-filter').select_option(fields['max_detour_minutes'])
            page.locator('#poi-limit-filter').select_option(fields['limit'])
            with page.expect_response(lambda response:'/map/poi-candidates?' in response.url) as filtered:
                page.locator('#poi-candidate-filter button[type=submit]').click()
            assert filtered.value.status==200
            remember_control('#poi-candidate-filter',before_layers,response=filtered.value,fields=fields)
        assert len(filtered.value.json()['candidates'])==2
        for selector in ('#poi-layer-toggle','#poi-layer-toggle','#layers-button','#layers-button'):
            before_layers=layers()
            expect(page.locator(selector)).to_be_enabled()
            page.locator(selector).click()
            remember_control(selector,before_layers)
        before_layers=layers()
        page.locator('#close-poi-candidates').click()
        expect(page.locator('#poi-candidate-heading')).to_have_count(0)
        remember_control('#close-poi-candidates',before_layers)
        with page.expect_response(lambda response:'/map/poi-candidates?' in response.url) as reopened:
            page.locator('#find-route-candidates').click()
        assert reopened.value.status==200
        remember_control('#find-route-candidates',layers(),response=reopened.value)
        assert reopened.value.json()['context']['kind']=='insertion'
        assert reopened.value.json()['context']['replacement_target_id'] is None
        assert reopened.value.json()['candidates']==[]
        expect(page.locator('[data-poi-candidate]')).to_have_count(0)
        expect(page.locator('.candidate-results')).to_contain_text('No candidates are available')
        assert _map_ack(page)==initial
        control_log={
            "events":page.evaluate("window.__poiControlEvents"),"requests":list(control_requests),
            "request_start":control_request_start,"request_end":len(control_requests),
            "parent":initial,"provider_calls":0}
        _write_json(evidence_dir / "poi_control_input_log.json",control_log)
        details=page.locator('details.textual-map-alternative')
        if details.get_attribute('open') is None:
            details.locator(':scope > summary').click()
        with page.expect_response(lambda response:response.url.endswith('/selection')
                and response.request.post_data_json.get('selected_stop_id')=='bixby_creek_bridge_viewpoint') as reentry_stop:
            page.locator('[data-geo-stop="bixby_creek_bridge_viewpoint"]').click()
        stop_events=page.evaluate('window.__poiControlEvents')
        expect(page.locator('#prepare-replacement-candidates')).to_be_enabled()
        page.locator('#prepare-replacement-candidates').click()
        preparation_events=page.evaluate('window.__poiControlEvents')
        if details.get_attribute('open') is None:
            details.locator(':scope > summary').click()
        with page.expect_response(lambda response:response.url.endswith('/selection')
                and response.request.post_data_json.get('selected_segment_id')==route_leg_id) as reentry_leg:
            page.locator(f'[data-geo-route-leg="{route_leg_id}"]').click()
        leg_events=page.evaluate('window.__poiControlEvents')
        with page.expect_response(lambda response:'/map/poi-candidates?' in response.url
                and 'replacement_target_id=bixby_creek_bridge_viewpoint' in response.url) as replacement_again:
            page.locator('#find-route-candidates').click()
        assert replacement_again.value.status==200
        replacement_payload=replacement_again.value.json()
        assert replacement_payload['context']['kind']=='replacement'
        assert replacement_payload['context']['replacement_target_id']=='bixby_creek_bridge_viewpoint'
        assert replacement_payload['context']['route_leg_id']==route_leg_id
        expect(page.locator('[data-poi-candidate]')).to_have_count(len(replacement_payload['candidates']))
        restored_layers=layers()
        control_log['replacement_reentry']={
            'schema_version':'poi-replacement-reentry-v1','target':'bixby_creek_bridge_viewpoint',
            'route_leg_id':route_leg_id,'generic_http':_http_record(reopened.value),
            'stop_http':_http_record(reentry_stop.value),'leg_http':_http_record(reentry_leg.value),
            'candidate_http':_http_record(replacement_again.value),'stop_events':stop_events,
            'preparation_events':preparation_events,'leg_events':leg_events,
            'events':page.evaluate('window.__poiControlEvents'),'map':_map_ack(page),
            'layers':restored_layers,'visible_candidate_ids':page.locator('[data-poi-candidate]').evaluate_all(
                'elements=>elements.map(e=>e.dataset.poiCandidate)'),'assertions_passed':True}
        _write_json(evidence_dir / "poi_control_input_log.json",control_log)
        assert {row['candidate_id'] for row in replacement_payload['candidates']}=={
            'pfeiffer_big_sur_state_park','point_lobos_state_natural_reserve'}
        with page.expect_response(
            lambda response: "/map/poi-candidates?" in response.url
            and response.request.method == "GET"
        ):
            page.locator(
                '[data-poi-candidate="pfeiffer_big_sur_state_park"]'
            ).click()
        expect(page.locator("#use-candidate-replacement")).to_be_enabled(
            timeout=15_000
        )
        _capture_ready(page, evidence_dir / "candidate_discovery.png")

        page.locator("#use-candidate-replacement").click()
        expect(page.locator("#toast-region")).to_contain_text(
            "Draft operation added", timeout=15_000
        )
        expect(page.locator("#draft-preview")).to_be_enabled(timeout=15_000)
        expect(stage).to_have_attribute(
            "data-preview-state", "PARENT_RENDERED", timeout=30_000
        )
        expect(stage).to_have_attribute(
            "data-rendered-map-plan-id", str(initial["rendered_map_plan_id"]), timeout=30_000
        )
        expect(stage).to_have_attribute(
            "data-rendered-marker-ids", re.compile(r"^\["), timeout=30_000
        )
        staged = _map_ack(page)
        assert staged["rendered_map_plan_id"] == initial["rendered_map_plan_id"]
        assert staged["rendered_marker_ids"] == initial["rendered_marker_ids"]
        assert staged["rendered_geometry_hash"] == initial["rendered_geometry_hash"]

        with page.expect_response(
            lambda response: response.url.endswith("/preview")
            and response.request.method == "POST"
        ) as preview_response_info:
            page.locator("#inline-preview").click()
        preview_response = preview_response_info.value
        preview_request = preview_response.request
        preview_request_body = preview_request.post_data or ""
        (evidence_dir / "browser_preview_request.json").write_text(
            json.dumps(
                {
                    "method": preview_request.method,
                    "url": preview_request.url,
                    "headers": {
                        "Content-Type": preview_request.headers.get("content-type", ""),
                        "X-Session-Token": "OMITTED_SECRET",
                    },
                    "body_utf8": preview_request_body,
                    "llm_call_made": False,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        assert preview_response.status == 200, preview_response.text()
        preview_payload = preview_response.json()
        (evidence_dir / "browser_preview_response.json" ).write_bytes(
            preview_response.body()
        )
        proposal = preview_payload["session"]["proposal"]
        assert proposal["state"] == "eligible"
        expect(stage).to_have_attribute(
            "data-preview-state", "PREVIEW_RENDERED", timeout=30_000
        )

        rendered = _map_ack(page)
        expect(day_five).to_contain_text("Pfeiffer Big Sur State Park")
        assert "Bixby" not in day_five.inner_text()
        assert rendered["rendered_map_plan_id"] == proposal["child_plan_id"]
        assert rendered["active_plan_id"] == proposal["child_plan_id"]
        assert (
            rendered["active_plan_content_hash"]
            == proposal["child_plan_content_hash"]
        )
        assert "pfeiffer_big_sur_state_park" in rendered["rendered_marker_ids"]
        assert "bixby_creek_bridge_viewpoint" not in rendered["rendered_marker_ids"]
        assert rendered["rendered_geometry_hash"] != initial["rendered_geometry_hash"]
        assert http_errors == []
        assert console_errors == []
        assert request_failures == []

        _capture_ready(page, evidence_dir / "candidate_preview_rendered.png")
        assertions = {
            "candidate_request_url": candidate_response.url,
            "candidate_status": candidate_response.status,
            "candidate_ids": [
                row["candidate_id"] for row in candidate_payload["candidates"]
            ],
            "parent_map": initial,
            "staged_map": staged,
            "rendered_map": rendered,
            "preview_child_plan_id": proposal["child_plan_id"],
            "preview_child_plan_content_hash": proposal[
                "child_plan_content_hash"
            ],
            "console_errors": console_errors,
            "request_failures": request_failures,
            "http_errors": http_errors,
            "llm_call_made": False,
            "assertions_passed": True,
        }
        (evidence_dir / "map_render_ack.json").write_text(
            json.dumps(rendered, indent=2, sort_keys=True), encoding="utf-8"
        )
        (evidence_dir / "trace.json").write_text(
            json.dumps(
                {
                    "parent_plan_id": initial["active_plan_id"],
                    "child_plan_id": proposal["child_plan_id"],
                    "child_plan_content_hash": proposal["child_plan_content_hash"],
                    "diff_id": proposal["diff_id"],
                    "certificate_id": proposal["certificate_id"],
                    "rendered_map_plan_id": rendered["rendered_map_plan_id"],
                    "geometry_hash": rendered["rendered_geometry_hash"],
                    "rendered_marker_ids": rendered["rendered_marker_ids"],
                    "assertions_passed": True,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        (evidence_dir / "browser_assertions.json").write_text(
            json.dumps(assertions, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        _write_proposal_artifacts(evidence_dir, proposal)
        _write_no_llm_io(
            evidence_dir,
            "Route-aware POI discovery and replacement preview; no Copilot/provider call.",
        )
        context.close()
        browser.close()
@pytest.mark.skipif(not BASE_URL, reason="Set PRODUCT_E2E_BASE_URL to the live loopback app")
def test_unsupported_removal_never_stages_or_changes_the_map(tmp_path: Path) -> None:
    evidence_dir = Path(os.environ.get("PRODUCT_E2E_GUARD_EVIDENCE_DIR", tmp_path))
    evidence_dir.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as runtime:
        launch: dict[str, object] = {"headless": True}
        if CHROME.is_file():
            launch["executable_path"] = str(CHROME)
        browser = runtime.chromium.launch(**launch)
        context = browser.new_context(viewport={"width": 1440, "height": 900})
        page = context.new_page()
        page.goto(f"{BASE_URL}/app/itinerary", wait_until="domcontentloaded")
        expect(page.locator("#app-shell")).to_be_visible(timeout=15_000)
        initial = _map_ack(page)

        page.locator('.day-card[data-day="6"]').click()
        expect(page.locator("#map-context-title")).to_contain_text(
            "Stearns Wharf", timeout=15_000
        )
        page.locator("#open-copilot").click()
        page.evaluate(
            """() => {
              const originalFetch = window.fetch.bind(window);
              window.__e2eUnsupportedCopilotStreamRaw = null;
              window.fetch = async (input, init = {}) => {
                const response = await originalFetch(input, init);
                const url = typeof input === "string" ? input : input.url;
                if (
                  url.endsWith("/copilot/messages")
                  && init.headers?.Accept === "text/event-stream"
                ) {
                  response.clone().text().then((raw) => {
                    window.__e2eUnsupportedCopilotStreamRaw = raw;
                  });
                }
                return response;
              };
            }"""
        )
        message = "I do not want to go to Stearns Wharf."
        page.locator("#copilot-input").fill(message)
        with page.expect_response(
            lambda response: response.url.endswith("/copilot/messages")
            and response.request.method == "POST"
        ) as response_info:
            page.locator("#copilot-form button[type='submit']").click()
        response = response_info.value
        assert response.status == 200, response.text()
        page.wait_for_function(
            "() => typeof window.__e2eUnsupportedCopilotStreamRaw === 'string'",
            timeout=30_000,
        )
        copilot_stream_raw = page.evaluate(
            "window.__e2eUnsupportedCopilotStreamRaw"
        )
        copilot_events = _sse_events(copilot_stream_raw)
        assert [event["type"] for event in copilot_events] == [
            "message.accepted",
            "assistant.status",
            "turn.completed",
        ]
        assert [event["sequence"] for event in copilot_events] == [1, 2, 3]
        assert len({event["request_id"] for event in copilot_events}) == 1
        payload = copilot_events[-1]["payload"]
        interpretation = payload["turn"]["interpretation"]
        assert interpretation["state"] == "unsupported"
        assert [row["type"] for row in interpretation["intents"]] == [
            "unsupported_request"
        ]
        assert payload["session"]["draft"] == []
        expect(page.locator("#copilot-lifecycle")).to_have_attribute(
            "data-state", "unsupported"
        )
        expect(page.locator(".message-assistant").last).to_contain_text(
            "Removing or excluding"
        )
        expect(page.locator(".copilot-proposal-actions")).to_have_count(0)
        after = _map_ack(page)
        assert after["rendered_map_plan_id"] == initial["rendered_map_plan_id"]
        assert after["rendered_geometry_hash"] == initial["rendered_geometry_hash"]
        assert after["rendered_marker_ids"] == initial["rendered_marker_ids"]

        request = response.request
        (evidence_dir / "copilot_removal_request.json").write_text(
            json.dumps(
                {
                    "method": request.method,
                    "url": request.url,
                    "headers": {
                        "Accept": "text/event-stream",
                        "Content-Type": request.headers.get("content-type", ""),
                        "X-Session-Token": "OMITTED_SECRET",
                    },
                    "body_utf8": request.post_data or "",
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        (evidence_dir / "copilot_removal_response.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )
        (evidence_dir / "copilot_stream_events.json").write_text(
            json.dumps(copilot_events, indent=2, sort_keys=True), encoding="utf-8"
        )
        (evidence_dir / "copilot_stream_raw.txt").write_text(
            copilot_stream_raw, encoding="utf-8"
        )
        (evidence_dir / "llm_io.json").write_text(
            json.dumps(
                {
                    "provider_call_count": 0,
                    "llm_input": None,
                    "llm_output": None,
                    "reason": "unsupported removal is rejected before the provider boundary",
                    "verification_test": (
                        "tests/product_app/test_openai_copilot.py::"
                        "test_removal_language_is_rejected_before_provider_call"
                    ),
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        (evidence_dir / "map_render_ack.json").write_text(
            json.dumps(after, indent=2, sort_keys=True), encoding="utf-8"
        )
        _capture_ready(page, evidence_dir / "unsupported_removal.png")
        context.close()
        browser.close()
@pytest.mark.skipif(not BASE_URL, reason="Set PRODUCT_E2E_BASE_URL to the live loopback app")
def test_ineligible_duration_keeps_parent_map_and_drafts(tmp_path: Path) -> None:
    evidence_dir = Path(os.environ.get("PRODUCT_E2E_INELIGIBLE_EVIDENCE_DIR", tmp_path))
    evidence_dir.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as runtime:
        launch: dict[str, object] = {"headless": True}
        if CHROME.is_file():
            launch["executable_path"] = str(CHROME)
        browser = runtime.chromium.launch(**launch)
        context = browser.new_context(viewport={"width": 1440, "height": 900})
        page = context.new_page()
        page.goto(f"{BASE_URL}/app/itinerary", wait_until="domcontentloaded")
        expect(page.locator("#app-shell")).to_be_visible(timeout=15_000)
        initial = _map_ack(page)

        page.locator('.day-card[data-day="4"]').click()
        expect(page.locator("#map-context-title")).to_contain_text(
            "Sequoia National Park", timeout=15_000
        )
        for stop_id in ("sequoia_national_park", "kings_canyon_national_park"):
            if stop_id == "kings_canyon_national_park":
                page.locator("details.textual-map-alternative > summary").click()
                page.locator(f'[data-geo-stop="{stop_id}"]').click()
            page.locator("#map-edit-button").click()
            page.locator("#inspect-edit-stop").click()
            dialog = page.locator("#map-edit-dialog")
            expect(dialog).to_be_visible(timeout=15_000)
            dialog.locator("details", has_text="Duration and time window").locator(
                "summary"
            ).click()
            form = dialog.locator(
                'form[data-typed-operation="set_stop_duration"]'
            )
            form.locator('input[name="duration-preferred"]').fill("480")
            with page.expect_response(
                lambda response: response.url.endswith("/draft/operations")
                and response.request.method == "POST"
            ) as draft_response_info:
                form.locator('button[type="submit"]').click()
            assert draft_response_info.value.status == 200, draft_response_info.value.text()
            expect(page.locator("#toast-region")).to_contain_text(
                "Draft operation added", timeout=15_000
            )
            page.locator("#close-map-edit").click()

        staged = _map_ack(page)
        assert staged["rendered_map_plan_id"] == initial["rendered_map_plan_id"]
        assert staged["rendered_geometry_hash"] == initial["rendered_geometry_hash"]
        assert staged["rendered_marker_ids"] == initial["rendered_marker_ids"]
        page.locator("#map-edit-button").click()
        expect(page.locator("#inline-preview")).to_be_visible(timeout=15_000)
        expect(page.locator("#inline-preview")).to_be_enabled(timeout=15_000)
        with page.expect_response(
            lambda response: response.url.endswith("/preview")
            and response.request.method == "POST"
        ) as response_info:
            page.locator("#inline-preview").click()
        response = response_info.value
        assert response.status == 200, response.text()
        payload = response.json()
        proposal = payload["proposal"]
        assert proposal["state"] == "ineligible"
        assert proposal["decision_eligible"] is False
        assert "day_time_exceeded" in proposal["certificate_schedule_evidence"][
            "failure_codes"
        ]
        expect(page.locator(".map-stage")).to_have_attribute(
            "data-preview-state", "PREVIEW_INELIGIBLE", timeout=30_000
        )
        after = _map_ack(page)
        assert after["rendered_map_plan_id"] == initial["rendered_map_plan_id"]
        assert after["rendered_geometry_hash"] == initial["rendered_geometry_hash"]
        assert after["rendered_marker_ids"] == initial["rendered_marker_ids"]
        expect(page.locator(".draft-bar")).to_contain_text("2 draft changes")

        request = response.request
        (evidence_dir / "preview_request.json").write_text(
            json.dumps(
                {
                    "method": request.method,
                    "url": request.url,
                    "headers": {
                        "Content-Type": request.headers.get("content-type", ""),
                        "X-Session-Token": "OMITTED_SECRET",
                    },
                    "body_utf8": request.post_data or "",
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        (evidence_dir / "preview_response.json").write_bytes(response.body())
        (evidence_dir / "map_render_ack.json").write_text(
            json.dumps(after, indent=2, sort_keys=True), encoding="utf-8"
        )
        (evidence_dir / "llm_io.json").write_text(
            json.dumps(
                {
                    "provider_call_count": 0,
                    "llm_input": None,
                    "llm_output": None,
                    "reason": "direct typed edits and deterministic preview do not invoke an LLM",
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        _capture_ready(page, evidence_dir / "ineligible_parent_preserved.png")
        _write_proposal_artifacts(evidence_dir, proposal)
        context.close()
        browser.close()
@pytest.mark.skipif(not BASE_URL, reason="Set PRODUCT_E2E_BASE_URL to the live loopback app")
def test_older_preview_response_cannot_overwrite_newer_preview(tmp_path: Path) -> None:
    evidence_dir = Path(os.environ.get("PRODUCT_E2E_STALE_EVIDENCE_DIR", tmp_path))
    evidence_dir.mkdir(parents=True, exist_ok=True)

    hold_first_preview = """
    (() => {
      const originalFetch = window.fetch.bind(window);
      window.__e2ePreviewHold = { held: false, release: null, requestBody: null };
      window.fetch = async (...args) => {
        const raw = args[0];
        const url = typeof raw === "string" ? raw : raw?.url || "";
        if (url.endsWith("/preview") && !window.__e2ePreviewHold.held) {
          window.__e2ePreviewHold.held = true;
          window.__e2ePreviewHold.requestBody = JSON.parse(args[1]?.body || "{}");
          await new Promise((resolve) => { window.__e2ePreviewHold.release = resolve; });
        }
        return originalFetch(...args);
      };
    })();
    """

    with sync_playwright() as runtime:
        launch: dict[str, object] = {"headless": True}
        if CHROME.is_file():
            launch["executable_path"] = str(CHROME)
        browser = runtime.chromium.launch(**launch)
        context = browser.new_context(viewport={"width": 1440, "height": 900})
        context.add_init_script(hold_first_preview)
        page = context.new_page()
        page.goto(f"{BASE_URL}/app/itinerary", wait_until="domcontentloaded")
        expect(page.locator("#app-shell")).to_be_visible(timeout=15_000)
        initial = _map_ack(page)

        day_five = page.locator('.day-card[data-day="5"]')
        day_five.click()
        page.locator("#map-edit-button").click()
        page.locator("#inspect-edit-stop").click()
        page.locator(".legacy-edit-controls summary").click()
        page.locator("#registered-candidate").select_option(
            "pfeiffer_big_sur_state_park"
        )
        with page.expect_response(
            lambda response: response.url.endswith("/draft/operations")
            and response.request.method == "POST"
        ):
            page.locator("#draft-replace").click()
        page.locator("#close-map-edit").click()
        expect(page.locator("#inline-preview")).to_be_enabled(timeout=15_000)

        page.evaluate("document.querySelector('#inline-preview').click()")
        expect(page.locator(".map-stage")).to_have_attribute(
            "data-preview-state", "PREVIEW_RUNNING", timeout=15_000
        )
        page.wait_for_function("window.__e2ePreviewHold.held === true")
        first_request_body = page.evaluate("window.__e2ePreviewHold.requestBody")

        with page.expect_response(
            lambda response: response.url.endswith("/draft/undo")
            and response.request.method == "POST"
        ) as undo_info:
            page.locator("#inline-undo").click()
        assert undo_info.value.status == 200, undo_info.value.text()
        expect(page.locator(".map-stage")).to_have_attribute(
            "data-preview-state", "PARENT_RENDERED", timeout=15_000
        )

        if not page.locator("#inspect-edit-stop").is_visible():
            page.locator("#map-edit-button").click()
        page.locator("#inspect-edit-stop").click()
        if not page.locator("#registered-candidate").is_visible():
            page.locator(".legacy-edit-controls summary").click()
        expect(page.locator("#registered-candidate")).to_be_visible(timeout=15_000)
        page.locator("#registered-candidate").select_option(
            "pfeiffer_big_sur_state_park"
        )
        with page.expect_response(
            lambda response: response.url.endswith("/draft/operations")
            and response.request.method == "POST"
        ):
            page.locator("#draft-replace").click()
        page.locator("#close-map-edit").click()
        expect(page.locator("#inline-preview")).to_be_enabled(timeout=15_000)

        with page.expect_response(
            lambda response: response.url.endswith("/preview")
            and response.request.method == "POST"
        ) as second_info:
            page.locator("#inline-preview").click()
        second_response = second_info.value
        assert second_response.status == 200, second_response.text()
        second_payload = second_response.json()
        second_proposal = second_payload["session"]["proposal"]
        expect(page.locator(".map-stage")).to_have_attribute(
            "data-preview-state", "PREVIEW_RENDERED", timeout=30_000
        )
        before_release = _map_ack(page)
        assert before_release["rendered_map_plan_id"] == second_proposal["child_plan_id"]

        with page.expect_response(
            lambda response: response.url.endswith("/preview")
            and response.request.method == "POST"
        ) as first_info:
            page.evaluate("window.__e2ePreviewHold.release()")
        first_response = first_info.value
        assert first_response.status == 409
        first_payload = first_response.json()
        assert first_payload["detail"] == "stale_session_revision"
        page.wait_for_timeout(750)

        after_release = _map_ack(page)
        stage = page.locator(".map-stage")
        expect(stage).to_have_attribute(
            "data-preview-request-id",
            second_proposal["compiled_request"]["source_request_id"],
        )
        assert after_release == before_release
        assert after_release["rendered_map_plan_id"] == second_proposal["child_plan_id"]
        assert after_release["rendered_geometry_hash"] != initial["rendered_geometry_hash"]
        assert first_request_body["expected_revision"] < second_response.request.post_data_json["expected_revision"]

        (evidence_dir / "preview_a_request.json").write_text(
            json.dumps(first_request_body, indent=2, sort_keys=True), encoding="utf-8"
        )
        (evidence_dir / "preview_a_response.json").write_text(
            json.dumps(first_payload, indent=2, sort_keys=True), encoding="utf-8"
        )
        (evidence_dir / "preview_b_request.json").write_text(
            json.dumps(second_response.request.post_data_json, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        (evidence_dir / "preview_b_response.json").write_text(
            json.dumps(second_payload, indent=2, sort_keys=True), encoding="utf-8"
        )
        (evidence_dir / "map_render_ack.json").write_text(
            json.dumps(after_release, indent=2, sort_keys=True), encoding="utf-8"
        )
        (evidence_dir / "trace.json").write_text(
            json.dumps(
                {
                    "older_expected_revision": first_request_body["expected_revision"],
                    "older_response_status": first_response.status,
                    "newer_request_id": second_proposal["compiled_request"]["source_request_id"],
                    "rendered_map_plan_id": after_release["rendered_map_plan_id"],
                    "rendered_geometry_hash": after_release["rendered_geometry_hash"],
                    "older_response_ignored": True,
                    "assertions_passed": True,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        (evidence_dir / "llm_io.json").write_text(
            json.dumps(
                {
                    "provider_call_count": 0,
                    "llm_input": None,
                    "llm_output": None,
                    "reason": "Direct repair preview path; no Copilot/provider call.",
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        _capture_ready(page, evidence_dir / "screenshot_after.png")
        _write_proposal_artifacts(evidence_dir, second_proposal)
        context.close()
        browser.close()

@pytest.mark.skipif(not BASE_URL, reason="Set PRODUCT_E2E_BASE_URL to the live loopback app")
def test_accepted_replacement_survives_reload_with_exact_map_identity(tmp_path: Path) -> None:
    evidence_dir = Path(os.environ.get("PRODUCT_E2E_ACCEPT_EVIDENCE_DIR", tmp_path))
    evidence_dir.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as runtime:
        launch: dict[str, object] = {"headless": True}
        if CHROME.is_file():
            launch["executable_path"] = str(CHROME)
        browser = runtime.chromium.launch(**launch)
        context = browser.new_context(viewport={"width": 1440, "height": 900})
        page = context.new_page()
        console_errors: list[str] = []
        session_creation_requests: list[str] = []
        page.on("request", lambda request: session_creation_requests.append(request.url)
                if request.method == "POST" and request.url.endswith("/api/sessions") else None)
        page.on("console", lambda message: console_errors.append(message.text) if message.type == "error" else None)
        page.on("pageerror", lambda error: console_errors.append(str(error)))

        def capture_diagnostics(label: str) -> None:
            observed = page.evaluate("""() => ({
              observed_at: new Date().toISOString(),
              geographic_map: {...document.querySelector('#geographic-map')?.dataset},
              map_stage: {...document.querySelector('.map-stage')?.dataset},
              map_status: document.querySelector('.map-stage [role=status]')?.textContent,
              map_status_code: document.querySelector('#map-render-status')?.dataset.mapStatusCode,
              inspector_title: document.querySelector('#map-context-title')?.textContent,
              toast: document.querySelector('#toast-region')?.textContent,
            })""")
            (evidence_dir / f"{label}_diagnostics.json").write_text(
                json.dumps({**observed, "console_errors": console_errors}, indent=2, sort_keys=True), encoding="utf-8"
            )

        def capture_http(response, label: str) -> None:
            (evidence_dir / f"{label}_request.json").write_text(
                json.dumps(json.loads(response.request.post_data or "{}"), indent=2, sort_keys=True), encoding="utf-8"
            )
            (evidence_dir / f"{label}_response.json").write_bytes(response.body())
            (evidence_dir / f"{label}_http_io.json").write_text(json.dumps({
                "method": response.request.method,
                "url": response.url,
                "request_body_utf8": response.request.post_data or "",
                "response_status_code": response.status,
                "response_body_utf8": response.body().decode("utf-8"),
                "headers_persisted": False,
            }, indent=2, sort_keys=True), encoding="utf-8")

        page.goto(f"{BASE_URL}/app/itinerary", wait_until="domcontentloaded")
        expect(page.locator("#app-shell")).to_be_visible(timeout=15_000)
        initial = _map_ack(page)

        day_five = page.locator('.day-card[data-day="5"]')
        expect(day_five).to_contain_text("Bixby Creek Bridge")
        day_five.click()
        page.locator("#map-edit-button").click()
        page.locator("#inspect-edit-stop").click()
        page.locator(".legacy-edit-controls summary").click()
        page.locator("#registered-candidate").select_option(
            "pfeiffer_big_sur_state_park"
        )
        with page.expect_response(
            lambda response: response.url.endswith("/draft/operations")
            and response.request.method == "POST"
        ):
            page.locator("#draft-replace").click()
        page.locator("#close-map-edit").click()
        expect(page.locator("#inline-preview")).to_be_enabled(timeout=15_000)

        with page.expect_response(
            lambda response: response.url.endswith("/preview")
            and response.request.method == "POST"
        ) as preview_info:
            page.locator("#inline-preview").click()
        preview_response = preview_info.value
        capture_http(preview_response, "preview")
        assert preview_response.status == 200, preview_response.text()
        preview_payload = preview_response.json()
        proposal = preview_payload["session"]["proposal"]
        expect(page.locator("#accept-repair")).to_be_enabled(timeout=30_000)
        preview_ack = _map_ack(page)
        session_id = preview_payload["session"]["session_id"]
        page.locator("#open-copilot").click()
        expect(page.locator("#copilot-dock")).to_be_visible()
        prompt = "Review the current itinerary without changing it."
        page.evaluate("""() => {
          const originalFetch = window.fetch.bind(window);
          window.__acceptanceCopilotRaw = null;
          window.fetch = async (...args) => {
            const response = await originalFetch(...args);
            const url = typeof args[0] === 'string' ? args[0] : args[0].url;
            if (url.endsWith('/copilot/messages')) {
              response.clone().text().then(raw => { window.__acceptanceCopilotRaw = raw; });
            }
            return response;
          };
        }""")
        page.locator("#copilot-input").fill(prompt)
        with page.expect_response(
            lambda response: response.url == f"{BASE_URL}/api/sessions/{session_id}/conversation"
            and response.request.method == "GET"
        ) as conversation_before_info:
            with page.expect_response(
                lambda response: response.url.endswith("/copilot/messages")
                and response.request.method == "POST"
            ) as chat_info:
                page.locator("#copilot-form button[type='submit']").click()
            page.wait_for_function("() => typeof window.__acceptanceCopilotRaw === 'string'")
        chat_response = chat_info.value
        chat_raw = page.evaluate("window.__acceptanceCopilotRaw")
        (evidence_dir / "copilot_message_http_io.json").write_text(json.dumps({
            "method": chat_response.request.method,
            "url": chat_response.url,
            "request_body_utf8": chat_response.request.post_data,
            "response_status_code": chat_response.status,
            "response_body_utf8": chat_raw,
            "headers_persisted": False,
        }, indent=2, sort_keys=True), encoding="utf-8")
        assert chat_response.status == 200
        chat_events = _sse_events(chat_raw)
        assert chat_events[-1]["type"] == "turn.completed"
        assert chat_events[-1]["payload"]["turn"]["provider"] == "deterministic"
        capture_http(conversation_before_info.value, "conversation_before")
        conversation_before = conversation_before_info.value.json()["conversation"]
        assert re.fullmatch(r"conversation_[0-9a-f]{32}", conversation_before["conversation_id"])
        assert any(turn["user_message"] == prompt for turn in conversation_before["turns"])
        expect(page.locator("#conversation .message-user")).to_contain_text(prompt)
        page.locator("#close-copilot").click()

        confirmation_messages: list[str] = []
        with page.expect_response(
            lambda response: response.url.endswith("/accept")
            and response.request.method == "POST"
        ) as accept_info:
            page.locator("#accept-repair").click()
            expect(page.get_by_role("dialog", name="Confirm trip decision", exact=True)).to_be_visible()
            confirmation_messages.append(page.locator("#acceptance-message").inner_text())
            page.get_by_role("dialog", name="Confirm trip decision", exact=True).get_by_role("button", name='Accept repair', exact=True).click()
        accept_response = accept_info.value
        capture_http(accept_response, "accept")
        capture_diagnostics("accept_response_received")
        assert accept_response.status == 200, accept_response.text()
        accept_payload = accept_response.json()
        assert confirmation_messages == [
            "Accept this independently evaluated repair as the durable trip plan?"
        ]
        assert accept_payload["session"]["accepted_plan_id"] == proposal["child_plan_id"]
        try:
            accepted_ack = _map_ack(page)
        except Exception:
            capture_diagnostics("accept_failure")
            raise
        capture_diagnostics("accept_rendered")
        assert accepted_ack["rendered_map_plan_id"] == proposal["child_plan_id"]
        assert accepted_ack["rendered_geometry_hash"] == preview_ack["rendered_geometry_hash"]
        assert accepted_ack["rendered_marker_ids"] == preview_ack["rendered_marker_ids"]

        creation_count_before_reload = len(session_creation_requests)
        with page.expect_response(
            lambda response: response.url == f"{BASE_URL}/api/sessions/{session_id}/conversation"
            and response.request.method == "GET",
            timeout=30_000,
        ) as conversation_after_info:
            with page.expect_response(
                lambda response: response.url == f"{BASE_URL}/api/sessions/{session_id}"
                and response.request.method == "GET",
                timeout=30_000,
            ) as reload_info:
                page.reload(wait_until="domcontentloaded")
        reload_response = reload_info.value
        capture_http(reload_response, "reload")
        capture_diagnostics("reload_response_received")
        assert reload_response.status == 200, reload_response.text()
        restored_session = reload_response.json()["session"]
        assert restored_session["session_id"] == session_id
        assert restored_session["conversation_id"] == conversation_before["conversation_id"]
        assert restored_session["selected_stop_id"] == "pfeiffer_big_sur_state_park"
        assert restored_session["selected_day"] == 5
        expect(page.locator("#loading-screen")).to_be_hidden(timeout=30_000)
        try:
            restored_ack = _map_ack(page)
        except Exception:
            capture_diagnostics("reload_failure")
            raise
        capture_diagnostics("reload_rendered")
        expect(page.locator('.day-card[data-day="5"]')).to_contain_text(
            "Pfeiffer Big Sur State Park", timeout=30_000
        )
        assert restored_ack["rendered_map_plan_id"] == proposal["child_plan_id"]
        assert restored_ack["active_plan_content_hash"] == proposal["child_plan_content_hash"]
        assert restored_ack["rendered_geometry_hash"] == accepted_ack["rendered_geometry_hash"]
        assert restored_ack["rendered_marker_ids"] == accepted_ack["rendered_marker_ids"]
        assert restored_ack["rendered_geometry_hash"] != initial["rendered_geometry_hash"]
        expect(page.locator("#map-context-title")).to_contain_text("Pfeiffer Big Sur State Park")
        page.locator("#open-copilot").click()
        capture_http(conversation_after_info.value, "conversation_after")
        assert conversation_after_info.value.status == 200
        conversation_after = conversation_after_info.value.json()["conversation"]
        assert conversation_after == conversation_before
        expect(page.locator("#conversation .message-user")).to_contain_text(prompt)
        page.locator("#close-copilot").click()
        assert len(session_creation_requests) == creation_count_before_reload == 1
        assert console_errors == []
        (evidence_dir / "session_reload_assertions.json").write_text(json.dumps({
            "session_id": session_id,
            "conversation_id": conversation_before["conversation_id"],
            "selected_stop_id": restored_session["selected_stop_id"],
            "selected_day": restored_session["selected_day"],
            "session_creation_count_before_reload": creation_count_before_reload,
            "session_creation_count_after_reload": len(session_creation_requests),
            "console_errors": console_errors,
            "assertions_passed": True,
        }, indent=2, sort_keys=True), encoding="utf-8")

        (evidence_dir / "map_render_ack.json").write_text(
            json.dumps(restored_ack, indent=2, sort_keys=True), encoding="utf-8"
        )
        (evidence_dir / "trace.json").write_text(
            json.dumps(
                {
                    "parent_plan_id": proposal["parent_plan_id"],
                    "child_plan_id": proposal["child_plan_id"],
                    "child_plan_content_hash": proposal["child_plan_content_hash"],
                    "geometry_hash": restored_ack["rendered_geometry_hash"],
                    "rendered_map_plan_id": restored_ack["rendered_map_plan_id"],
                    "rendered_marker_ids": restored_ack["rendered_marker_ids"],
                    "survived_reload": True,
                    "assertions_passed": True,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        (evidence_dir / "llm_io.json").write_text(
            json.dumps(
                {
                    "provider_call_count": 0,
                    "llm_input": None,
                    "llm_output": None,
                    "reason": "Direct repair and acceptance with a deterministic local Copilot transcript; exact chat I/O is in copilot_message_http_io.json and no external provider is invoked.",
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        _capture_ready(page, evidence_dir / "screenshot_after_reload.png")
        _write_proposal_artifacts(evidence_dir, proposal)
        context.close()
        browser.close()
