"""Production browser A -> accepted replacement B -> accepted added-stop C.

Every mutation is caused by a visible control. No API setup, mocked requests,
provider calls, or browser state injection is used. Run only with isolated state.
"""

from __future__ import annotations

import hashlib
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
DISPOSABLE = os.environ.get("PRODUCT_E2E_ADDITION_CONTINUATION_DISPOSABLE") == "1"
BIXBY = "bixby_creek_bridge_viewpoint"
PFEIFFER = "pfeiffer_big_sur_state_park"
LOBOS = "point_lobos_state_natural_reserve"


def _artifact_hash(record: dict) -> str:
    payload = {key: value for key, value in record.items() if key != "content_hash"}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()[
        :16
    ]


@pytest.mark.skipif(not BASE_URL or not DISPOSABLE, reason="Requires a fresh isolated addition-continuation app")
def test_browser_accept_replacement_then_add_candidate_and_preserve_lineage(tmp_path: Path) -> None:
    evidence = Path(os.environ.get("PRODUCT_E2E_ADDITION_CONTINUATION_EVIDENCE_DIR", tmp_path))
    evidence.mkdir(parents=True, exist_ok=True)
    errors = {"console_errors": [], "request_failures": [], "http_errors": [], "http_capture_errors": []}
    http_index, checkpoints, decisions, creations = [], {}, [], []
    decision_controls, decision_events = [], []
    current_session = None
    phase = "initial_parent"

    with sync_playwright() as runtime:
        launch = {
            "headless": os.environ.get("PRODUCT_E2E_HEADED") != "1",
            "slow_mo": int(os.environ.get("PRODUCT_E2E_SLOW_MO_MS", "0")),
        }
        if CHROME.is_file():
            launch["executable_path"] = str(CHROME)
        browser = runtime.chromium.launch(**launch)
        options = {"viewport": {"width": 1440, "height": 900}}
        if os.environ.get("PRODUCT_E2E_RECORD_VIDEO") == "1":
            options.update(record_video_dir=str(evidence), record_video_size={"width": 1440, "height": 900})
        context = browser.new_context(**options)
        context.add_init_script("""(() => {
          window.__decisionControlEvents=[];
          document.addEventListener('click',event=>{
            const button=event.target.closest?.('#accept-repair, #keep-original');
            if(!button) return;
            window.__decisionControlEvents.push({type:event.type,trusted:event.isTrusted,at:Date.now(),
              selector:'#'+button.id,path:location.pathname,visible:!!button.getClientRects().length,
              disabled:button.disabled,label:button.textContent.trim()});
          },true);
        })();""")
        page = context.new_page()
        page.on(
            "console",
            lambda message: errors["console_errors"].append(message.text) if message.type == "error" else None,
        )
        page.on("pageerror", lambda error: errors["console_errors"].append(str(error)))
        page.on(
            "requestfailed",
            lambda request: errors["request_failures"].append(f"{request.method} {request.url}: {request.failure}"),
        )

        def record_response(response):
            nonlocal current_session
            if response.status >= 400:
                errors["http_errors"].append(f"{response.status} {response.url}")
            path = urlsplit(response.url).path
            if not (path.startswith("/api/sessions") or path in {"/api/health", "/api/map/config"}):
                return
            try:
                row = {"phase": phase, "observed_at": datetime.now(UTC).isoformat(), **_http_record(response)}
                filename = f"http_{len(http_index) + 1:03d}.json"
                _write_json(evidence / filename, row)
                http_index.append(
                    {
                        "file": filename,
                        "phase": phase,
                        "method": row["method"],
                        "url": response.url,
                        "status": row["response_status_code"],
                    }
                )
                if response.status == 200:
                    body = json.loads(row["response_body_utf8"])
                    if isinstance(body, dict) and "session" in body:
                        current_session = body["session"]
                if path == "/api/sessions" and response.request.method == "POST":
                    creations.append(json.loads(row["response_body_utf8"])["session"]["session_id"])
            except Exception as error:
                errors["http_capture_errors"].append(
                    {"url": response.url, "type": type(error).__name__, "message": str(error)}
                )

        page.on("response", record_response)

        def capture(label):
            _capture_ready(page, evidence / f"{label}.png")
            checkpoints[label] = _map_ack(page)
            _write_json(evidence / "map_checkpoints.json", checkpoints)
            return checkpoints[label]

        def same_render(actual, expected):
            for field in (
                "active_plan_id",
                "active_plan_content_hash",
                "rendered_map_plan_id",
                "rendered_geometry_hash",
            ):
                assert actual[field] == expected[field], (field, actual, expected)
            assert sorted(actual["rendered_marker_ids"]) == sorted(expected["rendered_marker_ids"])

        def stage(kind, candidate):
            day = page.locator('.day-card[data-day="5"]')
            day.click()
            expected_title = "Bixby Creek Bridge" if kind == "replace_nearby" else "Pfeiffer Big Sur State Park"
            expect(page.locator("#map-context-title")).to_contain_text(expected_title, timeout=15_000)
            page.locator("#map-edit-button").click()
            page.locator("#inspect-edit-stop").click()
            expect(page.locator("#map-edit-dialog")).to_be_visible()
            shortcuts = page.locator(".legacy-edit-controls")
            if not shortcuts.evaluate("element => element.open"):
                shortcuts.locator("summary").click()
            page.locator("#registered-candidate").select_option(candidate)
            button = "#draft-replace" if kind == "replace_nearby" else "#draft-add"
            if kind == "add_candidate":
                expect(page.locator("#destination-day")).to_be_enabled()
                page.locator("#destination-day").select_option("5")
            expect(page.locator(button)).to_be_enabled()
            with page.expect_response(
                lambda response: response.url.endswith("/draft/operations") and response.request.method == "POST",
                timeout=30_000,
            ) as received:
                page.locator(button).click()
            response = received.value
            assert response.status == 200, response.text()
            request = json.loads(response.request.post_data)
            assert request["type"] == kind
            assert request["target"] == (BIXBY if kind == "replace_nearby" else candidate)
            assert request["parameters"] == ({"candidate_id": candidate} if kind == "replace_nearby" else {"day": 5})
            payload = response.json()
            assert len(payload["session"]["draft"]) == 1
            assert payload["operation"]["type"] == kind
            page.locator("#close-map-edit").click()
            return payload

        def visible_evidence(proposal, label):
            page.locator('.nav-item[data-route="/app/evidence"]').click()
            panel = page.locator("[data-evidence-plan-id]")
            expect(panel).to_have_attribute("data-evidence-plan-id", proposal["child_plan_id"], timeout=15_000)
            expect(panel).to_have_attribute("data-plan-content-hash", proposal["child_plan_content_hash"])
            expect(panel).to_have_attribute("data-evaluation-id", proposal["certificate_id"])
            expect(panel).to_contain_text(proposal["diff_id"])
            expect(panel).to_contain_text(proposal["certificate_id"])
            _write_json(
                evidence / f"{label}_evidence_dom.json",
                {
                    "attributes": panel.evaluate("element => ({...element.dataset})"),
                    "visible_text": panel.inner_text(),
                },
            )
            page.locator('.nav-item[data-route="/app/itinerary"]').click()

        def preview(parent_ack, label):
            expect(page.locator("#inline-preview")).to_be_enabled(timeout=30_000)
            started = datetime.now(UTC).isoformat()
            with page.expect_response(
                lambda response: response.url.endswith("/preview") and response.request.method == "POST", timeout=90_000
            ) as received:
                page.locator("#inline-preview").click()
            response = received.value
            assert response.status == 200, response.text()
            payload = response.json()
            proposal = payload["proposal"]
            assert proposal == payload["session"]["proposal"]
            execution = proposal["runtime_execution"]
            child, diff, certificate = (
                execution["child_plan"],
                execution["plan_diff"],
                execution["evaluation_certificate"],
            )
            assert execution["evaluation_executed"] is True
            assert execution["evaluator"] == "PlanEvaluator.evaluate_final_plan"
            assert started <= execution["started_at"] <= execution["completed_at"] <= datetime.now(UTC).isoformat()
            assert proposal["state"] == proposal["eligibility"] == certificate["comparison_eligibility"] == "eligible"
            assert (
                child["plan_id"]
                == proposal["child_plan_id"]
                == execution["child_plan_id"]
                == certificate["plan_id"]
                == diff["child_plan_id"]
            )
            assert (
                child["content_hash"]
                == _artifact_hash(child)
                == proposal["child_plan_content_hash"]
                == certificate["plan_content_hash"]
            )
            assert certificate["content_hash"] == _artifact_hash(certificate) == proposal["certificate_content_hash"]
            assert certificate["certificate_id"] == proposal["certificate_id"] == execution["evaluation_id"]
            assert diff["diff_id"] == proposal["diff_id"]
            assert _artifact_hash(diff) == proposal["diff_content_hash"]
            assert (
                child["parent_plan_id"]
                == proposal["parent_plan_id"]
                == diff["parent_plan_id"]
                == parent_ack["active_plan_id"]
            )
            assert proposal["parent_plan_content_hash"] == parent_ack["active_plan_content_hash"]
            assert certificate["route_validation"] == proposal["route_validation"]
            route = certificate["route_validation"]
            assert route["required_leg_count"] == route["road_validated_leg_count"] > 0
            assert route["missing_leg_count"] == route["fallback_leg_count"] == 0
            expect(page.locator(".map-stage")).to_have_attribute(
                "data-preview-state", "PREVIEW_RENDERED", timeout=30_000
            )
            expect(page.locator(".map-stage")).to_have_attribute("data-rendered-map-plan-id", child["plan_id"])
            ack = capture(f"{label}_preview")
            assert ack["active_plan_content_hash"] == child["content_hash"]
            assert ack["rendered_geometry_hash"] == _independent_geometry_hash(page, proposal["geography_plan"])
            assert ack["rendered_geometry_hash"] != parent_ack["rendered_geometry_hash"]
            markers = {feature["properties"]["stop_id"] for feature in proposal["geography_plan"]["stops"]["features"]}
            assert markers == set(ack["rendered_marker_ids"])
            metrics = page.locator("#metric-strip")
            expect(metrics).to_have_attribute("data-plan-id", child["plan_id"])
            expect(metrics).to_have_attribute("data-plan-content-hash", child["content_hash"])
            expect(metrics).to_have_attribute("data-evaluation-id", certificate["certificate_id"])
            _write_json(
                evidence / f"{label}_metrics_dom.json",
                {
                    "attributes": metrics.evaluate("element => ({...element.dataset})"),
                    "visible_text": metrics.inner_text(),
                },
            )
            _write_json(evidence / f"{label}_canonical_artifacts.json", {"proposal": proposal, "map_ack": ack})
            page.locator('.nav-item[data-route="/app/compare"]').click()
            expect(page.locator(f'[data-option-card="{child["plan_id"]}"]')).to_be_visible(timeout=30_000)
            option = page.locator('[data-compare-map-pane="option"] [data-compare-map-canvas]')
            expect(option).to_have_attribute("data-rendered-map-plan-id", child["plan_id"], timeout=30_000)
            expect(option).to_have_attribute("data-rendered-plan-content-hash", child["content_hash"])
            expect(option).to_have_attribute("data-rendered-geometry-hash", ack["rendered_geometry_hash"])
            _capture_ready(page, evidence / f"{label}_compare.png")
            visible_evidence(proposal, label)
            same_render(capture(f"{label}_after_evidence"), ack)
            return proposal, ack

        def accept_and_reload(proposal, preview_ack, label):
            dialogs = []

            expect(page.locator("#accept-repair")).to_be_enabled(timeout=30_000)
            before = json.loads(json.dumps(current_session))
            event_start = page.evaluate("window.__decisionControlEvents.length")
            with page.expect_response(
                lambda response: response.url.endswith("/accept") and response.request.method == "POST", timeout=60_000
            ) as received:
                page.locator("#accept-repair").click()
                expect(page.get_by_role("dialog", name="Confirm trip decision", exact=True)).to_be_visible()
                dialogs.append(page.locator("#acceptance-message").inner_text())
                page.get_by_role("dialog", name="Confirm trip decision", exact=True).get_by_role(
                    "button", name="Accept repair", exact=True
                ).click()
            response = received.value
            assert response.status == 200, response.text()
            payload = response.json()
            assert dialogs == ["Accept this independently evaluated repair as the durable trip plan?"]
            assert json.loads(response.request.post_data)["confirmation"] == "accept_evaluated_repair"
            session = payload["session"]
            assert session["accepted_plan_id"] == proposal["child_plan_id"]
            assert session["draft"] == [] and session["proposal"] is None
            decision = payload["result"]["decision"]
            assert decision["decision_type"] == "accept_repair"
            assert decision["previous_plan_id"] == proposal["parent_plan_id"]
            assert decision["selected_plan_id"] == proposal["child_plan_id"]
            assert decision["selected_plan_content_hash"] == proposal["child_plan_content_hash"]
            assert decision["certificate_id"] == proposal["certificate_id"]
            assert decision["certificate_content_hash"] == proposal["certificate_content_hash"]
            assert decision["diff_id"] == proposal["diff_id"]
            decision_payload = {key: value for key, value in decision.items() if key != "content_hash"}
            assert (
                hashlib.sha256(
                    json.dumps(
                        decision_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
                    ).encode()
                ).hexdigest()
                == decision["content_hash"]
            )
            decisions.append(decision)
            _write_json(evidence / "acceptance_decisions.json", decisions)
            same_render(capture(f"{label}_accepted"), preview_ack)
            clicked = page.evaluate("start=>window.__decisionControlEvents.slice(start)", event_start)
            decision_events.extend(page.evaluate("window.__decisionControlEvents"))
            session_url = f"{BASE_URL}/api/sessions/{session['session_id']}"
            with page.expect_response(
                lambda response: response.url == session_url and response.request.method == "GET", timeout=30_000
            ) as reloaded:
                page.reload(wait_until="domcontentloaded")
            assert reloaded.value.status == 200, reloaded.value.text()
            restored = reloaded.value.json()
            assert restored["session"]["session_id"] == session["session_id"]
            assert restored["session"]["accepted_plan_id"] == proposal["child_plan_id"]
            exact = restored["workspace"]["alternative_evidence"][proposal["child_plan_id"]]
            assert exact["certificate"] == proposal["runtime_execution"]["evaluation_certificate"]
            assert exact["diff"] == proposal["runtime_execution"]["plan_diff"]
            pointer = restored["workspace"]["accepted_plan_pointer"]
            assert pointer["decision_id"] == decision["decision_id"]
            assert pointer["accepted_plan_content_hash"] == proposal["child_plan_content_hash"]
            same_render(capture(f"{label}_after_reload"), preview_ack)
            expect(page.locator("#metric-strip")).to_have_attribute("data-evaluation-id", proposal["certificate_id"])
            assert creations == [session["session_id"]]
            decision_controls.append(
                {
                    "binding_id": "accept_repair",
                    "selector": "#accept-repair",
                    "context": "original" if label == "b" else "accepted_child",
                    "phase": phase,
                    "events": clicked,
                    "confirmation_messages": dialogs,
                    "session_before": before,
                    "request": response.request.post_data_json,
                    "response": payload,
                    "restored": restored,
                    "rendered_before": preview_ack,
                    "rendered_after": checkpoints[f"{label}_after_reload"],
                    "assertions_passed": True,
                }
            )
            return restored

        def keep_original(context_name, current):
            nonlocal phase
            phase = "keep_original_" + context_name
            before = json.loads(json.dumps(current_session))
            clicked_from = page.evaluate("window.__decisionControlEvents.length")
            dialogs = []
            expect(page.locator("#keep-original")).to_be_visible()
            expect(page.locator("#keep-original")).to_be_enabled()
            with page.expect_response(
                lambda response: response.url.endswith("/keep-original") and response.request.method == "POST"
            ) as received:
                page.locator("#keep-original").click()
                expect(page.get_by_role("dialog", name="Confirm trip decision", exact=True)).to_be_visible()
                dialogs.append(page.locator("#acceptance-message").inner_text())
                page.get_by_role("dialog", name="Confirm trip decision", exact=True).get_by_role(
                    "button", name="Keep original", exact=True
                ).click()
            response = received.value
            assert response.status == 200, response.text()
            payload = response.json()
            assert payload["result"]["decision"]["decision_type"] == "keep_original"
            assert payload["result"]["decision"]["previous_plan_id"] == before["accepted_plan_id"]
            assert payload["session"]["accepted_plan_id"] == initial["active_plan_id"]
            assert payload["session"]["draft"] == [] and payload["session"]["proposal"] is None
            same_render(capture(phase + "_rendered"), initial)
            clicked = page.evaluate("start=>window.__decisionControlEvents.slice(start)", clicked_from)
            decision_events.extend(page.evaluate("window.__decisionControlEvents"))
            session_url = f"{BASE_URL}/api/sessions/{before['session_id']}"
            with page.expect_response(
                lambda response: response.url == session_url and response.request.method == "GET"
            ) as reloaded:
                page.reload(wait_until="domcontentloaded")
            assert reloaded.value.status == 200
            restored = reloaded.value.json()
            assert restored["workspace"]["accepted_plan_pointer"] == payload["workspace"]["accepted_plan_pointer"]
            assert restored["session"]["accepted_plan_id"] == initial["active_plan_id"]
            same_render(capture(phase + "_reloaded"), initial)
            decision_controls.append(
                {
                    "binding_id": "keep_original",
                    "selector": "#keep-original",
                    "context": context_name,
                    "phase": phase,
                    "events": clicked,
                    "confirmation_messages": dialogs,
                    "session_before": before,
                    "request": response.request.post_data_json,
                    "response": payload,
                    "restored": restored,
                    "rendered_before": current,
                    "rendered_after": checkpoints[phase + "_reloaded"],
                    "assertions_passed": True,
                }
            )

        with _preserve_browser_evidence(context, page, evidence, errors):
            try:
                page.goto(f"{BASE_URL}/app/itinerary", wait_until="domcontentloaded")
                expect(page.locator("#app-shell")).to_be_visible(timeout=30_000)
                initial = capture("a_initial")
                assert BIXBY in initial["rendered_marker_ids"]
                assert PFEIFFER not in initial["rendered_marker_ids"] and LOBOS not in initial["rendered_marker_ids"]
                phase = "stage_replacement_b"
                stage("replace_nearby", PFEIFFER)
                same_render(capture("b_staged_parent_unchanged"), initial)
                phase = "preview_b"
                proposal_b, ack_b = preview(initial, "b")
                diff_b = proposal_b["runtime_execution"]["plan_diff"]
                assert {row["stop_id"] for row in diff_b["deleted_stops"]} == {BIXBY}
                assert {row["stop_id"] for row in diff_b["added_stops"]} == {PFEIFFER}
                assert BIXBY not in ack_b["rendered_marker_ids"] and PFEIFFER in ack_b["rendered_marker_ids"]
                expect(page.locator('.day-card[data-day="5"]')).to_contain_text("Pfeiffer Big Sur State Park")
                phase = "accept_b_and_reload"
                restored_b = accept_and_reload(proposal_b, ack_b, "b")
                readiness = restored_b["workspace"]["draft_capabilities"]["operations"]["add_candidate"][
                    "candidate_day_readiness"
                ]
                assert [(row["candidate_id"], row["day"]) for row in readiness if row["preview_executable"]] == [
                    (LOBOS, 5)
                ]
                phase = "stage_addition_c"
                stage("add_candidate", LOBOS)
                same_render(capture("c_staged_parent_unchanged"), ack_b)
                phase = "preview_c"
                proposal_c, ack_c = preview(ack_b, "c")
                diff_c = proposal_c["runtime_execution"]["plan_diff"]
                assert diff_c["deleted_stops"] == []
                assert {row["stop_id"] for row in diff_c["added_stops"]} == {LOBOS}
                assert set(ack_c["rendered_marker_ids"]) == set(ack_b["rendered_marker_ids"]) | {LOBOS}
                day_five = page.locator('.day-card[data-day="5"]')
                expect(day_five).to_contain_text("Pfeiffer Big Sur State Park")
                expect(day_five).to_contain_text("Point Lobos State Natural Reserve")
                assert "Bixby Creek Bridge" not in day_five.inner_text()
                phase = "accept_c_and_reload"
                accept_and_reload(proposal_c, ack_c, "c")
                assert decisions[1]["previous_plan_id"] == decisions[0]["selected_plan_id"]
                assert decisions[1]["pointer_revision_before"] == decisions[0]["pointer_revision_after"]
                phase = "accepted_c_visible_evidence"
                visible_evidence(proposal_c, "c_accepted_after_reload")
                same_render(capture("c_final_after_evidence"), ack_c)
                keep_original("accepted_child", ack_c)
                keep_original("original", initial)
                assert all(not rows for rows in errors.values()), errors
                _write_json(
                    evidence / "browser_assertions.json",
                    {
                        "schema_version": "addition-continuation-browser-v1",
                        "assertions_passed": True,
                        "mocked_api_responses": False,
                        "direct_api_setup_calls": 0,
                        "copilot_submissions": 0,
                        "session_creation_count": len(creations),
                        "acceptance_count": len(decisions),
                        "a": initial,
                        "b": ack_b,
                        "c": ack_c,
                        "decisions": decisions,
                        **errors,
                    },
                )
                _write_json(
                    evidence / "llm_io.json",
                    {
                        "provider_call_count": 0,
                        "llm_input": None,
                        "llm_output": None,
                        "reason": "Actual UI replacement and addition with two explicit acceptances and reloads; no Copilot submission or provider call.",
                    },
                )
            finally:
                _write_json(evidence / "decision_control_journeys.json", decision_controls)
                _write_json(
                    evidence / "decision_control_events.json",
                    decision_events + page.evaluate("window.__decisionControlEvents || []"),
                )
                _write_json(evidence / "http_index.json", http_index)
                _write_json(
                    evidence / "journey_trace.json",
                    {
                        "last_phase": phase,
                        "checkpoints": checkpoints,
                        "decisions": decisions,
                        "session_creation_ids": creations,
                        **errors,
                    },
                )
        browser.close()
