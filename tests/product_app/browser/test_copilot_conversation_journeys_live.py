"""Real website conversation journeys; separate deterministic and fake-SDK scopes.

This module owns fourteen sequential disposable app lifecycles on the reserved map
origin. Set PRODUCT_E2E_CONVERSATION_DISPOSABLE=1 only after reserving port 8127.
It must not run beside another browser suite or attach to a user's running app.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit

import pytest

pytest.importorskip("playwright.sync_api")
from copilot_conversation_fixture import conversation_server, observe_context, seed_capacity
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

BASE_URL = os.environ.get("PRODUCT_E2E_BASE_URL", "http://127.0.0.1:8127")
pytestmark = pytest.mark.skipif(os.environ.get("PRODUCT_E2E_CONVERSATION_DISPOSABLE") != "1",
    reason="Requires explicitly reserved, self-managed disposable conversation browser lifecycles")


@contextmanager
def journey(tmp_path, name, *, fake_sdk=False, plan_context="original", registry_path=None,
            expected_final_draft_types=()):
    assert plan_context in {"original", "accepted_child"}
    if plan_context == "accepted_child":
        name += "_accepted_child"
    combined = name.startswith("combined_chat")
    root = Path(os.environ.get("PRODUCT_E2E_CONVERSATION_EVIDENCE_DIR", tmp_path))
    evidence = root / name
    evidence.mkdir(parents=True, exist_ok=False)
    run_id = os.environ.get("PRODUCT_E2E_RUN_ID", "conversation_browser") + "_" + name
    http, turns, errors, failures, draft_requests = [], [], [], [], []
    # Keep complete state artifacts under this run while bounding Windows paths.
    state_root = root.parent / "_state" / hashlib.sha256(name.encode()).hexdigest()[:12]
    with conversation_server(evidence, base_url=BASE_URL, run_id=run_id, fake_sdk=fake_sdk,
                             state_root=state_root, registry_path=registry_path) as server:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True,
                **({"executable_path":str(CHROME)} if CHROME.is_file() else {}))
            context = browser.new_context(viewport={"width":1440,"height":900})
            page = context.new_page()
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.on("console", lambda message: errors.append(message.text) if message.type == "error" else None)
            page.on("requestfailed", lambda request: failures.append({"url":request.url,"failure":request.failure}))

            def record_draft_request(request):
                if request.method != "POST" or not request.url.endswith(("/draft/operations", "/draft/impact-preview", "/preview")):
                    return
                # Typed draft endpoint bodies contain no credentials; retain
                # starts even when their response is missing or times out.
                draft_requests.append({"method":request.method, "url":request.url,
                    "observed_at":datetime.now(UTC).isoformat(), "request_body_utf8":request.post_data,
                    "request_body_sha256":hashlib.sha256(request.post_data.encode()).hexdigest()})
                _write_json(evidence / "draft_request_starts.json", draft_requests)

            page.on("request", record_draft_request)

            def record(response, raw=None, *, complete=True):
                if not urlsplit(response.url).path.startswith("/api/"):
                    return
                if response.url.endswith("/copilot/messages") and raw is None:
                    return
                row = _http_record(response, response_body=raw)
                row["response_capture_complete"] = complete
                # Allocate only after body reading, which can dispatch nested responses.
                name = f"http_{len(http)+1:03d}.json"
                _write_json(evidence / name, row)
                http.append({"file":name,"method":row["method"],"url":row["url"],
                    "status":row["response_status_code"],"response_capture_complete":complete})

            page.on("response", record)
            # Passive tee, including received prefix if the real stream is aborted.
            page.add_init_script("""(() => {
              const originalFetch = window.fetch.bind(window);
              window.__conversationStreams = [];
              window.__conversationControlClicks = [];
              document.addEventListener('click',event=>{
                const button=event.target.closest?.('button');
                if(button) window.__conversationControlClicks.push({trusted:event.isTrusted,
                  id:button.id,label:button.textContent.trim(),aria_label:button.getAttribute('aria-label'),
                  action_group:button.parentElement.getAttribute('aria-label'),at:Date.now()});
              },true);
              window.fetch = async (...args) => {
                const response = await originalFetch(...args);
                const input = args[0], url = typeof input === 'string' ? input : input.url;
                if (url.endsWith('/copilot/messages')) {
                  const row = {raw:'',done:false,error:null};
                  window.__conversationStreams.push(row);
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

            def send(message, *, expected="turn.completed", action=None):
                index = page.evaluate("window.__conversationStreams.length")
                if action is None:
                    page.locator("#copilot-input").fill(message)
                with page.expect_response(lambda response: response.url.endswith("/copilot/messages"), timeout=60_000) as incoming:
                    (action or page.locator("#copilot-submit")).click()
                response = incoming.value
                assert response.status == 200
                page.wait_for_function("i => window.__conversationStreams[i]?.done === true", arg=index, timeout=65_000)
                stream = page.evaluate("i => window.__conversationStreams[i]", index)
                record(response, stream["raw"], complete=stream["error"] is None)
                events = _sse_events(stream["raw"])
                assert stream["error"] is None and events[-1]["type"] == expected, events
                row = {"input":response.request.post_data_json,"output":events[-1],"events":events}
                turns.append(row)
                _write_json(evidence / "exact_chat_io.json", turns)
                if expected == "turn.completed":
                    payload = events[-1]["payload"]
                    turn = payload["turn"]
                    assert turn["user_message"] == message
                    expect(page.locator("#copilot-lifecycle")).to_have_attribute("data-state", turn["interpretation"]["state"])
                    expect(page.locator(".message-user").last.locator("p").first).to_have_text(message)
                    expect(page.locator(".message-assistant").last.locator("p").first).to_have_text(turn["assistant_message"])
                    return payload
                return events[-1]

            def reset():
                page.locator("#transcript-settings-button").click()
                page.once("dialog", lambda dialog: dialog.accept())
                with page.expect_response(lambda response: response.request.method == "GET" and response.url.endswith("/conversation")) as restored:
                    with page.expect_response(lambda response: response.request.method == "DELETE" and response.url.endswith("/conversation")) as deleted:
                        page.locator("#delete-current-conversation").click()
                assert deleted.value.status == 200
                restoration = _http_record(restored.value)
                _write_json(evidence / "reset_restore_http.json", restoration)
                assert restored.value.status == 200
                payload = restored.value.json()
                removed = deleted.value.json()["session"]
                assert removed["conversation_id"] is None
                assert payload["conversation"]["conversation_id"] == payload["session"]["conversation_id"]
                assert all(payload["conversation"][key] == payload["session"][key] == removed[key]
                    for key in ("session_id", "trip_id", "run_id"))
                assert payload["conversation"]["turns"] == []
                assert {key:value for key,value in payload["session"].items() if key != "conversation_id"} == {
                    key:value for key,value in removed.items() if key != "conversation_id"}
                expect(page.locator("#copilot-lifecycle")).to_have_attribute("data-state","idle")
                expect(page.locator(".message-user")).to_have_count(0)
                expect(page.locator("#copilot-submit")).to_be_enabled()
                return deleted.value.json()

            with _preserve_browser_evidence(context, page, evidence,
                {"console_errors":errors,"request_failures":failures}):
                try:
                    page.goto(f"{BASE_URL}/app/itinerary",wait_until="domcontentloaded")
                    initial = _map_ack(page)
                    if plan_context == "accepted_child":
                        initial = _accepted_baseline(page, evidence, initial, http, server)
                    _capture_ready(page,evidence / "initial.png")
                    control_http_start = len(http)
                    page.locator("#open-copilot").click()
                    yield SimpleNamespace(page=page, evidence=evidence, server=server, http=http, turns=turns,
                        errors=errors, failures=failures, initial=initial, send=send, reset=reset, record=record,
                        scenario=name, plan_context=plan_context, control_http_start=control_http_start)
                    if plan_context == "accepted_child":
                        restored = json.loads((evidence / "accepted_setup_restored_http.json").read_text(encoding="utf-8"))
                        sid = json.loads(restored["response_body_utf8"])["session"]["session_id"]
                        final = observe_context(server,sid,evidence,"control_final")
                        final_map = _map_ack(page) if combined else initial
                        assert final["accepted_plan_id"] == final_map["active_plan_id"]
                        assert final["accepted_plan_content_hash"] == final_map["active_plan_content_hash"]
                        assert tuple(row["type"] for row in final["draft_operations"]) == expected_final_draft_types
                    if name != "clarification_reload" and not combined:
                        assert not [row for row in http[control_http_start:] if row["method"] == "POST"
                            and row["url"].endswith(("/draft/operations", "/preview", "/accept", "/keep-original", "/rollback"))]
                    assert len({row["file"] for row in http}) == len(http)
                    assert not errors
                    unexpected = [row for row in failures if not (fake_sdk
                        and row["url"].endswith("/copilot/messages") and "ERR_ABORTED" in str(row["failure"]))]
                    assert not unexpected, unexpected
                    assert not [row for row in http if row["status"] >= 400]
                finally:
                    _write_json(evidence / "control_clicks.json",page.evaluate("window.__conversationControlClicks || []"))
                    _write_json(evidence / "http_index.json",http)
                    _write_json(evidence / "console_errors.json",errors)
                    _write_json(evidence / "network_failures.json",failures)
                    _write_json(evidence / "provider_scope.json", {"live_provider_call_count":0,
                        "mode":"actual_openai_adapter_fake_sdk_transport" if fake_sdk else "deterministic",
                        "fake_sdk_attempt_count":len(server.sdk.calls) if fake_sdk else 0,
                        "llm_input_artifact":"fake_sdk_io.json" if fake_sdk else None,
                        "llm_output_artifact":"fake_sdk_io.json" if fake_sdk else None,
                        "real_chat_endpoint":True,"internal_api_mocked":False,"live_openai_connectivity_proven":False})
            # Closing the context may complete one final response callback. Seal
            # the index only after that boundary so it inventories every file.
            _write_json(evidence / "http_index.json",http)
            browser.close()
    guard = json.loads((evidence / "provider_isolation.json").read_text(encoding="utf-8"))
    assert guard["finalized"] and guard["sent_provider_calls"] == 0 and guard["blocked_attempts"] == []


def _accepted_baseline(page, evidence, original, http, server):
    """Four control journeys share the same real Replace/Preview/Accept/reload setup."""
    bixby, pfeiffer = "bixby_creek_bridge_viewpoint", "pfeiffer_big_sur_state_park"
    _capture_ready(page, evidence / "accepted_setup_original.png")
    with page.expect_response(lambda response: response.url.endswith("/draft/impact-preview")):
        page.locator('.day-card[data-day="5"]').click()
    _map_ack(page)
    page.locator("#map-edit-button").click()
    page.locator("#inspect-edit-stop").click()
    expect(page.locator("#map-edit-dialog")).to_be_visible()
    shortcuts = page.locator(".legacy-edit-controls")
    if shortcuts.get_attribute("open") is None:
        shortcuts.locator("summary").click()
    expect(page.locator("#registered-candidate")).to_be_enabled()
    expect(page.locator(f'#registered-candidate option[value="{pfeiffer}"]')).to_have_js_property("disabled", False)
    with page.expect_response(lambda response: response.url.endswith("/draft/impact-preview")) as candidate_impact:
        page.locator("#registered-candidate").select_option(pfeiffer)
    assert candidate_impact.value.status == 200
    candidate_revision = candidate_impact.value.request.post_data_json["expected_revision"]
    expect(page.locator("#registered-candidate")).to_have_value(pfeiffer)
    expect(page.locator("#draft-replace")).to_be_enabled()
    # Stage and impact are sequential real requests; do not share the default
    # 30-second budget. Bind the impact response to this exact staged revision.
    with page.expect_response(lambda response: response.url.endswith("/draft/impact-preview")
            and response.request.post_data_json == {"expected_revision":candidate_revision + 1}, timeout=90_000) as staged_impact:
        with page.expect_response(lambda response: response.url.endswith("/draft/operations")) as staged:
            page.locator("#draft-replace").click()
    assert staged_impact.value.status == 200
    assert staged_impact.value.json()["session_revision"] == candidate_revision + 1
    stage_io = _http_record(staged.value)
    _write_json(evidence / "accepted_setup_stage_http.json", stage_io)
    assert staged.value.status == 200
    request = staged.value.request.post_data_json
    assert request["type"] == "replace_nearby" and request["target"] == bixby
    assert request["parameters"] == {"candidate_id": pfeiffer}
    assert _map_ack(page)["active_plan_id"] == original["active_plan_id"]
    page.locator("#close-map-edit").click()
    expect(page.locator("#inline-preview")).to_be_enabled()
    with page.expect_response(lambda response: response.url.endswith("/preview"), timeout=90_000) as previewed:
        page.locator("#inline-preview").click()
    preview_io = _http_record(previewed.value)
    _write_json(evidence / "accepted_setup_preview_http.json", preview_io)
    assert previewed.value.status == 200
    proposal = previewed.value.json()["proposal"]
    execution = proposal["runtime_execution"]
    child, certificate, diff = execution["child_plan"], execution["evaluation_certificate"], execution["plan_diff"]
    assert execution["evaluation_executed"] is True and execution["evaluator"] == "PlanEvaluator.evaluate_final_plan"
    assert proposal["state"] == certificate["comparison_eligibility"] == "eligible"
    assert child["plan_id"] == proposal["child_plan_id"] == certificate["plan_id"] == diff["child_plan_id"]
    assert child["content_hash"] == proposal["child_plan_content_hash"] == certificate["plan_content_hash"]
    assert child["parent_plan_id"] == original["active_plan_id"] == diff["parent_plan_id"]
    expect(page.locator(".map-stage")).to_have_attribute("data-preview-state", "PREVIEW_RENDERED", timeout=45_000)
    preview_ack = _map_ack(page)
    assert preview_ack["active_plan_id"] == child["plan_id"]
    assert preview_ack["active_plan_content_hash"] == child["content_hash"]
    assert preview_ack["rendered_geometry_hash"] == _independent_geometry_hash(page, proposal["geography_plan"])
    assert set(preview_ack["rendered_marker_ids"]) == (set(original["rendered_marker_ids"]) - {bixby}) | {pfeiffer}
    _capture_ready(page, evidence / "accepted_setup_preview.png")
    with page.expect_response(lambda response: response.url.endswith("/accept"), timeout=60_000) as accepted:
        page.locator("#accept-repair").click()
        expect(page.get_by_role("dialog", name="Confirm trip decision", exact=True)).to_be_visible()
        expect(page.locator("#acceptance-message")).to_have_text('Accept this independently evaluated repair as the durable trip plan?')
        page.get_by_role("dialog", name="Confirm trip decision", exact=True).get_by_role("button", name='Accept repair', exact=True).click()
    accept_io = _http_record(accepted.value)
    _write_json(evidence / "accepted_setup_accept_http.json", accept_io)
    trusted_setup_clicks = page.evaluate("window.__conversationControlClicks")
    _write_json(evidence / "accepted_setup_clicks.json", trusted_setup_clicks)
    assert accepted.value.status == 200
    decision = accepted.value.json()["result"]["decision"]
    assert decision["selected_plan_id"] == child["plan_id"] and decision["selected_plan_content_hash"] == child["content_hash"]
    sid = accepted.value.json()["session"]["session_id"]
    with page.expect_response(lambda response: response.request.method == "GET" and response.url.endswith(f"/api/sessions/{sid}")) as restored:
        page.reload(wait_until="domcontentloaded")
    _write_json(evidence / "accepted_setup_restored_http.json", _http_record(restored.value))
    assert restored.value.status == 200
    saved = restored.value.json()
    assert saved["session"]["accepted_plan_id"] == child["plan_id"] and saved["session"]["draft"] == [] and saved["session"]["proposal"] is None
    assert saved["workspace"]["accepted_plan_pointer"]["decision_id"] == decision["decision_id"]
    assert saved["workspace"]["alternative_evidence"][child["plan_id"]]["certificate"] == certificate
    initial = _map_ack(page)
    for key in ("active_plan_id", "active_plan_content_hash", "rendered_map_plan_id", "rendered_geometry_hash", "rendered_marker_ids"):
        assert initial[key] == preview_ack[key]
    chat_count = sum(row["url"].endswith("/copilot/messages") for row in http)
    fake_attempts = len(server.sdk.calls) if server.sdk else 0
    assert chat_count == fake_attempts == 0
    observe_context(server,sid,evidence,"accepted_baseline")
    _write_json(evidence / "accepted_baseline_artifacts.json", {"schema_version": "conversation-accepted-baseline-v1",
        "context": "accepted_child", "original": original, "preview_map": preview_ack, "accepted_map": initial,
        "http_boundary_files": [row["file"] for row in http], "chat_request_count": chat_count,
        "fake_sdk_attempt_count": fake_attempts, "assertions_passed": True})
    return initial


def test_clarification_survives_reload_and_short_answer_renders_exact_child(tmp_path):
    with journey(tmp_path,"clarification_reload") as case:
        page = case.page
        first = case.send("Replace Bixby Creek Bridge.")
        assert first["turn"]["interpretation"]["state"] == "clarification_required"
        sid = first["session"]["session_id"]
        before = observe_context(case.server,sid,case.evidence,"pending_before_reload")
        pending = before["conversation_memory"]["pending_clarification"]
        assert pending["target_stop_id"] == "bixby_creek_bridge_viewpoint"
        assert list(pending["missing_slots"]) == ["candidate_id"]
        assert first["session"]["draft"] == [] and _map_ack(page) == case.initial
        page.reload(wait_until="domcontentloaded")
        assert _map_ack(page) == case.initial
        page.locator("#open-copilot").click()
        expect(page.locator("#conversation")).to_contain_text(first["turn"]["assistant_message"])
        restored = observe_context(case.server,sid,case.evidence,"pending_after_reload")
        assert restored["conversation_memory"] == before["conversation_memory"]
        answer = case.send("Pfeiffer Big Sur State Park.")
        intents = answer["turn"]["interpretation"]["intents"]
        assert [(row["type"],row["target_stop_id"],row["candidate_id"]) for row in intents] == [
            ("replace_nearby","bixby_creek_bridge_viewpoint","pfeiffer_big_sur_state_park")]
        assert answer["session"]["draft"] == []
        assert observe_context(case.server,sid,case.evidence,"answered")["conversation_memory"]["pending_clarification"] is None
        actions = page.locator(".copilot-proposal-actions").last
        with page.expect_response(lambda response: response.url.endswith("/draft/operations")) as staged:
            actions.get_by_role("button",name="Add to draft",exact=False).click()
        assert staged.value.status == 200
        assert staged.value.request.post_data_json["parameters"] == {"candidate_id":"pfeiffer_big_sur_state_park"}
        assert _map_ack(page) == case.initial
        with page.expect_response(lambda response: response.url.endswith("/preview"),timeout=90_000) as preview:
            actions.get_by_role("button",name="Preview this repair",exact=False).click()
        assert preview.value.status == 200
        proposal = preview.value.json()["proposal"]
        _write_json(case.evidence / "preview_response.json",preview.value.json())
        assert proposal["state"] == "eligible"
        expect(page.locator(".map-stage")).to_have_attribute("data-preview-state","PREVIEW_RENDERED",timeout=45_000)
        ack = _map_ack(page)
        assert ack["active_plan_id"] == ack["rendered_map_plan_id"] == proposal["child_plan_id"]
        assert ack["active_plan_content_hash"] == proposal["child_plan_content_hash"]
        assert ack["rendered_geometry_hash"] == _independent_geometry_hash(page,proposal["geography_plan"])
        assert "pfeiffer_big_sur_state_park" in ack["rendered_marker_ids"]
        assert "bixby_creek_bridge_viewpoint" not in ack["rendered_marker_ids"]
        page.locator("#close-copilot").click()
        expect(page.locator("#day-list")).to_contain_text("Pfeiffer Big Sur State Park")
        expect(page.locator("#day-list")).not_to_contain_text("Bixby Creek Bridge")
        expect(page.locator("#metric-strip")).to_have_attribute("data-plan-id",proposal["child_plan_id"])
        expect(page.locator("#metric-strip")).to_have_attribute("data-evaluation-id",proposal["certificate_id"])
        _capture_ready(page,case.evidence / "clarified_preview.png")
        _write_json(case.evidence / "trace.json", {"schema_version":"conversation-browser-v1",
            "scenario":"clarification_reload","initial":case.initial,"rendered":ack,
            "request_id":proposal.get("request_id"),"child_plan_id":proposal["child_plan_id"],
            "certificate_id":proposal["certificate_id"],"accepted":False,"assertions_passed":True})


def test_combined_chat_stages_all_edits_and_renders_exact_child(tmp_path):
    _combined_chat_journey(tmp_path)


@pytest.mark.parametrize("plan_context,stage_first", [
    pytest.param("original", False, id="original-one_click"),
    pytest.param("accepted_child", True, id="accepted_child-staged"),
    pytest.param("accepted_child", False, id="accepted_child-one_click"),
])
def test_combined_chat_continuation_and_one_click(tmp_path, plan_context, stage_first):
    _combined_chat_journey(tmp_path, plan_context=plan_context, stage_first=stage_first)


def _combined_chat_journey(tmp_path, *, plan_context="original", stage_first=True, registry_path=None):
    with journey(tmp_path, "combined_chat" + ("" if stage_first else "_one_click"),
                 plan_context=plan_context, registry_path=registry_path) as case:
        page = case.page
        continued = plan_context == "accepted_child"
        message = ("Set Pfeiffer Big Sur State Park duration to 90 minutes; set Stearns Wharf earliest arrival to 10:00."
            if continued else "Replace Bixby Creek Bridge with Pfeiffer Big Sur State Park; set Griffith Observatory duration to 90 minutes.")
        reply = case.send(message)
        sid = reply["session"]["session_id"]
        chat_context = observe_context(case.server, sid, case.evidence, "combined_before_stage")
        assert chat_context["accepted_plan_id"] == case.initial["active_plan_id"]
        assert chat_context["accepted_plan_content_hash"] == case.initial["active_plan_content_hash"]
        assert ("bixby_creek_bridge_viewpoint" in chat_context["allowed_stop_ids"]) is not continued
        assert ("pfeiffer_big_sur_state_park" in chat_context["allowed_stop_ids"]) is continued
        interpretation = reply["turn"]["interpretation"]
        assert interpretation["state"] == "proposal_ready"
        assert [row["type"] for row in interpretation["intents"]] == (
            ["set_stop_duration", "set_stop_time_window"] if continued else ["replace_nearby", "set_stop_duration"])
        assert reply["session"]["draft"] == [] and _map_ack(page) == case.initial
        expect(page.locator(".copilot-proposal-card")).to_have_count(2)
        # Observation only: snapshot the rendered parent immediately before the
        # real preview request, including the one-click stage->preview path.
        page.evaluate("""() => {
          const original = window.fetch.bind(window);
          window.__combinedBeforePreview = null;
          window.fetch = (...args) => {
            const input = args[0], url = typeof input === 'string' ? input : input.url;
            if (url.endsWith('/preview')) {
              const map = document.querySelector('.map-stage');
              window.__combinedBeforePreview = {request:JSON.parse(args[1].body),
                active_plan_id:map.dataset.activePlanId, active_plan_content_hash:map.dataset.activePlanContentHash,
                rendered_map_plan_id:map.dataset.renderedMapPlanId || null, rendered_geometry_hash:map.dataset.renderedGeometryHash || null,
                rendered_marker_ids:map.dataset.renderedMarkerIds ? JSON.parse(map.dataset.renderedMarkerIds) : null, observed_at:Date.now()};
            }
            return original(...args);
          };
        }""")
        actions = page.locator(".copilot-proposal-actions").last
        if stage_first:
            with page.expect_response(lambda response: response.url.endswith("/draft/operations")) as staged:
                actions.get_by_role("button", name="Add all to draft", exact=True).click()
            staged_ack = _map_ack(page)
            assert staged_ack == case.initial
            with page.expect_response(lambda response: response.url.endswith("/preview"), timeout=90_000) as preview:
                actions.get_by_role("button", name="Preview combined repair", exact=True).click()
        else:
            with page.expect_response(lambda response: response.url.endswith("/preview"), timeout=90_000) as preview:
                with page.expect_response(lambda response: response.url.endswith("/draft/operations")) as staged:
                    actions.get_by_role("button", name="Preview combined repair", exact=True).click()
        assert staged.value.status == 200, staged.value.text()
        stage_input, stage_output = staged.value.request.post_data_json, staged.value.json()
        assert stage_output["session"]["revision"] == stage_input["expected_revision"] + 1
        assert len(stage_output["session"]["draft"]) == len(stage_input["operations"]) == 2
        _write_json(case.evidence / "staging_io.json", {"input":stage_input,"output":stage_output})
        staged_ack = page.evaluate("window.__combinedBeforePreview")
        assert all(staged_ack[key] == case.initial[key] for key in ("active_plan_id", "active_plan_content_hash"))
        assert all(staged_ack[key] in (None, case.initial[key]) for key in (
            "rendered_map_plan_id", "rendered_geometry_hash", "rendered_marker_ids"))
        _write_json(case.evidence / "before_preview_observation.json", staged_ack)
        assert preview.value.status == 200, preview.value.text()
        payload = preview.value.json()
        proposal = payload["proposal"]
        assert proposal["state"] == "eligible" and proposal["execution_mode"] == "atomic_combined_independent_evaluation"
        _write_json(case.evidence / "preview_response.json", payload)
        expect(page.locator(".map-stage")).to_have_attribute("data-preview-state", "PREVIEW_RENDERED", timeout=45_000)
        rendered = _map_ack(page)
        assert rendered["active_plan_id"] == rendered["rendered_map_plan_id"] == proposal["child_plan_id"]
        assert rendered["active_plan_content_hash"] == proposal["child_plan_content_hash"]
        assert rendered["rendered_geometry_hash"] == _independent_geometry_hash(page, proposal["geography_plan"])
        # The acknowledgement digest includes plan identity, not just coordinates.
        assert rendered["rendered_geometry_hash"] != case.initial["rendered_geometry_hash"]
        if continued:
            accepted_setup = json.loads((case.evidence / "accepted_setup_accept_http.json").read_text(encoding="utf-8"))
            parent = next(plan for plan in json.loads(accepted_setup["response_body_utf8"])["workspace"]["geography"]["plans"]
                if plan["plan_id"] == case.initial["active_plan_id"])
            for collection, keys in (
                ("route_path", ("occurrence_index", "node_id", "selected_stop", "selected_sequence_index", "route_anchor",
                    "arrival_day", "departure_day", "coordinate_source")),
                ("validated_legs", ("leg_index", "day", "from_day", "to_day", "cross_day", "travel_mode", "origin_id",
                    "destination_id", "validation_status", "route_leg_id", "road_validated", "fallback_used", "geometry_source", "distance_m", "duration_s")),
                ("gaps", ()),
            ):
                assert [{"geometry":feature["geometry"], **{key:feature["properties"].get(key) for key in keys}}
                    for feature in parent[collection]["features"]] == [
                    {"geometry":feature["geometry"], **{key:feature["properties"].get(key) for key in keys}}
                    for feature in proposal["geography_plan"][collection]["features"]]
        expected_markers = (set(case.initial["rendered_marker_ids"]) if continued else
            (set(case.initial["rendered_marker_ids"]) - {"bixby_creek_bridge_viewpoint"}) | {"pfeiffer_big_sur_state_park"})
        assert set(rendered["rendered_marker_ids"]) == expected_markers
        child = proposal["runtime_execution"]["child_plan"]
        duration_operation = stage_input["operations"][0 if continued else 1]
        duration_stop = next(stop for stop in child["selected_stops"] if stop["stop_id"] == duration_operation["target"])
        assert duration_stop["visit_duration_minutes"] == 90
        assert duration_stop["duration_constraint"] == duration_operation["parameters"]["duration"]
        if continued:
            assert next(stop for stop in child["selected_stops"] if stop["stop_id"] == "stearns_wharf")["time_window_constraint"] == {
                "schema_version":"stop-time-window-constraint-v1", "earliest_arrival":"10:00", "latest_departure":None,
                "early_arrival_policy":"wait_until_earliest_arrival", "latest_departure_semantics":"departure_after_visit"}
        added = actions.get_by_role("button", name="All changes added to draft", exact=True)
        expect(added).to_be_visible()
        expect(added).to_be_disabled()
        _write_json(case.evidence / "combined_added_disabled.json", {"label":added.inner_text(),
            "disabled":added.is_disabled(), "visible":added.is_visible(), "session":payload["session"],
            "note":actions.locator(".copilot-proposal-note").inner_text()})
        page.locator("#close-copilot").click()
        expect(page.locator("#day-list")).to_contain_text("Pfeiffer Big Sur State Park")
        expect(page.locator("#day-list")).not_to_contain_text("Bixby Creek Bridge")
        expect(page.locator("#metric-strip")).to_have_attribute("data-plan-id", proposal["child_plan_id"])
        expect(page.locator("#metric-strip")).to_have_attribute("data-evaluation-id", proposal["certificate_id"])
        _capture_ready(page, case.evidence / "combined_preview.png")
        page.locator('[data-route="/app/compare"]').first.click()
        expect(page.locator('[data-compare-map-pane="option"] [data-compare-map-canvas]')).to_have_attribute("data-rendered-map-plan-id", proposal["child_plan_id"], timeout=30_000)
        _capture_ready(page, case.evidence / "combined_compare.png")
        page.locator("#compare-evidence").click()
        panel = page.locator('[data-evidence-plan-id]')
        expect(panel).to_have_attribute("data-evidence-plan-id", proposal["child_plan_id"])
        expect(panel).to_have_attribute("data-plan-content-hash", proposal["child_plan_content_hash"])
        _write_json(case.evidence / "evidence_panel.json", {"scope":panel.evaluate("e => ({...e.dataset})"),"text":panel.inner_text()})
        page.locator("#evidence-back").click()
        with page.expect_response(lambda response: response.url.endswith("/accept"), timeout=60_000) as accepted:
            page.locator("#accept-repair").click()
            expect(page.get_by_role("dialog", name="Confirm trip decision", exact=True)).to_be_visible()
            expect(page.locator("#acceptance-message")).to_have_text('Accept this independently evaluated repair as the durable trip plan?')
            page.get_by_role("dialog", name="Confirm trip decision", exact=True).get_by_role("button", name='Accept repair', exact=True).click()
        assert accepted.value.status == 200, accepted.value.text()
        _write_json(case.evidence / "acceptance_response.json", accepted.value.json())
        clicks = page.evaluate("window.__conversationControlClicks")
        assert sum(row["trusted"] is True and row["label"] == "Add all to draft" for row in clicks) == int(stage_first)
        assert sum(row["trusted"] is True and row["label"] == "Preview combined repair" for row in clicks) == 1
        _write_json(case.evidence / "combined_controls_before_reload.json", {
            "schema_version":"combined-chat-controls-v2", "clicks":clicks, "stage_first":stage_first,
            "preview_plan_id":rendered["active_plan_id"],
            "accepted_plan_id":accepted.value.json()["session"]["accepted_plan_id"],
        })
        page.reload(wait_until="domcontentloaded")
        reloaded = _map_ack(page)
        assert reloaded["preview_state"] == "PARENT_RENDERED"
        assert all(reloaded[field] == rendered[field] for field in ("active_plan_id", "active_plan_content_hash", "rendered_map_plan_id", "rendered_geometry_hash", "rendered_marker_ids"))
        _capture_ready(page, case.evidence / "combined_accepted_reload.png")
        _write_json(case.evidence / "trace.json", {"schema_version":"conversation-browser-v1", "scenario":case.scenario,
            "context":plan_context, "stage_first":stage_first,
            "initial":case.initial,"staged":staged_ack,"rendered":rendered,"accepted_map":reloaded,
            "child_plan_id":proposal["child_plan_id"],"accepted":True,"assertions_passed":True})


def test_preference_survives_seven_turns_corrects_and_resets_without_plan_change(tmp_path):
    _preference_journey(tmp_path)


def _preference_journey(tmp_path, *, plan_context="original"):
    with journey(tmp_path,"preference_memory",plan_context=plan_context) as case:
        first = case.send("I prefer less driving.")
        sid = first["session"]["session_id"]
        for _ in range(7):
            case.send("Review the safer repair.")
        memory = observe_context(case.server,sid,case.evidence,"after_seven_turns")["conversation_memory"]
        assert list(memory["proposed_preferences"]) == [{"preference":"reduce_driving",
            "source_turn_id":first["turn"]["turn_id"],"disposition":"proposed_only"}]
        remembered = case.send("What preferences do you remember?")
        assert "reduce driving" in remembered["turn"]["assistant_message"]
        assert "not accepted" in remembered["turn"]["assistant_message"]
        corrected = case.send("Actually, I prefer preserving the experience instead.")
        memory = observe_context(case.server,sid,case.evidence,"corrected")["conversation_memory"]
        assert list(memory["proposed_preferences"]) == [{"preference":"preserve_experience",
            "source_turn_id":corrected["turn"]["turn_id"],"disposition":"proposed_only"}]
        _capture_ready(case.page,case.evidence / "remembered_and_corrected.png")
        assert corrected["session"]["draft"] == [] and _map_ack(case.page) == case.initial
        deleted = case.reset()
        assert deleted["session"]["draft"] == []
        reset = observe_context(case.server,sid,case.evidence,"reset")["conversation_memory"]
        assert list(reset["proposed_preferences"]) == [] and reset["pending_clarification"] is None
        fresh = case.send("What preferences do you remember?")
        assert "reduce driving" not in fresh["turn"]["assistant_message"]
        assert "preserve experience" not in fresh["turn"]["assistant_message"]
        assert _map_ack(case.page) == case.initial
        # This owns an isolated repository; the global-local delete must never
        # be exercised against the user's running service or conversation store.
        before_delete_all = observe_context(case.server,sid,case.evidence,"before_delete_all")
        before_calls = len(case.turns)
        before_streams = case.page.evaluate("window.__conversationStreams.length")
        case.page.locator("#transcript-settings-button").click()
        case.page.once("dialog",lambda dialog:dialog.accept())
        with case.page.expect_response(lambda response:response.request.method == "GET"
                and response.url.endswith("/conversation")) as restored_all:
            with case.page.expect_response(lambda response:response.request.method == "DELETE"
                    and response.url.endswith("/api/conversations")) as deleted_all:
                case.page.locator("#delete-all-conversations").click()
        assert deleted_all.value.status == 200
        assert restored_all.value.status == 200
        _write_json(case.evidence / "delete_all_restore_http.json", _http_record(restored_all.value))
        delete_payload = deleted_all.value.json()
        assert delete_payload["deleted_count"] >= 1
        assert deleted_all.value.request.post_data_json["confirmation"] == "delete_all_conversations"
        expect(case.page.locator("#copilot-lifecycle")).to_have_attribute("data-state","idle")
        expect(case.page.locator(".message-user")).to_have_count(0)
        after_delete_all = observe_context(case.server,sid,case.evidence,"after_delete_all")
        assert after_delete_all["accepted_plan_id"] == before_delete_all["accepted_plan_id"]
        assert after_delete_all["draft_operations"] == []
        assert after_delete_all["conversation_memory"]["proposed_preferences"] == []
        assert len(case.turns) == before_calls and _map_ack(case.page) == case.initial
        after_streams = case.page.evaluate("window.__conversationStreams.length")
        assert after_streams == before_streams
        _write_json(case.evidence / "control_closure.json", {"schema_version":"conversation-control-closure-v1",
            "scenario":case.scenario,"context":case.plan_context,"delete_all_request":deleted_all.value.request.post_data_json,
            "delete_all_response":delete_payload,"before":before_delete_all,"after":after_delete_all,
            "raw_clicks":case.page.evaluate("window.__conversationControlClicks"),"visible_turn_count":0,
            "new_chat_request_count":0,"accepted_map":_map_ack(case.page),"assertions_passed":True,
            "streams_before":before_streams,"streams_after":after_streams,
            "scope":"Every conversation in this disposable local repository; not current-session-only deletion"})
        assert not [row for row in case.http[case.control_http_start:] if row["url"].endswith(("/draft/operations","/preview","/accept"))]
        _capture_ready(case.page,case.evidence / "reset.png")
        _write_json(case.evidence / "trace.json", {"schema_version":"conversation-browser-v1",
            "scenario":case.scenario,"context":case.plan_context,"filler_turn_count":7,"chat_turn_count":len(case.turns),
            "initial":case.initial,"final":_map_ack(case.page),"assertions_passed":True})


@pytest.mark.parametrize("kind",["turns","bytes"])
def test_capacity_blocks_real_chat_before_dispatch_and_settings_reset_recovers(tmp_path,kind):
    _capacity_journey(tmp_path,kind)


def _capacity_journey(tmp_path,kind,*,plan_context="original"):
    with journey(tmp_path,"capacity_" + kind,plan_context=plan_context) as case:
        seed = case.send("Review the safer repair.")
        sid = seed["session"]["session_id"]
        full = seed_capacity(case.server,sid,kind,case.evidence)
        repository = case.server.service.conversations
        before = repository.dispatch_history(full.conversation_id,sid)
        blocked = case.send("Review the safer repair.",expected="turn.failed")
        expected = "conversation_turn_limit_reached" if kind == "turns" else "conversation_size_limit_reached"
        assert blocked["code"] == expected and blocked["status"] == 422 and blocked["retryable"] is False
        expect(case.page.locator("#copilot-lifecycle")).to_have_attribute("data-state","blocked")
        last = case.page.locator(".message-assistant").last
        expect(last).to_contain_text("blocked before any new provider attempt")
        expect(last.get_by_role("button",name="Retry",exact=True)).to_have_count(0)
        expect(last.get_by_role("button",name="Conversation settings",exact=True)).to_be_visible()
        before_action_requests = len(case.turns)
        before_action_streams = case.page.evaluate("window.__conversationStreams.length")
        last.get_by_role("button",name="Edit message",exact=True).click()
        expect(case.page.locator("#copilot-input")).to_have_value("Review the safer repair.")
        last.get_by_role("button",name="Conversation settings",exact=True).click()
        expect(case.page.locator("#transcript-settings-dialog")).to_be_visible()
        case.page.locator("#close-transcript-settings").click()
        expect(case.page.locator("#transcript-settings-dialog")).to_be_hidden()
        assert len(case.turns) == before_action_requests
        assert case.page.evaluate("window.__conversationStreams.length") == before_action_streams
        _write_json(case.evidence / "control_closure.json", {"schema_version":"conversation-control-closure-v1",
            "scenario":case.scenario,"context":case.plan_context,"edit_message_composer":case.page.locator("#copilot-input").input_value(),
            "settings_opened":True,"settings_closed":True,"new_chat_request_count":0,
            "streams_before":before_action_streams,"streams_after":case.page.evaluate("window.__conversationStreams.length"),
            "raw_clicks":case.page.evaluate("window.__conversationControlClicks"),"accepted_map":_map_ack(case.page),
            "assertions_passed":True})
        assert repository.dispatch_history(full.conversation_id,sid) == before
        assert repository.get(full.conversation_id,sid) == full
        assert _map_ack(case.page) == case.initial
        _capture_ready(case.page,case.evidence / "capacity_blocked.png")
        case.reset()
        fresh = case.send("Review the safer repair.")
        assert fresh["session"]["draft"] == [] and _map_ack(case.page) == case.initial
        new_conversation = case.server.service.read_conversation(case.server.service.workspace.get(sid))
        assert new_conversation.conversation_id != full.conversation_id and len(new_conversation.turns) == 1
        _capture_ready(case.page,case.evidence / "capacity_reset.png")
        _write_json(case.evidence / "trace.json", {"schema_version":"conversation-browser-v1",
            "scenario":case.scenario,"context":case.plan_context,"blocked_code":expected,"attempts_before":len(before),
            "blocked_attempt_delta":0,"initial":case.initial,"final":_map_ack(case.page),
            "old_conversation_id":full.conversation_id,"new_conversation_id":new_conversation.conversation_id,
            "assertions_passed":True})


def test_fake_sdk_stop_waiting_reload_and_retry_reconciles_one_attempt(tmp_path):
    _fake_sdk_journey(tmp_path)


def _fake_sdk_journey(tmp_path,*,plan_context="original"):
    with journey(tmp_path,"fake_sdk_cancel",fake_sdk=True,plan_context=plan_context) as case:
        page, sdk = case.page, case.server.sdk
        message = ("Suggest a safe itinerary change for the accepted plan."
            if plan_context == "accepted_child" else
            "Replace Bixby Creek Bridge with Pfeiffer Big Sur State Park.")
        page.locator("#copilot-input").fill(message)
        with page.expect_response(lambda response: response.url.endswith("/copilot/messages")) as received:
            page.locator("#copilot-submit").click()
        response = received.value
        assert response.status == 200 and sdk.entered.wait(10)
        envelope = response.request.post_data_json
        sid = urlsplit(response.url).path.split("/")[3]
        expect(page.locator("#copilot-submit")).to_have_attribute("aria-label","Stop waiting for Copilot")
        page.locator("#copilot-submit").click()
        expect(page.locator("#copilot-lifecycle")).to_have_attribute("data-state","cancelled")
        expect(page.locator(".message-assistant").last).to_contain_text("Provider processing or usage may already have occurred")
        page.wait_for_function("() => window.__conversationStreams[0]?.done === true",timeout=15_000)
        interrupted = page.evaluate("window.__conversationStreams[0]")
        case.record(response,interrupted["raw"],complete=False)
        _write_json(case.evidence / "interrupted_chat_io.json", {"input":envelope,"received_stream":interrupted,
            "scope":"Exact received prefix only. Aborted stream is not presented as a complete provider response."})
        sdk.release.set()
        assert sdk.finished.wait(10)
        assert len(sdk.calls) == 1
        session = case.server.service.workspace.get(sid)
        conversation = case.server.service.read_conversation(session)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            journal = case.server.service.conversations.dispatch_record(conversation.conversation_id,sid,
                envelope["client_message_id"],message)
            if journal and journal["outcome"] in {"unknown","completed","failed"}:
                break
            time.sleep(0.02)
        assert journal["attempt_count"] == 1 and journal["attempt_budget"] == 1
        assert journal["outcome"] == "unknown", "Cancellation recovery must not claim a completed provider answer"
        _write_json(case.evidence / "dispatch_after_cancel.json",journal)
        assert _map_ack(page) == case.initial
        _capture_ready(page,case.evidence / "cancelled.png")
        page.reload(wait_until="domcontentloaded")
        assert _map_ack(page) == case.initial
        page.locator("#open-copilot").click()
        before_edit_attempts = len(sdk.calls)
        before_edit_streams = page.evaluate("window.__conversationStreams.length")
        page.locator(".copilot-recovery-actions").last.get_by_role("button",name="Edit message",exact=True).click()
        expect(page.locator("#copilot-input")).to_have_value(message)
        assert len(sdk.calls) == before_edit_attempts == 1
        assert page.evaluate("window.__conversationStreams.length") == before_edit_streams
        _write_json(case.evidence / "control_closure.json", {"schema_version":"conversation-control-closure-v1",
            "scenario":case.scenario,"context":case.plan_context,"edit_message_composer":page.locator("#copilot-input").input_value(),
            "new_chat_request_count":0,"fake_attempts_before":before_edit_attempts,"fake_attempts_after":len(sdk.calls),
            "streams_before":before_edit_streams,"streams_after":page.evaluate("window.__conversationStreams.length"),
            "raw_clicks":page.evaluate("window.__conversationControlClicks"),"accepted_map":_map_ack(page),"assertions_passed":True})
        retry = page.locator(".copilot-recovery-actions").last.get_by_role("button",name="Retry",exact=True)
        expect(retry).to_be_visible()
        reconciled = case.send(message,expected="turn.failed",action=retry)
        assert reconciled["code"] == "copilot_outcome_unknown" and reconciled["retryable"] is False
        assert case.turns[-1]["input"]["client_message_id"] == envelope["client_message_id"]
        assert len(sdk.calls) == 1
        after = case.server.service.conversations.dispatch_record(conversation.conversation_id,sid,
            envelope["client_message_id"],message)
        assert after == journal
        _write_json(case.evidence / "dispatch_after_retry.json",after)
        assert _map_ack(page) == case.initial
        assert not [row for row in case.http[case.control_http_start:] if row["url"].endswith(("/draft/operations","/preview","/accept"))]
        _capture_ready(page,case.evidence / "reconciled.png")
        _write_json(case.evidence / "trace.json", {"schema_version":"conversation-browser-v1",
            "scenario":case.scenario,"context":case.plan_context,"client_message_id":envelope["client_message_id"],
            "fake_sdk_attempt_count":1,"live_provider_call_count":0,"outcome":"unknown",
            "initial":case.initial,"final":_map_ack(page),"assertions_passed":True,
            "completed_at":datetime.now(UTC).isoformat()})


def test_late_selection_preserves_deleted_conversation_generation(tmp_path):
    from test_selection_conversation_generation_live import run_selection_conversation_generation

    root = Path(os.environ.get("PRODUCT_E2E_CONVERSATION_EVIDENCE_DIR", tmp_path))
    name = "selection_conversation_generation"
    run_selection_conversation_generation(root / name,
        os.environ.get("PRODUCT_E2E_RUN_ID", "conversation_browser") + "_" + name,
        state_root=root.parent / "_state" / hashlib.sha256(name.encode()).hexdigest()[:12])


@pytest.mark.parametrize("control", ["delete_all", "capacity_turns", "capacity_bytes", "recovery"])
def test_accepted_child_conversation_controls(tmp_path, control):
    if control == "delete_all":
        _preference_journey(tmp_path,plan_context="accepted_child")
    elif control == "recovery":
        _fake_sdk_journey(tmp_path,plan_context="accepted_child")
    else:
        _capacity_journey(tmp_path,control.removeprefix("capacity_"),plan_context="accepted_child")
