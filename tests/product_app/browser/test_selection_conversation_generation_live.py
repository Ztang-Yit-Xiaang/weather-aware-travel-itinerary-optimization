"""Real browser regression for same-revision conversation deletion vs late selection.

Only delivery timing changes: original fetch responses and real backend state are
never replaced. The conversation suite reuses this disposable lifecycle.
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
from copilot_conversation_fixture import conversation_server
from test_e2e1_live_copilot_chat import (
    CHROME,
    _capture_ready,
    _http_record,
    _map_ack,
    _preserve_browser_evidence,
    _sse_events,
    _write_json,
    expect,
    sync_playwright,
)

ROOT = Path(__file__).resolve().parents[3]
BASE_URL = "http://127.0.0.1:8127"
pytestmark = pytest.mark.skipif(os.environ.get("PRODUCT_E2E_SELECTION_METADATA_DISPOSABLE") != "1",
    reason="Requires explicitly reserved self-managed disposable port 8127")

# The initial conversation GET is delayed before dispatch. Selection delivery is
# delayed after reading its complete actual response, never a substituted body.
TIMING_OBSERVER = r"""(() => {
  const original = window.fetch.bind(window);
  const observed = window.__selectionConversation = {calls:[], clicks:[], streams:[], firstConversation:true, firstSelection:true};
  const now = () => new Date().toISOString();
  document.addEventListener('click', event => {
    const button = event.target.closest?.('button');
    if (button) observed.clicks.push({id:button.id, day:button.dataset.day || null,
      text:button.textContent.trim(), trusted:event.isTrusted, at:now()});
  }, true);
  window.fetch = async (...args) => {
    const url = new URL(typeof args[0] === 'string' ? args[0] : args[0].url, location.href);
    const method = (args[1]?.method || args[0]?.method || 'GET').toUpperCase();
    const firstRead = method === 'GET' && url.pathname.endsWith('/conversation') && observed.firstConversation;
    const selection = method === 'POST' && url.pathname.endsWith('/selection') && observed.firstSelection;
    const relevant = url.pathname.endsWith('/conversation') || selection
      || (method === 'GET' && /^\/api\/sessions\/[^/]+$/.test(url.pathname));
    const row = relevant ? {url:url.href,method,request_body_utf8:args[1]?.body || '',
      requested_at:now(),dispatch_at:null,response_body_utf8:null,response_status:null,
      body_completed_at:null,held_at:null,released_at:null,release_count:0,
      application_json_read_at:null,delivery_settled_at:null} : null;
    if (row) observed.calls.push(row);
    if (firstRead) {
      observed.firstConversation = false; row.held_at = now();
      await new Promise(resolve => {window.__releaseInitialConversation = () => {
        if (row.release_count !== 0) throw new Error('initial conversation already released');
        row.release_count++; row.released_at=now(); resolve();
      };});
    }
    if (selection) observed.firstSelection = false;
    if (row) row.dispatch_at=now();
    const response = await original(...args);
    if (url.pathname.endsWith('/copilot/messages')) {
      const stream={raw:null,done:false,error:null}; observed.streams.push(stream);
      response.clone().text().then(raw=>{stream.raw=raw;stream.done=true;},
        error=>{stream.error=error.name;stream.done=true;});
    }
    if (row) {
      row.response_status=response.status; row.response_body_utf8=await response.clone().text();
      row.body_completed_at=now();
      const actualJson = response.json.bind(response);
      response.json = async (...jsonArgs) => {
        const value=await actualJson(...jsonArgs); row.application_json_read_at=now();
        setTimeout(() => {row.delivery_settled_at=now();},0); return value;
      };
    }
    if (selection) {
      row.held_at=now();
      await new Promise(resolve => {window.__releaseLateSelection = () => {
        if (row.release_count !== 0) throw new Error('selection already released');
        row.release_count++; row.released_at=now(); resolve();
      };});
    }
    return response;
  };
})();"""


def run_selection_conversation_generation(evidence, run_id, *, state_root=None):
    """Reuse the actual delayed-delivery journey without mocking internal APIs."""
    evidence = Path(evidence)
    evidence.mkdir(parents=True, exist_ok=False)
    files = [Path(__file__), ROOT / "tests/product_app/browser/copilot_conversation_fixture.py"]
    files.extend(ROOT / "src/itinerary_system/product_app" / name for name in
        ("api.py", "service.py", "workspace.py", "static/js/app.js", "static/js/copilot.js"))
    sources = {str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    _write_json(evidence / "source_identity_before.json", sources)
    http, errors, failures, checkpoints = [], [], [], {}
    with conversation_server(evidence, base_url=BASE_URL, run_id=run_id, state_root=state_root) as server:
        with sync_playwright() as runtime:
            browser = runtime.chromium.launch(headless=True,
                **({"executable_path":str(CHROME)} if CHROME.is_file() else {}))
            context = browser.new_context(viewport={"width":1440,"height":900})
            page = context.new_page()
            page.on("pageerror", lambda error:errors.append(str(error)))
            page.on("console", lambda message:errors.append(message.text) if message.type == "error" else None)
            page.on("requestfailed", lambda request:failures.append({"url":request.url,"failure":request.failure}))

            def record(response, raw=None):
                if not urlsplit(response.url).path.startswith("/api/"):
                    return
                if response.url.endswith("/copilot/messages") and raw is None:
                    return
                row = _http_record(response, response_body=raw)
                name = f"http_{len(http)+1:03d}.json"
                _write_json(evidence / name, row)
                http.append({"file":name,"method":row["method"],"url":row["url"],"status":row["response_status_code"]})

            def snapshot(label):
                value = page.evaluate("""() => ({timing:window.__selectionConversation,
                  lifecycle:document.querySelector('#copilot-lifecycle')?.textContent,
                  lifecycle_state:document.querySelector('#copilot-lifecycle')?.dataset.state,
                  user_messages:[...document.querySelectorAll('.message-user p:first-of-type')].map(e=>e.textContent),
                  selected_day:document.querySelector('#day-list [aria-pressed="true"]')?.dataset.day,
                  observed_at:new Date().toISOString()})""")
                checkpoints[label] = value
                _write_json(evidence / "checkpoints.json", checkpoints)
                return value

            def read_current():
                return page.evaluate("""async () => {
                  const saved=JSON.parse(sessionStorage.getItem('itinerary-repair-copilot.session.v1'));
                  const headers={'X-Session-Token':saved.mutation_token,
                    'X-Product-Build-Id':document.querySelector('meta[name="product-build-id"]').content};
                  const response=await fetch(`/api/sessions/${saved.session_id}/conversation`,{headers});
                  return {status:response.status,body:await response.json()};
                }""")

            page.on("response", record)
            page.add_init_script(TIMING_OBSERVER)
            with _preserve_browser_evidence(context,page,evidence,{"console_errors":errors,"request_failures":failures}):
                try:
                    page.goto(BASE_URL + "/app/itinerary",wait_until="domcontentloaded")
                    page.wait_for_function("() => typeof window.__releaseInitialConversation === 'function'")
                    initial = _map_ack(page)
                    _write_json(evidence / "map_initial.json", initial)
                    _capture_ready(page,evidence / "before.png")
                    page.locator('.day-card[data-day="2"]').click()
                    page.wait_for_function("() => typeof window.__releaseLateSelection === 'function'")
                    held = snapshot("selection_held")["timing"]["calls"][-1]
                    original = json.loads(held["response_body_utf8"])["session"]
                    assert held["method"] == "POST" and held["response_status"] == 200
                    assert held["application_json_read_at"] is None and held["released_at"] is None
                    page.evaluate("window.__releaseInitialConversation()")
                    expect(page.locator("#copilot-lifecycle")).to_contain_text("Conversation restored")
                    restored = next(row for row in snapshot("original_conversation_restored")["timing"]["calls"]
                        if row["method"] == "GET" and row["url"].endswith("/conversation"))
                    assert json.loads(restored["response_body_utf8"])["session"] == original
                    page.locator("#open-copilot").click()
                    page.locator("#transcript-settings-button").click()
                    page.once("dialog",lambda dialog:dialog.accept())
                    with page.expect_response(lambda response:response.request.method == "DELETE"
                        and response.url.endswith("/conversation")) as removed:
                        page.locator("#delete-current-conversation").click()
                    assert removed.value.status == 200
                    expect(page.locator("#copilot-lifecycle")).to_contain_text("Conversation restored")
                    renewed = read_current()
                    _write_json(evidence / "persisted_after_delete.json", renewed)
                    snapshot("new_conversation_restored")
                    new_session = renewed["body"]["session"]
                    assert renewed["status"] == 200 and new_session["conversation_id"] != original["conversation_id"]
                    assert new_session["revision"] == original["revision"]
                    message = "What preferences do you remember?"
                    page.locator("#copilot-input").fill(message)
                    with page.expect_response(lambda response:response.url.endswith("/copilot/messages")) as chat:
                        page.locator("#copilot-submit").click()
                    page.wait_for_function("() => window.__selectionConversation.streams[0]?.done === true")
                    stream=page.evaluate("window.__selectionConversation.streams[0]")
                    record(chat.value,stream["raw"])
                    events = _sse_events(stream["raw"])
                    _write_json(evidence / "exact_chat_io.json", [{"input":chat.value.request.post_data_json,
                        "events":events,"output":events[-1],"stream":stream}])
                    assert stream["error"] is None and chat.value.status == 200
                    expect(page.locator(".message-user")).to_have_count(1)
                    expect(page.locator("#copilot-submit")).to_have_attribute("data-mode","send",timeout=30_000)
                    before_release = read_current()
                    _write_json(evidence / "persisted_before_release.json", before_release)
                    snapshot("new_conversation_message_visible")
                    _capture_ready(page,evidence / "before_late_selection.png")
                    page.evaluate("window.__releaseLateSelection()")
                    page.wait_for_function("() => window.__selectionConversation.calls.find(row=>row.method==='POST')?.delivery_settled_at !== null")
                    # The impact request is after updateSelection's assignment; its
                    # completion lets the real renderAll/update callbacks finish.
                    page.wait_for_function("() => window.__selectionConversation.calls.find(row=>row.method==='POST')?.application_json_read_at !== null")
                    expect(page.locator('#day-list [data-day="2"]')).to_have_attribute("aria-pressed","true")
                    final_map = _map_ack(page)
                    _capture_ready(page,evidence / "after_late_selection.png")
                    after_release = read_current()
                    _write_json(evidence / "persisted_after_release.json", after_release)
                    final = snapshot("late_selection_applied")
                    _write_json(evidence / "map_final.json", final_map)
                    _write_json(evidence / "runtime_trace.json", {"original_conversation_id":original["conversation_id"],
                        "new_conversation_id":new_session["conversation_id"],"session_revision":original["revision"],
                        "accepted_plan_id":original["accepted_plan_id"],"message":message,
                        "before_release":before_release,"after_release":after_release,"visible_after_release":final,
                        "map_initial":initial,"map_final":final_map,"live_provider_call_count":0})
                    assert after_release["body"] == before_release["body"]
                    assert after_release["body"]["session"]["draft"] == []
                    assert final_map == initial
                    assert final["user_messages"] == [message], "Late selection erased the new conversation from the visible chat"
                    assert "Previous local conversation cleared" not in final["lifecycle"]
                    assert all(row["trusted"] for row in final["timing"]["clicks"])
                    assert not errors and not failures
                    _write_json(evidence / "trace.json", {"schema_version":"conversation-browser-v1",
                        "scenario":"selection_conversation_generation","initial":initial,"final":final_map,
                        "assertions_passed":True,"completed_at":datetime.now(UTC).isoformat()})
                finally:
                    snapshot("final_observation")
                    _write_json(evidence / "http_index.json",http)
                    _write_json(evidence / "console_errors.json",errors)
                    _write_json(evidence / "network_failures.json",failures)
                    _write_json(evidence / "provider_scope.json", {"live_provider_call_count":0,
                        "mode":"deterministic","fake_sdk_attempt_count":0,
                        "llm_input_artifact":None,"llm_output_artifact":None,"real_chat_endpoint":True,
                        "internal_api_mocked":False,"live_openai_connectivity_proven":False})
                    _write_json(evidence / "source_identity_after.json", {str(path.relative_to(ROOT)):
                        hashlib.sha256(path.read_bytes()).hexdigest() for path in files})
            browser.close()
        assert server.sdk is None
    guard=json.loads((evidence / "provider_isolation.json").read_text())
    assert guard["finalized"] and guard["sent_provider_calls"] == 0 and guard["blocked_attempts"] == []


def test_late_selection_preserves_deleted_conversation_generation(tmp_path):
    run_selection_conversation_generation(
        os.environ.get("PRODUCT_E2E_SELECTION_METADATA_EVIDENCE_DIR", tmp_path / "evidence"),
        os.environ.get("PRODUCT_E2E_RUN_ID", "selection_conversation_diagnostic"))
