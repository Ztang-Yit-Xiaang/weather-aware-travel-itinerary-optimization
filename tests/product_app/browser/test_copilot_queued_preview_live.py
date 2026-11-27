"""Real queued-preview ownership gate; only original-response delivery is delayed."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

pytest.importorskip("playwright.sync_api")
from copilot_conversation_fixture import observe_context
from test_copilot_conversation_journeys_live import journey
from test_e2e1_live_copilot_chat import _capture_ready, _http_record, _map_ack, _write_json, expect

pytestmark = pytest.mark.skipif(
    os.environ.get("PRODUCT_E2E_QUEUED_PREVIEW_DISPOSABLE") != "1",
    reason="Requires explicitly reserved disposable port 8127; never attaches to an existing app",
)

TIMING_OBSERVER = r"""() => {
  const original=window.fetch.bind(window);
  const audit=window.__queuedPreviewAudit={calls:[],held:null,released:false,json_read:false};
  window.fetch=async (...args) => {
    const url=typeof args[0] === 'string' ? args[0] : args[0].url;
    const row={url,method:args[1]?.method || 'GET',request_body_utf8:args[1]?.body || '',
      dispatch_at:new Date().toISOString()};
    audit.calls.push(row);
    const response=await original(...args);
    if(!audit.held && url.endsWith('/draft/impact-preview')) {
      audit.held={...row,response_body_utf8:await response.clone().text(),response_status:response.status,
        held_at:new Date().toISOString()};
      const originalJson=response.json.bind(response);
      response.json=async (...jsonArgs) => {
        const value=await originalJson(...jsonArgs); audit.json_read=true; return value;
      };
      await new Promise(resolve => {window.__releaseQueuedImpact=() => {
        if(audit.released) throw new Error('Already released');
        audit.released=true; audit.released_at=new Date().toISOString(); resolve();
      };});
    }
    return response;
  };
}"""


@pytest.mark.parametrize("combined", [False, True], ids=["single", "combined"])
@pytest.mark.parametrize("plan_context", ["original", "accepted_child"])
def test_deleted_chat_cannot_resume_queued_preview(tmp_path, combined, plan_context):
    from e2e1_source_universe import ROOT, deterministic_source_inputs

    inputs = set(deterministic_source_inputs(Path(__file__).resolve().relative_to(ROOT).as_posix()))
    sources = {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in sorted(inputs)}
    root = Path(os.environ.get("PRODUCT_E2E_CONVERSATION_EVIDENCE_DIR", tmp_path))
    name = "combined_chat_async_delete_" + ("combined" if combined else "single")
    expected_types = ("set_stop_duration",) if plan_context == "accepted_child" else ("replace_nearby",)
    if combined:
        expected_types += ("set_stop_time_window" if plan_context == "accepted_child" else "set_stop_duration",)
    with journey(root, name, plan_context=plan_context, expected_final_draft_types=expected_types) as case:
        _write_json(case.evidence / "source_inputs_before.json", sources)
        page = case.page
        message = (
            "Set Pfeiffer Big Sur State Park duration to 90 minutes"
            if plan_context == "accepted_child"
            else "Replace Bixby Creek Bridge with Pfeiffer Big Sur State Park"
        )
        if combined:
            message += (
                "; set Stearns Wharf earliest arrival to 10:00."
                if plan_context == "accepted_child"
                else "; set Griffith Observatory duration to 90 minutes."
            )
        reply = case.send(message)
        sid = reply["session"]["session_id"]
        observe_context(case.server, sid, case.evidence, "queued_before_stage")
        page.evaluate(TIMING_OBSERVER)
        control = page.locator(".copilot-proposal-actions").last.get_by_role(
            "button", name="Preview combined repair" if combined else "Preview this repair", exact=combined
        )
        with page.expect_response(lambda response: response.url.endswith("/draft/operations")) as staged:
            control.click()
        assert staged.value.status == 200
        page.wait_for_function("() => typeof window.__releaseQueuedImpact === 'function'", timeout=90_000)
        before = observe_context(case.server, sid, case.evidence, "before_delete")
        assert len(before["draft_operations"]) == (2 if combined else 1)
        assert _map_ack(page) == case.initial
        deleted = case.reset()
        reset = observe_context(case.server, sid, case.evidence, "after_delete")
        assert deleted["session"]["conversation_id"] is None
        assert before["draft_operations"] == reset["draft_operations"]
        _capture_ready(page, case.evidence / "after_delete_before_release.png")
        page.evaluate("window.__releaseQueuedImpact()")
        page.wait_for_function("() => window.__queuedPreviewAudit.json_read === true")
        page.wait_for_timeout(2000)
        timing = page.evaluate("window.__queuedPreviewAudit")
        previews = [row for row in timing["calls"] if row["url"].endswith("/preview")]
        final = observe_context(case.server, sid, case.evidence, "after_release")
        ack = _map_ack(page)
        _write_json(
            case.evidence / "race_io.json",
            {
                "scope": "Real website and local APIs, delayed original response only",
                "combined": combined,
                "input": message,
                "turn": reply["turn"],
                "before_delete": before,
                "after_delete": reset,
                "after_release": final,
                "timing": timing,
                "preview_requests": previews,
                "initial_map": case.initial,
                "final_map": ack,
                "llm_input": None,
                "llm_output": None,
                "provider_attempts": 0,
            },
        )
        assert previews == []
        assert ack == case.initial
        assert final["draft_operations"] == before["draft_operations"]
        expect(page.locator(".message-user")).to_have_count(0)
        expect(page.locator(".map-stage")).to_have_attribute("data-preview-state", "PARENT_RENDERED")
        _capture_ready(page, case.evidence / "after_release.png")
        _write_json(case.evidence / "controls_before_reload.json", page.evaluate("window.__conversationControlClicks"))
        with page.expect_response(
            lambda response: response.request.method == "GET" and response.url.endswith("/conversation")
        ) as reloaded_chat:
            page.reload(wait_until="domcontentloaded")
        assert _map_ack(page) == case.initial
        restored = observe_context(case.server, sid, case.evidence, "after_reload")
        assert restored["draft_operations"] == before["draft_operations"]
        expect(page.locator(".message-user")).to_have_count(0)
        _capture_ready(page, case.evidence / "after_reload.png")
        assert reloaded_chat.value.json()["conversation"]["turns"] == []
        page.locator("#open-copilot").click()
        expect(page.locator(".message-user")).to_have_count(0)
        _write_json(case.evidence / "reload_conversation_http.json", _http_record(reloaded_chat.value))
        _write_json(
            case.evidence / "trace.json",
            {
                "schema_version": "conversation-browser-v1",
                "scenario": case.scenario,
                "context": plan_context,
                "initial": case.initial,
                "final": _map_ack(page),
                "combined": combined,
                "preview_request_count": 0,
                "accepted": False,
                "assertions_passed": True,
            },
        )
        final_sources = {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in sorted(inputs)}
        _write_json(case.evidence / "source_inputs_after.json", final_sources)
        assert sources == final_sources
