"""Real chat conflict rejection on original and accepted-child baselines."""
from __future__ import annotations

import os

import pytest
from copilot_conversation_fixture import observe_context
from test_copilot_conversation_journeys_live import journey
from test_e2e1_live_copilot_chat import _capture_ready, _map_ack, _write_json, expect

pytestmark = pytest.mark.skipif(os.environ.get("PRODUCT_E2E_CONVERSATION_DISPOSABLE") != "1",
    reason="Requires a reserved disposable app lifecycle and provider isolation")


@pytest.mark.parametrize("plan_context", ["original", "accepted_child"])
def test_conflicting_combined_chat_never_offers_or_stages_partial_edits(tmp_path, plan_context):
    with journey(tmp_path, "combined_conflicts", plan_context=plan_context) as case:
        page = case.page
        messages = [
            "Set Stearns Wharf duration to 90 minutes; set Stearns Wharf duration to 120 minutes.",
            "Set Stearns Wharf duration to 90 minutes; set Stearns Wharf duration to 90 minutes.",
            ("Set Stearns Wharf role to meal; set Stearns Wharf role to activity." if plan_context == "accepted_child" else
             "Replace Bixby Creek Bridge with Pfeiffer Big Sur State Park; set Bixby Creek Bridge duration to 90 minutes."),
        ]
        visible = []
        for message in messages:
            reply = case.send(message)
            sid = reply["session"]["session_id"]
            assert reply["turn"]["interpretation"]["state"] == "clarification_required"
            assert reply["session"]["draft"] == [] and reply["session"]["proposal"] is None
            assert reply["session"]["accepted_plan_id"] == case.initial["active_plan_id"]
            expect(page.locator(".copilot-proposal-card")).to_have_count(0)
            expect(page.get_by_role("button", name="Add all to draft", exact=True)).to_have_count(0)
            expect(page.get_by_role("button", name="Preview combined repair", exact=True)).to_have_count(0)
            assert _map_ack(page) == case.initial
            visible.append({"user_message":page.locator(".message-user").last.locator("p").first.inner_text(),
                "assistant_message":page.locator(".message-assistant").last.locator("p").first.inner_text(),
                "proposal_card_count":page.locator(".copilot-proposal-card").count(),
                "map":_map_ack(page)})
        _write_json(case.evidence / "blocked_visible_results.json", visible)
        _write_json(case.evidence / "conflict_controls_before_reload.json", page.evaluate("window.__conversationControlClicks"))
        before = observe_context(case.server, sid, case.evidence, "conflicts_before_reload")
        assert before["draft_operations"] == []
        _capture_ready(page, case.evidence / "conflicts_blocked.png")
        page.reload(wait_until="domcontentloaded")
        reloaded = _map_ack(page)
        assert reloaded == case.initial
        page.locator("#open-copilot").click()
        expect(page.locator(".message-user")).to_have_count(3)
        expect(page.locator(".copilot-proposal-card")).to_have_count(0)
        after = observe_context(case.server, sid, case.evidence, "conflicts_after_reload")
        assert after["draft_operations"] == []
        assert before["accepted_plan_id"] == after["accepted_plan_id"] == reloaded["active_plan_id"]
        _capture_ready(page, case.evidence / "conflicts_reloaded.png")
        _write_json(case.evidence / "trace.json", {"schema_version":"conversation-browser-v1",
            "scenario":case.scenario, "context":plan_context, "initial":case.initial, "final":reloaded,
            "chat_turn_count":3, "accepted":False, "assertions_passed":True})
