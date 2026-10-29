"""Actual website chat and map over registered, independently evaluated owned inputs."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from control_action_fixture import OBSERVE_CONTROL_ACTIONS, after_action, before_action, fill_native_text
from owned_product_fixture import build_owned_product_registry
from test_copilot_conversation_journeys_live import _combined_chat_journey, journey
from test_e2e1_live_copilot_chat import _capture_ready, _independent_geometry_hash, _map_ack, _write_json, expect

pytestmark = pytest.mark.skipif(os.environ.get("PRODUCT_E2E_CONVERSATION_DISPOSABLE") != "1",
    reason="Requires reserved port 8127 and disposable, provider-isolated app state")


def test_owned_replacement_and_duration_uses_existing_complete_browser_journey(tmp_path):
    registry = build_owned_product_registry(tmp_path)
    _combined_chat_journey(tmp_path, registry_path=registry)
    _write_json(Path(os.environ["PRODUCT_E2E_CONVERSATION_EVIDENCE_DIR"]) / "combined_chat" / "owned_fixture_input.json",
        {"registry_path": str(registry), "registry_body_utf8": registry.read_text(encoding="utf-8"),
         "family": "replacement_duration", "protected": False, "entry": "chat"})


@pytest.mark.parametrize("protected", [False, True])
@pytest.mark.parametrize("family,entry", [("mixed", "chat"), ("duration", "chat"), ("time_window", "chat"),
                                        ("duration", "editor"), ("time_window", "editor")])
def test_owned_mixed_chat_renders_one_child_or_preserves_parent(tmp_path, protected, family, entry):
    registry = build_owned_product_registry(tmp_path,
        protected_attribute=("duration" if family == "mixed" else family) if protected else None)
    name = "combined_chat_owned_" + family + "_" + ("protected" if protected else "eligible")
    if entry == "editor":
        name += "_editor"
    with journey(tmp_path, name, registry_path=registry,
                 plan_context="original" if family == "mixed" else "accepted_child",
                 expected_final_draft_types=("set_stop_" + family,) if protected and family != "mixed" else ()) as case:
        page = case.page
        _write_json(case.evidence / "owned_fixture_input.json", {"registry_path": str(registry),
            "registry_body_utf8": registry.read_text(encoding="utf-8"), "family": family,
            "protected": protected, "entry": entry})
        message = ("Replace Bixby Creek Bridge with Pfeiffer Big Sur State Park; "
                   "set Griffith Observatory duration to 90 minutes; "
                   "set Golden Gate Bridge earliest arrival to 10:00; "
                   "set Golden Gate Bridge role to scenic_stop.")
        if family == "duration":
            message = "Set Griffith Observatory duration to 90 minutes."
        elif family == "time_window":
            message = "Set Golden Gate Bridge earliest arrival to 10:00."
        if entry == "chat":
            reply = case.send(message)
            assert reply["turn"]["interpretation"]["state"] == "proposal_ready"
            assert {row["type"] for row in reply["turn"]["interpretation"]["intents"]} == ({
                "replace_nearby", "set_stop_duration", "set_stop_time_window", "set_stop_role"}
                if family == "mixed" else {"set_stop_" + family})
            actions = page.locator(".copilot-proposal-actions").last
            stage_control = actions.get_by_role("button", name="Add all to draft" if family == "mixed" else "Add to draft", exact=False)
        else:
            page.locator("#close-copilot").click()
            page.evaluate(OBSERVE_CONTROL_ACTIONS)
            page.locator(f'.day-card[data-day="{7 if family == "duration" else 1}"]').click()
            page.locator("#map-edit-button").click()
            page.locator("#inspect-edit-stop").click()
            _write_json(case.evidence / "editor_policy_before.json", {"text": page.locator('[data-combination-policy]').inner_text()})
            expect(page.locator('[data-combination-policy]')).not_to_contain_text("Mixing those families is not supported")
            page.locator("#typed-edit-surface summary").filter(has_text="Duration and time window").click()
            form = page.locator(f'[data-typed-operation="set_stop_{family}"]')
            fill_native_text(form.locator('[name="duration-preferred"]' if family == "duration" else '[name="earliest_arrival"]'),
                             "90" if family == "duration" else "10:00")
            stage_control = form.locator('button[type="submit"]')
            before = before_action(page, stage_control)
        assert _map_ack(page) == case.initial
        with page.expect_response(lambda response: response.url.endswith("/draft/operations")) as staged:
            stage_control.click()
        assert staged.value.status == 200
        assert len(staged.value.json()["session"]["draft"]) == (4 if family == "mixed" else 1)
        assert _map_ack(page) == case.initial
        _write_json(case.evidence / "staging_io.json", {"input":staged.value.request.post_data_json,"output":staged.value.json()})
        _write_json(case.evidence / "staged_map_ack.json", _map_ack(page))
        if entry == "editor":
            _write_json(case.evidence / "editor_control_io.json", after_action(page, before,
                context="accepted_child", phase="stage", request=staged.value.request.post_data_json, response=staged.value.json()))
            page.locator("#close-map-edit").click()
        preview_control = (actions.get_by_role("button", name="Preview combined repair" if family == "mixed" else "Preview this repair", exact=False)
                           if entry == "chat" else page.locator("#inline-preview"))
        with page.expect_response(lambda response: response.url.endswith("/preview"), timeout=90_000) as received:
            preview_control.click()
        assert received.value.status == 200, received.value.text()
        payload, proposal = received.value.json(), received.value.json()["proposal"]
        receipt = proposal["runtime_execution"]
        assert receipt["plan_diff"]["schema_version"] == ("plan-diff-v7" if family == "mixed" else "plan-diff-v5")
        assert receipt["evaluation_certificate"]["evaluator_version"] == "verify-001-plan-evaluator-v3"
        assert receipt["child_plan"]["plan_id"] == receipt["evaluation_certificate"]["plan_id"] == receipt["plan_diff"]["child_plan_id"]
        assert receipt["child_plan"]["content_hash"] == receipt["evaluation_certificate"]["plan_content_hash"]
        assert receipt["evaluation_certificate"]["metrics"]["day_1_waiting_minutes"] >= 0
        if family != "time_window":
            assert receipt["evaluation_certificate"]["metrics"]["day_7_visit_minutes"] >= 90
        _write_json(case.evidence / "preview_response.json", payload)
        expect(page.locator(".map-stage")).to_have_attribute("data-preview-state",
            "PREVIEW_INELIGIBLE" if protected else "PREVIEW_RENDERED", timeout=45_000)
        rendered = _map_ack(page)
        if entry == "chat":
            page.locator("#close-copilot").click()
        if protected:
            assert proposal["state"] == "ineligible"
            for key in ("active_plan_id", "active_plan_content_hash", "rendered_map_plan_id", "rendered_geometry_hash", "rendered_marker_ids"):
                assert rendered[key] == case.initial[key]
            expect(page.locator("#accept-repair")).to_be_disabled()
            expect(page.locator("#day-list")).to_contain_text(
                "Bixby Creek Bridge" if family == "mixed" else "Pfeiffer Big Sur State Park")
        else:
            assert proposal["state"] == "eligible"
            assert rendered["active_plan_id"] == rendered["rendered_map_plan_id"] == proposal["child_plan_id"]
            assert rendered["active_plan_content_hash"] == proposal["child_plan_content_hash"]
            assert rendered["rendered_geometry_hash"] == _independent_geometry_hash(page, proposal["geography_plan"])
            if family == "mixed":
                assert "bixby_creek_bridge_viewpoint" not in rendered["rendered_marker_ids"]
                assert "pfeiffer_big_sur_state_park" in rendered["rendered_marker_ids"]
                expect(page.locator("#day-list")).to_contain_text("Pfeiffer Big Sur State Park")
                expect(page.locator("#day-list")).not_to_contain_text("Bixby Creek Bridge")
            else:
                assert rendered["rendered_marker_ids"] == case.initial["rendered_marker_ids"]
                stop = next(row for row in receipt["child_plan"]["selected_stops"] if (row.get("stop_id") or row.get("poi_id")) ==
                    ("griffith_observatory" if family == "duration" else "golden_gate_bridge"))
                assert (stop["visit_duration_minutes"] == 90 if family == "duration"
                        else stop["time_window_constraint"]["earliest_arrival"] == "10:00")
            expect(page.locator("#metric-strip")).to_have_attribute("data-plan-id", proposal["child_plan_id"])
            expect(page.locator("#metric-strip")).to_have_attribute("data-evaluation-id", proposal["certificate_id"])
        _capture_ready(page, case.evidence / "owned_preview.png")
        _write_json(case.evidence / "presentation_state.json", {
            "accept_disabled": page.locator("#accept-repair").is_disabled(),
            "metrics": page.locator("#metric-strip").evaluate("e => ({...e.dataset})"),
            "timeline": page.locator("#day-list").inner_text()})
        if not protected:
            page.locator('[data-route="/app/compare"]').first.click()
            expect(page.locator('[data-compare-map-pane="option"] [data-compare-map-canvas]')).to_have_attribute(
                "data-rendered-map-plan-id", proposal["child_plan_id"], timeout=30_000)
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
            assert accepted.value.json()["session"]["accepted_plan_id"] == proposal["child_plan_id"]
            _write_json(case.evidence / "acceptance_response.json", accepted.value.json())
        if entry == "editor":
            _write_json(case.evidence / "editor_final_events.json", page.evaluate("window.__draftControlInputs"))
        sid = payload["session"]["session_id"]
        with page.expect_response(lambda response: response.request.method == "GET"
                and response.url.endswith(f"/api/sessions/{sid}")) as restored:
            page.reload(wait_until="domcontentloaded")
        assert restored.value.status == 200
        _write_json(case.evidence / "restoration_response.json", restored.value.json())
        assert restored.value.json()["session"]["draft"] == (payload["session"]["draft"] if protected else [])
        reloaded = _map_ack(page)
        for key in ("active_plan_id", "active_plan_content_hash", "rendered_map_plan_id", "rendered_geometry_hash", "rendered_marker_ids"):
            assert reloaded[key] == rendered[key]
        _capture_ready(page, case.evidence / "owned_reload.png")
        _write_json(case.evidence / "trace.json", {"schema_version":"owned-mixed-browser-v1", "initial":case.initial,
            "entry": entry, "family": family, "protected": protected,
            "context": "original" if family == "mixed" else "accepted_child",
            "rendered":rendered, "reloaded":reloaded, "request_id":receipt["request_id"],
            "child_plan_id":proposal["child_plan_id"], "evaluation_id":proposal["certificate_id"],
            "accepted":not protected, "assertions_passed":True, "live_openai_test":False})
