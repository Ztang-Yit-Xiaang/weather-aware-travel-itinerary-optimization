"""Actual typed controls through canonical evaluation, panels and rendered map."""
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


def _artifact_hash(record):
    payload = {key: value for key, value in record.items() if key != "content_hash"}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()[:16]


@pytest.mark.skipif(not BASE_URL or os.environ.get("PRODUCT_E2E_DIRECT_EDITS_DISPOSABLE") != "1",
    reason="Requires explicitly isolated direct-edit server state")
@pytest.mark.parametrize("kind,target,day", [
    ("combined", "bixby_creek_bridge_viewpoint", 5),
    ("duration", "griffith_observatory", 7),
    ("time_window", "stearns_wharf", 6),
    ("role", "stearns_wharf", 6),
    ("order", "kings_canyon_national_park", 4),
])
def test_direct_edit_browser_renders_exact_evaluated_child(tmp_path: Path, kind: str, target: str, day: int):
    evidence = Path(os.environ.get("PRODUCT_E2E_DIRECT_EDITS_EVIDENCE_DIR", tmp_path)) / kind
    evidence.mkdir(parents=True, exist_ok=True)
    errors, failures, requests, responses = [], [], [], []
    with sync_playwright() as runtime:
        browser = runtime.chromium.launch(headless=True, **({"executable_path": str(CHROME)} if CHROME.is_file() else {}))
        context = browser.new_context(viewport={"width":1440,"height":900})
        page = context.new_page()
        page.on("console", lambda message: errors.append(message.text) if message.type == "error" else None)
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.on("requestfailed", lambda request: failures.append({"url":request.url,"failure":request.failure}))
        page.on("request", lambda request: requests.append({"url":request.url,"method":request.method}))

        def record(response):
            if urlsplit(response.url).path.startswith("/api/"):
                row = _http_record(response)
                # Reading a body can dispatch another Playwright response callback.
                # Allocate only after that read, so nested captures cannot reuse a path.
                name = f"http_{len(responses)+1:03d}.json"
                _write_json(evidence / name, row)
                responses.append({"file":name,"url":row["url"],"status":row["response_status_code"]})
        page.on("response", record)

        def open_edit(selected_day):
            page.locator(f'.day-card[data-day="{selected_day}"]').click()
            page.locator("#map-edit-button").click()
            page.locator("#inspect-edit-stop").click()

        def preview():
            expect(page.locator("#inline-preview")).to_be_enabled(timeout=15_000)
            with page.expect_response(lambda response: response.url.endswith("/preview"), timeout=60_000) as completed:
                page.locator("#inline-preview").click()
            assert completed.value.status == 200, completed.value.text()
            payload = completed.value.json()
            assert payload["proposal"]["state"] == "eligible", payload["proposal"].get("reason")
            expect(page.locator(".map-stage")).to_have_attribute("data-preview-state", "PREVIEW_RENDERED", timeout=40_000)
            assert _map_ack(page)["rendered_map_plan_id"] == payload["proposal"]["child_plan_id"]
            return payload

        with _preserve_browser_evidence(context, page, evidence, {"console_errors":errors,"request_failures":failures}):
            try:
                page.goto(f"{BASE_URL}/app/itinerary", wait_until="domcontentloaded")
                original = _map_ack(page)
                # Resolve the inherited weather disruption using real UI actions first.
                # Subsequent parameter cases share only this explicitly accepted baseline.
                if kind != "combined" and "bixby_creek_bridge_viewpoint" in original["rendered_marker_ids"]:
                    open_edit(5)
                    page.locator(".legacy-edit-controls summary").click()
                    page.locator("#registered-candidate").select_option("pfeiffer_big_sur_state_park")
                    with page.expect_response(lambda response: response.url.endswith("/draft/operations")) as staged:
                        page.locator("#draft-replace").click()
                    assert staged.value.status == 200
                    page.locator("#close-map-edit").click()
                    baseline_preview = preview()
                    _write_json(evidence / "baseline_preview_response.json", baseline_preview)
                    expect(page.locator("#accept-repair")).to_be_enabled()
                    with page.expect_response(lambda response: response.url.endswith("/accept")) as accepted:
                        page.locator("#accept-repair").click()
                        expect(page.get_by_role("dialog", name="Confirm trip decision", exact=True)).to_be_visible()
                        expect(page.locator("#acceptance-message")).to_have_text('Accept this independently evaluated repair as the durable trip plan?')
                        page.get_by_role("dialog", name="Confirm trip decision", exact=True).get_by_role("button", name='Accept repair', exact=True).click()
                    assert accepted.value.status == 200, accepted.value.text()
                    _write_json(evidence / "baseline_acceptance_response.json", accepted.value.json())
                    page.reload(wait_until="domcontentloaded")
                parent = _map_ack(page)
                if kind == "combined":
                    assert "bixby_creek_bridge_viewpoint" in parent["rendered_marker_ids"]
                    assert "pfeiffer_big_sur_state_park" not in parent["rendered_marker_ids"]
                else:
                    assert "pfeiffer_big_sur_state_park" in parent["rendered_marker_ids"]
                    assert "bixby_creek_bridge_viewpoint" not in parent["rendered_marker_ids"]
                _capture_ready(page, evidence / "screenshot_before.png")
                open_edit(day)
                form = page.locator(f'[data-typed-operation="set_stop_{kind}"]')
                operation_inputs = []
                if kind == "combined":
                    page.locator(".legacy-edit-controls summary").click()
                    page.locator("#registered-candidate").select_option("pfeiffer_big_sur_state_park")
                    with page.expect_response(lambda response: response.url.endswith("/draft/operations")) as staged:
                        page.locator("#draft-replace").click()
                    assert staged.value.status == 200, staged.value.text()
                    operation_inputs.append(staged.value.request.post_data_json)
                    page.locator("#close-map-edit").click()
                    for edit_kind, edit_day, field, value in (
                        ("duration", 7, "duration-preferred", "90"),
                        ("time_window", 1, "earliest_arrival", "10:00"),
                        ("role", 1, "role", "scenic_stop"),
                        ("order", 4, "sequence_position", "2"),
                    ):
                        open_edit(edit_day)
                        if edit_kind in {"duration", "time_window"}:
                            page.locator("#typed-edit-surface summary").filter(has_text="Duration and time window").click()
                        edit_form = page.locator(f'[data-typed-operation="set_stop_{edit_kind}"]')
                        control = edit_form.locator(f'[name="{field}"]')
                        if edit_kind in {"role", "order"}:
                            control.select_option(value)
                        else:
                            control.fill(value)
                        with page.expect_response(lambda response: response.url.endswith("/draft/operations")) as staged:
                            edit_form.locator('button[type="submit"]').click()
                        assert staged.value.status == 200, staged.value.text()
                        operation_inputs.append(staged.value.request.post_data_json)
                        if edit_kind != "order":
                            page.locator("#close-map-edit").click()
                    actual_input = operation_inputs
                if kind in {"duration", "time_window"}:
                    page.locator("#typed-edit-surface summary").filter(has_text="Duration and time window").click()
                if kind == "duration":
                    form.locator('[name="duration-preferred"]').fill("60")
                    expected_parameters = {"duration":{"mode":"exact","preferred_minutes":60,"minimum_minutes":60,"maximum_minutes":60}}
                elif kind == "time_window":
                    form.locator('[name="earliest_arrival"]').fill("10:00")
                    expected_parameters = {"earliest_arrival":"10:00","latest_departure":None}
                elif kind == "role":
                    form.locator('[name="role"]').select_option("meal")
                    expected_parameters = {"role":"meal"}
                elif kind == "order":
                    form.locator('[name="sequence_position"]').select_option("2")
                    expected_parameters = {"day":day,"sequence_index":1}
                if kind != "combined":
                    with page.expect_response(lambda response: response.url.endswith("/draft/operations")) as staged:
                        form.locator('button[type="submit"]').click()
                    assert staged.value.status == 200, staged.value.text()
                    actual_input = staged.value.request.post_data_json
                    assert actual_input["type"] == f"set_stop_{kind}"
                    assert actual_input["target"] == target
                    assert actual_input["parameters"] == expected_parameters
                assert len(staged.value.json()["session"]["draft"]) == (5 if kind == "combined" else 1)
                # Supported families remain available; the complete draft is evaluated atomically.
                other_kind = "duration" if kind != "duration" else "role"
                other = page.locator(f'[data-typed-operation="set_stop_{other_kind}"]')
                expect(other).not_to_have_attribute("data-combination-blocking-code", "draft_evaluated_operation_combination_unsupported")
                expect(other.locator('button[type="submit"]')).to_be_enabled()
                page.locator("#close-map-edit").click()
                staged_ack = _map_ack(page)
                assert staged_ack["rendered_map_plan_id"] == parent["rendered_map_plan_id"]
                assert staged_ack["rendered_geometry_hash"] == parent["rendered_geometry_hash"]
                assert staged_ack["rendered_marker_ids"] == parent["rendered_marker_ids"]
                payload = preview()
                proposal = payload["proposal"]
                _write_json(evidence / "preview_response.json", payload)
                _write_json(evidence / "operation_input.json", actual_input)
                receipt = proposal.get("runtime_execution", {})
                certificate = receipt.get("evaluation_certificate") or proposal.get("evaluation_certificate") or proposal.get("certificate_schedule_evidence", {}).get("certificate_record") or proposal.get("certificate_role_evidence", {}).get("certificate_record")
                plan_diff = proposal.get("plan_diff") or receipt.get("plan_diff")
                assert certificate is not None
                assert certificate["plan_id"] == proposal["child_plan_id"]
                assert certificate["plan_content_hash"] == proposal["child_plan_content_hash"]
                assert certificate["comparison_eligibility"] == "eligible"
                assert certificate["certificate_id"] == proposal["certificate_id"]
                assert _artifact_hash(certificate) == certificate["content_hash"] == proposal["certificate_content_hash"]
                assert _artifact_hash(plan_diff) == proposal["diff_content_hash"]
                assert plan_diff["diff_id"] == proposal["diff_id"]
                assert plan_diff["child_plan_id"] == proposal["child_plan_id"]
                assert proposal["parent_plan_id"] == parent["rendered_map_plan_id"]
                changed = next(feature["properties"] for feature in proposal["geography_plan"]["stops"]["features"]
                    if feature["properties"]["stop_id"] == ("pfeiffer_big_sur_state_park" if kind == "combined" else target))
                if kind == "duration":
                    assert changed["visit_duration_minutes"] == 60
                elif kind == "role":
                    assert changed["itinerary_role"] == "meal"
                elif kind == "order":
                    assert changed["stop_order"] == 2
                elif kind == "time_window":
                    changes = proposal["plan_diff"]["time_window_changes"]
                    assert len(changes) == 1 and changes[0]["stop_id"] == target
                    assert changes[0]["to_earliest_arrival"] == "10:00"
                    assert changes[0]["to_latest_departure"] is None
                else:
                    child = receipt["child_plan"]
                    assert _artifact_hash(child) == child["content_hash"] == proposal["child_plan_content_hash"]
                    assert proposal["execution_mode"] == "atomic_combined_independent_evaluation"
                    assert len(proposal["compiled_request"]["operations"]) == len(operation_inputs) == 5
                    assert receipt["optimizer_execution"]["planner_run"]["method_executed"] == "deterministic_context_aware_heuristic"
                    assert plan_diff["schema_version"] == "plan-diff-v6"
                    assert all(plan_diff[key] for key in ("duration_changes", "time_window_changes", "role_changes", "reorder_changes"))
                    stops = {stop["stop_id"]: stop for stop in child["selected_stops"]}
                    assert "bixby_creek_bridge_viewpoint" not in stops and "pfeiffer_big_sur_state_park" in stops
                    assert stops["griffith_observatory"]["visit_duration_minutes"] == 90
                    assert stops["golden_gate_bridge"]["time_window_constraint"]["earliest_arrival"] == "10:00"
                    assert stops["golden_gate_bridge"]["itinerary_role"] == "scenic_stop"
                    assert child["ordered_days"][3]["stop_ids"] == ["kings_canyon_national_park", "sequoia_national_park"]
                rendered = _map_ack(page)
                geometry_hash = _independent_geometry_hash(page, proposal["geography_plan"])
                assert rendered["rendered_geometry_hash"] == geometry_hash
                expected_markers = set(parent["rendered_marker_ids"])
                if kind == "combined":
                    expected_markers.remove("bixby_creek_bridge_viewpoint")
                    expected_markers.add("pfeiffer_big_sur_state_park")
                    assert rendered["rendered_geometry_hash"] != parent["rendered_geometry_hash"]
                assert set(rendered["rendered_marker_ids"]) == expected_markers
                assert rendered["active_plan_content_hash"] == proposal["child_plan_content_hash"]
                expect(page.locator("#metric-strip")).to_have_attribute("data-plan-id", proposal["child_plan_id"])
                expect(page.locator("#metric-strip")).to_have_attribute("data-evaluation-id", proposal["certificate_id"])
                _capture_ready(page, evidence / "screenshot_after.png")
                page.locator('[data-route="/app/compare"]').first.click()
                compare = page.locator('[data-compare-map-pane="option"] [data-compare-map-canvas]')
                expect(compare).to_have_attribute("data-rendered-map-plan-id", proposal["child_plan_id"], timeout=30_000)
                _capture_ready(page, evidence / "compare.png")
                page.locator("#compare-evidence").click()
                view = page.locator('[data-evidence-plan-id]')
                expect(view).to_have_attribute("data-evidence-plan-id", proposal["child_plan_id"])
                expect(view).to_have_attribute("data-plan-content-hash", proposal["child_plan_content_hash"])
                expect(view).to_contain_text(proposal["certificate_id"])
                _write_json(evidence / "evidence_panel.json", {"scope":view.evaluate("e => ({...e.dataset})"),"text":view.inner_text()})
                page.locator("#evidence-back").click()
                page.reload(wait_until="domcontentloaded")
                expect(page.locator(".map-stage")).to_have_attribute("data-preview-state", "PREVIEW_RENDERED", timeout=40_000)
                reloaded = _map_ack(page)
                assert reloaded == rendered
                _capture_ready(page, evidence / "after_reload.png")
                # Explicit acceptance is a separate real UI action for every typed family.
                expect(page.locator("#accept-repair")).to_be_enabled(timeout=15_000)
                with page.expect_response(lambda response: response.url.endswith("/accept"), timeout=60_000) as accepted:
                    page.locator("#accept-repair").click()
                    expect(page.get_by_role("dialog", name="Confirm trip decision", exact=True)).to_be_visible()
                    expect(page.locator("#acceptance-message")).to_have_text('Accept this independently evaluated repair as the durable trip plan?')
                    page.get_by_role("dialog", name="Confirm trip decision", exact=True).get_by_role("button", name='Accept repair', exact=True).click()
                assert accepted.value.status == 200, accepted.value.text()
                accepted_payload = accepted.value.json()
                assert accepted_payload["session"]["accepted_plan_id"] == proposal["child_plan_id"]
                assert accepted_payload["session"]["proposal"] is None
                assert accepted_payload["result"]["decision"]["certificate_content_hash"] == certificate["content_hash"]
                _write_json(evidence / "acceptance_response.json", accepted_payload)
                page.reload(wait_until="domcontentloaded")
                accepted_ack = _map_ack(page)
                assert accepted_ack["preview_state"] == "PARENT_RENDERED"
                assert {key:value for key,value in accepted_ack.items() if key != "preview_state"} == {key:value for key,value in rendered.items() if key != "preview_state"}
                _capture_ready(page, evidence / "accepted_after_reload.png")
                page.locator('[data-route="/app/evidence"]').first.click()
                accepted_evidence = page.locator('[data-evidence-plan-id]')
                expect(accepted_evidence).to_have_attribute("data-evidence-plan-id", proposal["child_plan_id"])
                expect(accepted_evidence).to_have_attribute("data-evidence-scope", "accepted")
                expect(accepted_evidence).to_have_attribute("data-evaluation-id", proposal["certificate_id"])
                _write_json(evidence / "accepted_evidence_panel.json", {"scope":accepted_evidence.evaluate("e => ({...e.dataset})"),"text":accepted_evidence.inner_text()})
                assert errors == []
                assert failures == []
                assert len({row["file"] for row in responses}) == len(responses)
                assert not any(row["method"] == "POST" and "/copilot/messages" in row["url"] for row in requests)
                _write_json(evidence / "trace.json", {"schema_version":"direct-edit-browser-v1","kind":kind,
                    "parent":parent,"staged":staged_ack,"rendered":rendered,"reloaded":reloaded,"accepted":accepted_ack,
                    "child_plan_id":proposal["child_plan_id"],"child_plan_content_hash":proposal["child_plan_content_hash"],
                    "evaluation_id":proposal["certificate_id"],"geometry_hash":geometry_hash,
                    "operation_input":actual_input,"assertions_passed":True,"completed_at":datetime.now(UTC).isoformat()})
                _write_json(evidence / "llm_io.json", {"provider_call_count":0,"llm_input":None,"llm_output":None,
                    "reason":"Actual direct controls and deterministic optimizer/evaluator; no chat sent."})
            finally:
                _write_json(evidence / "http_index.json", responses)
                _write_json(evidence / "console_errors.json", errors)
                _write_json(evidence / "network_failures.json", failures)
        browser.close()
