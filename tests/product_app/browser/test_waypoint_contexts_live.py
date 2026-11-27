"""Remaining route-checked waypoint contexts through real UI and native pointer.

Two ordered cases share a fresh ordinary server: original, then accepted_child.
The latter accepts a real evaluated replacement before testing route-only and
custom drafts. These drafts are never called evaluated repairs. Overlay-release
regression coverage belongs to test_waypoint_pointer_live, not this file.
"""

from __future__ import annotations

import os
import re
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
    assert_overlay,
    browser_network_record,
    classify_browser_network,
    focus_with_pointer,
    map_cancellation_policy,
    nearby_road_coordinate,
    overlay_observation,
    point_pixels,
    wait_map,
)

BASE_URL = os.environ.get("PRODUCT_E2E_BASE_URL")
DISPOSABLE = os.environ.get("PRODUCT_E2E_WAYPOINT_CONTEXTS_DISPOSABLE") == "1"
BIXBY = "bixby_creek_bridge_viewpoint"
PFEIFFER = "pfeiffer_big_sur_state_park"


@pytest.mark.skipif(not BASE_URL or not DISPOSABLE, reason="Requires fresh deterministic waypoint-contexts server")
@pytest.mark.parametrize("plan_context", ["original", "accepted_child"])
def test_route_checked_waypoint_contexts_preserve_accepted_plan(tmp_path: Path, plan_context: str) -> None:
    evidence = Path(os.environ.get("PRODUCT_E2E_WAYPOINT_CONTEXTS_EVIDENCE_DIR", tmp_path)) / plan_context
    evidence.mkdir(parents=True, exist_ok=False)
    http, requests, checkpoints, operations, inputs, errors, failures = [], [], {}, [], [], [], []
    control_rows = []
    observed = {}
    map_configuration = None
    map_configuration_http_file = None
    network_classification = None
    phase = "initial"
    with sync_playwright() as runtime:
        browser = runtime.chromium.launch(
            headless=True, **({"executable_path": str(CHROME)} if CHROME.is_file() else {})
        )
        context = browser.new_context(viewport={"width": 1440, "height": 900})
        context.add_init_script(OBSERVE_REAL_MAPS)
        context.add_init_script("""(() => {
          window.__waypointContextCanvasClicks = [];
          window.__waypointCancelClicks = [];
          window.__waypointControls = [];
          const controls = ['[data-draft-waypoint]','[data-relocate-waypoint]','#relocate-custom-waypoint',
            '#remove-custom-waypoint','#exploratory-meaning','#check-road-access','#waypoint-duration-mode',
            '#confirm-map-waypoint-form','.draft-waypoint-marker'];
          for(const type of ['click','change','submit']) document.addEventListener(type,event=>{
            const selector=controls.find(item=>event.target.matches?.(item)||
              (type==='click'&&item!=='#confirm-map-waypoint-form'&&event.target.closest?.(item)));
            if(!selector) return;
            const element=event.target.closest(selector);
            window.__waypointControls.push({type,trusted:event.isTrusted,selector,id:element.id,
              waypoint_id:element.dataset.draftWaypoint||element.dataset.relocateWaypoint||null,
              value:element.value??null,label:element.getAttribute('aria-label')||element.textContent.trim(),at:Date.now()});
          },true);
          document.addEventListener('click', event => {
            if (event.target.closest?.('#cancel-exploratory')) {
              window.__waypointCancelClicks.push({trusted:event.isTrusted,id:'cancel-exploratory',at:Date.now()});
            }
            if (event.target === document.querySelector('#geographic-map canvas')) {
              window.__waypointContextCanvasClicks.push({trusted:event.isTrusted,
                x:event.clientX,y:event.clientY,target:'geographic-map canvas'});
            }
          }, true);
        })();""")
        page = context.new_page()
        page.on("console", lambda item: errors.append(item.text) if item.type == "error" else None)
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.on("requestfailed", lambda request: failures.append(browser_network_record(request, phase)))
        page.on("request", lambda request: requests.append({"method": request.method, "url": request.url}))

        def record(response):
            nonlocal map_configuration, map_configuration_http_file
            if not urlsplit(response.url).path.startswith("/api/"):
                return
            row = {"phase": phase, **_http_record(response)}
            name = f"http_{len(http) + 1:03d}.json"
            _write_json(evidence / name, row)
            http.append(
                {
                    "file": name,
                    "phase": row["phase"],
                    "method": row["method"],
                    "url": row["url"],
                    "status": row["response_status_code"],
                }
            )
            if response.status == 200:
                payload = response.json()
                if response.url == f"{BASE_URL}/api/map/config":
                    assert map_configuration is None or payload == map_configuration
                    map_configuration, map_configuration_http_file = payload, name
                if isinstance(payload, dict):
                    for key in ("workspace", "session"):
                        if key in payload:
                            observed[key] = payload[key]

        page.on("response", record)

        def capture(label, operation=None):
            before = overlay_observation(page, operation)
            assert_overlay(before, operation)
            _capture_ready(page, evidence / f"{label}.png")
            ack, overlay = _map_ack(page), overlay_observation(page, operation)
            assert before == overlay
            assert_overlay(overlay, operation)
            assert overlay["plan_id"] == ack["active_plan_id"] == ack["rendered_map_plan_id"]
            assert overlay["plan_content_hash"] == ack["active_plan_content_hash"]
            assert overlay["geometry_hash"] == ack["rendered_geometry_hash"]
            assert overlay["accepted_marker_ids"] == ack["rendered_marker_ids"]
            checkpoints[label] = {
                "map_ack": ack,
                "overlay": overlay,
                "draft_status": page.locator(".draft-bar").inner_text(),
                "context": page.locator(".map-context-card").inner_text(),
            }
            _write_json(evidence / "checkpoints.json", checkpoints)
            return ack

        def count(endpoint):
            return sum(row["method"] == "POST" and row["url"].endswith(endpoint) for row in requests)

        def native_option(selector, value):
            control = page.locator(selector)
            expect(control).to_be_enabled()
            options = control.locator("option").evaluate_all(
                "rows=>rows.map(row=>({value:row.value,disabled:row.disabled}))"
            )
            position = next(i for i, row in enumerate(options) if row["value"] == value)
            assert not options[position]["disabled"]
            control.click()
            control.press("Home")
            for _ in range(position):
                control.press("ArrowDown")
            control.press("Enter")
            expect(page.locator(selector)).to_have_value(value)

        def visible_route_leg(route_leg_id):
            details = page.locator(".textual-map-alternative")
            if details.get_attribute("open") is None:
                details.locator(":scope > summary").click()
            control = page.locator(f'[data-geo-route-leg="{route_leg_id}"]')
            expect(control).to_be_visible()
            return control

        def visible_text_route_control(selector):
            details = page.locator(".textual-map-alternative")
            if details.get_attribute("open") is None:
                details.locator(":scope > summary").click()
            control = page.locator(selector)
            expect(control).to_be_visible()
            return control

        def control_state():
            ui = page.evaluate("""() => ({
                  title:document.querySelector('#map-context-title')?.textContent,
                  meaning:document.querySelector('#exploratory-meaning')?.value??null,
                  snap:document.querySelector('.snap-preview-state')?.textContent??null,
                  form:!!document.querySelector('#confirm-map-waypoint-form'),
                  duration:document.querySelector('#waypoint-duration-mode')?.value??null,
                  duration_fields:[...document.querySelectorAll('#waypoint-duration-fields input')].map(el=>({
                    id:el.id,disabled:el.disabled,required:el.required,visible:el.getClientRects().length>0})),
                  selected_waypoints:[...document.querySelectorAll('[data-draft-waypoint][aria-pressed=true]')].map(el=>el.dataset.draftWaypoint),
                  selected_markers:[...document.querySelectorAll('.draft-waypoint-marker.is-selected')].map(el=>el.getAttribute('aria-label')),
                  toast:document.querySelector('#toast-region')?.textContent??null
                })""")
            return {"map": _map_ack(page), "draft": observed["session"]["draft"], "ui": ui}

        def begin_control():
            return {
                "state": control_state(),
                "request_start": len(requests),
                "event_start": page.evaluate("window.__waypointControls.length"),
            }

        def finish_control(binding, selector, before, *, response=None, operation=None):
            after = control_state()
            events = page.evaluate("window.__waypointControls")
            event_type = (
                "change"
                if binding in {"exploratory_meaning", "waypoint_duration"}
                else "submit"
                if binding == "confirm_waypoint"
                else "click"
            )
            candidates = [
                (i, event)
                for i, event in enumerate(events)
                if i >= before["event_start"] and event["selector"] == selector and event["type"] == event_type
            ]
            assert candidates and candidates[-1][1]["trusted"] is True
            index, event = candidates[-1]
            row = {
                "binding_id": binding,
                "context": plan_context,
                "selector": selector,
                "event_index": index,
                "event": event,
                "before": before["state"],
                "after": after,
                "request_start": before["request_start"],
                "request_end": len(requests),
                "requests": requests[before["request_start"] :],
                "http": _http_record(response) if response else None,
                "operation": operation,
                "assertions_passed": True,
                "evaluated_repair": False,
            }
            assert row["before"]["map"] == row["after"]["map"] == parent
            control_rows.append(row)
            _write_json(evidence / "waypoint_control_receipts.json", control_rows)
            _write_json(evidence / "waypoint_control_input_log.json", events)

        def wait_selected_marker(operation):
            wait_map(page)
            marker = page.locator(".draft-waypoint-marker.is-selected")
            expect(marker).to_be_visible()
            expect(marker).to_have_attribute("aria-label", re.compile(re.escape(operation["parameters"]["name"])))

        def confirm(label, kind, snap, expected_count):
            control_before = begin_control()
            with page.expect_response(
                lambda response: "/map/snap-previews/" in response.url and response.url.endswith("/confirm")
            ) as completed:
                page.get_by_role("button", name=label, exact=True).click()
            assert completed.value.status == 200, completed.value.text()
            payload = completed.value.json()
            operation = payload["operation"]
            assert payload["feedback_tier"] == "route_checked" and payload["evaluated_repair"] is False
            assert operation["type"] == kind and operation["source"] == "confirmed_map_interaction"
            assert operation["parameters"]["snap_preview_id"] == snap["snap_preview_id"]
            assert operation["parameters"]["raw_coordinate"] == snap["raw_coordinate"]
            assert operation["parameters"]["snapped_coordinate"] == snap["snapped_coordinate"]
            assert operation["parameters"]["selected_access_point"] == snap["selected_access_point"]
            assert operation["parameters"]["affected_route_legs"] == snap["affected_route_legs"]
            assert operation["parameters"]["day"] == 5
            assert payload["session"]["accepted_plan_id"] == parent["active_plan_id"]
            assert payload["session"]["proposal"] is None and len(payload["session"]["draft"]) == expected_count
            assert payload["session"]["draft"][-1] == operation
            expect(page.locator("#inline-preview")).to_be_disabled()
            expect(page.locator("#accept-repair")).to_be_disabled()
            expect(page.locator(".draft-bar")).to_contain_text("evaluated repair is not enabled")
            _write_json(
                evidence / f"{kind}_confirmation.json",
                {"request": completed.value.request.post_data_json, "response": payload},
            )
            operations.append(
                {
                    "context": plan_context,
                    "operation": kind,
                    "variant": "default",
                    "source": operation["source"],
                    "selector": "#confirm-map-waypoint-form button[type=submit]",
                    "feedback_tier": "route_checked",
                    "scope": "route_checked_draft_only",
                    "evaluated_repair": False,
                    "operation_input": completed.value.request.post_data_json,
                    "operation_output": operation,
                    "parent": parent,
                    "assertions_passed": True,
                }
            )
            _write_json(evidence / "operation_assertions.json", operations)
            finish_control(
                "confirm_waypoint",
                "#confirm-map-waypoint-form",
                control_before,
                response=completed.value,
                operation=operation,
            )
            return operation

        def undo(expected, remaining, label, *, selector="#inline-undo"):
            control_before = begin_control() if selector == "#remove-custom-waypoint" else None
            with page.expect_response(lambda response: response.url.endswith("/draft/impact-preview")):
                with page.expect_response(lambda response: response.url.endswith("/draft/undo")) as undone:
                    page.locator(selector).click()
            assert undone.value.status == 200
            result = undone.value.json()
            assert result["undone"] == expected
            assert result["session"]["accepted_plan_id"] == parent["active_plan_id"]
            assert result["session"]["proposal"] is None
            assert result["session"]["draft"] == ([remaining] if remaining else [])
            assert capture(label, remaining) == parent
            _write_json(evidence / f"{label}.json", result)
            if control_before:
                finish_control("remove_custom", selector, control_before, response=undone.value, operation=expected)

        with _preserve_browser_evidence(
            context, page, evidence, {"console_errors": errors, "network_failures": failures}
        ):
            try:
                page.goto(f"{BASE_URL}/app/itinerary", wait_until="domcontentloaded")
                initial = _map_ack(page)
                assert BIXBY in initial["rendered_marker_ids"]
                _capture_ready(page, evidence / "initial_parent.png")
                if plan_context == "accepted_child":
                    phase = "baseline_replace_preview_accept"
                    page.locator('.day-card[data-day="5"]').click()
                    page.locator("#map-edit-button").click()
                    page.locator("#inspect-edit-stop").click()
                    page.locator(".legacy-edit-controls summary").click()
                    page.locator("#registered-candidate").select_option(PFEIFFER)
                    with page.expect_response(lambda response: response.url.endswith("/draft/operations")) as staged:
                        page.locator("#draft-replace").click()
                    assert staged.value.status == 200
                    page.locator("#close-map-edit").click()
                    with page.expect_response(
                        lambda response: response.url.endswith("/preview"), timeout=90_000
                    ) as previewed:
                        page.locator("#inline-preview").click()
                    assert previewed.value.status == 200
                    proposal = previewed.value.json()["proposal"]
                    assert (
                        proposal["state"] == "eligible" and proposal["runtime_execution"]["evaluation_executed"] is True
                    )
                    expect(page.locator(".map-stage")).to_have_attribute(
                        "data-preview-state", "PREVIEW_RENDERED", timeout=45_000
                    )
                    preview_ack = _map_ack(page)
                    assert preview_ack["rendered_map_plan_id"] == proposal["child_plan_id"]
                    assert preview_ack["rendered_geometry_hash"] == _independent_geometry_hash(
                        page, proposal["geography_plan"]
                    )
                    _capture_ready(page, evidence / "baseline_preview_ready.png")
                    expect(page.locator("#accept-repair")).to_be_enabled()
                    with page.expect_response(
                        lambda response: response.url.endswith("/accept"), timeout=60_000
                    ) as accepted:
                        page.locator("#accept-repair").click()
                        expect(page.get_by_role("dialog", name="Confirm trip decision", exact=True)).to_be_visible()
                        expect(page.locator("#acceptance-message")).to_have_text(
                            "Accept this independently evaluated repair as the durable trip plan?"
                        )
                        page.get_by_role("dialog", name="Confirm trip decision", exact=True).get_by_role(
                            "button", name="Accept repair", exact=True
                        ).click()
                    assert accepted.value.status == 200
                    assert accepted.value.json()["result"]["decision"]["selected_plan_id"] == proposal["child_plan_id"]
                    _write_json(evidence / "baseline_preview_response.json", previewed.value.json())
                    _write_json(evidence / "baseline_acceptance_response.json", accepted.value.json())
                    page.reload(wait_until="domcontentloaded")
                    accepted_ack = _map_ack(page)
                    assert accepted_ack["active_plan_id"] == proposal["child_plan_id"]
                    assert accepted_ack["active_plan_content_hash"] == proposal["child_plan_content_hash"]
                    assert accepted_ack["rendered_geometry_hash"] == preview_ack["rendered_geometry_hash"]
                    assert set(accepted_ack["rendered_marker_ids"]) == (
                        set(initial["rendered_marker_ids"]) - {BIXBY}
                    ) | {PFEIFFER}
                parent = capture("context_baseline")
                workspace = observed["workspace"]
                geography = next(
                    plan for plan in workspace["geography"]["plans"] if plan["plan_id"] == parent["active_plan_id"]
                )
                incoming = next(
                    feature["properties"]
                    for feature in geography["validated_legs"]["features"]
                    if feature["properties"]["destination_id"] == (BIXBY if plan_context == "original" else PFEIFFER)
                )
                capabilities = workspace["map_edit_capabilities"]
                kinds = ["add_route_waypoint", "add_custom_waypoint"]
                for kind in kinds + ["relocate_custom_waypoint"]:
                    assert capabilities["operations"][kind]["enabled"] is True
                    assert capabilities["operations"][kind]["preview_executable"] is False
                    assert capabilities["operations"][kind]["evaluated_repair"] is False
                _write_json(
                    evidence / "capability_snapshot.json",
                    {
                        "schema_version": "runtime-capability-snapshot-v1",
                        "context": plan_context,
                        "session_id": observed["session"]["session_id"],
                        "plan_id": parent["active_plan_id"],
                        "plan_content_hash": parent["active_plan_content_hash"],
                        "capabilities": {"map_edit_capabilities": capabilities},
                    },
                )
                baseline_preview_count, baseline_accept_count = count("/preview"), count("/accept")
                for kind in kinds:
                    phase = kind
                    page.locator('.day-card[data-day="5"]').click()
                    route_leg_control = visible_route_leg(incoming["route_leg_id"])
                    with page.expect_response(lambda response: response.url.endswith("/selection")) as selected:
                        route_leg_control.click()
                    assert selected.value.status == 200
                    page.locator("#map-edit-button").click()
                    # This is an exploratory raw point already used by the
                    # existing public fixture, not claimed as road-valid input.
                    raw_coordinate = [-121.94165, 36.51710]
                    pixels = focus_with_pointer(page, raw_coordinate)
                    if kind == kinds[0]:
                        phase = "exploratory_cancel"
                        before_cancel = _map_ack(page)
                        request_start = len(requests)
                        page.mouse.click(pixels["x"], pixels["y"])
                        exploratory_click = page.evaluate("window.__waypointContextCanvasClicks.at(-1)")
                        expect(page.locator("#map-context-title")).to_have_text("Unconfirmed map point")
                        expect(page.locator("#cancel-exploratory")).to_be_enabled()
                        page.locator("#cancel-exploratory").click()
                        expect(page.locator("#cancel-exploratory")).to_have_count(0)
                        expect(page.locator("#map-edit-button")).to_be_focused()
                        wait_map(page)
                        after_cancel = _map_ack(page)
                        assert after_cancel == before_cancel
                        assert observed["session"]["draft"] == []
                        request_window = requests[request_start:]
                        assert all(row["method"] == "GET" for row in request_window)
                        cancel_click = page.evaluate("window.__waypointCancelClicks.at(-1)")
                        assert exploratory_click["trusted"] is True and cancel_click["trusted"] is True
                        _write_json(
                            evidence / "exploratory_cancel.json",
                            {
                                "schema_version": "waypoint-exploratory-cancel-v1",
                                "context": plan_context,
                                "raw_coordinate": raw_coordinate,
                                "pixels": pixels,
                                "exploratory_click": exploratory_click,
                                "cancel_click": cancel_click,
                                "before_map": before_cancel,
                                "after_map": after_cancel,
                                "requests": request_window,
                                "request_start": request_start,
                                "request_end": len(requests),
                                "draft": [],
                                "cancel_control_removed": True,
                                "focus_returned_to_edit": True,
                                "assertions_passed": True,
                            },
                        )
                        _write_json(
                            evidence / "exploratory_input_log.json",
                            page.evaluate("""() => ({
                            canvas_clicks:window.__waypointContextCanvasClicks,
                            cancel_clicks:window.__waypointCancelClicks
                        })"""),
                        )
                        phase = kind
                        pixels = focus_with_pointer(page, raw_coordinate)
                    inputs.append(
                        {
                            "action": "native_canvas_click",
                            "operation": kind,
                            "route_leg": incoming,
                            "raw_coordinate": raw_coordinate,
                            "pixels": pixels,
                        }
                    )
                    page.mouse.click(pixels["x"], pixels["y"])
                    click = page.evaluate("window.__waypointContextCanvasClicks.at(-1)")
                    assert click and click["trusted"] is True
                    assert abs(click["x"] - pixels["x"]) <= 1 and abs(click["y"] - pixels["y"]) <= 1
                    inputs[-1]["trusted_dom_event"] = click
                    expect(page.locator("#map-context-title")).to_have_text("Unconfirmed map point")
                    expect(page.locator(".draft-bar strong")).to_have_text("0 draft changes")
                    meaning = "route_waypoint" if kind == "add_route_waypoint" else "possible_stop"
                    control_before = begin_control()
                    native_option("#exploratory-meaning", meaning)
                    finish_control("exploratory_meaning", "#exploratory-meaning", control_before)
                    control_before = begin_control()
                    with page.expect_response(
                        lambda response: response.url.endswith("/map/snap-preview"), timeout=60_000
                    ) as snapped:
                        page.locator("#check-road-access").click()
                    assert snapped.value.status == 200, snapped.value.text()
                    snap = snapped.value.json()
                    assert snap["validation_state"] == "route_checked" and snap["persisted"] is False
                    assert snap["operation_intent"] == kind
                    expect(page.locator(".draft-bar strong")).to_have_text("0 draft changes")
                    expect(page.locator("#confirm-map-waypoint-form")).to_be_visible()
                    finish_control("check_road_access", "#check-road-access", control_before, response=snapped.value)
                    name = (
                        f"{plan_context} real-pointer {'via point' if kind == 'add_route_waypoint' else 'custom stop'}"
                    )
                    page.locator("#waypoint-name").fill(name)
                    if kind == "add_custom_waypoint":
                        page.locator("#waypoint-role").select_option("scenic_stop")
                        for mode in ("range", "exact"):
                            control_before = begin_control()
                            native_option("#waypoint-duration-mode", mode)
                            finish_control("waypoint_duration", "#waypoint-duration-mode", control_before)
                        page.locator("#waypoint-duration-preferred").fill("45")
                    else:
                        assert (
                            page.locator("#waypoint-role").count()
                            == page.locator("#waypoint-duration-mode").count()
                            == 0
                        )
                    if snap["confirmation_required"]:
                        page.locator("#snap-warning-ack").check()
                    added = confirm(
                        "Add route waypoint" if kind == "add_route_waypoint" else "Add custom stop", kind, snap, 1
                    )
                    assert added["parameters"]["name"] == name
                    if kind == "add_route_waypoint":
                        assert (
                            added["parameters"]["role"] == "route_waypoint" and added["parameters"]["duration"] is None
                        )
                    else:
                        assert added["parameters"]["role"] == "scenic_stop"
                        assert added["parameters"]["duration"] == {
                            "mode": "exact",
                            "preferred_minutes": 45,
                            "minimum_minutes": 45,
                            "maximum_minutes": 45,
                        }
                    assert capture(f"{kind}_staged", added) == parent
                    if kind == "add_custom_waypoint":
                        phase = "relocate_custom_waypoint"
                        point = added["parameters"]["snapped_coordinate"]
                        for binding, selector in (
                            ("select_draft_waypoint", f'[data-draft-waypoint="{added["target"]}"]'),
                            ("relocate_draft_waypoint", f'[data-relocate-waypoint="{added["target"]}"]'),
                        ):
                            route_leg_control = visible_route_leg(incoming["route_leg_id"])
                            with page.expect_response(lambda response: response.url.endswith("/draft/impact-preview")):
                                route_leg_control.click()
                            expect(page.locator(f'[data-draft-waypoint="{added["target"]}"]')).to_have_attribute(
                                "aria-pressed", "false"
                            )
                            control_before = begin_control()
                            visible_text_route_control(selector).click()
                            expect(page.locator(f'[data-draft-waypoint="{added["target"]}"]')).to_have_attribute(
                                "aria-pressed", "true"
                            )
                            expect(page.locator("#map-context-title")).to_have_text(added["parameters"]["name"])
                            wait_selected_marker(added)
                            if binding == "relocate_draft_waypoint":
                                expect(page.locator("#toast-region")).to_contain_text(
                                    "Drag the selected custom waypoint marker"
                                )
                            finish_control(
                                binding,
                                "[data-draft-waypoint]"
                                if binding == "select_draft_waypoint"
                                else "[data-relocate-waypoint]",
                                control_before,
                                operation=added,
                            )
                        route_leg_control = visible_route_leg(incoming["route_leg_id"])
                        with page.expect_response(lambda response: response.url.endswith("/draft/impact-preview")):
                            route_leg_control.click()
                        focus_with_pointer(page, [point["longitude"], point["latitude"]])
                        control_before = begin_control()
                        page.locator(".draft-waypoint-marker").click()
                        expect(page.locator("#map-context-title")).to_have_text(added["parameters"]["name"])
                        wait_selected_marker(added)
                        finish_control("select_draft_marker", ".draft-waypoint-marker", control_before, operation=added)
                        control_before = begin_control()
                        page.locator("#relocate-custom-waypoint").click()
                        expect(page.locator("#toast-region")).to_contain_text(
                            "Drag the selected custom waypoint marker"
                        )
                        wait_selected_marker(added)
                        finish_control("relocate_custom", "#relocate-custom-waypoint", control_before, operation=added)
                        target = nearby_road_coordinate(added)
                        # Entering relocation remounts the map at the full-trip
                        # bounds. Re-center the draggable marker itself so its
                        # trusted mouse-down cannot land outside the clipped map
                        # container; the nearby route coordinate then remains a
                        # visible, unobstructed release target.
                        focus_with_pointer(page, [point["longitude"], point["latitude"]])
                        wait_map(page)
                        destination = point_pixels(page, target)
                        bounds = page.locator(".draft-waypoint-marker").bounding_box()
                        assert bounds is not None
                        start = {"x": bounds["x"] + bounds["width"] / 2, "y": bounds["y"] + bounds["height"] / 2}
                        assert page.evaluate(
                            """p => document.elementFromPoint(p.x,p.y)?.closest('.draft-waypoint-marker')
                          === document.querySelector('.draft-waypoint-marker')""",
                            start,
                        )
                        assert page.evaluate(
                            "p => document.elementFromPoint(p.x,p.y) === document.querySelector('#geographic-map canvas')",
                            destination,
                        )
                        inputs.append(
                            {
                                "action": "native_marker_drag",
                                "waypoint_id": added["target"],
                                "start": start,
                                "target_coordinate_from_actual_road_geometry": target,
                                "destination": destination,
                                "overlay_release_tested": False,
                            }
                        )
                        with page.expect_response(
                            lambda response: response.url.endswith("/map/snap-preview"), timeout=60_000
                        ) as moved:
                            page.mouse.move(**start)
                            page.mouse.down()
                            try:
                                page.mouse.move(destination["x"], destination["y"], steps=20)
                            finally:
                                page.mouse.up()
                        assert moved.value.status == 200
                        movement = moved.value.json()
                        assert movement["operation_intent"] == "relocate_custom_waypoint"
                        assert movement["validation_state"] == "route_checked" and movement["persisted"] is False
                        gesture = page.evaluate("window.__e2eMarkerGestures.at(-1)")
                        assert gesture[0]["type"] == "mousedown" and gesture[-1]["type"] == "mouseup"
                        assert len(gesture) >= 3 and all(row["trusted"] for row in gesture)
                        assert any(row["type"] == "mousemove" and row["buttons"] == 1 for row in gesture)
                        assert abs(gesture[0]["x"] - start["x"]) <= 1 and abs(gesture[0]["y"] - start["y"]) <= 1
                        assert (
                            abs(gesture[-1]["x"] - destination["x"]) <= 1
                            and abs(gesture[-1]["y"] - destination["y"]) <= 1
                        )
                        assert (
                            (gesture[-1]["x"] - gesture[0]["x"]) ** 2 + (gesture[-1]["y"] - gesture[0]["y"]) ** 2
                        ) ** 0.5 > 8
                        inputs[-1]["trusted_dom_events"] = gesture
                        expect(page.locator("#map-context-title")).to_have_text("Unconfirmed relocation")
                        expect(page.locator(".draft-bar strong")).to_have_text("1 draft change")
                        if movement["confirmation_required"]:
                            page.locator("#snap-warning-ack").check()
                        relocated = confirm("Confirm relocation", "relocate_custom_waypoint", movement, 2)
                        assert relocated["target"] == added["target"]
                        for field in ("name", "role", "duration", "day", "insertion"):
                            assert relocated["parameters"][field] == added["parameters"][field]
                        assert (
                            relocated["parameters"]["snapped_coordinate"] != added["parameters"]["snapped_coordinate"]
                        )
                        assert capture("custom_relocated", relocated) == parent
                        undo(relocated, added, "undo_custom_relocation")
                        visible_text_route_control(f'[data-draft-waypoint="{added["target"]}"]').click()
                        expect(page.locator("#remove-custom-waypoint")).to_be_enabled()
                        page.once("dialog", lambda dialog: dialog.accept())
                        undo(added, None, f"undo_{kind}", selector="#remove-custom-waypoint")
                    else:
                        undo(added, None, f"undo_{kind}")
                    assert count("/preview") == baseline_preview_count and count("/accept") == baseline_accept_count
                phase = "final_reload"
                _write_json(evidence / "waypoint_control_input_log.json", page.evaluate("window.__waypointControls"))
                page.reload(wait_until="domcontentloaded")
                assert capture("parent_after_final_reload") == parent
                assert observed["session"]["draft"] == [] and observed["session"]["proposal"] is None
                assert observed["session"]["accepted_plan_id"] == parent["active_plan_id"]
                asset_sources = page.evaluate(
                    """() => {
                  const map = """
                    + CURRENT_MAP
                    + """;
                  const style=map.getStyle();
                  return {container_id:map.getContainer().id,
                    map_dataset:{...map.getContainer().dataset},
                    sprite:style.sprite??null,
                    vector_sources:Object.fromEntries(Object.entries(style.sources)
                      .filter(([,source])=>source.type==='vector'))};
                }"""
                )
                assert asset_sources["container_id"] == "geographic-map"
                network_policy = map_cancellation_policy(
                    map_configuration, {"sources": asset_sources["vector_sources"], "sprite": asset_sources["sprite"]}
                )
                network_classification = classify_browser_network(failures, network_policy)
                assert errors == [] and network_classification["blocking_failure_indexes"] == []
                assert not any(row["status"] >= 400 for row in http)
                assert count("/preview") == count("/accept") == (1 if plan_context == "accepted_child" else 0)
                assert not any(row["url"].endswith("/copilot/messages") for row in requests)
                _write_json(
                    evidence / "browser_assertions.json",
                    {
                        "schema_version": "waypoint-context-browser-v1",
                        "context": plan_context,
                        "parent": parent,
                        "operations": operations,
                        "checkpoints": checkpoints,
                        "native_pointer_inputs": inputs,
                        "route_checked_draft_only": True,
                        "overlay_regression_claimed": False,
                        "evaluated_waypoint_repair_claimed": False,
                        "assertions_passed": True,
                    },
                )
                _write_json(
                    evidence / "llm_io.json",
                    {
                        "provider_call_count": 0,
                        "llm_input": None,
                        "llm_output": None,
                        "reason": "Real UI draft/OSRM road checks and deterministic baseline repair; no chat/provider call.",
                    },
                )
                _write_json(
                    evidence / "map_asset_sources.json",
                    {
                        "schema_version": "real-map-asset-sources-v1",
                        "scope": "Read-only public getStyle() vector sources and actual ready container dataset",
                        "phase": phase,
                        "context": plan_context,
                        "map_configuration_http_file": map_configuration_http_file,
                        **asset_sources,
                    },
                )
                _write_json(
                    evidence / "network_classification.json",
                    {
                        "map_configuration_http_file": map_configuration_http_file,
                        "rendered_map_asset_sources_file": "map_asset_sources.json",
                        **network_classification,
                    },
                )
            finally:
                _write_json(evidence / "waypoint_control_receipts.json", control_rows)
                if phase != "final_reload":
                    _write_json(
                        evidence / "waypoint_control_input_log.json", page.evaluate("window.__waypointControls || []")
                    )
                _write_json(evidence / "browser_requests.json", requests)
                _write_json(evidence / "http_index.json", http)
                _write_json(evidence / "native_pointer_inputs.json", inputs)
                _write_json(evidence / "console_errors.json", errors)
                _write_json(evidence / "network_failures.json", failures)
        browser.close()
