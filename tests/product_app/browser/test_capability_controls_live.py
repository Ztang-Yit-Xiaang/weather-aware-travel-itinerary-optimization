"""Real controls prove draft-only semantics, disabled options, and family rejection.

One ordinary disposable server, two ordered cases. The accepted-child case builds
its baseline through actual Replace, Preview, Accept, and Reload controls.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from urllib.parse import urlsplit

import pytest

pytest.importorskip("playwright.sync_api")
from test_e2e1_live_copilot_chat import (
    CHROME,
    _capture_ready,
    _http_record,
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

ROOT = Path(__file__).resolve().parents[3]
BASE_URL = os.environ.get("PRODUCT_E2E_BASE_URL")
TARGET = "griffith_observatory"
ROUTE_LEG = "leg_portola_hotel_monterey_stearns_wharf"


def map_identity(ack):
    return {key: value for key, value in ack.items() if key != "preview_state"}


@pytest.mark.skipif(
    not BASE_URL or os.environ.get("PRODUCT_E2E_CAPABILITY_CONTROLS_DISPOSABLE") != "1",
    reason="Requires a fresh deterministic capability-controls server; never attach to user state",
)
@pytest.mark.parametrize("plan_context", ["original", "accepted_child"])
def test_real_control_variants_preserve_draft_only_and_disabled_semantics(tmp_path, plan_context):
    sys.path.insert(0, str(ROOT / "scripts"))
    from e2e1_interaction_coverage import required_variants

    evidence = Path(os.environ.get("PRODUCT_E2E_CAPABILITY_CONTROLS_EVIDENCE_DIR", tmp_path)) / plan_context
    evidence.mkdir(parents=True, exist_ok=False)
    registry = json.loads((ROOT / "configs/e2e1_interaction_coverage.json").read_text(encoding="utf-8"))
    http, operations, inventory, disabled, combinations, errors, failures = [], [], [], [], [], [], []
    general_inventory, general_journeys, prior_general_events = [], [], []
    responsive_journeys, control_actions = [], []
    bindings = registry["source_bindings"] + registry["native_controls"]
    observed = {}
    map_configuration = None
    map_configuration_http_file = None
    network_policy = None
    phase = "initial"
    with sync_playwright() as runtime:
        browser = runtime.chromium.launch(
            headless=True, **({"executable_path": str(CHROME)} if CHROME.is_file() else {})
        )
        context = browser.new_context(viewport={"width": 1440, "height": 900})
        context.add_init_script(OBSERVE_REAL_MAPS)
        from control_action_fixture import (
            OBSERVE_CONTROL_ACTIONS,
            after_action,
            before_action,
            select_native_option,
        )

        context.add_init_script(OBSERVE_CONTROL_ACTIONS)
        # Read-only capture records real trusted DOM input. Inventory presence is
        # kept separate from a completed semantic journey and never credits it.
        context.add_init_script(
            """(() => {
          const bindings = __BINDINGS__;
          window.__controlReceiptEvents = [];
          for (const type of ['click','change','submit']) document.addEventListener(type,event => {
            const target = event.target;
            if (!(target instanceof Element)) return;
            window.__controlReceiptEvents.push({type, trusted:event.isTrusted, at:Date.now(),
              path:location.pathname, target_tag:target.tagName.toLowerCase(), target_id:target.id,
              ancestors:[target,...Array.from((function*(){let p=target.parentElement;
                while(p){yield p;p=p.parentElement;}})())].map(e=>({tag:e.tagName.toLowerCase(),id:e.id,
                  classes:[...e.classList],attributes:Object.fromEntries([...e.attributes].map(a=>[a.name,a.value]))})),
              matches:bindings.flatMap(binding=>binding.event===type ? binding.selectors.filter(selector=>target.closest(selector))
                .map(selector=>({binding_id:binding.binding_id,selector})) : []),
              text:target.textContent.trim(), value:target.value ?? null});
          },true);
        })();""".replace("__BINDINGS__", json.dumps(bindings))
        )
        page = context.new_page()
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.on("console", lambda message: errors.append(message.text) if message.type == "error" else None)
        page.on("requestfailed", lambda request: failures.append(browser_network_record(request, phase)))

        def record(response):
            nonlocal map_configuration, map_configuration_http_file
            if response.status >= 400:
                failures.append(browser_network_record(response.request, phase, http_status=response.status))
            if not urlsplit(response.url).path.startswith("/api/"):
                return
            row = {"phase": phase, **_http_record(response)}
            name = f"http_{len(http) + 1:03d}.json"
            _write_json(evidence / name, row)
            http.append(
                {
                    "file": name,
                    "phase": phase,
                    "method": row["method"],
                    "url": row["url"],
                    "status": row["response_status_code"],
                }
            )
            if row["response_status_code"] == 200:
                payload = json.loads(row["response_body_utf8"])
                if response.url == f"{BASE_URL}/api/map/config":
                    assert map_configuration is None or payload == map_configuration, (
                        "Map configuration changed during the journey"
                    )
                    map_configuration, map_configuration_http_file = payload, name
                if isinstance(payload, dict):
                    for key in ("workspace", "session"):
                        if key in payload:
                            observed[key] = payload[key]
                    if row["url"].endswith("/draft/impact-preview"):
                        observed["impact"] = payload

        page.on("response", record)

        def expand(details):
            if details.get_attribute("open") is None:
                details.locator(":scope > summary").click()

        def wait_selected(response):
            assert response.status == 200, response.text()
            session = response.json()["session"]
            for _ in range(600):
                if observed.get("impact", {}).get("session_revision") == session["revision"]:
                    break
                page.wait_for_timeout(50)
            assert observed.get("impact", {}).get("session_revision") == session["revision"]
            expect(page.locator(f'.day-card[data-day="{session["selected_day"]}"]')).to_have_attribute(
                "aria-pressed", "true"
            )
            assert map_identity(_map_ack(page)) == map_identity(parent)

        def open_stop(day=7):
            if page.locator("#map-edit-dialog").is_visible():
                page.locator("#close-map-edit").click()
            with page.expect_response(
                lambda response: (
                    response.url.endswith("/selection") and response.request.post_data_json.get("selected_day") == day
                )
            ) as selected:
                page.locator(f'.day-card[data-day="{day}"]').click()
            wait_selected(selected.value)
            page.locator("#map-edit-button").click()
            _map_ack(page)
            page.locator("#inspect-edit-stop").click()
            expect(page.locator('[data-editor-kind="stop"]')).to_be_visible()

        def form(kind):
            selected = page.locator(f'[data-typed-operation="{kind}"]')
            details = selected.locator("xpath=ancestor::details[1]")
            if details.count():
                expand(details)
            expect(selected).to_be_visible()
            return selected

        def legacy():
            expand(page.locator(".legacy-edit-controls"))

        def count_mutations(suffix):
            return sum(row["method"] == "POST" and row["url"].endswith(suffix) for row in http)

        def take_inventory(label):
            take_general_inventory(label)
            for mapping in registry["rows"]:
                if plan_context not in mapping["contexts"]:
                    continue
                query = ", ".join(part.strip() for part in mapping["selector"].split(";"))
                controls = page.locator(query).evaluate_all("""elements => elements.map(element => ({
                  tag:element.tagName.toLowerCase(),id:element.id,disabled:element.matches(':disabled'),
                  visible:!!(element.offsetWidth || element.offsetHeight || element.getClientRects().length),
                  label:element.getAttribute('aria-label') || element.textContent.trim(),title:element.title,
                  form_operation:element.closest('[data-typed-operation]')?.dataset.typedOperation || null,
                  options:[...element.closest('form')?.querySelectorAll('select') || []].map(select => ({
                    name:select.name,value:select.value,options:[...select.options].map(option => ({
                      value:option.value,label:option.textContent,disabled:option.disabled,selected:option.selected}))}))
                }))""")
                inventory.append(
                    {
                        "phase": label,
                        "context": plan_context,
                        "action_id": mapping["action_id"],
                        "operation": mapping["operation"],
                        "capability_group": mapping["capability_group"],
                        "selector": mapping["selector"],
                        "actual_dom_query": query,
                        "controls": controls,
                        "observed_count": len(controls),
                    }
                )
            _write_json(
                evidence / "runtime_control_inventory.json",
                {
                    "schema_version": "runtime-control-inventory-v1",
                    "context": plan_context,
                    "observations": inventory,
                    "absent_controls_are_not_claimed_tested": True,
                },
            )

        def take_general_inventory(label):
            for mapping in bindings:
                for selector in mapping["selectors"]:
                    controls = page.locator(selector).evaluate_all("""elements => elements.map(element => ({
                      tag:element.tagName.toLowerCase(),id:element.id,
                      disabled:element.matches(':disabled'), aria_disabled:element.getAttribute('aria-disabled'),
                      visible:!!(element.offsetWidth || element.offsetHeight || element.getClientRects().length)
                        && getComputedStyle(element).visibility !== 'hidden',
                      label:element.getAttribute('aria-label') || element.textContent.trim(),
                      attributes:Object.fromEntries([...element.attributes].map(a=>[a.name,a.value]))
                    }))""")
                    general_inventory.append(
                        {
                            "phase": label,
                            "context": plan_context,
                            "binding_id": mapping["binding_id"],
                            "selector": selector,
                            "observed_count": len(controls),
                            "controls": controls,
                            "path": urlsplit(page.url).path,
                            "viewport": page.viewport_size,
                        }
                    )
            _write_json(
                evidence / "general_control_inventory.json",
                {
                    "schema_version": "general-control-inventory-v1",
                    "context": plan_context,
                    "observations": general_inventory,
                    "presence_is_not_execution": True,
                    "absent_is_not_unsupported": True,
                },
            )

        def settled_surface():
            route = urlsplit(page.url).path
            if route in {"/app/itinerary", "/app/map"}:
                _map_ack(page)
            if route == "/app/compare":
                expect(page.locator("[data-compare-workspace]")).to_be_visible(timeout=45_000)
                page.wait_for_function(
                    """() => {
                  const maps=[...document.querySelectorAll('[data-compare-map-canvas]')].filter(e=>e.getClientRects().length);
                  return maps.length>0 && maps.every(e=>e.dataset.mapReady==='true' && Number(e.dataset.mapTileCoverage)>=.95);
                }""",
                    timeout=45_000,
                )

        def compact_workspace_receipt(binding_id, workspace):
            if binding_id.startswith("evidence_back_"):
                return {"alternative_evidence": workspace["alternative_evidence"]}
            if binding_id not in {
                "select_day",
                "text_select_stop",
                "text_select_leg",
                "candidate_select",
                "compare_select",
            }:
                return None
            plans = []
            for plan in workspace["geography"]["plans"]:
                plans.append({key: plan[key] for key in ("plan_id", "content_hash", "stops", "validated_legs")})
            return {
                "geography": {"plans": plans},
                "alternatives": [
                    {"plan_id": row["plan_id"], "plan_content_hash": row["plan_content_hash"]}
                    for row in workspace["alternatives"]
                ],
                "draft_capabilities": {"candidate_choices": workspace["draft_capabilities"]["candidate_choices"]},
            }

        def general_action(
            binding_id, selector, outcome, *, select_value=None, variant="default", target_selector=None
        ):
            nonlocal phase
            phase = "general_" + binding_id + "_" + str(len(general_journeys))
            control = page.locator(target_selector or selector)
            expect(control).to_be_visible()
            expect(control).to_be_enabled()
            take_general_inventory(phase + "_before")
            event_start = page.evaluate("window.__controlReceiptEvents.length")
            http_start = len(http)
            input_attributes = control.evaluate("e=>Object.fromEntries([...e.attributes].map(a=>[a.name,a.value]))")
            session_before = json.loads(json.dumps(observed["session"]))
            workspace_before = compact_workspace_receipt(binding_id, observed["workspace"])
            evidence_before = (
                page.locator("#workspace-content").evaluate("""e=>({
              heading:e.querySelector('h2')?.textContent,
              evidence:[...e.querySelectorAll('[data-evidence-plan-id]')].map(p=>({...p.dataset})),
              code:e.querySelector('code')?.textContent??null
            })""")
                if binding_id.startswith("evidence_back_")
                else None
            )
            selection = None
            selected_plan = None
            if binding_id in {"compare_choose", "context_choose"}:
                selected_id = page.locator('[data-compare-option][aria-pressed="true"]').get_attribute(
                    "data-compare-option"
                )
                selected_plan = next(
                    plan for plan in observed["workspace"]["geography"]["plans"] if plan["plan_id"] == selected_id
                )
            if binding_id in {
                "select_day",
                "text_select_stop",
                "text_select_leg",
                "candidate_select",
                "compare_select",
            }:
                with page.expect_response(
                    lambda response: response.url.endswith("/selection") and response.request.method == "POST"
                ) as changed:
                    if select_value is None:
                        control.click()
                    elif binding_id == "candidate_select":
                        options = control.locator("option").evaluate_all("elements=>elements.map(option=>option.value)")
                        assert options[-1] == select_value, (
                            "Case requires the alternate candidate to be the final native option"
                        )
                        control.press("End")
                    else:
                        control.select_option(select_value)
                assert changed.value.status == 200, changed.value.text()
                selection = {"request": changed.value.request.post_data_json, "response": changed.value.json()}
                if binding_id != "compare_select":
                    wait_selected(changed.value)
                else:
                    expect(page.locator(target_selector)).to_have_attribute("aria-pressed", "true")
                assert observed["session"] == selection["response"]["session"]
            elif select_value is None:
                control.click()
            elif binding_id == "destination_day_select":
                options = control.locator("option").evaluate_all("elements=>elements.map(option=>option.value)")
                assert options[-1] == select_value, "Case requires the destination day to be the final native option"
                control.press("End")
            else:
                control.select_option(select_value)
            if "path" in outcome:
                expect(page).to_have_url(f"{BASE_URL}{outcome['path']}")
            if "selector" in outcome:
                result = page.locator(outcome["selector"])
                if "visible" in outcome:
                    (expect(result).to_be_visible if outcome["visible"] else expect(result).to_be_hidden)()
                if "attribute" in outcome:
                    expect(result).to_have_attribute(outcome["attribute"], outcome["value"])
                if "text" in outcome:
                    expect(result).to_contain_text(outcome["text"])
                if "value_text" in outcome:
                    expect(result).to_have_value(outcome["value_text"])
                if "open" in outcome:
                    assert result.evaluate("e=>(e.closest('details')||e).open") is outcome["open"]
            settled_surface()
            take_general_inventory(phase + "_after")
            events = page.evaluate("start=>window.__controlReceiptEvents.slice(start)", event_start)
            assert any(event["trusted"] for event in events)
            assert observed["session"]["accepted_plan_id"] == parent["active_plan_id"]
            assert observed["session"]["draft"] == []
            selected_map = _map_ack(page) if selected_plan else None
            if selected_plan:
                assert selected_map["rendered_map_plan_id"] == selected_plan["plan_id"]
                assert selected_map["active_plan_content_hash"] == selected_plan["content_hash"]
            dom_result = (
                page.locator(outcome["selector"]).evaluate("""e=>({
              attributes:Object.fromEntries([...e.attributes].map(a=>[a.name,a.value])),
              text:e.textContent,value:e.value??null,visible:!!e.getClientRects().length && getComputedStyle(e).visibility!=='hidden',
              open:(e.closest('details')||e).open??null
            })""")
                if "selector" in outcome
                else None
            )
            general_journeys.append(
                {
                    "binding_id": binding_id,
                    "selector": selector,
                    "context": plan_context,
                    "phase": phase,
                    "variant": variant,
                    "input": {"select_value": select_value},
                    "trusted_events": events,
                    "outcome": outcome,
                    "dom_result": dom_result,
                    "input_attributes": input_attributes,
                    "session_before": session_before,
                    "workspace_before": workspace_before,
                    "evidence_before": evidence_before,
                    "selection": selection,
                    "session_after": observed["session"],
                    "selection_map": _map_ack(page) if selection and binding_id != "compare_select" else None,
                    "compare_maps": page.locator("[data-compare-map-canvas]").evaluate_all(
                        "elements=>elements.map(e=>({...e.dataset}))"
                    )
                    if binding_id == "compare_select"
                    else [],
                    "path": urlsplit(page.url).path,
                    "http_files": [row["file"] for row in http[http_start:]],
                    "accepted_plan_id": parent["active_plan_id"],
                    "accepted_plan_content_hash": parent["active_plan_content_hash"],
                    "selected_plan": selected_plan,
                    "selected_map": selected_map,
                    "compare_visible_panes": page.locator("[data-compare-map-pane]").evaluate_all(
                        "elements=>elements.filter(e=>e.getClientRects().length).map(e=>e.dataset.compareMapPane)"
                    ),
                    "draft": [],
                    "assertions_passed": True,
                }
            )
            _write_json(
                evidence / "general_control_journeys.json",
                {
                    "schema_version": "general-control-journeys-v1",
                    "context": plan_context,
                    "journeys": general_journeys,
                    "inventory_is_not_execution": True,
                },
            )

        def general_navigation():
            # Inspect Evidence before Compare can choose its default. Original
            # state has no selected option; accepted state has exact fallback evidence.
            general_action("navigate_global", '.nav-item[data-route="/app/evidence"]', {"path": "/app/evidence"})
            initial_back = (
                "/app/compare" if "Compare" in page.locator("#evidence-back").inner_text() else "/app/itinerary"
            )
            general_action(
                "evidence_back_unavailable" if initial_back.endswith("compare") else "evidence_back_exact",
                "#evidence-back",
                {"path": initial_back},
            )
            if initial_back.endswith("compare"):
                general_action("compare_back", "#compare-back", {"path": "/app/itinerary"})
            assert page.locator(".textual-map-alternative").get_attribute("open") is None
            general_action(
                "text_route_details",
                ".textual-map-alternative > summary",
                {"selector": ".text-route-summary", "open": True},
                variant="open",
            )
            general_action(
                "text_route_details",
                ".textual-map-alternative > summary",
                {"selector": ".text-route-summary", "open": False},
                variant="closed",
            )
            general_action(
                "select_day",
                ".day-card",
                {"selector": '.day-card[data-day="6"]', "attribute": "aria-pressed", "value": "true"},
                target_selector='.day-card[data-day="6"]',
            )
            expand(page.locator(".textual-map-alternative"))
            general_action(
                "text_select_stop",
                "[data-geo-stop]",
                {"selector": '[data-geo-stop="stearns_wharf"]', "attribute": "aria-pressed", "value": "true"},
                target_selector='[data-geo-stop="stearns_wharf"]',
            )
            general_action(
                "map_edit",
                "#map-edit-button",
                {"selector": "#map-edit-button", "attribute": "aria-pressed", "value": "true"},
            )
            general_action(
                "inspect_stop", "#inspect-edit-stop", {"selector": '[data-editor-kind="stop"]', "visible": True}
            )
            legacy()
            general_action(
                "candidate_select",
                "#registered-candidate",
                {"selector": "#registered-candidate", "value_text": "point_lobos_state_natural_reserve"},
                select_value="point_lobos_state_natural_reserve",
            )
            if plan_context == "accepted_child":
                general_action(
                    "destination_day_select",
                    "#destination-day",
                    {"selector": "#destination-day", "value_text": "7"},
                    select_value="7",
                )
            else:
                expect(page.locator("#destination-day")).to_be_disabled()
            general_action("close_map_editor", "#close-map-edit", {"selector": "#map-edit-dialog", "visible": False})
            expand(page.locator(".textual-map-alternative"))
            general_action(
                "text_select_leg",
                "[data-geo-route-leg]",
                {"selector": f'[data-geo-route-leg="{ROUTE_LEG}"]', "attribute": "aria-pressed", "value": "true"},
                target_selector=f'[data-geo-route-leg="{ROUTE_LEG}"]',
            )
            general_action(
                "inspect_route", "#inspect-route-edit", {"selector": '[data-editor-kind="route"]', "visible": True}
            )
            general_action("close_map_editor", "#close-map-edit", {"selector": "#map-edit-dialog", "visible": False})
            general_action("help", "#help-button", {"selector": "#info-dialog", "visible": True})
            general_action(
                "about_close",
                '#info-dialog button[value="close"]',
                {"selector": "#info-dialog", "visible": False},
                target_selector='#info-dialog button[aria-label="Close"]',
            )
            general_action(
                "layers",
                "#layers-button",
                {"selector": "#toast-region", "text": "original route"},
                variant="empty_candidates",
            )
            general_action(
                "map_edit",
                "#map-edit-button",
                {"selector": "#map-edit-button", "attribute": "aria-pressed", "value": "true"},
            )
            general_action(
                "map_select",
                "#map-select-button",
                {"selector": "#map-select-button", "attribute": "aria-pressed", "value": "true"},
            )
            general_action("open_copilot", "#open-copilot", {"selector": "#copilot-dock", "visible": True})
            general_action(
                "transcript_open",
                "#transcript-settings-button",
                {"selector": "#transcript-settings-dialog", "visible": True},
            )
            general_action(
                "transcript_close",
                "#close-transcript-settings",
                {"selector": "#transcript-settings-dialog", "visible": False},
            )
            # The actual visible prompt becomes the exact composer expectation;
            # no message is sent, so this never dispatches a provider.
            for selector in (".quick-prompts button:first-child", ".quick-prompts button:last-child"):
                quick = page.locator(selector)
                expect(quick).to_be_visible()
                general_action(
                    "quick_prompt",
                    selector,
                    {"selector": "#copilot-input", "value_text": quick.get_attribute("data-prompt")},
                )
            general_action("close_copilot", "#close-copilot", {"selector": "#copilot-dock", "visible": False})
            for route in ("/app/map", "/app/repairs", "/app/compare", "/app/evidence", "/app/itinerary"):
                general_action("navigate_global", f'.nav-item[data-route="{route}"]', {"path": route})
                if route == "/app/map":
                    general_action(
                        "context_edit",
                        "#context-edit-mode",
                        {"selector": "#map-edit-button", "attribute": "aria-pressed", "value": "true"},
                    )
                    general_action(
                        "context_select",
                        "#context-select-mode",
                        {"selector": "#map-select-button", "attribute": "aria-pressed", "value": "true"},
                    )
                if route == "/app/repairs":
                    general_action("repairs_back", "#repairs-back", {"path": "/app/itinerary"})
                if route == "/app/compare":
                    # Change the actual selected option, not just click the default.
                    alternatives = page.locator('[data-compare-option][aria-pressed="false"]')
                    expect(alternatives.first).to_be_visible()
                    alternative_id = alternatives.first.get_attribute("data-compare-option")
                    selected_selector = f'[data-compare-option="{alternative_id}"]'
                    general_action(
                        "compare_select",
                        "[data-compare-option]",
                        {"selector": selected_selector, "attribute": "aria-pressed", "value": "true"},
                        target_selector=selected_selector,
                    )
                    responsive_compare()
                    general_action("context_evidence", "#context-evidence", {"path": "/app/evidence"})
                    general_action("context_back_compare", "#context-back-compare", {"path": "/app/compare"})
                    general_action("compare_evidence", "#compare-evidence", {"path": "/app/evidence"})
                    general_action("context_back_compare", "#context-back-compare", {"path": "/app/compare"})
                    general_action("compare_back", "#compare-back", {"path": "/app/itinerary"})
                if route == "/app/evidence":
                    target = (
                        "/app/compare" if "Compare" in page.locator("#evidence-back").inner_text() else "/app/itinerary"
                    )
                    general_action(
                        "evidence_back_unavailable" if target.endswith("compare") else "evidence_back_exact",
                        "#evidence-back",
                        {"path": target},
                    )
            general_action("open_compare", "#compare-repair", {"path": "/app/compare"})
            general_action("compare_back", "#compare-back", {"path": "/app/itinerary"})
            general_action("open_evidence", "#review-evidence", {"path": "/app/evidence"})
            general_action("context_back_compare", "#context-back-compare", {"path": "/app/compare"})
            general_action("compare_back", "#compare-back", {"path": "/app/itinerary"})
            page.set_viewport_size({"width": 390, "height": 844})
            _map_ack(page)
            general_action(
                "open_copilot", "#open-copilot", {"selector": "#copilot-dock", "visible": True}, variant="mobile"
            )
            general_action("close_copilot", "#close-copilot", {"selector": "#copilot-dock", "visible": False})
            for route in ("/app/map", "/app/repairs", "/app/compare", "/app/evidence", "/app/itinerary"):
                general_action("navigate_global", f'.mobile-nav button[data-route="{route}"]', {"path": route})
                if route == "/app/compare":
                    for pane in ("baseline", "option"):
                        general_action(
                            "compare_mobile_tab",
                            f'[data-compare-mobile-view="{pane}"]',
                            {
                                "selector": f'[data-compare-mobile-view="{pane}"]',
                                "attribute": "aria-pressed",
                                "value": "true",
                            },
                        )
            page.set_viewport_size({"width": 1440, "height": 900})
            assert map_identity(_map_ack(page)) == map_identity(parent)
            for selector, binding in (("#compare-choose", "compare_choose"), ("#context-choose", "context_choose")):
                general_action("open_compare", "#compare-repair", {"path": "/app/compare"})
                general_action(binding, selector, {"path": "/app/itinerary"})
                # Choose is transient. Reload must restore the unchanged accepted plan.
                prior_general_events.extend(page.evaluate("window.__controlReceiptEvents"))
                page.reload(wait_until="domcontentloaded")
                assert map_identity(_map_ack(page)) == map_identity(parent)

        def responsive_compare():
            nonlocal phase
            phase = "general_compare_responsive"
            request_start = len(http)
            views = []
            for width, height in ((1440, 900), (390, 844), (1440, 900)):
                page.set_viewport_size({"width": width, "height": height})
                settled_surface()
                expected = ["option"] if width == 390 else ["baseline", "option"]
                page.wait_for_function(
                    "expected => JSON.stringify([...document.querySelectorAll('[data-compare-map-pane]')].filter(e=>e.getClientRects().length).map(e=>e.dataset.compareMapPane))===JSON.stringify(expected)",
                    arg=expected,
                )
                page.wait_for_function("""() => [...document.querySelectorAll('[data-compare-map-pane]')]
                  .filter(p => p.getClientRects().length)
                  .every(p => {
                    const data = p.querySelector('[data-compare-map-canvas]').dataset;
                    return data.mapReady === 'true'
                      && Number(data.mapTileCoverage) >= .95
                      && !!data.renderedMapPlanId
                      && !!data.renderedPlanContentHash
                      && !!data.renderedGeometryHash;
                  })""")
                views.append(
                    page.locator("[data-compare-workspace]").evaluate("""e=>({
                  width:innerWidth,height:innerHeight,mobile_toggle_visible:!!e.querySelector('[data-compare-mobile-toggle]').getClientRects().length,
                  maps:[...e.querySelectorAll('[data-compare-map-pane]')].map(p=>({pane:p.dataset.compareMapPane,
                    visible:!!p.getClientRects().length,dataset:{...p.querySelector('[data-compare-map-canvas]').dataset}}))})""")
                )
            responsive_journeys.append(
                {
                    "context": plan_context,
                    "views": views,
                    "accepted_plan_id": parent["active_plan_id"],
                    "accepted_plan_content_hash": parent["active_plan_content_hash"],
                    "http_files": [row["file"] for row in http[request_start:]],
                    "assertions_passed": True,
                }
            )
            _write_json(evidence / "compare_responsive_journeys.json", responsive_journeys)

        def capture(label):
            _capture_ready(page, evidence / f"{label}.png")
            ack = _map_ack(page)
            assert map_identity(ack) == map_identity(parent)
            take_general_inventory(label)
            return ack

        def undo(expected):
            if page.locator("#map-edit-dialog").is_visible():
                page.locator("#close-map-edit").click()
            with page.expect_response(lambda response: response.url.endswith("/draft/undo")) as undone:
                page.locator("#inline-undo").click()
            assert undone.value.status == 200
            payload = undone.value.json()
            assert payload["session"]["draft"] == []
            assert {key: payload["undone"][key] for key in ("type", "target", "parameters")} == expected
            expect(page.locator(".draft-bar strong")).to_have_text("0 draft changes")
            assert map_identity(_map_ack(page)) == map_identity(parent)
            return payload

        def stage(kind, target, parameters, control):
            input_receipt = before_action(page, control)
            with page.expect_response(lambda response: response.url.endswith("/draft/operations")) as staged:
                control.click()
            assert staged.value.status == 200, staged.value.text()
            expected = {"type": kind, "target": target, "parameters": parameters}
            request = staged.value.request.post_data_json
            payload = staged.value.json()
            assert {key: request[key] for key in expected} == expected
            assert request["source"] == "map"
            assert {key: payload["operation"][key] for key in expected} == expected
            assert payload["session"]["draft"] == [payload["operation"]]
            assert payload["session"]["accepted_plan_id"] == parent["active_plan_id"]
            expect(page.locator(".draft-bar strong")).to_have_text("1 draft change", timeout=15_000)
            control_actions.append(
                after_action(
                    page,
                    input_receipt,
                    context="original" if phase == "baseline_replace_accept" else plan_context,
                    phase=phase,
                    request=request,
                    response=payload,
                )
            )
            _write_json(evidence / "control_action_receipts.json", control_actions)
            return expected, payload

        def draft_only(kind, variant, parameters, selected_form, target=TARGET):
            nonlocal phase
            phase = kind + "_" + variant.replace(":", "_")
            before_preview = count_mutations("/preview")
            take_inventory(phase + "_before")
            expect(selected_form.locator('button[type="submit"]')).to_be_enabled()
            expected, payload = stage(kind, target, parameters, selected_form.locator('button[type="submit"]'))
            expect(page.locator("#typed-edit-surface .typed-impact")).to_contain_text("1 draft only", timeout=15_000)
            for _ in range(300):
                if observed.get("impact", {}).get("session_revision") == payload["session"]["revision"]:
                    break
                page.wait_for_timeout(50)
            impact = observed["impact"]
            assert impact["session_revision"] == payload["session"]["revision"]
            assert impact["parent_plan_id"] == parent["active_plan_id"]
            assert impact["parent_plan_content_hash"] == parent["active_plan_content_hash"]
            assert impact["is_certified"] is False and impact["certification_status"] == "not_certified"
            assert impact["summary"]["operation_count"] == impact["summary"]["draft_only_count"] == 1
            assert impact["summary"]["evaluated_executable_count"] == 0
            assert impact["summary"]["can_run_evaluated_preview"] is False
            assert (
                len(impact["operations"]) == 1
                and impact["operations"][0]["operation_id"] == payload["operation"]["operation_id"]
            )
            assert impact["operations"][0]["feedback_tier"] == "draft_only"
            assert impact["operations"][0]["preview_executable"] is False
            assert impact["operations"][0]["evaluated_repair"] is False
            expect(page.locator("#draft-preview")).to_be_disabled()
            page.locator("#close-map-edit").click()
            expect(page.locator("#inline-preview")).to_be_disabled()
            expect(page.locator("#accept-repair")).to_be_disabled()
            rendered = capture(phase + "_staged")
            restored = undo(expected)
            assert count_mutations("/preview") == before_preview
            operations.append(
                {
                    "operation": kind,
                    "variant": variant,
                    "feedback_tier": "draft_only",
                    "context": plan_context,
                    "input": expected,
                    "staged": payload,
                    "impact": impact,
                    "rendered": rendered,
                    "undo": restored,
                    "preview_request_delta": 0,
                    "assertions_passed": True,
                }
            )
            _write_json(evidence / "operation_assertions.json", operations)

        with _preserve_browser_evidence(
            context, page, evidence, {"console_errors": errors, "request_failures": failures}
        ):
            try:
                page.goto(f"{BASE_URL}/app/itinerary", wait_until="domcontentloaded")
                initial = _map_ack(page)
                assert "bixby_creek_bridge_viewpoint" in initial["rendered_marker_ids"], (
                    "Case requires the unchanged fresh baseline"
                )
                parent = initial
                if plan_context == "accepted_child":
                    phase = "baseline_replace_accept"
                    open_stop(5)
                    legacy()
                    page.locator("#registered-candidate").select_option("pfeiffer_big_sur_state_park")
                    stage(
                        "replace_nearby",
                        "bixby_creek_bridge_viewpoint",
                        {"candidate_id": "pfeiffer_big_sur_state_park"},
                        page.locator("#draft-replace"),
                    )
                    page.locator("#close-map-edit").click()
                    expect(page.locator("#inline-preview")).to_be_enabled()
                    with page.expect_response(
                        lambda response: response.url.endswith("/preview"), timeout=90_000
                    ) as preview:
                        page.locator("#inline-preview").click()
                    assert preview.value.status == 200
                    proposal = preview.value.json()["proposal"]
                    assert proposal["state"] == "eligible"
                    expect(page.locator(".map-stage")).to_have_attribute(
                        "data-preview-state", "PREVIEW_RENDERED", timeout=40_000
                    )
                    assert _map_ack(page)["rendered_map_plan_id"] == proposal["child_plan_id"]
                    _capture_ready(page, evidence / "baseline_preview.png")
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
                    _write_json(evidence / "baseline_preview_response.json", preview.value.json())
                    _write_json(evidence / "baseline_acceptance_response.json", accepted.value.json())
                    page.reload(wait_until="domcontentloaded")
                    parent = _map_ack(page)
                    assert parent["rendered_map_plan_id"] == proposal["child_plan_id"]
                    assert parent["active_plan_content_hash"] == proposal["child_plan_content_hash"]
                    assert "pfeiffer_big_sur_state_park" in parent["rendered_marker_ids"]
                    assert "bixby_creek_bridge_viewpoint" not in parent["rendered_marker_ids"]
                phase = "capability_inventory"
                capture("baseline")
                asset_sources = page.evaluate(
                    """() => {
                  const map = """
                    + CURRENT_MAP
                    + """;
                  const style=map.getStyle();
                  return {container_id:map.getContainer().id,map_dataset:{...map.getContainer().dataset},
                    sprite:style.sprite??null,
                    vector_sources:Object.fromEntries(Object.entries(style.sources)
                      .filter(([,source])=>source.type==='vector'))};
                }"""
                )
                assert asset_sources["container_id"] == "geographic-map"
                assert asset_sources["map_dataset"]["mapReady"] == "true"
                assert float(asset_sources["map_dataset"]["mapTileCoverage"]) >= 0.95
                assert asset_sources["map_dataset"]["renderedMapPlanId"] == parent["active_plan_id"]
                assert asset_sources["map_dataset"]["renderedPlanContentHash"] == parent["active_plan_content_hash"]
                assert asset_sources["map_dataset"]["renderedGeometryHash"] == parent["rendered_geometry_hash"]
                network_policy = map_cancellation_policy(
                    map_configuration, {"sources": asset_sources["vector_sources"], "sprite": asset_sources["sprite"]}
                )
                _write_json(
                    evidence / "map_asset_sources.json",
                    {
                        "schema_version": "real-map-asset-sources-v1",
                        "scope": "Read-only public getStyle() vector sources and actual ready container dataset; no map-disposal cause inferred",
                        "phase": phase,
                        "context": plan_context,
                        "map_configuration_http_file": map_configuration_http_file,
                        **asset_sources,
                    },
                )
                workspace = observed["workspace"]
                capabilities = {
                    key: workspace[key]
                    for key in ("draft_capabilities", "typed_edit_capabilities", "map_edit_capabilities")
                }
                required = {
                    group: {
                        operation: required_variants(operation, capability, contract)
                        for operation, capability in contract["operations"].items()
                    }
                    for group, contract in capabilities.items()
                }
                snapshot = {
                    "schema_version": "runtime-capability-snapshot-v1",
                    "context": plan_context,
                    "session_id": observed["session"]["session_id"],
                    "plan_id": parent["active_plan_id"],
                    "plan_content_hash": parent["active_plan_content_hash"],
                    "capabilities": capabilities,
                    "required_variants": required,
                }
                _write_json(evidence / "capability_snapshot.json", snapshot)
                typed = capabilities["typed_edit_capabilities"]
                general_navigation()
                open_stop()
                for details in page.locator("#typed-edit-surface details").all():
                    expand(details)
                legacy()
                take_inventory("stop_controls_empty_draft")
                before_disabled = count_mutations("/draft/operations")
                for selector, operation, group in [
                    ("#draft-lock", "lock_stop", "draft_capabilities"),
                    ("#draft-flexible", "mark_flexible", "draft_capabilities"),
                    ("#draft-move", "move_day", "draft_capabilities"),
                    ("#draft-feedback", "route_feedback", "draft_capabilities"),
                    (
                        '[data-typed-operation="set_stop_day"] button[type="submit"]',
                        "set_stop_day",
                        "typed_edit_capabilities",
                    ),
                    (
                        "#draft-add" if plan_context == "original" else "#draft-replace",
                        "add_candidate" if plan_context == "original" else "replace_nearby",
                        "draft_capabilities",
                    ),
                ]:
                    assert capabilities[group]["operations"][operation]["enabled"] is False
                    control = page.locator(selector)
                    expect(control).to_be_visible()
                    expect(control).to_be_disabled()
                    disabled.append(
                        {
                            "operation": operation,
                            "variant": "disabled",
                            "selector": selector,
                            "context": plan_context,
                            "visible": True,
                            "disabled": True,
                            "label": control.inner_text(),
                            "title": control.get_attribute("title"),
                            "forced_interaction": False,
                            "draft_request_delta": 0,
                        }
                    )
                day = form("set_stop_day")
                expect(day).to_contain_text("Unavailable")
                day_before = before_action(page, day.locator('button[type="submit"]'))
                day_select = day.locator('[name="day"]')
                expect(day_select).to_have_value("7")
                day_select.press("ArrowUp")
                expect(day_select).to_have_value("6")
                expect(day.locator('button[type="submit"]')).to_be_disabled()
                _write_json(
                    evidence / "disabled_day_control.json",
                    {
                        "context": plan_context,
                        "before": day_before,
                        "after": before_action(page, day.locator('button[type="submit"]')),
                        "events": page.evaluate("window.__draftControlInputs"),
                        "capability": typed["operations"]["set_stop_day"],
                        "map": _map_ack(page),
                        "parent": parent,
                        "phase": phase,
                        "assertions_passed": True,
                    },
                )
                for operation in ("set_stop_commitment", "set_attribute_constraint"):
                    selected = form(operation)
                    for strength in typed["operations"][operation]["protected_strengths"]:
                        option = selected.locator(f'[name="strength"] option[value="{strength}"]')
                        # Playwright's actionability disabled check does not classify
                        # native option elements; inspect the browser's real state.
                        expect(option).to_have_js_property("disabled", True)
                        assert option.evaluate("element => element.matches(':disabled')")
                        expect(option).to_contain_text("permission required")
                        disabled.append(
                            {
                                "operation": operation,
                                "variant": strength if operation == "set_stop_commitment" else "strength:" + strength,
                                "context": plan_context,
                                "selector": f'[data-typed-operation="{operation}"] [name="strength"] option[value="{strength}"]',
                                "disabled": True,
                                "label": option.inner_text(),
                                "forced_interaction": False,
                                "draft_request_delta": 0,
                            }
                        )
                assert count_mutations("/draft/operations") == before_disabled
                page.locator("#close-map-edit").click()
                capture("disabled_and_protected_controls")

                phase = "legacy_keep_control"
                open_stop()
                legacy()
                expected, kept = stage("keep_stop", TARGET, {}, page.locator("#draft-keep"))
                kept_map = _map_ack(page)
                assert map_identity(kept_map) == map_identity(parent)
                reverted = undo(expected)
                _write_json(
                    evidence / "legacy_keep_control.json",
                    {
                        "context": plan_context,
                        "phase": phase,
                        "parent": parent,
                        "stage": kept,
                        "staged_map": kept_map,
                        "undo": reverted,
                        "undone_map": _map_ack(page),
                        "evaluated_claimed": False,
                        "assertions_passed": True,
                    },
                )

                for mode in typed["operations"]["set_stop_duration"]["draft_only_modes"]:
                    open_stop()
                    selected = form("set_stop_duration")
                    select_native_option(selected.locator('[name="duration_mode"]'), mode)
                    values = {"mode": mode, "preferred_minutes": None, "minimum_minutes": None, "maximum_minutes": None}
                    for field, value in (("preferred", 60), ("minimum", 45), ("maximum", 90)):
                        if (
                            (field == "preferred" and mode == "preferred")
                            or (field == "minimum" and mode in {"minimum", "range"})
                            or (field == "maximum" and mode in {"maximum", "range"})
                        ):
                            selected.locator(f'[name="duration-{field}"]').fill(str(value))
                            values[field + "_minutes"] = value
                    expect(selected.locator("[data-duration-capability-note]")).to_contain_text("draft only")
                    draft_only("set_stop_duration", mode, {"duration": values}, selected)
                for role in typed["operations"]["set_stop_role"]["draft_only_roles"]:
                    open_stop()
                    selected = form("set_stop_role")
                    select_native_option(selected.locator('[name="role"]'), role)
                    expect(selected.locator("[data-role-capability-note]")).to_contain_text("draft only")
                    draft_only("set_stop_role", role, {"role": role}, selected)
                protected = typed["operations"]["set_stop_commitment"]["protected_strengths"]
                for strength in typed["vocabularies"]["commitment_strengths"]:
                    if strength in protected:
                        continue
                    open_stop()
                    selected = form("set_stop_commitment")
                    select_native_option(selected.locator('[name="strength"]'), strength)
                    select_native_option(selected.locator('[name="scope_lifetime"]'), "current_repair_session")
                    draft_only(
                        "set_stop_commitment",
                        strength,
                        {"strength": strength, "scope_lifetime": "current_repair_session"},
                        selected,
                    )
                for attribute in typed["vocabularies"]["attributes"]:
                    open_stop()
                    selected = form("set_attribute_constraint")
                    select_native_option(selected.locator('[name="attribute"]'), attribute)
                    if attribute == "existence":
                        selected.locator('[name="attribute_existence"]').check()
                        value = True
                    elif attribute == "day":
                        select_native_option(selected.locator('[name="attribute_day"]'), "6")
                        value = 6
                    elif attribute == "duration":
                        # Exercise the second duration-mode listener alias, then restore the exact request.
                        select_native_option(selected.locator('[name="attribute_duration_mode"]'), "range")
                        select_native_option(selected.locator('[name="attribute_duration_mode"]'), "exact")
                        selected.locator('[name="attribute-duration-preferred"]').fill("60")
                        value = {"mode": "exact", "preferred_minutes": 60, "minimum_minutes": 60, "maximum_minutes": 60}
                    elif attribute == "sequence_order":
                        select_native_option(selected.locator('[name="attribute_order_day"]'), "7")
                        selected.locator('[name="attribute_sequence_index"]').fill("0")
                        value = {"day": 7, "sequence_index": 0}
                    elif attribute == "role":
                        select_native_option(selected.locator('[name="attribute_role"]'), "meal")
                        value = "meal"
                    else:
                        raise AssertionError(f"New attribute requires a real browser input definition: {attribute}")
                    draft_only(
                        "set_attribute_constraint",
                        "attribute:" + attribute,
                        {
                            "attribute": attribute,
                            "value": value,
                            "strength": "prefer_keep",
                            "scope_lifetime": "current_repair_session",
                        },
                        selected,
                    )

                for issue in typed["vocabularies"]["route_issue_types"]:
                    phase = "route_controls"
                    with page.expect_response(
                        lambda response: (
                            response.url.endswith("/selection")
                            and response.request.post_data_json.get("selected_day") == 6
                        )
                    ) as day_selection:
                        page.locator('.day-card[data-day="6"]').click()
                    wait_selected(day_selection.value)
                    expand(page.locator(".textual-map-alternative"))
                    with page.expect_response(
                        lambda response: (
                            response.url.endswith("/selection")
                            and response.request.post_data_json.get("selected_segment_id") == ROUTE_LEG
                        )
                    ) as selection:
                        page.locator(f'[data-geo-route-leg="{ROUTE_LEG}"]').click()
                    wait_selected(selection.value)
                    assert selection.value.json()["session"]["selected_segment_id"] == ROUTE_LEG
                    page.locator("#map-edit-button").click()
                    _map_ack(page)
                    page.locator("#inspect-route-edit").click()
                    expect(page.locator('[data-editor-kind="route"]')).to_be_visible()
                    take_inventory("route_controls_" + issue)
                    preference = page.get_by_role("region", name="Unavailable route preference editing")
                    expect(preference.get_by_role("button", name="Change route preference")).to_be_disabled()
                    expect(preference).to_contain_text("No simulated")
                    if issue == typed["vocabularies"]["route_issue_types"][0]:
                        disabled.append(
                            {
                                "operation": "change_route_preference",
                                "variant": "disabled",
                                "context": plan_context,
                                "selector": ".disabled-capability button",
                                "disabled": True,
                                "visible": True,
                                "label": preference.get_by_role("button", name="Change route preference").inner_text(),
                                "explanation": preference.inner_text(),
                                "forced_interaction": False,
                                "draft_request_delta": 0,
                            }
                        )
                    selected = form("report_route_issue")
                    select_native_option(selected.locator('[name="issue_type"]'), issue)
                    note = "Browser capability test: user observation only, not a verified closure."
                    selected.locator('[name="note"]').fill(note)
                    draft_only(
                        "report_route_issue",
                        issue,
                        {"route_leg_id": ROUTE_LEG, "issue_type": issue, "note": note},
                        selected,
                        target=ROUTE_LEG,
                    )

                if plan_context == "original":
                    duration_parameters = {
                        "duration": {
                            "mode": "range",
                            "preferred_minutes": None,
                            "minimum_minutes": 45,
                            "maximum_minutes": 90,
                        }
                    }
                    replacement_parameters = {"candidate_id": "pfeiffer_big_sur_state_park"}
                    for first in ("duration", "replacement"):
                        phase = "cross_family_" + first + "_first"
                        # First prove replacement is individually selectable with an empty draft.
                        open_stop(5)
                        legacy()
                        page.locator("#registered-candidate").select_option("pfeiffer_big_sur_state_park")
                        expect(page.locator("#draft-replace")).to_be_enabled()
                        if first == "duration":
                            open_stop()
                            selected = form("set_stop_duration")
                            select_native_option(selected.locator('[name="duration_mode"]'), "range")
                            selected.locator('[name="duration-minimum"]').fill("45")
                            selected.locator('[name="duration-maximum"]').fill("90")
                            initial_op, payload = stage(
                                "set_stop_duration",
                                TARGET,
                                duration_parameters,
                                selected.locator('button[type="submit"]'),
                            )
                            open_stop(5)
                            legacy()
                            page.locator("#registered-candidate").select_option("pfeiffer_big_sur_state_park")
                            blocked_control = page.locator("#draft-replace")
                            second_op = {
                                "type": "replace_nearby",
                                "target": "bixby_creek_bridge_viewpoint",
                                "parameters": replacement_parameters,
                            }
                        else:
                            initial_op, payload = stage(
                                "replace_nearby",
                                "bixby_creek_bridge_viewpoint",
                                replacement_parameters,
                                page.locator("#draft-replace"),
                            )
                            open_stop()
                            selected = form("set_stop_duration")
                            select_native_option(selected.locator('[name="duration_mode"]'), "range")
                            selected.locator('[name="duration-minimum"]').fill("45")
                            selected.locator('[name="duration-maximum"]').fill("90")
                            blocked_control = selected.locator('button[type="submit"]')
                            second_op = {
                                "type": "set_stop_duration",
                                "target": TARGET,
                                "parameters": duration_parameters,
                            }
                        before = count_mutations("/draft/operations")
                        expect(blocked_control).to_be_visible()
                        expect(blocked_control).to_be_disabled()
                        assert "combin" in (blocked_control.get_attribute("title") or "").lower()
                        second_control = before_action(page, blocked_control)
                        selected_candidate = (
                            page.locator("#registered-candidate").input_value() if first == "duration" else None
                        )
                        take_inventory(phase)
                        page.locator("#close-map-edit").click()
                        capture(phase)
                        assert count_mutations("/draft/operations") == before
                        assert observed["session"]["draft"] == [payload["operation"]]
                        undo(initial_op)
                        combinations.append(
                            {
                                "first": first,
                                "context": plan_context,
                                "first_operation": initial_op,
                                "second_operation": second_op,
                                "second_control": second_control,
                                "selected_candidate_id": selected_candidate,
                                "second_operation_disabled": True,
                                "second_stage_request_delta": 0,
                                "assertions_passed": True,
                            }
                        )

                capture("final_unchanged")
                assert not errors
                assert classify_browser_network(failures, network_policy)["blocking_failure_indexes"] == []
                assert not [row for row in http if row["status"] >= 400]
                assert len({row["file"] for row in http}) == len(http)
                assert count_mutations("/preview") == (1 if plan_context == "accepted_child" else 0)
                assert count_mutations("/accept") == (1 if plan_context == "accepted_child" else 0)
                _write_json(
                    evidence / "browser_assertions.json",
                    {
                        "schema_version": "capability-controls-browser-v1",
                        "context": plan_context,
                        "parent": parent,
                        "final": _map_ack(page),
                        "draft_only_variant_count": len(operations),
                        "disabled_variants": disabled,
                        "cross_family_cases": combinations,
                        "assertions_passed": True,
                        "evaluated_variant_coverage_claimed": False,
                        "remaining_evaluated_variants": {
                            operation: [row["id"] for row in variants if row["feedback_tier"] == "evaluated"]
                            for operation, variants in required["typed_edit_capabilities"].items()
                        },
                    },
                )
            finally:
                _write_json(
                    evidence / "general_control_inventory.json",
                    {
                        "schema_version": "general-control-inventory-v1",
                        "context": plan_context,
                        "observations": general_inventory,
                        "presence_is_not_execution": True,
                        "absent_is_not_unsupported": True,
                    },
                )
                _write_json(
                    evidence / "general_control_journeys.json",
                    {
                        "schema_version": "general-control-journeys-v1",
                        "context": plan_context,
                        "journeys": general_journeys,
                        "inventory_is_not_execution": True,
                    },
                )
                _write_json(
                    evidence / "general_control_events.json",
                    {
                        "schema_version": "general-control-events-v1",
                        "context": plan_context,
                        "events": prior_general_events + page.evaluate("window.__controlReceiptEvents || []"),
                        "trusted_event_is_not_semantic_completion": True,
                    },
                )
                _write_json(evidence / "http_index.json", http)
                _write_json(evidence / "operation_assertions.json", operations)
                _write_json(evidence / "disabled_control_assertions.json", disabled)
                _write_json(evidence / "cross_family_assertions.json", combinations)
                _write_json(evidence / "console_errors.json", errors)
                _write_json(evidence / "network_failures.json", failures)
                if network_policy is not None:
                    _write_json(
                        evidence / "network_classification.json",
                        {
                            "map_configuration_http_file": map_configuration_http_file,
                            "rendered_map_asset_sources_file": "map_asset_sources.json",
                            **classify_browser_network(failures, network_policy),
                        },
                    )
                _write_json(
                    evidence / "llm_io.json",
                    {
                        "provider_call_count": 0,
                        "llm_input": None,
                        "llm_output": None,
                        "reason": "Only actual direct-edit website controls are exercised; no chat or provider call.",
                    },
                )
        browser.close()
