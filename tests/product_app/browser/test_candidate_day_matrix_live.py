"""Real candidate/day readiness in original and browser-accepted contexts.

This is staging/readiness coverage, not a claim of evaluating every pair. The
addition-continuation journey owns eligible A -> B -> C addition/acceptance.
"""
from __future__ import annotations

import json
import math
import os
from copy import deepcopy
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
    browser_network_record,
    classify_browser_network,
    map_cancellation_policy,
    wait_map,
)

from itinerary_system.product_app.product_demo import load_product_demo_package
from itinerary_system.product_app.registry import ProductRunRegistry
from itinerary_system.product_app.route_delta import configured_route_matrix
from itinerary_system.routing.matrix import RouteMatrixCell, RouteMatrixNotPublicationEligible

ROOT = Path(__file__).resolve().parents[3]
BASE_URL = os.environ.get("PRODUCT_E2E_BASE_URL")
DISPOSABLE = os.environ.get("PRODUCT_E2E_CANDIDATE_DAY_MATRIX_DISPOSABLE") == "1"
BIXBY = "bixby_creek_bridge_viewpoint"
PFEIFFER = "pfeiffer_big_sur_state_park"


def _identity(ack):
    return {key: value for key, value in ack.items() if key != "preview_state"}


def _check_required_legs(row, parent, geography, matrix):
    """Cross-check the HTTP readiness against immutable parent and road cells."""
    assert row["parent_plan_id"] == parent["plan_id"]
    assert row["parent_plan_content_hash"] == parent["content_hash"]
    assert row["context_snapshot_id"] == parent["context_snapshot_id"] == matrix["context_snapshot_id"]
    assert row["matrix_id"] == matrix["matrix_id"]
    assert row["comparison_eligibility"] == "not_evaluated"
    assert row["insertion_mode"] == "append_after_day_stops"
    day = next(item for item in parent["ordered_days"] if item["day"] == row["day"])
    day_legs = sorted((feature["properties"] for feature in geography["validated_legs"]["features"]
                      if feature["properties"]["day"] == row["day"] and not feature["properties"]["cross_day"]),
                      key=lambda item: item["leg_index"])
    assert day_legs
    sequence = [day_legs[0]["origin_id"], *day["stop_ids"], row["candidate_id"], day_legs[-1]["destination_id"]]
    assert row["route_sequence"] == sequence
    pairs = list(zip(sequence[:-1], sequence[1:], strict=True))
    assert [(leg["origin_id"], leg["destination_id"]) for leg in row["required_legs"]] == pairs
    cells = {(cell["origin_id"], cell["destination_id"]): cell for cell in matrix["cells"]}
    expected_findings, evidence = [], []
    for pair, leg in zip(pairs, row["required_legs"], strict=True):
        raw = cells.get(pair)
        code = "missing_route_matrix_cell" if raw is None else None
        if raw is not None:
            cell = RouteMatrixCell(**{**raw, "geometry": tuple(tuple(point) for point in raw["geometry"])})
            try:
                cell.require_publication_eligible()
            except RouteMatrixNotPublicationEligible:
                code = "route_leg_not_road_validated"
            if code is None and cell.context_snapshot_id != parent["context_snapshot_id"]:
                code = "route_context_mismatch"
            if code is None and pair[0] != pair[1] and len(cell.geometry) < 2:
                code = "route_geometry_missing"
            if code is None and any(len(point) != 2
                    or not all(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
                               for value in point)
                    or not (-90 <= point[0] <= 90 and -180 <= point[1] <= 180) for point in cell.geometry):
                code = "route_geometry_invalid"
            if code == "route_leg_not_road_validated":
                assert "route_leg_id" not in leg
            else:
                assert leg["route_leg_id"] == cell.route_leg_id
        else:
            assert "route_leg_id" not in leg
        assert leg["ready"] is (code is None)
        if code:
            expected_findings.append((code, *pair))
        evidence.append({"origin_id": pair[0], "destination_id": pair[1], "actual_cell": raw, "blocking_code": code})
    assert [(item["code"], item["origin_id"], item["destination_id"]) for item in row["blocking_findings"]] == expected_findings
    for finding in row["blocking_findings"]:
        assert finding["context_snapshot_id"] == row["context_snapshot_id"]
        assert finding["matrix_id"] == row["matrix_id"] and finding["corrective_action"]
    assert row["preview_executable"] is (not expected_findings)
    expected_status = "route_evidence_ready" if not expected_findings else (
        "blocked_missing_route_evidence" if all(item[0] == "missing_route_matrix_cell" for item in expected_findings)
        else "blocked_invalid_route_evidence")
    assert row["status"] == expected_status
    return evidence


@pytest.mark.skipif(not BASE_URL or not DISPOSABLE, reason="Requires a fresh guarded candidate-day-matrix server")
def test_browser_candidate_day_readiness_in_original_and_accepted_child(tmp_path):
    directory = Path(os.environ.get("PRODUCT_E2E_CANDIDATE_DAY_MATRIX_EVIDENCE_DIR", tmp_path))
    directory.mkdir(parents=True, exist_ok=True)
    registry = ProductRunRegistry(ROOT, ROOT / "configs/product_app_registry.json")
    record = registry.default
    bundle = load_product_demo_package(ROOT, registry.run_dir(record.run_id)).primary_bundle
    matrix = configured_route_matrix(ROOT, record.run_id, bundle.route_matrix)
    immutable_parent = bundle.parent_plan
    _write_json(directory / "immutable_inputs.json", {"run_id": record.run_id,
        "manifest_sha256": record.manifest_hash, "run_directory": record.relative_path,
        "original_parent": immutable_parent, "configured_route_matrix": matrix,
        "scope": "Read-only pinned artifacts; no repair, provider, or setup API calls"})
    http, cases, inventories, checkpoints, snapshots = [], [], [], {}, {}
    errors = {"page_errors": [], "console_errors": [], "request_failures": [], "http_errors": [], "capture_errors": []}
    observed = {}
    map_configuration = None
    map_configuration_http_file = None
    phase = "initial_original"
    from control_action_fixture import OBSERVE_CONTROL_ACTIONS, after_action, before_action
    with sync_playwright() as runtime:
        browser = runtime.chromium.launch(headless=True, **({"executable_path": str(CHROME)} if CHROME.is_file() else {}))
        context = browser.new_context(viewport={"width": 1440, "height": 900})
        context.add_init_script(OBSERVE_REAL_MAPS)
        context.add_init_script(OBSERVE_CONTROL_ACTIONS)
        context.add_init_script("""window.__poiControlEvents=[];
          for(const type of ['click','change','submit']) document.addEventListener(type,event=>{
            const e=event.target.closest?.('button,input,form');
            if(e) window.__poiControlEvents.push({type,trusted:event.isTrusted,id:e.id,candidate:e.dataset.poiCandidate||null,
              form:e.closest('form')?.id||null,stop:e.dataset.geoStop||null,
              leg:e.dataset.geoRouteLeg||null,at:Date.now()});
          },true);""")
        page = context.new_page()
        control_requests=[]
        page.on('request',lambda request:control_requests.append({'method':request.method,'url':request.url}))
        page.on("pageerror", lambda error: errors["page_errors"].append(str(error)))
        page.on("console", lambda event: errors["console_errors"].append(event.text) if event.type == "error" else None)
        page.on("requestfailed", lambda request: errors["request_failures"].append(browser_network_record(request, phase)))

        def record_response(response):
            nonlocal map_configuration, map_configuration_http_file
            if response.status >= 400:
                errors["http_errors"].append({"url": response.url, "status": response.status})
            if not urlsplit(response.url).path.startswith("/api/"):
                return
            try:
                row = {"phase": phase, **_http_record(response)}
                name = f"http_{len(http)+1:03d}.json"
                _write_json(directory / name, row)
                http.append({"file": name, "phase": phase, "method": row["method"], "url": row["url"], "status": row["response_status_code"]})
                if response.status == 200:
                    payload = json.loads(row["response_body_utf8"])
                    if response.url == f"{BASE_URL}/api/map/config":
                        assert map_configuration is None or payload == map_configuration
                        map_configuration, map_configuration_http_file = payload, name
                    if isinstance(payload, dict):
                        for key in ("session", "workspace"):
                            if key in payload:
                                observed[key] = payload[key]
            except Exception as error:
                errors["capture_errors"].append({"url": response.url, "type": type(error).__name__, "message": str(error)})
        page.on("response", record_response)

        def capture(label, expected=None):
            _capture_ready(page, directory / f"{label}.png")
            ack = _map_ack(page)
            if expected:
                assert _identity(ack) == _identity(expected)
            checkpoints[label] = ack
            return ack

        def count(suffix):
            return sum(row["method"] == "POST" and row["url"].endswith(suffix) for row in http)

        def editor(day):
            if page.locator("#map-edit-dialog").is_visible():
                page.locator("#close-map-edit").click()
            with page.expect_response(lambda response: response.url.endswith("/draft/impact-preview")):
                page.locator(f'.day-card[data-day="{day}"]').click()
            page.locator("#map-edit-button").click()
            page.locator("#inspect-edit-stop").click()
            expect(page.locator("#map-edit-dialog")).to_be_visible()
            shortcuts = page.locator(".legacy-edit-controls")
            if shortcuts.get_attribute("open") is None:
                shortcuts.locator("summary").click()
            expect(page.locator("#registered-candidate")).to_be_visible()

        def select_candidate(candidate):
            expect(page.locator("#registered-candidate")).to_be_enabled()
            option = page.locator(f'#registered-candidate option[value="{candidate}"]')
            expect(option).to_have_js_property("disabled", False)
            assert not option.evaluate("element=>element.matches(':disabled')")
            with page.expect_response(lambda response: response.url.endswith("/draft/impact-preview")):
                with page.expect_response(lambda response: response.url.endswith("/selection")) as selected:
                    page.locator("#registered-candidate").select_option(candidate)
            assert selected.value.status == 200
            assert selected.value.json()["session"]["selected_candidate_id"] == candidate
            expect(page.locator("#registered-candidate")).to_have_value(candidate)
            expect(page.locator(".typed-impact")).to_contain_text("The active draft is empty.")

        def inventory():
            return page.locator("#registered-candidate, #destination-day, #draft-add, #draft-preview").evaluate_all("""elements => elements.map(element => ({
              id:element.id,tag:element.tagName.toLowerCase(),value:element.value,disabled:element.matches(':disabled'),
              visible:!!(element.offsetWidth || element.offsetHeight || element.getClientRects().length),title:element.title,
              label:element.labels?.[0]?.textContent || element.textContent.trim(),
              options:[...element.options || []].map(option=>({value:option.value,label:option.textContent,
                disabled:option.disabled,selected:option.selected}))}))""")

        def accepted_poi_controls(parent_map,geography):
            """Actual accepted-target discovery; this never stages or evaluates a replacement."""
            nonlocal phase
            with page.expect_response(lambda response:response.url.endswith('/selection')
                    and response.request.post_data_json.get('selected_day')==5):
                page.locator('.day-card[data-day="5"]').click()
            expect(page.locator('#map-context-title')).to_contain_text('Pfeiffer')
            page.locator('#map-edit-button').click()
            expect(page.locator('#prepare-replacement-candidates')).to_be_enabled()
            page.locator('#prepare-replacement-candidates').click()
            preparation=page.evaluate('window.__poiControlEvents')
            details=page.locator('.textual-map-alternative')
            if details.get_attribute('open') is None:
                details.locator(':scope > summary').click()
            leg=next(feature['properties'] for feature in geography['validated_legs']['features']
                if feature['properties']['day']==5 and feature['properties']['destination_id']==PFEIFFER)
            with page.expect_response(lambda response:response.url.endswith('/selection')
                    and response.request.post_data_json.get('selected_segment_id')==leg['route_leg_id']):
                page.locator(f'[data-geo-route-leg="{leg["route_leg_id"]}"]').click()
            expect(page.locator('#find-route-candidates')).to_be_enabled()
            with page.expect_response(lambda response:'/map/poi-candidates?' in response.url) as discovered:
                page.locator('#find-route-candidates').click()
            assert discovered.value.status==200
            initial=discovered.value.json()
            assert initial['context']['replacement_target_id']==PFEIFFER and initial['context']['day']==5
            discovery_events=page.evaluate('window.__poiControlEvents')
            rows=[]
            start=len(control_requests)

            def layers():
                wait_map(page)
                return page.evaluate("""()=>{const map="""+CURRENT_MAP+""";
                  return Object.fromEntries(['v2-poi-candidates','v2-poi-access-points'].map(id=>[id,map.getStyle().sources[id].data]));}""")

            def observation(selector,before,response=None,fields=None):
                assert _identity(_map_ack(page))==_identity(parent_map)
                rows.append({'selector':selector,'visible':True,'disabled':False,
                    'events':page.evaluate('window.__poiControlEvents'),'before_layers':before,'after_layers':layers(),
                    'map':_map_ack(page),'http':_http_record(response) if response else None,'fields':fields,
                    'drawer_visible':page.locator('#poi-candidate-heading').is_visible(),'assertions_passed':True})

            for fields in ({'role':'meal','max_detour_minutes':'120','limit':'3'},
                           {'role':'','max_detour_minutes':'60','limit':'5'}):
                before=layers()
                for selector,key in (('#poi-role-filter','role'),('#poi-detour-filter','max_detour_minutes'),('#poi-limit-filter','limit')):
                    page.locator(selector).select_option(fields[key])
                expect(page.locator('#poi-candidate-filter button[type=submit]')).to_be_enabled()
                with page.expect_response(lambda response:'/map/poi-candidates?' in response.url) as filtered:
                    page.locator('#poi-candidate-filter button[type=submit]').click()
                assert filtered.value.status==200
                observation('#poi-candidate-filter',before,filtered.value,fields)
            for selector in ('#poi-layer-toggle','#poi-layer-toggle','#layers-button','#layers-button'):
                before=layers()
                expect(page.locator(selector)).to_be_enabled()
                expect(page.locator(selector)).to_be_visible()
                page.locator(selector).click()
                observation(selector,before)
            before=layers()
            page.locator('#close-poi-candidates').click()
            expect(page.locator('#poi-candidate-heading')).to_have_count(0)
            observation('#close-poi-candidates',before)
            with page.expect_response(lambda response:'/map/poi-candidates?' in response.url) as reopened:
                page.locator('#find-route-candidates').click()
            assert reopened.value.status==200
            observation('#find-route-candidates',layers(),reopened.value)
            assert reopened.value.json()['context']['kind']=='insertion'
            assert reopened.value.json()['context']['replacement_target_id'] is None
            assert reopened.value.json()['candidates']==[]
            expect(page.locator('[data-poi-candidate]')).to_have_count(0)
            expect(page.locator('.candidate-results')).to_contain_text('No candidates are available')
            assert _map_ack(page)==parent_map
            _write_json(directory/'accepted_poi_control_journeys.json',rows)
            control_log={'events':page.evaluate('window.__poiControlEvents'),
                'requests':list(control_requests),'request_start':start,'request_end':len(control_requests),'parent':parent_map,
                'provider_calls':0,'context':'accepted_child','target':PFEIFFER,'selected_leg':leg,
                'preparation_events':preparation,'discovery_events':discovery_events,'discovery_http':_http_record(discovered.value),
                'initial_candidates':initial['candidates']}
            _write_json(directory/'accepted_poi_control_input_log.json',control_log)
            phase='accepted_poi_reentry'
            if details.get_attribute('open') is None:
                details.locator(':scope > summary').click()
            with page.expect_response(lambda response:response.url.endswith('/selection')
                    and response.request.post_data_json.get('selected_stop_id')==PFEIFFER) as reentry_stop:
                page.locator(f'[data-geo-stop="{PFEIFFER}"]').click()
            stop_events=page.evaluate('window.__poiControlEvents')
            expect(page.locator('#prepare-replacement-candidates')).to_be_enabled()
            page.locator('#prepare-replacement-candidates').click()
            reentry_preparation=page.evaluate('window.__poiControlEvents')
            if details.get_attribute('open') is None:
                details.locator(':scope > summary').click()
            with page.expect_response(lambda response:response.url.endswith('/selection')
                    and response.request.post_data_json.get('selected_segment_id')==leg['route_leg_id']) as reentry_leg:
                page.locator(f'[data-geo-route-leg="{leg["route_leg_id"]}"]').click()
            leg_events=page.evaluate('window.__poiControlEvents')
            with page.expect_response(lambda response:'/map/poi-candidates?' in response.url
                    and 'replacement_target_id='+PFEIFFER in response.url) as replacement_again:
                page.locator('#find-route-candidates').click()
            assert replacement_again.value.status==200
            replacement_payload=replacement_again.value.json()
            assert replacement_payload['context']['replacement_target_id']==PFEIFFER
            assert replacement_payload['context']['route_leg_id']==leg['route_leg_id']
            assert replacement_payload['context']['kind']=='replacement' and replacement_payload['candidates']
            expect(page.locator('[data-poi-candidate]')).to_have_count(len(replacement_payload['candidates']))
            restored_layers=layers()
            control_log['replacement_reentry']={
                'schema_version':'poi-replacement-reentry-v1','target':PFEIFFER,'route_leg_id':leg['route_leg_id'],
                'generic_http':_http_record(reopened.value),'stop_http':_http_record(reentry_stop.value),
                'leg_http':_http_record(reentry_leg.value),'candidate_http':_http_record(replacement_again.value),
                'stop_events':stop_events,'preparation_events':reentry_preparation,'leg_events':leg_events,
                'events':page.evaluate('window.__poiControlEvents'),'map':_map_ack(page),'layers':restored_layers,
                'visible_candidate_ids':page.locator('[data-poi-candidate]').evaluate_all(
                    'elements=>elements.map(e=>e.dataset.poiCandidate)'),'assertions_passed':True}
            _write_json(directory/'accepted_poi_control_input_log.json',control_log)
            phase='accepted_poi_inspection'
            inspection_start=len(control_requests)
            inspection_http_start=len(http)
            inspected=[]
            capability=observed['workspace']['draft_capabilities']['operations']['replace_nearby']
            for candidate in replacement_payload['candidates']:
                button=page.locator('[data-poi-candidate="'+candidate['candidate_id']+'"]')
                expect(button).to_be_visible()
                expect(button).to_be_enabled()
                with page.expect_response(lambda response:'/map/poi-candidates?' in response.url) as refreshed:
                    with page.expect_response(lambda response,candidate_id=candidate['candidate_id']:response.url.endswith('/selection')
                            and response.request.post_data_json.get('selected_candidate_id')==candidate_id) as selected:
                        button.click()
                assert selected.value.status==refreshed.value.status==200
                expect(button).to_have_attribute('aria-pressed','true')
                expect(page.locator('#poi-candidate-title')).to_have_text(candidate['place']['name'])
                replace=page.locator('#use-candidate-replacement')
                expect(replace).to_be_visible()
                registered=candidate.get('registered_replacement') or {}
                enabled=capability['enabled'] and registered.get('draft_type')=='replace_nearby' \
                    and registered.get('candidate_id')==candidate['candidate_id']==candidate['place']['place_id'] \
                    and registered.get('target_stop_id')==PFEIFFER
                assert not enabled, 'Accepted target unexpectedly has an executable replacement: add a real stage/Undo journey before crediting it'
                expect(replace).to_be_disabled()
                assert _identity(_map_ack(page))==_identity(parent_map)
                inspected.append({'candidate':candidate,'selected':button.get_attribute('aria-pressed'),
                    'title':page.locator('#poi-candidate-title').inner_text(),'events':page.evaluate('window.__poiControlEvents'),
                    'map':_map_ack(page),'replace_control':{'visible':True,'disabled':True,'title':replace.get_attribute('title')},
                    'selection_http':_http_record(selected.value),'refresh_http':_http_record(refreshed.value),
                    'forced_interaction':False,'assertions_passed':True})
            assert inspected
            _write_json(directory/'accepted_candidate_inspection.json',{'context':'accepted_child','parent':parent_map,
                'candidate_http':_http_record(replacement_again.value),'capability':capability,'rows':inspected,
                'events':page.evaluate('window.__poiControlEvents'),'requests':control_requests[inspection_start:],
                'http_records':[json.loads((directory/item['file']).read_text(encoding='utf-8')) for item in http[inspection_http_start:]],
                'accepted_plan_changed':False,'assertions_passed':True})
            page.locator('#close-poi-candidates').click()
            expect(page.locator('#poi-candidate-heading')).to_have_count(0)

        with _preserve_browser_evidence(context, page, directory, errors):
            try:
                page.goto(f"{BASE_URL}/app/itinerary", wait_until="domcontentloaded")
                baseline = capture("original_baseline")
                assert baseline["active_plan_id"] == immutable_parent["plan_id"]
                assert baseline["active_plan_content_hash"] == immutable_parent["content_hash"]
                parent = immutable_parent
                for plan_context in ("original", "accepted_child"):
                    workspace = deepcopy(observed["workspace"])
                    assert observed["session"]["accepted_plan_id"] == parent["plan_id"]
                    assert workspace["accepted_plan_pointer"]["accepted_plan_content_hash"] == parent["content_hash"]
                    geography = next(plan for plan in workspace["geography"]["plans"] if plan["plan_id"] == parent["plan_id"])
                    assert geography["content_hash"] == parent["content_hash"]
                    capability = workspace["draft_capabilities"]["operations"]["add_candidate"]
                    choices = workspace["draft_capabilities"]["candidate_choices"]
                    candidates = sorted(row["candidate_id"] for row in choices)
                    days = sorted(row["day"] for row in parent["ordered_days"])
                    readiness = capability["candidate_day_readiness"]
                    expected_pairs = {(candidate, day) for candidate in candidates for day in days}
                    assert len(readiness) == len(expected_pairs) > 0
                    assert {(row["candidate_id"], row["day"]) for row in readiness} == expected_pairs
                    assert capability["enabled"] is any(row["preview_executable"] for row in readiness)
                    snapshots[plan_context] = {"parent": parent, "map_ack": baseline, "session_id": observed["session"]["session_id"],
                        "candidate_choices": choices, "days": days, "capability": capability, "geography": geography}
                    _write_json(directory / "capability_snapshots.json", snapshots)
                    if plan_context=='accepted_child':
                        phase='accepted_poi_controls'
                        accepted_poi_controls(baseline,geography)
                    editor(7)  # No replacement candidate targets Griffith; expose the full add catalog.
                    controls = inventory()
                    initial_controls = {item["id"]: item for item in controls}
                    candidate_options = initial_controls["registered-candidate"]["options"]
                    assert {option["value"] for option in candidate_options if option["value"]} == set(candidates)
                    placeholders = [option for option in candidate_options if not option["value"]]
                    assert len(placeholders) <= 1
                    if placeholders:
                        assert placeholders[0]["label"] == "Choose a registered candidate"
                    assert {int(option["value"]) for option in initial_controls["destination-day"]["options"]} == set(days)
                    for candidate in candidates:
                        candidate_control = next(item for item in inventory() if item["id"] == "registered-candidate")
                        candidate_option = next(option for option in candidate_control["options"] if option["value"] == candidate)
                        candidate_selectable = not candidate_control["disabled"] and not candidate_option["disabled"]
                        if candidate_selectable:
                            select_candidate(candidate)
                        for day in days:
                            phase = f"{plan_context}_{candidate}_day_{day}"
                            row = next(item for item in readiness if item["candidate_id"] == candidate and item["day"] == day)
                            cell_evidence = _check_required_legs(row, parent, geography, matrix)
                            before = inventory()
                            by_id = {item["id"]: item for item in before}
                            day_option = next(option for option in by_id["destination-day"]["options"] if option["value"] == str(day))
                            can_select = candidate_selectable and not by_id["destination-day"]["disabled"] and not day_option["disabled"]
                            if can_select:
                                page.locator("#destination-day").select_option(str(day))
                                expect(page.locator("#destination-day")).to_have_value(str(day))
                            selected_controls = inventory()
                            mutations_before = count("/draft/operations")
                            case = {"context": plan_context, "candidate_id": candidate, "day": day, "readiness": row,
                                "required_leg_evidence": cell_evidence, "controls_before": before, "controls_selected": selected_controls,
                                "pair_selected_through_enabled_controls": can_select, "forced_interaction": False,
                                "accepted_plan_id": parent["plan_id"], "accepted_plan_content_hash": parent["content_hash"],
                                "scope": "candidate_readiness_not_independent_eligibility", "comparison_eligibility": "not_evaluated"}
                            cases.append(case)
                            inventories.append({"phase": phase, "controls": selected_controls})
                            if not can_select or page.locator("#draft-add").is_disabled():
                                assert row["preview_executable"] is False and row["blocking_findings"]
                                if can_select or not capability["enabled"]:
                                    expect(page.locator("#draft-add")).to_be_disabled()
                                expect(page.locator("#draft-preview")).to_be_disabled()
                                case.update(scope="disabled_pair_not_interacted", disabled_selectors=[item["id"] for item in selected_controls if item["disabled"]],
                                            candidate_option_disabled=candidate_option["disabled"], day_option_disabled=day_option["disabled"],
                                            visible_reason=page.locator("#edit-target").inner_text(), draft_request_delta=count("/draft/operations")-mutations_before)
                                assert case["draft_request_delta"] == 0
                            else:
                                input_receipt = before_action(page, page.locator("#draft-add"))
                                with page.expect_response(lambda response: response.url.endswith("/draft/impact-preview")) as impacted:
                                    with page.expect_response(lambda response: response.url.endswith("/draft/operations")) as staged:
                                        page.locator("#draft-add").click()
                                assert staged.value.status == impacted.value.status == 200
                                draft, impact = staged.value.json(), impacted.value.json()
                                request = staged.value.request.post_data_json
                                assert request["type"] == "add_candidate" and request["target"] == candidate and request["parameters"] == {"day": day}
                                assert draft["session"]["draft"] == [draft["operation"]]
                                assert draft["session"]["accepted_plan_id"] == parent["plan_id"] and draft["session"]["proposal"] is None
                                assert impact["parent_plan_id"] == parent["plan_id"] and impact["parent_plan_content_hash"] == parent["content_hash"]
                                assert impact["session_revision"] == draft["session"]["revision"]
                                assert impact["is_certified"] is False and impact["certification_status"] == "not_certified"
                                assert len(impact["operations"]) == impact["summary"]["operation_count"] == 1
                                item = impact["operations"][0]
                                assert item["operation_id"] == draft["operation"]["operation_id"]
                                assert item["type"] == "add_candidate" and item["target"] == candidate
                                assert item["preview_executable"] is row["preview_executable"]
                                assert item["evaluated_repair"] is row["preview_executable"]
                                assert item["feedback_tier"] == ("evaluated" if row["preview_executable"] else "draft_only")
                                assert impact["summary"]["can_run_evaluated_preview"] is row["preview_executable"]
                                assert impact["summary"]["evaluated_executable_count"] == int(row["preview_executable"])
                                assert impact["summary"]["draft_only_count"] == int(not row["preview_executable"])
                                assert item["blocking_codes"] == ([] if row["preview_executable"] else [row["status"]])
                                assert impact["summary"]["blocking_codes"] == item["blocking_codes"]
                                expect(page.locator(".typed-impact")).to_contain_text("Not certified.")
                                if row["preview_executable"]:
                                    expect(page.locator("#draft-preview")).to_be_enabled()
                                    expect(page.locator(".typed-impact")).to_contain_text("1 evaluated-preview capable")
                                else:
                                    expect(page.locator("#draft-preview")).to_be_disabled()
                                    expect(page.locator(".typed-impact-blocker")).to_contain_text("Preview blocked:")
                                    expect(page.locator(".typed-impact")).to_contain_text("1 draft only")
                                    assert row["status"].replace("_", " ").lower() in page.locator(".typed-impact-blocker").inner_text().lower()
                                control_receipt = after_action(page, input_receipt, context=plan_context, phase=phase,
                                    request=request, response=draft)
                                case.update(scope="real_staging_and_impact_only", request=request, operation=draft["operation"], impact=impact,
                                            control_action_receipt=control_receipt,
                                            visible_impact=page.locator(".typed-impact").inner_text(), controls_after_stage=inventory(),
                                            draft_request_delta=count("/draft/operations")-mutations_before)
                                assert case["draft_request_delta"] == 1
                                page.locator("#close-map-edit").click()
                                (expect(page.locator("#inline-preview")).to_be_enabled() if row["preview_executable"]
                                 else expect(page.locator("#inline-preview")).to_be_disabled())
                                expect(page.locator("#accept-repair")).to_be_disabled()
                                case["map_ack"] = capture(phase + "_staged", baseline)
                                with page.expect_response(lambda response: response.url.endswith("/draft/impact-preview")):
                                    with page.expect_response(lambda response: response.url.endswith("/draft/undo")) as undone:
                                        page.locator("#inline-undo").click()
                                assert undone.value.status == 200
                                assert undone.value.json()["undone"] == draft["operation"]
                                assert undone.value.json()["session"]["draft"] == []
                                assert undone.value.json()["session"]["accepted_plan_id"] == parent["plan_id"]
                                case["undo_response"] = undone.value.json()
                                editor(7)
                                select_candidate(candidate)
                            case["assertions_passed"] = True
                            _write_json(directory / "candidate_day_cases.json", cases)
                    assert {(item["candidate_id"], item["day"]) for item in cases if item["context"] == plan_context} == expected_pairs
                    page.locator("#close-map-edit").click()
                    capture(plan_context + "_matrix_complete", baseline)
                    if plan_context == "accepted_child":
                        break
                    phase = "establish_accepted_child_through_real_ui"
                    editor(5)
                    select_candidate(PFEIFFER)
                    with page.expect_response(lambda response: response.url.endswith("/draft/impact-preview")):
                        with page.expect_response(lambda response: response.url.endswith("/draft/operations")) as replacement:
                            page.locator("#draft-replace").click()
                    assert replacement.value.status == 200
                    assert replacement.value.request.post_data_json["target"] == BIXBY
                    assert replacement.value.request.post_data_json["parameters"] == {"candidate_id": PFEIFFER}
                    page.locator("#close-map-edit").click()
                    expect(page.locator("#inline-preview")).to_be_enabled()
                    with page.expect_response(lambda response: response.url.endswith("/preview"), timeout=90_000) as previewed:
                        page.locator("#inline-preview").click()
                    assert previewed.value.status == 200
                    proposal = previewed.value.json()["proposal"]
                    execution = proposal["runtime_execution"]
                    parent = execution["child_plan"]
                    certificate, diff = execution["evaluation_certificate"], execution["plan_diff"]
                    assert execution["evaluation_executed"] is True and execution["evaluator"] == "PlanEvaluator.evaluate_final_plan"
                    assert proposal["state"] == certificate["comparison_eligibility"] == "eligible"
                    assert parent["plan_id"] == proposal["child_plan_id"] == certificate["plan_id"] == diff["child_plan_id"]
                    assert parent["content_hash"] == proposal["child_plan_content_hash"] == certificate["plan_content_hash"]
                    assert parent["parent_plan_id"] == immutable_parent["plan_id"] == diff["parent_plan_id"]
                    expect(page.locator(".map-stage")).to_have_attribute("data-preview-state", "PREVIEW_RENDERED", timeout=45_000)
                    preview_ack = capture("accepted_child_preview_ready")
                    assert preview_ack["active_plan_id"] == parent["plan_id"] and preview_ack["active_plan_content_hash"] == parent["content_hash"]
                    assert preview_ack["rendered_geometry_hash"] == _independent_geometry_hash(page, proposal["geography_plan"])
                    assert set(preview_ack["rendered_marker_ids"]) == (set(baseline["rendered_marker_ids"])-{BIXBY}) | {PFEIFFER}
                    with page.expect_response(lambda response: response.url.endswith("/accept"), timeout=60_000) as accepted:
                        page.locator("#accept-repair").click()
                        expect(page.get_by_role("dialog", name="Confirm trip decision", exact=True)).to_be_visible()
                        expect(page.locator("#acceptance-message")).to_have_text('Accept this independently evaluated repair as the durable trip plan?')
                        page.get_by_role("dialog", name="Confirm trip decision", exact=True).get_by_role("button", name='Accept repair', exact=True).click()
                    assert accepted.value.status == 200
                    decision = accepted.value.json()["result"]["decision"]
                    assert decision["selected_plan_id"] == parent["plan_id"] and decision["selected_plan_content_hash"] == parent["content_hash"]
                    session_id = accepted.value.json()["session"]["session_id"]
                    with page.expect_response(lambda response, expected_id=session_id: response.url.endswith(f"/api/sessions/{expected_id}") and response.request.method == "GET") as restored:
                        page.reload(wait_until="domcontentloaded")
                    assert restored.value.status == 200
                    saved = restored.value.json()
                    assert saved["session"]["accepted_plan_id"] == parent["plan_id"] and saved["session"]["draft"] == [] and saved["session"]["proposal"] is None
                    assert saved["workspace"]["accepted_plan_pointer"]["decision_id"] == decision["decision_id"]
                    assert saved["workspace"]["alternative_evidence"][parent["plan_id"]]["certificate"] == certificate
                    baseline = capture("accepted_child_after_reload", preview_ack)
                    _write_json(directory / "accepted_baseline_artifacts.json", {"preview_response": previewed.value.json(),
                        "acceptance_response": accepted.value.json(), "restored_workspace": saved["workspace"], "map_ack": baseline})
                assert count("/preview") == count("/accept") == 1
                assert count("/copilot/messages") == 0
                phase = "final_candidate_matrix"
                asset_sources = page.evaluate("""() => {
                  const map = """ + CURRENT_MAP + """;
                  const style = map.getStyle();
                  return {container_id:map.getContainer().id,
                    map_dataset:{...map.getContainer().dataset},sprite:style.sprite??null,
                    vector_sources:Object.fromEntries(Object.entries(style.sources)
                      .filter(([,source])=>source.type==='vector'))};
                }""")
                policy = map_cancellation_policy(map_configuration,
                    {"sources": asset_sources["vector_sources"], "sprite": asset_sources["sprite"]})
                classification = classify_browser_network(errors["request_failures"], policy)
                _write_json(directory / "map_asset_sources.json", {"schema_version": "real-map-asset-sources-v1",
                    "scope": "Read-only public getStyle() vector sources and actual ready container dataset",
                    "phase": phase, "context": "accepted_child",
                    "map_configuration_http_file": map_configuration_http_file, **asset_sources})
                _write_json(directory / "network_classification.json", {
                    "map_configuration_http_file": map_configuration_http_file,
                    "rendered_map_asset_sources_file": "map_asset_sources.json", **classification})
                assert all(not values for key, values in errors.items() if key != "request_failures"), errors
                assert classification["blocking_failure_indexes"] == [], errors
                _write_json(directory / "browser_assertions.json", {"schema_version": "candidate-day-matrix-browser-v1", "assertions_passed": True,
                    "contexts": list(snapshots), "pair_count": len(cases), "pairs_by_context": {name: sum(row["context"]==name for row in cases) for name in snapshots},
                    "forced_disabled_interactions": 0, "direct_api_setup_calls": 0, "provider_call_count": 0,
                    "preview_count": 1, "acceptance_count": 1, "pair_scope": "readiness_and_staging_not_eligibility",
                    "eligible_addition_acceptance_owner": "addition-continuation", "all_pairs_recorded": True})
            finally:
                _write_json(directory / "http_index.json", http)
                _write_json(directory / "candidate_day_cases.json", cases)
                _write_json(directory / "runtime_control_inventory.json", inventories)
                _write_json(directory / "map_checkpoints.json", checkpoints)
                _write_json(directory / "browser_errors.json", errors)
                _write_json(directory / "llm_io.json", {"provider_call_count": 0, "llm_input": None, "llm_output": None,
                    "reason": "No Copilot submission. All mutations use actual candidate/day, draft, preview, accept and Undo controls."})
        browser.close()
