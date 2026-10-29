"""Preservation and family-level chat closure through real controls and HTTP/SSE.

Two ordered cases share one ordinary fresh deterministic server. The read-only
observer is not an API setup client: it reads that server's disposable journal
and reconstructs the exact provider-safe context to check the recorded hash.
No chat-family sample claims every possible natural-language/LLM variant.
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from urllib.parse import urlsplit

import pytest
from fastapi.encoders import jsonable_encoder

pytest.importorskip("playwright.sync_api")
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
from waypoint_ui_fixture import (
    CURRENT_MAP,
    OBSERVE_REAL_MAPS,
    browser_network_record,
    classify_browser_network,
    map_cancellation_policy,
)

ROOT = Path(__file__).resolve().parents[3]
BASE_URL = os.environ.get("PRODUCT_E2E_BASE_URL")
BIXBY = "bixby_creek_bridge_viewpoint"
PFEIFFER = "pfeiffer_big_sur_state_park"
POINT_LOBOS = "point_lobos_state_natural_reserve"
GOLDEN_GATE = "golden_gate_bridge"
STEARNS = "stearns_wharf"


def _hash(record):
    body = {key: value for key, value in record.items() if key != "content_hash"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()[:16]


def _identity(ack):
    return {key: value for key, value in ack.items() if key != "preview_state"}


def _session_wire(session):
    """Encode every public field as FastAPI does, including tuple evidence refs."""
    return jsonable_encoder(session.public_dict())


def _assert_accepted_reference(pointer, public, session, accepted, original, parent, initial, decision):
    """Check the full durable pointer, its public projection, and immutable plans."""
    public_fields = {"schema_version", "accepted_plan_id", "accepted_plan_content_hash", "decision_id", "revision"}
    assert set(public) == public_fields
    assert set(pointer) == public_fields | {"trip_id", "run_id", "original_plan_id", "original_plan_content_hash",
        "updated_at", "content_hash"}
    assert public == {key: pointer[key] for key in public_fields}
    unsigned = {key: value for key, value in pointer.items() if key != "content_hash"}
    assert pointer["content_hash"] == hashlib.sha256(json.dumps(unsigned, sort_keys=True,
        separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()
    assert pointer["schema_version"] == "workspace-pointer-v1"
    assert type(pointer["revision"]) is int and pointer["revision"] >= 0
    assert datetime.fromisoformat(pointer["updated_at"]).tzinfo is not None
    assert pointer["trip_id"] == session["trip_id"] and pointer["run_id"] == session["run_id"]
    assert pointer["original_plan_id"] == original["plan_id"] == initial["active_plan_id"]
    assert pointer["original_plan_content_hash"] == original["content_hash"] == _hash(original) == initial["active_plan_content_hash"]
    assert pointer["accepted_plan_id"] == accepted["plan_id"] == session["accepted_plan_id"] == parent["active_plan_id"]
    assert pointer["accepted_plan_content_hash"] == accepted["content_hash"] == _hash(accepted) == parent["active_plan_content_hash"]
    assert parent["rendered_map_plan_id"] == accepted["plan_id"]
    assert set(parent["rendered_marker_ids"]) == set(accepted["sequence"])
    if decision is None:
        assert pointer["decision_id"] is None and pointer["revision"] == 0
        assert accepted == original
    else:
        decision_body = {key: value for key, value in decision.items() if key != "content_hash"}
        assert decision["content_hash"] == hashlib.sha256(json.dumps(decision_body, sort_keys=True,
            separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()
        assert pointer["decision_id"] == decision["decision_id"]
        assert pointer["revision"] == decision["pointer_revision_after"]
        assert pointer["trip_id"] == decision["trip_id"] and pointer["run_id"] == decision["run_id"]
        assert accepted["plan_id"] == decision["selected_plan_id"]
        assert accepted["content_hash"] == decision["selected_plan_content_hash"]
    return {"schema_version": "chat-accepted-reference-witness-v1", "durable_pointer": pointer,
        "public_pointer": public, "session_identity": {key: session[key] for key in ("session_id", "trip_id", "run_id", "accepted_plan_id")},
        "accepted_plan": accepted, "original_plan": original, "accepted_map_reference": parent,
        "original_map_reference": initial, "decision": decision, "assertions_passed": True}


@pytest.mark.skipif(not BASE_URL or os.environ.get("PRODUCT_E2E_CHAT_CLOSURE_DISPOSABLE") != "1",
    reason="Requires a fresh isolated deterministic chat-closure server")
@pytest.mark.parametrize("plan_context", ["original", "accepted_child"])
def test_keep_and_chat_families_preserve_runtime_truth(tmp_path, plan_context):
    from itinerary_system.product_app.copilot import DeterministicCopilotAdapter
    from itinerary_system.product_app.copilot_memory import bound_records, context_binding
    from itinerary_system.product_app.registry import ProductRunRegistry
    from itinerary_system.product_app.service import ProductService

    evidence = Path(os.environ.get("PRODUCT_E2E_CHAT_CLOSURE_EVIDENCE_DIR", tmp_path)) / plan_context
    evidence.mkdir(parents=True, exist_ok=False)
    state_path = os.environ.get("PRODUCT_E2E_CHAT_CLOSURE_STATE_ROOT")
    assert state_path and Path(state_path).is_dir(), "Runner must expose its exact disposable state root for read-only journal evidence"
    observer = ProductService(ProductRunRegistry(ROOT, ROOT / "configs/product_app_registry.json"), Path(state_path),
        copilot=DeterministicCopilotAdapter())
    http, turns, journeys, negatives, errors, failures, requests = [], [], [], [], [], [], []
    observed = {}
    phase = "initial"
    late_conversation_race = None
    undo_selections = []
    generated_actions, generated_documents = [], {}
    map_configuration = None
    map_configuration_http_file = None
    network_policy = None
    with sync_playwright() as runtime:
        browser = runtime.chromium.launch(headless=True, **({"executable_path": str(CHROME)} if CHROME.is_file() else {}))
        context = browser.new_context(viewport={"width": 1440, "height": 900})
        context.add_init_script(OBSERVE_REAL_MAPS)
        page = context.new_page()
        page.on("console", lambda message: errors.append(message.text) if message.type == "error" else None)
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.on("requestfailed", lambda request: failures.append(browser_network_record(request, phase)))
        page.on("request", lambda request: requests.append({"method": request.method, "url": request.url}))

        def record(response, raw=None):
            nonlocal map_configuration, map_configuration_http_file
            if response.status >= 400:
                failures.append(browser_network_record(response.request, phase, http_status=response.status))
            if not urlsplit(response.url).path.startswith("/api/") or (response.url.endswith("/copilot/messages") and raw is None):
                return
            captured_phase = phase
            row = {"phase": captured_phase, **_http_record(response, response_body=raw)}
            name = f"http_{len(http)+1:03d}.json"  # Allocate after body reads, which can re-enter callbacks.
            _write_json(evidence / name, row)
            http.append({"file": name, "phase": captured_phase, "method": row["method"],
                "url": row["url"], "status": row["response_status_code"]})
            if raw is None and row["response_status_code"] == 200:
                payload = json.loads(row["response_body_utf8"])
                if response.url == f"{BASE_URL}/api/map/config":
                    assert map_configuration is None or payload == map_configuration, "Map configuration changed during the journey"
                    map_configuration, map_configuration_http_file = payload, name
                if isinstance(payload, dict):
                    for key in ("workspace", "session"):
                        if key in payload:
                            observed[key] = payload[key]
                    if row["url"].endswith("/draft/impact-preview"):
                        observed["impact"] = payload
        page.on("response", record)
        page.add_init_script("""(() => {
          const original = window.fetch.bind(window); window.__closureStreams = [];
          window.__closureLateConversation = null; window.__closureMapSelectClicks = [];
          window.__closureGeneratedClicks=[];
          document.addEventListener('click', event => {
            const button=event.target.closest?.('.copilot-proposal-actions button');
            if(button) window.__closureGeneratedClicks.push({trusted:event.isTrusted,at:Date.now(),path:location.pathname,
              label:button.textContent.trim(),aria_label:button.getAttribute('aria-label'),
              client_message_id:button.closest('.message-assistant')?.dataset.clientMessageId,
              selector:button.closest('.copilot-proposal-action-row')?'.copilot-proposal-action-row button':'.copilot-proposal-actions > button',
              intent_type:button.closest('.copilot-proposal-action-row')?.dataset.copilotIntentType||null});
            if (event.target.closest?.('#map-select-button')) window.__closureMapSelectClicks.push({
              selector:'#map-select-button',trusted:event.isTrusted,clicked_at:new Date().toISOString()});
          }, true);
          window.fetch = async (...args) => {
            const url = new URL(typeof args[0] === 'string' ? args[0] : args[0].url, location.href);
            const method = (args[1]?.method || args[0]?.method || 'GET').toUpperCase();
            const holdSession = sessionStorage.getItem('__closureHoldConversationOnce');
            const hold = method === 'GET' && holdSession && url.pathname === `/api/sessions/${holdSession}/conversation`;
            let held = null;
            if (hold) {
              sessionStorage.removeItem('__closureHoldConversationOnce');
              held = {url:url.href,method,request_body_utf8:'',status:null,body_utf8:null,body_sha256:null,
                requested_at:new Date().toISOString(),body_completed_at:null,held_at:null,released_at:null,
                application_json_read_at:null,delivery_settled_at:null,release_count:0,error:null};
              window.__closureLateConversation = held;
            }
            const response = await original(...args);
            if (url.pathname.endsWith('/copilot/messages')) {
              const row = {raw:null,error:null}; window.__closureStreams.push(row);
              response.clone().text().then(raw => {row.raw=raw;}, error => {row.error=error.name;});
            }
            if (held) {
              // Hold delivery only after receiving the complete real response.
              // The original response object/body is returned, never a fixture.
              held.status = response.status;
              held.body_utf8 = await response.clone().text();
              held.body_completed_at = new Date().toISOString();
              held.body_sha256 = Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',
                new TextEncoder().encode(held.body_utf8))), value => value.toString(16).padStart(2,'0')).join('');
              const actualJson = response.json.bind(response);
              response.json = async (...jsonArgs) => {
                try {
                  const payload = await actualJson(...jsonArgs);
                  held.application_json_read_at = new Date().toISOString();
                  // The next task is after api()/restore() promise continuations.
                  setTimeout(() => {held.delivery_settled_at = new Date().toISOString();}, 0);
                  return payload;
                } catch (error) { held.error = error.name; throw error; }
              };
              held.held_at = new Date().toISOString();
              await new Promise(resolve => {
                window.__releaseClosureConversation = () => {
                  if (held.release_count !== 0) throw new Error('Conversation response already released');
                  held.release_count += 1; held.released_at = new Date().toISOString(); resolve();
                };
              });
            }
            return response;
          };
        })();""")

        def mutations():
            return [row for row in requests if row["method"] == "POST"
                and row["url"].endswith(("/draft/operations", "/preview", "/accept"))]

        def open_chat():
            if not page.locator("#copilot-dock").is_visible():
                page.locator("#open-copilot").click()

        def close_chat():
            if page.locator("#copilot-dock").is_visible():
                page.locator("#close-copilot").click()

        def generated_before(control):
            observed_control=control.evaluate("""button=>({
              label:button.textContent.trim(),
              aria_label:button.getAttribute('aria-label'),client_message_id:button.closest('.message-assistant')?.dataset.clientMessageId,
              assistant_text:button.closest('.message-assistant')?.querySelector('p')?.textContent,
              selector:button.closest('.copilot-proposal-action-row')?'.copilot-proposal-action-row button':'.copilot-proposal-actions > button',
              intent_type:button.closest('.copilot-proposal-action-row')?.dataset.copilotIntentType||null,
              path:location.pathname,observed_at:Date.now()})""")
            observed_control['visible']=control.is_visible()
            observed_control['disabled']=control.is_disabled()
            return {'control':observed_control,
                  'event_start':page.evaluate('window.__closureGeneratedClicks.length'),
                  'document_id':page.evaluate('String(performance.timeOrigin)'),
                  'request_start':len(requests),'session':_session_wire(read_session())}

        def generated_finish(variant,before,*,response=None,map_ack=None,result_selector=None):
            document_id=page.evaluate('String(performance.timeOrigin)')
            assert document_id==before['document_id']
            events=page.evaluate('window.__closureGeneratedClicks')
            generated_documents[document_id]=events
            session=_session_wire(read_session())
            assert session['accepted_plan_id']==before['session']['accepted_plan_id']
            generated_actions.append({'variant':variant,'context':'original' if session['accepted_plan_id']==initial['active_plan_id'] else 'accepted_child',
                'phase':phase,'input':before,'output':{'session':session,'path':urlsplit(page.url).path,'map':map_ack,
                    'result_selector':result_selector,'view':page.locator(result_selector).evaluate("""e=>({visible:!!e.getClientRects().length,
                      attributes:Object.fromEntries([...e.attributes].map(a=>[a.name,a.value]))})""") if result_selector else None,
                    'observed_at':page.evaluate('Date.now()'),'http':_http_record(response) if response else None},
                'events':events,'requests':requests[before['request_start']:],'assertions_passed':True})
            _write_json(evidence/'generated_control_actions.json',generated_actions)
            _write_json(evidence/'generated_control_observer.json',{'schema_version':'generated-control-observer-v1','documents':generated_documents})

        def expand(details):
            if details.get_attribute("open") is None:
                details.locator(":scope > summary").click()

        def open_stop(day):
            close_chat()
            if page.locator("#map-edit-dialog").is_visible():
                page.locator("#close-map-edit").click()
            card = page.locator(f'.day-card[data-day="{day}"]')
            stop_id = card.get_attribute("data-stop")
            draft_before = observed["session"]["draft"]
            with page.expect_response(lambda response: response.url.endswith("/selection")
                    and response.request.post_data_json["selected_day"] == day
                    and response.request.post_data_json["selected_stop_id"] == stop_id, timeout=30_000) as selected:
                card.click()
            assert selected.value.status == 200, selected.value.text()
            selection = selected.value.json()["session"]
            # The impact response can precede Python's reading of the selection
            # body. Observe the existing response ledger, not a late listener.
            for _ in range(600):
                if observed.get("impact", {}).get("session_revision") == selection["revision"]:
                    break
                page.wait_for_timeout(50)
            checked = observed.get("impact", {})
            assert checked["session_revision"] == selection["revision"]
            assert (selection["selected_day"], selection["selected_stop_id"]) == (day, stop_id)
            assert selection["accepted_plan_id"] == parent["active_plan_id"] and selection["draft"] == draft_before
            expect(card).to_have_attribute("aria-pressed", "true", timeout=30_000)
            page.locator("#map-edit-button").click()
            page.locator("#inspect-edit-stop").click()
            expect(page.locator('[data-editor-kind="stop"]')).to_be_visible(timeout=30_000)
            expand(page.locator(".legacy-edit-controls"))
            _write_json(evidence / f"{phase}_selected_day_{day}.json", {"request": selected.value.request.post_data_json,
                "response": selected.value.json(), "impact": checked, "selected_day": day,
                "selected_stop_id": stop_id, "editor_visible": True, "assertions_passed": True})

        def read_session():
            # Workspace.get is process-cached. Read the actual server snapshot
            # under its existing lock; never use authentication or a write API.
            with observer.workspace._snapshots.locked():
                return observer.workspace._load_current_locked(observed["session"]["session_id"])

        def read_context(label):
            session = read_session()
            assert session.conversation_id, "The actual browser must create its conversation before observation"
            pointer = observer.acceptance.load_pointer(session.trip_id)
            accepted = observer.acceptance.plan_repository.load(pointer["accepted_plan_id"]).to_record(include_content_hash=True)
            original = observer.acceptance.plan_repository.load(pointer["original_plan_id"]).to_record(include_content_hash=True)
            decisions = observer.acceptance.history(session.trip_id)
            decision = next((row for row in decisions if row["decision_id"] == pointer["decision_id"]), None)
            reference_inputs = {"pointer": pointer, "public": observed["workspace"]["accepted_plan_pointer"],
                "session": session.public_dict(), "accepted": accepted, "original": original,
                "parent": parent, "initial": initial, "decision": decision}
            _write_json(evidence / f"accepted_reference_inputs_{label}.json", reference_inputs)
            witness = _assert_accepted_reference(**reference_inputs)
            _write_json(evidence / f"accepted_reference_{label}.json", witness)
            if session.run_id not in observer._bundles:
                # The browser has already created the pointer. Registry loading
                # now reads immutable source bundles rather than seeding a trip.
                observer.load(session.run_id)
                assert observer.acceptance.load_pointer(session.trip_id) == pointer
            records = observer.conversations.dispatch_history(session.conversation_id, session.session_id)
            # Supplying the browser's workspace and existing journal avoids the
            # normal context builder's conversation-creation/session-save path.
            projected = observer.copilot_context(session, view=observed["workspace"], dispatch_records=records)
            assert projected.accepted_plan_id == session.accepted_plan_id == parent["active_plan_id"]
            assert projected.accepted_plan_content_hash == parent["active_plan_content_hash"]
            assert set(projected.allowed_stop_ids) == set(parent["rendered_marker_ids"])
            _write_json(evidence / f"context_{label}.json", projected.as_dict())
            return projected

        def send(message, label):
            nonlocal phase
            phase = label
            open_chat()
            before = read_context(label + "_before")
            session = read_session()
            records = observer.conversations.dispatch_history(session.conversation_id, session.session_id)
            history = [observer.conversations._validate_turn(row["turn"]).as_dict() for row in bound_records(before, records)]
            digest = hashlib.sha256(json.dumps({"context": before.as_dict(), "history": history},
                sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
            original_mutations = len(mutations())
            index = page.evaluate("window.__closureStreams.length")
            page.locator("#copilot-input").fill(message)
            with page.expect_response(lambda response: response.url.endswith("/copilot/messages"), timeout=60_000) as completed:
                page.locator("#copilot-submit").click()
            response = completed.value
            assert response.status == 200
            page.wait_for_function("i => window.__closureStreams[i]?.raw !== null || window.__closureStreams[i]?.error !== null",
                arg=index, timeout=60_000)
            stream = page.evaluate("i => window.__closureStreams[i]", index)
            assert stream["error"] is None
            record(response, stream["raw"])
            events = _sse_events(stream["raw"])
            assert events[-1]["type"] == "turn.completed", events
            payload = events[-1]["payload"]
            turn = payload["turn"]
            assert turn["provider"] == "deterministic" and turn["user_message"] == message
            assert turn["context_revision"] == before.session_revision
            expect(page.locator("#copilot-lifecycle")).to_have_attribute("data-state", turn["interpretation"]["state"])
            expect(page.locator(".message-user").last.locator("p").first).to_have_text(message)
            expect(page.locator(".message-assistant").last.locator("p").first).to_have_text(turn["assistant_message"])
            assert len(mutations()) == original_mutations, "Sending a suggestion must not stage, preview, or accept"
            observed["session"] = payload["session"]
            journal = observer.conversations.dispatch_history(session.conversation_id, session.session_id)
            dispatch = next(row for row in journal if row["client_message_id"] == response.request.post_data_json["client_message_id"])
            assert dispatch["context_binding"] == context_binding(before)
            assert dispatch["context_sha256"] == digest
            assert dispatch["attempt_count"] == dispatch["attempt_budget"] == 1 and dispatch["outcome"] == "completed"
            turns.append({"label": label, "input": response.request.post_data_json, "output": payload, "events": events,
                "context": before.as_dict(), "bound_history": history, "context_sha256": digest, "dispatch_record": dispatch})
            _write_json(evidence / "exact_chat_io.json", turns)
            return payload

        def stage(kind, target, parameters, control, source="map", count=1):
            expect(control).to_be_visible(timeout=30_000)
            expect(control).to_be_enabled()
            generated_input=generated_before(control) if source=='copilot' else None
            with page.expect_response(lambda response: response.url.endswith("/draft/operations")) as completed:
                control.click()
            assert completed.value.status == 200, completed.value.text()
            actual, result = completed.value.request.post_data_json, completed.value.json()
            assert {key: actual[key] for key in ("type", "target", "parameters", "source")} == {
                "type": kind, "target": target, "parameters": parameters, "source": source}
            assert all(result["operation"][key] == actual[key] for key in ("type", "target", "parameters", "source"))
            assert result["session"]["draft"][-1] == result["operation"] and len(result["session"]["draft"]) == count
            expect(page.locator(".draft-bar strong")).to_have_text(f"{count} draft change" + ("s" if count != 1 else ""), timeout=15_000)
            assert result["session"]["accepted_plan_id"] == parent["active_plan_id"]
            assert _identity(_map_ack(page)) == _identity(parent)
            if generated_input:
                generated_finish('proposal_add',generated_input,response=completed.value,map_ack=_map_ack(page))
            return result["session"]["draft"]

        def stage_chat(payload):
            interpretation = payload["turn"]["interpretation"]
            assert interpretation["state"] == "proposal_ready" and len(interpretation["intents"]) == 1
            intent = interpretation["intents"][0]
            if intent["type"] == "replace_nearby":
                target, parameters = intent["target_stop_id"], {"candidate_id": intent["candidate_id"]}
            elif intent["type"] == "add_candidate":
                target, parameters = intent["candidate_id"], {"day": intent["destination_day"]}
            else:
                target, parameters = intent["target_stop_id"], intent.get("parameters") or {}
            actions = page.locator(".copilot-proposal-actions").last
            operations = stage(intent["type"], target, parameters, actions.get_by_role("button", name="Add to draft", exact=False), "copilot")
            expect(page.locator("#copilot-lifecycle")).to_contain_text("Draft staged", timeout=30_000)
            added = actions.get_by_role("button", name="Added to draft", exact=False)
            expect(added).to_be_visible(timeout=30_000)
            expect(added).to_be_disabled()
            generated_finish('proposal_added_disabled',generated_before(added),map_ack=_map_ack(page))
            expect(actions.get_by_role("button", name="Preview this repair", exact=False)).to_be_enabled()
            assert read_context(phase + "_staged").conversation_memory["last_proposal"]["disposition"] == "staged"
            return operations

        def surface(ack, geography):
            expect(page.locator("#metric-strip")).to_have_attribute("data-plan-id", ack["active_plan_id"])
            expect(page.locator("#metric-strip")).to_have_attribute("data-plan-content-hash", ack["active_plan_content_hash"])
            expand(page.locator(".textual-map-alternative"))
            text = page.locator(".text-route-summary")
            expect(text).to_be_visible()
            expect(text).to_have_attribute("data-plan-id", ack["active_plan_id"])
            expect(text).to_have_attribute("data-plan-content-hash", ack["active_plan_content_hash"])
            stops = [feature["properties"] for feature in geography["stops"]["features"]]
            for day in range(1, 8):
                ordered = sorted((stop for stop in stops if int(stop["day"]) == day), key=lambda stop: stop["stop_order"])
                card = page.locator(f'.day-card[data-day="{day}"]')
                expect(card.locator(".day-copy strong")).to_have_text(" · ".join(stop["name"] for stop in ordered) or "No recorded stop")
                expect(card).to_have_attribute("data-stop", ordered[0]["stop_id"] if ordered else "")
            return {"metrics": page.locator("#metric-strip").evaluate("e => ({...e.dataset})"),
                "timeline": page.locator("#day-list").inner_text(), "text_route": text.inner_text()}

        def preview(label, operations, outcome, *, chat=False):
            nonlocal phase, late_conversation_race
            phase = label
            case = evidence / label
            case.mkdir()
            if page.locator("#map-edit-dialog").is_visible():
                page.locator("#close-map-edit").click()
            _capture_ready(page, case / "staged_parent.png")
            control = (page.locator(".copilot-proposal-actions").last.get_by_role("button", name="Preview this repair", exact=False)
                if chat else page.locator("#inline-preview"))
            expect(control).to_be_enabled(timeout=15_000)
            generated_input=generated_before(control) if chat else None
            with page.expect_response(lambda response: response.url.endswith("/preview"), timeout=90_000) as completed:
                control.click()
            assert completed.value.status == 200, completed.value.text()
            payload = completed.value.json()
            _write_json(case / "preview_response.json", payload)
            proposal = payload["proposal"]
            assert proposal["state"] == proposal["eligibility"] == outcome, proposal.get("reason")
            assert proposal["parent_plan_id"] == parent["active_plan_id"]
            assert proposal["parent_plan_content_hash"] == parent["active_plan_content_hash"]
            assert payload["session"]["accepted_plan_id"] == parent["active_plan_id"]
            assert payload["session"]["draft"] == operations
            assert proposal["compiled_request"]["operations"] == [
                {key: value for key, value in operation.items() if key != "created_at"} for operation in operations]
            execution = proposal.get("runtime_execution") or {}
            certificate = (execution.get("evaluation_certificate") or proposal.get("evaluation_certificate")
                or proposal.get("certificate_schedule_evidence", {}).get("certificate_record")
                or proposal.get("certificate_role_evidence", {}).get("certificate_record"))
            diff = execution.get("plan_diff") or proposal["plan_diff"]
            assert certificate["plan_id"] == proposal["child_plan_id"]
            assert certificate["plan_content_hash"] == proposal["child_plan_content_hash"]
            assert certificate["certificate_id"] == proposal["certificate_id"]
            assert _hash(certificate) == certificate["content_hash"] == proposal["certificate_content_hash"]
            assert certificate["comparison_eligibility"] == outcome
            assert _hash(diff) == proposal["diff_content_hash"] and diff["diff_id"] == proposal["diff_id"]
            assert diff["parent_plan_id"] == parent["active_plan_id"] and diff["child_plan_id"] == proposal["child_plan_id"]
            if execution:
                assert execution["evaluation_executed"] is True
                assert execution["child_plan_id"] == proposal["child_plan_id"]
                assert _hash(execution["child_plan"]) == proposal["child_plan_content_hash"]
                assert execution["evaluation_id"] == certificate["certificate_id"]
                candidates = {row["stop_id"] for row in execution["repair_request"]["candidate_pois"]}
                expected_candidates = {row["parameters"]["candidate_id"] if row["type"] == "replace_nearby" else row["target"]
                    for row in operations if row["type"] in {"replace_nearby", "add_candidate"}}
                assert candidates == expected_candidates, "Keep-only must not inject optional registered candidates"
            route = certificate["route_validation"]
            assert route["required_leg_count"] > 0 and route["road_validated_leg_count"] == route["required_leg_count"]
            assert route["missing_leg_count"] == route["fallback_leg_count"] == 0
            geography = proposal["geography_plan"]
            assert geography["plan_id"] == proposal["child_plan_id"] and geography["content_hash"] == proposal["child_plan_content_hash"]
            assert geography["coverage"]["complete"] is True
            child_stops = {row["properties"]["stop_id"]: row["properties"] for row in geography["stops"]["features"]}
            expected_markers = set(parent["rendered_marker_ids"])
            for operation in operations:
                target, parameters = operation["target"], operation["parameters"]
                if operation["type"] == "replace_nearby":
                    expected_markers.remove(target)
                    expected_markers.add(parameters["candidate_id"])
                elif operation["type"] == "add_candidate":
                    expected_markers.add(target)
                elif operation["type"] == "set_stop_duration":
                    assert child_stops[target]["visit_duration_minutes"] == parameters["duration"]["preferred_minutes"]
                elif operation["type"] == "set_stop_role":
                    assert child_stops[target]["itinerary_role"] == parameters["role"]
                elif operation["type"] == "set_stop_order":
                    assert child_stops[target]["stop_order"] == parameters["sequence_index"] + 1
                elif operation["type"] == "set_stop_time_window":
                    change = next(row for row in diff["time_window_changes"] if row["stop_id"] == target)
                    assert change["to_earliest_arrival"] == parameters["earliest_arrival"]
                    assert change["to_latest_departure"] == parameters["latest_departure"]
            assert set(child_stops) == expected_markers
            state = "PREVIEW_RENDERED" if outcome == "eligible" else "PREVIEW_INELIGIBLE"
            expect(page.locator(".map-stage")).to_have_attribute("data-preview-state", state, timeout=45_000)
            rendered = _map_ack(page)
            geometry_hash = _independent_geometry_hash(page, geography)
            if outcome == "eligible":
                assert certificate["failures"] == []
                assert rendered["active_plan_id"] == rendered["rendered_map_plan_id"] == proposal["child_plan_id"]
                assert rendered["active_plan_content_hash"] == proposal["child_plan_content_hash"]
                assert rendered["rendered_geometry_hash"] == geometry_hash
                assert set(rendered["rendered_marker_ids"]) == expected_markers
                expect(page.locator("#accept-repair")).to_be_enabled()
            else:
                assert [row["code"] for row in certificate["failures"]] == ["weather_infeasible"]
                assert _identity(rendered) == _identity(parent)
                expect(page.locator("#accept-repair")).to_be_disabled()
            if chat:
                result = page.locator(".copilot-preview-result")
                expect(result).to_have_attribute("data-preview-state", state)
                expect(result).to_contain_text("Preview ready" if outcome == "eligible" else "Preview not eligible")
                expect(page.locator("#copilot-lifecycle")).to_have_attribute("data-state", "preview_rendered" if outcome == "eligible" else "preview_ineligible")
                disposition = read_context(label + "_after_preview").conversation_memory["last_proposal"]["disposition"]
                assert disposition == ("previewed" if outcome == "eligible" else "staged")
                generated_finish('proposal_preview',generated_input,response=completed.value,map_ack=rendered)
            close_chat()
            surfaces = surface(rendered, geography if outcome == "eligible" else parent_geography)
            _capture_ready(page, case / "preview_result.png")
            advanced = page.locator("#repair-panel .evaluated-edit-evidence")
            expand(advanced)
            for identifier in (proposal["child_plan_id"], proposal["diff_id"], proposal["certificate_id"]):
                expect(advanced).to_contain_text(identifier)
            _write_json(case / "visible_surfaces.json", {**surfaces, "advanced_evidence": advanced.inner_text()})
            if outcome == "eligible":
                page.locator('[data-route="/app/compare"]').first.click()
                expect(page.locator('[data-compare-map-pane="option"] [data-compare-map-canvas]')).to_have_attribute(
                    "data-rendered-map-plan-id", proposal["child_plan_id"], timeout=30_000)
                _capture_ready(page, case / "compare.png")
                page.locator("#compare-evidence").click()
                view = page.locator('[data-evidence-plan-id]')
                expect(view).to_have_attribute("data-evidence-plan-id", proposal["child_plan_id"])
                expect(view).to_have_attribute("data-plan-content-hash", proposal["child_plan_content_hash"])
                expect(view).to_have_attribute("data-evaluation-id", proposal["certificate_id"])
                _write_json(case / "evidence_panel.json", {"scope": view.evaluate("e => ({...e.dataset})"), "text": view.inner_text()})
                page.locator('[data-route="/app/itinerary"]').first.click()
            if label == "chat_window_preview":
                # Arm only the existing reload. No API setup, extra request, or
                # synthetic response is introduced to manufacture the race.
                assert late_conversation_race is None and len(operations) == 1
                page.evaluate("sessionId => sessionStorage.setItem('__closureHoldConversationOnce',sessionId)", payload["session"]["session_id"])
            page.reload(wait_until="domcontentloaded")
            expect(page.locator(".map-stage")).to_have_attribute("data-preview-state", state, timeout=40_000)
            assert _map_ack(page) == rendered
            assert observed["session"]["accepted_plan_id"] == parent["active_plan_id"]
            _capture_ready(page, case / "preview_after_reload.png")
            if label == "chat_window_preview":
                page.wait_for_function("() => window.__closureLateConversation?.held_at !== null && typeof window.__releaseClosureConversation === 'function'", timeout=30_000)
                held = page.evaluate("window.__closureLateConversation")
                # Retain the exact received bytes even if decoding or an identity
                # comparison below fails; never discard a negative race witness.
                _write_json(case / "held_conversation_response.json", held)
                late_conversation_race = {"schema_version": "real-conversation-undo-race-v1", "context": plan_context,
                    "case": label, "scope": "actual_response_delivery_delay_only", "api_response_mocked": False,
                    "response_body_modified": False, "original_response_object_returned": True,
                    "response_json_observation_only": True, "additional_session_reload_count": 0,
                    "parent": parent, "operation": operations[0], "held_response": held,
                    "before_undo": {"map_ack": _map_ack(page), "persisted_session": _session_wire(read_session())},
                    "assertions_passed": False}
                _write_json(case / "late_conversation_undo_race.json", late_conversation_race)
                held_payload = json.loads(held["body_utf8"])
                assert held["error"] is None and held["status"] == 200 and held["released_at"] is None
                assert held["application_json_read_at"] is None and held["release_count"] == 0
                assert held_payload["session"] == late_conversation_race["before_undo"]["persisted_session"]
                assert held_payload["session"]["draft"] == operations and held_payload["session"]["proposal"] == proposal
            _write_json(case / "evaluation_summary.json", certificate)
            _write_json(case / "plan_diff.json", diff)
            _write_json(case / "map_render_ack.json", rendered)
            journeys.append({"case": label, "context": "original" if label == "chat_baseline_replace" else plan_context,
                "scope": "evaluated_preview_only",
                "interaction_surface": "copilot_chat" if chat else "direct_controls",
                "ui_controls": {"stage": '.copilot-proposal-actions button:has-text("Add to draft")' if chat
                    else ["#draft-keep" if row["type"] == "keep_stop" else "#draft-replace" for row in operations],
                    "preview": '.copilot-proposal-actions button:has-text("Preview this repair")' if chat else "#inline-preview"},
                "operations": operations, "sources": sorted({row["source"] for row in operations}),
                "request_id": proposal["compiled_request"]["source_request_id"], "parent": parent,
                "child_plan_id": proposal["child_plan_id"], "child_plan_content_hash": proposal["child_plan_content_hash"],
                "evaluation_id": proposal["certificate_id"], "comparison_eligibility": outcome, "geometry_hash": geometry_hash,
                "rendered": rendered, "assertions_passed": True})
            _write_json(evidence / "journey_assertions.json", journeys)
            return proposal, rendered

        def undo(operations, label):
            close_chat()
            selection_receipt = {"schema_version": "chat-undo-selection-v1", "case": phase,
                "before_session": _session_wire(read_session()), "undo_http": [],
                "assertions_passed": False}
            undo_selections.append(selection_receipt)
            _write_json(evidence / "undo_selection_receipts.json", undo_selections)
            race = late_conversation_race if label == "chat_window" else None
            if race:
                assert page.evaluate("window.__closureLateConversation.released_at") is None
                race["undo_clicked_at"] = page.evaluate("new Date().toISOString()")
            for operation in reversed(operations):
                with page.expect_response(lambda response: response.url.endswith("/draft/undo")) as completed:
                    page.locator("#inline-undo").click()
                assert completed.value.status == 200
                selection_receipt["undo_http"].append({"phase": phase, **_http_record(completed.value)})
                _write_json(evidence / "undo_selection_receipts.json", undo_selections)
                assert completed.value.json()["undone"] == operation
                if race:
                    race["undo_response_received_at"] = page.evaluate("new Date().toISOString()")
                expect(page.locator(".draft-bar strong")).to_have_text(
                    f'{len(completed.value.json()["session"]["draft"])} draft changes' if len(completed.value.json()["session"]["draft"]) != 1 else "1 draft change")
            assert completed.value.json()["session"]["draft"] == []
            assert _identity(_map_ack(page)) == _identity(parent)
            selected_session = completed.value.json()["session"]
            selected_stop = selected_session["selected_stop_id"]
            selected_segment = selected_session["selected_segment_id"]
            selection_receipt.update({"after_session": _session_wire(read_session()),
                "map_ack": _map_ack(page), "controls": page.evaluate("""() => ({
                  stop_ids:[...document.querySelectorAll('button[data-geo-stop][aria-pressed="true"]')].map(e=>e.dataset.geoStop),
                  segment_ids:[...document.querySelectorAll('button[data-geo-route-leg][aria-pressed="true"]')].map(e=>e.dataset.geoRouteLeg),
                  days:[...document.querySelectorAll('.day-card[aria-pressed="true"]')].map(e=>Number(e.dataset.day))})"""),
                "editor": None})
            _write_json(evidence / "undo_selection_receipts.json", undo_selections)
            stops = {row["properties"]["stop_id"]: row["properties"] for row in parent_geography["stops"]["features"]}
            segments = {row["properties"]["route_leg_id"] for row in parent_geography["validated_legs"]["features"]}
            assert selection_receipt["after_session"] == selected_session
            assert selected_stop is None or selected_stop in stops
            assert selected_segment is None or selected_segment in segments
            assert selection_receipt["controls"] == {"stop_ids": [selected_stop] if selected_stop else [],
                "segment_ids": [selected_segment] if selected_segment else [],
                "days": [selected_session["selected_day"]] if selected_session["selected_day"] else []}
            if selected_stop:
                before_requests = [row for row in requests if urlsplit(row["url"]).path.startswith("/api/")]
                page.locator("#map-edit-button").click()
                editor_button = page.locator("#inspect-edit-stop")
                expect(editor_button).to_be_enabled()
                page.evaluate("""() => {window.__closureEditorClick=null;
                  document.querySelector('#inspect-edit-stop').addEventListener('click',event=>{
                    window.__closureEditorClick={selector:'#inspect-edit-stop',trusted:event.isTrusted,
                      clicked_at:new Date().toISOString()};},{once:true});} """)
                editor_button.click()
                expect(page.locator("#map-edit-dialog")).to_be_visible()
                expand(page.locator("#map-edit-dialog details.legacy-edit-controls"))
                expected_target = (
                    f"Editing accepted-plan stop: {stops[selected_stop]['name']}. "
                    "Changes remain drafts until Preview repair finishes."
                )
                expect(page.locator("#edit-target")).to_have_text(expected_target)
                selection_receipt["editor"] = {"click": page.evaluate("window.__closureEditorClick"),
                    "dialog_visible": page.locator("#map-edit-dialog").is_visible(),
                    "target_text": page.locator("#edit-target").inner_text(),
                    "keep_disabled": page.locator("#draft-keep").is_disabled()}
                _write_json(evidence / "undo_selection_receipts.json", undo_selections)
                assert selection_receipt["editor"]["click"]["trusted"] is True
                assert selection_receipt["editor"]["target_text"] == expected_target
                assert selection_receipt["editor"]["keep_disabled"] is False
                page.locator("#close-map-edit").click()
                after_requests = [row for row in requests if urlsplit(row["url"]).path.startswith("/api/")]
                selection_receipt["editor"].update({"api_requests_before": before_requests,
                    "api_requests_after": after_requests})
                assert after_requests == before_requests
            else:
                expect(page.locator("#inspect-edit-stop")).to_have_count(0)
            selection_receipt["assertions_passed"] = True
            _write_json(evidence / "undo_selection_receipts.json", undo_selections)
            if race:
                case = evidence / "chat_window_preview"
                race["undo_http"] = {"phase": phase, **_http_record(completed.value)}
                assert completed.value.json()["session"]["revision"] > json.loads(race["held_response"]["body_utf8"])["session"]["revision"]

                def preserved_ui():
                    ack = _map_ack(page)
                    assert _identity(ack) == _identity(parent) and ack["preview_state"] == "PARENT_RENDERED"
                    expect(page.locator(".draft-bar strong")).to_have_text("0 draft changes")
                    expect(page.locator("#inline-preview")).to_be_disabled()
                    expect(page.locator("#accept-repair")).to_be_disabled()
                    expect(page.locator("#repair-panel .evaluated-edit-evidence")).to_have_count(0)
                    surfaces = surface(parent, parent_geography)
                    return {"observed_at": page.evaluate("new Date().toISOString()"), "map_ack": ack, "surfaces": surfaces,
                        "draft_label": page.locator(".draft-bar strong").inner_text(),
                        "preview_disabled": page.locator("#inline-preview").is_disabled(),
                        "accept_disabled": page.locator("#accept-repair").is_disabled(),
                        "evaluated_draft_evidence_count": page.locator("#repair-panel .evaluated-edit-evidence").count(),
                        "repair_panel": page.locator("#repair-panel").inner_text(),
                        "text_route_scope": page.locator(".text-route-summary").evaluate("e => ({...e.dataset})"),
                        "timeline_cards": page.locator(".day-card").evaluate_all("cards => cards.map(e => ({day:Number(e.dataset.day),stop_id:e.dataset.stop,name:e.querySelector('.day-copy strong').textContent}))")}

                race["after_undo_before_release"] = preserved_ui()
                race["persisted_before_release"] = _session_wire(read_session())
                assert race["persisted_before_release"] == completed.value.json()["session"]
                _write_json(case / "late_conversation_undo_race.json", race)
                before_api = [row for row in requests if urlsplit(row["url"]).path.startswith("/api/")]
                page.evaluate("window.__releaseClosureConversation()")
                page.wait_for_function("() => window.__closureLateConversation.delivery_settled_at !== null || window.__closureLateConversation.error !== null", timeout=30_000)
                race["held_response"] = page.evaluate("window.__closureLateConversation")
                _write_json(case / "late_conversation_undo_race.json", race)
                assert race["held_response"]["error"] is None
                race["after_release_before_rerender"] = preserved_ui()
                click_count = page.evaluate("window.__closureMapSelectClicks.length")
                page.locator("#map-select-button").click()
                expect(page.locator("#map-select-button")).to_have_attribute("aria-pressed", "true")
                clicks = page.evaluate("window.__closureMapSelectClicks")
                assert len(clicks) == click_count + 1 and clicks[-1]["trusted"] is True
                race["rerender_control"] = clicks[-1]
                race["raw_after_rerender"] = page.evaluate("""() => ({
                  draft_label:document.querySelector('.draft-bar strong')?.textContent,
                  map:{...document.querySelector('.map-stage')?.dataset},
                  metrics:{...document.querySelector('#metric-strip')?.dataset},
                  repair_panel:document.querySelector('#repair-panel')?.textContent})""")
                _write_json(case / "late_conversation_undo_race.json", race)
                race["after_rerender"] = preserved_ui()
                race["persisted_after_rerender"] = _session_wire(read_session())
                assert race["persisted_after_rerender"] == race["persisted_before_release"]
                after_api = [row for row in requests if urlsplit(row["url"]).path.startswith("/api/")]
                race["api_requests_before_release"] = before_api
                race["api_requests_after_rerender"] = after_api
                assert after_api == before_api, "Delivering an old response and a local rerender must not redispatch any API work"
                race["assertions_passed"] = True
                _write_json(case / "late_conversation_undo_race.json", race)
            _capture_ready(page, evidence / f"{label}_undone.png")
            if race:
                race["ready_screenshot"] = {"image": f"{label}_undone.png", "readiness": f"{label}_undone.readiness.json",
                    "image_sha256": hashlib.sha256((evidence / f"{label}_undone.png").read_bytes()).hexdigest()}
                _write_json(evidence / "chat_window_preview" / "late_conversation_undo_race.json", race)

        def negative(message, label, *, expected="unsupported"):
            before = len(mutations())
            map_before = _map_ack(page)
            payload = send(message, label)
            assert payload["turn"]["interpretation"]["state"] == expected
            intents = payload["turn"]["interpretation"]["intents"]
            assert [row["type"] for row in intents] == ["unsupported_request" if expected == "unsupported" else "request_clarification"]
            last = page.locator(".message-assistant").last
            assert last.get_by_role("button", name="Add to draft", exact=False).count() == 0
            assert last.get_by_role("button", name="Preview this repair", exact=False).count() == 0
            assert payload["session"]["draft"] == [] and len(mutations()) == before
            assert _identity(_map_ack(page)) == _identity(parent)
            negatives.append({"case": label, "context": plan_context, "input": message, "output": payload,
                "interaction_surface": "copilot_chat", "selector": ".message-assistant:last-child",
                "controls": last.locator("button").evaluate_all("buttons => buttons.map(button => ({label:button.textContent, disabled:button.disabled}))"),
                "mutation_request_delta": 0, "map_before": map_before, "map": _map_ack(page), "assertions_passed": True})
            _write_json(evidence / "negative_assertions.json", negatives)
            return payload

        # Case bodies below use only public controls; observer calls never stage
        # operations, preview, accept, reset, or invoke any provider adapter.
        with _preserve_browser_evidence(context, page, evidence, {"console_errors": errors, "request_failures": failures}):
            try:
                page.goto(f"{BASE_URL}/app/itinerary", wait_until="domcontentloaded")
                parent = _map_ack(page)
                initial = dict(parent)
                assert BIXBY in parent["rendered_marker_ids"], "Each context case starts before any accepted mutation"
                parent_geography = next(plan for plan in observed["workspace"]["geography"]["plans"] if plan["plan_id"] == parent["active_plan_id"])
                _capture_ready(page, evidence / "initial.png")
                open_chat()
                expect(page.locator("#copilot-provider-label")).to_have_text("Deterministic demo")
                # Wait on the actual UI's conversation restore, not an observer-created session.
                for _ in range(200):
                    if read_session().conversation_id:
                        break
                    page.wait_for_timeout(25)
                assert read_session().conversation_id

                if plan_context == "original":
                    phase = "keep_only_original"
                    open_stop(1)
                    operations = stage("keep_stop", GOLDEN_GATE, {}, page.locator("#draft-keep"))
                    proposal, _ = preview(phase, operations, "ineligible")
                    assert proposal["runtime_execution"]["repair_request"]["candidate_pois"] == []
                    assert proposal["runtime_execution"]["repair_request"]["parsed_intent"]["must_keep"] == [GOLDEN_GATE]
                    undo(operations, phase)

                    phase = "keep_and_invalid_target"
                    open_stop(1)
                    operations = stage("keep_stop", GOLDEN_GATE, {}, page.locator("#draft-keep"))
                    before = len(mutations())
                    page.locator("#registered-candidate").select_option(PFEIFFER)
                    blocked = page.locator("#draft-replace")
                    expect(blocked).to_be_visible()
                    expect(blocked).to_be_disabled()
                    assert not any(row["candidate_id"] == PFEIFFER and row["replaces_stop_id"] == GOLDEN_GATE
                        for row in observed["workspace"]["draft_capabilities"]["candidate_choices"])
                    assert len(mutations()) == before
                    negatives.append({"case": phase, "context": plan_context, "operation": "replace_nearby",
                        "input": {"target": GOLDEN_GATE, "parameters": {"candidate_id": PFEIFFER}},
                        "selector": "#draft-replace", "visible": True, "disabled": True,
                        "label": blocked.inner_text(), "title": blocked.get_attribute("title"),
                        "forced_interaction": False, "mutation_request_delta": 0, "assertions_passed": True})
                    page.locator("#close-map-edit").click()
                    _capture_ready(page, evidence / "invalid_target_blocked.png")
                    open_stop(5)
                    page.locator("#registered-candidate").select_option(PFEIFFER)
                    operations = stage("replace_nearby", BIXBY, {"candidate_id": PFEIFFER}, page.locator("#draft-replace"), count=2)
                    proposal, _ = preview("keep_and_valid_replacement", operations, "eligible")
                    compiled = proposal["runtime_execution"]["repair_request"]
                    assert [row["stop_id"] for row in compiled["candidate_pois"]] == [PFEIFFER]
                    assert compiled["confirmed_constraints"]["preserved_stop_ids"] == [GOLDEN_GATE]
                    assert compiled["confirmed_constraints"]["affected_days"] == [5]
                    undo(operations, "combined")
                else:
                    preference = send("I prefer less driving.", "memory_preference_original")
                    memory = read_context("preference_original").conversation_memory
                    assert list(memory["proposed_preferences"]) == [{"preference": "reduce_driving",
                        "source_turn_id": preference["turn"]["turn_id"], "disposition": "proposed_only"}]
                    pending = send("Replace Bixby Creek Bridge.", "memory_pending_original")
                    assert pending["turn"]["interpretation"]["state"] == "clarification_required"
                    assert read_context("pending_original").conversation_memory["pending_clarification"]["target_stop_id"] == BIXBY
                    ready = send("Pfeiffer Big Sur State Park.", "memory_resolved_original")
                    intent = ready["turn"]["interpretation"]["intents"][0]
                    assert (intent["type"], intent["target_stop_id"], intent["candidate_id"]) == ("replace_nearby", BIXBY, PFEIFFER)
                    memory = read_context("suggested_original").conversation_memory
                    assert memory["pending_clarification"] is None and memory["last_proposal"]["disposition"] == "suggested"
                    operations = stage_chat(ready)
                    proposal, rendered = preview("chat_baseline_replace", operations, "eligible", chat=True)
                    expect(page.locator("#accept-repair")).to_be_enabled()
                    with page.expect_response(lambda response: response.url.endswith("/accept"), timeout=60_000) as accepted:
                        page.locator("#accept-repair").click()
                        expect(page.get_by_role("dialog", name="Confirm trip decision", exact=True)).to_be_visible()
                        expect(page.locator("#acceptance-message")).to_have_text('Accept this independently evaluated repair as the durable trip plan?')
                        page.get_by_role("dialog", name="Confirm trip decision", exact=True).get_by_role("button", name='Accept repair', exact=True).click()
                    assert accepted.value.status == 200, accepted.value.text()
                    accepted_result = accepted.value.json()
                    assert accepted_result["session"]["accepted_plan_id"] == proposal["child_plan_id"]
                    assert accepted_result["session"]["draft"] == [] and accepted_result["session"]["proposal"] is None
                    decision = accepted_result["result"]["decision"]
                    assert decision["selected_plan_id"] == proposal["child_plan_id"]
                    assert decision["certificate_content_hash"] == proposal["certificate_content_hash"]
                    _write_json(evidence / "acceptance_response.json", accepted_result)
                    journeys[-1]["scope"] = "preview_then_explicit_acceptance_and_reload"
                    page.reload(wait_until="domcontentloaded")
                    expect(page.locator(".map-stage")).to_have_attribute("data-preview-state", "PARENT_RENDERED", timeout=40_000)
                    parent = _map_ack(page)
                    assert _identity(parent) == _identity(rendered)
                    parent_geography = next(plan for plan in observed["workspace"]["geography"]["plans"] if plan["plan_id"] == parent["active_plan_id"])
                    assert PFEIFFER in parent["rendered_marker_ids"] and BIXBY not in parent["rendered_marker_ids"]
                    _capture_ready(page, evidence / "accepted_child_reload.png")
                    open_chat()
                    current = read_context("accepted_transition")
                    assert current.conversation_memory["pending_clarification"] is None
                    assert current.conversation_memory["last_proposal"] is None
                    assert not current.conversation_memory["proposed_preferences"]
                    places = {row["place_id"]: row for row in current.known_places}
                    assert places[PFEIFFER]["route_status"] == "accepted_stop"
                    assert places.get(BIXBY, {}).get("route_status") != "accepted_stop"
                    assert current.capabilities["registered_repair_current"] is False
                    old_actions = page.locator(".copilot-proposal-action-row").filter(has_text="Bixby")
                    expect(old_actions).to_have_count(1)
                    expect(old_actions.get_by_role("button", name="Cannot add to draft", exact=False)).to_be_disabled()
                    assert old_actions.get_by_role("button", name="Preview this repair", exact=False).count() == 0

                    recalled = send("What preferences do you remember?", "memory_after_acceptance")
                    assert "none for this accepted plan" in recalled["turn"]["assistant_message"]
                    assert "reduce driving" not in recalled["turn"]["assistant_message"]
                    missing = negative("I want to go to Bixby Creek Bridge.", "removed_bixby_after_acceptance", expected="clarification_required")
                    assert "already in" not in missing["turn"]["assistant_message"].lower()
                    kept = send("Keep Pfeiffer Big Sur State Park.", "keep_pfeiffer_accepted")
                    intent = kept["turn"]["interpretation"]["intents"][0]
                    assert (intent["type"], intent["target_stop_id"]) == ("keep_stop", PFEIFFER)
                    operations = stage_chat(kept)
                    kept_proposal, _ = preview("keep_only_accepted", operations, "eligible", chat=True)
                    assert kept_proposal["runtime_execution"]["repair_request"]["candidate_pois"] == []
                    # Preserving an existing stop is not a required geography change.
                    undo(operations, "keep_only_accepted")
                    other = negative("Do the same for the other stop.", "pronoun_missing_target", expected="clarification_required")
                    assert "Which accepted stop" in other["turn"]["assistant_message"]
                    pending = read_context("pronoun_pending").conversation_memory["pending_clarification"]
                    assert pending["intent_type"] == "keep_stop" and pending["target_stop_id"] is None
                    answer = send("Stearns Wharf.", "pronoun_named_answer")
                    assert [(intent["type"], intent["target_stop_id"]) for intent in answer["turn"]["interpretation"]["intents"]] == [("keep_stop", STEARNS)]
                    assert answer["session"]["draft"] == []
                    review = negative("Review the current repair.", "accepted_review_truthful_unsupported")
                    assert "original plan, not the accepted child" in review["turn"]["assistant_message"]
                    assert "advisory" not in review
                    _capture_ready(page, evidence / "accepted_context_chat.png")

                # One new real turn owns the navigation controls. Selecting a different
                # accepted stop first makes Show on map's actual selection observable.
                open_stop(1)
                page.locator('#close-map-edit').click()
                navigation=send('Keep Stearns Wharf.','generated_navigation')
                assert navigation['turn']['interpretation']['intents'][0]['target_stop_id']==STEARNS
                actions=page.locator('.copilot-proposal-actions').last
                show=actions.get_by_role('button',name='Show on map',exact=True)
                expect(show).to_be_enabled()
                before=generated_before(show)
                with page.expect_response(lambda response:response.url.endswith('/selection')
                        and response.request.post_data_json.get('selected_stop_id')==STEARNS) as selected:
                    show.click()
                assert selected.value.status==200
                expect(page).to_have_url(BASE_URL+'/app/map')
                expect(page.locator('#map-context-title')).to_contain_text('Stearns Wharf')
                generated_finish('proposal_show_map',before,response=selected.value,map_ack=_map_ack(page))
                open_chat()
                stale=page.locator('.copilot-proposal-actions').last.get_by_role('button',name='Cannot add to draft',exact=False)
                expect(stale).to_be_visible()
                expect(stale).to_be_disabled()
                generated_finish('proposal_cannot_add_disabled',generated_before(stale),map_ack=_map_ack(page))
                for name,variant,route,selector in [('Compare','proposal_compare','/app/compare','[data-compare-workspace]'),
                        ('Review evidence','proposal_evidence','/app/evidence','[data-evidence-plan-id]')]:
                    open_chat()
                    control=page.locator('.copilot-proposal-actions').last.get_by_role('button',name=name,exact=True)
                    expect(control).to_be_enabled()
                    before=generated_before(control)
                    control.click()
                    expect(page).to_have_url(BASE_URL+route)
                    expect(page.locator(selector)).to_be_visible(timeout=45_000)
                    if variant=='proposal_compare':
                        page.wait_for_function("""()=>{const maps=[...document.querySelectorAll('[data-compare-map-canvas]')].filter(e=>e.getClientRects().length);
                          return maps.length>0&&maps.every(e=>e.dataset.mapReady==='true'&&Number(e.dataset.mapTileCoverage)>=.95);} """,timeout=45_000)
                    generated_finish(variant,before,result_selector=selector)
                page.locator('.nav-item[data-route="/app/itinerary"]').click()
                assert _identity(_map_ack(page))==_identity(parent)

                capabilities = {key: observed["workspace"][key] for key in ("draft_capabilities", "typed_edit_capabilities")}
                _write_json(evidence / "capability_snapshot.json", {"schema_version": "runtime-capability-snapshot-v1",
                    "context": plan_context, "session_id": observed["session"]["session_id"],
                    "plan_id": parent["active_plan_id"], "plan_content_hash": parent["active_plan_content_hash"], "capabilities": capabilities})
                families = [
                    ("duration", "Spend 60 minutes at Griffith Observatory.", "set_stop_duration", "griffith_observatory",
                        {"duration": {"mode": "exact", "preferred_minutes": 60, "minimum_minutes": 60, "maximum_minutes": 60}}),
                    ("order", "Set Sequoia National Park to position 2.", "set_stop_order", "sequoia_national_park", {"day": 4, "sequence_index": 1}),
                    ("role", "Set Stearns Wharf role to meal.", "set_stop_role", STEARNS, {"role": "meal"}),
                    ("window", "Set Stearns Wharf earliest arrival to 10:00.", "set_stop_time_window", STEARNS,
                        {"earliest_arrival": "10:00", "latest_departure": None}),
                ]
                for label, message, kind, target, parameters in families:
                    response = send(message, "chat_" + label)
                    assert response["turn"]["interpretation"]["schema_version"] == "copilot-interpretation-v2"
                    intent = response["turn"]["interpretation"]["intents"][0]
                    assert (intent["type"], intent["target_stop_id"], intent["parameters"]) == (kind, target, parameters)
                    operations = stage_chat(response)
                    preview("chat_" + label + "_preview", operations, "ineligible" if plan_context == "original" else "eligible", chat=True)
                    undo(operations, "chat_" + label)
                if plan_context == "accepted_child":
                    response = send("Add Point Lobos State Natural Reserve to day 5.", "chat_addition")
                    intent = response["turn"]["interpretation"]["intents"][0]
                    assert (intent["type"], intent["candidate_id"], intent["destination_day"]) == ("add_candidate", POINT_LOBOS, 5)
                    operations = stage_chat(response)
                    preview("chat_addition_preview", operations, "eligible", chat=True)
                    undo(operations, "chat_addition")
                    negative("Replace Pfeiffer Big Sur State Park with Point Lobos State Natural Reserve.", "accepted_replacement_not_registered")
                else:
                    negative("Add Pfeiffer Big Sur State Park to day 5.", "original_addition_missing_routes")
                for label, message in [
                    ("lock", "Lock Stearns Wharf."), ("flexible", "Mark Stearns Wharf flexible."),
                    ("move", "Move Stearns Wharf to day 7."), ("route_feedback", "Reduce driving."),
                    ("duration_mode", "Spend at least 60 minutes at Griffith Observatory."),
                    ("structural_role", "Set Stearns Wharf role to lodging."),
                ]:
                    negative(message, "unsupported_" + label)
                for label, message in [
                    ("compound_duration", "Spend 1 hour 30 minutes at Stearns Wharf."),
                    ("alternative_duration", "Set Stearns Wharf duration to 60 or 90 minutes."),
                ]:
                    negative(message, label, expected="clarification_required")
                close_chat()
                assert _identity(_map_ack(page)) == _identity(parent)
                surface(parent, parent_geography)
                _capture_ready(page, evidence / "final_preserved_parent.png")
                phase = "final_network_observation"
                asset_sources = page.evaluate("""() => {
                  const map = """ + CURRENT_MAP + """;
                  const style = map.getStyle();
                  return {container_id:map.getContainer().id,map_dataset:{...map.getContainer().dataset},
                    sprite:style.sprite??null,
                    vector_sources:Object.fromEntries(Object.entries(style.sources)
                      .filter(([,source])=>source.type==='vector'))};
                }""")
                assert asset_sources["container_id"] == "geographic-map"
                assert asset_sources["map_dataset"]["mapReady"] == "true"
                assert float(asset_sources["map_dataset"]["mapTileCoverage"]) >= .95
                assert asset_sources["map_dataset"]["renderedMapPlanId"] == parent["active_plan_id"]
                assert asset_sources["map_dataset"]["renderedPlanContentHash"] == parent["active_plan_content_hash"]
                assert asset_sources["map_dataset"]["renderedGeometryHash"] == parent["rendered_geometry_hash"]
                network_policy = map_cancellation_policy(map_configuration,
                    {"sources":asset_sources["vector_sources"],"sprite":asset_sources["sprite"]})
                _write_json(evidence / "map_asset_sources.json", {"schema_version":"real-map-asset-sources-v1",
                    "scope":"Read-only public getStyle() vector sources and actual ready container dataset; no map-disposal cause inferred",
                    "phase":phase,"context":plan_context,"map_configuration_http_file":map_configuration_http_file,
                    **asset_sources})
                classification = classify_browser_network(failures, network_policy)
                assert classification["blocking_failure_indexes"] == []
                assert errors == []
                assert len({row["file"] for row in http}) == len(http)
                assert not [row for row in http if row["status"] >= 400]
                assert sum(row["method"] == "POST" and row["url"].endswith("/accept") for row in requests) == (1 if plan_context == "accepted_child" else 0)
                _write_json(evidence / "browser_assertions.json", {"schema_version": "chat-runtime-closure-browser-v1",
                    "context": plan_context, "parent": parent, "journeys": journeys, "negatives": negatives,
                    "chat_family_samples_only": True, "all_natural_language_or_llm_variants_claimed": False,
                    "all_direct_conditional_modes_claimed": False, "assertions_passed": True})
                _write_json(evidence / "llm_io.json", {"provider_call_count": 0, "llm_input": None, "llm_output": None,
                    "reason": "Actual browser chat uses deterministic adapter; exact chat and context/journal input-output retained separately.",
                    "actual_openai_connectivity_claimed": False})
            finally:
                _write_json(evidence / "http_index.json", http)
                _write_json(evidence / "journey_assertions.json", journeys)
                _write_json(evidence / "negative_assertions.json", negatives)
                _write_json(evidence / "console_errors.json", errors)
                _write_json(evidence / "network_failures.json", failures)
                if network_policy is not None:
                    _write_json(evidence / "network_classification.json", {
                        "map_configuration_http_file":map_configuration_http_file,
                        "rendered_map_asset_sources_file":"map_asset_sources.json",
                        **classify_browser_network(failures,network_policy)})
        browser.close()
