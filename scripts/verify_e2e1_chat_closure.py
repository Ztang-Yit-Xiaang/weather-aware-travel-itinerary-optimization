"""Independent website-chat/keep-family closure evidence checks."""
from __future__ import annotations

import hashlib
import json
import xml.etree.ElementTree as ET
from collections import Counter

import verify_e2e1_vertical_slice as core
from verify_e2e1_self_managed import _http, _stream

SCENARIO = "chat_closure"
TEST = "test_keep_and_chat_families_preserve_runtime_truth"
FAMILIES = {
    "duration": ("Spend 60 minutes at Griffith Observatory.", "set_stop_duration", "griffith_observatory",
        {"duration": {"mode": "exact", "preferred_minutes": 60, "minimum_minutes": 60, "maximum_minutes": 60}}),
    "order": ("Set Sequoia National Park to position 2.", "set_stop_order", "sequoia_national_park", {"day": 4, "sequence_index": 1}),
    "role": ("Set Stearns Wharf role to meal.", "set_stop_role", "stearns_wharf", {"role": "meal"}),
    "window": ("Set Stearns Wharf earliest arrival to 10:00.", "set_stop_time_window", "stearns_wharf",
        {"earliest_arrival": "10:00", "latest_departure": None}),
}
NEGATIVE_MESSAGES = {
    "unsupported_lock": "Lock Stearns Wharf.", "unsupported_flexible": "Mark Stearns Wharf flexible.",
    "unsupported_move": "Move Stearns Wharf to day 7.", "unsupported_route_feedback": "Reduce driving.",
    "unsupported_duration_mode": "Spend at least 60 minutes at Griffith Observatory.",
    "unsupported_structural_role": "Set Stearns Wharf role to lodging.",
}
NUMERIC_CLARIFICATIONS = {
    "compound_duration": "Spend 1 hour 30 minutes at Stearns Wharf.",
    "alternative_duration": "Set Stearns Wharf duration to 60 or 90 minutes.",
}


def require(condition, reason):
    core._require(condition, "e2e1_chat_closure_" + reason)


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def identity(first, second):
    return all(first.get(key) == second.get(key) for key in ("active_plan_id", "active_plan_content_hash",
        "rendered_map_plan_id", "rendered_geometry_hash", "rendered_marker_ids"))


def numeric_clarification(row, parent):
    interpretation = row["output"]["turn"]["interpretation"]
    session = row["output"]["session"]
    require(row["input"] == NUMERIC_CLARIFICATIONS[row["case"]]
        and interpretation["state"] == "clarification_required"
        and len(interpretation["intents"]) == 1
        and interpretation["intents"][0]["type"] == "request_clarification"
        and bool(interpretation["intents"][0].get("clarification_question")), "numeric_clarification_missing")
    require(session["draft"] == [] and session["proposal"] is None
        and session["accepted_plan_id"] == parent["active_plan_id"]
        and identity(row["map_before"], parent) and identity(row["map"], parent), "numeric_clarification_mutated_parent")


GENERATED_VARIANTS={
    'proposal_show_map':('Show on map','.copilot-proposal-actions > button'),
    'proposal_compare':('Compare','.copilot-proposal-actions > button'),
    'proposal_evidence':('Review evidence','.copilot-proposal-actions > button'),
    'proposal_add':('Add to draft','.copilot-proposal-action-row button'),
    'proposal_preview':('Preview this repair','.copilot-proposal-action-row button'),
    'proposal_added_disabled':('Added to draft','.copilot-proposal-action-row button'),
    'proposal_cannot_add_disabled':('Cannot add to draft','.copilot-proposal-action-row button'),
}


def generated_controls(rows,observer,turns,records,journeys,context,parent):
    require(rows and observer['schema_version']=='generated-control-observer-v1','generated_receipts_missing')
    known={row['output']['turn']['client_message_id']:row for row in turns}
    counts=Counter()
    used_operations=[]
    used_previews=[]
    proofs=[]
    parent_maps={item['parent']['active_plan_id']:item['parent'] for item in journeys if 'parent' in item}
    parent_maps[parent['active_plan_id']]=parent
    for row in rows:
        variant=row['variant']
        require(variant in GENERATED_VARIANTS and row['assertions_passed'] is True,'generated_variant_invalid')
        before,output=row['input'],row['output']
        control=before['control']
        turn_row=known.get(control['client_message_id'])
        require(turn_row is not None,'generated_turn_missing')
        turn=turn_row['output']['turn']
        label,selector=GENERATED_VARIANTS[variant]
        disabled=variant.endswith('_disabled')
        require(control['label']==label and control['selector']==selector and control['visible'] is True
            and control['disabled'] is disabled and control['assistant_text']==turn['assistant_message']
            and turn['interpretation']['state']=='proposal_ready'
            and before['session']['accepted_plan_id']==output['session']['accepted_plan_id']==turn_row['context']['accepted_plan_id']
            and before['session']['session_id']==output['session']['session_id']==turn_row['context']['session_id'],
            'generated_owner_or_control_unbound')
        expected_context=context if output['session']['accepted_plan_id']==parent['active_plan_id'] else 'original'
        require(row['context']==expected_context and row['context'] in {'original',context},'generated_context_unbound')
        if variant not in {'proposal_preview','proposal_compare','proposal_evidence'}:
            require(identity(output['map'],parent_maps[before['session']['accepted_plan_id']]),
                'generated_parent_map_disconnected')
        document=observer['documents'][before['document_id']]
        events,start=row['events'],before['event_start']
        require(type(start) is int and 0<=start<=len(events) and events==document[:len(events)]
            and control['observed_at']<=output['observed_at'],'generated_observer_unbound')
        clicks=events[start:]
        if disabled:
            require(clicks==[],'generated_disabled_forced')
        else:
            require(len(clicks)==1 and clicks[0]['trusted'] is True
                and all(clicks[0][key]==control[key] for key in ('client_message_id','label','selector','intent_type','path','aria_label'))
                and control['observed_at']<=clicks[0]['at']<=output['observed_at'],'generated_click_unbound')
        if variant in {'proposal_add','proposal_show_map'}:
            require(before['session']['revision']==turn['context_revision'],'generated_enabled_action_stale')
        http=output['http']
        if http:
            core._verify_product_http_capture(http)
            require(any(record['phase']==row['phase'] and all(record.get(key)==value for key,value in http.items())
                for record in records),'generated_http_unbound')
            require(http['method']=='POST' and http['response_status_code']==200,'generated_http_failed')
            request,payload=(json.loads(http[key+'_body_utf8']) for key in ('request','response'))
        else:
            request,payload=None,None
        relevant=[item for item in row['requests'] if item['method']=='POST'
            and item['url'].endswith(('/draft/operations','/preview','/accept'))]
        if variant in {'proposal_add','proposal_preview'}:
            suffix='/draft/operations' if variant=='proposal_add' else '/preview'
            require(http and http['url'].endswith(suffix) and relevant==[{'method':'POST','url':http['url']}],
                'generated_mutation_universe_invalid')
            intent=turn['interpretation']['intents'][0]
            require(control['intent_type']==intent['type'],'generated_wrong_intent')
            if variant=='proposal_add':
                operation=payload['operation']
                require(interpretation_operation(intent)==(operation['type'],operation['target'],operation['parameters'])
                    and request['source']==operation['source']=='copilot'
                    and payload['session']['draft']==output['session']['draft']==[operation]
                    and before['session']['draft']==[], 'generated_add_disconnected')
                used_operations.append(operation['operation_id'])
                require(output['map']['active_plan_id']==output['map']['rendered_map_plan_id']==before['session']['accepted_plan_id'],
                    'generated_stage_rendered_child')
            else:
                proposal=payload['proposal']
                matching=[journey for journey in journeys if journey['case']==row['phase'] and journey['interaction_surface']=='copilot_chat']
                require(len(matching)==1 and matching[0]['rendered']==output['map']
                    and matching[0]['child_plan_id']==proposal['child_plan_id']
                    and matching[0]['operations']==before['session']['draft']==output['session']['draft'],
                    'generated_preview_disconnected')
                used_previews.append(row['phase'])
        else:
            require(relevant==[] and before['session']['draft']==output['session']['draft'], 'generated_navigation_mutated')
            if variant=='proposal_added_disabled':
                require(http is None and len(output['session']['draft'])==1
                    and interpretation_operation(turn['interpretation']['intents'][0])==tuple(
                        output['session']['draft'][0][key] for key in ('type','target','parameters')),
                    'generated_added_not_same_operation')
            elif variant=='proposal_cannot_add_disabled':
                require(http is None and output['session']['draft']==[] and before['session']['revision']>turn['context_revision'],
                    'generated_stale_not_proven')
            elif variant=='proposal_show_map':
                require(http and http['url'].endswith('/selection') and output['path']=='/app/map'
                    and request['selected_stop_id']==payload['session']['selected_stop_id']==output['session']['selected_stop_id']=='stearns_wharf'
                    and 'stearns_wharf' in turn['interpretation']['highlights']['stop_ids']
                    and before['session']['selected_stop_id']!='stearns_wharf'
                    and output['map']['active_plan_id']==output['map']['rendered_map_plan_id']==before['session']['accepted_plan_id'],
                    'generated_show_map_disconnected')
                if 'selected_day' in request:
                    require(request['selected_day'] in turn['interpretation']['highlights']['day_ids']
                        and request['selected_day']==payload['session']['selected_day']==output['session']['selected_day'],
                        'generated_day_highlight_disconnected')
            else:
                path,view_selector=('/app/compare','[data-compare-workspace]') if variant=='proposal_compare' else ('/app/evidence','[data-evidence-plan-id]')
                require(http is None and output['path']==path and output['result_selector']==view_selector
                    and output['view']['visible'] is True,'generated_navigation_result_missing')
        if row['context']==context:
            counts[variant]+=1
            proofs.append({'variant':variant,'selector':selector,'context':context,'phase':row['phase'],
                'status':'disabled_observed' if disabled else 'executed','client_message_id':control['client_message_id']})
    require(set(counts)==set(GENERATED_VARIANTS) and all(counts[key]==1 for key in
        ('proposal_show_map','proposal_compare','proposal_evidence','proposal_cannot_add_disabled')),
        'generated_variant_universe_incomplete')
    staged=[json.loads(record['response_body_utf8'])['operation']['operation_id'] for record in records
        if record['method']=='POST' and record['url'].endswith('/draft/operations')
        and json.loads(record['request_body_utf8'])['source']=='copilot']
    previews=[journey['case'] for journey in journeys if journey['interaction_surface']=='copilot_chat']
    require(Counter(used_operations)==Counter(staged) and Counter(used_previews)==Counter(previews)
        and counts['proposal_add']==counts['proposal_added_disabled']==counts['proposal_preview'],
        'generated_exact_action_inventory_incomplete')
    return proofs


def context_digest(context, history):
    return hashlib.sha256(json.dumps({"context": context, "history": history}, sort_keys=True,
        separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def chat_turn(row, http, plan_identities):
    request, events = _stream(http)
    require(request == row["input"] and events == row["events"] and events[-1]["payload"] == row["output"], "exact_sse_unbound")
    context, history, output = row["context"], row["bound_history"], row["output"]
    turn, dispatch = output["turn"], row["dispatch_record"]
    key = (context["accepted_plan_id"], context["accepted_plan_content_hash"])
    require(key in plan_identities and set(context["allowed_stop_ids"]) == set(plan_identities[key]["rendered_marker_ids"])
        and context["conversation_memory"]["plan_id"] == key[0]
        and context["conversation_memory"]["plan_content_hash"] == key[1], "provider_context_not_current_plan")
    binding = {"plan_id": key[0], "plan_content_hash": key[1], **{name: context[name] for name in (
        "selected_day", "selected_stop_id", "selected_candidate_id", "selected_segment_id", "selected_alternative_id")}}
    require(turn["provider"] == "deterministic" and turn["user_message"] == request["message"]
        and turn["context_revision"] == context["session_revision"]
        and dispatch["context_binding"] == binding and dispatch["context_sha256"] == row["context_sha256"] == context_digest(context, history)
        and dispatch["client_message_id"] == request["client_message_id"]
        and dispatch["attempt_budget"] == dispatch["attempt_count"] == 1 and dispatch["outcome"] == "completed"
        and dispatch["turn"] == turn, "durable_dispatch_or_context_hash_mismatch")
    require(output["session"]["accepted_plan_id"] == key[0] and output["session"]["draft"] == [], "chat_send_mutated_draft")


def interpretation_operation(intent):
    kind = intent["type"]
    if kind == "replace_nearby":
        return kind, intent["target_stop_id"], {"candidate_id": intent["candidate_id"]}
    if kind == "add_candidate":
        return kind, intent["candidate_id"], {"day": intent["destination_day"]}
    return kind, intent["target_stop_id"], intent.get("parameters") or {}


def journey_lifecycle_http(row, payload, records):
    """Bind reload and Undo to real HTTP, not the browser's assertion booleans."""
    phase = [http for http in records if http["phase"] == row["case"]]
    session = payload["session"]
    reloads = [json.loads(http["response_body_utf8"])["session"] for http in phase
        if http["method"] == "GET" and http["url"].endswith("/api/sessions/" + session["session_id"])]
    accepts = [http for http in phase if http["method"] == "POST" and http["url"].endswith("/accept")]
    undos = [json.loads(http["response_body_utf8"]) for http in phase
        if http["method"] == "POST" and http["url"].endswith("/draft/undo")]
    baseline = row["case"] == "chat_baseline_replace"
    require(row["assertions_passed"] is True and row["scope"] == (
        "preview_then_explicit_acceptance_and_reload" if baseline else "evaluated_preview_only"), "journey_scope_mismatch")
    require(len(reloads) == 1 + int(baseline) and reloads[0]["accepted_plan_id"] == session["accepted_plan_id"]
        and reloads[0]["draft"] == row["operations"] and reloads[0]["proposal"] == payload["proposal"], "reload_not_bound_to_preview")
    if baseline:
        require(len(accepts) == 1 and not undos and reloads[1]["accepted_plan_id"] == row["child_plan_id"]
            and reloads[1]["draft"] == [] and reloads[1]["proposal"] is None, "accepted_reload_not_bound")
    else:
        require(not accepts and len(undos) == len(row["operations"]), "undo_count_or_unexpected_acceptance")
        remaining = list(row["operations"])
        for undone in undos:
            require(undone["undone"] == remaining.pop() and undone["session"]["draft"] == remaining
                and undone["session"]["proposal"] is None
                and undone["session"]["accepted_plan_id"] == session["accepted_plan_id"], "undo_changed_parent_or_wrong_operation")


def undo_selection_receipt(row, receipt, records, geography):
    """Recompute accepted selection from actual pre-Undo and mutation wire bodies."""
    phase = [record for record in records if record["phase"] == row["case"]]
    undos = [record for record in phase if record["method"] == "POST" and record["url"].endswith("/draft/undo")]
    before = receipt["before_session"]
    reads = [json.loads(record["response_body_utf8"])["session"] for record in phase
        if record["method"] == "GET" and record["url"].endswith("/api/sessions/" + before["session_id"])]
    require(receipt["schema_version"] == "chat-undo-selection-v1" and receipt["case"] == row["case"]
        and receipt["assertions_passed"] is True and reads == [before]
        and undos and receipt["undo_http"] == undos, "undo_selection_http_unbound")
    parent = row["parent"]
    require(geography["plan_id"] == before["accepted_plan_id"] == parent["active_plan_id"]
        and geography["content_hash"] == parent["active_plan_content_hash"]
        and core._completed_race_geometry_hash(geography) == parent["rendered_geometry_hash"]
        and identity(receipt["map_ack"], parent) and receipt["map_ack"]["preview_state"] == "PARENT_RENDERED",
        "undo_selection_parent_unbound")
    stops = {feature["properties"]["stop_id"]: feature["properties"] for feature in geography["stops"]["features"]}
    segments = {feature["properties"]["route_leg_id"] for feature in geography["validated_legs"]["features"]}
    for record in undos:
        request, result = json.loads(record["request_body_utf8"]), json.loads(record["response_body_utf8"])
        require(record["response_status_code"] == 200
            and record["request_body_sha256"] == hashlib.sha256(record["request_body_utf8"].encode()).hexdigest()
            and record["response_body_sha256"] == hashlib.sha256(record["response_body_utf8"].encode()).hexdigest()
            and request == {"expected_revision": before["revision"]}, "undo_selection_revision_unbound")
        expected = {**before, "revision": before["revision"] + 1, "draft": before["draft"][:-1], "proposal": None}
        proposal = before.get("proposal") or {}
        if proposal.get("state") == "eligible":
            selected_stop = before["selected_stop_id"] if before["selected_stop_id"] in stops else None
            expected["selected_stop_id"] = selected_stop
            expected["selected_segment_id"] = before["selected_segment_id"] if before["selected_segment_id"] in segments else None
            expected["selected_day"] = stops[selected_stop]["day"] if selected_stop else before["selected_day"]
        require(result["undone"] == before["draft"][-1] and result["session"] == expected,
            "undo_selection_not_projected_from_accepted_plan")
        before = expected
    require(receipt["after_session"] == before, "undo_selection_persistence_mismatch")
    stop, segment, day = before["selected_stop_id"], before["selected_segment_id"], before["selected_day"]
    require(receipt["controls"] == {"stop_ids": [stop] if stop else [], "segment_ids": [segment] if segment else [],
        "days": [day] if day else []}, "undo_selection_visible_controls_mismatch")
    editor = receipt["editor"]
    if not stop:
        require(editor is None, "undo_selection_orphan_editor")
        return
    require(editor["click"]["selector"] == "#inspect-edit-stop" and editor["click"]["trusted"] is True
        and editor["dialog_visible"] is True and editor["keep_disabled"] is False
        and editor["target_text"] == f"Editing accepted-plan stop: {stops[stop]['name']}. Changes remain drafts until Preview repair finishes.",
        "undo_selection_editor_stale")
    core._isolation_timestamp(editor["click"]["clicked_at"])
    end = max(index for index, record in enumerate(records) if record["phase"] == row["case"])
    require(editor["api_requests_before"] == editor["api_requests_after"]
        and Counter((item["method"], item["url"]) for item in editor["api_requests_before"])
        == Counter((item["method"], item["url"]) for item in records[:end + 1]), "undo_selection_editor_extra_api")


def late_conversation_undo_race(row, payload, race, records, geography, ready):
    """Independently bind an actual delayed read to a newer real Undo and UI."""
    parent, held = row["parent"], race["held_response"]
    require(row["case"] == race["case"] == "chat_window_preview" and len(row["operations"]) == 1
        and race["schema_version"] == "real-conversation-undo-race-v1" and race["context"] == row["context"]
        and race["scope"] == "actual_response_delivery_delay_only" and race["assertions_passed"] is True
        and race["api_response_mocked"] is False and race["response_body_modified"] is False
        and race["original_response_object_returned"] is True and race["response_json_observation_only"] is True
        and type(race["additional_session_reload_count"]) is int and race["additional_session_reload_count"] == 0
        and race["parent"] == parent and race["operation"] == row["operations"][0], "late_read_scope_invalid")
    session_id = payload["session"]["session_id"]
    window = [http for http in records if http["phase"] == row["case"]]
    reload_indexes = [index for index, http in enumerate(window) if http["method"] == "GET"
        and http["url"].endswith("/api/sessions/" + session_id)]
    require(len(reload_indexes) == 1, "late_read_reload_count_invalid")
    reads = [http for http in window[reload_indexes[0] + 1:] if http["method"] == "GET"
        and http["url"].endswith("/api/sessions/" + session_id + "/conversation")
        and http["response_body_utf8"] == held["body_utf8"]]
    require(len(reads) == 1 and held["url"] == reads[0]["url"] and held["method"] == "GET"
        and held["status"] == reads[0]["response_status_code"] == 200
        and held["request_body_utf8"] == reads[0]["request_body_utf8"] == ""
        and held["body_sha256"] == reads[0]["response_body_sha256"] == hashlib.sha256(held["body_utf8"].encode()).hexdigest()
        and held["error"] is None and type(held["release_count"]) is int and held["release_count"] == 1,
        "late_read_exact_http_unbound")
    old = json.loads(held["body_utf8"])["session"]
    undo = race["undo_http"]
    undos = [http for http in window if http["method"] == "POST" and http["url"].endswith("/api/sessions/" + session_id + "/draft/undo")]
    require(undos == [undo] and undo["response_status_code"] == 200
        and window.index(reads[0]) < window.index(undo)
        and undo["request_body_sha256"] == hashlib.sha256(undo["request_body_utf8"].encode()).hexdigest()
        and undo["response_body_sha256"] == hashlib.sha256(undo["response_body_utf8"].encode()).hexdigest(), "late_read_undo_http_unbound")
    request, undone = json.loads(undo["request_body_utf8"]), json.loads(undo["response_body_utf8"])
    current = undone["session"]
    require(old == race["before_undo"]["persisted_session"]
        and old["session_id"] == current["session_id"] == session_id
        and old["accepted_plan_id"] == current["accepted_plan_id"] == parent["active_plan_id"]
        and old["draft"] == row["operations"] and old["proposal"] == payload["proposal"]
        and type(old["revision"]) is int and type(current["revision"]) is int
        and request == {"expected_revision": old["revision"]} and current["revision"] == old["revision"] + 1
        and current["draft"] == [] and current["proposal"] is None and undone["undone"] == row["operations"][0]
        and race["persisted_before_release"] == race["persisted_after_rerender"] == current,
        "late_read_revision_or_persistence_mismatch")
    require(identity(race["before_undo"]["map_ack"], row["rendered"])
        and race["before_undo"]["map_ack"]["preview_state"] == row["rendered"]["preview_state"], "late_read_preview_checkpoint_unbound")
    require(geography["plan_id"] == parent["active_plan_id"] and geography["content_hash"] == parent["active_plan_content_hash"]
        and core._completed_race_geometry_hash(geography) == parent["rendered_geometry_hash"], "late_read_parent_geography_unbound")
    stops = [feature["properties"] for feature in geography["stops"]["features"]]
    expected_cards = []
    for day in range(1, 8):
        ordered = sorted((stop for stop in stops if int(stop["day"]) == day), key=lambda stop: stop["stop_order"])
        expected_cards.append({"day": day, "stop_id": ordered[0]["stop_id"] if ordered else "",
            "name": " · ".join(stop["name"] for stop in ordered) or "No recorded stop"})
    for label in ("after_undo_before_release", "after_release_before_rerender", "after_rerender"):
        surface = race[label]
        require(identity(surface["map_ack"], parent) and surface["map_ack"]["preview_state"] == "PARENT_RENDERED"
            and surface["draft_label"] == "0 draft changes" and surface["preview_disabled"] is True
            and surface["accept_disabled"] is True and type(surface["evaluated_draft_evidence_count"]) is int
            and surface["evaluated_draft_evidence_count"] == 0 and surface["timeline_cards"] == expected_cards
            and surface["surfaces"]["metrics"]["planId"] == surface["text_route_scope"]["planId"] == parent["active_plan_id"]
            and surface["surfaces"]["metrics"]["planContentHash"] == surface["text_route_scope"]["planContentHash"] == parent["active_plan_content_hash"]
            and all(identifier not in surface["repair_panel"] for identifier in (row["child_plan_id"], row["evaluation_id"], payload["proposal"]["diff_id"])),
            "late_read_visible_state_rolled_back")
    control = race["rerender_control"]
    require(control["selector"] == "#map-select-button" and control["trusted"] is True, "late_read_rerender_not_real_control")
    raw = race["raw_after_rerender"]
    require(raw["draft_label"] == "0 draft changes" and raw["map"]["activePlanId"] == parent["active_plan_id"]
        and raw["map"]["activePlanContentHash"] == parent["active_plan_content_hash"]
        and raw["metrics"]["planId"] == parent["active_plan_id"]
        and raw["metrics"]["planContentHash"] == parent["active_plan_content_hash"], "late_read_raw_rerender_rolled_back")
    before, after = race["api_requests_before_release"], race["api_requests_after_rerender"]
    end = max(index for index, http in enumerate(records) if http["phase"] == row["case"])
    require(before == after and Counter((item["method"], item["url"]) for item in before)
        == Counter((item["method"], item["url"]) for item in records[:end + 1]), "late_read_extra_api_work")
    stamps = [held["requested_at"], held["body_completed_at"], held["held_at"], race["undo_clicked_at"],
        race["undo_response_received_at"], race["after_undo_before_release"]["observed_at"], held["released_at"],
        held["application_json_read_at"], held["delivery_settled_at"], race["after_release_before_rerender"]["observed_at"],
        control["clicked_at"], race["after_rerender"]["observed_at"], ready["screenshot_started_at"]]
    stamps = [core._isolation_timestamp(value) for value in stamps]
    require(stamps == sorted(stamps) and ready["rendered_plan_id"] == parent["active_plan_id"]
        and ready["rendered_geometry_hash"] == parent["rendered_geometry_hash"], "late_read_delivery_or_capture_order_invalid")


def canonical_journey(row, payload, certificate, diff):
    proposal, parent, operations = payload["proposal"], row["parent"], row["operations"]
    outcome = row["comparison_eligibility"]
    require(outcome in {"eligible", "ineligible"} and proposal["state"] == proposal["eligibility"] == outcome
        and proposal["parent_plan_id"] == parent["active_plan_id"] and proposal["parent_plan_content_hash"] == parent["active_plan_content_hash"]
        and payload["session"]["accepted_plan_id"] == parent["active_plan_id"] and payload["session"]["draft"] == operations
        and proposal["compiled_request"]["operations"] == [{key: value for key, value in operation.items() if key != "created_at"} for operation in operations]
        and proposal["compiled_request"]["source_request_id"] == row["request_id"], "compiled_operation_identity_mismatch")
    require(certificate["plan_id"] == diff["child_plan_id"] == proposal["child_plan_id"] == row["child_plan_id"]
        and certificate["plan_content_hash"] == proposal["child_plan_content_hash"] == row["child_plan_content_hash"]
        and certificate["certificate_id"] == proposal["certificate_id"] == row["evaluation_id"]
        and certificate["comparison_eligibility"] == outcome
        and core._artifact_content_hash(certificate) == certificate["content_hash"] == proposal["certificate_content_hash"]
        and core._artifact_content_hash(diff) == proposal["diff_content_hash"] and diff["diff_id"] == proposal["diff_id"]
        and diff["parent_plan_id"] == parent["active_plan_id"], "evaluation_diff_mismatch")
    require([failure["code"] for failure in certificate["failures"]] == ([] if outcome == "eligible" else ["weather_infeasible"]),
        "unrelated_failure_hidden_as_ineligible")
    route = certificate["route_validation"]
    geo = proposal["geography_plan"]
    require(route["required_leg_count"] > 0 and route["road_validated_leg_count"] == route["required_leg_count"]
        and route["fallback_leg_count"] == route["missing_leg_count"] == 0
        and geo["plan_id"] == proposal["child_plan_id"] and geo["content_hash"] == proposal["child_plan_content_hash"]
        and geo["coverage"]["complete"] is True and row["geometry_hash"] == core._completed_race_geometry_hash(geo), "missing_or_foreign_geometry")
    execution = proposal.get("runtime_execution")
    if execution:
        expected_candidates = {operation["parameters"]["candidate_id"] if operation["type"] == "replace_nearby" else operation["target"]
            for operation in operations if operation["type"] in {"replace_nearby", "add_candidate"}}
        require(execution["evaluation_executed"] is True and execution["evaluation_certificate"] == certificate
            and execution["plan_diff"] == diff and execution["child_plan_id"] == proposal["child_plan_id"]
            and core._artifact_content_hash(execution["child_plan"]) == proposal["child_plan_content_hash"]
            and {poi["stop_id"] for poi in execution["repair_request"]["candidate_pois"]} == expected_candidates,
            "unrequested_candidates_or_unevaluated_child")
    expected_markers = set(parent["rendered_marker_ids"])
    stops = {feature["properties"]["stop_id"]: feature["properties"] for feature in geo["stops"]["features"]}
    for operation in operations:
        target, parameters, kind = operation["target"], operation["parameters"], operation["type"]
        if kind == "replace_nearby":
            expected_markers.remove(target)
            expected_markers.add(parameters["candidate_id"])
        elif kind == "add_candidate":
            expected_markers.add(target)
        elif kind == "set_stop_duration":
            require(stops[target]["visit_duration_minutes"] == parameters["duration"]["preferred_minutes"], "duration_not_applied")
        elif kind == "set_stop_role":
            require(stops[target]["itinerary_role"] == parameters["role"], "role_not_applied")
        elif kind == "set_stop_order":
            require(stops[target]["stop_order"] == parameters["sequence_index"] + 1, "order_not_applied")
        elif kind == "set_stop_time_window":
            changes = [change for change in diff["time_window_changes"] if change["stop_id"] == target]
            require(len(changes) == 1 and changes[0]["to_earliest_arrival"] == parameters["earliest_arrival"]
                and changes[0]["to_latest_departure"] == parameters["latest_departure"], "time_window_not_applied")
        else:
            require(kind == "keep_stop" and target in expected_markers, "unsupported_operation_masquerading_as_keep")
    require(set(stops) == expected_markers, "marker_semantics_mismatch")
    if outcome == "eligible":
        core._verify_plan_ack(row["rendered"], proposal)
    else:
        require(identity(row["rendered"], parent), "ineligible_child_rendered")
    require(row["rendered"]["preview_state"] == ("PREVIEW_RENDERED" if outcome == "eligible" else "PREVIEW_INELIGIBLE"), "preview_state_mismatch")


def _verify(root, base_url):
    junit = ET.parse(root / "pytest_junit.xml").getroot()
    cases = list(junit.iter("testcase"))
    require([case.get("name") for case in cases] == [TEST + "[original]", TEST + "[accepted_child]"]
        and all(case.get("classname") == "tests.product_app.browser.test_chat_runtime_closure_live" for case in cases)
        and not any(node.tag in {"failure", "error", "skipped"} for node in junit.iter()), "browser_test_universe_invalid")
    for context in ("original", "accepted_child"):
        base = root / SCENARIO / context
        assertions = core._load(base / "browser_assertions.json")
        require(core._load(base / "llm_io.json")["actual_openai_connectivity_claimed"] is False, "live_connectivity_falsely_claimed")
        require(assertions["schema_version"] == "chat-runtime-closure-browser-v1" and assertions["context"] == context
            and assertions["assertions_passed"] is True and assertions["chat_family_samples_only"] is True
            and assertions["all_natural_language_or_llm_variants_claimed"] is False
            and assertions["all_direct_conditional_modes_claimed"] is False, "false_completeness_claim")
        journeys, negatives = load(base / "journey_assertions.json"), load(base / "negative_assertions.json")
        require(journeys == assertions["journeys"] and negatives == assertions["negatives"], "assertion_projection_mismatch")
        expected = (["keep_only_original", "keep_and_valid_replacement"] if context == "original" else ["chat_baseline_replace", "keep_only_accepted"])
        expected += ["chat_" + family + "_preview" for family in FAMILIES]
        if context == "accepted_child":
            expected.append("chat_addition_preview")
        require([row["case"] for row in journeys] == expected
            and {path.name for path in base.iterdir() if path.is_dir()} == set(expected), "journey_universe_incomplete")
        records = _http(base, base_url)
        selections = load(base / "undo_selection_receipts.json")
        require([receipt["case"] for receipt in selections] == [row["case"] for row in journeys if row["case"] != "chat_baseline_replace"],
            "undo_selection_inventory_incomplete")
        posts = [http for http in records if http["method"] == "POST"]
        chats = [http for http in posts if http["url"].endswith("/copilot/messages")]
        turns = load(base / "exact_chat_io.json")
        require(len(chats) == len(turns) == (14 if context == "original" else 24)
            and len({row["label"] for row in turns}) == len(turns), "chat_inventory_incomplete")
        identities = {(row["parent"]["active_plan_id"], row["parent"]["active_plan_content_hash"]): row["parent"] for row in journeys}
        identities[(assertions["parent"]["active_plan_id"], assertions["parent"]["active_plan_content_hash"])] = assertions["parent"]
        for row in turns:
            matches = [http for http in chats if http["phase"] == row["label"]]
            require(len(matches) == 1, "chat_phase_unbound")
            chat_turn(row, matches[0], identities)
            before = core._load(base / ("context_" + row["label"] + "_before.json"))
            require(before == row["context"], "context_projection_mismatch")
        by_label = {row["label"]: row for row in turns}
        require(by_label['generated_navigation']['output']['turn']['user_message']=='Keep Stearns Wharf.',
            'generated_navigation_turn_missing')
        generated_controls(load(base/'generated_control_actions.json'),core._load(base/'generated_control_observer.json'),
            turns,records,journeys,context,assertions['parent'])
        for family, (message, kind, target, parameters) in FAMILIES.items():
            row = by_label["chat_" + family]
            require(row["output"]["turn"]["user_message"] == message
                and row["output"]["turn"]["interpretation"]["schema_version"] == "copilot-interpretation-v2"
                and len(row["output"]["turn"]["interpretation"]["intents"]) == 1
                and interpretation_operation(row["output"]["turn"]["interpretation"]["intents"][0]) == (kind, target, parameters), "typed_chat_mapping_mismatch")
        for row in journeys:
            case = base / row["case"]
            payload = core._load(case / "preview_response.json")
            matching = [http for http in posts if http["phase"] == row["case"] and http["url"].endswith("/preview")]
            require(len(matching) == 1 and json.loads(matching[0]["response_body_utf8"]) == payload, "preview_http_unbound")
            journey_lifecycle_http(row, payload, records)
            canonical_journey(row, payload, core._load(case / "evaluation_summary.json"), core._load(case / "plan_diff.json"))
            if row["case"] != "chat_baseline_replace":
                parent_views = [json.loads(http["response_body_utf8"]) for http in records if http["url"].endswith(
                    ("/api/sessions", "/api/sessions/" + payload["session"]["session_id"]))]
                parent_geography = next(plan for view in parent_views if view["session"]["accepted_plan_id"] == row["parent"]["active_plan_id"]
                    and view["session"]["proposal"] is None for plan in view["workspace"]["geography"]["plans"]
                    if plan["plan_id"] == row["parent"]["active_plan_id"] and plan["content_hash"] == row["parent"]["active_plan_content_hash"])
                undo_selection_receipt(row, next(receipt for receipt in selections if receipt["case"] == row["case"]), records, parent_geography)
            if row["case"] == "chat_window_preview":
                race = core._load(case / "late_conversation_undo_race.json")
                capture = race["ready_screenshot"]
                require(capture["image"] == "chat_window_undone.png" and capture["readiness"] == "chat_window_undone.readiness.json"
                    and capture["image_sha256"] == core._sha256(base / capture["image"]), "late_read_screenshot_unbound")
                parent_views = [json.loads(http["response_body_utf8"]) for http in records if http["url"].endswith(
                    ("/api/sessions", "/api/sessions/" + payload["session"]["session_id"]))]
                parent_geography = next(plan for view in parent_views if view["session"]["accepted_plan_id"] == row["parent"]["active_plan_id"]
                    and view["session"]["proposal"] is None for plan in view["workspace"]["geography"]["plans"]
                    if plan["plan_id"] == row["parent"]["active_plan_id"] and plan["content_hash"] == row["parent"]["active_plan_content_hash"])
                late_conversation_undo_race(row, payload, race, records, parent_geography, core._load(base / capture["readiness"]))
            require(core._load(case / "map_render_ack.json") == row["rendered"], "map_ack_projection_mismatch")
            for operation in row["operations"]:
                staged = [http for http in posts if http["url"].endswith("/draft/operations")
                    and json.loads(http["response_body_utf8"])["operation"] == operation]
                require(len(staged) == 1 and all(json.loads(staged[0]["request_body_utf8"])[key] == operation[key]
                    for key in ("type", "target", "parameters", "source")), "stage_http_unbound")
                if operation["source"] == "copilot":
                    turn = by_label[staged[0]["phase"]]["output"]["turn"]
                    require(turn["interpretation"]["state"] == "proposal_ready"
                        and interpretation_operation(turn["interpretation"]["intents"][0]) == (operation["type"], operation["target"], operation["parameters"]),
                        "chat_to_staged_operation_disconnected")
            require(row["sources"] == sorted({operation["source"] for operation in row["operations"]})
                and row["interaction_surface"] == ("copilot_chat" if row["sources"] == ["copilot"] else "direct_controls"), "surface_source_mismatch")
            labels = ["staged_parent", "preview_result", "preview_after_reload"]
            if row["comparison_eligibility"] == "eligible":
                panel = core._load(case / "evidence_panel.json")
                require(panel["scope"]["evidencePlanId"] == row["child_plan_id"]
                    and panel["scope"]["planContentHash"] == row["child_plan_content_hash"]
                    and panel["scope"]["evaluationId"] == row["evaluation_id"], "evidence_scope_mismatch")
                labels.append("compare")
            core._verify_journey_captures(case, labels, {row["parent"]["active_plan_id"], row["child_plan_id"]})
        expected_negative = set(NEGATIVE_MESSAGES) | set(NUMERIC_CLARIFICATIONS) | ({"keep_and_invalid_target", "original_addition_missing_routes"} if context == "original" else {
            "removed_bixby_after_acceptance", "pronoun_missing_target", "accepted_review_truthful_unsupported", "accepted_replacement_not_registered"})
        require(len(negatives) == len(expected_negative) and {row["case"] for row in negatives} == expected_negative, "negative_universe_incomplete")
        for row in negatives:
            require(row["assertions_passed"] is True and row["context"] == context and row["mutation_request_delta"] == 0, "negative_mutation_claim_invalid")
            if row["case"] == "keep_and_invalid_target":
                require(row["visible"] is True and row["disabled"] is True and row["forced_interaction"] is False, "invalid_target_not_disabled")
                continue
            require(not any(http["phase"] == row["case"] and http["url"].endswith(("/draft/operations", "/preview", "/accept", "/draft/undo"))
                for http in posts), "negative_http_mutation_hidden")
            turn = by_label[row["case"]]
            require(row["output"] == turn["output"] and row["input"] == turn["output"]["turn"]["user_message"]
                and not any("Add to draft" in button["label"] or "Preview this repair" in button["label"] for button in row["controls"]), "negative_has_executable_action")
            interpretation = row["output"]["turn"]["interpretation"]
            state = "clarification_required" if row["case"] in {"removed_bixby_after_acceptance", "pronoun_missing_target", *NUMERIC_CLARIFICATIONS} else "unsupported"
            require(interpretation["state"] == state and [intent["type"] for intent in interpretation["intents"]]
                == (["request_clarification"] if state == "clarification_required" else ["unsupported_request"]), "unsupported_semantics_altered")
            if row["case"] in NEGATIVE_MESSAGES:
                require(row["input"] == NEGATIVE_MESSAGES[row["case"]], "negative_prompt_mismatch")
            if row["case"] in NUMERIC_CLARIFICATIONS:
                numeric_clarification(row, assertions["parent"])
        accepts = [http for http in posts if http["url"].endswith("/accept")]
        require(len(accepts) == int(context == "accepted_child") and len([http for http in posts if http["url"].endswith("/preview")]) == len(journeys),
            "extra_preview_or_acceptance")
        if context == "accepted_child":
            transition = core._load(base / "context_accepted_transition.json")
            require(transition["accepted_plan_id"] == assertions["parent"]["active_plan_id"]
                and transition["accepted_plan_content_hash"] == assertions["parent"]["active_plan_content_hash"]
                and transition["conversation_memory"]["pending_clarification"] is None
                and transition["conversation_memory"]["last_proposal"] is None
                and transition["conversation_memory"]["proposed_preferences"] == []
                and transition["capabilities"]["registered_repair_current"] is False, "accepted_memory_not_invalidated")
            places = {place["place_id"]: place for place in transition["known_places"]}
            require(places["pfeiffer_big_sur_state_park"]["route_status"] == "accepted_stop"
                and places.get("bixby_creek_bridge_viewpoint", {}).get("route_status") != "accepted_stop", "removed_stop_still_accepted")
            acceptance = core._load(base / "acceptance_response.json")
            require(json.loads(accepts[0]["response_body_utf8"]) == acceptance
                and acceptance["result"]["decision"]["selected_plan_id"] == assertions["parent"]["active_plan_id"]
                and acceptance["session"]["draft"] == [] and acceptance["session"]["proposal"] is None, "accepted_lineage_unbound")
        labels = ["initial", "final_preserved_parent"] + ["chat_" + family + "_undone" for family in FAMILIES]
        labels += (["invalid_target_blocked", "keep_only_original_undone", "combined_undone"] if context == "original" else
            ["accepted_child_reload", "accepted_context_chat", "keep_only_accepted_undone", "chat_addition_undone"])
        core._verify_journey_captures(base, labels, {key[0] for key in identities})
        require(load(base / "console_errors.json") == [], "browser_errors")
        from verify_e2e1_network_observations import verify as verify_network
        from verify_e2e1_network_observations import verify_asset_observation

        assets = load(base / "map_asset_sources.json")
        verify_asset_observation(assets, assertions["parent"], context, phase="final_network_observation")
        verify_network(base, base_url, load(base / "network_failures.json"),
            {"sources":assets["vector_sources"],"sprite":assets["sprite"]},asset_observation=assets)
    return {"verified": True, "test_count": 2, "scope": "deterministic_chat_family_samples_and_preservation",
        "live_openai_connectivity_proven": False, "all_natural_language_variants_proven": False}


def verify(root, base_url):
    try:
        return _verify(root.resolve(), base_url)
    except (OSError, ValueError, TypeError, KeyError, AttributeError, IndexError, StopIteration, ET.ParseError) as error:
        if isinstance(error, core.E2E1VerificationError):
            raise
        raise core.E2E1VerificationError("e2e1_chat_closure_evidence_malformed") from error
