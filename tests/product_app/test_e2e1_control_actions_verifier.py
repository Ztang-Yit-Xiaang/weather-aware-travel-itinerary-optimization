"""Independent native-control receipts reject alias, form and request substitutions."""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import verify_e2e1_control_actions as verifier  # noqa: E402
import verify_e2e1_vertical_slice as core  # noqa: E402


def receipt():
    request = {"type": "set_stop_role", "source": "map", "target": "stop", "parameters": {"role": "meal"}}
    operation = {**deepcopy(request), "operation_id": "draft-1"}
    response = {"operation": operation, "session": {"draft": [operation]}}
    control = {
        "id": None,
        "visible": True,
        "disabled": False,
        "operation": "set_stop_role",
        "form_id": 8,
        "path": "/app/map",
        "fields": {"role": "meal"},
    }
    common = {
        "trusted": True,
        "operation": "set_stop_role",
        "form_id": 8,
        "path": "/app/map",
        "fields": {"role": "meal"},
    }
    events = [
        {**common, "type": "change", "at": 10, "name": "role", "value": "meal"},
        {**common, "type": "submit", "at": 11, "name": None, "value": None},
    ]
    row = {
        "context": "original",
        "phase": "role_meal",
        "assertions_passed": True,
        "request": request,
        "response": response,
        "control": control,
        "event_start": 1,
        "event_end": 2,
        "raw_events": events,
    }
    http = {
        "method": "POST",
        "url": "http://127.0.0.1:8127/api/sessions/s/draft/operations",
        "phase": row["phase"],
        "request_body_utf8": json.dumps(request),
        "response_body_utf8": json.dumps(response),
    }
    return row, http


def decision_fixture():
    import hashlib
    def signed(value):
        return {**value,'content_hash':hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),
            ensure_ascii=False,allow_nan=False).encode()).hexdigest()}
    assertions={name:{'active_plan_id':name,'active_plan_content_hash':'h-'+name,'rendered_map_plan_id':name,
        'rendered_geometry_hash':'g-'+name,'rendered_marker_ids':[name],'preview_state':'PREVIEW_RENDERED'} for name in ('a','b','c')}
    rows,events,records=[],[],[]
    previous='a'
    for index,(kind,context,name) in enumerate((('accept_repair','original','b'),('accept_repair','accepted_child','c'),
            ('keep_original','accepted_child','a'),('keep_original','original','a'))):
        accepted=kind=='accept_repair'
        selector='#accept-repair' if accepted else '#keep-original'
        endpoint='accept' if accepted else 'keep-original'
        before={'session_id':'s','trip_id':'trip','run_id':'run','accepted_plan_id':previous,'revision':index,'draft':[],'proposal':None}
        after={**before,'accepted_plan_id':name,'revision':index+1}
        decision=signed({'decision_id':'d'+str(index),'decision_type':kind,'previous_plan_id':previous,
            'selected_plan_id':name,'selected_plan_content_hash':'h-'+name,'session_revision_before':index,
            'session_revision_after':index+1,'pointer_revision_after':index+1})
        pointer=signed({'schema_version':'accepted-plan-pointer-v1','accepted_plan_id':name,'accepted_plan_content_hash':'h-'+name,
            'original_plan_id':'a','original_plan_content_hash':'h-a','decision_id':decision['decision_id'],'revision':index+1})
        public={key:pointer[key] for key in ('schema_version','accepted_plan_id','accepted_plan_content_hash','decision_id','revision')}
        payload={'session':after,'result':{'decision':decision,'pointer':pointer},'workspace':{'accepted_plan_pointer':public}}
        request={'confirmation':'accept_evaluated_repair' if accepted else 'keep_original_plan','expected_revision':index,
            'idempotency_key':endpoint+':unique'+str(index)}
        event={'trusted':True,'type':'click','at':100+index,'selector':selector,'path':'/app/itinerary','visible':True,'disabled':False}
        row={'binding_id':kind,'selector':selector,'context':context,'phase':str(index),'events':[event],
            'confirmation_messages':['Accept this independently evaluated repair as the durable trip plan?' if accepted else
                'Keep the immutable original plan as the durable trip plan?'],'assertions_passed':True,
            'session_before':before,'request':request,'response':payload,'restored':deepcopy(payload),
            'rendered_before':deepcopy(assertions[name if accepted else previous]),'rendered_after':deepcopy(assertions[name])}
        records.append({'method':'POST','url':'http://local/api/sessions/s/'+endpoint,'phase':str(index),
            'request_body_utf8':json.dumps(request),'response_body_utf8':json.dumps(payload)})
        rows.append(row)
        events.append(event)
        previous=name
    return rows,events,records,assertions


@pytest.mark.parametrize('fault',[None,'context','missing_click','untrusted','disabled','wrong_selector','no_confirmation',
    'wrong_request','wrong_response','stale_revision','foreign_session','wrong_lineage','wrong_target','wrong_pointer',
    'wrong_restored_pointer','wrong_restored_hash','wrong_marker','wrong_geometry','not_rendered'])
def test_exact_decision_controls_reject_false_durable_acceptance_credit(fault):
    rows,events,records,assertions=decision_fixture()
    row=rows[1]
    if fault=='context':
        row['context']='original'
    elif fault=='missing_click':
        events.pop()
    elif fault=='untrusted':
        row['events'][0]['trusted']=False
    elif fault=='disabled':
        row['events'][0]['disabled']=True
    elif fault=='wrong_selector':
        row['events'][0]['selector']='#compare-choose'
    elif fault=='no_confirmation':
        row['confirmation_messages']=[]
    elif fault=='wrong_request':
        records[1]['request_body_utf8']='{}'
    elif fault=='wrong_response':
        records[1]['response_body_utf8']='{}'
    elif fault=='stale_revision':
        row['session_before']['revision']=99
    elif fault=='foreign_session':
        row['session_before']['session_id']='foreign'
    elif fault=='wrong_lineage':
        row['session_before']['accepted_plan_id']='foreign'
    elif fault=='wrong_target':
        assertions['c']['active_plan_id']='other'
    elif fault=='wrong_pointer':
        row['response']['result']['pointer']['content_hash']='bad'
        records[1]['response_body_utf8']=json.dumps(row['response'])
    elif fault=='wrong_restored_pointer':
        row['restored']['workspace']['accepted_plan_pointer']['accepted_plan_id']='other'
    elif fault=='wrong_restored_hash':
        row['restored']['workspace']['accepted_plan_pointer']['accepted_plan_content_hash']='other'
    elif fault=='wrong_marker':
        row['rendered_after']['rendered_marker_ids']=[]
    elif fault=='wrong_geometry':
        row['rendered_after']['rendered_geometry_hash']='other'
    elif fault=='not_rendered':
        row['rendered_before']['preview_state']='PREVIEW_DATA_READY'
    if fault:
        with pytest.raises(core.E2E1VerificationError):
            verifier.decision_controls(rows,events,records,assertions)
    else:
        result=verifier.decision_controls(rows,events,records,assertions)
        assert {(row['binding_id'],row['context']) for row in result}=={
            (kind,context) for kind in ('accept_repair','keep_original') for context in ('original','accepted_child')}


@pytest.mark.parametrize(
    "fault",
    [
        None,
        "hidden",
        "disabled",
        "wrong_target",
        "wrong_request",
        "wrong_phase",
        "wrong_context",
        "untrusted_submit",
        "wrong_submit_form",
        "wrong_control_form",
        "wrong_submit_fields",
        "wrong_submit_path",
        "wrong_control_value",
        "missing_submit",
        "extra_response",
        "missing_receipt",
        "duplicate_receipt",
        "noninteger_range",
        "bad_range",
    ],
)
def test_exact_native_input_to_canonical_draft_binding(fault):
    row, http = receipt()
    rows, records = [row], [http]
    if fault == "hidden":
        row["control"]["visible"] = False
    elif fault == "disabled":
        row["control"]["disabled"] = True
    elif fault == "wrong_target":
        row["request"]["target"] = "foreign"
    elif fault == "wrong_request":
        http["request_body_utf8"] = "{}"
    elif fault == "wrong_phase":
        http["phase"] = "foreign"
    elif fault == "wrong_context":
        row["context"] = "accepted_child"
    elif fault == "untrusted_submit":
        row["raw_events"][1]["trusted"] = False
    elif fault == "wrong_submit_form":
        row["raw_events"][1]["form_id"] = 9
    elif fault == "wrong_control_form":
        row["control"]["operation"] = "set_stop_day"
    elif fault == "wrong_submit_fields":
        row["raw_events"][1]["fields"] = {"role": "attraction"}
    elif fault == "wrong_submit_path":
        row["raw_events"][1]["path"] = "/elsewhere"
    elif fault == "wrong_control_value":
        row["control"]["fields"]["role"] = "attraction"
    elif fault == "missing_submit":
        row["raw_events"][1]["type"] = "click"
    elif fault == "extra_response":
        records.append(deepcopy(http))
    elif fault == "missing_receipt":
        rows = []
    elif fault == "duplicate_receipt":
        rows.append(deepcopy(row))
    elif fault == "noninteger_range":
        row["event_start"] = True
    elif fault == "bad_range":
        row["event_end"] = 1
    if fault:
        with pytest.raises(core.E2E1VerificationError):
            verifier.verify_rows(rows, records, "original")
    else:
        assert {proof["binding_id"] for proof in verifier.verify_rows(rows, records, "original")} == {
            "typed_submit",
            "typed_combination",
            "role_value",
        }


@pytest.mark.parametrize("fault", ["foreign_form", "untrusted", "later", "wrong_name", "wrong_value", "foreign_path"])
def test_no_alias_credit_for_an_unrelated_or_stale_change(fault):
    row, http = receipt()
    change = row["raw_events"][0]
    if fault == "foreign_form":
        change["form_id"] = 7
    elif fault == "untrusted":
        change["trusted"] = False
    elif fault == "later":
        change["at"] = 12
    elif fault == "wrong_name":
        change["name"] = "another"
    elif fault == "wrong_value":
        change["value"] = "another"
    elif fault == "foreign_path":
        change["path"] = "/elsewhere"
    assert [proof["binding_id"] for proof in verifier.verify_rows([row], [http])] == ["typed_submit"]


def test_change_can_precede_filling_another_field_in_the_same_form():
    row, http = receipt()
    row["raw_events"][0]["fields"] = {"role": "meal", "other": "before"}
    assert "role_value" in {proof["binding_id"] for proof in verifier.verify_rows([row], [http])}


@pytest.mark.parametrize(
    "kind,parameters,fields",
    [
        ("set_stop_day", {"day": 6}, {"day": "6"}),
        ("set_stop_order", {"day": 7, "sequence_index": 0}, {"day": "7", "sequence_position": "1"}),
        (
            "set_stop_time_window",
            {"earliest_arrival": "10:00", "latest_departure": None},
            {"earliest_arrival": "10:00", "latest_departure": ""},
        ),
        (
            "set_stop_commitment",
            {"strength": "prefer_keep", "scope_lifetime": "current_repair_session"},
            {"strength": "prefer_keep", "scope_lifetime": "current_repair_session"},
        ),
        ("report_route_issue", {"issue_type": "closure", "note": "known"}, {"issue_type": "closure", "note": "known"}),
        (
            "set_stop_duration",
            {"duration": {"mode": "exact", "preferred_minutes": 60, "minimum_minutes": 60, "maximum_minutes": 60}},
            {"duration_mode": "exact", "duration-preferred": "60"},
        ),
        (
            "set_attribute_constraint",
            {
                "attribute": "existence",
                "value": True,
                "strength": "prefer_keep",
                "scope_lifetime": "current_repair_session",
            },
            {
                "attribute": "existence",
                "attribute_existence": True,
                "strength": "prefer_keep",
                "scope_lifetime": "current_repair_session",
            },
        ),
    ],
)
def test_real_public_field_names_and_one_based_order_are_bound(kind, parameters, fields):
    verifier.parameter_fields(kind, parameters, fields)
    bad = deepcopy(fields)
    key = next(
        (key for key in bad if key not in {"attribute", "strength", "scope_lifetime", "duration_mode"}), next(iter(bad))
    )
    bad[key] = "foreign"
    with pytest.raises((core.E2E1VerificationError, ValueError)):
        verifier.parameter_fields(kind, parameters, bad)


@pytest.mark.parametrize('fault',[None,'missing_input','untrusted','other_form','enabled','hidden','wrong_capability',
    'changed_map','different_day','request_post'])
def test_disabled_day_requires_actual_change_with_server_bound_rejection(fault):
    control={'operation':'set_stop_day','form_id':3,'disabled':True,'visible':True,'path':'/app/map','fields':{'day':'6'}}
    events=[{'type':'change','trusted':True,'operation':'set_stop_day','form_id':3,'name':'day','value':'6','path':'/app/map'}]
    cap={'enabled':False}
    parent={'active_plan_id':'parent'}
    value={'context':'original','assertions_passed':True,'capability':cap,'parent':parent,'map':deepcopy(parent),'phase':'inventory',
        'before':{'control':deepcopy(control),'event_start':0},'after':{'control':deepcopy(control),'event_start':1},'events':events}
    records=[]
    if fault=='missing_input':
        events.clear()
    elif fault=='untrusted':
        events[0]['trusted']=False
    elif fault=='other_form':
        events[0]['form_id']=4
    elif fault=='enabled':
        value['after']['control']['disabled']=False
    elif fault=='hidden':
        value['after']['control']['visible']=False
    elif fault=='wrong_capability':
        value['capability']={'enabled':True}
    elif fault=='changed_map':
        value['map']={'active_plan_id':'child'}
    elif fault=='different_day':
        value['after']['control']['fields']['day']='5'
    elif fault=='request_post':
        records=[{'phase':'inventory','method':'POST','url':'http://127.0.0.1:8127/api/sessions/s/draft/operations'}]
    if fault:
        with pytest.raises(core.E2E1VerificationError):
            verifier.disabled_day(value,cap,records,'original',parent)
    else:
        assert verifier.disabled_day(value,cap,records,'original',parent)['unsupported_claimed'] is False
