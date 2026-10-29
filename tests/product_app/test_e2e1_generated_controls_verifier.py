"""Counterexamples for generated chat control identity and actual action receipts."""
from __future__ import annotations

import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
import verify_e2e1_chat_closure as verifier  # noqa: E402
import verify_e2e1_vertical_slice as core  # noqa: E402


def http(phase,suffix,request,response):
    before,after=json.dumps(request),json.dumps(response)
    first,last=(hashlib.sha256(text.encode()).hexdigest() for text in (before,after))
    return {'schema_version':'e2e1-product-http-io-v1','phase':phase,'method':'POST',
        'url':'http://127.0.0.1:8127/api/sessions/s'+suffix,'request_body_utf8':before,
        'request_body_sha256':first,'request_original_body_sha256':first,'response_status_code':200,
        'response_body_utf8':after,'response_body_sha256':last,'response_original_body_sha256':last,
        'credential_redactions':{'request':[],'response':[]},'body_capture':'exact_utf8',
        'request_headers_persisted':False,'response_headers_persisted':False,'authorization_persisted':False,'session_token_persisted':False}


def fixture():
    parent={'active_plan_id':'parent','rendered_map_plan_id':'parent','active_plan_content_hash':'parent-hash',
        'rendered_geometry_hash':'parent-geometry','rendered_marker_ids':['stearns_wharf']}
    operation={'operation_id':'operation','type':'keep_stop','target':'stearns_wharf','parameters':{},'source':'copilot'}
    intent={'type':'keep_stop','target_stop_id':'stearns_wharf'}
    def turn(client,revision):
        return {'context':{'accepted_plan_id':'parent','session_id':'s'},'output':{'turn':{
            'client_message_id':client,'assistant_message':'Keep the stop.','context_revision':revision,
            'interpretation':{'state':'proposal_ready','intents':[intent],'highlights':{'stop_ids':['stearns_wharf']}}}}}
    turns=[turn('add',1),turn('navigate',4)]
    def session(revision,draft=(),stop='golden_gate_bridge'):
        return {'session_id':'s','accepted_plan_id':'parent','revision':revision,'draft':list(draft),'selected_stop_id':stop}
    stage=http('add','/draft/operations',{'source':'copilot','type':'keep_stop','target':'stearns_wharf','parameters':{}},
        {'operation':operation,'session':session(2,[operation])})
    preview=http('preview','/preview',{'request_id':'preview'}, {'proposal':{'child_plan_id':'child'},'session':session(3,[operation])})
    select=http('navigate','/selection',{'selected_stop_id':'stearns_wharf'}, {'session':session(5,stop='stearns_wharf')})
    rendered={'active_plan_id':'child','rendered_map_plan_id':'child'}
    journeys=[{'case':'preview','interaction_surface':'copilot_chat','rendered':rendered,'child_plan_id':'child','operations':[operation]}]
    rows,events=[],[]
    definitions=[('proposal_add','add',1,session(1),session(2,[operation]),stage,parent),
        ('proposal_added_disabled','add',1,session(2,[operation]),session(2,[operation]),None,parent),
        ('proposal_preview','add',1,session(2,[operation]),session(3,[operation]),preview,rendered),
        ('proposal_show_map','navigate',4,session(4),session(5,stop='stearns_wharf'),select,parent),
        ('proposal_cannot_add_disabled','navigate',4,session(5,stop='stearns_wharf'),session(5,stop='stearns_wharf'),None,parent),
        ('proposal_compare','navigate',4,session(5,stop='stearns_wharf'),session(5,stop='stearns_wharf'),None,None),
        ('proposal_evidence','navigate',4,session(5,stop='stearns_wharf'),session(5,stop='stearns_wharf'),None,None)]
    for index,(variant,client,_,before,after,record,map_ack) in enumerate(definitions):
        label,selector=verifier.GENERATED_VARIANTS[variant]
        disabled=variant.endswith('_disabled')
        control={'label':label,'selector':selector,'disabled':disabled,'visible':True,'client_message_id':client,
            'assistant_text':'Keep the stop.','intent_type':'keep_stop' if 'action-row' in selector else None,
            'path':'/app/itinerary','aria_label':label,'observed_at':index*10}
        start=len(events)
        if not disabled:
            events.append({**{key:control[key] for key in ('label','selector','client_message_id','intent_type','path','aria_label')},
                'trusted':True,'at':index*10+1})
        path='/app/map' if variant=='proposal_show_map' else '/app/compare' if variant=='proposal_compare' else '/app/evidence' if variant=='proposal_evidence' else '/app/itinerary'
        view='[data-compare-workspace]' if variant=='proposal_compare' else '[data-evidence-plan-id]' if variant=='proposal_evidence' else None
        rows.append({'variant':variant,'context':'original','phase':record['phase'] if record else 'navigate',
            'assertions_passed':True,'input':{'control':control,'event_start':start,'document_id':'doc','session':before},
            'output':{'session':after,'path':path,'map':map_ack,'observed_at':index*10+2,'http':record,
                'result_selector':view,'view':{'visible':True,'attributes':{}} if view else None},
            'events':deepcopy(events),'requests':[{'method':'POST','url':record['url']}] if record else []})
    return rows,{'schema_version':'generated-control-observer-v1','documents':{'doc':events}},turns,[stage,preview,select],journeys,'original',parent


@pytest.mark.parametrize('fault',[None,'missing_variant','missing_click','untrusted','wrong_owner','wrong_message',
    'wrong_context','hidden','enabled_disabled','forced_disabled','wrong_selector','wrong_label','wrong_intent',
    'wrong_phase','wrong_http','extra_mutation','wrong_parent','wrong_staged_operation','stale_enabled',
    'wrong_child','wrong_preview_map','wrong_stage_hash','wrong_show_markers','not_stale_disabled','wrong_highlight','wrong_selected_stop','show_not_changed',
    'wrong_path','hidden_view','wrong_view','missing_preview','missing_stage','duplicate_action'])
def test_generated_controls_require_exact_turn_native_action_and_runtime_result(fault):
    args=list(deepcopy(fixture()))
    rows,observer,turns,records,journeys,_,_=args
    add,added,preview,show,stale,compare,_=rows
    if fault=='missing_variant':
        rows.pop()
    elif fault=='missing_click':
        add['input']['event_start']=len(add['events'])
    elif fault=='untrusted':
        add['events'][0]['trusted']=False
    elif fault=='wrong_owner':
        add['input']['control']['client_message_id']='foreign'
    elif fault=='wrong_message':
        add['input']['control']['assistant_text']='foreign'
    elif fault=='wrong_context':
        add['context']='accepted_child'
    elif fault=='hidden':
        add['input']['control']['visible']=False
    elif fault=='enabled_disabled':
        added['input']['control']['disabled']=False
    elif fault=='forced_disabled':
        added['input']['event_start']=0
    elif fault=='wrong_selector':
        add['input']['control']['selector']='button'
    elif fault=='wrong_label':
        add['input']['control']['label']='Preview this repair'
    elif fault=='wrong_intent':
        add['input']['control']['intent_type']='replace_nearby'
    elif fault=='wrong_phase':
        add['phase']='other'
    elif fault=='wrong_http':
        records.pop(0)
    elif fault=='extra_mutation':
        show['requests'].append({'method':'POST','url':'http://127.0.0.1:8127/api/sessions/s/accept'})
    elif fault=='wrong_parent':
        add['output']['session']['accepted_plan_id']='other'
    elif fault=='wrong_staged_operation':
        added['output']['session']['draft'][0]['target']='other'
    elif fault=='stale_enabled':
        add['input']['session']['revision']=0
    elif fault=='wrong_child':
        journeys[0]['child_plan_id']='other'
    elif fault=='wrong_preview_map':
        preview['output']['map']={'active_plan_id':'parent','rendered_map_plan_id':'parent'}
    elif fault=='wrong_stage_hash':
        add['output']['map']={**add['output']['map'],'rendered_geometry_hash':'foreign'}
    elif fault=='wrong_show_markers':
        show['output']['map']={**show['output']['map'],'rendered_marker_ids':['foreign']}
    elif fault=='not_stale_disabled':
        stale['input']['session']['revision']=4
    elif fault=='wrong_highlight':
        turns[1]['output']['turn']['interpretation']['highlights']['stop_ids']=[]
    elif fault=='wrong_selected_stop':
        show['output']['session']['selected_stop_id']='other'
    elif fault=='show_not_changed':
        show['input']['session']['selected_stop_id']='stearns_wharf'
    elif fault=='wrong_path':
        compare['output']['path']='/app/itinerary'
    elif fault=='hidden_view':
        compare['output']['view']['visible']=False
    elif fault=='wrong_view':
        compare['output']['result_selector']='main'
    elif fault=='missing_preview':
        journeys.clear()
    elif fault=='missing_stage':
        records.pop(0)
    elif fault=='duplicate_action':
        rows.append(deepcopy(show))
    if fault:
        with pytest.raises(core.E2E1VerificationError):
            verifier.generated_controls(*args)
    else:
        proofs=verifier.generated_controls(*args)
        assert {row['variant'] for row in proofs}==set(verifier.GENERATED_VARIANTS)
        assert sum(row['status']=='disabled_observed' for row in proofs)==2
