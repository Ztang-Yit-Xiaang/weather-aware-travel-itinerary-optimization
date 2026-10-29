"""Independent, explicitly fault-scoped optional Evidence Back verification."""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import verify_e2e1_vertical_slice as core
from verify_e2e1_bootstrap_recovery import acceptance_hash, fingerprint
from verify_e2e1_self_managed import _http, _lifecycle, _private_input_universe, _proposal, _runtime, load, stamp

TEST='test_optional_evidence_projection_fault_preserves_accepted_child_and_back_navigation'
RELATIVE='src/itinerary_system/product_app/service.py'
ANCHOR='        view["geography"] = geography\n        if snapshot is not None:'
FAULT='        view["geography"] = geography\n        # Disposable optional-evidence projection fault; accepted snapshot is untouched.\n        for option in view.get("alternatives", []):\n            if option.get("role") == "low_driving":\n                view.get("alternative_evidence", {}).pop(option["plan_id"], None)\n        if snapshot is not None:'
PROTECTED={'plan_repository','workspaces','decisions','transactions','acceptance_snapshots'}
CATEGORY='declared_backend_projection_failure_path'
BASE_URL='http://127.0.0.1:8127'


def require(condition,reason):
    core._require(condition,'e2e1_optional_evidence_'+reason)


def patch_contract(row,original):
    newline = "\r\n" if b"\r\n" in original else "\n"
    anchor, fault = (text.replace("\n", newline) for text in (ANCHOR, FAULT))
    require(row['schema_version']=='optional-evidence-projection-fault-v1' and row['path']==RELATIVE
        and row['before_utf8'].encode()==original and original.decode().count(anchor)==1
        and row['after_utf8']==original.decode().replace(anchor,fault)
        and row['target_role']=='low_driving'
        and all(row[key] is False for key in ('production_source_modified','client_state_injection','fetch_mocking')),
        'fault_scope_invalid')
    for side in ('before','after'):
        require(hashlib.sha256(row[side+'_utf8'].encode()).hexdigest()==row[side+'_sha256'],'fault_hash_invalid')


def accepted_geography(proposal):
    expected=deepcopy(proposal['geography_plan'])
    expected['role']='accepted'
    expected['label']='Accepted plan'
    for name in ('stops','route_path','validated_legs','gaps'):
        for feature in expected[name]['features']:
            feature['properties']['role']='accepted'
    return expected


def protected_contract(state,value):
    require(state['before']==state['after'] and state['before'],'protected_state_changed')
    documents={}
    for name,row in state['before'].items():
        path=Path(name)
        require(not path.is_absolute() and '..' not in path.parts and path.parts[0] in PROTECTED,
            'protected_path_invalid')
        body=row['utf8'].encode()
        require(len(body)==row['size_bytes'] and hashlib.sha256(body).hexdigest()==row['sha256'],
            'protected_bytes_invalid')
        documents[name]=[json.loads(line) for line in row['utf8'].splitlines() if line.strip()] if name.endswith('.jsonl') else json.loads(row['utf8'])
    accepted=value['acceptance']
    pointer,decision=accepted['result']['pointer'],accepted['result']['decision']
    trip=hashlib.sha256(pointer['trip_id'].encode()).hexdigest()[:32]
    require(documents.get('workspaces/'+trip+'.json')==pointer
        and documents.get('decisions/'+trip+'/ledger.jsonl')==[decision]
        and documents.get('plan_repository/'+pointer['accepted_plan_id']+'.json')==value['proposal']['runtime_execution']['child_plan']
        and documents.get('plan_repository/'+pointer['original_plan_id']+'.json',{}).get('content_hash')==pointer['original_plan_content_hash']
        and 'acceptance_snapshots/'+decision['decision_id']+'.json' in documents,'protected_canonical_artifacts_unbound')
    snapshot=documents['acceptance_snapshots/'+decision['decision_id']+'.json']
    execution=value['proposal']['runtime_execution']
    require(snapshot.get('geography_plan')==accepted_geography(value['proposal'])
        and snapshot.get('plan')==execution['child_plan'] and snapshot.get('diff')==execution['plan_diff']
        and snapshot.get('certificate')==execution['evaluation_certificate']
        and snapshot.get('proposal_identity')=={key:value['proposal'].get(key) for key in (
            'proposal_id','draft_content_hash','diff_id','diff_content_hash','certificate_id','certificate_content_hash')}
        and snapshot.get('route_legs')==[{key:feature['properties'][key] for key in (
            'day','origin_id','destination_id','evidence_scope')} for feature in snapshot['geography_plan']['validated_legs']['features']]
        and snapshot['content_hash']==acceptance_hash(snapshot)==decision['snapshot_content_hash'],
        'accepted_snapshot_changed')


def semantic_contract(value,events,records,patch):
    require(value['schema_version']=='optional-evidence-fault-browser-v1' and value['context']=='accepted_child'
        and value['coverage_category']==CATEGORY and value['natural_interaction_claimed'] is False
        and value['assertions_passed'] is True,'scope_invalid')
    before,after=value['initial'],value['faulted']
    accepted,proposal=value['acceptance'],value['proposal']
    sid=accepted['session']['session_id']
    require(before['session']==after['session']==accepted['session'] and before['session']['draft']==[],
        'accepted_session_changed')
    option=value['optional_plan_id']
    choices=[row for row in before['workspace']['alternatives'] if row['role']=='low_driving']
    require(len(choices)==1 and choices[0]['plan_id']==option==patch['target_plan_id']
        and option!=before['session']['accepted_plan_id'],'optional_target_invalid')
    expected=dict(before['workspace']['alternative_evidence'])
    require(option in expected,'optional_evidence_not_initially_present')
    del expected[option]
    require(after['workspace']['alternative_evidence']==expected
        and after['workspace']['alternatives']==before['workspace']['alternatives']
        and after['workspace']['geography']==before['workspace']['geography']==accepted['workspace']['geography'],
        'fault_modified_more_than_optional_evidence')
    pointer,decision=accepted['result']['pointer'],accepted['result']['decision']
    public={'schema_version','accepted_plan_id','accepted_plan_content_hash','decision_id','revision'}
    require(pointer['content_hash']==acceptance_hash(pointer) and decision['content_hash']==acceptance_hash(decision)
        and before['workspace']['accepted_plan_pointer']==after['workspace']['accepted_plan_pointer']
        ==accepted['workspace']['accepted_plan_pointer']=={key:pointer[key] for key in public}
        and pointer['accepted_plan_id']==decision['selected_plan_id']==proposal['child_plan_id']
        and pointer['accepted_plan_content_hash']==decision['selected_plan_content_hash']==proposal['child_plan_content_hash']
        and pointer['original_plan_id']==decision['previous_plan_id']==proposal['parent_plan_id']
        and pointer['original_plan_content_hash']==value['checkpoints']['initial_parent']['active_plan_content_hash']
        and pointer['decision_id']==decision['decision_id'] and decision['certificate_id']==proposal['certificate_id']
        and all(pointer[key]==accepted['session'][key] for key in ('trip_id','run_id')),'accepted_lineage_invalid')
    plans={row['plan_id']:row for row in after['workspace']['geography']['plans']}
    require(len(plans)==len(after['workspace']['geography']['plans'])
        and plans[proposal['child_plan_id']]==accepted_geography(proposal),'accepted_geography_not_canonical')
    checkpoints=value['checkpoints']
    require(set(checkpoints)=={'initial_parent','evaluated_child','accepted_child','accepted_before_fault',
        'accepted_during_fault','accepted_after_back'},'checkpoint_universe_invalid')
    _proposal(proposal,checkpoints['evaluated_child'])
    for label in ('accepted_child','accepted_before_fault','accepted_during_fault','accepted_after_back'):
        core._verify_plan_ack(checkpoints[label],proposal)
        require(checkpoints[label]==checkpoints['accepted_child'] and checkpoints[label]['preview_state']=='PARENT_RENDERED',
            'accepted_map_changed')
        require(len(checkpoints[label]['rendered_marker_ids'])==len(set(checkpoints[label]['rendered_marker_ids'])),
            'accepted_map_duplicate_markers')
    original=plans[proposal['parent_plan_id']]
    first=checkpoints['initial_parent']
    require(first['active_plan_id']==first['rendered_map_plan_id']==original['plan_id']
        and first['active_plan_content_hash']==original['content_hash']
        and first['rendered_geometry_hash']==core._completed_race_geometry_hash(original)
        and len(first['rendered_marker_ids'])==len(set(first['rendered_marker_ids']))
        and set(first['rendered_marker_ids'])=={item['properties']['stop_id'] for item in original['stops']['features']},
        'original_map_not_canonical')
    require(checkpoints['initial_parent']['active_plan_id']==proposal['parent_plan_id']
        and 'bixby_creek_bridge_viewpoint' in checkpoints['initial_parent']['rendered_marker_ids']
        and 'bixby_creek_bridge_viewpoint' not in checkpoints['accepted_child']['rendered_marker_ids']
        and 'pfeiffer_big_sur_state_park' in checkpoints['accepted_child']['rendered_marker_ids'],'accepted_replacement_missing')
    def responses(suffix,method=None,phase=None):
        return [row for row in records if row['url'].endswith(suffix)
            and (method is None or row['method']==method) and (phase is None or row['phase']==phase)]
    previews,accepts=responses('/preview','POST'),responses('/accept','POST')
    require(len(previews)==len(accepts)==1 and json.loads(previews[0]['response_body_utf8'])['proposal']==proposal
        and json.loads(accepts[0]['response_body_utf8'])==accepted
        and previews[0]['phase']==accepts[0]['phase']=='generation_a','accepted_http_unbound')
    accept_input=json.loads(accepts[0]['request_body_utf8'])
    require(accept_input['confirmation']=='accept_evaluated_repair' and accept_input['idempotency_key']==decision['idempotency_key'],
        'accept_confirmation_unbound')
    for payload,phase in ((before,'generation_a'),(after,'generation_b_reload')):
        require(any(json.loads(row['response_body_utf8'])==payload for row in responses('/api/sessions/'+sid,'GET',phase)),
            'reload_http_unbound')
    selections=responses('/selection','POST','select_optional')
    matches=[row for row in selections if json.loads(row['request_body_utf8']).get('selected_alternative_id')==option]
    require(len(matches)==1 and json.loads(matches[0]['response_body_utf8'])==value['selected'],'option_selection_http_unbound')
    selected=value['selected']['session']
    request=json.loads(matches[0]['request_body_utf8'])
    require(selected['session_id']==sid and selected['accepted_plan_id']==pointer['accepted_plan_id'] and selected['draft']==[]
        and selected['selected_alternative_id']==option and selected['revision']==request['expected_revision']+1,
        'selection_changed_accepted_state')
    post_restart_contract(records,before['session'],pointer)
    unavailable=value['unavailable']
    require(unavailable['path']==BASE_URL+'/app/evidence' and unavailable['heading']=='Evidence unavailable'
        and unavailable['code']=='compare_option_evidence_unavailable' and unavailable['back_visible'] is True
        and unavailable['back_disabled'] is False and unavailable['evidence_panel_count']==0
        and 'id="evidence-back"' in unavailable['html'],'unavailable_branch_not_visible')
    require(len(events)==5 and all(event['trusted'] is True and event['type']=='click' and event['visible'] is True
        and event['disabled'] is False and type(event['at']) in (int,float) for event in events)
        and [event['at'] for event in events]==sorted(event['at'] for event in events),'native_input_invalid')
    require([event['path'] for event in events]==['/app/itinerary','/app/compare','/app/compare','/app/evidence','/app/compare']
        and [event['route'] for event in events]==['/app/compare',None,'/app/evidence',None,'/app/itinerary']
        and events[1]['option']==option and events[3]['id']=='evidence-back'
        and events[3]['label']=='Back to Compare','native_input_sequence_invalid')
    maps=value['compare_after_back']
    require(value['compare_labels']=={'baseline_heading':'Current trip route','baseline_card':'Current accepted trip',
        'baseline_plan_id':pointer['accepted_plan_id'],'original_plan_id':pointer['original_plan_id']},'compare_baseline_label_misleading')
    require(len(maps)==2 and {data['renderedMapPlanId'] for data in maps}=={pointer['accepted_plan_id'],option},
        'compare_map_universe_invalid')
    for data in maps:
        plan=plans[data['renderedMapPlanId']]
        markers=json.loads(data['renderedMarkerIds'])
        require(data['mapReady']=='true' and .95<=float(data['mapTileCoverage'])<=1
            and data['renderedPlanContentHash']==plan['content_hash']
            and data['renderedGeometryHash']==core._completed_race_geometry_hash(plan)
            and len(markers)==len(set(markers))
            and set(markers)=={item['properties']['stop_id'] for item in plan['stops']['features']},'compare_map_not_canonical')
    return {'scope':'Accepted-child Evidence Back after declared optional backend projection fault; canonical accepted state unchanged',
        'coverage_category':CATEGORY,'natural_interaction_claimed':False,'accepted_plan_id':pointer['accepted_plan_id'],
        'accepted_plan_content_hash':pointer['accepted_plan_content_hash'],'selected_optional_plan_id':option}


def post_restart_contract(records,session,pointer):
    """Impact preview is a read-only POST, never a repair or durable mutation."""
    current=session
    prefix=BASE_URL+'/api/sessions/'+session['session_id']
    require(not any(row['url'].endswith('/copilot/messages') for row in records),'unexpected_chat')
    for row in records:
        if row['phase']=='generation_a' or row['method']=='GET':
            continue
        require(row['method']=='POST' and row['url'] in {prefix+'/selection',prefix+'/draft/impact-preview'},
            'fault_unexpected_mutation')
        request=json.loads(row['request_body_utf8'])
        response=json.loads(row['response_body_utf8'])
        require(request['expected_revision']==current['revision'],'post_restart_revision_unbound')
        if row['url'].endswith('/selection'):
            selected=response['session']
            require(all(selected[key]==current[key] for key in ('session_id','trip_id','run_id','accepted_plan_id','draft'))
                and selected['draft']==[] and selected['revision']==current['revision']+1
                and all(selected[key]==value for key,value in request.items() if key!='expected_revision'),
                'post_restart_selection_mutated_parent')
            current=selected
        else:
            require(request=={'expected_revision':current['revision']}
                and response=={'schema_version':'product-draft-impact-preview-v1','session_id':session['session_id'],
                    'session_revision':current['revision'],'parent_plan_id':pointer['accepted_plan_id'],
                    'parent_plan_content_hash':pointer['accepted_plan_content_hash'],'certification_status':'not_certified',
                    'is_certified':False,'operations':[],'summary':{'operation_count':0,'evaluated_executable_count':0,
                        'draft_only_count':0,'can_run_evaluated_preview':False,'blocking_codes':[]}},'impact_not_exact_empty_draft')


def raw_http_contract(requests,records,witnesses):
    require([row['sequence'] for row in requests]==list(range(1,len(requests)+1)),'request_sequence_invalid')
    indexed={row['request_sequence']:row for row in records}
    require(len(indexed)==len(records),'duplicate_http_binding')
    observed=set()
    for row in requests:
        require(row['phase'] in {'generation_a','generation_b_reload','select_optional','evidence_unavailable','evidence_back','return_to_accepted'},
            'request_phase_invalid')
        lifecycle=witnesses[0 if row['phase']=='generation_a' else 1]
        require(row['failed_at'] is None and row['failure'] is None and type(row['status']) is int and 200<=row['status']<400
            and stamp(lifecycle['installed_at'])<=stamp(row['started_at'])<=stamp(row['response_at'])
            <=stamp(row['finished_at'])<=stamp(lifecycle['finalized_at']),'request_failed_or_incomplete')
        if row['url'].startswith(BASE_URL+'/api/'):
            require(row['sequence'] in indexed,'orphan_api_request')
            record=indexed[row['sequence']]
            require(all(row[key]==record[key] for key in ('phase','method','url'))
                and row['request_body_sha256']==record['request_original_body_sha256']
                and row['status']==record['response_status_code'] and row['http_file']==f'http_{records.index(record)+1:03d}.json',
                'raw_http_unbound')
            observed.add(row['sequence'])
        else:
            require(row['http_file'] is None and row['method']=='GET','non_api_mutation')
    require(observed==set(indexed),'http_without_raw_request')


def verify_scenarios(root,run_id,process,nodes):
    base=root/'optional_evidence_fault'
    value=core._load(base/'browser_assertions.json')
    patch=core._load(base/'fault_patch.json')
    patch_contract(patch,(core.ROOT/RELATIVE).read_bytes())
    records=_http(base,BASE_URL)
    events=load(base/'control_inputs.json')
    semantic_contract(value,events,records,patch)
    protected_contract(core._load(base/'protected_state.json'),value)
    copied=core._load(base/'private_fixture_inputs.json')
    rows=copied['files']
    require(copied['schema_version']=='build-coherence-private-inputs-v1' and copied['production_source_modified'] is False
        and copied['api_response_mocking'] is False and len({row['path'] for row in rows})==len(rows)
        and {row['path'] for row in rows}==_private_input_universe(core.ROOT),'private_source_universe_invalid')
    for row in rows:
        path=core._safe_path(core.ROOT,row['path'])
        require(row['sha256']==row['copied_sha256']==core._sha256(path) and row['size_bytes']==path.stat().st_size,
            'private_source_hash_invalid')
    require(load(base/'private_restored_inputs.json')==[{key:row[key] for key in ('path','sha256','size_bytes')} for row in rows]
        and value['private_source_restored_sha256']==patch['before_sha256'],'private_source_not_restored')
    private=Path(copied['private_repository'])
    require(private.is_absolute() and private.resolve().is_relative_to((core.ROOT/'tmp').resolve()) and private.name=='repo',
        'private_directory_invalid')
    bindings=load(base/'generation_bindings.json')
    require([row['generation'] for row in bindings]==['generation_a','generation_b'],'generation_universe_invalid')
    builds=[fingerprint(rows,{}),fingerprint(rows,{RELATIVE:patch['after_utf8'].encode()})]
    witnesses=[]
    ids={value['proposal']['parent_plan_id'],value['proposal']['child_plan_id'],value['optional_plan_id']}
    for index,binding in enumerate(bindings):
        generation=binding['generation']
        directory=base/generation
        identity=run_id+'_'+generation
        receipt=_lifecycle(directory,identity,process,child_process=True)
        require(_runtime(directory,receipt,nodes,BASE_URL)==nodes,'runtime_node_unbound')
        health=core._load(directory/'health.json')
        command=core._load(directory/'server_command.json')
        expected=[process['command'][0],'scripts/e2e1_provider_isolation.py','--port','8127','--state-root',str(private.parent/'state'),
            '--output',str(directory/'provider_isolation.json'),'--run-id',identity]
        require(command['command']==expected and command['cwd_scope']=='private_product_copy' and command['run_id']==identity
            and command['generation']==generation and binding['run_id']==identity and binding['pid']==receipt['process_id']
            and binding['build_id']==health['build_id']==builds[index] and binding['provider_call_count']==0
            and binding['provider_receipt']==generation+'/provider_isolation.json'
            and binding['provider_receipt_sha256']==core._sha256(directory/'provider_isolation.json')
            and all(binding[key]==receipt[key] for key in ('installed_at','finalized_at')),'generation_binding_invalid')
        labels=('initial_parent','evaluated_child','accepted_child','accepted_before_fault') if index==0 else (
            'accepted_during_fault','compare_after_back','accepted_after_back')
        require(binding['readiness_artifacts']==sorted(generation+'/'+label+'.readiness.json' for label in labels),
            'capture_binding_invalid')
        core._verify_journey_captures(directory,labels,ids)
        witnesses.append({**binding,'provider_calls':0})
    require(builds[0]!=builds[1] and stamp(process['started_at'])<=stamp(copied['copied_at'])<=stamp(witnesses[0]['installed_at'])
        and stamp(witnesses[0]['finalized_at'])<=stamp(witnesses[1]['installed_at']),'generation_order_invalid')
    raw_http_contract(load(base/'browser_requests.json'),records,witnesses)
    require(stamp(witnesses[1]['installed_at']).timestamp()*1000<=events[0]['at']<=events[-1]['at']
        <=stamp(core._load(base/'generation_b/accepted_after_back.readiness.json')['screenshot_started_at']).timestamp()*1000,
        'input_outside_lifecycle')
    llm=core._load(base/'llm_io.json')
    require(llm['provider_call_count']==0 and llm['llm_input'] is llm['llm_output'] is None
        and load(base/'console_errors.json')==load(base/'network_failures.json')==[],'errors_or_provider_calls')
    return witnesses
