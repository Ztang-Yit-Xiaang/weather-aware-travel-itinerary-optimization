"""Independent private fault provenance, stopped lifecycles and real Retry gate."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import verify_e2e1_vertical_slice as core
from verify_e2e1_self_managed import _http, _lifecycle, _private_input_universe, _proposal, _runtime, load, stamp

CASES=('core_not_ready','invalid_health')
CONTEXTS=('original','accepted_child')
TEST='test_actual_bootstrap_fault_retry_renders_ready_parent'
SCHEMA='e2e1-bootstrap-fault-isolation-v1'


def require(condition,reason):
    core._require(condition,'e2e1_bootstrap_'+reason)


def acceptance_hash(value):
    """Acceptance ledger hashes are full SHA-256, unlike 16-character plan hashes."""
    try:
        body=json.dumps({key:item for key,item in value.items() if key!='content_hash'},
            ensure_ascii=False,allow_nan=False,sort_keys=True,separators=(',',':')).encode('utf-8')
    except (TypeError,ValueError,UnicodeError):
        require(False,'acceptance_hash_input_invalid')
    return hashlib.sha256(body).hexdigest()


def restored_acceptance(payload,transition):
    accepted=transition['accept_response']
    full_pointer=accepted['result']['pointer']
    public_fields={'schema_version','accepted_plan_id','accepted_plan_content_hash','decision_id','revision'}
    require(full_pointer['content_hash']==acceptance_hash(full_pointer)
        and full_pointer['accepted_plan_id']==transition['proposal']['child_plan_id']
        and full_pointer['accepted_plan_content_hash']==transition['proposal']['child_plan_content_hash']
        and full_pointer['original_plan_id']==transition['proposal']['parent_plan_id']
        and full_pointer['original_plan_content_hash']==transition['parent']['active_plan_content_hash']
        and payload['workspace']['accepted_plan_pointer']==accepted['workspace']['accepted_plan_pointer']
        =={key:full_pointer[key] for key in public_fields},'accepted_pointer_lineage_missing')
    require(payload['session']==accepted['session']
        and payload['session']['session_id']==transition['session']['session_id']
        and all(payload['session'][key]==full_pointer[key] for key in ('trip_id','run_id'))
        and payload['session']['accepted_plan_id']==full_pointer['accepted_plan_id'],
        'accepted_session_owner_invalid')
    plans=payload['workspace']['geography']['plans']
    require(len(plans)==len({plan['plan_id'] for plan in plans})
        and payload['workspace']['geography']==accepted['workspace']['geography'],
        'accepted_geography_not_exact')
    matches=[plan for plan in plans if plan['plan_id']==full_pointer['accepted_plan_id']]
    require(len(matches)==1 and matches[0]['content_hash']==full_pointer['accepted_plan_content_hash'],
        'accepted_geography_identity_invalid')
    markers=[feature['properties']['stop_id'] for feature in matches[0]['stops']['features']]
    require(len(markers)==len(set(markers)),'accepted_geography_duplicate_markers')


def fault_lifecycle(base,run_id,process,case):
    receipt=core._load(base/'provider_isolation.json')
    lifecycle=core._load(base/'server_lifecycle.json')
    require(receipt['schema_version']==SCHEMA and receipt['case']==case and receipt['run_id']==lifecycle['run_id']==run_id
        and type(receipt['process_id']) is int and receipt['process_id']==lifecycle['pid'] and lifecycle['exit_code']==0,
        'fault_lifecycle_identity_invalid')
    require(all(receipt.get(key) is True for key in ('finalized','api_key_disabled','app_initialized','server_started','server_stopped'))
        and receipt['failure_code'] is None and receipt['guard_boundaries']==core.ISOLATION_BOUNDARIES
        and receipt['scope']=='application_server' and type(receipt['sent_provider_calls']) is int and receipt['sent_provider_calls']==0
        and receipt['blocked_attempts']==[] and receipt['normal_guard_relaxed'] is False
        and type(receipt['allowed_local_connections']) is int and receipt['allowed_local_connections']>=0
        and receipt['fixture_scope']=='private_copy_bootstrap_fault','fault_isolation_incomplete')
    require(receipt['actual_service_available'] is (case=='invalid_health') and receipt['core_ready'] is (case=='invalid_health')
        and receipt['actual_adapter']==('DeterministicCopilotAdapter' if case=='invalid_health' else None)
        and receipt['effective_adapter']==('deterministic' if case=='invalid_health' else 'unavailable_core_not_ready'),
        'unavailable_service_falsely_claimed_ready')
    require(lifecycle['receipt_sha256']==core._sha256(base/'provider_isolation.json')
        and stamp(process['started_at'])<=stamp(lifecycle['process_started_at'])<=stamp(receipt['installed_at'])
        <=stamp(lifecycle['fault_observed_at'])<=stamp(receipt['finalized_at'])<=stamp(lifecycle['process_stopped_at'])
        <=stamp(process['completed_at']),'fault_lifecycle_time_or_hash_invalid')
    return receipt


def source_change(change,restored,case):
    expected='configs/product_app_registry.json' if case=='core_not_ready' else 'src/itinerary_system/product_app/api.py'
    require(change['schema_version']=='bootstrap-private-source-fault-v1' and change['case']==case and change['path']==expected
        and change['private_copy_only'] is True and change['frontend_fetch_mocked'] is False,'fault_scope_invalid')
    before,fault=(change[key+'_utf8'].encode() for key in ('before','fault'))
    require(hashlib.sha256(before).hexdigest()==change['before_sha256']==restored['restored_sha256']
        and hashlib.sha256(fault).hexdigest()==change['fault_sha256']==restored['fault_sha256']
        and before==(core.ROOT/expected).read_bytes(),'fault_source_binding_invalid')
    if case=='core_not_ready':
        original,broken=json.loads(before),json.loads(fault)
        require(broken.get('schema_version')=='bootstrap-intentionally-invalid-registry-v1','registry_fault_missing')
        broken['schema_version']=original['schema_version']
        require(broken==original,'unrelated_registry_fault')
    else:
        original='        return runtime.health().as_dict()'
        replacement='        return {**runtime.health().as_dict(), "schema_version": "bootstrap-intentionally-invalid-health-v1"}'
        require(before.decode().count(original)==1 and fault==before.decode().replace(original,replacement).encode(),
            'unrelated_health_fault')
    return expected,before,fault


def fingerprint(rows,override):
    suffixes={'.css','.html','.js','.json','.py','.svg','.webmanifest'}
    paths={'scripts/run_product_app.py','configs/product_app_registry.json','configs/product_route_delta.json'}
    paths.add(core._load(core.ROOT/'configs/product_route_delta.json')['descriptor_path'])
    paths.update(row['path'] for row in rows if row['path'].startswith('src/itinerary_system/product_app/')
        and Path(row['path']).suffix.lower() in suffixes)
    require(paths<={row['path'] for row in rows},'fingerprint_input_missing')
    digest=hashlib.sha256()
    for name in sorted(paths,key=lambda name:core.ROOT/name):
        body=override[name] if name in override else (core.ROOT/name).read_bytes()
        digest.update(name.encode()+b'\0'+body+b'\0')
    return digest.hexdigest()[:16]


def command_binding(commands,private,directory,identity,process,case):
    require(private.is_absolute() and private.resolve().is_relative_to((core.ROOT/'tmp').resolve())
        and private.name=='repo' and all(row['cwd_scope']=='private_product_copy' for row in commands.values()),
        'private_command_scope_invalid')
    common=['--port','8127']
    expected={
        'fault':[process['command'][0],str(core.ROOT/'tests/product_app/browser/bootstrap_recovery_fixture.py'),
            '--repository',str(private),'--state',str(private.parent/'state'),*common,
            '--output',str(directory/'fault/provider_isolation.json'),'--run-id',identity+'_fault','--case',case],
        'recovered':[process['command'][0],'scripts/e2e1_provider_isolation.py',*common,
            '--state-root',str(private.parent/'state'),'--output',str(directory/'recovered/provider_isolation.json'),
            '--run-id',identity+'_recovered']}
    require(set(commands)==set(expected) and all(commands[key]['command']==expected[key]
        and commands[key]['run_id']==identity+'_'+key for key in expected),'private_command_not_exact')


def blocked_result(blocked,health,case,context='original'):
    title='The workspace is not ready' if case=='core_not_ready' else 'Copilot could not start'
    require(blocked['case']==case and blocked['title']==title and 'id="retry-runtime"' in blocked['html']
        and blocked['app_hidden'] is True and blocked['ready_maps']==blocked['rendered_ack_count']==0
        and blocked['session_present'] is (context=='accepted_child')
        and blocked['screenshot_taken'] is False,'blocked_ui_falsely_ready')
    if case=='core_not_ready':
        require(health['schema_version']=='product-health-v2' and health['core_ready'] is False
            and health['components']['registry']['status']=='failed','core_fault_not_real')
    else:
        require(health['schema_version']=='bootstrap-intentionally-invalid-health-v1' and health['core_ready'] is True
            and 'invalid_product_health' in blocked['html'],'health_contract_fault_not_real')


def raw_requests(rows,records,fault,recovered):
    require(isinstance(rows,list) and rows and [row['sequence'] for row in rows]==list(range(1,len(rows)+1)),
        'raw_request_sequence_invalid')
    by_sequence={row['request_sequence']:row for row in records}
    require(len(by_sequence)==len(records),'duplicate_http_request_binding')
    observed=set()
    for row in rows:
        require(row['phase'] in {'fault','recovered','accept_setup'},'raw_request_phase_invalid')
        lifecycle=fault if row['phase']=='fault' else recovered
        require(row['failed_at'] is None and row['failure'] is None and type(row['status']) is int
            and 200<=row['status']<400 and stamp(lifecycle['installed_at'])<=stamp(row['started_at'])
            <=stamp(row['response_at'])<=stamp(row['finished_at'])<=stamp(lifecycle['finalized_at']),
            'orphan_failed_or_out_of_lifecycle_request')
        if '/api/' in row['url']:
            require(row['sequence'] in by_sequence,'unanswered_api_request')
            record=by_sequence[row['sequence']]
            require(row['http_file']==f"http_{records.index(record)+1:03d}.json"
                and all(row[key]==record[key] for key in ('phase','method','url'))
                and row['status']==record['response_status_code']
                and row['request_body_sha256']==record['request_original_body_sha256'],'raw_http_unbound')
            observed.add(row['sequence'])
        else:
            require(row['http_file'] is None and row['method']=='GET','unexpected_non_api_mutation')
    require(observed==set(by_sequence),'http_without_request')


def retry_observer(observer,blocked,result,recovered,ready):
    before,after=observer['before'],observer['after']
    require(isinstance(before,list) and isinstance(after,list) and before==blocked['observer_before']
        and len(after)==len(before)+1 and after[:-1]==before and after[-1]==result['retry']
        and [row['sequence'] for row in after]==list(range(1,len(after)+1)), 'retry_raw_observer_unbound')
    click=after[-1]
    require(click['trusted'] is True and click['id']=='retry-runtime' and click['title']==blocked['title']
        and click['path']=='/app/itinerary' and stamp(blocked['observed_at'])<=stamp(recovered['installed_at'])
        <=stamp(click['clicked_at'])<=stamp(ready['screenshot_started_at']),'retry_click_missing')


def accepted_transition(value,records):
    previews=[json.loads(row['response_body_utf8']) for row in records if row['url'].endswith('/preview')]
    accepts=[json.loads(row['response_body_utf8']) for row in records if row['url'].endswith('/accept')]
    require(len(previews)==len(accepts)==1 and previews[0]['proposal']==value['proposal']
        and accepts[0]==value['accept_response'],'accepted_setup_http_unbound')
    proposal=value['proposal']
    _proposal(proposal,value['preview_map'])
    core._verify_plan_ack(value['accepted_map'],proposal)
    decision=accepts[0]['result']['decision']
    pointer=accepts[0]['result']['pointer']
    accept_record=next(row for row in records if row['url'].endswith('/accept'))
    accept_input=json.loads(accept_record['request_body_utf8'])
    require(proposal['parent_plan_id']==value['parent']['active_plan_id']
        and accept_record['phase']=='accept_setup' and accept_input['confirmation']=='accept_evaluated_repair'
        and decision['idempotency_key']==accept_input['idempotency_key']
        and decision['previous_plan_id']==proposal['parent_plan_id']
        and decision['selected_plan_id']==proposal['child_plan_id']
        and decision['selected_plan_content_hash']==proposal['child_plan_content_hash']
        and decision['certificate_id']==proposal['certificate_id']
        and decision['content_hash']==acceptance_hash(decision)
        and pointer['content_hash']==acceptance_hash(pointer)
        and pointer['accepted_plan_id']==proposal['child_plan_id']
        and pointer['accepted_plan_content_hash']==proposal['child_plan_content_hash']
        and pointer['original_plan_id']==proposal['parent_plan_id'] and pointer['decision_id']==decision['decision_id']
        and value['accepted_map']['preview_state']=='PARENT_RENDERED'
        and 'pfeiffer_big_sur_state_park' in value['accepted_map']['rendered_marker_ids']
        and 'bixby_creek_bridge_viewpoint' not in value['accepted_map']['rendered_marker_ids'],
        'accepted_setup_lineage_invalid')
    restored_acceptance(accepts[0],value)
    return decision


def verify_scenarios(root,run_id,process,nodes):
    base=root/'bootstrap_recovery'
    require({path.name for path in base.iterdir() if path.is_dir()}==set(CASES),'case_directory_universe_invalid')
    witnesses=[]
    observed=set()
    for case,plan_context in [(case,context) for case in CASES for context in CONTEXTS]:
        require({path.name for path in (base/case).iterdir() if path.is_dir()}==set(CONTEXTS),'context_universe_invalid')
        directory=base/case/plan_context
        identity=run_id+'_'+case+'_'+plan_context
        fault=fault_lifecycle(directory/'fault',identity+'_fault',process,case)
        if plan_context=='accepted_child':
            require(stamp(core._load(base/case/'original/recovered/provider_isolation.json')['finalized_at'])
                <=stamp(fault['installed_at']),'accepted_context_started_before_original_stopped')
        recovered=_lifecycle(directory/'recovered',identity+'_recovered',process,child_process=True)
        observed.update(_runtime(directory/'recovered',recovered,nodes,'http://127.0.0.1:8127'))
        change,restore=(core._load(directory/name) for name in ('fault_source_change.json','fault_source_restored.json'))
        relative,before,broken=source_change(change,restore,case)
        copied=core._load(directory/'private_fixture_inputs.json')
        require(copied==core._load(base/case/'private_fixture_inputs.json'),'context_private_copy_unbound')
        commands={name:core._load(directory/name/'server_command.json') for name in ('fault','recovered')}
        private=Path(copied['private_repository'])
        command_binding(commands,private,directory,identity,process,case)
        rows=copied['files']
        require(copied['production_source_modified'] is False and copied['api_response_mocking'] is False
            and len(rows)==len({row['path'] for row in rows}) and {row['path'] for row in rows}==_private_input_universe(core.ROOT),
            'private_input_universe_invalid')
        for row in rows:
            path=core._safe_path(core.ROOT,row['path'])
            require(path.stat().st_size==row['size_bytes'] and core._sha256(path)==row['sha256']==row['copied_sha256'],
                'private_copy_not_exact')
        require(stamp(process['started_at'])<=stamp(copied['copied_at'])<=stamp(change['created_at'])<=stamp(fault['installed_at'])
            and stamp(fault['finalized_at'])<=stamp(restore['restored_at'])<=stamp(recovered['installed_at']),
            'fault_restore_restart_order_invalid')
        probe=core._load(directory/'fault/health_probe.json')
        require(probe['method']=='GET' and probe['url']=='http://127.0.0.1:8127/api/health' and probe['status']==200
            and hashlib.sha256(probe['response_body_utf8'].encode()).hexdigest()==probe['response_body_sha256'],
            'fault_probe_io_invalid')
        health=json.loads(probe['response_body_utf8'])
        repaired_health=core._load(directory/'recovered/health.json')
        require(health['build_id']==fingerprint(rows,{relative:broken})
            and repaired_health['build_id']==fingerprint(rows,{relative:before}) and repaired_health['core_ready'] is True,
            'private_build_identity_invalid')
        blocked=core._load(directory/'blocked_dom.json')
        blocked_result(blocked,health,case,plan_context)
        require(blocked['fault_health']==health and stamp(fault['installed_at'])<=stamp(blocked['observed_at'])<=stamp(fault['finalized_at']),
            'blocked_health_or_time_unbound')
        records=_http(directory,'http://127.0.0.1:8127')
        raw_requests(load(directory/'browser_requests.json'),records,fault,recovered)
        require(stamp(probe['observed_at'])==stamp(core._load(directory/'fault/server_lifecycle.json')['fault_observed_at'])
            and stamp(fault['installed_at'])<=stamp(probe['observed_at'])<=stamp(blocked['observed_at']),
            'fault_probe_time_unbound')
        fault_records=[row for row in records if row['phase']=='fault']
        require(fault_records and all(row['method']=='GET' and row['url'].endswith('/api/health') for row in fault_records),
            'fault_created_session_or_other_request')
        for row in fault_records:
            browser_health=json.loads(row['response_body_utf8'])
            blocked_result(blocked,browser_health,case,plan_context)
            require(browser_health['build_id']==health['build_id'],'browser_fault_build_unbound')
        asserts=core._load(directory/'browser_assertions.json')
        require(asserts['schema_version']=='bootstrap-recovery-browser-v1' and asserts['case']==case
            and asserts['context']==blocked['context']==plan_context
            and asserts['assertions_passed'] is True and asserts['accepted_plan_changed'] is False
            and asserts['frontend_fetch_mocked'] is False and asserts['blocked_screenshot_taken'] is False
            and type(asserts['provider_calls']) is int and asserts['provider_calls']==0,'browser_scope_invalid')
        click=asserts['retry']
        require(click['trusted'] is True and click['id']=='retry-runtime' and click['title']==blocked['title']
            and click['path']=='/app/itinerary' and stamp(recovered['installed_at'])<=stamp(click['clicked_at']),
            'retry_click_missing')
        creations=[row for row in records if row['method']=='POST' and row['url'].endswith('/api/sessions')]
        if plan_context=='original':
            require(len(creations)==1 and creations[0]['phase']=='recovered'
                and asserts['accepted_transition'] is None,'session_creation_universe_invalid')
            payload=json.loads(creations[0]['response_body_utf8'])
            accepted_transition(core._load(directory/'accepted_transition.json'),records)
        else:
            transition=core._load(base/case/'original/accepted_transition.json')
            require(not creations and asserts['accepted_transition']==transition
                and asserts['parent']==transition['accepted_map'] and asserts['session']==blocked['session']==transition['session'],
                'accepted_context_reset_or_mixed')
            restores=[row for row in records if row['method']=='GET' and row['url'].endswith('/api/sessions/'+asserts['session']['session_id'])]
            require(len(restores)==1 and restores[0]['phase']=='recovered','accepted_session_restore_missing')
            payload=json.loads(restores[0]['response_body_utf8'])
            restored_acceptance(payload,transition)
        require(asserts['session']['present'] is True and payload['session']['session_id']==asserts['session']['session_id'],
            'restored_session_identity_invalid')
        parent=asserts['parent']
        plans=payload['workspace']['geography']['plans']
        require(len(plans)==len({plan['plan_id'] for plan in plans}),'recovered_geography_duplicate_plans')
        matching=[plan for plan in plans if plan['plan_id']==payload['session']['accepted_plan_id']]
        require(len(matching)==1,'recovered_active_plan_missing')
        plan=matching[0]
        marker_ids=[feature['properties']['stop_id'] for feature in plan['stops']['features']]
        require(parent['preview_state']=='PARENT_RENDERED' and parent['active_plan_id']==parent['rendered_map_plan_id']==plan['plan_id']
            and parent['active_plan_content_hash']==plan['content_hash'] and parent['rendered_geometry_hash']==core._completed_race_geometry_hash(plan)
            and len(marker_ids)==len(set(marker_ids)) and len(parent['rendered_marker_ids'])==len(marker_ids)
            and set(parent['rendered_marker_ids'])==set(marker_ids)
            and payload['session']['draft']==[] and payload['session']['proposal'] is None,'recovered_parent_not_canonical')
        require(all(row['method']=='GET' or (row['method']=='POST' and row['url'].endswith(('/api/sessions','/draft/impact-preview')))
            or (plan_context=='original' and row['phase']=='accept_setup' and row['method']=='POST'
                and row['url'].endswith(('/selection','/draft/operations','/preview','/accept')))
            for row in records),'unexpected_mutation')
        labels=['recovered_parent']+(['evaluated_setup','accepted_setup'] if plan_context=='original' else [])
        ids={plan['plan_id']}
        if plan_context=='original':
            ids.add(core._load(directory/'accepted_transition.json')['proposal']['child_plan_id'])
        core._verify_journey_captures(directory/'recovered',labels,ids)
        capture_acks={'recovered_parent':parent}
        if plan_context=='original':
            transition=core._load(directory/'accepted_transition.json')
            capture_acks.update(evaluated_setup=transition['preview_map'],accepted_setup=transition['accepted_map'])
        for label,ack in capture_acks.items():
            capture=core._load(directory/'recovered'/f'{label}.readiness.json')
            require(capture['rendered_plan_id']==ack['rendered_map_plan_id']
                and capture['rendered_geometry_hash']==ack['rendered_geometry_hash']
                and len(capture['visible_maps'])==1
                and capture['visible_maps'][0]['plan_content_hash']==ack['active_plan_content_hash']
                and set(capture['visible_maps'][0]['marker_ids'])==set(ack['rendered_marker_ids']),
                'capture_label_not_canonical')
        ready=core._load(directory/'recovered/recovered_parent.readiness.json')
        retry_observer(core._load(directory/'retry_observer.json'),blocked,asserts,recovered,ready)
        require(stamp(click['clicked_at'])<=stamp(ready['screenshot_started_at']) and not list(directory.glob('*.png'))
            and not list((directory/'fault').glob('*.png')),'pre_ready_screenshot')
        require(core._load(directory/'browser_errors.json')=={'errors':[],'request_failures':[]},'browser_errors')
        llm=core._load(directory/'llm_io.json')
        require(llm['provider_call_count']==0 and llm['llm_input'] is llm['llm_output'] is None,'provider_scope_changed')
        witnesses.extend([fault,recovered])
    require(observed==nodes,'runtime_node_universe_invalid')
    return witnesses
