"""Real bootstrap faults and Retry in original and persistently accepted contexts."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip('playwright.sync_api')
from bootstrap_recovery_fixture import fault_generation, install_fault, restore_fault
from build_coherence_fixture import copy_product_inputs, guarded_generation
from test_e2e1_live_copilot_chat import (
    CHROME,
    _capture_ready,
    _http_record,
    _map_ack,
    _write_json,
    expect,
    sync_playwright,
)

BASE_URL=os.environ.get('PRODUCT_E2E_BASE_URL')
DISPOSABLE=os.environ.get('PRODUCT_E2E_BOOTSTRAP_RECOVERY_DISPOSABLE')=='1'
TEST_KEY='e2e.bootstrap.retry.observations'
SESSION_KEY='itinerary-repair-copilot.session.v1'


def now():
    return datetime.now(UTC).isoformat()


def session_identity(page):
    return page.evaluate("""async key=>{
      const raw=sessionStorage.getItem(key),parsed=JSON.parse(raw||'null');
      const hash=await crypto.subtle.digest('SHA-256',new TextEncoder().encode(raw||''));
      return {present:raw!==null,session_id:parsed?.session_id||null,
        sha256:[...new Uint8Array(hash)].map(x=>x.toString(16).padStart(2,'0')).join('')};
    }""",SESSION_KEY)


@pytest.mark.skipif(not BASE_URL or not DISPOSABLE,reason='Requires isolated self-managed bootstrap-recovery mode')
@pytest.mark.parametrize('case_name',['core_not_ready','invalid_health'])
def test_actual_bootstrap_fault_retry_renders_ready_parent(tmp_path,case_name):
    case_dir=Path(os.environ.get('PRODUCT_E2E_BOOTSTRAP_RECOVERY_EVIDENCE_DIR',tmp_path))/case_name
    case_dir.mkdir(parents=True,exist_ok=False)
    root_run_id=os.environ['PRODUCT_E2E_RUN_ID']+'_'+case_name
    private_base=Path(__file__).resolve().parents[3]/'tmp'
    private_base.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='b',dir=private_base,delete=False) as temporary,sync_playwright() as runtime:
        repository,state=Path(temporary)/'repo',Path(temporary)/'state'
        copy_product_inputs(repository,case_dir)
        copied=json.loads((case_dir/'private_fixture_inputs.json').read_text(encoding='utf-8'))
        browser=runtime.chromium.launch(headless=True,**({'executable_path':str(CHROME)} if CHROME.is_file() else {}))
        context=browser.new_context(viewport={'width':1440,'height':900})
        context.add_init_script("""document.addEventListener('click',event=>{
          const button=event.target.closest?.('#retry-runtime');
          if(button){const log=JSON.parse(sessionStorage.getItem('__KEY__')||'[]');
            log.push({sequence:log.length+1,trusted:event.isTrusted,id:button.id,
              title:document.querySelector('#recovery-title')?.textContent,path:location.pathname,
              clicked_at:new Date().toISOString()});sessionStorage.setItem('__KEY__',JSON.stringify(log));}
        },true);""".replace('__KEY__',TEST_KEY))
        page=context.new_page()
        http,requests,errors,failures,request_rows=[],[],[],[],{}
        phase='fault'
        evidence=case_dir
        transition=None

        def requested(request):
            row={'sequence':len(requests)+1,'phase':phase,'method':request.method,'url':request.url,
                'request_body_sha256':hashlib.sha256((request.post_data or '').encode()).hexdigest(),
                'started_at':now(),'response_at':None,'finished_at':None,'failed_at':None,
                'status':None,'failure':None,'http_file':None}
            requests.append(row)
            request_rows[request]=row

        def failed(request):
            row=request_rows[request]
            row.update(failed_at=now(),failure=request.failure)
            failures.append(dict(row))

        def record(response):
            raw=request_rows[response.request]
            raw.update(response_at=now(),status=response.status)
            if '/api/' not in response.url:
                return
            value={'phase':raw['phase'],'request_sequence':raw['sequence'],**_http_record(response)}
            name=f'http_{len(http)+1:03d}.json'
            raw['http_file']=name
            _write_json(evidence/name,value)
            http.append({'file':name,'phase':raw['phase'],'request_sequence':raw['sequence'],
                'method':value['method'],'url':value['url'],'status':value['response_status_code']})

        page.on('pageerror',lambda error:errors.append(str(error)))
        page.on('console',lambda message:errors.append(message.text) if message.type=='error' else None)
        page.on('request',requested)
        page.on('requestfailed',failed)
        page.on('requestfinished',lambda request:request_rows[request].update(finished_at=now()))
        page.on('response',record)

        def capture(name,run_id):
            with patch.dict(os.environ,{'PRODUCT_E2E_RUN_ID':run_id+'_recovered'}):
                _capture_ready(page,evidence/'recovered'/f'{name}.png')
            return _map_ack(page)

        try:
            for plan_context in ('original','accepted_child'):
                evidence=case_dir/plan_context
                evidence.mkdir()
                _write_json(evidence/'private_fixture_inputs.json',copied)
                http,requests,errors,failures,request_rows=[],[],[],[],{}
                run_id=root_run_id+'_'+plan_context
                path,before=install_fault(repository,evidence,case_name)
                phase='fault'
                try:
                    with fault_generation(repository,state,evidence,BASE_URL,run_id,case_name) as fault:
                        page.goto(BASE_URL+'/app/itinerary',wait_until='domcontentloaded')
                        title='The workspace is not ready' if case_name=='core_not_ready' else 'Copilot could not start'
                        expect(page.locator('#recovery-title')).to_have_text(title,timeout=30_000)
                        expect(page.locator('#retry-runtime')).to_be_enabled()
                        expect(page.locator('#app-shell')).to_be_hidden()
                        page.wait_for_load_state('networkidle')
                        blocked=page.evaluate("""()=>({title:document.querySelector('#recovery-title').textContent,
                          html:document.querySelector('#loading-screen').innerHTML,app_hidden:document.querySelector('#app-shell').hidden,
                          ready_maps:document.querySelectorAll('[data-map-ready="true"]').length,
                          rendered_ack_count:document.querySelectorAll('[data-rendered-map-plan-id]').length})""")
                        credentials=session_identity(page)
                        assert credentials['present'] is (plan_context=='accepted_child')
                        if transition:
                            assert credentials==transition['session']
                        assert blocked['ready_maps']==blocked['rendered_ack_count']==0
                        assert not any(row['method']!='GET' for row in requests)
                        observer_before=json.loads(page.evaluate('key=>sessionStorage.getItem(key)||"[]"',TEST_KEY))
                        _write_json(evidence/'blocked_dom.json',{'case':case_name,'context':plan_context,**blocked,
                            'session_present':credentials['present'],'session':credentials,'observed_at':now(),
                            'screenshot_taken':False,'fault_health':fault,'observer_before':observer_before})
                    restore_fault(path,before,evidence)
                    phase='recovered'
                    with guarded_generation(repository,state,evidence,BASE_URL,run_id,'recovered'):
                        with page.expect_navigation(wait_until='domcontentloaded'):
                            page.locator('#retry-runtime').click()
                        parent=capture('recovered_parent',run_id)
                        assert parent['preview_state']=='PARENT_RENDERED'
                        expect(page.locator('#retry-runtime')).to_have_count(0)
                        observer_after=json.loads(page.evaluate('key=>sessionStorage.getItem(key)||"[]"',TEST_KEY))
                        assert len(observer_after)==len(observer_before)+1 and observer_after[:-1]==observer_before
                        click=observer_after[-1]
                        assert click['trusted'] is True and click['id']=='retry-runtime' and click['title']==title
                        current_session=session_identity(page)
                        if transition:
                            assert parent==transition['accepted_map'] and current_session==transition['session']
                        _write_json(evidence/'retry_observer.json',{'before':observer_before,'after':observer_after})
                        _write_json(evidence/'browser_assertions.json',{'schema_version':'bootstrap-recovery-browser-v1',
                            'case':case_name,'context':plan_context,'parent':parent,'retry':click,'session':current_session,
                            'accepted_plan_changed':False,'assertions_passed':True,'provider_calls':0,
                            'blocked_screenshot_taken':False,'frontend_fetch_mocked':False,'accepted_transition':transition})
                        if plan_context=='original':
                            phase='accept_setup'
                            with page.expect_response(lambda response:response.url.endswith('/selection') and response.request.post_data_json.get('selected_day')==5):
                                page.locator('.day-card[data-day="5"]').click()
                            expect(page.locator('#map-context-title')).to_contain_text('Bixby Creek Bridge')
                            page.locator('#map-edit-button').click()
                            page.locator('#inspect-edit-stop').click()
                            page.locator('.legacy-edit-controls summary').click()
                            page.locator('#registered-candidate').select_option('pfeiffer_big_sur_state_park')
                            with page.expect_response(lambda response:response.url.endswith('/draft/operations')) as staged:
                                page.locator('#draft-replace').click()
                            assert staged.value.status==200
                            page.locator('#close-map-edit').click()
                            with page.expect_response(lambda response:response.url.endswith('/preview'),timeout=90_000) as previewed:
                                page.locator('#inline-preview').click()
                            assert previewed.value.status==200
                            proposal=previewed.value.json()['proposal']
                            assert proposal['state']=='eligible' and proposal['parent_plan_id']==parent['active_plan_id']
                            expect(page.locator('.map-stage')).to_have_attribute('data-preview-state','PREVIEW_RENDERED',timeout=40_000)
                            preview_map=capture('evaluated_setup',run_id)
                            with page.expect_response(lambda response:response.url.endswith('/accept'),timeout=90_000) as accepted:
                                page.locator('#accept-repair').click()
                                expect(page.get_by_role("dialog", name="Confirm trip decision", exact=True)).to_be_visible()
                                expect(page.locator("#acceptance-message")).to_have_text('Accept this independently evaluated repair as the durable trip plan?')
                                page.get_by_role("dialog", name="Confirm trip decision", exact=True).get_by_role("button", name='Accept repair', exact=True).click()
                            assert accepted.value.status==200
                            expect(page.locator('.map-stage')).to_have_attribute('data-preview-state','PARENT_RENDERED',timeout=30_000)
                            accepted_map=capture('accepted_setup',run_id)
                            assert accepted_map['active_plan_id']==proposal['child_plan_id']
                            transition={'parent':parent,'proposal':proposal,'preview_map':preview_map,'accepted_map':accepted_map,
                                'accept_response':accepted.value.json(),'session':session_identity(page)}
                            _write_json(evidence/'accepted_transition.json',transition)
                        page.wait_for_load_state('networkidle',timeout=30_000)
                        assert errors==failures==[]
                        assert all(row['finished_at'] and row['status'] and row['status']<400 for row in requests)
                        # Disconnect the page while its own server is still healthy; no background request is cut off.
                        page.goto('about:blank')
                finally:
                    _write_json(evidence/'http_index.json',http)
                    _write_json(evidence/'browser_requests.json',requests)
                    _write_json(evidence/'browser_errors.json',{'errors':errors,'request_failures':failures})
                    _write_json(evidence/'llm_io.json',{'provider_call_count':0,'llm_input':None,'llm_output':None,
                        'reason':'Actual bootstrap Retry plus explicit deterministic accepted-baseline setup; no chat/provider dispatch.'})
        finally:
            browser.close()
