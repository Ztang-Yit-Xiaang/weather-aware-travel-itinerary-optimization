"""Explicit private-copy bootstrap faults; never relax the normal launch guard."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'scripts'))
from e2e1_provider_isolation import ProviderIsolation, deterministic_environment  # noqa: E402

SCHEMA='e2e1-bootstrap-fault-isolation-v1'
CASES=('core_not_ready','invalid_health')
FAULT_HEALTH='bootstrap-intentionally-invalid-health-v1'
HEALTH_ORIGINAL='        return runtime.health().as_dict()'
HEALTH_FAULT='        return {**runtime.health().as_dict(), "schema_version": "'+FAULT_HEALTH+'"}'


def write(path,value):
    path.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n',encoding='utf-8')


def install_fault(repository,evidence,case):
    relative='configs/product_app_registry.json' if case=='core_not_ready' else 'src/itinerary_system/product_app/api.py'
    path=repository/relative
    before=path.read_bytes()
    if case=='core_not_ready':
        value=json.loads(before)
        value['schema_version']='bootstrap-intentionally-invalid-registry-v1'
        after=(json.dumps(value,indent=2,sort_keys=True)+'\n').encode()
    else:
        assert case=='invalid_health' and before.decode().count(HEALTH_ORIGINAL)==1
        after=before.decode().replace(HEALTH_ORIGINAL,HEALTH_FAULT).encode()
    path.write_bytes(after)
    write(evidence/'fault_source_change.json',{'schema_version':'bootstrap-private-source-fault-v1','case':case,
        'path':relative,'before_utf8':before.decode(),'fault_utf8':after.decode(),
        'before_sha256':hashlib.sha256(before).hexdigest(),'fault_sha256':hashlib.sha256(after).hexdigest(),
        'private_copy_only':True,'frontend_fetch_mocked':False,'created_at':datetime.now(UTC).isoformat()})
    return path,before


def restore_fault(path,before,evidence):
    fault=path.read_bytes()
    path.write_bytes(before)
    write(evidence/'fault_source_restored.json',{'fault_sha256':hashlib.sha256(fault).hexdigest(),
        'restored_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'restored_at':datetime.now(UTC).isoformat()})


@contextmanager
def fault_generation(repository,state,evidence,base_url,run_id,case):
    from urllib.parse import urlsplit

    from run_e2e1_vertical_slice import _port_is_available
    port=urlsplit(base_url).port
    assert _port_is_available(port)
    directory=evidence/'fault'
    directory.mkdir()
    command=[sys.executable,str(Path(__file__).resolve()),'--repository',str(repository),'--state',str(state),
        '--port',str(port),'--output',str(directory/'provider_isolation.json'),'--run-id',run_id+'_fault','--case',case]
    env=deterministic_environment(os.environ)
    env.update(PRODUCT_APP_ORIGIN=base_url,PYTHONUTF8='1',PYTHONDONTWRITEBYTECODE='1')
    write(directory/'server_command.json',{'command':command,'cwd_scope':'private_product_copy','run_id':run_id+'_fault','case':case})
    started=datetime.now(UTC).isoformat()
    with (directory/'server.stdout.log').open('w',encoding='utf-8') as stdout,(directory/'server.stderr.log').open('w',encoding='utf-8') as stderr:
        process=subprocess.Popen(command,cwd=repository,env=env,stdout=stdout,stderr=stderr,stdin=subprocess.PIPE,text=True)
        observed_at=None
        try:
            deadline=time.monotonic()+60
            while time.monotonic()<deadline:
                assert process.poll() is None,'Bootstrap fault process exited before real HTTP was observable'
                try:
                    with urlopen(base_url+'/api/health',timeout=2) as response:
                        raw=response.read().decode()
                        health=json.loads(raw)
                        assert response.status==200
                    break
                except (URLError,TimeoutError):
                    time.sleep(.1)
            else:
                raise AssertionError('Bootstrap fault endpoint never became observable')
            assert (health['schema_version']=='product-health-v2' and health['core_ready'] is False) if case=='core_not_ready' else health['schema_version']==FAULT_HEALTH
            observed_at=datetime.now(UTC).isoformat()
            write(directory/'health_probe.json',{'method':'GET','url':base_url+'/api/health','status':200,'response_body_utf8':raw,
                'response_body_sha256':hashlib.sha256(raw.encode()).hexdigest(),'observed_at':observed_at})
            yield health
        finally:
            if process.poll() is None:
                try:
                    process.communicate('stop\n',timeout=20)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            receipt=directory/'provider_isolation.json'
            write(directory/'server_lifecycle.json',{'run_id':run_id+'_fault','pid':process.pid,'case':case,
                'process_started_at':started,'fault_observed_at':observed_at,'process_stopped_at':datetime.now(UTC).isoformat(),
                'exit_code':process.returncode,'receipt_sha256':hashlib.sha256(receipt.read_bytes()).hexdigest() if receipt.exists() else None})
            assert process.returncode==0,'Fault server must still finalize network isolation and stop normally'


def main():
    parser=argparse.ArgumentParser()
    for name in ('repository','state','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--port',type=int,required=True)
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--case',choices=CASES,required=True)
    args=parser.parse_args()
    os.environ.update(deterministic_environment(os.environ))
    sys.path.insert(0,str(args.repository/'src'))
    with ProviderIsolation(args.output,args.run_id) as guard:
        import uvicorn

        from itinerary_system.product_app.api import create_product_app
        app=create_product_app(repository_root=args.repository,state_root=args.state,application_port=args.port)
        runtime=app.state.product_runtime
        health=runtime.health()
        if args.case=='core_not_ready':
            assert health.core_ready is False and runtime.service is None
            assert runtime.config.copilot_adapter=='deterministic' and os.environ['OPENAI_API_KEY']==''
            guard.record.update(api_key_disabled=True,app_initialized=True,effective_adapter='unavailable_core_not_ready',actual_adapter=None)
        else:
            guard.check_adapter(runtime)
            assert health.core_ready is True
        guard.record.update(schema_version=SCHEMA,case=args.case,actual_service_available=runtime.service is not None,
            core_ready=health.core_ready,fixture_scope='private_copy_bootstrap_fault',normal_guard_relaxed=False)
        guard._write()
        server=uvicorn.Server(uvicorn.Config(app,host='127.0.0.1',port=args.port,log_level='info'))
        thread=threading.Thread(target=server.run,daemon=True)
        thread.start()
        try:
            assert sys.stdin.readline().strip()=='stop'
        finally:
            guard.record['server_started']=server.started
            server.should_exit=True
            thread.join(timeout=15)
            guard.record['server_stopped']=not thread.is_alive()
        assert server.started and not thread.is_alive()
    return 0


if __name__=='__main__':
    raise SystemExit(main())
