import assert from "node:assert/strict";
import test from "node:test";
import {createMutationRecovery} from "../../src/itinerary_system/product_app/static/js/mutation-recovery.js";

const flush = async () => { for (let i=0;i<20;i++) await Promise.resolve(); };
const deferred = () => { let resolve;const promise=new Promise(yes=>{resolve=yes;});return {promise,resolve}; };
const copy = value => structuredClone(value);

function fixture() {
  let current={session_id:"session_one",trip_id:"trip",run_id:"run",accepted_plan_id:"parent",revision:7,draft:[]};
  let visible=copy(current), fault=null, reads=null;
  const durable=new Map(),calls=[],presentations=[],values=new Map();
  const storage={getItem:key=>values.get(key)||null,setItem:(key,value)=>values.set(key,value),removeItem:key=>values.delete(key)};
  const workspace=()=>({session:copy(current),workspace:{accepted_plan_pointer:{accepted_plan_id:current.accepted_plan_id}},registry:{run_id:"run"}});
  const api=async (path,options={})=>{
    calls.push({path,options:copy({...options,signal:undefined})});
    if(reads && options.method!=="POST") return reads(path,options);
    if(options.method!=="POST"){
      if(path.includes("/mutations/")){
        const id=path.split("/").at(-1),proof=durable.get(id);
        return {schema_version:"product-mutation-status-v1",state:proof?"applied":"unseen",mutation_id:id,
          mutation_receipt:proof?.mutation_receipt||null,result:proof?.result||null};
      }
      return workspace();
    }
    if(path.endsWith("/cancel")){
      assert.ok(values.get("pending"));
      const {action,request}=options.body;
      if(fault?.cancelBefore)await fault.cancelBefore(options);
      const proof=durable.get(request.mutation_id);
      if(!proof && current.revision===request.expected_revision)current.revision++;
      const value={schema_version:"product-mutation-resolution-v1",state:proof?"applied":"not_applied",
        mutation_id:request.mutation_id,action,expected_revision:request.expected_revision,resolved_revision:current.revision,
        mutation_receipt:proof?.mutation_receipt||null,result:proof?.result||null};
      if(fault?.cancelAfter)return fault.cancelAfter(options,value);
      return value;
    }
    assert.ok(values.get("pending"),"The durable journal must precede dispatch");
    const body=copy(options.body),action=path.endsWith("/selection")?"selection":path.split("/").at(-1);
    if(fault?.before) await fault.before(options);
    let proof=durable.get(body.mutation_id);
    if(!proof){
      if(body.expected_revision!==current.revision) throw Object.assign(new Error("Stale revision"),{status:409,code:"stale_session_revision"});
      current.revision++;
      if(action==="operations") current.draft.push({operation_id:"op_one",...body});
      if(action==="undo") current.draft.pop();
      if(action==="selection") current.selected_day=body.selected_day;
      proof={mutation_receipt:{mutation_id:body.mutation_id,action,request_sha256:"a".repeat(64),
        expected_revision:body.expected_revision,committed_revision:current.revision},result:{operation:{operation_id:"op_one"}}};
      durable.set(body.mutation_id,proof);
    }
    if(fault?.after) return fault.after(options,proof);
    return {...copy(proof.result),session:copy(current),mutation_receipt:copy(proof.mutation_receipt)};
  };
  const create=()=>createMutationRecovery({api,storage,key:"pending",getSession:()=>visible,onChange:value=>presentations.push(value)});
  return {api,storage,values,calls,presentations,durable,workspace,create,
    setFault:value=>{fault=value;},setReads:value=>{reads=value;},
    get current(){return current;},get visible(){return visible;},setVisible:value=>{visible=copy(value);}};
}
const bodyFor=action=>({expected_revision:7,...(action==="selection"?{selected_day:4}:action==="operations"?{operations:[{type:"set_stop_duration",target:"stop_one",parameters:{duration:{mode:"exact",minutes:75}}}]}:{})});



async function damaged(f) {
  const r=f.create();f.values.set("pending","{");await assert.rejects(r.restore());return r;
}
const posts=f=>f.calls.filter(c=>c.options.method==="POST");

test("protected refresh retains unreadable bytes until a verified fence and preserves the draft",async()=>{
 const f=fixture(),r=await damaged(f),original=copy(f.current);
 f.setFault({cancelBefore:()=>{const saved=JSON.parse(f.values.get("pending"));assert.equal(saved.unreadable_record,"{");assert.equal(saved.action,"recovery");assert.equal(r.blocked,true);}});
 const value=await r.repair();assert.equal(value.recovery_outcome,"refreshed");assert.equal(value.session.revision,8);
 assert.deepEqual({...value.session,revision:7},original);assert.equal(r.blocked,false);assert.equal(f.values.size,0);
 assert.equal(posts(f).length,1);assert.ok(posts(f)[0].path.endsWith("/cancel"));assert.equal(posts(f)[0].options.body.action,"undo");

 assert.deepEqual(f.current.draft,original.draft);
});

for(const raw of ['{"schema_version":"bad"}','null','[]'])test(`incompatible stored envelope can be refreshed: ${raw}`,async()=>{
 const f=fixture(),r=f.create();f.values.set('pending',raw);await assert.rejects(r.restore());
 assert.equal((await r.repair()).recovery_outcome,'refreshed');assert.equal(r.blocked,false);assert.equal(f.current.revision,8);
});

test("already saved batch is preserved without reporting the lost original outcome",async()=>{
 const f=fixture();f.current.revision=9;f.current.draft=[{operation_id:'one'},{operation_id:'two'}];
 const r=await damaged(f),before=copy(f.current.draft);const value=await r.repair();
 assert.equal(value.recovery_outcome,'refreshed');assert.equal(value.session.revision,10);assert.deepEqual(value.session.draft,before);assert.equal(f.current.accepted_plan_id,'parent');
});

test("lost fence reply survives reload and settles through reads without duplicate writes",async()=>{
 const f=fixture(),r=await damaged(f);f.setFault({cancelAfter:()=>{throw new TypeError('Reply lost');}});
 await assert.rejects(r.repair(),/Reply lost/);assert.equal(f.current.revision,8);const raw=f.values.get('pending');
 assert.equal(JSON.parse(raw).unreadable_record,'{');const restored=f.create();await restored.restore();
 const count=posts(f).length;assert.equal((await restored.check()).recovery_outcome,'refreshed');assert.equal(posts(f).length,count);assert.equal(restored.blocked,false);
});

test("failed fence dispatch retries only the same cancellation identity and never sends Undo",async()=>{
 const f=fixture(),r=await damaged(f);f.setFault({cancelBefore:()=>{throw Error('Not delivered');}});
 await assert.rejects(r.repair(),/Not delivered/);const raw=f.values.get('pending');assert.equal(f.current.revision,7);
 const restored=f.create();await restored.restore();f.setFault(null);assert.equal((await restored.retry()).recovery_outcome,'refreshed');
 assert.equal(posts(f).length,2);assert.deepEqual(posts(f)[0].options.body,posts(f)[1].options.body);
 assert.ok(posts(f).every(c=>c.path.endsWith('/cancel')));assert.equal(JSON.parse(raw).unreadable_record,'{');
});

test("unavailable storage sends nothing and can recover when access returns",async()=>{
 const f=fixture(),r=f.create(),get=f.storage.getItem;f.storage.getItem=()=>{throw Error('Denied');};
 await assert.rejects(r.restore());await assert.rejects(r.repair(),{code:'mutation_journal_unavailable'});assert.equal(f.calls.length,0);assert.equal(r.blocked,true);
 f.storage.getItem=get;assert.equal((await r.repair()).recovery_outcome,'refreshed');assert.equal(r.blocked,false);
});

test("quota failure preserves original bytes and never dispatches a fence",async()=>{
 const f=fixture(),r=await damaged(f);f.storage.setItem=()=>{throw Error('Quota');};
 await assert.rejects(r.repair(),{code:'mutation_journal_unavailable'});assert.equal(posts(f).length,0);assert.equal(f.values.get('pending'),'{');assert.equal(r.blocked,true);
});

test("failure to clear the resolved journal remains recoverable after storage returns",async()=>{
 const f=fixture(),r=await damaged(f),remove=f.storage.removeItem;f.storage.removeItem=()=>{throw Error('Denied');};
 await assert.rejects(r.repair(),/Denied/);assert.equal(f.current.revision,8);assert.equal(r.blocked,true);assert.equal(JSON.parse(f.values.get('pending')).unreadable_record,'{');
 f.storage.removeItem=remove;assert.equal((await r.check()).recovery_outcome,'refreshed');assert.equal(r.blocked,false);
});

test("workspace older than the fence cannot clear the recovery journal",async()=>{
 const f=fixture(),r=await damaged(f);f.setFault({cancelAfter:(_options,value)=>{f.setReads(path=>path.includes('/mutations/')?{schema_version:'product-mutation-status-v1',state:'unseen',mutation_id:r.pending.mutation_id,mutation_receipt:null,result:null}:{...f.workspace(),session:{...f.visible}});return value;}});
 await assert.rejects(r.repair(),/current workspace is not yet available/);assert.equal(r.blocked,true);assert.ok(f.values.get('pending'));
 f.setReads(null);assert.equal((await r.check()).recovery_outcome,'refreshed');
});

test("a journal changed during the initial read is retained without a fence",async()=>{
 const f=fixture(),r=await damaged(f);f.setReads(()=>{f.values.set('pending','changed');return f.workspace();});
 await assert.rejects(r.repair(),{code:'mutation_journal_changed'});assert.equal(posts(f).length,0);assert.equal(f.values.get('pending'),'changed');assert.equal(r.blocked,true);
});

test("parallel protected refresh is rejected while the first fence remains active",async()=>{
 const f=fixture(),r=await damaged(f),hold=deferred();f.setFault({cancelBefore:()=>hold.promise});
 const first=r.repair();await flush();await assert.rejects(r.repair(),{code:'mutation_recovery_busy'});hold.resolve();await first;
 assert.equal(posts(f).length,1);assert.equal(r.blocked,false);
});

test("past and future revisions for every submitted action are refused before journaling",async()=>{
 for(const action of ['selection','operations','undo'])for(const expected_revision of [6,8]){
  const f=fixture(),r=f.create();await assert.rejects(r.submit(action,{...bodyFor(action),expected_revision}),{code:'invalid_mutation_request'});
  assert.equal(f.calls.length,0);assert.equal(f.values.size,0);
 }
});
