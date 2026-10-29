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


for(const action of ["selection","operations","undo"]){
  for(const reason of ["undelivered","rejected"]){
    test(`${action}: cancel settles ${reason} request without applying it`,async()=>{
      const f=fixture(),r=f.create();
      f.setFault({before:()=>{throw reason==="rejected"?Object.assign(new Error("Unsupported edit"),{status:422,code:"unsupported_draft_operation"}):new TypeError("Not delivered");}});
      await assert.rejects(r.submit(action,bodyFor(action)));const record=r.pending;
      await assert.rejects(r.check(),/not confirmed yet/);assert.equal(r.blocked,true);
      const before=copy(f.current.draft),value=await r.cancel();
      assert.equal(value.recovery_outcome,"not_applied");assert.equal(value.session.revision,8);
      assert.deepEqual(value.session.draft,before);assert.equal(r.blocked,false);assert.equal(f.values.size,0);
      const cancellations=f.calls.filter(c=>c.path.endsWith("/cancel"));
      assert.equal(cancellations.length,1);assert.deepEqual(cancellations[0].options.body,{action,request:JSON.parse(record.body)});
      assert.equal(f.calls.filter(c=>c.options.method==="POST"&&!c.path.endsWith("/cancel")).length,1);
      assert.equal(f.current.accepted_plan_id,"parent");
    });
  }
  test(`${action}: cancel checks an applied request and never reverses it`,async()=>{
    const f=fixture(),r=f.create();f.setFault({after:()=>{throw new TypeError("Lost saved reply");}});
    await assert.rejects(r.submit(action,bodyFor(action)));const expected=copy(f.current);
    const value=await r.cancel();assert.equal(value.recovery_outcome,"applied");assert.deepEqual(value.session,expected);
    assert.equal(f.calls.filter(c=>c.path.endsWith("/cancel")).length,0);assert.equal(r.blocked,false);
  });
}

test("lost cancellation reply survives reload and is settled by reads",async()=>{
  const f=fixture(),r=f.create();f.setFault({before:()=>{throw new TypeError("Not delivered");}});
  await assert.rejects(r.submit("operations",bodyFor("operations")));
  f.setFault({cancelAfter:()=>{throw new TypeError("Cancellation reply lost");}});
  await assert.rejects(r.cancel(),/Cancellation reply lost/);assert.equal(r.blocked,true);assert.equal(f.current.revision,8);
  const restored=f.create();await restored.restore();assert.equal((await restored.check()).recovery_outcome,"not_applied");
  assert.equal(restored.blocked,false);assert.equal(f.calls.filter(c=>c.options.method==="POST").length,2);
});

test("failed cancellation before persistence retains uncertainty and permits exact retry",async()=>{
  const f=fixture(),r=f.create();f.setFault({before:()=>{throw new TypeError("Not delivered");}});
  await assert.rejects(r.submit("undo",bodyFor("undo")));
  f.setFault({cancelBefore:()=>{throw new TypeError("Cancellation not delivered");}});
  await assert.rejects(r.cancel());assert.equal(f.current.revision,7);assert.equal(r.blocked,true);
  await assert.rejects(r.check(),/not confirmed yet/);f.setFault(null);
  assert.equal((await r.cancel()).recovery_outcome,"not_applied");assert.equal(f.current.revision,8);
});

test("original commit after the first unseen read is reported applied",async()=>{
  const f=fixture(),r=f.create();f.setFault({before:()=>{throw new TypeError("Not delivered");}});
  await assert.rejects(r.submit("operations",bodyFor("operations")));const record=r.pending;
  f.setFault({cancelBefore:()=>{
    f.current.revision=8;f.current.draft=[{operation_id:"op_one"}];
    f.durable.set(record.mutation_id,{mutation_receipt:{mutation_id:record.mutation_id,action:record.action,
      request_sha256:"a".repeat(64),expected_revision:7,committed_revision:8},result:{operation:{operation_id:"op_one"}}});
  }});
  const value=await r.cancel();assert.equal(value.recovery_outcome,"applied");assert.equal(value.session.draft.length,1);
  assert.equal(f.current.revision,8);assert.equal(r.blocked,false);
});

for(const change of [v=>{v.mutation_id="wrong";},v=>{v.action="selection";},v=>{v.resolved_revision=7;},
  v=>{v.resolved_revision=true;},v=>{v.state="unseen";},v=>{v.result={};},v=>{v.expected_revision=8;}]){
  test(`incompatible cancellation proof retains journal: ${change.toString()}`,async()=>{
    const f=fixture(),r=f.create();f.setFault({before:()=>{throw new TypeError("Not delivered");}});
    await assert.rejects(r.submit("undo",bodyFor("undo")));const raw=f.values.get("pending");
    f.setFault({cancelAfter:(_options,value)=>{change(value);return value;}});
    await assert.rejects(r.cancel(),/safely resolved/);assert.equal(r.blocked,true);assert.equal(f.values.get("pending"),raw);
    f.setFault(null);assert.equal((await r.check()).recovery_outcome,"not_applied");
  });
}

test("cancellation proof cannot clear a journal using a workspace older than its fence",async()=>{
  const f=fixture(),r=f.create();f.setFault({before:()=>{throw new TypeError("Not delivered");}});
  await assert.rejects(r.submit("undo",bodyFor("undo")));
  f.setReads(path=>path.includes("/mutations/")
    ?{schema_version:"product-mutation-status-v1",state:"unseen",mutation_id:r.pending.mutation_id,mutation_receipt:null,result:null}
    :{...f.workspace(),session:{...f.visible}});
  await assert.rejects(r.cancel(),/current workspace is not yet available/);assert.equal(r.blocked,true);
});
