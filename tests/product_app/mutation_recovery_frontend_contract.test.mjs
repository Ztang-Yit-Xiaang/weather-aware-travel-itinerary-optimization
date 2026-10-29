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
  test(`${action}: ordinary keyed confirmation installs its validated session without an extra read`,async()=>{
    const f=fixture(),r=f.create();
    const value=await r.submit(action,bodyFor(action));
    assert.equal(value.recovery_outcome,"applied");assert.equal(value.session.revision,8);
    assert.equal(r.blocked,false);assert.equal(f.values.size,0);
    assert.equal(f.calls.filter(c=>c.options.method==="POST").length,1);
    assert.equal(f.calls.length,1);
  });
  test(`${action}: lost committed reply survives reload and checks without another write`,async()=>{
    const f=fixture(),first=f.create();f.setFault({after:()=>{throw new TypeError("Reply lost");}});
    await assert.rejects(first.submit(action,bodyFor(action)),/Reply lost/);
    assert.equal(first.blocked,true);const raw=f.values.get("pending");
    const restored=f.create();await restored.restore();assert.equal(f.values.get("pending"),raw);
    f.current.revision=10;f.current.draft=[];
    const value=await restored.check();assert.equal(value.session.revision,10);assert.deepEqual(value.session.draft,[]);
    assert.equal(value.mutation_receipt.committed_revision,8);assert.equal(restored.blocked,false);
    assert.equal(f.calls.filter(c=>c.options.method==="POST").length,1);
  });
  test(`${action}: explicit retry reuses exact ID body and revision`,async()=>{
    const f=fixture(),r=f.create();f.setFault({after:()=>{throw new TypeError("Reply lost");}});
    await assert.rejects(r.submit(action,bodyFor(action)));const before=copy(f.calls[0]);
    f.setFault(null);const value=await r.retry();
    const posts=f.calls.filter(c=>c.options.method==="POST");
    assert.equal(JSON.stringify(posts[1]),JSON.stringify(before));assert.equal(value.session.revision,8);assert.equal(f.durable.size,1);
  });
}

test("unseen at the same revision never clears uncertainty or sends another write",async()=>{
  const f=fixture(),r=f.create();f.setFault({before:()=>{throw new TypeError("Request not delivered");}});
  await assert.rejects(r.submit("undo",bodyFor("undo")));
  await assert.rejects(r.check(),/not confirmed yet/);assert.equal(r.blocked,true);
  assert.equal(f.calls.filter(c=>c.options.method==="POST").length,1);
  await assert.rejects(r.submit("undo",bodyFor("undo")),/Resolve/);
});

for(const code of ["invalid_selected_day","invalid_selected_stop","invalid_selected_segment","invalid_selected_candidate","invalid_selected_alternative"]){
  test(`${code}: a definite initial selection rejection does not lock the workspace`,async()=>{
    const f=fixture(),r=f.create();
    f.setFault({before:()=>{throw Object.assign(new Error(code),{status:code==="invalid_selected_candidate"?422:400,code});}});
    await assert.rejects(r.submit("selection",bodyFor("selection")),new RegExp(code));
    assert.equal(r.blocked,false);assert.equal(f.values.size,0);assert.equal(f.current.revision,7);
    f.setFault(null);
    assert.equal((await r.submit("selection",bodyFor("selection"))).session.revision,8);
  });
}

test("a rejected selection retry cannot clear an earlier uncertain attempt",async()=>{
  const f=fixture(),r=f.create();
  f.setFault({before:()=>{throw new TypeError("Reply unavailable");}});
  await assert.rejects(r.submit("selection",bodyFor("selection")));
  const pending=f.values.get("pending");
  f.setFault({before:()=>{throw Object.assign(new Error("invalid_selected_segment"),{status:400,code:"invalid_selected_segment"});}});
  await assert.rejects(r.retry(),/invalid_selected_segment/);
  assert.equal(r.blocked,true);assert.equal(f.values.get("pending"),pending);
});

test("a later revision followed by an unseen read settles an unapplied old request",async()=>{
  const f=fixture(),r=f.create();f.setFault({before:()=>{throw new TypeError("Request not delivered");}});
  await assert.rejects(r.submit("undo",bodyFor("undo")));f.current.revision=8;
  const value=await r.check();assert.equal(value.recovery_outcome,"not_applied");assert.equal(r.blocked,false);
  assert.equal(f.calls.filter(c=>c.path.includes("/mutations/")).length,2);
  assert.equal(f.calls.filter(c=>c.options.method==="POST").length,1);
});

test("an in-flight commit between unseen and refresh is recovered without another write",async()=>{
  const f=fixture(),r=f.create();f.setFault({before:()=>{throw new TypeError("Request not delivered");}});
  await assert.rejects(r.submit("selection",bodyFor("selection")));
  const record=r.pending;let statuses=0,workspaces=0;
  const proof={mutation_id:record.mutation_id,action:"selection",request_sha256:"a".repeat(64),expected_revision:7,committed_revision:8};
  f.setReads(path=>{
    if(path.includes("/mutations/")) {statuses++;return {schema_version:"product-mutation-status-v1",state:statuses===1?"unseen":"applied",mutation_id:record.mutation_id,mutation_receipt:statuses===1?null:proof,result:statuses===1?null:{}};}
    workspaces++;if(workspaces===2)f.current.revision=8;return f.workspace();
  });
  assert.equal((await r.check()).session.revision,8);assert.equal(workspaces,2);assert.equal(r.blocked,false);
});

test("deadline covers a response that never completes, retains journal and ignores its late completion",async()=>{
  const originalSet=globalThis.setTimeout,originalClear=globalThis.clearTimeout,timers=new Map();let id=0;
  globalThis.setTimeout=(fn,delay)=>{timers.set(++id,{fn,delay});return id;};globalThis.clearTimeout=id=>timers.delete(id);
  try{
    const f=fixture(),r=f.create(),late=deferred();let signal;
    f.setFault({after:options=>{signal=options.signal;return late.promise;}});
    const pending=r.submit("selection",bodyFor("selection"));await flush();
    assert.equal(timers.size,1);const timer=[...timers.values()][0];assert.equal(timer.delay,20000);timer.fn();
    await assert.rejects(pending,/did not finish/);assert.equal(signal.aborted,true);assert.equal(r.blocked,true);assert.equal(timers.size,0);
    f.setFault(null);await r.retry();const after=copy(f.visible);
    late.resolve({unexpected:"late first reply"});await flush();
    assert.equal(r.blocked,false);assert.deepEqual(f.visible,after);assert.equal(f.current.revision,8);assert.equal(f.values.size,0);
  }finally{globalThis.setTimeout=originalSet;globalThis.clearTimeout=originalClear;}
});

test("storage failure before dispatch sends nothing",async()=>{
  const f=fixture(),r=f.create();f.storage.setItem=()=>{throw new Error("Quota");};
  await assert.rejects(r.submit("undo",bodyFor("undo")),/Nothing was sent/);assert.equal(f.calls.length,0);assert.equal(r.blocked,false);
});

test("failure to clear the saved record retains the request for recovery",async()=>{
  const f=fixture(),r=f.create();f.storage.removeItem=()=>{throw new Error("Storage unavailable");};
  await assert.rejects(r.submit("undo",bodyFor("undo")),/Storage unavailable/);assert.equal(r.blocked,true);assert.equal(f.current.revision,8);
});

for(const mutate of [value=>{value.schema_version="bad";},value=>{value.expected_revision=false;},value=>{value.mutation_id="bad";},value=>{value.body="{}";}]){
  test(`corrupt durable record stays blocked: ${mutate.toString()}`,async()=>{
    const f=fixture(),r=f.create();f.setFault({before:()=>{throw new Error("Lost");}});await assert.rejects(r.submit("undo",bodyFor("undo")));
    const raw=JSON.parse(f.values.get("pending"));mutate(raw);f.values.set("pending",JSON.stringify(raw));
    const restored=f.create();await assert.rejects(restored.restore());assert.equal(restored.blocked,true);
    await assert.rejects(restored.retry());assert.equal(f.calls.length,1);
  });
}

test("pending request cannot be replayed into another workspace",async()=>{
  const f=fixture(),r=f.create();f.setFault({before:()=>{throw new Error("Lost");}});await assert.rejects(r.submit("undo",bodyFor("undo")));
  f.setVisible({...f.visible,session_id:"session_other"});await assert.rejects(r.retry(),/different workspace/);assert.equal(r.blocked,true);assert.equal(f.calls.length,1);
});

for(const change of [proof=>{proof.action="selection";},proof=>{proof.committed_revision=99;},proof=>{proof.mutation_id="mutation_"+"0".repeat(32);}]){
  test(`mismatched confirmation does not clear the request: ${change.toString()}`,async()=>{
    const f=fixture(),r=f.create();f.setFault({after:(_options,proof)=>{const result=copy(proof);change(result.mutation_receipt);return result;}});
    await assert.rejects(r.submit("undo",bodyFor("undo")),/did not match/);assert.equal(r.blocked,true);assert.equal(f.values.size,1);
  });
}

test("a foreign direct confirmation cannot replace the journal's pending result",async()=>{
  const f=fixture(),r=f.create();f.setFault({after:(_options,proof)=>({...proof.result,mutation_receipt:proof.mutation_receipt,session:{...f.current,session_id:"foreign"}})});
  await assert.rejects(r.submit("undo",bodyFor("undo")),/matching current workspace/);assert.equal(r.blocked,true);
});

test("recovery refuses a foreign current workspace even with an applied receipt",async()=>{
  const f=fixture(),r=f.create();f.setFault({after:()=>{throw new Error("Lost");}});await assert.rejects(r.submit("undo",bodyFor("undo")));
  const record=r.pending,proof=f.durable.get(record.mutation_id);
  f.setReads(path=>path.includes("/mutations/")?{schema_version:"product-mutation-status-v1",state:"applied",mutation_id:record.mutation_id,...proof}:
    {...f.workspace(),session:{...f.current,session_id:"foreign"}});
  await assert.rejects(r.check(),/could not be verified/);assert.equal(r.blocked,true);
});

test("nonfinite edit values are refused before a journal or request is created",async()=>{
  const f=fixture(),r=f.create();await assert.rejects(r.submit("selection",{expected_revision:7,selected_day:NaN}),/invalid numeric/);
  assert.equal(f.calls.length,0);assert.equal(f.values.size,0);assert.equal(r.blocked,false);
});

test("initial stale-revision rejection clears the never-committed ID for the existing guarded preview retry",async()=>{
  const f=fixture(),r=f.create();f.current.revision=8;
  await assert.rejects(r.submit("undo",bodyFor("undo")),/Stale/);assert.equal(r.blocked,false);assert.equal(f.values.size,0);assert.equal(f.durable.size,0);
});

test("a status-read failure preserves the original request",async()=>{
  const f=fixture(),r=f.create();f.setFault({after:()=>{throw new Error("Lost");}});await assert.rejects(r.submit("undo",bodyFor("undo")));
  const record=f.values.get("pending");f.setReads(()=>{throw new Error("Offline");});
  await assert.rejects(r.check(),/Offline/);assert.equal(r.blocked,true);assert.equal(f.values.get("pending"),record);
});
