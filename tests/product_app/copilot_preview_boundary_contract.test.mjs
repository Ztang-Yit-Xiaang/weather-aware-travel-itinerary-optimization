import assert from "node:assert/strict";
import fs from "node:fs";
import { sameCopilotDraftOperation } from "../../src/itinerary_system/product_app/static/js/copilot.js";

// Execute the actual submission function, controlling only its asynchronous
// selection boundary. API/renderer stubs stop the test after the request decision.
const source = fs.readFileSync(new URL("../../src/itinerary_system/product_app/static/js/app.js", import.meta.url), "utf8");
const start = source.indexOf("async function previewDraft(");
const end = source.indexOf("\nfunction setupDockResize()", start);
assert.ok(start >= 0 && end > start);
const actual = source.slice(start, end);
const operation = {type:"set_stop_duration",target:"griffith_observatory",parameters:{duration:{
  mode:"exact",preferred_minutes:60,minimum_minutes:60,maximum_minutes:60,
}}};
const extra = {...structuredClone(operation),target:"stearns_wharf"};

async function exercise({after,options,mutateOptions,selection=true,unscoped=false,sessionPatch={}}) {
  if (!unscoped && options === undefined) options={expectedCopilotOperation:structuredClone(operation)};
  let release;
  const pendingSelection = new Promise(resolve => { release = resolve; });
  const state = {session:{session_id:"session_fixture",conversation_id:"conversation_fixture",trip_id:"trip_fixture",
    run_id:"run_fixture",accepted_plan_id:"plan_fixture",revision:1,draft:[structuredClone(operation)]},previewRuntime:{epoch:0}};
  const requests = [], messages = [];
  let contextReads = 0;
  const run = new Function("pendingSelection","state","toast","hasRouteCheckedOnlyDraft","canRunEvaluatedPreview",
    "renderAll","copilot","previewExpectedContext","api","sameCopilotDraftOperation","options",actual + "\nreturn previewDraft(options);");
  const pending = run(pendingSelection,state,(message)=>messages.push(message),()=>false,()=>true,()=>{},
    {previewStateChanged(){}},()=>{contextReads++;return structuredClone(state.session);},
    async (url,request)=>{requests.push({url,request,draft:structuredClone(state.session.draft)});throw new Error("Captured request boundary");},
    sameCopilotDraftOperation,options);
  state.session={...state.session,...sessionPatch,revision:2,draft:structuredClone(after)};
  mutateOptions?.(options);
  release(selection);
  await pending;
  return {requests,messages,contextReads,state};
}

for (const [name,after] of [
  ["additional operation",[operation,extra]],
  ["duplicate matching operations",[operation,operation]],
  ["removed operation",[]],
  ["changed target",[extra]],
  ["changed type",[{...operation,type:"set_stop_order"}]],
  ["changed parameters",[{...operation,parameters:{duration:{mode:"exact",preferred_minutes:90,minimum_minutes:90,maximum_minutes:90}}}]],
]) {
  const result = await exercise({after});
  assert.equal(result.requests.length,0,`${name}: no broader or different preview may be sent`);
  assert.equal(result.contextReads,0,`${name}: reject before constructing a new expected preview context`);
  assert.deepEqual(result.state.session.draft,after,`${name}: rejection must retain the draft`);
  assert.ok(result.messages.some(message=>message.includes("exact staged suggestion")),name);
}
for (const expectedCopilotOperation of [null,undefined,{}]) {
  const result = await exercise({after:[operation],options:{expectedCopilotOperation}});
  assert.equal(result.requests.length,0,"An explicitly supplied invalid guard must fail closed");
}
const mutated = await exercise({after:[extra],mutateOptions:options=>{options.expectedCopilotOperation.target=extra.target;}});
assert.equal(mutated.requests.length,0,"Snapshot the caller's expectation before the asynchronous wait");
const exact = await exercise({after:[operation]});
assert.equal(exact.requests.length,1);
assert.equal(exact.requests[0].request.body.expected_revision,2,"Use current revision when exact draft survives selection");
assert.deepEqual(exact.requests[0].draft,[operation]);
assert.equal(exact.state.previewRuntime.status,"PREVIEW_FAILED");
assert.equal(exact.state.previewRuntime.message,"Captured request boundary",
  "The actual submission function retains its failure reason after the toast disappears");
const reorderedKeys = structuredClone(operation);
reorderedKeys.parameters.duration = {maximum_minutes:60,minimum_minutes:60,preferred_minutes:60,mode:"exact"};
assert.equal((await exercise({after:[reorderedKeys]})).requests.length,1,"Canonical object key order is not semantic change");
for (const options of [undefined,null,new Event("click")]) {
  const result = await exercise({after:[operation,extra],options,unscoped:true});
  assert.equal(result.requests.length,1,"Ordinary DOM click Preview still includes all valid staged operations");
  assert.deepEqual(result.requests[0].draft,[operation,extra]);
}
const failedSelection = await exercise({after:[operation],selection:false});
assert.equal(failedSelection.requests.length,0);
for (const combined of [false,true]) {
  const after=combined ? [operation,extra] : [operation];
  const options=combined ? {expectedCopilotOperations:structuredClone(after)} : {expectedCopilotOperation:structuredClone(operation)};
  assert.equal((await exercise({after,options})).requests.length,1,"A legitimate selection may advance revision without changing ownership");
  for (const [field,value] of [["session_id","foreign_session"],["conversation_id","foreign_conversation"],
    ["conversation_id",null],["trip_id","foreign_trip"],["run_id","foreign_run"],["accepted_plan_id","other_plan"]]) {
    const result=await exercise({after,options,sessionPatch:{[field]:value}});
    assert.equal(result.requests.length,0,`${combined ? "Combined" : "Single"} queued Copilot preview must not cross ${field}`);
    assert.equal(result.contextReads,0,"Reject changed ownership before constructing a replacement preview context");
    assert.equal(result.state.previewRuntime.epoch,0,"Rejection must not replace the new workspace's preview state");
    assert.deepEqual(result.state.session.draft,after,"The new workspace's draft remains unchanged");
  }
}
console.log("Actual preview boundary rejects changed Copilot drafts after selection; ordinary DOM Preview remains preview-all.");
