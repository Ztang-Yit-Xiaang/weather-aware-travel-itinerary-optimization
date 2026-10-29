import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import test from "node:test";
import {fileURLToPath} from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "..");
const source = fs.readFileSync(process.argv[2] || path.join(root, "src/itinerary_system/product_app/static/js/app.js"), "utf8");
const fixture = fs.readFileSync(path.join(root, "tests/product_app/typed_edit_frontend_contract.test.mjs"), "utf8");
const capabilities = new Function(fixture.slice(fixture.indexOf("const common ="), fixture.indexOf("state.workspace =")) + "return capabilities;")();
function section(start, end) {
  const first = source.indexOf(start), last = source.indexOf(end, first + start.length);
  assert.ok(first >= 0 && last > first, start);
  return source.slice(first, last);
}
const deferred = () => {
  let resolve, reject;
  const promise = new Promise((yes, no) => {resolve = yes; reject = no;});
  return {promise, resolve, reject};
};
const flush = async () => {for (let index = 0; index < 15; index++) await Promise.resolve();};

function harness() {
  const operation = {operation_id:"op_day", type:"set_stop_day", target:"stop_one", parameters:{day:4}};
  const state = {session:{session_id:"session_fixture", trip_id:"trip", run_id:"run", revision:7,
    accepted_plan_id:"parent", draft:[operation], proposal:null},
    workspace:{typed_edit_capabilities:capabilities, draft_capabilities:{operations:{}}, map_edit_capabilities:{operations:{}}},
    typedImpact:{status:"idle", response:null, error:null}, undoPending:false, mapInteraction:{},
    previewRuntime:{epoch:1,status:"PARENT_RENDERED",requestId:null}};
  const before = structuredClone(state.session), calls = [], timers = new Map(), queue = [], presentations = [], messages = [];
  const retry = {listener:null, addEventListener(_event, listener) {this.listener = listener;}};
  const surface = {querySelectorAll:()=>[], querySelector:selector=>selector === "[data-retry-draft-impact]" ? retry : null};
  const modal = {open:true};
  let nextTimer = 0, functions, undoCalls = 0, copilotUpdates = 0, dialogUpdates = 0;
  const snapshot = () => ({status:state.typedImpact.status, revision:state.session.revision,
    canPreview:functions.canRunEvaluatedPreview(), draft:structuredClone(state.session.draft)});
  const dependencies = {state,
    fetch:async (url, options) => {
      calls.push({url, options});
      assert.ok(url.endsWith("/draft/impact-preview"), "Only the declared read endpoint is allowed");
      assert.equal(options.method, "POST");
      return queue.shift()(options);
    },
    setTimeout:(callback, delay) => {const id=++nextTimer; timers.set(id,{callback, delay}); return id;},
    clearTimeout:id=>timers.delete(id), AbortController,
    apiErrorMessage:code=>code, isBuildMismatch:()=>false, renderBuildRecovery:()=>{},
    renderAll:()=>presentations.push(snapshot()), refreshMapEditDialog:()=>dialogUpdates++,
    $:selector=>selector === "#typed-edit-surface" ? surface : modal,
    submitTypedOperation:()=>{}, bindDurationModeSelects:()=>{}, bindRoleSelects:()=>{},
    pendingSelection:null, copilot:{update:()=>copilotUpdates++},
    submitDraftMutation:async action=>{
      assert.equal(action,"undo"); undoCalls++;
      return {session:{...structuredClone(state.session), revision:state.session.revision+1, draft:[], proposal:null}};
    },
    installDraftMutationSession:session=>{state.session=session;}, syncPreviewRuntimeFromSession:()=>{},
    resetPoiDiscovery:()=>{}, selectedDraftWaypoint:()=>null, toast:message=>messages.push(message),
  };
  const code = section("const TYPED_EDIT_OPERATIONS", "function safeExternalLink")
    + section("async function api(", "function copilotStreamError(")
    + section("function draftImpactRequestIsCurrent(", "function selectContext(")
    + section("function bindTypedEditSurface(", "async function updateSelection(")
    + section("async function undoDraft()", "async function previewDraft(");
  functions = new Function(...Object.keys(dependencies),code+"return {loadDraftImpactPreview,canRunEvaluatedPreview,bindTypedEditSurface,undoDraft};")(...Object.values(dependencies));
  const payload = () => ({schema_version:"product-draft-impact-preview-v1", session_id:state.session.session_id,
    session_revision:state.session.revision, parent_plan_id:state.session.accepted_plan_id, parent_plan_content_hash:"a".repeat(16),
    certification_status:"not_certified", is_certified:false,
    operations:state.session.draft.map(row=>({operation_id:row.operation_id, type:row.type, target:row.target,
      feedback_tier:"evaluated", preview_executable:true, evaluated_repair:true, changed_attributes:["day"], blocking_codes:[]})),
    summary:{operation_count:state.session.draft.length, evaluated_executable_count:state.session.draft.length,
      draft_only_count:0, can_run_evaluated_preview:state.session.draft.length>0, blocking_codes:[]}});
  const response = (value=payload(), status=200) => ({ok:status===200,status,headers:{get:()=>"application/json"},json:async()=>value});
  return {state, before, calls, queue, timers, presentations, messages, retry, functions, payload, response,
    expire:()=>{assert.equal(timers.size,1,"The unresolved read must have one deadline"); for(const timer of timers.values()){assert.equal(timer.delay,20000);timer.callback();}},
    undoCalls:()=>undoCalls, copilotUpdates:()=>copilotUpdates, dialogUpdates:()=>dialogUpdates};
}

for (const boundary of ["headers", "body"]) {
  test(`${boundary}: unresolved read finishes at its deadline without retry or draft mutation`, async () => {
    const h=harness(), held=deferred();
    h.queue.push(()=>boundary === "headers" ? held.promise : {...h.response(), json:()=>held.promise});
    const pending=h.functions.loadDraftImpactPreview(); await flush();
    assert.equal(h.state.typedImpact.status,"loading"); assert.equal(h.functions.canRunEvaluatedPreview(),false);
    h.expire(); assert.equal(await pending,null);
    assert.equal(h.state.typedImpact.status,"failed"); assert.match(h.state.typedImpact.error,/timed out/);
    assert.equal(h.calls[0].options.signal.aborted,true); assert.equal(h.timers.size,0);
    assert.deepEqual(h.state.session,h.before); assert.equal(h.calls.length,1);
    held.resolve(boundary === "headers" ? h.response() : h.payload()); await flush();
    assert.equal(h.state.typedImpact.status,"failed"); assert.equal(h.functions.canRunEvaluatedPreview(),false);
  });
}
for (const outcome of ["resolve", "reject"]) {
  test(`late ${outcome} after deadline cannot erase a successful explicit recheck`, async () => {
    const h=harness(), held=deferred(); h.queue.push(()=>held.promise);
    const first=h.functions.loadDraftImpactPreview(); await flush(); h.expire(); await first;
    h.queue.push(()=>h.response()); await h.functions.loadDraftImpactPreview();
    const ready=structuredClone(h.state.typedImpact);
    if(outcome === "resolve") held.resolve(h.response()); else held.reject(new Error("Late lost read"));
    await flush(); assert.deepEqual(h.state.typedImpact,ready); assert.equal(h.calls.length,2);
    assert.equal(h.timers.size,0); assert.deepEqual(h.state.session,h.before);
  });
}
for (const outcome of ["success", "http_error", "invalid_contract"]) {
  test(`${outcome}: settled read clears its deadline and preserves strict validation`, async () => {
    const h=harness(); h.queue.push(()=>outcome === "http_error" ? h.response({detail:"state_lock_timeout"},503)
      : h.response(outcome === "invalid_contract" ? {...h.payload(),is_certified:true} : h.payload()));
    const result=await h.functions.loadDraftImpactPreview();
    assert.equal(h.timers.size,0); assert.equal(h.state.typedImpact.status,outcome === "success" ? "ready" : "failed");
    assert.equal(Boolean(result),outcome === "success"); assert.equal(h.functions.canRunEvaluatedPreview(),outcome === "success");
    assert.equal(h.calls[0].options.signal.aborted,false); assert.deepEqual(h.state.session,h.before);
  });
}
test("an older deadline cannot overwrite a newer same-revision read", async () => {
  const h=harness(), old=deferred(), current=deferred(); h.queue.push(()=>old.promise,()=>current.promise);
  const first=h.functions.loadDraftImpactPreview(); await flush(); const oldTimer=[...h.timers.values()][0];
  const second=h.functions.loadDraftImpactPreview(); await flush(); current.resolve(h.response()); await second;
  const ready=structuredClone(h.state.typedImpact); oldTimer.callback(); await first;
  assert.deepEqual(h.state.typedImpact,ready); assert.equal(h.timers.size,0); old.resolve(h.response()); await flush();
});
test("a deadline after workspace switch cannot render or alter the new workspace", async () => {
  const h=harness(), held=deferred(); h.queue.push(()=>held.promise);
  const pending=h.functions.loadDraftImpactPreview(); await flush();
  h.state.session={...h.state.session,session_id:"different",revision:0};
  h.state.typedImpact={status:"ready",response:{newWorkspace:true},request:{different:true}};
  const next=structuredClone(h.state); h.expire(); await pending;
  assert.deepEqual(h.state,next); assert.equal(h.presentations.length,0); held.resolve(h.response()); await flush();
});
test("explicit retry binds the visible control and ignores duplicate clicks while loading", async () => {
  const h=harness(), held=deferred(); h.state.typedImpact.status="failed"; h.queue.push(()=>held.promise);
  h.functions.bindTypedEditSurface(); assert.equal(typeof h.retry.listener,"function");
  const pending=h.retry.listener(); await flush(); await h.retry.listener();
  assert.equal(h.calls.length,1); assert.equal(h.state.typedImpact.status,"loading");
  held.resolve(h.response()); await pending;
  assert.equal(h.state.typedImpact.status,"ready"); assert.equal(h.functions.canRunEvaluatedPreview(),true);
  assert.equal(h.dialogUpdates(),2); assert.equal(h.undoCalls(),0); assert.deepEqual(h.state.session,h.before);
});
test("confirmed Undo unlocks after stalled impact expires and never sends another Undo", async () => {
  const h=harness(), held=deferred(); h.queue.push(()=>held.promise);
  const pending=h.functions.undoDraft(); await flush();
  assert.equal(h.undoCalls(),1); assert.equal(h.state.session.revision,8); assert.deepEqual(h.state.session.draft,[]);
  assert.equal(h.state.undoPending,true,h.messages.join("; ")); h.expire(); await pending;
  assert.equal(h.state.undoPending,false); assert.equal(h.copilotUpdates(),2); assert.equal(h.undoCalls(),1);
  assert.equal(h.state.typedImpact.status,"failed"); assert.equal(h.functions.canRunEvaluatedPreview(),false);
  held.resolve(h.response()); await flush(); assert.equal(h.state.typedImpact.status,"failed");
});
test("late retry completion cannot refresh a changed editor workspace", async () => {
  const h=harness(), held=deferred(); h.queue.push(()=>held.promise); h.functions.bindTypedEditSurface();
  const pending=h.retry.listener(); await flush(); const rendered=h.dialogUpdates();
  h.state.session.revision++; h.state.typedImpact={status:"idle",request:{newer:true}};
  held.resolve(h.response()); await pending; assert.equal(h.dialogUpdates(),rendered);
  assert.equal(h.state.typedImpact.status,"idle"); assert.equal(h.timers.size,0);
});
