import assert from 'node:assert/strict';
import test from 'node:test';
import fs from 'node:fs';
import vm from 'node:vm';

const root=process.env.TWO_TAB_CANDIDATE;
assert.ok(root);
const source=fs.readFileSync(`${root}/src/itinerary_system/product_app/static/js/app.js`,'utf8');
const names=['refreshStaleWorkspace','api','updateSelection','undoDraft','submitDraftMutation'];
const extract=name=>{
 const start=source.search(new RegExp(`(?:async )?function ${name}\\(`));
 assert.ok(start>=0,name);
 const rest=source.slice(start),next=rest.slice(1).search(/\n(?:async )?function \w+\(/);
 return next<0?rest:rest.slice(0,next+1);
};
const stale=()=>Object.assign(new Error('stale'),{status:409,code:'stale_session_revision'});
const flush=async()=>{for(let i=0;i<30;i++)await Promise.resolve();};
function fixture(){
 const origin={session_id:'one',trip_id:'trip',run_id:'run',accepted_plan_id:'parent',conversation_id:'chat',revision:5,draft:[{operation_id:'op'}],proposal:{state:'eligible'},selected_day:5};
 const state={session:structuredClone(origin),workspace:{},registry:{},token:'synthetic',clientBuildId:'test',workspaceConflict:null,
  mutationRecovery:{blocked:false},previewRuntime:{epoch:3,status:'PREVIEW_RENDERED'},tripPreviewPlanId:'child',acknowledgedPreviewPlanId:'child',mapInteraction:{},undoPending:false};
 const current={session:{...structuredClone(origin),revision:6,draft:[],proposal:null},workspace:{accepted_plan_pointer:{accepted_plan_id:'parent'}},registry:{run_id:'run'}};
 const calls=[],renders=[],notices=[],timers=new Map();let reads=async()=>structuredClone(current),submitError=stale(),submits=0,id=0;
 const context={workspaceNoticeEpoch:0,state,console,AbortController,structuredClone,encodeURIComponent,pendingSelection:null,
  setTimeout:fn=>{timers.set(++id,fn);return id;},clearTimeout:id=>timers.delete(id),
  fetch:async(path,options)=>{calls.push({path,method:options.method});const payload=await reads(path,options);return {ok:true,headers:{get:()=> 'application/json'},json:async()=>payload};},
  renderAll:()=>renders.push({conflict:structuredClone(state.workspaceConflict),status:state.previewRuntime.status}),
  copilot:{previewStateChanged(){},update(){},async restore(){}},
  installDraftMutationSession:(session,payload)=>{state.session=session;state.workspace=payload.workspace;state.registry=payload.registry;},
  syncPreviewRuntimeFromSession:()=>{state.previewRuntime.status=state.session.proposal?'PREVIEW_RENDERED':'PARENT_RENDERED';},
  resetPoiDiscovery(){},selectedDraftWaypoint:()=>false,loadDraftImpactPreview:async()=>null,
  toast:(message,error)=>notices.push({message,error}),
  mutationRecovery:{submit:async()=>{submits++;throw submitError;}},
  isBuildMismatch:()=>false,apiErrorMessage:code=>code,
  $:()=>({open:false}),draftImpactRequestIsCurrent:()=>true,
  previewExpectedContext:()=>structuredClone(state.session),
 };
 vm.createContext(context);vm.runInContext(names.map(extract).join('\n'),context);
 return {context,state,current,calls,renders,notices,timers,setReads:fn=>{reads=fn;},setSubmitError:error=>{submitError=error;},get submits(){return submits;}};
}

test('a stale revision reads the normal workspace once and withdraws the old eligible preview',async()=>{
 const f=fixture();assert.equal(await f.context.refreshStaleWorkspace(stale()),true);
 assert.equal(f.state.session.revision,6);assert.equal(f.state.session.draft.length,0);assert.equal(f.state.session.proposal,null);
 assert.equal(f.state.previewRuntime.status,'PARENT_RENDERED');assert.equal(f.state.acknowledgedPreviewPlanId,null);
 assert.equal(f.state.workspaceConflict,null);assert.deepEqual(f.calls,[{path:'/api/sessions/one',method:'GET'}]);
 assert.ok(f.renders.some(r=>r.conflict?.refreshing&&r.status==='PREVIEW_FAILED'));
 assert.ok(f.notices.some(n=>n.message.includes('not applied')));
});
test('ordinary errors do not refresh or hide a valid preview',async()=>{
 const f=fixture();assert.equal(await f.context.refreshStaleWorkspace(Object.assign(new Error('down'),{status:503})),false);
 assert.equal(f.calls.length,0);assert.equal(f.state.previewRuntime.status,'PREVIEW_RENDERED');
});
test('an unresolved mutation keeps precedence over conflict recovery',async()=>{
 const f=fixture();f.state.mutationRecovery.blocked=true;
 assert.equal(await f.context.refreshStaleWorkspace(stale()),false);assert.equal(f.calls.length,0);
});
for(const key of ['session_id','trip_id','run_id'])test(`a mismatched ${key} never replaces the visible workspace`,async()=>{
 const f=fixture();f.current.session[key]='foreign';await f.context.refreshStaleWorkspace(stale());
 assert.equal(f.state.session.revision,5);assert.equal(f.state.workspaceConflict.refreshing,false);
 assert.equal(f.state.previewRuntime.status,'PREVIEW_FAILED');
});
test('an older response cannot roll the visible revision back',async()=>{
 const f=fixture();f.current.session.revision=4;await f.context.refreshStaleWorkspace(stale());
 assert.equal(f.state.session.revision,5);assert.ok(f.state.workspaceConflict);
});
test('a response racing a newer local revision cannot roll back that revision',async()=>{
 const f=fixture();f.setReads(async()=>{f.state.session.revision=8;return f.current;});
 await f.context.refreshStaleWorkspace(stale());assert.equal(f.state.session.revision,8);assert.ok(f.state.workspaceConflict);
});
test('a failed read blocks writes and a manual read retry recovers without any write',async()=>{
 const f=fixture();f.setReads(async()=>{throw new TypeError('Network lost');});await f.context.refreshStaleWorkspace(stale());
 assert.equal(f.state.workspaceConflict.refreshing,false);assert.equal(f.state.previewRuntime.status,'PREVIEW_FAILED');
 await assert.rejects(f.context.api('/api/sessions/one/accept',{method:'POST',body:{}}),/Refresh/);
 await assert.rejects(f.context.submitDraftMutation('undo'),/Refresh/);
 assert.equal(await f.context.updateSelection({selected_day:7}),false);assert.equal(f.submits,0);
 f.setReads(async()=>f.current);await f.context.refreshStaleWorkspace();assert.equal(f.state.workspaceConflict,null);
 assert.deepEqual(f.calls.map(r=>r.method),['GET','GET']);
});
test('a stalled read times out, ignores its late response and retains the retry barrier',async()=>{
 const f=fixture();let release;f.setReads(()=>new Promise(yes=>{release=yes;}));
 const pending=f.context.refreshStaleWorkspace(stale());await flush();assert.equal(f.timers.size,1);
 [...f.timers.values()][0]();await pending;release(f.current);await flush();
 assert.equal(f.state.session.revision,5);assert.equal(f.state.workspaceConflict.refreshing,false);assert.equal(f.timers.size,0);
});
test('overlapping failures share the pending read rather than starting another refresh',async()=>{
 const f=fixture();let release;f.setReads(()=>new Promise(yes=>{release=yes;}));
 const first=f.context.refreshStaleWorkspace(stale());await flush();await f.context.refreshStaleWorkspace(stale());
 assert.equal(f.calls.length,1);release(f.current);await first;
});
test('actual selection handler recovers after rejection and never repeats the selection',async()=>{
 const f=fixture();assert.equal(await f.context.updateSelection({selected_day:7}),false);
 assert.equal(f.submits,1);assert.equal(f.state.session.revision,6);assert.equal(f.calls.length,1);
});
test('actual Undo handler recovers after rejection without removing another draft',async()=>{
 const f=fixture();await f.context.undoDraft();assert.equal(f.submits,1);
 assert.equal(f.state.session.revision,6);assert.equal(f.state.undoPending,false);assert.equal(f.calls.length,1);
});
test('a different trip selected during the read is never overwritten',async()=>{
 const f=fixture();f.setReads(async()=>{f.state.session={...f.state.session,session_id:'two',revision:9};return f.current;});
 await f.context.refreshStaleWorkspace(stale());assert.equal(f.state.session.session_id,'two');assert.equal(f.state.session.revision,9);
});
