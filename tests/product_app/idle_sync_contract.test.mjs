import assert from 'node:assert/strict';
import test from 'node:test';
import fs from 'node:fs';
import vm from 'node:vm';

const root=process.env.IDLE_SYNC_CANDIDATE;assert.ok(root);
const source=fs.readFileSync(`${root}/src/itinerary_system/product_app/static/js/app.js`,'utf8');
const copilotSource=fs.readFileSync(`${root}/src/itinerary_system/product_app/static/js/copilot.js`,'utf8');
const extract=(name,code=source)=>{
 const start=code.search(new RegExp(`(?:async )?function ${name}\\(`));assert.ok(start>=0,name);
 const rest=code.slice(start),next=rest.slice(1).search(/\n(?:  )?(?:async )?function \w+\(/);
 return next<0?rest:rest.slice(0,next+1);
};
const microtasks=async()=>{for(let i=0;i<40;i++)await Promise.resolve();};
function fixture({channelUnavailable=false}={}){
 const session={session_id:'one',trip_id:'trip',run_id:'run',revision:5,conversation_id:'chat',accepted_plan_id:'parent',draft:[{operation_id:'op'}],proposal:{state:'eligible',acceptance_eligible:true,child_plan_id:'child'}};
 const state={session:structuredClone(session),workspace:{},registry:{},token:'synthetic-not-a-real-token',clientBuildId:'test',workspaceConflict:null,workspaceSyncPending:false,
  mutationRecovery:{blocked:false},undoPending:false,mapInteraction:{},previewRuntime:{epoch:3,status:'PREVIEW_RENDERED'},acknowledgedPreviewPlanId:'child',tripPreviewPlanId:'child'};
 const current={session:{...structuredClone(session),revision:6,draft:[],proposal:null},workspace:{accepted_plan_pointer:{accepted_plan_id:'parent'}},registry:{run_id:'run'}};
 const calls=[],renders=[],notices=[],sent=[],listeners={window:{},document:{}},timers=new Map();let timerId=0,read=async()=>structuredClone(current),busy=false,restores=0,impacts=0,poiResets=0,receive=null,channelThrows=false;
 class Channel{constructor(name){if(channelUnavailable)throw new Error('Channel denied');this.name=name;}addEventListener(name,fn){assert.equal(name,'message');receive=fn;}postMessage(value){if(channelThrows)throw new Error('Channel closed');sent.push(structuredClone(value));}}
 const context={state,console,AbortController,structuredClone,encodeURIComponent,BroadcastChannel:Channel,workspaceChannel:null,workspaceRefreshQueued:false,workspaceRefreshTimer:null,workspaceNoticeEpoch:0,pendingSelection:null,
  window:{addEventListener:(name,fn)=>{listeners.window[name]=fn;}},document:{visibilityState:'visible',addEventListener:(name,fn)=>{listeners.document[name]=fn;}},
  setTimeout:(fn,delay)=>{timers.set(++timerId,{fn,delay});return timerId;},clearTimeout:id=>timers.delete(id),
  fetch:async(path,options)=>{calls.push({path,method:options.method,body:options.body});const value=await read(path,options);return {ok:true,headers:{get:()=> 'application/json'},json:async()=>value};},
  renderAll:()=>{renders.push({pending:state.workspaceSyncPending,conflict:structuredClone(state.workspaceConflict),status:state.previewRuntime.status});context.flushQueuedWorkspaceRefresh();},
  copilot:{isBusy:()=>busy,previewStateChanged(){},async restore(){restores++;}},
  installDraftMutationSession:(session,payload)=>{state.session=session;state.workspace=payload.workspace;state.registry=payload.registry;},
  syncPreviewRuntimeFromSession:()=>{state.previewRuntime.status=state.session.proposal?'PREVIEW_RENDERED':'PARENT_RENDERED';},
  resetPoiDiscovery:()=>{poiResets++;},selectedDraftWaypoint:()=>false,loadDraftImpactPreview:async()=>{impacts++;return null;},
  toast:(message,error)=>notices.push({message,error}),isBuildMismatch:()=>false,apiErrorMessage:code=>code,
 };
 vm.createContext(context);vm.runInContext(['flushQueuedWorkspaceRefresh','requestWorkspaceRefresh','bindWorkspaceSynchronization','announceWorkspaceChange','refreshStaleWorkspace','api','previewCanBeAccepted','activeProposalPlan'].map(n=>extract(n)).join('\n'),context);
 context.bindWorkspaceSynchronization();
 const tick=async()=>{const zero=[...timers].filter(([,t])=>t.delay===0);for(const [id,t]of zero){timers.delete(id);t.fn();}await microtasks();};
 const drain=async()=>{for(let n=0;n<20&&[...timers.values()].some(t=>t.delay===0);n++)await tick();};
 return {context,state,current,calls,renders,notices,sent,listeners,timers,tick,drain,
  deliver:value=>receive?.({data:value}),setBusy:value=>{busy=value;},setRead:fn=>{read=fn;},setChannelThrows:()=>{channelThrows=true;},
  get restores(){return restores;},get impacts(){return impacts;},get poiResets(){return poiResets;}};
}
const notice=(extra={})=>({schema_version:'product-workspace-invalidation-v1',session_id:'one',revision:6,conversation_id:'chat',...extra});

test('a newer notice withdraws the old preview and reads authoritative state without an edit',async()=>{
 const f=fixture();f.deliver(notice());assert.equal(f.state.workspaceSyncPending,true);
 assert.equal(f.context.previewCanBeAccepted(),false);assert.equal(f.context.activeProposalPlan(),null);
 assert.equal(f.state.session.revision,5);await f.drain();assert.equal(f.state.session.revision,6);assert.equal(f.state.session.draft.length,0);
 assert.equal(f.state.workspaceSyncPending,false);assert.equal(f.state.workspaceConflict,null);
 assert.deepEqual(f.calls.map(r=>r.method),['GET']);
});
for(const [name,value]of [
 ['wrong session',notice({session_id:'other'})],['older revision',notice({revision:4})],['fractional revision',notice({revision:6.5})],
 ['untrusted workspace body',notice({session:{revision:100}})],['wrong schema',notice({schema_version:'other'})],['invalid conversation',notice({conversation_id:{}})],['null',null]
])test(`${name} is ignored`,async()=>{const f=fixture();f.deliver(value);await f.drain();assert.equal(f.calls.length,0);assert.equal(f.state.workspaceSyncPending,false);});
test('a matching revision and conversation is ignored',async()=>{
 const f=fixture();f.deliver(notice({revision:5}));await f.drain();assert.equal(f.calls.length,0);
});
test('same-revision conversation rebinding is read from the server',async()=>{
 const f=fixture();f.current.session.revision=5;f.current.session.conversation_id='new-chat';f.deliver(notice({revision:5,conversation_id:'notice-is-not-authority'}));
 await f.drain();assert.equal(f.state.session.conversation_id,'new-chat');assert.equal(f.restores,1);
});
test('several notices in one task coalesce to one read',async()=>{
 const f=fixture();f.deliver(notice());f.deliver(notice({revision:7}));f.deliver(notice({revision:8}));await f.drain();assert.equal(f.calls.length,1);
});
test('focus checks unchanged state without resetting a valid preview or reloading conversation',async()=>{
 const f=fixture();Object.assign(f.current.session,structuredClone(f.state.session));f.listeners.window.focus();await f.drain();
 assert.equal(f.calls.length,1);assert.equal(f.state.previewRuntime.status,'PREVIEW_RENDERED');assert.equal(f.state.acknowledgedPreviewPlanId,'child');
 assert.equal(f.restores,0);assert.equal(f.impacts,0);assert.equal(f.poiResets,0);assert.equal(f.notices.length,0);
});
test('visibility checks only when the document becomes visible',async()=>{
 const f=fixture();f.context.document.visibilityState='hidden';f.listeners.document.visibilitychange();await f.drain();assert.equal(f.calls.length,0);
 f.context.document.visibilityState='visible';f.listeners.document.visibilitychange();await f.drain();assert.equal(f.state.session.revision,6);
});
test('focus recovery remains usable when BroadcastChannel is unavailable',async()=>{
 const f=fixture({channelUnavailable:true});assert.equal(f.context.workspaceChannel,null);f.listeners.window.focus();await f.drain();assert.equal(f.state.session.revision,6);
});
test('an unresolved saved action keeps precedence and the queued read resumes after settlement',async()=>{
 const f=fixture();f.state.mutationRecovery.blocked=true;f.deliver(notice());await f.drain();assert.equal(f.calls.length,0);assert.equal(f.state.workspaceSyncPending,true);
 f.state.mutationRecovery.blocked=false;f.context.flushQueuedWorkspaceRefresh();await f.drain();assert.equal(f.calls.length,1);assert.equal(f.state.session.revision,6);
});
test('active chat defers refresh while immediately disabling stale acceptance',async()=>{
 const f=fixture();f.setBusy(true);f.deliver(notice());await f.drain();assert.equal(f.calls.length,0);assert.equal(f.context.previewCanBeAccepted(),false);
 f.setBusy(false);f.context.flushQueuedWorkspaceRefresh();await f.drain();assert.equal(f.state.session.revision,6);
});
test('selection, Undo and running Preview each defer the queued read',async()=>{
 for(const kind of ['selection','undo','preview']){
  const f=fixture();if(kind==='selection')f.context.pendingSelection=Promise.resolve();if(kind==='undo')f.state.undoPending=true;if(kind==='preview')f.state.previewRuntime.status='PREVIEW_RUNNING';
  f.deliver(notice());await f.drain();assert.equal(f.calls.length,0,kind);
  f.context.pendingSelection=null;f.state.undoPending=false;f.state.previewRuntime.status='PARENT_RENDERED';f.context.flushQueuedWorkspaceRefresh();await f.drain();assert.equal(f.calls.length,1,kind);
 }
});
test('an automatic read failure retains the barrier and manual retry does not claim a rejected action',async()=>{
 const f=fixture();f.setRead(async()=>{throw new TypeError('Private diagnostic should not enter product copy');});f.deliver(notice());await f.drain();
 assert.equal(f.state.workspaceConflict.refreshing,false);assert.equal(f.state.workspaceSyncPending,true);assert.equal(f.context.activeProposalPlan(),null);
 assert.ok(!f.state.workspaceConflict.message.includes('Private diagnostic'));
 await assert.rejects(f.context.api('/api/sessions/one/accept',{method:'POST',body:{}}),/Refresh/);
 f.setRead(async()=>f.current);await f.context.refreshStaleWorkspace();await f.drain();assert.equal(f.state.workspaceSyncPending,false);
 assert.ok(f.notices.some(n=>n.message.includes('updated from another action')));assert.ok(!f.notices.some(n=>n.message.includes('rejected action')));
 assert.ok(f.calls.every(r=>r.method==='GET'));
});
test('existing pending-action retry is allowed while unrelated new writes remain blocked',async()=>{
 const f=fixture();f.state.workspaceSyncPending=true;f.state.mutationRecovery.blocked=true;
 await f.context.api('/api/sessions/one/draft/operations',{method:'POST',body:{mutation_id:'existing'},pendingMutation:true});assert.equal(f.calls.length,1);
 await assert.rejects(f.context.api('/api/sessions/one/accept',{method:'POST',body:{}}),/Resolve|Refresh/);assert.equal(f.calls.length,1);
 f.state.mutationRecovery.blocked=false;await assert.rejects(f.context.api('/api/sessions/one/draft/operations',{method:'POST',body:{},pendingMutation:true}),/Refresh/);
});
test('a notice received during a read is reconciled again after the first read settles',async()=>{
 const f=fixture();let release,reads=0;f.setRead(()=>{reads++;return reads===1?new Promise(yes=>{release=yes;}):{...f.current,session:{...f.current.session,revision:7}};});
 f.deliver(notice());await f.tick();f.deliver(notice({revision:7}));await f.tick();release(structuredClone(f.current));await microtasks();await f.drain();
 assert.equal(reads,2);assert.equal(f.state.session.revision,7);
});
test('a write announces only the bounded invalidation fields; reads never announce',async()=>{
 const f=fixture();await f.context.api('/api/sessions/one/selection',{method:'POST',body:{selected_day:7}});
 assert.equal(f.sent.length,1);assert.deepEqual(Object.keys(f.sent[0]).sort(),['conversation_id','revision','schema_version','session_id']);
 assert.ok(!JSON.stringify(f.sent).includes('synthetic-not-a-real-token'));
 await f.context.api('/api/sessions/one');assert.equal(f.sent.length,1);
});
test('notification failure never turns a committed response into an action failure',async()=>{
 const f=fixture();f.setChannelThrows();assert.equal((await f.context.api('/api/sessions/one/selection',{method:'POST',body:{}})).session.revision,6);
 assert.equal(f.context.workspaceChannel,null);f.listeners.window.focus();await f.drain();assert.equal(f.state.session.revision,6);
});
test('actual controller idle callback fires only when chat and preview actions are idle',()=>{
 let idle=0;const node={disabled:false,dataset:{},setAttribute(){}};
 const ctx={requestActive:false,previewActionActive:false,pendingConversationReset:null,$:()=>node,$$:()=>[],available:()=>true,getState:()=>({}),onIdle:()=>{idle++;}};
 vm.createContext(ctx);vm.runInContext(extract('setSubmitting',copilotSource),ctx);
 ctx.setSubmitting(true);assert.equal(idle,0);ctx.setSubmitting(false);assert.equal(idle,1);
 ctx.previewActionActive=true;ctx.setSubmitting(false);assert.equal(idle,1);ctx.previewActionActive=false;
 ctx.pendingConversationReset=Promise.resolve();ctx.setSubmitting(false);assert.equal(idle,1);ctx.pendingConversationReset=null;ctx.setSubmitting(false);assert.equal(idle,2);
});

test('read-only impact POST never broadcasts a saved-workspace change',async()=>{
 const f=fixture();await f.context.api('/api/sessions/one/draft/impact-preview',{method:'POST',body:{}});assert.equal(f.sent.length,0);
});
test('recovery barrier disables inline Undo, Preview and Accept while preserving pending recovery controls',()=>{
 const controls=new Map(['#inline-undo','#inline-preview','#accept-repair','#copilot-submit[data-mode=send]'].map(id=>[id,{disabled:false,dataset:{}}]));
 const check={disabled:false};const banner={hidden:true,innerHTML:'',querySelector:()=>({addEventListener(){}})};
 const shell={classList:{toggle(){}}};
 const state={mutationRecovery:{blocked:false},workspaceConflict:{refreshing:false,message:'Check current workspace'},workspaceSyncPending:true};
 const ctx={state,$:selector=>selector==='#mutation-recovery-banner'?banner:shell,
  $$:selector=>[...controls].filter(([id])=>selector.split(', ').includes(id)).map(([,control])=>control),
  updateMutationRecoveryLayout(){},escapeHtml:value=>value,refreshStaleWorkspace(){}};
 vm.createContext(ctx);vm.runInContext(extract('renderMutationRecovery'),ctx);ctx.renderMutationRecovery();
 assert.equal(banner.hidden,false);for(const [id,control]of controls){assert.equal(control.disabled,true,id);assert.equal(control.dataset.pendingMutationDisabled,'true');}
 assert.equal(check.disabled,false);
 vm.runInContext(extract('restoreMutationControls'),ctx);ctx.$$=selector=>selector==='[data-pending-mutation-disabled]'?[...controls.values()]:[];ctx.restoreMutationControls();
 for(const control of controls.values()){assert.equal(control.disabled,false);assert.equal(control.dataset.pendingMutationDisabled,undefined);}
});

test('a newer notice while an authoritative read is pending cannot briefly restore old eligibility',async()=>{
 const f=fixture();let releaseFirst,releaseSecond,reads=0;
 const older={...structuredClone(f.current),session:{...structuredClone(f.current.session),proposal:structuredClone(f.state.session.proposal)}};
 f.setRead(()=>{reads++;return new Promise(yes=>{if(reads===1)releaseFirst=yes;else releaseSecond=yes;});});
 f.deliver(notice());await f.tick();f.deliver(notice({revision:7}));await f.tick();releaseFirst(older);await microtasks();
 assert.equal(f.state.workspaceSyncPending,true,'A newer notice must keep the old preview blocked until a new read settles');
 assert.equal(f.context.activeProposalPlan(),null);assert.equal(f.context.previewCanBeAccepted(),false);
 assert.equal(f.state.session.revision,5,'A superseded read must not install its older snapshot');
 await f.tick();assert.equal(reads,2);releaseSecond({...f.current,session:{...f.current.session,revision:7}});await microtasks();await f.drain();
 assert.equal(f.state.session.revision,7);assert.equal(f.state.workspaceSyncPending,false);
});
test('same-revision conversation notice also supersedes an in-flight no-op focus read',async()=>{
 const f=fixture();let releaseFirst,reads=0;const older={...structuredClone(f.current),session:structuredClone(f.state.session)};
 f.setRead(()=>{reads++;return reads===1?new Promise(yes=>{releaseFirst=yes;}):{...f.current,session:{...f.current.session,revision:5,conversation_id:'new-chat',proposal:null}};});
 f.listeners.window.focus();await f.tick();f.deliver(notice({revision:5,conversation_id:'new-chat'}));await f.tick();releaseFirst(older);await microtasks();
 assert.equal(f.state.workspaceSyncPending,true);assert.equal(f.context.previewCanBeAccepted(),false);
 await f.drain();assert.equal(f.state.session.conversation_id,'new-chat');assert.equal(f.state.workspaceSyncPending,false);
});

test('chat adopting the new revision cannot make a known invalidation skip draft impact reconciliation',async()=>{
 const f=fixture();f.setBusy(true);f.deliver(notice());await f.drain();assert.equal(f.calls.length,0);
 // A real terminal chat reply can adopt the newer session before the idle callback.
 f.state.session=structuredClone(f.current.session);
 f.setBusy(false);f.context.flushQueuedWorkspaceRefresh();await f.drain();
 assert.equal(f.impacts,1,'The authoritative workspace must refresh draft impact even when chat already adopted its revision');
 assert.equal(f.restores,1);assert.equal(f.state.workspaceSyncPending,false);
 assert.equal(f.state.workspace.accepted_plan_pointer.accepted_plan_id,'parent');
});
