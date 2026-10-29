import { readFileSync } from "node:fs";
import assert from "node:assert/strict";
import { createCopilotController, copilotOperationForIntent } from "../../src/itinerary_system/product_app/static/js/copilot.js";

// Exercise the actual controller: the low-level eligibility helper alone did not
// catch that staging invalidates a conversation turn's revision.
class Element {
  constructor() {
    this.children = []; this.listeners = {}; this.dataset = {}; this.style = {};
    this.attributes = {}; this.value = ""; this.textContent = ""; this.hidden = false;
    this.scrollHeight = 100; this.scrollTop = 0; this.clientHeight = 100;
    this.classList = { add() {}, remove() {} };
  }
  append(...children) { this.children.push(...children); }
  replaceChildren(...children) { this.children = children; }
  setAttribute(key, value) { this.attributes[key] = value; }
  addEventListener(name, action) { this.listeners[name] = action; }
  focus() {}
}
const descendants = element => [element, ...element.children.flatMap(descendants)];
// Accepted replacements leave candidate_choices. Restored historical cards
// still have the place name in the accepted timeline or geography.
for (const nameSource of ["timeline", "geography"]) {
  const elements = new Map();
  const element = selector => {
    if (!elements.has(selector)) elements.set(selector, new Element());
    return elements.get(selector);
  };
  globalThis.document = {querySelector:element,querySelectorAll:() => [],createElement:() => new Element()};
  globalThis.window = {addEventListener() {}};
  const candidateId = "pfeiffer_big_sur_state_park";
  const placeName = "Pfeiffer Big Sur State Park";
  const state = {session:{session_id:"synthetic",conversation_id:"conversation_fixture",trip_id:"trip",run_id:"run",
    revision:2,selected_day:5,draft:[]},workspace:{interaction:{enabled:true,provider:"deterministic"},
    timeline:nameSource === "timeline" ? [{stops:[{id:candidateId,name:placeName}]}] : [],
    geography:{plans:[{stops:{features:[
      {properties:{stop_id:"bixby",name:"Bixby Creek Bridge Viewpoint"}},
      ...(nameSource === "geography" ? [{properties:{stop_id:candidateId,name:placeName}}] : []),
    ]}}]},draft_capabilities:{operations:{replace_nearby:{enabled:true}},parent_targets:[candidateId],candidate_choices:[]}}};
  const conversation = {schema_version:"product-conversation-v1",conversation_id:"conversation_fixture",
    session_id:"synthetic",trip_id:"trip",run_id:"run",revision:1,turns:[{client_message_id:"turn-1",
      context_revision:0,state:"completed",user_message:"Replace Bixby with Pfeiffer.",assistant_message:"Suggestion only.",
      interpretation:{state:"proposal_ready",intents:[{type:"replace_nearby",target_stop_id:"bixby",candidate_id:candidateId}]}}]};
  let additions = 0;
  const controller = createCopilotController({getState:() => state,api:async () => ({session:state.session,conversation}),
    addDraft:async () => { additions++; },getPreviewState:() => ({status:"PARENT_RENDERED"}),
    toast(){},renderAll(){},navigate(){}});
  await controller.restore({announce:false});
  const rendered = descendants(element("#conversation"));
  assert.ok(rendered.some(node => node.textContent === `Replace Bixby Creek Bridge Viewpoint with ${placeName}.`),
    `A restored accepted replacement must use its ${nameSource} place name`);
  const unavailable = rendered.find(node => node.textContent === "Cannot add to draft");
  assert.equal(unavailable?.disabled,true,"The old suggestion must stay disabled after acceptance");
  await unavailable.listeners.click({});
  assert.equal(additions,0);
}
console.log("Historical accepted replacement cards retain place names and reject stale additions.");

for (const variant of ["exact", "changed_after_render", "unrelated", "disabled_capability"]) {
  const elements = new Map();
  const element = selector => {
    if (!elements.has(selector)) elements.set(selector, new Element());
    return elements.get(selector);
  };
  globalThis.document = {querySelector:element,querySelectorAll:() => [],createElement:() => new Element()};
  globalThis.window = {addEventListener() {}};
  const operation = {type:"replace_nearby",target:"bixby",parameters:{candidate_id:"pfeiffer"}};
  const state = {session:{session_id:"synthetic",conversation_id:"conversation_fixture",trip_id:"synthetic-trip",run_id:"synthetic-run",
    revision:0,selected_day:5,selected_stop_id:"bixby",draft:[]},
    workspace:{interaction:{enabled:true,provider:"deterministic"},timeline:[],
      draft_capabilities:{operations:{replace_nearby:{enabled:true}},parent_targets:["bixby"],
        candidate_choices:[{candidate_id:"pfeiffer",replaces_stop_id:"bixby"}]}}};
  const conversation = {schema_version:"product-conversation-v1",conversation_id:"conversation_fixture",
    session_id:"synthetic",trip_id:"synthetic-trip",run_id:"synthetic-run",revision:1,
    turns:[{client_message_id:"turn-1",context_revision:0,state:"completed",
    user_message:"Replace Bixby with Pfeiffer.",assistant_message:"Suggestion only.",
    interpretation:{state:"proposal_ready",intents:[{type:"replace_nearby",target_stop_id:"bixby",candidate_id:"pfeiffer"}]}}]};
  let additions = 0, previews = 0;
  const previewOptions = [];
  const controller = createCopilotController({getState:() => state,
    api:async () => ({session:state.session,conversation}),
    addDraft:async () => { additions++; state.session.revision++; state.session.draft=[structuredClone(operation)]; return true; },
    previewDraft:async (options) => { previews++; previewOptions.push(structuredClone(options)); return true; },getPreviewState:() => ({status:"PARENT_RENDERED"}),
    toast() {},renderAll() {},navigate() {}});
  await controller.restore({announce:false});
  const find = label => descendants(element("#conversation")).find(node => node.textContent === label);
  await find("Add to draft").listeners.click({});
  assert.equal(additions,1);
  assert.equal(state.session.revision,1);
  assert.equal(find("Added to draft").disabled,true,"A stale suggestion must not restage its operation");
  await find("Added to draft").listeners.click({});
  assert.equal(additions,1,"The stale Add handler must reject even if invoked directly");
  let preview = find("Preview this repair");
  assert.ok(preview && !preview.disabled,"The exact server-owned staged operation remains previewable");
  if (variant === "changed_after_render") {
    state.session.draft=[];
    await preview.listeners.click({currentTarget:preview});
    assert.equal(previews,0,"A captured button handler must recheck the current draft");
  } else if (variant !== "exact") {
    if (variant === "unrelated") state.session.draft.push({type:"keep_stop",target:"other",parameters:{}});
    else state.workspace.draft_capabilities.operations.replace_nearby.enabled=false;
    await controller.restore({announce:false});
    preview=find("Preview this repair");
    assert.ok(!preview || preview.disabled);
    assert.equal(previews,0);
  } else {
    await preview.listeners.click({currentTarget:preview});
    assert.equal(previews,1);
    assert.deepEqual(previewOptions,[{expectedCopilotOperation:operation}],"Controller must bind its exact operation at the final async submission boundary");
    assert.equal(additions,1,"Preview must not stage a duplicate operation");
  }
}
console.log("Actual controller staging preserves exact Preview; stale additions and changed drafts remain blocked.");

// The real map can return to the accepted parent after Undo while its old
// Copilot success banner remains visible. Exercise update(), used by renderAll.
for (const initialStatus of ["PREVIEW_RENDERED", "PREVIEW_INELIGIBLE", "PREVIEW_FAILED", "PREVIEW_BLOCKED"]) {
  for (const remainingDrafts of [0, 1]) {
    const elements = new Map();
    const element = selector => {
      if (!elements.has(selector)) elements.set(selector, new Element());
      return elements.get(selector);
    };
    globalThis.document = {querySelector:element,querySelectorAll:() => [],createElement:() => new Element()};
    globalThis.window = {addEventListener() {}};
    const state = {session:{session_id:"synthetic",conversation_id:"conversation_fixture",trip_id:"trip",run_id:"run",
      revision:1,selected_day:5,draft:[]},registry:{label:"Synthetic trip"},
      workspace:{interaction:{enabled:true,provider:"deterministic"},timeline:[]}};
    let preview = {status:initialStatus,planId:"child",mapPlanId:"child"};
    const controller = createCopilotController({getState:() => state,getPreviewState:() => preview,
      api:async () => ({session:state.session,conversation:{schema_version:"product-conversation-v1",
        conversation_id:"conversation_fixture",session_id:"synthetic",trip_id:"trip",run_id:"run",revision:0,turns:[]}}),
      getSelectedStop:() => null,toast() {},renderAll() {},navigate() {}});
    controller.previewStateChanged();
    if (initialStatus === "PREVIEW_RENDERED") assert.match(element("#copilot-lifecycle").textContent, /Preview ready/);
    state.session.revision++;
    state.session.draft = remainingDrafts ? [{type:"replace_nearby",target:"bixby",parameters:{candidate_id:"pfeiffer"}}] : [];
    preview = {status:"PARENT_RENDERED",planId:null,mapPlanId:null};
    controller.update();
    assert.match(element("#copilot-lifecycle").textContent,
      remainingDrafts ? /Draft changed.*Preview repair/ : /accepted itinerary.*No evaluated preview/,
      `${initialStatus} must not outlive the actual preview after a draft mutation`);
    assert.doesNotMatch(element("#copilot-lifecycle").textContent, /Preview ready/);
    if (remainingDrafts) {
      state.session.revision++;
      state.session.draft = [];
      controller.update();
      assert.match(element("#copilot-lifecycle").textContent,/accepted itinerary.*No evaluated preview/,
        "A second Undo must not leave a draft-ready message after the final draft is removed");
    }
    await controller.restore({announce:true});
    const restoredMessage = element("#copilot-lifecycle").textContent;
    assert.match(restoredMessage,/Conversation restored/);
    controller.update();
    assert.equal(element("#copilot-lifecycle").textContent,restoredMessage,
      "Parent rendering must not overwrite an unrelated chat lifecycle");
  }
}
console.log("Copilot preview status follows parent restoration without overwriting chat status.");

// Inspect the actual rendered suggestion cards and persistent preview region.
for (const [type, parameters, expected] of [
  ["set_stop_duration", {duration:{mode:"exact",preferred_minutes:75,minimum_minutes:75,maximum_minutes:75}}, "Set Griffith Observatory to exactly 75 minutes."],
  ["set_stop_order", {day:4,sequence_index:0}, "Place Griffith Observatory at position 1 on Day 4."],
  ["set_stop_role", {role:"scenic_stop"}, "Use Griffith Observatory as a scenic stop."],
  ["set_stop_time_window", {earliest_arrival:"09:00",latest_departure:"18:00"}, "At Griffith Observatory, arrive no earlier than 09:00 and depart no later than 18:00."],
  ["set_stop_time_window", {earliest_arrival:null,latest_departure:"18:00"}, "At Griffith Observatory, depart no later than 18:00."],
]) {
  const elements = new Map();
  const element = selector => {
    if (!elements.has(selector)) elements.set(selector, new Element());
    return elements.get(selector);
  };
  globalThis.document = {querySelector:element,querySelectorAll:() => [],createElement:() => new Element()};
  globalThis.window = {addEventListener() {}};
  const state = {session:{session_id:"synthetic",conversation_id:"conversation_fixture",trip_id:"trip",run_id:"run",
    revision:0,selected_day:4,draft:[]},registry:{label:"Synthetic trip"},
    workspace:{interaction:{enabled:true,provider:"deterministic"},timeline:[{stops:[{id:"griffith",name:"Griffith Observatory"}]}],
      draft_capabilities:{parent_targets:["griffith"],operations:{}},typed_edit_capabilities:{operations:{[type]:{enabled:true,preview_executable:true}}}}};
  const conversation = {schema_version:"product-conversation-v1",conversation_id:"conversation_fixture",session_id:"synthetic",
    trip_id:"trip",run_id:"run",revision:1,turns:[{client_message_id:"turn-1",context_revision:0,state:"completed",
      user_message:"Synthetic typed request",assistant_message:"Suggestion only.",interpretation:{state:"proposal_ready",
        intents:[{type,target_stop_id:"griffith",parameters}]}}]};
  let preview = {status:"PARENT_RENDERED"};
  const controller = createCopilotController({getState:() => state,getPreviewState:() => preview,
    api:async () => ({session:state.session,conversation}),getSelectedStop:() => null,toast(){},renderAll(){},navigate(){}});
  await controller.restore({announce:false});
  assert.ok(descendants(element("#conversation")).some(node=>node.textContent===expected),expected);
  preview = {status:"PREVIEW_FAILED",message:"Choose different positions: two stops cannot occupy position 1."};
  controller.previewStateChanged();
  assert.ok(element("#copilot-lifecycle").textContent.includes(preview.message));
  assert.ok(descendants(element("#conversation")).some(node=>node.textContent===preview.message));
  controller.update();
  assert.ok(element("#copilot-lifecycle").textContent.includes(preview.message),"Rerendering must retain the error reason");
}
console.log("Typed cards show precise values and preview failures remain visible.");

// Sending your own message must reveal the new turn even while reading history.
// Status updates must still preserve a deliberate scroll back up afterward.
{
  const elements = new Map();
  const element = selector => {
    if (!elements.has(selector)) elements.set(selector, new Element());
    return elements.get(selector);
  };
  globalThis.document = {querySelector:element,querySelectorAll:() => [],createElement:() => new Element()};
  globalThis.window = {addEventListener() {}};
  const state = {session:{session_id:"synthetic",conversation_id:"conversation_fixture",trip_id:"trip",run_id:"run",
    revision:0,selected_day:4,draft:[]},workspace:{interaction:{enabled:true,provider:"deterministic"},timeline:[]}};
  const conversation = {schema_version:"product-conversation-v1",conversation_id:"conversation_fixture",session_id:"synthetic",
    trip_id:"trip",run_id:"run",revision:0,turns:[]};
  let atSend, whileReading;
  const controller = createCopilotController({getState:() => state,
    api:async () => ({session:state.session,conversation}),getPreviewState:() => ({status:"PARENT_RENDERED"}),
    streamMessage:async (_path,{onEvent}) => {
      atSend = element("#conversation").scrollTop;
      element("#conversation").scrollTop = 200;
      onEvent({type:"assistant.status",state:"sending",message:"Reading the itinerary"});
      whileReading = element("#conversation").scrollTop;
      throw Object.assign(new Error("conversation_turn_limit_reached"),{code:"conversation_turn_limit_reached",status:422});
    },toast(){},renderAll(){},navigate(){}});
  controller.bind();
  await controller.restore({announce:false});
  element("#conversation").scrollHeight = 1000;
  element("#conversation").scrollTop = 0;
  element("#copilot-input").value = "Set Griffith Observatory duration to 75 minutes.";
  await element("#copilot-form").listeners.submit({preventDefault(){}});
  assert.equal(atSend,1000,"Own message submission must reveal the latest turn");
  assert.equal(whileReading,200,"Later status updates must preserve deliberate history reading");
}
console.log("Own messages reveal the latest turn; subsequent history reading remains stable.");

// Keep the controller's async boundary unresolved, so this regression does not
// depend on whether a local optimizer happens to finish before the next click.
for (const action of ["add", "preview"]) {
  for (const succeeded of [true, false]) {
    const elements = new Map();
    const element = selector => {
      if (!elements.has(selector)) elements.set(selector, new Element());
      return elements.get(selector);
    };
    globalThis.document = { querySelector: element, querySelectorAll: () => [], createElement: () => new Element() };
    globalThis.window = { addEventListener() {} };
    const operation = { type: "replace_nearby", target: "bixby", parameters: { candidate_id: "pfeiffer" } };
    const state = { session: { session_id: "synthetic", conversation_id: "conversation_fixture", trip_id: "trip", run_id: "run",
      revision: 0, selected_day: 5, selected_stop_id: "bixby", draft: [] }, registry: { label: "Synthetic trip" },
      workspace: { interaction: { enabled: true, provider: "deterministic" }, timeline: [],
        draft_capabilities: { operations: { replace_nearby: { enabled: true } }, parent_targets: ["bixby"],
          candidate_choices: [{ candidate_id: "pfeiffer", replaces_stop_id: "bixby" }] } } };
    const conversation = { schema_version: "product-conversation-v1", conversation_id: "conversation_fixture",
      session_id: "synthetic", trip_id: "trip", run_id: "run", revision: 1, turns: [{ client_message_id: "turn-1",
        context_revision: 0, state: "completed", user_message: "Replace Bixby with Pfeiffer.", assistant_message: "Suggestion only.",
        interpretation: { state: "proposal_ready", intents: [{ type: "replace_nearby", target_stop_id: "bixby", candidate_id: "pfeiffer" }] } }] };
    let finish, streamCalls = 0, entered = false;
    const pending = new Promise(resolve => { finish = resolve; });
    const controller = createCopilotController({ getState: () => state,
      api: async () => ({ session: state.session, conversation }),
      streamMessage: async () => { streamCalls++; throw new Error("Unexpected send during pending draft action"); },
      addDraft: async () => {
        if (action === "add") { entered = true; if (!await pending) return false; }
        state.session.revision++; state.session.draft = [structuredClone(operation)]; return true;
      },
      previewDraft: async () => { entered = true; return await pending; },
      getPreviewState: () => ({ status: "PARENT_RENDERED" }), getSelectedStop: () => null,
      toast() {}, renderAll() {}, navigate() {} });
    controller.bind();
    await controller.restore({ announce: false });
    controller.update();
    const find = label => descendants(element("#conversation")).find(node => node.textContent === label);
    const control = find(action === "add" ? "Add to draft" : "Preview this repair");
    const actionPromise = control.listeners.click({ currentTarget: control });
    await Promise.resolve(); await Promise.resolve();
    assert.equal(entered, true);
    assert.equal(element("#copilot-submit").disabled, true, `${action}: Send must be disabled while a draft action is pending`);
    element("#copilot-input").value = "Give Griffith Observatory 75 minutes.";
    controller.update();
    assert.equal(element("#copilot-submit").disabled, true, `${action}: render updates must preserve the Send block`);
    await element("#copilot-form").listeners.submit({ preventDefault() {} });
    assert.equal(streamCalls, 0, `${action}: a direct form submission must not race the draft revision`);
    assert.equal(element("#copilot-input").value, "Give Griffith Observatory 75 minutes.", "Keep unsent text for the user");
    finish(succeeded);
    await actionPromise;
    assert.equal(element("#copilot-submit").disabled, false, `${action}: success and failure both release the Send block`);
    assert.equal(element("#copilot-input").value, "Give Griffith Observatory 75 minutes.");
    assert.equal(streamCalls, 0);
  }
}
console.log("Four pending Add/Preview success/failure boundaries preserve unsent text and prevent concurrent dispatch.");



// A separate workspace action can finish while Add is restoring its transcript.
// Hold the actual controller's restore promise, then deliver an older/current
// response after Preview or Undo. Completion must preserve the newer lifecycle.
const delayedLifecycleCases = [];
for (const boundary of ["restore", "add_return"]) {
for (const shape of ["single", "batch"]) {
for (const outcome of ["PREVIEW_RUNNING", "PREVIEW_DATA_READY", "PREVIEW_RENDERED", "PREVIEW_INELIGIBLE", "PREVIEW_BLOCKED", "PREVIEW_FAILED", "UNDO_EMPTY", "UNDO_REMAINING"]) {
  for (const response of ["stale", "current"]) {
    const elements = new Map();
    const element = selector => {
      if (!elements.has(selector)) elements.set(selector, new Element());
      return elements.get(selector);
    };
    globalThis.document = {
      querySelector: element,
      querySelectorAll: selector => selector === ".copilot-proposal-actions button"
        ? descendants(element("#conversation")).filter(node => node.tagName === "button") : [],
      createElement: tag => new Element(tag),
    };
    globalThis.window = { addEventListener() {} };
    const operation = { type: "set_stop_duration", target: "griffith", parameters: {
      duration: { mode: "exact", minimum_minutes: 75, maximum_minutes: 75, preferred_minutes: 75 } } };
    const operations = shape === "single" ? [operation] : [
      {type:"set_stop_order",target:"kings",parameters:{day:4,sequence_index:0}},
      {type:"set_stop_order",target:"sequoia",parameters:{day:4,sequence_index:1}},
    ];
    const state = { session: { session_id: "synthetic", conversation_id: "conversation_fixture", trip_id: "trip", run_id: "run",
      revision: 0, selected_day: 5, accepted_plan_id: "accepted", draft: [] }, registry: { label: "Synthetic trip" },
      workspace: { interaction: { enabled: true, provider: "deterministic" }, timeline: [],
        draft_capabilities: { parent_targets: ["griffith", "kings", "sequoia"], operations: {}, executable_combinations: [{operations:["set_stop_order"],cardinality:2}] },
        typed_edit_capabilities: { operations: { set_stop_duration: { enabled: true, preview_executable: true }, set_stop_order: {enabled:true,preview_executable:true} } } } };
    const conversation = { schema_version: "product-conversation-v1", conversation_id: "conversation_fixture",
      session_id: "synthetic", trip_id: "trip", run_id: "run", revision: 1, turns: [{ client_message_id: "turn-1",
        context_revision: 0, state: "completed", user_message: "Give Griffith 75 minutes.", assistant_message: "Suggestion only.",
        interpretation: { state: "proposal_ready", intents: operations.map(op=>({type:op.type,target_stop_id:op.target,parameters:op.parameters})) } }] };
    let resolveRestore, finishAdd, apiCalls = 0, additions = 0;
    const held = new Promise(resolve => { resolveRestore = resolve; });
    const pendingAdd = new Promise(resolve => { finishAdd = resolve; });
    let preview = { status: "PARENT_RENDERED" }, controller;
    controller = createCopilotController({ getState: () => state,
      api: async () => { apiCalls++; return apiCalls === 1 ? { session: structuredClone(state.session), conversation } : held; },
      addDraft: async () => { additions++; state.session.revision++; state.session.draft = structuredClone(operations); if (boundary === "add_return") await pendingAdd; return true; },
      getPreviewState: () => preview, getSelectedStop: () => null, toast() {},
      renderAll() { controller.update(); }, navigate() {} });
    controller.bind();
    await controller.restore({ announce: false });
    const add = descendants(element("#conversation")).find(node => node.textContent === (shape === "single" ? "Add to draft" : "Add all to draft"));
    assert.ok(add && !add.disabled);
    const adding = add.listeners.click({ currentTarget: add });
    for (let i = 0; apiCalls < (boundary === "restore" ? 2 : 1) && i < 20; i++) await Promise.resolve();
    assert.equal(apiCalls, boundary === "restore" ? 2 : 1);
    assert.equal(additions, 1);
    assert.equal(element("#copilot-submit").disabled, true);
    const stagedSession = structuredClone(state.session);
    state.session.revision++;
    let expected;
    if (outcome.startsWith("UNDO_")) {
      preview = { status: "PREVIEW_RENDERED", planId: "prior-child", mapPlanId: "prior-child" };
      controller.previewStateChanged();
      state.session.draft = outcome === "UNDO_EMPTY" ? [] : shape === "batch" ? [structuredClone(operations[0])] : [{ type: "keep_stop", target: "other", parameters: {} }];
      preview = { status: "PARENT_RENDERED" };
      expected = outcome === "UNDO_EMPTY"
        ? "idle: Showing the accepted itinerary. No evaluated preview is active."
        : "Proposal ready: Draft changed. Run Preview repair to update the itinerary and map.";
    } else {
      preview = { status: outcome, planId: "child", mapPlanId: "child", message: "Independent evaluation result" };
    }
    controller.previewStateChanged();
    if (!expected) expected = element("#copilot-lifecycle").textContent;
    finishAdd();
    for (let i = 0; apiCalls < 2 && i < 20; i++) await Promise.resolve();
    assert.equal(apiCalls, 2);
    resolveRestore({ session: structuredClone(response === "stale" ? stagedSession : state.session), conversation });
    await adding;
    const actual = element("#copilot-lifecycle").textContent;
    assert.equal(additions, 1);
    assert.equal(state.session.revision, 2, "Late transcript restoration must not downgrade the workspace revision");
    assert.equal(state.session.accepted_plan_id, "accepted");
    assert.equal(element("#copilot-submit").disabled, false);
    assert.equal(state.session.draft.length, outcome === "UNDO_EMPTY" ? 0 : outcome === "UNDO_REMAINING" ? 1 : operations.length);
    delayedLifecycleCases.push({ boundary, shape, outcome, response, expected, actual, matched: actual === expected });
  }
}
}
}

assert.ok(delayedLifecycleCases.every(row => row.matched), JSON.stringify(delayedLifecycleCases.filter(row => !row.matched)));
console.log("64 deferred Add/restore cases preserve newer single/batch Preview and Undo lifecycle state.");


// A revision-safe chat must also wait for the outer app's Undo operation.
// Use the actual Undo function and controller with controlled asynchronous boundaries.
{
const appCode = readFileSync(new URL("../../src/itinerary_system/product_app/static/js/app.js", import.meta.url), "utf8");
const undoCode = appCode.slice(appCode.indexOf("async function undoDraft() {"), appCode.indexOf("async function previewDraft("));
assert.ok(undoCode.startsWith("async function undoDraft() {"));
const makeUndo = new Function("state", "copilot", "pendingSelection", "submitDraftMutation", "installDraftMutationSession",
  "syncPreviewRuntimeFromSession", "resetPoiDiscovery", "selectedDraftWaypoint", "loadDraftImpactPreview",
  "draftImpactRequestIsCurrent", "renderAll", "refreshMapEditDialog", "toast", "$", `${undoCode}\nreturn undoDraft;`);
const deferred = () => { let resolve, reject; const promise = new Promise((a, b) => { resolve = a; reject = b; }); return { promise, resolve, reject }; };
function harness(boundary, holdChat = false) {
  const gate = deferred(), elements = new Map(), quick = new Element(), dispatches = [], messages = [];
  const element = selector => { if (!elements.has(selector)) elements.set(selector, new Element()); return elements.get(selector); };
  globalThis.document = { querySelector: element, querySelectorAll: selector => selector === ".quick-prompts button" ? [quick] : [], createElement: () => new Element() };
  globalThis.window = { addEventListener() {} };
  const state = { undoPending: false, session: { session_id: "synthetic", conversation_id: "conversation_fixture", trip_id: "trip", run_id: "run",
    revision: 7, accepted_plan_id: "parent", selected_day: 4, selected_stop_id: "kings", draft: [{ type: "set_stop_order", target: "kings", parameters: { day: 4, sequence_index: 0 } }], proposal: null },
    mapInteraction: {}, typedImpact: { request: "impact_fixture" }, registry: { label: "Synthetic trip" },
    workspace: { interaction: { enabled: true, provider: "deterministic" }, timeline: [] } };
  const conversation = { schema_version: "product-conversation-v1", conversation_id: "conversation_fixture", session_id: "synthetic", trip_id: "trip", run_id: "run", revision: 0, turns: [] };
  let undoDispatches = 0, pendingSubmit, chatSignal;
  const controller = createCopilotController({ getState: () => state,
    api: async () => ({ session: structuredClone(state.session), conversation: structuredClone(conversation) }),
    streamMessage: async (_path, options) => {
      dispatches.push(structuredClone(options.body));
      chatSignal = options.signal;
      if (holdChat) await new Promise((_resolve, reject) => options.signal.addEventListener("abort",
        () => reject(new DOMException("Stopped locally", "AbortError")), { once: true }));
      const turn = { client_message_id: options.body.client_message_id, context_revision: options.body.expected_revision,
        user_message: options.body.message, assistant_message: "Review only.", state: "completed", interpretation: { state: "explanation_ready", intents: [] } };
      conversation.revision++; conversation.turns.push(turn);
      return { session: structuredClone(state.session), turn };
    }, getPreviewState: () => ({ status: "PARENT_RENDERED" }), getSelectedStop: () => null,
    toast: text => messages.push(text), renderAll() {}, navigate() {} });
  controller.bind();
  element("#copilot-form").requestSubmit = () => { pendingSubmit = element("#copilot-form").listeners.submit({ preventDefault() {} }); };
  const undo = makeUndo(state, controller, boundary === "selection" ? gate.promise : null,
    async () => { undoDispatches++; if (boundary === "request") await gate.promise; return { session: { ...structuredClone(state.session), revision: state.session.revision + 1, draft: [], proposal: null } }; },
    incoming => { state.session = incoming; }, () => {}, () => {}, () => null,
    async () => { if (boundary === "impact") await gate.promise; }, () => true,
    () => controller.update(), () => {}, text => messages.push(text), element);
  return { gate, state, controller, undo, element, quick, dispatches, messages, undoDispatches: () => undoDispatches,
    chatSignal: () => chatSignal,
    submit: async mode => { if (mode === "form") await element("#copilot-form").listeners.submit({ preventDefault() {} });
      else { element("#copilot-input").listeners.keydown({ key: "Enter", preventDefault() {} }); await pendingSubmit; } } };
}

let undoCases = 0;
for (const boundary of ["selection", "request", "impact"]) {
  for (const outcome of ["success", "failure"]) {
    for (const submit of ["form", "enter"]) {
      const h = harness(boundary);
      await h.controller.restore({ announce: false }); h.controller.update();
      const pending = h.undo();
      await Promise.resolve(); await Promise.resolve();
      h.element("#copilot-input").value = "Give Griffith Observatory 75 minutes.";
      h.controller.update();
      assert.equal(h.element("#copilot-submit").disabled, true);
      assert.equal(h.element("#copilot-input").attributes["aria-busy"], "true");
      assert.equal(h.quick.disabled, true);
      await h.submit(submit);
      assert.equal(h.element("#copilot-input").value, "Give Griffith Observatory 75 minutes.");
      assert.equal(h.dispatches.length, 0);
      if (outcome === "failure" && boundary !== "selection") h.gate.reject(new Error("Controlled Undo failure"));
      else h.gate.resolve(outcome === "success");
      await pending;
      assert.equal(h.element("#copilot-submit").disabled, false);
      await h.submit(submit);
      assert.equal(h.dispatches.length, 1);
      assert.equal(h.dispatches[0].expected_revision, outcome === "success" || boundary === "impact" ? 8 : 7);
      assert.equal(h.undoDispatches(), boundary === "selection" && outcome === "failure" ? 0 : 1);
      undoCases++;
    }
    const h = harness(boundary);
    await h.controller.restore({ announce: false }); h.controller.update();
    const first = h.undo();
    await Promise.resolve(); await Promise.resolve();
    const second = h.undo();
    if (outcome === "failure" && boundary !== "selection") h.gate.reject(new Error("Controlled Undo failure"));
    else h.gate.resolve(outcome === "success");
    await first; await second;
    assert.equal(h.undoDispatches(), boundary === "selection" && outcome === "failure" ? 0 : 1);
    assert.equal(h.state.session.revision, outcome === "success" || boundary === "impact" ? 8 : 7);
    assert.equal(h.element("#copilot-submit").disabled, false);
    undoCases++;
  }
}
for (const boundary of ["request", "impact"]) {
  const h = harness(boundary, true);
  await h.controller.restore({ announce: false }); h.controller.update();
  h.element("#copilot-input").value = "Review this trip";
  const chat = h.submit("form");
  await Promise.resolve(); await Promise.resolve();
  const pending = h.undo();
  await Promise.resolve(); await Promise.resolve(); h.controller.update();
  assert.equal(h.element("#copilot-submit").disabled, false);
  assert.equal(h.element("#copilot-submit").dataset.mode, "stop");
  await h.submit("form"); await chat;
  assert.equal(h.chatSignal().aborted, true);
  assert.equal(h.element("#copilot-submit").disabled, true);
  h.gate.resolve(true); await pending;
  assert.equal(h.undoDispatches(), 1);
  assert.equal(h.state.session.revision, 8);
  assert.equal(h.element("#copilot-submit").disabled, false);
  undoCases++;
}
assert.equal(undoCases, 20);
console.log("20 outer Undo/controller cases preserve input, revision binding, duplicate protection and Stop.");
}


// A new trip-length proposal must never borrow an older draft's preview badge.
{
 const elements=new Map();
 const element=selector=>{if(!elements.has(selector))elements.set(selector,new Element());return elements.get(selector);};
 globalThis.document={querySelector:element,querySelectorAll:()=>[],createElement:()=>new Element()};
 globalThis.window={addEventListener(){}};
 const params={start_city:null,coastal:false,required_place_ids:[],excluded_place_ids:[],avoid_regions:[],verify_official:false,max_daily_drive_minutes:720,total_days:14,pace:"balanced",interests:[],profile_updates:[]};
 const draft=[{type:"replan_trip",target:"trip",parameters:{...params,total_days:10}}];
 const state={session:{session_id:"synthetic",conversation_id:"conversation_fixture",trip_id:"trip",run_id:"run",revision:1,selected_day:1,draft},registry:{label:"Synthetic trip"},workspace:{interaction:{enabled:true,provider:"deterministic"},timeline:[],draft_capabilities:{operations:{replan_trip:{enabled:true,preview_executable:true}},parent_targets:[]}}};
 const conversation={schema_version:"product-conversation-v1",conversation_id:"conversation_fixture",session_id:"synthetic",trip_id:"trip",run_id:"run",revision:1,turns:[{client_message_id:"duration-turn",context_revision:1,state:"completed",user_message:"我想花兩週去旅行",assistant_message:"十四天提案",interpretation:{state:"proposal_ready",intents:[{type:"replan_trip",parameters:params}]}}]};
 const controller=createCopilotController({getState:()=>state,getPreviewState:()=>({status:"PREVIEW_RENDERED",planId:"old10",mapPlanId:"old10"}),api:async()=>({session:state.session,conversation}),getSelectedStop:()=>null,toast(){},renderAll(){},navigate(){}});
 await controller.restore({announce:false});controller.previewStateChanged();
 assert.ok(!descendants(element("#conversation")).some(n=>n.textContent==="Preview ready"));
 assert.ok(descendants(element("#conversation")).some(n=>n.textContent.includes("Plan 14 days")));
 assert.ok(element("#copilot-lifecycle").textContent.includes("This trip proposal has not been previewed"));
 assert.deepEqual(state.session.draft,draft);
}
console.log("New duration proposals cannot inherit an unrelated evaluated-preview badge.");

// UI copy and place labels must not alter the exact parameters sent to the draft.
for (const explicitPreferences of [false, true]) {
 const elements = new Map();
 const element = selector => { if (!elements.has(selector)) elements.set(selector, new Element()); return elements.get(selector); };
 globalThis.document = {querySelector:element,querySelectorAll:()=>[],createElement:()=>new Element()};
 globalThis.window = {addEventListener(){}};
 const params = {start_city:"Los Angeles",coastal:true,required_place_ids:["hearst_castle","channel_islands","redwood_national_park"],excluded_place_ids:["kings_canyon_national_park"],avoid_regions:[],verify_official:true,max_daily_drive_minutes:explicitPreferences ? 300 : 720,total_days:explicitPreferences ? 14 : 10,pace:explicitPreferences ? "relaxed" : "balanced",interests:explicitPreferences ? ["nature","culture","coast"] : [],profile_updates:explicitPreferences ? ["pace","interests","max_daily_drive_minutes","coastal"] : []};
 const originalParameters = structuredClone(params);
 const state = {session:{session_id:"synthetic",conversation_id:"conversation_fixture",trip_id:"trip",run_id:"run",revision:0,selected_day:1,draft:[]},registry:{label:"Synthetic trip"},workspace:{interaction:{enabled:true,provider:"deterministic"},timeline:[],draft_capabilities:{operations:{replan_trip:{enabled:true,preview_executable:true}},parent_targets:[]}}};
 const conversation = {schema_version:"product-conversation-v1",conversation_id:"conversation_fixture",session_id:"synthetic",trip_id:"trip",run_id:"run",revision:0,turns:[{client_message_id:"english-plan",context_revision:0,state:"completed",user_message:"Plan a coastal trip from LA including Hearst, Channel Islands and Redwood, excluding Kings Canyon.",assistant_message:"Suggestion only.",interpretation:{state:"proposal_ready",intents:[{type:"replan_trip",parameters:params}]}}]};
 const additions = [];
 const previews = [];
 const controller = createCopilotController({getState:()=>state,getPreviewState:()=>({status:"PARENT_RENDERED"}),api:async()=>({session:state.session,conversation}),addDraft:async intent=>{additions.push(structuredClone(intent));state.session.draft=[copilotOperationForIntent(intent)];state.session.revision++;return true;},previewDraft:async options=>{previews.push(structuredClone(options));return true;},getSelectedStop:()=>null,toast(){},renderAll(){},navigate(){}});
 await controller.restore({announce:false}); controller.previewStateChanged();
 const nodes = descendants(element("#conversation"));
 const summary = nodes.find(node=>node.textContent.startsWith(`Plan ${params.total_days} days`))?.textContent;
 assert.ok(summary?.includes("Required: Hearst Castle, Channel Islands, Redwood National Park"));
 assert.ok(summary.includes("Excluded: Kings Canyon National Park"));
 assert.ok(summary.includes(explicitPreferences ? "Pace: relaxed" : "keep accepted pace (balanced for the first plan)"));
 assert.ok(summary.includes(explicitPreferences ? "Daily driving limit: 300 minutes" : "keep accepted limit (720 minutes for the first plan)"));
 assert.ok(summary.includes(explicitPreferences ? "Interests: nature, culture, coast" : "Interests: keep accepted interests"));
 assert.ok(summary.includes("nothing has been applied"));
 assert.ok(!/[\u3400-\u9fff]/.test(summary + element("#copilot-lifecycle").textContent));
 const preview = nodes.find(node=>node.textContent === "Preview this repair");
 const add = nodes.find(node=>node.textContent === "Add to draft");
 assert.equal(preview.attributes["aria-label"], "Preview this repair: trip plan");
 assert.equal(add.attributes["aria-label"], "Add to draft: trip plan");
 assert.deepEqual(params, originalParameters);
 assert.deepEqual(state.session.draft, []);
 await add.listeners.click({});
 assert.deepEqual(additions, [{type:"replan_trip",parameters:originalParameters}]);
 assert.deepEqual(state.session.draft, [{type:"replan_trip",target:"trip",parameters:originalParameters}]);
 const stagedNodes = descendants(element("#conversation"));
 assert.equal(stagedNodes.find(node=>node.textContent === "Added to draft").attributes["aria-label"], "Added to draft: trip plan");
 await stagedNodes.find(node=>node.textContent === "Preview this repair").listeners.click({});
 assert.deepEqual(previews, [{expectedCopilotOperation:{type:"replan_trip",target:"trip",parameters:originalParameters}}]);
 assert.deepEqual(params, originalParameters);
}
console.log("English trip summaries and concise actions preserve exact typed parameters.");
