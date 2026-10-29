// Independent actual-controller counterexamples. Controlled I/O and a minimal DOM
// make async boundaries deterministic; these are not browser/provider acceptance.
import assert from "node:assert/strict";
import test from "node:test";
import { createCopilotController } from "../../src/itinerary_system/product_app/static/js/copilot.js";

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
  focus() {} close() {} showModal() {}
}
const descendants = element => [element, ...element.children.flatMap(descendants)];
const deferred = () => {
  let resolve, reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
};
const tick = () => new Promise(resolve => setImmediate(resolve));
const session = (id = "session_a", conversationId = "conversation_a", revision = 24) => ({
  session_id: id, conversation_id: conversationId, revision, accepted_plan_id: "parent",
  trip_id: "synthetic-trip", run_id: "synthetic-run",
  draft: [], proposal: null, selected_day: 5, selected_stop_id: "bixby",
});
const conversation = (owner, turns = [], revision = turns.length) => ({
  schema_version: "product-conversation-v1", conversation_id: owner.conversation_id, session_id: owner.session_id,
  trip_id: owner.trip_id, run_id: owner.run_id, revision, turns,
});
const reply = (owner, turns = []) => ({ session: structuredClone(owner), conversation: conversation(owner, turns) });
const canonicalTurn = (envelope, text = "OLD_DELETED_TURN") => ({
  client_message_id: envelope.client_message_id, user_message: envelope.message,
  assistant_message: text, state: "completed", context_revision: envelope.expected_revision,
  interpretation: { state: "proposal_ready", intents: [
    { type: "replace_nearby", target_stop_id: "bixby", candidate_id: "pfeiffer" },
  ] },
});

function harness(api) {
  const elements = new Map();
  const element = selector => { if (!elements.has(selector)) elements.set(selector, new Element()); return elements.get(selector); };
  globalThis.document = { querySelector: element, querySelectorAll: () => [], createElement: () => new Element() };
  globalThis.window = { addEventListener() {}, confirm: () => true };
  const h = { state: { session: session(), workspace: {
    interaction: { enabled: true, provider: "deterministic" }, timeline: [],
    draft_capabilities: { operations: { replace_nearby: { enabled: true } }, parent_targets: ["bixby"],
      candidate_choices: [{ candidate_id: "pfeiffer", replaces_stop_id: "bixby" }] },
  } }, element, calls: [], notices: [], stream: deferred(), streamCalls: 0, renderCalls: 0, addCalls: 0 };
  h.controller = createCopilotController({ getState: () => h.state,
    api: (path, options) => { h.calls.push({ path, options }); return api(h, path, options); },
    streamMessage: (path, options) => { h.streamCalls++; h.streamPath = path; h.streamOptions = options; return h.stream.promise; },
    getPreviewState: () => ({ status: "PARENT_RENDERED" }),
    addDraft: async intent => { h.addCalls++; return h.stageHook ? h.stageHook(intent) : true; },
    previewDraft: async options => { h.previewCalls=(h.previewCalls || 0)+1; h.previewOptions=options; return true; },
    toast: text => h.notices.push(text), renderAll: () => { h.renderCalls++; },
    syncPreviewRuntime: () => { h.syncedPlan = h.state.session.accepted_plan_id; },
  });
  h.controller.bind();
  h.submit = message => { element("#copilot-input").value = message; return element("#copilot-form").listeners.submit({ preventDefault() {} }); };
  h.text = () => descendants(element("#conversation")).map(node => node.textContent).join(" ");
  h.buttons = () => descendants(element("#conversation")).filter(node => typeof node.listeners.click === "function");
  return h;
}

for (const selector of ["#delete-current-conversation", "#delete-all-conversations"]) {
  test(`${selector}: deleted completed turn must not reappear when an older refresh loses its epoch`, async () => {
    const oldRead = deferred(); let reads = 0;
    const recreated = session("session_a", selector.includes("-all-") ? "conversation_a" : "conversation_new");
    const h = harness(async (_h, _path, options) => {
      if (options?.method === "DELETE") return { session: session("session_a", selector.includes("-all-") ? "conversation_a" : null), deleted: true, deleted_count: 1 };
      reads++;
      return reads === 1 ? oldRead.promise : reply(recreated);
    });
    const sending = h.submit("OLD_USER_MESSAGE");
    const turn = canonicalTurn(h.streamOptions.body);
    h.stream.resolve({ session: session(), turn });
    await tick();
    assert.equal(reads, 1, "Completed response must really be awaiting its original transcript GET");
    await h.element(selector).listeners.click();
    assert.deepEqual(h.state.session, recreated);
    assert.doesNotMatch(h.text(), /OLD_DELETED_TURN/);
    oldRead.resolve(reply(session(), [turn]));
    await sending;
    assert.equal(h.streamCalls, 1);
    assert.deepEqual(h.state.session, recreated);
    assert.doesNotMatch(h.text(), /OLD_DELETED_TURN|OLD_USER_MESSAGE/, "A superseded restore is not permission to reinsert a deleted turn");
    assert.ok(!h.buttons().some(button => button.textContent === "Add to draft" && !button.disabled),
      "A deleted same-revision proposal must not regain an executable action");
  });
}

test("confirmed deletion clears a pending local turn without asserting provider cancellation", async () => {
  const recreated = session("session_a", "conversation_new");
  const h = harness(async (_h, _path, options) => options?.method === "DELETE"
    ? { session: session("session_a", null), deleted: true } : reply(recreated));
  const sending = h.submit("OLD_PENDING_USER_MESSAGE");
  await h.element("#delete-current-conversation").listeners.click();
  const afterDelete = h.text();
  h.streamOptions.onEvent({ type: "assistant.status", state: "interpreting", message: "OLD_PENDING_STATUS" });
  const afterLateStatus = h.text();
  h.stream.resolve({ session: session(), turn: canonicalTurn(h.streamOptions.body) });
  await sending;
  assert.equal(h.streamCalls, 1);
  assert.deepEqual(h.state.session, recreated);
  assert.doesNotMatch(afterDelete, /OLD_PENDING_USER_MESSAGE/);
  assert.doesNotMatch(afterLateStatus, /OLD_PENDING_USER_MESSAGE|OLD_PENDING_STATUS/);
  assert.ok(!h.notices.some(text => /provider.*(cancelled|not billed)/i.test(text)));
});

test("late completed refresh from session A cannot append its turn to session B", async () => {
  const oldRead = deferred(); let reads = 0;
  const other = session("session_b", "conversation_b");
  const otherTurn = { client_message_id: "b", user_message: "B_ONLY", assistant_message: "B_ONLY", state: "completed" };
  const h = harness(async () => (++reads === 1 ? oldRead.promise : reply(other, [otherTurn])));
  const sending = h.submit("A_ONLY_USER");
  const turn = canonicalTurn(h.streamOptions.body, "A_ONLY_ANSWER");
  h.stream.resolve({ session: session(), turn });
  await tick();
  assert.equal(reads, 1);
  h.state.session = structuredClone(other);
  assert.equal(await h.controller.restore({ announce: false }), true);
  oldRead.resolve(reply(session(), [turn]));
  await sending;
  assert.deepEqual(h.state.session, other);
  assert.match(h.text(), /B_ONLY/);
  assert.doesNotMatch(h.text(), /A_ONLY_ANSWER|A_ONLY_USER/);
});

test("same workspace revision does not authorize a foreign conversation envelope", async () => {
  const owner = session(), foreign = session("session_b", "conversation_b");
  const turn = canonicalTurn({ client_message_id: "foreign", message: "FOREIGN_TURN", expected_revision: owner.revision });
  const h = harness(async () => ({ session: owner, conversation: conversation(foreign, [turn]) }));
  const restored = await h.controller.restore({ announce: false });
  assert.equal(restored, false);
  assert.deepEqual(h.state.session, owner);
  assert.doesNotMatch(h.text(), /FOREIGN_TURN/);
});

test("late failed restore cannot overwrite a newer valid restore lifecycle", async () => {
  const oldRead = deferred(); let reads = 0;
  const owner = session();
  const h = harness(async () => (++reads === 1 ? oldRead.promise : reply(owner)));
  const old = h.controller.restore();
  assert.equal(await h.controller.restore(), true);
  oldRead.reject(new Error("OLD_RESTORE_FAILURE"));
  assert.equal(await old, false);
  assert.equal(h.element("#copilot-lifecycle").dataset.state, "idle");
  assert.ok(!h.notices.some(text => text.includes("OLD_RESTORE_FAILURE")));
});

test("a completed historical same-session turn cannot regain Add after a newer Undo revision", async () => {
  const current = session("session_a", "conversation_a", 25);
  const turn = canonicalTurn({ client_message_id: "old", message: "Historical", expected_revision: 24 });
  const h = harness(async () => reply(current, [turn]));
  h.state.session = current;
  assert.equal(await h.controller.restore({ announce: false }), true);
  assert.ok(!h.buttons().some(button => button.textContent === "Add to draft" && !button.disabled));
  assert.match(h.text(), /older workspace state/);
});

test("completed canonical turn remains visible after a genuine offline transcript refresh without resending", async () => {
  const h = harness(async () => { throw new TypeError("Failed to fetch"); });
  const sending = h.submit("KEEP_CONFIRMED_REPLY");
  h.stream.resolve({ session: session(), conversation_revision: 1, turn: canonicalTurn(h.streamOptions.body, "CONFIRMED_OFFLINE_REPLY") });
  await sending;
  assert.match(h.text(), /CONFIRMED_OFFLINE_REPLY/);
  assert.equal(h.streamCalls, 1);
  assert.ok(h.notices.some(text => text.includes("No retry was sent")));
  assert.equal(h.element("#copilot-lifecycle").dataset.state, "proposal_ready");
});

test("HTTP rejection is not an offline fallback license to display a deleted response", async () => {
  const h = harness(async () => { throw Object.assign(new Error("unknown_conversation"), { status: 404 }); });
  const sending = h.submit("DO_NOT_REINSERT");
  h.stream.resolve({ session: session(), conversation_revision: 1, turn: canonicalTurn(h.streamOptions.body, "DELETED_HTTP_REPLY") });
  await sending;
  assert.doesNotMatch(h.text(), /DELETED_HTTP_REPLY/);
  assert.equal(h.streamCalls, 1);
  assert.equal(h.element("#copilot-lifecycle").dataset.state, "failed");
});

test("captured proposal action cannot act after its conversation has been deleted", async () => {
  const owner = session();
  const turn = canonicalTurn({ client_message_id: "old", message: "Old proposal", expected_revision: 24 });
  let deleted = false;
  const h = harness(async (_h, _path, options) => {
    if (options?.method === "DELETE") { deleted = true; return { session: owner, deleted: true }; }
    return reply(owner, deleted ? [] : [turn]);
  });
  assert.equal(await h.controller.restore({ announce: false }), true);
  const add = h.buttons().find(button => button.textContent === "Add to draft");
  assert.ok(add && !add.disabled);
  await h.element("#delete-current-conversation").listeners.click();
  await add.listeners.click({});
  assert.equal(h.addCalls, 0);
  assert.match(h.notices.at(-1), /older workspace state/);
});

test("old SSE callback cannot update an explicit Retry reusing the same message identity", async () => {
  const h = harness(async () => reply(session()));
  const first = h.submit("RETRY_SAME_ID");
  const firstOptions = h.streamOptions;
  h.stream.reject(Object.assign(new Error("Aborted"), { name: "AbortError" }));
  await first;
  const retry = h.buttons().find(button => button.textContent === "Retry");
  assert.ok(retry);
  h.stream = deferred();
  const second = retry.listeners.click({});
  const secondOptions = h.streamOptions;
  assert.equal(secondOptions.body.client_message_id, firstOptions.body.client_message_id);
  firstOptions.onEvent({ type: "assistant.status", state: "interpreting", message: "OLD_ATTEMPT_CALLBACK" });
  const afterOldCallback = h.text();
  h.stream.resolve({ session: session(), conversation_revision: 1, turn: canonicalTurn(secondOptions.body, "RETRY_COMPLETED") });
  await second;
  assert.doesNotMatch(afterOldCallback, /OLD_ATTEMPT_CALLBACK/);
  assert.equal(h.streamCalls, 2, "Only the user's explicit Retry starts a second transport request");
});

test("first legitimate conversation binding is accepted without an itinerary revision increment", async () => {
  const owner = session("session_a", "conversation_created");
  const h = harness(async () => reply(owner));
  h.state.session = session("session_a", null);
  assert.equal(await h.controller.restore({ announce: false }), true);
  assert.deepEqual(h.state.session, owner);
});

for (const [field, value] of [["schema_version", "foreign"], ["conversation_id", "conversation_foreign"],
  ["trip_id", "foreign-trip"], ["run_id", "foreign-run"], ["revision", -1], ["turns", {}]]) {
  test(`invalid conversation ${field} is rejected before session adoption`, async () => {
    const incoming = session("session_a", "conversation_a", 25);
    const payload = reply(incoming);
    payload.conversation[field] = value;
    const h = harness(async () => payload);
    const original = structuredClone(h.state.session);
    assert.equal(await h.controller.restore({ announce: false }), false);
    assert.deepEqual(h.state.session, original);
  });
}

test("new message waits for authoritative post-delete restoration without being discarded or sent", async () => {
  const restored = deferred();
  const h = harness(async (_h, _path, options) => options?.method === "DELETE"
    ? { session: session("session_a", null), deleted: true } : restored.promise);
  const deleting = h.element("#delete-current-conversation").listeners.click();
  await tick();
  assert.equal(h.element("#copilot-submit").disabled, true);
  await h.submit("PRESERVE_TYPED_MESSAGE");
  assert.equal(h.streamCalls, 0);
  assert.equal(h.element("#copilot-input").value, "PRESERVE_TYPED_MESSAGE");
  assert.ok(h.notices.some(text => text.includes("Your message has not been sent")));
  restored.resolve(reply(session("session_a", "conversation_new")));
  await deleting;
  assert.equal(h.element("#copilot-submit").disabled, false);
  assert.equal(h.element("#copilot-input").value, "PRESERVE_TYPED_MESSAGE");
});

test("a captured Retry from a deleted conversation cannot dispatch another request", async () => {
  const h = harness(async (_h, _path, options) => options?.method === "DELETE"
    ? { session: session("session_a", null), deleted: true } : reply(session("session_a", "conversation_new")));
  const sending = h.submit("OLD_RETRY_MESSAGE");
  h.stream.reject(Object.assign(new Error("Aborted"), { name: "AbortError" }));
  await sending;
  const retry = h.buttons().find(button => button.textContent === "Retry");
  assert.ok(retry);
  await h.element("#delete-current-conversation").listeners.click();
  await retry.listeners.click({});
  assert.equal(h.streamCalls, 1);
  assert.match(h.notices.at(-1), /previous conversation/);
});

for (const field of ["trip_id", "run_id"]) {
  test(`a matching conversation envelope cannot change the current session's immutable ${field}`, async () => {
    const incoming = { ...session("session_a", "conversation_a", 25), [field]: "foreign-owner" };
    const h = harness(async () => reply(incoming));
    const original = structuredClone(h.state.session);
    assert.equal(await h.controller.restore({ announce: false }), false);
    assert.deepEqual(h.state.session, original);
  });
}

for (const combined of [false,true]) {
  for (const action of ["add","preview"]) {
    for (const field of ["session_id","conversation_id","trip_id","run_id","accepted_plan_id"]) {
      test(`${combined ? "combined" : "single"} captured ${action} rejects changed ${field}`, async () => {
        const turn=canonicalTurn({client_message_id:"captured",message:"Captured edit",expected_revision:24});
        if(combined) turn.interpretation.intents.push({type:"set_stop_duration",target_stop_id:"griffith",parameters:{
          duration:{mode:"exact",preferred_minutes:90,minimum_minutes:90,maximum_minutes:90}}});
        const h=harness(async () => reply(h.state.session,[turn]));
        h.state.workspace.draft_capabilities.parent_targets.push("griffith");
        h.state.workspace.draft_capabilities.executable_combinations=[{operations:["replace_nearby","set_stop_duration"],cardinality:2}];
        h.state.workspace.typed_edit_capabilities={operations:{set_stop_duration:{enabled:true,preview_executable:true}}};
        await h.controller.restore({announce:false});
        const label=action === "add" ? (combined ? "Add all to draft" : "Add to draft")
          : (combined ? "Preview combined repair" : "Preview this repair");
        const control=h.buttons().find(button => button.textContent === label);
        assert.ok(control && !control.disabled);
        h.state.session[field]="foreign_owner";
        await control.listeners.click({currentTarget:control});
        assert.equal(h.addCalls,0);
        assert.equal(h.previewCalls || 0,0);
        assert.deepEqual(h.state.session.draft,[]);
      });
    }
  }
  for (const change of ["none","conversation_id","session_id","trip_id","run_id","accepted_plan_id"]) {
    test(`${combined ? "combined" : "single"} queued preview rechecks ${change} after async staging`, async () => {
      const staged=deferred();
      const turn=canonicalTurn({client_message_id:"queued",message:"Queued edit",expected_revision:24});
      if (combined) turn.interpretation.intents.push({type:"set_stop_duration",target_stop_id:"griffith",parameters:{
        duration:{mode:"exact",preferred_minutes:90,minimum_minutes:90,maximum_minutes:90}}});
      const h=harness(async () => reply(h.state.session,[turn]));
      h.state.workspace.draft_capabilities.parent_targets.push("griffith");
      h.state.workspace.draft_capabilities.executable_combinations=[{operations:["replace_nearby","set_stop_duration"],cardinality:2}];
      h.state.workspace.typed_edit_capabilities={operations:{set_stop_duration:{enabled:true,preview_executable:true}}};
      h.stageHook=() => staged.promise;
      assert.equal(await h.controller.restore({announce:false}),true);
      const preview=h.buttons().find(button => button.textContent === (combined ? "Preview combined repair" : "Preview this repair"));
      assert.ok(preview && !preview.disabled);
      const action=preview.listeners.click({currentTarget:preview});
      await tick();
      assert.equal(h.addCalls,1);
      h.state.session.revision++;
      h.state.session.draft=turn.interpretation.intents.map(intent => ({type:intent.type,target:intent.target_stop_id,
        parameters:intent.type === "replace_nearby" ? {candidate_id:intent.candidate_id} : intent.parameters}));
      if(change !== "none") h.state.session[change]="foreign_owner";
      staged.resolve(true);
      await action;
      assert.equal(h.previewCalls || 0,change === "none" ? 1 : 0,"A queued action cannot migrate to another chat, trip or accepted baseline");
      assert.equal(h.addCalls,1,"No duplicate staging or automatic retry");
      assert.equal(h.state.session.draft.length,combined ? 2 : 1,"Preserve the already committed draft");
    });
  }
}

const accepted = () => ({ ...session(), revision: 25, accepted_plan_id: "accepted_child" });
const projection = owner => ({ session: structuredClone(owner), registry: { run_id: owner.run_id }, workspace: {
  schema_version: "product-dashboard-data-v1", interaction: { enabled: true, provider: "deterministic" },
  geography: { plans: [{ plan_id: owner.accepted_plan_id, content_hash: "child_hash" }] },
  accepted_plan_pointer: { accepted_plan_id: owner.accepted_plan_id, accepted_plan_content_hash: "child_hash" },
} });

test("conversation adoption installs the accepted map projection before advancing its session", async () => {
  const incoming = accepted(), full = projection(incoming);
  const h = harness(async (_h, path) => path.endsWith("/conversation") ? reply(incoming) : full);
  assert.equal(await h.controller.restore({ announce: false }), true);
  assert.deepEqual(h.state.session, incoming);
  assert.deepEqual(h.state.workspace, full.workspace);
  assert.deepEqual(h.state.registry, full.registry);
  assert.equal(h.calls.length, 2, "Only a changed accepted plan requires a complete workspace read");
  assert.ok(h.renderCalls > 0, "The newer map/context must render without a reload");
  assert.equal(h.syncedPlan, incoming.accepted_plan_id, "Preview/map readiness must follow the accepted session");
});

for (const mismatch of ["missing workspace", "wrong pointer", "wrong geometry hash", "wrong revision", "wrong conversation"]) {
  test(`invalid accepted projection (${mismatch}) cannot advance session or map`, async () => {
    const incoming = accepted(), full = projection(incoming);
    if (mismatch === "missing workspace") delete full.workspace;
    if (mismatch === "wrong pointer") full.workspace.accepted_plan_pointer.accepted_plan_id = "other";
    if (mismatch === "wrong geometry hash") full.workspace.geography.plans[0].content_hash = "wrong";
    if (mismatch === "wrong revision") full.session.revision++;
    if (mismatch === "wrong conversation") full.session.conversation_id = "rebound";
    const h = harness(async (_h, path) => path.endsWith("/conversation") ? reply(incoming) : full);
    const before = structuredClone(h.state);
    assert.equal(await h.controller.restore({ announce: false }), false);
    assert.deepEqual(h.state, before);
  });
}

test("a selection that finishes during projection recovery wins over the older conversation read", async () => {
  const incoming = accepted(), pending = deferred();
  const h = harness(async (_h, path) => path.endsWith("/conversation") ? reply(incoming) : pending.promise);
  const restoring = h.controller.restore({ announce: false });
  await tick();
  const newer = { ...session(), revision: 26, selected_day: 6 };
  h.state.session = newer;
  const workspace = h.state.workspace;
  pending.resolve(projection(incoming));
  assert.equal(await restoring, false);
  assert.deepEqual(h.state.session, newer);
  assert.equal(h.state.workspace, workspace);
});

test("stale chat recovery clears its temporary Reading bubble and shows the current trip", async () => {
  const incoming = accepted(), full = projection(incoming);
  const h = harness(async (_h, path) => path.endsWith("/conversation") ? reply(incoming) : full);
  const sending = h.submit("Give Griffith up to 90 minutes.");
  h.streamOptions.onEvent({ type: "message.accepted" });
  h.stream.reject(new Error("stale_revision"));
  await sending;
  assert.deepEqual(h.state.session, incoming);
  assert.deepEqual(h.state.workspace, full.workspace);
  assert.doesNotMatch(h.text(), /Reading the current itinerary|Give Griffith up to 90/);
  assert.equal(h.element("#copilot-lifecycle").dataset.state, "idle");
  assert.equal(h.element("#copilot-submit").disabled, false);
  assert.equal(h.streamCalls, 1);
});

test("an old request finishing after deletion cannot clear a newer request's temporary turn", async () => {
  const recreated = session("session_a", "conversation_new");
  const h = harness(async (_h, _path, options) => options?.method === "DELETE"
    ? { session: session("session_a", null), deleted: true } : reply(recreated));
  const oldStream = h.stream;
  const oldSending = h.submit("OLD_WAIT");
  await h.element("#delete-current-conversation").listeners.click();
  h.stream = deferred();
  const newSending = h.submit("NEW_WAIT");
  h.streamOptions.onEvent({ type: "message.accepted" });
  oldStream.reject(new Error("OLD_FAILURE"));
  await oldSending;
  assert.match(h.text(), /NEW_WAIT.*Reading the current itinerary/);
  assert.equal(h.element("#copilot-submit").attributes["aria-label"], "Stop waiting for Copilot");
  h.stream.reject(new Error("NEW_FAILURE"));
  await newSending;
});

for (const replacement of [session("session_b", "conversation_b"), session("session_a", "conversation_new")]) {
  test(`projection recovery cannot overwrite ${replacement.session_id}/${replacement.conversation_id}`, async () => {
    const incoming = accepted(), pending = deferred();
    const h = harness(async (_h, path) => path.endsWith("/conversation") ? reply(incoming) : pending.promise);
    const restoring = h.controller.restore({ announce: false });
    await tick();
    h.state.session = replacement;
    const workspace = h.state.workspace;
    pending.resolve(projection(incoming));
    assert.equal(await restoring, false);
    assert.deepEqual(h.state.session, replacement);
    assert.equal(h.state.workspace, workspace);
  });
}

test("a failed map recovery leaves both projections unchanged and a later read can recover", async () => {
  const incoming = accepted(), full = projection(incoming);
  let mapReads = 0;
  const h = harness(async (_h, path) => {
    if (path.endsWith("/conversation")) return reply(incoming);
    if (++mapReads === 1) throw new Error("Map read unavailable");
    return full;
  });
  const before = structuredClone(h.state);
  assert.equal(await h.controller.restore({ announce: false }), false);
  assert.deepEqual(h.state, before);
  assert.equal(await h.controller.restore({ announce: false }), true);
  assert.deepEqual(h.state.session, incoming);
  assert.deepEqual(h.state.workspace, full.workspace);
  assert.equal(h.streamCalls, 0, "Recovery must never resend a provider request");
});

