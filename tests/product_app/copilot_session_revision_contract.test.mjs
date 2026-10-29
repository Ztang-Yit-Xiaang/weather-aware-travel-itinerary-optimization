// Actual controller with controlled asynchronous replies; no provider or server.
// The 24 -> 25 sequence reproduces the retained website Undo/restore race.
import assert from "node:assert/strict";
import test from "node:test";
import { createCopilotController } from "../../src/itinerary_system/product_app/static/js/copilot.js";

class Element {
  constructor() {
    this.children = []; this.listeners = {}; this.dataset = {}; this.style = {};
    this.attributes = {}; this.value = ""; this.textContent = ""; this.hidden = true;
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
const deferred = () => { let resolve; const promise = new Promise(done => { resolve = done; }); return { promise, resolve }; };
const session = (revision = 24) => ({ session_id: "synthetic-session", conversation_id: "conversation_fixture",
  trip_id: "synthetic-trip", run_id: "synthetic-run", revision, accepted_plan_id: "parent",
  draft: revision === 24 ? [{ type: "set_stop_time_window", target: "stearns", parameters: { earliest_arrival: "10:00" } }] : [],
  proposal: null, selected_day: 6, selected_stop_id: "stearns" });
const conversation = text => ({ schema_version: "product-conversation-v1", conversation_id: "conversation_fixture",
  session_id: "synthetic-session", trip_id: "synthetic-trip", run_id: "synthetic-run", revision: 1,
  turns: [{ client_message_id: text, user_message: text,
  assistant_message: text, state: "completed", context_revision: 24, interpretation: { state: "explanation_ready", intents: [] } }] });

function harness(api, streamMessage) {
  const elements = new Map();
  const element = selector => { if (!elements.has(selector)) elements.set(selector, new Element()); return elements.get(selector); };
  globalThis.document = { querySelector: element, querySelectorAll: () => [], createElement: () => new Element() };
  globalThis.window = { addEventListener() {}, confirm: () => true };
  const state = { session: session(), workspace: { interaction: { enabled: true, provider: "deterministic" } } };
  const messages = [];
  const controller = createCopilotController({ getState: () => state,
    api: (...args) => api(state, ...args), streamMessage, toast: text => messages.push(text), renderAll() {} });
  return { state, controller, element, messages,
    text: () => descendants(element("#conversation")).map(node => node.textContent).join(" ") };
}

test("late conversation snapshot cannot resurrect a draft already undone at a newer revision", async () => {
  const reply = deferred();
  const h = harness(() => reply.promise);
  const restoring = h.controller.restore({ announce: false });
  const afterUndo = session(25); h.state.session = afterUndo;
  reply.resolve({ session: session(24), conversation: conversation("stale snapshot") });
  assert.equal(await restoring, false);
  assert.deepEqual(h.state.session, afterUndo);
  assert.doesNotMatch(h.text(), /stale snapshot/);
});

test("older restore completion cannot replace a newer transcript at the same session revision", async () => {
  const first = deferred(), second = deferred(); let count = 0;
  const h = harness(() => (++count === 1 ? first : second).promise);
  const a = h.controller.restore({ announce: false }), b = h.controller.restore({ announce: false });
  second.resolve({ session: session(), conversation: conversation("new transcript") });
  assert.equal(await b, true);
  first.resolve({ session: session(), conversation: conversation("old transcript") });
  assert.equal(await a, false);
  assert.match(h.text(), /new transcript/); assert.doesNotMatch(h.text(), /old transcript/);
});

for (const [label, incoming] of [
  ["missing", null], ["foreign", { ...session(), session_id: "different-session" }],
  ["missing revision", { ...session(), revision: undefined }], ["string revision", { ...session(), revision: "26" }],
  ["negative revision", { ...session(), revision: -1 }], ["fractional revision", { ...session(), revision: 25.5 }],
]) {
  test(`invalid ${label} response leaves canonical session untouched`, async () => {
    const h = harness(async () => ({ session: incoming, conversation: conversation("invalid response") }));
    const before = structuredClone(h.state.session);
    assert.equal(await h.controller.restore({ announce: false }), false);
    assert.deepEqual(h.state.session, before);
    assert.doesNotMatch(h.text(), /invalid response/);
  });
}

for (const revision of [24, 25]) {
  test(`valid current or newer revision ${revision} is adopted`, async () => {
    const incoming = session(revision);
    const h = harness(async () => ({ session: incoming, conversation: conversation("valid response") }));
    assert.equal(await h.controller.restore({ announce: false }), true);
    assert.deepEqual(h.state.session, incoming); assert.match(h.text(), /valid response/);
  });
}

test("late completed chat response does not roll back a newer visible draft mutation", async () => {
  const reply = deferred(); let envelope, calls = 0;
  const h = harness(async state => ({ session: state.session, conversation: conversation("completed response") }),
    async (_path, options) => { calls++; envelope = options.body; return reply.promise; });
  h.controller.bind(); h.element("#copilot-input").value = "Review this trip";
  const pending = h.element("#copilot-form").listeners.submit({ preventDefault() {} });
  h.state.session = session(25);
  reply.resolve({ session: session(24), turn: { client_message_id: envelope.client_message_id,
    user_message: envelope.message, assistant_message: "Review only", state: "completed",
    interpretation: { state: "explanation_ready", intents: [] } } });
  await pending;
  assert.deepEqual(h.state.session, session(25)); assert.equal(calls, 1);
});

for (const selector of ["#delete-current-conversation", "#delete-all-conversations"]) {
  test(`late ${selector} acknowledgement cannot restore an older trip draft`, async () => {
    const reply = deferred(); let deletes = 0;
    const h = harness(async (state, _path, options) => {
      if (options?.method === "DELETE") { deletes++; return reply.promise; }
      return { session: state.session, conversation: { ...conversation(""), revision: 0, turns: [] } };
    });
    h.controller.bind();
    const pending = h.element(selector).listeners.click();
    h.state.session = session(25);
    reply.resolve({ session: session(24), deleted_count: 1 });
    await pending;
    assert.deepEqual(h.state.session, session(25)); assert.equal(deletes, 1);
  });
}
