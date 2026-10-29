import assert from "node:assert/strict";
import { copilotCapacityBlock, createCopilotController } from "../../src/itinerary_system/product_app/static/js/copilot.js";

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
  focus() { document.activeElement = this; }
  showModal() { this.hidden = false; }
  close() { this.hidden = true; }
}
function descendants(element) { return [element, ...element.children.flatMap(descendants)]; }

for (const code of ["conversation_turn_limit_reached", "conversation_size_limit_reached"]) {
  const elements = new Map();
  const element = (selector) => {
    if (!elements.has(selector)) elements.set(selector, new Element());
    return elements.get(selector);
  };
  globalThis.document = { querySelector: element, querySelectorAll: () => [], createElement: () => new Element() };
  globalThis.window = { addEventListener() {}, confirm: () => true };
  const state = { workspace: { interaction: { enabled: true, provider: "deterministic" } },
    session: { session_id: "synthetic", conversation_id: "conversation_fixture", trip_id: "synthetic-trip", run_id: "synthetic-run",
      revision: 0, accepted_plan_id: "parent", draft: [{ type: "keep_stop", target: "bixby" }] } };
  const before = structuredClone(state.session);
  let streamCalls = 0;
  const apiCalls = [];
  const controller = createCopilotController({
    getState: () => state,
    api: async (path, options) => { apiCalls.push({ path, options }); return { session: state.session,
      conversation: { schema_version: "product-conversation-v1", conversation_id: "conversation_fixture",
        session_id: "synthetic", trip_id: "synthetic-trip", run_id: "synthetic-run", revision: 0, turns: [] } }; },
    streamMessage: async () => { streamCalls += 1; throw Object.assign(new Error(code), { code, status: 422 }); },
    toast() {}, renderAll() {},
  });
  controller.bind();
  element("#copilot-input").value = "Review my itinerary";
  await element("#copilot-form").listeners.submit({ preventDefault() {} });
  assert.equal(streamCalls, 1);
  assert.equal(element("#copilot-lifecycle").dataset.state, "blocked");
  const rendered = descendants(element("#conversation"));
  assert.ok(rendered.some(node => node.dataset.deliveryState === "blocked"));
  assert.ok(rendered.some(node => node.textContent.includes("blocked before any new provider attempt")));
  assert.ok(!rendered.some(node => node.textContent.includes("result was not confirmed")));
  assert.ok(!rendered.some(node => node.textContent === "Retry"));
  assert.deepEqual(state.session, before);
  const settings = rendered.find(node => node.textContent === "Conversation settings");
  assert.ok(settings);
  settings.listeners.click();
  assert.equal(element("#transcript-settings-dialog").hidden, false);
  await element("#delete-current-conversation").listeners.click();
  assert.ok(apiCalls.some(call => call.options?.method === "DELETE"));
  assert.ok(!descendants(element("#conversation")).some(node => node.dataset.deliveryState === "blocked"));
  assert.equal(streamCalls, 1, "Reset must not resend the blocked message");
  assert.deepEqual(state.session, before);
}
for (const error of [null, { code: "openai_timeout", status: 504 }, { code: "conversation_turn_limit_reached", status: 503 },
  { code: "conversation_size_limit_reached", status: "422" }, { code: "unknown", status: 422 }]) {
  assert.equal(copilotCapacityBlock(error), null, "Unknown outcomes must not be claimed as confirmed capacity blocks");
}
console.log("copilot capacity controller contract passed");
