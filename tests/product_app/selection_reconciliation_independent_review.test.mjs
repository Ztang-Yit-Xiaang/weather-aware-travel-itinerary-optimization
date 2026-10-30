// Independent controller checks: real extracted handler, controlled local replies.
// These are not browser/API journeys and make no server or provider calls.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const source = readFileSync(new URL("../../src/itinerary_system/product_app/static/js/app.js", import.meta.url), "utf8");
const start = source.indexOf("async function updateSelection(");
const end = source.indexOf("function operationEnabled(", start);
assert.ok(start >= 0 && end > start);
const handler = "const mutationRecovery = {submit: (action, body) => api(`/api/sessions/${state.session.session_id}/${action === \"selection\" ? \"selection\" : `draft/${action}`}`, {method:\"POST\", body})};\n" + source.slice(start, end);
const session = (conversation_id, revision = 1, extra = {}) => ({
  session_id: "s", trip_id: "t", run_id: "r", revision, conversation_id,
  accepted_plan_id: "parent", selected_day: 5, selected_stop_id: "bixby",
  selected_segment_id: null, selected_candidate_id: null, selected_alternative_id: null,
  draft: [], proposal: null, ...extra,
});
const deferred = () => { let resolve; const promise = new Promise(done => { resolve = done; }); return { promise, resolve }; };
const tick = async () => { for (let step = 0; step < 6; step++) await Promise.resolve(); };
function harness(read) {
  const post = deferred();
  const state = { session: session("C", 0), workspace: { draft_capabilities: {} }, registry: { marker: "original" } };
  const requests = [], rendered = [], messages = [];
  const update = new Function("state", "api", "syncPreviewRuntimeFromSession", "loadDraftImpactPreview", "renderAll", "toast",
    `${handler}; return updateSelection;`)(state, async (path, options) => {
      requests.push({ path, options });
      return options?.method === "POST" ? post.promise : read();
    }, () => {}, async () => {}, () => rendered.push(structuredClone(state)), message => messages.push(message));
  const begin = () => {
    const result = update({ selected_day: 6 });
    state.session = session("D");
    post.resolve({ session: session("C", 1, { selected_day: 6 }) });
    return result;
  };
  return { state, requests, rendered, messages, update, post, begin };
}

test("draft mutation during a current GET discards its entire older session/workspace pair", async () => {
  const firstRead = deferred(); let reads = 0;
  const canonical = { session: session("D", 2, { draft: [{ type: "keep_stop", target: "bixby" }], selected_day: 6 }),
    workspace: { marker: "current canonical workspace" }, registry: { marker: "current canonical registry" } };
  const h = harness(() => ++reads === 1 ? firstRead.promise : canonical);
  const result = h.begin(); await tick(); assert.equal(reads, 1);
  h.state.session = canonical.session;
  firstRead.resolve({ session: session("D", 1, { selected_day: 6 }), workspace: { marker: "stale" }, registry: { marker: "stale" } });
  assert.equal(await result, true);
  assert.equal(reads, 2);
  assert.strictEqual(h.state.session, canonical.session);
  assert.strictEqual(h.state.workspace, canonical.workspace);
  assert.strictEqual(h.state.registry, canonical.registry);
  assert.deepEqual(h.rendered, [canonical]);
});

for (const invalid of [{ run_id: "foreign" }, { revision: -1 }, { revision: "1" }]) {
  test(`invalid current GET ${JSON.stringify(invalid)} cannot install session or workspace`, async () => {
    const h = harness(() => ({ session: session("D", 1, invalid), workspace: { marker: "foreign" }, registry: { marker: "foreign" } }));
    const originalWorkspace = h.state.workspace, originalRegistry = h.state.registry;
    const result = h.begin(); const current = h.state.session;
    assert.equal(await result, false);
    assert.strictEqual(h.state.session, current);
    assert.strictEqual(h.state.workspace, originalWorkspace);
    assert.strictEqual(h.state.registry, originalRegistry);
    assert.equal(h.rendered.length, 0);
    assert.match(h.messages.at(-1), /Selection was saved/);
  });
}

test("three stale current reads preserve a newer accepted child and never fabricate a merged session", async () => {
  const h = harness(() => ({ session: session("D", 1), workspace: { marker: "older" } }));
  const result = h.begin();
  const current = session("D", 3, { accepted_plan_id: "accepted-child", selected_stop_id: "pfeiffer" });
  h.state.session = current;
  assert.equal(await result, false);
  assert.strictEqual(h.state.session, current);
  assert.equal(h.requests.filter(row => !row.options?.method).length, 3);
  assert.equal(h.rendered.length, 0);
  assert.match(h.messages.at(-1), /workspace kept changing/i);
});

test("deletion to null during a GET forces another read and installs the exact durable null conversation", async () => {
  const firstRead = deferred(); let reads = 0;
  const canonical = { session: session(null, 1, { selected_day: 6 }), workspace: { marker: "after deletion" } };
  const h = harness(() => ++reads === 1 ? firstRead.promise : canonical);
  const result = h.begin(); await tick(); assert.equal(reads, 1);
  h.state.session = session(null);
  firstRead.resolve({ session: session("D", 1, { selected_day: 6 }) });
  assert.equal(await result, true);
  assert.equal(reads, 2);
  assert.strictEqual(h.state.session, canonical.session);
  assert.equal(h.rendered.length, 1);
  assert.equal(h.rendered[0].session.conversation_id, null);
  assert.equal(h.rendered[0].session.selected_day, 6);
});
