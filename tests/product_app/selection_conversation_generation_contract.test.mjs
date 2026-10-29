// Actual app selection handler; controlled reply timing only, no server/provider.
import assert from "node:assert/strict";
import fs from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";

const source = fs.readFileSync(fileURLToPath(new URL("../../src/itinerary_system/product_app/static/js/app.js", import.meta.url)), "utf8");
const code = "const mutationRecovery = {submit: (action, body) => api(`/api/sessions/${state.session.session_id}/${action === \"selection\" ? \"selection\" : `draft/${action}`}`, {method:\"POST\", body})};\n" + source.slice(source.indexOf("async function updateSelection("), source.indexOf("function operationEnabled("));
const deferred = () => { let resolve; const promise = new Promise(done => { resolve = done; }); return { promise, resolve }; };
const session = (conversation_id = "C", revision = 0, extra = {}) => ({
  session_id: "session", trip_id: "trip", run_id: "run", conversation_id, revision,
  accepted_plan_id: "parent", selected_day: 1, selected_stop_id: "first", selected_segment_id: null,
  selected_candidate_id: null, selected_alternative_id: null, draft: [], proposal: null, ...extra,
});
const tick = async () => { for (let i = 0; i < 5; i++) await Promise.resolve(); };

function harness(api) {
  const state = { session: session(), workspace: { draft_capabilities: { candidate_choices: [] } } };
  const calls = [], messages = [], rendered = [];
  const update = new Function("state", "api", "syncPreviewRuntimeFromSession", "loadDraftImpactPreview", "renderAll", "toast",
    `${code}; return updateSelection;`)(state, async (...args) => { calls.push(args); return api(...args); },
    () => {}, async () => {}, () => rendered.push(structuredClone(state.session)), text => messages.push(text));
  return { state, calls, messages, rendered, update };
}

test("ordinary selection uses its exact committed response without an extra GET", async () => {
  const committed = session("C", 1, { selected_day: 2 });
  const h = harness(async () => ({ session: committed }));
  assert.equal(await h.update({ selected_day: 2 }), true);
  assert.strictEqual(h.state.session, committed);
  assert.equal(h.calls.length, 1);
});

test("late C selection reconciles exact canonical D without dropping committed selection", async () => {
  const pending = deferred();
  const canonical = session("D", 1, { selected_day: 2 });
  const workspace = { exact_canonical_workspace: true };
  const h = harness(async (_url, options) => options?.method === "POST" ? pending.promise : { session: canonical, workspace });
  const work = h.update({ selected_day: 2 });
  h.state.session = session("D", 1);
  pending.resolve({ session: session("C", 1, { selected_day: 2 }) });
  assert.equal(await work, true);
  assert.strictEqual(h.state.session, canonical);
  assert.strictEqual(h.state.workspace, workspace);
  assert.equal(h.calls.length, 2);
  assert.deepEqual(h.rendered, [canonical]);
});

test("a second deletion during reconciliation requires another current GET", async () => {
  const post = deferred(), firstGet = deferred(); let reads = 0;
  const canonical = session("E", 1, { selected_day: 2 });
  const h = harness(async (_url, options) => options?.method === "POST" ? post.promise
    : ++reads === 1 ? firstGet.promise : { session: canonical });
  const work = h.update({ selected_day: 2 });
  h.state.session = session("D", 1);
  post.resolve({ session: session("C", 1, { selected_day: 2 }) });
  await tick();
  assert.equal(reads, 1);
  h.state.session = session("E", 1);
  firstGet.resolve({ session: session("D", 1, { selected_day: 2 }) });
  assert.equal(await work, true);
  assert.equal(reads, 2);
  assert.deepEqual(h.rendered, [canonical]);
});

test("newer draft revision is not overwritten by a late selection snapshot", async () => {
  const post = deferred();
  const canonical = session("C", 2, { selected_day: 2, draft: [{ type: "keep_stop", target: "first" }] });
  const h = harness(async (_url, options) => options?.method === "POST" ? post.promise : { session: canonical });
  const work = h.update({ selected_day: 2 }); h.state.session = canonical;
  post.resolve({ session: session("C", 1, { selected_day: 2 }) });
  assert.equal(await work, true); assert.strictEqual(h.state.session, canonical);
  assert.equal(h.calls.length, 2);
});

for (const field of ["session_id", "trip_id", "run_id"]) {
  test(`foreign ${field} response cannot install a session`, async () => {
    const h = harness(async () => ({ session: session("C", 1, { [field]: "foreign" }) }));
    const before = h.state.session;
    assert.equal(await h.update({ selected_day: 2 }), false);
    assert.strictEqual(h.state.session, before); assert.equal(h.rendered.length, 0);
  });
}

test("a workspace switch while reconciliation is pending cannot adopt its old response", async () => {
  const post = deferred(), get = deferred();
  const h = harness(async (_url, options) => options?.method === "POST" ? post.promise : get.promise);
  const work = h.update({ selected_day: 2 }); h.state.session = session("D", 1);
  post.resolve({ session: session("C", 1) }); await tick();
  const other = session("other", 0, { session_id: "other-session" }); h.state.session = other;
  get.resolve({ session: session("D", 1) });
  assert.equal(await work, false); assert.strictEqual(h.state.session, other);
  assert.equal(h.rendered.length, 0);
});

test("failed reconciliation preserves D and says the selection was saved", async () => {
  const post = deferred();
  const h = harness(async (_url, options) => { if (options?.method === "POST") return post.promise;
    throw new Error("read unavailable"); });
  const work = h.update({ selected_day: 2 }); const current = session("D", 1); h.state.session = current;
  post.resolve({ session: session("C", 1) });
  assert.equal(await work, false); assert.strictEqual(h.state.session, current);
  assert.match(h.messages.at(-1), /selection was saved/i);
  assert.equal(h.rendered.length, 0);
});

test("continued generation changes terminate without installing a stale GET", async () => {
  const post = deferred(); let reads = 0;
  const h = harness(async (_url, options) => {
    if (options?.method === "POST") return post.promise;
    const previous = h.state.session; reads++; h.state.session = session(`generation-${reads}`, 1);
    return { session: previous };
  });
  const work = h.update({ selected_day: 2 }); h.state.session = session("D", 1);
  post.resolve({ session: session("C", 1) });
  assert.equal(await work, false); assert.equal(reads, 3);
  assert.equal(h.state.session.conversation_id, "generation-3");
  assert.equal(h.rendered.length, 0); assert.match(h.messages.at(-1), /selection was saved/i);
});
