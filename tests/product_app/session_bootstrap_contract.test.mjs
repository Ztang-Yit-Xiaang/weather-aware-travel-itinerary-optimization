import assert from "node:assert/strict";
import fs from "node:fs";
import vm from "node:vm";
import {test} from "node:test";

const source = fs.readFileSync(new URL("../../src/itinerary_system/product_app/static/js/app.js", import.meta.url), "utf8");
function section(start, end) {
  const begin = source.indexOf(start);
  const finish = source.indexOf(end, begin);
  assert.ok(begin >= 0 && finish > begin);
  return source.slice(begin, finish);
}
const restore = section("async function restoreOrCreateSession()", "\nconst COMPONENT_COPY");
const composer = section("function savePendingMutationInput()", "\nfunction restoreMutationControls()");
function harness({code = "unknown_session", status = 404, pending = false} = {}) {
  const original = JSON.stringify({session_id: "old", mutation_token: "old-token"});
  const values = new Map([["session", original]]);
  if (pending) values.set("pending", "durable-pending-journal");
  const calls = [];
  const input = {value: ""};
  const context = {
    state: {session: {session_id: "old"}, token: null, mutationRecovery: {blocked: false}},
    SESSION_STORAGE_KEY: "session", PENDING_MUTATION_KEY: "pending", PENDING_INPUT_KEY: "input",
    sessionStorage: {getItem: key => values.get(key) ?? null, setItem: (key, value) => values.set(key, value), removeItem: key => values.delete(key)},
    $: () => input, toast() {},
    isBuildMismatch: value => value.startsWith("product_"),
    api: async (url, options = {}) => {
      calls.push(options.method || "GET");
      if (!options.method) throw Object.assign(new Error(code), {code, status});
      return {session: {session_id: "new"}, mutation_token: "new-token"};
    },
  };
  vm.createContext(context);
  vm.runInContext(restore + composer, context);
  return {context, values, calls, input, original};
}
test("a removed session starts a new workspace without a pending mutation", async () => {
  const h = harness();
  const result = await h.context.restoreOrCreateSession();
  assert.equal(result.session.session_id, "new");
  assert.deepEqual(h.calls, ["GET", "POST"]);
});
test("an expired session explains retained drafts without automatically adopting them", async () => {
  const h = harness({code: "session_expired"});
  const result = await h.context.restoreOrCreateSession();
  assert.equal(result.session.session_id, "new");
  assert.match(h.context.state.sessionRecoveryNotice, /retained for recovery/);
  assert.match(h.context.state.sessionRecoveryNotice, /have not been reapplied/);
});
for (const row of [
  {code: "unknown_session", status: 404, pending: true},
  {code: "session_expired", status: 404, pending: true},
  {code: "invalid_session_token", status: 403},
  {code: "state_lock_timeout", status: 503},
  {code: "product_server_build_changed", status: 503},
]) test(`does not replace protected credentials: ${row.code}, pending=${Boolean(row.pending)}`, async () => {
  const h = harness(row);
  await assert.rejects(h.context.restoreOrCreateSession());
  assert.deepEqual(h.calls, ["GET"]);
  assert.equal(h.values.get("session"), h.original);
  if (row.pending) assert.equal(h.values.get("pending"), "durable-pending-journal");
});
test("unsent text survives reload in the same session without a pending mutation", () => {
  const h = harness();
  h.input.value = "Please keep my coast preference 海岸線";
  h.context.savePendingMutationInput();
  h.input.value = "";
  h.context.restorePendingMutationInput();
  assert.equal(h.input.value, "Please keep my coast preference 海岸線");
});
test("empty composer clears saved text, and another session cannot read it", () => {
  const h = harness();
  h.input.value = "old message";
  h.context.savePendingMutationInput();
  h.context.state.session.session_id = "other";
  h.input.value = "";
  h.context.restorePendingMutationInput();
  assert.equal(h.input.value, "");
  h.context.state.session.session_id = "old";
  h.context.savePendingMutationInput();
  h.input.value = "";
  h.context.restorePendingMutationInput();
  assert.equal(h.input.value, "");
});
test("page hide persists programmatic composer changes before reload", () => {
  assert.match(source, /window\.addEventListener\("pagehide", savePendingMutationInput\)/);
});
