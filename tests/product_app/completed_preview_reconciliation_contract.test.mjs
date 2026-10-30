import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

// Component-level adversarial tests reuse an actual backend response; the
// separate browser journey exercises the real HTTP/optimizer/rendering path.
assert.ok(process.env.R19_PREVIEW_IO, "R19_PREVIEW_IO must name a recorded real preview JSON");
const capture = JSON.parse(readFileSync(process.env.R19_PREVIEW_IO, "utf8"));
const actual = capture.first || capture.preview || capture;
assert.equal(actual.proposal.state, "eligible");
const source = readFileSync(new URL("../../src/itinerary_system/product_app/static/js/app.js", import.meta.url), "utf8");
const contract = source.slice(source.indexOf("const TYPED_EDIT_OPERATIONS"), source.indexOf("function safeExternalLink"));
const apiCode = source.slice(source.indexOf("function isBuildMismatch("), source.indexOf("function copilotStreamError("));
const helper = source.slice(source.indexOf("function previewExpectedContext("), source.indexOf("async function addDraft("));
const create = new Function("state", "fetch", "renderAll", "copilot", "apiErrorMessage", "announceWorkspaceChange",
  `const mutationRecovery = {submit: (action, body) => api(\`/api/sessions/\${state.session.session_id}/draft/\${action}\`, {method:"POST", body, pendingMutation:true})};\n${contract}\n${apiCode}\n${helper}\nreturn { submitDraftMutation, previewExpectedContext, previewRequestIsCurrent };`);
const routeNormalizer = new Function(`${contract}; return normalizeRouteValidation;`)();

test("frozen additive route identity binds matrix and composite bundle to the same descriptor hash", () => {
  const route = structuredClone(actual.proposal.route_validation);
  const prefix = route.source_content_sha256.slice(0, 16);
  route.matrix_id = `route_matrix_delta_${prefix}`;
  route.source_bundle_id = `route_bundle_composite_${prefix}`;
  assert.deepEqual(routeNormalizer(route), route);
});

for (const [name, mutate] of [
  ["mixed original and composite identifiers", route => { route.matrix_id = "route_matrix_" + route.source_content_sha256.slice(0, 16); }],
  ["matrix descriptor mismatch", route => { route.matrix_id = "route_matrix_delta_1111111111111111"; }],
  ["bundle descriptor mismatch", route => { route.source_bundle_id = "route_bundle_composite_1111111111111111"; }],
  ["descriptor content hash mismatch", route => { route.source_content_sha256 = "0".repeat(64); }],
  ["invented evidence prefix", route => { route.matrix_id = "route_matrix_dynamic_" + route.source_content_sha256.slice(0, 16); }],
]) {
  test(`composite route evidence rejects ${name}`, () => {
    const route = structuredClone(actual.proposal.route_validation);
    const prefix = route.source_content_sha256.slice(0, 16);
    route.matrix_id = `route_matrix_delta_${prefix}`;
    route.source_bundle_id = `route_bundle_composite_${prefix}`;
    mutate(route);
    assert.equal(routeNormalizer(route), null);
  });
}

function setup({ status = 409, code = "stale_session_revision", mutate = () => {}, phase = "PREVIEW_RUNNING", retryStatus = 200, onGet = () => {} } = {}) {
  const before = structuredClone(actual.session);
  before.revision = actual.proposal.expected_revision;
  before.proposal = null;
  before.selected_stop_id = "bixby_creek_bridge_viewpoint";
  const state = {
    session: before, token: "unit-only-not-a-secret", clientBuildId: "build_unit_fixture", workspace: structuredClone(capture.workspace || { timeline: [] }),
    typedImpact: { response: { parent_plan_content_hash: actual.proposal.parent_plan_content_hash } },
    previewRuntime: { epoch: 8, status: phase, requestId: null },
  };
  const current = structuredClone(actual);
  mutate(current);
  const calls = [];
  let posts = 0;
  const after = { session: { ...structuredClone(current.session), revision: current.session.revision + 1, draft: [], proposal: null } };
  const fetch = async (path, options) => {
    calls.push({ path, method: options.method, body: options.body ? JSON.parse(options.body) : null });
    if (options.method === "GET") {
      onGet(state);
      return new Response(JSON.stringify({ session: current.session }), { status: 200, headers: { "Content-Type": "application/json" } });
    }
    posts += 1;
    const httpStatus = posts === 1 ? status : retryStatus;
    return new Response(JSON.stringify(httpStatus === 200 ? after : { detail: code }), {
      status: httpStatus, headers: { "Content-Type": "application/json" },
    });
  };
  return { state, calls, after, ...create(state, fetch, () => {}, { previewStateChanged() {} }, code => code, () => {}) };
}

test("completed exact preview permits only one unchanged draft retry and never installs old proposal", async () => {
  const h = setup();
  const before = structuredClone(h.state.session);
  const expected = h.previewExpectedContext();
  const originalEpoch = h.state.previewRuntime.epoch;
  assert.deepEqual(await h.submitDraftMutation("undo"), h.after);
  assert.deepEqual(h.state.session, before);
  assert.equal(h.previewRequestIsCurrent(originalEpoch, expected), false);
  assert.deepEqual(h.calls.map(row => row.method), ["POST", "GET", "POST"]);
  assert.equal(h.calls[0].body.expected_revision + 1, h.calls[2].body.expected_revision);
  assert.equal(h.calls[0].path, h.calls[2].path);
});

test("add retry preserves exact operation semantics", async () => {
  const h = setup();
  const body = { type: "replace_nearby", target: "bixby_creek_bridge_viewpoint", parameters: { candidate_id: "point_lobos_state_natural_reserve" }, source: "map" };
  await h.submitDraftMutation("operations", body);
  const first = { ...h.calls[0].body };
  const retry = { ...h.calls[2].body };
  delete first.expected_revision;
  delete retry.expected_revision;
  assert.deepEqual(first, body);
  assert.deepEqual(retry, body);
});

for (const [name, mutate] of [
  ["different draft", value => { value.session.draft = []; }],
  ["different accepted plan", value => { value.session.accepted_plan_id = "plan_1111111111111111"; }],
  ["different session", value => { value.session.session_id += "x"; }],
  ["different conversation", value => { value.session.conversation_id += "x"; }],
  ["permission change", value => { value.session.permission_decisions = [{ permission: "changed" }]; }],
  ["two intervening revisions", value => { value.session.revision += 1; }],
  ["no evaluated proposal", value => { value.session.proposal = null; }],
  ["wrong proposal hash", value => { value.session.proposal.child_plan_content_hash = "1111111111111111"; }],
  ["wrong evaluation hash", value => { value.session.proposal.runtime_execution.evaluation_certificate.plan_content_hash = "1111111111111111"; }],
]) {
  test(`does not rebase ${name}`, async () => {
    const h = setup({ mutate });
    await assert.rejects(h.submitDraftMutation("undo"));
    assert.deepEqual(h.calls.map(row => row.method), ["POST", "GET"]);
    assert.equal(h.state.session.proposal, null);
    assert.equal(h.state.previewRuntime.status, "PARENT_RENDERED");
  });
}

test("does not retry if a newer UI action starts during reconciliation", async () => {
  const h = setup({ onGet: state => { state.previewRuntime.epoch += 1; } });
  await assert.rejects(h.submitDraftMutation("undo"));
  assert.equal(h.calls.length, 2);
});

for (const options of [{ status: 503 }, { code: "other_conflict" }, { phase: "PARENT_RENDERED" }]) {
  test(`does not retry unrelated failure ${JSON.stringify(options)}`, async () => {
    const h = setup(options);
    await assert.rejects(h.submitDraftMutation("undo"), error => error.status === (options.status || 409));
    assert.equal(h.calls.length, 1);
  });
}

test("a second revision conflict is returned without a retry loop", async () => {
  const h = setup({ retryStatus: 409 });
  await assert.rejects(h.submitDraftMutation("undo"));
  assert.equal(h.calls.length, 3);
});

test("normal draft success does not perform a reconciliation read", async () => {
  const h = setup({ status: 200 });
  assert.deepEqual(await h.submitDraftMutation("undo"), h.after);
  assert.equal(h.calls.length, 1);
});

test("this helper cannot dispatch preview, acceptance, or billed conversation retries", async () => {
  for (const action of ["preview", "accept", "copilot/messages"]) {
    const h = setup();
    await assert.rejects(h.submitDraftMutation(action), /unsupported_draft_mutation/);
    assert.equal(h.calls.length, 0);
  }
});
