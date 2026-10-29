import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import path from "node:path";
import test from "node:test";

// Use the actual website captures, so the positive case crosses the complete
// production normalizer rather than a test-built approximation of the response.
const evidenceDir = process.env.RUNTIME_PREVIEW_EVIDENCE_DIR;
assert.ok(evidenceDir, "RUNTIME_PREVIEW_EVIDENCE_DIR must identify a live_copilot_chat evidence directory");
function loadHttp(name) {
  const record = JSON.parse(readFileSync(path.join(evidenceDir, name), "utf8"));
  assert.equal(record.response_status_code, 200, name);
  for (const kind of ["request", "response"]) {
    assert.equal(createHash("sha256").update(record[`${kind}_body_utf8`]).digest("hex"), record[`${kind}_body_sha256`], `${name} exact ${kind} bytes`);
  }
  return { record, payload: JSON.parse(record.response_body_utf8) };
}
const draft = loadHttp("product_draft_http_io.json");
const preview = loadHttp("product_preview_http_io.json");
const actual = preview.payload;
const before = draft.payload.session;
assert.equal(draft.record.method, "POST");
assert.equal(preview.record.method, "POST");
assert.equal(new URL(preview.record.url).pathname, `/api/sessions/${before.session_id}/preview`);
assert.equal(new URL(preview.record.url).origin, new URL(draft.record.url).origin);
assert.equal(actual.session.session_id, before.session_id);
assert.deepEqual(JSON.parse(preview.record.request_body_utf8), { expected_revision: before.revision });
const expected = {
  session_id: before.session_id,
  trip_id: before.trip_id,
  run_id: before.run_id,
  revision: before.revision,
  accepted_plan_id: before.accepted_plan_id,
  parent_plan_content_hash: actual.proposal.parent_plan_content_hash,
  draft: before.draft,
  permission_decisions: before.permission_decisions,
  conversation_id: before.conversation_id,
};
const source = readFileSync(new URL("../../src/itinerary_system/product_app/static/js/app.js", import.meta.url), "utf8");
const start = source.indexOf("const TYPED_EDIT_OPERATIONS");
const end = source.indexOf("function safeExternalLink");
assert.ok(start >= 0 && end > start);
const normalize = new Function("state", `${source.slice(start, end)}\nreturn normalizeEvaluatedPreviewResponse;`)(
  { workspace: { timeline: [] }, session: before },
);
assert.ok(await normalize(structuredClone(actual), structuredClone(expected)), "Actual website preview must pass the complete production normalizer before adversarial checks");

test("exact real website response passes the production preview contract", async () => {
  const result = await normalize(structuredClone(actual), structuredClone(expected));
  assert.equal(result.proposal.child_plan_id, actual.proposal.child_plan_id);
  assert.equal(result.session.selected_stop_id, actual.session.selected_stop_id);
});

const mutations = [
  ["missing runtime receipt", (proposal) => { delete proposal.runtime_execution; }],
  ["historical runtime receipt schema", (proposal) => { proposal.runtime_execution.schema_version = "registered-repair-execution-v0"; }],
  ["registered demo downgrade", (proposal) => {
    proposal.execution_mode = "registered_independent_evaluation";
    proposal.provenance = "registered_independent_evaluation";
    delete proposal.runtime_execution;
  }],
  ["old receipt parent", (proposal) => { proposal.runtime_execution.parent_plan_id = "plan_1111111111111111"; }],
  ["old receipt parent hash", (proposal) => { proposal.runtime_execution.parent_plan_content_hash = "1111111111111111"; }],
  ["old receipt request", (proposal) => { proposal.runtime_execution.request_id = "older_request"; }],
  ["old executed request", (proposal) => { proposal.runtime_execution.repair_request.request_id = "older_request"; }],
  ["old planner request", (proposal) => { proposal.runtime_execution.planner_run.planning_request_id = "older_request"; }],
  ["old receipt child", (proposal) => { proposal.runtime_execution.child_plan_id = "plan_1111111111111111"; }],
  ["old child content hash", (proposal) => { proposal.runtime_execution.child_plan.content_hash = "1111111111111111"; }],
  ["old child source run", (proposal) => { proposal.runtime_execution.child_plan.source_run_id = "older_run"; }],
  ["old evaluation identity", (proposal) => { proposal.runtime_execution.evaluation_id = "cert_1111111111111111"; }],
  ["old evaluation plan", (proposal) => { proposal.runtime_execution.evaluation_certificate.plan_id = "plan_1111111111111111"; }],
  ["old evaluation source run", (proposal) => { proposal.runtime_execution.evaluation_certificate.source_run_id = "older_run"; }],
  ["old evaluation context", (proposal) => { proposal.runtime_execution.evaluation_certificate.route_validation.context_snapshot_id = "older_context"; }],
  ["different executed method", (proposal) => {
    proposal.runtime_execution.planner_run.method_executed = "different_optimizer";
    proposal.runtime_execution.planner_trace.planner_run = structuredClone(proposal.runtime_execution.planner_run);
  }],
  ["evaluation did not execute", (proposal) => { proposal.runtime_execution.evaluation_executed = false; }],
  ["different evaluator", (proposal) => { proposal.runtime_execution.evaluator = "stored_certificate_lookup"; }],
  ["execution timestamp missing", (proposal) => { proposal.runtime_execution.started_at = null; }],
  ["execution interval reversed", (proposal) => { proposal.runtime_execution.started_at = new Date(Date.parse(proposal.runtime_execution.completed_at) + 1000).toISOString(); }],
  ["evaluation predates current execution", (proposal) => { proposal.runtime_execution.evaluation_certificate.evaluated_at = new Date(Date.parse(proposal.runtime_execution.started_at) - 1000).toISOString(); }],
  ["evaluation after execution", (proposal) => { proposal.runtime_execution.evaluation_certificate.evaluated_at = new Date(Date.parse(proposal.runtime_execution.completed_at) + 1000).toISOString(); }],
  ["replacement removal constraint missing", (proposal) => { proposal.runtime_execution.repair_request.confirmed_constraints.must_delete = []; }],
  ["replacement inclusion constraint missing", (proposal) => { proposal.runtime_execution.repair_request.confirmed_constraints.must_include = []; }],
  ["frozen child requirement ID", (proposal) => { proposal.geography_plan.validated_legs.features[0].properties.requirement_id = "plan_8ae77712483a709e:required-leg:1"; }],
  ["old route feature plan", (proposal) => { proposal.geography_plan.validated_legs.features[0].properties.plan_id = "plan_1111111111111111"; }],
];
for (const [name, mutate] of mutations) {
  test(`rejects ${name}`, async () => {
    const forged = structuredClone(actual);
    mutate(forged.proposal);
    // Mutate both API projections identically: the rejection must come from
    // the cross-layer invariant rather than a duplicate-envelope difference.
    forged.session.proposal = structuredClone(forged.proposal);
    assert.equal((await normalize(forged, structuredClone(expected))) === null, true, name);
  });
}
