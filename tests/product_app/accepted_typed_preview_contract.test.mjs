import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFileSync, readdirSync } from "node:fs";
import path from "node:path";
import test from "node:test";

// Replay the exact failed browser responses, not substitute optimizer results.
const root = process.env.ACTUAL_TYPED_BROWSER_EVIDENCE_DIR;
assert.ok(root, "ACTUAL_TYPED_BROWSER_EVIDENCE_DIR must identify a real direct_edits capture");
const hash = value => createHash("sha256").update(value).digest("hex");
function captures(kind) {
  return readdirSync(path.join(root, kind)).filter(name => /^http_\d+\.json$/.test(name)).sort()
    .map(name => {
      const record = JSON.parse(readFileSync(path.join(root, kind, name), "utf8"));
      let payload;
      try { payload = JSON.parse(record.response_body_utf8); } catch { return null; }
      assert.equal(hash(record.response_body_utf8), record.response_body_sha256, `${kind}/${name} exact response`);
      assert.equal(hash(record.request_body_utf8), record.request_body_sha256, `${kind}/${name} exact request`);
      return {record, payload};
    }).filter(Boolean);
}
const durationRows = captures("duration");
const accepted = durationRows.find(row => row.record.url.endsWith("/accept") && row.payload.workspace).payload;
const source = readFileSync(new URL("../../src/itinerary_system/product_app/static/js/app.js", import.meta.url), "utf8");
const code = source.slice(source.indexOf("const TYPED_EDIT_OPERATIONS"), source.indexOf("function safeExternalLink"));
const make = new Function("state", `${code}; return { normalizeEvaluatedPreviewResponse, sameJson, computedRoleProposalIntegrityDigest, diffChangedDays };`);

test("JSON object key order is not artifact identity; sequence, types, and values remain strict", () => {
  const {sameJson} = make({});
  assert.equal(sameJson({type:"Point", coordinates:[1,2]}, {coordinates:[1,2], type:"Point"}), true);
  for (const [left, right] of [[[1,2],[2,1]], [{x:1},{x:"1"}], [{x:null},{}], [{x:1},{x:2}]]) {
    assert.equal(sameJson(left, right), false);
  }
});

for (const kind of ["duration", "time_window", "role"]) {
  const rows = kind === "duration" ? durationRows : captures(kind);
  const preview = rows.findLast(row => row.record.url.endsWith("/preview")
    && row.payload.proposal?.execution_mode === "typed_direct_edit_independent_evaluation");
  const before = rows.findLast(row => row.record.url.endsWith("/draft/operations")
    && row.payload.session?.revision === preview.payload.proposal.expected_revision).payload.session;
  const actual = preview.payload;
  assert.equal(before.accepted_plan_id, accepted.session.accepted_plan_id);
  assert.equal(actual.proposal.parent_plan_content_hash, accepted.workspace.geography.plans
    .find(plan => plan.plan_id === before.accepted_plan_id).content_hash);
  assert.equal(preview.record.response_status_code, 200);
  assert.deepEqual(JSON.parse(preview.record.request_body_utf8), {expected_revision:before.revision});
  const expected = {session_id:before.session_id, trip_id:before.trip_id, run_id:before.run_id,
    revision:before.revision, accepted_plan_id:before.accepted_plan_id,
    parent_plan_content_hash:actual.proposal.parent_plan_content_hash, draft:before.draft,
    permission_decisions:before.permission_decisions, conversation_id:before.conversation_id};
  const normalize = async (value, resign = true) => {
    const functions = make({workspace:structuredClone(accepted.workspace), session:structuredClone(before),
      token:"synthetic-unit-key-not-a-browser-credential"});
    // The real session credential was correctly redacted. Re-sign only the role
    // transport envelope under a synthetic unit key; do not change its artifact body.
    if (kind === "role" && resign) {
      value.proposal.role_proposal_integrity.digest = await functions.computedRoleProposalIntegrityDigest(value.proposal);
      value.session.proposal.role_proposal_integrity = structuredClone(value.proposal.role_proposal_integrity);
    }
    return functions.normalizeEvaluatedPreviewResponse(value, structuredClone(expected));
  };
  function candidate() {
    const value = structuredClone(actual);
    // Historical v2 has a separately reproduced false-zero summary. Test the
    // corrected canonical projection, without changing plan/diff/certificate.
    value.proposal.repair.changed.affected_day_count = make({}).diffChangedDays(value.proposal.plan_diff).length;
    value.session.proposal = structuredClone(value.proposal);
    return value;
  }
  test(`${kind}: captured artifacts with corrected summary pass (role uses a synthetic unit key)`, async () => {
    const result = await normalize(candidate());
    assert.ok(result, "Backend eligible child must not become PREVIEW_FAILED");
    assert.equal(result.proposal.child_plan_id, actual.proposal.child_plan_id);
    assert.equal(result.proposal.repair.tradeoffs.weather_risk_delta, null, "Missing evidence is not zero improvement");
  });
  test(`${kind}: historical false-zero summary is rejected`, async () => {
    assert.equal(await normalize(structuredClone(actual)), null);
  });
  if (kind === "role") test("role: a signature for another session still fails", async () => {
    assert.equal(await normalize(candidate(), false), null);
  });
  for (const [name, mutation] of [
    ["invented zero weather delta", p => { p.repair.tradeoffs.weather_risk_delta = 0; }],
    ["invented positive weather delta", p => { p.repair.tradeoffs.weather_risk_delta = 1; }],
    ["changed coordinate", p => { p.geography_plan.validated_legs.features[0].geometry.coordinates[0][0] += 0.001; }],
    ["reversed route coordinates", p => { p.geography_plan.validated_legs.features[0].geometry.coordinates.reverse(); }],
    ["different parent content hash", p => { p.parent_plan_content_hash = "0".repeat(16); }],
    ["different route evidence", p => { p.route_validation.source_content_sha256 = "0".repeat(64); }],
    ["different certificate hash", p => { p.certificate_content_hash = "0".repeat(16); }],
  ]) {
    test(`${kind}: rejects ${name}`, async () => {
      const changed = candidate();
      mutation(changed.proposal);
      changed.session.proposal = structuredClone(changed.proposal);
      assert.equal(await normalize(changed), null);
    });
  }
}
