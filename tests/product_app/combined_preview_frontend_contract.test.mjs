import assert from "node:assert/strict";
import fs from "node:fs";
import test from "node:test";

// This is the complete owned or mixed draft and response returned by the real
// API. It is not a mocked browser journey.
const contractPath = process.env.ACTUAL_COMBINED_PREVIEW_CONTRACT_PATH;
assert.ok(contractPath, "ACTUAL_COMBINED_PREVIEW_CONTRACT_PATH is required");
const actual = JSON.parse(fs.readFileSync(contractPath, "utf8"));
const source = fs.readFileSync(new URL("../../src/itinerary_system/product_app/static/js/app.js", import.meta.url), "utf8");
const code = source.slice(source.indexOf("const TYPED_EDIT_OPERATIONS"), source.indexOf("function safeExternalLink"));
const normalizers = new Function("state", `${code}\nreturn {
  normalizeEvaluatedPreviewResponse, normalizeTypedEditCapabilities, normalizeDraftImpactPreview,
  draftCombinationBlocker, computedStableContentHash, normalizeCombinedAttributeOwnership,
};`)({workspace: actual.workspace, session: actual.before_session});

test("actual combined draft is permitted by the production controls and impact contract", () => {
  assert.ok(normalizers.normalizeTypedEditCapabilities(actual.workspace.typed_edit_capabilities));
  assert.equal(normalizers.draftCombinationBlocker(actual.expected.draft), null);
  assert.equal(actual.impact.summary.can_run_evaluated_preview, true);
  assert.ok(normalizers.normalizeDraftImpactPreview(actual.impact));
});

test("real combined API result passes the complete production frontend normalizer", async () => {
  assert.ok(await normalizers.normalizeEvaluatedPreviewResponse(structuredClone(actual.preview), actual.expected));
});

const mutations = [
  ["missing combined execution receipt", p => { delete p.runtime_execution; }],
  ["wrong parent", p => { p.runtime_execution.parent_plan_id = "plan_1111111111111111"; }],
  ["wrong complete request", p => { p.runtime_execution.operations.pop(); }],
  ["missing final independent evaluation", p => { p.runtime_execution.evaluation_executed = false; }],
  ["wrong final evaluator", p => { p.runtime_execution.evaluator = "stored_candidate_lookup"; }],
  ["foreign final child", p => { p.runtime_execution.child_plan.plan_id = "plan_1111111111111111"; }],
  ["foreign certificate plan", p => { p.runtime_execution.evaluation_certificate.plan_id = "plan_1111111111111111"; }],
  ["foreign diff plan", p => { p.runtime_execution.plan_diff.child_plan_id = "plan_1111111111111111"; }],
  ...(actual.preview.proposal.runtime_execution.optimizer_execution ? [
    ["missing optimizer for replacement", p => { p.runtime_execution.optimizer_execution = null; }],
    ["wrong optimizer request", p => { p.runtime_execution.optimizer_execution.repair_request.confirmed_constraints.must_delete = []; }],
  ] : [["invented selection optimizer", p => { p.runtime_execution.optimizer_execution = {}; }]]),
  ["foreign final route feature", p => { p.geography_plan.stops.features[0].properties.plan_id = "plan_1111111111111111"; }],
];
for (const [name, mutate] of mutations) {
  test(`rejects ${name}`, async () => {
    const forged = structuredClone(actual.preview);
    mutate(forged.proposal);
    forged.session.proposal = structuredClone(forged.proposal);
    assert.equal(await normalizers.normalizeEvaluatedPreviewResponse(forged, actual.expected), null);
  });
}

for (const family of ["duration", "time_window", "role"]) {
  if (actual.preview.proposal.runtime_execution.plan_diff[`${family}_changes`]) continue;
  test(`rejects an unrequested empty ${family} family`, async () => {
    const forged = structuredClone(actual.preview);
    forged.proposal.runtime_execution.plan_diff[`${family}_changes`] = [];
    forged.session.proposal = structuredClone(forged.proposal);
    assert.equal(await normalizers.normalizeEvaluatedPreviewResponse(forged, actual.expected), null);
  });
}

for (const [name, mutate] of [
  ["a dropped duration", child => { child.selected_stops.find(s => (s.stop_id || s.poi_id) === "griffith_observatory").visit_duration_minutes = 45; }],
  ["a dropped time window", child => { delete child.selected_stops.find(s => (s.stop_id || s.poi_id) === "golden_gate_bridge").time_window_constraint; }],
  ["a dropped role", child => { delete child.selected_stops.find(s => (s.stop_id || s.poi_id) === "golden_gate_bridge").itinerary_role; }],
]) {
  const type = name.includes("duration") ? "set_stop_duration" : name.includes("time window") ? "set_stop_time_window" : "set_stop_role";
  if (!actual.expected.draft.some(operation => operation.type === type)) continue;
  test(`rejects ${name} even with internally rebound content hashes`, async () => {
    const forged = structuredClone(actual.preview);
    const p = forged.proposal;
    const r = p.runtime_execution;
    mutate(r.child_plan);
    const childPayload = {...r.child_plan};
    delete childPayload.content_hash;
    const hash = await normalizers.computedStableContentHash(childPayload);
    r.child_plan.content_hash = r.child_plan_content_hash = p.child_plan_content_hash = hash;
    p.certificate_identity.plan_content_hash = r.evaluation_certificate.plan_content_hash = hash;
    p.geography_plan.content_hash = hash;
    for (const name of ["stops", "route_path", "validated_legs", "gaps"]) {
      for (const feature of p.geography_plan[name].features) feature.properties.content_hash = hash;
    }
    const certificatePayload = {...r.evaluation_certificate};
    delete certificatePayload.content_hash;
    const certificateHash = await normalizers.computedStableContentHash(certificatePayload);
    r.evaluation_certificate.content_hash = r.evaluation_content_hash = p.certificate_content_hash = certificateHash;
    p.certificate_identity.content_hash = certificateHash;
    forged.session.proposal = structuredClone(p);
    assert.equal(await normalizers.normalizeEvaluatedPreviewResponse(forged, actual.expected), null);
  });
}

if (["plan-diff-v5", "plan-diff-v7"].includes(actual.preview.proposal.runtime_execution.plan_diff.schema_version)) {
  const ownedMutations = [
    ["foreign ownership parent", e => { e.parent_plan_content_hash = "1111111111111111"; }],
    ["omitted accepted rule", e => { e.constraints.pop(); }],
    ["weakened accepted rule", e => { e.constraints[0].strength = "weak"; }],
    ["different accepted rule value", e => { e.constraints[0].canonical_value = null; }],
    ["different cost policy", e => { e.ownership_policy.change_type_weights.duration_change = 0; }],
    ["invented permission grant", e => { e.semantic_scope.permission_claim = "granted"; }],
    ["invented feasibility verdict", e => { e.semantic_scope.feasibility_claim = "eligible"; }],
    ["altered before value", e => { e.changes[0].before_value.existence = true; }],
    ["dropped cost row", e => { e.changes.pop(); }],
  ];
  for (const [name, mutate] of ownedMutations) {
    test(`rejects ${name} even after rehashing ownership evidence`, async () => {
      const p = structuredClone(actual.preview.proposal);
      const diff = p.runtime_execution.plan_diff;
      const evidence = diff.attribute_ownership_evidence;
      mutate(evidence);
      const payload = {...evidence}; delete payload.content_hash;
      evidence.content_hash = await normalizers.computedStableContentHash(payload);
      assert.equal(await normalizers.normalizeCombinedAttributeOwnership(diff, p, p.runtime_execution.child_plan), null);
    });
  }
  test("rejects child constraint removal rather than treating it as user permission", async () => {
    const p = structuredClone(actual.preview.proposal);
    p.runtime_execution.child_plan.owned_constraints = [];
    assert.equal(await normalizers.normalizeCombinedAttributeOwnership(p.runtime_execution.plan_diff, p, p.runtime_execution.child_plan), null);
  });
  test("rejects an ownership format downgrade after coherent diff rehashing", async () => {
    const forged = structuredClone(actual.preview), p = forged.proposal, diff = p.runtime_execution.plan_diff;
    delete diff.attribute_ownership_evidence;
    diff.schema_version = diff.schema_version === "plan-diff-v7" ? "plan-diff-v6"
      : diff.duration_changes ? "plan-diff-v2" : "plan-diff-v3";
    const seed = {parent_plan_id: diff.parent_plan_id, child_plan_id: diff.child_plan_id,
      added: diff.added_stops, deleted: diff.deleted_stops, day_moves: diff.day_moves, time_shifts: diff.time_shifts,
      reorder: diff.reorder_changes, lodging: diff.lodging_changes, road: diff.road_changes,
      unchanged_days: diff.unchanged_days, weighted_edit_cost: diff.weighted_edit_cost};
    for (const key of ["duration_changes", "time_window_changes", "role_changes"]) if (diff[key]) seed[key] = diff[key];
    p.diff_id = p.diff_identity.diff_id = diff.diff_id = `diff_${await normalizers.computedStableContentHash(seed)}`;
    p.diff_content_hash = p.diff_identity.content_hash = await normalizers.computedStableContentHash(diff);
    forged.session.proposal = structuredClone(p);
    assert.equal(await normalizers.normalizeEvaluatedPreviewResponse(forged, actual.expected), null);
  });
  if (actual.preview.proposal.state === "ineligible") {
    test("cost evidence cannot turn a protected mismatch into eligibility", async () => {
      const p = structuredClone(actual.preview.proposal);
      p.eligibility = "eligible";
      assert.equal(await normalizers.normalizeCombinedAttributeOwnership(p.runtime_execution.plan_diff, p, p.runtime_execution.child_plan), null);
    });
  }
}
