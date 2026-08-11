import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const contractPath = process.env.ACTUAL_EVALUATED_DURATION_CONTRACT_PATH;
assert.ok(contractPath, "ACTUAL_EVALUATED_DURATION_CONTRACT_PATH is required");
const actual = JSON.parse(fs.readFileSync(contractPath, "utf8"));
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "..");
const appPath = path.join(root, "src", "itinerary_system", "product_app", "static", "js", "app.js");
const source = fs.readFileSync(appPath, "utf8");
const contractCode = source.slice(
  source.indexOf("const TYPED_EDIT_OPERATIONS"),
  source.indexOf("function safeExternalLink"),
);
const state = { workspace: actual.workspace, session: actual.before_session };
const normalizers = new Function(
  "state",
  `${contractCode}\nreturn {
    normalizeTypedEditCapabilities, normalizeDraftImpactPreview, normalizeEvaluatedPreviewResponse,
    normalizeRouteValidation, normalizeDurationPlanDiff, normalizeDurationScheduleImpact,
    normalizeCertificateScheduleEvidence, normalizePreviewRepair,
    normalizePreviewGeography, durationRouteEvidenceUnchanged, previewProposalBinding,
    exactKeys, sameJson, canonicalJson, computedDraftContentHash, computedStableContentHash,
    draftHashRecords,
  };`,
)(state);

assert.ok(normalizers.normalizeTypedEditCapabilities(
  structuredClone(actual.workspace.typed_edit_capabilities),
));
assert.ok(normalizers.normalizeDraftImpactPreview(structuredClone(actual.impact)));
const actualProposal = actual.preview.proposal;
const routeValidation = normalizers.normalizeRouteValidation(actualProposal.route_validation);
assert.ok(routeValidation, "route validation");
const planDiff = await normalizers.normalizeDurationPlanDiff(
  actualProposal.plan_diff, actualProposal, actual.expected.draft,
);
assert.ok(planDiff, "duration plan diff");
assert.ok(await normalizers.normalizeDurationScheduleImpact(
  actualProposal.schedule_impact,
  actualProposal.schedule_impact_identity,
  actualProposal,
  planDiff,
  routeValidation,
), "duration schedule impact");
assert.ok(await normalizers.normalizeCertificateScheduleEvidence(
  actualProposal.certificate_schedule_evidence,
  actualProposal,
  actualProposal.schedule_impact,
), "certificate schedule evidence");
assert.ok(normalizers.normalizePreviewRepair(actualProposal.repair, actualProposal), "preview repair");
assert.deepEqual(
  normalizers.previewProposalBinding(actual.preview.session.proposal),
  normalizers.previewProposalBinding(actualProposal),
  "session proposal binding",
);
assert.ok(normalizers.normalizePreviewGeography(
  actualProposal.geography_plan,
  actualProposal,
  routeValidation,
  actual.expected.draft,
), "preview geography");
assert.ok(normalizers.durationRouteEvidenceUnchanged(actualProposal.geography_plan), "unchanged route evidence");
const durationProposalKeys = [
  "schema_version", "state", "reason", "eligibility", "provenance", "execution_mode",
  "expected_revision", "session_revision", "draft_content_hash", "draft_operation_ids",
  "message", "parent_plan_id", "parent_plan_content_hash", "child_plan_id",
  "child_plan_content_hash", "certificate_id", "certificate_content_hash", "diff_id",
  "diff_content_hash", "diff_identity", "evaluation_status", "certificate_identity",
  "route_validation", "route_validation_identity", "evidence", "geography_plan",
  "compiled_request", "repair", "plan_diff", "schedule_impact", "schedule_impact_identity",
  "certificate_schedule_evidence", "decision_eligible", "ranking_eligible",
  "acceptance_eligible", "acceptance_blocking_code",
];
assert.ok(normalizers.exactKeys(actualProposal, durationProposalKeys), "proposal exact keys");
assert.ok(normalizers.exactKeys(actual.preview.session.proposal, durationProposalKeys), "session proposal exact keys");
assert.equal(actualProposal.draft_content_hash, await normalizers.computedDraftContentHash(actual.expected.draft));
assert.equal(
  normalizers.canonicalJson(actualProposal.compiled_request.operations),
  normalizers.canonicalJson(normalizers.draftHashRecords(actual.expected.draft)),
  "compiled duration request",
);
assert.deepEqual(actualProposal.route_validation_identity, {
  matrix_id: routeValidation.matrix_id,
  context_snapshot_id: routeValidation.context_snapshot_id,
  source_bundle_id: routeValidation.source_bundle_id,
  source_content_sha256: routeValidation.source_content_sha256,
});
assert.ok(normalizers.sameJson(actualProposal.evidence.route_validation, routeValidation));
assert.equal(actualProposal.repair.changed.affected_day_count, actualProposal.schedule_impact.affected_days.length);
const fullPreview = await normalizers.normalizeEvaluatedPreviewResponse(
  structuredClone(actual.preview), structuredClone(actual.expected),
);
assert.ok(fullPreview);
assert.equal(fullPreview.proposal.repair.tradeoffs.utility_retained, null);
assert.equal(fullPreview.proposal.repair.tradeoffs.weather_risk_delta, null);

const ineligiblePreview = await normalizers.normalizeEvaluatedPreviewResponse(
  structuredClone(actual.ineligible.preview), structuredClone(actual.ineligible.expected),
);
assert.ok(ineligiblePreview, "truthful ineligible duration preview");
const ineligibleProposal = ineligiblePreview.proposal;
assert.equal(ineligibleProposal.state, "ineligible");
assert.equal(ineligibleProposal.reason, "independent_evaluation_failed");
assert.equal(ineligibleProposal.evaluation_status, "FAILED");
assert.equal(ineligibleProposal.decision_eligible, false);
assert.equal(ineligibleProposal.ranking_eligible, false);
assert.equal(ineligibleProposal.acceptance_eligible, false);
assert.equal(ineligibleProposal.acceptance_blocking_code, "acceptance_not_enabled_until_w5");
assert.equal(ineligibleProposal.repair.tradeoffs.utility_retained, null);
assert.equal(ineligibleProposal.repair.tradeoffs.weather_risk_delta, null);
assert.deepEqual(ineligibleProposal.schedule_impact.child.blocking_codes, ["day_time_exceeded"]);
assert.deepEqual(ineligibleProposal.certificate_schedule_evidence.failure_codes, ["day_time_exceeded"]);
assert.ok(ineligibleProposal.schedule_impact.child.metrics.day_4_overrun_minutes > 0);

for (const [name, row] of [
  ["preferred", actual.preferred],
  ["mixed", actual.mixed],
]) {
  const scoped = new Function(
    "state",
    `${contractCode}\nreturn {normalizeDraftImpactPreview};`,
  )({ workspace: actual.workspace, session: row.session });
  assert.ok(scoped.normalizeDraftImpactPreview(structuredClone(row.impact)), name);
}
assert.deepEqual(actual.preferred.impact.summary, {
  operation_count: 1,
  evaluated_executable_count: 0,
  draft_only_count: 1,
  can_run_evaluated_preview: false,
  blocking_codes: ["duration_mode_evaluation_not_supported"],
});
assert.equal(actual.mixed.impact.summary.can_run_evaluated_preview, false);
assert.ok(actual.mixed.impact.summary.blocking_codes.includes(
  "draft_evaluated_operation_combination_unsupported",
));

const forge = (mutate) => {
  const payload = structuredClone(actual.preview);
  mutate(payload.proposal);
  payload.session.proposal = structuredClone(payload.proposal);
  return payload;
};
const contentPayload = (value) => {
  const payload = structuredClone(value);
  delete payload.content_hash;
  return payload;
};
const rehashPlanDiff = async (proposal) => {
  const hash = await normalizers.computedStableContentHash(proposal.plan_diff);
  proposal.diff_content_hash = hash;
  proposal.diff_identity.content_hash = hash;
};
const rehashSchedule = async (proposal) => {
  for (const side of ["parent", "child"]) {
    proposal.schedule_impact[side].content_hash = await normalizers.computedStableContentHash(
      contentPayload(proposal.schedule_impact[side]),
    );
  }
  const hash = await normalizers.computedStableContentHash(contentPayload(proposal.schedule_impact));
  proposal.schedule_impact.content_hash = hash;
  proposal.schedule_impact_identity.content_hash = hash;
};
const rehashCertificateSchedule = async (proposal) => {
  proposal.certificate_schedule_evidence.content_hash = await normalizers.computedStableContentHash(
    contentPayload(proposal.certificate_schedule_evidence),
  );
};
const coherentForge = async (basePreview, mutate, rehash) => {
  const payload = structuredClone(basePreview);
  await mutate(payload.proposal);
  await rehash(payload.proposal);
  payload.session.proposal = structuredClone(payload.proposal);
  return payload;
};
const duration = (proposal) => proposal.plan_diff.duration_changes[0];
const affectedDay = actual.preview.proposal.schedule_impact.affected_days[0];
const dayVisitMetric = `day_${affectedDay}_visit_minutes`;
const dayTotalMetric = `day_${affectedDay}_total_minutes`;
const firstLegIndex = 0;
const forgeries = [
  ["v1_duration_smuggling", (proposal) => { proposal.plan_diff.schema_version = "plan-diff-v1"; }],
  ["arbitrary_from_constraint", (proposal) => { duration(proposal).from_constraint = { mode: "exact", arbitrary: 45 }; }],
  ["to_constraint", (proposal) => { duration(proposal).to_constraint.preferred_minutes += 15; }],
  ["fallback_minutes", (proposal) => { duration(proposal).accounting_from_minutes += 5; }],
  ["raw_source_minutes", (proposal) => { duration(proposal).from_minutes = 45; }],
  ["raw_source", (proposal) => { duration(proposal).from_source = "visit_duration_minutes"; }],
  ["accounting_source", (proposal) => { duration(proposal).accounting_from_source = "visit_duration_minutes"; }],
  ["time_shift_smuggling", (proposal) => { proposal.plan_diff.time_shifts.push({ forged: true }); }],
  ["road_change_smuggling", (proposal) => { proposal.plan_diff.road_changes.push({ forged: true }); }],
  ["diff_schema_removed", (proposal) => { delete proposal.plan_diff.schema_version; }],
  ["schedule_default", (proposal) => { proposal.schedule_impact.configured_default_visit_minutes += 5; }],
  ["schedule_child_visit", (proposal) => { proposal.schedule_impact.child.metrics[dayVisitMetric] += 5; }],
  ["schedule_child_total", (proposal) => { proposal.schedule_impact.child.metrics[dayTotalMetric] += 5; }],
  ["schedule_scope", (proposal) => { proposal.schedule_impact.child.modeled_components.pop(); }],
  ["schedule_unavailable", (proposal) => { proposal.schedule_impact.child.unavailable_components[0] = "parking"; }],
  ["opening_missing", (proposal) => { proposal.schedule_impact.child.missing_opening_window_stop_ids.pop(); }],
  ["schedule_hash", (proposal) => { proposal.schedule_impact.content_hash = "0".repeat(16); }],
  ["schedule_identity", (proposal) => { proposal.schedule_impact_identity.route_matrix_id = `route_matrix_${"1".repeat(16)}`; }],
  ["certificate_identity", (proposal) => { proposal.certificate_identity.content_hash = "2".repeat(16); }],
  ["missing_risk_as_zero", (proposal) => { proposal.repair.tradeoffs.weather_risk_delta = 0; }],
  ["compiled_duration", (proposal) => { proposal.compiled_request.operations[0].parameters.duration.maximum_minutes += 15; }],
  ["child_route", (proposal) => {
    proposal.geography_plan.validated_legs.features[firstLegIndex].properties.duration_s += 1;
  }],
  ["route_geometry", (proposal) => {
    const geometry = proposal.geography_plan.validated_legs.features[firstLegIndex].geometry.coordinates;
    geometry.splice(1, Math.max(0, geometry.length - 2));
  }],
];
for (const [name, mutate] of forgeries) {
  assert.equal(
    await normalizers.normalizeEvaluatedPreviewResponse(forge(mutate), actual.expected),
    null,
    name,
  );
}

const coherentForgeries = [
  ["rehashed_from_constraint", actual.preview, actual.expected, (proposal) => {
    duration(proposal).from_constraint = { mode: "exact", arbitrary: 45 };
  }, rehashPlanDiff],
  ["rehashed_raw_source", actual.preview, actual.expected, (proposal) => {
    duration(proposal).from_minutes = 45;
    duration(proposal).from_source = "visit_duration_minutes";
  }, rehashPlanDiff],
  ["rehashed_owner_cost", actual.preview, actual.expected, (proposal) => {
    duration(proposal).owner_strength = "soft";
    duration(proposal).cost = 2.5;
    proposal.plan_diff.weighted_edit_cost = 2.5;
  }, rehashPlanDiff],
  ["rehashed_default", actual.preview, actual.expected, (proposal) => {
    proposal.schedule_impact.configured_default_visit_minutes = 50;
    duration(proposal).accounting_from_minutes = 50;
    duration(proposal).delta_minutes = 10;
  }, async (proposal) => {
    await rehashPlanDiff(proposal);
    await rehashSchedule(proposal);
  }],
  ["rehashed_cost_policy", actual.preview, actual.expected, (proposal) => {
    proposal.schedule_impact.duration_cost_policy.base_change_cost = 0.5;
  }, rehashSchedule],
  ["rehashed_scope", actual.preview, actual.expected, (proposal) => {
    proposal.schedule_impact.child.modeled_components[0] = "straight_line_travel";
  }, rehashSchedule],
  ["rehashed_warning", actual.preview, actual.expected, (proposal) => {
    proposal.certificate_schedule_evidence.warning_codes = ["forged_warning"];
  }, rehashCertificateSchedule],
  ["rehashed_failure", actual.ineligible.preview, actual.ineligible.expected, (proposal) => {
    proposal.schedule_impact.child.blocking_codes = ["forged_failure"];
    proposal.certificate_schedule_evidence.failure_codes = ["forged_failure"];
  }, async (proposal) => {
    await rehashSchedule(proposal);
    await rehashCertificateSchedule(proposal);
  }],
  ["coherent_proposal_binding", actual.preview, actual.expected, (proposal) => {
    proposal.decision_eligible = false;
  }, async () => {}],
];
for (const [name, basePreview, expected, mutate, rehash] of coherentForgeries) {
  assert.equal(
    await normalizers.normalizeEvaluatedPreviewResponse(
      await coherentForge(basePreview, mutate, rehash), expected,
    ),
    null,
    name,
  );
}

const waitingProposal = structuredClone(actualProposal);
const waitingSchedule = waitingProposal.schedule_impact;
const waitingDay = waitingSchedule.affected_days[0];
const waitingMetrics = waitingSchedule.child.metrics;
waitingMetrics[`day_${waitingDay}_waiting_minutes`] += 7;
waitingMetrics[`day_${waitingDay}_total_minutes`] += 7;
waitingMetrics[`day_${waitingDay}_slack_minutes`] = Math.max(
  0,
  waitingMetrics[`day_${waitingDay}_limit_minutes`]
    - waitingMetrics[`day_${waitingDay}_total_minutes`],
);
waitingMetrics[`day_${waitingDay}_overrun_minutes`] = Math.max(
  0,
  waitingMetrics[`day_${waitingDay}_total_minutes`]
    - waitingMetrics[`day_${waitingDay}_limit_minutes`],
);
await rehashSchedule(waitingProposal);
const waitingImpact = await normalizers.normalizeDurationScheduleImpact(
  waitingSchedule,
  waitingProposal.schedule_impact_identity,
  waitingProposal,
  planDiff,
  routeValidation,
);
assert.ok(waitingImpact, "opening-window waiting may change modeled total");
for (const [name, metric] of Object.entries(waitingMetrics)) {
  waitingProposal.certificate_schedule_evidence.schedule_metrics[name] = metric;
}
await rehashCertificateSchedule(waitingProposal);
assert.ok(await normalizers.normalizeCertificateScheduleEvidence(
  waitingProposal.certificate_schedule_evidence,
  waitingProposal,
  waitingImpact,
), "certificate schedule permits evaluator-owned waiting changes");

console.log("evaluated duration frontend contract and forgeries passed");
