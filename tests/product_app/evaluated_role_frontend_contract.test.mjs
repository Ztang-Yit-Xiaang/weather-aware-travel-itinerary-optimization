import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const contractPath = process.env.ACTUAL_EVALUATED_ROLE_CONTRACT_PATH;
assert.ok(contractPath, "ACTUAL_EVALUATED_ROLE_CONTRACT_PATH is required");
const mutationToken = process.env.ACTUAL_EVALUATED_ROLE_TOKEN;
assert.ok(mutationToken, "ACTUAL_EVALUATED_ROLE_TOKEN is required");
const actual = JSON.parse(fs.readFileSync(contractPath, "utf8"));
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "..");
const appPath = path.join(root, "src", "itinerary_system", "product_app", "static", "js", "app.js");
const source = fs.readFileSync(appPath, "utf8");
const contractCode = source.slice(
  source.indexOf("const TYPED_EDIT_OPERATIONS"),
  source.indexOf("function safeExternalLink"),
);
const state = {
  workspace: actual.workspace,
  session: actual.eligible.before_session,
  token: mutationToken,
};
const normalizers = new Function(
  "state",
  `${contractCode}\nreturn {
    normalizeTypedEditCapabilities, normalizeDraftImpactPreview, normalizeEvaluatedPreviewResponse,
    normalizeRoleConstraintEvidence, normalizeRolePlanDiff, normalizeRoleCostPolicy,
    normalizeRoleStates, normalizeRoleImpact, normalizeRoleCertificateEvidence,
    normalizeRoleCertificateRecord,
    normalizeRoleGeography, normalizeRouteValidation, normalizePreviewRepair,
    normalizePreviewGeography, normalizeTimeWindowRouteEvidence,
    normalizeTimeWindowScheduleAccounting, durationRouteEvidenceUnchanged,
    previewProposalBinding, exactKeys, sameJson, canonicalJson,
    computedDraftContentHash, computedStableContentHash,
    computedRoleProposalIntegrityDigest, normalizeRoleProposalIntegrity, draftHashRecords,
  };`,
)(state);

assert.ok(normalizers.normalizeTypedEditCapabilities(
  structuredClone(actual.workspace.typed_edit_capabilities),
));
assert.ok(await normalizers.normalizeRoleConstraintEvidence(
  structuredClone(actual.workspace.role_constraint_evidence),
));
assert.ok(normalizers.normalizeDraftImpactPreview(structuredClone(actual.eligible.impact)));

const proposalKeys = [
  "schema_version", "state", "reason", "eligibility", "provenance", "execution_mode",
  "expected_revision", "session_revision", "draft_content_hash", "draft_operation_ids",
  "message", "parent_plan_id", "parent_plan_content_hash", "child_plan_id",
  "child_plan_content_hash", "certificate_id", "certificate_content_hash", "diff_id",
  "diff_content_hash", "diff_identity", "evaluation_status", "certificate_identity",
  "route_validation", "route_validation_identity", "evidence", "geography_plan",
  "compiled_request", "repair", "plan_diff", "role_impact", "role_impact_identity",
  "certificate_role_evidence", "role_proposal_integrity", "decision_eligible", "ranking_eligible",
  "acceptance_eligible", "acceptance_blocking_code",
];

const row = actual.eligible;
const proposal = row.preview.proposal;
assert.ok(normalizers.exactKeys(proposal, proposalKeys));
assert.ok(normalizers.exactKeys(row.preview.session.proposal, proposalKeys));
assert.ok(await normalizers.normalizeRoleProposalIntegrity(
  proposal.role_proposal_integrity, proposal,
));
assert.equal(proposal.draft_content_hash, await normalizers.computedDraftContentHash(row.expected.draft));
assert.equal(
  normalizers.canonicalJson(proposal.compiled_request.operations),
  normalizers.canonicalJson(normalizers.draftHashRecords(row.expected.draft)),
);
const routeValidation = normalizers.normalizeRouteValidation(proposal.route_validation);
assert.ok(routeValidation);
const planDiff = await normalizers.normalizeRolePlanDiff(
  proposal.plan_diff, proposal, row.expected.draft,
);
assert.ok(planDiff, "PlanDiff v4");
assert.ok(normalizers.exactKeys(proposal.role_impact, [
  "schema_version", "evaluator_version", "route_matrix_id", "configured_default_visit_minutes",
  "configured_day_start_minute", "supported_evaluated_roles", "semantic_scope",
  "role_constraint_evidence", "route_leg_evidence", "role_cost_policy", "target_stop_ids",
  "affected_days", "parent_role_states", "child_role_states", "parent_schedule",
  "child_schedule", "invariance", "content_hash",
]), "role impact keys");
assert.ok(normalizers.normalizeRoleStates(
  proposal.role_impact.parent_role_states, planDiff, false,
), "parent role states");
assert.ok(normalizers.normalizeRoleStates(
  proposal.role_impact.child_role_states, planDiff, true,
), "child role states");
assert.ok(normalizers.normalizeRoleCostPolicy(proposal.role_impact.role_cost_policy));
assert.ok(await normalizers.normalizeRoleConstraintEvidence(
  proposal.role_impact.role_constraint_evidence,
), "impact role constraints");
assert.deepEqual(
  proposal.role_impact.role_constraint_evidence,
  actual.workspace.role_constraint_evidence,
  "workspace and impact role constraints",
);
assert.deepEqual(proposal.role_impact.supported_evaluated_roles, [
  "attraction", "activity", "meal", "rest_stop", "scenic_stop",
]);
assert.deepEqual(proposal.role_impact.affected_days, [4]);
assert.deepEqual(proposal.role_impact.target_stop_ids, ["stearns_wharf"]);
assert.equal(proposal.role_impact.configured_default_visit_minutes, 45);
assert.equal(proposal.role_impact.configured_day_start_minute, 540);
assert.ok(normalizers.normalizeTimeWindowRouteEvidence(
  proposal.role_impact.route_leg_evidence, routeValidation,
), "role route evidence");
const roleScheduleCommon = {
  evaluatorVersion: proposal.role_impact.evaluator_version,
  routeMatrixId: proposal.role_impact.route_matrix_id,
  routeValidation,
  configuredDefaultVisitMinutes: proposal.role_impact.configured_default_visit_minutes,
  configuredDayStartMinute: proposal.role_impact.configured_day_start_minute,
  routeLegEvidence: proposal.role_impact.route_leg_evidence,
};
assert.ok(await normalizers.normalizeTimeWindowScheduleAccounting(
  proposal.role_impact.parent_schedule,
  {
    ...roleScheduleCommon,
    planId: proposal.parent_plan_id,
    planContentHash: proposal.parent_plan_content_hash,
  },
), "role parent schedule");
assert.ok(await normalizers.normalizeTimeWindowScheduleAccounting(
  proposal.role_impact.child_schedule,
  {
    ...roleScheduleCommon,
    planId: proposal.child_plan_id,
    planContentHash: proposal.child_plan_content_hash,
  },
), "role child schedule");
assert.ok(normalizers.exactKeys(proposal.role_impact_identity, [
  "content_hash", "evaluator_version", "route_matrix_id", "parent_plan_id",
  "parent_plan_content_hash", "child_plan_id", "child_plan_content_hash",
  "certificate_id", "certificate_content_hash", "role_constraint_evidence_content_hash",
]), "role impact identity keys");
const roleImpactPayload = structuredClone(proposal.role_impact);
delete roleImpactPayload.content_hash;
assert.equal(
  proposal.role_impact.content_hash,
  await normalizers.computedStableContentHash(roleImpactPayload),
  "role impact content hash",
);
assert.equal(
  proposal.role_impact_identity.role_constraint_evidence_content_hash,
  proposal.role_impact.role_constraint_evidence.content_hash,
  "role constraint identity",
);
const parentScheduleInvariant = structuredClone(proposal.role_impact.parent_schedule);
const childScheduleInvariant = structuredClone(proposal.role_impact.child_schedule);
for (const value of [parentScheduleInvariant, childScheduleInvariant]) {
  delete value.plan_id;
  delete value.plan_content_hash;
  delete value.content_hash;
}
assert.deepEqual(parentScheduleInvariant, childScheduleInvariant, "role schedule invariance");
assert.equal(proposal.role_impact_identity.content_hash, proposal.role_impact.content_hash);
assert.equal(proposal.role_impact_identity.evaluator_version, proposal.role_impact.evaluator_version);
assert.equal(proposal.role_impact_identity.route_matrix_id, proposal.role_impact.route_matrix_id);
assert.equal(proposal.role_impact_identity.parent_plan_id, proposal.parent_plan_id);
assert.equal(proposal.role_impact_identity.parent_plan_content_hash, proposal.parent_plan_content_hash);
assert.equal(proposal.role_impact_identity.child_plan_id, proposal.child_plan_id);
assert.equal(proposal.role_impact_identity.child_plan_content_hash, proposal.child_plan_content_hash);
assert.equal(proposal.role_impact_identity.certificate_id, proposal.certificate_id);
assert.equal(proposal.role_impact_identity.certificate_content_hash, proposal.certificate_content_hash);
const roleImpact = await normalizers.normalizeRoleImpact(
  proposal.role_impact, proposal.role_impact_identity, proposal, planDiff, routeValidation,
);
assert.ok(roleImpact, "role impact");
assert.ok(await normalizers.normalizeRoleCertificateEvidence(
  proposal.certificate_role_evidence, proposal, roleImpact, planDiff,
), "role certificate evidence");
assert.ok(normalizers.normalizePreviewRepair(proposal.repair, proposal));
assert.ok(normalizers.normalizePreviewGeography(
  proposal.geography_plan, proposal, routeValidation, row.expected.draft,
));
assert.ok(normalizers.normalizeRoleGeography(proposal.geography_plan, planDiff, roleImpact));
assert.ok(normalizers.durationRouteEvidenceUnchanged(proposal.geography_plan));
assert.deepEqual(
  normalizers.previewProposalBinding(row.preview.session.proposal),
  normalizers.previewProposalBinding(proposal),
);
const full = await normalizers.normalizeEvaluatedPreviewResponse(
  structuredClone(row.preview), structuredClone(row.expected),
);
assert.ok(full, "actual API response passes full JavaScript trust boundary");
assert.equal(full.proposal.state, "eligible");
assert.equal(full.proposal.decision_eligible, true);
assert.equal(full.proposal.ranking_eligible, false);
assert.equal(full.proposal.acceptance_eligible, false);
assert.equal(full.proposal.acceptance_blocking_code, "acceptance_not_enabled_until_w5");
assert.equal(full.proposal.repair.tradeoffs.utility_retained, null);
assert.equal(full.proposal.repair.tradeoffs.weather_risk_delta, null);
assert.deepEqual(full.proposal.role_impact.role_constraint_evidence.constraints, []);
assert.deepEqual(full.proposal.role_impact.invariance, {
  sequence_unchanged: true,
  ordered_days_unchanged: true,
  route_ids_by_day_unchanged: true,
  route_legs_unchanged: true,
  schedule_accounting_unchanged: true,
});
const parentTarget = actual.workspace.geography.plans
  .find((plan) => plan.plan_id === full.proposal.parent_plan_id)
  .stops.features.find((feature) => feature.properties.stop_id === "stearns_wharf");
const childTarget = full.proposal.geography_plan.stops.features
  .find((feature) => feature.properties.stop_id === "stearns_wharf");
assert.equal(parentTarget.properties.role, "original");
assert.equal(parentTarget.properties.itinerary_role, null);
assert.equal(parentTarget.properties.itinerary_role_source, "unavailable");
assert.equal(childTarget.properties.role, "draft_preview");
assert.equal(childTarget.properties.itinerary_role, "meal");
assert.equal(childTarget.properties.itinerary_role_source, "user_declared_itinerary_role");

const structuralState = { workspace: actual.workspace, session: actual.structural.session };
const structuralNormalizer = new Function(
  "state", `${contractCode}\nreturn {normalizeDraftImpactPreview};`,
)(structuralState);
assert.ok(structuralNormalizer.normalizeDraftImpactPreview(structuredClone(actual.structural.impact)));
assert.deepEqual(actual.structural.impact.summary.blocking_codes, [
  "stop_role_evaluation_not_supported",
]);
assert.equal(actual.structural.impact.summary.can_run_evaluated_preview, false);

const mixedState = { workspace: actual.workspace, session: actual.mixed.session };
const mixedNormalizer = new Function(
  "state", `${contractCode}\nreturn {normalizeDraftImpactPreview};`,
)(mixedState);
assert.ok(mixedNormalizer.normalizeDraftImpactPreview(structuredClone(actual.mixed.impact)));
assert.equal(actual.mixed.impact.summary.can_run_evaluated_preview, false);
assert.ok(actual.mixed.impact.summary.blocking_codes.includes(
  "draft_evaluated_operation_combination_unsupported",
));

const withoutContentHash = (value) => {
  const payload = structuredClone(value);
  delete payload.content_hash;
  return payload;
};
const rehashDiff = async (candidate) => {
  const hash = await normalizers.computedStableContentHash(candidate.plan_diff);
  candidate.diff_content_hash = hash;
  candidate.diff_identity.content_hash = hash;
};
const rehashRoleImpact = async (candidate) => {
  const hash = await normalizers.computedStableContentHash(
    withoutContentHash(candidate.role_impact),
  );
  candidate.role_impact.content_hash = hash;
  candidate.role_impact_identity.content_hash = hash;
};
const rehashRoleConstraints = async (candidate) => {
  candidate.role_impact.role_constraint_evidence.content_hash = await normalizers.computedStableContentHash(
    withoutContentHash(candidate.role_impact.role_constraint_evidence),
  );
  candidate.role_impact_identity.role_constraint_evidence_content_hash =
    candidate.role_impact.role_constraint_evidence.content_hash;
  candidate.certificate_role_evidence.role_constraint_evidence_content_hash =
    candidate.role_impact.role_constraint_evidence.content_hash;
};
const rehashCertificate = async (candidate) => {
  candidate.certificate_role_evidence.content_hash = await normalizers.computedStableContentHash(
    withoutContentHash(candidate.certificate_role_evidence),
  );
};
const rehashCertificateRecord = async (candidate) => {
  const record = candidate.certificate_role_evidence.certificate_record;
  const hash = await normalizers.computedStableContentHash(withoutContentHash(record));
  record.content_hash = hash;
  candidate.certificate_content_hash = hash;
  candidate.certificate_identity.content_hash = hash;
  candidate.role_impact_identity.certificate_content_hash = hash;
  candidate.certificate_role_evidence.certificate_content_hash = hash;
};
const rehashSchedule = async (candidate, side) => {
  candidate.role_impact[side].content_hash = await normalizers.computedStableContentHash(
    withoutContentHash(candidate.role_impact[side]),
  );
};
const resignProposal = async (candidate) => {
  candidate.role_proposal_integrity.digest =
    await normalizers.computedRoleProposalIntegrityDigest(candidate);
};
const reject = async (label, mutate) => {
  const forged = structuredClone(row.preview);
  await mutate(forged.proposal);
  await resignProposal(forged.proposal);
  forged.session.proposal = structuredClone(forged.proposal);
  assert.equal(
    await normalizers.normalizeEvaluatedPreviewResponse(forged, structuredClone(row.expected)),
    null,
    label,
  );
};

const softConstraintResponse = structuredClone(row.preview);
const softConstraintEvidencePayload = {
  schema_version: "evaluated-role-constraint-evidence-v1",
  parent_plan_id: proposal.parent_plan_id,
  parent_plan_content_hash: proposal.parent_plan_content_hash,
  constraints: [{
    constraint_id: "role_soft_stearns_wharf",
    target_stop_id: "stearns_wharf",
    required_role: "activity",
    strength: "soft",
    scope: "stop",
    relation: "role",
    relaxation_policy: "always",
    permission_semantics: "weighted_mismatch_allowed",
  }],
};
const softConstraintEvidence = {
  ...softConstraintEvidencePayload,
  content_hash: await normalizers.computedStableContentHash(softConstraintEvidencePayload),
};
softConstraintResponse.proposal.plan_diff.role_changes[0].owner_strength = "soft";
softConstraintResponse.proposal.plan_diff.role_changes[0].cost = 2.5;
softConstraintResponse.proposal.plan_diff.weighted_edit_cost = 2.5;
softConstraintResponse.proposal.repair.tradeoffs.weighted_edit_cost = 2.5;
softConstraintResponse.proposal.role_impact.role_constraint_evidence = softConstraintEvidence;
softConstraintResponse.proposal.certificate_role_evidence.warning_codes = [
  "owned_role_constraint_unsatisfied", "opening_window_evidence_missing",
];
softConstraintResponse.proposal.certificate_role_evidence.certificate_record.warnings.unshift({
  code: "owned_role_constraint_unsatisfied",
  message: "Owned itinerary-role constraint 'role_soft_stearns_wharf' is not satisfied.",
  severity: "warning",
  category: "soft",
  evidence_refs: ["role_soft_stearns_wharf"],
});
softConstraintResponse.proposal.certificate_role_evidence.certificate_record.warning_count = 2;
softConstraintResponse.proposal.certificate_role_evidence.certificate_record.nonblocking_warning_count = 2;
await rehashDiff(softConstraintResponse.proposal);
await rehashRoleConstraints(softConstraintResponse.proposal);
await rehashRoleImpact(softConstraintResponse.proposal);
await rehashCertificateRecord(softConstraintResponse.proposal);
await rehashCertificate(softConstraintResponse.proposal);
await resignProposal(softConstraintResponse.proposal);
softConstraintResponse.session.proposal = structuredClone(softConstraintResponse.proposal);
const softState = {
  workspace: { ...structuredClone(actual.workspace), role_constraint_evidence: softConstraintEvidence },
  session: actual.eligible.before_session,
  token: mutationToken,
};
const softNormalizers = new Function(
  "state", `${contractCode}\nreturn {normalizeEvaluatedPreviewResponse};`,
)(softState);
const normalizedSoftConstraint = await softNormalizers.normalizeEvaluatedPreviewResponse(
  structuredClone(softConstraintResponse), structuredClone(row.expected),
);
assert.ok(normalizedSoftConstraint, "valid nonempty soft role-constraint evidence");
assert.equal(normalizedSoftConstraint.proposal.plan_diff.role_changes[0].owner_strength, "soft");
assert.equal(normalizedSoftConstraint.proposal.plan_diff.weighted_edit_cost, 2.5);
assert.deepEqual(normalizedSoftConstraint.proposal.certificate_role_evidence.warning_codes, [
  "owned_role_constraint_unsatisfied", "opening_window_evidence_missing",
]);
assert.equal(normalizedSoftConstraint.proposal.ranking_eligible, false);
assert.equal(normalizedSoftConstraint.proposal.acceptance_eligible, false);

const hardMismatchWorkspace = structuredClone(softState.workspace);
hardMismatchWorkspace.role_constraint_evidence.constraints[0].strength = "locked";
hardMismatchWorkspace.role_constraint_evidence.constraints[0].relaxation_policy = "never";
hardMismatchWorkspace.role_constraint_evidence.constraints[0].permission_semantics =
  "explicit_permission_required_for_mismatch";
hardMismatchWorkspace.role_constraint_evidence.content_hash = await normalizers.computedStableContentHash(
  withoutContentHash(hardMismatchWorkspace.role_constraint_evidence),
);
const hardMismatchState = { workspace: hardMismatchWorkspace, session: actual.eligible.before_session };
const hardMismatchNormalizer = new Function(
  "state", `${contractCode}\nreturn {normalizeRolePlanDiff};`,
)(hardMismatchState);
assert.equal(
  await hardMismatchNormalizer.normalizeRolePlanDiff(
    structuredClone(proposal.plan_diff), proposal, row.expected.draft,
  ),
  null,
  "explicit-permission role mismatch cannot normalize as executable",
);

await reject("PlanDiff v3 cannot smuggle role changes", async (candidate) => {
  candidate.plan_diff.schema_version = "plan-diff-v3";
  await rehashDiff(candidate);
});
await reject("structural role cannot become evaluated", async (candidate) => {
  candidate.plan_diff.role_changes[0].to_role = "lodging";
  await rehashDiff(candidate);
});
await reject("role diff must match ordered draft", async (candidate) => {
  candidate.plan_diff.role_changes[0].to_role = "activity";
  await rehashDiff(candidate);
});
await reject("role source provenance is exact", async (candidate) => {
  candidate.plan_diff.role_changes[0].to_source = "place_category_inference";
  await rehashDiff(candidate);
});
await reject("generic owner cannot fabricate role ownership", async (candidate) => {
  candidate.plan_diff.role_changes[0].owner_strength = "locked";
  candidate.plan_diff.role_changes[0].cost = 250;
  await rehashDiff(candidate);
});
await reject("forged role constraint evidence is rejected", async (candidate) => {
  candidate.role_impact.role_constraint_evidence.constraints = [{
    constraint_id: "forged_role_constraint",
    target_stop_id: "stearns_wharf",
    required_role: "meal",
    strength: "soft",
    scope: "stop",
    relation: "role",
    relaxation_policy: "always",
    permission_semantics: "weighted_mismatch_allowed",
  }];
  await rehashRoleConstraints(candidate);
  await rehashCertificate(candidate);
  await rehashRoleImpact(candidate);
});
await reject("existence constraint cannot enter role evidence", async (candidate) => {
  candidate.role_impact.role_constraint_evidence.constraints = [{
    constraint_id: "forged_existence_constraint",
    target_stop_id: "stearns_wharf",
    required_role: "meal",
    strength: "soft",
    scope: "stop",
    relation: "must_keep",
    relaxation_policy: "always",
    permission_semantics: "weighted_mismatch_allowed",
  }];
  await rehashRoleConstraints(candidate);
  await rehashCertificate(candidate);
  await rehashRoleImpact(candidate);
});
await reject("semantic scope cannot be self-rehashed", async (candidate) => {
  candidate.role_impact.semantic_scope.semantic_fit_claim = "high";
  candidate.certificate_role_evidence.semantic_scope.semantic_fit_claim = "high";
  await rehashRoleImpact(candidate);
  await rehashCertificate(candidate);
});
await reject("non-target role state is immutable", async (candidate) => {
  candidate.role_impact.child_role_states[0].itinerary_role = "activity";
  candidate.role_impact.child_role_states[0].itinerary_role_source = "user_declared_itinerary_role";
  await rehashRoleImpact(candidate);
});
await reject("route evidence cannot change", async (candidate) => {
  candidate.role_impact.route_leg_evidence[0].duration_minutes += 1;
  await rehashRoleImpact(candidate);
});
await reject("schedule metrics remain invariant", async (candidate) => {
  candidate.role_impact.child_schedule.metrics.day_4_visit_minutes += 1;
  await rehashSchedule(candidate, "child_schedule");
  await rehashRoleImpact(candidate);
});
await reject("certificate warning set is exact", async (candidate) => {
  candidate.certificate_role_evidence.warning_codes = ["fabricated_role_warning"];
  await rehashCertificate(candidate);
});
await reject("uniform forged certificate hash is recomputed and rejected", async (candidate) => {
  const forgedHash = "f".repeat(16);
  candidate.certificate_content_hash = forgedHash;
  candidate.certificate_identity.content_hash = forgedHash;
  candidate.role_impact_identity.certificate_content_hash = forgedHash;
  candidate.certificate_role_evidence.certificate_content_hash = forgedHash;
  candidate.certificate_role_evidence.certificate_record.content_hash = forgedHash;
  await rehashCertificate(candidate);
});
await reject("certificate record rejects unknown fields", async (candidate) => {
  candidate.certificate_role_evidence.certificate_record.raw_provider_payload = "forged";
  await rehashCertificateRecord(candidate);
  await rehashCertificate(candidate);
});
const staleIntegrity = structuredClone(row.preview);
staleIntegrity.proposal.message = `${proposal.message} forged`;
staleIntegrity.session.proposal = structuredClone(staleIntegrity.proposal);
assert.equal(
  await normalizers.normalizeEvaluatedPreviewResponse(
    staleIntegrity, structuredClone(row.expected),
  ),
  null,
  "proposal mutation with a stale integrity digest is rejected",
);
const replacedIntegrity = structuredClone(row.preview);
replacedIntegrity.proposal.role_proposal_integrity.digest = "0".repeat(64);
replacedIntegrity.session.proposal = structuredClone(replacedIntegrity.proposal);
assert.equal(
  await normalizers.normalizeEvaluatedPreviewResponse(
    replacedIntegrity, structuredClone(row.expected),
  ),
  null,
  "attacker-replaced integrity digest is rejected",
);
const wrongIntegrityAlgorithm = structuredClone(row.preview);
wrongIntegrityAlgorithm.proposal.role_proposal_integrity.algorithm = "hmac-sha1";
wrongIntegrityAlgorithm.session.proposal = structuredClone(wrongIntegrityAlgorithm.proposal);
assert.equal(
  await normalizers.normalizeEvaluatedPreviewResponse(
    wrongIntegrityAlgorithm, structuredClone(row.expected),
  ),
  null,
  "proposal integrity algorithm is exact",
);
const expandedIntegrity = structuredClone(row.preview);
expandedIntegrity.proposal.role_proposal_integrity.key_id = "forged";
expandedIntegrity.session.proposal = structuredClone(expandedIntegrity.proposal);
assert.equal(
  await normalizers.normalizeEvaluatedPreviewResponse(
    expandedIntegrity, structuredClone(row.expected),
  ),
  null,
  "proposal integrity rejects unknown fields",
);
const forgedEvaluatedAt = structuredClone(row.preview);
forgedEvaluatedAt.proposal.certificate_role_evidence.certificate_record.evaluated_at =
  "2026-08-08T00:00:00Z";
await rehashCertificateRecord(forgedEvaluatedAt.proposal);
await rehashCertificate(forgedEvaluatedAt.proposal);
forgedEvaluatedAt.session.proposal = structuredClone(forgedEvaluatedAt.proposal);
assert.equal(
  await normalizers.normalizeEvaluatedPreviewResponse(
    forgedEvaluatedAt, structuredClone(row.expected),
  ),
  null,
  "evaluated_at and certificate rehash cannot bypass the proposal HMAC",
);
await reject("ranking evidence cannot be fabricated", async (candidate) => {
  candidate.ranking_eligible = true;
});
await reject("W5 acceptance stays disabled", async (candidate) => {
  candidate.acceptance_eligible = true;
  candidate.acceptance_blocking_code = null;
});
await reject("missing weather risk cannot become zero", async (candidate) => {
  candidate.repair.tradeoffs.weather_risk_delta = 0;
});
await reject("place/category properties are immutable", async (candidate) => {
  const target = candidate.geography_plan.stops.features
    .find((feature) => feature.properties.stop_id === "stearns_wharf");
  target.properties.place_categories = ["lodging"];
});
await reject("map plan role cannot become itinerary role", async (candidate) => {
  const target = candidate.geography_plan.stops.features
    .find((feature) => feature.properties.stop_id === "stearns_wharf");
  target.properties.role = "meal";
});
await reject("child role source must remain user declared", async (candidate) => {
  const target = candidate.geography_plan.stops.features
    .find((feature) => feature.properties.stop_id === "stearns_wharf");
  target.properties.itinerary_role_source = "category_inference";
});
await reject("route geometry cannot change", async (candidate) => {
  candidate.geography_plan.validated_legs.features[0].geometry.coordinates[0][0] += 0.01;
});
const forgedSessionBinding = structuredClone(row.preview);
forgedSessionBinding.session.proposal.message = `${proposal.message} forged`;
assert.equal(
  await normalizers.normalizeEvaluatedPreviewResponse(
    forgedSessionBinding, structuredClone(row.expected),
  ),
  null,
  "session proposal binding is exact",
);

console.log("evaluated role frontend contract and forgeries passed");
