import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const contractPath = process.env.ACTUAL_EVALUATED_REORDER_CONTRACT_PATH;
assert.ok(contractPath, "ACTUAL_EVALUATED_REORDER_CONTRACT_PATH is required");
const actual = JSON.parse(fs.readFileSync(contractPath, "utf8"));
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "..");
const appPath = path.join(root, "src", "itinerary_system", "product_app", "static", "js", "app.js");
const source = fs.readFileSync(appPath, "utf8");
const contractCode = source.slice(
  source.indexOf("const TYPED_EDIT_OPERATIONS"),
  source.indexOf("function safeExternalLink"),
);
const runtimeState = {
  workspace: {
    typed_edit_capabilities: actual.capabilities,
    draft_capabilities: actual.draft_capabilities,
    map_edit_capabilities: actual.map_edit_capabilities,
    timeline: actual.timeline,
  },
  session: actual.before_session,
};
const normalizers = new Function(
  "state",
  `${contractCode}\nreturn {normalizeTypedEditCapabilities, normalizeDraftImpactPreview, normalizeEvaluatedPreviewResponse};`,
)(runtimeState);

assert.ok(normalizers.normalizeTypedEditCapabilities(structuredClone(actual.capabilities)));
assert.ok(normalizers.normalizeDraftImpactPreview(structuredClone(actual.impact)));
assert.ok(await normalizers.normalizeEvaluatedPreviewResponse(
  actual.preview, actual.expected,
));
assert.ok(await normalizers.normalizeEvaluatedPreviewResponse(
  actual.legacy.preview, actual.legacy.expected,
));
const mixedNormalizers = new Function(
  "state",
  `${contractCode}\nreturn {normalizeDraftImpactPreview};`,
)({
  workspace: runtimeState.workspace,
  session: actual.mixed.session,
});
assert.ok(mixedNormalizers.normalizeDraftImpactPreview(actual.mixed.impact));
for (const [name, impact] of [
  ["mixed_can_run", {
    ...actual.mixed.impact,
    summary: { ...actual.mixed.impact.summary, can_run_evaluated_preview: true },
  }],
  ["mixed_blocker_removed", {
    ...actual.mixed.impact,
    summary: { ...actual.mixed.impact.summary, blocking_codes: [] },
  }],
  ["mixed_blocker_attached_to_row", {
    ...actual.mixed.impact,
    operations: actual.mixed.impact.operations.map((row, index) => index === 0 ? {
      ...row,
      blocking_codes: ["draft_evaluated_operation_combination_unsupported"],
    } : row),
  }],
]) {
  assert.equal(mixedNormalizers.normalizeDraftImpactPreview(impact), null, name);
}

const withProposal = (patch = {}, sessionPatch = {}) => {
  const proposal = { ...actual.preview.proposal, ...patch };
  return { proposal, session: { ...actual.preview.session, proposal: { ...proposal }, ...sessionPatch } };
};
const forgedRoute = {
  ...actual.preview.proposal.route_validation,
  matrix_id: `route_matrix_${"5".repeat(16)}`,
};
const dayFourLegIndex = actual.preview.proposal.geography_plan.validated_legs.features.findIndex(
  (feature) => feature.properties.day === 4,
);
const dayFourLeg = actual.preview.proposal.geography_plan.validated_legs.features[dayFourLegIndex];
const substitutedDayFourLeg = {
  ...dayFourLeg,
  geometry: {
    type: "LineString",
    coordinates: [dayFourLeg.geometry.coordinates[0], dayFourLeg.geometry.coordinates.at(-1)],
  },
};
const forgeries = [
  ["top_extra", () => ({ ...actual.preview, extra: true })],
  ["proposal_extra", () => withProposal({ extra: true })],
  ["session_extra", () => withProposal({}, { extra: true })],
  ["session_revision", () => withProposal(
    { session_revision: actual.preview.proposal.session_revision + 1 },
    { revision: actual.preview.session.revision + 1 },
  )],
  ["draft_hash", () => withProposal({ draft_content_hash: "0".repeat(16) })],
  ["operation_order", () => withProposal({ draft_operation_ids: [] })],
  ["parent_hash", () => withProposal({ parent_plan_content_hash: "1".repeat(16) })],
  ["child_id", () => withProposal({ child_plan_id: `plan_${"2".repeat(16)}` })],
  ["diff_parent", () => withProposal({
    diff_identity: {
      ...actual.preview.proposal.diff_identity,
      parent_plan_id: `plan_${"3".repeat(16)}`,
    },
  })],
  ["certificate_child", () => withProposal({
    certificate_identity: {
      ...actual.preview.proposal.certificate_identity,
      plan_id: `plan_${"4".repeat(16)}`,
    },
  })],
  ["route_matrix", () => withProposal({
    route_validation: forgedRoute,
    evidence: { ...actual.preview.proposal.evidence, route_validation: forgedRoute },
  })],
  ["route_source", () => withProposal({
    route_validation_identity: {
      ...actual.preview.proposal.route_validation_identity,
      source_content_sha256: "6".repeat(64),
    },
  })],
  ["geography_hash", () => withProposal({
    geography_plan: {
      ...actual.preview.proposal.geography_plan,
      content_hash: "7".repeat(16),
    },
  })],
  ["route_geometry", () => withProposal({
    geography_plan: {
      ...actual.preview.proposal.geography_plan,
      validated_legs: {
        ...actual.preview.proposal.geography_plan.validated_legs,
        features: actual.preview.proposal.geography_plan.validated_legs.features.map(
          (feature, index) => index === dayFourLegIndex ? substitutedDayFourLeg : feature,
        ),
      },
    },
  })],
  ["compiled_target", () => withProposal({
    compiled_request: {
      ...actual.preview.proposal.compiled_request,
      operations: [{
        ...actual.preview.proposal.compiled_request.operations[0],
        target: "forged_stop",
      }],
    },
  })],
  ["repair_certificate", () => withProposal({
    repair: {
      ...actual.preview.proposal.repair,
      certificate: {
        ...actual.preview.proposal.repair.certificate,
        id: `cert_${"8".repeat(16)}`,
      },
    },
  })],
  ["eligible_state", () => withProposal({ state: "ineligible", eligibility: "ineligible" })],
];
for (const [name, build] of forgeries) {
  const forged = build();
  assert.equal(
    await normalizers.normalizeEvaluatedPreviewResponse(forged, actual.expected),
    null,
    name,
  );
}

console.log("evaluated reorder frontend contract and forgeries passed");
