import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "..");
const appPath = path.join(root, "src", "itinerary_system", "product_app", "static", "js", "app.js");
const source = fs.readFileSync(appPath, "utf8");
const contractCode = source.slice(
  source.indexOf("const TYPED_EDIT_OPERATIONS"),
  source.indexOf("function safeExternalLink"),
);
const parameterCode = source.slice(
  source.indexOf("function durationPayload"),
  source.indexOf("async function submitTypedOperation"),
);
const routeMetricCode = source.slice(
  source.indexOf("function routeDurationLabel"),
  source.indexOf("function draftTargetLabel"),
);
const state = {
  workspace: null,
  session: {
    session_id: `session_${"b".repeat(32)}`,
    revision: 7,
    accepted_plan_id: "plan_parent",
    draft: [],
  },
  mapInteraction: { selectedRouteLeg: { route_leg_id: "leg_parent_child" } },
};
const buildForState = (runtimeState) => new Function(
  "state",
  `${contractCode}\n${routeMetricCode}\n${parameterCode}\nreturn {normalizeTypedEditCapabilities, normalizeDraftImpactPreview, typedOperationParameters, routeDurationLabel, routeDistanceLabel};`,
)(runtimeState);
const { normalizeTypedEditCapabilities, normalizeDraftImpactPreview, typedOperationParameters } = buildForState(state);
const { routeDurationLabel, routeDistanceLabel } = buildForState(state);

const common = (feedbackTier, executable, evaluated, blockingCode) => ({
  enabled: true,
  feedback_tier: feedbackTier,
  preview_executable: executable,
  evaluated_repair: evaluated,
  blocking_code: blockingCode,
});
const protectedCapability = () => ({
  ...common("draft_only", false, false, "full_evaluation_not_supported"),
  protected_strengths: ["must_keep", "booked"],
  protected_strengths_blocking_code: "commitment_permission_required",
});
const capabilities = {
  schema_version: "product-typed-edit-capabilities-v1",
  vocabularies: {
    stop_roles: ["attraction", "activity", "meal", "lodging", "transport_hub", "rest_stop", "scenic_stop", "route_waypoint", "origin", "destination"],
    duration_modes: ["exact", "preferred", "minimum", "maximum", "range"],
    commitment_strengths: ["optional", "prefer_keep", "strong_preference", "must_keep", "booked", "excluded"],
    scope_lifetimes: ["current_draft_only", "current_repair_session", "remainder_of_trip"],
    attributes: ["existence", "day", "duration", "sequence_order", "role"],
    route_issue_types: ["suspected_closure", "access_problem", "route_quality"],
  },
  operations: {
    set_stop_role: {
      ...common("conditional", true, true, null),
      supported_evaluated_roles: ["attraction", "activity", "meal", "rest_stop", "scenic_stop"],
      draft_only_roles: ["lodging", "transport_hub", "route_waypoint", "origin", "destination"],
      unsupported_role_blocking_code: "stop_role_evaluation_not_supported",
      typed_plan_field: "itinerary_role",
      typed_source_field: "itinerary_role_source",
      typed_source_value: "user_declared_itinerary_role",
      combinable_operation_types: ["set_stop_role"],
    },
    set_stop_day: common("evaluated", true, true, null),
    set_stop_order: {
      ...common("evaluated", true, true, null),
      supported_scope: "same_day",
      sequence_index_base: 0,
    },
    set_stop_duration: {
      ...common("conditional", true, true, null),
      supported_evaluated_modes: ["exact"],
      draft_only_modes: ["preferred", "minimum", "maximum", "range"],
      unsupported_mode_blocking_code: "duration_mode_evaluation_not_supported",
      scalar_plan_field: "visit_duration_minutes",
      typed_plan_field: "duration_constraint",
    },
    set_stop_time_window: {
      ...common("evaluated", true, true, null),
      typed_plan_field: "time_window_constraint",
      constraint_schema_version: "stop-time-window-constraint-v1",
      early_arrival_policy: "wait_until_earliest_arrival",
      latest_departure_semantics: "departure_after_visit",
      combinable_operation_types: ["set_stop_time_window"],
    },
    set_stop_commitment: protectedCapability(),
    set_attribute_constraint: protectedCapability(),
    change_route_preference: { ...common("draft_only", false, false, "route_preference_not_supported"), enabled: false },
    report_route_issue: common("draft_only", false, false, "full_evaluation_not_supported"),
  },
};
state.workspace = {
  typed_edit_capabilities: capabilities,
  draft_capabilities: { operations: {} },
  map_edit_capabilities: { operations: {} },
};
assert.equal(normalizeTypedEditCapabilities(structuredClone(capabilities)).schema_version, capabilities.schema_version);
for (const mutate of [
  (value) => { value.extra = true; },
  (value) => { value.vocabularies.stop_roles.reverse(); },
  (value) => { value.operations.set_stop_day.feedback_tier = "likely"; },
  (value) => { value.operations.set_stop_order.supported_scope = "cross_day"; },
  (value) => { value.operations.set_stop_order.sequence_index_base = 1; },
  (value) => { value.operations.set_stop_role.supported_evaluated_roles = ["lodging"]; },
  (value) => { value.operations.set_stop_duration.supported_evaluated_modes = ["preferred"]; },
  (value) => { value.operations.set_stop_time_window.latest_departure_semantics = "latest_start"; },
  (value) => { value.operations.set_stop_commitment.protected_strengths = ["booked"]; },
  (value) => { value.operations.report_route_issue.unknown = true; },
]) {
  const forged = structuredClone(capabilities);
  mutate(forged);
  assert.equal(normalizeTypedEditCapabilities(forged), null);
}

const impact = {
  schema_version: "product-draft-impact-preview-v1",
  session_id: state.session.session_id,
  session_revision: 7,
  parent_plan_id: "plan_parent",
  parent_plan_content_hash: "a".repeat(16),
  certification_status: "not_certified",
  is_certified: false,
  operations: [{
    operation_id: "operation_one",
    type: "set_stop_day",
    target: "stop_one",
    feedback_tier: "evaluated",
    preview_executable: true,
    evaluated_repair: true,
    changed_attributes: ["day"],
    blocking_codes: [],
  }],
  summary: {
    operation_count: 1,
    evaluated_executable_count: 1,
    draft_only_count: 0,
    can_run_evaluated_preview: true,
    blocking_codes: [],
  },
};
state.session.draft = [{
  operation_id: "operation_one",
  type: "set_stop_day",
  target: "stop_one",
  parameters: { day: 4 },
}];
assert.equal(normalizeDraftImpactPreview(structuredClone(impact)).summary.can_run_evaluated_preview, true);
for (const mutate of [
  (value) => { value.eligibility = "eligible"; },
  (value) => { value.is_certified = true; },
  (value) => { value.parent_plan_content_hash = "short"; },
  (value) => { value.session_revision = 8; },
  (value) => { value.operations[0].unknown = true; },
  (value) => { value.operations[0].operation_id = "operation_forged"; },
  (value) => { value.operations[0].type = "unknown_operation"; },
  (value) => { value.operations[0].feedback_tier = "draft_only"; },
  (value) => { value.operations[0].changed_attributes = ["duration"]; },
  (value) => { value.summary.evaluated_executable_count = 0; },
  (value) => { value.summary.can_run_evaluated_preview = false; },
  (value) => { value.summary.operation_count = 2; },
]) {
  const forged = structuredClone(impact);
  mutate(forged);
  assert.equal(normalizeDraftImpactPreview(forged), null);
}

const form = (elements) => ({ elements });
assert.deepEqual(typedOperationParameters(form({ role: { value: "meal" } }), "set_stop_role"), { role: "meal" });
assert.deepEqual(typedOperationParameters(form({ day: { value: "4" } }), "set_stop_day"), { day: 4 });
assert.deepEqual(typedOperationParameters(form({ day: { value: "4" }, sequence_position: { value: "3" } }), "set_stop_order"), { day: 4, sequence_index: 2 });
assert.deepEqual(typedOperationParameters(form({
  duration_mode: { value: "exact" },
  "duration-preferred": { disabled: false, value: "45" },
  "duration-minimum": { disabled: true, value: "" },
  "duration-maximum": { disabled: true, value: "" },
}), "set_stop_duration"), { duration: { mode: "exact", preferred_minutes: 45, minimum_minutes: 45, maximum_minutes: 45 } });
assert.deepEqual(typedOperationParameters(form({
  duration_mode: { value: "range" },
  "duration-preferred": { disabled: true, value: "" },
  "duration-minimum": { disabled: false, value: "30" },
  "duration-maximum": { disabled: false, value: "90" },
}), "set_stop_duration"), { duration: { mode: "range", preferred_minutes: null, minimum_minutes: 30, maximum_minutes: 90 } });
assert.equal(typedOperationParameters(form({
  earliest_arrival: { value: "15:00" }, latest_departure: { value: "14:00" },
}), "set_stop_time_window"), null);
assert.deepEqual(typedOperationParameters(form({
  earliest_arrival: { value: "10:00" }, latest_departure: { value: "" },
}), "set_stop_time_window"), { earliest_arrival: "10:00", latest_departure: null });
assert.deepEqual(typedOperationParameters(form({
  earliest_arrival: { value: "" }, latest_departure: { value: "17:30" },
}), "set_stop_time_window"), { earliest_arrival: null, latest_departure: "17:30" });
assert.equal(typedOperationParameters(form({
  earliest_arrival: { value: "9:00" }, latest_departure: { value: "" },
}), "set_stop_time_window"), null);
assert.deepEqual(typedOperationParameters(form({
  issue_type: { value: "route_quality" }, note: { value: "  rough access  " },
}), "report_route_issue"), {
  route_leg_id: "leg_parent_child", issue_type: "route_quality", note: "rough access",
});
for (const invalid of [null, undefined, "3600", 0, -1, Number.NaN, Number.POSITIVE_INFINITY]) {
  assert.equal(routeDurationLabel(invalid), "Unavailable");
  assert.equal(routeDistanceLabel(invalid), "Unavailable");
}
assert.equal(routeDurationLabel(3600), "60 min");
assert.equal(routeDistanceLabel(1609.344), "1.0 mi");

if (process.env.ACTUAL_TYPED_CONTRACT_PATH) {
  const actual = JSON.parse(fs.readFileSync(process.env.ACTUAL_TYPED_CONTRACT_PATH, "utf8"));
  const actualState = {
    workspace: {
      typed_edit_capabilities: actual.capabilities,
      draft_capabilities: actual.draft_capabilities,
      map_edit_capabilities: actual.map_edit_capabilities,
    },
    session: {
      session_id: actual.impact.session_id,
      revision: actual.impact.session_revision,
      accepted_plan_id: actual.impact.parent_plan_id,
      draft: actual.draft,
    },
    mapInteraction: { selectedRouteLeg: null },
  };
  const actualNormalizers = buildForState(actualState);
  assert.ok(actualNormalizers.normalizeTypedEditCapabilities(structuredClone(actual.capabilities)));
  assert.ok(actualNormalizers.normalizeDraftImpactPreview(structuredClone(actual.impact)));
}

console.log("typed edit frontend adversarial cases passed");
