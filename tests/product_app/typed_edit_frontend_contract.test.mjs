import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import test from "node:test";

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
  `${contractCode}\n${routeMetricCode}\n${parameterCode}\nreturn {normalizeTypedEditCapabilities, normalizeDraftImpactPreview, typedOperationParameters, routeDurationLabel, routeDistanceLabel, draftCombinationBlocker};`,
)(runtimeState);
const { normalizeTypedEditCapabilities, normalizeDraftImpactPreview, typedOperationParameters } = buildForState(state);
const { routeDurationLabel, routeDistanceLabel } = buildForState(state);
const { draftCombinationBlocker } = buildForState(state);
const op = (type, mode) => ({ type, parameters: { duration: { mode } } });
for (const draft of [
  [op("replace_nearby"), op("set_stop_duration", "exact")],
  [op("set_stop_order"), op("set_stop_time_window")],
  [op("set_stop_duration", "exact"), op("set_stop_duration", "preferred")],
  [op("set_stop_role"), op("set_stop_day")],
]) assert.equal(draftCombinationBlocker(draft), "draft_evaluated_operation_combination_unsupported");
for (const draft of [[], [op("replace_nearby"), op("keep_stop")],
  [op("set_stop_duration", "exact"), op("set_stop_duration", "exact")],
  [op("set_stop_order"), op("set_stop_order")]]) assert.equal(draftCombinationBlocker(draft), null);

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

// Component regression for the real dg1/dg2 failure: the mutation was committed
// and installed, but no renderer was called until the impact request completed.
// Only HTTP and DOM adapters are controlled here; browser evidence remains separate.
function productionSection(start, end) {
  const first = source.indexOf(start), last = source.indexOf(end, first + start.length);
  assert.ok(first >= 0 && last > first, `Missing production section ${start}`);
  return source.slice(first, last);
}

function stagedImpactHarness(action, { staleMutation = false } = {}) {
  const operation = { operation_id: "operation_order", type: "set_stop_order", target: "stop_one",
    parameters: { day: 4, sequence_index: 1 }, source: "map", evidence_refs: [] };
  const runtime = {
    undoPending: false,
    session: { session_id: "session_staged", trip_id: "trip", run_id: "run", revision: 6,
      accepted_plan_id: "plan_parent", selected_stop_id: "stop_one", draft: action === "undo" ? [operation] : [],
      proposal: null, permission_decisions: [], conversation_id: "conversation_current" },
    workspace: { typed_edit_capabilities: structuredClone(capabilities), geography: { plan_id: "plan_parent" } },
    mapInteraction: { selectedWaypointId: null },
    previewRuntime: { epoch: 1, status: "PARENT_RENDERED", requestId: null },
    typedImpact: { status: "ready", error: null,
      response: { session_revision: 6, parent_plan_content_hash: "a".repeat(16), summary: { can_run_evaluated_preview: true } } },
  };
  const before = structuredClone(runtime), committed = structuredClone(runtime.session);
  committed.revision = 7;
  committed.draft = action === "undo" ? [] : [operation];
  let releaseImpact, rejectImpact, functions, button, form;
  const heldImpact = new Promise((resolve, reject) => { releaseImpact = resolve; rejectImpact = reject; });
  const impactRequests = [];
  let mutations = 0;
  const calls = [], presentations = [], dialogs = [], messages = [];
  const surface = { querySelectorAll: () => [form], querySelector: () => null };
  const snapshot = () => ({ revision: runtime.session.revision, draft: structuredClone(runtime.session.draft),
    impactStatus: runtime.typedImpact.status, impactHtml: functions.typedImpactSummaryHtml(),
    previewEnabled: functions.canRunEvaluatedPreview(), parent: structuredClone(runtime.workspace.geography) });
  const dependencies = { state: runtime, pendingSelection: null,
    copilot: { update() {}, previewStateChanged() {} },
    api: async (url, options) => {
      calls.push({ url, ...structuredClone(options) });
      if (url.endsWith("/impact-preview")) {
        if (!impactRequests.length) {
          impactRequests.push({ resolve: releaseImpact, reject: rejectImpact });
          return heldImpact;
        }
        return new Promise((resolve, reject) => impactRequests.push({ resolve, reject }));
      }
      if (staleMutation) runtime.session.revision += 1;
      if (mutations++ > 0) return { session: { ...structuredClone(runtime.session),
        revision: runtime.session.revision + 1, draft: runtime.session.draft.slice(0, -1), proposal: null } };
      return { session: committed };
    },
    apiErrorMessage: value => value, resetPoiDiscovery() {},
    renderAll: () => presentations.push(snapshot()),
    refreshMapEditDialog: () => {
      button = { disabled: false, title: "" };
      form = { dataset: { typedOperation: "set_stop_duration" },
        elements: { duration_mode: { value: "exact" } }, addEventListener() {}, querySelector: () => button };
      functions.bindTypedEditSurface();
      dialogs.push({ ...snapshot(), disabled: button.disabled, blocker: form.dataset.combinationBlockingCode });
    },
    $: selector => selector === "#map-edit-dialog" ? { open: true } : surface,
    toast: (message, error = false) => messages.push({ message, error }),
    submitTypedOperation() {}, bindDurationModeSelects() {}, bindRoleSelects() {},
    humanLabel: value => value,
  };
  const code = "const mutationRecovery = {submit: (action, body) => api(`/api/sessions/${state.session.session_id}/${action === \"selection\" ? \"selection\" : `draft/${action}`}`, {method:\"POST\", body})};\n" + contractCode
    + productionSection("function escapeHtml(", "function unavailable(")
    + productionSection("function operationDisplayName(", "function draftStateHtml(")
    + productionSection("function typedImpactSummaryHtml(", "function stopTypedEditorHtml(")
    + productionSection("function bindTypedEditSurface(", "async function updateSelection(")
    + productionSection("function draftImpactRequestIsCurrent(", "function selectContext(")
    + productionSection("function previewExpectedContext(", "async function addCopilotDraft(")
    + productionSection("async function undoDraft()", "async function previewDraft(");
  functions = new Function(...Object.keys(dependencies), `${code}\nreturn {addDraft,undoDraft,loadDraftImpactPreview,canRunEvaluatedPreview,typedImpactSummaryHtml,bindTypedEditSurface,draftImpactRequestIsCurrent};`)(...Object.values(dependencies));
  const impactResponse = { schema_version: "product-draft-impact-preview-v1", session_id: committed.session_id,
    session_revision: 7, parent_plan_id: "plan_parent", parent_plan_content_hash: "a".repeat(16),
    certification_status: "not_certified", is_certified: false,
    operations: action === "undo" ? [] : [{ operation_id: operation.operation_id, type: operation.type,
      target: operation.target, feedback_tier: "evaluated", preview_executable: true, evaluated_repair: true,
      changed_attributes: ["day", "sequence_order"], blocking_codes: [] }],
    summary: { operation_count: committed.draft.length, evaluated_executable_count: committed.draft.length,
      draft_only_count: 0, can_run_evaluated_preview: action !== "undo", blocking_codes: [] } };
  return { runtime, before, committed, calls, presentations, dialogs, messages, releaseImpact, rejectImpact, impactResponse,
    functions, impactRequests,
    run: () => action === "undo" ? functions.undoDraft()
      : functions.addDraft(operation.type, operation.parameters, operation.target, action === "copilot" ? "copilot" : "map") };
}

for (const newer of ["undo", "same_revision_refresh"]) {
  for (const lateOutcome of ["resolve", "reject"]) {
    test(`${newer}: a late impact ${lateOutcome} cannot erase newer impact or refresh the form`, async () => {
      const h = stagedImpactHarness("map"), first = h.run();
      await new Promise(setImmediate);
      const second = newer === "undo" ? h.functions.undoDraft() : h.functions.loadDraftImpactPreview();
      await new Promise(setImmediate);
      const latest = newer === "undo" ? { ...structuredClone(h.impactResponse), session_revision: 8, operations: [],
        summary: { operation_count: 0, evaluated_executable_count: 0, draft_only_count: 0,
          can_run_evaluated_preview: false, blocking_codes: [] } } : structuredClone(h.impactResponse);
      h.impactRequests[1].resolve(latest);
      await second;
      assert.equal(h.runtime.typedImpact.status, "ready");
      const currentImpact = h.runtime.typedImpact.response;
      const rendered = h.presentations.length, refreshed = h.dialogs.length;
      if (lateOutcome === "reject") h.rejectImpact(new Error("older impact failed"));
      else h.releaseImpact(h.impactResponse);
      await first;
      assert.equal(h.runtime.typedImpact.status, "ready");
      assert.equal(h.runtime.typedImpact.response, currentImpact, "Keep the newer canonical response object");
      assert.equal(h.presentations.length, rendered, "A late outer addDraft continuation must not re-render");
      assert.equal(h.dialogs.length, refreshed, "Do not erase newer form input by refreshing it");
      assert.equal(h.runtime.session.revision, newer === "undo" ? 8 : 7);
    });
  }
}

test("same-revision recheck disables Preview until the current impact finishes", async () => {
  const h = stagedImpactHarness("map"), pending = h.run();
  await new Promise(setImmediate);
  h.releaseImpact(h.impactResponse);
  await pending;
  assert.equal(h.functions.canRunEvaluatedPreview(), true);
  const recheck = h.functions.loadDraftImpactPreview();
  assert.equal(h.runtime.typedImpact.status, "loading");
  const enabledWhileLoading = h.functions.canRunEvaluatedPreview();
  h.impactRequests[1].resolve(h.impactResponse);
  await recheck;
  assert.equal(enabledWhileLoading, false);
  assert.equal(h.functions.canRunEvaluatedPreview(), true);
});

for (const interruption of ["none", "undo_then_resolve", "undo_then_reject"]) {
  test(`preview post-impact completion respects the installed child ownership: ${interruption}`, async () => {
    const h = stagedImpactHarness("map");
    h.runtime.session.draft = structuredClone(h.committed.draft);
    const proposal = { schema_version: "draft-preview-v1", state: "eligible", child_plan_id: "child",
      compiled_request: { source_request_id: "preview_request" } };
    const normalized = { session: { ...structuredClone(h.runtime.session), revision: 7, proposal }, proposal };
    const guards = new Function("state", `${contractCode}\nreturn {previewRequestIsCurrent,previewStateForSession};`)(h.runtime);
    const expectedContext = new Function("state", `${productionSection("function previewExpectedContext(", "async function submitDraftMutation(")}\nreturn previewExpectedContext;`)(h.runtime);
    let closes = 0, renders = 0;
    const notices = [];
    // Supply the already-normalized child at this unit boundary. Independent
    // evaluator/geometry correctness is covered by the real browser gates.
    const dependencies = { pendingSelection: null, state: h.runtime, toast: message => notices.push(message),
      hasRouteCheckedOnlyDraft: () => false, canRunEvaluatedPreview: h.functions.canRunEvaluatedPreview,
      renderAll: () => { renders++; }, copilot: { previewStateChanged() {} },
      previewExpectedContext: expectedContext, api: async () => normalized,
      normalizeEvaluatedPreviewResponse: async () => normalized, apiErrorMessage: value => value,
      ...guards, loadDraftImpactPreview: h.functions.loadDraftImpactPreview,
      draftImpactRequestIsCurrent: h.functions.draftImpactRequestIsCurrent,
      $: () => ({ close() { closes++; } }) };
    const preview = new Function(...Object.keys(dependencies),
      `${productionSection("async function previewDraft(", "function setupDockResize(")}\nreturn previewDraft;`)(...Object.values(dependencies));
    const pending = preview();
    await new Promise(setImmediate);
    assert.equal(h.runtime.session, normalized.session);
    assert.equal(h.runtime.previewRuntime.status, "PREVIEW_DATA_READY");
    if (interruption !== "none") {
      h.committed.revision = 8;
      h.committed.draft = [];
      const undo = h.functions.undoDraft();
      await new Promise(setImmediate);
      h.impactRequests[1].resolve({ ...h.impactResponse, session_revision: 8, operations: [],
        summary: { operation_count: 0, evaluated_executable_count: 0, draft_only_count: 0,
          can_run_evaluated_preview: false, blocking_codes: [] } });
      await undo;
    }
    const before = { closes, renders, notices: [...notices], session: h.runtime.session };
    if (interruption === "undo_then_reject") h.rejectImpact(new Error("old preview impact failed"));
    else h.releaseImpact(h.impactResponse);
    const completed = await pending;
    if (interruption === "none") {
      assert.equal(completed, true);
      assert.equal(closes, 1);
      assert.equal(renders, 2);
      assert.ok(notices.some(message => message.startsWith("Preview evaluated.")));
    } else {
      assert.equal(completed, false);
      assert.deepEqual({ closes, renders, notices, session: h.runtime.session }, before);
      assert.equal(h.runtime.session.revision, 8);
      assert.equal(h.runtime.session.proposal, null);
    }
  });
}

for (const action of ["map", "copilot", "undo"]) {
  for (const outcome of ["ready", "failed", "wrong_revision"]) {
    test(`${action}: committed draft is visible before held impact becomes ${outcome}`, async () => {
      const h = stagedImpactHarness(action), pending = h.run();
      await new Promise(setImmediate);
      try {
        assert.equal(h.calls.length, 2);
        assert.equal(h.calls[1].body.expected_revision, 7);
        assert.equal(h.runtime.session, h.committed);
        assert.equal(h.runtime.typedImpact.status, "loading");
        assert.equal(h.presentations.length, 1, "Render the committed state before awaiting impact");
        assert.deepEqual(h.presentations[0].draft, h.committed.draft);
        assert.equal(h.presentations[0].previewEnabled, false);
        assert.match(h.presentations[0].impactHtml, /Checking draft impact/);
        assert.deepEqual(h.presentations[0].parent, h.before.workspace.geography);
        assert.equal(h.dialogs.length, 1, "Refresh the actual combination-control binding immediately");
        assert.equal(h.dialogs[0].disabled, action !== "undo");
        assert.equal(h.dialogs[0].blocker, action === "undo" ? undefined : "draft_evaluated_operation_combination_unsupported");
        assert.equal(h.messages.some(row => row.error), false);
        assert.ok(h.messages.some(row => /staged|added|removed/i.test(row.message)));
      } finally {
        if (outcome === "failed") h.rejectImpact(new Error("impact unavailable"));
        else h.releaseImpact(outcome === "wrong_revision" ? { ...h.impactResponse, session_revision: 6 } : h.impactResponse);
        await pending;
      }
      assert.equal(h.presentations.length, 2);
      assert.equal(h.dialogs.length, 2);
      const final = h.presentations.at(-1);
      assert.equal(final.previewEnabled, outcome === "ready" && action !== "undo");
      assert.equal(final.impactStatus, outcome === "ready" ? "ready" : "failed");
      assert.match(final.impactHtml, outcome === "ready" ? /Not certified/ : /Draft impact unavailable/);
      assert.deepEqual(final.draft, h.committed.draft, "Impact failure cannot erase a committed draft");
      assert.deepEqual(final.parent, h.before.workspace.geography);
      assert.equal(h.runtime.session.proposal, null);
      assert.equal(h.calls.length, 2, "No provider, preview, acceptance, or automatic retry");
    });
  }
  test(`${action}: stale mutation is still rejected before staged rendering`, async () => {
    const h = stagedImpactHarness(action, { staleMutation: true });
    await h.run();
    assert.equal(h.calls.length, 1);
    assert.equal(h.presentations.length, 0);
    assert.equal(h.dialogs.length, 0);
    assert.deepEqual(h.runtime.session.draft, h.before.session.draft);
    assert.deepEqual(h.messages, [{ message: "stale_session_revision", error: true }]);
  });
}
