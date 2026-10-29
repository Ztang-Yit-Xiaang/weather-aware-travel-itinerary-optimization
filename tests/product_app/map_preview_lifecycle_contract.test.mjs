import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { recordedJson } from "./recorded-runtime-fixture.mjs";
import { invalidateMapReadiness } from "../../src/itinerary_system/product_app/static/js/map-controller-v2.js";

const appSource = readFileSync(new URL("../../src/itinerary_system/product_app/static/js/app.js", import.meta.url), "utf8");
const mapSource = readFileSync(new URL("../../src/itinerary_system/product_app/static/js/map-controller-v2.js", import.meta.url), "utf8");

function sourceSection(source, startMarker, endMarker) {
  const start = source.indexOf(startMarker);
  const end = source.indexOf(endMarker, start + startMarker.length);
  assert.ok(start >= 0 && end > start, `${startMarker} must be extractable`);
  return source.slice(start, end);
}

function loadFunction(source, name, dependencies) {
  return new Function(...Object.keys(dependencies), `${source}\nreturn ${name};`)(...Object.values(dependencies));
}

test("historical edit-disabled accepted-plan capabilities cannot enable a mutation", () => {
  const accepted = recordedJson("results/e2e/runtime_optimizer_repair_deterministic_20260905_v2/accepted_reload/accept_response.json");
  const capabilities = accepted.workspace.typed_edit_capabilities;
  const contractSource = sourceSection(appSource, "const TYPED_EDIT_OPERATIONS", "function safeExternalLink");
  const normalize = loadFunction(contractSource, "normalizeTypedEditCapabilities", { state: {} });
  assert.equal(normalize(capabilities), capabilities);
  for (const [name, capability] of Object.entries(capabilities.operations)) {
    assert.equal(capability.blocking_code, "accepted_plan_requires_new_repair_baseline");
    for (const flag of ["enabled", "preview_executable", "evaluated_repair"]) {
      assert.equal(capability[flag], false, `${name}.${flag} remains disabled`);
      const forged = structuredClone(capabilities);
      forged.operations[name][flag] = true;
      assert.equal(normalize(forged), null, `reject contradictory ${name}.${flag}`);
    }
  }
});

test("an accepted real child remains displayable without permitting unknown plan roles", {
  skip: !process.env.RUNTIME_PREVIEW_EVIDENCE_DIR,
}, () => {
  const directory = process.env.RUNTIME_PREVIEW_EVIDENCE_DIR;
  const capturedPayload = (name) => JSON.parse(JSON.parse(readFileSync(`${directory}/${name}`, "utf8")).response_body_utf8);
  const preview = capturedPayload("product_preview_http_io.json").proposal.geography_plan;
  const geography = capturedPayload("product_reload_http_io.json").workspace.geography;
  const accepted = structuredClone(preview);
  accepted.role = "accepted";
  accepted.label = "Accepted plan";
  for (const name of ["stops", "route_path", "validated_legs", "gaps"]) {
    for (const feature of accepted[name].features) feature.properties.role = "accepted";
  }
  const source = mapSource.slice(0, mapSource.indexOf("function localAssetUrl"));
  const { validatePlan, selectVisiblePlans } = loadFunction(
    `${source}\nconst validators = { validatePlan, selectVisiblePlans };`, "validators", {},
  );
  assert.equal(validatePlan(accepted), accepted);
  const selected = selectVisiblePlans({
    ...geography,
    plans: [...geography.plans.filter((plan) => plan.plan_id !== accepted.plan_id), accepted],
  }, accepted.plan_id);
  assert.equal(selected.active, accepted);
  assert.equal(selected.original.role, "original");
  assert.notEqual(selected.original.plan_id, accepted.plan_id);
  assert.throws(() => validatePlan({ ...accepted, role: "unknown_accepted_role" }), /geography_plan_invalid/);
});

test("a stale geometry digest cannot publish over a newer map render", async () => {
  const pending = new Map();
  const fallbacks = [];
  const controllerSource = sourceSection(mapSource, "export function createGeographicMapController(", "/** Link two pane-owned controllers").replace("export function", "function");
  const empty = () => ({ type: "FeatureCollection", features: [] });
  const container = { dataset: {}, isConnected: true };
  const create = loadFunction(controllerSource, "createGeographicMapController", {
    invalidateMapReadiness,
    selectVisiblePlans: (_geography, planId) => ({ original: { plan_id: "parent" }, plans: [{ plan_id: planId }] }),
    collectFeatureCollection: empty,
    deriveRouteAnchors: empty,
    resolveGapMarkers: () => ({ descriptions: [] }),
    exploratoryPinFeature: () => null,
    draftRouteFeatureCollection: empty,
    poiCandidateFeatureCollections: empty,
    validateDraftWaypoint: () => {},
    textFallbackFor: (plans) => ({ planId: plans[0].plan_id }),
    diagnosticsFor: (_plans, _gaps, planId) => new Promise((resolve) => pending.set(planId, resolve)),
    loadMapRuntime: async () => { container.isConnected = false; return {}; },
  });
  // Unmount after digest publication to avoid a visual runtime; the real
  // generation guard and asynchronous publication order are exercised.
  const controller = create({ container, mapConfig: {}, onTextFallback: (value) => fallbacks.push(value.planId) });
  const older = controller.render({ geography: {}, activePlanId: "older" });
  const newer = controller.render({ geography: {}, activePlanId: "newer" });
  assert.deepEqual([...pending.keys()], ["older", "newer"]);
  pending.get("newer")({ activePlanId: "newer", activeGeometryHash: "newer-hash" });
  await newer;
  // Keep connectivity from masking the generation guard being tested.
  container.isConnected = true;
  pending.get("older")({ activePlanId: "older", activeGeometryHash: "older-hash" });
  await older;
  assert.equal(controller.diagnostics().activePlanId, "newer");
  assert.deepEqual(fallbacks, ["newer"], "old render must not publish textual state either");
});

function lifecycleHarness() {
  const plan = { plan_id: "child", content_hash: "child-hash", stops: { features: [{ properties: { stop_id: "pfeiffer" } }] } };
  const state = {
    session: { proposal: { state: "eligible", acceptance_eligible: true, child_plan_id: "child" } },
    previewRuntime: { status: "PREVIEW_DATA_READY", epoch: 1, requestId: "request" },
    acknowledgedPreviewPlanId: null,
  };
  const stage = { dataset: {} };
  const element = { dataset: {}, classList: { add() {}, remove() {} }, closest: () => stage };
  const source = sourceSection(appSource, "function updateGeographicMapStatus(", "\nfunction timeWindowLabel(")
    + sourceSection(appSource, "function previewCanBeAccepted()", "\nasync function submitAcceptance(")
    + sourceSection(appSource, "function copilotPreviewState()", "\nfunction activeGeography()");
  const functions = loadFunction(source + "\nconst exports = { updateGeographicMapStatus, previewCanBeAccepted, copilotPreviewState };", "exports", {
    state,
    $: () => element,
    activeRenderPlan: () => plan,
    activeProposalPlan: () => state.session.proposal?.state === "eligible" ? plan : null,
    copilot: { previewStateChanged() {} },
    toast() {},
    renderContextPanel() {},
  });
  const ready = () => functions.updateGeographicMapStatus({ state: "ready", activePlanId: "child", activePlanContentHash: "child-hash", activeGeometryHash: "a".repeat(64), activeMarkerIds: ["pfeiffer"], visiblePlanCount: 2, gapCount: 0 });
  return { ...functions, state, stage, ready };
}

test("map loading and failure revoke ready state and acceptance until a new acknowledgement", () => {
  const h = lifecycleHarness();
  const evaluationMessage = "The typed draft produced an immutable in-memory child that passed independent evaluation.";
  h.state.session.proposal.message = evaluationMessage;
  assert.equal(h.previewCanBeAccepted(), false);
  h.ready();
  assert.equal(h.previewCanBeAccepted(), true);
  h.updateGeographicMapStatus({ state: "loading", code: "map_resize_revalidation_pending" });
  assert.equal(h.state.previewRuntime.status, "PREVIEW_DATA_READY");
  assert.equal(h.state.acknowledgedPreviewPlanId, null);
  assert.equal(h.stage.dataset.renderedMapPlanId, undefined);
  assert.equal(h.previewCanBeAccepted(), false);
  h.ready();
  assert.equal(h.previewCanBeAccepted(), true);
  h.updateGeographicMapStatus({ state: "failed", code: "map_tile_coverage_incomplete" });
  assert.equal(h.state.previewRuntime.status, "MAP_RENDER_FAILED");
  assert.equal(h.state.acknowledgedPreviewPlanId, null);
  assert.equal(h.previewCanBeAccepted(), false);
  assert.match(h.copilotPreviewState().message, /map.*(?:rendered|verified)/i);
  assert.match(h.copilotPreviewState().message, /disabled.*retry/i);
  assert.notEqual(h.copilotPreviewState().message, evaluationMessage,
    "A map failure must not reuse the evaluator's success message as its failure reason");
  h.ready();
  assert.equal(h.previewCanBeAccepted(), true);
  assert.equal(h.copilotPreviewState().message, evaluationMessage,
    "A fresh exact acknowledgement clears the map failure explanation");
  h.updateGeographicMapStatus({ state: "ready", activePlanId: "older", activePlanContentHash: "older-hash", activeGeometryHash: "b".repeat(64), activeMarkerIds: ["bixby"] });
  assert.equal(h.state.previewRuntime.status, "MAP_IDENTITY_FAILED");
  assert.equal(h.state.acknowledgedPreviewPlanId, null);
  assert.equal(h.previewCanBeAccepted(), false);
  assert.match(h.copilotPreviewState().message, /map.*identity/i);
  assert.match(h.copilotPreviewState().message, /disabled.*retry/i);
});

test("acceptance requires a rendered matching eligible child, including handler entry", async () => {
  const h = lifecycleHarness();
  for (const status of ["PREVIEW_DATA_READY", "PREVIEW_RUNNING", "PREVIEW_INELIGIBLE", "MAP_RENDER_FAILED", "MAP_IDENTITY_FAILED"]) {
    h.state.previewRuntime.status = status;
    h.state.acknowledgedPreviewPlanId = "child";
    assert.equal(h.previewCanBeAccepted(), false, status);
  }
  h.state.previewRuntime.status = "PREVIEW_RENDERED";
  h.state.acknowledgedPreviewPlanId = "older";
  assert.equal(h.previewCanBeAccepted(), false);
  let requests = 0;
  let confirmations = 0;
  const submit = loadFunction(sourceSection(appSource, "async function submitAcceptance(", "\nfunction renderRepair("), "submitAcceptance", {
    state: h.state,
    pendingSelection: null,
    previewCanBeAccepted: h.previewCanBeAccepted,
    toast() {},
    window: { confirm: () => { confirmations += 1; return true; } },
    api: async () => { requests += 1; throw new Error("unexpected acceptance request"); },
  });
  await submit("accept_repair");
  assert.equal(requests, 0);
  assert.equal(confirmations, 0);
  h.state.acknowledgedPreviewPlanId = "child";
  assert.equal(h.previewCanBeAccepted(), true);
  h.state.session.proposal.acceptance_eligible = false;
  assert.equal(h.previewCanBeAccepted(), false);
});

function acceptanceHarness() {
  const state = {
    session: { session_id: "session", revision: 4, accepted_plan_id: "parent", proposal: {
      state: "eligible", acceptance_eligible: true, child_plan_id: "child", child_plan_content_hash: "child-hash",
    } },
    previewRuntime: { status: "PREVIEW_RENDERED" }, acknowledgedPreviewPlanId: "child",
  };
  const dialog = new EventTarget();
  Object.assign(dialog, { open: false, dataset: {}, returnValue: "", shows: 0 });
  dialog.showModal = () => { dialog.open = true; dialog.shows += 1; };
  dialog.close = (value) => {
    if (value !== undefined) dialog.returnValue = value;
    dialog.open = false;
    dialog.dispatchEvent(new Event("close"));
  };
  const message = { textContent: "" };
  const confirm = { textContent: "" };
  const requests = [];
  const toasts = [];
  let deliver;
  const submit = loadFunction(sourceSection(appSource, "async function submitAcceptance(", "\nfunction renderRepair("), "submitAcceptance", {
    state, pendingSelection: null,
    $: selector => ({ "#acceptance-dialog": dialog, "#acceptance-message": message, "#acceptance-confirm": confirm })[selector],
    previewCanBeAccepted: () => state.previewRuntime.status === "PREVIEW_RENDERED"
      && state.session.proposal?.acceptance_eligible === true
      && state.acknowledgedPreviewPlanId === state.session.proposal.child_plan_id,
    window: { confirm() { throw new Error("native_confirmation_unavailable"); } },
    toast: (...args) => toasts.push(args), crypto: { randomUUID: () => "test-request" },
    api: (url, request) => {
      requests.push({ url, request });
      return new Promise((resolve, reject) => { deliver = { resolve, reject }; });
    },
    normalizeAcceptanceAction: payload => payload,
    syncPreviewRuntimeFromSession() {}, renderAll() {},
  });
  const start = (type = "accept_repair") => {
    const action = submit(type);
    action.catch(() => {}); // Preserve a clean assertion failure against the old native-dialog implementation.
    return action;
  };
  const completedSession = structuredClone(state.session);
  const succeed = () => deliver.resolve({ session: { ...completedSession, revision: 5, accepted_plan_id: "child", draft: [], proposal: null }, workspace: {} });
  return { state, dialog, message, confirm, requests, toasts, start, succeed, fail: error => deliver.reject(error) };
}

test("decision waits for an in-page confirmation; Cancel and dismissal send nothing", async () => {
  const h = acceptanceHarness();
  for (const outcome of ["cancel", undefined]) {
    const action = h.start();
    assert.equal(h.dialog.open, true);
    assert.equal(h.requests.length, 0);
    assert.match(h.message.textContent, /independently evaluated repair/);
    assert.equal(h.confirm.textContent, "Accept repair");
    h.dialog.close(outcome);
    await action;
    assert.equal(h.requests.length, 0);
    assert.equal(h.state.session.accepted_plan_id, "parent");
    assert.equal(h.dialog.dataset.submitting, undefined);
  }
});

test("explicit confirmation sends one revision-bound decision and updates the accepted state", async () => {
  const h = acceptanceHarness();
  const action = h.start();
  h.dialog.close("confirm");
  await Promise.resolve();
  assert.equal(h.requests.length, 1);
  assert.equal(h.requests[0].url, "/api/sessions/session/accept");
  assert.deepEqual(h.requests[0].request.body, {
    expected_revision: 4, confirmation: "accept_evaluated_repair", idempotency_key: "accept:test-request",
  });
  h.succeed();
  await action;
  assert.equal(h.state.session.accepted_plan_id, "child");
  assert.equal(h.dialog.dataset.submitting, undefined);
});

test("repeated actions cannot open duplicate dialogs or dispatch while a decision is pending", async () => {
  const h = acceptanceHarness();
  const action = h.start();
  await h.start();
  assert.equal(h.dialog.shows, 1);
  h.dialog.close("confirm");
  await Promise.resolve();
  await h.start();
  assert.equal(h.dialog.shows, 1);
  assert.equal(h.requests.length, 1);
  h.succeed();
  await action;
});

for (const decisionType of ["accept_repair", "keep_original"]) {
  for (const [name, change] of [
    ["newer selection revision", h => { h.state.session.revision = 8; h.state.session.selected_day = 7; }],
    ["different session", h => { h.state.session = { ...h.state.session, session_id: "other-session", revision: 0 }; }],
    ["different accepted parent", h => { h.state.session.accepted_plan_id = "newer-parent"; }],
    ["rebound conversation", h => { h.state.session.conversation_id = "new-conversation"; }],
  ]) test(`late ${decisionType} response preserves ${name}`, async () => {
    const h = acceptanceHarness();
    const action = h.start(decisionType);
    h.dialog.close("confirm");
    await Promise.resolve();
    assert.equal(h.requests.length, 1);
    change(h);
    const current = structuredClone(h.state.session);
    const workspace = { newerWorkspace: true };
    h.state.workspace = workspace;
    h.succeed();
    await action;
    assert.deepEqual(h.state.session, current, "A delayed decision response must not replace newer visible state");
    assert.equal(h.state.workspace, workspace);
    assert.equal(h.dialog.dataset.submitting, undefined);
    assert.match(h.toasts.at(-1)[0], /changed.*saved decision/i);
  });
}

for (const [name, change] of [
  ["session revision", h => { h.state.session.revision += 1; }],
  ["session identity", h => { h.state.session.session_id = "other"; }],
  ["accepted parent", h => { h.state.session.accepted_plan_id = "other"; }],
  ["preview identity", h => { h.state.session.proposal.child_plan_id = "other"; h.state.acknowledgedPreviewPlanId = "other"; }],
  ["preview content", h => { h.state.session.proposal.child_plan_content_hash = "other-hash"; }],
  ["map readiness", h => { h.state.previewRuntime.status = "MAP_RENDER_FAILED"; }],
]) test(`confirmation refuses a changed ${name} before dispatch`, async () => {
  const h = acceptanceHarness();
  const action = h.start();
  change(h);
  h.dialog.close("confirm");
  await action;
  assert.equal(h.requests.length, 0);
  assert.match(h.toasts.at(-1)[0], /changed.*confirm again/i);
  assert.equal(h.dialog.dataset.submitting, undefined);
});

test("Keep original uses the same explicit confirmation boundary", async () => {
  const h = acceptanceHarness();
  const action = h.start("keep_original");
  assert.match(h.message.textContent, /immutable original/);
  assert.equal(h.confirm.textContent, "Keep original");
  assert.equal(h.requests.length, 0);
  h.dialog.close("confirm");
  await Promise.resolve();
  assert.equal(h.requests[0].url, "/api/sessions/session/keep-original");
  assert.equal(h.requests[0].request.body.confirmation, "keep_original_plan");
  h.fail(new Error("controlled_server_failure"));
  await action;
  assert.equal(h.state.session.accepted_plan_id, "parent");
  assert.equal(h.dialog.dataset.submitting, undefined);
  const retry = h.start("keep_original");
  assert.equal(h.dialog.returnValue, "cancel");
  h.dialog.close();
  await retry;
  assert.equal(h.requests.length, 1);
});

test("boot restores the selected child stop inspector before the first render", async () => {
  const state = { route: "/app/itinerary", mapInteraction: { selectionKind: null, selectedStopPlanId: null } };
  const selected = { id: "pfeiffer", plan_id: "child", day: 5 };
  let firstRender = null;
  let startupError = null;
  const boot = loadFunction(sourceSection(appSource, "async function boot()", "\nasync function restoreOrCreateSession()"), "boot", {
    state,
    api: async (path) => path === "/api/health"
      ? { schema_version: "product-health-v2", product_id: "itinerary-repair-copilot", core_ready: true }
      : { schema_version: "product-map-configuration-v2" },
    restorePendingMutationInput() {},
    mutationRecovery: { restore: async () => null, pending: null },
    restoreOrCreateSession: async () => ({ workspace: {}, registry: {}, session: { selected_stop_id: "pfeiffer", selected_day: 5 } }),
    syncPreviewRuntimeFromSession() {},
    loadDraftImpactPreview: async () => {},
    bindEvents() {},
    applyRuntimeAvailability() {},
    $: () => ({}),
    findSelectedStop: () => state.session?.selected_stop_id === selected.id ? selected : null,
    requestAnimationFrame: (callback) => callback(),
    renderAll: () => { firstRender = structuredClone(state.mapInteraction); },
    copilot: { restore: async () => {} },
    renderStartupFailure: (error) => { startupError = error; },
  });
  await boot();
  assert.equal(startupError, null);
  assert.deepEqual(firstRender, { selectionKind: "stop", selectedStopPlanId: "child" });
});
