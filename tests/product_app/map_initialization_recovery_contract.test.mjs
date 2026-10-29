// Fault injection only: execute the real application functions against local
// synthetic health/config responses and captured accepted-session artifacts.
// This does not claim browser tile coverage, real network, or provider testing.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { recordedJson } from "./recorded-runtime-fixture.mjs";

const appSource = readFileSync(new URL("../../src/itinerary_system/product_app/static/js/app.js", import.meta.url), "utf8");
const capture = (run, name) => recordedJson(`results/e2e/${run}/accepted_reload/${name}`);
const failedCapture = capture("r01_provider_isolation_20260907_v1", "reload_failure_diagnostics.json");
const acceptedPayload = capture("r01_provider_isolation_20260907_v2", "reload_response.json");
const successfulCapture = capture("r01_provider_isolation_20260907_v2", "reload_rendered_diagnostics.json");
const acceptedPlan = acceptedPayload.workspace.geography.plans.find(
  (plan) => plan.plan_id === acceptedPayload.session.accepted_plan_id,
);
assert.ok(acceptedPlan && acceptedPlan.role === "accepted", "use the actual persisted child, not a synthetic parent");
assert.equal(failedCapture.map_stage.renderedMapPlanId, undefined, "the original failure has no verified map identity");
assert.equal(successfulCapture.map_stage.renderedMapPlanId, acceptedPlan.plan_id);
assert.equal(successfulCapture.map_stage.activePlanContentHash, acceptedPlan.content_hash);

function functionSource(name, required = true) {
  const marker = new RegExp(`^(?:async )?function ${name}\\(`, "m").exec(appSource);
  if (!marker) {
    assert.equal(required, false, `${name} must exist in the actual application`);
    return "";
  }
  const remaining = appSource.slice(marker.index + marker[0].length);
  const end = /\n(?:async )?function \w+\(/.exec(remaining);
  return appSource.slice(marker.index, end ? marker.index + marker[0].length + end.index : undefined);
}

const healthy = () => ({
  schema_version: "product-health-v2", product_id: "itinerary-repair-copilot", core_ready: true,
  components: { map: { name: "map", status: "ready", code: "maplibre_ready" } },
});
const readyConfig = () => ({ schema_version: "product-map-configuration-v2", status: "ready", code: "maplibre_ready" });
const unavailableConfig = () => ({
  schema_version: "product-map-configuration-v2", status: "unavailable", code: "local_map_health_timeout",
});

function fakeElement() {
  const classes = new Set();
  const listeners = new Map();
  return {
    dataset: {}, hidden: false, innerHTML: "", textContent: "", disabled: false, title: "", listeners,
    classList: {
      add: (...names) => names.forEach((name) => classes.add(name)),
      remove: (...names) => names.forEach((name) => classes.delete(name)),
      contains: (name) => classes.has(name),
      toggle: (name, enabled) => enabled ? classes.add(name) : classes.delete(name),
    },
    addEventListener: (name, callback) => listeners.set(name, callback),
  };
}

function harness({ configs = [readyConfig()], mapHealth = "ready" } = {}) {
  let queue = [...configs];
  const requests = [];
  const delays = [];
  const renders = [];
  const messages = [];
  const elements = new Map();
  const stage = fakeElement();
  const element = (selector) => {
    if (!elements.has(selector)) elements.set(selector, fakeElement());
    const value = elements.get(selector);
    value.closest = () => stage;
    return value;
  };
  const state = {
    route: "/app/itinerary", mapConfig: null, health: null, session: null,
    mapInteraction: { selectionKind: null, selectedStopPlanId: null },
    previewRuntime: { status: "PARENT_RENDERED", epoch: 0, requestId: null },
    acknowledgedPreviewPlanId: null,
  };
  let restores = 0;
  let startupError = null;
  let functions;
  const dependencies = {
    state, $: element,
    api: async (path, options = {}) => {
      requests.push({ path, method: options.method || "GET" });
      if (path === "/api/health") {
        const response = healthy();
        response.components.map.status = mapHealth;
        return response;
      }
      assert.equal(path, "/api/map/config", "no new session, preview, acceptance, or provider request is allowed");
      const response = queue.length > 1 ? queue.shift() : queue[0];
      if (response instanceof Error) throw response;
      return structuredClone(response);
    },
    setTimeout: (callback, milliseconds) => { delays.push(milliseconds); callback(); },
    clearTimeout() {},
    requestAnimationFrame: (callback) => callback(),
    restoreOrCreateSession: async () => { restores += 1; return structuredClone(acceptedPayload); },
    syncPreviewRuntimeFromSession() {}, loadDraftImpactPreview: async () => {}, bindEvents() {},
    restorePendingMutationInput() {},
    mutationRecovery: {restore: async () => {}, pending: null},
    findSelectedStop: () => state.session ? { id: state.session.selected_stop_id, plan_id: acceptedPlan.plan_id } : null,
    activeRenderPlan: () => state.session ? acceptedPlan : null,
    activeProposalPlan: () => null,
    renderContextPanel() {},
    renderAll: () => {
      renders.push({ config: structuredClone(state.mapConfig), sessionId: state.session.session_id });
      functions.updateGeographicMapStatus(state.mapConfig?.status === "ready"
        ? { state: "loading", code: "injected_map_tiles_pending" }
        : { state: "failed", code: "map_configuration_unavailable" });
    },
    copilot: { restore: async () => {}, update() {}, previewStateChanged() {} },
    toast: (message) => messages.push(message),
    renderStartupFailure: (error) => { startupError = error; },
    renderCoreRecovery: () => { throw new Error("unexpected core failure in map-only fault injection"); },
    escapeHtml: (value) => String(value),
  };
  const copyStart = appSource.indexOf("const COMPONENT_COPY =");
  const copyEnd = appSource.indexOf("function componentMessage(", copyStart);
  const implementation = appSource.slice(copyStart, copyEnd) + [
    "componentMessage", "componentEntries", "component", "applyRuntimeAvailability", "updateGeographicMapStatus", "boot",
  ].map((name) => functionSource(name)).join("\n") + functionSource("refreshMapRuntime", false);
  functions = new Function(...Object.keys(dependencies), `${implementation}\nreturn {
    boot, applyRuntimeAvailability, updateGeographicMapStatus,
    refreshMapRuntime: typeof refreshMapRuntime === "function" ? refreshMapRuntime : null,
  };`)(...Object.values(dependencies));
  const acknowledge = (overrides = {}) => functions.updateGeographicMapStatus({
    state: "ready", code: "geographic_map_v2_ready", activePlanId: acceptedPlan.plan_id,
    activePlanContentHash: acceptedPlan.content_hash,
    activeGeometryHash: successfulCapture.map_stage.renderedGeometryHash,
    activeMarkerIds: JSON.parse(successfulCapture.map_stage.renderedMarkerIds),
    visiblePlanCount: 2, gapCount: 0, ...overrides,
  });
  return {
    ...functions, state, stage, element, requests, delays, renders, messages, acknowledge,
    setConfigs: (responses) => { queue = [...responses]; },
    restoreCount: () => restores,
    startupError: () => startupError,
    configCount: () => requests.filter((request) => request.path === "/api/map/config").length,
  };
}

function assertAcceptedSessionUnchanged(h) {
  assert.equal(h.startupError(), null);
  assert.equal(h.restoreCount(), 1);
  assert.equal(h.state.session.session_id, acceptedPayload.session.session_id);
  assert.equal(h.state.session.accepted_plan_id, acceptedPlan.plan_id);
  assert.equal(h.state.session.conversation_id, acceptedPayload.session.conversation_id);
  assert.equal(h.state.session.selected_stop_id, "pfeiffer_big_sur_state_park");
  assert.equal(h.state.session.selected_day, 5);
  assert.equal(h.state.mapInteraction.selectedStopPlanId, acceptedPlan.plan_id);
  assert.equal(h.state.session.proposal, null, "reload does not manufacture a new preview or acceptance");
}

test("fault injection: transient non-ready config recovers before rendering the same accepted child", async () => {
  const h = harness({ configs: [unavailableConfig(), readyConfig()] });
  await h.boot();
  assertAcceptedSessionUnchanged(h);
  assert.equal(h.configCount(), 2, "a transient failure must not be cached permanently");
  assert.equal(h.state.mapConfig.status, "ready");
  assert.equal(h.renders.length, 1);
  assert.equal(h.renders[0].config.status, "ready");
  assert.deepEqual(h.delays, [1000]);
  assert.equal(h.stage.dataset.renderedMapPlanId, undefined, "config readiness alone is not a map acknowledgement");
  assert.deepEqual(h.messages, []);
  h.acknowledge();
  assert.equal(h.stage.dataset.renderedMapPlanId, acceptedPlan.plan_id);
  assert.equal(h.stage.dataset.activePlanContentHash, acceptedPlan.content_hash);
  assert.equal(h.stage.dataset.renderedGeometryHash, successfulCapture.map_stage.renderedGeometryHash);
  assert.deepEqual(JSON.parse(h.stage.dataset.renderedMarkerIds), JSON.parse(successfulCapture.map_stage.renderedMarkerIds));
  assert.equal(h.state.previewRuntime.status, "PARENT_RENDERED", "persisted accepted plan is not a new preview");
});

test("fault injection: rejected config request retries without creating or replacing the accepted session", async () => {
  const h = harness({ configs: [new Error("injected local health timeout"), readyConfig()] });
  await h.boot();
  assertAcceptedSessionUnchanged(h);
  assert.equal(h.configCount(), 2);
  assert.equal(h.state.mapConfig.status, "ready");
  assert.equal(h.stage.dataset.renderedMapPlanId, undefined);
});

test("ready on first attempt does not incur unnecessary retry or publish pre-ready success", async () => {
  const h = harness();
  await h.boot();
  assertAcceptedSessionUnchanged(h);
  assert.equal(h.configCount(), 1);
  assert.deepEqual(h.delays, []);
  assert.equal(h.stage.dataset.renderedMapPlanId, undefined);
  assert.match(h.element("#map-render-status").textContent, /Loading/i);
  assert.deepEqual(h.messages, []);
});

test("fault injection: failed current map health cannot retain an apparently ready configuration", async () => {
  const h = harness({ configs: [readyConfig()], mapHealth: "failed" });
  await h.boot();
  assertAcceptedSessionUnchanged(h);
  assert.equal(h.configCount(), 3);
  assert.deepEqual(h.delays, [1000, 2000]);
  assert.notEqual(h.state.mapConfig?.status, "ready", "failed recovery must not leave renderable ready config cached");
  assert.equal(h.stage.dataset.renderedMapPlanId, undefined);
  assert.equal(h.element("#layers-button").disabled, true);
  assert.match(h.element("#runtime-banner").innerHTML, /retry-map-runtime/);
});

for (const [label, configuration] of [
  ["unavailable", unavailableConfig()],
  ["wrong schema", { schema_version: "wrong-version", status: "ready" }],
  ["request error", new Error("injected persistent config timeout")],
]) {
  test(`fault injection: persistent ${label} is bounded and exposes explicit retry without false success`, async () => {
    const h = harness({ configs: [configuration] });
    await h.boot();
    assertAcceptedSessionUnchanged(h);
    assert.equal(h.configCount(), 3);
    assert.deepEqual(h.delays, [1000, 2000]);
    assert.notEqual(h.state.mapConfig?.status, "ready");
    assert.equal(h.stage.dataset.renderedMapPlanId, undefined);
    assert.equal(h.element("#layers-button").disabled, true);
    assert.equal(h.element("#runtime-banner").hidden, false);
    assert.match(h.element("#runtime-banner").innerHTML, /retry-map-runtime/);
    assert.equal(h.element("#map-render-status").hidden, false);
    assert.match(h.element("#map-render-status").textContent, /could not be verified|unavailable/i);
    assert.deepEqual(h.messages, []);
  });
}

test("fault injection: recovered config clears a stale warning and keeps the session through explicit retry", async () => {
  const h = harness({ configs: [unavailableConfig()] });
  await h.boot();
  assertAcceptedSessionUnchanged(h);
  assert.match(h.element("#runtime-banner").innerHTML, /retry-map-runtime/);
  const retry = h.element("#retry-map-runtime").listeners.get("click");
  assert.equal(typeof retry, "function", "the visible retry control must have executable behavior");
  const before = structuredClone(h.state.session);
  h.setConfigs([readyConfig()]);
  await retry();
  assert.deepEqual(h.state.session, before);
  assert.equal(h.restoreCount(), 1);
  assert.equal(h.state.mapConfig.status, "ready");
  assert.equal(h.element("#runtime-banner").hidden, true);
  assert.equal(h.element("#app-shell").classList.contains("has-runtime-banner"), false);
  assert.equal(h.element("#layers-button").disabled, false);
  assert.equal(h.renders.length, 2);
  assert.equal(h.stage.dataset.renderedMapPlanId, undefined);
  h.acknowledge();
  assert.equal(h.stage.dataset.renderedMapPlanId, before.accepted_plan_id);
});

test("recovery cannot acknowledge the old parent or a mismatched accepted marker set", async () => {
  const h = harness({ configs: [unavailableConfig(), readyConfig()] });
  await h.boot();
  h.acknowledge({ activePlanId: "plan_359968d255ff9288" });
  assert.equal(h.state.previewRuntime.status, "MAP_IDENTITY_FAILED");
  assert.equal(h.stage.dataset.renderedMapPlanId, undefined);
  h.acknowledge({ activeMarkerIds: ["bixby_creek_bridge_viewpoint"] });
  assert.equal(h.stage.dataset.renderedMapPlanId, undefined);
  h.acknowledge();
  assert.equal(h.stage.dataset.renderedMapPlanId, acceptedPlan.plan_id);
});
