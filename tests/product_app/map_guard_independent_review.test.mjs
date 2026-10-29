// Independent lower-layer review only: real controller code with synthetic DOM,
// map and tile inputs. This is not a browser pointer or WebGL acceptance claim.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const mapSource = readFileSync(new URL("../../src/itinerary_system/product_app/static/js/map-controller-v2.js", import.meta.url), "utf8");
const appSource = readFileSync(new URL("../../src/itinerary_system/product_app/static/js/app.js", import.meta.url), "utf8");
function section(source, from, to) {
  const start = source.indexOf(from), end = source.indexOf(to, start + from.length);
  assert.ok(start >= 0 && end > start);
  return source.slice(start, end).replaceAll("export function", "function");
}
function extract(source, name, deps) {
  return new Function(...Object.keys(deps), `${source}\nreturn ${name};`)(...Object.values(deps));
}
const invalidation = section(mapSource, "export function invalidateMapReadiness(", "function clippedPolygonArea(");
const invalidate = extract(invalidation, "invalidateMapReadiness", {});
const controllerCode = section(mapSource, "export function createGeographicMapController(", "/** Link two pane-owned controllers");

function harness(onStatus = () => {}) {
  const statuses = [], maps = [], frames = new Map(), collectedStopPlans = [];
  const box = { width: 640, height: 600, left: 0, top: 0, right: 640, bottom: 600 };
  const container = { dataset: {}, isConnected: true, clientWidth: 640, clientHeight: 600, getBoundingClientRect: () => box };
  let coverage = 1, pendingDigest = null;
  class FakeElement {
    constructor(qualifies = true) { this.qualifies = qualifies; this.captures = []; }
    closest(selector) { assert.equal(selector, ".maplibregl-canvas, .draft-waypoint-marker"); return this.qualifies ? this : null; }
    setPointerCapture(id) { this.captures.push(id); }
  }
  class FakeMap {
    constructor() {
      maps.push(this); this.events = new Map(); this.domEvents = new Map();
      this.canvas = { setAttribute() {}, getBoundingClientRect: () => box };
    }
    on(name, ...args) { const callback = args.at(-1); this.events.set(name, [...(this.events.get(name) || []), { callback, once: false }]); }
    once(name, callback) { this.events.set(name, [...(this.events.get(name) || []), { callback, once: true }]); }
    emit(name) {
      const callbacks = [...(this.events.get(name) || [])];
      this.events.set(name, callbacks.filter(row => !row.once));
      callbacks.forEach(row => row.callback({}));
    }
    getCanvasContainer() { return { addEventListener: (name, callback) => this.domEvents.set(name, callback) }; }
    getCanvas() { return this.canvas; }
    resize() {} fitBounds() {} triggerRepaint() {}
    addSource() {} addLayer() {}
    remove() { this.emit("remove"); }
  }
  const empty = () => ({ type: "FeatureCollection", features: [] });
  const noop = () => {};
  const diagnostics = planId => ({ activePlanId: planId, activePlanContentHash: `${planId}-hash`,
    activeGeometryHash: "a".repeat(64), activeMarkerIds: ["stop"], gapCount: 0 });
  const create = extract(controllerCode, "createGeographicMapController", {
    Element: FakeElement,
    window: { maplibregl: { Map: FakeMap }, addEventListener: noop, removeEventListener: noop },
    ResizeObserver: class { observe() {} disconnect() {} },
    requestAnimationFrame: callback => { const id = frames.size + 1; frames.set(id, callback); return id; },
    cancelAnimationFrame: id => frames.delete(id),
    selectVisiblePlans: (_geography, id) => ({ original: { plan_id: "parent" },
      plans: id === "parent" ? [{plan_id:id}] : [{plan_id:"parent"},{plan_id:id}] }),
    collectFeatureCollection: (plans, field) => {
      if (field === "stops") collectedStopPlans.push(plans.map(plan => plan.plan_id));
      return empty();
    }, deriveRouteAnchors: empty, resolveGapMarkers: () => ({ descriptions: [] }),
    exploratoryPinFeature: empty, draftRouteFeatureCollection: empty, poiCandidateFeatureCollections: empty,
    validateDraftWaypoint: noop, textFallbackFor: empty,
    diagnosticsFor: (_plans, _gaps, id) => pendingDigest ? new Promise(resolve => { pendingDigest.resolve = () => resolve(diagnostics(id)); }) : diagnostics(id),
    loadMapRuntime: async () => ({}), boundsForDay: () => null, boundsForPlans: () => [0, 0, 1, 1],
    validateBounds: extract(section(mapSource, "function isFiniteNumber(", "function validateCoverage("), "validateBounds", {}),
    styleForRetainedCoverage: async () => ({ style: {}, overview: false }),
    addInterestAreaLayers: noop, addInterestPointLayers: noop,
    invalidateMapReadiness: invalidate,
    hideVisualCanvas: extract(section(mapSource, "function hideVisualCanvas(", "function canvasViewportCoverage("), "hideVisualCanvas", {}),
    canvasViewportCoverage: () => coverage,
    mapTileReadiness: () => ({ tilesLoaded: coverage >= 0.95, tileCoverageRatio: coverage, tileSourceCount: 1 }),
    addRouteLayers: noop, addDraftRouteLayers: noop, addPointLayers: noop, addPoiCandidateLayers: noop,
    addDraftWaypointMarkers: () => [],
  });
  const controller = create({ container, mapConfig: {}, onStatus: status => { statuses.push(status); onStatus(status); } });
  const render = async id => {
    const result = await controller.render({ geography: {}, activePlanId: id });
    assert.ok(result, JSON.stringify(statuses)); result.emit("load"); result.emit("idle"); return result;
  };
  return { controller, container, statuses, maps, FakeElement, render, collectedStopPlans,
    setCoverage: value => { coverage = value; },
    deferDigest: () => { pendingDigest = {}; return pendingDigest; } };
}

test("drawn stop features belong only to the active plan, not the baseline overlay", async () => {
  const h = harness();
  await h.render("child");
  await h.render("parent");
  assert.deepEqual(h.collectedStopPlans, [["child"],["parent"]],
    "Removed baseline stops must not remain visible or clickable in a child preview");
});

test("selected-day route highlighting excludes the baseline when a child is active", () => {
  const layers = [];
  const add = extract(section(mapSource,"function addRouteLayers(","function exploratoryPinFeature("),"addRouteLayers", {
    roleFilter: role => ["==",["get","display_role"],role],
  });
  add({addSource() {},addLayer: layer => layers.push(layer)}, {features:[]}, null, 5, "child");
  const filter = layers.find(layer => layer.id === "v2-route-selected-day").filter;
  const matches = (expression, properties) => {
    if (!Array.isArray(expression)) return expression;
    if (expression[0] === "get") return properties[expression[1]];
    if (expression[0] === "all") return expression.slice(1).every(item => matches(item,properties));
    return matches(expression[1],properties) === matches(expression[2],properties);
  };
  assert.equal(matches(filter,{day:5,plan_id:"child"}),true);
  assert.equal(matches(filter,{day:5,plan_id:"parent"}),false);
  assert.equal(matches(filter,{day:6,plan_id:"child"}),false);
});
function assertWithdrawn(container) {
  assert.notEqual(container.dataset.mapReady, "true");
  for (const key of ["mapReadyAt", "mapTileCoverage", "mapCoverageMethod", "renderedMapPlanId",
    "renderedPlanContentHash", "renderedGeometryHash", "renderedMarkerIds"]) {
    assert.equal(container.dataset[key], undefined, `${key} must be withdrawn`);
  }
}

test("pointer capture is restricted to primary mouse canvas/marker elements", async () => {
  const h = harness(); const map = await h.render("child");
  const pointerdown = map.domEvents.get("pointerdown");
  for (const target of [new h.FakeElement(), new h.FakeElement()]) {
    pointerdown({ target, pointerType: "mouse", button: 0, pointerId: 17 });
    assert.deepEqual(target.captures, [17]);
  }
  for (const event of [
    { target: new h.FakeElement(false), pointerType: "mouse", button: 0 },
    { target: new h.FakeElement(), pointerType: "touch", button: 0 },
    { target: new h.FakeElement(), pointerType: "pen", button: 0 },
    { target: new h.FakeElement(), pointerType: "mouse", button: 2 },
    { target: {}, pointerType: "mouse", button: 0 },
  ]) {
    pointerdown({ pointerId: 18, ...event });
    if (event.target.captures) assert.deepEqual(event.target.captures, []);
  }
});

test("WebGL loss withdraws identity and idle cannot republish before restoration", async () => {
  const h = harness(); const map = await h.render("child");
  assert.equal(h.container.dataset.mapReady, "true");
  map.emit("webglcontextlost"); assertWithdrawn(h.container);
  assert.equal(h.statuses.at(-1).code, "map_webgl_context_lost");
  map.emit("idle"); assertWithdrawn(h.container);
  assert.equal(h.statuses.at(-1).code, "map_webgl_context_lost");
  map.emit("webglcontextrestored"); assertWithdrawn(h.container);
  assert.equal(h.statuses.at(-1).state, "loading");
  h.setCoverage(0.94); map.emit("idle"); assertWithdrawn(h.container);
  h.setCoverage(0.95); map.emit("idle");
  assert.equal(h.container.dataset.mapReady, "true");
  assert.equal(h.container.dataset.renderedMapPlanId, "child");
});

test("old map loss/error/idle callbacks cannot withdraw or overwrite a newer acknowledgement", async () => {
  const h = harness(); const old = await h.render("old"), current = await h.render("current");
  const expected = structuredClone(h.container.dataset), count = h.statuses.length;
  for (const name of ["webglcontextlost", "webglcontextrestored", "error", "movestart", "idle"]) old.emit(name);
  assert.deepEqual(h.container.dataset, expected); assert.equal(h.statuses.length, count);
  current.emit("error"); assertWithdrawn(h.container);
});

test("disposing a map withdraws every previous acknowledgement field", async () => {
  const h = harness(); await h.render("child"); h.controller.dispose();
  assertWithdrawn(h.container);
});

test("replacement render revokes the previous ready status before awaiting its geometry digest", async () => {
  const h = harness(); await h.render("child"); const digest = h.deferDigest();
  const rendering = h.controller.render({ geography: {}, activePlanId: "child" });
  const snapshot = { dataset: structuredClone(h.container.dataset), status: h.statuses.at(-1) };
  digest.resolve(); await rendering;
  assert.equal(snapshot.status.state, "loading", "callers must lose their old acceptance acknowledgement immediately");
  assertWithdrawn({ dataset: snapshot.dataset });
});

test("real application acceptance gate is revoked throughout an asynchronous same-plan map replacement", async () => {
  const plan = { plan_id: "child", content_hash: "child-hash", stops: { features: [{ properties: { stop_id: "stop" } }] } };
  const state = { session: { proposal: { state: "eligible", acceptance_eligible: true, child_plan_id: "child" } },
    previewRuntime: { status: "PREVIEW_DATA_READY", epoch: 1, requestId: "request" }, acknowledgedPreviewPlanId: null };
  const stage = { dataset: {} }, statusElement = { dataset: {}, closest: () => stage, classList: { remove() {}, add() {} } };
  const actualCode = section(appSource, "function updateGeographicMapStatus(", "function timeWindowLabel(")
    + section(appSource, "function previewCanBeAccepted()", "async function submitAcceptance(");
  const functions = extract(`${actualCode}\nconst result = { updateGeographicMapStatus, previewCanBeAccepted };`, "result", {
    state, $: () => statusElement, activeRenderPlan: () => plan, activeProposalPlan: () => plan,
    copilot: { previewStateChanged() {} }, toast() {}, renderContextPanel() {},
  });
  const h = harness(functions.updateGeographicMapStatus);
  await h.render("child"); assert.equal(functions.previewCanBeAccepted(), true);
  // renderMap replaces the stage as well as the map; old identity attributes
  // cannot be carried into this check by the synthetic DOM fixture.
  stage.dataset = {};
  const digest = h.deferDigest();
  const replacement = h.controller.render({ geography: {}, activePlanId: "child" });
  const duringReplacement = { acceptable: functions.previewCanBeAccepted(), status: state.previewRuntime.status,
    acknowledged: state.acknowledgedPreviewPlanId, mapReady: h.container.dataset.mapReady,
    activePlanId: stage.dataset.activePlanId, activePlanContentHash: stage.dataset.activePlanContentHash };
  digest.resolve(); await replacement;
  assert.deepEqual(duringReplacement, { acceptable: false, status: "PREVIEW_DATA_READY", acknowledged: null, mapReady: "false",
    activePlanId: "child", activePlanContentHash: "child-hash" });
});

test("Trip metrics hide only in Compare/Evidence and return with the same canonical identity", () => {
  const state = { route: "/app/itinerary" }, strip = { dataset: {}, hidden: false };
  const plan = { plan_id: "accepted-child", content_hash: "accepted-hash" };
  const render = extract(section(appSource, "function renderMetrics()", "function evaluatedProposalCompareContext()"), "renderMetrics", {
    state, $: () => strip, activeRenderPlan: () => plan,
    planPresentationEvidence: () => ({ metrics: {}, eligibility: "eligible", certificateId: "certificate-child" }),
    planRouteMinutes: () => 1992.62, formatMetric: String, escapeHtml: String,
  });
  for (const route of ["/app/itinerary", "/app/compare", "/app/evidence", "/app/map", "/app/repairs", "/app/itinerary"]) {
    state.route = route; render();
    assert.equal(strip.hidden, ["/app/compare", "/app/evidence"].includes(route));
    assert.equal(strip.dataset.planId, plan.plan_id); assert.equal(strip.dataset.planContentHash, plan.content_hash);
    assert.equal(strip.dataset.evaluationId, "certificate-child"); assert.match(strip.innerHTML, /1992.62 min/);
  }
});
