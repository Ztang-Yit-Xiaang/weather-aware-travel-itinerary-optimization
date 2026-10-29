import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { invalidateMapReadiness } from "../../src/itinerary_system/product_app/static/js/map-controller-v2.js";

const root = new URL("../../src/itinerary_system/product_app/static/js/", import.meta.url);
const source = name => readFileSync(new URL(name, root), "utf8");
function section(text, start, end) {
  const from = text.indexOf(start), to = end ? text.indexOf(end, from + start.length) : text.length;
  assert.ok(from >= 0 && to > from);
  return text.slice(from, to).replaceAll("export function", "function").replaceAll("export async function", "async function");
}
const load = (text, name, deps) => new Function(...Object.keys(deps), `${text}\nreturn ${name};`)(...Object.values(deps));
const camera = { center: [-121.89, 36.25], zoom: 12, bearing: 0, pitch: 0 };

function controllerHarness() {
  const box = { width: 800, height: 600 }, observers = [], frames = [], statuses = [];
  const container = { dataset: {}, isConnected: true, clientWidth: 800, clientHeight: 600,
    getBoundingClientRect: () => box };
  class FakeMap {
    constructor(options) { this.options = options; this.events = new Map(); this.fits = 0;
      this.view = options.center ? structuredClone(camera) : { ...camera, zoom: 6 }; }
    on(name, ...args) { this.listen(name, args.at(-1), false); }
    once(name, callback) { this.listen(name, callback, true); }
    listen(name, callback, once) { this.events.set(name, [...(this.events.get(name) || []), { callback, once }]); }
    emit(name) { const rows = this.events.get(name) || []; this.events.set(name, rows.filter(r => !r.once)); rows.forEach(r => r.callback({})); }
    getCenter() { return { lng: this.view.center[0], lat: this.view.center[1] }; }
    getZoom() { return this.view.zoom; } getBearing() { return this.view.bearing; } getPitch() { return this.view.pitch; }
    getCanvasContainer() { return { addEventListener() {} }; }
    getCanvas() { return { setAttribute() {} }; }
    addSource() {} addLayer() {} resize() {} triggerRepaint() {}
    fitBounds() { this.fits += 1; this.view.zoom = 6; }
    remove() { this.emit("remove"); }
  }
  const noop = () => {}, empty = () => ({ features: [] });
  const create = load(section(source("map-controller-v2.js"), "export function createGeographicMapController(", "/** Link two pane-owned controllers"), "createGeographicMapController", {
    window: { maplibregl: { Map: FakeMap }, addEventListener() {}, removeEventListener() {} },
    ResizeObserver: class { constructor(callback) { observers.push(callback); } observe() {} disconnect() {} },
    requestAnimationFrame: callback => { frames.push(callback); return frames.length; }, cancelAnimationFrame: noop,
    selectVisiblePlans: () => ({ original: { plan_id: "parent" }, plans: [{ plan_id: "parent" }] }),
    collectFeatureCollection: empty, deriveRouteAnchors: empty, resolveGapMarkers: () => ({ descriptions: [] }),
    exploratoryPinFeature: empty, draftRouteFeatureCollection: empty, poiCandidateFeatureCollections: empty,
    validateDraftWaypoint: noop, textFallbackFor: empty,
    diagnosticsFor: async () => ({ activePlanId: "parent", activePlanContentHash: "hash", gapCount: 0 }),
    loadMapRuntime: async () => ({}), boundsForDay: () => null, boundsForPlans: () => [-122, 36, -121, 37],
    styleForRetainedCoverage: async () => ({ style: {}, overview: false }), validateBounds: Array.isArray,
    invalidateMapReadiness, hideVisualCanvas: noop, canvasViewportCoverage: () => 1,
    mapTileReadiness: () => ({ tilesLoaded: true, tileCoverageRatio: 1 }),
    addInterestAreaLayers: noop, addInterestPointLayers: noop, addRouteLayers: noop,
    addDraftRouteLayers: noop, addPointLayers: noop, addPoiCandidateLayers: noop, addDraftWaypointMarkers: () => [],
    isPosition: v => Array.isArray(v) && v.length === 2 && v.every(Number.isFinite), isFiniteNumber: Number.isFinite,
  });
  const controller = create({ container, mapConfig: {}, onStatus: status => statuses.push(status) });
  return { container, controller, statuses,
    render: async initialCamera => { const map = await controller.render({ geography: {}, activePlanId: "parent", initialCamera });
      assert.ok(map, JSON.stringify(statuses)); map.emit("load"); map.emit("idle"); return map; },
    resize: () => { box.width = 600; container.clientWidth = 600; observers.at(-1)(); frames.splice(0).forEach(f => f()); } };
}

test("resizing a zoomed editor preserves the camera and still revalidates readiness", async () => {
  const h = controllerHarness(), map = await h.render();
  map.view = structuredClone(camera);
  const fits = map.fits;
  h.resize();
  assert.deepEqual(map.view, camera);
  assert.equal(map.fits, fits);
  assert.notEqual(h.container.dataset.mapReady, "true");
  map.emit("idle"); assert.equal(h.container.dataset.mapReady, "true");
});

test("a retained camera is installed before drawing and is not overwritten on load", async () => {
  const h = controllerHarness(), map = await h.render(camera);
  assert.deepEqual(map.options.center, camera.center);
  assert.equal(map.options.zoom, camera.zoom);
  assert.equal(map.options.bounds, undefined);
  assert.equal(map.fits, 0);
  assert.deepEqual(map.view, camera);
});

function adapterHarness() {
  const controllers = [];
  const code = section(source("map.js"), "let activeController", "\nconst CONFIRMED_PARAMETER_KEYS")
    + section(source("map.js"), "function activePlanId(");
  const api = load(code + "\nconst api = {renderGeographicMap, disposeGeographicMap};", "api", {
    createGeographicMapController: () => {
      const controller = { view: structuredClone(camera), camera() { return structuredClone(this.view); },
        dispose() {}, render: async args => args };
      controllers.push(controller); return controller;
    },
  });
  const geography = { plans: [{ plan_id: "parent", content_hash: "hash", role: "original" }] };
  return { ...api, geography, controllers, render: patch => api.renderGeographicMap({ geography, selectedDay: 5, ...patch }) };
}

test("same-day edit rerenders retain the live camera; new day, plan hash and interest focus refit", async () => {
  const h = adapterHarness();
  await h.render();
  assert.deepEqual((await h.render({ interactionMode: "edit" })).initialCamera, camera);
  assert.equal((await h.render({ selectedDay: 6 })).initialCamera, null);
  assert.deepEqual((await h.render({ selectedDay: 6 })).initialCamera, camera);
  h.geography.plans[0].content_hash = "changed";
  assert.equal((await h.render({ selectedDay: 6 })).initialCamera, null);
  assert.equal((await h.render({ selectedDay: 6, selectedInterestId: "new-place" })).initialCamera, null);
  h.disposeGeographicMap();
  assert.equal((await h.render({ selectedDay: 6, selectedInterestId: "new-place" })).initialCamera, null);
});

test("rapid same-view rerenders retain the camera while an intermediate controller is loading", async () => {
  const h = adapterHarness();
  assert.equal((await h.render()).initialCamera, null);
  assert.deepEqual((await h.render()).initialCamera, camera);
  h.controllers.at(-1).view = null;
  assert.deepEqual((await h.render()).initialCamera, camera);
  const moved = { ...camera, center: [-120.5, 37], zoom: 13 };
  h.controllers.at(-1).view = moved;
  assert.deepEqual((await h.render()).initialCamera, moved);
  h.controllers.at(-1).view = null;
  assert.deepEqual((await h.render()).initialCamera, moved);
});

test("a retained fallback is cleared at day, plan, focus, and external disposal boundaries", async () => {
  const changes = [
    () => ({ selectedDay: 6 }),
    h => { h.geography.plans[0].content_hash = "changed"; return {}; },
    h => { h.geography.plans.push({ plan_id: "other", content_hash: "other-hash" });
      return { selectedAlternativeId: "other" }; },
    () => ({ selectedInterestId: "new-place" }),
    () => ({ interestRegions: { exploration_plan_id: "parent", exploration_plan_content_hash: "hash", context: { day: 5 } } }),
    h => { h.disposeGeographicMap(); return {}; },
  ];
  for (const change of changes) {
    const h = adapterHarness();
    await h.render();
    await h.render();
    h.controllers.at(-1).view = null;
    const patch = change(h);
    assert.equal((await h.render(patch)).initialCamera, null);
    h.controllers.at(-1).view = null;
    assert.equal((await h.render(patch)).initialCamera, null);
  }
});

function refreshHarness({ changed = false, failure = false } = {}) {
  const session = { session_id: "session", trip_id: "trip", run_id: "run", revision: 1,
    accepted_plan_id: "parent", conversation_id: "conversation" };
  const state = { session, mutationRecovery: { blocked: false }, workspaceConflict: null, workspaceSyncPending: false,
    previewRuntime: { epoch: 0, status: "PREVIEW_RENDERED" }, mapInteraction: {} };
  let renders = 0, installs = 0;
  const refresh = load(section(source("app.js"), "async function refreshStaleWorkspace(", "\nfunction escapeHtml("), "refreshStaleWorkspace", {
    state, workspaceNoticeEpoch: 0,
    renderAll: () => { renders += 1; }, copilot: { previewStateChanged() {}, async restore() {} },
    api: async () => { if (failure) throw new Error("offline"); return { session: { ...session, revision: changed ? 2 : 1 }, workspace: {}, registry: {} }; },
    installDraftMutationSession: value => { installs += 1; state.session = value; },
    syncPreviewRuntimeFromSession() {}, resetPoiDiscovery() {}, selectedDraftWaypoint() {},
    loadDraftImpactPreview: async () => {}, toast() {},
  });
  return { state, run: () => refresh(null, true), counts: () => ({ renders, installs }) };
}

test("a background check with unchanged session does not rebuild the editing map", async () => {
  const h = refreshHarness(); await h.run();
  assert.deepEqual(h.counts(), { renders: 0, installs: 0 });
  assert.equal(h.state.previewRuntime.status, "PREVIEW_RENDERED");
});

test("real external changes and failed checks still protect the editor", async () => {
  const changed = refreshHarness({ changed: true }); await changed.run();
  assert.equal(changed.state.session.revision, 2);
  assert.equal(changed.counts().installs, 1); assert.ok(changed.counts().renders > 0);
  const failed = refreshHarness({ failure: true }); await failed.run();
  assert.equal(failed.state.workspaceSyncPending, true);
  assert.equal(failed.state.previewRuntime.status, "PREVIEW_FAILED");
});
