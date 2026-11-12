import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync } from "node:fs";
import { mapTileReadiness } from "../../src/itinerary_system/product_app/static/js/map-controller-v2.js";

const quadrant = (x, y, state = "loaded", z = 1) => ({ canonical: { x, y, z }, wrap: 0, state });
const full = [quadrant(0, 0), quadrant(0, 1), quadrant(1, 0), quadrant(1, 1)];
function mapWith(tiles, { width = 256, height = 256, allTiles = true, second = null } = {}) {
  const sourceTiles = { base: tiles, ...(second ? { labels: second } : {}) };
  return {
    areTilesLoaded: () => allTiles,
    isSourceLoaded: () => true,
    getStyle: () => ({ sources: Object.fromEntries(Object.keys(sourceTiles).map(id => [id, { type: "vector" }])) }),
    getContainer: () => ({ getBoundingClientRect: () => ({ left: 0, top: 0, width, height }) }),
    getProjection: () => ({ type: "mercator" }),
    project: ([lng, lat]) => ({ x: (lng + 180) / 360 * 256,
      y: (1 - Math.asinh(Math.tan(lat * Math.PI / 180)) / Math.PI) / 2 * 256 }),
    style: { tileManagers: Object.fromEntries(Object.entries(sourceTiles).map(([id, rows]) => [id, {
      getVisibleCoordinates: () => rows,
      getTile: tile => ({ state: tile.state, isRenderable: () => tile.state === "loaded" }),
    }])) },
  };
}

test("all sources loaded cannot disguise a missing quarter of the visible map", () => {
  const ready = mapTileReadiness(mapWith(full.slice(0, 3)));
  assert.ok(Math.abs(ready.tileCoverageRatio - 0.75) < 1e-8);
  assert.equal(ready.tilesLoaded, false);
});
test("full loaded projected area is ready", () => {
  const ready = mapTileReadiness(mapWith(full));
  assert.ok(Math.abs(ready.tileCoverageRatio - 1) < 1e-12);
  assert.equal(ready.tilesLoaded, true);
});
test("failed tile is not covered even when renderer aggregate reports loaded", () => {
  const ready = mapTileReadiness(mapWith([...full.slice(0, 3), quadrant(1, 1, "errored")]));
  assert.ok(Math.abs(ready.tileCoverageRatio - 0.75) < 1e-8);
  assert.equal(ready.tilesLoaded, false);
});
test("overlapping parent and child tiles are counted only once", () => {
  assert.ok(Math.abs(mapTileReadiness(mapWith([...full, quadrant(0, 0, "loaded", 0)])).tileCoverageRatio - 1) < 1e-12);
});
test("coverage is clipped to the actual map viewport", () => {
  assert.ok(Math.abs(mapTileReadiness(mapWith(full.slice(0, 2), { width: 128 })).tileCoverageRatio - 1) < 1e-12);
});
test("every visible tile source must cover the viewport", () => {
  assert.ok(Math.abs(mapTileReadiness(mapWith(full, { second: full.slice(0, 3) })).tileCoverageRatio - 0.75) < 1e-8);
});
test("absent renderer tile access fails closed instead of calling source count coverage", () => {
  const map = mapWith(full); delete map.style.tileManagers;
  assert.equal(mapTileReadiness(map).tilesLoaded, false);
  assert.equal(mapTileReadiness(map).tileCoverageRatio, 0);
});
test("all projected tiles are insufficient while tile loading is still in flight", () => {
  assert.equal(mapTileReadiness(mapWith(full, { allTiles: false })).tilesLoaded, false);
});

test("the observed narrow Compare tile footprint remains below the unchanged ready threshold", () => {
  const map = mapWith([quadrant(2, 6, "loaded", 4)], { width: 203.5, height: 334 });
  map.project = ([lng, lat]) => ({
    x: -232.39240214546714 + ((lng+180)/360-2/16)*8238.22388088522,
    y: 19.26875102105541 + ((1-Math.asinh(Math.tan(lat*Math.PI/180))/Math.PI)/2-6/16)*8238.22388088522,
  });
  const result = mapTileReadiness(map);
  assert.ok(Math.abs(result.tileCoverageRatio-0.9423091286794749) < 1e-10);
  assert.equal(result.tilesLoaded, false);
});

test("camera padding scales to the actual pane instead of consuming half a narrow viewport", () => {
  const source = readFileSync(new URL("../../src/itinerary_system/product_app/static/js/map-controller-v2.js", import.meta.url), "utf8").replaceAll("\r\n", "\n");
  const match = source.match(/const fitBoundsOptions = (\{[\s\S]*?\n      \});/);
  assert.ok(match, "Both construction and final fitting must use the same bounded viewport options");
  const options = new Function("container", `return (${match[1]});`);
  assert.deepEqual(options({ clientWidth:204, clientHeight:334 }), {padding:25.5,maxZoom:13,duration:0});
  assert.deepEqual(options({ clientWidth:720, clientHeight:650 }), {padding:54,maxZoom:13,duration:0});
  const construction = source.match(/new window\.maplibregl\.Map\((\{[\s\S]*?\n      \})\)/);
  assert.ok(construction, "Map construction must retain the canonical fit options");
  const fitOptions = options({clientWidth:204, clientHeight:334});
  const initial = new Function("container", "basemap", "initialCamera", "fittedBounds", "fitBoundsOptions", `return (${construction[1]});`)(
    {}, {style: "test-style"}, null, [-123, 34, -118, 39], fitOptions);
  assert.equal(initial.fitBoundsOptions, fitOptions, "Initial fitting uses the same bounded options as refitting");
  assert.ok(source.includes("nextMap.fitBounds(fittedBounds, fitBoundsOptions)"));
  const observe = source.slice(source.indexOf("  function observeMapSize("), source.indexOf("  async function render("));
  assert.ok(observe.startsWith("  function observeMapSize("));
  const frames = [];
  let resizes = 0, repaints = 0;
  const nextMap = {resize: () => resizes++, triggerRepaint: () => repaints++,
    fitBounds() { assert.fail("Pane resizing must preserve the user's current camera"); }};
  new Function("nextMap", "container", "ResizeObserver", "requestAnimationFrame", "window", `
    let generation = 1, map = nextMap, resizeFrame = null, resizeObserver;
    ${observe}
    observeMapSize(nextMap, 1, () => {});
  `)(nextMap, {isConnected: true, dataset: {}, getBoundingClientRect: () => ({width:204, height:334})},
    class {constructor(callback) { this.callback = callback; } observe() { this.callback(); }},
    callback => { frames.push(callback); return 1; }, {addEventListener() {}, removeEventListener() {}});
  assert.equal(frames.length, 1);
  frames[0]();
  assert.equal(resizes, 1);
  assert.equal(repaints, 1);
});
