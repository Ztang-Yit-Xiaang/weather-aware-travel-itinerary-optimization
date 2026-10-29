import assert from "node:assert/strict";
import {
  invalidateMapReadiness,
  mapTileReadiness,
  routeGeometrySha256,
} from "../../src/itinerary_system/product_app/static/js/map-controller-v2.js";

const tileMap = ({ allTiles = true, sources = { base: { type: "vector" } }, loaded = [] } = {}) => ({
  areTilesLoaded: () => allTiles,
  getStyle: () => ({ sources }),
  isSourceLoaded: (sourceId) => loaded.includes(sourceId),
  getContainer: () => ({ getBoundingClientRect: () => ({ width: 256, height: 256 }) }),
  project: ([lng, lat]) => ({ x: (lng + 180) / 360 * 256, y: lat > 0 ? 0 : 256 }),
  style: { tileManagers: Object.fromEntries(Object.keys(sources).map(sourceId => [sourceId, {
    getVisibleCoordinates: () => loaded.includes(sourceId) ? [{ canonical: { x: 0, y: 0, z: 0 }, wrap: 0 }] : [],
    getTile: () => ({ state: "loaded", isRenderable: () => true }),
  }])) },
});

assert.deepEqual(mapTileReadiness({}), {
  tilesLoaded: false,
  tileCoverageRatio: 0,
  tileSourceCount: 0,
});

assert.deepEqual(mapTileReadiness(tileMap({ sources: {}, loaded: [] })), {
  tilesLoaded: false,
  tileCoverageRatio: 0,
  tileSourceCount: 0,
});

assert.deepEqual(mapTileReadiness(tileMap({ loaded: ["base"] })), {
  tilesLoaded: true,
  tileCoverageRatio: 1,
  tileSourceCount: 1,
});

assert.deepEqual(mapTileReadiness(tileMap({
  sources: { base: { type: "vector" }, labels: { type: "raster" } },
  loaded: ["base"],
})), {
  tilesLoaded: false,
  tileCoverageRatio: 0,
  tileSourceCount: 2,
});

assert.deepEqual(mapTileReadiness(tileMap({ allTiles: false, loaded: ["base"] })), {
  tilesLoaded: false,
  tileCoverageRatio: 1,
  tileSourceCount: 1,
});

const readyContainer = {
  dataset: { mapReady: "true", mapReadyAt: "2026-08-12T04:08:26.723Z", mapTileCoverage: "1" },
};
invalidateMapReadiness(readyContainer);
assert.deepEqual(readyContainer.dataset, { mapReady: "false" });

const routePlan = (role, coordinates = [[-122.4, 37.7], [-121.9, 36.6]]) => ({
  validated_legs: {
    type: "FeatureCollection",
    features: [{
      type: "Feature",
      id: "plan_child:leg:1",
      geometry: { type: "LineString", coordinates },
      properties: {
        plan_id: "plan_child",
        route_leg_id: "leg_a_b",
        origin_id: "a",
        destination_id: "b",
        role,
        validation_status: "road_validated",
      },
    }],
  },
});
const previewGeometryHash = await routeGeometrySha256(routePlan("draft_preview"));
const acceptedGeometryHash = await routeGeometrySha256(routePlan("accepted"));
assert.equal(previewGeometryHash, acceptedGeometryHash);
const reverseObjectKeys = (value) => {
  if (Array.isArray(value)) return value.map(reverseObjectKeys);
  if (value === null || typeof value !== "object") return value;
  return Object.fromEntries(Object.keys(value).reverse().map((key) => [key, reverseObjectKeys(value[key])]));
};
assert.equal(
  previewGeometryHash,
  await routeGeometrySha256(reverseObjectKeys(routePlan("draft_preview"))),
  "Persistence may reorder object keys without changing geometry identity",
);
assert.notEqual(
  previewGeometryHash,
  await routeGeometrySha256(routePlan("accepted", [[-121.9, 36.6], [-122.4, 37.7]])),
  "Travel direction and coordinate array order remain part of geometry identity",
);
assert.notEqual(
  previewGeometryHash,
  await routeGeometrySha256(routePlan("accepted", [[-122.4, 37.7], [-121.8, 36.6]])),
);

console.log("map tile readiness contract passed");
