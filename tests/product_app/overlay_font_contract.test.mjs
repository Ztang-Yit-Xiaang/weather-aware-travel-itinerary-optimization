import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { existsSync, readFileSync } from "node:fs";
import test from "node:test";

const root = new URL("../../", import.meta.url);
const source = readFileSync(new URL("src/itinerary_system/product_app/static/js/map-controller-v2.js", root), "utf8");
const font = ["Noto Sans Regular"];

function overlayLayers() {
  const layers = [];
  const map = { addSource() {}, addLayer(layer) { layers.push(layer); } };
  const start = source.indexOf("function addPointLayers(");
  const end = source.indexOf("function exactRouteLegSelection(", start);
  assert.ok(start >= 0 && end > start);
  const addPoints = new Function("roleFilter", `${source.slice(start, end)}; return addPointLayers;`)(() => null);
  const empty = { type: "FeatureCollection", features: [] };
  addPoints(map, empty, empty, empty, empty, 14);
  const offshore = source.indexOf('nextMap.addLayer({id:"v2-offshore-labels"');
  assert.ok(offshore >= 0);
  new Function("nextMap", source.slice(offshore, source.indexOf(";", offshore) + 1))(map);
  return layers;
}

for (const id of ["v2-route-gap-labels", "v2-offshore-labels"]) {
  test(`${id} explicitly uses the bundled font in its actual layer layout`, () => {
    const layer = overlayLayers().find(layer => layer.id === id);
    assert.equal(layer.type, "symbol");
    assert.ok(layer.layout["text-field"]);
    assert.deepEqual(layer.layout["text-font"], font);
  });
}

const assets = new URL("docker/maplibre/assets/", root);
test("overlay font matches the bundled style and hash-verified glyph provenance", {
  skip: !existsSync(new URL("provenance.json", assets)),
}, () => {
  const style = JSON.parse(readFileSync(new URL("styles/protomaps-light.json", assets), "utf8"));
  assert.ok(style.layers.some(layer => JSON.stringify(layer.layout?.["text-font"]) === JSON.stringify(font)));
  const path = `fonts/${font.join(",")}/0-255.pbf`;
  const provenance = JSON.parse(readFileSync(new URL("provenance.json", assets), "utf8"));
  assert.ok(provenance.glyph_ranges.includes(path));
  const asset = provenance.assets.find(asset => asset.path === path);
  assert.equal(asset.license, "OFL-1.1");
  assert.equal(createHash("sha256").update(readFileSync(new URL(path, assets))).digest("hex"), asset.sha256);
});
