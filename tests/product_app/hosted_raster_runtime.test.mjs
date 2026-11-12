import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import {test} from "node:test";

const source = readFileSync(new URL("../../src/itinerary_system/product_app/static/js/map-controller-v2.js", import.meta.url), "utf8");
const constants = source.slice(source.indexOf("const EXPECTED_MAP_SCHEMA"), source.indexOf("function isFiniteNumber("));
const loader = source.slice(source.indexOf("function localAssetUrl("), source.indexOf("export function retainedOverview("));
const API = "https://api.example.test";
const raster = (base = API) => ({schema_version: "product-map-configuration-v2", status: "ready", provider: "maplibre_raster",
  base_url: base, script_url: `${base}/static/vendor/maplibre-gl.js`, stylesheet_url: `${base}/static/vendor/maplibre-gl.css`,
  style_url: `${base}/public/map-style.json`, range_requests_required: false, protocol_script_url: null});

function runtime({page = "https://owner.github.io/repo/", api = API} = {}) {
  const nodes = new Map(), loads = [], protocols = [];
  const window = api == null ? {} : {PRODUCT_DEPLOYMENT: {api_origin: api}};
  const document = {
    getElementById: id => nodes.get(id),
    createElement(tag) {
      const events = new Map();
      return {tag, dataset: {}, addEventListener: (event, callback) => events.set(event, callback),
        fire: event => events.get(event)(), remove() { nodes.delete(this.id); }};
    },
    head: {append(node) {
      nodes.set(node.id, node);
      loads.push(node.href || node.src);
      queueMicrotask(() => {
        if (node.src?.includes("pmtiles")) window.pmtiles = {Protocol: class {tile() {}}};
        else if (node.src) window.maplibregl = {Map: class {}, addProtocol: (...args) => protocols.push(args)};
        node.fire("load");
      });
    }},
  };
  const load = new Function("window", "document", "location", `${constants}\n${loader}\nreturn loadMapRuntime;`)(window, document, new URL(page));
  return {load, loads, protocols};
}

test("hosted raster runtime loads only the configured server's fixed vendor paths and public style", async () => {
  const h = runtime({page: "https://owner.github.io/repo/?api_origin=https://evil.test#main-workspace"});
  assert.equal(await h.load(raster()), `${API}/public/map-style.json`);
  assert.deepEqual(h.loads, [`${API}/static/vendor/maplibre-gl.css`, `${API}/static/vendor/maplibre-gl.js`]);
  assert.deepEqual(h.protocols, []);
});

test("hosted raster runtime rejects cross-origin, altered paths, credential URLs and public mixed content before loading", async () => {
  const invalid = [
    ...["https://evil.test", `${API}/nested/`, `${API}/?x=1`, `${API}/#fragment`, "https://user:password@api.example.test"]
      .map(base_url => ({...raster(), base_url})),
    ...["script_url", "stylesheet_url", "style_url"].flatMap(field => [
      {...raster(), [field]: `https://evil.test/${field}`},
      {...raster(), [field]: `${raster()[field]}?version=unsafe`},
      {...raster(), [field]: `${raster()[field]}#fragment`},
      {...raster(), [field]: `${API}/nested/../${raster()[field].slice(API.length + 1)}`},
      {...raster(), [field]: `//api.example.test${raster()[field].slice(API.length)}`},
    ]),
    {...raster(), protocol_script_url: `${API}/static/vendor/pmtiles.js`},
    {...raster(), range_requests_required: true},
  ];
  for (const config of invalid) {
    const h = runtime();
    await assert.rejects(h.load(config), /hosted_map_asset_invalid/);
    assert.deepEqual(h.loads, []);
  }
  for (const api of ["http://api.example.test", "http://127.0.0.1:8127"]) {
    const h = runtime({api});
    await assert.rejects(h.load(raster(api)), /hosted_map_asset_invalid/);
    assert.deepEqual(h.loads, []);
  }
});

test("loopback raster and existing local PMTiles runtime retain their original asset flow", async () => {
  const page = "http://127.0.0.1:8127/app";
  const localRaster = runtime({page, api: null});
  assert.equal(await localRaster.load(raster("http://127.0.0.1:8127")), "http://127.0.0.1:8127/public/map-style.json");
  const h = runtime({page, api: null});
  const config = {...raster("http://127.0.0.1:8848"), provider: "maplibre_pmtiles", range_requests_required: true,
    protocol_script_url: "http://127.0.0.1:8848/pmtiles.js"};
  assert.equal(await h.load(config), config.style_url);
  assert.deepEqual(h.loads, [config.stylesheet_url, config.protocol_script_url, config.script_url]);
  assert.equal(h.protocols[0][0], "pmtiles");
});
