import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const staticRoot = new URL("../../src/itinerary_system/product_app/static/", import.meta.url);
const css = readFileSync(new URL("css/app.css", staticRoot), "utf8");
const app = readFileSync(new URL("js/app.js", staticRoot), "utf8");
const stage = ".map-stage:has(.destination-details.is-expanded)";
const openRouteStage = ".map-stage:has(.destination-details):has(.textual-map-alternative[open])";
function rule(selector) {
  const block = [...css.replace(/\/\*[\s\S]*?\*\//g, "").matchAll(/([^{}]+)\{([^{}]*)\}/g)].find(([, selectors]) =>
    selectors.split(",").some(part => part.trim() === selector));
  assert.ok(block, `Missing bounded-panel layout: ${selector}`);
  return block[2];
}

// Source layout contracts only; rendered geometry remains a browser acceptance gate.
for (const layoutStage of [stage, openRouteStage]) {
  test(`destination panels use separate bounded flow boxes for ${layoutStage}`, () => {
    assert.match(rule(layoutStage), /display: flex; flex-direction: column/);
    assert.match(rule(layoutStage), /min-height: 560px/);
    const route = rule(`${layoutStage} > .map-accessibility-panel`);
    const destination = rule(`${layoutStage} > .destination-details`);
    for (const box of [route, destination]) {
      assert.match(box, /position: static/);
      assert.match(box, /width: 100%/);
      assert.match(box, /min-width: 0/);
    }
    assert.match(route, /order: -1/);
    assert.match(route, /max-height: min\(24dvh, 160px\)/);
    assert.match(route, /min-height: 52px/);
    assert.match(destination, /flex: 1 1 0/);
    assert.match(destination, /height: auto/);
    assert.match(destination, /min-height: 180px/);
  });
}

test("text route and destination content retain independent scrolling and native disclosure", () => {
  assert.match(css, /\.textual-map-alternative \{[^}]*overflow: auto/);
  assert.match(css, /\.destination-content \{[^}]*overflow-y: auto/);
  assert.match(app, /<details class="textual-map-alternative"/);
  assert.match(app, /<summary>Text route description<\/summary>/);
  assert.match(rule(`${stage} > .map-accessibility-panel`), /overflow: auto/);
});

test("details expansion and repeated keyboard tab navigation preserve state without map rendering", () => {
  const nodes = new Map();
  let focused;
  const node = (key, dataset = {}) => {
    const value = { dataset, handlers: {}, attributes: {}, hidden: false,
      classList: { toggle(name, enabled) { this[name] = enabled; } },
      addEventListener(name, handler) { this.handlers[name] = handler; },
      setAttribute(name, text) { this.attributes[name] = text; },
      focus() { focused = key; } };
    nodes.set(key, value);
    return value;
  };
  const tabs = ["overview", "visit", "photos", "reviews", "sources"].map(key => {
    node(`#destination-panel-${key}`);
    return node(key, { destinationTab: key });
  });
  node("#destination-content");
  const details = node(".destination-details");
  const expand = node("#expand-destination-details");
  const state = { destinationDetails: { tab: "visit", expanded: false } };
  const begin = app.indexOf('  const destinationTabs = $$("[data-destination-tab]");');
  const end = app.indexOf('  $("#prepare-replacement-candidates")', begin);
  assert.ok(begin >= 0 && end > begin);
  new Function("state", "$", "$$", app.slice(begin, end))(
    state, selector => nodes.get(selector), () => tabs,
  );
  for (let repeat = 0; repeat < 3; repeat += 1) {
    expand.handlers.click({ currentTarget: expand });
    assert.equal(details.classList["is-expanded"], true);
    assert.equal(expand.attributes["aria-expanded"], "true");
    tabs[1].handlers.keydown({ key: "ArrowRight", preventDefault() {} });
    assert.equal(state.destinationDetails.tab, "photos");
    assert.equal(focused, "photos");
    assert.equal(nodes.get("#destination-panel-photos").hidden, false);
    assert.equal(tabs[2].attributes["aria-selected"], "true");
    assert.equal(tabs[2].tabIndex, 0);
    tabs[2].handlers.keydown({ key: "End", preventDefault() {} });
    assert.equal(state.destinationDetails.tab, "sources");
    tabs[4].handlers.keydown({ key: "Home", preventDefault() {} });
    assert.equal(state.destinationDetails.tab, "overview");
    expand.handlers.click({ currentTarget: expand });
    assert.equal(details.classList["is-expanded"], false);
    assert.equal(expand.attributes["aria-expanded"], "false");
  }
});
