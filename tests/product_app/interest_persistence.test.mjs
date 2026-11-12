import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import {test} from "node:test";
const app = readFileSync(new URL("../../src/itinerary_system/product_app/static/js/app.js", import.meta.url), "utf8");
const section = (start, end) => app.slice(app.indexOf(start), app.indexOf(end, app.indexOf(start)));
const reset = section("function resetInterestRegions(", "async function fetchInterestRegions(");
test("changing a day invalidates estimates while retaining enabled interest and radius", () => {
  let aborted = 0, saved = 0;
  const state = {interestRegions: {enabled: true, radiusKm: 2, searchArea: "visitor_area", controller: {abort() {aborted++;}}, generation: 1, response: {old: true}}};
  const resetLayer = new Function("state", "saveInterestPreferences", reset + ";return resetInterestRegions;")(state, () => saved++);
  resetLayer();
  assert.equal(state.interestRegions.enabled, true);
  assert.equal(state.interestRegions.status, "idle");
  assert.equal(state.interestRegions.response, null);
  assert.equal(state.interestRegions.radiusKm, 2);
  assert.equal(aborted, 1);
  resetLayer({hide: true});
  assert.equal(state.interestRegions.enabled, false);
  assert.equal(saved, 1);
});
test("day selection completes before the enabled layer refreshes automatically", async () => {
  const body = section("function selectContext(", "function resetPoiDiscovery(");
  let pendingSelection = null, finish, refreshes = 0;
  const state = {session: {session_id: "saved"}};
  const select = new Function("state", "updateSelection", "scheduleInterestRefresh", "let pendingSelection=null;" + body + ";return selectContext;")(
    state, () => new Promise(resolve => {finish = resolve;}), () => refreshes++);
  const waiting = select({selected_day: 2});
  await Promise.resolve(); await Promise.resolve();
  assert.equal(refreshes, 0);
  finish(true);
  assert.equal(await waiting, true);
  assert.equal(refreshes, 1);
});
test("a day change during an old request schedules the new context instead of leaving an error", async () => {
  const body = section("async function fetchInterestRegions(", "async function loadInterestRegions(");
  const state = {session: {session_id: "saved", revision: 1}, interestRegions: {enabled: true, status: "loading"}};
  let finish, refreshes = 0;
  const explore = new Function("state", "loadInterestRegions", "scheduleInterestRefresh", "renderMap",
    "let interestRefreshRunning=false, interestRefreshTimer=null;" + body + ";return fetchInterestRegions;")(
      state, () => new Promise(resolve => {finish = resolve;}), () => {if (state.interestRegions.status === "idle") refreshes++;}, () => {});
  const request = explore();
  state.session.revision++;
  state.interestRegions.status = "idle";
  finish(false);
  assert.equal(await request, false);
  assert.equal(state.interestRegions.status, "idle");
  assert.equal(state.interestRegions.error, null);
  assert.equal(refreshes, 1);
});
