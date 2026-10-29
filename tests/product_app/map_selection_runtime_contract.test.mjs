import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

const source = readFileSync(new URL("../../src/itinerary_system/product_app/static/js/app.js", import.meta.url), "utf8");
const start = source.indexOf("function findSelectedStop()");
const end = source.indexOf("\nfunction metricCell", start);
assert.ok(start >= 0 && end > start);
const stop = (id, name, day) => ({ properties: { stop_id: id, name, day } });
const parent = { plan_id: "parent", stops: { features: [stop("bixby", "Bixby", 5), stop("stanford", "Old Stanford", 2)] } };
const child = { plan_id: "child", stops: { features: [stop("pfeiffer", "Pfeiffer", 5), stop("stanford", "Current Stanford", 3)] } };
const state = {
  session: { selected_stop_id: "pfeiffer" },
  mapInteraction: { selectedStopPlanId: null },
  workspace: { timeline: [{ stops: [{ id: "pfeiffer", name: "Stale timeline name" }] }] },
};
const find = new Function("state", "activeGeography", "activeRenderPlan",
  source.slice(start, end) + "\nreturn findSelectedStop;")(
    state, () => ({ plans: [parent, child] }), () => child,
  );
assert.deepEqual(find(), { id: "pfeiffer", name: "Pfeiffer", day: 5, plan_id: "child" });
state.mapInteraction.selectedStopPlanId = "parent";
assert.equal(find().plan_id, "child", "A stale parent plan selection cannot override the selected child stop");
state.session.selected_stop_id = "stanford";
state.mapInteraction.selectedStopPlanId = null;
assert.equal(find().day, 3, "Current child day and details take precedence over the parent timeline");
state.session.selected_stop_id = "bixby";
assert.equal(find(), null, "A removed stop is not implicitly recovered from the parent");
state.mapInteraction.selectedStopPlanId = "parent";
assert.equal(find().plan_id, "parent", "An explicit original-plan inspection remains possible");
state.session.selected_stop_id = null;
assert.equal(find(), null);
console.log("active-plan stop selection contract passed");

