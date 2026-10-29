import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import test from "node:test";

const source = readFileSync(new URL("../../src/itinerary_system/product_app/static/js/app.js", import.meta.url), "utf8");
const code = source.slice(source.indexOf("const TYPED_EDIT_OPERATIONS"), source.indexOf("function safeExternalLink"));
const {diffChangedDays, normalizePreviewRepair} = new Function("state", `${code}; return {diffChangedDays, normalizePreviewRepair};`)({});
function repair(diff, count) {
  const proposal = {plan_diff:diff, certificate_id:"certificate", evaluation_status:"PASSED", state:"eligible"};
  const value = {status:"Eligible repair", status_state:"eligible_repair", result:"Edited itinerary",
    changed:{added:[], removed:[], affected_day_count:count, road_change_count:0, lodging_change_count:0},
    unchanged:{days:diff.unchanged_days, day_count:diff.unchanged_days.length},
    permissions:{booked_change_count:0, locked_change_count:0, permission_required:false, message:"No booking change"},
    tradeoffs:{weighted_edit_cost:0, preservation_rate:1, utility_retained:null, weather_risk_delta:null},
    accepted_radius:null, certificate:{id:"certificate", eligible:true, evaluation_status:"PASSED", failure_count:0, failures:[]},
    primary_action:"Review evidence"};
  return {proposal,value};
}
for (const [label, changes, days] of [
  ["duration", {duration_changes:[{day:7}]}, [7]],
  ["time window", {time_window_changes:[{day:6}]}, [6]],
  ["role", {role_changes:[{day:6}]}, [6]],
  ["same-day order and road", {reorder_changes:[{day:4},{day:4}],road_changes:[{day:4}]}, [4]],
  ["day move", {day_moves:[{from_day:2,to_day:5}]}, [2,5]],
  ["multiple days", {duration_changes:[{day:6},{day:7}],role_changes:[{day:6}]}, [6,7]],
  ["unchanged", {}, []],
]) {
  test(`${label}: summary equals distinct canonical changed days`, () => {
    const diff = {...changes, unchanged_days:[1,3]};
    assert.deepEqual(diffChangedDays(diff), days);
    const {proposal,value} = repair(diff, days.length);
    assert.ok(normalizePreviewRepair(value, proposal));
    assert.equal(normalizePreviewRepair({...value,changed:{...value.changed,affected_day_count:days.length+1}}, proposal), null);
    if (days.length) assert.equal(normalizePreviewRepair({...value,changed:{...value.changed,affected_day_count:0}}, proposal), null);
    assert.equal(normalizePreviewRepair({...value,unchanged:{days:[1],day_count:1}}, proposal), null);
  });
}
for (const invalid of [{duration_changes:"not rows"},{role_changes:[null]},
  {day_moves:[{from_day:2,to_day:true}]},{reorder_changes:[{day:0}]}]) {
  test(`malformed canonical day evidence is rejected: ${JSON.stringify(invalid)}`, () => {
    assert.equal(diffChangedDays(invalid), null);
  });
}
