import assert from "node:assert/strict";
import fs from "node:fs";
import { fileURLToPath } from "node:url";

const source = fs.readFileSync(fileURLToPath(new URL("../../src/itinerary_system/product_app/static/js/app.js", import.meta.url)), "utf8");
const code = source.slice(source.indexOf("function selectContext("), source.indexOf("function resetPoiDiscovery("));
const state = {session:{session_id:"same-session", revision:0, selected_stop_id:null}};
const calls = [];
let release;
const held = new Promise(resolve => { release = resolve; });
async function updateSelection(patch) {
  calls.push({revision:state.session.revision, stop:state.session.selected_stop_id, ...patch});
  if (calls.length === 1) await held;
  Object.assign(state.session, patch, {revision:state.session.revision+1});
  return true;
}
const select = new Function("state", "updateSelection", "scheduleInterestRefresh", `let pendingSelection=null; ${code}; return selectContext;`)(state,updateSelection,() => {});
const first = select({selected_stop_id:"bixby"});
const second = select({selected_alternative_id:"recommended"});
await Promise.resolve();
await Promise.resolve();
assert.equal(calls.length, 1);
release();
assert.deepEqual(await Promise.all([first,second]),[true,true]);
assert.equal(calls[1].revision,1);
assert.equal(calls[1].stop,"bixby");
assert.equal(state.session.revision,2);
assert.equal(state.session.selected_alternative_id,"recommended");
const abandoned = select({selected_stop_id:"obsolete"});
state.session = {session_id:"different-session",revision:0};
assert.equal(await abandoned,false);
assert.equal(calls.length,2);
console.log("Selection writes serialize current revisions and preserve the selected stop; switched-session work is discarded.");
