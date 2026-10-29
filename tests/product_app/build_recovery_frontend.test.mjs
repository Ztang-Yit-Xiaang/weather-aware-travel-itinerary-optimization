import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

const source = readFileSync(new URL("../../src/itinerary_system/product_app/static/js/app.js", import.meta.url), "utf8");
const start = source.indexOf("function renderBuildRecovery(");
const end = source.indexOf("async function api(", start);
assert.ok(start >= 0 && end > start);
for (const code of ["product_client_build_mismatch", "product_server_build_changed"]) {
  const state = {previewRuntime:{epoch:2},session:{draft:[{id:"retained"}]},token:"retained-in-memory"};
  const events = [];
  const elements = {
    "#app-shell":{hidden:false},
    "#loading-screen":{hidden:true,classList:{add:() => {}},innerHTML:""},
    "#reload-current-build":{addEventListener:(event,fn) => events.push([event,fn]),focus:() => events.push("focus")},
  };
  let closed = 0;
  const recover = new Function("state", "$", "$$", "location", `${source.slice(start,end)};return renderBuildRecovery;`)(
    state, selector => elements[selector], selector => {
      assert.equal(selector,"dialog[open]");
      return [{close:() => closed++},{close:() => closed++}];
    }, {reload:() => events.push("reload")},
  );
  recover(code);
  assert.equal(closed,2);
  assert.equal(state.buildRecovery,code);
  assert.equal(state.previewRuntime.epoch,3);
  assert.equal(elements["#app-shell"].hidden,true);
  assert.equal(elements["#loading-screen"].hidden,false);
  assert.match(elements["#loading-screen"].innerHTML,new RegExp(`data-build-recovery="${code}"`));
  assert.deepEqual(state.session.draft,[{id:"retained"}]);
  assert.equal(state.token,"retained-in-memory");
  assert.equal(events[1],"focus");
  events[0][1]();
  assert.equal(events[2],"reload");
}
console.log("Build recovery closes top-layer dialogs, focuses explicit reload, and preserves session state.");
