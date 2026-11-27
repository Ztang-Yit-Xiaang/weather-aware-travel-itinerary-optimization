import assert from "node:assert/strict";
import {test} from "node:test";
import {createWorkspaceStorage, restoreInterestPreferences} from "../../src/itinerary_system/product_app/static/js/workspace-storage.js";

const memory = () => {
  const data = new Map();
  return {getItem: key => data.get(key) ?? null, setItem: (key, value) => data.set(key, value), removeItem: key => data.delete(key)};
};
const storage = (persistent, tab = memory()) => createWorkspaceStorage({persistent, tab, sessionKey: "session", scopedKeys: ["pending", "input"]});
test("legacy workspace and journal survive closing all tabs and reopening", () => {
  const persistent = memory(), tab = memory();
  tab.setItem("session", JSON.stringify({session_id: "first", mutation_token: "token"}));
  tab.setItem("pending", "protected-action");
  const old = storage(persistent, tab);
  assert.ok(old.getItem("session"));
  assert.equal(old.getItem("pending"), "protected-action");
  old.setItem("input", "Unsent 海岸線");
  const reopened = storage(persistent);
  assert.equal(JSON.parse(reopened.getItem("session")).session_id, "first");
  assert.equal(reopened.getItem("pending"), "protected-action");
  assert.equal(reopened.getItem("input"), "Unsent 海岸線");
});
test("another workspace cannot inherit or remove the first workspace's unfinished action", () => {
  const persistent = memory(), tab = memory();
  const first = storage(persistent, tab);
  first.setItem("session", JSON.stringify({session_id: "first"}));
  first.setItem("pending", "first-action");
  const second = storage(persistent);
  second.setItem("session", JSON.stringify({session_id: "second"}));
  assert.equal(second.getItem("pending"), null);
  assert.equal(first.getItem("pending"), "first-action");
  first.removeItem("session");
  assert.equal(JSON.parse(persistent.getItem("session")).session_id, "second");
});
test("interest settings survive reopening, including explicit hide", () => {
  const persistent = memory();
  const first = storage(persistent);
  const preferences = {enabled: false, radiusKm: 2, naturePreference: 1, massLevel: .95, searchArea: "itinerary_pin"};
  first.setItem("interest:trip", JSON.stringify(preferences));
  assert.deepEqual(restoreInterestPreferences(storage(persistent), "interest:trip"), preferences);
  assert.equal(restoreInterestPreferences(storage(persistent), "interest:another").enabled, true);
  first.setItem("interest:trip", '{"enabled":true,"radiusKm":999}');
  assert.equal(restoreInterestPreferences(storage(persistent), "interest:trip").radiusKm, 5);
});
test("editing fails closed when neither browser store can save a journal", () => {
  const broken = {getItem() {throw new Error("denied");}, setItem() {throw new Error("denied");}};
  assert.throws(() => storage(broken, broken).setItem("pending", "action"), /storage is unavailable/);
});


test("a freshly reopened tab pins its own journal owner before another workspace opens", () => {
  const persistent = memory();
  storage(persistent).setItem("session", JSON.stringify({session_id: "first"}));
  const reopened = storage(persistent);
  assert.equal(JSON.parse(reopened.getItem("session")).session_id,"first");
  storage(persistent).setItem("session", JSON.stringify({session_id: "second"}));
  reopened.setItem("pending","first-action");
  assert.equal(persistent.getItem("pending:first"),"first-action");
  assert.equal(persistent.getItem("pending:second"),null);
});
