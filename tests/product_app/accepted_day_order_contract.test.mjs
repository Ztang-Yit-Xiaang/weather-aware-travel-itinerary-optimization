import assert from "node:assert/strict";
import fs from "node:fs";
import test from "node:test";

const source=fs.readFileSync(new URL("../../src/itinerary_system/product_app/static/js/app.js",import.meta.url),"utf8");
const code=source.slice(source.indexOf("function acceptedStopOrderByDay()"),source.indexOf("function normalizePreviewGeography("));
function order(count, change=()=>{}) {
  const features=Array.from({length:count},(_,i)=>({properties:{day:i+1,stop_id:`stop-${i+1}`,sequence_index:i}}));
  change(features);
  const state={session:{accepted_plan_id:"accepted"},workspace:{
    timeline:Array.from({length:7},(_,i)=>({day:i+1})),
    geography:{plans:[{plan_id:"accepted",stops:{features}}]},
  }};
  return new Function("state",code+"\nreturn acceptedStopOrderByDay();")(state);
}
for(const count of [5,7,10,14]) test(`accepted ${count}-day geography defines the editable days`,()=>{
  const result=order(count);
  assert.equal(result.size,count);
  assert.deepEqual([...result.keys()],Array.from({length:count},(_,i)=>i+1));
  assert.deepEqual(result.get(count),[`stop-${count}`]);
});
test("duplicate stop identities remain rejected",()=>{
  assert.equal(order(7,rows=>rows[6].properties.stop_id=rows[0].properties.stop_id),null);
});
for(const day of [0,-1,22,"8",null]) test(`invalid accepted day ${JSON.stringify(day)} is rejected`,()=>{
  assert.equal(order(7,rows=>rows[6].properties.day=day),null);
});
