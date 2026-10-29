import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

const input = JSON.parse(readFileSync(process.env.INSPECTOR_INPUT, "utf8"));
const source = readFileSync(new URL("../../src/itinerary_system/product_app/static/js/app.js", import.meta.url), "utf8");
const extract = (name, next) => {
  const begin = source.indexOf(`function ${name}(`);
  const end = source.indexOf(`\nfunction ${next}(`, begin);
  assert.ok(begin >= 0 && end > begin);
  return source.slice(begin, end);
};
const plan = input.geography.plans.find(item => item.plan_id === input.plan_id);
const properties = plan.stops.features.find(item => item.properties.stop_id === "griffith_observatory").properties;
const state = { session:{accepted_plan_id:plan.plan_id,draft:[]}, mapInteraction:{selectionKind:"stop"},
  workspace:{draft_capabilities:{parent_targets:["griffith_observatory"]}} };
const helper = new Function(`${extract("escapeHtml", "unavailable")}\n${extract("humanLabel", "stopNameForId")}\n${source.slice(source.indexOf("function unavailable("),source.indexOf("const TYPED_EDIT_OPERATIONS"))}\nreturn {escapeHtml,unavailable,humanLabel};`)();
const render = new Function("state","findSelectedStop","escapeHtml","unavailable","formatMetric","humanLabel",
  "safeExternalLink","sourceFreshnessHtml","stopDraftOperations","draftStateHtml","TYPED_EDIT_VOCABULARIES",
  `${extract("formatDurationConstraint", "stopDraftOperations")}\n${extract("mapInspectorHtml", "renderMap")}\nreturn mapInspectorHtml;`)(
    state, () => ({id:properties.stop_id,name:properties.name,day:properties.day,plan_id:plan.plan_id}),
    helper.escapeHtml, helper.unavailable, String, helper.humanLabel,
    () => "Unavailable", () => "Unavailable · Unavailable", () => [], () => "No draft",
    {duration_modes:["exact","preferred","minimum","maximum","range"]},
  );
const html = render(input.geography, plan);
assert.match(html, /Day 7 · position 1 in day/);
assert.doesNotMatch(html, /position 9/);
assert.match(html, /<dt>Visit duration<\/dt><dd>45 min · plan value/);
assert.match(html, /<dt>Duration rule<\/dt><dd>Not specified/);
assert.match(html, /<dt>Itinerary role<\/dt><dd>Unavailable/);
assert.match(html, /<dt>Place categories<\/dt><dd>Unavailable/);
assert.match(html, /Recorded weather feasibility: feasible/);
assert.match(html, /Recorded contextual risk: 0.2/);
assert.match(html, /recorded plan context, not a live forecast/);
assert.match(html, /repository curated product fixture · confidence 1 · freshness unavailable/i);
assert.equal((html.match(/role="tab" /g) || []).length, 5);
assert.equal((html.match(/role="tabpanel" /g) || []).length, 5);
assert.equal((html.match(/tabindex="0" hidden/g) || []).length, 4);
assert.match(html, /No sourced photos are available/);
assert.match(html, /No verified visitor reports are available/);

// Exercise the actual tab and expansion handlers without rebuilding the map.
const nodes = new Map();
let focused = null;
function node(key, dataset = {}) {
  const element = {dataset, attributes:{}, handlers:{}, hidden:false,
    classList:{toggle(name,value){this[name]=value;}},
    setAttribute(name,value){this.attributes[name]=value;},
    addEventListener(name,callback){this.handlers[name]=callback;},
    focus(){focused=key;}};
  nodes.set(key,element);
  return element;
}
const tabs = ["overview","visit","photos","reviews","sources"].map(key => {
  node(`#destination-panel-${key}`);
  return node(key,{destinationTab:key});
});
node("#destination-content"); node(".destination-details");
const expand = node("#expand-destination-details");
const begin = source.indexOf('  const destinationTabs = $$("[data-destination-tab]");');
const end = source.indexOf('  $("#prepare-replacement-candidates")',begin);
assert.ok(begin > 0 && end > begin);
new Function("state","$","$$",source.slice(begin,end))(state,key=>nodes.get(key),()=>tabs);
tabs[2].handlers.click();
assert.equal(state.destinationDetails.tab,"photos");
assert.equal(tabs[2].attributes["aria-selected"],"true");
assert.equal(nodes.get("#destination-panel-photos").hidden,false);
assert.equal(nodes.get("#destination-panel-visit").hidden,true);
tabs[2].handlers.keydown({key:"End",preventDefault(){}});
assert.equal(state.destinationDetails.tab,"sources"); assert.equal(focused,"sources");
tabs[4].handlers.keydown({key:"ArrowRight",preventDefault(){}});
assert.equal(state.destinationDetails.tab,"overview"); assert.equal(focused,"overview");
expand.handlers.click({currentTarget:expand});
assert.equal(expand.attributes["aria-expanded"],"true");
assert.equal(nodes.get(".destination-details").classList["is-expanded"],true);
assert.match(render(input.geography,plan),/destination-details is-expanded/);
assert.equal(state.destinationDetails.tab,"overview");
const oldHash=plan.content_hash;
plan.content_hash="different plan";
render(input.geography,plan);
assert.equal(state.destinationDetails.expanded,false);
plan.content_hash=oldHash;
state.session.proposal={runtime_execution:{planning_bundle:{plan:{plan_id:plan.plan_id,
  content_hash:plan.content_hash,selected_stops:[{stop_id:properties.stop_id,description:"Recovered saved preview description"}]}}}};
assert.match(render(input.geography,plan),/Recovered saved preview description/);
state.session.proposal.runtime_execution.planning_bundle.plan.content_hash="wrong hash";
assert.doesNotMatch(render(input.geography,plan),/Recovered saved preview description/);
delete state.session.proposal;

const mapSource = readFileSync(new URL("../../src/itinerary_system/product_app/static/js/map-controller-v2.js", import.meta.url), "utf8");
const validatorEnd = mapSource.indexOf("function selectVisiblePlans(");
assert.ok(validatorEnd > 0);
const validateSource = mapSource.slice(mapSource.indexOf("const EXPECTED_GEOGRAPHY_SCHEMA"), validatorEnd);
new Function(validateSource + "\nreturn validatePlan;")()(plan);
const module = await import(new URL("../../src/itinerary_system/product_app/static/js/map-controller-v2.js", import.meta.url));
const withoutMetadata = structuredClone(plan);
withoutMetadata.stops.features.forEach(item => delete item.properties.inspector_metadata);
assert.equal(await module.routeGeometrySha256(plan), await module.routeGeometrySha256(withoutMetadata), "Stop metadata must not alter route geometry identity");
const retained = properties.inspector_metadata;
properties.inspector_metadata = {schema_version:"unknown"};
assert.match(render(input.geography, plan), /<dt>Weather \/ context<\/dt><dd>Unavailable/);
properties.inspector_metadata = retained;
console.log("Actual accepted inspector canonical projection, unknowns, native map validation and route hash passed");
