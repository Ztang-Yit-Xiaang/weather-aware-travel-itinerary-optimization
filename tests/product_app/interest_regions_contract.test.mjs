import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import {test} from "node:test";
import {addInterestAreaLayers, addInterestPointLayers, interestPlacesInRegion, normalizeInterestResponse, REQUIRED_INTEREST_COLOR} from "../../src/itinerary_system/product_app/static/js/interest-regions.js";

const empty = () => ({type:"FeatureCollection",features:[]});
function example() {
  return {schema_version:"product-interest-regions-v1",session_id:"session_test",session_revision:2,
    accepted_plan_id:"plan_test",accepted_plan_content_hash:"abc",nature_preference:0,
    exploration_plan_id:"plan_test",exploration_plan_content_hash:"abc",plan_scope:"accepted",
    context:{day:1,route_leg_id:"leg_test",replacement_target_id:null},
    search_context:{stop_id:"stop1",name:"Current stop",coordinate:{longitude:-119.7,latitude:34.4},radius_km:5,distance_kind:"straight_line"},
    candidate_universe:{count:1,truncated:false,sha256:"a".repeat(64)},
    model:{calibration_status:"not_calibrated",evidence_mode:"prior_only"},
    poi_predictions:[{place_id:"p1",name:"Optional place",display_coordinate:{longitude:-119.7,latitude:34.4},
      interest_probability_mean:.3,interest_probability_interval:[.1,.8],interval_level:.9,distance_from_focus_km:0}],
    required_points:[{place_id:"required",name:"Required",coordinate:[-119.69,34.42],constraint_ids:["c1"],conflict_status:"required_stop_missing"}],
    surface:{status:"unavailable",density_geojson:empty(),regions_geojson:empty()},
  };
}
const expected={sessionId:"session_test",revision:2,planId:"plan_test",planHash:"abc",explorationPlanId:"plan_test",explorationPlanHash:"abc",day:1,routeLegId:"leg_test",replacementTargetId:null,naturePreference:0};

test("nearby responses are bound to the focus and radius and reject distant or misreported points",()=>{
  const nearby={...expected,focusStopId:"stop1",radiusKm:5};
  assert.equal(normalizeInterestResponse(example(),nearby).search_context.name,"Current stop");
  for(const change of [{focusStopId:"other"},{radiusKm:2}]) {
    assert.throws(()=>normalizeInterestResponse(example(),{...nearby,...change}),/stale/);
  }
  const far=example();far.poi_predictions[0].display_coordinate.longitude=-122.5;
  assert.throws(()=>normalizeInterestResponse(far,nearby),/distance/);
  const misleading=example();misleading.poi_predictions[0].distance_from_focus_km=1;
  assert.throws(()=>normalizeInterestResponse(misleading,nearby),/distance/);
});

test("the selected stop survives automatic route selection and the chosen radius is sent",async()=>{
  const app=readFileSync(new URL("../../src/itinerary_system/product_app/static/js/app.js",import.meta.url),"utf8");
  const body=app.slice(app.indexOf("async function fetchInterestRegions("),app.indexOf("function prepareReplacementDiscovery("));
  const state={session:{session_id:"session_test",revision:2,accepted_plan_id:"plan_test",selected_day:1,selected_stop_id:"stop2",selected_segment_id:null},
    mapInteraction:{mode:"select",selectedRouteLeg:null},interestRegions:{naturePreference:0,radiusKm:2,generation:0},poiDiscovery:{maxDetourMinutes:60,replacementTargetId:null}};
  const leg={plan_id:"plan_test",route_leg_id:"leg_test",day:1,destination_id:"stop1"};
  const plan={plan_id:"plan_test",content_hash:"abc",validated_legs:{features:[{properties:leg}]},
    stops:{features:[{properties:{stop_id:"stop1",day:1}},{properties:{stop_id:"stop2",day:1}}]}};
  let request;
  const payload=example();Object.assign(payload.search_context,{stop_id:"stop2",radius_km:2});
  const explore=new Function("state","pendingSelection","activeGeography","acceptedPlan","toast","renderMap","api","normalizeInterestResponse","activeRenderPlan","selectRouteLeg",("let interestRefreshRunning=false, interestRefreshTimer=null; const scheduleInterestRefresh=()=>{}, saveInterestPreferences=()=>{};"+body)+";return fetchInterestRegions;")(
    state,null,()=>({}),()=>plan,message=>assert.fail(message),()=>{},async url=>{request=url;return payload;},normalizeInterestResponse,()=>plan,
    async value=>{state.session.selected_stop_id=null;state.session.selected_segment_id=value.route_leg_id;state.mapInteraction.selectedRouteLeg=value;return true;});
  assert.equal(await explore(),true);
  assert.match(request,/focus_stop_id=stop2/);assert.match(request,/radius_km=2/);
  assert.equal(state.interestRegions.focusStopId,"stop2");
});

test("rejects stale revision, accepted parent, route, day, and preference",()=>{
  for(const changed of [{revision:3},{planId:"other"},{planHash:"different"},{explorationPlanId:"other"},{explorationPlanHash:"different"},{day:2},{routeLegId:"other"},{naturePreference:1}]) {
    assert.throws(()=>normalizeInterestResponse(example(),{...expected,...changed}),/stale/);
  }
  assert.equal(normalizeInterestResponse(example(),expected).model.evidence_mode,"prior_only");
});

test("rejects fabricated calibration, truncated universes and invalid numeric probabilities",()=>{
  for(const mutate of [p=>p.model.calibration_status="calibrated",p=>p.candidate_universe.truncated=true,
    p=>p.poi_predictions[0].interest_probability_mean=NaN,p=>p.poi_predictions[0].interest_probability_interval=[.8,.1],
    p=>p.required_points[0].coordinate=[200,0]]) {
    const value=example();mutate(value);assert.throws(()=>normalizeInterestResponse(value,expected));
  }
});

test("required color remains fixed and selection waits for map event dispatch",async()=>{
  const sources=new Map(),layers=[],events=[];
  const map={addSource:(id,value)=>sources.set(id,value),addLayer:layer=>layers.push(layer),on:(...args)=>events.push(args)};
  const value=example(), selected=[];
  addInterestAreaLayers(map,value,.8);
  assert.equal(layers.length,0);
  addInterestPointLayers(map,value,null,id=>selected.push(id));
  assert.equal(layers.at(-1).id,"interest-required");
  assert.equal(layers.at(-1).paint["circle-color"],REQUIRED_INTEREST_COLOR);
  assert.equal(sources.get("interest-required").data.features[0].properties.conflict_status,"required_stop_missing");
  events[0][2]({features:[{properties:{place_id:"p1"}}]});assert.deepEqual(selected,[]);
  await Promise.resolve();assert.deepEqual(selected,["p1"]);
  value.poi_predictions[0].interest_probability_mean=.99;
  addInterestPointLayers(map,value,"p1",()=>{});
  assert.equal(layers.at(-1).paint["circle-color"],REQUIRED_INTEREST_COLOR);
  map.queryRenderedFeatures=()=>[{properties:{place_id:"required"}}];
  events[0][2]({features:[{properties:{place_id:"p1"}}]});
  events[1][2]({features:[{properties:{place_id:"required"}}]});
  await Promise.resolve();assert.deepEqual(selected,["p1","required"]);
});

test("application discards superseded requests and keeps previews on their parent",()=>{
  const app=readFileSync(new URL("../../src/itinerary_system/product_app/static/js/app.js",import.meta.url),"utf8");
  assert.match(app,/generation !== view\.generation/);
  assert.match(app,/view\.controller\?\.abort\(\)/);
  const fetchBody=app.slice(app.indexOf("async function fetchInterestRegions"),app.indexOf("function prepareReplacementDiscovery"));
  assert.doesNotMatch(fetchBody,/acceptDraft|acceptProposal|addDraft\(/);
  assert.match(fetchBody,/normalizeInterestResponse\(raw, expected\)/);
});

test("place browsing respects the chosen polygon, holes, separate islands and boundary points",()=>{
  const payload=example();
  const point=(name,x,y)=>({...payload.poi_predictions[0],place_id:name,name,display_coordinate:{longitude:x,latitude:y}});
  payload.poi_predictions=[point("inside",1,1),point("hole",5,5),point("outside",11,1),point("edge",0,3),point("island",21,21)];
  payload.surface={status:"ready",regions_geojson:{type:"FeatureCollection",features:[{
    properties:{mass_level:.8},geometry:{type:"MultiPolygon",coordinates:[
      [[[0,0],[10,0],[10,10],[0,10],[0,0]],[[4,4],[6,4],[6,6],[4,6],[4,4]]],
      [[[20,20],[22,20],[22,22],[20,22],[20,20]]],
    ]},
  }]}};
  assert.deepEqual(interestPlacesInRegion(payload,.8).map(p=>p.name),["inside","edge","island"]);
  assert.equal(interestPlacesInRegion(payload,.5),payload.poi_predictions,"Unavailable boundaries keep all points inspectable");
});

test("place list exposes locations without map picking and supports search and full-neighborhood scope",()=>{
  const app=readFileSync(new URL("../../src/itinerary_system/product_app/static/js/app.js",import.meta.url),"utf8");
  const body=app.slice(app.indexOf("function interestPlaceListHtml("),app.indexOf("function interestExplorerHtml("));
  const state={interestRegions:{placeSearch:"",placeScope:"region",massLevel:.8,selectedId:"p1"}};
  const render=new Function("state","interestPlacesInRegion","escapeHtml","humanLabel",body+";return interestPlaceListHtml;")(
    state,p=>p.poi_predictions.slice(0,1),s=>String(s).replaceAll("<","&lt;"),s=>s);
  const payload=example();payload.surface.status="ready";
  payload.poi_predictions[0].name="Hidden <beach>";
  payload.poi_predictions[0].place_categories=["outdoor"];
  payload.poi_predictions.push({...payload.poi_predictions[0],place_id:"p2",name:"Other museum",interest_probability_mean:.7});
  let html=render(payload);
  assert.match(html,/Hidden &lt;beach>/);assert.doesNotMatch(html,/Other museum/);
  assert.match(html,/aria-pressed="true"/);assert.match(html,/90% model interval: 10%–80%/);
  state.interestRegions.placeScope="all";html=render(payload);
  assert.ok(html.indexOf("Other museum")<html.indexOf("Hidden &lt;beach>"));
  assert.match(html,/Outside selected boundary/);
  state.interestRegions.placeSearch="museum";html=render(payload);
  assert.match(html,/Other museum/);assert.doesNotMatch(html,/Hidden &lt;beach>/);
  state.interestRegions.placeSearch="absent";assert.match(render(payload),/No catalog places match/);
  payload.poi_predictions=[];
  assert.match(render(payload),/No nearby places are recorded for this route yet/);
});

test("map labels preserve place names and keep a selected point readable",()=>{
  const layers=[],sources=new Map();
  const map={addSource:(id,value)=>sources.set(id,value),addLayer:l=>layers.push(l),on:()=>{},
    getStyle:()=>({layers:[{type:"symbol",layout:{"text-font":["Local font"]}}]})};
  addInterestPointLayers(map,example(),"p1",()=>{});
  assert.equal(sources.get("interest-pois").data.features[0].properties.name,"Optional place");
  assert.equal(layers.find(l=>l.id==="interest-selected-label").layout["text-allow-overlap"],true);
  assert.equal(layers.find(l=>l.id==="interest-required").paint["circle-color"],REQUIRED_INTEREST_COLOR);
});

test("interest exploration works in Select mode without staging or acceptance",async()=>{
  const app=readFileSync(new URL("../../src/itinerary_system/product_app/static/js/app.js",import.meta.url),"utf8");
  const body=app.slice(app.indexOf("async function fetchInterestRegions("),app.indexOf("function prepareReplacementDiscovery("));
  const value=example(), requests=[];
  const state={session:{session_id:"session_test",revision:2,accepted_plan_id:"plan_test",selected_day:1,selected_segment_id:"leg_test"},
    mapInteraction:{mode:"select",selectedRouteLeg:{plan_id:"plan_test",route_leg_id:"leg_test",day:1}},
    interestRegions:{naturePreference:0,generation:0,controller:null},poiDiscovery:{replacementTargetId:null,maxDetourMinutes:60}};
  const plan={plan_id:"plan_test",content_hash:"abc",validated_legs:{features:[{properties:{route_leg_id:"leg_test",day:1}}]}};
  const explore=new Function("state","pendingSelection","activeGeography","acceptedPlan","toast","renderMap","api","normalizeInterestResponse","activeRenderPlan",
    ("let interestRefreshRunning=false, interestRefreshTimer=null; const scheduleInterestRefresh=()=>{}, saveInterestPreferences=()=>{};"+body)+";return fetchInterestRegions;")(state,null,()=>({}),()=>({plan_id:"plan_test",content_hash:"abc"}),
      message=>assert.fail(message),()=>{},async url=>{requests.push(url);return value;},normalizeInterestResponse,()=>plan);
  assert.equal(await explore(),true);
  assert.equal(requests.length,1);
  assert.match(requests[0],/\/map\/interest-regions\?/);
  assert.equal(state.mapInteraction.mode,"select");
  assert.equal(state.session.revision,2);
  assert.equal(state.interestRegions.response,value);
});

test("map summary makes loading, failure and empty results visible without scrolling",()=>{
  const app=readFileSync(new URL("../../src/itinerary_system/product_app/static/js/app.js",import.meta.url),"utf8");
  const body=app.slice(app.indexOf("function interestMapSummaryHtml("),app.indexOf("function interestExplorerHtml("));
  const state={interestRegions:{enabled:false,status:"idle",error:null,selectedId:null}};
  const render=new Function("state","escapeHtml",body+";return interestMapSummaryHtml;")(
    state,s=>String(s).replaceAll("<","&lt;"));
  assert.equal(render(null),"");
  Object.assign(state.interestRegions,{enabled:true,status:"loading"});
  assert.match(render(null),/Loading interest region/);
  assert.match(render(null),/role="status"/);
  Object.assign(state.interestRegions,{status:"failed",error:"poi_route_matrix_invalid"});
  const failed=render(null);
  assert.match(failed,/Interest region unavailable/);
  assert.match(failed,/route evidence could not be read/);
  assert.match(failed,/role="alert"/);
  assert.match(failed,/id="retry-interest-regions"/);
  assert.match(app,/#retry-interest-regions"\)\?\.addEventListener\("click", fetchInterestRegions\)/);
  Object.assign(state.interestRegions,{status:"ready",error:null});
  const empty=render({...example(),poi_predictions:[],required_points:[]});
  assert.match(empty,/0 nearby places/);
  assert.match(empty,/No nearby places are recorded within this radius/);
  assert.doesNotMatch(empty,/Click a blue point/);
  state.interestRegions.selectedId="p1";
  const ready=render(example());
  assert.match(ready,/Optional place/);
  assert.match(ready,/90% interval 10%–80%/);
  const unavailable=example();unavailable.surface.status="unavailable";
  assert.match(render(unavailable),/Continuous shading is unavailable for this route/);
  state.interestRegions.error="<untrusted>";
  assert.match(render(null),/&lt;untrusted>/);
});

test("must-go details are keyboard reachable and preserve the model probability",()=>{
  const app=readFileSync(new URL("../../src/itinerary_system/product_app/static/js/app.js",import.meta.url),"utf8");
  const body=app.slice(app.indexOf("function interestExplorerHtml("),app.indexOf("function humanLabel("));
  const response=example();
  response.required_points[0].interest_probability_mean=.2;
  response.candidate_universe.route_checked_candidate_count=0;
  const state={mapInteraction:{mode:"select"},interestRegions:{enabled:true,selectedId:"required",massLevel:.8}};
  const render=new Function("state","currentInterestResponse","escapeHtml","humanLabel","interestPlaceListHtml","safeExternalLink",body+";return interestExplorerHtml;")(
    state,()=>response,s=>String(s),s=>s,()=>"",()=>"");
  const html=render();
  assert.match(html,/<details open>/);
  assert.match(html,/<button type="button" data-interest-place="required" aria-pressed="true">Must-go: Required<\/button>/);
  assert.match(html,/Conflict: required_stop_missing/);
  assert.match(html,/Estimated interest: 20%/);
  assert.match(html,/not a probability of 100%/);
  assert.match(app,/\$\("\.interest-region-panel"\)\?\.addEventListener\("click"/);
  state.interestRegions.selectedId="p1";
  assert.match(render(),/Choose Edit in the map toolbar/);
  assert.match(render(),/Custom stops are draft only/);
});

test("Interest region selects the displayed 14-day preview route without accepting it",async()=>{
  const app=readFileSync(new URL("../../src/itinerary_system/product_app/static/js/app.js",import.meta.url),"utf8");
  const body=app.slice(app.indexOf("async function fetchInterestRegions("),app.indexOf("function prepareReplacementDiscovery("));
  const value=example(), requests=[], selections=[];
  const plan={plan_id:"preview14",content_hash:"preview-hash",validated_legs:{features:[{properties:{route_leg_id:"preview-leg14",day:14}}]}};
  const state={session:{session_id:"session_test",revision:2,accepted_plan_id:"plan_test",selected_day:14,selected_segment_id:null},
    mapInteraction:{mode:"edit",selectedRouteLeg:null},interestRegions:{naturePreference:0,generation:0},poiDiscovery:{replacementTargetId:null,maxDetourMinutes:60}};
  Object.assign(value,{session_revision:3,exploration_plan_id:plan.plan_id,exploration_plan_content_hash:plan.content_hash,plan_scope:"evaluated_preview"});
  Object.assign(value.context,{day:14,route_leg_id:"preview-leg14"});
  const explore=new Function("state","pendingSelection","activeGeography","acceptedPlan","activeRenderPlan","selectRouteLeg","toast","renderMap","api","normalizeInterestResponse",("let interestRefreshRunning=false, interestRefreshTimer=null; const scheduleInterestRefresh=()=>{}, saveInterestPreferences=()=>{};"+body)+";return fetchInterestRegions;")(
    state,null,()=>({}),()=>({plan_id:"plan_test",content_hash:"abc"}),()=>plan,
    async route=>{selections.push(route);state.mapInteraction.selectedRouteLeg=route;state.session.selected_segment_id=route.route_leg_id;state.session.revision++;return true;},
    message=>assert.fail(message),()=>{},async url=>{requests.push(url);return value;},normalizeInterestResponse);
  assert.equal(await explore(),true);
  assert.equal(selections.length,1);
  assert.equal(selections[0].plan_id,"preview14");
  assert.match(requests[0],/plan_id=preview14/);
  assert.match(requests[0],/day=14/);
  assert.equal(state.session.accepted_plan_id,"plan_test");
  assert.equal(state.interestRegions.response.plan_scope,"evaluated_preview");
});

test("the displayed preview route card renders an enabled interest action",()=>{
  const app=readFileSync(new URL("../../src/itinerary_system/product_app/static/js/app.js",import.meta.url),"utf8");
  const body=app.slice(app.indexOf("function mapInspectorHtml("),app.indexOf("function renderMap("));
  const route={plan_id:"preview14",route_leg_id:"leg14",day:14};
  const scope={state:{mapInteraction:{selectionKind:"route_leg",selectedRouteLeg:route,mode:"select"},interestRegions:{enabled:false},poiDiscovery:{}},
    findRouteLeg:()=>({properties:route}),routeDurationLabel:()=>"1 min",routeDistanceLabel:()=>"1 mi",
    routeNamePair:()=>({origin:"A",destination:"B"}),routeDraftOperations:()=>[],typedEditCapability:()=>null,
    selectedAcceptedRouteFeature:()=>null,escapeHtml:String,unavailable:v=>v??"Unavailable",sourceFreshnessHtml:()=>"source",draftStateHtml:()=>""};
  const render=new Function(...Object.keys(scope),body+";return mapInspectorHtml;")(...Object.values(scope));
  const html=render({plans:[{plan_id:"preview14"}]},{plan_id:"preview14"});
  assert.match(html,/<button type="button" id="explore-interest-regions" >Explore interest region<\/button>/);
  assert.match(html,/id="find-route-candidates" disabled/);
});


test("reopening uses the saved route without a selection write or revision loop",async()=>{
  const app=readFileSync(new URL("../../src/itinerary_system/product_app/static/js/app.js",import.meta.url),"utf8");
  const body=app.slice(app.indexOf("async function fetchInterestRegions("),app.indexOf("function prepareReplacementDiscovery("));
  const state={session:{session_id:"session_test",revision:2,accepted_plan_id:"plan_test",selected_day:1,selected_segment_id:"leg_test"},
    mapInteraction:{selectedRouteLeg:null},interestRegions:{naturePreference:0,generation:0},poiDiscovery:{replacementTargetId:null,maxDetourMinutes:60}};
  const plan={plan_id:"plan_test",content_hash:"abc",validated_legs:{features:[{properties:{route_leg_id:"leg_test",day:1}}]}};
  const explore=new Function("state","pendingSelection","activeGeography","acceptedPlan","activeRenderPlan","selectRouteLeg","toast","renderMap","api","normalizeInterestResponse",
    "let interestRefreshRunning=false, interestRefreshTimer=null; const scheduleInterestRefresh=()=>{}, saveInterestPreferences=()=>{};"+body+";return fetchInterestRegions;")(
      state,null,()=>({}),()=>plan,()=>plan,()=>assert.fail("Reopening must not write selection"),message=>assert.fail(message),()=>{},async()=>example(),normalizeInterestResponse);
  assert.equal(await explore(),true);
  assert.equal(state.session.revision,2);
  assert.equal(state.mapInteraction.selectedRouteLeg.route_leg_id,"leg_test");
});


test("place source shows live, cached and offline freshness without pretending access is verified",()=>{
  const app=readFileSync(new URL("../../src/itinerary_system/product_app/static/js/app.js",import.meta.url),"utf8");
  const body=app.slice(app.indexOf("function interestPlaceDataHtml("),app.indexOf("function interestMapSummaryHtml("));
  const render=new Function("escapeHtml",body+";return interestPlaceDataHtml;")(String);
  assert.match(render({status:"live",fetched_at:"2026-10-29T12:00:00Z"}),/Online places.*fetched/);
  assert.match(render({status:"fresh_cache"}),/Saved online places/);
  assert.match(render({status:"stale_cache",error_code:"nearby_api_unavailable"}),/Older saved online places.*showing saved places/);
  assert.match(render({status:"bundled_fallback",error_code:"nearby_api_rate_limited"}),/Saved place catalog.*unavailable/);
  assert.match(render({status:"live",error_code:"nearby_cache_write_failed"}),/could not be saved/);
  assert.match(app,/Refresh online places/);
  assert.match(app,/query.set\("refresh_places", "true"\)/);
});
