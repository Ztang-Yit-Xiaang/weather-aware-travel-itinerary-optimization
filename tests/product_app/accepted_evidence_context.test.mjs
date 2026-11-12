import assert from "node:assert/strict";
import fs from "node:fs";
import { fileURLToPath } from "node:url";

const source = fs.readFileSync(fileURLToPath(new URL("../../src/itinerary_system/product_app/static/js/app.js", import.meta.url)), "utf8");
const actual = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
const code = source.slice(source.indexOf("function evidencePresentationContext("), source.indexOf("function renderCompare("));
function resolve(input, preview=null) {
  return new Function("state", "evaluatedProposalCompareContext", "activeGeography", `${code}; return evidencePresentationContext();`)(
    input, () => preview, () => input.workspace.geography);
}
actual.session.selected_alternative_id = null;
const selected = resolve(actual);
assert.equal(selected.scope, "accepted");
assert.equal(selected.session.selected_alternative_id, actual.session.accepted_plan_id);
assert.equal(selected.workspace.alternatives[0].plan_content_hash, actual.workspace.accepted_plan_pointer.accepted_plan_content_hash);
assert.equal(selected.workspace.alternatives[0].method_label, "Accepted evaluated itinerary");
const id = actual.session.accepted_plan_id;
for (const mutate of [
  input => { input.workspace.accepted_plan_pointer.accepted_plan_content_hash = "foreign"; },
  input => { input.workspace.alternative_evidence[id].certificate.plan_id = "foreign"; },
  input => { input.workspace.alternative_evidence[id].certificate.plan_content_hash = "foreign"; },
  input => { input.workspace.alternative_evidence[id].diff.child_plan_id = "foreign"; },
  input => { input.workspace.alternative_evidence[id].diff.diff_id = "foreign"; },
  input => { input.workspace.alternative_evidence[id].parent_plan_id = "foreign"; },
  input => { input.workspace.geography.plans.push(input.workspace.geography.plans.find(plan => plan.plan_id === id)); },
  input => { input.session.selected_alternative_id = "explicit-registered-choice"; },
]) {
  const crossed = structuredClone(actual);
  mutate(crossed);
  assert.equal(resolve(crossed), null);
}
const livePreview = {scope:"preview", marker:"exact currently rendered child context"};
assert.equal(resolve(actual,livePreview),livePreview);
console.log("Actual accepted evidence resolves; crossed identities reject; explicit selection and current preview retain precedence.");

// Timeline status uses the exact current diff, including non-geographic edits.
const timelineCode = source.slice(source.indexOf("function timelineForPlan("), source.indexOf("function renderTrip("));
const daysCode = source.slice(source.indexOf("function diffChangedDays("), source.indexOf("function normalizePreviewRepair("));
const acceptedPlan = actual.workspace.geography.plans.find(plan => plan.plan_id === actual.session.accepted_plan_id);
const child = structuredClone(acceptedPlan);
child.plan_id = "typed-preview";
child.content_hash = "typed-preview-hash";
const input = structuredClone(actual);
input.session.proposal = {state:"eligible",child_plan_id:child.plan_id,child_plan_content_hash:child.content_hash,
  plan_diff:{duration_changes:[{day:7,stop_id:"griffith_observatory"}]}};
const timeline = new Function("state", "acceptedPlan", `${daysCode};${timelineCode};return timelineForPlan;`)(input,() => acceptedPlan);
const typed = timeline(child);
assert.deepEqual(typed.filter(day => day.states.includes("changed")).map(day => day.day),[7]);
assert.equal(typed.some(day => day.states.includes("affected")),false);
input.session.proposal.child_plan_content_hash = "foreign";
assert.equal(timeline(child).some(day => day.states.includes("changed")),false);
input.session.proposal = null;
assert.equal(timeline(acceptedPlan).some(day => day.states.includes("affected")),false);
assert.ok(source.includes('hasRenderedRepair ? "Original repair context"'));
assert.ok(source.includes('<span>Original issue days</span>'));
assert.ok(source.includes('<details class="evaluated-edit-evidence">'));
assert.ok(!source.includes('<details class="evaluated-edit-evidence" open>'));

const repairCode = source.slice(source.indexOf("function renderRepair("), source.indexOf("function renderContextPanel("));
const panel = {innerHTML:"",addEventListener() {}};
const proposal = {schema_version:"draft-preview-v1",state:"eligible",child_plan_id:child.plan_id,
  child_plan_content_hash:child.content_hash,repair:{result:"Exact preview",changed:{affected_day_count:1},
    permissions:{booked_change_count:0,locked_change_count:0},tradeoffs:{weather_risk_delta:null}}};
let displayed = child;
const render = new Function("state","acceptedPlan","activeRenderPlan","planPresentationEvidence",
  "previewCanBeAccepted","escapeHtml","formatMetric","evaluatedEditEvidenceHtml","$",
  `${repairCode};return renderRepair;`)({workspace:{issue:{summary:"Original Bixby disruption"}},session:{proposal}},
    () => null,() => displayed,() => ({}),() => true,String,String,() => "",() => panel);
render();
assert.ok(panel.innerHTML.includes("Original repair context"));
assert.ok(!panel.innerHTML.includes("Issue detected"));
displayed = acceptedPlan;
render();
assert.ok(panel.innerHTML.includes("Issue detected"),"A child response not actually rendered must not relabel the parent view");

const toastStart = source.indexOf("function toast(");
const startup = source.indexOf("\nensureHostedLogin()", toastStart);
assert.ok(toastStart >= 0 && startup > toastStart);
const toastCode = source.slice(toastStart, startup);
const items = [];
const region = {querySelectorAll:() => items.filter(item => !item.className.includes("error")),append:item => items.push(item)};
const notify = new Function("document","$","setTimeout",`${toastCode};return toast;`)(
  {createElement:() => ({remove() {const index=items.indexOf(this);if(index>=0) items.splice(index,1);}})},
  () => region,() => {});
notify("Draft staged");
notify("Rendering preview");
notify("Preview ready");
assert.deepEqual(items.map(item => item.textContent),["Preview ready"]);
notify("Retained error",true);
notify("Current status");
assert.deepEqual(items.map(item => item.textContent),["Retained error","Current status"]);

// Compare/Evidence own their selected-artifact summaries; the Trip strip must
// not display an unlabeled different plan there, or change Trip selection.
const metricsCode = source.slice(source.indexOf("function renderMetrics("), source.indexOf("function evaluatedProposalCompareContext("));
const metricsState = {route:"/app/itinerary"};
const metricsStrip = {dataset:{},hidden:false,innerHTML:""};
const metricsFacts = {repair:null,metrics:{},certificateId:"exact-certificate",weightedEditCost:6,eligibility:"eligible"};
const renderSummary = new Function("state","activeRenderPlan","planPresentationEvidence","planRouteMinutes","formatMetric","$","escapeHtml",
  `${metricsCode};return renderMetrics;`)(metricsState,() => acceptedPlan,() => metricsFacts,() => 1992.62,String,() => metricsStrip,String);
for (const route of ["/app/compare","/app/evidence","/app/itinerary","/app/map","/app/repairs","/app/compare","/app/itinerary"]) {
  metricsState.route = route;
  renderSummary();
  assert.equal(metricsStrip.hidden,["/app/compare","/app/evidence"].includes(route),`Correct metric visibility in ${route}`);
  assert.equal(metricsStrip.dataset.planId,acceptedPlan.plan_id);
  assert.equal(metricsStrip.dataset.planContentHash,acceptedPlan.content_hash);
  assert.equal(metricsStrip.dataset.evaluationId,metricsFacts.certificateId);
}
