import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { recordedJson } from "./recorded-runtime-fixture.mjs";

const source = readFileSync(new URL("../../src/itinerary_system/product_app/static/js/app.js", import.meta.url), "utf8");
const captured = recordedJson("results/e2e/r01_provider_isolation_20260907_v2/accepted_reload/reload_response.json");
const original = captured.workspace.geography.plans.find((plan) => plan.role === "original");
assert.ok(original);
const state = { workspace: captured.workspace, session: { selected_day: 5, proposal: null, accepted_plan_id: original.plan_id } };
const start = source.indexOf("function routeSummaryForPlan(");
const end = source.indexOf("function activeDisplayedPlan(", start);
assert.ok(start >= 0 && end > start);
const functions = new Function("state", "formatMetric", "escapeHtml", "routeNamePair", `${source.slice(start, end)}\nreturn { routeSummaryForPlan, textualRouteEvidence, textRoutePanelHtml, planPresentationEvidence, planRouteMinutes };`)(
  state, (value) => String(value), (value) => String(value), (leg) => ({ origin: leg.origin_id, destination: leg.destination_id }),
);

test("the original route never borrows registered-child eligibility", () => {
  const text = functions.textualRouteEvidence(original, functions.routeSummaryForPlan(original));
  assert.match(text, /Bixby Creek Bridge/);
  assert.doesNotMatch(text, /Eligibility: Eligible repair/i);
  assert.match(text, /Eligibility: Not evaluated/i);
});

test("the original text panel identifies its exact unevaluated plan", () => {
  const html = functions.textRoutePanelHtml(original, original);
  assert.match(html, /Bixby Creek Bridge/);
  assert.doesNotMatch(html, /<dt>Eligibility<\/dt><dd>Eligible repair<\/dd>/i);
  assert.match(html, new RegExp(`data-plan-id="${original.plan_id}"`));
  assert.match(html, new RegExp(`data-plan-content-hash="${original.content_hash}"`));
});

test("an unrelated ineligible candidate cannot label the parent evaluated", () => {
  state.session.proposal = { child_plan_id: "plan_unrelated", state: "ineligible", eligibility: "ineligible" };
  const text = functions.textualRouteEvidence(original, functions.routeSummaryForPlan(original));
  assert.match(text, /Eligibility: Not evaluated/i);
});

test("missing plan and mismatched certificate have no invented evaluation", () => {
  assert.equal(functions.planPresentationEvidence(null).eligibility, "Not evaluated");
  const registered = captured.workspace.geography.plans.find((plan) => plan.role === "registered_repair");
  assert.equal(functions.planPresentationEvidence(registered).eligibility, "eligible");
  assert.equal(functions.planPresentationEvidence({ ...registered, content_hash: "other" }).eligibility, "Not evaluated");
});

test("route duration uses the actual seconds and refuses incomplete coverage", () => {
  const expected = original.validated_legs.features.reduce((sum, leg) => sum + leg.properties.duration_s / 60, 0);
  assert.equal(functions.planRouteMinutes(original), expected);
  assert.equal(functions.planRouteMinutes({ ...original, coverage: { complete: false } }), null);
  const invalid = structuredClone(original);
  delete invalid.validated_legs.features[0].properties.duration_s;
  assert.equal(functions.planRouteMinutes(invalid), null);
});

test("metric strip references the displayed parent rather than an unrelated candidate", () => {
  const strip = { dataset: {}, innerHTML: "" };
  const metricStart = source.indexOf("function renderMetrics()");
  const metricEnd = source.indexOf("function evaluatedProposalCompareContext()", metricStart);
  const render = new Function("state", "$", "activeRenderPlan", "planPresentationEvidence", "planRouteMinutes", "formatMetric", "escapeHtml", `${source.slice(metricStart, metricEnd)}\nreturn renderMetrics;`)(
    state, () => strip, () => original, functions.planPresentationEvidence, functions.planRouteMinutes, String, String,
  );
  render();
  assert.equal(strip.dataset.planId, original.plan_id);
  assert.equal(strip.dataset.planContentHash, original.content_hash);
  assert.equal(strip.dataset.evaluationId, "");
  assert.doesNotMatch(strip.innerHTML, /88%/);
  assert.match(strip.innerHTML, /Not evaluated for this plan/);
});
