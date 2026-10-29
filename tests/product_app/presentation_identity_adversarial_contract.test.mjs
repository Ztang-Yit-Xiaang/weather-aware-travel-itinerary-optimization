// Independent R09/R10 counterexamples. Captured artifacts are real; deliberately
// changed hashes, missing legs, and crossed certificates are labeled mutations.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { recordedJson } from "./recorded-runtime-fixture.mjs";

const read = (path) => readFileSync(new URL(`../../${path}`, import.meta.url), "utf8");
const appSource = read("src/itinerary_system/product_app/static/js/app.js");
const compareSource = read("src/itinerary_system/product_app/static/js/compare.js");
const evidenceRoot = "results/e2e/r01_provider_isolation_20260907_v2/accepted_reload";
const captured = recordedJson(`${evidenceRoot}/reload_response.json`);
const preview = recordedJson(`${evidenceRoot}/preview_response.json`).session.proposal;
const original = captured.workspace.geography.plans.find((plan) => plan.role === "original");
const accepted = captured.workspace.geography.plans.find((plan) => plan.plan_id === captured.session.accepted_plan_id);
const registered = captured.workspace.geography.plans.find((plan) => plan.role === "registered_repair");
assert.ok(original && accepted && registered);

function section(source, from, to) {
  const start = source.indexOf(from);
  const end = source.indexOf(to, start + from.length);
  assert.ok(start >= 0 && end > start);
  return source.slice(start, end);
}

function appHarness(workspace = structuredClone(captured.workspace), session = structuredClone(captured.session)) {
  const state = { workspace, session };
  const source = section(appSource, "function routeSummaryForPlan(", "function activeDisplayedPlan(");
  const functions = new Function("state", "formatMetric", "escapeHtml", "routeNamePair", `${source}
    return {planPresentationEvidence, textRoutePanelHtml, planRouteMinutes, textualRouteEvidence, routeSummaryForPlan};`)(
    state, String, String, (leg) => ({ origin: leg.origin_id, destination: leg.destination_id }),
  );
  return { ...functions, state };
}

const compare = new Function(`${section(compareSource, "function escapeHtml(", "function diffSummary(")
  .replaceAll("export function", "function")}
  return {exactBaseline, selectedMetricsTable, resolveCompareSelection};`)();

function tableRow(html, label) {
  const row = html.match(new RegExp(`<th scope="row">${label}<\\/th>([\\s\\S]*?)<\\/tr>`));
  assert.ok(row, `metric ${label} is present`);
  return [...row[1].matchAll(/<td[^>]*>([\s\S]*?)<\/td>/g)].map((match) => match[1]);
}

test("actual accepted reload uses the persisted child as Compare baseline", () => {
  const baseline = compare.exactBaseline(captured.workspace, captured.session);
  assert.equal(baseline?.plan_id, accepted.plan_id, "Compare must not lose the baseline after accepting a real child");
  assert.equal(baseline?.content_hash, captured.workspace.accepted_plan_pointer.accepted_plan_content_hash);
});

test("Compare names its accepted baseline truthfully after a child is accepted", () => {
  assert.ok(compareSource.includes('mapPane("baseline", "Current trip route", baseline, baselineUnavailable)'));
  assert.ok(compareSource.includes('<h3>Current accepted trip</h3>'));
  assert.ok(!compareSource.includes('mapPane("baseline", "Original route"'));
  assert.ok(!compareSource.includes('<h3>Keep original</h3>'));
  assert.ok(compareSource.includes('When no option is selected, an eligible recommended repair is selected automatically.'));
  assert.ok(compareSource.includes('Accepting a repair still requires an eligible evaluated preview and confirmation.'));
});

test("mutated accepted-pointer hash prevents Compare from presenting a mismatched baseline", () => {
  const workspace = structuredClone(captured.workspace);
  workspace.accepted_plan_pointer = {
    ...workspace.accepted_plan_pointer,
    accepted_plan_id: original.plan_id,
    accepted_plan_content_hash: "mismatched-content-hash",
  };
  assert.equal(compare.exactBaseline(workspace, { accepted_plan_id: original.plan_id })?.plan_id ?? null, null);
});

test("Compare baseline duration comes from the actual complete road legs", () => {
  const table = compare.selectedMetricsTable(original, { state: "unavailable" });
  const minutes = original.validated_legs.features.reduce((sum, leg) => sum + leg.properties.duration_s / 60, 0);
  const row = tableRow(table, "Strict route time");
  assert.equal(row[1], `${Number(minutes.toFixed(2))} minutes`);
  assert.equal(tableRow(table, "Road-validated route")[1], "Validated");
});

test("mutated route-coverage count cannot advertise a validated Compare baseline", () => {
  const missing = structuredClone(original);
  missing.validated_legs.features.pop();
  missing.coverage.complete = false;
  // A stale display status is deliberately retained to expose field disagreement.
  missing.coverage.status = "complete";
  const table = compare.selectedMetricsTable(missing, { state: "unavailable" });
  assert.equal(tableRow(table, "Road-validated route")[1], "Unavailable");
  assert.equal(tableRow(table, "Strict route time")[1], "Unavailable");
});

test("mutated certificate inner identity does not remain exact Compare evidence", () => {
  const workspace = structuredClone(captured.workspace);
  workspace.alternative_evidence[registered.plan_id].certificate.plan_id = accepted.plan_id;
  const selection = compare.resolveCompareSelection(workspace, { selected_alternative_id: registered.plan_id });
  assert.equal(selection.evidenceExact, false, "outer record identity cannot vouch for a crossed evaluator certificate");
  assert.equal(selection.evidence === null, true);
});

test("a persisted accepted certificate remains eligible with no active proposal", () => {
  const workspace = structuredClone(captured.workspace);
  workspace.alternative_evidence[accepted.plan_id] = {
    plan_id: accepted.plan_id, plan_content_hash: accepted.content_hash,
    certificate: structuredClone(preview.runtime_execution.evaluation_certificate),
    diff: structuredClone(preview.plan_diff),
  };
  const h = appHarness(workspace);
  assert.equal(h.state.session.proposal, null);
  const facts = h.planPresentationEvidence(accepted);
  assert.equal(facts.eligibility, "eligible");
  assert.equal(facts.certificateId, preview.certificate_id);
  assert.equal(facts.metrics.preservation_rate, preview.runtime_execution.evaluation_certificate.metrics.preservation_rate);
  const html = h.textRoutePanelHtml(accepted, original);
  assert.match(html, new RegExp(`data-plan-id="${accepted.plan_id}"`));
  assert.match(html, /Pfeiffer Big Sur State Park/);
});

test("a current unrelated preview cannot lend eligibility or booking metrics to the original", () => {
  const h = appHarness();
  h.state.session.proposal = structuredClone(preview);
  const facts = h.planPresentationEvidence(original);
  assert.equal(facts.eligibility, "Not evaluated");
  assert.equal(facts.certificateId, null);
  assert.equal(facts.permissions, null);
  const html = h.textRoutePanelHtml(original, original);
  assert.match(html, /<dt>Eligibility<\/dt><dd>Not evaluated<\/dd>/);
  assert.match(html, /<dt>Booking impact<\/dt><dd>Unavailable<\/dd>/);
});

test("mutated selected-day route gap does not present the remaining legs as a complete travel duration", () => {
  const incomplete = structuredClone(accepted);
  const missingIndex = incomplete.validated_legs.features.findIndex((leg) => leg.properties.day === 5);
  assert.ok(missingIndex >= 0);
  const [removed] = incomplete.validated_legs.features.splice(missingIndex, 1);
  incomplete.coverage.complete = false;
  incomplete.coverage.status = "incomplete";
  incomplete.coverage.gap_count = 1;
  incomplete.coverage.road_validated_leg_count -= 1;
  incomplete.gaps.features.push({
    type: "Feature", geometry: null,
    properties: { ...removed.properties, road_validated: false, validation_status: "missing_route" },
  });
  const h = appHarness();
  assert.equal(h.planRouteMinutes(incomplete), null);
  const html = h.textRoutePanelHtml(incomplete, original);
  const duration = html.match(/<dt>Travel duration<\/dt><dd>(.*?)<\/dd>/)?.[1];
  assert.match(duration || "", /Unavailable|partial|incomplete/i,
    "the duration must disclose the omitted required leg instead of summing only present legs");
});
