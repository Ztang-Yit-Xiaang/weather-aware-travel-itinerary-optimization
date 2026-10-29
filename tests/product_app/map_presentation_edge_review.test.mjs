// Independent extracted-production-code checks. Synthetic DOM only; these do
// not claim browser rendering, pointer interaction, or provider connectivity.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const source = readFileSync(new URL("../../src/itinerary_system/product_app/static/js/app.js", import.meta.url), "utf8");
function section(from, to) {
  const start = source.indexOf(from), end = source.indexOf(to, start + from.length);
  assert.ok(start >= 0 && end > start);
  return source.slice(start, end);
}
function extract(code, result, dependencies) {
  return new Function(...Object.keys(dependencies), `${code}\nreturn ${result};`)(...Object.values(dependencies));
}
function plan(id) {
  return { plan_id: id, content_hash: `${id}-hash`, role: "original", stops: { features: [] } };
}

test("actual map markup preserves a native details toggle in either direction while updating its contents", () => {
  let details = null, html = "", current = plan("first");
  const workspace = { set innerHTML(value) {
    html = value;
    const attributes = value.match(/<details class="textual-map-alternative"([^>]*)>/)?.[1];
    assert.notEqual(attributes, undefined);
    details = { open: /\bopen\b/.test(attributes) };
  } };
  const code = section("function renderMap()", "  $$('[data-geo-stop]')") + "\n}";
  const render = extract(code, "renderMap", {
    $: selector => selector === ".textual-map-alternative" ? details : selector === "#workspace-content" ? workspace : null,
    $$: () => [],
    state: { session: { selected_day: 1, draft: [], proposal: null }, workspace: { issue: {} },
      mapInteraction: { mode: "select" }, typedImpact: {}, previewRuntime: { status: "PARENT_RENDERED", epoch: 1 }, poiDiscovery: {} },
    activeGeography: () => ({ schema_version: "product-geography-v2", status: "ready", plans: [current] }),
    confirmedMapDraftState: () => ({ waypoints: [], routeLegs: [] }), acceptedPlan: () => current,
    activeAlternativeId: () => null, routeSummaryForPlan: p => p.plan_id,
    textualRouteEvidence: p => `Current route ${p.plan_id}`, textRoutePanelHtml: p => `<p>${p.plan_id}</p>`,
    validatedRouteFeatures: () => [], routeGapFeatures: () => [], hasRouteCheckedOnlyDraft: () => false,
    mapInspectorHtml: () => "", candidateExplorerHtml: () => "", mapAttribution: () => "", escapeHtml: value => String(value ?? ""),
    currentInterestResponse: () => null, interestExplorerHtml: () => "", yosemiteGuideHtml: () => "",
  });
  render(); assert.equal(details.open, false, "first load defaults closed");
  details.open = true; current = plan("second"); render();
  assert.equal(details.open, true); assert.match(html, /Current route second/); assert.doesNotMatch(html, /Current route first/);
  details.open = false; current = plan("third"); render();
  assert.equal(details.open, false); assert.match(html, /Current route third/);
  // Absence after another module is not a stale preference for open.
  details = null; render(); assert.equal(details.open, false);
});

function statusHarness() {
  let current = plan("child");
  const state = { session: { proposal: { state: "eligible", acceptance_eligible: true, child_plan_id: "child" } },
    acknowledgedPreviewPlanId: null, previewRuntime: { status: "PREVIEW_DATA_READY", epoch: 1, requestId: "request" } };
  const stage = { dataset: {} }, element = { dataset: {}, classList: { add() {}, remove() {} }, closest: () => stage };
  const code = section("function updateGeographicMapStatus(", "function timeWindowLabel(")
    + section("function previewCanBeAccepted()", "async function submitAcceptance(");
  const functions = extract(code, "({updateGeographicMapStatus,previewCanBeAccepted})", {
    state, $: () => element, activeRenderPlan: () => current, activeProposalPlan: () => current,
    copilot: { previewStateChanged() {} }, toast() {}, renderContextPanel() {},
  });
  const ready = overrides => functions.updateGeographicMapStatus({ state: "ready", activePlanId: current?.plan_id,
    activePlanContentHash: current?.content_hash, activeGeometryHash: "a".repeat(64), activeMarkerIds: [], ...overrides });
  return { ...functions, state, stage, ready, setPlan: value => { current = value; } };
}

function noAcknowledgement(h) {
  assert.equal(h.previewCanBeAccepted(), false);
  assert.equal(h.state.acknowledgedPreviewPlanId, null);
  for (const key of ["renderedMapPlanId", "renderedGeometryHash", "renderedMarkerIds"]) {
    assert.equal(h.stage.dataset[key], undefined, `withdraw ${key} instead of retaining old success evidence`);
  }
}

test("every non-ready state retains only the canonical identity and revokes acceptance", () => {
  for (const state of ["loading", "failed", "unavailable", "unexpected_state"]) {
    const h = statusHarness(); h.ready(); assert.equal(h.previewCanBeAccepted(), true);
    h.setPlan(plan("new-child")); h.state.session.proposal.child_plan_id = "new-child";
    h.updateGeographicMapStatus({ state, activePlanId: "untrusted-old-child", activePlanContentHash: "old-hash" });
    assert.equal(h.stage.dataset.activePlanId, "new-child");
    assert.equal(h.stage.dataset.activePlanContentHash, "new-child-hash");
    noAcknowledgement(h);
  }
});

test("absence of a canonical plan clears active identity and all readiness acknowledgement", () => {
  const h = statusHarness(); h.ready(); h.setPlan(null);
  h.updateGeographicMapStatus({ state: "failed" });
  assert.equal(h.stage.dataset.activePlanId, undefined);
  assert.equal(h.stage.dataset.activePlanContentHash, undefined);
  noAcknowledgement(h);
});

test("mismatched ready payload withdraws every previous rendered acknowledgement", () => {
  const h = statusHarness(); h.ready(); assert.equal(h.previewCanBeAccepted(), true);
  h.setPlan(plan("new-child")); h.state.session.proposal.child_plan_id = "new-child";
  h.ready({ activePlanId: "child", activePlanContentHash: "child-hash" });
  assert.equal(h.state.previewRuntime.status, "MAP_IDENTITY_FAILED");
  assert.equal(h.stage.dataset.activePlanId, "new-child");
  assert.equal(h.stage.dataset.activePlanContentHash, "new-child-hash");
  noAcknowledgement(h);
});

test("ready payload without canonical plan cannot leave old map identity published", () => {
  const h = statusHarness(); h.ready(); h.setPlan(null);
  h.ready({ activePlanId: "child", activePlanContentHash: "child-hash" });
  assert.equal(h.state.previewRuntime.status, "MAP_IDENTITY_FAILED");
  assert.equal(h.stage.dataset.activePlanId, undefined);
  noAcknowledgement(h);
});
