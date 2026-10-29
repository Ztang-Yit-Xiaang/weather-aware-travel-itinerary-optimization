import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const source = readFileSync(new URL("../../src/itinerary_system/product_app/static/js/app.js", import.meta.url), "utf8");
function section(start, end) {
  const first = source.indexOf(start), last = source.indexOf(end, first + start.length);
  assert.ok(first >= 0 && last > first, `${start} must be extracted from production`);
  return source.slice(first, last);
}

function harness(action, { stop = "shared", segment = null, proposal = "eligible", waypoint = false, failure = false } = {}) {
  const parent = { plan_id: "parent", content_hash: "parent-hash",
    stops: { features: [{ properties: { stop_id: "shared", name: "Accepted stop", day: 2, plan_id: "parent" } }] },
    validated_legs: { features: [{ properties: { route_leg_id: "shared-leg", plan_id: "parent", day: 2, origin_id: "a", destination_id: "b" } }] } };
  const preview = { state: "eligible", child_plan_id: "child", child_plan_content_hash: "child-hash",
    geography_plan: { plan_id: "child", content_hash: "child-hash" } };
  if (proposal === "ineligible") preview.state = "ineligible";
  if (proposal === "unbound") preview.geography_plan.plan_id = "foreign";
  if (proposal === "empty") { preview.child_plan_id = ""; preview.geography_plan.plan_id = ""; }
  const operation = { type: "keep_stop", target: "shared", parameters: {} };
  const draftWaypoint = { type: "add_custom_waypoint", target: "waypoint" };
  const state = { undoPending: false, session: { session_id: "session", trip_id: "trip", run_id: "run", revision: 1,
    accepted_plan_id: "parent", selected_stop_id: stop,
    selected_segment_id: segment, selected_day: 3, draft: waypoint ? [draftWaypoint, operation] : [operation],
    proposal: proposal === "none" ? null : preview }, workspace: { geography: { plans: [parent] } },
    typedImpact: { request: null, status: "idle", response: null, error: null },
    mapInteraction: { selectionKind: waypoint ? "custom_waypoint" : segment ? "route_leg" : "stop",
      selectedStopPlanId: segment || waypoint ? null : "child", selectedWaypointId: waypoint ? "waypoint" : null,
      selectedRouteLeg: segment ? { route_leg_id: segment, plan_id: "child", day: 3 } : null } };
  const before = structuredClone(state), next = structuredClone(state.session);
  next.revision = 2;
  next.proposal = null;
  next.draft = action === "undoDraft" ? (waypoint ? [draftWaypoint] : []) : [...next.draft, operation];
  next.selected_stop_id = stop === "shared" ? stop : null;
  next.selected_segment_id = segment === "shared-leg" ? segment : null;
  next.selected_day = next.selected_stop_id ? 2 : 3;
  const errors = [], impact = [], failures = [], renders = [], refreshes = [];
  const dependencies = { state, pendingSelection: null, copilot: { update() {} },
    submitDraftMutation: async () => { if (failure) throw new Error("synthetic mutation rejection"); return { session: next }; },
    syncPreviewRuntimeFromSession() {}, resetPoiDiscovery() {}, draftCombinationBlocker: () => null,
    // This fixture isolates selection reconciliation, not impact validation.
    // Model the real asynchronous request ownership so completion is exercised.
    loadDraftImpactPreview: async () => {
      state.typedImpact.request = Object.fromEntries(
        ["session_id", "trip_id", "run_id", "revision", "accepted_plan_id"].map(key => [key, state.session[key]]),
      );
      state.typedImpact.status = "loading";
      state.typedImpact.response = null;
      impact.push(structuredClone(state.mapInteraction));
      await Promise.resolve();
      state.typedImpact.status = "ready";
      state.typedImpact.response = { session_revision: state.session.revision };
      return state.typedImpact.response;
    },
    toast: (message, error = false) => { errors.push(message); if (error) failures.push(message); },
    renderAll: () => renders.push(state.session.revision), refreshMapEditDialog: () => refreshes.push(state.session.revision),
    selectedDraftWaypoint: () => state.session.draft.find(row => row.target === state.mapInteraction.selectedWaypointId),
    validatedRouteFeatures: plan => plan?.validated_legs?.features || [],
    activeGeography: () => state.workspace.geography, activeRenderPlan: () => parent,
    $: () => ({ open: true }) };
  const helper = source.includes("function installDraftMutationSession(")
    ? section("function installDraftMutationSession(", "async function addDraft(") : "";
  const code = section("function acceptedPlan(", "function selectedAcceptedRouteFeature(")
    + section("function selectedAcceptedRouteFeature(", "function ")
    + section("function findRouteLeg(", "const MAP_DRAFT_OPERATION_TYPES")
    + section("function findSelectedStop()", "function metricCell(") + helper
    + section("function draftImpactRequestIsCurrent(", "async function loadDraftImpactPreview(")
    + section("async function addDraft(", "async function addCopilotDraft(")
    + section("async function undoDraft()", "async function previewDraft(");
  const functions = new Function(...Object.keys(dependencies), `${code}\nreturn {addDraft, undoDraft, installDraftMutationSession, selectedAcceptedStopFeature, selectedAcceptedRouteFeature, findSelectedStop};`)(...Object.values(dependencies));
  return { state, before, next, parent, errors, impact, functions, dependencies,
    run: async () => {
      const renderedBefore = renders.length, refreshedBefore = refreshes.length;
      const result = await (action === "addDraft" ? functions.addDraft("keep_stop", {}, "shared") : functions.undoDraft());
      if (!failure) {
        assert.deepEqual(failures, [], "A caught continuation exception must not masquerade as successful withdrawal");
        if (action === "addDraft") assert.equal(result, true);
        assert.equal(renders.length - renderedBefore, 2, "Render both committed and impact-complete states");
        assert.equal(refreshes.length - refreshedBefore, 2, "Refresh both phases of the open editor");
      }
      return result;
    } };
}

for (const stop of ["shared", "child-only"]) {
  test(`confirmed waypoint withdrawal and later Undo preserve the corrected ${stop} hint`, async () => {
    const h = harness("undoDraft", { stop });
    const confirmed = { ...structuredClone(h.next), draft: [...h.before.session.draft, { type: "add_route_waypoint", target: "waypoint" }] };
    h.next.revision = confirmed.revision + 1;
    h.next.draft = structuredClone(h.before.session.draft);
    h.state.mapInteraction.snapPreview = { snap_preview_id: "snap", operation_intent: "add_route_waypoint", confirmation_required: false };
    h.state.mapInteraction.exploratoryPin = { longitude: -121, latitude: 36 };
    const dependencies = { ...h.dependencies,
      installDraftMutationSession: h.functions.installDraftMutationSession,
      roadCheckedPreviewCanConfirm: () => true, mapEditCapability: () => ({ enabled: true }),
      api: async () => ({ session: confirmed, operation: { target: "waypoint" } }),
      requestAnimationFrame: callback => callback(),
      $: () => ({ value: "Route waypoint", open: false, focus() {} }) };
    const confirm = new Function(...Object.keys(dependencies), `${section("async function confirmMapWaypoint(", "function refreshMapEditDialog(")}\nreturn confirmMapWaypoint;`)(...Object.values(dependencies));
    let prevented = 0;
    await confirm({ preventDefault() { prevented++; } });
    assert.equal(prevented, 1);
    assert.equal(h.state.session, confirmed);
    assert.equal(h.state.mapInteraction.selectionKind, "custom_waypoint");
    assert.equal(h.state.mapInteraction.selectedWaypointId, "waypoint");
    assert.equal(h.state.mapInteraction.selectedStopPlanId, stop === "shared" ? "parent" : null);
    await h.run();
    assert.equal(h.state.session, h.next);
    assert.equal(h.state.mapInteraction.selectedWaypointId, null);
    assert.equal(h.state.mapInteraction.selectedStopPlanId, stop === "shared" ? "parent" : null);
    assert.equal(h.functions.selectedAcceptedStopFeature(h.state.workspace.geography), stop === "shared" ? h.parent.stops.features[0] : null);
  });
}

for (const action of ["undoDraft", "addDraft"]) {
  test(`${action}: a valid accepted stop regains its accepted-plan editor context`, async () => {
    const h = harness(action);
    await h.run();
    assert.equal(h.state.session, h.next);
    assert.equal(h.state.mapInteraction.selectedStopPlanId, "parent");
    assert.equal(h.impact[0].selectedStopPlanId, "parent", "Correct the hint before the asynchronous impact load");
    assert.equal(h.functions.selectedAcceptedStopFeature(h.state.workspace.geography), h.parent.stops.features[0]);
    assert.deepEqual(h.functions.findSelectedStop(), { id: "shared", name: "Accepted stop", day: 2, plan_id: "parent" });
  });
  test(`${action}: orphan preview stop hints are cleared`, async () => {
    const h = harness(action, { stop: "child-only" }); await h.run();
    assert.equal(h.state.mapInteraction.selectedStopPlanId, null);
    assert.equal(h.state.mapInteraction.selectionKind, null);
    assert.equal(h.functions.selectedAcceptedStopFeature(h.state.workspace.geography), null);
  });
  for (const segment of ["shared-leg", "child-only-leg"]) {
    test(`${action}: ${segment} selection is reconciled to accepted route evidence`, async () => {
      const h = harness(action, { stop: null, segment }); await h.run();
      assert.deepEqual(h.state.mapInteraction.selectedRouteLeg,
        segment === "shared-leg" ? h.parent.validated_legs.features[0].properties : null);
      assert.equal(h.state.mapInteraction.selectionKind, segment === "shared-leg" ? "route_leg" : null);
      assert.equal(h.functions.selectedAcceptedRouteFeature(h.state.workspace.geography),
        segment === "shared-leg" ? h.parent.validated_legs.features[0] : null);
    });
  }
  for (const proposal of ["none", "ineligible", "unbound", "empty"]) {
    test(`${action}: ${proposal} does not globally override explicit inspection`, async () => {
      const h = harness(action, { proposal }); await h.run();
      assert.deepEqual(h.state.mapInteraction, h.before.mapInteraction);
    });
  }
  test(`${action}: a selected remaining draft waypoint is preserved`, async () => {
    const h = harness(action, { waypoint: true }); await h.run();
    assert.deepEqual(h.state.mapInteraction, h.before.mapInteraction);
  });
  test(`${action}: rejected mutation cannot alter the selected plan hint`, async () => {
    const h = harness(action, { failure: true }); await h.run();
    assert.deepEqual(h.state, h.before);
    assert.deepEqual(h.errors, ["synthetic mutation rejection"]);
  });
}
