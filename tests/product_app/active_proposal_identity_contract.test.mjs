import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

const source = readFileSync(
  new URL("../../src/itinerary_system/product_app/static/js/app.js", import.meta.url),
  "utf8",
);
const start = source.indexOf("function activeProposalPlan()");
const end = source.indexOf("\nfunction activeGeography()", start);
assert.ok(start >= 0 && end > start, "activeProposalPlan source must be extractable");
const exactKeysStart = source.indexOf("function exactKeys(");
const exactKeysEnd = source.indexOf("\nfunction ", exactKeysStart + 10);
const routeValidationStart = source.indexOf("function normalizeRouteValidation(");
const routeValidationEnd = source.indexOf("\nfunction ", routeValidationStart + 10);
assert.ok(
  exactKeysStart >= 0 && exactKeysEnd > exactKeysStart
  && routeValidationStart >= 0 && routeValidationEnd > routeValidationStart,
  "active proposal dependencies must be extractable",
);

const state = { session: { proposal: null } };
const activeProposalPlan = new Function(
  "state",
  source.slice(exactKeysStart, exactKeysEnd)
    + "\n" + source.slice(routeValidationStart, routeValidationEnd)
    + "\n" + source.slice(start, end)
    + "\nreturn activeProposalPlan;",
)(state);
const childPlanId = "plan_1111111111111111";
const childHash = "2222222222222222";
const collectionNames = ["route_path", "stops", "validated_legs", "gaps"];

function proposal() {
  const geographyPlan = {
    plan_id: childPlanId,
    content_hash: childHash,
    role: "draft_preview",
    status: "ready",
    coverage: {
      status: "complete",
      complete: true,
      itinerary_sequence_accounted: true,
      route_path_node_count: 2,
      required_leg_count: 1,
      road_validated_leg_count: 1,
      gap_count: 0,
    },
  };
  for (const name of collectionNames) {
    geographyPlan[name] = {
      type: "FeatureCollection",
      features: name === "gaps" ? [] : [{
        properties: {
          plan_id: childPlanId,
          content_hash: childHash,
          role: "draft_preview",
        },
      }],
    };
  }
  geographyPlan.route_path.features.push(structuredClone(geographyPlan.route_path.features[0]));
  return {
    schema_version: "draft-preview-v1",
    state: "eligible",
    child_plan_id: childPlanId,
    child_plan_content_hash: childHash,
    route_validation: {
      matrix_id: "route_matrix_aaaaaaaaaaaaaaaa",
      context_snapshot_id: "context_demo",
      publication_ready: true,
      required_leg_count: 1,
      present_leg_count: 1,
      road_validated_leg_count: 1,
      fallback_leg_count: 0,
      invalid_value_count: 0,
      missing_leg_count: 0,
      source_bundle_id: "route_bundle_bbbbbbbbbbbbbbbb",
      source_content_sha256: "c".repeat(64),
      errors: [],
      warnings: [],
    },
    geography_plan: geographyPlan,
  };
}

state.session.proposal = proposal();
assert.equal(activeProposalPlan(), state.session.proposal.geography_plan);

state.session.proposal = proposal();
state.session.proposal.geography_plan.content_hash = "3333333333333333";
assert.equal(activeProposalPlan(), null);

state.session.proposal = proposal();
state.session.proposal.geography_plan.plan_id = "plan_4444444444444444";
assert.equal(activeProposalPlan(), null);

state.session.proposal = proposal();
state.session.proposal.geography_plan.role = "original";
assert.equal(activeProposalPlan(), null);

state.session.proposal = proposal();
state.session.proposal.geography_plan.status = "failed";
assert.equal(activeProposalPlan(), null);

state.session.proposal = proposal();
state.session.proposal.geography_plan.coverage.road_validated_leg_count = 0;
assert.equal(activeProposalPlan(), null);

for (const name of ["route_path", "stops", "validated_legs"]) {
  for (const [key, value] of [
    ["plan_id", "plan_5555555555555555"],
    ["content_hash", "6666666666666666"],
    ["role", "original"],
  ]) {
    state.session.proposal = proposal();
    state.session.proposal.geography_plan[name].features[0].properties[key] = value;
    assert.equal(activeProposalPlan(), null, name + " must reject nested " + key);
  }
  state.session.proposal = proposal();
  delete state.session.proposal.geography_plan[name];
  assert.equal(activeProposalPlan(), null, name + " must be present");

  state.session.proposal = proposal();
  state.session.proposal.geography_plan[name].type = "NotGeoJSON";
  assert.equal(activeProposalPlan(), null, name + " must be a FeatureCollection");

  state.session.proposal = proposal();
  state.session.proposal.geography_plan[name].features = [null];
  assert.equal(activeProposalPlan(), null, name + " must reject null features");
}

state.session.proposal = proposal();
delete state.session.proposal.geography_plan.gaps;
assert.equal(activeProposalPlan(), null, "gaps must be present");

state.session.proposal = proposal();
state.session.proposal.geography_plan.gaps.type = "NotGeoJSON";
assert.equal(activeProposalPlan(), null, "gaps must be a FeatureCollection");

state.session.proposal = proposal();
state.session.proposal.geography_plan.gaps.features = [null];
assert.equal(activeProposalPlan(), null, "gaps must reject null features");

for (const name of ["route_path", "stops"]) {
  state.session.proposal = proposal();
  state.session.proposal.geography_plan[name].features = [];
  assert.equal(activeProposalPlan(), null, name + " must not be empty");
}

state.session.proposal = proposal();
state.session.proposal.state = "ineligible";
assert.equal(activeProposalPlan(), null);

state.session.proposal = proposal();
delete state.session.proposal.geography_plan.coverage.route_path_node_count;
assert.equal(activeProposalPlan(), null, "route path count must be present");

state.session.proposal = proposal();
state.session.proposal.geography_plan.validated_legs.features = [];
assert.equal(activeProposalPlan(), null, "leg count must bind to validated legs");

state.session.proposal = proposal();
state.session.proposal.geography_plan.route_path.features.pop();
assert.equal(activeProposalPlan(), null, "route path must contain legs plus one node");

state.session.proposal = proposal();
state.session.proposal.route_validation.required_leg_count = 9;
assert.equal(activeProposalPlan(), null, "coverage must bind to route validation");

state.session.proposal = proposal();
state.session.proposal.route_validation.required_leg_count = 0;
state.session.proposal.route_validation.present_leg_count = 0;
state.session.proposal.route_validation.road_validated_leg_count = 0;
state.session.proposal.geography_plan.coverage.route_path_node_count = 1;
state.session.proposal.geography_plan.coverage.required_leg_count = 0;
state.session.proposal.geography_plan.coverage.road_validated_leg_count = 0;
state.session.proposal.geography_plan.route_path.features.pop();
state.session.proposal.geography_plan.validated_legs.features = [];
assert.equal(activeProposalPlan(), null, "zero-leg preview must fail closed");

console.log("active proposal identity contract passed");
