import assert from "node:assert/strict";

import { copilotCombinedEligibility, copilotDraftability, copilotOperationForIntent } from "../../src/itinerary_system/product_app/static/js/copilot.js";

const workspace = {
  draft_capabilities: {
    operations: {
      move_day: { enabled: true },
      replace_nearby: { enabled: true },
      add_candidate: { enabled: true },
      route_feedback: { enabled: true },
    },
    parent_targets: ["bixby_creek_bridge_viewpoint", "golden_gate_bridge"],
    candidate_choices: [
      {
        candidate_id: "pfeiffer_big_sur_state_park",
        replaces_stop_id: "bixby_creek_bridge_viewpoint",
      },
    ],
  },
};

assert.equal(copilotDraftability({
  type: "move_day",
  target_stop_id: "golden_gate_bridge",
  destination_day: 2,
}, workspace).allowed, true);

const previewStop = copilotDraftability({
  type: "move_day",
  target_stop_id: "pfeiffer_big_sur_state_park",
  destination_day: 6,
}, workspace);
assert.equal(previewStop.allowed, false);
assert.match(previewStop.reason, /repair-preview/);

assert.equal(copilotDraftability({
  type: "replace_nearby",
  target_stop_id: "bixby_creek_bridge_viewpoint",
  candidate_id: "pfeiffer_big_sur_state_park",
}, workspace).allowed, true);

assert.equal(copilotDraftability({
  type: "replace_nearby",
  target_stop_id: "golden_gate_bridge",
  candidate_id: "pfeiffer_big_sur_state_park",
}, workspace).allowed, false);

assert.equal(copilotDraftability({
  type: "move_day",
  target_stop_id: "golden_gate_bridge",
  destination_day: 2,
}, { draft_capabilities: { operations: {}, parent_targets: [] } }).allowed, false);

const combinedWorkspace = structuredClone(workspace);
combinedWorkspace.draft_capabilities.operations.keep_stop = {enabled:true, preview_executable:true};
combinedWorkspace.draft_capabilities.executable_combinations = [{operations:["keep_stop","replace_nearby"], cardinality:32}];
const combined = [
  {type:"replace_nearby",target_stop_id:"bixby_creek_bridge_viewpoint",candidate_id:"pfeiffer_big_sur_state_park"},
  {type:"keep_stop",target_stop_id:"golden_gate_bridge"},
];
assert.deepEqual(copilotCombinedEligibility(combined, combinedWorkspace, {draft:[]}), {allowed:true,alreadyStaged:false,reason:null});
const completeDraft = combined.map(intent => copilotOperationForIntent(intent));
assert.equal(copilotCombinedEligibility(combined, combinedWorkspace, {draft:completeDraft}).alreadyStaged, true);
assert.equal(copilotCombinedEligibility(combined, combinedWorkspace, {draft:[completeDraft[0]]}).allowed, false);
assert.equal(copilotCombinedEligibility(combined, workspace, {draft:[]}).allowed, false);
for (const malformed of [{}, {operations:null,cardinality:32}, {operations:["keep_stop","replace_nearby"],cardinality:"32"}]) {
  const invalid = structuredClone(combinedWorkspace);
  invalid.draft_capabilities.executable_combinations = [malformed];
  assert.equal(copilotCombinedEligibility(combined, invalid, {draft:[]}).allowed, false);
}
assert.equal(copilotCombinedEligibility([...combined, {type:"move_day",target_stop_id:"golden_gate_bridge",destination_day:3}], combinedWorkspace, {draft:[]}).allowed, false);
assert.equal(copilotCombinedEligibility([combined[0], {...combined[0],target_stop_id:"golden_gate_bridge"}], combinedWorkspace, {draft:[]}).allowed, false);
console.log("copilot draftability and atomic combined eligibility contract passed");
