import assert from "node:assert/strict";
import {
  copilotOneClickEligibility,
  copilotOperationForIntent,
  sameCopilotDraftOperation,
} from "../../src/itinerary_system/product_app/static/js/copilot.js";

const workspace = {
  draft_capabilities: {
    operations: {
      move_day: { enabled: true },
      replace_nearby: { enabled: true },
    },
    parent_targets: ["bixby"],
    candidate_choices: [
      { candidate_id: "pfeiffer", replaces_stop_id: "bixby" },
      { candidate_id: "lobos", replaces_stop_id: "bixby" },
    ],
  },
};

const replacement = {
  type: "replace_nearby",
  target_stop_id: "bixby",
  candidate_id: "pfeiffer",
};
const replacementOperation = copilotOperationForIntent(replacement, null, 5);
assert.deepEqual(replacementOperation, {
  type: "replace_nearby",
  target: "bixby",
  parameters: { candidate_id: "pfeiffer" },
});
assert.equal(
  sameCopilotDraftOperation(
    replacementOperation,
    { ...replacementOperation, parameters: { candidate_id: "lobos" } },
  ),
  false,
);
assert.equal(
  sameCopilotDraftOperation(
    { ...replacementOperation, parameters: { candidate_id: "pfeiffer" } },
    replacementOperation,
  ),
  true,
);

const moveDayTwo = copilotOperationForIntent(
  { type: "move_day", target_stop_id: "bixby", destination_day: 2 },
);
const moveDayThree = copilotOperationForIntent(
  { type: "move_day", target_stop_id: "bixby", destination_day: 3 },
);
assert.equal(sameCopilotDraftOperation(moveDayTwo, moveDayThree), false);

assert.deepEqual(
  copilotOneClickEligibility(replacement, workspace, {
    selected_day: 5,
    selected_stop_id: "bixby",
    draft: [],
  }),
  { allowed: true, alreadyStaged: false, reason: null },
);
assert.deepEqual(
  copilotOneClickEligibility(replacement, workspace, {
    selected_day: 5,
    selected_stop_id: "bixby",
    draft: [replacementOperation],
  }),
  { allowed: true, alreadyStaged: true, reason: null },
);

const unrelatedDraft = {
  type: "move_day",
  target: "bixby",
  parameters: { day: 4 },
};
const blocked = copilotOneClickEligibility(replacement, workspace, {
  selected_day: 5,
  selected_stop_id: "bixby",
  draft: [unrelatedDraft],
});
assert.equal(blocked.allowed, false);
assert.equal(blocked.alreadyStaged, false);
assert.match(blocked.reason, /must not silently include unrelated or conflicting changes/);

const conflictingReplacement = copilotOneClickEligibility(replacement, workspace, {
  selected_day: 5,
  selected_stop_id: "bixby",
  draft: [{
    type: "replace_nearby",
    target: "bixby",
    parameters: { candidate_id: "lobos" },
  }],
});
assert.equal(conflictingReplacement.allowed, false);
assert.equal(conflictingReplacement.alreadyStaged, false);

console.log("copilot interaction state contract passed");
