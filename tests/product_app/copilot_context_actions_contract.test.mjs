import assert from "node:assert/strict";
import test from "node:test";
import { copilotDraftability, copilotTurnIsCurrent } from "../../src/itinerary_system/product_app/static/js/copilot.js";

test("old, future, and missing chat revisions cannot stage as current suggestions", () => {
  assert.equal(copilotTurnIsCurrent({ context_revision: 3 }, { revision: 3 }), true);
  assert.equal(copilotTurnIsCurrent({ context_revision: 2 }, { revision: 3 }), false);
  assert.equal(copilotTurnIsCurrent({ context_revision: 4 }, { revision: 3 }), false);
  assert.equal(copilotTurnIsCurrent({}, { revision: 3 }), false);
});
test("addition needs exact candidate, day, parent ID/hash and directed road readiness", () => {
  const readiness = { candidate_id: "candidate", day: 5, parent_plan_id: "accepted", parent_plan_content_hash: "hash",
    preview_executable: false, blocking_findings: [{ origin_id: "stop", destination_id: "candidate", corrective_action: "Load road evidence." }] };
  const workspace = { accepted_plan_pointer: { accepted_plan_id: "accepted", accepted_plan_content_hash: "hash" },
    draft_capabilities: { operations: { add_candidate: { enabled: true, candidate_day_readiness: [readiness] } },
      candidate_choices: [{ candidate_id: "candidate" }] } };
  const intent = { type: "add_candidate", candidate_id: "candidate", destination_day: 5 };
  assert.equal(copilotDraftability(intent, workspace).allowed, false);
  assert.match(copilotDraftability(intent, workspace).reason, /stop → candidate/);
  readiness.preview_executable = true;
  assert.equal(copilotDraftability(intent, workspace).allowed, true);
  assert.equal(copilotDraftability({ ...intent, destination_day: 4 }, workspace).allowed, false);
  workspace.accepted_plan_pointer.accepted_plan_content_hash = "new-hash";
  assert.equal(copilotDraftability(intent, workspace).allowed, false);
});
