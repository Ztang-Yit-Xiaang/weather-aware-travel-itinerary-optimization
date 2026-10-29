# W5 Acceptance and Persistence Implementation Report

**Date:** 2026-08-13  
**Status:** W5 `implemented`; G5 independent verification `pending`  
**Authority boundary:** this report is implementation evidence, not a G5 PASS

## Outcome

W5 now provides explicit, local, durable Accept and Keep-original decisions. A
successful decision saves or reuses an immutable plan artifact, appends one
decision to the trip ledger, atomically advances the accepted-plan pointer,
increments the session revision, and clears draft/proposal/permission state.
Rollback is a separate confirmed append-only decision. Accept is enabled only
for a current independently evaluated eligible proposal; Keep original remains
available as an explicit durable decision.

## Exact API Inputs and Outputs

All three mutation endpoints require `X-Session-Token` and reject unknown JSON
fields.

```json
POST /api/sessions/{session_id}/accept
{
  "expected_revision": 0,
  "confirmation": "accept_evaluated_repair",
  "idempotency_key": "<16-128 character caller key>"
}
```

```json
POST /api/sessions/{session_id}/keep-original
{
  "expected_revision": 0,
  "confirmation": "keep_original_plan",
  "idempotency_key": "<16-128 character caller key>"
}
```

```json
POST /api/sessions/{session_id}/acceptance-decisions/{decision_id}/rollback
{
  "expected_revision": 1,
  "confirmation": "rollback_to_prior_decision",
  "idempotency_key": "<16-128 character caller key>"
}
```

Successful mutation output is exactly the `acceptance-action-v1` envelope:

```json
{
  "schema_version": "acceptance-action-v1",
  "result": {
    "schema_version": "acceptance-result-v1",
    "decision": "<acceptance-decision-v1 exact record>",
    "pointer": "<workspace-pointer-v1 exact record>",
    "recovery_performed": false
  },
  "session": "<public session with revision advanced by one and transient state cleared>",
  "workspace": "<workspace whose accepted_plan_pointer exactly equals result.pointer>"
}
```

Read-only decision history input/output:

```text
GET /api/sessions/{session_id}/acceptance-decisions
X-Session-Token: <session mutation token>
```

```json
{
  "schema_version": "acceptance-history-v1",
  "pointer": "<workspace-pointer-v1 exact record>",
  "decisions": ["<ordered acceptance-decision-v1 records>"]
}
```

## Exact Persistent File Inputs and Outputs

Inputs are the immutable parent/selected `PlanArtifactV2`, current session
revision, current accepted pointer revision/plan identity, explicit decision
type, idempotency key, and—only for Accept—the freshly recompiled child, diff,
certificate, route legs, geography, and proposal identities.

Outputs under the configured product state root are:

- `plan_repository/plans/{plan_id}.json`: append-only immutable plan artifact;
- `plan_repository/index.json`: atomically replaced repository index;
- `decisions/{trip_key}/ledger.jsonl`: append-only canonical
  `acceptance-decision-v1` records;
- `workspaces/{trip_key}.json`: atomically replaced `workspace-pointer-v1`;
- `acceptance_snapshots/{decision_id}.json`: append-only accepted evidence
  snapshot when an evaluated repair supplies one; and
- `transactions/{trip_key}.json`: prepared recovery record, removed only after
  ledger and pointer completion.

The transaction recovery tests cover interruption after plan save, prepare,
ledger append, and pointer replace. Concurrent writers produce exactly one
decision; stale pointer writers fail. An idempotency key cannot select a
different plan, and an immutable plan-ID collision produces the stable
`acceptance_plan_repository_conflict` code before ledger mutation.

## LLM Input and Output

W5 made **no OpenAI or other LLM call**. Exact LLM input: none. Exact LLM
output: none. The previously verified G4 v6 exact-I/O evidence remains unchanged
and is not reused as W5 transaction input.

## Verification Performed

- Focused W5/repository/product/Copilot/frontend boundary: **52 passed**, one
  known Starlette/httpx deprecation warning.
- Post-browser W5 API/store/frontend regression: **15 passed**, same known
  warning.
- Browser fallback journey: **1 passed**; Keep original was confirmed, recorded,
  and restored after reload with the same pointer decision ID and no console
  errors.
- Full current-file UI/browser/Compare/geography/responsive matrix: **55 passed**.
- Additional evaluated reorder/duration/time-window/role cross-check: **66 passed,
  20 failed**. The failures are not W5 ledger/CAS failures: order tests still
  target pre-coherence benchmark IDs, while duration/time-window/role API tests
  reach the coherent parent whose Bixby stop is intentionally weather-infeasible
  and then fail closed at the older specialized route/schedule evidence binding.
  This is an explicit G5 blocker/follow-up; it is not averaged into a PASS.
- JavaScript syntax: PASS.
- Scoped Ruff: PASS.
- `git diff --check`: PASS, with only existing Windows LF-to-CRLF notices.

The in-app browser helper was unavailable because its Windows sandbox helper
exited during startup. The repository's installed Playwright/Chromium harness
was used against an isolated local server instead.

## Remaining Gate and Limitations

- G5 remains pending. No independent G5 PASS is claimed here.
- G5 cannot pass until the specialized evaluated-edit fixtures/contracts are
  reconciled with the coherent California parent and the 20-failure cross-check
  above is green or intentionally re-scoped with independent evidence.
- Accepted generated children are durable and restorable. Further typed/map
  edits on a generated accepted child fail closed with
  `accepted_plan_requires_new_repair_baseline` until a compiler baseline for
  arbitrary accepted children is implemented.
- Legacy decision-file import remains preserved for manual review; it is not
  silently trusted or merged.
- Durable records use structural validation and content hashes within the
  documented loopback single-user filesystem trust model; they are not an
  external signature or transparency log.
- W6-W8 remain planned.
