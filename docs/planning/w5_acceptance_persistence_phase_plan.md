# W5 Implementation Plan: Durable Acceptance and Keep Original

**Status:** Implemented; G5 blocked on coherent evaluated-edit integration  
**Product gate:** W5 implementation -> G5 independent decision-integrity audit  
**Research boundary:** This product phase does not change `D1 -> E3.1 -> E3.3 -> E4`.

## Goal

Implement explicit, durable, revision-safe Accept and Keep original decisions for
the local Itinerary Repair Copilot. A successful decision must preserve immutable
plan artifacts, append an auditable decision, atomically advance a trip-scoped
accepted-plan pointer, survive restart, reconcile stale sessions, and return a
server-authoritative decision summary. W5 is code-complete only when both actions
work through the product API and UI; it is verified only after the G5 audit passes.

## Pipeline Context

- Parent track: `AUD-0 -> ... -> W4/G4 -> W5/G5 -> W6/G6`.
- Entry gate: current-source G4 v6 is `verified / passed`.
- Current W5 state: `implemented`; confirmed `/accept`, `/keep-original`, history,
  and rollback endpoints are active for `local_acceptance` runs. Accept remains
  eligibility-gated; G5 is pending.
- Exit gate: G5 independently verifies immutable artifacts, exact proposal and
  certificate binding, repository writes, decision history, pointer CAS,
  restart recovery, concurrency, tamper rejection, confirmation semantics, and
  server-authoritative UI refresh.
- Later dependents: W6 mobile/PWA and W7 launcher work may consume the accepted
  pointer but may not weaken this transaction.

## Entry State (Before W5 Implementation)

- `ProductSession`, `SessionSnapshotStore`, and `WorkspaceStore` persist
  token-authenticated, expected-revision sessions and ordered drafts under the
  cross-process product-state lock.
- `ProductService.preview()` persists an evaluated proposal bound to the parent,
  draft hash/operation IDs, child plan/hash, diff/hash, certificate/hash, route
  validation, evaluation state, and rendered geography.
- `PlanRepository` stores immutable `PlanArtifactV2` JSON by plan ID but its
  product use is not yet wrapped in a decision transaction.
- `.product_app_state/` already reserves `plan_repository/`, `workspaces/`,
  `sessions/`, `locks/`, and `temporary/`; W5 adds decisions and transaction
  recovery without importing legacy decision content.
- At entry, the API authenticated tokens and revisions but intentionally rejected
  acceptance. The entry UI presented read-only preview/Compare actions and
  disabled Accept/Keep controls.
- The independent acceptance audit identifies the required closure: repository
  hash/lineage revalidation, trip-scoped pointer CAS, restart authority,
  explicit Keep confirmation, proposal immutability, crash recovery,
  concurrent-writer behavior, pointer/ledger validation, and server-owned UI
  state.

## Scope

1. Accept the current session's exact eligible evaluated proposal.
2. Keep the immutable original registered parent through a distinct confirmed
   decision.
3. Persist the accepted child through `PlanRepository` before publishing it.
4. Persist append-only `AcceptanceDecisionV1` records and a separate
   trip-scoped `WorkspacePointerV1`.
5. Use a prepared transaction record and roll-forward recovery for crashes
   between plan, decision, pointer, and session writes.
6. Reconcile new/restored sessions with the authoritative pointer.
7. Reject stale, replayed, ineligible, mismatched, tampered, unconfirmed, or
   cross-trip decisions without advancing revision or changing files.
8. Return a strict server-authoritative decision response and render it in the
   Trip/Compare/Evidence UI.
9. Preserve immutable source parent/child artifacts and all G4 evidence.

## Non-Goals

- Booking, payment, cancellation, or another external action.
- Automatic acceptance from Copilot, map interaction, or preview.
- Learned ranking, recommendation, or permission escalation.
- Editing plan artifact bytes in place.
- Importing or interpreting legacy pre-W5 decision files.
- W6 mobile/PWA installation, W7 launcher replacement, or W8 final replacement.
- Research-pipeline status changes.

## Dependencies and Prerequisites

- Verified G4 v6 exact-I/O evidence and independent audits: satisfied.
- Existing `StateFileLock`, atomic JSON writer, session snapshots, and bounded
  request security: satisfied.
- Existing evaluated proposal with exact artifact/evaluation identities:
  satisfied for registered and executable typed-draft previews.
- Product state root must pass readiness and remain writable; failure is a
  sanitized 503 and never falls back to browser state.
- The selected proposal must be `eligible`, match the authenticated session and
  current accepted pointer, and recompile/revalidate to the same identities.

## New Components or Changes

### Acceptance store

Create `src/itinerary_system/product_app/acceptance.py` with strict versioned
decision, pointer, and transaction schemas. The store owns the product use of
`PlanRepository`, trip-scoped locks, append-only decision records, atomic
pointer replacement, recovery, idempotency, and readback validation.

### Workspace and service integration

`ProductService` becomes the only acceptance orchestrator. It recompiles and
revalidates the proposal, builds the immutable plan record, invokes the
acceptance store, then advances/clears the authenticated session. `WorkspaceStore`
continues to own session CAS and reconciles restored sessions to the pointer.

### API and UI

The API accepts exact confirmation envelopes. The UI requires a confirmation
dialog, disables actions while pending, handles stale/recovery responses, and
renders only the returned decision/pointer/session summary.

## Class / Registry Diagrams

```text
+-------------------------------------------------------------------------------+
|                            AcceptanceStore                                    |
+-------------------------------------------------------------------------------+
|  Category: Persistent Registry / Manager                                      |
+-------------------------------------------------------------------------------+
|  Storage owner: .product_app_state/{plan_repository,decisions,workspaces,...}  |
|  - plan_repository: PlanRepository                                             |
|  - state_root: Path                                                           |
|  - lock_timeout_seconds: float                                                |
+-------------------------------------------------------------------------------+
|  + ensure_pointer(...): WorkspacePointerV1  --> Initialize/read trip pointer  |
|  + load_pointer(trip_id): WorkspacePointerV1 --> Validate pointer authority   |
|  + accept(...): AcceptanceResultV1          --> Commit eligible child         |
|  + keep_original(...): AcceptanceResultV1   --> Record explicit keep decision |
|  + history(trip_id): tuple[...]             --> Read strict append-only ledger |
|  + recover(trip_id): WorkspacePointerV1      --> Roll prepared txn forward     |
+-------------------------------------------------------------------------------+
```

```text
+-------------------------------------------------------------------------------+
|                             WorkspaceStore                                    |
+-------------------------------------------------------------------------------+
|  Category: Runtime State Registry                                             |
+-------------------------------------------------------------------------------+
|  Lifecycle owner: authenticated product session                               |
|  - _snapshots: SessionSnapshotStore                                           |
|  - _pointer_resolver: Callable                                                |
+-------------------------------------------------------------------------------+
|  + create_session(...): (ProductSession, str) --> Start from pointer authority |
|  + authenticate(...): ProductSession           --> Reconcile then authenticate |
|  + apply_decision(...): ProductSession          --> CAS/clear obsolete state   |
+-------------------------------------------------------------------------------+
```

```text
+-------------------------------------------------------------------------------+
|                              ProductService                                   |
+-------------------------------------------------------------------------------+
|  Category: Service Class                                                      |
+-------------------------------------------------------------------------------+
|  - acceptance: AcceptanceStore                                                |
|  - workspace: WorkspaceStore                                                  |
|  - _draft_compilers: dict[str, FrozenDraftCompiler | None]                    |
+-------------------------------------------------------------------------------+
|  + accept(session,...): dict              --> Reverify and transact proposal  |
|  + keep_original(session,...): dict       --> Confirm immutable parent choice |
|  + decision_history(session): dict        --> Return trip-scoped history      |
+-------------------------------------------------------------------------------+
```

## Class Diagram Rules

- `AcceptanceStore` owns durable decision state; browser state is never an
  authority.
- `WorkspaceStore` owns session lifecycle and revision CAS, not plan artifacts.
- `ProductService` coordinates the boundaries but does not write files itself.
- `PlanRepository` remains append-only; acceptance never rewrites a plan.
- API handlers remain thin validators and do not implement transaction logic.

## Data Contracts

### `AcceptanceDecisionV1`

Exact fields:

- `schema_version = "acceptance-decision-v1"`
- `decision_id`, `idempotency_key`, `decision_type`
- `trip_id`, `run_id`, `session_id`
- `session_revision_before`, `session_revision_after`
- `pointer_revision_before`, `pointer_revision_after`
- `previous_plan_id`, `previous_plan_content_hash`
- `selected_plan_id`, `selected_plan_content_hash`
- nullable `proposal_id`, `draft_content_hash`, `diff_id`,
  `diff_content_hash`, `certificate_id`, `certificate_content_hash`
- `created_at`, `content_hash`

`decision_type` is exactly `accept_repair`, `keep_original`, or
`rollback_to_eligible_plan`. W5 UI exposes the first two; rollback remains an
authenticated API/history capability until later UX work.

### `WorkspacePointerV1`

Exact fields:

- `schema_version = "workspace-pointer-v1"`
- `trip_id`, `run_id`, `original_plan_id`, `original_plan_content_hash`
- `accepted_plan_id`, `accepted_plan_content_hash`
- nullable `decision_id`
- monotonic `revision`, `updated_at`, `content_hash`

### Requests

- Accept: `{expected_revision, confirmation:"accept_evaluated_repair",
  idempotency_key}`.
- Keep original: `{expected_revision, confirmation:"keep_original",
  idempotency_key}`.
- Idempotency keys are 16-128 characters from the closed identifier alphabet.
- Unknown, missing, extra, boolean-as-integer, or malformed fields fail 422.

### Response

`acceptance-result-v1` contains the exact decision, pointer, reconciled public
session, accepted-plan summary, and `recovery_performed`. It contains no local
path, token verifier, raw exception, or mutable artifact body.

## Storage / State

- `plan_repository/`: immutable `PlanArtifactV2` JSON and index.
- `decisions/<trip-key>/ledger.jsonl`: append-only canonical decision records.
- `workspaces/<trip-key>.json`: one atomic pointer document.
- `transactions/<trip-key>.json`: at most one prepared transaction, deleted
  after successful pointer publication; recovery rolls it forward.
- `locks/acceptance-<trip-hash>.lock`: process-safe trip transaction lock.
- Sessions remain under `sessions/`; a committed pointer is authoritative if a
  crash occurs before session snapshot update.
- Legacy files remain counted but uninterpreted.

## Required Interfaces

- `POST /api/sessions/{session_id}/accept`
- `POST /api/sessions/{session_id}/keep-original`
- `GET /api/sessions/{session_id}/decisions`
- `POST /api/sessions/{session_id}/decisions/{decision_id}/rollback`

The first two are exposed in W5 UI. History supports Evidence. Rollback is
server-only until a later reviewed interaction adds a confirmation surface.

## Validation Rules

- Token and expected session revision must match.
- Pointer trip/run and accepted plan must match the session before mutation.
- Accept requires a current nonempty draft and proposal for exactly that draft.
- Recompilation must reproduce child/hash, diff/hash, certificate/hash,
  route-validation identity, eligible comparison status, and complete validated
  geography.
- Certificate plan/hash must equal the selected child; diff parent/child must
  equal pointer/child; child parent lineage must equal the pointer plan.
- Keep original requires the exact confirmation and original plan hash.
- Idempotent replay returns the existing result only when every bound identity
  matches; reuse with different content fails 409.
- Pointer revision is monotonic. Cross-process stale writers fail CAS.
- Ledger records are unique by decision and idempotency key, hash-valid,
  finite, ordered, and pointer-consistent.
- Ineligible, missing, stale, fallback, partial, ungrounded, or tampered
  proposals fail before plan/decision/pointer/session writes.
- Successful decisions clear draft, proposal, scoped permission decisions, and
  object selections, then advance session revision exactly once.

## UI / API / Internal Integration

- Trip: Accept evaluated repair opens explicit confirmation and commits the
  current eligible proposal.
- Compare: Keep original opens a distinct explicit confirmation; Choose option
  remains preview selection, not acceptance.
- Evidence: shows decision summary and history without leading with raw IDs.
- Copilot remains advisory and cannot invoke either endpoint.
- After success, the client replaces session state only from the response,
  reloads workspace/geography/alternatives, and announces the durable outcome.
- 409 stale/pointer conflicts trigger a safe reload and never synthesize
  accepted state.

## Implementation Workflow

1. Freeze schemas, failure codes, pointer/ledger invariants, and phase plan.
2. Add strict acceptance store, trip locks, transaction marker, recovery, and
   append-only PlanRepository integration.
3. Add pointer-aware session creation/reconciliation and exact decision CAS.
4. Add service-side proposal recompilation and acceptance verification.
5. Replace API hard failures with strict endpoints and history/rollback reads.
6. Add explicit UI confirmations and server-authoritative refresh.
7. Add unit, API, restart, fault-injection, concurrency, tamper, and frontend
   tests; run focused regressions and static checks.
8. Run independent G5 artifact, acceptance, content, and web audits.
9. Only after all pass, mark W5/G5 verified. Do not advance W6 automatically.

## Files to Create

- `src/itinerary_system/product_app/acceptance.py`
- `tests/product_app/test_w5_acceptance.py`
- `tests/product_app/test_w5_acceptance_frontend_contract.py`

## Files to Modify

- `src/itinerary_system/plans/repository.py`: strict product-use save/readback
  and atomic index behavior under the acceptance lock.
- `src/itinerary_system/product_app/persistence.py`: W5 directories and strict
  atomic helpers reused by the acceptance store.
- `src/itinerary_system/product_app/models.py`: versioned decision/pointer
  public response contracts where appropriate.
- `src/itinerary_system/product_app/workspace.py`: pointer reconciliation and
  successful-decision session CAS.
- `src/itinerary_system/product_app/service.py`: exact revalidation and
  acceptance orchestration.
- `src/itinerary_system/product_app/api.py`: strict W5 endpoints.
- `src/itinerary_system/product_app/static/js/app.js`: confirmation, submit,
  recovery, and authoritative refresh.
- `src/itinerary_system/product_app/static/index.html`: replace obsolete W5
  disabled copy and bump asset cache token.
- Active status documents only after implementation/audit evidence exists.

## Error Handling and Recovery

- Every externally visible error is a stable code with no raw path or exception.
- The prepared transaction contains the exact next pointer and decision.
  Recovery validates it and rolls forward; it never guesses or rolls back an
  immutable decision.
- A plan save without a decision is harmless immutable orphan data.
- A decision append without pointer publication is completed from the prepared
  transaction on restart.
- A pointer publication without session update is authoritative; the next
  session authentication reconciles and clears stale draft/proposal state.
- Corrupt pointer, ledger, plan, or transaction bytes block that trip with 409
  or 503 and are never silently replaced.
- Rollback appends a new decision; it never deletes history or plan files.

## Observability

Reuse sanitized health/error codes and evidence files. Tests and black-box
evidence record decision ID, pointer revision, plan/hash identities, revision
transition, recovery stage, restart result, and source-parent byte hashes. No
token, local absolute path, or plan body is logged by the API.

## Testing and Verification

Required automated coverage:

- Normal Accept and Keep original.
- Missing/extra fields, bad confirmation, invalid token, stale revision.
- Ineligible, missing, stale, mismatched, and coherently tampered proposal.
- Source parent and child bytes unchanged.
- Same idempotency key replay and conflicting reuse.
- Concurrent threads and separate processes; exactly one pointer advance.
- Fault injection after plan save, transaction prepare, ledger append, pointer
  replace, and before session save; restart converges to one decision/pointer.
- Pointer, ledger, transaction, index, and plan tampering fail closed.
- New session and restored session use the accepted pointer.
- Keep original is explicit and Inspect/Choose preview remain non-mutating.
- Frontend confirmation, pending, success, stale, and failure states.
- Copilot cannot trigger acceptance; Accept/Keep API responses are `no-store`.

Commands:

```powershell
python -m pytest tests/product_app/test_w5_acceptance.py tests/product_app/test_w5_acceptance_frontend_contract.py -q
python -m pytest tests/plans/test_repository.py tests/product_app/test_product_app.py tests/product_app/test_workspace_w3.py tests/product_app/test_security_api.py -q
python -m ruff check --no-cache src/itinerary_system/plans/repository.py src/itinerary_system/product_app tests/product_app/test_w5_acceptance.py tests/product_app/test_w5_acceptance_frontend_contract.py
node --check src/itinerary_system/product_app/static/js/app.js
git diff --check
```

G5 also requires a disposable clean-state browser journey: preview -> Accept ->
restart -> accepted pointer; separate workspace -> Keep original; no console or
network failures; no provider call.

## Pipeline / Roadmap Update

- Creating this plan and beginning implementation moves W5 from `planned` to
  `in-progress`; it does not verify W5 or G5.
- Code-complete with focused checks moves W5 to `implemented` only.
- W5/G5 become `verified / passed` only after independent artifact,
  persistence, content, API/security, and black-box reviews report zero blockers.
- W6 remains planned regardless of W5 implementation progress.

## Acceptance Criteria

### Implementation complete

- Accept and Keep original are explicit, authenticated, revision-safe server
  transactions rather than hard-coded failures.
- Eligible child plan, exact decision, and accepted pointer persist under the
  documented W5 schemas.
- Restart and session reconciliation use the pointer as authority.
- UI renders only the server response and never treats Inspect/Choose/Copilot as
  acceptance.
- Rollback is append-only and limited to a previously eligible stored plan.

### Verification complete

- Focused/regression/static/browser checks pass.
- Fault injection and cross-process concurrency converge without duplicate
  decisions or pointer loss.
- Tampering fails closed with source artifacts unchanged.
- Independent G5 auditors report zero blockers.
- Active status documents say W5/G5 verified only after that evidence exists;
  W6 and research statuses remain unchanged.
