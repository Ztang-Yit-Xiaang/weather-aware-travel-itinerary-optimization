# Runtime coherence repair checkpoint — September 8, 2026

Status: **in progress; the complete E2E-1/stability gate is not passed.** This checkpoint supplements, rather than rewrites, the September 7 concern audit and retained failed runs. Work is continuing in verified batches. No research-baseline blocker is closed by these product repairs.

## Confirmed runtime progress

### September 8, 23:50 UTC — four-size layout journey independently passed

`results/e2e/ws2/` completed **4 real browser cases passed in 128.23 seconds**.
Although launched as a diagnostic while other harness work continued, its
recorded source universe remained unchanged and the full independent verifier
passed: 259 artifacts, 192 source inputs, four LLM no-call records, finalized
zero-provider isolation. Manifest content hash:
`e024c219f7e80855743635ee0c99d7b7b21a706417c434f536a6a6ba8d1da1f5`.
This is a bounded workspace-layout pass, not the full integration gate.

All recorded required draft controls and selected-day heading/stop-order elements
were fully visible and unobstructed at the asserted checkpoints. Initial map
dimensions were 651x630 (1440x900), 507x537 (1280x800), 613x505 (1024x768), and
368x592 (390x844), with projected loaded-tile coverage 1.0. Each screenshot
retains before/after stable readiness and timestamps. For example, the 1024x768
initial map was ready at 23:44:29.902Z and its screenshot started at
23:44:35.169Z, rendering parent `plan_359968d255ff9288` / `750dae617fc4c1e3`.

Independent read-only review additionally passed 78 frontend/map/responsive
tests and 20 text/presentation Node checks. It identified two bounded follow-ups:
record and actually exercise the new metric footer's intentional internal scroll,
and avoid claiming universal support for shorter or runtime-banner desktop
windows without their own checks. The four specified normal-runtime viewport
results do not establish that broader claim. Full-size visual inspection also
found underlying inspector text faintly visible through the open text panel;
an opaque panel background is a pending readability correction.

The fresh direct-edit browser suite `de1` is still running. Its completed duration
case generated and rendered child `plan_532d0d19a075b4c5` / `b6a0879dff8a445d`,
evaluation `cert_7a245c16575e2380`, geometry
`2f09f309f74b9286c6e5aa461675fbcb37f70ef33c9cd74863ccd0b06e044418`.
The actual input was `set_stop_duration` for `griffith_observatory`, exact 60
minutes, expected revision 6, source `map`. Its trace confirms the parent remains
rendered while staged, and the child is consistent after preview, reload and
explicit acceptance. Remaining direct cases and independent bundle verification
are not yet credited. Exact input/output is retained in `de1/direct_edits/duration/`.

### September 8, 23:44 UTC — staged-state repair reviewed; actual viewport failures retained

The committed-draft/impact-wait defect is corrected in production: Add and Undo
render the server-confirmed draft immediately with impact checking in progress.
Preview remains disabled until the current impact response is ready. Unique
request ownership and exact session/run/trip/revision/accepted-parent checks
reject older success and failure responses. Add, Undo and Preview continuations
also recheck ownership after awaiting impact. The independent source reviewer
cleared the four reported race risks at app.js SHA-256
`c0c8e65527401d972e6d24c0df9eb27a9b00ec017c4679a9bd0e662dc7e7e9ca`.
The subsequent text-panel ordering edit changes that hash and still needs fresh
browser verification; this is not a full-phase completion statement.

Retained focused evidence: 44 Node tests passed; the normal Python wrapper first
failed decoding Node output with Windows GBK, then passed 9 focused tests with
explicit UTF-8 (one real-backend test intentionally deselected). Exact outputs
are `staged_impact_final_focused.txt`, `staged_impact_pytest.xml`,
`staged_impact_pytest_utf8.xml`, and `staged_impact_fixture_final_focused.txt` under
`results/repairs/runtime_concern_batch_20260908/`. The conversation creation/bind
lock correction passed 30 focused tests; POI/workspace harness corrections passed
258 focused tests. These are not substitutes for browser journeys.

The strengthened four-size real browser run `results/e2e/ws1/` is **RED: 1 passed,
3 failed in 91.31 seconds**. It measured Preview visibility at 62.5% for 1280x800
and 31.8% for 1024x768; mobile selected-day stop-order visibility was 0%. Exact
HTTP bodies, screenshot readiness, clipping rectangles, hit tests and scroll
owners are retained in its viewport directories. Provider isolation finalized
with zero calls. No failure or screenshot was rewritten.

A narrow layout correction is now under diagnostic browser retest in `ws2`:
selected-day route details precede the long general description; the opened text
panel uses a bounded wider overlay; short desktop windows use 8px workspace
padding, a 500px minimum map and a 110px maximum metric footer. The metric footer
and long text details intentionally scroll internally. Normal-height desktop
map sizing is unchanged. This trades a slightly shorter short-window map for
fully visible user controls; measured dimensions and final acceptance remain
pending. All visibility and 95% tile-readiness requirements remain unchanged.

The new missing/foreign preview-geography fault gate has been registered but is
not yet validated: current proposed inventory is 20 suites / 50 browser cases.
The complete deterministic aggregate, complete product regression and separately
authorized one-attempt paid website journey plus exact-output replay remain open.
No new billed provider call has been invoked in this continuation.

### September 8, 23:25 UTC — two runtime causes confirmed; targeted repairs in progress

Product regression v2 was deliberately interrupted after five fully completed,
source-unchanged batches: **676 passes** (144 + 128 + 165 + 139 + 100). Batch six
started but has no final result. The full run remains incomplete and cannot be
credited as a 1,325-test pass. No owned batch-runner/pytest child remained after
the interruption. This releases source for the confirmed browser repairs;
fresh final-source regression remains required.

Two separate real-browser diagnostics, `results/e2e/dg1/` (passive) and `dg2/`
(explicitly debugger-timing-perturbed), reproduced the direct-edit failure.
Both used unchanged served source, exited their servers normally and retained
finalized zero-provider receipts. The debugger shows the ownership guard
passing and the committed order draft installed at revision 7 while the DOM
still says zero drafts. `addDraft()` then waits for impact loading to finish
before its first visible refresh. The five-second assertion fails during that
wait. Thus the hypothesis of a discarded draft response is **not** supported
by this trace; delayed visible synchronization is the demonstrated cause.
The planned correction immediately shows the server-confirmed draft with
impact-checking state and Preview disabled, then renders ready/failed impact.
The equivalent Undo structure is included as an adjacent source-confirmed risk.

Exact diagnostic input: `set_stop_order(sequoia_national_park, day=4,
sequence_index=1)`, source map, expected revision 6. Output: HTTP 200, revision 7,
operation `operation_c4111bb778e34122b586ac68fbb37079`; final real readback agrees.
The accepted map remains `plan_bc13d7ed8c7ec60b`. Complete input/output and
ownership observations remain under `dg2/browser/order/`; this red diagnostic
is not clean acceptance evidence.

The original preference/reset failure has a second concrete cause. Two real
service counterexamples—concurrent first reads and observation racing a read
after deletion—both reproduce `409 conversation_session_mismatch` and two
persisted conversation records. The existing creation and binding steps are
not atomic with concurrent authentication. Proposed repair uses the existing
workspace RLock, makes observation non-creating and waits for the actual
post-delete website restore response. Retained red: two failures in 12.76
seconds, exact I/O in `conversation_binding_red_v1/`, no network attempts.
This targets the demonstrated single-service threading race; no cross-process
transaction guarantee or green browser result is claimed yet.

### September 8, 23:12 UTC — aggregate g1 failed; remaining suites were not run

`results/e2e/g1/` finished with exit **1**, unchanged source, and
`assertions_passed=false`. It executed 31 browser cases: 20 individual passes,
11 failures, zero skips. Only six cases belong to fully verified passing
child suites. The other 17 registered cases in 11 later suites were **not run**,
not passed or silently excluded. Aggregate content hash:
`30c55fa64c88a90c53f9f0ad2f374fbf4135aef05e85166fe72c207545f80460`.

Conversation finished with six passes/four failures. The failures are original
preference/reset, accepted-child delete-all, accepted-child byte capacity and
accepted-child recovery. Four failed lifecycles retain
`e2e1_guard_lifecycle_incomplete`; the aggregate stopped with
`e2e1_self_lifecycle_isolation_incomplete`. Its conversation provider count is
therefore unavailable, not a certified zero. No live billed-provider command
was invoked. Exact requests, outputs, browser failures and incomplete receipts
remain untouched under the failed bundle.

The port was checked free after aggregate exit. A separately labeled passive
direct-edit diagnostic is next, to capture the transient error, response
ownership state and final server readback for the committed-but-unrendered
draft. It does not count as a clean acceptance rerun. Product regression v2
continues separately with source unchanged; it is not yet a complete pass.

The unapplied POI context correction passed 63 prospective verifier regressions
in 4.67 seconds by loading patched modules in memory, with all four bound
source files unchanged (`poi_prospective_v1.xml`). This is prospective unit
evidence only. The workspace timing/visibility correction also remains
unapplied; its new visible-content requirements must be enforced by the
independent verifier as well as by browser assertions.

### September 8, 22:58 UTC — additional bounded passes and explicit geometry-fault coverage gap

The fresh aggregate's addition-continuation journey passed one browser case in
174.92 seconds plus its full semantic verifier: 107 artifacts, 193 source
inputs, zero provider calls. Mobile chat passed one browser case in 73.56
seconds plus full verification: 52 artifacts, 192 source inputs, zero provider
calls. These do not cancel the aggregate's already-recorded failed cases.

All 33 standalone frontend test files passed **234 cases, zero failures/errors
or skips**, exit zero (`node_current_20260908_v4.xml`). Inputs match
`selection_node_independent_review.md` except the older-response payload now
comes from this aggregate's actual
`g1/suites/completed-response-race/completed_preview_race/preview_a_response.json`.
Other retained API payloads remain explicitly replay inputs, not fresh browser
evidence. The two API-input-generating Node wrappers remain separately covered.

Independent completeness review identified one unclosed literal browser gate:
an otherwise eligible backend result with missing or foreign route geometry.
Contract/controller guards exist, but they do not substitute for the required
real-browser blocked-parent/disabled-Accept journey. A narrowly labeled private
backend-fault test is being prepared outside frozen source. Real OS virtual
keyboard behavior also remains untested; mobile viewport simulation is not
credited as hardware keyboard testing.

### September 8, 22:54 UTC — full browser run exposes remaining failures; no completion claim

The frozen 19-suite / 48-case aggregate under `results/e2e/g1/` is still
running and already cannot pass. Its first vertical slice produced six passes
and one discovery-context failure. Exact retained HTTP shows that closing
replacement discovery clears the target; generic reopening correctly requests
insertion and returns no candidates. The test incorrectly tried to select a
candidate from the earlier replacement response. A pending test/verifier
correction will explicitly verify that reset and re-enter replacement through
real controls; no source has been changed during this aggregate.

The completed-response race suite passed one case and full verification, and
map recovery passed three cases and full verification. Workspace stability
then produced two passes and two failures at 1280/1024 during a detached DOM
scroll action. All four direct-edit cases failed after staging while checking
the visible incompatible-family state. Those failures are under independent
diagnosis, not assumed to be test-only problems. Their exact outputs remain in
the suite directories. Remaining suites continue to collect failure evidence.

The separate optional-Evidence bundle `results/e2e/of2/` has now also passed an
independent full verifier: 74 artifacts, 197 current source inputs, seven
map-ready captures, two finalized private server lifecycles and zero provider
calls. This remains a bounded fault-and-recovery result, not the whole gate.

Product batch v1 is retained as an incomplete failed run. Its third batch had
163 passes and two obsolete Compare-copy assertions, independently reproduced
as two failures/12 passes. The test-only correction now checks the actual
conditional default-selection, explicit-acceptance and current-baseline labels;
all 14 affected checks pass. A fresh complete product run is underway under
`results/repairs/runtime_concern_batch_20260908/product_batches_v2/`, using
`tmp/p2/`. Its first three batches passed 144, 128 and 165 tests respectively
with unchanged source. Neither this run nor the browser aggregate has a final
passing result. The authorized paid website call and exact-output replay remain
pending; zero billed calls have been made in this continuation.

### September 8 — frozen harness and registered optional-Evidence browser pass; complete runs underway

The complete frozen acceptance-harness suite passed **1,499 tests in 76.20
seconds**, with no failed or skipped tests (`gate_units_v8.xml`). This includes
the final aggregate cleanup/final-verification safeguards, separate real
selection-response bindings and optional-Evidence fault contracts.

The fresh registered browser run `results/e2e/of2/` passed **1 test in 86.62
seconds** and its full in-run verifier. The source-bound manifest reports
74 artifacts, 197 source inputs, unchanged source and zero provider calls;
content hash `76afa302cb2718f00cc1053a269eec3b9d63d2e2414a3de8bac06d2fe5f587cf`.
It requires the corrected rendered Compare labels, seven map-ready full-size
captures, exact canonical accepted artifacts and two clean private-server
lifecycles. A separate reviewer is re-verifying the bundle. This bounded pass
does not complete the full interaction gate.

The complete **19-suite / 48-case** deterministic browser aggregate is running
under `results/e2e/g1/`. The independently collected **1,325 product tests / 95
modules** are running through the fresh-process batch driver under
`results/repairs/runtime_concern_batch_20260908/product_batches_v1/`, with
temporary artifacts under `tmp/p1/`. Neither long run has a final result yet.
The authorized one-attempt live website-chat test and exact-output replay are
still pending; no billed call has occurred in this continuation.

### September 8 — real optional-Evidence diagnostic passed; screenshot exposed a baseline-label defect

The standalone real-browser diagnostic in `results/e2e/of1/` passed one test in
83.85 seconds. It used a private project copy, real local APIs and browser
controls, and no provider invocation. Both private server generations stopped
normally. The browser test's SHA-256 was unchanged before/after:
`5bf213d2d8ffbbc670fda0cf5377e08c8aa4c08fcd1c8babbc529b6183b41764`.
This direct diagnostic has no formal outer manifest and is **not** a complete
source-bound gate pass. The seven full-size screenshots each retain their
map-readiness receipt.

Exact input: replace Bixby with Pfeiffer Big Sur State Park, Preview, Accept,
reload, omit only the optional low-driving comparison's evidence projection in
the private server, select that real option, open global Evidence, click Back,
and return to Trip. Output: `compare_option_evidence_unavailable` is visible for
that optional option; the accepted child remains `plan_bc13d7ed8c7ec60b`, content
hash `ab610263c7f5a68f`, certificate `cert_90e853fe0e27d774`, and geometry hash
`d0a3d9a141aef73782d1843541a9c57a3543a5ca10b9b92ba40eea409f7bbc39`.
The accepted plan files remain byte-for-byte unchanged. Exact HTTP, fault
source before/after, protected file bodies and trusted clicks are retained in
`results/e2e/of1/optional_evidence_fault/`. Fault injection is explicitly
labeled, not presented as naturally missing production data.

Visual review of the real Compare screenshot exposed a separate copy defect:
the correct accepted-child baseline was labeled `Original route`, and its
disabled reference card was titled `Keep original`. These labels are now
`Current trip route` and `Current accepted trip`. The actual Keep original
action still means returning to immutable original A and was not changed.
The introductory text now distinguishes default selection from preserved
selection and explicit acceptance. Retained label regression: 1 failure out of
9 before the fix, then 9 passes with zero skips (`compare_baseline_label_red.xml`,
`compare_baseline_label_green.xml`, `compare_baseline_label_green_v2.xml`).
The final registered browser run must verify the corrected rendered labels;
the diagnostic screenshot deliberately remains unchanged historical evidence.

### September 8, 22:06 UTC — current frontend checks pass; aggregate false-pass repair in progress

All 35 current JavaScript test files have been exercised. The 33 standalone
files passed 233 tests with zero failures/skips (`node_current_20260908_v3.xml`),
using the explicitly labeled retained API capture inputs listed in
`selection_node_independent_review.md`. The remaining two files passed through
their real API input-generating wrappers in 34.90 seconds
(`node_api_wrappers_current_v4.xml`). The latter reported one existing
Starlette TestClient deprecation warning. Production Ruff and syntax checks
for all five shipped JavaScript files passed. These are not fresh browser or
live-provider acceptance claims.

Independent collection retained the exact 1,325 product-test node inventory
across 95 modules in `product_collection_v1.txt`. The full memory-bounded
execution has not started yet.

The accepted-child unavailable-Evidence branch cannot occur through normal
controls with a valid accepted snapshot. A narrowly labeled private-server
optional-comparison-evidence fault test is being added instead; accepted plan,
evaluation, pointer, ledger and geometry must remain intact. No client fetch
mock or fabricated accepted artifact is allowed. Separate real session and
workspace responses must be independently bound by the selection verifier.

The independent aggregate review found a final-suite cleanup false-pass: the
launcher could mark a child passed after a failed port-release check, although
the separate aggregate verifier would reject it. Its fail-closed correction
and negative regression are in progress before final runtime execution.
No billed provider request has been made.

### September 8, 21:57 UTC — duration/retention fixes verified; complete harness still under validation

The two production counterexamples now have retained red/green evidence:
ambiguous compound, alternative, signed or ranged visit durations clarify before
provider dispatch, and transcript expiry removes the matching dispatch journal.
The focused final batch passed 121 checks with zero live provider calls. Exact
19 synthetic request/response records are retained under
`results/repairs/runtime_concern_batch_20260908/duration_retention_green_v2/`;
the matching JUnit is `duration_retention_green_v2.xml`. Fake-SDK checks are not
live OpenAI evidence.

All 81 interaction binding groups now have implementation paths, but the
accepted-child unavailable-Evidence branch is still being investigated. Static
mapping is not runtime acceptance. The expanded browser registration currently
contains 18 suites and 47 cases; their fresh aggregate has not run.

The complete acceptance-harness unit run retained **1,426 passes and 2 failures**
in `gate_units_v6.xml`. One was a stale 43-case expectation, corrected to the
actual 47-case registration. The other was Windows error 5 while atomically
replacing a provider-isolation receipt. The guard correctly latched
`e2e1_telemetry_write_failed` and `finalized=false`; the failed receipt remains
under `tmp/h6/test_plain_http_loopback_remai5/`. Its inherited ACL was inspected
read-only. Both affected modules passed **126 checks** in a fresh process
(`gate_failures_fresh_v7.xml`) without changing isolation behavior or ACLs.
This does not establish the transient file-sharing cause or erase the red run.

Final-source product batches, the actual complete browser aggregate, one
authorized paid website-chat attempt, and exact-output replay remain pending.
No billed call has been used in this continuation.

### September 8, 21:12 UTC — monolithic regression interrupted; exact timeout case passes fresh

Broad product regression v4 was interrupted after severe memory pressure and is
**incomplete, not passed**. It produced no final JUnit file. Its retained exact
accept/add/preview sequence returned HTTP 504 `preview_execution_timeout` at
revision 4; no failure was removed. The owned test process was confirmed stopped,
and free physical memory increased from a nearby 511,652 KiB observation to
15,360,880 KiB. No application server was stopped. Full command, observations
and exact artifact path are recorded in
`results/repairs/runtime_concern_batch_20260908/product_broad_v4_interruption.md`.

The identical test, `test_real_api_accept_pfeiffer_add_point_lobos_accept_restart`,
then passed in a fresh process in **35.16 seconds**, with unchanged production
source and deadlines. Complete exact I/O is retained under `tmp/rdeltafresh/`;
JUnit is `addition_timeout_fresh_v1.xml`. The run emitted one third-party
Starlette TestClient deprecation warning, not a test failure. Resource pressure
is a plausible explanation for the earlier timeout, not a proven causal result.
The final full product regression will run all modules in smaller fresh-process
batches without changing assertions, evaluator gates or timeout limits.

The duration/retention counterexample batch retained **9 failures / 8 passes**
with 17 exact synthetic I/O records before fixes. It includes both adapters,
fake-SDK pre-dispatch assertions, exact-duration positive controls and the
expiry-boundary journal. Production changes are now released for these two
minimal fixes. Accepted conversation controls and generated proposal controls
still require their fresh source-bound browser execution; no billed call ran.

### September 8, 21:02 UTC — actual waypoint proof; two new semantic/retention defects remain open

Waypoint v6 passed its actual browser journey in 80.24 seconds and a separate
fresh-process independent verifier: 61 artifacts, 197 source inputs, zero
provider calls. Evidence: `results/e2e/r20_waypoint_pointer_20260908_v6/`.
Real canvas/panel release, addition, marker drag, confirmed relocation, both
Undo actions and reload retained accepted plan `plan_359968d255ff9288`, content
hash `750dae617fc4c1e3` and geometry hash
`75874f1813cfdcc2236358a45815b09005c534959411ac50028af49cdc824d92`.
Actual Undo impact requests used revisions 5 and 6 and returned HTTP 200; final
draft count was zero. The arbitrary waypoint remained explicitly route-checked
draft-only, not an evaluated or accepted itinerary change. Root inspected the
actual after-Undo full-size screenshot and exact pointer/map records.

V5's browser passed but its separate semantic verifier failed because the
renderer-only `display_role` decoration was included in a canonical hash.
The correction validates exact role and identity before normalizing that one
decoration; geometry and remaining canonical fields still participate. Two red
counterexamples and 124 focused green checks are retained. V6 is fresh evidence
for its recorded sources, not a substitute for the final whole-source aggregate.

This exposed a runner reporting defect: it could print verified after manifest
checks without invoking all semantic checks. The launcher now calls the complete
independent verifier before reporting success. A failure retains exact reason,
original pytest exit code and nonzero runner status in a bound failure artifact
and regenerates the final manifest/report as failed. **93 affected runner tests
passed in 10.74 seconds**. The preceding 92-pass/1-fail run retained an obsolete
single-context map fixture; it now requires both original and accepted contexts.
Evidence: `runner_semantic_gate_v1.xml` through `runner_semantic_gate_v3.xml` in
`results/repairs/runtime_concern_batch_20260908/`.

The final adversarial implementation review found two more genuine defects.
`Spend 1 hour 30 minutes at Stearns Wharf.` currently becomes an exact 60-minute
deterministic proposal; `Set Stearns Wharf duration to 60 or 90 minutes.` becomes
90 minutes instead of clarification. The valid 90-minute control remains correct.
No incorrect paid response is claimed: these are zero-provider executions of
the actual deterministic adapter and a shared preflight risk. The smallest safe
fix is explicit clarification for unresolved multi-quantity/ambiguous expressions
before proposal or paid dispatch; counterexample regressions are in progress.

The transcript append-at-expiry branch also deletes its transcript without the
paired dispatch journal. A clock-boundary in-memory reproduction recreated the
same empty conversation with one old completed outcome still recoverable.
The planned narrow correction matches existing expiry cleanup and deletes both
records. This does not imply ordinary explicit deletion is broken. Both defects
remain open at this checkpoint; production edits are held briefly until the
already-running frozen-source broad regression finishes.

Generated proposal-button and accepted-context conversation-control evidence is
still being completed. No final aggregate or paid website sample has run yet.

### September 8, 20:48 UTC — recovery guard review and registered race evidence

The registered conversation suite v3 retained **5 passed / 1 failed** in
251.67 seconds with unchanged source inputs. Its new delayed-selection/delete/
new-chat case passed both the actual browser journey and the independent exact
HTTP/SSE/state verifier. The unrelated preference-memory case failed on a test
assertion comparing an in-process empty tuple with a JSON-style empty list; the
whole suite remains failed until a fresh run. Evidence remains unchanged under
`results/e2e/conversation_generation_registered_20260908_v3/`. The exact input
was `What preferences do you remember?`; the reply preserved the new conversation
and stated that no proposed preferences existed. There were zero external
provider calls in this case, not a live OpenAI sample.

The two-context bootstrap browser run completed **2 tests in 155.35 seconds**
and eight real server lifecycles, but its independent verifier rejected a
full acceptance-record hash compared incorrectly with a shortened artifact hash.
`results/e2e/boot_ctx_20260908_v1/` remains a failed, unmodified run. The corrected
verifier uses the actual full canonical acceptance hash and stronger restored
session/pointer/geography guards. Its focused tests passed 95 checks; a separate
scenario-only diagnostic verified the retained actual inputs. That diagnostic
is not a new current-source acceptance bundle. A fresh registered run is required.

Independent review also exposed missing restored-session identity rejection in
the build-restart verifier. The correction now checks session/trip/run ownership,
accepted pointer ID/hash, full canonical geography equality, and recomputed
geometry/marker bindings. The portable regression passed **67 checks in 1.33
seconds**; counterexamples and exact mutated inputs are retained in
`results/repairs/runtime_concern_batch_20260908/build_reload_guard_audit.md`.
Production acceptance rules were not weakened or changed for either verifier fix.

The map Retry journey is now prepared for both original and accepted-child
contexts, plus the separate deliberately incomplete-tile case. The existing
configuration fault is explicitly injected only before Retry; Retry retrieves
the real healthy configuration. Trusted click, exact response, canonical plan,
and readiness ordering are required. **112 focused checks passed in 4.42 seconds**;
three browser cases collected, but actual execution and final coverage
registration remain pending. These are prepared tests, not a runtime pass.

All five production JavaScript modules passed syntax checks, and scoped Python
lint passed. Broad nonbrowser product regression v4 continues against frozen
production sources. Remaining control journeys, the complete source-bound
aggregate, and the authorized one-attempt paid website sample with exact-output
replay remain open. No billed call has run in this continuation.

### September 8, 20:19 UTC — race browser green; two-context real restart green

The real delayed-selection/delete/new-chat sequence now preserves the replacement
conversation on screen and in persisted state. Root independently inspected
`results/e2e/selection_conversation_generation_green_20260908_v1/runtime_trace.json`,
its finalized isolation receipt and the unchanged full-size after screenshot.
Exact chat input remains `What preferences do you remember?`; the visible reply
states that no proposed preferences exist for this accepted plan and asks what
itinerary change to review. Conversation D is
`conversation_1009580b97784acab212a8473fbbe2a5`, revision 1; accepted plan
`plan_359968d255ff9288`, hash `750dae617fc4c1e3`, marker set and geometry remain
unchanged. The actual deterministic adapter sent zero provider calls and shut
down with finalized isolation. This is a bounded green browser reproduction,
not yet the final registered source-bound aggregate or live OpenAI evidence.

`updateSelection` now re-reads the authoritative session when the conversation
identity changes or a newer revision overtakes its response. Owner/revision/
conversation checks also protect that asynchronous read. Reconciliation is
bounded; a failed refresh says the selection was saved instead of falsely
claiming the committed write failed. No session fields are merged or guessed.
Frozen app.js SHA-256:
`1ca59158d61da55484e5d56e65b0a77b9c2ca7459a0a043c42437b1fb572c5c3`.

The independent Node review exercised all 35 current Node files: 233 standalone
checks passed without skips, plus the two required real-service fixture wrappers
passed. Some controller inputs are retained historical browser/API artifacts,
explicitly identified in `selection_node_independent_review.md`; this is not a
new browser acceptance claim. A missing actual readiness helper in the lifecycle
test harness was corrected without changing production readiness rules. The
independent reviewer found no further defect in the bounded reconciliation cases.

The existing real-build restart journey now exercises both original and accepted
trip contexts. Both passed in **107.28 seconds**:
`results/e2e/build_contexts_20260908_v1/pytest_junit.xml`. Actual stale-page edits
returned 409 without durable-state mutation; explicit trusted Reload restored
the same session and canonical plan. A separate private in-process source drift
returned 503 and remained latched after exact file restoration. Exact HTTP,
state hashes, reload input, generation lifecycles and readiness-gated images are
under `build_coherence/original/` and `build_coherence/accepted_child/`. This
standalone browser run precedes final registration verification. The first
registration unit batch retained 44 passes / 5 failures while shared registration
was being edited (missing new verifier input and evolving inventory expectations);
it is not recorded as a complete harness pass.

Broad product regression v4 is running against frozen production code. Recovery
and control evidence extensions are still being independently checked. The full
runtime gate and the paid website sample/replay remain open; no billed call ran.

### September 8 — real conversation-generation failure and selection-hint correction

The previously source-only conversation ordering concern is now reproduced through
actual website controls and unchanged HTTP response bodies in
`results/e2e/selection_conversation_generation_red_20260908_v9/`. The user selects
Day 2, deletes conversation C, starts conversation D, and sends the exact text
`What preferences do you remember?`. Releasing an older completed selection
response then clears the visible message, although the backend still stores D
and its one turn. Both responses have itinerary revision 1, so a revision-only
guard cannot distinguish the obsolete conversation identity. The intended browser
assertion failed (1 failed in 23.22 seconds); repair and a fresh green run remain
required. The accepted plan and geometry did not change, tile coverage was 1.0,
and console/network errors were zero. This was the deterministic adapter with
zero external provider calls, not paid OpenAI evidence. The red lifecycle receipt
truthfully remains incomplete after the failed assertion. Earlier v1-v8 attempts
remain setup/harness failures and are not product reproductions.

Exact HTTP bodies, chat input/SSE output, persisted conversation states, ordering
checkpoints and readiness-gated before/after screenshots are retained in that
directory (`http_index.json`, `exact_chat_io.json`, `runtime_trace.json`,
`checkpoints.json`). No saved response was altered to produce the ordering.

The separate preview-withdrawal selection hint correction is implemented across
Add, Undo and confirmed waypoint staging. It restores valid accepted stop/segment
inspector hints only when a bound eligible preview is withdrawn; intentional
Compare inspection and remaining draft-waypoint selection are preserved. Retained
Node counterexamples showed 8/20 failures and then 2/22 confirmed-waypoint failures.
The final targeted batch passed 48 checks with no failures/skips, and independent
selection-receipt, delayed-response and wire-serialization checks passed 75.
Evidence: `selection_hint_green_v4.xml` and `chat_selection_receipts_v2.xml` under
`results/repairs/runtime_concern_batch_20260908/`. The existing two-case chat browser
journey now verifies actual Undo sessions, accepted geography, map/timeline
selection and editor targets, but its new browser run is still pending. These
focused passes do not close the complete runtime gate.

Remaining work includes the demonstrated conversation race repair, real recovery
and control journeys in the required accepted-plan contexts, coordinated final
source-frozen regression/aggregate runs, and the separately authorized one-attempt
paid website sample plus exact-output replay. No September 8 billed call has run.

### September 8, 19:43 UTC — complete focused concurrency rerun passed

With API, service and workspace sources held unchanged during execution, the
entire concurrency module passed **17 tests in 151.20 seconds**. Evidence:
`results/repairs/runtime_concern_batch_20260908/concurrency_root_frozen_v1.xml`,
with exact request/response, stage, heartbeat, worker and cancellation records
under `tmp/r22rootfrozen/`. This includes both event-controlled cancellation-
during-submission cases, full-worker capacity release, stale/dropped publication,
responsive preflight I/O, timeout and the real eligible retry. It supersedes the
overlapping focused rerun for these recorded sources, not the retained broad
failure's unknown historical cause. Full broad regression remains separate.

The backend selection correction also has a remaining frontend connection under
repair: preview rendering sets a child-plan selection hint; Undo installs the
correct accepted session but previously retained that hint. A valid accepted
stop can consequently be rejected by the editor's local selection lookup.
The correction must preserve intentional Compare inspection and remaining
draft-waypoint selection. Its Node counterexample and the real chat/Undo browser
selection assertions are still pending here; backend checks alone do not close it.

### September 8, 19:38 UTC — focused repairs verified; final browser gate remains open

The preview-invalidation selection correction passes **56 focused checks in
238.257 seconds**, with no failures, errors or skips
(`selection_invalidation_green_v3.xml` in the repair-batch directory). Undo,
appending a new operation and clearing a preview now reconcile stop/day/segment
selection against the actual accepted plan. Original-parent preview-only Pfeiffer
selection is explicitly cleared; accepted-child Pfeiffer selection remains valid.
Remaining draft operations are not treated as a second unofficial plan. Ordinary
draft-waypoint selection is preserved when no eligible preview is withdrawn.
Failure checks confirm that invalid selection context or failed persistence does
not partially commit the draft mutation. The canonical geography resolver is
invoked only at preview invalidation, not on ordinary authentication reads.
The existing chat browser journey still needs fresh selection-bound evidence.

Current service/workspace SHA-256 values are respectively
`0cfe8ea072f89546ce989229210eb5be779b2c6262305d43e8eefed910f608b5`
and `7f78bb744005817e7252d374edebc3e23b4e1ac0963276c5199966a18066575a`.
Exact synthetic inputs/outputs remain in the focused test directories and XML
records; these focused results are not substituted for browser acceptance.

Executor submission now runs off the event loop and registers worker-slot
release before the awaiting request can observe its result. The focused initial
compatibility group passed **51 checks in 195.530 seconds**. The later test-only
revision uses an explicitly released event instead of a 0.5-second automatic
submission release; the separate responsiveness counterexample remains unchanged.
Its overlapping rerun retained **15 passed / 1 failed / 1 setup error** because
production files changed during execution. Both new cancellation-during-submission
cases passed, but the complete concurrency rerun is still required after the
global source freeze. The retained red report is `tmp/r22_event_pause_freeze.xml`.
API SHA-256: `f40492cab630ab90b0a3d0da86ffb1cf212e2504f0052ef248d8ca97ad2233eb`.

Harness v5 likewise retained **905 passed / 1 source-binding failure** during
concurrent editing (`harness_regression_v5.xml`). This is not a current-source
pass. All final whole-suite/browser runs remain held until their complete source
universe is frozen. The two reviewers interrupted by a Codex usage error have
been resumed after the user's renewed Proceed; no reset credit was consumed.

A separate source-level risk is being checked in the actual browser: conversation
deletion changes conversation identity without changing the itinerary revision,
so an older selection response could potentially reintroduce the deleted identity.
The reachable ordering is identified, but no browser reproduction or correction
is claimed at this checkpoint. Ordinary selection-to-Undo operations already wait
for pending selection; no hypothetical newer-draft rollback is claimed.

Remaining work: actual missing control/Retry journeys, that bounded conversation-
identity check, source-frozen pointer/chat/candidate/waypoint runs, the complete
deterministic aggregate and broad regression, then the separately authorized paid
website-chat sample and its exact-output replay. **No new billed provider call has
occurred. Research statuses and the evaluator/route/map-readiness gates are unchanged.**

### September 8 — final-review failures retained; source freeze still pending

Further independent checks reproduced an invalid shared selection after Undo:
the session still selected preview-only Pfeiffer while its accepted parent
contained Bixby. Copilot filtered it to null, but that consumer-side filtering
did not repair the authoritative state. Reconciliation against accepted canonical
content is being added at preview invalidation, preserving still-valid selections.

The paid completion-unknown correction now passes **97 zero-provider checks in
19.72 seconds**, including timeout, cancellation, pre-dispatch rejection, complete
response and rejection of a second attempt. The red two-case reproduction and
green exact fake transport I/O are retained (`live_completion_flag_red_v1.xml`,
`live_completion_flag_green_v1.xml`). This is not new OpenAI connectivity evidence.

The complete runner/verifier batch v4 retained **813 passed / 2 failed in 128.49
seconds**. Both were outdated test expectations: the old September 7 bundle must
be rejected by today's source universe, and the explicit `OMITTED_SECRET` marker
is permitted while real sensitive values are not. Test-only corrections retain
historical source rejection and add adversarial nested/encoded/near-marker leak
checks; **134 focused checks in 7.11 seconds** pass. No historical manifest or
security/source guard was weakened. A complete harness rerun is underway.

Broad product regression v3 completed with **1,155 passed / 1 failed in
2,421.82 seconds**. The failure is the cancellation scenario's precondition:
neither compiler worker had entered its controlled pause before the deadline
check resumed after 20.703 seconds. Cancellation had **not yet been requested**;
the later cleanup-time publications therefore do not prove cancelled work was
published. The independent two-worker responsiveness check measured 0.187
seconds, and the timeout-abandonment case passed. The remaining pre-compiler
delay is under diagnosis; no timeout or capacity assertion is being relaxed.
Exact result: `results/repairs/runtime_concern_batch_20260908/product_broad_v3.xml`.

Chat-closure v4 completed **2 real browser cases in 663.03 seconds**, including
the held old conversation response delivered after Undo in both original and
accepted-child contexts. However, its overall evidence gate is **not passed**:
`scripts/verify_e2e1_self_managed.py` changed during the run, and the source-binding
guard correctly rejected it. The unchanged red bundle contains 701 artifacts,
194 recorded source inputs and measured zero provider calls:
`results/e2e/r04_chat_closure_20260908_v4`, manifest content hash
`e928ac28fbbd3f52774562f88376a853560f534c35f51bd3564392508de9b1fb`.
Its exact chat/SSE/HTTP and both `chat_window_preview/late_conversation_undo_race.json`
witnesses remain available; a fresh frozen-source run is required.

Waypoint-pointer v4 remains **red: 1 failed in 72.29 seconds**, 36 artifacts,
197 sources and zero provider calls. The native canvas release, native marker
release, addition and relocation completed. Undo returned the correct revision
5 and one remaining original operation, but the test inspected the old overlay
before waiting for the matching impact response/redraw. The retained run does
not establish persistent stale rendering. The corrected test waits for the exact
session revision's impact response and visible draft count, and writes full
expected/observed overlays before assertions. Its 117 focused checks and Ruff
pass; the fresh complete pointer journey remains required. Evidence:
`results/e2e/r20_waypoint_pointer_20260908_v4`, manifest content hash
`1069ac079dedfe1b639145303cc478cbdcdd9944bd03fe97869be922474723c2`.

The paid/replay harness now checks the complete approved synthetic context
before transport, preserves exact request/response and failure bodies, measures
actual mutation/acceptance observations, invokes the complete replay verifier,
and uses finalized replay provider-isolation receipts. The first focused group
passed 51 checks; independent review passed 92 checks and reproduced one further
reporting defect with a fake transport: an authorized transport timeout was
incorrectly labeled `provider_completion_unknown=false`. That reporting repair
is underway before any paid call. The exact fake request/failure is retained in
`results/repairs/runtime_concern_batch_20260908/paid_review_timeout_completion_v1`.
Configured-rate cost estimates are not invoice-confirmed charges. **No actual
provider call has occurred in this September 8 continuation.**

### September 8 — fresh website checks and pre-dispatch evidence audit

Waypoint-pointer v3 is retained red: **1 failed in 11.44 seconds**, 24 artifacts,
197 source inputs, measured zero provider calls. The actual parent map was ready
with coverage **1.0**, matching plan/hash/geometry and no page/network errors.
The native-pan setup found no sampled visible canvas start and dereferenced
`null`; pointer framing is being diagnosed without weakening the real capture/
bubble or map-readiness requirements. Evidence:
`results/e2e/r20_waypoint_pointer_20260908_v3`, manifest content hash
`87e0e278b2bdfe57191b78928013da1c5659e5abaf05b27e86e36e1dfacdd230`.

The follow-up real framing diagnostic confirmed **overlay occlusion, not page
scrolling**: scroll position was zero and map bounds were 651 by 630 at (397,152).
Text/context overlays covered all 16 old sample positions; a visible-region hit
check found two exposed canvas points. The unchanged trusted native canvas-to-
context release passed, then the next helper encountered the same old sampling
assumption. That diagnostic remains red (26 artifacts, 197 sources, zero provider
calls) in `results/e2e/r20_waypoint_pointer_frame_diagnostic_20260908_v1`.
The narrowed test keeps the overlay open for the release proof, then closes the
text route with its actual visible summary control before continuing map edits.
Its final browser rerun and independent transition verification remain pending.

Chat-closure v3 and broad product regression v3 were started against frozen
production sources. The original chat case reached the new held-response check,
then failed an in-memory-versus-HTTP representation comparison: draft
`evidence_refs` is a Python tuple from `dataclasses.asdict`, but an array on the
actual JSON wire. All fields must remain compared after faithful wire
serialization; the assertion is not being removed. Both contexts stopped at that
same assertion: **2 failed in 541.19 seconds**, 565 artifacts, 194 source inputs,
measured zero provider calls. The finalized red manifest content hash is
`9d6a1fa6c6d27de54292dc9976e23a15ce0667fff1b745b6d5438aa3c802bffd`.
The broad regression is still running. No new race-pass claim is made until the
actual old response has been released after Undo and all visible/persisted
surfaces have been verified.

The full-session wire comparison is now corrected using FastAPI's existing
`jsonable_encoder`, including every field. Held-response bytes and the initial
not-yet-passed race witness are saved before comparisons. **91 focused tests**
pass (`chat_session_wire_v1.xml` in the repair-batch directory); the two-case
fresh website rerun is still required.

The independent paid/replay audit passed its **23** existing zero-provider
regressions, then identified three additional acceptance gaps: full normalized
synthetic-context authorization must occur **before dispatch**, standalone replay
must invoke the full independent verifier, and early/finalization failures must
retain exact HTTP/SSE/provider bodies before assertions. These harness repairs
are in progress before the authorized billed sample. Cost wording will distinguish
configured-rate estimates from an invoice-confirmed charge; actual attempt and
token counts remain separate evidence. **No billed call has occurred.**

### September 8 — conversation lifecycle source freeze; real website rerun pending

The five additional conversation-lifecycle counterexamples are corrected. The
final complete Copilot Node batch passes **42 tests**, including the original
13 revision checks and 22 generation/ownership/recovery cases. Confirmed deletion
invalidates pending local callbacks, clears ephemeral transcript state and stops
only the browser wait; it does not claim provider cancellation. A superseded
restore cannot reinsert a completed turn. Only an actual offline refresh failure
can retain the already completed canonical reply locally, without sending it again.
Conversation/session ownership is checked, including immutable trip/run fields.
Both new Node files run through normal pytest discovery.

The focused frontend/API/conversation-store/dispatch-safety batch passes **46
tests in 65.73 seconds**; the final frontend hooks pass **15 in 0.87 seconds**.
These are focused checks, not real-browser lifecycle acceptance. Existing fixture
envelopes were supplied the owner/schema fields present in actual API responses;
no compatibility default or weakened assertion was introduced in production.
Frozen `copilot.js` SHA-256:
`fa2bb75598aa33b076dd0d8a094e8aa4bd4518583ca0df1ebc3077c5116c10b0`.

Exact checks/results are retained in `tmp/copilot_all_generation_freeze.tap`,
`tmp/copilot_generation_python_final.xml`,
`tmp/copilot_generation_frontend_freeze.xml` and the original
`tmp/copilot_generation_independent_red.tap`. The independent network/marker/
control verification group was also rerun: **119 passed in 0.73 seconds**,
`results/repairs/runtime_concern_batch_20260908/network_review_current_v3.xml`.
No server or provider request was made by this focused freeze verification.

### September 8 — real chat Undo rollback reproduced; revision and backend I/O corrections

Chat-closure v2 remains **red: 1 failed / 1 passed in 581.00 seconds**, 657
artifacts, 194 source inputs and measured zero provider calls. The original
case's exact sequence establishes a real browser-state race:

- `chat_closure/original/http_106.json`: actual Undo returned HTTP 200,
  revision **25**, and an empty draft.
- `http_107.json`: a delayed conversation GET returned revision **24**, still
  containing the time-window draft.
- `http_108.json`: impact correctly reported revision **25** and zero operations,
  while the visible draft bar incorrectly remained at one.

These are paths within `results/e2e/r04_chat_closure_20260908_v2/`. The complete
raw bodies, request parameters, response hashes and failure-time map identity
remain there. No operation or provider response was invented to reproduce this.
The accepted-child case passed its browser assertions; the two-case suite and
complete gate are not passed.

Copilot now guards session adoption against older/foreign/malformed revisions
at all four asynchronous assignment boundaries and orders transcript restores.
The initial focused revision suite retained **11 failures / 2 passes** before
the correction and then passed all **13** cases. Existing Copilot Node files
also passed (20 reported test cases at that checkpoint). Independent review
subsequently found five further conversation-lifecycle counterexamples involving
deletion, late fallback replies, and foreign conversation ownership. Those are
being corrected before a fresh website rerun; revision monotonicity alone is
not reported as complete lifecycle protection.

The map's mismatched-ready branch now clears its three old rendered attributes;
the independent edge/guard set passes **12** tests. Active plan identity and map
render acknowledgement remain different concepts.

The preview API's blocking readiness, authentication, baseline validation and
durable publication work is now offloaded from its event-loop thread. Final
revision checks and publication remain one locked transaction. Two admitted
optimizer workers, authentication, cancellation/timeout behavior and the
responsiveness threshold are unchanged. Four previously red timing checks now
pass within **14 concurrency tests in 110.98 seconds**; **62** security, preview,
selection, responsiveness and accepted-baseline compatibility tests also pass in
180.07 seconds. The earlier broad-run 31-second overrun is not retrospectively
attributed to this path without the missing historical thread evidence.

Exact unit/API timing inputs and outputs are retained by those tests; reports:
`concurrency_preflight_completion_green_v1.xml` and
`concurrency_security_preview_compatibility_v1.xml` under the repair-batch
directory. The network/marker/control independent verifier group passes **119**
tests in 0.72 seconds: only exact configured local map GET cancellations can be
classified nonblocking; raw observations remain present. Cancelled API requests,
HTTP errors, different hosts/assets, malformed metadata and wrong/unready maps
still block. No map-disposal cause is inferred by this classification.

The registered aggregate now contains **17 suites / 38 browser cases**, but has
not been run as a complete final-source gate. Registration and prepared verifier
tests do not supply missing runtime receipts. The paid website sample and exact
output replay remain separate pending gates; **no billed call has occurred**.

### September 8 — fresh remount proof and expanded controls checkpoint

`results/e2e/r18_same_plan_remount_20260908_v2` passes the real browser test
in **26.37 seconds** and full independent verification: 41 artifacts, 194 source
inputs, measured zero provider calls. Native Select/Edit produced 31 and 33
observed loading samples respectively: canonical child
`plan_bc13d7ed8c7ec60b` / `ab610263c7f5a68f` remained active; old renderer fields
were absent; Accept was disabled. The same independently evaluated child, marker
set and geometry returned only after actual map readiness. Manifest SHA-256:
`96a944a1aa209252bd38821d7e1c6205c9c2a575f59767512dbf5eb5776cb07b`;
exact-I/O report SHA-256:
`75a0a0f7f69c5b64dd8413d8f3401c5a0a8c8e6a370c8596f691a449cc166111`.
Focused Python map/presentation regressions pass **42 tests in 14.73 seconds**
(`map_loading_focused_v2.xml`).

Capability-controls v4 retains **1 failed / 1 passed in 435.18 seconds**, 450
artifacts, 196 source inputs and zero provider calls. The original case reached
all its control checks, including both pre-staging cross-family blockers, then
failed its strict empty-network-events assertion on one `net::ERR_ABORTED` GET
of the local map style. Final actual coverage was 1.0 with the exact parent
identity. The source-fixed text panel did not collapse. Raw cancellation evidence
is preserved; cancellation classification and explicitly sequential selection/
impact/map synchronization are being validated, not silently suppressed.

The waypoint v2 observer failure is also now precisely diagnosed: trusted native
mouseup capture hit the old overlay and original captured marker, then the real
drag-end handler replaced that panel before the same event bubbled. Requiring
the old physical panel to remain at bubble time was invalid. The corrected
observer binds one native event, the pre-release physical hit, old-panel identity
and disconnection, and the new unconfirmed-relocation panel. Its independent
verification and fresh actual browser run are still required. Four map archive
cancellations in that red run are retained, not described as zero network events.

An additional independent synthetic-DOM review found that a mismatched ready
callback correctly blocks acceptance but leaves three previous stage renderer
attributes behind. The retained review has 10 passes and 2 failures
(`map_presentation_edge_combined_v1.xml`); cleanup and verification are pending
after the active chat browser source boundary. This is not an acceptance bypass
or a browser reproduction claim.

### September 8 continuation — two presentation fixes and retained broad/pointer failures

The full non-browser product regression v2 completed: **1,043 passed / 1 failed
in 1,991.28 seconds** (`product_broad_v2.xml`). The failure is the two-running-
preview admission/responsiveness test; it is being diagnosed, not discarded as
flakiness. Waypoint-pointer v2 also failed (31 retained artifacts, 196 source
inputs, measured zero provider calls), manifest content hash
`ea03e72e1d605935e771c5d0535e18695cdba7e5578b5f5166a08db6551c3757`.

Production now preserves the user's native text-route open/closed state across
map updates. The map status handler also publishes the authoritative active
plan ID/hash during loading, separately from renderer acknowledgement, which
remains withdrawn until readiness. The strengthened coupled controller/status/
acceptance regression retained its failing run and now passes all seven checks
(`map_loading_identity_red_v1.xml`, `map_loading_identity_green_v1.xml`).
Browser remount v1 passed its behavioral assertions but failed complete-manifest
finalization; subsequent independent inspection also caught missing active IDs
during loading. Registration and the production identity issue are corrected;
the original v1 evidence is unchanged. A fresh remount v2 and strengthened
four-viewport text-panel test remain required at this checkpoint.

All named XML files are under `results/repairs/runtime_concern_batch_20260908/`.
Exact input/output for each browser run remains in its dedicated run directory.
No billed provider call has occurred in this continuation.

### September 8 next batch — graphics and real restart independently passed; two expanded suites retained red

- **Real graphics recovery:** `results/e2e/r18_webgl_recovery_20260908_v1`
  passes 1 browser test in 43.93 seconds and independent verification of 47
  artifacts / 195 source inputs. Genuine graphics loss withdraws acknowledgement
  immediately and throughout the 1.2-second loss interval. Genuine restoration
  returns to loading, then the existing tile/renderer checks acknowledge the same
  evaluated child `plan_bc13d7ed8c7ec60b` / `ab610263c7f5a68f`, certificate
  `cert_90e853fe0e27d774`, and geometry
  `d0a3d9a141aef73782d1843541a9c57a3543a5ca10b9b92ba40eea409f7bbc39`.
  Actual before/after rendered sources, layers and features match; no screenshot
  is captured while graphics are lost. Manifest SHA-256:
  `0dd2c54011eb81cf97488f3eb8c07332c1416171bafd8556c88efb7fe40c5f2c`;
  exact-I/O report SHA-256:
  `a5511d93c9c562d7ea7a70f7631f3bb4b8c734d2b97637ac8ac9092417fb61b9`.
- **Cached page and real server restart:**
  `results/e2e/r24_build_coherence_20260908_v2` passes 1 browser test in 56.58
  seconds plus independent verification (68 artifacts / 195 source inputs).
  Actual private generation A `2555a862458126ed` stops, generation B
  `6af5578963c49e49` starts, stale-page writes are rejected without state changes,
  and explicit reload retains the accepted child and staged draft. Source drift
  remains latched unavailable even after bytes are restored, until actual
  restart. Both lifecycle shutdown receipts are finalized with zero provider
  calls; no fictional outer application server is claimed. Manifest SHA-256:
  `2f9c246916aac692207f2f05e357d90d04e2f8df960c797fd2144af1c7a31f65`;
  report SHA-256:
  `0f7c0e1c7ebc785e4ab8a5a0cf58aedb1433b50f05e3afecd369ac62aa1c62c0`.
- **Read-only asset discovery:** production now uses immutable registered
  dashboard/package loaders without constructing a mutable acceptance store.
  The unchanged real regression passes in 6.88 seconds, and all five assigned
  frontend/map files pass 52 tests in 18.37 seconds (`tmp/massetro_green1.xml`,
  `tmp/mfront3.xml`). The fresh-state failures remain preserved. Validator SHA-256:
  `9bc024e45818f0f54334679fe8ede51ac8d66001f4b40266d8d83838b1e0efc1`.
- **Capability-controls v3 is not a pass:** 1 failed / 1 passed in 421.48
  seconds; 392 retained artifacts, 196 source inputs, measured zero provider
  calls. Original context reproduced a timing-dependent text-route collapse:
  a later selection render replaces the native open details element with a
  closed one and hides route-leg controls. Accepted context completed its browser
  assertions, but independent inspection also found a test evidence label mismatch:
  the route-preference record stored the whole explanatory region as its button
  label. The corrected test now records the actual button label and retains the
  full explanation separately. The text-panel production fix and full rerun are
  still required; no forced click or weakened verifier is used.
- **Chat-closure v1 is not a pass:** 2 failed in 62.50 seconds; 72 retained
  artifacts, 194 source inputs, measured zero provider calls. Initial harness
  assumptions disagreed with actual behavior: exact HTTP record 027 showed the
  original case had selected Day 5, but the follow-up impact/render had not yet
  completed when the test tried to open its editor. The accepted-child observer
  expected the public five-field projection to equal the richer canonical
  accepted-plan reference. The corrected harness waits for exact selection and
  impact, and preserves both complete records with strict projection/identity/
  hash checks; 44 focused checks pass. The fresh browser rerun is still required.
  These failures do not establish live OpenAI connectivity or a provider
  failure; no paid call occurred.

The full non-browser product regression v2 is running with deterministic/fake
provider configuration. The small text-panel production change is held until the
active source boundary completes. Newly prepared remount, marker-overlay,
waypoint-context and candidate-readiness journeys are not browser passes merely
because their independent verifier unit tests pass. The final aggregate and
separately authorized billed website journey remain required.

### September 8, 14:42 UTC — 20 evaluated variants independently passed; replacement-map gap corrected

`results/e2e/r20_evaluated_variants_20260908_v1` passed both real browser cases
in **746.71 seconds**: ten exact variants on the original trip and the same ten
on a child accepted through the actual website. Independent verification passed
792 artifacts, 195 source inputs and finalized zero-provider instrumentation.
The inventory is five evaluated roles, three time-window forms, exact duration,
and same-day ordering. Every variant exercises real staging, the real optimizer
and independent evaluator, canonical diff/evaluation/geography bindings, visible
timeline/metrics/evidence/map, reload, and actual Undo. Original-trip variants
are correctly ineligible for the existing `weather_infeasible` finding and keep
the parent visible; accepted-child variants render their eligible child.
Per-variant acceptance is deliberately not claimed. Example accepted duration
input `set_stop_duration / exact` produced child `plan_532d0d19a075b4c5`, content
hash `b6a0879dff8a445d`, evaluation `cert_7a245c16575e2380`, from parent
`plan_bc13d7ed8c7ec60b`; the complete parameters, responses and renderer identities
are in that variant's artifacts, not reconstructed from this summary.

Manifest SHA-256:
`206fedd3579eedf9ec1ad79dc3546a03cce939c3ddfe3629824a05240e5537a2`.
Exact-I/O report SHA-256:
`a1cfc94e9cef11d7caa86d3668e5ed41e94a3f1d83b7d8fff01e17a944fd751b`.

After this source-bound pass, an independent executable review coupled the actual
map controller to the actual frontend status handler and acceptance predicate.
It reproduced stale acceptance during a same-plan map replacement: the old map
was removed, but the old child acknowledgement and `PREVIEW_RENDERED` state
survived until a later event. Production now clears all renderer acknowledgement
fields on disposal and emits loading before the first asynchronous render step.
The unchanged seven independent tests now pass
(`map_guard_independent_review_green1.xml`); the three prior failures remain in
`map_guard_independent_review_v2.xml`. A real-browser remount regression and
fresh final-source matrix are still required. The 20-variant result above remains
a pass for its recorded pre-correction snapshot, not a claim about later source.

The test-only frontend/map repair group passes 51 tests in 20.55 seconds
(`tmp/mfront2.xml`). Its fresh-state red run also found a separate real asset
preflight problem: two historical runs share a trip ID, but read-only coordinate
discovery initializes mutable acceptance state. The genuine context guard rejects
the second run. The validator must use immutable registered map artifacts without
creating or touching acceptance state; isolated passing fixtures alone do not
close this production defect.

### September 8, 14:32 UTC — real pointer/graphics/selection corrections applied; final browser gate running

This checkpoint supersedes the implementation status of the earlier private-only
diagnostics below; it does not rewrite their source bindings or declare a pass.

- The original production controller now withdraws rendered-map acknowledgement
  on genuine WebGL loss, keeps it withdrawn while loss is latched, and allows
  restoration only through the existing readiness/tile checks. Formal registered
  graphics-loss execution remains pending at this checkpoint.
- The real pointer defect is independently reproduced: a canvas pan released
  over `<dt>Distance</dt>` in `.map-context-card` delivered trusted document/window
  mouseup events but no MapLibre mouseup/dragend/moveend. The same map remained
  moving despite loaded tiles. Native pointer capture for primary mouse input
  starting on the canvas or draft marker now keeps that trusted release with
  its original target. The private corrected overlay-release and unobstructed
  aiming branches returned 100% coverage with the exact same canonical map
  identity. Exact events, coordinates, input/output source hashes, ready-only
  screenshots and finalized zero-provider receipts are in
  `tmp/cwayrelease1/{diagnostic,aim_fixed}/`. A formal recurring overlay-release
  browser regression remains required; a safe-path aiming helper alone is not
  closure.
- A genuine shared-selection bug was found in `WorkspaceStore.set_proposal`:
  missing child and geography IDs compared equal as `None == None`, allowing a
  partial eligible proposal to clear the current stop/segment/alternative.
  Explicit nonempty child identity is now required before projecting selection.
  `selection_identity_red1.xml` retains 3 failures/3 passes; the corrected focused
  workspace/preview/recorder run passes 23 tests in 1.28 seconds
  (`selection_green2.xml`).
- Direct inspection of the actual 1280px Compare screenshot found the Trip
  metrics footer still visible beside a different selected Compare option.
  Compare and Evidence now hide that unrelated footer and retain their own
  artifact-bound facts. Returning to Trip restores its own footer. The executable
  frontend regression was red before the correction (`metric_scope_red1.xml`);
  real product/acceptance and presentation tests now pass 13 tests in 43.68
  seconds (`product_metric_green1.xml`). Fresh visual verification is pending.
- The broad product regression finished normally: **1007 passed, 24 failed in
  2221.28 seconds**, not a pass (`product_broad_v1.xml`). Its failures include the
  genuine missing-ID selection bug, outdated UI/capability/status expectations,
  old fake-provider schema and stale exact-I/O test baselines. Failures remain
  retained. Corrected privacy/exact-I/O tests pass 12 cases in 34.10 seconds;
  each tampering case first verifies an untampered current-input baseline and
  then requires its specific wire-rejection reason. No real provider call is
  implied by these fake-SDK tests. Full broad rerun remains required.
- Capability-controls v2 progressed past 18 draft-only variants in each context,
  then its listener consumed an earlier day-selection response instead of the
  requested segment response. The corrected test waits for each actual request
  identity separately; v2 remains failed, not passed.
- The registered one-command aggregate now contains 13 suites / 32 exact browser
  cases, including evaluated variants and real WebGL recovery; its verifier
  regression passes 162 focused tests in 6.09 seconds. The aggregate has not yet
  been executed, and missing-coverage flags remain fail-closed. Additional
  keep/accepted-memory/chat-family journeys are prepared but not registered or
  browser-verified.

Exact test outputs above live under
`results/repairs/runtime_concern_batch_20260908/`. Current evaluated-variant
browser inputs and outputs are being captured under
`results/e2e/r20_evaluated_variants_20260908_v1/`: real control values, staged
operations, requests, previews, evaluations, diffs, timeline/metric/evidence/map
identities, reload and Undo traces. No billed OpenAI call has occurred in this
continuation. The current source is frozen for this formal browser batch.

### September 8 conversation v1 — five real website journeys independently passed

`results/e2e/r14_conversation_20260908_v1` passed **5 browser tests in 172.18
seconds**, then independent verification (242 artifacts, 195 source inputs).
The five sequential, individually guarded application lifecycles exercise:

1. `Replace Bixby Creek Bridge.` -> a candidate-only clarification; reload ->
   `Pfeiffer Big Sur State Park.` -> the exact replacement proposal -> actual
   Add/Preview -> the independently evaluated child rendered in the website.
2. Proposed driving preferences survive seven intervening turns, can be
   corrected, and are cleared through the actual conversation settings control.
3. A real serialized 200-turn fixture rejects submission before provider dispatch
   with `conversation_turn_limit_reached`; actual settings reset restores use.
4. A real serialized byte-capacity fixture similarly rejects before dispatch with
   `conversation_size_limit_reached`, then recovers after explicit reset.
5. Real SDK transport is paused at an explicitly fake external HTTP boundary;
   actual Stop waiting, reload and Retry retain the same logical turn, durable
   `unknown` outcome and one-attempt budget. Retry returns
   `copilot_outcome_unknown` without a second SDK/provider attempt. The accepted
   plan, map geometry and markers remain unchanged.

All live provider call counts are measured zero. Case 5 uses real `AsyncOpenAI`
over never-fallthrough `httpx.MockTransport`; it **does not prove live OpenAI
connectivity**. Exact submitted chat envelopes, complete SSE events or explicitly
labeled interrupted prefixes, fake SDK request/response bytes, context/memory,
dispatch journals, file/source hashes, lifecycle receipts and ready-only captures
are retained. No secret headers are stored. Manifest SHA-256:
`7417827d7bf9164db8d0867137ca7db3cdf22afd9af9b2bf7d0d4805e2346f76`;
report SHA-256:
`1564c0f935ea78b9988ff9fcf46b1c503cc19cb0ff0e6979fba0f11a88e909b4`.

The following new batch remains red: build-coherence v1 failed during private
fixture copying before application startup because pytest's workspace temp prefix
exceeded Windows path limits (provider count remains unknown/null, not an invented
measured zero). Waypoint-pointer v1 timed out after actual camera movement with
readiness withdrawn; it recorded zero provider calls and no browser/network errors.
Capability-controls v1 failed both contexts because the browser test used a generic
Playwright disabled assertion on native `option` elements; their actual disabled
property and permission label were present. That test now checks both the native
`disabled` property and `:disabled` state, without forcing protected selections.
Fresh reruns are required. These failures are retained and not counted as passes.

### September 8 formal workspace v5 — four viewports independently passed

`results/e2e/r21_workspace_stability_20260908_v5` passed **4 real browser tests
in 87.43 seconds**, followed by independent verification: 231 artifacts, 192
source inputs, four null/null LLM disclosures, measured zero provider calls and
finalized isolation. The exact suite covers 1440x900, 1280x800, 1024x768 and
390x844; initial Trip, three reloads, selected-day text route, selected Compare,
module return, Copilot opening/closing, and desktop 1440 -> 1024 -> 1440 resizing.
Mobile includes the explicitly simulated reduced keyboard viewport, not an OS
virtual-keyboard claim. Every visible map has at least 95% loaded projected tile
coverage, exact canonical geometry/marker/plan identity and screenshot-after-ready
timestamps. Mobile Compare verifies its one selected visible pane; desktop
verifies both original and selected panes.

Manifest SHA-256: `cf351d97e5566fbfddc6def6246a458412effc9126a13135492b1788edebe80a`.
Exact-I/O report SHA-256: `a34d488205ba7edcd42eba1893ac9e1bbb13b244fe3e70d777c0ee454e746307`.
The failed 94.23% workspace v4 remains untouched. This closes the narrow-pane
Compare reproduction for v5's recorded source, **not the whole phase**: the
separately reproduced graphics-context-loss correction is still private-only
pending the active broad regression, and later source edits require revalidation.

### September 8 continuation — addition verified; narrow Compare gap corrected, formal rerun pending

- `results/e2e/r24_addition_continuation_20260908_v2`: **1 real browser journey passed in 74.07 seconds and independent verification passed** (83 artifacts, 192 source inputs, measured zero provider calls). The actual UI accepts replacement B, reloads it, adds Point Lobos to child C, previews and accepts C, reloads again, and verifies accepted Evidence and preserved A -> B -> C lineage. C is `plan_92f22bd7e77d251f` / `f48522f2d49b1878`; all 16 required road legs are validated. Manifest SHA-256: `e7f3d1266c252a7418ad5f429e1950075027fa0fd9283b2693952979d5909d28`. Exact request/response and file I/O remain inside that immutable run.
- Workspace v4 retained **3 passed / 1 failed in 111.55 seconds**, not a stability pass. At 1280x800, both Compare panes measured only `0.9423091286794749` loaded coverage in a 203.5x334 container. The ready guard correctly refused the uncovered 19.26875-pixel strip. Four fixed 54-pixel fitting margins forced an unnecessarily distant camera and exposed map area outside the available local tile.
- The map fitting margin is now `min(54, min(container width, container height) / 8)` at construction and final fit. A separate private-copy real-browser diagnostic covered the visible Compare maps at 1440x900, 1280x800, 1024x768 and 390x844 with effectively 100% loaded coverage and no console errors (`tmp/cmapdiag5/observed_maps.json`). The 95% threshold is unchanged; the exact 94.23% footprint still fails its regression. Focused map/G4R/Compare regressions passed 28 tests. These are corrective diagnostics, **not the final source-frozen workspace pass**.
- The captured chat Preview callback now rechecks its exact single expected server-owned operation after pending selection synchronization and immediately before sending Preview. A changed/duplicate/removed operation cannot be submitted through that stale action; unscoped ordinary Preview retains its existing semantics. Boundary regression: 16 passed, 1 deliberately deselected real-API case (`copilot_boundary_green2.xml` under the repair-batch directory). This scoped result is not the deselected case's execution evidence.
- Real screenshot inspection found stacked progress toasts obscuring the mobile map and default-expanded technical details making acceptance unnecessarily distant. New progress replaces older non-error progress, errors remain visible, advanced details are collapsed but retained, and the displayed evaluated child labels the original disruption as historical context. Presentation regressions passed 3 tests in 8.63 seconds (`results/repairs/runtime_concern_batch_20260908/presentation_v6.xml`); a fresh browser capture is still required after these changes.
- Clarification/reload, memory/capacity/cancellation, cached-client restart and actual pointer-waypoint browser tests are prepared. Collection, fake-SDK checks and private diagnostics are explicitly not their complete browser execution. The shared runner, independent verifier and operation/variant coverage are being connected before the source-frozen run.

The independent R18 review then reproduced a second real readiness defect with
the browser's genuine `WEBGL_lose_context` extension. After a trusted context-loss
event and `gl.isContextLost() === true`, the unchanged controller still published
`mapReady=true`, coverage 1 and `PREVIEW_RENDERED` more than 1.2 seconds later.
The isolated correction withdraws all rendered-map identity on loss, prevents
readiness publication while loss is latched, and returns to loading on genuine
restoration until the existing idle/tile/route checks pass. Private real-browser
evidence under `tmp/cwebgldiag1/` records red and corrected runs, exact controller
patch and input/output hashes, preview HTTP I/O, trusted events, exact before/after
rendered source/layer/feature equality, ready-only screenshots, and two finalized
zero-provider server lifecycles. Corrected coverage was 1.0 with the same child
`plan_bc13d7ed8c7ec60b`, hash `ab610263c7f5a68f`, geometry
`d0a3d9a141aef73782d1843541a9c57a3543a5ca10b9b92ba40eea409f7bbc39`.
No screenshot was captured while the context was lost. The main production file
remains unchanged during the active broad regression; applying the correction and
passing a registered final-source WebGL-loss browser journey are still required.

No billed OpenAI call has occurred in this continuation. All prior passing bundles below remain evidence for their recorded sources after these subsequent edits. The full E2E-1/stability gate remains open.

### September 8 source-frozen checkpoint — real mobile and typed acceptance verified

The earlier checkpoint below is retained as history. **The complete gate remains in progress.** This checkpoint is supported by fresh source-frozen browser runs, not by re-labeling earlier bundles.

- `results/e2e/r21_mobile_chat_20260908_v6`: **1 real touch-browser journey passed in 43.60 seconds; independent verification passed** (52 artifacts, 192 source inputs, measured zero provider calls). At 390x844 it sent three actual chat messages, staged the exact replacement, kept the parent map unchanged until Preview, displayed the independently evaluated child in the map/metrics and visible text route, explicitly accepted it, and reloaded that same child with exact accepted Evidence. Five full-resolution captures have map-ready/coverage/timestamp receipts. An OS virtual keyboard was not tested.
- `results/e2e/r24_direct_edits_20260908_v4`: **4 real browser journeys passed in 244.94 seconds; independent verification passed** (206 artifacts, 192 source inputs, measured zero provider calls). Duration, time window, role and same-day order edits each traverse the actual controls, API, evaluation, Compare, map, explicit acceptance, reload and accepted Evidence. The current role acceptance replay no longer fails on a tuple-versus-JSON-list representation difference; semantic/hash/timestamp forgery checks remain strict.
- The same-day reorder now changes only the requested day. Full untouched stop/route records remain unchanged; the regression verifies edit cost 4.0 rather than the unintended 4.5. Typed affected-day summaries and timeline change markers derive from the actual canonical diff, including duration/window/role changes. Historical disruption scope is labeled separately.
- The real mobile run found that staging increments the revision and disabled its own chat Preview action. The controller now permits Preview of exactly one matching server-owned staged operation; stale Add, unrelated drafts, changed handlers and disabled capabilities remain blocked. The actual controller regression and full focused semantic/Compare/presentation/Copilot set passed **71 tests in 28.91 seconds** (`results/repairs/runtime_concern_batch_20260908/focused_source_frozen_v1.xml`).
- Compound replacement plus positive stop-removal language now explicitly identifies removal as unavailable and states that no part was staged. Negated removal, ordinary replacement phrasing, and unrelated clauses retain separate guarded handling. Red failures and subsequent green tests are preserved; this does not claim removal has been implemented.
- The Compare ready callback no longer triggers a resize that invalidates readiness and recursively fires another ready callback. Loading is not labeled Map ready. The full 14-case focused Compare regression passed; the fresh four-viewport runtime rerun is still pending at this checkpoint.
- The frozen internal HTTP chain replacement -> duration -> time window -> role -> acceptance/restart passed separately with 21 actual exchanges (`tmp/crrole4-results.xml`, exact I/O under `tmp/crrole4/test_real_replacement_duration0/`). The four browser journeys above supply the browser proof instead of substituting that internal check.

Exact mobile input/output:

1. `Keep Stearns Wharf.` -> a `keep_stop` suggestion targeting `stearns_wharf`; no draft mutation.
2. `Replace Bixby Creek Bridge with Pfeiffer Big Sur State Park and remove Stearns Wharf.` -> `unsupported`; no partial draft or map change.
3. `Replace Bixby Creek Bridge with Pfeiffer Big Sur State Park.` -> `proposal_ready`; explicit Add sends `replace_nearby`, target `bixby_creek_bridge_viewpoint`, parameters `{"candidate_id":"pfeiffer_big_sur_state_park"}`, source `copilot`, expected revision 0.

Preview request ID `w3_preview_0235fb2f54ef1ba7`; parent `plan_359968d255ff9288` / `750dae617fc4c1e3`; child and rendered map `plan_bc13d7ed8c7ec60b` / `ab610263c7f5a68f`; evaluation `cert_90e853fe0e27d774` / `c974dbc8f53339af`; diff `diff_320fae8e32e4c503`; geometry SHA-256 `d0a3d9a141aef73782d1843541a9c57a3543a5ca10b9b92ba40eea409f7bbc39`; acceptance `decision_2412bd9a180f4d54aae0b7500b99411e`. Preview and accepted reload have the same marker set, with Pfeiffer present and Bixby absent.

Complete chat/SSE, API, artifact and file I/O are retained in each run's `exact_io_report.md`, `file_io_manifest.json`, `http_*.json`, and scenario artifacts. Mobile manifest SHA-256 is `199c8be7b4fbb2e117c3eb46e92cae2eeb2536d66b8ade6c8b9877b0851bee30`; direct-edit manifest SHA-256 is `2be53ff7cf93bbd20337fce879db430aee4fd8e5007d725986940f7b8bb82e7b`. Provider input/output are null/null for these deterministic runs; they are real website chat tests, not new GPT-call evidence. No billed call has occurred in this continuation.

Retained new failures: mobile v1 (consumed-SSE recorder), v2 (test predicate blocked by CSP), v3 (ambiguous unsupported-clause copy), v4 (staging disabled chat Preview). Mobile v5 passed its browser test but its first independent check exposed a list-versus-object loader bug; that bundle remains untouched, and v6 is the fresh independently verified run. Direct v3 retained the role replay failure and unintended untouched-day reorder. Addition v1 passed its browser but lacked HTTP status in its evidence index; workspace v3 retained the Compare readiness loop. Fresh addition v2 and workspace v4 are running, not yet reported passed here.

Still required: final four-size/addition results; actual multi-turn/capacity/cancellation, pointer-waypoint and cached-client restart journeys; complete fail-closed capability/interaction coverage; broad final regression; a separately bounded authorized billed website sample and final-source replay; final authoritative status reconciliation. Prepared or collected new tests are not execution evidence. Later source edits will make the runs above historical evidence for their recorded snapshots, not automatic current-source passes.

### Later September 8 checkpoint (not a full-gate pass)

- Workspace v2 retained **3 passed / 1 failed**: all twelve reload checks passed, but Compare exposed a queued-selection revision race. The selection queue is repaired and awaits a fresh four-size browser run. Visual inspection also found a mobile draft-count/Undo overlap; CSS and rectangle assertions are repaired but not yet recaptured.
- R19 v3 retained a failed completed-response race: the actual eligible result used the newly registered composite route identity, which the frontend rejected. Exact composite identity validation is repaired; a fresh race run remains mandatory.
- The authentic Pfeiffer-to-Point-Lobos directed route was captured once from the local OSRM service and frozen in `results/route_deltas/product_pfeiffer_point_lobos_20260908_v1/`. The original 22 cells remain unchanged. The real API A-to-B-to-C addition chain passed with C `plan_92f22bd7e77d251f`, hash `f48522f2d49b1878`, certificate `cert_76f82ba3c9c11e62`, and 16/16 validated road legs. Its actual browser addition/acceptance chain remains to run.
- Build-coherence guards passed 110 server/security/runtime checks. The original HTML build identity is required for browser API calls; stale clients are blocked before dispatch, and server-source drift is latched until restart. The running-browser restart journey remains separate and unverified.
- Structured memory/durable dispatch regressions passed 177 tests; runtime receipt tests passed 57; map transport/configuration tests passed 66; accepted-evidence/selection-queue checks passed 2. The packaging/runtime-receipt batch passed 128 tests after retaining a real telemetry-write failure from its preceding run. These are scoped checks, not proof that the complete browser gate passes.
- The new direct-edit suite has four real browser cases, each requiring precise changed values, independent certificate/diff hashes, visible Preview, explicit Accept, reload, and accepted Evidence identity. The addition suite has one real A-to-B-to-C browser case. Both are implemented and collected, not yet executed at this checkpoint.
- No billed LLM call has occurred in this September 8 continuation. Fake-provider outputs remain labeled fake; a bounded actual website call is pending after the deterministic repairs.

- The source-frozen real Chrome map-recovery run `results/e2e/map_area_recovery_guarded_20260908_v2` passed **2 browser tests in 69.60 seconds** and independent verification: 94 artifact files, 171 source inputs, measured zero provider calls, no blocked provider attempt. It exercised real replacement, independent evaluation, explicit acceptance, repeated reload, transient/persistent map-config failure and the visible Retry action.
- The same run deliberately hid renderable base tiles after a ready map. Readiness fell to zero and its acknowledgement was withdrawn. Restoring the tiles restored the **same** child, markers and geometry. Coverage now measures projected loaded-tile union area clipped to the map viewport, not loaded-source count. This is a 2D, locally pinned renderer contract; missing renderer access fails closed.
- A real API chain now reaches A → eligible B → accept/restart → eligible C duration edit → accept/restart. C has 15 required road legs, 15 validated legs, no missing or fallback legs. This API result is not yet a new browser A→B→C acceptance claim.
- Direct-duration and time-window previews now carry their full independently evaluated certificate, bound inside the existing evidence hash. Acceptance re-evaluates, restores only the inspected timestamp, and requires exact equality. It does not skip evaluation or accept semantic changes.
- The bounded preview worker no longer blocks independent requests; excess work is rejected, timeouts are explicit, and cancelled/stale results cannot publish. Focused concurrency checks passed; final whole-application regression remains pending.
- Long Windows acceptance paths exposed an overlong temporary filename and a collision-cleanup ownership bug. Both were reproduced, repaired, and tested. The original long-path replacement/acceptance failures now pass. Old-baseline expectation and Copilot regressions discovered in that broader run remain separately tracked.

Map-run manifest content hash: `04204545fd627acca03e190575671c7235cc08d454b851ce834982648d3af43b`.
Manifest SHA-256: `2b57fb968e6cc73a27af0ed5d33ff35dc0afaf4caccbf433646b5b1f25574dbc`.
Exact-I/O report SHA-256: `c2a5e906e85bc2b9072c344f5c782c7b6bcdfc68d9876a6c41bcab17e8518a63`.
Subsequent source edits mean this is evidence for its recorded source snapshot, not an automatic pass for later changes.

## Historical early-September-8 concern register

This table is retained as the earlier checkpoint, not the latest per-concern
status. Later results, newly discovered failures and remaining final-source
requirements are recorded in the dated continuation sections above. In
particular, the completed-response race, four-size workspace, addition,
conversation, graphics recovery and build-restart suites have bounded passes
for their recorded snapshots; newer repairs still require fresh aggregate proof.

| Concern | Current bounded result | Work still required before whole-gate closure |
|---|---|---|
| R01 provider isolation | Previously verified; measured again in map run | Retain for every aggregate run |
| R02 honest outcomes | Compiler/API/JS separate blocked, not evaluated, failed and ineligible | Actual browser negative journeys |
| R03 executable additions | Exact candidate/day/parent readiness; authentic directed route frozen; real A -> B -> C addition and independent verification passed | Final-source aggregate and all contextual candidate/day blockers |
| R04–R05 semantic draft validation | Keep-only no unrelated replacement; each candidate-target pair checked; focused real compiler/API tests passed | Include mixed valid/invalid browser journey |
| R06–R08 accepted context/capabilities/semantics | Accepted-plan context and conservative whole-message guard implemented | Final v2 regressions and multi-turn real chatbox check |
| R09–R10 presentation | Parent/child eligibility separated; canonical metrics/diffs; four direct-edit and mobile text/Evidence journeys independently passed | Recapture latest collapsed details/context/toast changes; final Compare and all-state aggregate |
| R11 continued acceptance | Real A→B→C API and browser addition/acceptance/reload chain independently passed | Final-source broad regression and aggregate |
| R12 combined operations | Existing post-staging combination blocker remains | Pre-staging restriction and complete supported/unsupported journey coverage |
| R13–R14 clarification/memory | Versioned capability and plan-bound structured memory wired; initial 5 checks passed | Adversarial/legacy-journal regression and real browser multi-turn result |
| R15–R16 dispatch/capacity | Durable pre-dispatch reservation and result recovery implemented; fake-provider tests passed | Real chatbox retry/cancel/capacity behavior; separately bounded live sample |
| R17 retention | Sliding-expiry and evidence-deletion scope clarified | Final clock/deletion and UI-copy regression |
| R18 map coverage | Real missing-tile/reload/recovery gate passed; narrow Compare gap reproduced and fixed in private four-size diagnostics | Formal four-size rerun; actual graphics-context-loss diagnosis and renderer-failure aggregate |
| R19 late successful response | Real A-success-held -> distinct B -> release-A browser v4 and independent verification passed | Preserve exact request/child/geometry identity in final-source aggregate |
| R20 aggregate traceability | Existing deterministic modes retained | Complete action matrix and one-command aggregate gate |
| R21 full-size/mobile | Mobile v6 real three-message touch journey independently passed; workspace v4 retained 3 passed / 1 failed | Final four-size rerun and latest-source mobile capture; OS virtual keyboard is not claimed tested |
| R22 responsiveness | Bounded workers and focused concurrency verified; Windows persistence fix verified | Final aggregate/source-frozen regression |
| R23 truthful tracking | This checkpoint records current limits without rewriting old evidence | Reconcile authoritative roadmap/traceability after final results |
| R24 environment/cache identity | Current local OSRM image/PBF checked by independent agent | Final secret-free environment manifest and cached-client coherence gate |

## Exact inputs and outputs

The browser map run's `exact_io_report.md`, `file_io_manifest.json`, and `map_initialization_recovery/http_*.json` identify actual requests/responses, source hashes and artifacts. `map_initialization_recovery/tile_coverage/fault_input.json` labels the renderer fault; `blocked_output_no_screenshot.json` records its result. Every saved screenshot has a corresponding readiness/timestamp receipt. No blocked-state screenshot is presented as map-ready evidence.

Representative real input: replace `bixby_creek_bridge_viewpoint` with `pfeiffer_big_sur_state_park`, then explicitly Preview and Accept. Result: child/map `plan_c9a1dffb7461d665`, plan hash `e0c7162763e71edd`, evaluation `cert_05e7851692e958f9`, rendered geometry SHA-256 `9163802f28fe20da03c40e1a022ab5fdad2d0524fe423844255dd8321056d680`.

The continued-edit API exchange is retained at `tmp/c11e/test_real_replace_accept_resta0/exact_chain_io.json`: A `plan_359968d255ff9288` → B above → C `plan_ad66f4db9c853461` / `a4ebb081a9cca7d1`; C evaluation `cert_db345b2a902aa61b`, diff `diff_8f151dccc99ff7ca`. The route-object hash in that test is explicitly **not** a browser render acknowledgement.

Direct-edit API-to-JavaScript inputs/outputs are retained under `tmp/schedcert1/`; the first test batch's one failure was a synthetic summary mutation that contradicted the newly retained full certificate. That counterexample now rejects as required, and a separately labeled coherent synthetic record still passes. It does not claim an evaluator ran for synthetic mutations.

Provider input/output for the map run is `null`/`null`: no chat was submitted, and the enclosing finalized guard measured zero provider calls. Fake-provider records for dispatch/memory tests remain labeled fake. No billed call has been made in this September 8 repair continuation at this checkpoint.

Credentials are excluded from new captures. Ordinary bodies remain exact UTF-8; JSON credential fields use `OMITTED_SECRET`, with explicit redaction paths, stored-body hash, and distinct original-body hash. The original map-area v1 and earlier recovery capture contained a disposable session token in a local HTTP record despite a false metadata flag. Those earlier snapshots are superseded and must not be shared as secret-safe evidence. They have not been silently rewritten or relabeled as passing. New captures and both evidence verifiers check embedded JSON credential fields as well as headers.

## Retained failures and limitations

- `results/e2e/map_area_recovery_guarded_20260908_v1`: 2 failed; one-shot readiness remained true after tile removal, and one normal reload could not verify its map runtime. The map-readiness listener is now continuous; map probing has a bounded 3-second timeout with a distinct timeout code. The exact original generic probe failure is not asserted to have been conclusively attributed to timeout.
- `results/repairs/runtime_concern_batch_20260908/r22_regressions_v2.xml`: original long-path persistence failures, retained.
- `tmp/c11c/`: C acceptance timestamp/hash failure, retained before its repair.
- Research D1/E3.1/E3.3/E4 statuses and the limited frozen candidate universe are unchanged. A statewide discovery catalog is not a promise that every California insertion has executable frozen route evidence.

Historical next-work list from that early checkpoint: the real completed-response race, four-size UI gate, authentic addition route delta, final multi-turn Copilot regressions, capability/action traceability, current environment receipt and one separately authorized billed website sample. Current work is the remaining conversation-lifecycle correction, deterministic late-restore/Undo browser regression, fresh expanded control/waypoint/candidate journeys, complete fail-closed aggregate coverage, broad final regression, and the separately authorized paid website sample plus exact-output replay. A component test count will not substitute for those runtime results.

### September 18 CT / September 19 UTC — bad eligible-child geography now fails closed in the real website

The next runtime-integration blocker is closed for the current recorded source:
an independently eligible child can no longer be presented as a successful
preview when the response has either no child geography or geography belonging
to the parent plan. The final source-bound bundle is
`results/e2e/preview_geography_fault_20260918_v4`.

The browser used the real map controls at 1440x900, selected Day 5/Bixby,
selected Pfeiffer, staged the repair, reloaded the actual session, and clicked
the actual Preview control. The optimizer and evaluator ran before the declared
private-copy fault changed only the response geography. Both cases passed:

- canonical missing-geometry case: eligible child
  `plan_bc13d7ed8c7ec60b` / `ab610263c7f5a68f` was produced and evaluated, then
  the wire response changed `geography_plan` to `null`;
- canonical foreign-geometry case: the same eligible child was produced and
  evaluated, then the wire response substituted parent geography
  `plan_359968d255ff9288`;
- in both cases the website entered `PREVIEW_FAILED`, kept active and rendered
  map plan `plan_359968d255ff9288`, disabled Accept, retained `1 draft change`,
  and displayed the mismatch failure instead of claiming Preview ready;
- the backend's canonical child, evaluation, diff, protected accepted artifacts,
  session owner and draft were unchanged by the response-only fault;
- the saved session verifier is recorded only as `OMITTED_SECRET`, with exact
  redaction path `/token_verifier`, separate projected/original hashes, and no
  plaintext verifier in the shareable evidence.

The general evidence scanner now treats `token_verifier` as a credential field,
and the geography recorder requires exactly its declared redaction path. Raw or
misclaimed verifier evidence fails verification; an already-redacted receipt is
accepted idempotently. The adversarial verifier fixture now binds the exact
staged operation, canonical repair request, candidate, affected day, no-fallback
policy, baseline route, child, evaluation, diff and response-only fault.

Focused verification passed **199 tests in 9.01 seconds**. The four edited
Python files also pass Ruff. The real browser bundle passed **2 tests**, followed
by a separate post-run verification over **117 artifact files, 198 source
inputs and four finalized isolated application lifecycles**. It measured zero
provider calls and no browser console or network failures. This deterministic
fault gate does **not** claim live OpenAI connectivity and made no billed call.

Exact website/API input:

```json
{"type":"replace_nearby","target":"bixby_creek_bridge_viewpoint","parameters":{"candidate_id":"pfeiffer_big_sur_state_park"},"source":"map","expected_revision":2}
```

Exact Preview input:

```json
{"expected_revision":3}
```

Canonical optimizer/evaluator output identity for both cases:

```json
{"request_id":"w3_preview_0235fb2f54ef1ba7","repair_request_hash":"6b67772f460ad86d","parent_plan_id":"plan_359968d255ff9288","child_plan_id":"plan_bc13d7ed8c7ec60b","child_plan_content_hash":"ab610263c7f5a68f","evaluation_id":"cert_90e853fe0e27d774","diff_id":"diff_320fae8e32e4c503","canonical_geography_plan_id":"plan_bc13d7ed8c7ec60b"}
```

Exact faulted response and visible output summary:

```json
{"missing":{"wire_geography_plan":null,"preview_state":"PREVIEW_FAILED","rendered_map_plan_id":"plan_359968d255ff9288","accept_disabled":true,"draft_text":"1 draft change"},"foreign":{"wire_geography_plan_id":"plan_359968d255ff9288","preview_state":"PREVIEW_FAILED","rendered_map_plan_id":"plan_359968d255ff9288","accept_disabled":true,"draft_text":"1 draft change"}}
```

Exact LLM input and output for each case:

```json
{"provider_call_count":0,"llm_input":null,"llm_output":null,"reason":"Real deterministic repair and independent evaluation; only private response geography is faulted."}
```

The complete unabridged file and HTTP input/output ledger is
`results/e2e/preview_geography_fault_20260918_v4/exact_io_report.md`; the two
stage requests are each `http_011.json`, the two Preview exchanges are each
`http_019.json`, canonical-versus-wire bytes are in `backend_boundary_bytes.json`,
visible assertions are in `browser_assertions.json`, redacted stored state is in
`persisted_session.json`, and provider input/output is in `llm_io.json` beneath
the `missing` and `foreign` case directories. Run ID:
`e2e1_self_20260919T024611873416Z`. Manifest content hash:
`037afd4730e3efea4060a070dfc8e66bbefebbf16256e8a91c841a2e27ff94af`;
manifest SHA-256:
`3199ea34fa38a0c3b2c1fa5b6e9071f53779b0399fa57360a04119feecb8f28b`;
exact-I/O report SHA-256:
`039c9e745b923a48105187f3b53361e1c678aacf3e36c1071da031b41d7ae36a`.

Retained red evidence is not relabeled. v1 could not read the protected
canonical demo artifacts under the restricted process token and never started
the app. v2 passed before the generic verifier-field scanner was added and is
historical. v3 ran the website but correctly exposed that the recorder rejected
its own expected `/token_verifier` redaction report; both browser cases failed.
v4 is the first bundle that binds the strengthened scanner, recorder,
current-source browser journey and independent verification together.

This closes the geography-response failure-path blocker only. It does not close
the complete E2E-1/stability gate: the registered full aggregate, remaining
capability/action coverage, broad final regression and separately bounded billed
website-chat sample retain their own acceptance requirements.

### September 20 CT — deterministic aggregate runtime gate is green

#### Executive result

The complete registered deterministic browser gate now passes. A real user
action is connected through the draft, repair engine, independent evaluator,
plan diff, itinerary, evidence surfaces and map for the supported test universe.
The result is **50 passed browser tests across 20 suites, 127 accepted runtime
evidence rows and zero cross-suite coverage findings**.

This closes the deterministic aggregate blocker. It does not claim that the
separately authorized billed OpenAI website-chat gate ran: the manifest retains
`full_phase_completion_claimed=false`, and no billed call was made.

#### Problem and smallest correction

The v10 browser journeys had already passed, but the aggregate discarded most
evaluated-role and time-window receipts because it required the independently
created capability and evaluated-variant suites to share a disposable browser
`session_id`. The canonical identity is the context, parent plan ID and parent
plan content hash. The correction now binds on those three values while the
regression proves that a different session is allowed and a different context,
plan ID or plan hash still fails.

Focused result:

```text
20 passed in 0.43 seconds
```

Read-only replay over the retained v10 evidence after the correction:

```json
{"passed":true,"finding_count":0,"runtime_evidence_count":127}
```

#### Exact source-file input and output

| File | Exact input | Exact output |
| --- | --- | --- |
| `scripts/e2e1_interaction_coverage.py` | 27,521 bytes; SHA-256 `66d140e13dc7b7e81d518aaad472642dcf25821f7e11d16e621b19cdaae1840a` | 27,584 bytes; SHA-256 `a1b6f4a5bca8ac5501a94686671f052cae7f565b6deb2f5a7a55a16b9304ffd1` |
| `tests/product_app/test_e2e1_interaction_coverage.py` | 11,139 bytes; SHA-256 `a5665525c90584f1567c6d1b3eb4dbb39db3060bb60d7f8b3b439ba8b9b1f7e8` | 11,551 bytes; SHA-256 `4aa4014af76ae95e367c397f3d09acc9cc0f87babf746987064752e09707bb40` |

The F-drive copy was restored from the byte-verified C-drive mirror after disk
space was made available. Both restored hashes matched the tested mirror before
the focused and aggregate runs began. The aggregate recorded
`source_unchanged=true` from start to completion.

#### Exact aggregate command and output

Input command:

```powershell
python scripts/run_e2e1_vertical_slice.py --browser-test aggregate --output-dir "C:\Users\Ztang_Yit_Xiaang\Documents\weather-itinerary-e2e1-results\e2e1_aggregate_20260920_v11" --port 8127
```

Exact process output:

```json
{
  "output_dir": "C:\\Users\\Ztang_Yit_Xiaang\\Documents\\weather-itinerary-e2e1-results\\e2e1_aggregate_20260920_v11",
  "assertions_passed": true,
  "passed_test_count": 50,
  "coverage_findings": 0
}
```

The run started at `2026-09-20T14:21:03.55724-05:00`, completed its test and
coverage manifest at `2026-09-20T16:06:49.142338-05:00`, and exited with code
zero after independent verification. All 20 suite stderr logs are empty.

#### Exact LLM input and output

The artifact tree contains 40 `llm_io.json` receipts. Every receipt has this
effective input/output:

```json
{
  "provider_call_count": 0,
  "llm_input": null,
  "llm_output": null
}
```

All 43 finalized `provider_isolation.json` lifecycle receipts report zero sent
provider calls and zero blocked provider attempts. Therefore this gate proves
deterministic website-to-optimizer integration, not live OpenAI connectivity.

#### Final evidence identity

Evidence directory:

```text
C:\Users\Ztang_Yit_Xiaang\Documents\weather-itinerary-e2e1-results\e2e1_aggregate_20260920_v11
```

Hashes:

```text
aggregate content hash: 01b73d3fc40a4f7b2eb0dbfeea56e6ed23737cea7f339b4ae385ed1326bed35e
aggregate_manifest.json SHA-256: cbe5c83c41607977e9d0670cd832c3994373abd07a87ce45471d88f3e27e9574
aggregate_exact_io_report.md SHA-256: 980f85dd18ff5469db55a5e2f3296c500c63305e4ca406f2be1bda7e2e48a54f
interaction_coverage.json SHA-256: 64b2ba05609a259f80a10e6c085c2346d5e66563adb7272590e07a655b4dbcd8
```

The previous v10 bundle remains retained as red history: its 50 browser tests
passed, but its aggregate correctly remained red with four coverage findings.
It is not relabeled as passing.

### October 7 — fresh website GPT-to-optimizer/map proof

The current readable checkpoint and complete R01–R24 register are in
[`runtime_coherence_status_20261007.md`](runtime_coherence_status_20261007.md).
Two separately bounded billed synthetic-demo calls ran through the actual
website. v1 produced an eligible child and changed map but failed an obsolete
metrics-copy assertion; it remains red. The test now verifies exact displayed
values and child/evaluation identity. Its 77-test evidence regression and lint
pass. Zero-provider replay v1 and fresh live v2 both pass the real browser
journey and independent verification, including Compare, Evidence, reload and
four screen sizes. Full file, API and provider inputs/outputs are retained in
their `results/e2e/e2e1_{live,replay}_20261007_*` bundles.

The full offline product regression is running. September's aggregate remains
valid historical evidence, but today's shared browser-test change requires a
fresh final-source aggregate. Cross-family composition, native mobile-keyboard
evidence and phase-level audit remain open. G5, W6-W8 and all research gate
statuses are unchanged. No historical failure is rewritten as a pass.
