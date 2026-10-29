# Runtime coherence concern audit and repair sequence — 2026-09-07

## Summary and scope

Status: **audit complete; proposed repairs not implemented by this audit**. The September 5 replacement/preview/acceptance slice remains supported by current-source evidence, but the complete E2E-1/stability gate is not established. Several other advertised interactions still fail or misrepresent the active plan.

**Subsequent repair update, September 7:** R01 is now implemented and independently verified for the current application-server Python transport boundaries; see [the R01 repair report](r01_provider_isolation_repair_20260907.md). This does not revise the original audit observations below. R02-R24 remain open. A first new browser run failed accepted-reload map acknowledgement; a diagnostic-only rerun passed. The intermittent reload failure remains unresolved, and the complete stability/E2E-1 gate is still not established. Old bundles remain historical evidence for their recorded source snapshots after this harness change.

This audit covers the local product website, Copilot interpretation/context/history, draft compiler, existing optimizer/evaluator, accepted-plan persistence, map/text/metrics presentation, and acceptance-test harness. It is not an exhaustive security certification, an all-California data validation, a full repository regression, or a research-baseline closeout. No new solver is proposed. D1/E3.1/E3.3/E4 research blockers remain separate.

Method: source inspection, three independent reviewers, zero-provider in-memory counterexamples, current manifest verification, and a fresh disposable-session browser inspection. Application source, user trip state, historical bundles, and authoritative status documents were not edited. Only this report and new audit artifacts were written. No billed provider call was made.

Evidence labels used below:

- **Reproduced:** observed in an executed current-code probe or current browser.
- **Source-confirmed:** control flow or a declared restriction was inspected; the full failure journey was not executed.
- **Coverage gap/risk:** required evidence is absent; this does not by itself prove a runtime defect.

## What remains verified

**September 8 continuation:** fixes and runtime tests are now in progress. See [the dated repair checkpoint](runtime_coherence_repair_progress_20260908.md) for bounded passes, retained new failures, exact input/output locations and remaining gates. The original audit observations below remain historical failure evidence; this link does not mark the complete phase passed.

The two September 5 bundles were reverified against their defined current source universe on September 7:

| Bundle | Current verification | Scope |
|---|---|---|
| `results/e2e/runtime_optimizer_repair_deterministic_20260905_v4` | PASS; 97 artifacts, 168 source inputs | Recorded seven browser tests, including accepted child/same-session/transcript reload; no provider calls |
| `results/e2e/runtime_optimizer_repair_replay_20260905_v5` | PASS; 42 artifacts, 170 source inputs | Recorded exact paid-output replay through real local optimizer/evaluator/browser; no new calls |
| `results/e2e/runtime_optimizer_repair_live_20260905_v1` | Historical paid evidence, not current-source standalone evidence | All 42 artifact hashes remain intact; current-source verifier rejects the two subsequent production-source changes, as previously disclosed |

The valid Bixby-to-Pfeiffer replacement was also recompiled in memory today and again generated eligible child `plan_c9a1dffb7461d665`. Strict route checks and canonical child/diff/evaluation bindings remain present. The existing execution method is `deterministic_context_aware_heuristic`, not an optimality certificate.

The new UI inspection exercised eight states: initial Trip, text route open, Compare, return to Trip, 1280, 1024, 390, and mobile Copilot open. All reached observed map readiness, with no page exceptions or chat/provider request. No horizontal document overflow was measured. Actual Trip map sizes were 651×630 at 1440×900, 491×558 at 1280×800, 605×558 at 1024×768, and 353×592 at 390×844. These observations do not prove visible-pixel tile coverage or mobile keyboard usability.

## Priority 1: actual semantic/runtime defects and immediate safeguards

### R01 — The nominally zero-cost test runner does not force zero-provider execution

**Source-confirmed risk.** `scripts/run_e2e1_vertical_slice.py:316` inherits the entire caller environment, does not override the adapter, and launches the ordinary `.env.local`-loading launcher. Its manifest writes a constant provider count of zero at line 157. Some browser scenarios actually submit a chat message. A caller configured for OpenAI can therefore reach provider dispatch before the later deterministic assertions reject it.

**Fix:** explicitly select the deterministic adapter before launching; preflight the adapter identity; install an outbound-provider transport guard; derive attempt counts from instrumentation. Do not rely on removing one API-key variable because the launcher can reload local configuration. Keep opt-in billed runs separate.

**Gate:** run the deterministic command under a deliberately OpenAI-configured parent environment with a fake transport. It must dispatch zero provider attempts and must never write a false zero-count manifest. No real billed failure test is necessary.

**Repair status:** verified within the application-server HTTPX/socket scope. Final focused regressions: 145 passed. Real browser v2: 7 passed; independent artifact/source/receipt verification passed, zero sent/blocked provider calls, 633 observed local connections. The preceding browser failure is retained; see the [complete report and exact I/O](r01_provider_isolation_repair_20260907.md). This is not an OS-wide egress or all-interaction completion claim.

### R02 — Missing evidence/execution failure is mislabeled as evaluated infeasibility

**Reproduced.** All candidate-addition counterexamples below returned `state=ineligible`, `reason=no_feasible_evaluated_child`, with `evaluation_executed=false`. `draft_compiler.py:250` classifies every zero-output execution this way. This is not proof that the proposed trip is infeasible.

**Fix:** distinguish `blocked_missing_route_evidence`, execution failure, timeout, unsupported request, not evaluated, and independently evaluated ineligibility. Return structured missing-leg findings and the corrective action. Keep the parent rendered, retain the draft, and disable Accept.

**Gate:** a missing route cell must produce blocked/not-evaluated, while an actually evaluated hard-constraint violation must produce ineligible with a certificate/finding. Update the regression that currently codifies the conflation (`tests/product_app/test_runtime_optimizer_mapping.py:121`).

### R03 — Add-candidate is advertised as executable, but every offered candidate/day pair fails

**Reproduced:** two exposed compiler candidates × seven days = **14 failing combinations**. Representative exact request: `add_candidate(pfeiffer_big_sur_state_park, day=5)`. Result: no child, no evaluator run, missing matrix cell `bixby_creek_bridge_viewpoint -> pfeiffer_big_sur_state_park`.

Sources: `draft_compiler.py:136` advertises enabled/executable capabilities; lines 318–360 obtain candidates from registered replacement bundles; lines 424–428 permit any day 1–7; lines 561–588 compile the addition.

**Fix:** make readiness contextual to operation/candidate/target/day/parent/context. Validate all insertion legs before offering execution. Provide a specific route-evidence blocker until those legs exist. A statewide discovery catalog is not an executable statewide optimizer universe; the same route/readiness process must onboard further California POIs.

**Gate:** one actual browser addition preserves the original stop, adds a distinct new stop, runs the real evaluator, and changes the canonical markers/route. Every unsupported candidate/day combination is explained before an allegedly ready preview.

### R04 — Preservation-only instructions inject unrelated replacement candidates

**Reproduced:** `keep_stop(golden_gate_bridge)` attempts an unrelated San Francisco airport-to-Pfeiffer leg and fails before evaluation. `draft_compiler.py:574` uses `requested_candidates or set(self._candidates)`; a candidate-free preservation request consequently explores all registered replacements.

**Fix:** derive candidate generation from the actual operation. An already-preserved stop should be reported as already preserved or retained as an explicit constraint; it should not trigger optional replacement exploration.

**Gate:** keep-Golden-Gate adds no Pfeiffer/Point-Lobos candidate. Keep-Golden-Gate + replace-Bixby executes exactly the intended replacement and preserves Golden Gate.

### R05 — Replacement target validation depends on unrelated draft length

**Reproduced:** replacing Golden Gate with the Bixby-only Pfeiffer candidate is rejected alone (`draft_candidate_target_mismatch`), but adding an unrelated keep-Stearns operation bypasses that check and enters the optimizer. `draft_compiler.py:798` performs target compatibility only for a single-operation draft.

**Fix:** validate every operation's target/candidate compatibility before compilation, independent of draft length. If arbitrary pairings are intentional, replace the contradictory restriction with one contextual route/day-readiness policy.

**Gate:** the same invalid pair receives the same pre-execution finding alone and in every mixed draft.

### R06 — Copilot context can identify a removed stop as still accepted

**Reproduced in a synthetic in-memory accepted-plan context; not a new accepted-browser test.** `service.py:638` builds Copilot context from the registered run view, and line 724 classifies immutable original-parent stops as `accepted_stop`. A different accepted plan ID does not change that list. The probe consequently answered that Bixby was already in the accepted itinerary. Registered-repair advice at line 988 likewise uses the original bundle.

**Fix:** project provider-safe context from the actual accepted/rendered plan and current evaluation: plan ID/hash, ordered stops/day assignments, route IDs, capability state, and acceptance/read-only state. Review advice must be explicitly scoped to the active artifact.

**Gate:** accept Pfeiffer, reload, ask about Bixby/Pfeiffer and the current repair. Prompt context, response, advisory IDs, map, and timeline must all agree with the accepted child.

### R07 — LLM and optimizer capabilities are separate and contradictory contracts

**Source-confirmed.** `copilot_prompt.py:32` permits lock/flexible/day-move/route-feedback operations that the coherent compiler rejects at `draft_compiler.py:312`. `models.py:47` has no operation-capability or target-compatibility map. Interpretation validation checks IDs but not all current execution restrictions. Conversely, supported direct duration/order edits are missing from the chat intent vocabulary.

**Fix:** reuse the compiler's capability contract as the source for prompt descriptions, structured validation, draftability, and visible actions. Include context-specific restrictions, disabled reasons, draft-only versus evaluated status, and allowed combinations. Reject or clarify unsupported intentions before `proposal_ready`, keeping the backend guard as a final check.

**Gate:** machine-check every enabled/disabled operation through chatbox → interpretation → draft → API → optimizer/evaluation → UI. A valid JSON intent alone must not be described as executable.

### R08 — Pre-model keyword shortcuts misunderstand negation and discard clauses

**Reproduced with zero provider attempts:**

| Exact input | Actual disposition |
|---|---|
| `Please do not remove Golden Gate Bridge.` | Unsupported stop-removal response |
| `Skip the explanation and review the itinerary.` | Unsupported stop-removal response |
| `Keep Golden Gate Bridge and move Bixby Bridge to day 3.` | `proposal_ready` containing only keep-stop; move clause omitted |

Sources: `copilot.py:84`, `openai_copilot.py:245`, `openai_copilot.py:467`, `copilot.py:590`.

**Fix:** restrict shortcuts to unambiguous single-action forms, or classify the complete message before applying deterministic semantic guards. Preserve a disposition for every requested clause: supported, unsupported, or clarification required. Never silently omit a clause.

**Gate:** negation, quoted verbs, non-stop objects of “skip”, conjunctions, multiple entities, and mixed supported/unsupported requests. The exact input and per-clause result must remain inspectable.

### R09 — Text route and initial repair summaries still mix parent and registered-child truth

**Reproduced in the actual browser.** With `PARENT_RENDERED` and Bixby in the map/timeline, opening Text route description shows “Day 5 (affected): Pfeiffer Big Sur State Park” and “Eligibility: Eligible repair”; the same panel's stop order says Bixby. The initial right panel uses past-tense “Replaced Bixby … with Pfeiffer” while preview is still required.

Source: `app.js:3833` falls back to `workspace.map_alternative` instead of deriving the description from `displayed`; `app.js:3247` and `app.js:3278` fall back to workspace repair eligibility. Evidence: `results/audits/runtime_concerns_20260907/text_route_1440.png` and `ui_observations.json`.

**Fix:** one active presentation context must own map, text, status, explanation, metrics, and selection. A registered recommendation may be shown separately as a proposed change, never as the current route or an executed change. Eligibility must be exact to plan/hash/evaluation; otherwise say not evaluated for this context.

**Gate:** fresh parent, staged draft, eligible preview, ineligible preview, Compare selection, acceptance, reload. Every textual stop name and status must agree with its labeled artifact scope.

### R10 — Metrics and Compare do not fully represent the evaluated selected plan

**Browser/source-confirmed.** Before preview, the global strip shows preservation 88% and edit cost 6 from registered repair while the parent map is rendered. `app.js:4431` uses fixture repair whenever no evaluated proposal exists and uses candidate metrics for ineligible proposals while parent remains rendered. Metric DOM has no plan/hash provenance. `compare.js:92` hard-codes baseline travel time unavailable; `app.js:4459` explicitly sets evaluated-preview travel time to null. The text route's duration also appears unavailable.

**Fix:** expose backend-owned plan-bound metrics with units, baseline/selected values, delta, missingness reason, evaluation identity, and exact scope. Do not fabricate missing values. Label ineligible-candidate diagnostics separately from the rendered parent's metrics. Add stable identity attributes and compare visible values to payload facts in tests.

**Gate:** inspect actual numbers, duration, booking changes, tradeoff and explanations—not just the presence of a certificate label. Selecting a different option updates every panel or explicitly labels any baseline/reference panel.

## Workflow and conversation limitations

### R11 — Acceptance is currently the end of editing, not the next conversational turn

**Source-confirmed, already disclosed limitation.** `service.py:274` disables editing after acceptance; `ensure_editable_baseline` returns `accepted_plan_requires_new_repair_baseline`; `draft_compiler.py:183` requires the original compiler parent.

**Fix:** construct the next repair baseline from the immutable accepted child, persisted context/route/schedule artifacts and decision lineage. Until then, disclose the limitation before acceptance.

**Gate:** A → accept B → reload → different repair → accept C → restart; B.parent=A and C.parent=B, with preserved history and exact render/evaluation identities. “Do edits one at a time” is not a complete workaround until this exists.

### R12 — Normal multi-part edits are not supported as one evaluated draft

**Reproduced.** The valid replacement succeeds alone; replacement + independent Golden Gate duration edit raises `draft_evaluated_operation_combination_unsupported`. Duration/order/role/time-window families are currently restricted to homogeneous combinations (`draft_compiler.py:188`, `service.py:1375`).

**Fix:** first expose combination restrictions before staging. Then compile allowed combinations into one child and independently evaluate the combined result once. Do not silently drop a clause or accept a partial result.

**Gate:** replacement + duration, order + time window, valid + unsupported. Each either produces the full requested child or a complete draft-level blocking explanation.

### R13 — Clarification and explanation specificity are erased

**Reproduced with an injected valid response, no real LLM.** “Which destination day should I use?” becomes “Which itinerary day, stop, or trade-off should I review?” `openai_copilot.py:301`, `:422`, and `:522` replace provider text/questions with broad fixed copy.

**Fix:** retain the safety boundary, but represent resolved target, missing fields, allowed choices, blocking reasons, and evidence facts structurally; render specific local questions/explanations from them. Do not merely permit unrestricted provider prose.

**Gate:** a supported incomplete request asks only for its missing slot; the next answer completes that same pending proposal. Unsupported moves must first explain the real capability restriction, not request details for an action that cannot run.

### R14 — Conversational memory loses structured context and decisions

**Source-confirmed limitation.** History is a bounded recent text window (12 messages / 12,000 characters), excluding typed interpretations, their resolved targets, disposition, and structured decisions (`openai_copilot.py:369`, `copilot_prompt.py:225`). Repeated canned assistant messages make that window less informative.

**Fix:** add small explicit plan-bound state for pending clarification, resolved IDs, confirmed preferences and proposal disposition. Keep proposed memory distinct from accepted plan constraints. Expose clear/reset and invalidate stale plan-bound references.

**Gate:** pronouns, corrections, preferences beyond six turns, explicit reset, and accepted-plan transitions; “do the same for the other stop” must resolve or ask a precise question.

### R15 — Provider retry/cancellation has no durable attempt reservation

**Retry behavior confirmed by executed tests; duplicate billing after uncertain failure is an unmeasured risk.** `openai_copilot.py:339` explicitly allows two attempts. `service.py:772` uses process-local locks and completed-turn lookup, not a durable pre-dispatch record. Cancellation or revision change can discard a paid response before persistence.

**Fix:** persist client-message ID, input/context hash, allowed attempt budget, attempt count, and pending/completed/unknown outcome before dispatch. Default and UI retry policy must agree. Recover a recorded result when available; do not interpret an unknown outcome as fresh billing permission.

**Gate:** fake-provider completion followed by disconnect/cancel/restart/revision change; repeated same-ID request cannot exceed its budget or lose the truthful unknown/completed state.

### R16 — Conversation capacity is checked after provider dispatch

**Source-confirmed risk.** The 200-turn limit is enforced in `conversations.py:200` during append, after `service.py:776` has called the provider. File-size/write failures can similarly strand an obtained result.

**Fix:** preflight and reserve capacity before dispatch, with an explicit start-new-conversation state. Preserve recoverable outcomes if storage later fails.

**Gate:** synthetic full 200-turn conversation → submit → zero provider attempts; separately test 1 MiB capacity and write failure without making paid failure calls.

### R17 — Retention wording does not describe actual expiration

**Source-confirmed policy mismatch.** The UI says transcripts are stored locally for 30 days. Each new turn extends expiry for the entire conversation (`conversations.py:219`), and cleanup occurs on application initialization/access. This is effectively 30 days after last activity, not strict per-message deletion.

**Fix:** choose and accurately disclose sliding or absolute retention. Distinguish transcript deletion from separately retained exact-I/O audit evidence. Keep synthetic evidence local and secrets excluded.

**Gate:** clock-controlled active/inactive expiry tests, explicit deletion-scope tests, and UI copy matching the chosen policy. No secret exposure was discovered by the inspected boundaries, but this was not a complete penetration test.

## Verification, UI and maintainability concerns

### R18 — The quantitative visible-map-coverage gate is still unproven

**Coverage gap, not a demonstrated gray-map regression.** `map-controller-v2.js:433` computes loaded source count / source count. One loaded source yields 1.0; this is not 100% visible tile pixels. Canvas/container size equality is useful but insufficient for the user's 95% visible-area requirement.

**Fix:** name source readiness accurately and add independent visible-region coverage or blank-region checks using rendered tile bounds/visual evidence. Cover missing/delayed tiles, source errors, WebGL loss, resizing and panel transitions. Report measurements separately.

**Gate:** a deliberately incomplete visible map cannot be marked ready even if a source-level proxy says loaded. Preserve the existing observer/generation/geometry checks.

### R19 — The completed-response A/B race is not actually tested

**Coverage gap.** `test_e2e1_vertical_slice_live.py:1218` delays A before sending it. B finishes; A is then sent and rejected with `409 stale_session_revision`. That proves stale-request rejection, not handling of an eligible A response already completed by the real backend and delivered late.

**Fix:** retain and rename the existing test; add A-completes/response-held, distinct B-completes/renders, release-A. Hold only response delivery, not the optimizer/API internals; use distinguishable children such as Pfeiffer and Point Lobos.

**Gate:** request/plan/hash/marker/geometry/metrics/explanation/selection all remain B after late eligible A delivery.

### R20 — Enabled interactions and negative cases are not a complete one-command browser gate

**Coverage gap.** The runner executes one browser file. Its core lacks combined valid/invalid drafts, eligible result with missing geography, eligible addition/order/duration/time-window journeys, and an aggregated HCI/waypoint/recovery suite. The documented matrix at phase-plan line 714 lists mandatory rows without concrete handlers/selectors/tests and contradicts current move-day support.

**Fix:** make capability-to-handler/state/API/service/artifact/selector/map/test coverage machine-checkable. Aggregate deterministic interactions, exact-output replay and existing recovery tests under one strictly zero-provider command. Keep live billing separately opt-in.

**Gate:** an enabled action without a passing corresponding journey fails the gate. Unsupported actions have explicit negative journeys. No HTTP 200, mounted map or screenshot-only shortcut qualifies.

### R21 — Screenshot/readiness and mobile interaction coverage are inconsistent

**Coverage gap plus observed usability concern.** The original final replay omitted a Compare screenshot and did not click the text route control. The fresh audit now captures both and found R09. Some deterministic tests still use raw screenshots/identity-only waits rather than the stable capture helper. Current mobile evidence mostly resizes a desktop journey; it does not establish touch, keyboard, focus, draft/preview/accept usability.

The fresh desktop text panel is visually crowded by the status banner and duplicate Select/Edit map controls; mobile Copilot fully covers the map. The latter can be an intentional chat layout, but there must be a clear route back to the preview.

**Fix:** reuse one readiness/timestamp/stable-geometry capture helper; test actual mobile interactions, virtual-keyboard viewport, focus return/trap, Escape, and accessible text. Consolidate overlapping controls and document intentional panel scroll ownership. Do not treat hidden horizontal overflow as proof that controls are reachable.

**Gate:** full-size initial/reload/module/Compare/text/map-ready/mobile evidence with identities, measured dimensions, console/network records and screenshot-after-ready timestamps. New screenshots supplement structural assertions rather than replacing them.

### R22 — Synchronous optimization can block the async web server

**Source-confirmed architecture; performance impact unmeasured.** `api.py:471` is async but calls synchronous `service.preview`, which runs the compiler/solver directly.

**Fix:** use bounded worker execution appropriate for CPU/blocking work, with request/revision guards, cancellation and truthful timeouts. Do not introduce a new solver or remove strict validation.

**Gate:** a deliberately slow repair must not freeze independent health/status/chat requests. Bound concurrency and verify late/cancelled results cannot overwrite newer state.

### R23 — Authoritative tracking documents are stale and overstate parts of the gate

**Source-confirmed.** CP-010 still points to August evidence and pending review; the phase plan describes the retired precomputed/registered path, old IDs, and categorical implementation completion. Its operation table includes unsupported move-day behavior.

**Fix:** add a dated current-truth checkpoint, mark historical checkpoints as historical, correct the actual capability/traceability matrix, and link September evidence plus these unresolved concerns. Preserve research statuses and old evidence. Do not tick whole-phase completion merely because the primary replacement passed.

**Gate:** source capabilities, test manifest, current status, UI copy, and public completion claims must agree.

### R24 — Reproducibility stops short of the full runtime environment

**Coverage gap.** The source universe in `scripts/e2e1_source_universe.py:14` does not bind the entire dependency/browser/renderer/tile archive/OSRM dataset environment. Source-current is not a byte-for-byte runtime attestation. Static JS version query strings also remain manually dated; mixed-generation/cached-client behavior is not verified.

**Fix:** add environment versions/lock hashes, browser and renderer versions, map archive/routing dataset identity, and an allowlisted secret-free test environment. Use a consistent build/content fingerprint and verify cache revalidation across an update rather than relying on Ctrl+F5.

**Gate:** clean-start reproduction reports all external inputs, and a cached prior page either updates coherently or explicitly requests reload. Do not claim the old runtime can be recreated from source hashes alone.

## Proposed repair sequence

These are proposed phases, not implemented changes. Each must preserve proposal → draft → evaluated preview → explicit acceptance.

| Phase | Status / dependencies | Scope and entry | Exit evidence |
|---|---|---|---|
| A — Spend safety and truthful baseline | In progress: R01 verified in its bounded scope; R02/R15/R16 remain open | R01, R02, R15, R16; isolate deterministic provider execution, distinguish failure states, preflight capacity and attempt budgets | Fake-transport/no-cost guard tests; missing-evidence versus evaluated-ineligible browser tests; no misleading zero-call report |
| B — One current-plan and capability contract | Ready; can begin alongside A using zero-provider tests | R05–R10, R13; reuse canonical plan/evaluation/capability data for compiler, chat, text, metrics and suggestions | Accepted-context and initial-parent truth tests; complete per-clause disposition; consistent rejection independent of draft length |
| C — Executable operation/data coverage | Depends on B's contract | R03, R04, R12; correct candidate-free keep, validate insertion route evidence, explicitly define supported combinations | At least one eligible real addition; candidate/day negative matrix; compound draft either fully executed or explicitly blocked |
| D — Continued repair and structured conversation | Depends on B/C and persistence evidence | R11, R14; child-as-next-parent baseline, pending clarification, confirmed preferences and proposal history | A→B→C acceptance/restart plus multi-turn clarification/correction/reset; no stale-plan context |
| E — Map/UI assurance and responsiveness | Can proceed alongside B/C; depends on active identity contract for final checks | R18, R21, R22; real coverage measurement, shared capture helper, mobile/accessibility, bounded worker execution | Failure injection, visible-map coverage, four-width actual interactions, responsive concurrent requests |
| F — Recurring integration/reproducibility gate | Joins A–E; currently blocked by their unresolved findings | R17, R19, R20, R23, R24; complete action matrix, true A/B response race, exact I/O/environment inventory, policy/status reconciliation | One zero-provider aggregate command plus separately authorized paid website sample; independent review; no new action may bypass its traceability row |

Engineering dependency chain: **B → C → D → F**. A and zero-provider B work may proceed concurrently; A is a mandatory spend-safety prerequisite for billable/live rollout and the final combined gate F. Map/UI work E can run in parallel after identity/capability interfaces stabilize. Retention wording and dated documentation corrections are small independent tasks, but must not advance completion labels. Solver performance/optimality research remains a separate path.

Recommended next implementation: **A and the active-plan/capability corrections in B**, before broader chat intelligence or additional California POIs. Add reproductions first; fix guards and presentation truth; then widen the executable operation surface.

Rollout/rollback: retain immutable plans and old evidence, create new evidence roots, version any public context/response schema change, and keep new operations disabled until their browser gate passes. Never restore a precomputed shortcut to obtain green tests.

## Exact inputs/outputs, commands and audit artifacts

New audit artifacts are under `results/audits/runtime_concerns_20260907/`:

- `inspect_ui.py` and `ui_observations.json`: exact browser actions, displayed page text, dimensions, dataset identities, request URLs/methods, page exceptions and explicit no-chat/provider observations.
- `*.png` and `*.readiness.json`: actual website captures after observed readiness; Trip captures reuse the stable current-source helper; Compare records before/after acknowledgement equality. These are diagnostic UI captures, not independent pixel-coverage proof.
- `optimizer_probes.py` and `optimizer_counterexamples.json`: complete reproducible synthetic operation probes and exact public compiled outputs/failure artifacts.
- `copilot_probes.py` and `copilot_counterexamples.json`: exact synthetic context/message/injected typed response and returned interpretation; fake transport is labeled, not represented as GPT output.

Commands actually executed by the audit/reviewers:

```powershell
python scripts/verify_e2e1_vertical_slice.py --output-dir results/e2e/runtime_optimizer_repair_deterministic_20260905_v4
python scripts/verify_e2e1_live_copilot_chat.py --output-dir results/e2e/runtime_optimizer_repair_replay_20260905_v5
python scripts/verify_e2e1_live_copilot_chat.py --output-dir results/e2e/runtime_optimizer_repair_live_20260905_v1
python -m pytest -q -p no:cacheprovider --assert=plain tests/product_app/test_openai_copilot.py -k 'transient_status_retries or transient_network_failure or canonical or provider_claim or cancellation_is_propagated or total_deadline or quota_exhaustion'
python results/audits/runtime_concerns_20260907/inspect_ui.py
python results/audits/runtime_concerns_20260907/optimizer_probes.py --output results/audits/runtime_concerns_20260907/optimizer_counterexamples.json
python results/audits/runtime_concerns_20260907/copilot_probes.py
```

Results: first two verifiers PASS; historical paid current-source verifier fails as expected with `e2e1_live_source_binding_invalid` while all paid artifact hashes remain intact. Selected adapter regression: **10 passed, 33 deselected**; this confirms current retry/canonicalization behavior, not an absence of semantic concerns. Final UI inspection: **8 recorded states, no page exceptions, no chat/provider requests**. Optimizer reproduction: **22 compiler calls, seven counterexample assertions reproduced**, including the valid replacement control. Copilot reproduction: **five counterexamples reproduced, zero real provider attempts, one explicitly injected fake response**. Counterexample assertions passing means defects were reproduced, not product acceptance passed. Browser backend used `PRODUCT_COPILOT_ADAPTER=deterministic`, origin 8127, and isolated `tmp/runtime_concern_audit_20260907`; it did not use the user's persistent state.

The optimizer evidence is approximately 15.7 MB because complete canonical inputs and outputs are retained. Its script refuses to overwrite an existing output; use a new output filename for reproduction. Audit probe execution also set `PYTHONUTF8=1`, `PYTHONIOENCODING=utf-8`, and `PYTHONDONTWRITEBYTECODE=1`. No report JSON contains a real provider credential. Fake request objects are labeled as injected client arguments, not an actual paid HTTP wire capture.

Audit setup failures are not product failures: the normal sandbox helper failed; the installed patch-tool location had changed; initial in-memory reviewer probes needed `src` on the import path; and the first UI wait targeted the map-stage wrapper rather than the geographic map element owning readiness. Those were corrected before the evidence above. An early diagnostic capture lacked stable before/after resize checking; the final audit reran with the existing stable helper. The normal app was not running at audit start; no claim of continuous uptime is made.

The full regression suite, an actual new billed call, mobile keyboard journey, concurrency load test, and a completed-response A/B race were not run in this audit. Findings describe what was demonstrated or remains unknown, not a guarantee that no other concerns exist.
