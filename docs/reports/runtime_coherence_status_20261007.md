# Runtime integration status — October 7, 2026

## October 8 exploratory live chat review — wording and preview gaps remain

The user-approved variation review completed **13 cases / 14 website messages** in OpenAI mode. The Chinese replacement-plus-90-minute request produced the correct eligible child and restored it after reload. Two English variations were rejected by the local guard; the Day 4 order draft failed at preview; an arrival window became a latest-departure bound; preference recall omitted nature. Duration and role edits evaluated correctly but remained ineligible because the seeded Bixby weather failure was unresolved. Five ambiguity, conflict, external-action and keep-original safeguard cases behaved safely.

The [screenshot report](chat_input_review_20261008.html) records every tested action and its result, with 69 original screenshots, conversation text, stored interpretations and certificate reasons. [Readable findings](chat_input_review_20261008.md) distinguish observed defects from expected safety blocks. At the completed-test checkpoint (14:08:53 UTC), all 143 production-source hashes and both saved workspace files matched their pre-review hashes. During report assembly, `static/js/copilot.js` changed concurrently at 14:10:56 UTC; the newer bytes are untested by this run. This review made no product fix. No repair was accepted and no phase gate is promoted. Some messages are handled by local guards; this message count is not a paid provider-call count. Earlier full regression and browser gate evidence remains separate.

## Current result — saved-trip access isolated from optional map probes

**Bottom line:** the fresh late-response website test passed and independently verified twice. The conversation rerun passed 13 of 14 cases, including the previously failing accepted-child chat recovery, but failed one accepted-trip map reload. That suite remains failed; capability controls were not run after it.

The reload retained the accepted child but received repeated `map_probe_timeout` health/config results. Source review and four red counterexamples confirmed a separate blocking boundary: normal trip access re-ran optional map probes and waited behind an in-flight map probe. A two-file correction now reads initialized core readiness without that network wait. Core failure or absent service still returns 503; explicit map/routing health and actual tile readiness are unchanged. **91 focused core/map checks and repository-wide Ruff pass**, with exact unchanged inputs. This is not yet a passing reload-browser result.

The real API-to-Node harnesses also passed 48 checks, including four successive accepted edits and restarts. A bare Node invocation retained 200 passes / ten missing-required-input failures / one optional skip; it is not counted as a complete frontend gate. The existing fixture-generating harness, not weaker assertions, supplied the required canonical inputs.

Next: fresh map-recovery, conversation and capability-control browser suites, then complete frozen regression and the 72-case aggregate. No phase promotion. Exact red/green source, inputs and outputs: `results/repairs/core_map_probe_isolation_20261008_v1/`; retained failed browser run and exact chat/SDK fixture I/O: `results/e2e/adopted_runtime_conversation_20261008_v1/`; frontend I/O: `results/repairs/adopted_runtime_gate_fix_20261008_v1/frontend_api_check_io.json`. No new live/billed LLM input or output; SDK transport fixtures are explicitly non-live. Saved user workspaces were unchanged when the normal service was restored after the failed rerun.

The earlier checkpoints below retain their recorded source scope; they do not attest the latest production correction.

## Current result — verification blockers repaired; website rerun next

**Bottom line:** eight focused test/harness corrections are adopted. The current repository passes 347 focused tests and repository-wide Ruff, with all checked source inputs unchanged during that run. This is not full product acceptance.

- **What changed:** repeated provider-isolation guards no longer accumulate permanent inactive audit hooks; the browser waits for the actual editor dialog; unsupported combination evidence now tests a genuinely unsupported range-duration edit, not a supported exact-duration edit. Independent readers reject misleading control/input receipts. No feature, provider credential, production timeout or saved trip was changed.
- **Why it matters:** the real API/restart sequence that took about 56 seconds in preview now takes about 3.2 seconds with the corrected guard, under the same 60-second deadline. This isolated sequence passes 37 tests; it is not yet proof that the entire browser suite is fixed.
- **Honest baseline:** the previous frozen regression ended at 2,819 passed / one failed / 75 skipped, with all 977 frozen inputs unchanged. It stopped on a preview timeout before persistence tampering or restart. The previous browser aggregate credited 21 cases from six passing suites; the late-response and conversation suites failed, and 15 suites were not run. Those runs remain failed and preserved.
- **Next gate:** rerun the affected real-browser suites, then the complete frozen regression and 72-case / 23-suite website aggregate. Fresh source-bound paid website proof, native phone-keyboard evidence, crowded overlays and the independent per-concern audit remain open. R12/R20/CP-010/E2E-1 stay in progress; G5 is blocked; W6-W8 and research do not advance.

Exact before/candidate/adopted file bytes, hashes, incremental diffs, commands and complete outputs: `results/repairs/adopted_runtime_gate_fix_20261008_v1/check_io.json`. Exact real API timing requests/responses and old-versus-corrected sequences: `results/repairs/role_preview_timeout_20261008_v1/`. **New LLM input/output: null / null; no new live or billed call.** Credentials and credential verifiers are omitted. The normal service was checked ready on port 8127, PID 4244, build `9cba8e46f6fc5ef1`, before the disposable browser rerun; its lifecycle record will identify the restored process. Saved user workspaces remain unchanged.

Older checkpoints below retain their historical source scope and process IDs; they are not current service-state claims.

## Latest result — recorded-test inputs work without old result folders

**Bottom line:** the verifier and presentation tests no longer depend on ignored historical results on this computer. Their historical inputs are now portable, hash-checked fixtures. The website and original evidence are unchanged.

- **Proof:** 319 Python tests pass both locally and in an isolated copied tree with historical E2E/repair runs absent. All 807 copied input files remain unchanged; the file-open audits record zero historical-directory read attempts. Four Node contract files produce 29 passes and one explicitly optional external-preview-receipt skip in both trees. The skip is not counted as acceptance evidence. Repository-wide Ruff passes.
- **Data:** the baseline audit found 1,213 historical inputs totaling 903.2 MB. The selected fixtures retain 367 exact original files totaling 403.1 MB, deduplicated into 323 gzip blobs totaling 66.7 MB. Compressed and decompressed hashes are checked, and originals are independently rechecked unchanged. Current authorization-test context comes from the actual service, with fresh copies for each counterexample. The historical fourteen-case source-rejection unit is narrowed to the actual selection-generation verifier; it does not certify that old whole browser bundle. The full real-browser inventory is unchanged.
- **Scope:** these are historical unit-test inputs, not fresh website or paid-provider proof. The previous four-case real website safety bundle is independently rechecked twice against its unchanged 198 source inputs. The complete fresh frozen regression and **72-case / 23-suite** aggregate remain required. Browser-harness setup source-pinning review, native OS keyboard evidence, crowded overlays and independent per-concern audit remain open. No R12/R20/CP-010/E2E-1/G5/W6-W8/research status advances.

<details>
<summary>Exact inputs, outputs and retained failure</summary>

Exact original file paths, byte counts, hashes and compressed output locations: `tests/product_app/fixtures/recorded_runtime/index.json` and `results/repairs/portable_recorded_fixtures_20261007/fixture_generation_io.json`. Exact before/after code, incremental diffs, command/stdout/stderr, original-file readback, copied-tree inventory, unchanged saved-workspace hashes and normal-service health: `results/repairs/portable_recorded_fixtures_20261007/repair_io.json` and `exact_io_inventory.json`. The isolated copied tree uses existing installed dependencies; this is not a new-environment installation test. Only configured, source-pinned route-delta product inputs remain under its results tree; historical acceptance result folders are absent.

The first migrated run remains rejected: 316 Python passes / three failures; Node 29 passes / one optional skip; lint rejected fixture imports. Two failures identified report inputs missed by the audit (including a Windows-native copy not observed as a Python file-open event). The third used the generic core credential scanner on a replay manifest's structured billing-authorization policy object. Inspection found a policy object, not a leaked secret. The corrected check reuses the existing live-evidence credential scanner, which validates that exact policy shape; no production verifier is relaxed. Fixtures are registered once through pytest's existing plugin mechanism. Exact failures and initial fixture index are retained in `red_migration_v1/`.

**New LLM input/output: null / null. No new live or billed provider call.** Historical exact provider requests and raw structured outputs remain unchanged in the recorded fixtures; they are explicitly historical, not authorization for another call or evidence of current connectivity. API keys, authorization headers, mutation tokens and credential verifiers remain explicitly omitted. The website stays on port 8127, PID 3544, build `9cba8e46f6fc5ef1`; saved user workspaces are unchanged. Minimal-implementation reused existing verifiers and native gzip readers shared by the affected tests, without adding a dependency or changing product behavior.

</details>

**Next:** review browser-harness setup source pinning, then complete the fresh frozen regression and real website aggregate. Older checkpoints below retain their original evidence scope.

## Latest result — queued-preview safety is now an automatic website gate

**Bottom line:** the previous website fix is now protected by a mandatory, independently verified browser suite. Deleting chat while one-click Preview waits cannot launch the old chat's queued repair. This passes for **one or two edits, on both the original trip and a genuinely accepted child**. The displayed trip stays unchanged, and the staged draft survives reload.

- **Actual website:** four cases passed in 317.93 seconds. Real controls, local APIs and accepted-child setup are used; only delivery of an already-completed original draft-impact response is delayed. Response bodies are not replaced. No internal API is mocked.
- **Independent evidence:** the runner accepted 304 artifacts and 198 unchanged source inputs. The saved bundle is rechecked twice separately. Twenty full-size 1440×900 captures occur after map readiness; projected loaded-tile area coverage is 1.0 in every capture. This is not pixel classification or native mobile-keyboard evidence. A fresh screenshot was visually inspected: tiles fill the bounded map, while crowded overlays remain an open usability issue.
- **Local checks:** 420 passed, including 40 deliberately disconnected/misleading receipt counterexamples which must be rejected. Repository-wide Ruff and exact four-node browser collection pass.
- **Recurring protection:** the one-command aggregate now requires **72 cases / 23 suites**, including these four safety cases. Registration and this focused pass are not a completed 72-case aggregate.

Normal website is restored on port 8127, PID 3544, build `9cba8e46f6fc5ef1`; map/routing are ready and OpenAI is configured. Both saved user workspace hashes remain unchanged. This increment changes the existing runner, verifier and tests, not production behavior or credentials. **LLM input/output: null / null; measured live provider sends and blocked attempts: zero. No new billed call.** Prior paid connectivity evidence retains its earlier source scope and is not relabeled as a fresh call.

<details>
<summary>Exact input and output — all four browser journeys</summary>

| Context | Exact chat input | Exact visible chat output before staging |
|---|---|---|
| Original, single | `Replace Bixby Creek Bridge with Pfeiffer Big Sur State Park` | `I prepared this typed suggestion for review. Nothing was staged or changed.` |
| Original, combined | `Replace Bixby Creek Bridge with Pfeiffer Big Sur State Park; set Griffith Observatory duration to 90 minutes.` | `I prepared all 2 requested changes as one combined suggestion. Nothing was staged or changed.` |
| Accepted child, single | `Set Pfeiffer Big Sur State Park duration to 90 minutes` | `I prepared this typed stop-edit suggestion for review. Nothing was staged or changed.` |
| Accepted child, combined | `Set Pfeiffer Big Sur State Park duration to 90 minutes; set Stearns Wharf earliest arrival to 10:00.` | `I prepared all 2 requested changes as one combined suggestion. Nothing was staged or changed.` |

After the user clicks Preview and then confirms deletion while impact delivery is held, each control journey makes **zero `/preview` requests** and does not accept another child. Original trip/map remains `plan_359968d255ff9288` / `750dae617fc4c1e3`; accepted-trip/map remains `plan_bc13d7ed8c7ec60b` / `ab610263c7f5a68f`. One or two complete canonical draft operations, respectively, remain unchanged. Accepted-child setup itself really runs replacement/preview/accept/reload before the safety test; that is distinct from the blocked queued control action.

Full exact HTTP/SSE requests and responses, pre-reload trusted clicks, context, draft, delayed bytes, map acknowledgements, source hashes, console/network records and readiness-bound screenshots: `results/e2e/queued_preview_registered_20261007_v2/conversation/`. Exact file inputs/outputs, incremental diffs, command outputs, independent readbacks and all retained failures: `results/repairs/queued_gate_registration_20261007/repair_io.json` and `exact_io_inventory.json`. API keys, headers, mutation tokens and disposable-state credential verifiers are explicitly omitted.

The first registered browser run remains failed: all four reached the map/draft/reload checks but timed out because the harness wrongly expected a new history request on opening Copilot; history was already fetched during reload. The corrected test binds that real reload response and is rerun fresh. The first unit reader also remains rejected: identical restored/reloaded GET bodies were mistaken for one occurrence; the correction uses the indexed occurrence, preserving strict ordering. Neither earlier failure is overwritten or relabeled.

</details>

**Next:** remove unit-test dependence on ignored historical result folders, then run the complete fresh frozen regression and aggregate. Native keyboard, crowded overlays and independent per-concern phase audit remain open. R12/R20/CP-010/E2E-1 stay in progress; G5 blocked; W6–W8 planned; research statuses unchanged. Minimal-implementation reused the existing lifecycle, provider-isolation, source-pinning and map-readiness checks instead of introducing another test framework. Older checkpoints below retain their original evidence scope.

## Current result — deleted chat cannot launch a queued preview

**Bottom line:** a real website race is repaired. Previously, deleting the conversation while one-click Preview waited on its real draft-impact response still launched the repair and changed the map. This happened for both one edit and a two-edit batch. The existing controller now rechecks conversation, session, trip, run and accepted-plan ownership after asynchronous boundaries; final preview submission also snapshots that ownership before waiting for selection. A legitimate selection may still advance the revision. Deletion preserves the already staged draft, but the old chat cannot launch its queued preview.

| Proof on the repaired production code | Result |
|---|---|
| Actual website deletion race | Two fresh cases pass, with trusted clicks, complete real HTTP, unchanged parent map, retained draft and reload; readback rejects the original failing captures and validates the corrected receipts |
| Actual-controller/submission checks | 55 Node test entries pass; a separate exact-I/O diagnostic passes all 14 single/grouped ownership cases |
| Focused product/chat regression | 156 passed; repository-wide Ruff and both changed JavaScript syntax checks pass |
| Normal successful repair journey | Seven real-browser tests pass; independent readback twice binds 119 artifacts / 194 unchanged inputs and zero measured provider sends |
| Normal billed website chat | One GPT-5.4-mini attempt, no retry; both edits reach real optimization/final evaluation and the same child in itinerary/map/metrics/Compare/Evidence, reload and four viewport captures; no acceptance |

### Exact input and output

Race input: `Replace Bixby Creek Bridge with Pfeiffer Big Sur State Park; set Griffith Observatory duration to 90 minutes.` The user clicks **Preview combined repair**, then confirms **Delete this local conversation** while the original draft-impact response is held after completion. Before the fix, one preview request executes and changes the parent map to a child. After the fix, **zero preview requests**, parent `plan_359968d255ff9288` / `750dae617fc4c1e3` remains rendered, both staged operations remain unchanged and survive reload. The single-edit case has the same result with one retained operation. Original response bytes are never replaced; only delivery timing changes. **LLM input/output: null / null; measured live provider sends and blocked attempts: zero.** This is pre-dispatch protection, not a claim that deleting chat cancels a provider or optimizer request already sent.

Full exact HTTP/SSE, controls, before/after context, unchanged source inputs, map acknowledgements and readiness-bound 1440×900 screenshots: `results/e2e/queued_preview_green_20261007_v2/conversation/`. The map screenshot was visually inspected; bounded tiles cover the map, while crowded overlays remain a usability issue. Coverage is loaded-tile projected union, not pixel classification or native mobile-keyboard proof. The first browser pass lacked a preserved pre-reload click receipt; its generated readback remains rejected. The corrected harness saves those clicks before reload, distinguishes the lean context projection from full draft artifacts and rebinds the delayed body to the real HTTP history. The fresh rerun passes without further production changes. Original failed browser and reader runs are retained.

Paid website input is the same complete two-edit sentence above. Exact raw provider assistant field: `Understood: replace Bixby Creek Bridge Viewpoint with Pfeiffer Big Sur State Park, and set Griffith Observatory to 90 minutes.` Exact locally validated visible response: `I prepared a typed itinerary suggestion for your review. Nothing was changed or executed automatically.` Full developer instructions, context, schema, request and raw structured response: `results/e2e/queued_guard_paid_20261007_v1/live_copilot_chat/provider_exact_llm_io.json`. Actual model `gpt-5.4-mini-2026-03-17`, low reasoning; 6,110 input / 428 output / 6,538 total tokens, zero cached tokens. Configured-rate usage estimate **$0.0065085**, not an invoice or newly verified price quote. Request SHA-256 `46160db25adcc73c69c74660279f84e6abaa99bd1e3b489b64f7d022761aae21`; response SHA-256 `3fea775f81cbe48ae308e0dc6c8543013867dae9434af75da5e2ca6a3257e650`. Child and map `plan_a31708c1c9ba0ab9`; independent verification twice binds 50 artifacts / 193 unchanged inputs. This proves the normal paid website path, not paid cancellation or an accepted-child race.

**Full gate remains open.** Frozen audit v2 was intentionally stopped before this repair, after the actual runtime defect was reproduced. Its 647 pinned files remained unchanged; the partial roughly 73% progress is not a passing test total or a completed full regression. No final JUnit was produced. Next: register these new safety journeys in the recurring aggregate, finish grouped/accepted-context and recorded-fixture portability gaps, then run the complete fresh frozen regression and aggregate. The existing inventory is still 68 cases / 22 suites and is not relabeled as containing these new tests. Native keyboard, overlays and independent phase audit remain open. R12/R20/CP-010/E2E-1 remain in progress; G5 blocked; W6–W8 and research statuses unchanged.

Exact before/after files, incremental diffs, API/LLM I/O references, command outputs, evidence readback and retained failures: `results/repairs/grouped_async_guard_20261007/repair_io.json` and `exact_io_inventory.json`. API keys, authorization headers, mutation tokens and disposable-state credential verifiers are explicitly omitted. Normal website restored on 8127, PID 12956, build `9cba8e46f6fc5ef1`; map/routing ready and OpenAI configured. Both saved user workspace hashes and revision zero remain unchanged. Minimal-implementation kept the fix in existing ownership guards; the API-key safeguard reused the approved key and bounded the paid check to one attempt. Earlier checkpoints below retain their historical source scope.

## Current result — whole-plan chat fixed; complete gate still open

**Bottom line:** the website no longer mistakes “Keep the original plan” for a request to preserve one stop. One shared, whole-message rule handles this in both deterministic and OpenAI adapters. It returns a non-mutating `keep_original` advisory, clears an obsolete pending clarification, and neither stages a draft nor spends a provider call. Negation and additional clauses are not discarded. The broad old keyword fallback was removed.

| Verified on this source | Result |
|---|---|
| Focused product, semantic and test-harness regression | 431 passed, including both adapters, pending clarification, negation and multi-clause counterexamples |
| Actual website chat regression | 13 passed, including whole-plan interpretation, Stop/Retry, truncated-response recovery, deletion and width checks |
| Fresh repair vertical slice | Seven actual browser journeys passed; independent readback verified twice, 119 artifacts / 194 unchanged source inputs, zero measured provider sends or blocked attempts |
| Fresh paid website journey | One GPT-5.4-mini attempt, no retry; both edits reach the real optimizer/evaluator and the same child in itinerary/map/metrics/Compare/Evidence, reload and four viewports; no acceptance |

The two recovery failures were stale selectors matching **Retry map** as well as **Retry**; the recovery implementation was not changed. Other harness repairs update the explicit 68-case / 22-suite inventory, include the combined-edit evidence record, and distinguish normal modes from the separately verified self-managed modes. Current-context authorization tests now obtain context from the actual service; the retained older fake context must be rejected, not relabeled current. Historical evidence bytes and production verification strictness are unchanged.

**Full-regression limit:** `results/regression/frozen_20261007_v1/` was intentionally stopped after confirmed failures, before source edits. All 647 pinned files were unchanged during that audit. It is failed/incomplete, not a completed full regression, and its partial progress is not a passing test total. Exact failure readbacks and the stop reason are retained. The corrected source still requires a complete fresh frozen regression and the full 68-case aggregate. Remaining grouped variants, recorded-fixture portability, native OS keyboard, crowded overlays and independent phase audit remain open. R12/R20/CP-010/E2E-1 stay in progress; G5, W6-W8 and research statuses do not advance.

### Exact input and output

Whole-plan website input: `Keep the original plan`. Exact visible output: `Keep original leaves the current accepted itinerary unchanged. It requires a separate confirmation; no decision was recorded and no draft was changed.` The real response contains only `keep_original`, with an empty draft and no preview. Full HTTP/SSE: `results/repairs/global_plan_intent_20261007/website_keep_original_http_io.json`. Complete before/after context and adapter outputs are in `red_exact_io/` and `green_exact_io/` under that repair directory. **LLM input/output for this local whole-plan rule: null / null; provider attempts: zero.** The 13 random-port chat tests exercise degraded-map UI, not tile-readiness acceptance.

Fresh billed website input: `Replace Bixby Creek Bridge with Pfeiffer Big Sur State Park; set Griffith Observatory duration to 90 minutes.` Exact provider assistant field: `Understood: replace Bixby Creek Bridge with Pfeiffer Big Sur State Park, and set Griffith Observatory to 90 minutes.` Exact locally validated website response: `I prepared a typed itinerary suggestion for your review. Nothing was changed or executed automatically.` The visible typed cards retain both edits. Full developer instructions, context, schema, request and raw structured output: `results/e2e/global_plan_paid_20261007_v1/live_copilot_chat/provider_exact_llm_io.json`. Actual model `gpt-5.4-mini-2026-03-17`, low reasoning; 6,112 input / 293 output / 6,405 total tokens, zero cached tokens. Configured-rate estimate **$0.0059025**, not an invoice or newly verified price quote. Request SHA-256 `bb577e950734961d24f5e042d64392d8077f47cc53df15787910829c94a8c7da`; response SHA-256 `8bed4f2c22e2439f1801f05a0c3220e1a2c266be01f703272c1e98e0cd984750`.

Paid runtime identity: child/map `plan_a31708c1c9ba0ab9`, content hash `dde0f994990b834a`, certificate `cert_8569ff5902645ae2`, geometry `f07301071e3038d8cb98f4ab7b33a03fa39b303f932ce2b861e38bbbbf5f3ff9`. Independent verification twice binds 50 artifacts / 193 unchanged inputs. The separate deterministic replacement renders child/map `plan_bc13d7ed8c7ec60b`; its 1440×900 capture has a 571×614 map, projected loaded-tile coverage 1.0, map ready `19:48:52.100Z`, screenshot starts `19:48:52.174Z`. This is projected tile union, not pixel classification. The real screenshot was inspected; crowded floating overlays remain disclosed.

Exact file input/output snapshots, incremental diffs, checks, retained red runs and complete I/O references are indexed in `results/repairs/global_plan_intent_20261007/repair_io.json` and `exact_io_inventory.json`. Retained failures include the initial stream-capture reader mistake; the corrected test observes a clone of the actual response without replacing it. API keys, authorization headers, mutation tokens and credential verifiers are omitted explicitly. Normal website restored on 8127, PID 2224, build `bc4efde7030a28e3`; map/routing ready and OpenAI configured. Both real saved workspace SHA-256 values and revision zero are unchanged. Minimal-implementation kept the repair in the shared parser; the API-key safeguard reused the approved credential and bounded the paid check to one attempt. Earlier checkpoints below remain source-bound history.

## Current result — direct editors checked end to end; full gate still open

**Bottom line:** the direct duration and time-window controls now have real-browser proof alongside chat. Eleven journeys passed in 1,794.87 seconds: seven chat paths and four native-editor paths. Allowed changes render the evaluated child and survive acceptance/reload; protected violations preserve the accepted parent and draft. The only production change in this increment corrects obsolete editor guidance that falsely said mixed edits were unsupported.

**Important evidence limit:** the original run's formal manifest remains **rejected**. Its new validator omitted the original A plan from the setup-screenshot allowlist for A → B → C cases. The corrected reader includes A only after verifying the actual baseline HTTP/lineage, requires the exact screenshot inventory, and rechecks complete child/diff/evaluation, persisted child, acceptance hashes, map, metrics, evidence and trusted controls. Corrected post-run readback passes all eleven cases. The sole post-run source difference is that reader; the 202 recorded inputs were unchanged during the browser run. This is not a fresh frozen-harness pass. The next complete aggregate must rerun the journeys with the corrected reader. Neither the rejected manifest nor a historical result is relabeled.

| Verified in this increment | Result |
|---|---|
| Actual website chat/editor journeys | 11 passed; real API/compiler/evaluator, no internal API mocking; zero measured provider sends and zero blocked attempts |
| Separate live GPT website journey | One actual attempt; same evaluated child in itinerary/map/metrics/Compare/Evidence, reload and four viewports; no acceptance |
| Local regression | 150 runner/validator checks, including 28 portable editor counterexamples; 207 domain checks; repository-wide Ruff and five script syntax checks pass |
| Recurring test inventory | Owned journeys added to the one-command suite: now 68 cases / 22 suites; inventory is not a passing full-gate claim |

### Exact input and output

Direct editor input: Day 7 → Edit stop → exact duration **90 minutes** → Save duration draft → Preview repair. The exact request uses `type=set_stop_duration`, `target=griffith_observatory`, `source=map`, and an exact 90/90/90-minute duration. Output is child/map `plan_ed40f482c56953dc` / `e1628f2e2ca66d9a`, certificate `cert_219512be029b737e`, geometry `920ed252efeda5c42d1df6edf16d5e4fe8453cd85e021be4926193bd23c24f07`; the persisted child equals the full response artifact. Day 1's editor input is earliest service admission **10:00**, latest departure blank; the exact window request targets Golden Gate Bridge with `latest_departure=null`. It produces child/map `plan_4346a2580338c931` / `843e58fd719c56e0`, certificate `cert_e619e6c7ad4702c8`. Trusted input, submit and Preview receipts, complete HTTP/SSE, rules, artifacts, screenshots and restoration responses are retained under `results/e2e/owned_ui_gate_20261007_v2/`. **LLM input/output: null for these deterministic cases.**

The 1440×900 duration-editor capture has a 651×630 map and projected loaded-tile coverage 1.0; map ready `18:42:23.399Z`, capture starts `18:42:26.386Z`. Coverage is projected tile union, not pixel classification. An actual full-size preview was inspected: tiles fill the bounded map, while crowded floating overlays remain a usability concern. Owned-rule cases are desktop journeys, not a new native-mobile-keyboard claim.

Fresh billed website input: `Replace Bixby Creek Bridge with Pfeiffer Big Sur State Park; set Griffith Observatory duration to 90 minutes.` Exact assistant output: `Understood: replace Bixby Creek Bridge with Pfeiffer Big Sur State Park, and set Griffith Observatory to 90 minutes.` Full developer instructions, context, schema and provider request/response bytes: `results/e2e/owned_editor_paid_20261007_v1/live_copilot_chat/provider_exact_llm_io.json`. Actual model `gpt-5.4-mini-2026-03-17`, low reasoning; 6,111 input / 365 output / 6,476 total tokens, no cached tokens, one attempt and zero retries. Configured-rate estimate **$0.00622575**, not an invoice or newly verified pricing quote. Request SHA-256 `e645ffb648a9b0f45719c8528abee5a91fdac4b0c6e5714093501fba6313d275`; response SHA-256 `0cc284880e723a90db54fd2ad4de866f6c1131d35372b66095a9e4e1f50385b6`. Child/map `plan_a31708c1c9ba0ab9`; the separate paid manifest binds 50 artifacts / 193 unchanged inputs. This is normal synthetic-demo context, not a live owned-rule fixture test.

Exact before/after files, the original and corrected reader, complete I/O locations, repeated readback and rejected runs are indexed in `results/repairs/owned_editor_gate_20261007/repair_io.json` and `exact_io_inventory.json`. API keys, authorization headers, mutation tokens and session credential verifiers are explicitly omitted. The initial red browser test reproduced the obsolete guidance; a runner setup failure and an incorrect unit-test outside-repository assumption are also retained.

**Next:** full frozen repository regression and the complete 68-case aggregate with the corrected reader; remaining grouped stale/foreign-owner variants, recorded-fixture portability, native OS keyboard, overlays and independent phase audit remain open. R12/R20/CP-010/E2E-1 stay in progress; G5/W6-W8/research statuses are unchanged. Normal service is restored on 8127, PID 12068, build `711ca6732009907f`; map/routing ready and OpenAI configured. Both saved user workspace hashes/revision zero are unchanged. No credentials, ACLs, Git publication or user trip were changed. Minimal-implementation guided reuse of the existing controls and canonical pipeline; the API-key safeguard reused the approved key and bounded the live call. Earlier checkpoints below are historical and source-bound.

## Current result — single schedule edits now reach the website

**Bottom line:** standalone duration and time-window edits on stops with saved attribute rules now reach the existing canonical-child/evaluation/map path. Allowed changes render and persist; protected violations leave the accepted trip visible and retain the draft for correction. This is a tested **R12 increment**, not completion of R12, R20, CP-010 or E2E-1.

The fix is small: the compiler and website choose the existing full-artifact receipt when a single schedule edit has an applicable accepted-parent rule. Ordinary unowned direct edits keep their existing detailed schedule contract. The website still validates immutable rules, before/after values, costs, child/diff/certificate identities and map evidence. Cost evidence does not grant permission. A standalone typed edit does not invoke a selection optimizer unnecessarily; its actual compiler and independent evaluator execute. Selection-containing repairs still run the real optimizer.

| Check on this production source | Result |
|---|---|
| Owned API / actual Node normalizer / persistence | Four new standalone cases passed; 12 mixed cases passed; soft and protected rules, real internal stack |
| Owned website journeys | Seven passed in 955.80 seconds: existing replacement/duration, mixed allowed/protected, standalone duration and window allowed/protected |
| Independent artifact rebuild | All seven complete child plans and diffs match; certificate semantics match except evaluation timestamp/hash; standalone edits preserve all unrelated stop fields and have no road changes |
| Regression | 207 domain and 100 product checks passed; seven standard browser tests passed in 229.331 seconds and independently verified twice |
| Fresh paid website test | One real GPT-5.4-mini attempt; real optimizer/final evaluation; exact child in itinerary/map/metrics/Compare/Evidence; reload and four viewport captures; independently verified twice |

The standalone allowed cases first fix and accept the demo's weather disruption through the real website/API, then edit the accepted child, accept again and reload: A → B → C. The protected cases keep B, keep the draft and disable acceptance. Changing a duration alone on the still-weather-infeasible original trip remains correctly ineligible; the evaluator was not weakened to make the test green. Owned-fixture chat is deterministic with zero measured provider sends and no mocked internal API/optimizer/evaluator. The paid call uses the normal synthetic demo, **not** an owned-rule fixture. The new owned cases are desktop chat journeys, not proof of every owned typed-editor control or native mobile keyboard.

### Exact input and output

Website input 1: `Set Griffith Observatory duration to 90 minutes.` Deterministic assistant output: `I prepared this typed stop-edit suggestion for review. Nothing was staged or changed.` After explicit staging and Preview, parent `plan_39bd61f60b6c3a84` produces eligible child/map `plan_ed40f482c56953dc` / `e1628f2e2ca66d9a`, certificate `cert_219512be029b737e`, diff `diff_46f51e2a61a84fa2`, geometry `920ed252efeda5c42d1df6edf16d5e4fe8453cd85e021be4926193bd23c24f07`. The actual child records 90 minutes and persists after acceptance/reload. Its 1440×900 screenshot has a 651×630 map, projected loaded-tile coverage 1.0, ready at `2026-10-07T17:33:32.931Z`, capture starts `17:33:36.040Z`. Tiles fill the map; crowded overlays remain a usability concern. Coverage is projected tile union, not pixel classification.

Website input 2: `Set Golden Gate Bridge earliest arrival to 10:00.` The same deterministic review message is returned. Preview produces eligible child/map `plan_4346a2580338c931` / `843e58fd719c56e0`, certificate `cert_e619e6c7ad4702c8`, diff `diff_9785301e6ed49f0c`, geometry `342204b0ec1054fb81216b79f1213a412860d2d11008f7a83efbe3d5f445c236`; acceptance/reload preserves that child. Complete HTTP/SSE, staging, preview, acceptance, map acknowledgements, evidence scope, independently rebuilt artifacts and readiness-bound full-size screenshots are retained under `results/e2e/standalone_owned_connected_20261007_v2/conversation/`. LLM input/output are explicitly null for these deterministic cases. Final isolation receipts confirm zero sends and zero blocked attempts. Runtime build guards are checked, but this custom run does not claim a pre-run full-source inventory.

Fresh billed website input: `Replace Bixby Creek Bridge with Pfeiffer Big Sur State Park; set Griffith Observatory duration to 90 minutes.` Exact provider assistant output: `Understood: replace Bixby Creek Bridge Viewpoint with Pfeiffer Big Sur State Park, and set Griffith Observatory to 90 minutes.` Both typed intents are retained. Full developer instructions, context, schema, request/response UTF-8 and structured output are in `results/e2e/standalone_owned_paid_20261007_v1/live_copilot_chat/provider_exact_llm_io.json`. Actual model `gpt-5.4-mini-2026-03-17`, low reasoning; 6,111 input / 331 output / 6,442 total tokens, zero cached tokens, one attempt, zero retries, no acceptance. Configured-rate usage estimate **$0.00607275**, not an invoice or newly verified price quote. Request SHA-256 `1bc1b642f6f12fbf0e34df22de781230ebe550d7c2e5a58941f7159075556510`; response SHA-256 `f85594f49176a23419b2d7b29e52cc50193bca02a955df90b111ed890ad1cc5f`. The child/map is `plan_a31708c1c9ba0ab9`. Independent verification binds 50 artifacts / 193 unchanged sources; manifest content hash `d0bac9441d0d1afbd4494c4ae70d54342ffc0ca9bfe6c40e1c8010d59afb04d5`.

Exact pre/final file bytes, complete API/Node inputs and outputs, fixture inputs, test summaries and retained failures are indexed in `results/repairs/standalone_owned_edits_20261007/repair_io.json` and `exact_io_inventory.json`. Secrets/authentication headers and saved session credential verifiers are explicitly omitted. Retained failures include the original HTTP 409 schedule-envelope mismatch, a correctly ineligible weather case whose first test expectation was wrong, an adversarial test that assumed `stop_id` rather than canonical `poi_id`, and the interrupted first browser run whose added assertion used a nonexistent metric. The alternate-port standard regression stopped at map-readiness preflight; the serial normal-port rerun passed. No failed run is relabeled. The passing standard regression independently verifies 119 artifacts / 194 unchanged sources; manifest content hash `c84d5c7dda0c6ae2d47fbb09ef63b403edb156466d5466ad85d8c5fd97767af6`.

**Next:** verify remaining owned typed-control/grouped stale/foreign-owner variants, then the complete frozen project regression and 57-case/21-suite recurring aggregate. Portability, native keyboard, overlay usability and independent phase audit remain open. No G5 promotion; W6-W8 and research statuses stay unchanged. Normal service restored on 8127, process 9652, build `377e7e10a590a0d0`; final health is checked below in the repair evidence. Both original user workspace hashes/revision zero are preserved. No key, ACL, Git publication or user trip was changed. Minimal-implementation guided reuse of the existing canonical path; API-key skill reused the approved key and bounded the paid test. Earlier checkpoints below remain historical and source-bound.

## Current result — mixed edits now reach the real website

**Bottom line:** the broken connection between evaluated mixed-owned edits and the website is repaired for the tested combinations. The browser now checks the change record against the accepted trip's actual rules. Staging keeps the parent visible; eligible Preview renders the backend's child; protected violations keep the parent visible; explicit acceptance persists the child through reload. This advances **R12**, but does not close R12 or the full E2E-1 gate.

**Proof on this production source:**

| Check | Result |
|---|---|
| New owned API / actual website-normalizer cases | 12 passed; six duration/window/role combinations, each soft and protected; real API, optimizer, evaluator, persistence and Node |
| New real website journeys | Three passed: owned replacement + duration; owned four-change request; protected four-change rejection |
| Existing checks | 207 domain and 100 product regressions passed; seven real-browser regression tests passed and independently verified twice |
| Fresh live GPT website journey | One GPT-5.4-mini attempt; both edits preserved through the real optimizer/evaluator and itinerary/map/Compare/Evidence; independently verified twice |

The three new browser journeys use registered disposable fixtures, actual chat controls and real local endpoints; internal fetch/API/optimizer/evaluator boundaries are not mocked. Their Copilot interpretation is deterministic, with **zero measured provider sends**. Afterward, each saved draft was recompiled using the actual configured route matrix: the complete child and diff match, and certificate semantics match except evaluation timestamp/content hash. The full-size owned preview was visually inspected: tiles fill the map, but crowded overlays remain a usability issue.

### What changed and what remains

The backend recomputes v5/v7 against the immutable parent and actual child, retaining evaluator-v2 and route-evidence checks. Bare cost evidence still cannot pass that boundary. The read-only `accepted-attribute-diff-context-v1` projection reuses domain normalization; the frontend checks parent identity, immutable constraints, applicable rules, before/after values, policy, costs and explicit non-permission semantics. Rehashed rule omissions, weaker rules, invented permission/feasibility, removed child constraints and format downgrades are rejected. No second frontend child plan or ownership policy was introduced.

**Next limitation:** standalone owned duration/window direct-preview envelopes are not closed by this mixed-edit increment. Remaining grouped stale/foreign-owner variants, the full frozen project regression and 57-case/21-suite recurring aggregate, portability, native mobile-keyboard evidence, overlay usability and phase audit remain open. No G5 promotion; W6-W8 and research statuses are unchanged.

### Exact inputs and outputs

Owned website input: `Replace Bixby Creek Bridge with Pfeiffer Big Sur State Park; set Griffith Observatory duration to 90 minutes; set Golden Gate Bridge earliest arrival to 10:00; set Golden Gate Bridge role to scenic_stop.` Full exact HTTP/SSE, staging, preview, acceptance, readback, screenshot-readiness and structured outputs are in `results/e2e/mixed_ownership_connected_20261007_v1/conversation/`. LLM input/output are null for these deterministic cases; `provider_scope.json` and finalized isolation receipts record the distinction.

Owned eligible truth: parent `plan_385f6f47a4359771` / `0ff390bd074092ac`; child/map `plan_8a1a4722f9b15151` / `1b202ff53fc0982f`; request `combined_e37e7b630b6a7af8`; certificate `cert_d128ef71c5fc9703`; geometry `eadbb41590e2e7b1bf050ff93f8f93fdd942fa460b7d424a591317c7cc42d11d`. Bixby is absent and Pfeiffer present. At 1440×900, the map is 651×630, projected loaded-tile coverage is 1.0, ready at `16:49:01.458Z`, capture starts `16:49:04.400Z`; readiness is stable before/after. This is projected tile union, not pixel classification or a new four-viewport owned-fixture matrix. The protected child is ineligible; its parent map/markers/hash survive reload and acceptance stays disabled.

Fresh billed website input: `Replace Bixby Creek Bridge with Pfeiffer Big Sur State Park; set Griffith Observatory duration to 90 minutes.` Exact provider `assistant_message`: `Understood: replace Bixby Creek Bridge Viewpoint with Pfeiffer Big Sur State Park, and set Griffith Observatory to 90 minutes.` The complete developer instructions, context, schema, typed intents, request and response bytes—not just that message—are in `results/e2e/mixed_ownership_connection_paid_20261007_v1/live_copilot_chat/provider_exact_llm_io.json`. Actual model `gpt-5.4-mini-2026-03-17`, low reasoning; 6,104 input / 468 output / 6,572 total tokens; one attempt, zero retries, no acceptance. Configured-rate usage estimate **$0.006684**, not an invoice or newly verified price quote. Request SHA-256 `5ae029bec98b2ae654135ee5c254963459ca5bc7afcc4a0209939963aeb51522`; response SHA-256 `7d717166c97f7b596c9b5074e8f7b4a238025f1dd706ce01c8601de3dbb1dd5a`. This paid journey uses the normal synthetic demo, **not** the added owned-rule fixtures. Its child/map is `plan_a31708c1c9ba0ab9` / `dde0f994990b834a`; request `combined_1b86dba18789f3ad`; certificate `cert_8569ff5902645ae2`; four full-size viewport captures pass readiness checks.

Exact pre/final file bytes, complete API/Node inputs and outputs, registered fixture inputs and retained failures are indexed in `results/repairs/mixed_ownership_product_20261007/repair_io.json` and `exact_io_inventory.json`. Retained failures include the original backend 409, the website's rejection of a valid backend result, a deliberate build-fingerprint interruption caused by an import-order edit during testing, and an adversarial fixture that incorrectly retained an old child hash. Corrected final tests use coherently rehashed counterexamples; no production check was weakened. The initial offline readback omitted the configured route delta; its differing result is retained as `readback_unconfigured_matrix_failure_io.json`, and corrected readbacks use the actual product configuration. Two evidence-script field-lookup mistakes were corrected before verification. Failed runs are not relabeled.

Normal website restored on 8127, process 10052, build `66d72845a11dcfcd`; health reports map/routing ready and OpenAI configured. Both saved user workspace files retain their original SHA-256 and revision zero. No key, ACL, Git publication or user trip was changed. The minimal-implementation skill guided reuse of canonical normalization; the API-key skill reused the approved key and bounded the fresh provider test to one attempt. Earlier checkpoints below remain historical and source-bound.

## Latest result — mixed protected edits retain one complete change record

**Progress, not phase completion:** the canonical diff now supports mixed relation-specific ownership through `plan-diff-v7`, with hash-bound ownership evidence v2. Duration, trip-specific time-window and role changes no longer fail merely because they occur together. Every change retains its exact before/after value, applicable rule IDs, attribute-specific strength, satisfaction and recomputable cost. The immutable parent's constraints cannot be dropped, weakened, retargeted, unconfirmed or rewritten. Existing v5/v6 shapes remain strict; no protection metadata is stripped to fit them.

**Important boundary:** this is the data-contract foundation. The current product verifier still rejects v5/v7 without a fully connected product path. Mixed-owned website Preview/Apply is **not enabled or verified** by this increment. Compiler, permission, accepted-parent frontend bindings and dedicated browser counterexamples are next. Independent evaluation still determines eligibility; a valid diff or high cost is neither permission nor feasibility.

| Check on this source | Actual result | Evidence |
|---|---|---|
| Evaluator, plans, repair and benchmark | 207 passed in 4.648 seconds | `results/repairs/mixed_ownership_contract_20261007/domain_regression_v2_junit.xml` |
| Real API/compiler/optimizer/evaluator/persistence and actual Node normalizer | 100 passed in 541.607 seconds; one existing Starlette deprecation warning | `product_regression_junit.xml` in the same repair directory |
| Real website regression | Seven tests passed in 228.446 seconds; independently verified twice; 119 artifacts / 194 unchanged source inputs / zero measured provider sends | `results/e2e/mixed_ownership_regression_20261007_v1/` |

The browser tests cover replacement through chat/staging/real Preview, one-click replacement, route-aware discovery, unsupported removal, ineligible duration, stale-response guards and accepted-child reload. These protect the **existing supported product**, not the new mixed-owned path. The four existing combined API cases also retain complete exact requests/responses, Node inputs/stdout/stderr, certificate round trips and forged-certificate rejection under `product_io/`; map readiness is a separate browser check, not the API fixture's health stub.

### Exact inputs and outputs

Canonical serialization example: stop `a` changes from exact **45 to 60 minutes**, required window **09:00–12:00 to 09:00–11:30**, and user-declared role **activity to meal**. All three parent rules are soft/automatically relaxable. Output is `plan-diff-v7`, with three change rows and weighted cost **7.5**. Full parent, child, policy, options and output: `diff_io/99c81132f6d0f49c.json`. This is a domain test, not an HTTP or website claim.

Independent-evaluation example: the immutable parent requires **30 minutes**, window **09:00–11:00**, role **activity**; the child requests **45 minutes**, **09:00–11:30**, **meal**. With locked rules, the exact certificate is **ineligible** with two `owned_attribute_constraint_unsatisfied` findings and one `owned_role_constraint_unsatisfied`; with automatically relaxable soft rules, it is **eligible with those rule warnings**, not silently preserved. Full evaluated child, reference parent, configuration, route matrix, planner run and certificate are `evaluator_io/e1e3292db55234d3.json` and `evaluator_io/957d38e1be49ca2b.json`. Eight mixed-family/protection combinations pass. Other checks reject contradictory/duplicate role rules, malformed role values/scope/confirmation, omitted or altered change evidence, incompatible version labels and attempted parent-rule rewrites; a lock on duration cannot become a role lock.

Exact website input was `Replace Bixby Creek Bridge with Pfeiffer Big Sur State Park`. Its complete HTTP/SSE/draft/preview outputs are in `registered_replacement/` under the browser run. **LLM input: null; LLM output: null; provider sends: 0** for this deterministic run. No new GPT/billed call occurred; older paid evidence remains historical and source-bound, not current-source proof after this change. API keys and mutation tokens are omitted from evidence; the already-approved key was not changed.

Browser truth: parent `plan_359968d255ff9288` / `750dae617fc4c1e3`; child and rendered map `plan_bc13d7ed8c7ec60b` / `ab610263c7f5a68f`; request `w3_preview_0235fb2f54ef1ba7`; certificate `cert_90e853fe0e27d774`; diff `diff_320fae8e32e4c503`; geometry `d0a3d9a141aef73782d1843541a9c57a3543a5ca10b9b92ba40eea409f7bbc39`. Staging leaves the parent rendered; Preview removes Bixby/adds Pfeiffer; acceptance/reload preserves the child. The actual full-size 1440×900 image was visually inspected. Its map is 571×614, ready at `2026-10-07T16:00:20.486Z`, capture starts `16:00:20.846Z`, and loaded-tile projected union coverage is 1.0. This is not pixel classification, a fresh four-viewport matrix, or native mobile-keyboard proof. Crowded overlays and unavailable fields remain visible usability/data limits. The replacement trace has no console errors or failed requests.

Browser manifest content hash `acedef1793acbe52a8f17f0042d44b12c9dd1f96708fbc1edb8e8da7618954ee`, SHA-256 `e930922117cf8b6b14abf7e9eaf973bf287b7019318a67f4145b8833adf3f967`; report SHA-256 `ac18439ffceea7829e862e3939a8b42750e4b16abd5507b4c8c4bd834627eac1`. Exact pre/final file bytes, hashes, domain/API exchanges and outcomes are indexed in `results/repairs/mixed_ownership_contract_20261007/repair_io.json` and `exact_io_inventory.json`. The optional observers call the real diff/evaluator; they do not alter results. Full terminal transcripts are not claimed for every unit/API test; complete selected exchange bodies and all JUnit outcomes are retained.

Retained failures: the initial reproducer had 10 failures / five passes / 19 deselections before production changes. Then 44 diff regressions, 111 evaluator/diff checks, 199 initial domain checks and the final expanded 207 passed. Ruff's single import-order finding was mechanically corrected; all five changed Python files and `git diff --check` pass. No rejected run is relabeled. The minimal-implementation skill reused existing ownership/diff validation rather than introducing another policy; the API-key skill retained the approved credential and deterministic provider isolation.

**Still open:** product mixed-owned compiler/permission/UI/map integration; remaining grouped stale/foreign-owner variants; complete frozen project regression and 57-case / 21-suite aggregate; recorded-fixture portability; native keyboard evidence, overlay usability and independent phase audit. R12/R20/CP-010/E2E-1 remain in progress; G5 blocked; W6-W8 planned; research statuses unchanged. The normal website is restored on 8127, process 29512, build `ee33c9433c542438`; configuration health is not GPT-call proof. Both saved user workspace pointers retain revision zero and their original SHA-256. Earlier checkpoints below are preserved as historical, source-bound records.

## Latest result — protected trip rules are independently enforced

**Fixed:** recording a high edit cost no longer substitutes for checking an owned rule. Evaluator `verify-001-plan-evaluator-v2` reuses the canonical ownership parser and checks existence, day, one-based within-day order, computed visit start/end, actual visit duration and the declared trip-specific time window. Locked/hard/booked and permission-gated violations block eligibility; automatically relaxable preferences produce warnings. Active child constraints cannot be dropped, weakened, retargeted, unconfirmed or rewritten against the supplied immutable parent. An existence lock does not lock unrelated attributes. Existing diff/permission gates were not loosened.

**Concrete input/output:** an active locked exact-duration rule requires **45 minutes**, while the canonical stop uses **30 minutes**. The preserved pre-fix evaluator returns `eligible`; the repaired evaluator returns `ineligible`, with `owned_attribute_constraint_unsatisfied`. Complete inputs/configuration/route matrix and both certificates are `results/repairs/owned_attribute_evaluator_20261007/{before_evaluator_io,evaluator_io}/40882c1e8de4a258.json`. There are 46 paired exact cases; 29 eligibility decisions change, while allowed edits and matching/inactive rules remain allowed. The pre-fix replay is a labeled in-process diagnostic, not a website journey.

### Evidence and limits

| Check | Actual result | Evidence |
|---|---|---|
| Evaluator, plans, repair and benchmark regression | 172 passed in 4.33 seconds | `results/repairs/owned_attribute_evaluator_20261007/domain_regression_v3_junit.xml` |
| Real product compiler/API/optimizer/evaluator/continuation regression | Final frozen-production-source rerun: 100 passed in 544.13 seconds; one Starlette deprecation warning | `product_regression_v2_junit.xml` in the same repair directory; earlier 99 passed / one stale expectation retained in `product_regression_junit.xml` |
| Actual website controls, preview, reload and Undo | Two browser journeys passed in 895.867 seconds; 10 variants each on original and accepted-child baselines; independently verified twice, 818 artifacts / 198 unchanged source inputs / zero measured provider sends | `results/e2e/owned_attribute_variants_20261007_v1/` |
| Fresh billed website chat | One passed in 46.910 seconds; independently verified twice; one GPT-5.4-mini attempt, no retry, no acceptance; 50 artifacts / 193 unchanged source inputs | `results/e2e/owned_attribute_paid_20261007_v1/` |

The 20 variants are five roles, three time-window forms, exact duration and same-day order, each on two baselines. Original-weather-ineligible results leave the parent rendered; accepted-baseline eligible results render the exact child, survive reload and Undo back to the accepted parent. The accepted baseline itself is created by real replacement/Preview/Accept controls. This does **not** establish acceptance of every typed variant, a new paid accepted-child continuation, or browser support for mixed relation-specific ownership. Those owned-rule counterexamples are evaluator/domain evidence; the browser journeys are regression evidence for the current demo and supported controls.

Two stale tests were corrected to current contracts, without weakening production rules: the benchmark cannot invent a weather-risk delta for changed selections with incomplete risk evidence; duration plus independent order is supported and must be evaluated, not declared unsupported. The latter still verifies an ineligible child is not accepted, the parent remains accepted and both drafts are retained. The pre-repair evaluator reproduces the stale benchmark assertion too. Failed runs remain retained: original red 25 failed / 8 passed; first green 69 passed / one error-message mismatch; recording-only run 64 failed / 19 passed due to a test serializer; labeled pre-fix replay 36 failed / 10 passed; first broader domain run 170 passed / one stale benchmark assertion; first product run 99 passed / one stale assertion; first corrected mixed test failed on its test-only response-envelope assumption. None is relabeled as a pass. Ruff passes all five changed Python files.

### Exact fresh LLM input and output

Exact website prompt:

```text
Replace Bixby Creek Bridge with Pfeiffer Big Sur State Park; set Griffith Observatory duration to 90 minutes.
```

Exact provider `assistant_message`:

```text
Understood: replace Bixby Creek Bridge Viewpoint with Pfeiffer Big Sur State Park, and set Griffith Observatory to 90 minutes.
```

The raw structured output contains the ordered `replace_nearby` and `set_stop_duration` intents, exact target/candidate IDs, exact 90/90/90-minute duration and complete highlights. All fields, developer instructions, current context, schema and UTF-8 wire bodies are preserved in `results/e2e/owned_attribute_paid_20261007_v1/live_copilot_chat/provider_exact_llm_io.json`; this paragraph is not a substitute for that complete output. Exact locally validated visible response:

```text
I prepared a typed itinerary suggestion for your review. Nothing was changed or executed automatically.
```

Actual model `gpt-5.4-mini-2026-03-17`, low reasoning; 6,106 input, zero cached, 385 output, 6,491 total tokens. Configured-rate cost estimate **$0.006312**, not an invoice or newly verified price quote. One logical website message, one actual provider transport attempt, zero retries. Request 29,790 bytes / SHA-256 `4f549a9d7fc065bd822a99a0257b196e18dfaead24efb20d5fa2eb92ec18cd07`; response 22,828 bytes / SHA-256 `63fe7f04dcf011b9b1f6a9028879db9e9c18d389f888acdf2a8bb77b4051151c`. Request/response headers and credentials are not persisted. The already-approved key was reused without modification.

Runtime truth: parent `plan_359968d255ff9288` / `750dae617fc4c1e3`; child and rendered map `plan_a31708c1c9ba0ab9` / `dde0f994990b834a`; request `combined_1b86dba18789f3ad`; certificate `cert_8569ff5902645ae2` / content hash `68e4e85e02e04807`, evaluator v2; diff `diff_fdf9468f97bac522`; geometry `f07301071e3038d8cb98f4ab7b33a03fa39b303f932ce2b861e38bbbbf5f3ff9`. Bixby is absent, Pfeiffer present and Griffith exactly 90 minutes. Itinerary, map, metrics, Compare and Evidence agree on that backend-owned child. Fresh acceptance revalidation checks full certificate content, including evaluator version; certificate ID alone is not treated as sufficient proof. Missing opening-hours/context remains disclosed; modeled eligibility is not live real-world availability.

| Viewport | Visible map | Loaded-tile projected area ratio | Map ready UTC | Capture started UTC |
|---|---|---|---|---|
| 1440×900 | 651×630 | 1.0 | 15:33:38.474 | 15:33:38.499 |
| 1280×800 | 507×537 | 1.0 | 15:33:36.445 | 15:33:36.470 |
| 1024×768 | 613×505 | 1.0 | 15:33:37.077 | 15:33:37.094 |
| 390×844 | 353×592 | 1.0 | 15:33:37.842 | 15:33:37.867 |

All are October 7, 2026, stable before/after map readiness. This coverage is the projected union of loaded tile footprints, not pixel classification or native mobile-keyboard proof. No unexpected console/network errors. The full-size 1440×900 screenshot was visually inspected: complete tiles and the evaluated Pfeiffer child are visible; crowded overlays and unavailable fields remain visible limitations, not invented data.

Browser manifest content hash `460004e8302512c13166b68df4ba917cff2f0024bf9419de69151edd8d76408c`, SHA-256 `cc65197d52e0f65f01aaaadeb0f47410af0edd17dcb7667294ef759869e35046`. Paid manifest content hash `3ad89d7709384dd8b03a2bde068c3d02783b70f5c549dcfa2ab3907c5c46fd6c`, SHA-256 `4d6662e6b0d6b99c7c1c3a207bd01ecb611f3f4fd1335cbac95ced80191dd8a4`. Exact file pre/final bytes, hashes, all synthetic evaluator/API exchanges, checks and retained failures are indexed in `results/repairs/owned_attribute_evaluator_20261007/repair_io.json` and `exact_io_inventory.json`. The observer records real evaluator results without changing them; the optional evidence recorder is not a substitute for the real-browser gate.

**Still open:** product-side mixed ownership/diff/permission/UI bindings and their browser counterexamples; remaining grouped stale/foreign-owner variants; complete frozen project regression and 57-case / 21-suite aggregate; recorded-fixture portability; native OS keyboard evidence and phase audit. R12/R20/CP-010/E2E-1 remain in progress, G5 blocked, W6-W8 planned and research statuses unchanged. Normal website restored on 8127; both saved user pointers remain revision zero with their exact pre-test SHA-256 values. The minimal-implementation skill led to reuse of the canonical parser, not a second ownership policy; the API-key skill retained the existing approved credential and bounded the fresh test to one attempt. Earlier checkpoints below remain historical, source-bound evidence.

## Bottom line — conflicting combined requests now stop safely

**Fixed:** Copilot no longer offers a conflicting multi-edit request as ready and then fails when it reaches the draft. Chat now reuses the canonical draft conflict validator. The server also rejects a batch that both replaces and edits the same stop, atomically: no first edit is retained. Valid combined requests still reach GPT, the real optimizer, independent evaluation and the exact child on the map.

**Verified this increment:** two real-browser safety journeys passed in 105.55 seconds on original and accepted-child baselines; standalone independent verification passed twice, binding 138 artifacts / 198 unchanged source inputs / zero measured provider sends. The complete typed-chat/API/Node/optimizer module passed 27 tests in 110.23 seconds; the scoped audit, aggregate, semantic and JavaScript regression passed 226 in 101.40 seconds. A separate fresh paid website journey passed in 43.02 seconds and was independently verified twice: exactly one GPT-5.4-mini request, both edits preserved, real final evaluation, the same child on itinerary/map/metrics/Compare/Evidence, reload and four full-size viewport captures. No retry and no acceptance in that paid test.

**Your normal website is restored:** port 8127, process 29656, build `ee33c9433c542438`; health reports core, map, routing and OpenAI configured. Both real saved workspace files retain their original SHA-256 and revision zero. Configuration health is not a provider-call proof; the separate paid website evidence supplies that proof. Refresh an older open page to load this build.

**Not complete:** R12, R20, CP-010 and E2E-1 remain in progress. Mixed relation-specific ownership, remaining conflicting/stale/foreign-owner grouped variants, complete frozen project regression and the fresh aggregate, recorded-fixture portability, native OS mobile-keyboard evidence and the phase-level audit remain open. The recurring aggregate now requires 57 browser cases across 21 suites; this increment did not run or pass that complete aggregate. G5 remains blocked, W6-W8 planned and research statuses unchanged. Older checkpoints below remain source-bound history.

### Exact conflict inputs and outputs

These were entered through the actual website chatbox. Each output below is exact visible local copy; no provider request was sent for these conflicts, no partial proposal/action was offered, no draft or accepted plan changed, and the canonical map/markers/hash remained unchanged after reload.

| Exact website input | Exact website output |
| --- | --- |
| `Set Stearns Wharf duration to 90 minutes; set Stearns Wharf duration to 120 minutes.` | `Different values were requested for the same stop attribute. Which value should I use? No part was staged or changed.` |
| `Set Stearns Wharf duration to 90 minutes; set Stearns Wharf duration to 90 minutes.` | `The same edit was requested more than once. Should I keep just one copy? No part was staged or changed.` |
| `Replace Bixby Creek Bridge with Pfeiffer Big Sur State Park; set Bixby Creek Bridge duration to 90 minutes.` | `A stop cannot be preserved or edited and replaced in the same request. Which change should I keep? No part was staged or changed.` |
| `Set Stearns Wharf role to meal; set Stearns Wharf role to activity.` | `Different values were requested for the same stop attribute. Which value should I use? No part was staged or changed.` |

The first two run on both baselines. The third runs on the original; the fourth on the accepted child, where replacement is not an enabled capability. Full exact SSE/HTTP bodies, visible text, trusted-click receipts, before/after-reload context, screenshots and lifecycle/provider-isolation records are under `results/e2e/combined_conflict_browser_20261007_v2/conversation/`. The original parent is `plan_359968d255ff9288` / `750dae617fc4c1e3`; the accepted-child safety baseline is `plan_bc13d7ed8c7ec60b` / `ab610263c7f5a68f`, created only in disposable test state through real Preview/Accept/reload.

The direct batch bypass is also checked against the real authenticated API. Its exact two-operation input and output are `results/repairs/combined_conflict_safety_20261007/api_io/atomic_replaced_target_batch.json`: HTTP 409, body `{"detail":"draft_conflicts_with_replacement"}`, revision zero, empty draft, null proposal. It rejects the whole batch, not a misleading successful subset. This additional exact-I/O recording rerun passed one test in 4.64 seconds.

### Exact fresh billed website input and output

Exact submitted input:

```text
Replace Bixby Creek Bridge with Pfeiffer Big Sur State Park; set Griffith Observatory duration to 90 minutes.
```

Exact provider structured text output:

```json
{"schema_version":"copilot-interpretation-v2","state":"proposal_ready","assistant_message":"Understood: replace Bixby Creek Bridge with Pfeiffer Big Sur State Park, and set Griffith Observatory to 90 minutes.","intents":[{"type":"replace_nearby","target_stop_id":"bixby_creek_bridge_viewpoint","destination_day":null,"candidate_id":"pfeiffer_big_sur_state_park","preference":null,"clarification_question":null,"parameters":null},{"type":"set_stop_duration","target_stop_id":"griffith_observatory","destination_day":null,"candidate_id":null,"preference":null,"clarification_question":null,"parameters":{"duration":{"mode":"exact","preferred_minutes":90,"minimum_minutes":90,"maximum_minutes":90}}}],"permission":null,"highlights":{"day_ids":[],"stop_ids":["bixby_creek_bridge_viewpoint","griffith_observatory"],"segment_ids":[],"candidate_ids":["pfeiffer_big_sur_state_park"]}}
```

Exact validated visible website output:

```text
I prepared a typed itinerary suggestion for your review. Nothing was changed or executed automatically.
```

Actual model `gpt-5.4-mini-2026-03-17`, low reasoning, one logical message, one actual transport attempt and zero retries. Usage: 6,112 input, zero cached, 471 output, 6,583 total tokens. The harness's configured-rate estimate is **$0.0067035**, not an invoice or a newly verified price quote. Complete developer instructions, workspace context, strict schema, provider request and response UTF-8 bodies are retained in `results/e2e/combined_conflict_paid_20261007_v1/live_copilot_chat/provider_exact_llm_io.json`; request 29,790 bytes / SHA-256 `c819a6ee5dd874d3f0c99a6c82e9becd2b0bce6328513b17f65d70c6f067392a`; response 23,522 bytes / SHA-256 `f42a5ecb6969c54636ad6923ab88a9431b5ab1197c3a62a6eea943cb989ec71c`. Request/response headers and credentials are not persisted.

The current-source paid child is `plan_a31708c1c9ba0ab9` / `dde0f994990b834a`; request `combined_1b86dba18789f3ad`, certificate `cert_8569ff5902645ae2`, diff `diff_fdf9468f97bac522`, rendered map plan the same child, geometry hash `f07301071e3038d8cb98f4ab7b33a03fa39b303f932ce2b861e38bbbbf5f3ff9`. Bixby is absent, Pfeiffer present, Griffith's canonical exact duration is 90 minutes. This paid test did not accept a repair and does not establish paid accepted-child continuation; the separate safety journey does establish unchanged accepted-child state under invalid requests. Missing opening-hours/context remains disclosed; modeled eligibility is not live real-world availability.

| Full-size paid viewport | Map dimensions | Projected loaded-tile visible-area coverage | Ready UTC | Capture start UTC |
| --- | --- | --- | --- | --- |
| 1440×900 | 651×630 | 100% | `2026-10-07T14:53:46.605Z` | `14:53:46.619Z` |
| 1280×800 | 507×537 | 100% | `2026-10-07T14:53:44.574Z` | `14:53:44.593Z` |
| 1024×768 | 613×505 | 100% | `2026-10-07T14:53:45.207Z` | `14:53:45.234Z` |
| 390×844 | 353×592 | 100% | `2026-10-07T14:53:45.990Z` | `14:53:46.015Z` |

The negative original-trip screenshot was visually inspected: complete tiles, unchanged Bixby, zero staged changes, disabled repair Preview and the correct visible clarification. All captures have stable map readiness before/after; no unexpected console/network errors. Tile coverage is projected loaded-tile union area, not pixel classification or native keyboard evidence.

### Retained failures and exact file I/O

- The red API run reproduced four failures before the production change. The first post-fix run had three passes and a wrong test-envelope assertion; the actual canonical error key is `detail`. Both runs are retained, not relabeled.
- Browser v1 had two behavior passes but independent audit rejected the missing pre-reload trusted-click log. V2 captures that log before reload and binds all imported helper/verifier sources; it independently passes. V1 remains rejected.
- Scoped regression v1 had 222 passes and two failures: an obsolete lifecycle count and a historical bundle correctly rejected after source changes. V2 had 225 passes and the new-mode fixture incorrectly demanding a child process. V3 has 226 passes. The stale-source rejection is preserved; a portable actual scenario-reader entry test separately checks list-valued artifact loading. Some other recorded audit fixtures still require dated local artifacts, so clean-clone portability is not claimed.
- Independent portable counterexamples reject fake ready states, partial drafts, wrong questions, changed messages/maps/context, foreign owners and untrusted clicks. These unit fixtures are not browser certification.

The verified negative manifest has content hash `35cbe15807188faf8c5233429e110b3c25470676a290bcfa6729bdf30f6ec3a7`, SHA-256 `9cf33fcbbba98cf7b2ec020be05ee3ebd3eff5d8541d035233aeadc7190b7711`, 138 artifacts / 198 sources. The paid manifest has content hash `d982b4d77700737cdd398d3ce86fd0f7c1839280455367cd810c9e1c930f6d42`, SHA-256 `f1ba5d7f2d817afc33526db1149366a07b9fd01b5aee33464b59b1b2d93d4173`, 50 artifacts / 193 sources. Both standalone verifications passed twice against unchanged runtime/harness inputs.

Full pre-edit and final file bytes, hashes, exact API outputs, commands, test results, verification receipts and retained failed-run references are indexed in [the additive exact-I/O ledger](../../results/repairs/combined_conflict_safety_20261007/repair_io.json). The minimal-implementation approach reused the existing canonical validator rather than introducing another conflict policy; the API safeguard retained the already-approved key and enforced one bounded synthetic paid attempt. No permission/diff ownership contract was loosened and no research gate advanced.

## Earlier checkpoint — accepted-trip continuation repair

**The next integration defect is fixed and verified in the real website.** A valid second repair was reaching the backend but being rejected by the browser. The website now renders that exact evaluated child, and both **Add all to draft → Preview** and **one-click combined Preview** work on the original trip and on an already accepted repair.

**Evidence:** 14 real-browser journeys passed in 1,445.04 seconds; independent artifact verification passed twice (965 artifacts, 197 source inputs, unchanged source, zero measured live provider calls). A separate fresh billed website test passed in 40.71 seconds with exactly one GPT request, real optimizer/final evaluation, the same child on all checked surfaces, reload and four viewports; it did not accept the child. The normal website is restored on 8127, process 3872, build `4947e37cfc8b8aaf`; both real saved workspace pointers and their file hashes are unchanged at revision zero.

**Still open:** special ownership/conflicting combined-edit boundaries, complete frozen project regression and aggregate, native OS mobile-keyboard evidence, and the independent phase audit. R12, R20, CP-010 and E2E-1 remain in progress. G5 stays blocked; W6-W8 and research statuses are unchanged.

## Latest R12 continuation checkpoint — exact I/O and evidence limits

The backend's `plan-diff-v6` deliberately serializes only non-empty typed change families. The browser incorrectly required all three families even for a valid duration-plus-time-window child. The application fix aligns that exact-key check with the canonical serializer; plan/hash/request/evaluation/diff/geography and semantic checks remain enforced. Real API tests now cover all three families and every two-family subset, with 64 embedded production-JavaScript valid/forged assertions, acceptance and restart.

| Real browser entry path | Original accepted parent | Previously accepted repair |
| --- | --- | --- |
| Add all to draft, then Preview combined repair | Passed; accept and reload | Passed; A → B → C, accept and reload |
| Preview combined repair directly | Passed; atomic staging, accept and reload | Passed; atomic staging, A → B → C, accept and reload |

Exact continuation website input:

```text
Set Pfeiffer Big Sur State Park duration to 90 minutes; set Stearns Wharf earliest arrival to 10:00.
```

Exact deterministic website output:

```text
I prepared all 2 requested changes as one combined suggestion. Nothing was staged or changed.
```

The resulting canonical stop duration is 90 minutes and the earliest-arrival constraint is 10:00; no exact scheduled arrival is invented. A is `plan_359968d255ff9288` / `750dae617fc4c1e3`; accepted B is `plan_bc13d7ed8c7ec60b` / `ab610263c7f5a68f`; child/rendered/accepted-after-reload C is `plan_a01375842a4b5f6b` / `ed08c8647299a1af`. Request `combined_c70b5af047ecfaa7`, certificate `cert_d30db27214d4b073`, diff `diff_a7e51b81c2386536`, map geometry hash `90ee1317be58687e63fbb4d71de8a6edd200a38f34d469811cdc77f8bace478e`. Timing-only edits preserve road coordinates and road evidence; the plan-bound geometry hash changes for C. Bixby remains absent and Pfeiffer present.

The full-size 1440×900 continuation screenshot records a 651×630 map and 100% **projected loaded-tile visible-area coverage**. Ready: `2026-10-07T13:39:05.250Z`; capture start: `13:39:05.266Z`; completion: `13:39:05.350Z`. Readiness is stable before/after capture. The preview and Compare screenshots were visually inspected; those cases have no console or network errors. This is not a pixel-classification or native keyboard claim.

| Check | Actual result / retained evidence |
| --- | --- |
| Exact failed response replay | Old browser source rejected it; corrected source accepts the identical response. `results/repairs/combined_continuation_20261007/exact_normalizer_replay_io.json` |
| API / optimizer / evaluator / JavaScript subset matrix | 4 passed in 125.05 seconds, 64 embedded JavaScript assertions. Complete exact API and Node I/O: `results/repairs/combined_continuation_20261007/api_and_node/` |
| Focused frontend/API/identity/outcome/acceptance regression | 130 passed in 717.16 seconds; not a complete-project regression. `results/e2e/combined_continuation_regression_20261007_v2/` |
| Independent-audit regression | 283 passed in 98.68 seconds; includes the real 14-case scenario-reader entry point. `results/e2e/combined_continuation_unit_20261007_v4/` |
| First full browser run | 13 passed / one failed in accepted-recovery setup. Preserved as failed: `combined_continuation_browser_20261007_v1` |
| Second full browser run | 14 browser tests passed, but independent audit rejected an object-only reader used for list-valued inventories. Preserved as unverified: `combined_continuation_browser_20261007_v2`; regression reproduced that exact failure before repair |
| Fresh frozen-source full run | 14 passed; independent verification passed twice; 965 artifacts / 197 source inputs / zero live sends. `results/e2e/combined_continuation_browser_20261007_v3/` |
| Fresh billed combined website journey | One passed, independently verified twice; 50 artifacts / 193 source inputs / exactly one actual provider attempt. `results/e2e/combined_continuation_paid_20261007_v1/` |

The audit repair uses the existing general JSON loader for list-valued chat and HTTP inventories; it does not weaken identity, content hash, semantics, lifecycle, acceptance or ready-state checks. The v2 rejected manifest remains rejected. Recorded-evidence unit tests require their retained local run artifacts and are not clean-clone portability or current-source browser certification. The current-source recurring browser journey and standalone manifest verification are separate. All seven changed Python files passed Ruff.

### Fresh billed website input and output

Exact user input:

```text
Replace Bixby Creek Bridge with Pfeiffer Big Sur State Park; set Griffith Observatory duration to 90 minutes.
```

Exact provider structured text output (the full Responses API envelope is separately retained):

```json
{"schema_version":"copilot-interpretation-v2","state":"proposal_ready","assistant_message":"Understood: replace Bixby Creek Bridge Viewpoint with Pfeiffer Big Sur State Park, and set Griffith Observatory to 90 minutes.","intents":[{"type":"replace_nearby","target_stop_id":"bixby_creek_bridge_viewpoint","destination_day":null,"candidate_id":"pfeiffer_big_sur_state_park","preference":null,"clarification_question":null,"parameters":null},{"type":"set_stop_duration","target_stop_id":"griffith_observatory","destination_day":null,"candidate_id":null,"preference":null,"clarification_question":null,"parameters":{"duration":{"mode":"exact","preferred_minutes":90,"minimum_minutes":90,"maximum_minutes":90}}}],"permission":null,"highlights":{"day_ids":[],"stop_ids":["bixby_creek_bridge_viewpoint","griffith_observatory"],"segment_ids":[],"candidate_ids":["pfeiffer_big_sur_state_park"]}}
```

Exact locally validated, visible website output:

```text
I prepared a typed itinerary suggestion for your review. Nothing was changed or executed automatically.
```

This increment made **one** actual provider attempt, one logical website message, no hidden retry, and no acceptance. Model `gpt-5.4-mini`, low reasoning; raw response model `gpt-5.4-mini-2026-03-17`; 6,108 input tokens, zero cached, 474 output, 6,582 total. The harness's `billed_cost_usd=0.006714` is its configured-rate calculation, not an invoice/current-price verification. The earlier paid-combined increment's two attempts remain historical and separately reported below.

This fresh paid test proves the original-parent combined GPT path, **not** a new paid accepted-child continuation. The separate provider-isolated 14-case gate proves acceptance/reload and continued-edit controls. Its two cancellation cases use one fake external SDK attempt each, not live OpenAI; their exact input/output are `conversation/fake_sdk_cancel/fake_sdk_io.json` and `conversation/fake_sdk_cancel_accepted_child/fake_sdk_io.json`.

The paid child and rendered map are `plan_a31708c1c9ba0ab9` / `dde0f994990b834a`, request `combined_1b86dba18789f3ad`, certificate `cert_8569ff5902645ae2`, diff `diff_fdf9468f97bac522`, geometry `f07301071e3038d8cb98f4ab7b33a03fa39b303f932ce2b861e38bbbbf5f3ff9`. Bixby is removed, Pfeiffer added, Griffith's canonical visit value and exact constraint both encode 90 minutes. No unexpected console/network errors occurred. Missing opening-hours/context remains disclosed; modeled eligibility is not real-world availability.

| Paid full-size viewport | Map dimensions | Projected loaded-tile coverage | Ready UTC | Capture start UTC |
| --- | --- | --- | --- | --- |
| 1440×900 | 651×630 | 100% | `2026-10-07T14:03:20.534Z` | `14:03:20.918Z` |
| 1280×800 | 507×537 | 100% | `2026-10-07T14:03:18.486Z` | `14:03:18.499Z` |
| 1024×768 | 613×505 | 100% | `2026-10-07T14:03:19.119Z` | `14:03:19.145Z` |
| 390×844 | 353×592 | 100% | `2026-10-07T14:03:19.901Z` | `14:03:19.926Z` |

Complete live LLM wire input/output, including instructions, context, strict schema and both UTF-8 bodies: `results/e2e/combined_continuation_paid_20261007_v1/live_copilot_chat/provider_exact_llm_io.json`. Complete actual website/API exchanges, validated turn, runtime child, diff, evaluation, map acknowledgements, reload and readiness-timestamped screenshots are beside it; `exact_io_report.md` and `file_io_manifest.json` bind them all. Paid manifest content hash `3f1110d00f8469aade1655aa350c109d1bc70352d9f6a030b5f2afdc07d51fff`; SHA-256 `465457356cb35ee00e33f5210541f7696db35e0c95c16874c977dab15c68c4e2`. Deterministic manifest content hash `8977c8d8fd4ee17b17836ae4ab19a5bbcbfe39d94af76418a85ed3a67d536b9c`; SHA-256 `48e8ac29dc846c273b87ec4919275072e00cfbcaa8184c21670461438a3c4b38`.

Full before/after bytes for all 15 changed files, hashes, exact API/JavaScript I/O, commands, verification receipts and retained failures are indexed in [the additive exact-I/O ledger](../../results/repairs/combined_continuation_20261007/repair_io.json). Nothing below is relabeled as this increment's source-bound proof.

## Earlier paid/workspace checkpoint — source-bound historical evidence

**The combined-chat website path now also passes a fresh paid, independently verified browser journey.** The actual website sent the exact two-edit prompt to GPT, staged both returned edits together, ran the real optimizer and independent evaluation, and displayed that same eligible child in itinerary/map/metrics/Compare/Evidence, after reload and at all four required viewport sizes. The fresh paid run did not accept the child. The earlier 11-case deterministic conversation gate separately verifies acceptance and reload. Neither scoped result closes the full R12/E2E-1 gate.

**The desktop workspace repair now passes its real-browser gate.** Copilot reserves its own resizable column, the map refits after resizing, and the selected stop is verified inside the canvas and unobscured by panels. All four required viewport journeys and independent artifact verification passed, with zero provider attempts. A fresh billed website journey also passed against the repaired source: one actual GPT request, real runtime repair and evaluation, the exact child on the map and all visible surfaces, reload and four viewport sizes; no acceptance.

**The complete repair program is still in progress.** The original non-frozen regression finished with 3,000 passes, 28 failures and one error. Thirteen failing traces explicitly identify a changed source build; the others are not automatically attributed to drift. Fresh checks pass 17 concurrency cases and all 14 targeted outcome/map/security/persistence/status cases after repairing stale test inputs. The final scoped Python regression passes 445 checks and the Node frontend run passes 18. These do not replace a complete final-source regression and aggregate. Native mobile-keyboard evidence remains untested. G5 and later product waves remain blocked; research gates are unchanged.

## Earlier R12 paid combined-chat checkpoint — source-bound history, with exact I/O

Exact website input:

```text
Replace Bixby Creek Bridge with Pfeiffer Big Sur State Park; set Griffith Observatory duration to 90 minutes.
```

Exact locally validated, visible assistant output:

```text
I prepared a typed itinerary suggestion for your review. Nothing was changed or executed automatically.
```

The provider returned the two ordered typed intents: `replace_nearby(bixby_creek_bridge_viewpoint, pfeiffer_big_sur_state_park)` and `set_stop_duration(griffith_observatory, exact 90 minutes)`. All parameter fields and the complete raw provider response are preserved, not replaced by this shorthand. User-facing prose and complete action highlights are locally canonicalized; the raw provider body remains separately available. **Add all to draft** stages one complete revision-guarded batch and leaves the parent map unchanged. **Preview combined repair** runs the real optimizer plus final independent evaluation and renders only the returned eligible child. Trusted button-click receipts, the complete typed interpretation, batch request/response and optimizer/final-evaluator receipts are checked independently.

The existing bounded paid harness now offers two closed profiles, `replacement` and `combined`; arbitrary prompts remain disallowed. The selected profile is bound before transport, with one logical website message, one provider attempt, no accepted-state mutations and no saved user-trip changes. Both profiles reuse the same browser journey, map-ready capture and independent verifier. The combined optimizer runs against an attribute-adjusted temporary baseline; the final child is independently evaluated against the original accepted parent. The verifier checks both receipts and their content relationship instead of incorrectly requiring those two baselines to have the same ID. No application/optimizer implementation changed in this increment.

| Current check | Actual result | Evidence |
| --- | --- | --- |
| New closed-profile and complete-clause safety baseline | Two failures before implementation | `results/e2e/combined_paid_red_20261007_v1/` |
| Offline profile/identity/transport checks | 27 passed before the first paid run | `results/e2e/combined_paid_offline_20261007_v2/` |
| First paid two-edit browser journey | One browser test passed, but the independent gate failed on a leftover one-operation assertion. Preserved as unverified; one actual provider call, 6,110 input / 545 output tokens | `results/e2e/combined_paid_website_20261007_v1/` |
| Audit regression baseline | Initial recorded-bundle fixture was incomplete; corrected fixture reproduced `e2e1_live_compiled_operation_mismatch`. Both red runs preserved | `results/e2e/combined_paid_bundle_red_20261007_v1/` and `v2/` |
| Final focused harness regression | 138 passed in 10.01 seconds; Ruff passed all four changed Python files | `results/e2e/combined_paid_harness_20261007_v2/pytest_junit.xml` |
| Fresh final-source paid website journey | One passed in 37.85 seconds; independently verified twice; 50 artifacts / 193 source inputs; one measured actual provider attempt | `results/e2e/combined_paid_website_20261007_v2/` |

There were **two actual provider attempts across this increment**, one per separate run, not one overall. Both used `gpt-5.4-mini`, low reasoning; both raw responses identify `gpt-5.4-mini-2026-03-17`. The verified v2 used 6,112 input tokens (5,888 cached), 367 output tokens, 6,479 total. Across both runs: 12,222 input, 912 output, 13,134 total tokens. The existing harness's field named `billed_cost_usd` is a calculation from its configured rates, not an invoice: v1 `0.007035`, v2 `0.0022611`, combined `0.0092961` USD. No current price/invoice verification is claimed.

Verified identities: parent `plan_359968d255ff9288` / `750dae617fc4c1e3`; request `combined_1b86dba18789f3ad`; child and rendered map `plan_a31708c1c9ba0ab9`; child hash `dde0f994990b834a`; certificate `cert_8569ff5902645ae2`; diff `diff_fdf9468f97bac522`; geometry `f07301071e3038d8cb98f4ab7b33a03fa39b303f932ce2b861e38bbbbf5f3ff9`. Bixby is removed, Pfeiffer is added, and Griffith's canonical visit duration and exact duration constraint are both 90 minutes. Console errors, failed requests and HTTP errors are empty. No Accept or Keep action occurred. Missing opening-hour/context evidence remains disclosed; eligibility is modeled, not a guarantee of real-world availability.

| Full-size viewport | Actual map dimensions | Visible projected tile-area coverage | Map ready UTC | Screenshot started UTC |
| --- | --- | --- | --- | --- |
| 1440×900 | 651×630 | 100% | `2026-10-07T10:52:03.435Z` | `2026-10-07T10:52:03.449Z` |
| 1280×800 | 507×537 | 100% | `2026-10-07T10:52:01.338Z` | `2026-10-07T10:52:01.364Z` |
| 1024×768 | 613×505 | 100% | `2026-10-07T10:52:01.970Z` | `2026-10-07T10:52:01.996Z` |
| 390×844 | 353×592 | 100% | `2026-10-07T10:52:02.737Z` | `2026-10-07T10:52:02.960Z` |

All four screenshots have matching canvas/container sizes, stable readiness before/after capture and the same child/hash/geometry. Browser identity is recorded as installed Chrome `154.0.8037.98`. Full-size proposal and rendered-preview screenshots were also visually inspected. These viewport checks do not prove native mobile-keyboard behavior.

Complete exact LLM input/output: `results/e2e/combined_paid_website_20261007_v2/live_copilot_chat/provider_exact_llm_io.json` (both UTF-8 wire bodies, instructions, context, schema and usage; no headers or keys). Complete exact website/API I/O: `product_copilot_http_io.json`, `product_draft_http_io.json`, `product_preview_http_io.json`, `product_reload_http_io.json`, with the separate typed/local output, child, diff, evaluation, map acknowledgements, trusted clicks and screenshots in the same directory. `exact_io_report.md` includes the complete provider bodies; `file_io_manifest.json` binds every evidence file and source input. Manifest content hash `eb57ef90732ca030721ddc4b4fb0588f1005bffd9b7c5c7772a076564ed51204`; manifest SHA-256 `73242dde08f965eb0beecac37a798a52eec22fd82c66d8932967e6c8fc398abc`; report SHA-256 `da8e5c244b2c34e4f1948517a32409375f5b15bd812f0533b0b93887856aff04`.

Exact input/output file snapshots and all retained failures/commands are indexed in `results/repairs/combined_chat_paid_20261007/repair_io.json`. The first normal-server restore lacked protected fixture access and returned `ready=false`; it was stopped by its owned handle and restored with approved read access, without ACL changes. The normal website is now ready on 8127, PID 5768, build `bdb0302f66dedf47`, with the California and legacy user workspace pointers still at revision zero and their original accepted IDs/hashes. The disposable paid preview did not alter either saved trip.

**Next:** accepted-child and one-click grouped-action inventory coverage; then remaining ownership boundaries and a complete frozen regression/aggregate. R12, R20, CP-010 and E2E-1 remain in progress; G5, W6-W8 and the research gates are unchanged. The prior deterministic, single-replacement paid and failed runs below remain historical and source-bound; their then-open paid-combined requirement is superseded only by this fresh checkpoint.

## R12 earlier deterministic combined-chat increment — implementation and real website sample

Exact website input:

```text
Replace Bixby Creek Bridge with Pfeiffer Big Sur State Park; set Griffith Observatory duration to 90 minutes.
```

Exact deterministic assistant output:

```text
I prepared all 2 requested changes as one combined suggestion. Nothing was staged or changed.
```

Executable combinations now come from the same capability contract in the prompt, interpretation guard and controls. The existing draft endpoint accepts one revision-guarded batch: every operation is validated before any is persisted. **Add all to draft** leaves the parent map unchanged. **Preview combined repair** uses the real optimizer and independent evaluator, then renders only the returned child. No unofficial frontend child is constructed. Conflicting day instructions are blocked before staging or provider dispatch. Replies that drop, retarget or reorder clauses are rejected; highlights derive from the complete validated edit set. Unsupported, negated or unresolved compound language remains fail-closed, not unrestricted ChatGPT-style language support.

| Interaction | Handler / state | Real endpoint and backend | Artifact / expected map effect | Recurring test |
| --- | --- | --- | --- | --- |
| Add all to draft | `addIntentToDraft -> addCopilotDraft -> addDraft`; canonical complete batch | `POST /draft/operations -> WorkspaceStore.add_operation`; whole-batch validation and one revision/CAS | All staged operations or none; accepted plan/map unchanged | `test_combined_chat_stages_all_edits_and_renders_exact_child` |
| Preview combined repair | `previewIntentFromCopilot -> previewDraft`; exact ordered draft match | `POST /preview -> ProductService.preview -> compiler/optimizer -> independent evaluator -> geography` | One child/diff/certificate/geography; itinerary, metrics, Compare, Evidence and map agree | Same real-browser test, including acceptance and reload |

Sample identities: request `combined_1b86dba18789f3ad`; child `plan_a31708c1c9ba0ab9`; hash `dde0f994990b834a`; certificate `cert_8569ff5902645ae2`; diff `diff_fdf9468f97bac522`; rendered geometry `f07301071e3038d8cb98f4ab7b33a03fa39b303f932ce2b861e38bbbbf5f3ff9`. The accepted and rendered IDs remain that child after reload. Griffith's canonical duration constraint and visit duration both encode 90 minutes. Missing opening-hour evidence remains disclosed; modeled eligibility is not real-world availability.

| Check | Actual result | Evidence |
| --- | --- | --- |
| Compound-chat baseline | Two failed / one passed | `results/e2e/combined_chat_contract_20261007_red_v1/` |
| Contradictory-day baseline | Two failed / four passed; a day instruction was silently ignored | `results/e2e/combined_chat_day_guard_20261007_red_v1/` |
| First conversation browser run | Ten passed / one failed: test used the wrong canonical stop field | `results/e2e/combined_chat_browser_20261007_v1/` |
| Second conversation browser run | All 11 passed in 1119.53 seconds; independent gate failed because reload erased trusted-click receipts. Preserved as unverified | `results/e2e/combined_chat_browser_20261007_v2/` |
| Fresh durable-receipt browser run | All 11 passed in 976.07 seconds; independently verified twice, 704 artifacts / 197 source inputs, zero measured live provider calls, source unchanged. Manifest content hash `369322b7061d363f38f6b021a87fdb7ad08e8175c2e684f9f1a51b2b9d672b8f` | `results/e2e/combined_chat_browser_20261007_v3/` |
| Final frozen scoped regression | 445 passed in 246.81 seconds; one third-party deprecation warning | `results/e2e/combined_chat_final_regression_20261007_v4/pytest_junit.xml` |
| Frontend contracts | 18 passed; zero failed/skipped | Command/output in the additive ledger |
| Fresh concurrency recheck | 17 passed in 180.59 seconds | `results/e2e/preview_concurrency_recheck_20261007_v1/pytest_junit.xml` |
| Initial targeted recheck | Three failed / 11 passed: empty-workspace fixture and stale status assertions | `results/e2e/regression_failure_recheck_20261007_v1/` |
| Recheck after fixture repairs | 14 passed in 101.58 seconds; missing workspace still rejected | `results/e2e/regression_failure_recheck_20261007_v3/pytest_junit.xml` |

Exact browser/API input and output are under `results/e2e/combined_chat_browser_20261007_v3/conversation/combined_chat/`: `exact_chat_io.json`, `staging_io.json`, `preview_response.json`, `acceptance_response.json`, `combined_controls_before_reload.json`, `http_index.json`, `trace.json`, `evidence_panel.json` and full-size readiness-timestamped screenshots. The 1440×900 preview records a 651×630 map, 100% visible projected tile-area coverage, ready at `2026-10-07T10:15:21.006Z`, capture starting at `2026-10-07T10:15:21.025Z`. Console and network errors were empty. Reload's new-page click log is separate from the durable pre-reload receipt.

**Real LLM input/output this increment: `null` / `null`; live provider calls: zero.** The browser uses the real chat endpoint and unmocked internal repair path with provider transport isolated. Two cancellation journeys explicitly use fake external HTTP through the real SDK, not live OpenAI. Complete injected SDK requests, strict schemas, raw injected outputs and validated local outputs for complete/drop/retarget/reorder cases are retained in `results/e2e/combined_chat_final_regression_20261007_v4/exact_io/`; they are not billed-call evidence. Existing billed synthetic-context authorization remains available for the next bounded website test.

Full before-file bytes, after-file bytes, hashes, commands, retained failures and I/O references are indexed by `results/repairs/combined_chat_20261007/repair_io.json`. User workspace pointers remain at revision zero; acceptance uses disposable state only. The normal website is restored on port 8127, PID 29164, build `bdb0302f66dedf47`; health reports map/routing ready and OpenAI configured, not a new provider-call proof. Remaining: paid combined-prompt website proof; accepted-child and one-click grouped-control inventory coverage; mixed relation-specific attribute ownership; complete frozen regression/aggregate; native mobile-keyboard evidence. No full R12, E2E-1, G5 or research promotion is claimed.

## R12 website increment — actual controls and independent browser gate verified

The production website now accepts the existing backend combined-execution receipt instead of constructing another child. The same backend capability list drives combination availability and draft-impact validation. The browser checks the complete request, final evaluation, diff identity/cost, requested attributes, stop order, markers and geometry bindings before rendering the backend child. Staging remains separate from Preview and acceptance.

The real test exposed and corrected three connected defects:

- Initial prewritten timeline data already contained the recommended replacement. Validation now derives accepted stop order from the accepted plan's geography, not that recommendation.
- A subsequent time-window edit incorrectly required the accepted parent's schedule to contain no time windows. Existing parent constraints are now allowed and still checked through the full route/schedule accounting.
- A later role edit failed its integrity check when a validated road leg had an exact whole-second duration. Python serialized that declared float as `13468.0`; JavaScript serialized it as `13468`. The declared `duration_s` float is now serialized consistently; HMAC verification is retained.

Validation and retained failures:

| Check | Actual result | Evidence |
| --- | --- | --- |
| API-to-production-JavaScript baseline | Failed: valid combined API child was blocked/rejected by the website | `results/e2e/combined_frontend_20261007_red_v1/` |
| Real combined API, 16 JavaScript valid/forged cases, acceptance and restart | Passed; valid case is required before negative cases count | `results/e2e/combined_frontend_20261007_green_v2/` |
| First five-journey browser run | One passed / four failed; exposed continuation defects and two incorrect test expectations | `results/e2e/combined_browser_20261007_v1/` |
| Second frozen-source browser run | Five passed in 390.22 seconds; independent gate failed because its object-only loader rejected the five-operation input list. Preserved as unverified | `results/e2e/combined_browser_20261007_v2/` |
| Focused existing contracts and registered manifests | 48 passed in 505.02 seconds; one third-party deprecation warning | `results/e2e/combined_frontend_regression_20261007_v2/pytest_junit.xml` |
| Complete canonical plans, compiler audit and real API mapping | 75 passed in 91.97 seconds; one third-party deprecation warning | `results/e2e/combined_final_contracts_20261007_v2/pytest_junit.xml` |
| Fresh browser run with corrected verifier | Five passed in 350.62 seconds; independent verification passed, 253 artifacts / 192 source inputs / zero provider calls | `results/e2e/combined_browser_20261007_v3/` |

The final verified run's exact five-operation input is `direct_edits/combined/operation_input.json`; exact API exchanges and output bytes are indexed in `http_index.json`. The preview, acceptance, evidence DOM, map acknowledgement and readiness-timestamped full-size screenshots are retained beside them. Its combined child is `plan_814b66fb63960420`, content hash `90de0702244c63a6`, evaluation `cert_b5960f03e8f03810`, diff `diff_821b40e17e117fc6`, geometry hash `13157c7a2cea271d7d339b37c6dcdadd938687cb277c3f8aae18bfe025450752`. The screenshot records a 651×630 map inside a 1440×900 viewport, 100% visible projected tile-area coverage, map-ready timestamp `2026-10-07T09:00:24.648Z`, and capture starting at `2026-10-07T09:00:27.163Z`. These identifiers are browser observations, not backend-only inference. Final manifest content hash: `b1eefa9baa9cec5b77c1caf3c236e49a244fad9aed53a7317b4e4183f1549542`.

LLM input and output for this increment are both `null`: no chat was submitted. The deterministic browser server disabled the API key, blocked outbound provider transport, and measured zero provider sends. This is not fresh GPT or combined-chat evidence. Main user workspace pointers stayed at revisions zero; all acceptance tests used disposable state. The normal user server was restored on port 8127 as process 29376, build `c137f61832c5d493`; health reports map/routing ready and OpenAI configured. Configured is not a new billed-call test. File input/output hashes, exact test commands and preserved failures are in `results/repairs/combined_frontend_20261007/repair_io.json`; full pre-increment source bytes were not snapshotted and that evidence limit is explicit.

## Earlier R12 backend checkpoint — website boundary superseded by the increment above

The engine now composes supported California-demo replacement/addition/preservation and exact-duration, time-window, visit-role and same-day-order edits in memory. It validates all clauses before selection search, retains the original immutable parent, computes one final diff, independently evaluates the complete child, and exposes an explicit combined-execution receipt. Intermediate eligibility is not a final decision. A new `plan-diff-v6` is necessary because the older attribute diff formats could serialize only one attribute family; no requested field is dropped to fit an older format. Mixed relation-specific attribute-ownership evidence remains explicitly unsupported rather than being stripped.

An actual API test staged five edits, ran the real repair/evaluator/geography pipeline, accepted the exact evaluated child, restarted the application against disposable persisted state, and restored the same child and marker set. It **did not run in a browser, send a chat message, or acknowledge a rendered map**. The map-health probe alone was pinned for this backend test; internal repair, evaluator, repository, serializer and acceptance boundaries were not mocked. Existing pre-staging website blockers and the compound-message guard remain in place until the new browser/LLM contract passes.

The first API run failed at acceptance despite an eligible preview. Exact diagnostics confirmed a representation mismatch: the freshly recomputed warning held one tuple-valued evidence reference, whereas JSON persistence held an array. Raw semantic dictionaries compared unequal; JSON-normalized dictionaries compared equal. Acceptance now compares the same JSON representation while retaining every evaluation field and allowing only the existing timestamp replay. A validly rehashed but semantically forged warning was rejected; the accepted parent stayed unchanged before the genuine acceptance succeeded.

| Check | Actual outcome | Evidence |
| --- | --- | --- |
| Original combined-edit baseline | Three failed: the compiler rejected all combinations | `results/e2e/combined_edits_20261007_red_v1/pytest_junit.xml` |
| First implementation check | One failed / two passed: legacy duration-mode guard accidentally included the replacement row; corrected without weakening the tests | `results/e2e/combined_edits_20261007_v1/pytest_junit.xml` |
| Final compiler and complete canonical-plan directory | 65 passed in 2.26 seconds: complete diffs, final-only eligibility, invalid-clause preflight, removed-target conflicts, missing-route blocking, stable identity, ownership and serialization checks | `results/e2e/combined_edits_canonical_20261007_v2/pytest_junit.xml` |
| First API vertical slice | Failed with `acceptance_proposal_invalid`; preserved | `results/e2e/atomic_combined_api_20261007_v1/pytest_junit.xml` |
| Final API vertical slice, adversarial check, acceptance and restart | One passed in 34.88 seconds; one third-party deprecation warning | `results/e2e/atomic_combined_api_20261007_v4/pytest_junit.xml` |
| Scoped Python lint | All seven changed production/test Python files passed | Exact command in the additive repair ledger below |

Exact logical inputs, in staging order:

```json
[
  {"type":"replace_nearby","target":"bixby_creek_bridge_viewpoint","parameters":{"candidate_id":"pfeiffer_big_sur_state_park"}},
  {"type":"set_stop_duration","target":"griffith_observatory","parameters":{"duration":{"mode":"exact","preferred_minutes":90,"minimum_minutes":90,"maximum_minutes":90}}},
  {"type":"set_stop_time_window","target":"golden_gate_bridge","parameters":{"earliest_arrival":"10:00","latest_departure":null}},
  {"type":"set_stop_role","target":"golden_gate_bridge","parameters":{"role":"scenic_stop"}},
  {"type":"set_stop_order","target":"sequoia_national_park","parameters":{"day":4,"sequence_index":1}}
]
```

Exact final identities from the successful disposable run:

```json
{
  "request_id":"combined_26b8c75415500444",
  "parent_plan_id":"plan_359968d255ff9288",
  "child_plan_id":"plan_814b66fb63960420",
  "child_plan_content_hash":"90de0702244c63a6",
  "evaluation_id":"cert_b5960f03e8f03810",
  "diff_id":"diff_821b40e17e117fc6",
  "comparison_eligibility":"eligible",
  "route_geometry_plan_id":"plan_814b66fb63960420",
  "rendered_map_plan_id":null,
  "map_render_acknowledgement":"not_tested_in_this_backend_run",
  "accepted_and_restored":true,
  "llm_input":null,
  "llm_output":null,
  "llm_scope":"No LLM message was sent in this backend increment"
}
```

Complete HTTP bodies, revisions, returned artifacts and statuses—not only these summaries—are retained in `tmp/atomic-combined-api-20261007-d/test_atomic_combined_api_previ0/exact_atomic_combined_api_io.json` (approximately 86 MB). Authentication headers and the session token are not exported. The same directory retains `exact_certificate_roundtrip_io.json` and `exact_forged_certificate_io.json`; the accepted snapshot and immutable plan repository are under its `atomic-state/` directory. Compiler inputs/outputs are retained under `tmp/combined-edits-canonical-20261007-b/`. Additive file before/after hashes and exact commands are in `results/repairs/atomic_combined_20261007/repair_io.json`. Earlier file and billed-LLM reports remain unchanged and linked below.

## What changed today

- Replaced an obsolete metrics-copy assertion with checks of the child plan ID, content hash, evaluation ID, all five visible values, and each metric's plan identity.
- Added an independent metric-evidence verifier and adversarial regression checks. That first repair changed only test/evidence code. The later mobile investigation reproduced a real scroll-chaining defect and added one CSS declaration to contain conversation scrolling; no optimizer code was changed.
- Corrected stale current-status and README statements. Historical runs and previous reports remain intact.
- Fixed three connected desktop defects in the existing layout and map-resize code: dock overlap, a selected marker pushed outside the canvas, and a marker covered by the details card or legend. Added real DOM hit tests and canonical-coordinate/plan checks; screenshots alone and tile coverage alone are insufficient.
- Fixed Windows test-report decoding: deterministic children emit UTF-8 and both runner layers decode UTF-8 explicitly. The original Unicode reporting failure, original browser failures, and an intermediate independent-verifier failure are retained; none is relabeled as a pass. No new runtime dependency or optimizer behavior was introduced by these workspace repairs.
- Aligned the Git-ignored local launch setting from `OPENAI_COPILOT_MODEL=gpt-5.6-terra` to the successfully tested `OPENAI_COPILOT_MODEL=gpt-5.4-mini`. A no-spend configuration check returned `adapter=openai`, `model=gpt-5.4-mini`, `api_key_present=true`. The key value was not printed or changed. A configured key is not itself provider proof; the separate live journey supplies that proof.

## Results that actually ran

| Check | Result | Evidence |
| --- | --- | --- |
| Offline live-chat preflight | 100 passed; first restricted-token attempt retained 2 fixture-access failures / 98 passes | `tmp/live-preflight-20261007-a` and `tmp/live-preflight-20261007-b`; tool transcript |
| Metric/evidence regression after repair | 77 passed | `results/e2e/live_metrics_20261007_v1/pytest_junit.xml` |
| Changed-file lint | Passed | `ruff check` of the three changed Python files; tool transcript |
| First billed website journey | GPT, staging, evaluated child and changed map succeeded; test failed on obsolete metrics wording | `results/e2e/e2e1_live_20261007_v1/` — retained red, not relabeled |
| Exact-output website replay | Passed browser journey and independent verification; zero new provider attempts | `results/e2e/e2e1_replay_20261007_v1/` |
| Fresh billed website journey after repair | Passed browser journey and independent verification; exactly one provider attempt | `results/e2e/e2e1_live_20261007_v2/` |
| Full offline product regression | Finished: 3,000 passed / 28 failed / one error in 17926.12 seconds; source changed during the run, so not final-source certification. Fresh targeted rechecks are reported above | `results/e2e/product_regression_20261007_v1/pytest_junit.xml` |
| Complete deterministic browser aggregate | Deliberately stopped after seven finalized suites / 21 passing cases: vertical slice 7, completed-response race 1, map recovery 3, workspace stability 4, direct edits 4, addition/continuation 1, mobile chat 1. Conversation was interrupted; later suites did not run. Stopped to repair confirmed desktop overlap, not due to a timeout; owned process tree only. No complete-gate claim | Retained incomplete: `results/e2e/e2e1_aggregate_20261007_v1/`; historical complete: `C:/Users/Ztang_Yit_Xiaang/Documents/weather-itinerary-e2e1-results/e2e1_aggregate_20260920_v11/` |
| Mobile browser typing, focus and reduced-height journey | Passed; independently verified 53 artifacts / 192 source inputs; zero provider attempts | `results/e2e/mobile_keyboard_20261007_v1/` — historical checkpoint before the stronger boundary check |
| Stronger mobile scroll-boundary test, before fix | Failed: conversation remained at its bottom, but the itinerary behind it scrolled from 0 to 200 px | `results/e2e/mobile_keyboard_boundary_20261007_v1/` — retained red |
| Same complete mobile journey, after scroll fix | Passed in 49.17 seconds; independently verified 53 artifacts / 192 source inputs; zero provider attempts | `results/e2e/mobile_keyboard_boundary_20261007_v2/` |
| Mobile evidence, runner and responsive checks | 320 passed; changed Python files lint clean | `results/e2e/mobile_evidence_regression_20261007_v2/pytest_junit.xml` |
| Isolated continued-acceptance/restart chain | 1 passed in 254.27 seconds: replacement → duration → time window → role, each accepted and restored; one third-party deprecation warning | `results/e2e/acceptance_chain_20261007_v2/pytest_junit.xml`; exact API I/O: `tmp/acceptance-chain-20261007-b/test_real_replacement_duration0/exact_role_chain_io.json` |
| Restoration of copied failed-run state | Internal diagnostic passed: the accepted duration child restored with the exact original ID/hash and eight expected markers; original test state preserved | `results/repairs/regression_checkpoint_20261007_v1/restoration_diagnostic.json` — not a browser test or a claim that the original failure is fixed |
| Desktop overlap reproduction | Three desktop cases failed; mobile passed. Original run also exposed a UTF-8/GBK reporting failure; a second reproduction retained readable failure output | `results/e2e/desktop_overlay_20261007_red_v1/` and `desktop_overlay_20261007_red_v2/` |
| Dock-layout repair gate | Four browser cases and independent verification passed. The preceding intermediate bundle had four browser passes but failed independent verification due to an array-loading error; that failed bundle remains failed | `results/e2e/desktop_overlay_20261007_green_v2/`; intermediate `desktop_overlay_20261007_green_v1/` retained |
| Selected-marker resize reproduction and repair | Before refitting, two cases failed / two passed: Bixby projected to x=-39.49 at 1440 and x=-4.95 at 1024. After refitting, four cases and independent verification passed; inside-canvas placement alone was not treated as unobscured visibility | `results/e2e/selected_marker_resize_20261007_red_v1/` and `selected_marker_resize_20261007_green_v1/` |
| Selected-marker occlusion reproduction and final repair | Before the overlay correction, two cases failed / two passed: legend at 1280 and details card at 1024 intercepted marker hit tests. After correction, four cases passed in 113.42 seconds; 322 artifacts / 192 source inputs independently verified; zero provider attempts | `results/e2e/selected_marker_occlusion_20261007_red_v1/` and `selected_marker_occlusion_20261007_green_v1/` |
| Final focused desktop/runner/verifier checks | 281 passed in 23.83 seconds; changed Python files lint clean. Earlier 84- and 184-test support runs retained separately, not added as unique coverage | `results/e2e/desktop_marker_units_20261007_v3/pytest_junit.xml` |
| General Node frontend contracts | 249 passed, zero failed/skipped when supplied with the correct real evidence inputs. Six specialized API/token-driven file entry points are excluded from this invocation; this is not a complete Node/API gate. Initial incorrect invocations are not product-regression evidence | `results/repairs/desktop_overlay_20261007/node_replay_io.json`; historical response replay inputs are distinct from new browser evidence |
| Fresh billed website journey after all workspace repairs | One real browser journey passed in 44.90 seconds; independent verification of 49 artifacts / 193 source inputs; exactly one provider attempt; no acceptance, browser errors or failed requests | `results/e2e/e2e1_live_20261007_v3/` |

The live and replay bundles each contain 49 verified artifacts and 193 source inputs. The replay manifest content hash is `e04692a5d5cde37b77b584bf510e5c35372ccec7354b67ebf6ed7409a7009891`; the fresh live manifest content hash is `32f95e0741c2105eed591498977125a5ac88bc1f50185c54d7d693bc9ab0df08`.

The broad regression began before today's repairs; its completed result is not frozen-final-source certification. Fresh focused rechecks and their retained failures are recorded above. The earlier aggregate started at 06:13 UTC, then its verified owned process tree was stopped before editing the confirmed workspace defect. New workspace and conversation bundles bind their subsequently repaired source independently. Documentation-only checkpoint changes are separate. A complete frozen regression, fresh aggregate, remaining combined-edit boundaries and native-keyboard evidence still prevent full phase closure. No running, stopped, intended or browser-only intermediate result is a complete-gate pass.

Exact regression command:

```powershell
python -m pytest tests/product_app --ignore=tests/product_app/browser -q -p no:cacheprovider --basetemp tmp/product-regression-20261007-a --junitxml results/e2e/product_regression_20261007_v1/pytest_junit.xml
```

Exact earlier aggregate command; deliberately stopped, not a completed-result claim:

```powershell
python scripts/run_e2e1_vertical_slice.py --browser-test aggregate --output-dir results/e2e/e2e1_aggregate_20261007_v1 --port 8127
```

The regression, mobile and workspace runs explicitly disable the provider key and force deterministic mode. Deterministic browser runners enforce provider-transport isolation; the final workspace receipt records zero dispatched and zero blocked attempts. The separately billed tests retain their own provider counts.

## Mobile interaction repair — latest checkpoint

The existing mobile journey now types every character through real browser keyboard events instead of filling the field directly. It uses touch Send for the first two messages and Enter for the third. Recorded observations prove that Shift+Enter inserts a newline without a request, Tab/Shift+Tab move focus between input and Send, Escape closes the dock and restores focus, and all three requests are submitted exactly once. At a simulated reduced viewport of 390 × 480, the input and Send are both 44 px high, fully visible and unobstructed; the composer spans y=357 through 412. This is not a physical phone or native virtual-keyboard test.

The first boundary run reproduced a real problem: at conversation scrollTop 865 / content 988 / visible height 123, another browser wheel event moved the underlying page from y=0 to y=200. Adding `overscroll-behavior-y: contain` to the existing `.conversation` rule kept both conversation and page unchanged at the boundary. The corrected complete journey still staged the exact replacement, rendered the independently evaluated child, accepted it explicitly, reloaded it, and verified accepted Evidence scope. Console errors and failed requests were empty. All five full-size screenshots were taken after map readiness.

Exact website inputs and visible outputs for the corrected deterministic mobile run:

| Exact input | Exact assistant output | Disposition |
| --- | --- | --- |
| `Keep Stearns Wharf.` | `I prepared this typed suggestion for review. Nothing was staged or changed.` | `keep_stop`; no automatic draft or map change |
| `Replace Bixby Creek Bridge with Pfeiffer Big Sur State Park and remove Stearns Wharf.` | `Your request contains replace, remove. Removing or excluding an accepted stop is not enabled in this repair workflow. No part of this combined request was staged or changed. You can submit a supported edit separately; the removal will not be substituted with another operation.` | Unsupported; no partial draft or map change |
| `Replace Bixby Creek Bridge with Pfeiffer Big Sur State Park.` | `I prepared this typed suggestion for review. Nothing was staged or changed.` | Proposal ready; only explicit Add/Preview/Accept changed state |

The corrected mobile run rendered and reloaded `plan_bc13d7ed8c7ec60b`, content hash `ab610263c7f5a68f`, with Bixby absent and Pfeiffer present. Acceptance decision `decision_4b1d69cd78714262b2347b3ee2278697` retained the immutable original parent. Full exact chat HTTP/SSE inputs and outputs are in `mobile_chat/exact_chat_io.json` and the HTTP capture files; keyboard observations are in `mobile_chat/mobile_keyboard_assertions.json`. The independently verified manifest hash is `4dc5ab4b88de3eefe682ffeabbd84ebf851aca687affdb01d9506a99556a177c`.

LLM input: `null`. LLM output: `null`. Measured provider dispatches: `0`; blocked attempts: `[]`. These are deterministic browser tests, not new billed GPT evidence. The earlier billed v1/v2 and replay bundles remain unchanged, but because the CSS and shared verifier changed afterward, they are historical source-bound checkpoints, not the new final-source claim. The additive mobile repair ledger is `results/repairs/mobile_keyboard_20261007/file_io.json`; the earlier closeout ledger is preserved.

## Desktop workspace repair — actual browser evidence

Input: the real workspace-stability browser journey selected Day 5 / Bixby Creek Bridge Viewpoint, returned from Compare to Trip, opened Copilot, and typed `Unsaved UI test text; do not send.` without submitting it. Viewport: 1024 × 768.

Output: the full-size `workspace_stability/1024x768/copilot_open.png` visibly shows the fixed right-hand dock covering the right portion of the selected-stop title, details and bottom Preview control. This is a visual-overlap observation, not a fabricated DOM hit-test result. The map itself was ready: container and canvas 613 × 505, projected loaded-tile coverage 1.0, plan `plan_359968d255ff9288`, content hash `750dae617fc4c1e3`. Map-ready time was `2026-10-07T06:22:58.542Z`; screenshot started at `06:22:58.734Z` and completed at `06:22:58.821Z`. The PNG is 193,813 bytes, SHA-256 `517d70b2d17f4befa0ad276851d73bb6618c6fbf5a19588da61159c4c7daf6a9`.

That earlier four-case pass did not test dock-open control visibility. The repaired journey now checks five real DOM hit points on Preview, Undo, selected-stop title, Send and Close, plus clipping against every scrolling ancestor. Copilot takes the right grid column rather than covering it; at narrower desktop widths the timeline is hidden while Copilot is open and restored on close. Keyboard resizing changes the shared column width between 320, 440 and 560 pixels. The details card retains an intentional internal scroll; the legend is positioned above it rather than across the selected marker.

The stronger test then exposed two more genuine failures. First, resizing repainted tiles without refitting selected-day bounds, moving Bixby outside the canvas despite 100% tile coverage. Second, a marker inside the canvas could still be covered by the details card or legend. The final test reads the real MapLibre instance without mocking it, binds the exact stop coordinates to the backend plan, checks its projected canvas position and verifies five real hit points on the marker area return the actual map canvas. The independent verifier rejects wrong plan IDs, changed coordinates, off-canvas points, stale observation times, panel hits and invented probe positions.

Final deterministic input: the same unsent text above, actual module navigation/reloads, keyboard dock resizing and window resizing 1440 → 1024 → 1440. Output: all four viewport cases passed, every selected-marker probe clear, all control visibility checks passed, all captures after ready state, coverage at least 95%, no console errors or blocking network failures. The final manifest hash is `a99b343906444c1c98219dcc751720086f652f758f414dbb550501113f8ba889`. Full-size screenshots, exact HTTP bodies, state, marker projections, readiness timestamps and hashes remain in `results/e2e/selected_marker_occlusion_20261007_green_v1/`; no reduced poster is used as acceptance evidence.

LLM input: `null`. LLM output: `null`. Measured provider dispatches: `0`; blocked attempts: `[]`. No chat was submitted in this workspace gate. The separate fresh billed website journey below supplies actual GPT evidence; no result is inferred from configuration alone. A device-availability question for native phone-keyboard testing is still pending and does not block the other repairs.

## Latest billed website test after workspace repairs — v3

Exact website input, submitted by the browser through the real chatbox:

```text
Replace Bixby Creek Bridge with Pfeiffer Big Sur State Park on Day 5, preserving the rest of the seven-day trip.
```

Exact assistant output:

```text
I prepared a typed itinerary suggestion for your review. Nothing was changed or executed automatically.
```

Exact structured output: `replace_nearby`, target `bixby_creek_bridge_viewpoint`, candidate `pfeiffer_big_sur_state_park`, highlighted day `5`, state `proposal_ready`. No operation was staged until the browser clicked Add; no map changed until it clicked Preview. GPT interpreted the request; the actual backend executed `deterministic_context_aware_heuristic` and the independent evaluator produced the eligibility certificate. This is not a claim that GPT itself optimizes the route.

Current execution receipt: `execution_41fb0a01a7704e5094a3356563229896`. Preview request `w3_preview_0235fb2f54ef1ba7` produced child `plan_bc13d7ed8c7ec60b`, hash `ab610263c7f5a68f`, evaluation `cert_90e853fe0e27d774` and diff `diff_320fae8e32e4c503`. The rendered map ID equals the child ID, and geometry SHA-256 is `d0a3d9a141aef73782d1843541a9c57a3543a5ca10b9b92ba40eea409f7bbc39`. Eight markers: Bixby absent, Pfeiffer present. The current receipt is distinct from previous executions even though deterministic canonical plan IDs match the same supported edit.

The itinerary, metrics, Compare, Evidence and map referenced that same child. All four viewport screenshots and reload preserved the identity. Console errors, failed requests and HTTP errors were all `[]`; Accept and Keep were not invoked. The test used disposable synthetic state, not the user's persisted accepted trip. Source-bound independent verification passed with manifest hash `c7d188d1ad268fd5a51bcacefa17545cbdef967f8197fe7b0bad07f4ed18e6b3`.

Exactly one provider attempt: 5,991 input tokens, 0 cached, 234 output. The recorded estimated cost is `$0.00554625`, calculated from the harness's recorded rates, not an invoice-confirmed charge. Full, unabridged LLM request/response bodies are in `results/e2e/e2e1_live_20261007_v3/live_copilot_chat/provider_exact_llm_io.json`; exact website/API/file I/O, source hashes and all screenshots are linked by `results/e2e/e2e1_live_20261007_v3/exact_io_report.md`. Credentials and headers were not persisted. Earlier v1/v2/replay evidence below remains historical and unchanged.

## Exact website input and visible output

User input, sent by clicking the website chatbox's Submit control:

```text
Replace Bixby Creek Bridge with Pfeiffer Big Sur State Park on Day 5, preserving the rest of the seven-day trip.
```

Fresh live assistant message:

```text
Understood: replace Bixby Creek Bridge Viewpoint with Pfeiffer Big Sur State Park on Day 5, keeping the rest of the trip unchanged.
```

The typed intent was `replace_nearby`, targeting `bixby_creek_bridge_viewpoint` with candidate `pfeiffer_big_sur_state_park`. Adding to draft left the parent map unchanged. Clicking **Preview this repair** produced this exact identity chain:

| Artifact | Exact output |
| --- | --- |
| Preview request | `w3_preview_0235fb2f54ef1ba7` |
| Parent plan | `plan_359968d255ff9288` |
| Child plan / rendered map plan | `plan_bc13d7ed8c7ec60b` |
| Child content hash | `ab610263c7f5a68f` |
| Independent evaluation | `cert_90e853fe0e27d774`; eligible |
| Diff | `diff_320fae8e32e4c503` |
| Rendered geometry SHA-256 | `d0a3d9a141aef73782d1843541a9c57a3543a5ca10b9b92ba40eea409f7bbc39` |
| Marker change | Bixby absent; Pfeiffer present; eight markers total |
| Visible metrics | Route time `1992.62 min`; contextual risk `Unavailable`; preservation `88%`; edit cost `6`; certificate `eligible` |
| Browser errors / failed requests / HTTP errors | `[]` / `[]` / `[]` |
| Acceptance | Not invoked; accepted parent remained unchanged throughout |

The map reported loaded-tile projected-union coverage of `1.0` at every screenshot checkpoint. This is a measured map-space coverage signal, not an assertion that every screenshot pixel is unobstructed by controls.

| Viewport | Map container | Map ready (UTC) | Screenshot started (UTC) |
| --- | --- | --- | --- |
| 1440 × 900 | 651 × 630 | `2026-10-07T05:14:41.077Z` | `2026-10-07T05:14:41.092Z` |
| 1280 × 800 | 507 × 537 | `2026-10-07T05:14:39.163Z` | `2026-10-07T05:14:39.191Z` |
| 1024 × 768 | 613 × 505 | `2026-10-07T05:14:39.795Z` | `2026-10-07T05:14:39.812Z` |
| 390 × 844 | 353 × 592 | `2026-10-07T05:14:40.445Z` | `2026-10-07T05:14:40.463Z` |

Full-size images, not a poster, are in the fresh live bundle's `live_copilot_chat/` directory. Mobile images include the full scrollable page; the readiness receipts record the actual browser viewport separately.

## Exact LLM and file input/output

The two billed attempts were explicit, separate website test runs using the approved synthetic demo and `gpt-5.4-mini` with low reasoning. There was no automatic provider retry. The first browser failure did not erase its successful provider attempt. The replay used the first response exactly and blocked new provider transport.

| Run | Actual model returned | Provider attempts | Input / cached / output tokens |
| --- | --- | --- | --- |
| Live v1, browser red | `gpt-5.4-mini-2026-03-17` | 1 | 5994 / 0 / 228 |
| Replay v1 | No new provider call | 0 | 0 / 0 / 0 |
| Live v2, browser green | `gpt-5.4-mini-2026-03-17` | 1 | 5997 / 4864 / 197 |

Full, unabridged request and response bodies, including the developer prompt, complete context, strict schema, model settings, usage and structured output, are preserved in each live bundle's `live_copilot_chat/provider_exact_llm_io.json`. Each bundle's `exact_io_report.md` includes the exact recorded API/file inputs and outputs and source manifest. Credentials and authorization headers are not persisted; disposable session credentials are explicitly redacted, with original and stored body hashes distinguished.

The runner's `billed_cost_usd` field is a **calculation at its recorded token rates, not a billing-invoice measurement**: v1 calculates `$0.00552150`, v2 `$0.00210105`, total `$0.00762255`. No current pricing or final account charge is inferred from these calculations.

Changed test/evidence files:

| File | Input SHA-256 | Output SHA-256 |
| --- | --- | --- |
| `tests/product_app/browser/test_e2e1_live_copilot_chat.py` | `a3144713ab5e87f5a398008792e2a5ac4421f2aed1ad015938ebc0d642a26feb` | `235b3ef515bd8c08cf9ceb6908edd4f6357432c2eaa63adc39eb37fea5340d65` |
| `scripts/verify_e2e1_live_copilot_chat.py` | `9f77d8adabcc41807a034f6484cbd39b9de52399eb1e1562e1b17b4b667c5bc1` | `a7a1384d7a97fd3c54dc4f7e504bff923255601410721d72636be13ffc97bdca` |
| `tests/product_app/test_e2e1_live_copilot_evidence.py` | `1a27fd318054f303f2a48dc795853adc9d4d9dda712e67ead3f08de8d0212ee9` — reconstructed by removing the exact additive test block, not a pre-edit observed receipt | `2f41c69a57c548376436de18787440a3dfd9facc8a26e3eb915f2c4a8270ed0d` |

Input behavior was an assertion that the metrics contained `Current evaluated draft certificate`. Output behavior checks actual plan/evaluation identities and backend-derived route time, risk missingness, preservation, edit cost and eligibility. Adversarial tests reject a parent ID/hash, foreign evaluation, a single metric's foreign plan and an incorrect value in each of the five metrics.

The current edit ledger is `results/repairs/runtime_closeout_20261007/file_io.json`. It records before/after identities for the changed test/evidence and documentation files. The `.env.local` entry records only the exact non-secret model line; the existing credential file is neither copied nor exported.

## Concern register — no blanket completion claim

"September aggregate" below means the retained, independently verified 50-test registered deterministic browser gate. It is not relabeled as today's unchanged-source gate: shared test/evidence files and existing production CSS, dock sizing and map resizing changed today.

| Concern | Current evidence and remaining boundary |
| --- | --- |
| R01 provider isolation | The fresh 14-case gate measures zero live sends; its two fake-SDK cases are separately labeled. The fresh paid combined website gate measures exactly one actual attempt. Historical runs retain their own counts; no constant-zero or combined total is substituted for telemetry. |
| R02 honest outcomes | Blocked, failed, not evaluated and ineligible paths are covered by deterministic runtime gates; final regression/fresh aggregate remain required. |
| R03 addition readiness | Exact candidate/day/parent/context readiness and continued-addition journeys passed September aggregate. Discovery alone is not executable routing. |
| R04 preservation-only search | Compiler checks and September chat/runtime coverage; final regression pending. |
| R05 pair validation | Each candidate/target pair is validated; invalid/mixed cases remain fail-closed; final regression pending. |
| R06 accepted-plan context | Accepted child is the context baseline; continued-edit and restart journeys in September aggregate; final regression pending. |
| R07 capability consistency | Prompt, validation, controls and compilation are capability-bound; complete control inventory in September aggregate. Unsupported controls stay explicit. |
| R08 negation and clauses | Conservative whole-message guard and chat tests; a single live replacement does not prove arbitrary language correctness. |
| R09 presentation truth | Today's live and replay agree across itinerary, text/context, Compare, Evidence and map. |
| R10 metric provenance | Strengthened today: five visible values and child/evaluation identity independently verified in the real browser. Missing risk stays unavailable. |
| R11 continued acceptance | The fresh 14-case gate verifies A → B → C and acceptance/reload for both staged and one-click combined controls. Earlier restart evidence remains source-bound. The fresh paid sample intentionally did not accept or prove a paid continuation. |
| R12 combined edits | Four original/accepted-parent × staged/one-click combined paths and the fresh 14-case browser gate pass independent verification twice; A → B → C, canonical child/map, acceptance and reload. A separate fresh paid two-edit website test passes with one actual GPT attempt and no acceptance. Mixed relation-specific ownership and remaining conflicting/stale/foreign-owner grouped variants remain open; no full composition closure. |
| R13 precise clarification | Structured missing fields and choices covered by conversation tests/September aggregate; broad regression pending. |
| R14 structured memory | Plan-bound clarification/preferences/proposal disposition covered by multi-turn and lifecycle gates; broad regression pending. |
| R15 durable dispatch | Reservations, recovery and uncertain outcomes tested locally; today's live attempts were counted explicitly, without hidden retry. |
| R16 capacity before dispatch | Capacity and no-spend rejection tests remain in the full regression; current paid journey used one normal turn. |
| R17 retention | Sliding expiry is explicitly 30 days after last activity, distinct from audit evidence. README corrected; clock/deletion regression pending. |
| R18 map readiness | Missing-tile/recovery/graphics gates in September aggregate; today's full-size captures all followed ready state and measured 100% tile coverage. |
| R19 late eligible response | Completed-A-held, B-rendered, A-released scenario passed September aggregate; no stale override accepted. |
| R20 complete gate | September aggregate green for its source. Original non-frozen regression finished 3,000 passed / 28 failed / one error; fresh focused repairs pass, but a complete frozen final-source regression and aggregate remain required. |
| R21 mobile and overlays | Mobile browser keyboard/touch/focus and scroll ownership verified; reproduced background scroll fixed. Desktop dock overlap, off-canvas selected marker and marker occlusion reproduced and repaired. Final four-size workspace gate independently verifies control and selected-marker hit tests, shared sizing, canonical coordinates and readiness. **Native OS virtual keyboard is not tested**; viewport resize is not a substitute. |
| R22 bounded execution | Bounded workers, cancellation and responsiveness have focused tests; full regression is the current check. |
| R23 truthful status | Current manifest, execution checkpoint, E2E-1 checkpoint and README reconciled; historical evidence preserved. No release or research promotion. |
| R24 environment/cache identity | Today's live/replay bind actual browser, dependencies, map and OSRM receipts; cached-client restart is in September aggregate. Fresh aggregate remains required. |

## Next gate

Close the mixed relation-specific ownership and remaining conflicting/stale/foreign-owner grouped-action boundaries. Resolve the recorded-audit fixture portability boundary, then run the complete frozen project regression and fresh source-bound aggregate. Obtain native OS mobile-keyboard evidence and complete the independent phase-level decision/persistence audit before claiming all concerns closed. Do not advance G5, W6-W8 or research milestones on this scoped success.
