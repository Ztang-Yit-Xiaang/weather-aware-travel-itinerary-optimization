# Chat input review — October 8, 2026

The chat can produce a correct evaluated repair, but varied wording is not robust yet. This user-approved review completed **13 cases / 14 website messages**, with **69 original step screenshots**. It found two English wording gaps, one order-preview failure, an arrival/departure semantic mismatch and incomplete preference recall. No product fixes were made during this evidence run.

Open the [interactive screenshot report](chat_input_review_20261008.html) to inspect each input, response, draft and preview. Every image opens at original size. Search filters the case matrix and screenshot sections. Full conversation text is included for content that scrolls outside a viewport screenshot.

| Case | Request family | Observed result |
|---|---|---|
| C01 | Paraphrased replacement | **Wording gap** — A replacement plus a preservation clause was rejected as multiple edits. |
| C02 | Two edits, reversed order | **Wording gap** — A supported replacement and duration combination failed with this English wording. |
| C03 | Chinese combined request | **Preview verified** — The Chinese combined request produced the requested replacement and exact 90-minute visit. |
| C04 | Standalone duration | **Expected safety block** — 75 minutes was interpreted and evaluated correctly; the existing Bixby weather failure blocked acceptance. |
| C05 | Arrival time window | **Semantic gap** — An arrival window was translated into earliest arrival plus latest departure. |
| C06 | Day-four stop order | **Preview failure** — Both Day 4 order changes staged, but the preview failed without an evaluated proposal. |
| C07 | Stop role | **Expected safety block** — Stearns Wharf was assigned scenic_stop; the existing Bixby weather failure blocked acceptance. |
| C08 | Ambiguous target and candidate | **Safeguard verified** — The assistant asked which accepted stop rather than guessing an ambiguous replacement. |
| C09 | Conflicting duration values | **Safeguard verified** — Conflicting 60- and 90-minute values did not produce an arbitrary edit. |
| C10 | Unsupported booking | **Safeguard verified** — The unsupported hotel booking was refused; no booking or payment occurred. |
| C11 | Supported edit mixed with booking | **Safeguard verified** — A replacement mixed with hotel booking was refused as a whole. |
| C12 | Keep the whole original trip | **Safeguard verified** — Keep original was recognized without recording an unconfirmed decision. |
| C13 | Preference memory | **Partial memory** — Less driving was remembered, but nature was silently omitted. |

## What the successful case establishes

C03 replaced Bixby Creek Bridge Viewpoint with Pfeiffer Big Sur State Park and set Griffith Observatory to exactly 90 minutes. The child is `plan_a31708c1c9ba0ab9`, content hash `dde0f994990b834a`; the certificate is `cert_8569ff5902645ae2` with `PASSED_WITH_WARNINGS` and eligible comparison status. The recorded child, diff, displayed stop details and rendered map agree. Days 1, 2, 3, 4 and 6 are unchanged. Reload restored the same child and geometry. Missing opening-window evidence remains a certificate warning. No repair was accepted.

## Interpretation is separate from feasibility

C04 and C07 correctly represented the requested duration and role. Their certificates reject the child because Bixby remains weather-infeasible, which is an existing disruption in the seeded parent. C05 also reached evaluation, but it used a latest-departure upper bound for an arrival-window request. That interpretation gap is distinct from its expected weather-related acceptance block. The parent remained active for all three ineligible previews.

C06 staged the requested two order rows, then showed an unverified-preview error. Its saved proposal is null and the Repairs view has no evaluated proposal. The captured interface does not expose the underlying error, so this review does not claim a diagnosed backend or verification root cause.

## Recommended next work

1. Expand wording coverage while preserving complete-request validation; rerun C01 and C02 verbatim.
2. Diagnose C06 with the exact stored two-operation draft and expose the actionable preview error.
3. Clarify arrival-only bounds instead of silently translating them into a departure deadline.
4. Report unsupported preference terms such as nature rather than silently dropping them.
5. Show requested duration, order, time-window and role values in the suggestion cards. They currently use a generic advisory sentence.

## Evidence and limits

- Target: the normal local website on `http://127.0.0.1:8127/app/itinerary`, build `d41101983d42457a`, seeded `california_coast_coherent_demo_v1` parent `plan_359968d255ff9288` / `750dae617fc4c1e3`.
- The user explicitly approved these synthetic public California tests. All 14 messages are recorded once, with no automatic message retries. Some requests return through local guards before an external model call. Application records label the provider OpenAI and model `gpt-5.4-mini`; this is not an independent count or capture of raw external exchanges or charges.
- All cases have input, submission and response screenshots. Executable suggestions also have staging and preview captures; C03 and C13 include reload captures. Intermediate screenshots retain their actual loading/pending state. After failed full-page capture attempts, captures use the actual 1280 × 720 viewport. C01 retains its preparation screenshot from before approval as well as the fresh approved test input.
- All 13 accepted-plan pointers still identify the original parent. At test completion (14:08:53 UTC), all 143 production-source/static/config hashes and both saved workspace file hashes matched the pre-review snapshot. During report assembly, `static/js/copilot.js` changed concurrently at 14:10:56 UTC; this review did not edit that product file, and its newer bytes are untested here. The current integrity record retains both scopes. No booking, acceptance, commit, push or deployment occurred.
- No overall release or phase-gate promotion follows from this exploratory sample. Earlier regression and browser results remain separate.

Evidence: [case outcomes](../../results/chat_variation_review_20261008/outcomes.json), [record checks](../../results/chat_variation_review_20261008/evidence_checks.json), [integrity comparison](../../results/chat_variation_review_20261008/integrity.json), [screenshot index](../../results/chat_variation_review_20261008/screenshots.json), [file hash manifest](../../results/chat_variation_review_20261008/evidence_manifest.json).
