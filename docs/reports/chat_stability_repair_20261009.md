# Copilot stability repair review — October 9, 2026

[Latest additional boundary review: 20 new cases and 29 screenshots](chat_stability_boundary_20261009.html). The evidence below retains its earlier source scope.

Known interpretation and order-compilation defects are repaired within the verified scope. Fresh live OpenAI accuracy, the full browser aggregate and G5 remain unverified.

165 screenshots; 24 completed deterministic browser messages; zero new paid calls. 159 preexisting saved-state files unchanged.

[Open the interactive screenshot report](chat_stability_repair_20261009.html)

## Implementation

- **Order batch:** Validate all requested final slots, reject collisions, and construct the whole day order together. A truly unchanged whole batch still fails as no effect.
- **English interpretation:** Recognize bounded forms, distinguish labelled old values, and ground Bixby to the existing stop alias.
- **Place-name boundary:** Use word boundaries for place substitution. Mixed booking requests remain blocked as a whole.
- **Time semantics:** Ask for clarification before provider dispatch or staging when the contract cannot express the arrival upper bound.
- **OpenAI reply boundary:** Compare locally resolved stop, operation and parameters with the provider reply; retargeting, changed values and extra edits require clarification.
- **Review and errors:** Show exact duration, day/position, role and clock bounds; retain the backend error in the preview state and card.
- **Preference transparency:** Explain unsupported terms, retain supported preferences as proposed only, and do not label the trailing correction word instead as a preference.
- **Latest-message display:** Reveal the latest turn on your own submission; preserve deliberate history reading during later status updates. A failing actual-controller regression now passes.

## Verification

- **Final Python memory + semantic + order API check: 171 passed.** After the last Python edit; includes 120 variations, 13 exact recorded-prompt adapter replays, 3 provider-mismatch mutations, memory tests and the order API preview. The later one-line JavaScript scroll fix has separate controller and browser checks.
- **Safety and domain regression: 287 passed.** Contracts, dispatch safety, duration retention, semantic coherence, privacy, OpenAI adapter, plans and evaluation. Before the last two-line preference wording repair; the affected memory/semantic suite was rerun afterward.
- **Earlier broad affected check: 230 passed; one new assertion failed.** The new test used the wrong certificate field, violations. Corrected to failures and rerun in the 137-test semantic check, then the final 171-test run. This log is retained as a failed run.
- **Intermediate semantic check: 137 passed.** After correcting that test assertion and before the final preference wording fix. Overlaps the final run; counts must not be added.
- **Actual browser: 24 completed deterministic messages.** 11 exact original primary prompts, the exact preference follow-up, booking and recall variants, explicit order wording, preference correction, and repeated final-source checks. Focused reply/card captures supplement the earlier scroll-position-limited screenshots. No fresh C03 Chinese or original C06 relative-order model claim.
- **Actual JavaScript controllers: Four contract files passed.** Exact typed cards, persistent error copy, staging/preview ownership, changed-draft rejection, conversation generation and capacity. The new own-message scroll regression fails before the one-line fix and passes afterward; deliberate history reading remains preserved.

The counts overlap; do not add them. The 29 reserved cases were inspected during repair and are no longer a blind held-out set. The last Python change corrected trailing preference wording. A subsequent one-line JavaScript fix reveals your own new messages while preserving deliberate history reading. Final-source browser repeats include readable replacement, duration, role and order cards, combined preview/reload and the atomic order batch.

## Current limits

The relative Day 4 before wording and Chinese combined request have injected expected-reply adapter tests, not fresh live responses. Weather blocks are preserved. Early conversation screenshots sometimes show older scroll positions. Focused readbacks and repeated exact cards provide readable visual evidence; all earlier attempts remain in the gallery. A few immediate UI actions needed a fresh state read and retry. This is not a completed project-wide gate.

## Next live checkpoint

Proposed: the 13 exact original requests and the preference follow-up; maximum 14 website messages and 14 provider attempts, no automatic retry, synthetic seeded context, fresh session per case, screenshots at each relevant step, no repair acceptance or booking. The prior bounded live batch is complete; this is a new batch for review.

[Exact proposed live batch](../../results/repairs/chat_stability_20261009_v1/proposed_live_batch.json)
