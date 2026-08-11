# W4 Copilot Accessibility and Content Audit

**Audit role:** `ACCESSIBILITY_CONTENT_AUDITOR`  
**Initial audit:** 2026-08-06  
**Post-fix re-audit:** 2026-08-06  
**Implementation independence:** The auditor did not implement the reviewed
frontend, API, provider, conversation, or test code. The only repository edit
made by this audit is this report. No live OpenAI request was made.  
**Offline implementation verdict:** **PASS after independent re-audit**  
**G4 gate verdict:** **BLOCKED** on the separately authorized live OpenAI
evidence and remaining required independent sign-offs; this audit has no
remaining offline blocker.

## Scope

This audit compared the W4 accessibility and content requirements in
`docs/planning/w4_copilot_provider_transcript_phase_plan.md` with the current
Copilot markup, controller, styles, and browser behavior. It covered:

- provider labels and pre-send disclosure;
- lifecycle, error, refusal, and recovery copy;
- safe rendering of user/provider text;
- keyboard access, Escape behavior, and focus restoration;
- mobile full-screen behavior and composer reachability;
- transcript deletion wording and confirmation;
- proposal-action truthfulness;
- resizer semantics; and
- Copilot touch-target sizing.

The audit did not assess prompt quality, provider transport correctness,
transcript filesystem integrity, or map-artifact validity except where those
surfaces directly affected user-facing Copilot content.

## Commands and evidence

### Initial audit evidence

| Check | Result |
| --- | --- |
| `python -m pytest tests\product_app\test_copilot_frontend_contract.py tests\product_app\browser\test_w4_copilot_journey.py -q --basetemp <unique-temp>` | **PASS — 15 passed in 22.77 seconds.** This established the original disclosure, safe-DOM, persistence/deletion, desktop-focus, UTF-8, console, and normal-height responsive baseline. |
| Initial live headless Chromium keyboard audit at 390×900 | **FAIL.** Escape left the full-screen Copilot open and Tab moved into visually covered background controls. This produced W4-A11Y-002. |
| Initial live headless Chromium provider-failure simulation | **FAIL.** An intercepted HTTP 504 ended as `idle: Conversation restored`. This produced W4-A11Y-001. |
| Initial live headless Chromium reduced-height check at 390×360 | **FAIL.** The composer controls ended at y=364 in a 360-pixel viewport. This produced W4-A11Y-005. |
| Initial live transcript injection probe using `<img src=x onerror=...>` as user text | **PASS.** The string remained literal text, no nested image appeared, and no script flag was set. |

### Post-fix independent re-audit

| Check | Result |
| --- | --- |
| `python -m pytest tests\product_app\test_copilot_frontend_contract.py tests\product_app\browser\test_w4_copilot_journey.py -q --basetemp <unique-temp>` | **PASS — 17 passed in 22.05 seconds.** The two new browser regressions cover persistent provider failure and reduced-height mobile focus containment/Escape behavior. |
| Independent intercepted-504 browser replay | **PASS.** Final lifecycle text was `Failed: openai_timeout`, `data-state` was `failed`, and the submitted text remained in the composer. |
| Independent mobile replay at 390×360 | **PASS.** The open dock exposed `role="dialog"` and `aria-modal="true"`; textarea and send controls ended at y=287; both were completely inside the 360-pixel viewport. |
| Independent forward and reverse focus-loop replay | **PASS.** Tab from Send moved to Transcript settings; Shift+Tab from Transcript settings returned to Send. No covered background control received focus. |
| Independent Escape/focus-restoration replay | **PASS.** Escape hid the mobile dialog and returned focus to `#mobile-copilot`. |
| Independent breakpoint-semantic replay | **PASS.** Resizing the open surface from 390 to 1024 pixels changed its role from modal dialog to complementary and removed `aria-modal`. |
| Independent resizer-semantic replay | **PASS.** The separator exposed min 320, max 560, current 440; ArrowLeft resized it and updated current value to 460. |
| Copilot target-size inspection | **PASS.** Header icon buttons, proposal actions, quick prompts, composer textarea, and Send now use at least a 44-pixel relevant dimension; short-height mode hides the quick prompts rather than clipping them. |
| Focused color-pair calculation | **PASS for sampled Copilot text pairs.** Muted text on white was 4.69:1, disclosure text/background 7.82:1, white on teal 4.86:1, and failure text on white 5.79:1. This is not a complete visual-state audit. |

The custom browser checks used the documented launcher, a disposable state
root, explicit deterministic mode, and local Chromium. They did not call
OpenAI, inspect `.env.local`, or persist audit conversations in product state.

## Controls verified

- The active provider is truthfully labeled `Deterministic demo` or
  `OpenAI Copilot`; fixture behavior is not presented as live intelligence.
- The OpenAI disclosure appears before the composer and states that visible
  trip context, the user's message, and bounded recent conversation history
  are sent to OpenAI. Both modes disclose local 30-day transcript retention
  and deletion controls.
- The deterministic disclosure states that requests remain local. There is no
  silent provider fallback copy.
- Empty-state text explicitly says Copilot cannot change the trip without a
  separate user action.
- Proposal controls are review actions: show on map, add to draft, preview,
  compare, and review evidence. The controller contains no Accept or
  Keep-original mutation endpoint.
- Adding a typed intent and previewing a repair each require explicit
  confirmation.
- Transcript deletion uses irreversible-action confirmation and explains that
  trip, draft, evaluation, and research artifacts are preserved.
- Closing the desktop dock and closing transcript settings restore focus to
  their openers.
- At mobile widths, the open Copilot is a named modal dialog, contains forward
  and reverse sequential focus, closes with Escape, and restores its opener.
- The open surface updates between mobile dialog and desktop complementary
  semantics when the viewport crosses the breakpoint.
- Conversation text, context chips, labels, lifecycle text, and toast text use
  `textContent`; the live injection probe confirmed that user markup is not
  interpreted as HTML.
- The lifecycle region is an atomic polite status region and names the planned
  sending, interpreting, clarification, permission, proposal, refusal, and
  failure states.
- A provider failure remains visibly failed after conversation restoration and
  preserves the submitted text for retry or correction.
- The desktop separator is keyboard operable and exposes current/minimum/
  maximum width semantics that update with keyboard resizing.
- Copilot controls satisfy the current 44-pixel target contract.
- Normal-height 1024/768/430/390/360 layouts and the reduced-height 390×360
  replay have no unreachable composer or horizontal overflow in the executed
  checks.
- Changed product files contain valid UTF-8 and the passing browser journey
  observed no unexpected console or page errors. The intercepted 504 regression
  allows only its deliberate `/copilot/messages` console failure.

## Findings and closure status

### W4-A11Y-001 — High — failed lifecycle overwritten by recovery

**Status:** **Resolved and independently re-verified.**

The original failure path set a failed lifecycle and then called `restore()`,
which replaced it with `idle: Conversation restored`. The revised path calls
`restore({ announce: false })` and asserts the failed lifecycle afterward. A
new browser regression and the auditor's separate intercepted-504 replay both
confirmed that the final state remains failed and the input is preserved.

### W4-A11Y-002 — High — mobile full-screen Copilot exposed covered background focus

**Status:** **Resolved and independently re-verified.**

The initial full-screen surface was a non-modal aside. Escape did nothing and
Tab moved behind it. The revised controller applies modal-dialog semantics at
mobile widths, traps forward and reverse Tab within the surface, closes on
Escape, restores the opener, and updates semantics when the breakpoint changes.
The new browser regression and independent replay passed.

### W4-A11Y-003 — Low — lifecycle uses a stable code rather than friendly failure copy

**Status:** Open; **non-blocking.**

The final lifecycle now truthfully remains `Failed`, but the immediate detail
is the stable API code `openai_timeout`. Persisted assistant failure turns use
friendlier copy such as “OpenAI Copilot did not respond before the local
timeout.” Reusing that sanitized mapping in the live status would improve
recovery copy, but the current text is truthful, stable, and no longer hides the
failed state. This does not block the offline W4 sign-off.

### W4-A11Y-004 — Medium — resizer lacked value semantics

**Status:** **Resolved and independently re-verified.**

The separator now exposes 320/560 minimum and maximum values, initializes at
440, and updates `aria-valuenow` during pointer or arrow-key resizing. The
independent keyboard replay observed 440 change to 460.

### W4-A11Y-005 — High for W6 — reduced-height composer clipping

**Status:** **Resolved for the current W4 hardening; real-device W6 evidence is
still required.**

The compact height media query reduces header/context/disclosure/lifecycle
space, hides quick prompts, and bounds the composer. At 390×360 the independent
replay observed both textarea and Send end at y=287. W6 must still verify a real
mobile virtual keyboard and safe-area behavior; the deterministic counterexample
that failed this audit is closed.

### W4-A11Y-006 — High for W6 — Copilot targets below 44×44

**Status:** **Resolved for the reviewed Copilot controls.**

Header icons are 44×44; proposal and quick-prompt buttons have 44-pixel minimum
height; the composer input and Send control meet the target. Broader W6 target
measurement remains a separate whole-product gate.

## Remaining uncertainty

- No automated axe-core dependency or current project axe harness exists, so
  this audit does not claim an axe pass. The executed browser checks directly
  exercise the highest-risk W4 interactions.
- A real screen-reader pass was not performed. Announcement order and modal
  interaction should still be manually checked during the final accessibility
  gate.
- The reduced-height check approximates a virtual keyboard by shrinking the
  viewport; it is not physical-device evidence.
- Sampled contrast calculations do not cover every hover, focus, selected,
  disabled, forced-colors, or high-contrast-mode state.
- No live OpenAI response, refusal, authentication error, or rate-limit event
  was triggered. Live-provider evidence requires explicit user authorization
  and remains separate from this offline accessibility result.

## Verdict

The two original blocking counterexamples now pass independently. Safe text
rendering, provider disclosure, explicit advisory actions, transcript deletion
wording, failure persistence, desktop/mobile focus restoration, mobile focus
containment, Escape handling, resizer semantics, target sizing, and the tested
normal/reduced-height composer layouts satisfy the offline W4 accessibility and
content boundary.

The offline accessibility/content verdict is therefore **PASS**. G4 remains
**BLOCKED**, but not by this audit: it still requires the separately authorized
live OpenAI evidence and every other required independent audit sign-off before
W4/G4 can be marked verified or W5 can become ready.
