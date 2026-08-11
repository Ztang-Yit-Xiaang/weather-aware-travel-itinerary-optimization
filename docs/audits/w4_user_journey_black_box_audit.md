# W4 User-Journey Black-Box Audit

**Audit date:** 2026-08-06  
**Audit role:** `USER_JOURNEY_BLACK_BOX_AUDITOR`  
**Implementation independence:** The auditor did not implement or modify the
reviewed launcher, backend, frontend, provider, conversation storage, or test
code. The only repository edit made by this audit is this report.  
**Offline implementation verdict:** **PASS**  
**G4 gate verdict:** **BLOCKED** pending the separately authorized live OpenAI
smoke/evaluation evidence and the remaining required independent G4 sign-offs.

## Scope

This audit exercised the served product as a user through a real Chromium
browser. It covered:

- startup through `scripts/run_product_app.py` on an explicit free loopback
  port and a clean temporary state root;
- navigation to the HTTP `/app` route, with an explicit assertion that the
  browser did not open a `file://` artifact;
- Day 3 and stop-context selection;
- Copilot open/close behavior, deterministic-provider labeling, and local-only
  disclosure;
- typed request submission and proposal-ready rendering;
- conversation and selection persistence after close/reopen and browser reload
  (`Ctrl+R` equivalent);
- deletion of the current conversation and deletion of all conversations;
- focus restoration after closing the Copilot and after both transcript
  deletion dialogs;
- responsive use at 1280, 430, 390, and 360 CSS pixels;
- horizontal-overflow and composer-visibility geometry; and
- browser console errors and uncaught page errors.

The test process explicitly selected `PRODUCT_COPILOT_ADAPTER=deterministic`.
No OpenAI request was made and no API key was read, printed, or exercised.

## Command and result

```powershell
$auditBase = Join-Path ([System.IO.Path]::GetTempPath()) `
    ('w4-user-journey-audit-' + [guid]::NewGuid().ToString('N'))
python -m pytest tests\product_app\browser\test_w4_copilot_journey.py `
    -q --basetemp $auditBase
```

Result:

```text
......                                                                   [100%]
6 passed in 22.37s
```

The successful suite did not produce screenshots. Its failure handlers are
configured to save a full-page screenshot for desktop, responsive-width, and
UTF-8 failures; none was triggered.

## Evidence matrix

| Journey requirement | Browser evidence | Verdict |
| --- | --- | --- |
| Real served application | Launcher subprocess exposes `/api/health`; Chromium opens `{loopback-origin}/app`; the test asserts the final URL begins with that HTTP origin and `/app` | Pass |
| Clean local state | Launcher receives a newly created temporary `--state-root` | Pass |
| Context selection | Day 3 is clicked and becomes `aria-pressed=true`; Copilot chips display `Day 3` and `Griffith Observatory` | Pass |
| Provider truthfulness | Visible label is exactly `Deterministic demo`; disclosure says requests stay on this computer | Pass |
| Proposal lifecycle | User submits `Review a safer weather repair`; lifecycle reaches `Proposal ready`; a user turn and explicit proposal actions render | Pass |
| Close/reopen persistence | Dock closes, focus returns to its opener, and the submitted user turn remains after reopening | Pass |
| Reload persistence | `page.reload()` performs the browser reload; reopening the Copilot retains the turn and Day 3/stop chips | Pass |
| Delete current | Confirmation is accepted, dialog closes, focus returns to transcript settings, and the user-turn count becomes zero | Pass |
| Delete all | A second conversation is created, delete-all is confirmed, dialog closes, focus returns, and the transcript becomes empty | Pass |
| Desktop and mobile widths | Separate clean browser contexts pass at 1280, 430, 390, and 360 px | Pass |
| No horizontal overflow | Both document and body scroll widths remain at or below the viewport width at every required width | Pass |
| Composer reachable | Textarea and Send button are visible, have valid bounding boxes, and remain fully inside the viewport at every width | Pass |
| Focus restoration | Copilot close restores the desktop/mobile opener; transcript deletion restores the settings control | Pass |
| Console/page errors | Console `error` messages and uncaught `pageerror` events are captured; every test asserts an empty collection | Pass |
| Encoding | Rendered body is checked for the project's known malformed UTF-8 markers; none is present | Pass |

## Findings

### Critical, high, and medium

None.

### Informational — W4-UJ-001: this is deterministic browser evidence

The audit proves the local deterministic W4 user journey and browser-state
contracts. It does not prove that the saved OpenAI credential authenticates,
that the selected project can access `gpt-5.6-terra`, or that a live structured
response completes in the browser. Those are deliberately excluded because a
live provider call requires explicit user authorization and may incur cost.

### Informational — W4-UJ-002: responsive functional geometry, not visual sign-off

The four-width checks prove absence of document overflow and that the composer
controls remain reachable in a 900-pixel-high browser viewport. This audit does
not replace the later visual-fidelity rubric, physical-device/virtual-keyboard
assessment, or W6 mobile/PWA verification.

## Verdict and gate consequence

The W4 deterministic Copilot journey is **PASS** for this audit. The product
opens through the launcher and HTTP application boundary, carries the selected
itinerary context into Copilot, renders an explicitly deterministic proposal,
persists it across close/reopen and reload, deletes local transcripts through
both supported controls, restores focus, avoids horizontal overflow at all four
required widths, and emits no observed console or page errors.

This result supports marking the offline W4 implementation complete once the
orchestrator's other deterministic checks and independent audits pass. It does
not verify G4. **G4 remains BLOCKED** until the user explicitly authorizes and
the project records the bounded live OpenAI smoke/evaluation evidence, all
remaining independent G4 sign-offs pass, and the orchestrator durably updates
the gate status. W5 must not be opened from this browser audit alone.
