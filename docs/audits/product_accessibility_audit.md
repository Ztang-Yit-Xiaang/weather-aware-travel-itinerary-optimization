# Itinerary Repair Copilot Accessibility Audit

**Audit date:** 2026-08-03

**Audit role:** Independent read-only accessibility specialist

**Scope:** Semantics, names, keyboard/focus behavior, live status, target size,
contrast, responsive routes, and automated/manual verification readiness.

**Gate verdict:** **BLOCKING — accessibility portions of G3, G6, and G8 fail.**

## Method

The auditor inspected semantic DOM snapshots from disposable desktop/mobile
sessions and reviewed `static/index.html`, `app.css`, and `app.js`. No current
product browser accessibility suite was found. The local Python environment did
not have Playwright installed despite its optional dependency declaration, so no
claim of automated axe/Playwright passage is made.

## Findings

| ID | Severity | Gates | Evidence | Required closure |
|---|---|---|---|---|
| A11Y-001 | Critical | G3/G6 | Interactive day buttons use `role="listitem"`, overriding native button semantics. | Use a real list container with native buttons and selected-state semantics. |
| A11Y-002 | Critical | G6 | Mobile toolbar controls expose symbol-only or ambiguous names. | Provide stable visible labels/accessibility names. |
| A11Y-003 | High | G6 | Mobile route composition does not expose a reliable page-level `h1`. | Give every real route one logical heading hierarchy. |
| A11Y-004 | Critical | G3/G6 | Focusable SVG groups are nested inside a surface presented as `role="img"`; state and target size are fragile. | Keep canvas/map visual semantics separate from DOM stop/action controls. |
| A11Y-005 | Critical | G6 | Mobile Copilot leaves background content tabbable and lacks complete Escape/focus restoration behavior. | Use correct full-screen dialog/route semantics and focus management. |
| A11Y-006 | High | G3/G6 | Some dialogs are not programmatically named. | Bind each dialog to a visible title with `aria-labelledby`. |
| A11Y-007 | High | G3/G6 | Live regions are broad in some areas and absent for important draft/provider/evaluator transitions. | Announce concise state changes without re-reading whole panels. |
| A11Y-008 | High | G6 | Multiple interactive targets are below 44x44 px. | Enforce minimum target dimensions at all supported widths. |
| A11Y-009 | High | G6/G8 | Focus indication and small-text contrast are not consistently AA. | Test tokens and components in default, focus, selected, disabled, and error states. |
| A11Y-010 | Medium | G6 | Resizable separator lacks complete value semantics. | Supply orientation, min/max/current value, and keyboard resizing. |
| A11Y-011 | High | G3/G6 | Selected, affected, warning, and preserved workflow states rely too heavily on color. | Pair color with text, icon, and programmatic state. |

## Closure evidence

Required evidence includes keyboard-only journeys, focus restoration, zoom/reflow,
contrast, target-size inspection, automated accessibility scans, and a manual
screen-reader rubric. Installing a tool is not itself verification; recorded
results must cover the actual user workflows.
