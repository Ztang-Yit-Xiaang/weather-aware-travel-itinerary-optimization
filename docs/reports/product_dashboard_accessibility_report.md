# Product Dashboard Accessibility Report

**Status:** `verified` for E3.UX4 read-only scope  
**Artifact:** `runs/e3ux-weather-repair-demo-v6/dashboard_product/`

## Verified

- Semantic page landmarks, one `h1`, ordered section headings, native buttons,
  data tables, and native disclosure widgets.
- Skip link, live load/map status, `aria-pressed` mode state,
  `aria-current` selected day, and explicit map label/text alternative.
- Customer/research switching and day selection preserve review state.
- Visible focus: solid `rgb(11, 95, 255)` outline with offset.
- No interactive target below 44×44px at 390px.
- Reduced-motion media query removes animation, transition, and smooth scroll.
- No color-only meaning: all states have text labels.
- Six-width browser matrix has no horizontal document overflow or clipped
  control.
- Automated computed-style review checked 196 visible leaf-text nodes at 390px
  with zero WCAG AA threshold failures.

## Correction made during audit

The initial warning foreground `#a65a10` measured 4.16:1 on `#f7e5cb`, below
the 4.5:1 normal-text threshold. The final token is `#854306`; the repeated
computed-style audit has zero failures.

## Keyboard evidence boundary

Controls are native buttons/summary elements, focus order remains DOM order,
and browser inspection verified visible focus. The browser automation backend
did not synthesize a default click from its locator-level Enter helper, so the
report does not claim a separate physical-key activation observation. Native
semantics, focus state, click flows, and static contracts are covered; a future
participant-facing study should repeat activation with physical assistive
technology before making a usability claim.

## Not claimed

This is an engineering accessibility gate, not a user study or formal WCAG
conformance certification. No screen-reader comprehension or participant
outcome claim is made.
