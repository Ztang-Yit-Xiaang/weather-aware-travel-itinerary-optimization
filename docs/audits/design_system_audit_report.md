# Itinerary Repair Copilot Design-System Audit

**Audit date:** 2026-08-03

**Audit role:** Independent read-only design-system specialist

**Scope:** Current `/app` tokens, components, state styling, responsive behavior,
and correspondence with the approved eight-panel mockup.

**Gate verdict:** **BLOCKING — G2, G6, and G8 fail.**

## Method

The auditor inspected `static/index.html`, `static/css/app.css`, and the rendered
desktop/mobile application in a disposable local session. Source presence was
not treated as proof of usable behavior. No product files were changed.

## Findings

| ID | Severity | Gates | Evidence | Required closure |
|---|---|---|---|---|
| DS-001 | High | G2/G6 | `app.css` contains a small set of local variables but no complete semantic token system for type, spacing, state, elevation, motion, or responsive density. | Define reviewed semantic tokens and require components to consume them. |
| DS-002 | Critical | G2/G8 | `app.js` builds an index-positioned SVG schematic; the dominant visual surface is not the geographic map required by panels 1, 3, and 6. | Render artifact-grounded geography and retain a textual equivalent. |
| DS-003 | High | G2/G6 | Orange is reused for selection, affected state, warning, and change emphasis. | Separate state semantics and pair every color state with text/icon treatment. |
| DS-004 | High | G6/G8 | Focus, border, and small-text combinations are inconsistent and include low-contrast treatments. | Establish AA contrast pairs and one visible focus-ring contract. |
| DS-005 | High | G6 | Several toolbar, navigation, chip, and evidence actions render below the 44x44 px touch target. | Enforce minimum target size without relying on neighboring whitespace. |
| DS-006 | Medium | G6/G8 | Typography is systematically undersized in dense rails, metric cards, and Evidence. | Adopt a readable type scale and validate zoom/reflow. |
| DS-007 | Medium | G2/G6 | Buttons and cards lack consistent variants for selected, busy, unavailable, eligible, and destructive states. | Specify component variants and async state behavior. |
| DS-008 | Critical | G6/G8 | Responsive CSS removes desktop regions without providing the required mobile Overview, Day Details, bottom-sheet map, and Evidence compositions. | Implement purpose-built 430/390/360 px layouts before responsive sign-off. |
| DS-009 | Medium | G6 | Unicode symbols act as the icon system, producing ambiguous names and platform-dependent rendering. | Use a reviewed icon set with visible labels or stable accessible names. |

## Preserved foundations

The existing visual identity, compact metric-card language, resizable desktop
Copilot concept, and overall navigation grouping can be retained. They are
foundations, not evidence that the design-system or visual-fidelity gates pass.

## Closure evidence

G2 and G6 require token/component documentation, viewport screenshots, contrast
and target-size results, and keyboard-visible states. G8 additionally requires
the user-approved mockup rubric. The implementing design agent cannot approve
its own closure.
