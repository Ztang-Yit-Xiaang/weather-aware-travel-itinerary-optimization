# Itinerary Repair Copilot UI/UX Baseline Audit

**Audit date:** 2026-08-03  
**Audit gate:** AUD-0  
**Disposition:** **BLOCKING — product replacement acceptance fails**  
**Auditor role:** Independent read-only UI/UX audit; no product files were changed.

## Scope and evidence

This report audits the current `/app` experience against the approved eight-panel
Itinerary Repair Copilot mockup. It preserves two evidence classes:

- **Observed:** behavior recorded during the 2026-08-02 user-like browser audit.
- **Source-confirmed:** behavior established by current HTML, CSS, JavaScript, API,
  and artifact inspection on 2026-08-03.

An observed result is not generalized beyond the journey that produced it. A
source-confirmed handler is not treated as proof that the user journey works.

## Scorecard

Scale: `0` absent or fundamentally wrong; `1` present but blocking; `2` partial;
`3` acceptance-ready.

| Area | Score | Audit result |
|---|---:|---|
| Product visual identity | 2 | Clear brand and experimental label; not yet mockup-fidelity evidence |
| Navigation and discoverability | 2 | Primary routes and toolbar exist; mobile information architecture is incomplete |
| Desktop hierarchy | 2 | Major shell regions exist; the map-centered hierarchy is undermined by the schematic |
| Real geographic map | 0 | No geographic map runtime or basemap in `/app` |
| Map editing | 1 | Dialog, drafts, and undo exist; interaction and executability are incomplete |
| Desktop Copilot | 1 | Resizable dock exists; fixture-only behavior and incomplete lifecycle block acceptance |
| Compare | 2 | Cards and matrix exist; alternatives are not proven distinct and Inspect is unsafe |
| Evidence | 2 | Structured cards exist; technical data precedes decision-useful evidence |
| Mobile experience | 1 | Responsive styling exists; required dedicated screens and bottom-sheet flow do not |
| Accessibility | 2 | Good foundations with blocking semantic and interaction defects |
| Feedback and recovery | 2 | Toasts and fail-closed messages exist; provider/map/PWA recovery is incomplete |
| Eight-panel mockup fidelity | 1 | Several modules are recognizable, but required real-map and mobile workflows are absent |

## Blocking findings

### UIUX-B01 — Primary map is a schematic, not a geographic map

**Evidence:** Source-confirmed.

- `static/js/app.js:132-176` constructs an inline SVG whose node positions are
  derived from itinerary index.
- `static/css/app.css:114-116` labels the surface `ROUTE CONTEXT · SCHEMATIC` and
  supplies a decorative gradient rather than a basemap.
- The registered plans contain latitude/longitude values, and the run contains
  cached road geometry, so the gap is in product use rather than artifact absence.

**Impact:** Mockup panels 1, 3, and 6 fail their central geographic requirement.
Original, repaired, draft, and alternative routes cannot be evaluated as
geographic layers.

**Required closure:** Render artifact-grounded coordinates and route geometry in
the approved map runtime; provide separate route layers, geographic selection,
fit-bounds, failure recovery, and a textual equivalent.

### UIUX-B02 — Mobile hides desktop regions without supplying the required flows

**Evidence:** Source-confirmed.

- `static/css/app.css:229-255` hides the navigation rail, itinerary rail, and
  repair panel below 820px.
- `static/js/app.js:26` supports only itinerary, map, repairs, compare, and evidence
  routes; there is no mobile Day Details or full-screen mobile Copilot route.
- The mobile map remains the same schematic, and no editing bottom sheet exists.

**Impact:** Mockup panels 6–8 are not acceptance-complete even though the page can
avoid horizontal document overflow.

**Required closure:** Build dedicated Overview, Day Details, full-screen Map with
bottom sheet, full-screen Copilot, stacked Compare, and accordion Evidence at
430/390/360px.

### UIUX-B03 — Map selection is not reliably demonstrated end to end

**Evidence:** Observed plus source-confirmed discrepancy.

- During the recorded browser journey, activating a map marker did not update the
  visible selected state.
- `static/js/app.js:169-175` does register click and keyboard handlers that call the
  selection API.

**Impact:** Source presence cannot close the interaction gate; a user may not know
which stop or segment subsequent edits affect.

**Required closure:** Reproduce and fix the journey, then verify selected stop and
segment state through API revision, DOM state, Copilot context, and map styling.

### UIUX-B04 — Core map tools do not produce usable repairs

**Evidence:** Observed and source-confirmed.

- Move Day, lock, flexible, route feedback, Undo, and Preview controls are present.
- `service.py:58-84` makes only route-feedback-only drafts eligible; other edits are
  preserved but fail as `operation_not_executable_in_frozen_fixture_universe`.
- The recorded journey confirmed route-feedback draft and Undo, but not a general
  edit-to-evaluated-repair lifecycle.

**Impact:** Panel 3 looks interactive but does not deliver the requested core
editing outcome.

**Required closure:** Compile every approved controlled operation into repair
execution, independent evaluation, and truthful eligible/ineligible states.

### UIUX-B05 — Compare contains a destructive Inspect path

**Evidence:** Source-confirmed.

- `static/js/app.js:239` labels eligible cards `Inspect option`.
- `static/js/app.js:245` invokes `keepOriginal()` when that card represents the
  baseline.
- `keepOriginal()` posts a durable decision at `static/js/app.js:424-431`.

**Impact:** A review affordance mutates decision state. This is a critical action-
semantics failure.

**Required closure:** Make Inspect preview-only. Keep original must be a separately
labeled, confirmed decision.

### UIUX-B06 — Copilot lacks the required real-provider lifecycle

**Evidence:** Source-confirmed.

- The dock, context chips, composer, focus restoration, and resize affordance exist.
- The UI labels every response `Fixture Copilot`; the backend always constructs
  `FixtureCopilotAdapter`.
- No provider-not-configured, authentication, timeout, rate-limit, structured-output,
  cancellation, or live-provider state is implemented.

**Impact:** Mockup panels 2 and 7 remain a demo rather than the approved product
workflow.

**Required closure:** Preserve the deterministic adapter for tests, add the approved
provider adapter, and render the complete request/permission/solve/evaluate lifecycle.

## Accessibility and design-system findings

| ID | Severity | Finding | Evidence |
|---|---|---|---|
| UIUX-A01 | High | Day buttons override their native role with `role="listitem"` | `static/js/app.js:113` |
| UIUX-A02 | High | Interactive SVG groups inside an SVG declared `role="img"` form a fragile accessibility model | `static/js/app.js:146-154` |
| UIUX-A03 | High | Mobile toolbar visually hides text, leaving symbol glyphs without explicit accessible names | `static/css/app.css:242`; toolbar markup in `index.html` |
| UIUX-A04 | Medium | Several controls specify 34–42px minimum height, below the required 44px target | `static/css/app.css:164`, `:177`, `:191`, `:206` |
| UIUX-D01 | Medium | Selected and affected stops both use the same orange fill | `static/css/app.css:121-122` |
| UIUX-D02 | Medium | Evidence typography is dense (10px) and identifiers dominate the visual texture | `static/css/app.css:171-176` |
| UIUX-D03 | Medium | Responsive behavior is primarily omission and stacking, not a dedicated mobile composition | `static/css/app.css:229-276` |

Positive foundations to preserve include the skip link, semantic page landmarks,
native dialogs, explicit focus restoration for the Copilot dock, reduced-motion
support, and a textual map-alternative field.

## Eight-panel coverage

| Panel | Current evidence | Acceptance status |
|---|---|---|
| 1. Main Repair Review | Shell, itinerary, repair panel, metrics, schematic | **Fail** — real map and reliable selection absent |
| 2. Desktop Copilot | Resizable fixture dock and context chips | **Fail** — real lifecycle absent |
| 3. Interactive Map Editing | Draft dialog, four operations, undo/preview | **Fail** — non-geographic and mostly non-executable |
| 4. Compare | Cards and metric table | **Fail** — distinct-plan proof and safe actions absent |
| 5. Evidence | Four evidence cards and download | **Partial** — hierarchy and failure-state coverage incomplete |
| 6. Mobile Overview | Same main workspace with hidden rails | **Fail** — dedicated overview absent |
| 7. Mobile Copilot | Full-width fixture dock | **Fail** — route/lifecycle/keyboard evidence absent |
| 8. Mobile Evidence | Single-column evidence cards | **Fail** — required disclosure hierarchy and journey absent |

## Gate recommendation

Do not mark the product visually or interactively verified. G2 must close the
geographic map; G3 the controlled edit lifecycle; G4 the Copilot lifecycle; G5
Compare/Evidence actions; and G6 the dedicated mobile and accessibility contract.
Final user sign-off is required only after those gates have objective evidence.
