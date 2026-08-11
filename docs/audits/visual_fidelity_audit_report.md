# Itinerary Repair Copilot Visual-Fidelity Audit

**Audit date:** 2026-08-03

**Audit role:** Independent read-only visual-fidelity specialist

**Reference:** Approved eight-panel Itinerary Repair Copilot mockup

**Gate verdict:** **BLOCKING — G2, G6, and G8 fail.**

## Method

The auditor compared the live disposable `/app` at desktop and mobile widths
against each numbered mockup panel. Scores use `0` absent/wrong, `1` blocking
approximation, `2` recognizable partial implementation, `3` close with minor
defects, and `4` acceptance-ready. Dynamic basemap pixels could not be masked
because the current product has no basemap.

## Panel scorecard

| Mockup panel | Score / 4 | Verdict |
|---|---:|---|
| 1. Main Repair Review | 2 | Shell recognizable; geographic map and hierarchy fail |
| 2. Desktop Copilot | 2 | Dock recognizable; lifecycle and proposal detail incomplete |
| 3. Interactive Map Editing | 1 | Controls exist; required geographic edit experience absent |
| 4. Compare Alternatives | 2 | Cards/table partial; distinct alternatives and hierarchy unproven |
| 5. Evidence View | 2 | Content partial; technical identifiers dominate |
| 6. Mobile Overview | 0 | Required composition absent |
| 7. Mobile Copilot | 2 | Partial overlay; dedicated full-screen experience incomplete |
| 8. Mobile Evidence | 2 | Partial stacked content; route/accordion polish incomplete |
| **Total** | **13 / 32** | **Fail** |

## Findings

| ID | Severity | Gates | Evidence | Required closure |
|---|---|---|---|---|
| VF-001 | Critical | G2/G8 | Central schematic does not resemble or function as the approved geographic map. | Implement geographic basemap, route layers, stops, selection, and truthful fallback. |
| VF-002 | Critical | G6/G8 | Mobile Overview panel is absent. | Build the issue/day/repair/metrics/bottom-nav composition. |
| VF-003 | Critical | G2/G3 | Geographic interactive edit panel is absent. | Deliver controlled map edit states with draft bar and bottom sheet. |
| VF-004 | High | G2/G8 | Route labels collide or truncate and do not communicate distinct plans. | Establish collision-safe labels and distinguish route states. |
| VF-005 | High | G4/G8 | Desktop Copilot lacks the mockup's complete conversational/proposal hierarchy. | Render provider, clarification, permission, proposal, and action states. |
| VF-006 | High | G5/G8 | Compare lacks clear alternative identity and decision hierarchy. | Show baseline plus two distinct artifact-backed children and one primary choice. |
| VF-007 | High | G5/G8 | Evidence is technical-first rather than decision-first. | Reorder content progressively; move hashes below eligibility/checks/routes. |
| VF-008 | High | G6/G8 | Mobile Copilot is not the required dedicated full-screen screen. | Add route-safe composer, context, proposal, and keyboard behavior. |
| VF-009 | High | G6 | Mobile toolbar labels are ambiguous. | Use clear labels and accessible icon treatment. |
| VF-010 | High | G6 | Touch targets are inconsistent. | Meet 44x44 px minimum. |
| VF-011 | Medium | G2/G8 | High-level workflow header is under-specified relative to the mockup. | Restore the five-step flow with responsive simplification. |
| VF-012 | Critical | G8 | No approved `/app` screenshot baseline exists. | Capture deterministic target widths and obtain user rubric sign-off. |

## Closure evidence

Visual closure requires deterministic screenshots at 1440, 1280, 1024, 768,
430, 390, and 360 px, panel-by-panel rubric evidence, and final user approval.
Automated diffs may pre-screen but cannot replace that approval.
