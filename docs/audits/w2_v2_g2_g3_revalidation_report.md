# Corrected-v2 W2/G2 and W3/G3 Revalidation Report

**Generated:** 2026-08-05 (America/Chicago)  
**Track:** CP-010 corrective local product  
**Default run:** `california_coast_product_demo_v2`  
**Manifest SHA-256:** `925eea6e5722a782d48f657efa931e18536e502a7803a839a983747ed79b5e40`  
**Gate result:** corrected-v2 W2/G2 and W3/G3 `verified`; W4 `ready`

## Scope and non-claims

This report revalidates the immutable v2 route-continuity correction and every
W2/W3 behavior that consumes it. It verifies route/artifact integrity,
solver/evaluator lineage, persistent preview, live desktop/mobile geography,
security/state boundaries, accessibility/content, and phase/status truth.

It does not implement or verify OpenAI transport/transcripts (W4), repository
acceptance (W5), complete mobile/PWA behavior (W6), launcher migration (W7), or
replacement verification and user visual sign-off (W8). It does not advance
E3.1, E3.3, E4, E3.UX5, or E5.

## Corrected artifact evidence

- The registry pins v2 by exact manifest hash; 26 declared files match 26 actual
  files with no missing, undeclared, or mismatched content.
- Parent, recommended, and low-driving plans each contain 16 continuous,
  road-validated route legs and zero fallback legs.
- The corrected inter-day relocation is `the_line_la -> stearns_wharf`, 6,784.4
  seconds, 2,108 geometry points, from the pinned cached-OSRM matrix.
- Recomputed route totals are 654.536667 minutes for the parent, 650.57 for the
  recommended child, and 595.463333 for the low-driving child.
- Recommended child `plan_f5ee52459659dcb5` binds diff
  `diff_367bd571d9a8b665` and certificate `cert_686ef65d376b2867`.
- Low-driving child `plan_8aa919c8323dbac0` binds diff
  `diff_ea97896a586cb3af` and certificate `cert_5a6deef4c159d346`.
- Both children are independently eligible and record 16/16 road-validated
  legs. v1 remains byte-identical historical evidence.

## User-like and accessibility verification

- At 1440 x 900, MapLibre/OSM loaded, the repaired route visibly connected Los
  Angeles to Santa Barbara and onward, route styles were distinguishable, OSM
  attribution was present, and no console warning/error appeared.
- Compare and Evidence displayed the exact v2 plans, diffs, certificates, and
  selected-plan lineage.
- A typed draft preview produced an eligible evaluated proposal, survived
  refresh, and was removed by undo without changing any accepted pointer or
  decision ledger.
- At 390 x 844, the document width remained 390 pixels, bottom navigation fit,
  and toolbar/table overflow stayed inside intentional scroll containers.
- The visual map is `aria-hidden`; all 13 post-render canvas, marker, and zoom
  descendants are non-focusable. The separate DOM stop list remains the keyboard
  interface.
- The textual route names Griffith Observatory, The LINE LA, and Stearns Wharf,
  and states eligibility, 16/16 road validation, no fallback, and zero booked or
  locked changes.

## Blocking findings resolved during revalidation

1. The research gate map and planning index still exposed historical v1
   verified/ready wording. They were corrected, historical v1 reports were
   labeled at-generation-time, and per-document negative status assertions were
   added.
2. The hidden MapLibre subtree retained focusable canvas/zoom controls. The
   visual subtree is now made non-focusable after construction and after load.
3. The text map alternative omitted inter-day continuity, booking impact, and
   eligibility. It now derives those facts from the selected artifact and
   evaluation state.

Every resolved blocker received an independent re-audit. Blocking findings were
not averaged away.

## Independent audit verdicts

| Independent role | Scope | Final result |
| --- | --- | --- |
| `v2_route_artifact_auditor` | Manifest, inventories, v1 immutability, route cells, marker truth | PASS |
| `v2_solver_evaluator_auditor` | Totals, children, diffs, certificates, restart persistence | PASS |
| `v2_browser_journey_auditor` | Desktop/mobile map, Compare/Evidence, preview/refresh/undo | PASS |
| `v2_security_state_auditor` | Host/Origin/cache, tokens/revisions, state and evidence boundaries | PASS |
| `v2_accessibility_content_auditor` | Source/API/content/accessibility review and blocker discovery | Source/API/tests PASS after fixes; live recheck reassigned when its browser backend became unavailable |
| `v2_accessibility_live_dom_auditor` | Post-fix dynamic DOM at 1440 and 390 pixels | PASS |
| `v2_phase_gate_status_auditor` | Authority chain, historical labels, research boundary | PASS after fixes |

All auditors were read-only and independent of implementation. The first
accessibility auditor's environment-only live-check block was not treated as a
PASS; the fresh dynamic-DOM auditor repeated and closed that exact scope.

## Automated closeout

- Focused product and `PlanRepository`: **195 passed**, one existing Starlette
  TestClient deprecation warning.
- Repository-wide checks: Ruff passed; context snapshot **5 passed**; full pytest
  **506 passed**, with the same warning.
- JavaScript syntax, scoped diff checks, live health/default-run checks, and the
  immutable v2 manifest hash check passed.
- Live health is core-ready with MapLibre ready. Overall status remains degraded
  only because preserved legacy state awaits validated W5 migration; OpenAI is
  explicitly disabled by the deterministic adapter.

## Gate decision

Corrected-v2 W2/G2 and W3/G3 are verified. W4 is ready for an approved
implementation-ready phase plan. CP-010 remains `in-progress`; G4 and W5-W8
remain planned; all research statuses remain unchanged.
