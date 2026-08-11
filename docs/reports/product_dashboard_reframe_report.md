# Product Dashboard Reframe Report

## Outcome

E3.UX0–E3.UX4 produced a separately versioned, read-only product dashboard at
`runs/e3ux-weather-repair-demo-v6/dashboard_product/`. It presents the accepted
itinerary, disruption, child repair, changed/unchanged scope, tradeoffs,
certificate, evidence, method failures, and route context without replacing
any frozen research UI.

For Windows discoverability, `OPEN_PRODUCT_DASHBOARD.cmd` at the repository
root verifies the v6 entry file, starts the existing local dashboard server,
and opens the correct product URL. The README distinguishes this launcher from
the preserved legacy dashboard command.

## Implemented

- Strict adapter for canonical run artifacts, lineage, finite values, and
  SHA-256 readback.
- Shared customer/research view model with explicit metric owner and
  directionality.
- Customer timeline, issue/result, repair/tradeoff, comparison, evidence, and
  synchronized original/repaired map.
- Research view with requested/executed methods, planner diagnostics, exact-cap
  state, ranking eligibility, route/certificate lineage, `PlanDiff`, evidence
  references, run-relative source paths, and hashes.
- All required truth states in the stable data contract.
- Mobile-first layout, 44px targets, visible focus, reduced motion, map text
  alternative, and six-width browser verification.
- Non-overwritable renderer, product manifest/version, source/asset/screenshot
  hashes, validator, and dedicated tests.

## Preserved

- E3.C4 normalized Folium HTML signature and day/route-debug contracts.
- Modular dashboard and evaluation export.
- Optimizer, evaluator, benchmark, publication, route-model, plan-repository,
  and interaction semantics.
- D1/E3.1/E3.3/E4 gate status.

## Interaction boundary

E3.UX5 was not started. The final UI is read-only and has no accept, keep,
permission, clarification, persistence, or mutation control. E5 remains
`deferred`.

## Evidence

- [Artifact/UI audit](../current/e3ux0_artifact_and_ui_audit.md)
- [Design contract](../current/e3ux1_product_design_contract.md)
- [Browser matrix](product_dashboard_browser_matrix.md)
- [Accessibility report](product_dashboard_accessibility_report.md)
- [Artifact integrity report](product_dashboard_artifact_integrity_report.md)
- [Testing report](product_dashboard_testing_report.md)
