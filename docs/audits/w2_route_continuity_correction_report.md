# W2 Route-Continuity Correction Report

**Date:** 2026-08-05  
**Status:** corrected-v2 G2/G3 verified  
**Default run:** `california_coast_product_demo_v2`

## Finding

The v1 product demo compiled road-validated legs within each day but did not require the end of one day to equal the start of the next. All three plans therefore ended Day 3 at `the_line_la` and began Day 4 at `hotel_milo_santa_barbara`, leaving one Los Angeles-to-Santa Barbara continuity gap. The required OSRM cell already existed; this was a demo route-compilation and metric-boundary defect, not a routing-provider failure.

## Correction

- Day 4 now starts at `the_line_la` and ends at `hotel_milo_santa_barbara` after its two Santa Barbara stops.
- Route totals and independent child evaluations now include the relocation.
- The product-demo loader fails closed on any adjacent-leg discontinuity.
- W3 draft preview uses the same corrected day-anchor contract.
- The map keeps exact GeoJSON coordinates and adds presentation-only leader lines for collision-offset markers.
- v1 remains immutable historical evidence; v2 is a new package and the pinned registry default.

## Immutable v2 Evidence

| Item | Value |
|---|---|
| Manifest SHA-256 | `925eea6e5722a782d48f657efa931e18536e502a7803a839a983747ed79b5e40` |
| Parent | 16 road-validated legs; 0 continuity gaps |
| Recommended child | `plan_f5ee52459659dcb5`; `cert_686ef65d376b2867`; 650.57 minutes |
| Low-driving child | `plan_8aa919c8323dbac0`; `cert_5a6deef4c159d346`; 595.463333 minutes |
| Day 4 first leg | `the_line_la` to `stearns_wharf` |
| Fallback legs | 0 for every plan |

## Local Verification

- 194 focused product and `PlanRepository` tests passed.
- Full repository validation passed: 505 tests, Ruff, and context-snapshot checks.
- Ruff, JavaScript syntax, registry/hash validation, and scoped diff checks passed.
- Live health and run registry selected v2 as the default.
- Live API inspection found 0 continuity gaps across all three plans.
- Desktop browser inspection showed the Los Angeles-to-Santa Barbara route and collision-marker leader lines.

Historical v1 independent audits do not verify v2. Corrected-v2 artifact/route,
solver/evaluator, browser journey, security/state, accessibility/content, dynamic
DOM, and phase/status audits now pass. The durable decision is recorded in
[`w2_v2_g2_g3_revalidation_report.md`](w2_v2_g2_g3_revalidation_report.md); W4 is
ready for its own implementation-ready phase plan.
