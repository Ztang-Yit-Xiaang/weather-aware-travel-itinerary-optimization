# E3 Publication Benchmark Contract Status

**Status:** Evidence bundle publication-ready; complete four-method performance comparison blocked  
**Updated:** 2026-07-27  
**Authority:** `docs/planning/current_execution_plan.md`

## Implemented Contract

- Publication mode requires exactly four distinct method IDs: `context_blind_solver`, `deterministic_context_aware_heuristic`, `progressive_sequential_lexicographic_repair`, and `full_reoptimization`.
- The four IDs resolve to separate executors. Exact methods either complete the declared finite universe or refuse optimality when the candidate cap is exceeded.
- Every method receives an isolated copy of the same frozen parent, scenario, catalog, context, and route matrix. Parent mutation, split leakage, and relabeled method provenance fail closed.
- Failed and strict-ineligible rows are retained with diagnostics and cannot enter rankings.
- Preservation values come only from `PlanDiff`; quality metrics come only from the independent evaluator.
- Numeric benchmark/evaluator values must be finite. Unknown certificate status, missing utility retention, invalidated planner certification, duplicate selected stop IDs, and selected records omitted from the displayed sequence fail closed.
- A row is ranking-eligible only when method provenance passes, the certificate status is `PASSED` or `PASSED_WITH_WARNINGS`, independent eligibility passes, route validation passes, and finite evaluator-owned `utility_retained` exists.
- Manifest readiness is evidence completeness, not universal method success. It requires method coverage, unique scenario-method pairs, complete provenance, complete route lineage, and identical route inputs.

## Latest Immutable Diagnostic — v14

Immutable output:

`tmp_test/research_pipeline_raw/e3-real-production-20260725-optimized-v14-cap50000`

Frozen routing lineage:

- route bundle: `route_bundle_a60c80047098a3b6`
- validated cache SHA-256: `3d2ded2d2a862aa6bcf026429b0a0376386928cfa3af109758301d1647584f71`
- route matrix: `route_matrix_68ab535465b06808`
- cells: 223/223 road validated and endpoint-snap validated; no fallback

The v14 closeout records:

- `method_provenance_complete: true`
- `route_evidence_complete: true`
- `publication_ready: true`
- 8 ranking-eligible rows
- 16 retained failed rows

Eligible rows are the deterministic context-aware heuristic and progressive repair for weather deterioration, hotel unavailability, attraction closure, and new must-visit. Road closure and reduced driving tolerance remain infeasible for both non-exact methods in the frozen universe.

Both exact methods used `max_complete_candidates=50000`. All 12 exact rows still report `complete_candidate_limit_exceeded:50000`; they have no child plan, ranking eligibility, or optimality certificate. The cap increase is a truthful scalability diagnostic, not E3.1 completion.

## E3.M Integrity Hardening — 2026-07-27

The source contract now adds:

- rejection of NaN and infinity in benchmark, evaluator, and interaction paths;
- no fallback from planner-owned values into `PlanDiff` or evaluator-owned metrics;
- sequence-grounded utility totals with one consistent utility field and no fabricated 1.0 ratio for a zero denominator;
- fail-closed planner execution/certification checks and duplicate/off-sequence stop checks;
- missing probe diffs are non-equivalent, permission questions retain interpretation provenance, and unmeasured interaction outcomes remain null;
- canonical four-method evaluation UI, honest no-data state, correct higher/lower-is-better bar direction, and escaped rendered artifact text.

E3.M is verified: the focused matrix passes 45/45, Ruff passes, the full suite passes 265/265, documented project checks pass, and the regenerated real dashboard export passes validation.

The v14 artifacts predate this hardening. Preserve them as historical diagnostic evidence, but do not relabel them as generated under the E3.M contract. E3.3 must use a new immutable root.

## Interpretation Boundary

The v14 bundle is publication-ready under its evidence-completeness contract because all required methods, failures, provenance, and route inputs are present and consistent. It is not a complete four-method performance comparison: only 8/24 rows are rankable, neither exact baseline completed, and two disruption families have no eligible non-exact output. No method-superiority claim is allowed.

## Next Gate

1. Preserve the verified E3.M contract and regenerate all future benchmark/UI artifacts under it.
2. Preserve the verified E3.2 adapter regressions and their retained physical-cause evidence.
3. Resolve D1-A versus D1-B in `docs/current/e3_exact_baseline_strategy_decision.md`; do not raise the cap blindly.
4. Run a new immutable 24-cell E3.3 design only after E3.1 is verified; preserve the verified E3.M and E3.2 contracts.
5. Publish only independently eligible rows and state method/scenario noncoverage and failure counts explicitly.