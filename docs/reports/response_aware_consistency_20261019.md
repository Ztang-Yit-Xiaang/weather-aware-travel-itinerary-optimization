# Response-aware repair: representation consistency and matched-pool selection

Date: October 19, 2026. Follow-up to [the initial audit and experiment](response_aware_repair_20261019.md).

## Result and scope

The follow-up fixes inconsistent child day/order representations and adds an explicit, opt-in full-PlanDiff preservation objective. On an identical supplied pool of two complete, independently model-feasible weather repairs, the existing proxy selector chooses weighted edit cost 7 when the more disruptive route appears first; full-PlanDiff selection chooses cost 6. Reversing pool order gives 6 versus 6. Restricting the pool to the disruptive route gives 7 versus 7. This demonstrates a selection-objective blind spot and its limited correction, with no additional candidate coverage or follower search.

This does **not** show that the stock Progressive Repair generator currently returns the worse route: that generator does not supply this reordered response. The original A/B/C benchmark outcomes remain unchanged after the fixes. There is still no demonstrated additional preservation benefit from a separate bilevel algorithm over a correctly scored preservation-first candidate selector. All experiments remain synthetic and publication-ineligible.

## Implemented changes

| Area | Change and evidence |
|---|---|
| Child extraction | [RepairModel.extract_plan](../../src/itinerary_system/repair/master_model.py) writes consistent raw day, per-day one-based stop order, existing order aliases, sequence, day assignments and ordered days. Explicit solved orders also work on previously empty days. Duplicate, unselected or wrong-day explicit sequence entries are rejected. The parent is not mutated. |
| PlanDiff fallback | [_plan_stop_index](../../src/itinerary_system/plans/diff.py) derives implicit order separately per day. An unchanged multiday parent/child pair has zero disruption. Explicit historical fields are not silently overwritten when reading a parent. |
| Independent evaluator | [PlanEvaluator._check_plan_shape](../../src/itinerary_system/evaluation/plan_evaluator.py) rejects contradictory day/order representations with `plan_representation_conflict`. Evaluator identity is now `verify-001-plan-evaluator-v3`. |
| Certificate consumer | [draft_compiler._verify_diff](../../src/itinerary_system/product_app/draft_compiler.py) requires the current evaluator identity. A freshly evaluated v3 certificate passes the direct verifier regression; changing it to v2 and recomputing its hash is rejected. Existing version assertions were updated; no historical certificate was rewritten. |
| Preservation objective | [solve_lexicographically](../../src/itinerary_system/repair/lexicographic.py), with `use_plan_diff=True`, uses actual canonical weighted PlanDiff plus existing non-hard relaxation penalties at the preservation stage. Hard/booked priority, stage tolerances and later objectives remain unchanged. The builder's supplied ownership policy is retained for extraction and scoring. |
| Provenance | Results expose `preservation_cost_basis`; opt-in stage runs use specification `repair-003-plan-diff-lexicographic-v1` and distinct run IDs. Existing requested/executed backend, statuses, bounds, gaps and stage lineage remain available. |
| Experiment | [compare_matched_pool](../../src/itinerary_system/benchmark/response_aware.py) constructs both complete routes with the actual day-route solver, independently checks them and gives exactly the same pool to both selectors. [The runner](../../scripts/run_response_aware_benchmark.py) saves the new comparison with the existing scenarios. |

The production controller still uses its existing default proxy mode. This change adds no dependency, product setting, follower model, route provider or deployment. Representation validation applies whenever the current evaluator runs; old v2 receipts require reevaluation rather than a version-label edit.

## Objective and guarantee

For supplied candidates \(\mathcal C\), the opt-in selector replaces only the preservation stage by

\[
L_{\rm preserve}(x)=D(x,P)+\sum_{j\in J_{\rm soft}}\lambda_j r_j(x),\qquad x\in\mathcal C.
\]

Here \(D\) is the existing ownership-weighted, typed PlanDiff and the second term contains the existing non-hard constraint-relaxation terms in that stage. Those penalties cannot be dropped just because PlanDiff is now computed on a complete artifact. The earlier hard/booked stage still dominates preservation; existing declared tolerances still control which candidates survive each stage. Later context and utility proxy objectives retain their order and meaning. Tests exercise both hard/booked priority and a soft relaxation penalty that would otherwise disappear.

Enumeration is exact over the supplied model-valid pool under those tolerances. It provides no global itinerary guarantee, no extra candidate coverage and no independent feasibility guarantee. Callers must independently evaluate complete final plans. Default proxy metadata is retained for inspection; actual canonical `diff.weighted_edit_cost` is the authoritative comparison value in this experiment.

This is a correction to measuring \(D(R(z),P)\) on available complete responses. It does not establish an independent traveler objective or justify a new bilevel model. The direct/secondary/outside partition defined in the initial report remains unchanged; the cost differences below are total \(D\), not a claim that all edits are induced or outside the intended neighborhood.

## Identical-pool experiment

The accepted order is `l,a,p,b,o`. Weather blocks optional `a`. `l` is locked, `p` is strongly preferred and flexible, and booked `b` must remain within 10:30--10:45. Both alternatives use an indoor replacement and the same context, matrix, ownership policy and independent evaluation rules.

| Complete response | Travel, minutes | Proxy preservation cost | Actual weighted PlanDiff |
|---|---:|---:|---:|
| X: `l,p,b,x,o` | 26 | 6 | 7 |
| Y: `l,y,p,b,o` | 28 | 6 | 6 |

The proxy sees the same addition, deletion and changed route for both alternatives. Actual PlanDiff also counts the two retained-stop order changes in X, at 0.5 each. Both candidates preserve the locked, preferred and booked visits, and have zero modeled feasibility failures. Choosing Y trades two travel minutes for one unit of preservation, consistent with preservation-before-travel semantics. The complete route traces and evaluations are in [matched_pool.json](../../results/response_aware_consistency_20261019/experiment_v2/matched_pool.json).

| Supplied pool order | Existing proxy: actual D | Full PlanDiff: actual D | Interpretation |
|---|---:|---:|---|
| X, Y | 7 | 6 | Correct scoring resolves the proxy tie in favor of preservation. |
| Y, X | 6 | 6 | No additional benefit when the existing tie order already preserves more. |
| X only | 7 | 7 | Objective correction cannot recover an absent alternative. |

These are deterministic controls, not a statistical effectiveness estimate. The route-duration comparison is descriptive: travel was not added as a new selector objective. Neither selector receives additional downstream calls. The synthetic wrapper's model-feasibility flag excludes the expected missing-road finding only; each canonical certificate remains `FAILED`, comparison-ineligible, with synthetic edges explicitly not road-validated.

## Executed validation and reproduction

[The machine-readable receipt](../../results/response_aware_consistency_20261019/validation.json) distinguishes source epochs, test counts, fingerprints and limits.

- Before editing: 239 relevant tests passed. The first 12 new regression tests failed before implementation; this includes missing opt-in functionality and does not mean 12 distinct defects.
- Final source: [19 focused tests passed](../../results/response_aware_consistency_20261019/focused_final.xml). [The test source](../../tests/repair/test_plan_consistency.py) covers unchanged multiday cost, canonical extraction and aliases, conflicting representations, explicit destination-day order, invalid solved sequences, custom policy, same-pool selection, booking priority, retained soft penalties and the actual product certificate verifier.
- Final source: [463 tests outside `tests/product_app` passed](../../results/response_aware_consistency_20261019/nonproduct.xml), in 117.33 seconds. This suite includes the direct product verifier test above without loading inaccessible demo data.
- Repository-wide `python -m ruff check src tests scripts` passed. Scoped whitespace checks passed.
- A fresh nonbrowser fail-fast probe completed with [207 passed and one setup error](../../results/response_aware_consistency_20261019/broad_probe.txt): `PermissionError` reading `runs/california-coast-coherent-demo-v1/manifest.json`. It is not a complete product regression or a full-suite pass. No browser test was rerun. Dataset permissions and frozen historical evidence were preserved.
- Both complete experiments ran six base and twelve perturbed contexts, 54 strategy cells per run, plus the matched-pool controls. Inputs agree; stability JSON is byte-identical. Recomputed A/B/C semantic fingerprint: `a5c883895f06b646`, unchanged from the initial experiment. Recomputed matched-pool fingerprint: `18793c75c310c3fb`.
- Independent readback verified all 70 manifest-listed outputs per run and all 141 frozen source/config/test files against current files. The receipt also records hashes for the other modified version/fixture tests. No source mismatch was found.

To reproduce, use a fresh output directory; the runner refuses to overwrite evidence:

```powershell
python scripts/run_response_aware_benchmark.py --output-root results/response_aware_consistency_reproduction_fresh
python -m pytest -q tests/repair/test_plan_consistency.py
python -m pytest -q --ignore=tests/product_app
python -m ruff check src tests scripts
```

## Decision and next gate

Keep full-PlanDiff mode opt-in until a matched-pool experiment with validated road evidence and accepted parents checked by evaluator v3 succeeds. Ambiguous historical parent fields require an evidence-based migration decision; automatically normalizing the accepted parent would change the reference against which preservation is measured. Also distinguish a route identifier changing from a physical road path changing before drawing road-disruption conclusions.

The highest-value next research step is that small road-backed comparison, with identical supplied candidates and explicit generation-versus-selection ablations. Product regression must separately recover access to its existing demo fixture and rerun its affected flows before product adoption. A separate follower, bilevel solver, column generation or Branch-and-Price-and-Cut remains unjustified by these results. No commit, push, deployment or research/product gate promotion was performed.
