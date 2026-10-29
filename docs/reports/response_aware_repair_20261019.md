# Response-aware itinerary repair and induced disruption

Date: October 19, 2026. Research scope: Context-Aware, Inspectable Itinerary Repair.

Follow-up: [representation consistency and matched-pool selection](response_aware_consistency_20261019.md) fixes the extraction issue identified below and evaluates an opt-in full-PlanDiff selector. This report and its original evidence describe the initial source snapshot; follow-up validation is recorded separately.

## Finding and decision

The phenomenon exists in the executed toy model: improving one directed travel edge from 40 to 1 minutes makes the existing full-reoptimization executor choose a 26-minute itinerary instead of the accepted 30-minute itinerary. Its parent-preservation cost rises from 0 to 4.5, including an unrelated stop's order change. A feasible, unchanged alternative still exists. Existing Progressive Repair already prevents that change. These experiments do **not** establish an additional preservation advantage from downstream-response awareness over Progressive Repair.

The useful immediate extensions are complete-plan disruption accounting, explicit counterfactual traces, and better candidate coverage/selection. One weather case reduces travel from 30 to 28 minutes at unchanged preservation cost, but reranking B's existing candidates obtains the same result. Another weather case succeeds where B produces no repair, because the experiment combines edits that B's generator does not combine. That is evidence about candidate coverage, not proof of a necessary bilevel model.

This is a synthetic directed-graph experiment. Every edge has `road_validated=False`, no geographic distance is invented, and every canonical independent certificate remains `FAILED` for missing road evidence. Model feasibility, exact finite-search completion, and publication eligibility are separate fields. Nothing here promotes an existing research/product gate, deploys code, or establishes road-network performance.

## Audit of the actual working tree

Audit precedence followed [the active roadmap](../planning/research_pipeline_and_gate_map.md): current source/tests/executed artifacts first, then current status and execution documents, then the canonical technical contract. Read the current planning index, execution plan, technical specification, problem manifest, E3 method/benchmark contracts, contributing and quality instructions. The working tree already contained substantial unrelated edits; they were preserved.

| Component | Actual implementation | Status and limit |
|---|---|---|
| Parent/child and immutable persistence | `research_artifacts.PlanArtifactV2`; [plans/repository.py](../../src/itinerary_system/plans/repository.py), `PlanRepository.save` | Implemented, append-only records with hashes; frozen dataclasses still contain mutable dictionaries, so immutability also depends on caller discipline. Parent hash is checked in the experiment. |
| Typed PlanDiff | [plans/diff.py](../../src/itinerary_system/plans/diff.py), `compute_plan_diff`; [plans/models.py](../../src/itinerary_system/plans/models.py), `PlanDiff` | Implemented, including typed attributes and v1--v7 evidence. Cost is attribute-specific; ownership of existence does not automatically own order. |
| Ownership, priority, permission | `OwnedConstraint.can_relax`, `OwnershipPolicy`; [plans/attribute_ownership.py](../../src/itinerary_system/plans/attribute_ownership.py); `repair.master_model` | Implemented but partial across attributes. Audit reproduced a booked-stop deletion gap in master validation. This patch adds a fixed selection constraint for booked STOP/POI commitments when relaxation is not authorized. |
| Lexicographic optimization | [repair/lexicographic.py](../../src/itinerary_system/repair/lexicographic.py), `solve_lexicographically`; `master_model._objective_components` | Exact selection over supplied candidates. Current stages are hard/booked relaxation, weighted change, contextual-burden **proxy**, utility **proxy**. Master change variables omit actual within-day order/time/typed attribute edits; proxy change need not equal final PlanDiff. No general global-optimality conclusion. |
| Progressive neighborhoods | [repair/neighborhood.py](../../src/itinerary_system/repair/neighborhood.py), `REPAIR_RADIUS_ORDER`; [repair/progressive.py](../../src/itinerary_system/repair/progressive.py), `repair_progressively` | Implemented: stops at first evaluator-approved candidate in the first successful radius. Radius names overstate enforcement: the first `same_stop_time_shift` neighborhood can already generate replacements. |
| Candidate generation | `day_route_solver._candidate_sequences`, `progressive._generate_candidates` | Partial: parent sequence, one substitution, or append; no general permutation search, pure deletion pool, or Cartesian combination of repairs across days. Wider radius alone does not imply complete search. |
| Validated travel evidence | [routing/matrix.py](../../src/itinerary_system/routing/matrix.py), `RouteMatrix`, `validate_route_matrix`; routing evidence bundle modules | Implemented strict road/fallback gates. No historical real-road artifact was promoted or revalidated by this toy experiment. |
| Independent final evaluation | [evaluation/plan_evaluator.py](../../src/itinerary_system/evaluation/plan_evaluator.py), `PlanEvaluator.evaluate_final_plan`; [certificate.py](../../src/itinerary_system/evaluation/certificate.py) | Implemented: lineage, shape, ownership, context, lodging, road, opening/day limits and trip-specific windows. It independently rejects synthetic roads and invalid commitments. Budget checks consume a modeled cost field; not a universal independent cost reconstruction. |
| Full reoptimization | [repair/exact_baselines.py](../../src/itinerary_system/repair/exact_baselines.py), `plan_full_reoptimization` | Separate executable finite-universe search, context then utility/travel/cost, no preservation objective. Candidate cap refusal retains incompleteness. Does not independently validate its own final child; the experiment does so afterward. |
| Solver evidence/fallbacks | `PlannerRun`; `LexicographicStageResult`; `ExactBaselinePlanningResult`; [benchmark/methods.py](../../src/itinerary_system/benchmark/methods.py) | Requested/executed methods, status, bounds, gaps and lineage exist. Zero gap from enumeration is scoped to its declared candidate universe. The progressive extractor's source-run ID is not its stage-run ID; this experiment records an explicit heuristic adapter receipt and retains the original stages. |
| Separate traveler follower, induced metrics, response comparison | No general separate traveler-choice model found; new files linked below | Missing before this task. This patch adds metrics and an isolated bounded experiment, not a production traveler simulator. |

Document conflicts: technical specification §1.2 still describes parent-aware repair, ownership, progressive neighborhoods, generalized diff and independent evaluation as unimplemented. That section is a June snapshot, contradicted by current source and the executed baseline. The E3 benchmark contract's older passing counts and v14 real-road results are historical, not this working tree's acceptance evidence. Its reference to `docs/current/e3_exact_baseline_strategy_decision.md` is currently absent. Planning checkboxes/statuses were not rewritten.

Additional correctness gate before general experiments: `master_model.extract_plan` retains raw stop `stop_order` values while changing sequence; `compute_plan_diff` prefers those raw values over sequence-derived positions. Raw `day` similarly takes precedence over `day_assignments`. The toy deliberately has consistent day data and omits redundant order fields. Production generalization needs canonical shape consistency, not an assumption that every artifact measures changed order correctly.

This was reproduced on the current source: the child sequence changed to `l,p,b,a,o` while raw parent orders remained, and the diff reported zero reorder events and only the route-ID cost of 3. See [artifact_consistency_receipt.json](../../results/response_aware_20261019/artifact_consistency_receipt.json). This unresolved diagnostic does not invalidate the consistent toy fixture, but blocks a claim that this metric measures all arbitrary production artifacts correctly. The toy also omits stored arrival/departure timestamps, so PlanDiff does not charge recomputed schedule shifts at other flexible stops; only recorded attributes are measured. Independent schedule checks still enforce the booking.

## Definition and mathematical scope

Let P be the accepted parent, z an action in the updated frozen context, and R(z) the **complete** result after a specified downstream response. Let E(P,x) be the canonical typed PlanDiff events. For event e, c(e) is its existing ownership- and attribute-sensitive cost. Do not assign one uniform stop cost or transfer existence ownership to every attribute.

Before solving, declare (i) the intended neighborhood N, with stop IDs and days, and (ii) an exact set A of direct edit keys `(diff_family, target)`. STOP edits target stop IDs; road/lodging edits target a day string. A is a declaration of intended changes, not a permission grant or proof of necessity.

\[
D(x,P)=\sum_{e\in E(P,x)}c(e),\qquad
D_{direct}=\sum_{e\in E(P,x)\cap A}c(e),\qquad
I_A(x,P)=\sum_{e\in E(P,x)\setminus A}c(e).
\]

\[
I_{out}(x,P;A,N)=\sum_{e\in E(P,x)\setminus A}c(e)\,1[e\text{ affects a stop/day outside }N].
\]

Every event appears once in the direct/secondary partition, so `D = direct + secondary`, and `outside <= secondary`. **Do not optimize `D + I` as though I were an independent cost**: that double-counts secondary edits. Use I as a diagnostic, explicit constraint, or subordinate tie-break. Existing additive attribute facets are retained: a day move and an order change are different charged attributes, not two deleted stops. Unique secondary stop IDs are separately deduplicated.

Within direct edits, distinguish context-required edits M from elective local choices `A\M`. Thus `D = D_required + D_elective_direct + I`. In the weather fixtures M consists only of deleting the declared infeasible parent stops (cost 2 per optional stop); adding an indoor alternative is an elective local repair choice, and the expected changed route-ID is lineage accounting. Neither is asserted mathematically necessary. In the edge-update fixtures no itinerary edit is required and A is empty. The serialized request, scope and typed events retain these distinctions without inventing a necessity oracle. For full regeneration, the common scope labels permitted local additions; it does not certify that every such addition was needed. The single-edge controlled counterfactual gives the causal response evidence; a scope partition alone does not establish causation.

The implementation reports total/direct/secondary/outside cost, typed events and ownership, unique secondary/outside stop IDs, changed hotels/day assignments, and propagation radius. Radius is the maximum parent-sequence index distance of a secondary parent stop from the nearest scoped parent stop; it is null without a parent center. It is **not** road distance, days of propagation, or a causal-network radius. Added stops have no parent position.

PlanDiff positional changes can arise from deleting an earlier stop even when surviving unrelated activities keep their relative order. `unrelated_order_inversions` therefore counts pair inversions among surviving unscoped stops that remain on the same day. It is a separate diagnostic, never another added cost. A booked activity changing list position is also distinct from losing its reservation or missing its window.

A verified nonmonotonic response has a local routing improvement L(z1)<L(z0), feasible responses, and `D(R(z1),P)>D(R(z0),P)` (or the analogous secondary metric). To distinguish avoidable spillover from unavoidable repair, retain a feasible counterfactual Q under the **same** updated context with lower D. A useful finite-pool regret is `D(R(z),P) - min_Q D(Q,P)` over declared comparable feasible responses; it is not regret against an unknown global optimum. Exogenous edge-update experiments identify the response to that update for this deterministic planner, not human behavior.

Implementation: [repair/induced_disruption.py](../../src/itinerary_system/repair/induced_disruption.py). Focused invariants and ownership tests: [test_response_aware.py](../../tests/repair/test_response_aware.py).

## Strategies and controlled inputs

All strategies consume the same parent projection, updated context, candidate records, RouteMatrix and day/evaluation configuration within each scenario. Parent metadata is constructed in the shared comparison snapshot because existing PlanDiff requires identical context snapshot IDs; this is not mutating an accepted real-world artifact across contexts. Complete inputs, hashes, outputs and failed attempts are serialized.

* A calls the existing finite-universe `plan_full_reoptimization`. Context burden precedes utility, travel and cost; preservation does not influence its selection.
* B calls the existing `ProgressiveRepairController`. A benchmark-only `SyntheticModelEvaluator` delegates to the real independent evaluator and reports model acceptance only if the sole failures are the explicit synthetic-road gate. Its original independent certificate remains failed and publication-ineligible. This adapter is not installed in product or publication paths.
* C is an experimental complete-response selector. It retains B's model-feasible incumbent and also examines a deliberately separate minimum-travel response to local drop/replacement actions. Each action fixes selected stops; all their one-day permutations are evaluated by the existing route solver and the independent booking/window evaluator. Follower ties use tuple ordering. The selector filters hard model failures, then uses radius, actual D, secondary cost, context burden, utility retention and travel. Secondary cost is a subordinate tie-break, and preservation still precedes utility. All experiments forbid commitment relaxation.

C's safe incumbent and action pool are explicit parts of this policy. It is allowed to keep the inherited order rather than submit to the travel follower. Therefore **C is not a solution of a mandatory pure shortest-route-follower bilevel model**. Its non-worsening preservation when B is in the same-radius pool follows candidate inclusion and selection; it is not independent empirical evidence of superiority. Distinct actions with the same selected set remain distinct trace entries, not independent samples.

The experiment additionally reranks only B's generated candidates and retains the greedy shortest follower response. These ablations distinguish actual-response inspection from proxy reranking and broader action coverage. C timing includes B, downstream generation and the ablation work; this single-run diagnostic does not support speed comparisons.

## Executed benchmark

The accepted sequence is `s → l → a → p → b → o → t`: locked l, optional outdoor a, strongly preferred flexible p, booked b, optional o; s/t are hotel anchors. Each visit takes 15 minutes, the day starts at 09:00 and lasts at most 180 minutes, and b must occupy 10:30--10:45. x/y are indoor alternatives in weather scenarios. Hotel and day remain fixed. Ownership protects existence; p's strong existence preference does not inflate order-change cost. The complete directed edge table is retained in each JSON input.

These are declared leg durations consumed by the actual RouteMatrix implementation, not a calibrated shortest-path closure of a geographic network. The experiment tests the implementation's response to those values. It does not establish the magnitude or frequency of the effect on real road matrices, nor calibrate PlanDiff scores as traveler-reported disutility.

Initially the parent path has six 5-minute legs, totaling 30. The alternative `s → l → p → b → a → o → t` has five 5-minute legs plus `b→a=40`, totaling 65. After `b→a=1`, that alternative totals 26. Enumeration confirms the implementation's actual selected sequences. Reordering a, p and b costs `3 × 0.5`; the changed route-ID event costs 3, yielding D=4.5. Only p is outside the declared edge-endpoint scope `{a,b}`, yielding outside=0.5. b still starts/departs within its fixed window. Of D=4.5, the route event is an evidence-identity change, not a physical geometry-change measurement.

Each table entry is **D / travel minutes / utility retained**, with model feasibility assessed separately from road certification.

| Scenario | A: full reoptimization | B: progressive | C: response-aware |
|---|---:|---:|---:|
| Travel before, edge=40 | 0 / 30 / 1.0 | 0 / 30 / 1.0 | 0 / 30 / 1.0 |
| Travel after, edge=1 | 4.5 / 26 / 1.0 | 0 / 30 / 1.0 | 0 / 30 / 1.0 |
| Travel threshold, edge=5 | 4.5 / 30 / 1.0 | 0 / 30 / 1.0 | 0 / 30 / 1.0 |
| Weather: a unavailable | 7.5 / 29 / 1.2 | 6 / 30 / 1.0 | 6 / 28 / 1.0 |
| Weather: a and o unavailable | 9 / 24 / 1.0 | No feasible candidate produced | 8 / 58 / 0.8 |
| Booked b weather-infeasible | No plan | No plan | No plan |

All returned model-feasible rows preserve l, b, p, their owned constraints, and b's fixed window, with zero modeled hard failures. Their canonical certificate status is `FAILED`, not feasibility-certified for road use. In the booked conflict, fixed inclusion and context exclusion conflict: no relaxation permission is supplied, so refusing a plan is the correct outcome. In the two-stop weather case, B's failure is a generator-coverage failure, not proof that the repair problem is infeasible.

In the single-weather counterfactual, replacing a with x looks attractive to a travel follower: `l,p,b,x,o` takes 26 minutes but D=7 with secondary cost 1. Replacing it with y returns `l,y,p,b,o`, travel=28, D=6, secondary=0. B already attains D=6 using x in the inherited slot at travel=30. C's two-minute improvement is exactly reproduced by reranking B's existing candidates; extra downstream permutation search is unnecessary for that gain. The greedy local pool can instead drop a, travel=25, D=6.5, illustrating an additional utility/preservation tradeoff.

The two-stop case is also a negative quality result for C: it selects D=8 at travel=58 and utility=0.8 rather than A's D=9 at travel=24 and utility=1.0. This follows the specified preservation hierarchy, not a demonstrated traveler preference. It shows why a lower preservation cost alone should not be called a better trip.

Perturb `b→a` by ±0.1 minute in all six scenarios, producing twelve additional contexts and 54 strategy cells per run. Only A at the threshold changes sequence between 4.9 and 5.1 minutes (and one perturbation differs from its deterministic tie outcome at 5). B and C keep the parent order. Other measured base-to-perturbation sequence changes are zero. Both-infeasible outcomes are retained as such, not rewarded as stable feasible plans. Stability here covers sequence response to travel perturbations; context classification thresholds and stochastic traveler responses are untested.

Example observed times in the first run, in milliseconds: travel-after A/B/C=28.4/16.7/32.6; single-weather=189.8/17.7/56.0; two-weather=29.6/3.9/63.8. C includes the work described above. No statistical runtime conclusion is justified.

Primary outputs: [summary.csv](../../results/response_aware_20261019/experiment_v1/summary.csv), [travel_after.json](../../results/response_aware_20261019/experiment_v1/travel_after.json), [weather_choice.json](../../results/response_aware_20261019/experiment_v1/weather_choice.json), [stability.json](../../results/response_aware_20261019/experiment_v1/stability.json), and [manifest.json](../../results/response_aware_20261019/experiment_v1/manifest.json). These generated local results are ignored by Git; regenerate them from source for sharing. Independent reproduction is in `results/response_aware_20261019/reproduction_v1/`.

## Is bilevel optimization warranted?

The reference paper has a genuine objective separation: planner infrastructure investment versus cyclists' perceived-time route response. The publisher abstract supports that structure and the use of a branch-and-price-and-cut algorithm; it does not by itself establish the specific paradox attributed to the full paper. Full text was not accessible during this audit, so that claim is not asserted here. [Publisher abstract](https://www.sciencedirect.com/science/article/pii/S1366554526005570), DOI `10.1016/j.tre.2026.105219`.

Here a locally shorter route improving its own travel objective while worsening preservation is a conflict between two objectives. It is not a proof that total travel worsens, a strict Braess paradox, or a paper-level novelty result.

If the planner controls both local edits and final routing, a single complete lexicographic model can capture the requirement. At a fixed feasible radius, let `x* ∈ argmin_{x∈X} lex(H(x),D(x,P),B(x),-U(x),T(x))`. If X0 is contained in X1 after a travel-time reduction, and earlier arrival may wait, the minimum D at the fixed optimal H cannot increase: the former optimum remains available. If the accepted parent is still feasible and D(P,P)=0, exact preservation-first optimization has no reason to disrupt it for a lower-priority travel gain. This theorem requires the **actual** D, a complete feasible universe and the same hard-policy/radius restrictions. Current proxies, bounded candidate generation, and early stopping do not provide that theorem for arbitrary current executions.

| Approach | Validity and guarantee | Cost / decision |
|---|---|---|
| Existing lexicographic repair + true final D and explicit stability limits | Directly expresses planner-controlled preservation; a true minimum depends on complete modeling/search. Relative-order or outside-change caps can be subordinate constraints. | Highest immediate value; first align proxy vs final metrics and canonical artifact consistency. |
| Local candidates + downstream response evaluation | Useful when routing is a black box or costly, and full responses differ. Optimal only over the supplied actions/responses; include infeasible and tied follower outcomes. | Implemented experimental slice here. Measure matched-pool improvement and additional response-call cost before product integration. |
| True bilevel `min_z D(x,P), x∈argmin_u C_traveler(u;z)` | Appropriate only for a separate traveler/router whose response cannot be overridden by the preservation controller. Need follower objective evidence, feasible-set semantics and optimistic/pessimistic/deterministic tie treatment. | Deferred. Current C can retain B and therefore does not enforce this follower optimality constraint. No real independent follower evidence currently exists. |
| Decomposition / column generation | Could treat feasible day routes as columns, but a useful master, duals, a correct pricing problem and completeness/bounds are prerequisites. | Deferred until profiling demonstrates a route-column bottleneck at representative scale. Branch-and-Price-and-Cut is not implemented or justified by this toy. |

One practical bilevel tie specification is to choose the least disruptive x among minimum-travel follower optima (optimistic), or bound against the most disruptive (pessimistic). Both differ from an arbitrary stable-ID tie. Any future claim must name which process the data support. Candidate enumeration is a valid finite evaluation procedure; it is not automatically a scalable bilevel algorithm or a global guarantee.

## Implementation, reproduction and verification

The only production behavior change is eleven lines adding `booked_select_*` in `master_model._constraints`. Tests reproduce deletion refusal and booked/context infeasibility. New `repair/induced_disruption.py` reuses PlanDiff; `benchmark/response_aware.py` is a synthetic-only experiment; `scripts/run_response_aware_benchmark.py` freezes inputs/source hashes and writes immutable evidence. New tests cover the counterexample, zero/partition/cost/ownership/radius metrics, relative-order vs reindexing, hotel/day changes, counterfactuals, locked/booked preservation, priority, infeasibility, synthetic evidence rejection and reproduction.

Run from repository root with a **fresh** output directory:

```powershell
python scripts/run_response_aware_benchmark.py --output-root results/response_aware_local_01
python -m pytest tests/repair/test_response_aware.py tests/repair/test_master_model.py -q
python -m ruff check src tests scripts
```

Before editing, the relevant existing plans/repair/routing/evaluation/benchmark baseline executed **228 passed**. After the patch, focused tests executed **17 passed** and the same core areas executed **239 passed**; repository-wide Ruff passed. Evidence: `results/response_aware_20261019/{baseline,focused,core}.xml`. Two independent experiment runs have identical input hashes and semantic result fingerprint **`a5c883895f06b646`**; all 69 files listed in the first output manifest were read back and SHA-256 checked. Exact wall-clock times and artifact timestamps are intentionally not claimed byte-identical. Code hashes before/after each experiment matched.

Additional readback independently recomputed booking schedules for all 14 returned base-case plans: service starts at minute 630 (10:30) and departure is minute 645 (10:45), with identical stability outputs across the two runs. [booking_schedule_readback.json](../../results/response_aware_20261019/booking_schedule_readback.json) retains the schedules. [correctness_receipt.json](../../results/response_aware_20261019/correctness_receipt.json) verifies that exactly one duration cell changed (b→a, 40→1), and reconstructs the prepatch master constraint set by removing only the added booked-selection row in memory: deletion then has no validation violation, whereas the patched model rejects it.

Full-repository regression was started separately and its terminal result is recorded in the validation closeout below. A partial progress file never counts as a pass. Optional/live/provider skips are not coverage evidence. No fresh paid LLM call or live road query was part of the experiment.

## Ranked next steps and gates

Scope of the next increment: strengthen the existing preservation-first research path; keep UI/LLM work and a solver replacement outside this increment. Tracking documents retain their preexisting gate states. The dependency chain is `canonical artifact / objective correctness → matched candidate coverage → frozen road benchmark → separate follower evidence → any bilevel/decomposition decision`.

| Priority / status | Concrete task and area | Entry / exit gate | Research value / effort |
|---|---|---|---|
| 0 / verified for this slice | Booked-stop inclusion fix, scope-relative metrics and synthetic experiment | Current source + deterministic tests; executed evidence above. Does not imply a publication gate. | Correctness foundation / small |
| 1 / ready | Canonicalize extracted order/day fields and test adversarial redundant-field disagreement in `master_model`, PlanDiff and evaluator | Enter with preserved existing artifacts; exit when no reordered route can retain a misleading unchanged typed diff and route-ID-only churn is distinguished from physical change. | Essential correctness / small--medium |
| 2 / ready after priority 1 for claims | Align selection with complete final PlanDiff within each radius; retain the earlier hard policy, independent evaluator and actual quality metrics | Matched-pool tests must isolate proxy mismatch; include order/booking/window cases and held-out deterministic scenarios. An explicitly subordinate I cap/tie-break must not override D. | Highest immediate research value / medium |
| 3 / planned | Expand candidate combinations/deletion/permutations only where existing coverage fails; record candidate budget and incomplete search honestly | Priority 1--2; exit with multi-disruption and multi-day feasible counterfactuals, plus a same-pool ablation showing which gain survives identical generation. | High / medium |
| 4 / planned | Frozen road-backed paired experiment, using existing E3 adapters/evidence bundle contracts | Priorities 1--3 and validated duration/road/endpoint evidence for **every** required candidate leg. Exit with strict independent eligible outputs, retained failed rows, bounded scope, latency distributions and perturbation sweeps. | Required for empirical claims / medium--high |
| 5 / deferred | Elicit a genuinely separate traveler or external-router objective/response and tie process | Separate-response evidence; exit with a tested disagreement that cannot be removed by the planner's existing lexicographic control. Only then select a bilevel model. | Potential novelty / high and uncertain |
| 6 / deferred | Decomposition/column generation | Representative profiling, master/pricing formulation and a valid bound strategy. | Scale contingent / high |

Priorities 1--3 control the critical path to a defensible road comparison. Scenario curation and real-route evidence inventory may proceed independently, but no performance ranking should cross the strict evidence gate. The recommended next implementation is canonical artifact consistency plus a matched-pool final-PlanDiff selector; the present experiment provides counterexamples and ablations for its acceptance tests.

## Validation closeout

The broader nonbrowser run **completed with exit code 1**: **3,457 passed, 276 failed, 842 errors**, 4,575 tests total, in 17 minutes 24 seconds by the pytest terminal summary. All 1,118 failed/error cases are in `tests/product_app`; no failure/error occurred outside that package. The 17 new-response/master-model tests also passed inside this completed run. Of the failed/error traces, 795 include `PermissionError` for the preexisting `runs/california-coast-coherent-demo-v1/manifest.json`. A separate completed diagnostic reproduces that exact setup failure (four passes, one setup error). Other product failures were retained, not reclassified or repaired as part of this research task.

There was no complete prepatch whole-suite baseline. Consequently, do not label every remaining failure either a new regression from this patch or a proven preexisting defect. The independently reproduced fixture-access error is established; blanket causal attribution is not. No fixture permissions or historical data were changed.

The initial full invocation including browsers was stopped after more than six minutes without new output in its browser stage. It exited 1 and has **no completed JUnit**. It is interrupted, not passed. The completed nonbrowser run excludes 89 collected browser cases; it cannot replace browser acceptance. No full-project, product, publication or deployment gate passes on these results.

All 140 experiment source/configuration/test SHA-256 hashes were rechecked after regression and remained unchanged. Structured status, affected-test counts and failure-module attribution: [validation.json](../../results/response_aware_20261019/validation.json). Raw complete nonbrowser result: [nonbrowser.xml](../../results/response_aware_20261019/nonbrowser.xml) and [nonbrowser_output.txt](../../results/response_aware_20261019/nonbrowser_output.txt). The interrupted browser-inclusive output remains `full_output.txt` beside them. [repository_identity.json](../../results/response_aware_20261019/repository_identity.json) confirms the requested repository on `main` at HEAD `f0e294cb664c2df991955ec29ac1b55d6b810bbb`; the hashed dirty working tree, rather than HEAD alone, is the tested source.

The implemented synthetic slice is verified by its scoped tests and repeated evidence. General real-road/product integration remains gated by canonical artifact correctness, candidate-pool controls, validated road evidence and a clean applicable regression baseline.
