# Phase E3.1 Implementation Plan: Exact-Baseline Scalability and Completion

**Status:** `blocked`  
**Blocker:** D1-A versus D1-B research decision  
**Prepared:** 2026-07-22  
**Parent roadmap:** `docs/planning/research_pipeline_and_gate_map.md`  
**Near-term authority:** `docs/planning/current_execution_plan.md`

## Goal

Replace the current diagnostic exact-baseline refusal with a complete or
validly bounded execution path for `context_blind_solver` and
`full_reoptimization`, while preserving method identity, frozen-input parity,
failure retention, route validation, and independent evaluation.

This phase is complete only when all six real scenarios for both exact methods
either:

1. return a child plan from a completed declared finite universe; or
2. return a complete infeasibility result for that same declared universe.

`complete_candidate_limit_exceeded` is a safe diagnostic outcome, not an E3.1
completion outcome.

## Pipeline Context

- **Parent objective:** close the technical four-method comparison before E4
  robustness or E5 interaction work.
- **Current phase:** E3.1, `blocked` on the D1-A versus D1-B research
  decision.
- **Earlier evidence:** E1 and E2 are verified; E3.0 produced a real 24-row
  evidence-complete diagnostic.
- **Parallel phase:** E3.2 is `verified`; its non-exact road-closure and
  reduced-driving-tolerance failure policy must remain frozen.
- **Later dependent:** E3.3 reruns and closes the immutable 24-cell comparison.
- **Entry gate:** the v14 diagnostic, expanded route matrix, 50,000-candidate
  exact-method refusal records, and focused exact-baseline tests are
  reproducible.
- **Exit gate:** both exact methods complete or prove infeasibility over the
  approved common universe for all six frozen scenarios, without a cap-based
  optimality claim.

## Verified Current State

### Existing contracts

- `src/itinerary_system/repair/exact_baselines.py` exposes
  `plan_context_blind_solver()` and `plan_full_reoptimization()`.
- Both functions delegate to `_plan_exact_baseline()` and return immutable
  `ExactBaselinePlanningResult` records.
- `ExactBaselinePlanningResult` already records `search_complete`,
  `candidate_count`, `candidate_space_lower_bound`, `failure_reasons`, the
  `PlannerRun`, and any child/diff evidence.
- `_plan_exact_baseline()` preflights assignment/lodging combinations, then
  enumerates day permutations and evaluates each route sequence. It refuses
  certification when the declared cap is exceeded.
- `src/itinerary_system/benchmark/methods.py` binds the two public exact
  functions to the locked publication method IDs.
- `scripts/run_e3_publication_benchmark.py` rejects an existing output root,
  freezes inputs, runs the four-method suite, and emits closeout evidence.

### Existing verification

- `tests/repair/test_exact_baselines.py` covers distinct context semantics,
  complete small-universe search, and cap refusal.
- `tests/repair/test_exact_baseline_pipeline.py` covers exact output through the
  package pipeline and independent route certificate.
- `tests/benchmark/test_publication_method_factory.py` exercises all four
  adapters together.
- `docs/current/e3_publication_benchmark_contract.md` records the v14 real
  diagnostic: 12 exact rows refused under
  `max_complete_candidates=50000`.
- `docs/current/e3_exact_baseline_test_strategy_and_audit.md` records that the
  safe local pruning test passes but the real E3.1 gate remains blocked. The
  v14 raw candidate-space lower bounds range from 134,217,727 to
  1,073,741,823.

### Known performance boundary

The product-of-factorials preflight prevents materializing a candidate universe
that already exceeds the cap. That fix is necessary but insufficient: prior
real attempts show that model construction and candidate evaluation are also
costly. Raising the cap without a stage profile is therefore prohibited.

## Scope

1. Measure exact-baseline time and candidate growth for one frozen scenario and
   one method before changing the search strategy.
2. Record separate costs for model construction, assignment expansion, route
   sequence evaluation, feasibility checking, scoring, and artifact creation.
3. Decide between:
   - **D1-A:** implement a complete solver/search strategy for the existing
     declared universe; or
   - **D1-B:** formally narrow the finite universe using a method-independent,
     frozen rule, then rerun all four methods on that common universe.
4. Implement and test the approved branch behind the existing public exact
   method functions.
5. Prove completion or infeasibility for the 12 real exact
   scenario-method cells.
6. Preserve every refusal, exception, timeout, and ineligible result as
   evidence.
7. Hand a stable exact strategy and its evidence to E3.3.

## Non-Goals

- Do not alter the progressive repair or deterministic heuristic objectives.
- Do not relabel an approximate, interrupted, capped, or partial search as
  exact.
- Do not weaken route/snap validation, independent evaluation, or publication
  ranking rules.
- Do not use interaction, LLM, memory, participant, or preference-learning
  components.
- Do not run the full E4 sensitivity or ablation matrix in this phase.
- Do not choose D1-B merely because it is faster; the narrowed universe must be
  frozen, common to all methods, and compatible with the research question.

## Dependencies and Prerequisites

| Dependency | How it is satisfied or detected |
|---|---|
| Frozen parent, scenarios, catalog, context, and route matrix | Reuse the v14 input lineage recorded in `docs/current/e3_publication_benchmark_contract.md` and its immutable `inputs/manifest.json`; do not rewrite v14. |
| Exact method semantics | Preserve the method IDs and context modes in `src/itinerary_system/repair/exact_baselines.py` and `src/itinerary_system/benchmark/methods.py`. |
| Independent evaluator | Continue through the existing pipeline adapter and evaluator-owned `benchmark_ranking_eligible` gate. |
| Compute envelope | Record wall time, peak memory if available from the existing environment, candidate counts, and termination reason before approving a matrix run. |
| D1 research decision | Resolve after the one-scenario profile. Record the approved universe, exactness argument, tradeoff, and reviewer in `docs/current/e3_exact_baseline_strategy_decision.md`. |
| E3.2 handoff | Satisfied: E3.2 is `verified`. Preserve its retained physical-cause tokens and unranked failure rows in E3.3. |

## Architecture and Required Changes

### Exact-baseline module

```text
+-------------------------------------------------------------------------------+
|                         exact_baselines module                                |
+-------------------------------------------------------------------------------+
|  Category: Stateless Utility Module                                           |
+-------------------------------------------------------------------------------+
|  - No persistent internal state                                               |
|  - Inputs: PlanArtifactV2, request, RouteMatrix, DayRouteSolverConfig          |
|  - Output: ExactBaselinePlanningResult                                         |
+-------------------------------------------------------------------------------+
|  + plan_context_blind_solver(...): ExactBaselinePlanningResult                 |
|      --> Complete static-context exact baseline or retain a truthful failure   |
|  + plan_full_reoptimization(...): ExactBaselinePlanningResult                  |
|      --> Complete context-aware full replanning or retain a truthful failure   |
+-------------------------------------------------------------------------------+
```

Rules:

- Keep both public signatures and method IDs stable unless the decision record
  proves a compatibility change is required.
- Keep search state local to one method/scenario call.
- Keep `search_complete` false for any cap, interruption, timeout without a
  valid bound, or unhandled exception.
- Set `OPTIMALITY_CERTIFIED` only when the approved universe has been completely
  solved and a feasible optimum exists.
- Represent complete infeasibility separately from cap refusal.
- Continue generating the child plan and `PlanDiff` only after exact completion.

### Publication benchmark boundary

```text
+-------------------------------------------------------------------------------+
|                    run_e3_publication_benchmark.py                             |
+-------------------------------------------------------------------------------+
|  Category: Major Integration Boundary                                         |
+-------------------------------------------------------------------------------+
|  - Frozen input owner: copied input manifests under a new immutable run root   |
|  - Runtime owner: one bounded benchmark invocation                             |
|  - Final output: closeout plus benchmark/pipeline artifacts                    |
+-------------------------------------------------------------------------------+
|  + parse_args(): argparse.Namespace                                            |
|      --> Validate required frozen inputs and exact-search controls             |
|  + main(): int                                                                 |
|      --> Preflight, run, retain failures, and emit deterministic closeout       |
+-------------------------------------------------------------------------------+
```

Rules:

- Do not mutate the v14 diagnostic.
- A profiling or partial root is not publication output.
- E3.3 must use a new immutable output root.
- Before E3.3, add or verify a bounded per-cell execution/checkpoint design so
  an interrupted long run can resume without rewriting a completed immutable
  cell or silently changing frozen inputs.
- Final closeout must reject duplicate scenario-method cells, mixed route
  matrices, mixed source bundles, or changed universe definitions.

## Data Contracts

### `ExactBaselinePlanningResult`

Preserve the existing schema fields:

| Field | Required meaning |
|---|---|
| `method_id` | Locked requested/executed exact method identity. |
| `status` | Completed only when a child exists and the search is complete; otherwise failed. |
| `planner_run` | Requested method, executed method, backend, raw status, certificate, and result plan linkage. |
| `search_complete` | True only when the approved universe is exhausted or validly solved/bounded. |
| `candidate_count` | Number of fully evaluated candidates or the strategy-equivalent explored count. |
| `candidate_space_lower_bound` | Existing conservative preflight evidence; do not reinterpret it as the exact universe size. |
| `failure_reasons` | Stable machine-readable cap, infeasibility, timeout, interruption, or exception reasons. |
| `child_plan` and `diff_record` | Present only for a completed feasible exact result. |

If the selected strategy needs new diagnostic fields, add them through a new
schema version and retain backward-readable v1 fields. At minimum, the new
record must distinguish:

- complete feasible optimum;
- complete infeasibility;
- incomplete due to candidate cap;
- incomplete due to time or interruption;
- invalid due to route/input/contract failure.

### Strategy decision record

`docs/current/e3_exact_baseline_strategy_decision.md` must record:

- v14 input hashes and scenario/method profiled;
- candidate-space estimate and observed counts;
- elapsed time by stage;
- peak memory when measurable;
- D1-A or D1-B decision;
- exactness or common-universe argument;
- rejected alternative and reason;
- compute cap and cancellation behavior;
- compatibility effects on all four methods;
- reviewer/approval boundary;
- commands and artifact paths used as evidence.

## Storage and State

The exact planning functions remain stateless. They receive frozen inputs,
return one immutable result, and do not persist data between calls.

Profiling data and completed method/scenario cells are research artifacts:

- write each record to a run-specific staging location;
- include input and configuration hashes;
- write a cell atomically before marking it complete;
- never reuse a completed cell with different hashes;
- preserve interrupted or failed cells with their termination reason;
- create the final immutable E3.3 root only from a validated complete staging
  set.

No credentials, provider secrets, or hidden model reasoning belong in these
records.

## Implementation Workflow

### Step 1 - Reproduce and profile one exact cell

1. Select one v14 scenario and `context_blind_solver`.
2. Reuse the exact frozen parent, candidate catalog, context, and route matrix.
3. Capture model-build, assignment-expansion, route-evaluation, scoring, and
   artifact-finalization time separately.
4. Record `candidate_space_lower_bound`, evaluated candidate count, termination
   reason, and memory observations.
5. Repeat with `full_reoptimization` on the same scenario only if the first
   profile finishes within the approved diagnostic envelope.
6. Do not start the 12-cell exact matrix at this step.

### Gate D1 - Approve the exact universe and strategy

Choose D1-A only if a complete strategy can solve the existing universe with a
defensible completeness/optimality argument and bounded compute.

Choose D1-B only if the common universe can be defined before method execution
using frozen, method-independent rules. The rule must apply to all four methods,
retain the ownership and context contrasts required by the research question,
and be versioned in every run record.

If neither branch is defensible, mark E3.1 `blocked` and narrow the research
question explicitly; do not present the v14 diagnostic as a completed
comparison.

### Step 2 - Implement the approved branch

1. Keep the two public exact methods stable.
2. Isolate the new search/universe logic from child-plan extraction,
   `PlanDiff`, and `PlannerRun` construction.
3. Fail closed on route gaps, hash changes, invalid universe metadata,
   cancellation, or incomplete bounds.
4. Emit stable completion and infeasibility evidence.
5. Retain the current cap-refusal path as a safety control and regression case.

### Step 3 - Verify small and adversarial universes

1. Preserve existing exact small-universe expected optima.
2. Add complete-infeasibility coverage.
3. Add cancellation/time-limit coverage that cannot certify optimality.
4. Add universe/hash mismatch coverage.
5. Cross-check the selected strategy against enumeration on tiny fixtures.
6. Confirm context-blind and full-reoptimization objectives remain distinct.

### Step 4 - Scale one scenario, then the exact matrix

1. Complete both exact methods on one real scenario.
2. Review time, memory, result semantics, route evidence, and independent
   certificates.
3. Run the remaining five scenarios only after the single-scenario gate passes.
4. Retain all complete infeasibility and other failed rows.
5. Produce an E3.1 closeout that lists 12 exact cells and their completion class.

### Step 5 - Hand off to E3.3

1. Freeze the approved strategy/universe identifier and configuration.
2. Update the E3 contract and current execution plan with evidence, not intent.
3. Preserve the already verified E3.2 failure-policy contract.
4. Leave E3.3 blocked until E3.1 is verified.
5. Do not mark E4 ready during this phase.

## Files to Create

| Path | Responsibility |
|---|---|
| `scripts/profile_e3_exact_baseline.py` | Reproduce one frozen exact method/scenario cell and emit bounded stage-level profile evidence without creating a publication closeout. |
| `docs/current/e3_exact_baseline_strategy_decision.md` | Durable D1 decision, universe definition, compute boundary, rejected alternative, and evidence links. |

Create a separate benchmark execution/checkpoint module only if the approved
strategy requires it. Its location must follow the existing
`src/itinerary_system/benchmark/` boundary and be named in the D1 record before
implementation.

## Files to Modify

| Path | Planned change |
|---|---|
| `src/itinerary_system/repair/exact_baselines.py` | Add the approved complete strategy or common-universe boundary while preserving truthful result/certificate behavior. |
| `src/itinerary_system/benchmark/methods.py` | Pass only versioned exact-strategy controls through the existing method factory; keep locked method IDs. |
| `scripts/run_e3_publication_benchmark.py` | Freeze strategy/universe metadata and support bounded recoverable execution before the E3.3 rerun. |
| `tests/repair/test_exact_baselines.py` | Add completeness, infeasibility, interruption, and tiny-fixture cross-check cases. |
| `tests/repair/test_exact_baseline_pipeline.py` | Verify child/diff/certificate behavior for completed exact results and no certificate for incomplete results. |
| `tests/benchmark/test_publication_method_factory.py` | Verify the four-method factory preserves strategy metadata and method identity. |
| `tests/benchmark/test_publication_contract.py` | Reject mixed universe/strategy lineage if the result schema adds those fields. |
| `docs/current/e3_publication_benchmark_contract.md` | Record E3.1 evidence after verification; retain the v14 interpretation boundary. |
| `docs/planning/current_execution_plan.md` | Move E3.1 only when its exit evidence exists and update the immediate queue. |
| `docs/current/current_problem_manifest.md` | Update CP-007 only after the exact matrix evidence exists. |

## Error Handling and Recovery

- **Candidate cap:** return an incomplete failed result with no optimality
  certificate.
- **Time limit or cancellation:** preserve partial diagnostics, mark
  `search_complete=false`, and emit no optimum claim unless the selected solver
  provides a valid exact bound accepted by D1.
- **Complete infeasibility:** record a complete termination distinct from cap,
  timeout, or exception.
- **Route or input mismatch:** fail before method execution when possible;
  otherwise retain the failed cell and block closeout.
- **Interrupted multi-cell run:** reuse only hash-matching, atomically completed
  staging cells; never edit a finalized immutable output root.
- **Artifact-write failure:** keep the staging root nonfinal and do not emit a
  publication-ready closeout.
- **Unexpected exception:** retain type and safe summary without credentials;
  no child plan or certificate.
- **Rollback:** the v14 diagnostic remains immutable. Revert the new strategy
  configuration or code without rewriting v14 evidence.

## Observability

Reuse the repository's immutable manifests, `PlannerRun`, benchmark rows,
closeout findings, and structured failure reasons. E3.1 evidence must expose:

- input, route matrix, source bundle, strategy, and universe hashes;
- model-build and candidate-evaluation timing;
- candidate lower bound, explored/evaluated count, and termination class;
- requested and executed method identity;
- complete feasible, complete infeasible, or incomplete status;
- route certificate and independent ranking eligibility;
- artifact paths sufficient to reproduce the result.

The profile script is diagnostic only. Its output cannot set
`publication_ready`.

## Testing and Verification

Run targeted checks while implementing:

```powershell
python -m pytest tests/repair/test_exact_baselines.py
python -m pytest tests/repair/test_exact_baseline_pipeline.py
python -m pytest tests/benchmark/test_publication_method_factory.py tests/benchmark/test_publication_contract.py
```

Run repository checks before marking the code implemented:

```powershell
python -m ruff check --no-cache src tests scripts
python -m pytest
python scripts/run_project_checks.py
```

Required real evidence before `verified`:

1. a stored one-scenario profile for each exact method;
2. an approved D1 decision record;
3. tiny-fixture equivalence against enumeration;
4. 12 real exact cells with no `complete_candidate_limit_exceeded` outcome;
5. complete-search or complete-infeasibility status for each exact cell;
6. unchanged method IDs and frozen route/input lineage;
7. independent certificates for every returned child plan;
8. an E3.1 closeout read back after the process exits.

## Pipeline and Roadmap Update

Planning this phase does not change E3.1 from `blocked` on D1, E3.3 from
`blocked`, E4 from `blocked`, or E5 from `deferred`.

After D1 is approved, record the chosen branch and move E3.1 to `in-progress`.
After all code-complete criteria pass, E3.1 may move to `implemented`. Move it
to `verified` only after the real 12-cell exact evidence and regression checks
above pass. Because E3.2 is already verified, E3.3 may then move from
`blocked` to `ready`.

## Acceptance Criteria

### Code-complete (`implemented`)

- A reviewed D1 strategy/universe decision exists.
- Both public exact methods use the approved path without changing their
  research semantics or method IDs.
- Result records distinguish complete feasible, complete infeasible, capped,
  timed-out/interrupted, and invalid outcomes.
- The safety cap still refuses false optimality.
- Bounded recoverable execution cannot mix hashes or rewrite finalized output.
- Targeted and regression tests for the new behavior are present.

### Verification-complete (`verified`)

- Ruff, full pytest, and project checks pass.
- The approved strategy matches exhaustive enumeration on tiny fixtures.
- Both exact methods complete all six frozen real scenarios or prove complete
  infeasibility within the approved universe.
- None of the 12 exact cells relies on a diagnostic candidate-cap refusal.
- Every returned child plan has valid lineage, `PlanDiff`, route evidence, and
  independent evaluation.
- The E3.1 closeout and D1 record make no claim beyond the observed universe.
- The v14 diagnostic remains unchanged and available for comparison.
