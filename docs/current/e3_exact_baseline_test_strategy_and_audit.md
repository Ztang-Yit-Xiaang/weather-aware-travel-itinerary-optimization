# E3.1 Exact-Baseline Test Strategy and Gate Audit

**Date:** 2026-07-26  
**Phase:** E3.1 exact-baseline scalability and completion  
**Gate result:** Not passed

## Test Strategy

Use a risk-based, contract-first sequence:

1. **Exact-search unit contracts**
   - completed finite search may certify optimality;
   - cap refusal must set `search_complete=false`, return no child, and emit no
     optimality certificate;
   - complete infeasibility must remain distinct from incomplete search.
2. **Pruning-equivalence characterization**
   - infeasible assignments may be pruned before route evaluation;
   - pruning must not change the complete-search result;
   - a raw candidate-space cap must still refuse before any unproved pruning is
     treated as completion.
3. **Pipeline integration**
   - completed exact output retains parent/child lineage, `PlanDiff`, route
     evidence, and independent evaluation;
   - context-blind output may be independently ineligible even when its own
     exact objective completes.
4. **Publication contracts**
   - all four method identities must be present;
   - failed and ineligible rows remain visible;
   - only independently eligible rows enter rankings;
   - route matrix and source-bundle lineage must be shared.
5. **Regression**
   - run focused Ruff;
   - run the full test suite and project checks only after the focused gates
     pass and before moving the phase to `verified`.

## Specific Test Added

`tests/repair/test_exact_baseline_scalability_gate.py` adds:

`test_budget_pruning_is_exact_but_does_not_relax_raw_space_preflight`

The fixture has three selectable stops, each with positive cost, under a zero
budget:

- with `max_complete_candidates=1`, the raw assignment-space lower bound is 7,
  so the exact baseline refuses before pruning with
  `complete_candidate_limit_exceeded:1`;
- with `max_complete_candidates=10`, every nonempty assignment is proven
  budget-infeasible before route evaluation, the search completes with zero
  evaluated candidates, and the result reports
  `no_feasible_complete_candidate` with no optimality certificate.

This confirms that budget pruning is a safe local optimization. It also
demonstrates that the current raw-space preflight occurs before pruning and
therefore cannot solve the real E3 scalability problem.

## Commands and Results

```powershell
python -m pytest tests/repair/test_exact_baseline_scalability_gate.py -q
# 1 passed

python -m pytest tests/repair/test_exact_baselines.py tests/repair/test_exact_baseline_scalability_gate.py tests/repair/test_exact_baseline_pipeline.py -q
# 6 passed

python -m pytest tests/benchmark/test_publication_method_factory.py tests/benchmark/test_publication_contract.py -q
# 8 passed

python -m ruff check --no-cache src/itinerary_system/repair/exact_baselines.py tests/repair/test_exact_baselines.py tests/repair/test_exact_baseline_scalability_gate.py tests/repair/test_exact_baseline_pipeline.py
# All checks passed
```

## Real v14 Evidence

Inspected immutable output:

`tmp_test/research_pipeline_raw/e3-real-production-20260725-optimized-v14-cap50000`

The closeout reports:

- 24 total rows;
- 8 independently ranking-eligible rows;
- 16 retained failed rows;
- `publication_ready: true` for evidence completeness;
- `all_runs_ranking_eligible: false`.

All 12 exact method/scenario cells have:

- `execution_status: FAILED`;
- `solver_status_raw: candidate_limit_exceeded`;
- `solver_certification: NO_CERTIFICATE`;
- `complete_candidate_limit_exceeded:50000`.

Their recorded raw candidate-space lower bounds range from 134,217,727 to
1,073,741,823.

## Standards Finding

The v14 run is a valid evidence-complete diagnostic, but it does not satisfy the
E3.1 exit gate in
`docs/planning/e3_exact_baseline_scalability_phase_plan.md`.

The labels `[PASSED] Profile and optimize exact baselines`,
`[PASSED] Rerun immutable 24-row benchmark suite`, and “proceed to E4” in
`docs/planning/current_execution_plan.md` overstate the phase status:

- pruning was implemented and tested;
- the diagnostic rerun completed;
- the exact baseline gate did not pass;
- E3.3 remains blocked;
- E4 remains blocked.

## Required Next Gate

Keep E3.1 `in-progress` or `blocked on D1`, not `verified`.

The next accepted evidence must come from one of the phase plan's two decisions:

1. **D1-A:** a complete solver/search strategy for the existing declared
   universe; or
2. **D1-B:** a formally approved, method-independent narrowed universe used by
   all four methods.

Do not rerun the 24-cell matrix again until a single real exact cell completes
or proves complete infeasibility without candidate-cap refusal.
