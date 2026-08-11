# E3 Method Semantics

**Status:** Locked implementation contract  
**Updated:** 2026-07-13  
**Applies to:** The first paired publication benchmark

All methods consume the same frozen parent plan, disruption record, candidate catalog, context snapshot, and route-matrix hash. They may differ only in the information and decision rule explicitly assigned below. Every output is evaluated afterward by the same context-aware independent evaluator.

## `context_blind_solver`

- Uses the same solver-grade feasibility model for static time, budget, opening-window, lodging, and route constraints.
- Does not consume the disrupted weather, closure, mobility, or user-context fields while planning.
- Does not optimize ownership or preservation cost.
- Produces a new plan from the full editable decision space.
- Is still scored by the common context-aware evaluator, so blindness can reduce eligibility rather than being hidden.

Acceptance evidence must show that context fields were excluded from planner inputs and that planner provenance uses the canonical method ID.

## `deterministic_context_aware_heuristic`

- Consumes the same frozen disruption/context and validated route matrix as the proposed method.
- Applies a documented one-pass deterministic repair rule; it does not call a mathematical optimizer.
- Uses stable tie-breaking by feasibility, contextual burden, utility, added travel, then entity ID.
- May edit only the disruption-affected day(s) unless no feasible local action exists, in which case it emits a failed run rather than silently becoming full reoptimization.

Acceptance evidence must include the ordered candidate trace and prove that repeated runs on the same hashes return the same child-plan hash.

## `progressive_sequential_lexicographic_repair`

- Uses the implemented ownership-aware repair model.
- Tries configured neighborhoods from smallest to largest and stops at the first independently eligible child plan.
- Orders objectives as hard/booked commitment preservation, ownership-weighted change, contextual burden, then utility.
- Keeps full-radius repair only as the last fallback inside this method; that fallback remains part of progressive repair and is not the full-reoptimization baseline.

The pipeline executor now emits the canonical method ID in its planner provenance.

## `full_reoptimization`

- Consumes the same frozen disruption/context and validated route matrix.
- Opens the full decision space from the first solve.
- Optimizes feasibility, contextual burden, utility, travel, and cost without an ownership or parent-preservation objective.
- Computes preservation metrics only after solving; preservation must not influence selection or tie-breaking.

This is implemented as a separate finite-universe exact-search executor, not as `RepairRadius.FULL_REOPTIMIZATION` inside the preservation-first controller. It certifies optimality only after complete enumeration and refuses the claim when the declared candidate cap is exceeded.

## Shared Publication Gates

- Exact method ID and method-specific planner provenance.
- Identical frozen input hashes and no parent/scenario mutation.
- Identical nonempty E2 source-bundle lineage and content-addressed route-matrix hashes across all four methods.
- Complete provider-backed road and endpoint-snap validation with no geodesic fallback.
- Independent evaluation certificate and evaluator-owned quality metrics.
- Failed runs retained with diagnosis.
- No ranked row unless method provenance, evaluator eligibility, route validation, and numeric evaluator quality evidence all pass.

