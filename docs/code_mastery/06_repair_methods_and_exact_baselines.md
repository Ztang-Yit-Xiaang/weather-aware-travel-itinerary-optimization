# Repair Methods and Exact Baselines

## Progressive Repair Modules

### `RepairNeighborhoodBuilder` / `build_repair_neighborhood()`

- **Why/category:** stateless service defining what may change at each radius.
- **Caller:** `ProgressiveRepairController`.
- **Inputs/outputs:** parent + request + radius in; editable/frozen days,
  lodging, route boundaries, and constraints out.
- **State/invariants:** no persistent state; unaffected content stays frozen;
  full reoptimization is last.
- **Failures/tests/gate:** unsafe booked/locked relaxation is excluded;
  `tests/repair/test_neighborhood.py`; E3.

### `RepairModel` in `master_model.py`

- **Why/category:** service/domain model that compiles selections, day/lodging
  assignments, change variables, owned constraints, and context overlays.
- **Caller:** day-route and lexicographic repair execution.
- **Inputs/outputs:** neighborhood, parent, candidates, route/context evidence
  in; candidate solution and objective components out.
- **State:** per repair attempt.
- **Invariants:** locked content cannot silently change; output child retains
  lineage.
- **Failures/tests:** infeasible/unsafe solution; `test_master_model.py`.

### `SequentialLexicographicSolver`

- **Why/category:** service enforcing stage priority instead of one weighted
  sum.
- **Caller:** progressive attempt.
- **Inputs/outputs:** candidate solutions/objective stages in; selected solution
  plus stage trace out.
- **State:** one solve.
- **Invariant:** later stages cannot degrade earlier stages beyond tolerance.
- **Tests/gate:** `test_lexicographic.py`; E3.

### `DayRouteSolver`

- **Why/category:** RouteMatrix-backed candidate generator/service for affected
  days.
- **Caller:** repair and exact baselines.
- **Inputs/outputs:** selected stops, anchors, time/opening constraints, matrix
  in; route candidate/solution out.
- **Invariant:** publication mode requires strict matrix evidence.
- **Tests:** `test_day_route_solver.py`.

### `ProgressiveRepairController`

- **Why/category:** orchestrator that tries radii in order.
- **Caller:** pipeline progressive executor and interaction continuation.
- **Inputs/outputs:** parent/request/repository/evaluator/candidate generator in;
  `RepairOutcome` with attempts, child, diff, evaluation, or diagnosis out.
- **State:** attempt-local; accepted child stored append-only.
- **Invariant:** stop at the smallest independently eligible radius.
- **Failure:** retain every failed attempt and return a diagnosis, not fake
  success.
- **Tests/gate/state:** `test_progressive.py`; E3; current.

## Four Publication Methods

| Method ID | Information boundary | Completion meaning |
|---|---|---|
| `context_blind_solver` | Exact declared universe without contextual repair information | Complete search or complete infeasibility; independent evaluator may still reject |
| `deterministic_context_aware_heuristic` | Context-aware deterministic heuristic | Candidate result, not global optimality |
| `progressive_sequential_lexicographic_repair` | Context + ownership + smallest-radius progression | Smallest independently eligible configured radius |
| `full_reoptimization` | Exact context-aware declared universe without preservation objective | Complete search or complete infeasibility |

## `exact_baselines.py`

- **Why/category:** exact finite-universe baseline service.
- **Caller:** publication adapters and pipeline executors.
- **Inputs/outputs:** parent/request/model/matrix/candidate cap in;
  `ExactBaselinePlanningResult` out.
- **State:** no cross-run state.
- **Invariant:** raw-space lower-bound preflight occurs before enumeration;
  incomplete/capped search has `search_complete=false`, no child, and no
  optimality certificate.
- **Failure:** `complete_candidate_limit_exceeded:<cap>` is diagnostic refusal.
- **Tests:** exact baseline, scalability gate, pipeline tests.
- **Gate/state:** implemented; E3.1 blocked.

See [four-method flow](diagrams/four_method_benchmark_flow.md) and
[D1 refusal](diagrams/exact_baseline_refusal_and_d1.md).

> **Beginner note / 初学者提示:** “Exact method” describes the intended search
> contract. If the method stops at a safety cap, that particular run is not an
> exact completion.

