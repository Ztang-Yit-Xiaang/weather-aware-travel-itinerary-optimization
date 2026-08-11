# Independent Evaluator and Metrics

## `PlanEvaluator`

1. **Why:** recompute whether the displayed child is independently eligible.
2. **Category:** service.
3. **Called by:** repair executors and pipeline adapters.
4. **Inputs:** child, reference parent, planner run, owned constraints,
   `RouteMatrix`, evaluator config/context requirements.
5. **Outputs:** `PlanEvaluationCertificate`.
6. **State ownership:** stateless evaluator; certificate persists in the run.
7. **Invariants:** does not trust solver feasibility; checks content hash,
   lineage, planner status/certification, constraints, route/timing/opening
   windows, lodging, context, duplicates, sequence, and utility units.
8. **Failure:** findings make comparison ineligible; warnings remain separate.
9. **Tests:** `tests/evaluation/test_plan_evaluator.py`.
10. **Gate:** E3.M.
11. **State:** current and verified.
12. **Read next:** `evaluation/certificate.py`.

## `PlanEvaluationCertificate`

- **Why/category:** immutable domain evidence for evaluation status and
  comparison eligibility.
- **Caller:** benchmark metrics, explanation builder, dashboard evaluation.
- **Inputs/outputs:** evaluator findings and metrics in; serialized certificate
  out.
- **State:** run-owned.
- **Invariant:** `plan_content_hash` must match the displayed plan.
- **Failure:** missing/stale/mismatched certificate is not success.
- **Tests/gate:** evaluator, pipeline, dashboard, benchmark tests; E3.M.

## Metric Ownership

| Metric family | Owner |
|---|---|
| Changed/unchanged days, weighted edit cost, booked/locked changes | `PlanDiff` |
| Route validity, hard eligibility, utility retained, contextual burden | Independent evaluator/certificate |
| Runtime, candidate counts, search complete, failure reason | Planner/benchmark execution |
| Method rank | Benchmark layer only after independent eligibility |

Null remains null. NaN/Inf is rejected. Lower-is-better bars must invert
direction correctly; zero is not a replacement for missing data.

See [independent evaluation](diagrams/independent_evaluation.md).

> **Beginner note / 初学者提示:** The planner proposes; the evaluator judges.
> Keeping them separate prevents a solver from grading its own output.

