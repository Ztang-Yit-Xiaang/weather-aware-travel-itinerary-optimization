# Parent, Child Plan, and Diff

## `PlanArtifactV2`

1. **Why:** canonical accepted or repaired itinerary artifact.
2. **Category:** immutable domain model.
3. **Called by:** repository, repair, evaluator, explanation, benchmark,
   pipeline, and interaction code.
4. **Inputs:** stable IDs, snapshots, selected stops, ordered days, lodging,
   routes, owned constraints, metrics, change/certificate metadata.
5. **Outputs:** serialized record and stable content hash.
6. **State ownership:** the run/`PlanRepository` owns serialized copies.
7. **Invariants:** parent ID and content hash match content; mutation invalidates
   certification.
8. **Failure:** hash/lineage/certificate mismatch fails validation/evaluation.
9. **Tests:** `tests/test_artifact_lineage.py`,
   `tests/plans/test_models.py`.
10. **Gate:** E1/E3.
11. **State:** current; v1 migration compatibility remains.
12. **Read next:** `plans/models.py`.

## `OwnedConstraint` and `OwnershipPolicy`

- **Why/category:** typed domain rules describing who controls a commitment and
  what relaxation is permitted.
- **Callers:** neighborhood/master/evaluator/interaction policy.
- **Inputs/outputs:** constraint target, owner/strength/status/provenance in;
  active/frozen/relaxable decisions out.
- **State:** stored inside the plan; no hidden permission state.
- **Invariants:** locked/booked changes require allowed relaxation and explicit
  permission where required; unconfirmed LLM constraints are inactive.
- **Failures/tests/gate:** invalid enum/policy or unsafe relaxation is rejected;
  plan, master-model, neighborhood, permission tests; E3/E5.
- **Lifecycle/read next:** current; read `repair/neighborhood.py`.

## `PlanRepository`

- **Why/category:** append-only repository/adapter for plan JSON.
- **Caller:** progressive repair and authorized interaction continuation.
- **Inputs/outputs:** `PlanArtifactV2` in; saved/loaded/verified plan out.
- **State:** owns its root and index.
- **Invariants:** identical save is idempotent; same ID with different content
  raises `PlanRepositoryConflict`.
- **Tests/gate/state:** `tests/plans/test_repository.py`; E3; current.
- **Read next:** `plans/diff.py`.

## `compute_plan_diff()` / `PlanDiffBuilder`

- **Why/category:** stateless typed comparison between compatible plans.
- **Callers:** progressive/exact/heuristic executors, pipeline, evaluator-facing
  metrics, explanation, interaction consequence.
- **Inputs:** parent, child, optional ownership policy.
- **Outputs:** added/deleted stops, day/time/order/lodging/road changes,
  unchanged days, and weighted edit cost.
- **State:** none.
- **Invariants:** catalog/context compatibility and correct parent/child IDs;
  preservation metrics are diff-owned.
- **Failure:** incompatible snapshots raise; missing diff is not equivalent to
  no change.
- **Tests/gate/state:** `tests/plans/test_diff.py`; E3.M; current.
- **Read next:** [repair methods](06_repair_methods_and_exact_baselines.md).

See [parent–child repair](diagrams/parent_child_repair.md).

> **Beginner note / 初学者提示:** “No diff file” and “zero changes” are different.
> Zero changes is a computed result; missing evidence is an error/unknown state.

