# Pipeline and Immutable Artifacts

## Core Records

### `PipelineRunContext`

- **Why/category:** immutable runtime input record for an executor; domain model.
- **Caller:** `run_research_pipeline()`.
- **Inputs/outputs:** resolved run/config/snapshot/output metadata in; executor
  behavior out through `PipelineExecutionResult`.
- **State:** points at the run directory but does not own mutable global state.
- **Invariant/failure:** one context belongs to one run; executor failures are
  captured in failed run diagnostics.
- **Tests/gate/state:** pipeline tests; E1; current.
- **Read next:** `PipelineExecutionResult`.

### `PipelineExecutionResult`

- **Why/category:** typed handoff from execution to artifact writing; domain
  model.
- **Caller:** generation, progressive, heuristic, and exact executors.
- **Inputs/outputs:** planner/plan/diff/route/evaluation/explanation/request/
  metric/dashboard records in; canonical serialized files out.
- **State:** immutable in memory; run directory owns persistence.
- **Invariant/failure:** missing repair output becomes failed status; nonfinite
  or invalid evidence is handled by owning contracts.
- **Tests/gate/state:** `tests/test_pipeline_runner.py`; E1/E3; current.
- **Read next:** `_write_execution_artifacts()`.

### `PipelineRun`

- **Why/category:** return summary and durable run locator; domain model.
- **Caller:** CLI, benchmark adapter, interaction continuation.
- **Inputs/outputs:** completed run metadata in; `to_record()` summary out.
- **State:** immutable record referencing manifest/metrics paths.
- **Invariant/failure:** status reflects strict failures and missing repair
  output.
- **Tests/gate/state:** pipeline/benchmark/interaction tests; current.

## Write Order

1. create a non-existing run directory;
2. create canonical subdirectories;
3. resolve config with live APIs disabled when refresh is `never`;
4. write redacted config and dataset manifest;
5. call the injected executor;
6. write requests, planner runs, plans, diffs, routes, evaluations,
   explanations, dashboard records, and metrics;
7. write the manifest with run-relative paths;
8. return `PipelineRun` or raise strict failure with the written run attached.

See [immutable artifact flow](diagrams/immutable_artifact_flow.md).

## Hash and Lineage Rule

`PlanArtifactV2.content_hash` identifies plan content. A changed plan must
become a new child and receive a new independent evaluation. A stale
certificate is not repaired by changing a display flag.

> **Beginner note / 初学者提示:** “Immutable” does not mean the file system is
> magical. It means the program refuses overwrite and all later changes create
> new identified evidence.

## Known Limitation

The current manifest has the legacy `dashboard` category only. E3.UX plans an
additive `dashboard_product` category; it must not rename `dashboard`.

