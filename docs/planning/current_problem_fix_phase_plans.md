# Current Problem Fix Phase Plans

Generated from `docs/current/current_score_audit.md`, `docs/planning/research_stabilization_and_publication_plan.md`, `docs/planning/travel_itinerary_repair_technical_specification.md`, `docs/reference/data_dictionary.md`, and source/test inspection on 2026-07-06.

Validation snapshot:

- `python -m ruff check src tests scripts`: passed.
- `python -m pytest`: reached 77 passed and 4 failed, with the failures in `tests/data/test_context_snapshot.py` caused by `PermissionError` while creating temporary directories under `C:\Users\1\AppData\Local\Temp`; pytest cache writes also warned on `.pytest_cache`.
- Current implementation has a Phase 0 artifact scaffold in `src/itinerary_system/research_artifacts.py`, road-validation contracts in `src/itinerary_system/routing/models.py`, and a deterministic repair scaffold in `src/itinerary_system/repair_planner.py`.
- Current implementation still has publication blockers: post-solve route mutation risk in `src/itinerary_system/experiment_runner.py`, geodesic travel inside solver paths, no parent-plan-aware repair model, no generalized plan diff, no complete final-plan evaluator, and source-coverage/utility semantics that still keep legacy `data_confidence` aliases.

These plans are planning artifacts only. Do not tick roadmap or pipeline checkboxes until the implementation and validation for that phase are complete.

# Phase 0.0 Implementation Plan: Validation Harness And Problem Manifest

## Goal

Implement **validation harness and current problem manifest**.

This feature should allow the system to:

1. Run lint and tests with a repository-controlled temporary directory on Windows.
2. Classify validation failures as environment, test-fixture, or product-code failures.
3. Produce a compact current-problem manifest that links each fix phase to real files and observed blockers.

Keep the implementation modular, easy to test, and consistent with the existing project structure.

## Current State

The project already has:

* `pyproject.toml`: configures ruff and pytest with `testpaths = ["tests"]`.
* `docs/reference/code_quality_workflow.md`: documents lint, tests, coverage, and artifact policy.
* `tests/data/test_context_snapshot.py`: uses `tempfile.TemporaryDirectory()` and copies data snapshots into a temporary root.
* `.codex_tmp_pytest/`: an existing workspace-local temporary directory, but Python still resolved `tempfile.gettempdir()` to `C:\Users\1\AppData\Local\Temp` during inspection.

The missing part is:

* A stable test temp/cache policy that works in the managed Windows workspace.
* A scripted check runner that records real validation outcomes without requiring generated artifacts to be committed.
* A current problem manifest that separates architecture blockers from validation-environment blockers.

## New Components to Add

### Component 1

`WorkspaceTempConfigurator`

Responsibility:

Configure pytest and Python temporary-directory behavior to use a writable workspace temp root during tests and local check runs.

### Component 2

`ProjectCheckRunner`

Responsibility:

Run ruff, focused pytest, and full pytest with stable environment variables, then write a concise JSON/Markdown check summary under an ignored quality directory.

### Component 3

`CurrentProblemManifest`

Responsibility:

Record current obvious and latent problems with phase ownership, evidence file paths, severity, and acceptance checks.

## Class / Registry Diagrams

```text
+-------------------------------------------------------------------------------+
|                         WorkspaceTempConfigurator                              |
+-------------------------------------------------------------------------------+
|  - workspace_root: Path                                                        |
|  - temp_root: Path                                                             |
|  - cache_root: Path                                                            |
+-------------------------------------------------------------------------------+
|  + configure(): dict[str, str]              --> Returns env overrides          |
|  + ensure_writable(): None                  --> Creates and probes temp roots  |
|  + pytest_addopts(): str                    --> Produces cache-safe addopts    |
+-------------------------------------------------------------------------------+
```

```text
+-------------------------------------------------------------------------------+
|                              ProjectCheckRunner                                 |
+-------------------------------------------------------------------------------+
|  - workspace_root: Path                                                        |
|  - temp_config: WorkspaceTempConfigurator                                      |
+-------------------------------------------------------------------------------+
|  + run_ruff(): CheckResult                  --> Runs ruff check                |
|  + run_pytest(targets): CheckResult         --> Runs pytest target list        |
|  + write_summary(results): Path             --> Writes ignored check report    |
+-------------------------------------------------------------------------------+
```

```text
+-------------------------------------------------------------------------------+
|                         CurrentProblemManifest                                  |
+-------------------------------------------------------------------------------+
|  - No persistent internal state beyond emitted manifest                         |
+-------------------------------------------------------------------------------+
|  + load_observed_findings(): tuple[ProblemRecord, ...] --> Reads known issues   |
|  + write_markdown(records): Path            --> Writes docs/current manifest    |
|  + write_json(records): Path                --> Writes machine-readable output  |
+-------------------------------------------------------------------------------+
```

## Class Diagram Rules

1. Create one diagram block for every important utility or script module.
2. Keep class names specific to validation and problem-manifest work.
3. List only durable fields such as paths and environment settings.
4. List public methods Codex should implement.
5. Mark temp configuration and check running as utility/service components.
6. If a class does not need persistent state, explicitly say `No persistent internal state`.
7. Skip `CurrentProblemManifest` JSON output if the Markdown manifest is enough for this repository.

## Data Model

```python
type CheckResult = {
    "name": str,
    "command": list[str],
    "exit_code": int,
    "duration_seconds": float,
    "passed": bool,
    "stdout_excerpt": str,
    "stderr_excerpt": str,
    "failure_class": str,  # environment | test_fixture | product_code | timeout | unknown
}

type ProblemRecord = {
    "problem_id": str,
    "title": str,
    "severity": str,  # blocker | high | medium | low
    "evidence_paths": list[str],
    "phase": str,
    "observed_status": str,
    "acceptance_check": str,
}
```

## Storage / State

Persistent state:

* `docs/current/current_problem_manifest.md` for human-readable problem tracking.
* Optional ignored generated output under `results/quality/project_check_summary.json`.

Temporary state:

* `.codex_tmp_pytest/` for local test temp directories.
* No committed pytest cache artifacts.

## Required Methods

```python
def configure_workspace_temp(workspace_root: Path) -> dict[str, str]
def ensure_temp_root_writable(temp_root: Path) -> None
def run_project_check(command: list[str], env: Mapping[str, str]) -> CheckResult
def classify_validation_failure(result: CheckResult) -> str
def write_current_problem_manifest(records: tuple[ProblemRecord, ...], path: Path) -> Path
```

## Validation Rules

1. Temp root must be inside the repository or another approved writable root.
2. Temp root probe must create and delete a file and directory.
3. Check summary must not store secrets or full massive logs.
4. Failed pytest runs must preserve enough excerpt text to identify failing tests.
5. Manifest records must include evidence paths that exist when the manifest is generated.
6. Generated quality outputs must remain ignored or out of commit scope.

## UI / API Integration

Internal only:

* `scripts/run_project_checks.py` should be callable from PowerShell and CI.
* `tests/conftest.py` can set `tempfile.tempdir` early if Windows temp resolution remains outside writable roots.
* No dashboard or user-facing UI changes are required.

## Workflow

1. Create/validate workspace temp and pytest cache roots.
2. Run ruff.
3. Run focused context snapshot tests.
4. Run full pytest.
5. Classify failures.
6. Write a compact quality summary.
7. Update `docs/current/current_problem_manifest.md` only when problem ownership changes.

## Files to Create

* `tests/conftest.py`
* `scripts/run_project_checks.py`
* `docs/current/current_problem_manifest.md`

## Files to Modify

* `pyproject.toml` only if pytest cache/temp behavior needs explicit repository configuration.
* `.gitignore` only if new quality or temp outputs are not already ignored.
* `docs/reference/code_quality_workflow.md` to document the new check runner.

## Error Handling

* If the workspace temp root is not writable, fail with a clear setup error before running tests.
* If Python still chooses the user temp directory, set `tempfile.tempdir` in `tests/conftest.py`.
* If ruff passes but pytest fails, classify the failure by traceback and failing test names.
* If checks time out, write `failure_class="timeout"` and preserve the collected test count.
* If the manifest cannot be written, fail the script instead of silently skipping evidence.

## Testing Checklist

- [ ] `tests/conftest.py` sets temp behavior before tests use `tempfile.TemporaryDirectory()`.
- [ ] `python -m pytest tests/data/test_context_snapshot.py` passes without writing to `C:\Users\1\AppData\Local\Temp`.
- [ ] `python -m pytest` passes or reports product-code failures rather than temp permission failures.
- [ ] `python -m ruff check src tests scripts` still passes.
- [ ] Check summaries are written only under ignored quality/temp paths.
- [ ] `docs/current/current_problem_manifest.md` links every blocker to a phase plan.

## Roadmap / Full Pipeline Update

Do not mark any research roadmap item complete for this planning step. After implementation and passing validation, update `docs/reference/code_quality_workflow.md` and optionally add a short note to `docs/current/README.md`.

## Acceptance Criteria

1. The test suite uses a writable temp/cache location in this Windows workspace.
2. Full validation can distinguish true code failures from environment permission failures.
3. A current problem manifest exists and points to the phase plans below.
4. The generated quality artifacts are not committed accidentally.
5. Existing ruff and pytest behavior remains compatible with normal developer machines.

# Phase 0.1 Implementation Plan: Truthful Artifact Lineage And Post-Solve Mutation Gate

## Goal

Implement **truthful artifact lineage and post-solve mutation gate**.

This feature should allow the system to:

1. Separate planner attempts, immutable plans, child plans, and evaluations.
2. Prevent post-solve route mutations from retaining stale solver certification.
3. Store requested/executed method lineage and fallback reasons consistently.

Keep the implementation modular, easy to test, and consistent with the existing project structure.

## Current State

The project already has:

* `src/itinerary_system/research_artifacts.py`: defines `PlannerRun`, `PlanArtifact`, `ResearchEvaluationReport`, and `evaluate_phase0_plan()`.
* `src/itinerary_system/phase0_exporter.py`: writes Phase 0 artifact tables and gates route validation.
* `src/itinerary_system/experiment_runner.py`: builds production routes and can ensure required anchor stops after route generation.
* `docs/planning/research_stabilization_and_publication_plan.md`: identifies post-solve anchor mutation as an explicit blocker.
* `tests/test_research_foundation.py`: covers Phase 0 artifact and route-validation behavior.

The missing part is:

* `PlanArtifact` still lacks canonical v2 fields such as `ordered_days`, `route_ids_by_day`, `owned_constraints`, and `certificate_id`.
* Post-solve mutations do not universally create a child run, child plan, and new evaluation.
* Fallback lineage is distributed across production helpers rather than centralized.

## New Components to Add

### Component 1

`PlanArtifactV2`

Responsibility:

Extend the existing plan artifact with backward-compatible v2 fields required for parent-child lineage, certificates, route IDs, owned constraints, and stable hashes.

### Component 2

`PlannerRunRecorder`

Responsibility:

Create consistent requested/executed method records for solver runs, fallbacks, post-processing runs, and failed attempts.

### Component 3

`PostSolveMutationGate`

Responsibility:

Detect material route changes after solve and either block publication eligibility or create a child lineage record before evaluation.

## Class / Registry Diagrams

```text
+-------------------------------------------------------------------------------+
|                              PlanArtifactV2                                     |
+-------------------------------------------------------------------------------+
|  - plan_id: str                                                                |
|  - parent_plan_id: str | None                                                  |
|  - source_run_id: str                                                          |
|  - route_ids_by_day: dict[int, str]                                            |
|  - owned_constraints: tuple[OwnedConstraint, ...]                              |
|  - certificate_id: str | None                                                  |
|  - content_hash: str                                                           |
+-------------------------------------------------------------------------------+
|  + to_record(): dict[str, Any]              --> Serializes stable artifact     |
|  + content_hash(): str                      --> Hashes artifact without hash   |
|  + from_v1(plan): PlanArtifactV2            --> Migrates old records safely    |
+-------------------------------------------------------------------------------+
```

```text
+-------------------------------------------------------------------------------+
|                             PlannerRunRecorder                                  |
+-------------------------------------------------------------------------------+
|  - clock: Callable[[], datetime]                                                |
|  - planner_specification_id: str                                                |
+-------------------------------------------------------------------------------+
|  + start_run(request, method): PlannerRun     --> Creates attempt record        |
|  + complete_run(run, result): PlannerRun      --> Adds status/certification    |
|  + fallback_run(parent, reason): PlannerRun   --> Creates child fallback run    |
|  + failed_run(parent, error): PlannerRun      --> Creates failed run record     |
+-------------------------------------------------------------------------------+
```

```text
+-------------------------------------------------------------------------------+
|                            PostSolveMutationGate                                |
+-------------------------------------------------------------------------------+
|  - No persistent internal state                                                |
+-------------------------------------------------------------------------------+
|  + compare_before_after(parent, candidate): MutationReport --> Detects edits    |
|  + require_child_run(report): bool          --> Classifies material changes     |
|  + invalidate_certificate(plan): PlanArtifactV2 --> Clears stale certificate    |
+-------------------------------------------------------------------------------+
```

## Class Diagram Rules

1. Create one diagram block for each lineage component.
2. Keep plan and run classes in artifact modules, not optimizer modules.
3. List only stable fields that influence lineage or certification.
4. Public methods must create new records rather than mutate existing records.
5. Mark mutation detection as a stateless utility.
6. If legacy v1 compatibility is needed, implement it as explicit migration methods.
7. Skip new repository storage in this phase if Phase 1.0 introduces it first.

## Data Model

```python
@dataclass(frozen=True)
class MutationReport:
    parent_plan_id: str
    candidate_plan_id: str
    material_change: bool
    changed_fields: tuple[str, ...]
    reason_codes: tuple[str, ...]
    requires_child_run: bool

@dataclass(frozen=True)
class PlanArtifactV2:
    plan_id: str
    parent_plan_id: str | None
    source_run_id: str
    planning_request_id: str
    catalog_snapshot_id: str
    context_snapshot_id: str
    selected_stops: tuple[dict[str, Any], ...]
    ordered_days: tuple[dict[str, Any], ...]
    lodging_assignments: dict[str, str]
    route_ids_by_day: dict[int, str]
    owned_constraints: tuple[dict[str, Any], ...]
    modeled_metrics: dict[str, float]
    change_components: dict[str, float]
    certificate_id: str | None
    created_at: str
    schema_version: str = "plan-artifact-v2"
```

## Storage / State

Persistent state:

* Existing Phase 0 CSV/JSONL outputs remain supported.
* V2 records should be emitted as JSONL under current Phase 0/exporter outputs, then later under `runs/<run_id>/plans/`.

No hidden mutable global state should be introduced.

## Required Methods

```python
def stable_content_hash(payload: Any) -> str
def plan_artifact_from_v1(plan: PlanArtifact) -> PlanArtifactV2
def detect_post_solve_mutation(before: PlanArtifactV2, after: PlanArtifactV2) -> MutationReport
def create_child_plan_after_mutation(parent: PlanArtifactV2, changed_plan: PlanArtifactV2, run: PlannerRun) -> PlanArtifactV2
def mark_solver_certificate_invalidated(run: PlannerRun, reason: str) -> PlannerRun
```

## Validation Rules

1. Plan content hash must change when selected stops, sequence, lodging, route IDs, constraints, or certificate changes.
2. `source_run_id` must equal the run that produced the exact plan content.
3. A plan edited after solve must have a new child run or `solver_certification="INVALIDATED_AFTER_EDIT"`.
4. Fallback runs must preserve `method_requested`, `method_executed`, and `fallback_reason`.
5. V1 records must remain readable by current Phase 0 tests.
6. A displayed plan without valid lineage is comparison-ineligible.

## UI / API Integration

Internal artifact integration:

* `src/itinerary_system/phase0_exporter.py` emits v2-compatible records.
* `src/itinerary_system/experiment_runner.py` calls the mutation gate around required-anchor insertion or replacement.
* `src/itinerary_system/map_exporter.py` can display method/fallback labels from canonical run records in a later phase.

## Workflow

1. Solver creates a `PlannerRun`.
2. Solver creates a candidate `PlanArtifactV2`.
3. Any post-solve edit is compared against the candidate artifact.
4. If material, create child run and child plan or mark certification invalid.
5. Run independent evaluation against the exact final displayed plan.
6. Export plan, run, mutation report, and evaluation together.

## Files to Create

* `tests/test_artifact_lineage.py`

## Files to Modify

* `src/itinerary_system/research_artifacts.py`
* `src/itinerary_system/phase0_exporter.py`
* `src/itinerary_system/experiment_runner.py`
* `src/itinerary_system/map_exporter.py` only if labels must read v2 fields immediately.
* `tests/test_research_foundation.py`

## Error Handling

* Missing parent plan ID for a repair child should make the child comparison-ineligible.
* Unknown solver status should map to `NO_CERTIFICATE` and store raw status.
* Mutation without child lineage should fail strict validation.
* Hash mismatch between emitted plan and stored metadata should fail evaluation.
* Legacy v1 records should raise a clear migration error only when a v2-only field is required.

## Testing Checklist

- [ ] V1 `PlanArtifact` tests still pass.
- [ ] V2 plan hash changes on selected stop, sequence, route ID, lodging, and certificate changes.
- [ ] Post-solve anchor insertion creates a child run or invalidates certification.
- [ ] Fallback run records preserve requested/executed method names.
- [ ] `evaluate_phase0_plan()` rejects a plan whose `source_run_id` does not match the final run.
- [ ] Strict Phase 0 validation blocks mutated plans without fresh evaluation.
- [ ] Existing route-validation tests still pass.

## Roadmap / Full Pipeline Update

After implementation and tests pass, update the Phase 0 foundation status in `docs/planning/research_stabilization_and_publication_plan.md`. Do not mark minimal-change repair complete; this phase only makes artifacts truthful.

## Acceptance Criteria

1. Every displayed publication-path plan has an exact source run.
2. Material post-solve edits cannot keep stale solver certification.
3. Fallback and failure attempts are recorded as first-class planner runs.
4. Existing Phase 0 tests remain compatible.
5. Strict validation fails closed when lineage is incomplete.

# Phase 0.2 Implementation Plan: RouteMatrix And Road-Validation Boundary

## Goal

Implement **route matrix and road-validation boundary**.

This feature should allow the system to:

1. Feed solver travel times and distances from explicit route records rather than hidden geodesic estimates.
2. Keep geodesic fallback available only in demo/approximate mode.
3. Make optimizer, map, and evaluator route totals agree with the same route provenance.

Keep the implementation modular, easy to test, and consistent with the existing project structure.

## Current State

The project already has:

* `src/itinerary_system/routing/models.py`: `RouteLegResult` and `RouteResult` with road-validation fields.
* `src/itinerary_system/routing/cache.py`: `RoadRouteCache` for validated route-leg lookup.
* `src/itinerary_system/routing/road_cache_builder.py`: creates road route caches and audit files.
* `src/itinerary_system/multi_objective_route.py`: computes `travel_minutes()` from geodesic distance.
* `src/itinerary_system/hierarchical_gurobi.py`: uses geodesic drive proxies in city-level planning.
* `src/itinerary_system/route_gurobi_oracle.py`: returns `solve_multi_objective_route()` before a legacy implementation block.
* `docs/reference/data_dictionary.md`: states geodesic fallback is not final-comparison eligible.

The missing part is:

* A canonical `RouteMatrix` indexed by stable stop/entity IDs.
* Injection of route durations into solver paths.
* Publication-mode refusal when a road-valid matrix is missing.
* Removal or quarantine of unreachable route-oracle legacy code after equivalence tests.

## New Components to Add

### Component 1

`RouteMatrix`

Responsibility:

Store route-leg cells with duration, distance, validation, fallback, and provenance for every pair used by a solver or evaluator.

### Component 2

`RoutingProvider`

Responsibility:

Define provider protocol methods for table, route, nearest, and activation/probe behavior.

### Component 3

`SolverRouteMatrixAdapter`

Responsibility:

Map solver node IDs to `RouteMatrix` cells and enforce publication versus demo fallback policy.

## Class / Registry Diagrams

```text
+-------------------------------------------------------------------------------+
|                                  RouteMatrix                                    |
+-------------------------------------------------------------------------------+
|  - matrix_id: str                                                              |
|  - context_snapshot_id: str                                                    |
|  - entity_ids: tuple[str, ...]                                                 |
|  - cells: dict[tuple[str, str], RouteMatrixCell]                               |
+-------------------------------------------------------------------------------+
|  + duration_minutes(origin, dest): float    --> Solver-ready travel time       |
|  + distance_m(origin, dest): float          --> Solver-ready distance          |
|  + leg(origin, dest): RouteLegResult        --> Provenance-bearing route leg   |
|  + require_road_validated(): None           --> Fails if any cell invalid      |
+-------------------------------------------------------------------------------+
```

```text
+-------------------------------------------------------------------------------+
|                                RoutingProvider                                  |
+-------------------------------------------------------------------------------+
|  - provider_id: str                                                            |
+-------------------------------------------------------------------------------+
|  + probe(): RoutingActivation              --> Checks provider readiness       |
|  + table(request): RouteMatrix             --> Returns matrix durations        |
|  + route(request): RouteLegResult          --> Returns selected leg geometry   |
|  + nearest(request): SnapResult            --> Returns snap diagnostics        |
+-------------------------------------------------------------------------------+
```

```text
+-------------------------------------------------------------------------------+
|                           SolverRouteMatrixAdapter                              |
+-------------------------------------------------------------------------------+
|  - route_matrix: RouteMatrix                                                   |
|  - mode: str                                                                   |
+-------------------------------------------------------------------------------+
|  + travel_minutes(i, j): float             --> Reads a matrix cell             |
|  + assert_publication_ready(): None        --> Blocks fallback cells           |
|  + route_result(sequence): RouteResult     --> Builds final audited route      |
+-------------------------------------------------------------------------------+
```

## Class Diagram Rules

1. Create diagrams for `RouteMatrix`, provider protocol, and solver adapter.
2. Store all route provenance in routing modules, not optimization modules.
3. Optimizers may read durations but may not invent route provenance.
4. Demo fallback must be explicit and auditable.
5. Publication mode must fail before optimization if required route cells are missing.
6. Skip OSRM Docker files until the provider interface and matrix tests are stable.

## Data Model

```python
@dataclass(frozen=True)
class RouteMatrixCell:
    origin_id: str
    destination_id: str
    distance_m: float | None
    duration_s: float | None
    route_leg_id: str
    road_validated: bool
    fallback_used: bool
    provider: str
    query_hash: str
    context_snapshot_id: str

@dataclass(frozen=True)
class RouteMatrix:
    matrix_id: str
    context_snapshot_id: str
    entity_ids: tuple[str, ...]
    cells: Mapping[tuple[str, str], RouteMatrixCell]
```

## Storage / State

Persistent state:

* Route matrices are context-snapshot artifacts and should eventually live under `data/contexts/<context_snapshot_id>/route_options.csv` or `runs/<run_id>/routing/`.
* Existing `production_road_route_cache.csv` remains a Phase 0 source for validated cells.

Temporary state:

* Solver adapters hold matrix references only for one run.

## Required Methods

```python
def load_route_matrix_from_cache(path: Path, context_snapshot_id: str) -> RouteMatrix
def build_route_matrix_from_context(bundle: DatasetBundle) -> RouteMatrix
def geodesic_fallback_matrix(entity_points: Mapping[str, tuple[float, float]]) -> RouteMatrix
def route_minutes_from_matrix(matrix: RouteMatrix, origin_id: str, destination_id: str) -> float
def route_result_for_sequence(matrix: RouteMatrix, sequence: tuple[str, ...]) -> RouteResult
```

## Validation Rules

1. Missing matrix cell must raise a clear error; never substitute zero.
2. `duration_s` and `distance_m` must be positive when present.
3. Fallback cells must have `road_validated=False` and `fallback_used=True`.
4. Publication mode requires all selected cells to be road validated.
5. Matrix entity IDs must be stable POI, lodging, airport, or depot IDs.
6. Optimizer totals must match route result totals within tolerance.

## UI / API Integration

Internal integration:

* `multi_objective_route.solve_multi_objective_route()` accepts an optional route-matrix adapter and publication/demo mode.
* `hierarchical_gurobi.py` reads intercity/city route durations from the same interface.
* `route_gurobi_oracle.py` either becomes a thin documented wrapper or loses unreachable legacy code after tests.
* `phase0_exporter.py` and evaluators consume `RouteResult` created from the matrix.

## Workflow

1. Load catalog/context bundle.
2. Load or build route matrix.
3. Validate route matrix against selected candidate IDs.
4. Pass adapter into optimizer.
5. Solver optimizes using matrix durations.
6. Final plan route IDs and route result are emitted.
7. Evaluator checks the same route result for eligibility.

## Files to Create

* `src/itinerary_system/routing/matrix.py`
* `src/itinerary_system/routing/provider.py`
* `tests/routing/test_route_matrix.py`

## Files to Modify

* `src/itinerary_system/routing/models.py`
* `src/itinerary_system/routing/__init__.py`
* `src/itinerary_system/multi_objective_route.py`
* `src/itinerary_system/hierarchical_gurobi.py`
* `src/itinerary_system/route_gurobi_oracle.py`
* `src/itinerary_system/phase0_exporter.py`
* `tests/test_research_foundation.py`

## Error Handling

* Missing matrix: publication mode raises `RouteMatrixMissing`.
* Missing cell: raise `RouteMatrixCellMissing` with origin/destination IDs.
* Invalid fallback in strict mode: raise `RouteMatrixNotPublicationEligible`.
* Provider unavailable: return activation status and do not send normal requests.
* Unreachable legacy code: remove only after regression tests prove current wrapper behavior.

## Testing Checklist

- [ ] Empty matrix rejects solver usage.
- [ ] Missing cell rejects solver usage.
- [ ] Fallback matrix works in demo mode and is blocked in publication mode.
- [ ] Validated cache matrix produces `RouteResult.evaluation_eligible=True` when solver/schedule/snapshot gates pass.
- [ ] `multi_objective_route.py` no longer calls geodesic travel in publication mode.
- [ ] `hierarchical_gurobi.py` can run with injected matrix durations.
- [ ] Route oracle wrapper behavior is covered before deleting unreachable code.
- [ ] Existing road-cache builder tests still pass.

## Roadmap / Full Pipeline Update

After implementation and tests pass, update the technical specification status for `ROUTE-001` and `ROUTE-004`. Do not claim road-valid publication results until a validated context matrix exists for every compared route.

## Acceptance Criteria

1. Publication-mode solvers receive route durations only from `RouteMatrix`.
2. Geodesic fallback is explicit, labeled, and comparison-ineligible.
3. Optimizer, map, and evaluator route totals use the same route records.
4. Missing or invalid route evidence fails closed.
5. Legacy route-oracle dead code is removed or quarantined only after tests.

# Phase 0.3 Implementation Plan: Utility Source-Missingness And Claim-Safe Scores

## Goal

Implement **utility source-missingness and claim-safe scores**.

This feature should allow the system to:

1. Stop penalizing missing source families as numeric zero when the source is unavailable.
2. Separate `source_coverage_score` from utility and uncertainty.
3. Keep legacy `data_confidence` compatibility without using it as a calibrated truth claim.

Keep the implementation modular, easy to test, and consistent with the existing project structure.

## Current State

The project already has:

* `src/itinerary_system/utility_model.py`: builds signals and utility scores; still uses `data_confidence` aliases.
* `src/itinerary_system/data_enrichment.py`: computes `source_coverage_score` and then writes `data_confidence` as an alias.
* `src/itinerary_system/schemas.py`: includes `data_confidence` in `EnrichedPOI`.
* `docs/current/current_score_audit.md`: documents missing Yelp as effectively zero and warns that weights are heuristic.
* `tests/test_research_foundation.py`: tests that source coverage and model uncertainty remain separate for one case.

The missing part is:

* Per-source availability masks for Yelp, Wikipedia, curated, OSM, weather, and route features.
* Utility fusion that excludes unavailable source signals from denominators.
* Tests proving equivalent non-Yelp POIs do not diverge because one city lacks Yelp coverage.

## New Components to Add

### Component 1

`SourceSignalMask`

Responsibility:

Represent which source families are available for each POI and which utility terms should be active.

### Component 2

`MaskedUtilitySignalBuilder`

Responsibility:

Build normalized utility signals with missingness masks and separate coverage metrics.

### Component 3

`SourceAblationReport`

Responsibility:

Produce deterministic audit rows that show how utility changes when each source family is included or excluded.

## Class / Registry Diagrams

```text
+-------------------------------------------------------------------------------+
|                              SourceSignalMask                                   |
+-------------------------------------------------------------------------------+
|  - poi_id: str                                                                 |
|  - available_sources: frozenset[str]                                           |
|  - missing_sources: frozenset[str]                                             |
|  - source_weights: dict[str, float]                                            |
+-------------------------------------------------------------------------------+
|  + is_available(source): bool              --> Checks source availability      |
|  + active_weight_sum(): float              --> Weight denominator              |
|  + to_record(): dict[str, Any]             --> Audit serialization             |
+-------------------------------------------------------------------------------+
```

```text
+-------------------------------------------------------------------------------+
|                         MaskedUtilitySignalBuilder                              |
+-------------------------------------------------------------------------------+
|  - No persistent internal state                                                |
+-------------------------------------------------------------------------------+
|  + build_masks(frame): pd.DataFrame        --> Adds source mask columns        |
|  + build_signals(frame, config): pd.DataFrame --> Computes masked signals      |
|  + score_masked_mcda(signals): pd.Series   --> Scores using active sources     |
+-------------------------------------------------------------------------------+
```

```text
+-------------------------------------------------------------------------------+
|                              SourceAblationReport                               |
+-------------------------------------------------------------------------------+
|  - No persistent internal state                                                |
+-------------------------------------------------------------------------------+
|  + compute(frame, config): pd.DataFrame    --> Source-family ablation rows     |
|  + write(output_dir): Path                 --> Writes audit output             |
+-------------------------------------------------------------------------------+
```

## Class Diagram Rules

1. Create diagrams for mask data, signal builder, and ablation report.
2. Treat this phase as stateless utility work except for optional audit output.
3. Do not remove legacy `data_confidence` until downstream dashboards are migrated.
4. Prefer explicit mask columns over inferring missingness from zeros.
5. Keep utility weights configurable through existing config patterns.

## Data Model

```python
type SourceMaskColumns = {
    "has_osm": bool,
    "has_yelp": bool,
    "has_curated": bool,
    "has_wikidata": bool,
    "has_wikipedia": bool,
    "has_weather": bool,
    "source_coverage_score": float,
    "model_uncertainty": float | None,
}

type MaskedUtilityOutput = {
    "poi_id": str,
    "utility_masked_mcda": float,
    "source_coverage_score": float,
    "data_confidence": float,  # compatibility alias only
    "active_source_weight": float,
    "missing_source_list": str,
}
```

## Storage / State

Stateless feature:

This feature receives enriched POI frames, returns scored frames, and does not persist data by default.

Optional persistent audit:

* `production_utility_source_ablation.csv` under run/output directories.

## Required Methods

```python
def build_source_masks(enriched_df: pd.DataFrame) -> pd.DataFrame
def normalize_source_signal(series: pd.Series, mask: pd.Series, default: float = 0.0) -> pd.Series
def score_masked_weighted_utility(signal_df: pd.DataFrame, config: TripConfig) -> pd.Series
def write_source_ablation_audit(enriched_df: pd.DataFrame, output_dir: Path, config: TripConfig) -> pd.DataFrame
```

## Validation Rules

1. A missing source is excluded from that source term and denominator.
2. A present source with a bad/low value may score low; missing and low are different states.
3. All missing sources should return a conservative explicit fallback and low coverage.
4. `source_coverage_score` must remain in `[0, 1]`.
5. `data_confidence` must be documented as a compatibility alias for coverage only.
6. Utility scoring must not mutate original input frames in place unless documented.

## UI / API Integration

Internal integration:

* `src/itinerary_system/utility_model.py` uses masks for MCDA/TOPSIS/Bayesian UCB inputs.
* `src/itinerary_system/data_enrichment.py` emits mask columns and preserves legacy aliases.
* `src/itinerary_system/map_exporter.py` can keep displaying `source_confidence`/`data_confidence` until migrated.

## Workflow

1. Receive enriched POI frame.
2. Add source availability masks.
3. Normalize each source only over available rows.
4. Score utility with active source weights.
5. Emit source coverage separately from utility.
6. Write ablation audit if output directory is provided.

## Files to Create

* `tests/test_utility_missingness.py`

## Files to Modify

* `src/itinerary_system/utility_model.py`
* `src/itinerary_system/data_enrichment.py`
* `src/itinerary_system/schemas.py`
* `src/itinerary_system/map_exporter.py` only if compatibility names change.
* `docs/current/current_score_audit.md` after implementation to reflect the fixed behavior.

## Error Handling

* Missing required identity columns should raise a clear validation error in strict mode.
* Non-numeric source columns should coerce with warnings only when permissive mode is requested.
* All-source-missing rows should get explicit fallback utility and missingness flags.
* Configured source weights that sum to zero should raise `ValueError`.
* Unknown source family in config should be ignored only with an audit warning.

## Testing Checklist

- [ ] Identical non-Yelp POIs do not differ only because Yelp is unavailable.
- [ ] Present but poor Yelp score is still allowed to lower utility.
- [ ] All-source-missing row returns fallback utility and low source coverage.
- [ ] `source_coverage_score` and `model_uncertainty` remain separate.
- [ ] Legacy `data_confidence` alias equals source coverage where required by old dashboards.
- [ ] Source ablation audit is deterministic.
- [ ] Existing utility and Phase 0 tests still pass.

## Roadmap / Full Pipeline Update

After implementation and tests pass, update `docs/current/current_score_audit.md` and `docs/reference/data_dictionary.md` with the repaired missingness behavior. Do not claim calibrated satisfaction or learned preferences.

## Acceptance Criteria

1. Missing source data is not silently treated as zero quality.
2. Utility, source coverage, and uncertainty are separate fields.
3. Legacy compatibility is preserved for dashboards/tests.
4. Source ablation makes utility provenance inspectable.
5. Claim language remains limited to heuristic contextual utility.

# Phase 1.0 Implementation Plan: Parent Plan, Ownership, And Plan Diff Foundation

## Goal

Implement **parent plan, ownership, and plan diff foundation**.

This feature should allow the system to:

1. Store immutable parent and child plans with lineage.
2. Attach ownership labels to stops, lodging, routes, time windows, and user constraints.
3. Compute typed plan differences and weighted edit cost.

Keep the implementation modular, easy to test, and consistent with the existing project structure.

## Current State

The project already has:

* `src/itinerary_system/research_artifacts.py`: lightweight plan artifacts but no full ownership schema.
* `src/itinerary_system/repair_planner.py`: repair records over route dictionaries and parent route hashes.
* `src/itinerary_system/request_schema.py`: structured trip request overrides, but no canonical parsed request or ownership semantics.
* `docs/current/current_score_audit.md`: recommends `ParentPlan`/`ChildPlan`, ownership labels, typed edit variables, and parent-child metrics.
* `docs/planning/travel_itinerary_repair_technical_specification.md`: specifies `OwnedConstraint`, `RepairRequest`, and `PlanDiff` shapes.

The missing part is:

* A canonical `plans` package for immutable plans and repositories.
* `OwnedConstraint` and ownership/relaxation policy vocabularies.
* Generalized plan diff over stops, days, times, lodging, routes, and ordering.

## New Components to Add

### Component 1

`OwnedConstraint`

Responsibility:

Represent who owns a constraint, how strong it is, what it targets, and whether it may be relaxed.

### Component 2

`PlanRepository`

Responsibility:

Persist immutable parent/child plan records append-only and reject duplicate IDs with different content.

### Component 3

`PlanDiffBuilder`

Responsibility:

Compute added/deleted stops, day moves, time shifts, reorder changes, lodging changes, road changes, unchanged days, and weighted edit cost.

## Class / Registry Diagrams

```text
+-------------------------------------------------------------------------------+
|                                OwnedConstraint                                  |
+-------------------------------------------------------------------------------+
|  - constraint_id: str                                                          |
|  - origin: ConstraintOrigin                                                    |
|  - strength: ConstraintStrength                                                |
|  - scope: ConstraintScope                                                      |
|  - target_id: str                                                              |
|  - confirmed: bool                                                             |
|  - relaxation_policy: RelaxationPolicy                                         |
+-------------------------------------------------------------------------------+
|  + is_active(): bool                       --> Checks confirmation/strength    |
|  + can_relax(request): bool                --> Applies relaxation rules        |
|  + to_record(): dict[str, Any]             --> Stable serialization            |
+-------------------------------------------------------------------------------+
```

```text
+-------------------------------------------------------------------------------+
|                                PlanRepository                                   |
+-------------------------------------------------------------------------------+
|  - root: Path                                                                  |
|  - index_path: Path                                                            |
+-------------------------------------------------------------------------------+
|  + load(plan_id): PlanArtifactV2           --> Reads immutable plan            |
|  + save(plan): Path                        --> Append-only save                |
|  + exists(plan_id): bool                   --> Checks registry                 |
|  + verify_hash(plan_id): bool              --> Validates stored content hash   |
+-------------------------------------------------------------------------------+
```

```text
+-------------------------------------------------------------------------------+
|                                PlanDiffBuilder                                  |
+-------------------------------------------------------------------------------+
|  - ownership_policy: OwnershipPolicy                                           |
+-------------------------------------------------------------------------------+
|  + diff(parent, child): PlanDiff           --> Computes typed changes          |
|  + weighted_cost(diff): float              --> Applies ownership weights       |
|  + unchanged_days(diff): tuple[int, ...]   --> Identifies preserved days       |
+-------------------------------------------------------------------------------+
```

## Class Diagram Rules

1. Create diagrams for persistent plan storage, ownership records, and diff utilities.
2. Put plan artifacts under `src/itinerary_system/plans/`, not notebooks.
3. Keep ownership vocabularies closed with `StrEnum`.
4. Do not allow unconfirmed LLM constraints to become active.
5. Parent plans are immutable; any change creates a child plan.
6. If repository persistence is deferred, still implement JSON round-trip tests for the models.

## Data Model

```python
@dataclass(frozen=True)
class OwnedConstraint:
    constraint_id: str
    origin: ConstraintOrigin
    strength: ConstraintStrength
    scope: ConstraintScope
    target_id: str
    relation: str
    value: JsonValue
    confirmed: bool
    relaxation_policy: RelaxationPolicy
    evidence_refs: tuple[str, ...] = ()
    schema_version: str = "owned-constraint-v1"

@dataclass(frozen=True)
class PlanDiff:
    diff_id: str
    parent_plan_id: str
    child_plan_id: str
    added_stops: tuple[StopChange, ...]
    deleted_stops: tuple[StopChange, ...]
    day_moves: tuple[DayMove, ...]
    time_shifts: tuple[TimeShift, ...]
    reorder_changes: tuple[OrderChange, ...]
    lodging_changes: tuple[LodgingChange, ...]
    road_changes: tuple[RoadChange, ...]
    unchanged_days: tuple[int, ...]
    weighted_edit_cost: float
```

## Storage / State

Persistent state:

* `data/benchmark/parent_plans/*.json` for reviewed benchmark parents.
* Later `runs/<run_id>/plans/*.json` for generated child plans.

Append-only rule:

* Saving the same `plan_id` with different content must fail.
* Saving the same `plan_id` with identical content may be idempotent.

## Required Methods

```python
def plan_from_phase0_route_stops(route_stops_df: pd.DataFrame, *, plan_id: str) -> PlanArtifactV2
def load_plan(path: Path) -> PlanArtifactV2
def save_plan_append_only(plan: PlanArtifactV2, root: Path) -> Path
def compute_plan_diff(parent: PlanArtifactV2, child: PlanArtifactV2, policy: OwnershipPolicy) -> PlanDiff
def validate_owned_constraints(constraints: tuple[OwnedConstraint, ...]) -> tuple[str, ...]
```

## Validation Rules

1. Required IDs must be nonempty and stable.
2. Parent and child plans must share compatible catalog/context IDs unless context change is explicit.
3. `LLM_INTERPRETATION` constraints with `confirmed=False` are inactive.
4. `LOCKED` constraints have no automatic relaxation variable.
5. `BOOKED` constraints require explicit relaxation permission.
6. Diff must handle identical plans, partial overlap, reordered stops, and missing route IDs.
7. Weighted edit cost must be deterministic.

## UI / API Integration

Internal integration:

* `repair_planner.py` can migrate from route-hash-only parents to `PlanArtifactV2`.
* Future dashboard export can use diff components for original/repaired views.
* The benchmark runner can load canonical parents from `PlanRepository`.

## Workflow

1. Build or load immutable parent plan.
2. Validate owned constraints.
3. Save parent plan append-only.
4. Create child plan through a planner/repair run.
5. Compute typed diff.
6. Store diff with child lineage and metrics.

## Files to Create

* `src/itinerary_system/plans/__init__.py`
* `src/itinerary_system/plans/models.py`
* `src/itinerary_system/plans/repository.py`
* `src/itinerary_system/plans/diff.py`
* `tests/plans/test_models.py`
* `tests/plans/test_repository.py`
* `tests/plans/test_diff.py`
* `data/benchmark/parent_plans/plan_demo_current.json`

## Files to Modify

* `src/itinerary_system/research_artifacts.py`
* `src/itinerary_system/schemas.py`
* `src/itinerary_system/repair_planner.py`
* `tests/test_repair_planner.py`
* `tests/test_research_foundation.py`

## Error Handling

* Duplicate plan ID with different content raises `PlanRepositoryConflict`.
* Missing parent plan raises `PlanNotFound`.
* Unknown constraint enum value raises validation error.
* Inactive unconfirmed constraints are preserved in the artifact but excluded from active solver constraints.
* Diff on incompatible catalog snapshots raises unless explicitly configured as migration mode.

## Testing Checklist

- [ ] `OwnedConstraint` JSON round trip preserves enum values.
- [ ] Unconfirmed LLM constraints are inactive.
- [ ] Locked constraints cannot be relaxed automatically.
- [ ] Plan repository append-only behavior rejects conflicting duplicate IDs.
- [ ] Identical parent/child diff has zero weighted cost.
- [ ] Added/deleted/moved/reordered/lodging/road changes are detected.
- [ ] Phase 0 plan artifacts can migrate into v2 records.
- [ ] Existing repair planner scaffold still passes during migration.

## Roadmap / Full Pipeline Update

After implementation and verification, update `docs/planning/travel_itinerary_repair_technical_specification.md` status for `PLAN-001`, `PLAN-002`, and `DIFF-001`. Do not claim repair improvement yet; this phase only creates the parent-child substrate.

## Acceptance Criteria

1. Parent plans are immutable and append-only.
2. Ownership and relaxation semantics are explicit and tested.
3. Plan diffs produce typed change components and weighted edit costs.
4. Legacy Phase 0 artifacts can migrate without breaking current tests.
5. Downstream repair/evaluator phases have canonical plan inputs.

# Phase 2.0 Implementation Plan: Progressive Ownership-Aware Repair Solver

## Goal

Implement **progressive ownership-aware repair solver**.

This feature should allow the system to:

1. Repair a child plan relative to an immutable parent plan.
2. Minimize hard/booked relaxations and ownership-weighted changes before contextual burden and utility.
3. Expand from narrow repair neighborhoods to full reoptimization only when smaller repairs fail.

Keep the implementation modular, easy to test, and consistent with the existing project structure.

## Current State

The project already has:

* `src/itinerary_system/repair_planner.py`: deterministic route-dictionary scaffold with `ADD`, `DELETE`, `REPLACE`, `MOVE`, `RELAX`, and `KEEP`.
* `src/itinerary_system/multi_objective_route.py`: day-level candidate route optimizer, but no parent plan or typed edit variables.
* `src/itinerary_system/hierarchical_gurobi.py`: city-level planning, but no parent ownership semantics.
* `docs/current/current_score_audit.md`: states current route objective is not parent-aware repair.
* `docs/planning/travel_itinerary_repair_technical_specification.md`: defines progressive neighborhoods and sequential objective stages.

The missing part is:

* Repair neighborhoods that freeze unaffected days.
* Change variables relative to parent assignments.
* Lexicographic solve stages.
* Day-route subproblem using `RouteMatrix`.
* Progressive controller that stops at the smallest independently eligible radius.

## New Components to Add

### Component 1

`RepairNeighborhoodBuilder`

Responsibility:

Construct editable sets for radius 0, same-day replacement, adjacent-day move, hotel-preserving reroute, hotel-changing repair, and full reoptimization.

### Component 2

`RepairMasterModel`

Responsibility:

Create selection, day, lodging, relaxation, and typed change variables relative to the parent plan.

### Component 3

`LexicographicRepairSolver`

Responsibility:

Solve sequential objective stages and store each stage status, objective, bound, gap, and tolerance.

### Component 4

`ProgressiveRepairController`

Responsibility:

Try allowed neighborhoods in order, evaluate each candidate independently, and return the smallest eligible repair.

## Class / Registry Diagrams

```text
+-------------------------------------------------------------------------------+
|                           RepairNeighborhoodBuilder                             |
+-------------------------------------------------------------------------------+
|  - No persistent internal state                                                |
+-------------------------------------------------------------------------------+
|  + build(parent, request, radius): RepairNeighborhood --> Editable sets        |
|  + freeze_constraints(parent, neighborhood): tuple[OwnedConstraint, ...]        |
|  + affected_days(request): tuple[int, ...]  --> Computes disruption scope      |
+-------------------------------------------------------------------------------+
```

```text
+-------------------------------------------------------------------------------+
|                                RepairMasterModel                                |
+-------------------------------------------------------------------------------+
|  - parent: PlanArtifactV2                                                      |
|  - request: RepairRequest                                                      |
|  - neighborhood: RepairNeighborhood                                            |
|  - route_matrix: RouteMatrix                                                   |
+-------------------------------------------------------------------------------+
|  + build(): RepairModel                    --> Creates variables/constraints   |
|  + export_components(solution): dict       --> Change/context/utility metrics  |
|  + extract_plan(solution): PlanArtifactV2  --> Builds child plan candidate     |
+-------------------------------------------------------------------------------+
```

```text
+-------------------------------------------------------------------------------+
|                           LexicographicRepairSolver                             |
+-------------------------------------------------------------------------------+
|  - solver_backend: SolverBackend                                               |
|  - tolerances: ObjectiveTolerances                                             |
+-------------------------------------------------------------------------------+
|  + solve(model): LexicographicResult       --> Runs all stages in order        |
|  + fix_stage(stage, value): None           --> Preserves prior optimum         |
|  + to_planner_runs(result): tuple[PlannerRun, ...] --> Stage lineage          |
+-------------------------------------------------------------------------------+
```

```text
+-------------------------------------------------------------------------------+
|                           ProgressiveRepairController                           |
+-------------------------------------------------------------------------------+
|  - plan_repository: PlanRepository                                             |
|  - solver: LexicographicRepairSolver                                           |
|  - evaluator: PlanEvaluator                                                    |
+-------------------------------------------------------------------------------+
|  + repair_progressively(request): RepairOutcome --> Returns eligible repair    |
|  + diagnose_failure(attempts): RepairDiagnosis --> Smallest-relaxation report  |
+-------------------------------------------------------------------------------+
```

## Class Diagram Rules

1. Diagram every important repair module.
2. Keep model construction separate from progressive orchestration.
3. Use `RouteMatrix` for travel, not geodesic helpers.
4. Stage objectives must be explicit and persisted.
5. If Gurobi is unavailable, fallback must be recorded as a separate method, not hidden.
6. Skip enhanced branch-and-check cuts until the baseline progressive solver is stable.

## Data Model

```python
class RepairRadius(StrEnum):
    SAME_STOP_TIME_SHIFT = "same_stop_time_shift"
    SAME_DAY_REPLACEMENT = "same_day_replacement"
    ADJACENT_DAY_MOVE = "adjacent_day_move"
    HOTEL_PRESERVING_REROUTE = "hotel_preserving_reroute"
    HOTEL_CHANGING_REPAIR = "hotel_changing_repair"
    FULL_REOPTIMIZATION = "full_reoptimization"

@dataclass(frozen=True)
class RepairNeighborhood:
    radius: RepairRadius
    editable_days: tuple[int, ...]
    editable_stop_ids: tuple[str, ...]
    editable_lodging_nights: tuple[int, ...]
    frozen_constraint_ids: tuple[str, ...]

@dataclass(frozen=True)
class LexicographicStageResult:
    stage: str
    status: str
    objective_value: float | None
    best_bound: float | None
    optimality_gap: float | None
    fixed_tolerance: float
```

## Storage / State

Temporary state:

* Solver models and variables exist only during one repair attempt.

Persistent state:

* Successful and failed attempts are stored as `PlannerRun` records.
* Child plans, diffs, and evaluation reports are stored by the pipeline/repository.

## Required Methods

```python
def build_repair_neighborhood(parent: PlanArtifactV2, request: RepairRequest, radius: RepairRadius) -> RepairNeighborhood
def build_repair_master_model(parent: PlanArtifactV2, request: RepairRequest, neighborhood: RepairNeighborhood, matrix: RouteMatrix) -> RepairModel
def solve_lexicographically(model: RepairModel, tolerances: ObjectiveTolerances) -> LexicographicResult
def extract_child_plan(parent: PlanArtifactV2, result: LexicographicResult) -> PlanArtifactV2 | None
def repair_progressively(request: RepairRequest, plan_repository: PlanRepository, evaluator: PlanEvaluator) -> RepairOutcome
```

## Validation Rules

1. Outside-neighborhood parent assignments must remain fixed.
2. Locked constraints cannot be deleted or relaxed.
3. Booked constraints require explicit permission before relaxation.
4. Stage 1 cannot degrade Stage 0 beyond tolerance; later stages cannot degrade earlier stages beyond tolerance.
5. Full reoptimization is only used as the final configured fallback.
6. Every candidate must pass independent evaluation before being accepted.
7. Failed solver stages must still emit `PlannerRun` records.

## UI / API Integration

Internal integration:

* `repair_planner.py` remains a compatibility scaffold or is wrapped behind the new controller as a deterministic baseline.
* Future dashboard/UI can call the progressive controller through the pipeline runner.
* The benchmark runner compares this repair against context-blind solver, heuristic, and full reoptimization.

## Workflow

1. Load parent plan.
2. Validate repair request and active constraints.
3. For each allowed radius, build neighborhood.
4. Build repair model.
5. Solve lexicographic stages.
6. Extract child plan and diff.
7. Run independent evaluator.
8. Return the first eligible child plan.
9. If none succeed, return diagnosis and all attempts.

## Files to Create

* `src/itinerary_system/repair/__init__.py`
* `src/itinerary_system/repair/neighborhood.py`
* `src/itinerary_system/repair/change_variables.py`
* `src/itinerary_system/repair/master_model.py`
* `src/itinerary_system/repair/lexicographic.py`
* `src/itinerary_system/repair/day_route_solver.py`
* `src/itinerary_system/repair/progressive.py`
* `tests/repair/test_neighborhood.py`
* `tests/repair/test_master_model.py`
* `tests/repair/test_lexicographic.py`
* `tests/repair/test_progressive.py`

## Files to Modify

* `src/itinerary_system/repair_planner.py`
* `src/itinerary_system/multi_objective_route.py`
* `src/itinerary_system/route_gurobi_oracle.py`
* `src/itinerary_system/__init__.py`
* Existing repair tests during migration.

## Error Handling

* Missing parent plan raises `PlanNotFound`.
* Missing route matrix cell raises before solve in publication mode.
* Infeasible stage returns failed `PlannerRun` and tries next radius.
* Solver backend unavailable creates a recorded fallback only if configured.
* No eligible radius returns `RepairOutcome.infeasible` with diagnosis, not a silent full replan.

## Testing Checklist

- [ ] Radius 0 freezes unaffected days.
- [ ] Adjacent-day move includes boundary route legs and lodging nights.
- [ ] Locked POI cannot be deleted.
- [ ] Booked hotel cannot change without permission.
- [ ] Lexicographic solver preserves prior stage optimum within tolerance.
- [ ] Failed stage emits a failed planner run.
- [ ] Progressive controller stops at smallest eligible radius.
- [ ] Full reoptimization is only final fallback.
- [ ] Existing deterministic repair scaffold still works as a baseline.

## Roadmap / Full Pipeline Update

After implementation and validation, update `REPAIR-001` through `REPAIR-005` status in the technical specification. Only then can roadmap language move from "route alternatives" toward "parent-plan-aware repair implemented."

## Acceptance Criteria

1. Repairs are computed relative to an immutable parent plan.
2. Typed edit variables and weighted change costs are exported.
3. Lexicographic priorities are enforced and tested.
4. Progressive neighborhoods stop at the smallest eligible repair.
5. Every accepted repair has child plan, diff, run lineage, and independent evaluation.

# Phase 3.0 Implementation Plan: Independent Evaluator, Certificate, And Explanation Evidence

## Goal

Implement **independent evaluator, certificate, and explanation evidence**.

## Implementation Status

VERIFY-001, EXPLAIN-001, and EXPLAIN-002 are implemented in the current workspace. The evaluator and certificate package recompute final-plan eligibility independently, structured explanation evidence fails closed without artifact refs, and `ExplanationEvidenceBuilder` now creates default why/contrastive evidence from `PlanDiff`, certificate eligibility, and route-validation records. Dependency-injected counterfactual runners plus deterministic template verbalization support why-not/what-if explanations from stored run evidence. Phase 4.0 still must wire these artifacts into immutable pipeline run directories and benchmark exports before publication or UI claims are complete.

This feature should allow the system to:

1. Recompute final displayed plan validity independently of solver booleans.
2. Issue comparison eligibility certificates with route, snapshot, constraint, and mutation evidence.
3. Generate why, why-not, changed, and what-if explanation evidence constrained to real artifact IDs.

Keep the implementation modular, easy to test, and consistent with the existing project structure.

## Current State

The project already has:

* `src/itinerary_system/research_artifacts.py`: `evaluate_phase0_plan()` gates empty plans, lineage mismatches, solver status, and route eligibility.
* `src/itinerary_system/repair_planner.py`: local `EvaluationReport` and `CounterfactualExplanation` for the scaffold.
* `tests/test_research_foundation.py`: Phase 0 evaluator tests.
* `docs/planning/research_stabilization_and_publication_plan.md`: requires independent hard/soft/preference/robustness evaluation.
* `docs/current/current_score_audit.md`: recommends parent-child preservation, disruption mitigation, and utility layers.

The missing part is:

* A final-plan evaluator that checks all hard constraints, route connectivity, lodging continuity, opening windows, budget, closures, weather gates, and post-solve mutation.
* A certificate object with evidence IDs and fallback flags.
* Explanation evidence that fails closed when claims lack artifact references.

## New Components to Add

### Component 1

`PlanEvaluator`

Responsibility:

Independently recompute artifact grounding, hard feasibility, soft quality, preference fit, and comparison eligibility from final artifacts.

### Component 2

`PlanCertificateBuilder`

Responsibility:

Create certificate records that summarize route validation, snapshot validity, mutation status, solver certification, and warnings.

### Component 3

`ExplanationEvidenceBuilder`

Responsibility:

Build structured explanation claims that reference constraints, route records, plan diffs, evaluations, and counterfactual runs.

## Class / Registry Diagrams

```text
+-------------------------------------------------------------------------------+
|                                  PlanEvaluator                                  |
+-------------------------------------------------------------------------------+
|  - evaluator_version: str                                                      |
|  - route_matrix: RouteMatrix                                                   |
|  - data_bundle: DatasetBundle                                                  |
+-------------------------------------------------------------------------------+
|  + evaluate(plan): ResearchEvaluationReport --> Full eligibility report        |
|  + check_hard_constraints(plan): tuple[Finding, ...]                           |
|  + compute_soft_metrics(plan): dict[str, float]                                |
|  + compute_preference_metrics(plan): dict[str, float]                          |
+-------------------------------------------------------------------------------+
```

```text
+-------------------------------------------------------------------------------+
|                             PlanCertificateBuilder                              |
+-------------------------------------------------------------------------------+
|  - No persistent internal state                                                |
+-------------------------------------------------------------------------------+
|  + build(plan, eval, route): PlanCertificate --> Certificate record            |
|  + invalidate(reason): PlanCertificate       --> Failed certificate            |
|  + evidence_refs(certificate): tuple[str, ...] --> Supporting artifacts        |
+-------------------------------------------------------------------------------+
```

```text
+-------------------------------------------------------------------------------+
|                          ExplanationEvidenceBuilder                             |
+-------------------------------------------------------------------------------+
|  - No persistent internal state                                                |
+-------------------------------------------------------------------------------+
|  + why(plan, evaluation): WhyEvidence       --> Why selected/kept              |
|  + why_not(option, run): WhyNotEvidence     --> Why omitted                    |
|  + changed(diff): ContrastiveEvidence       --> Parent-child changes           |
|  + what_if(request): WhatIfEvidence         --> Counterfactual sandbox result  |
+-------------------------------------------------------------------------------+
```

## Class Diagram Rules

1. Diagram evaluator, certificate builder, and explanation builder separately.
2. Evaluator must not reuse solver feasibility booleans as proof.
3. Certificates summarize evidence but do not replace detailed reports.
4. Every explanation claim must reference at least one artifact or evidence ID.
5. LLM verbalization is optional and must receive only structured evidence.
6. If counterfactual solving is not ready, template why/changed evidence still must work.

## Data Model

```python
@dataclass(frozen=True)
class PlanCertificate:
    certificate_id: str
    plan_id: str
    source_run_id: str
    evaluation_id: str
    route_result_id: str | None
    comparison_eligibility: str
    road_validated: bool
    fallback_used: bool
    mutation_valid: bool
    blocking_findings: tuple[str, ...]
    warning_findings: tuple[str, ...]
    evidence_refs: tuple[str, ...]
    schema_version: str = "plan-certificate-v1"

@dataclass(frozen=True)
class ExplanationClaim:
    claim_id: str
    claim_type: str
    text_template: str
    evidence_refs: tuple[str, ...]
    confidence: str
    unsupported: bool = False
```

## Storage / State

Persistent state:

* Certificates under `runs/<run_id>/evaluations/` or Phase 0 CSV/JSONL output.
* Explanation evidence under `runs/<run_id>/explanations/`.

No hidden state:

* Evaluator version and input artifact hashes must fully determine outputs.

## Required Methods

```python
def evaluate_final_plan(plan: PlanArtifactV2, bundle: DatasetBundle, route_matrix: RouteMatrix) -> ResearchEvaluationReport
def check_route_connectivity(plan: PlanArtifactV2, route_matrix: RouteMatrix) -> tuple[Finding, ...]
def check_owned_constraints(plan: PlanArtifactV2) -> tuple[Finding, ...]
def build_plan_certificate(plan: PlanArtifactV2, evaluation: ResearchEvaluationReport, route_result: RouteResult | None) -> PlanCertificate
def build_explanation_evidence(plan: PlanArtifactV2, diff: PlanDiff, evaluation: ResearchEvaluationReport) -> tuple[ExplanationClaim, ...]
```

## Validation Rules

1. A hard-infeasible or ungrounded plan is comparison-ineligible.
2. Soft and preference metrics are emitted only after hard eligibility is known.
3. Unvalidated route cells block final comparison.
4. Mutation/hash mismatch invalidates certificate.
5. Explanation without evidence references is marked unsupported and hidden from publication outputs.
6. Warnings and failures must be separate.
7. Rule-based and LLM-judged metrics must be stored separately if LLM judges are added later.

## UI / API Integration

Internal and dashboard integration:

* `phase0_exporter.py` writes certificates alongside evaluation reports.
* `map_exporter.py` can display certificate badges and evidence panels after `UI-001`.
* Benchmark runner uses evaluator-owned metrics, not solver totals, for final comparison.

## Workflow

1. Load final plan, planner run, data bundle, route matrix, and optional diff.
2. Recompute artifact and hard feasibility checks.
3. Compute soft/preference metrics only after hard checks.
4. Build certificate and attach it to plan artifact.
5. Generate explanation evidence from diff/evaluation/counterfactual records.
6. Export reports and evidence with stable hashes.

## Files to Create

* `src/itinerary_system/evaluation/__init__.py`
* `src/itinerary_system/evaluation/plan_evaluator.py`
* `src/itinerary_system/evaluation/certificate.py`
* `src/itinerary_system/explanation/__init__.py`
* `src/itinerary_system/explanation/evidence.py`
* `src/itinerary_system/explanation/counterfactual.py`
* `src/itinerary_system/explanation/verbalizer.py`
* `tests/evaluation/test_plan_evaluator.py`
* `tests/evaluation/test_certificate.py`
* `tests/explanation/test_evidence.py`
* `tests/explanation/test_evidence_builder.py`

## Files to Modify

* `src/itinerary_system/research_artifacts.py`
* `src/itinerary_system/phase0_exporter.py`
* `src/itinerary_system/repair_planner.py`
* `src/itinerary_system/map_exporter.py` when certificate badge is added.
* `tests/test_research_foundation.py`

## Error Handling

* Missing route matrix returns hard failure, not a soft warning.
* Missing optional opening-hours data can be a warning if the plan does not claim opening-window validity.
* Missing owned constraints in old artifacts should produce compatibility warning, not crash, unless strict mode requires v2.
* Unsupported explanation claim is emitted as unsupported and excluded from user-facing verbalization.
* Counterfactual solver failure becomes explanation failure evidence, not fabricated why-not text.

## Testing Checklist

- [ ] Empty plan fails artifact grounding.
- [ ] Source-run mismatch fails artifact grounding.
- [ ] Unvalidated route blocks comparison eligibility.
- [ ] Post-solve mutation/hash mismatch invalidates certificate.
- [ ] Locked/confirmed owned constraints are checked.
- [ ] Soft metrics do not override hard failures.
- [ ] Explanation claims without evidence refs fail closed.
- [ ] Why/changed evidence references diff and evaluation IDs.
- [ ] Existing Phase 0 evaluator compatibility remains intact.

## Roadmap / Full Pipeline Update

After implementation and tests pass, update `VERIFY-001` and `EXPLAIN-001` status in the technical specification. Only mark explanation depth beyond evidence scaffolding after counterfactual runner and UI display are verified.

## Acceptance Criteria

1. Final displayed plans receive independent eligibility reports.
2. Certificates make road validation, fallback, mutation, and warning states explicit.
3. Explanation evidence cannot make unsupported numerical or causal claims.
4. Benchmark comparisons use evaluator-owned metrics.
5. Strict mode blocks ineligible plans.

# Phase 4.0 Implementation Plan: Authoritative Pipeline, Benchmark, And Notebook Migration

## Goal

Implement **authoritative pipeline, benchmark, and notebook migration**.

## Implementation Status

PIPE-001 package-runner, Phase 0 generation adapter, production optimizer adapter, progressive repair adapter, BENCH-001 disruption-generator, BENCH-002 paired-runner, and BENCH-002 pipeline-method-adapter slices are implemented in the current workspace. `src/itinerary_system/pipeline_runner.py` provides a typed package-level run boundary with immutable run directories, redacted config export, injected generation/repair executors, canonical artifact subdirectories, overwrite protection, refresh-policy live-API disabling, strict-mode blocking after diagnostic artifact writes, `build_phase0_generation_executor()` for adapting existing Phase 0 evidence exports, `build_production_generation_executor()` for adapting the existing production optimizer output into `PipelineExecutionResult`, and `build_progressive_repair_executor()` for adapting the REPAIR-005 progressive controller into child plan, diff, certificate, routing, explanation, metrics, and dashboard artifacts without notebook dependency. `src/itinerary_system/benchmark/disruptions.py` now provides deterministic six-family disruption scenarios and repair requests with explicit observed/synthetic status. `run_benchmark_suite()` now runs injected method adapters over identical frozen disruption scenarios, enforces parent-plan/disruption-family split isolation, and exports long-form benchmark metrics plus a run manifest. `build_pipeline_benchmark_method_adapter()` now lets benchmark methods execute through `run_research_pipeline()` and load the resulting artifacts back into benchmark rows. Notebook migration remains open Phase 4.0 work, and publication claims still require complete validated benchmark matrices from later provider/pipeline runs.

This feature should allow the system to:

1. Run generation and repair through one package-level pipeline runner.
2. Produce immutable run directories with requests, plans, diffs, routing, evaluations, explanations, and metrics.
3. Run reproducible disruption benchmarks and keep notebooks as thin display layers.

Keep the implementation modular, easy to test, and consistent with the existing project structure.

## Current State

The project already has:

* `src/itinerary_system/experiment_runner.py`: large notebook-oriented production pipeline helper.
* `notebook/production_system_blueprint.ipynb`: current production notebook execution path.
* `scripts/run_phase0_evidence_pipeline.py`: Phase 0 evidence pipeline.
* `scripts/validate_phase0_artifacts.py`: strict Phase 0 validation.
* `README.md`: documents notebook execution and dashboard serving.
* `docs/planning/travel_itinerary_repair_technical_specification.md`: defines `run_research_pipeline()` target interface.

The missing part is:

* Paired benchmark method runner.
* Notebook cells that call package functions instead of carrying business logic.

## New Components to Add

### Component 1

`ResearchPipelineRunner`

Responsibility:

Resolve config, load snapshots, execute generation or repair, route selected legs, evaluate, explain, export manifests, and never overwrite run directories.

### Component 2

`DisruptionGenerator`

Responsibility:

Create deterministic weather, road closure, hotel unavailability, attraction closure, must-go conflict, and pace/accessibility disruption requests.

### Component 3

`BenchmarkRunner`

Responsibility:

Run paired methods on identical snapshots, enforce no leakage, and export long-form metrics.

### Component 4

`ThinNotebookAdapter`

Responsibility:

Let notebooks call the package pipeline and load completed outputs for display only.

## Class / Registry Diagrams

```text
+-------------------------------------------------------------------------------+
|                            ResearchPipelineRunner                               |
+-------------------------------------------------------------------------------+
|  - config_path: Path                                                           |
|  - output_root: Path                                                           |
|  - clock: Callable[[], datetime]                                               |
+-------------------------------------------------------------------------------+
|  + run(config): PipelineRun                 --> Executes canonical pipeline    |
|  + create_run_dir(run_id): Path             --> Immutable output directory     |
|  + write_manifest(run): Path                --> Records hashes and outputs     |
+-------------------------------------------------------------------------------+
```

```text
+-------------------------------------------------------------------------------+
|                              DisruptionGenerator                                |
+-------------------------------------------------------------------------------+
|  - seed: int                                                                   |
|  - catalog_snapshot_id: str                                                    |
|  - context_snapshot_id: str                                                    |
+-------------------------------------------------------------------------------+
|  + weather(): RepairRequest                --> Weather deterioration case      |
|  + closure(): RepairRequest                --> Road/attraction closure case    |
|  + hotel_unavailability(): RepairRequest   --> Lodging disruption case         |
|  + pace_accessibility(): RepairRequest     --> User-approved pace case         |
+-------------------------------------------------------------------------------+
```

```text
+-------------------------------------------------------------------------------+
|                                BenchmarkRunner                                  |
+-------------------------------------------------------------------------------+
|  - methods: tuple[BenchmarkMethod, ...]                                        |
|  - evaluator: PlanEvaluator                                                    |
+-------------------------------------------------------------------------------+
|  + run_suite(scenarios): BenchmarkResult    --> Paired method comparison       |
|  + check_no_leakage(result): None           --> Validates split isolation      |
|  + export_metrics(result): Path             --> Writes long-form table         |
+-------------------------------------------------------------------------------+
```

```text
+-------------------------------------------------------------------------------+
|                              ThinNotebookAdapter                                |
+-------------------------------------------------------------------------------+
|  - No persistent internal state                                                |
+-------------------------------------------------------------------------------+
|  + run_from_notebook(config_path): PipelineRun --> Calls package runner        |
|  + load_completed_run(run_dir): dict        --> Loads outputs for display      |
+-------------------------------------------------------------------------------+
```

## Class Diagram Rules

1. Pipeline orchestration belongs in `src/itinerary_system/pipeline_runner.py`.
2. Notebook cells may call package functions and display outputs only.
3. Benchmark components must use identical frozen snapshots across methods.
4. Generated run artifacts must be written under run/output directories, not source folders.
5. A run directory is immutable; rerunning uses a new run ID or fails.
6. Skip user-study logging until computational benchmark outputs are stable.

## Data Model

```python
@dataclass(frozen=True)
class PipelineRun:
    run_id: str
    data_bundle: DatasetBundle
    parent_plan: PlanArtifactV2 | None
    planner_runs: tuple[PlannerRun, ...]
    output_plans: tuple[PlanArtifactV2, ...]
    evaluations: tuple[ResearchEvaluationReport, ...]
    output_dir: Path

@dataclass(frozen=True)
class BenchmarkResult:
    benchmark_id: str
    scenario_ids: tuple[str, ...]
    method_ids: tuple[str, ...]
    run_ids: tuple[str, ...]
    metrics_path: Path
    manifest_path: Path
```

## Storage / State

Persistent state:

```text
runs/<run_id>/
|-- manifest.json
|-- resolved_config.redacted.json
|-- dataset_manifest.json
|-- requests/
|-- planner_runs.jsonl
|-- plans/
|-- diffs/
|-- routing/
|-- evaluations/
|-- explanations/
|-- metrics/
`-- dashboard/
```

Temporary state:

* Intermediate solver/model objects remain in memory only.

## Required Methods

```python
def run_research_pipeline(
    *,
    config_path: Path,
    catalog_snapshot_id: str,
    context_snapshot_id: str,
    parent_plan_id: str | None = None,
    repair_request_id: str | None = None,
    refresh_policy: RefreshPolicy = RefreshPolicy.NEVER,
    run_id: str | None = None,
) -> PipelineRun

def generate_disruption_requests(parent_plan: PlanArtifactV2, bundle: DatasetBundle, seed: int) -> tuple[RepairRequest, ...]
def run_benchmark_suite(
    *,
    scenarios: tuple[DisruptionScenario, ...],
    methods: tuple[BenchmarkMethodAdapter, ...],
    output_dir: Path,
) -> BenchmarkResult
def migrate_notebook_to_pipeline(notebook_path: Path) -> None
```

## Validation Rules

1. No run directory overwrite.
2. Resolved config must be redacted before serialization.
3. Refresh policy `never` must prevent live API calls.
4. Every output plan must have run, route, evaluation, and manifest entries.
5. Strict mode must block ineligible plans.
6. Benchmark methods must share identical catalog/context/parent/disruption inputs.
7. Notebooks may not construct optimizer models, call external APIs, or mutate canonical snapshots.

## UI / API Integration

Internal and notebook integration:

* `README.md` should prefer the pipeline runner once implemented.
* Existing notebook command remains available during migration.
* `scripts/serve_dashboard.py` can keep serving exported dashboard outputs.
* Dashboard comparison views should load from run artifacts after migration.

## Workflow

1. Resolve config and create immutable run directory.
2. Load and validate catalog/context snapshots.
3. Resolve providers according to refresh policy.
4. Load or generate parent plan.
5. Run generation or repair.
6. Build route results from route matrix.
7. Evaluate and certificate final plans.
8. Generate diff/explanation evidence.
9. Export manifest, metrics, and dashboard inputs.
10. For benchmarks, repeat paired methods over frozen scenarios and export long-form metrics.

## Files to Create

* `src/itinerary_system/pipeline_runner.py`
* `src/itinerary_system/benchmark/__init__.py`
* `src/itinerary_system/benchmark/disruptions.py`
* `src/itinerary_system/benchmark/runner.py`
* `src/itinerary_system/benchmark/splits.py`
* `src/itinerary_system/benchmark/metrics.py`
* `src/itinerary_system/benchmark/methods.py`
* `tests/test_pipeline_runner.py`
* `tests/benchmark/test_disruptions.py`
* `tests/benchmark/test_no_leakage.py`
* `tests/benchmark/test_method_adapters.py`

## Files to Modify

* `src/itinerary_system/experiment_runner.py`
* `src/itinerary_system/__init__.py`
* `notebook/production_system_blueprint.ipynb`
* `README.md`
* `docs/reference/code_quality_workflow.md`

## Error Handling

* Existing run directory raises `RunDirectoryExists`.
* Missing snapshot raises typed snapshot error.
* Ineligible plan in strict mode raises pipeline failure after writing diagnostic artifacts.
* Provider disabled or missing credentials records unavailable status and sends no normal request.
* Notebook migration keeps old command working until the new CLI/pipeline path is verified.

## Testing Checklist

- [ ] Pipeline creates immutable run directory.
- [ ] Pipeline refuses to overwrite existing run.
- [ ] Resolved config is redacted.
- [ ] Refresh policy `never` sends no live API requests.
- [ ] Generation mode emits plan/run/evaluation/manifest.
- [ ] Repair mode emits parent/child/diff/evaluation/manifest.
- [ ] Strict mode blocks ineligible plan.
- [ ] Six disruption families have deterministic IDs.
- [x] Benchmark paired methods use identical frozen inputs.
- [ ] Notebook calls package runner and contains no business logic.

## Roadmap / Full Pipeline Update

After implementation and full validation, update `PIPE-001`, `NOTEBOOK-001`, `BENCH-001`, and `BENCH-002` status in the technical specification. Do not claim publication readiness until benchmark results and evaluator gates pass.

## Acceptance Criteria

1. A package-level pipeline can reproduce generation and repair runs.
2. Run artifacts are immutable, complete, and redacted.
3. Benchmarks compare methods on identical frozen inputs.
4. Notebooks are display wrappers over package outputs.
5. Strict validation prevents ineligible plans from being ranked or published.
