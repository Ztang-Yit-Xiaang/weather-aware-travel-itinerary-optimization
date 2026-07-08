# Context-Aware, Inspectable Itinerary Repair Detailed Phase Plan

Prepared: 2026-07-06
Repository: `Ztang-Yit-Xiaang/weather-aware-travel-itinerary-optimization`
Status: planning only; no code implementation is complete because of this document.

## Executive Summary

This project should be framed as **Context-Aware, Inspectable Itinerary Repair**. It is not TripTide. TripTide is a related disruption benchmark and threat paper in the literature, not the system name.

The core research thesis is:

> Repair, do not regenerate. Treat the accepted itinerary as a persistent parent artifact. After a disruption or user edit, produce a child itinerary that preserves locked, booked, strong-preference, weak-preference, and flexible commitments as much as possible, expands the repair neighborhood only when needed, solves preservation before utility, independently evaluates the final plan, and explains the repair using evidence from plan diffs, constraints, route records, context snapshots, solver stages, and counterfactual attempts.

The immediate priority is to fix the repository truth layer, canonical parent/child plan semantics, road-valid routing context, independent evaluation, and typed repair diff before building more solver, benchmark, UI, or study claims. The actual gate order is:

1. G0 - Repository truth and Phase 0 closeout.
2. G1 - Canonical parent/child plans and typed plan diff.
3. G2 - Road-valid routing context.
4. G3 - Single-day ownership-aware repair.
5. G4 - Progressive multi-day repair.
6. G5 - Benchmark package.
7. G6 - Authoritative pipeline runner and interactive prototype.
8. G6.5 - Evidence-bounded LLM preference-to-commitment layer.
9. G7 - Explanation and study readiness.
10. G8 - IUI/CHI extension readiness.

G6.5 is a work package, not a replacement for the core repair architecture. The LLM must not be the itinerary generator, optimizer, feasibility judge, route validator, hotel-availability judge, or final evaluator. Its role is a bounded mixed-initiative layer: interpret repair language, infer inspectable taste hypotheses from the accepted parent itinerary, suggest inactive commitments, ask targeted clarification questions, verbalize already-valid evidence, and translate why-not/what-if questions into structured counterfactual requests.

No implementation checklist item should be marked complete yet unless it was already completed and verified before this planning task.

## Repository-Grounded Current State Summary

The repository already has a Phase 0 research foundation, but it is not yet a complete parent-child itinerary repair system.

Planning and current-state documents inspected:

- `docs/planning/research_stabilization_and_publication_plan.md`: identifies the truthful artifact chain, requested/executed method lineage, independent evaluator need, and post-solve mutation blocker.
- `docs/planning/travel_itinerary_repair_technical_specification.md`: defines the target artifact, ownership, lodging, routing, repair, evaluator, explanation, pipeline, benchmark, and notebook contracts.
- `docs/planning/current_problem_fix_phase_plans.md`: documents near-term problem-fix phases and live validation results.
- `docs/current/current_score_audit.md`: warns that current utility is a transparent proxy, that geodesic travel is not road-valid, and that current "repair" naming is not parent-child repair.
- `docs/reference/data_dictionary.md`: defines catalog/context separation, routing eligibility, route cache artifacts, and source coverage semantics.
- `docs/reference/code_quality_workflow.md`: documents Ruff, pytest, coverage, dashboard validation, and generated-artifact policy.
- `README.md` and `docs/README.md`: describe the current California-focused prototype, notebook pipeline, dashboards, and documentation index.
- `doc/`: checked and not present.

Literature documents inspected:

- `docs/literature/repair_gap_review.md`: canonical repair-gap framing and safe contribution claim.
- `docs/literature/literature_matrix_repair_gap.md`: compact closest-work matrix showing empty cells around ownership-aware parent-child repair.
- `docs/literature/evidence_matrix.md`: current implementation versus literature-supported claims.
- `docs/literature/literature_onboarding_guide.md`: plain-language literature structure and next-stage roadmap.
- `docs/literature/core_paper_reading_cards.md`: integrated eight-paper spine covering TTDP, TravelEval, TRIP-PAL, TripTide, and explanation.
- `docs/literature/literature_deep_read_study_report.md`: detailed evidence bank and claim cautions.

Source and test evidence inspected:

- `src/itinerary_system/repository_state.py`: `RepositoryState` and `capture_repository_state()` already exist.
- `src/itinerary_system/research_artifacts.py`: `PlannerRun`, `PlanArtifact` v1, `ResearchEvaluationReport`, and `evaluate_phase0_plan()` exist.
- `src/itinerary_system/data/schemas.py`, `src/itinerary_system/data/snapshot.py`, `src/itinerary_system/data/context.py`: catalog/context/dataset bundle and validation scaffolds exist.
- `src/itinerary_system/routing/models.py`, `src/itinerary_system/routing/cache.py`, `src/itinerary_system/routing/road_cache_builder.py`: road-validation contracts, road-route cache, OSRM cache builder, public OSRM policy gate, and geodesic fallback labeling exist.
- `src/itinerary_system/phase0_exporter.py`, `scripts/run_phase0_evidence_pipeline.py`, `scripts/validate_phase0_artifacts.py`, `scripts/summarize_phase0_readiness.py`, `scripts/check_route_source.py`: Phase 0 export, validation, readiness, and route-source checks exist.
- `src/itinerary_system/repair_planner.py`: deterministic repair scaffold exists with `ParsedRepairIntent`, `RepairRequest`, `RepairPlan`, repair operations, evidence conflict detection, and simple evaluation, but it is not an ownership-aware optimization model.
- `src/itinerary_system/multi_objective_route.py`: route optimizer uses geodesic travel proxies internally and solves generation/selection, not parent-child repair.
- `src/itinerary_system/hierarchical_gurobi.py`: hierarchical route generation still uses geodesic drive proxies and fallback enumeration.
- `src/itinerary_system/route_gurobi_oracle.py`: `solve_enriched_route_with_gurobi()` returns `solve_multi_objective_route()` before a legacy block, leaving unreachable old code.
- `src/itinerary_system/utility_model.py`: keeps source coverage and model uncertainty separate in some fields, but missing Yelp-like signals can still behave as numeric zero in utility fusion paths.
- `src/itinerary_system/request_schema.py`: `TripPlanningRequest` supports structured interest/config overrides, but there is no full repair/edit request schema.
- `tests/test_repository_state.py`, `tests/test_research_foundation.py`, `tests/test_repair_planner.py`, `tests/data/test_context_snapshot.py`: current tests cover repository identity, Phase 0 artifact gating, road fallback ineligibility, public OSRM policy, deterministic repair scaffold, and separated context snapshots.

Current missing pieces:

- No canonical `PlanArtifact` v2 with owned constraints, route IDs, certificate ID, immutable content hash, and parent-child migration.
- No `PlanRepository`.
- No generalized `PlanDiff`.
- No ownership policy layer with locked/booked/strong/weak/test-only semantics.
- No route-matrix provider abstraction or OSRM Table/Route/Nearest client.
- No road-valid travel injection into publication solvers.
- No provider-independent lodging package.
- No missing-source-aware utility fusion.
- No ownership-aware repair master model.
- No sequential lexicographic solver trace.
- No progressive multi-day controller.
- No independent final-plan evaluator/certificate layer beyond Phase 0 eligibility.
- No structured explanation evidence layer.
- No LLM preference-to-commitment layer that converts parent-plan evidence into inactive, user-confirmable commitment suggestions.
- No six-disruption benchmark package.
- No authoritative package-level `run_research_pipeline()`.
- Notebook and dashboard paths are still artifact-oriented and notebook-era.

## Claim Guardrails From The Literature

Safe contribution claim:

> This project combines ownership-labeled commitments, progressive repair neighborhoods, lexicographic preservation-before-utility objectives, independent validation, and evidence-grounded explanations for user-owned itinerary repair under localized disruptions.

Claims to avoid until later evidence exists:

- First disruption-aware itinerary system.
- First itinerary modification system.
- First LLM travel planner.
- First weather-aware route planner.
- First explainable optimizer.
- First scenic route planner.
- Real-time hotel booking or availability.
- Road-valid evaluation when geodesic fallback is used.
- Online bandit personalization without real sequential feedback.
- Calibrated uncertainty when the field is only source coverage.
- Accurate real-world congestion when demand is simulated or proxied.

Literature framing:

- OP/TTDP supplies the optimization backbone, but not persistent parent-child repair with ownership-weighted typed edits.
- Dynamic/RL routing supplies disruption/reoptimization background, but not human-owned itinerary preservation.
- LLM travel planning supplies language interfaces and benchmarks, but not independent route certification or lexicographic preservation objectives.
- LLM preference inference is related prior art; the project-specific contribution is using bounded taste hypotheses to support editable repair commitments rather than direct itinerary generation.
- iTIMO and TripTide make "first modification" and "first disruption benchmark" claims unsafe.
- TravelEval and TripScore supply evaluation vocabulary, but not a repair mechanism.
- Explainable optimization supports sensitivity, infeasibility, and counterfactual explanation, but not this travel-specific diff/certificate architecture.

## Implementation-Order Locks

1. Do not implement repair master logic before canonical `PlanArtifact`, `PlanDiff`, and independent evaluator contracts exist.
2. Do not make benchmark, transportation, or publication claims until local or pinned road-valid routing evidence passes strict validation.
3. Do not start UI or user-study work until pipeline, diff, repair, evaluator, and explanation contracts are stable enough to replay from artifacts.
4. Preserve existing Phase 0 script names and artifact names unless a compatibility wrapper is provided.
5. Treat public OSRM as disabled by default. Use local OSRM or pinned route evidence unless a human explicitly approves another routing source for a non-publication run.
6. Geodesic or straight-line fallback is never `road_validated=True`.
7. A plan modified after solve is not solver-certified until a new independent evaluation.
8. Do not put business logic in notebooks.
9. Do not make live API calls inside optimization model builders.
10. Parent plans are immutable. Any change creates a child plan with a new ID and content hash.
11. LLM output is inactive until deterministic validation and user confirmation produce solver-safe `OwnedConstraint` records.
12. Inferred taste alone cannot create `LOCKED` or `BOOKED` constraints; locked needs explicit user lock evidence, and booked needs booking evidence.
13. LLM explanations and counterfactual answers must cite structured artifact evidence; unsupported prose is rejected or hidden.

## Master Artifact Flow

```text
+-------------------+      +-------------------+      +----------------------+
| RepositoryState   | ---> | CatalogBundle     | ---> | DatasetBundle        |
| PlannerRun        |      | ContextBundle     |      | RouteMatrix          |
+-------------------+      +-------------------+      +----------------------+
          |                         |                           |
          v                         v                           v
+-------------------+      +-------------------+      +----------------------+
| Parent PlanArtifact| ---> | RepairRequest     | ---> | PlannerRun attempts  |
| OwnedConstraints   |      | DisruptionSnapshot|      | SolverStageRecords   |
+-------------------+      +-------------------+      +----------------------+
          |                                                     |
          v                                                     v
+-------------------+      +-------------------+      +----------------------+
| TasteEvidencePack | ---> | SuggestedCommitment| ---> | UserCommitmentDecision|
| PromptRun records |      | inactive labels   |      | confirmed constraints |
+-------------------+      +-------------------+      +----------------------+
          |                                                     |
          v                                                     v
+-------------------+      +-------------------+      +----------------------+
| Child PlanArtifact | ---> | PlanDiff          | ---> | ResearchEvaluation   |
| immutable hash     |      | weighted edits    |      | EvaluationCertificate|
+-------------------+      +-------------------+      +----------------------+
          |                                                     |
          v                                                     v
+-------------------+      +--------------------------------------------+
| ExplanationEvidence| ---> | dashboard / notebook / benchmark reports  |
| claim refs         |      | consume artifacts only                     |
+-------------------+      +--------------------------------------------+
```

## Roadmap Update Note

This document is a planning deliverable. It creates implementation-ready work packages but does not complete any package. If future Codex sessions modify `ROADMAP.md`, `roadmap.md`, `docs/planning/*`, `docs/*pipeline*`, or `CODEX_EDIT_LOG.md`, they must not change `[ ]` to `[x]` unless implementation and tests for that item are actually complete.

# Phase G0.0 Implementation Plan: Repository Truth And Phase 0 Closeout

## Goal

Implement **repository truth and Phase 0 closeout**.

This gate should allow the system to:

1. Preserve existing Phase 0 script names and artifact names.
2. Make repository state, run lineage, route-source status, and final-comparison readiness explicit.
3. Close current truthfulness blockers before any parent-child repair implementation begins.

Keep the implementation modular, easy to test, and consistent with the existing `src/itinerary_system` package.

## Current State

Repository evidence found:

- `src/itinerary_system/repository_state.py`: already defines `RepositoryState`, `RepositoryStateUnavailable`, `capture_repository_state()`, environment overrides, dirty flag capture, and package version capture.
- `src/itinerary_system/research_artifacts.py`: already defines `PlannerRun`, `PlanArtifact` v1, `ResearchEvaluationReport`, `stable_content_hash()`, and `evaluate_phase0_plan()`.
- `src/itinerary_system/phase0_exporter.py`: writes `production_phase0_dataset_validation.json`, `production_phase0_planner_runs.csv`, `production_phase0_plan_artifacts.jsonl`, `production_phase0_route_audit.csv`, `production_phase0_evaluation_reports.csv`, and `production_phase0_evidence_summary.csv`.
- `scripts/run_phase0_evidence_pipeline.py`: runs route-cache build, Phase 0 export, validation, and readiness summary.
- `scripts/validate_phase0_artifacts.py`: validates Phase 0 output and blocks false eligible route claims.
- `scripts/summarize_phase0_readiness.py`: reports strict comparison readiness.
- `scripts/build_road_route_cache.py` and `scripts/check_route_source.py`: preserve local OSRM default and public OSRM opt-in policy.
- `tests/test_research_foundation.py`: verifies route fallback ineligibility, public OSRM policy, route cache conversion, Phase 0 strict mode, and readiness summaries.
- `docs/planning/research_stabilization_and_publication_plan.md`: names post-solve mutation and lineage as blockers.
- `docs/planning/current_problem_fix_phase_plans.md`: records prior validation: Ruff passed; pytest had four temp-permission failures in `tests/data/test_context_snapshot.py`.

Missing pieces:

- Phase 0 artifacts are still flat production outputs, not immutable `runs/<run_id>/` artifacts.
- `PlanArtifact` is v1 and lacks v2 fields needed by repair, diff, certificate, and ownership.
- Post-solve mutation detection is not centralized.
- Fallback lineage is present in fields but not yet governed by one recorder.
- Road-valid route evidence is available by cache, but current demo snapshots still include non-road-validated fallback route options, making final comparison ineligible unless a complete road cache is built.
- No canonical "Phase 0 closeout checklist" exists as a single source of truth.

What should not be claimed or implemented yet:

- Do not claim road-valid transportation evidence unless `scripts/run_phase0_evidence_pipeline.py --require-final-eligible` passes with local or pinned road evidence.
- Do not implement the repair master model in this gate.
- Do not rename `production_phase0_*` artifacts without compatibility wrappers and tests.

## New Components to Add

### Component 1

`Phase0CloseoutReport`

Responsibility:

Summarize repository state, dataset state, route-source state, Phase 0 artifact hashes, strict-readiness status, current blockers, and next gate eligibility.

### Component 2

`PlannerRunRecorder`

Responsibility:

Centralize creation of requested/executed method lineage records for primary solver attempts, fallback attempts, skipped attempts, failed attempts, and post-processing child runs.

### Component 3

`PostSolveMutationGate`

Responsibility:

Compare candidate solver output with final displayed output. If material content changed, either create a child run/plan or mark the old solver certificate invalid.

### Component 4

`Phase0CompatibilityAdapter`

Responsibility:

Keep existing Phase 0 CSV/JSONL artifacts readable while allowing v2 plan/run fields to be added gradually.

## Class / Registry Diagrams

```text
+-------------------------------------------------------------------------------+
|                         Phase0CloseoutReport                                   |
+-------------------------------------------------------------------------------+
|  - repository_state: RepositoryState                                           |
|  - dataset_report: DatasetValidationReport                                     |
|  - readiness: dict[str, Any]                                                   |
|  - blocker_codes: tuple[str, ...]                                              |
+-------------------------------------------------------------------------------+
|  + to_record(): dict[str, Any]              --> Serialize closeout state       |
|  + strict_ready(): bool                     --> True only when final eligible  |
|  + write_markdown(path: Path): Path         --> Human-readable closeout        |
+-------------------------------------------------------------------------------+
```

```text
+-------------------------------------------------------------------------------+
|                             PlannerRunRecorder                                  |
+-------------------------------------------------------------------------------+
|  - planner_specification_id: str                                               |
|  - clock: Callable[[], datetime]                                               |
+-------------------------------------------------------------------------------+
|  + start_run(request_id, method): PlannerRun --> Create attempt record         |
|  + complete_run(run, status): PlannerRun     --> Add solver status/cert        |
|  + fallback_run(parent, reason): PlannerRun  --> Record executed fallback      |
|  + failed_run(parent, exc): PlannerRun       --> Record failed attempt         |
+-------------------------------------------------------------------------------+
```

```text
+-------------------------------------------------------------------------------+
|                            PostSolveMutationGate                                |
+-------------------------------------------------------------------------------+
|  - No persistent internal state                                                |
+-------------------------------------------------------------------------------+
|  + compare(before, after): MutationReport   --> Identify material edits        |
|  + requires_child_run(report): bool         --> True for displayed changes     |
|  + invalidate(run, reason): PlannerRun      --> Mark solver cert invalid       |
+-------------------------------------------------------------------------------+
```

```text
+-------------------------------------------------------------------------------+
|                         Phase0CompatibilityAdapter                              |
+-------------------------------------------------------------------------------+
|  - No persistent internal state                                                |
+-------------------------------------------------------------------------------+
|  + plan_to_phase0_row(plan): dict           --> Existing JSONL-compatible row  |
|  + plan_from_phase0_row(row): PlanArtifact  --> Read old records               |
|  + enrich_v2_defaults(row): dict            --> Add safe v2 defaults           |
+-------------------------------------------------------------------------------+
```

## Class Diagram Rules

1. Create one diagram block for every new closeout, recorder, adapter, or mutation component.
2. Keep persistent artifact state in dataclasses and JSON/CSV outputs, not in module globals.
3. List only fields that influence artifact identity, eligibility, or strict readiness.
4. Mark stateless utilities with `No persistent internal state`.
5. Compatibility code must read old Phase 0 artifacts without changing their meaning.
6. Skip run-directory migration until G6; G0 only closes the current Phase 0 truth layer.

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
class Phase0CloseoutReport:
    closeout_id: str
    repository_state: RepositoryState
    catalog_snapshot_id: str
    context_snapshot_id: str
    phase0_artifact_hashes: dict[str, str]
    strict_comparison_ready: bool
    blocker_codes: tuple[str, ...]
    warnings: tuple[str, ...] = ()
    schema_version: str = "phase0-closeout-v1"
```

Existing models to preserve:

- `RepositoryState`
- `PlannerRun`
- `PlanArtifact`
- `ResearchEvaluationReport`
- `RouteLegResult`
- `RouteResult`
- `DatasetBundle`
- `DatasetValidationReport`

## Storage / State

Persistent state:

- Existing `results/outputs/production_phase0_*` files remain the Phase 0 compatibility artifacts.
- Add `results/quality/phase0_closeout_report.md` and `results/quality/phase0_closeout_report.json` when generated.
- Later G6 will create `runs/<run_id>/`; do not require that layout here.

Temporary state:

- Route-cache build temp state remains in existing cache directories.
- No hidden in-memory registry should outlive one command.

## Required Methods

```python
def collect_phase0_artifact_hashes(output_dir: Path) -> dict[str, str]
def build_phase0_closeout_report(output_dir: Path, quality_dir: Path) -> Phase0CloseoutReport
def detect_post_solve_mutation(before: PlanArtifact, after: PlanArtifact) -> MutationReport
def mark_run_invalidated_after_edit(run: PlannerRun, reason: str) -> PlannerRun
def assert_phase0_compatibility(output_dir: Path) -> None
```

Preserve existing public commands:

```bash
python scripts/build_road_route_cache.py --output-dir results/outputs --require-complete
python scripts/check_route_source.py --output-dir results/outputs
python scripts/run_phase0_evidence_pipeline.py --config configs/default_trip_config.yaml --output-dir results/outputs --quality-dir results/quality
python scripts/validate_phase0_artifacts.py --output-dir results/outputs --require-final-eligible
python scripts/summarize_phase0_readiness.py --output-dir results/outputs --write-dir results/quality
```

## Validation Rules

1. `RepositoryState.commit_sha` must be real or explicit `"unknown"`; never fabricate a SHA.
2. Dirty worktree state must be recorded.
3. `PlannerRun.method_requested` and `PlannerRun.method_executed` must differ when fallback is used.
4. A post-solve material edit invalidates the solver certificate unless a new child run and evaluation are created.
5. Existing Phase 0 files must remain readable.
6. Strict final comparison requires route, solver, schedule, dataset, and road-validation eligibility.
7. A geodesic fallback route must produce `road_validated=False` and `comparison_eligibility="ineligible"`.
8. Public OSRM fetch remains blocked unless an explicit non-publication approval flag is provided.

## UI / API Integration

Internal only:

- `scripts/run_phase0_evidence_pipeline.py` can call the closeout report after readiness output is written.
- `README.md` can later link to the closeout command, but this gate does not change UI.
- Dashboards must continue consuming existing Phase 0-compatible artifacts.

## Workflow

1. Capture `RepositoryState`.
2. Run or read dataset/context validation through `load_dataset_bundle()` and `validate_dataset_bundle()`.
3. Build or validate `production_road_route_cache.csv`.
4. Export Phase 0 artifacts with existing `write_phase0_research_artifacts()`.
5. Validate Phase 0 artifacts.
6. Summarize strict readiness.
7. Hash Phase 0 outputs.
8. Write closeout report and current blockers.
9. Proceed to G1 only when parent/child artifact work can extend this foundation without breaking Phase 0.

## Files to Create

- `src/itinerary_system/phase0_closeout.py`
- `tests/test_phase0_closeout.py`

## Files to Modify

- `src/itinerary_system/research_artifacts.py`
- `src/itinerary_system/phase0_exporter.py`
- `scripts/run_phase0_evidence_pipeline.py`
- `scripts/summarize_phase0_readiness.py`
- `tests/test_research_foundation.py`
- `docs/reference/data_dictionary.md`
- `docs/planning/travel_itinerary_repair_technical_specification.md` after implementation only

## Error Handling

- Missing Phase 0 files: report as blocking `missing_phase0_artifact`.
- Hash mismatch: report as `phase0_artifact_hash_mismatch`.
- Unvalidated route: report as `route_not_road_validated`.
- Public OSRM without approval: raise the existing policy error and mark non-publication approval required.
- Post-solve mutation without child run: mark `INVALIDATED_AFTER_EDIT` and ineligible.
- Unknown repository state in strict mode: raise `RepositoryStateUnavailable`.

## Testing Checklist

- [ ] `RepositoryState` is embedded in closeout output.
- [ ] Existing `production_phase0_*` artifacts remain compatible.
- [ ] `python scripts/run_phase0_evidence_pipeline.py --require-final-eligible` passes with complete cached OSRM evidence.
- [ ] `python scripts/run_phase0_evidence_pipeline.py` reports blockers without strict failure when road cache is incomplete.
- [ ] Post-solve mutation invalidates solver certification.
- [ ] Fallback lineage records requested and executed methods separately.
- [ ] Public OSRM fetch remains blocked without explicit approval.
- [ ] `python -m ruff check src tests scripts`
- [ ] `python -m pytest`

## Roadmap / Full Pipeline Update

After implementation and validation pass, update Phase 0 status notes in `docs/planning/current_problem_fix_phase_plans.md` and the technical specification. Do not mark G1-G8 complete. This gate connects to G1 by making sure plan-v2 and diff work can reuse truthful run/source/evaluation lineage.

## Acceptance Criteria

1. Phase 0 scripts and artifact names are preserved.
2. Strict readiness clearly distinguishes demo/fallback from publication-eligible evidence.
3. Fallback lineage and post-solve mutation status are explicit.
4. Existing tests remain compatible.
5. The repository has a single closeout report that future gates can cite.
6. No parent-child repair, benchmark, transportation, UI, or publication claim is made beyond the validated Phase 0 evidence.

# Phase G1.0 Implementation Plan: Canonical Parent/Child Plans And Typed Plan Diff

## Goal

Implement **canonical parent/child plans and typed plan diff**.

This gate should allow the system to:

1. Treat the accepted itinerary as an immutable parent artifact.
2. Create child plans with new IDs, content hashes, source runs, and parent lineage.
3. Compute typed plan diffs and preservation metrics before any repair master logic exists.

Keep the implementation modular, easy to test, and consistent with existing dataclass style.

## Current State

Repository evidence found:

- `src/itinerary_system/research_artifacts.py`: `PlanArtifact` v1 stores `plan_id`, `source_run_id`, `planning_request_id`, snapshots, selected stops, day assignments, sequence, lodging assignments, modeled metrics, exposure components, change components, and content hash.
- `docs/planning/travel_itinerary_repair_technical_specification.md`: specifies `PlanArtifact` v2, `PlanRepository`, ownership schemas, and `PlanDiff`.
- `src/itinerary_system/repair_planner.py`: has route hash and simple repair operations, but not canonical plan objects or typed ownership diff.
- `tests/test_research_foundation.py`: builds v1 `PlanArtifact` fixtures.
- `docs/reference/data_dictionary.md`: states material post-processing must create a new plan.

Missing pieces:

- No `src/itinerary_system/plans/` package.
- No `PlanRepository`.
- No `PlanArtifact` v2 or migration wrapper.
- No `PlanDiff` class.
- No ownership vocabulary implementation.
- No `EvaluationCertificate` ID on plans.
- No immutable append-only storage.
- No canonical parent fixtures such as `plan_demo_current.json` or `plan_california_repair_v1.json`.

What should not be claimed or implemented yet:

- Do not solve repair optimization here.
- Do not compute benchmark metrics beyond diff/preservation primitives.
- Do not use LLM-interpreted constraints unless confirmed and represented as inactive by default before confirmation.

## New Components to Add

### Component 1

`PlanArtifactV2`

Responsibility:

Extend plan artifacts with ordered days, route IDs by day, owned constraints, certificate ID, immutable content hash, parent plan ID, and v1 compatibility.

### Component 2

`PlanRepository`

Responsibility:

Append-only persistent registry for immutable plans. Reject duplicate IDs with different content and expose parent-child lineage.

### Component 3

`OwnedConstraint`

Responsibility:

Represent user, booking, LLM, solver, external-data, and experiment constraints with origin, strength, scope, confirmation, relaxation policy, and evidence refs.

### Component 4

`OwnershipPolicy`

Responsibility:

Classify active constraints, compute typed change costs, enforce booked-relaxation permission rules, and reject test-only constraints in accepted plans.

### Component 5

`PlanDiff`

Responsibility:

Compute additions, deletions, replacements, moves, reorders, time shifts, lodging changes, road/corridor changes, constraint relaxations, unchanged days, and weighted edit cost.

## Class / Registry Diagrams

```text
+-------------------------------------------------------------------------------+
|                              PlanArtifactV2                                     |
+-------------------------------------------------------------------------------+
|  - plan_id: str                                                                |
|  - parent_plan_id: str | None                                                  |
|  - source_run_id: str                                                          |
|  - catalog_snapshot_id: str                                                    |
|  - context_snapshot_id: str                                                    |
|  - ordered_days: tuple[DayPlan, ...]                                           |
|  - lodging_assignments: dict[str, str]                                         |
|  - route_ids_by_day: dict[int, str]                                            |
|  - owned_constraints: tuple[OwnedConstraint, ...]                              |
|  - certificate_id: str | None                                                  |
+-------------------------------------------------------------------------------+
|  + to_record(): dict[str, Any]              --> Stable JSON record             |
|  + content_hash: str                        --> Hash excluding hash field      |
|  + from_v1(plan: PlanArtifact): PlanArtifactV2 --> Compatibility migration     |
+-------------------------------------------------------------------------------+
```

```text
+-------------------------------------------------------------------------------+
|                    PlanRepository (Persistent Registry / Manager)              |
+-------------------------------------------------------------------------------+
|  - root_dir: Path                                                              |
|  - index_path: Path                                                            |
+-------------------------------------------------------------------------------+
|  + save(plan: PlanArtifactV2): Path         --> Append immutable plan          |
|  + load(plan_id: str): PlanArtifactV2       --> Read by ID                     |
|  + children(parent_id: str): tuple[str, ...] --> Parent-child lineage          |
|  + assert_immutable(plan): None             --> Reject conflicting content     |
+-------------------------------------------------------------------------------+
```

```text
+-------------------------------------------------------------------------------+
|                                  OwnedConstraint                                |
+-------------------------------------------------------------------------------+
|  - constraint_id: str                                                          |
|  - origin: ConstraintOrigin                                                    |
|  - strength: ConstraintStrength                                                |
|  - scope: ConstraintScope                                                      |
|  - target_id: str                                                              |
|  - confirmed: bool                                                             |
|  - relaxation_policy: RelaxationPolicy                                         |
+-------------------------------------------------------------------------------+
|  + active(): bool                           --> LLM unconfirmed is inactive    |
|  + to_record(): dict[str, Any]              --> Stable JSON record             |
+-------------------------------------------------------------------------------+
```

```text
+-------------------------------------------------------------------------------+
|                       OwnershipPolicy (Stateless Utility Module)                |
+-------------------------------------------------------------------------------+
|  - No persistent internal state                                                |
+-------------------------------------------------------------------------------+
|  + is_active(c): bool                       --> Apply confirmation rules       |
|  + can_relax(c, request): bool              --> Locked/booked permission gate  |
|  + change_cost(change, constraints): float  --> Ownership-weighted edit cost   |
|  + validate_plan_constraints(plan): tuple[Finding, ...] --> Hard checks       |
+-------------------------------------------------------------------------------+
```

```text
+-------------------------------------------------------------------------------+
|                            PlanDiff (Stateless Utility Module)                  |
+-------------------------------------------------------------------------------+
|  - No persistent internal state                                                |
+-------------------------------------------------------------------------------+
|  + compute(parent, child, policy): PlanDiff --> Typed parent-child diff        |
|  + weighted_cost(): float                   --> Preservation metric            |
|  + unchanged_days(): tuple[int, ...]        --> Locality metric                |
|  + evidence_refs(): tuple[str, ...]         --> Explanation-ready refs         |
+-------------------------------------------------------------------------------+
```

## Class Diagram Rules

1. Diagram every persistent registry, artifact dataclass, policy utility, and diff module.
2. Do not include solver state in plan models.
3. Include only identity, lineage, ownership, route, lodging, and certification fields that define plan content.
4. Mark policy and diff as stateless utilities.
5. Keep v1 compatibility explicit and tested.
6. Skip `PlanRepository` only if the implementation must first extend Phase 0 JSONL, but then add a compatibility storage adapter.

## Data Model

```python
from enum import StrEnum

class ConstraintOrigin(StrEnum):
    USER = "user"
    USER_BOOKING = "user_booking"
    LLM_INTERPRETATION = "llm_interpretation"
    SOLVER = "solver"
    EXTERNAL_DATA = "external_data"
    EXPERIMENT = "experiment"

class ConstraintStrength(StrEnum):
    LOCKED = "locked"
    BOOKED = "booked"
    STRONG = "strong"
    WEAK = "weak"
    TEST_ONLY = "test_only"

class ConstraintScope(StrEnum):
    POI = "poi"
    DAY = "day"
    TIME = "time"
    LODGING = "lodging"
    ROAD = "road"
    REGION = "region"
    TRIP = "trip"

@dataclass(frozen=True)
class RelaxationPolicy:
    automatic_allowed: bool
    requires_user_permission: bool
    max_relaxation: float | None = None
    reason_required: bool = True

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
class StopRef:
    stop_id: str
    poi_id: str
    day: int
    sequence_index: int
    arrival_time: str | None = None
    departure_time: str | None = None

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
    owned_constraints: tuple[OwnedConstraint, ...]
    modeled_metrics: dict[str, float]
    change_components: dict[str, float]
    certificate_id: str | None
    created_at: str
    schema_version: str = "plan-artifact-v2"

@dataclass(frozen=True)
class PlanDiff:
    diff_id: str
    parent_plan_id: str
    child_plan_id: str
    parent_content_hash: str
    child_content_hash: str
    added_stops: tuple[StopChange, ...]
    deleted_stops: tuple[StopChange, ...]
    replaced_stops: tuple[ReplaceChange, ...]
    day_moves: tuple[DayMove, ...]
    time_shifts: tuple[TimeShift, ...]
    reorder_changes: tuple[OrderChange, ...]
    lodging_changes: tuple[LodgingChange, ...]
    road_changes: tuple[RoadChange, ...]
    constraint_relaxations: tuple[ConstraintRelaxation, ...]
    unchanged_days: tuple[int, ...]
    weighted_edit_cost: float
```

Typed changes to support:

- add stop;
- delete stop;
- replace stop;
- move stop to another day;
- reorder stops;
- time shift;
- lodging change;
- road/corridor change;
- constraint relaxation.

## Storage / State

Persistent state:

- `data/benchmark/parent_plans/plan_demo_current.json`
- `data/benchmark/parent_plans/plan_california_repair_v1.json`
- `data/benchmark/parent_plans/index.json`
- Later G6 mirrors records under `runs/<run_id>/plans/`.

Append-only rules:

- Parent plans are immutable.
- Saving an existing `plan_id` with the same content hash is idempotent.
- Saving an existing `plan_id` with a different content hash raises `PlanImmutabilityError`.
- Child plans must have a non-null `parent_plan_id`.

## Required Methods

```python
def plan_artifact_from_v1(plan: PlanArtifact) -> PlanArtifactV2
def canonical_plan_hash(plan: PlanArtifactV2) -> str
def validate_plan_artifact(plan: PlanArtifactV2, *, strict: bool = False) -> tuple[Finding, ...]
def save_plan(plan: PlanArtifactV2, repository: PlanRepository) -> Path
def compute_plan_diff(parent: PlanArtifactV2, child: PlanArtifactV2, policy: OwnershipPolicy) -> PlanDiff
def weighted_edit_cost(diff: PlanDiff, policy: OwnershipPolicy) -> float
def active_owned_constraints(plan: PlanArtifactV2) -> tuple[OwnedConstraint, ...]
```

## Validation Rules

1. `PlanArtifactV2.content_hash` changes when selected stops, order, time, lodging, route IDs, constraints, metrics, or certificate ID change.
2. Parent plans cannot be overwritten.
3. Child plan must reference a loadable parent.
4. Child plan snapshot IDs must match parent unless a documented context refresh creates a new child with new context lineage.
5. Unconfirmed `ConstraintOrigin.LLM_INTERPRETATION` constraints are inactive by default.
6. `ConstraintStrength.LOCKED` has no automatic relaxation.
7. `ConstraintStrength.BOOKED` requires request-level booked-relaxation permission.
8. `ConstraintStrength.TEST_ONLY` cannot be accepted into a user-owned plan unless separately confirmed or stored as experiment-only.
9. Diff identity case returns zero weighted edit cost.
10. Reordering must be detected even if selected stops are unchanged.
11. Missing Yelp/source signals must not be encoded as ownership constraints.

## UI / API Integration

Internal integration:

- `phase0_exporter.py` can migrate v1 plans into v2-compatible records.
- `repair_planner.py` should later accept `PlanArtifactV2` parent IDs rather than raw route tuples.
- `map_exporter.py` and dashboards should eventually consume `PlanDiff`, not ad hoc route comparison.

No external API calls are involved in this gate.

## Workflow

1. Add plan and ownership models.
2. Add v1-to-v2 migration.
3. Add append-only repository.
4. Create canonical parent plan fixtures from existing route artifacts or hand-reviewed seed plans.
5. Add typed diff computation.
6. Add identity and known-change fixtures.
7. Add preservation metric helpers.
8. Run validation.
9. Proceed to G2 only after route IDs and road context can attach to canonical plans.

## Files to Create

- `src/itinerary_system/plans/__init__.py`
- `src/itinerary_system/plans/models.py`
- `src/itinerary_system/plans/ownership.py`
- `src/itinerary_system/plans/repository.py`
- `src/itinerary_system/plans/diff.py`
- `src/itinerary_system/plans/errors.py`
- `data/benchmark/parent_plans/plan_demo_current.json`
- `data/benchmark/parent_plans/plan_california_repair_v1.json`
- `tests/plans/test_models.py`
- `tests/plans/test_ownership.py`
- `tests/plans/test_repository.py`
- `tests/plans/test_diff.py`

## Files to Modify

- `src/itinerary_system/research_artifacts.py`
- `src/itinerary_system/phase0_exporter.py`
- `src/itinerary_system/schemas.py`
- `src/itinerary_system/__init__.py`
- `tests/test_research_foundation.py`
- `docs/reference/data_dictionary.md`
- `docs/planning/travel_itinerary_repair_technical_specification.md` after implementation only

## Error Handling

- Duplicate plan ID with different content: `PlanImmutabilityError`.
- Missing parent: `PlanLineageError`.
- Unknown constraint enum value: `ConstraintValidationError`.
- Inactive LLM constraint used by solver/evaluator: hard validation failure.
- Booked constraint relaxation without permission: hard validation failure.
- Test-only constraint in accepted plan: hard validation failure unless experiment mode explicitly marks plan non-user-owned.
- Diff on unrelated plans without declared parent-child relationship: warning in permissive mode, error in strict mode.

## Testing Checklist

- [ ] v1 `PlanArtifact` migrates to v2 without losing Phase 0 fields.
- [ ] Content hash is stable across JSON round trips.
- [ ] Parent plan overwrite with different content fails.
- [ ] Child plan stores parent ID and parent hash.
- [ ] Unconfirmed LLM constraints are inactive.
- [ ] Locked constraints cannot be relaxed automatically.
- [ ] Booked constraints require explicit permission to relax.
- [ ] Test-only constraints cannot enter accepted plan.
- [ ] Identity diff has zero cost.
- [ ] Add/delete/replace/move/reorder/time/lodging/road/relax changes are detected.
- [ ] Weighted edit cost follows ownership strength.
- [ ] `python -m ruff check src tests scripts`
- [ ] `python -m pytest`

## Roadmap / Full Pipeline Update

After implementation and tests pass, update `PLAN-001`, `PLAN-002`, and `DIFF-001` status in the technical specification. This gate connects to G2 by giving route-matrix work stable plan stop IDs, day IDs, route IDs, and content hashes. Do not mark repair implementation complete.

## Acceptance Criteria

1. Parent and child plans are immutable, content-hashed, and lineage-linked.
2. Existing Phase 0 plan artifacts remain readable.
3. Owned constraints are typed and active/inactive rules are enforced.
4. `PlanDiff` computes typed changes and weighted edit cost.
5. Future repair and evaluator code can depend on stable plan/diff contracts.
6. No solver repair, benchmark, UI, or publication claim is made solely from this gate.

# Phase G2.0 Implementation Plan: Road-Valid Routing Context

## Goal

Implement **road-valid routing context**.

This gate should allow the system to:

1. Build and validate route matrices and selected route records from road evidence.
2. Use the same route/context snapshot in optimizer, evaluator, renderer, and benchmark.
3. Keep geodesic fallback available only as explicit non-publication/demo evidence.

Keep the implementation modular, easy to test, and consistent with the existing `routing` package.

## Current State

Repository evidence found:

- `src/itinerary_system/routing/models.py`: `RouteLegResult` and `RouteResult` already mark provider, source fields, fallback, and road validation.
- `src/itinerary_system/routing/cache.py`: `RoadRouteCache` loads `production_road_route_cache.csv`.
- `src/itinerary_system/routing/road_cache_builder.py`: builds validated route-cache rows from cached OSRM responses, preserves local OSRM default, and blocks public OSRM unless explicitly approved.
- `scripts/build_road_route_cache.py`: command wrapper for road cache build.
- `scripts/check_route_source.py`: manifest/policy precheck and optional probe.
- `src/itinerary_system/phase0_exporter.py`: uses road cache when available; otherwise creates explicit geodesic fallback legs.
- `docs/reference/data_dictionary.md`: states geodesic fallback is never road-valid and defines `production_road_route_cache.csv`, audit, and request artifacts.
- `src/itinerary_system/multi_objective_route.py`: computes travel with `geodesic(...).km * 1.25 / 38 * 60`.
- `src/itinerary_system/hierarchical_gurobi.py`: computes drive proxies with geodesic formulas.
- `src/itinerary_system/route_gurobi_oracle.py`: has geodesic helper and unreachable legacy model after early return.

Missing pieces:

- No `RoutingProvider` protocol.
- No `RouteMatrix`, `RouteMatrixCell`, `RouteMatrixRequest`, `RouteRequest`, or `NearestRequest`.
- No typed OSRM Table/Route/Nearest client.
- No local OSRM Docker assets.
- No route-source validation object shared by optimizer/evaluator/renderer.
- Solvers still use geodesic travel internally.

What should not be claimed or implemented yet:

- Do not make transportation comparison claims until strict route validation passes.
- Do not treat OSRM `fallback_speed` cells as road-valid.
- Do not use public OSRM for publication unless a human explicitly approves a non-publication run; preferred publication path is local or pinned evidence.
- Do not refactor benchmark or UI before route matrix contracts are stable.

## New Components to Add

### Component 1

`RouteMatrix`

Responsibility:

Represent stable entity-to-entity distance/duration/provenance cells with explicit missing and fallback states.

### Component 2

`RoutingProvider`

Responsibility:

Common protocol for pinned cache, local OSRM, explicitly approved remote OSRM, and demo fallback providers.

### Component 3

`OSRMClient`

Responsibility:

Implement OSRM Table for solver matrices, Route for selected geometry, and Nearest for snap diagnostics using injected HTTP client and route-source policy checks.

### Component 4

`RouteSourceValidator`

Responsibility:

Validate route matrix cells, selected route legs, snap distances, provider policy, and publication eligibility.

### Component 5

`RoutingContext`

Responsibility:

Bundle route matrix, selected route records, snap diagnostics, provider activation, and context snapshot ID for one run.

## Class / Registry Diagrams

```text
+-------------------------------------------------------------------------------+
|                                RouteMatrix                                      |
+-------------------------------------------------------------------------------+
|  - matrix_id: str                                                              |
|  - context_snapshot_id: str                                                    |
|  - entity_ids: tuple[str, ...]                                                  |
|  - cells: dict[tuple[str, str], RouteMatrixCell]                               |
+-------------------------------------------------------------------------------+
|  + cell(origin, destination): RouteMatrixCell --> Read one cell                |
|  + require_road_valid(origin, dest): RouteMatrixCell --> Strict lookup         |
|  + to_record(): dict[str, Any]              --> Stable artifact record         |
|  + validation_summary(): dict[str, float]    --> Coverage and fallback counts  |
+-------------------------------------------------------------------------------+
```

```text
+-------------------------------------------------------------------------------+
|                              RouteMatrixCell                                    |
+-------------------------------------------------------------------------------+
|  - origin_id: str                                                              |
|  - destination_id: str                                                         |
|  - distance_m: float | None                                                    |
|  - duration_s: float | None                                                    |
|  - provider: str                                                               |
|  - road_validated: bool                                                        |
|  - fallback_used: bool                                                         |
|  - query_hash: str                                                             |
|  - evidence_ref: str | None                                                    |
+-------------------------------------------------------------------------------+
|  + evaluation_eligible(): bool             --> True only if road-valid values  |
|  + to_leg_result(): RouteLegResult         --> Convert selected leg            |
+-------------------------------------------------------------------------------+
```

```text
+-------------------------------------------------------------------------------+
|                         RoutingProvider (Service Class)                         |
+-------------------------------------------------------------------------------+
|  - provider_id: str                                                            |
|  - policy: RoutingPolicy                                                       |
+-------------------------------------------------------------------------------+
|  + probe(): RoutingActivation              --> Provider availability           |
|  + table(request): RouteMatrix             --> OSRM Table or cache matrix      |
|  + route(request): RouteLegResult          --> Selected route geometry         |
|  + nearest(request): SnapResult            --> Snap diagnostics                |
+-------------------------------------------------------------------------------+
```

```text
+-------------------------------------------------------------------------------+
|                              OSRMClient                                         |
+-------------------------------------------------------------------------------+
|  - base_url: str                                                               |
|  - http_client: HttpClient                                                     |
|  - allow_public_osrm: bool                                                     |
+-------------------------------------------------------------------------------+
|  + table(request): RouteMatrix             --> `/table/v1/driving` result      |
|  + route(request): RouteLegResult          --> `/route/v1/driving` result      |
|  + nearest(request): SnapResult            --> `/nearest/v1/driving` result    |
|  + query_hash(payload): str                --> Stable cache key                |
+-------------------------------------------------------------------------------+
```

```text
+-------------------------------------------------------------------------------+
|                       RouteSourceValidator (Stateless Utility)                  |
+-------------------------------------------------------------------------------+
|  - No persistent internal state                                                |
+-------------------------------------------------------------------------------+
|  + validate_matrix(matrix): ValidationReport --> Matrix coverage gate          |
|  + validate_route(route): ValidationReport   --> Selected leg gate             |
|  + publication_eligible(context): bool        --> Strict route-source status    |
+-------------------------------------------------------------------------------+
```

## Class Diagram Rules

1. Diagram matrix, cell, provider, client, validator, and context objects.
2. Use existing `RouteLegResult`, `RouteResult`, and `RoadRouteCache` rather than replacing them.
3. Every class must expose source, fallback, and road-validation state.
4. Keep live network access out of model builders.
5. Route matrix cells with missing values must not become zero values.
6. Mark validators as stateless.

## Data Model

```python
@dataclass(frozen=True)
class RouteMatrixCell:
    origin_id: str
    destination_id: str
    distance_m: float | None
    duration_s: float | None
    provider: str
    routing_profile: str
    routing_status: str
    geometry_source: str
    distance_source: str
    duration_source: str
    road_validated: bool
    fallback_used: bool
    fallback_reason: str | None
    query_hash: str
    evidence_ref: str | None = None

@dataclass(frozen=True)
class RouteMatrix:
    matrix_id: str
    context_snapshot_id: str
    entity_ids: tuple[str, ...]
    cells: dict[tuple[str, str], RouteMatrixCell]
    created_at: str
    schema_version: str = "route-matrix-v1"

@dataclass(frozen=True)
class SnapResult:
    point_id: str
    provider: str
    snapped_latitude: float | None
    snapped_longitude: float | None
    snap_distance_m: float | None
    road_validated: bool
    query_hash: str

@dataclass(frozen=True)
class RoutingContext:
    context_snapshot_id: str
    route_matrix: RouteMatrix
    selected_routes: tuple[RouteResult, ...]
    snap_results: tuple[SnapResult, ...]
    provider_activation: dict[str, RoutingActivation]
```

## Storage / State

Persistent state:

- `data/contexts/<context_snapshot_id>/route_options.csv` remains context snapshot evidence.
- `production_road_route_cache.csv` remains Phase 0 compatibility route cache.
- Add future run artifacts under `runs/<run_id>/routing/route_matrix.json`, `selected_routes.jsonl`, `snap_diagnostics.jsonl`, and `route_source_validation.json`.
- Add local OSRM setup under `docker/osrm/`.

Temporary state:

- HTTP responses are cached as pinned evidence before optimizer/evaluator use.
- Optimizers receive only `RouteMatrix`, not live clients.

## Required Methods

```python
def route_matrix_from_cache(cache: RoadRouteCache, entity_points: tuple[RoutePoint, ...]) -> RouteMatrix
def build_route_matrix(provider: RoutingProvider, request: RouteMatrixRequest) -> RouteMatrix
def validate_route_source(matrix: RouteMatrix, *, require_complete: bool) -> RouteSourceValidationReport
def route_result_from_plan(plan: PlanArtifactV2, matrix: RouteMatrix) -> RouteResult
def require_publication_route_matrix(matrix: RouteMatrix) -> None
def replace_geodesic_travel_with_matrix(candidate_df: pd.DataFrame, matrix: RouteMatrix) -> pd.DataFrame
```

OSRM methods:

```python
def osrm_table(points: tuple[RoutePoint, ...], *, base_url: str) -> RouteMatrix
def osrm_route(origin: RoutePoint, destination: RoutePoint, *, base_url: str) -> RouteLegResult
def osrm_nearest(point: RoutePoint, *, base_url: str) -> SnapResult
```

## Validation Rules

1. Every matrix row/column ID must match a stable catalog, lodging, or route anchor ID.
2. Missing cells are explicit and fail strict publication validation.
3. Geodesic fallback cells have `road_validated=False`.
4. OSRM `fallback_speed` cells have `road_validated=False`.
5. Public OSRM is disabled by default and requires explicit approval for non-publication runs.
6. Selected route geometry must align with the plan sequence and route IDs.
7. Optimizer, evaluator, and renderer must consume the same route/context snapshot.
8. `RouteResult.evaluation_eligible` remains false unless solver feasible, schedule feasible, dataset valid, and all legs road validated.

## UI / API Integration

Internal integration:

- `multi_objective_route.py` receives a `RouteMatrix` or explicit `ApproximateRoutingMode`.
- `hierarchical_gurobi.py` receives route matrix costs for intercity/base-city travel.
- `route_gurobi_oracle.py` removes unreachable legacy code after equivalence tests.
- `phase0_exporter.py` continues to support `RoadRouteCache`.
- Future dashboard renderers consume selected route IDs and route audit rows.

## Workflow

1. Define route matrix/cell models and provider protocol.
2. Add cache provider adapter for `RoadRouteCache`.
3. Add OSRM client with Table, Route, and Nearest.
4. Add route-source validation reports.
5. Add local OSRM Docker assets and documentation.
6. Inject route matrix into solver entry points.
7. Keep demo fallback mode explicit and non-publication.
8. Run route-source strict validation.
9. Proceed to G3 only after publication-mode route matrix validation passes for the first parent plan slice.

## Files to Create

- `src/itinerary_system/routing/matrix.py`
- `src/itinerary_system/routing/provider.py`
- `src/itinerary_system/routing/osrm_client.py`
- `src/itinerary_system/routing/validation.py`
- `src/itinerary_system/routing/errors.py`
- `docker/osrm/docker-compose.yml`
- `docker/osrm/.env.example`
- `docker/osrm/README.md`
- `docker/osrm/scripts/download_extract.ps1`
- `docker/osrm/scripts/preprocess.ps1`
- `docker/osrm/scripts/healthcheck.ps1`
- `tests/routing/test_route_matrix.py`
- `tests/routing/test_osrm_client.py`
- `tests/routing/test_route_source_validation.py`
- `tests/routing/test_solver_route_matrix_injection.py`

## Files to Modify

- `src/itinerary_system/routing/models.py`
- `src/itinerary_system/routing/cache.py`
- `src/itinerary_system/routing/road_cache_builder.py`
- `src/itinerary_system/routing/__init__.py`
- `src/itinerary_system/multi_objective_route.py`
- `src/itinerary_system/hierarchical_gurobi.py`
- `src/itinerary_system/route_gurobi_oracle.py`
- `src/itinerary_system/phase0_exporter.py`
- `scripts/build_road_route_cache.py`
- `scripts/check_route_source.py`
- `tests/test_research_foundation.py`
- `.gitignore` for large `.osm.pbf` and `.osrm*` assets

## Error Handling

- Missing matrix cell in strict mode: `RouteMatrixMissingCell`.
- Non-road-valid cell in publication mode: `RouteValidationError`.
- Public OSRM without approval: existing `ValueError` text or typed `RoutingPolicyError`.
- Snap distance beyond threshold: warning or hard failure depending on publication policy.
- OSRM timeout: provider unavailable; do not silently fall back in publication mode.
- Geodesic fallback requested in publication mode: hard failure.

## Testing Checklist

- [ ] Route matrix rejects missing cells in strict mode.
- [ ] Route matrix never substitutes zero duration for missing route evidence.
- [ ] Geodesic fallback cell has `road_validated=False`.
- [ ] OSRM Table fallback-speed cells are nonvalidated.
- [ ] OSRM Route geometry produces `RouteLegResult`.
- [ ] OSRM Nearest snap diagnostics store snap distance.
- [ ] Public OSRM is blocked without approval.
- [ ] Local/pinned route cache can satisfy strict Phase 0 validation.
- [ ] `multi_objective_route.py` publication mode refuses missing road-valid matrix.
- [ ] `hierarchical_gurobi.py` no longer uses geodesic proxy in publication mode.
- [ ] Optimizer/evaluator/renderer totals match selected route records within tolerance.
- [ ] `python -m ruff check src tests scripts`
- [ ] `python -m pytest`

## Roadmap / Full Pipeline Update

After implementation and strict route validation pass, update `ROUTE-001`, `ROUTE-002`, `ROUTE-003`, and `ROUTE-004` in the technical specification. This gate connects to G3 by giving the repair solver road-valid travel and route continuity evidence. Do not claim benchmark readiness until G5.

## Acceptance Criteria

1. Publication solvers can consume only road-valid route matrices.
2. Demo fallback remains available but cannot be final-comparison eligible.
3. Local or pinned OSRM evidence can generate strict route artifacts.
4. Optimizer, evaluator, and renderer use the same route/context snapshot.
5. Current Phase 0 routing artifacts remain compatible.

# Phase G3.0 Implementation Plan: Single-Day Ownership-Aware Repair

## Goal

Implement **single-day ownership-aware repair**.

This gate should allow the system to:

1. Accept a typed edit or disruption against a canonical parent plan.
2. Build a radius 0 or radius 1 repair neighborhood for one affected day.
3. Solve preservation-before-utility repair with independent evaluation.

Keep the implementation modular, easy to test, and consistent with G1 and G2 contracts.

## Current State

Repository evidence found:

- `src/itinerary_system/repair_planner.py`: supports `ParsedRepairIntent`, `EvidenceConflict`, `RepairOperation`, `RepairRequest`, `RepairPlan`, `build_repair_plan()`, and `evaluate_repair_plan()`, but works on raw route tuples and deterministic heuristics.
- `tests/test_repair_planner.py`: verifies unconfirmed parser blocking, evidence conflict labels, replacement of high-weather-risk stop, frontier alternatives, move operations, and hard feasibility gates.
- `src/itinerary_system/multi_objective_route.py`: has a Gurobi/heuristic route selector, but no parent assignment variables or typed change variables.
- `docs/planning/travel_itinerary_repair_technical_specification.md`: specifies repair request, neighborhoods, master variables, lexicographic objective stages, and progressive controller.

Missing pieces:

- No `EditRequest`, `DisruptionSnapshot`, `RepairRadius`, `RepairNeighborhood`, or v2 `RepairOutcome`.
- No parent assignment index.
- No child assignment variables.
- No typed change variables tied to `PlanDiff`.
- No relaxation variables for owned constraints.
- No route matrix injection in repair model.
- No opening-window, visit-duration, route-continuity, budget/time/closure/weather checks in an independent final-plan evaluator.

What should not be claimed or implemented yet:

- Do not implement multi-day progressive repair here.
- Do not implement benchmark baselines here.
- Do not use one large weighted sum as the final repair objective.
- Do not allow an LLM interpretation to control the solver unless confirmed.

## New Components to Add

### Component 1

`EditRequest`

Responsibility:

Represent typed user edits such as add, delete, replace, move, reorder, time shift, lodging change, road/corridor change, and constraint relaxation.

### Component 2

`DisruptionSnapshot`

Responsibility:

Represent frozen weather, road closure, hotel unavailability, attraction closure, fatigue, or new must-visit disruption evidence tied to catalog/context snapshots.

### Component 3

`RepairRequest`

Responsibility:

Bind parent plan, edits, disruption snapshot, allowed radii, and relaxation permissions.

### Component 4

`RepairNeighborhood`

Responsibility:

Freeze all plan elements outside radius 0 or radius 1 and expose editable stops, days, route legs, lodging nights, and constraints.

### Component 5

`OwnershipAwareRepairModel`

Responsibility:

Build single-day assignment, typed change, relaxation, route, time, and objective expressions relative to the parent plan.

### Component 6

`LexicographicRepairSolver`

Responsibility:

Solve Stage 0 through Stage 3 sequentially and store status, objective value, bound, gap, infeasibility evidence, and fixed-stage tolerances.

## Class / Registry Diagrams

```text
+-------------------------------------------------------------------------------+
|                                 EditRequest                                     |
+-------------------------------------------------------------------------------+
|  - edit_id: str                                                                |
|  - edit_type: EditType                                                         |
|  - target_id: str | None                                                       |
|  - requested_value: JsonValue                                                  |
|  - origin: ConstraintOrigin                                                    |
|  - strength: ConstraintStrength                                                |
+-------------------------------------------------------------------------------+
|  + validate_against(parent): tuple[Finding, ...] --> Entity/scope validation   |
|  + to_owned_constraint(): OwnedConstraint | None --> Constraint edits          |
+-------------------------------------------------------------------------------+
```

```text
+-------------------------------------------------------------------------------+
|                              DisruptionSnapshot                                 |
+-------------------------------------------------------------------------------+
|  - disruption_snapshot_id: str                                                 |
|  - context_snapshot_id: str                                                    |
|  - disruption_type: DisruptionType                                             |
|  - affected_ids: tuple[str, ...]                                               |
|  - evidence_refs: tuple[str, ...]                                              |
+-------------------------------------------------------------------------------+
|  + affected_scope(): tuple[ConstraintScope, ...] --> Scope classifier          |
|  + to_constraints(): tuple[OwnedConstraint, ...] --> External-data constraints |
+-------------------------------------------------------------------------------+
```

```text
+-------------------------------------------------------------------------------+
|                                RepairRequest                                    |
+-------------------------------------------------------------------------------+
|  - repair_request_id: str                                                      |
|  - parent_plan_id: str                                                         |
|  - disruption_snapshot_id: str                                                 |
|  - edits: tuple[EditRequest, ...]                                              |
|  - allowed_radii: tuple[RepairRadius, ...]                                     |
|  - allow_booked_relaxation: bool                                               |
+-------------------------------------------------------------------------------+
|  + validate(parent): tuple[Finding, ...]    --> Request eligibility            |
|  + requested_changes(): tuple[EditRequest, ...] --> Typed edit list            |
+-------------------------------------------------------------------------------+
```

```text
+-------------------------------------------------------------------------------+
|                  RepairNeighborhood (Runtime State Registry)                    |
+-------------------------------------------------------------------------------+
|  - radius: RepairRadius                                                        |
|  - editable_days: tuple[int, ...]                                              |
|  - editable_stop_ids: tuple[str, ...]                                          |
|  - editable_route_leg_ids: tuple[str, ...]                                     |
|  - frozen_stop_ids: tuple[str, ...]                                            |
+-------------------------------------------------------------------------------+
|  + contains_stop(stop_id): bool             --> Editable/frozen check          |
|  + freeze_constraints(): tuple[Constraint, ...] --> Outside-neighborhood fix   |
|  + evidence_refs(): tuple[str, ...]         --> Explanation-ready refs         |
+-------------------------------------------------------------------------------+
```

```text
+-------------------------------------------------------------------------------+
|                        OwnershipAwareRepairModel                                |
+-------------------------------------------------------------------------------+
|  - parent_index: ParentPlanIndex                                               |
|  - neighborhood: RepairNeighborhood                                            |
|  - route_matrix: RouteMatrix                                                   |
|  - ownership_policy: OwnershipPolicy                                           |
+-------------------------------------------------------------------------------+
|  + build_variables(): None                  --> Parent/child/change variables  |
|  + add_constraints(): None                  --> Feasibility and ownership      |
|  + objective(stage): LinExpr               --> Stage-specific expression       |
|  + extract_child_plan(): PlanArtifactV2     --> Candidate child artifact       |
+-------------------------------------------------------------------------------+
```

```text
+-------------------------------------------------------------------------------+
|                         LexicographicRepairSolver                               |
+-------------------------------------------------------------------------------+
|  - tolerances: ObjectiveTolerances                                             |
|  - solver_backend: str                                                         |
+-------------------------------------------------------------------------------+
|  + solve(model): LexicographicResult        --> Solve all stages               |
|  + fix_stage(model, stage, value): None     --> Preserve prior optimum         |
|  + stage_record(stage): SolverStageRecord   --> Store status/bound/gap         |
+-------------------------------------------------------------------------------+
```

## Class Diagram Rules

1. Diagram request, disruption, neighborhood, model, and solver classes.
2. Keep request parsing separate from solver model building.
3. Solvers consume `PlanArtifactV2`, `PlanDiff`, `RouteMatrix`, and evaluator contracts from prior gates.
4. Do not put live API or data refresh logic in model builders.
5. Mark neighborhoods as runtime state, not persistent source-of-truth.
6. Every solver result must produce `PlannerRun` and `SolverStageRecord` artifacts.

## Data Model

```python
class EditType(StrEnum):
    ADD_STOP = "add_stop"
    DELETE_STOP = "delete_stop"
    REPLACE_STOP = "replace_stop"
    MOVE_STOP_TO_DAY = "move_stop_to_day"
    REORDER_STOPS = "reorder_stops"
    TIME_SHIFT = "time_shift"
    LODGING_CHANGE = "lodging_change"
    ROAD_CORRIDOR_CHANGE = "road_corridor_change"
    CONSTRAINT_RELAXATION = "constraint_relaxation"

class DisruptionType(StrEnum):
    WEATHER_DETERIORATION = "weather_deterioration"
    ROAD_CLOSURE = "road_closure"
    HOTEL_UNAVAILABILITY = "hotel_unavailability"
    ATTRACTION_CLOSURE = "attraction_closure"
    OPENING_WINDOW_CONFLICT = "opening_window_conflict"
    REDUCED_DRIVING_TOLERANCE = "reduced_driving_tolerance"
    NEW_MUST_VISIT = "new_must_visit"

class RepairRadius(StrEnum):
    RADIUS_0_DIRECT = "radius_0_direct"
    RADIUS_1_SAME_DAY = "radius_1_same_day"
    RADIUS_2_ADJACENT_DAYS = "radius_2_adjacent_days"
    FULL_EDITABLE_SCOPE = "full_editable_scope"

class ObjectiveStage(StrEnum):
    HARD_RELAXATION = "hard_relaxation"
    PLAN_CHANGE = "plan_change"
    CONTEXTUAL_BURDEN = "contextual_burden"
    ROUTE_UTILITY = "route_utility"

@dataclass(frozen=True)
class SolverStageRecord:
    stage: ObjectiveStage
    status: str
    objective_value: float | None
    best_bound: float | None
    optimality_gap: float | None
    runtime_seconds: float
    fixed_prior_stage_constraints: tuple[str, ...]
    infeasibility_evidence_refs: tuple[str, ...] = ()

@dataclass(frozen=True)
class RepairOutcome:
    repair_outcome_id: str
    parent_plan_id: str
    child_plan_id: str | None
    repair_request_id: str
    accepted_radius: RepairRadius | None
    planner_runs: tuple[PlannerRun, ...]
    stage_records: tuple[SolverStageRecord, ...]
    diff_id: str | None
    evaluation_id: str | None
    status: str
```

Solver variables to plan:

- parent assignment variables/index: `x0[i,d]`;
- child assignment variables: `x[i,d]`;
- typed change variables: `delta_minus[i]`, `delta_plus[i]`, `replace[i,j]`, `move[i,d]`, `time_shift[i]`, `reorder[i,j]`;
- relaxation variables: `relax[c]`;
- lodging variables: `z[h,d]`, `hotel_change[h,d]`;
- route variables: `y[i,j,d]`, `road_change[i,j,d]`;
- arrival/departure variables: `T[i,d]`;
- route matrix injection: `tau[i,j]`, `dist[i,j]` from `RouteMatrix`.

Sequential objective:

```text
Stage 0: minimize hard/booked relaxations
Stage 1: minimize ownership-weighted typed plan changes
Stage 2: minimize contextual burden: weather risk, closure risk, road-valid travel, hotel switches, cost
Stage 3: maximize utility, diversity, scenic/nature fit
```

Each stage is solved separately. After each solve, add a constraint fixing that stage objective to the prior optimum within tolerance before optimizing the next stage.

## Storage / State

Persistent state:

- `runs/<run_id>/planner_runs.jsonl` later in G6; for G3 tests use temporary fixtures or Phase 0-compatible output.
- `runs/<run_id>/requests/<repair_request_id>.json`
- `runs/<run_id>/plans/<child_plan_id>.json`
- `runs/<run_id>/diffs/<diff_id>.json`
- `runs/<run_id>/evaluations/<evaluation_id>.json`
- `runs/<run_id>/routing/route_matrix.json`

Temporary state:

- Gurobi/solver model objects exist only during solve.
- `RepairNeighborhood` is runtime-only, except its summary is stored in `PlannerRun` metadata.

## Required Methods

```python
def build_parent_plan_index(parent: PlanArtifactV2) -> ParentPlanIndex
def build_repair_neighborhood(parent: PlanArtifactV2, request: RepairRequest, radius: RepairRadius) -> RepairNeighborhood
def build_single_day_repair_model(
    parent: PlanArtifactV2,
    request: RepairRequest,
    neighborhood: RepairNeighborhood,
    bundle: DatasetBundle,
    route_matrix: RouteMatrix,
    ownership_policy: OwnershipPolicy,
) -> OwnershipAwareRepairModel
def solve_lexicographically(model: OwnershipAwareRepairModel, tolerances: ObjectiveTolerances) -> LexicographicResult
def extract_repair_child_plan(result: LexicographicResult) -> PlanArtifactV2
def evaluate_repair_child(plan: PlanArtifactV2, evaluator: PlanEvaluator) -> ResearchEvaluationReport
```

## Validation Rules

1. Radius 0 edits only the direct disrupted item or direct user edit.
2. Radius 1 edits the same day and adjacent route legs.
3. Outside-neighborhood assignments are fixed to the parent.
4. Locked constraints cannot be relaxed.
5. Booked constraints require explicit permission to relax.
6. Unconfirmed LLM constraints are inactive.
7. Test-only constraints cannot create accepted child plans.
8. Publication solver uses only road-valid route matrix values.
9. Opening windows are enforced when available.
10. Visit duration is included in daily feasibility.
11. Route continuity is enforced with start/end anchors.
12. Budget/time/closure/weather checks are recomputed by evaluator after solve.
13. No post-solve mutation is allowed without new evaluation.

## UI / API Integration

Internal only:

- `repair_planner.py` can remain a compatibility/demo scaffold, but production repair should move into `src/itinerary_system/repair/`.
- Existing tests can continue covering deterministic scaffold behavior.
- No dashboard, notebook, or API endpoint is required until G6/G7.

## Workflow

1. Load parent plan from `PlanRepository`.
2. Validate `RepairRequest`.
3. Build radius 0 or radius 1 neighborhood.
4. Build ownership-aware repair model.
5. Solve Stage 0 and fix optimum.
6. Solve Stage 1 and fix optimum.
7. Solve Stage 2 and fix optimum.
8. Solve Stage 3.
9. Extract child plan.
10. Compute `PlanDiff`.
11. Run independent evaluator.
12. Store stage records and repair outcome.
13. Stop here; do not implement multi-day expansion until G4.

## Files to Create

- `src/itinerary_system/repair/__init__.py`
- `src/itinerary_system/repair/models.py`
- `src/itinerary_system/repair/neighborhood.py`
- `src/itinerary_system/repair/parent_index.py`
- `src/itinerary_system/repair/change_variables.py`
- `src/itinerary_system/repair/master_model.py`
- `src/itinerary_system/repair/lexicographic.py`
- `src/itinerary_system/repair/day_route_solver.py`
- `src/itinerary_system/repair/errors.py`
- `tests/repair/test_models.py`
- `tests/repair/test_neighborhood.py`
- `tests/repair/test_master_model.py`
- `tests/repair/test_lexicographic.py`
- `tests/repair/test_single_day_repair.py`

## Files to Modify

- `src/itinerary_system/repair_planner.py` for compatibility wrappers only if needed.
- `src/itinerary_system/research_artifacts.py`
- `src/itinerary_system/plans/diff.py`
- `src/itinerary_system/routing/matrix.py`
- `src/itinerary_system/__init__.py`
- `tests/test_repair_planner.py` only if compatibility behavior changes.

## Error Handling

- Missing parent plan: `RepairInputError`.
- Repair request references unknown stop/lodging/route: validation failure.
- No road-valid route matrix in publication mode: hard failure before model build.
- Solver infeasible at Stage 0: failed `PlannerRun` with infeasibility evidence.
- Later stage cannot preserve prior optimum within tolerance: solver failure.
- Booked relaxation without permission: hard validation failure.
- Child plan fails independent evaluation: return outcome with `status="ineligible_child"`.

## Testing Checklist

- [ ] Radius 0 only edits direct disrupted item.
- [ ] Radius 1 includes same-day stops and adjacent legs.
- [ ] Locked POI cannot be deleted.
- [ ] Booked lodging cannot change without permission.
- [ ] Strong preference incurs higher change cost than weak preference.
- [ ] Unconfirmed LLM constraint does not constrain solver.
- [ ] Stage 1 cannot worsen Stage 0 beyond tolerance.
- [ ] Stage 2 cannot worsen Stage 1 beyond tolerance.
- [ ] Stage 3 cannot worsen Stage 2 beyond tolerance.
- [ ] Road-valid route matrix is required in publication mode.
- [ ] Evaluator recomputes final child eligibility.
- [ ] Post-solve mutation invalidates certification.
- [ ] `python -m ruff check src tests scripts`
- [ ] `python -m pytest`

## Roadmap / Full Pipeline Update

After implementation and validation pass, update `REPAIR-001`, `REPAIR-002`, `REPAIR-003`, and `REPAIR-004` only for the single-day slice. This gate connects to G4 by proving one-day ownership-aware repair before expanding neighborhoods across days. Do not claim progressive repair or benchmark superiority yet.

## Acceptance Criteria

1. A single-day child plan can be produced from a typed repair request.
2. Locked/booked/strong/weak/test-only ownership behavior is enforced.
3. Sequential lexicographic stage records are stored.
4. The child plan has a typed diff and independent evaluation.
5. The result is not a weighted-sum reoptimization.
6. The system does not mutate the parent plan.

# Phase G4.0 Implementation Plan: Progressive Multi-Day Repair

## Goal

Implement **progressive multi-day repair**.

This gate should allow the system to:

1. Expand repair neighborhoods deterministically from radius 0 to full editable scope.
2. Stop at the smallest radius that produces an independently eligible child plan.
3. Integrate lodging boundaries, adjacent days, and route/corridor protection.

Keep the implementation modular and build directly on G1-G3 contracts.

## Current State

Repository evidence found:

- `docs/planning/travel_itinerary_repair_technical_specification.md`: defines radius 0, radius 1, radius 2, and full editable scope.
- `src/itinerary_system/repair_planner.py`: has `generate_repair_alternatives()` but not progressive radius attempts or independent eligibility.
- `src/itinerary_system/experiment_runner.py`: supports multi-day route artifacts and hotel debug outputs but not parent-child repair.
- `src/itinerary_system/data_enrichment.py`: creates curated hotel fallback rows and marks fallback reasons, but lodging is not provider-independent.
- `docs/current/current_score_audit.md`: recommends progressive neighborhoods and independent certificates.

Missing pieces:

- No progressive controller.
- No attempt JSONL.
- No smallest-radius selection rule.
- No multi-day lodging boundary handling.
- No route/corridor protected arc handling.
- No smallest-relaxation diagnosis after all radii fail.

What should not be claimed or implemented yet:

- Do not jump straight to full reoptimization as default.
- Do not make benchmark claims until G5.
- Do not claim live hotel availability.
- Do not add UI/user study work before G6/G7.

## New Components to Add

### Component 1

`ProgressiveRepairController`

Responsibility:

Try allowed repair radii in order, store every attempt, evaluate each candidate independently, and return the smallest eligible child.

### Component 2

`MultiDayRepairNeighborhoodBuilder`

Responsibility:

Build radius 0, radius 1, radius 2, and full editable neighborhoods with stops, route legs, lodging nights, day boundaries, and frozen scopes.

### Component 3

`LodgingRepairPolicy`

Responsibility:

Protect booked lodging, estimate lodging boundary effects, and permit lodging changes only when neighborhood and ownership rules allow.

### Component 4

`CorridorProtectionPolicy`

Responsibility:

Protect scenic or user-owned route/corridor commitments unless disruption or evaluator failure requires road changes.

### Component 5

`SmallestRelaxationDiagnosis`

Responsibility:

Explain why no radius succeeded and list the smallest constraint relaxations or scope expansions needed.

## Class / Registry Diagrams

```text
+-------------------------------------------------------------------------------+
|                       ProgressiveRepairController                               |
+-------------------------------------------------------------------------------+
|  - solver: LexicographicRepairSolver                                           |
|  - evaluator: PlanEvaluator                                                    |
|  - neighborhood_builder: MultiDayRepairNeighborhoodBuilder                     |
+-------------------------------------------------------------------------------+
|  + repair(request): RepairOutcome           --> Smallest eligible child        |
|  + attempt_radius(radius): RepairAttempt    --> One radius attempt             |
|  + stop_condition(attempt): bool            --> Eligible child gate            |
|  + diagnose_failure(attempts): SmallestRelaxationDiagnosis --> Failure reason  |
+-------------------------------------------------------------------------------+
```

```text
+-------------------------------------------------------------------------------+
|                  MultiDayRepairNeighborhoodBuilder                              |
+-------------------------------------------------------------------------------+
|  - parent_index: ParentPlanIndex                                               |
|  - ownership_policy: OwnershipPolicy                                           |
+-------------------------------------------------------------------------------+
|  + radius0(request): RepairNeighborhood     --> Direct disrupted item          |
|  + radius1(request): RepairNeighborhood     --> Same day and adjacent legs     |
|  + radius2(request): RepairNeighborhood     --> Adjacent days/lodging boundary |
|  + full_scope(request): RepairNeighborhood  --> Final fallback editable scope  |
+-------------------------------------------------------------------------------+
```

```text
+-------------------------------------------------------------------------------+
|                         LodgingRepairPolicy                                     |
+-------------------------------------------------------------------------------+
|  - No persistent internal state                                                |
+-------------------------------------------------------------------------------+
|  + editable_nights(radius): tuple[int, ...] --> Lodging scope by radius        |
|  + can_change_lodging(constraint, request): bool --> Booked permission gate    |
|  + lodging_switch_cost(parent, child): float --> Contextual burden component   |
+-------------------------------------------------------------------------------+
```

```text
+-------------------------------------------------------------------------------+
|                       CorridorProtectionPolicy                                  |
+-------------------------------------------------------------------------------+
|  - No persistent internal state                                                |
+-------------------------------------------------------------------------------+
|  + protected_arcs(parent): tuple[RouteArc, ...] --> Scenic/user route arcs     |
|  + can_change_arc(arc, request, disruption): bool --> Road edit permission     |
|  + road_change_cost(arc): float             --> Stage 1/2 burden component     |
+-------------------------------------------------------------------------------+
```

```text
+-------------------------------------------------------------------------------+
|                       SmallestRelaxationDiagnosis                               |
+-------------------------------------------------------------------------------+
|  - failed_radii: tuple[RepairRadius, ...]                                      |
|  - binding_constraints: tuple[str, ...]                                        |
|  - proposed_relaxations: tuple[ConstraintRelaxation, ...]                      |
+-------------------------------------------------------------------------------+
|  + to_explanation_seed(): dict[str, Any]    --> Why-not evidence input         |
|  + to_record(): dict[str, Any]              --> Stored failure artifact        |
+-------------------------------------------------------------------------------+
```

## Class Diagram Rules

1. Diagram the controller, neighborhood builder, lodging policy, corridor policy, and diagnosis objects.
2. Controller stores attempt summaries, not mutable solver models.
3. Radius definitions must be deterministic and testable.
4. Lodging and route protection are policies, not ad hoc objective weights.
5. Full editable scope is final fallback only.
6. Failed attempts are first-class evidence for explanation and benchmarks.

## Data Model

```python
@dataclass(frozen=True)
class RepairAttempt:
    attempt_id: str
    repair_request_id: str
    radius: RepairRadius
    planner_run_id: str
    child_plan_id: str | None
    diff_id: str | None
    evaluation_id: str | None
    eligible: bool
    failure_codes: tuple[str, ...]
    stage_records: tuple[SolverStageRecord, ...]

@dataclass(frozen=True)
class ProgressiveRepairOutcome(RepairOutcome):
    attempts: tuple[RepairAttempt, ...]
    accepted_attempt_id: str | None
    smallest_successful_radius: RepairRadius | None
    diagnosis: SmallestRelaxationDiagnosis | None
```

Progressive neighborhoods:

```text
radius 0: only direct disrupted item or direct user edit
radius 1: same day and adjacent route legs
radius 2: adjacent days and affected lodging boundary
full editable scope: final fallback only
```

## Storage / State

Persistent state:

- `runs/<run_id>/repair_attempts.jsonl`
- `runs/<run_id>/requests/<repair_request_id>.json`
- `runs/<run_id>/planner_runs.jsonl`
- `runs/<run_id>/plans/`
- `runs/<run_id>/diffs/`
- `runs/<run_id>/evaluations/`
- `runs/<run_id>/diagnostics/smallest_relaxation_<id>.json`

Temporary state:

- One solver model per attempt.
- Controller state discarded after persisted outcome.

## Required Methods

```python
def repair_progressively(
    request: RepairRequest,
    plan_repository: PlanRepository,
    data_bundle: DatasetBundle,
    routing_context: RoutingContext,
    solver: LexicographicRepairSolver,
    evaluator: PlanEvaluator,
) -> ProgressiveRepairOutcome

def build_radius_sequence(request: RepairRequest) -> tuple[RepairRadius, ...]
def evaluate_attempt(candidate: PlanArtifactV2 | None, evaluator: PlanEvaluator) -> ResearchEvaluationReport | None
def stop_at_smallest_eligible(attempts: tuple[RepairAttempt, ...]) -> RepairAttempt | None
def compute_smallest_relaxation(attempts: tuple[RepairAttempt, ...]) -> SmallestRelaxationDiagnosis
```

## Validation Rules

1. Radius attempts run in deterministic order.
2. Controller stops at the first independently eligible child plan.
3. Attempts with solver feasible but evaluator ineligible do not stop the controller.
4. Full editable scope is attempted only after smaller radii fail.
5. Adjacent-day edits include affected lodging boundary nights.
6. Booked lodging changes require explicit permission.
7. Scenic/protected route arcs are preserved unless disruption/evaluator failure permits road change.
8. All attempts store planner run, radius, stage records, and failure codes.
9. Parent plan remains immutable across attempts.

## UI / API Integration

Internal integration:

- Benchmark runner will later consume progressive attempts.
- Explanation layer will use failed radius attempts as why-not/contrastive evidence.
- UI is not required in this gate.

## Workflow

1. Validate parent plan and repair request.
2. Create radius sequence.
3. For each radius, build neighborhood.
4. Solve lexicographic repair.
5. Extract child if possible.
6. Compute diff and evaluate independently.
7. Store attempt.
8. Stop only when evaluator returns eligible.
9. If all radii fail, compute smallest-relaxation diagnosis.
10. Return `ProgressiveRepairOutcome`.

## Files to Create

- `src/itinerary_system/repair/progressive.py`
- `src/itinerary_system/repair/multiday_neighborhood.py`
- `src/itinerary_system/repair/lodging_policy.py`
- `src/itinerary_system/repair/corridor_policy.py`
- `src/itinerary_system/repair/diagnostics.py`
- `tests/repair/test_progressive.py`
- `tests/repair/test_multiday_neighborhood.py`
- `tests/repair/test_lodging_policy.py`
- `tests/repair/test_corridor_policy.py`

## Files to Modify

- `src/itinerary_system/repair/models.py`
- `src/itinerary_system/repair/master_model.py`
- `src/itinerary_system/plans/diff.py`
- `src/itinerary_system/evaluation/plan_evaluator.py`
- `src/itinerary_system/__init__.py`

## Error Handling

- No allowed radii: request validation error.
- Solver timeout at one radius: store failed attempt and continue if policy allows.
- Evaluator ineligible: store hard failure and continue.
- Booked lodging blocks all radii: diagnosis identifies booked-relaxation permission as required.
- Route matrix missing adjacent-day legs: fail the attempt, do not substitute geodesic.
- Full scope fails: return `ProgressiveRepairOutcome` with diagnosis and no child.

## Testing Checklist

- [ ] Radius sequence is deterministic.
- [ ] Radius 0 freezes all unaffected stops/days.
- [ ] Radius 1 includes same-day adjacent legs.
- [ ] Radius 2 includes adjacent days and lodging boundary.
- [ ] Full scope is attempted last.
- [ ] Controller stops at smallest independently eligible radius.
- [ ] Evaluator-ineligible candidate does not stop controller.
- [ ] Every failed attempt is persisted.
- [ ] Booked lodging requires permission.
- [ ] Protected scenic route arc change is penalized/blocked according to policy.
- [ ] Failure diagnosis lists binding constraints and proposed relaxations.
- [ ] `python -m ruff check src tests scripts`
- [ ] `python -m pytest`

## Roadmap / Full Pipeline Update

After implementation and tests pass, update `REPAIR-005` in the technical specification. This gate connects to G5 by producing the progressive repair method and attempt artifacts needed for benchmark comparisons. Do not claim benchmark results until G5 runs paired baselines.

## Acceptance Criteria

1. Progressive repair always starts local and expands only when needed.
2. The accepted child is the smallest independently eligible repair.
3. Failed attempts are stored and explanation-ready.
4. Lodging and route/corridor ownership rules are enforced.
5. Full reoptimization is a final fallback, not the default.

# Phase G5.0 Implementation Plan: Benchmark Package

## Goal

Implement **benchmark package**.

This gate should allow the system to:

1. Generate the six required disruption families from immutable parent plans.
2. Run paired baselines on identical frozen catalog/context/parent/disruption inputs.
3. Report preservation, utility, risk, radius, certificate, explanation, fallback, runtime, route-valid travel, and solver status metrics without leakage.

Keep the implementation modular and do not make benchmark claims before strict validation passes.

## Current State

Repository evidence found:

- `docs/literature/repair_gap_review.md`: defines metrics such as locked preservation, booked preservation, unaffected-day preservation, weighted edit cost, utility retained, utility regret, weather risk reduction, nature exposure reduction, repair radius, certificate coverage, explanation evidence coverage, and fallback rate.
- `docs/planning/travel_itinerary_repair_technical_specification.md`: specifies benchmark parent plans, six disruptions, ownership profiles, baselines, and split rule.
- `src/itinerary_system/experiment_runner.py`: already builds method comparisons and dashboards, but not repair benchmark suites.
- `tests/test_research_foundation.py`: contains Phase 0 artifact fixtures that can seed benchmark fixture style.

Missing pieces:

- No `src/itinerary_system/benchmark/` package.
- No canonical disruptions.
- No ownership profiles.
- No baseline adapters for LLM-only, full reoptimization, current hybrid pipeline, weighted-sum repair, fixed-radius repair, and progressive sequential lexicographic repair.
- No parent/disruption-family split rules.
- No benchmark long-form metrics table.
- No leakage tests.

What should not be claimed or implemented yet:

- Do not include LLM-only rewrite unless frozen inputs and evaluator are available.
- Do not split rows from the same parent/disruption family across train/dev/test.
- Do not rank plans that are evaluator-ineligible.
- Do not compare route travel time if route evidence is geodesic fallback.

## New Components to Add

### Component 1

`DisruptionGenerator`

Responsibility:

Create deterministic typed disruptions for weather deterioration, road closure, hotel unavailability, attraction closure/opening-window conflict, reduced driving tolerance/fatigue, and newly added user must-visit.

### Component 2

`OwnershipProfileFactory`

Responsibility:

Generate flexible, mixed, booking-constrained, scenic-route-protective, and high-lock ownership configurations over parent plans.

### Component 3

`BenchmarkMethodAdapter`

Responsibility:

Wrap each baseline method behind a common input/output contract.

### Component 4

`RepairBenchmarkRunner`

Responsibility:

Execute paired benchmark cases, enforce identical snapshots, call evaluator, store metrics, and block ineligible rankings.

### Component 5

`BenchmarkSplitManager`

Responsibility:

Split by parent-plan/disruption family and prove no leakage across train/dev/test.

### Component 6

`BenchmarkMetrics`

Responsibility:

Compute the required preservation, utility, risk, route, certificate, explanation, fallback, and solver metrics from artifacts.

## Class / Registry Diagrams

```text
+-------------------------------------------------------------------------------+
|                             DisruptionGenerator                                 |
+-------------------------------------------------------------------------------+
|  - seed: int                                                                   |
|  - catalog_snapshot_id: str                                                    |
|  - context_snapshot_id: str                                                    |
+-------------------------------------------------------------------------------+
|  + weather(parent): RepairRequest          --> Weather deterioration           |
|  + road_closure(parent): RepairRequest     --> Road/corridor disruption        |
|  + hotel_unavailable(parent): RepairRequest --> Lodging disruption             |
|  + attraction_closed(parent): RepairRequest --> Closure/window conflict        |
|  + fatigue(parent): RepairRequest          --> Reduced driving tolerance       |
|  + new_must_visit(parent): RepairRequest   --> Added locked/strong POI         |
+-------------------------------------------------------------------------------+
```

```text
+-------------------------------------------------------------------------------+
|                         OwnershipProfileFactory                                 |
+-------------------------------------------------------------------------------+
|  - No persistent internal state                                                |
+-------------------------------------------------------------------------------+
|  + flexible(parent): PlanArtifactV2        --> Mostly weak/flexible            |
|  + mixed(parent): PlanArtifactV2           --> Mixed strengths                 |
|  + booking_constrained(parent): PlanArtifactV2 --> Booked lodging/tickets      |
|  + scenic_route_protective(parent): PlanArtifactV2 --> Route commitments       |
|  + high_lock(parent): PlanArtifactV2       --> Many locked commitments         |
+-------------------------------------------------------------------------------+
```

```text
+-------------------------------------------------------------------------------+
|                       BenchmarkMethodAdapter (Service Class)                    |
+-------------------------------------------------------------------------------+
|  - method_id: str                                                              |
|  - method_kind: str                                                            |
+-------------------------------------------------------------------------------+
|  + run(case): BenchmarkRunResult           --> Produce plan/evaluation         |
|  + supports(case): bool                    --> Whether baseline is allowed     |
|  + required_evidence(): tuple[str, ...]    --> Inputs needed for validity      |
+-------------------------------------------------------------------------------+
```

```text
+-------------------------------------------------------------------------------+
|                            RepairBenchmarkRunner                                |
+-------------------------------------------------------------------------------+
|  - method_adapters: tuple[BenchmarkMethodAdapter, ...]                         |
|  - evaluator: PlanEvaluator                                                    |
|  - route_validator: RouteSourceValidator                                       |
+-------------------------------------------------------------------------------+
|  + run_suite(cases): BenchmarkResult        --> Paired method suite            |
|  + run_case(case): tuple[BenchmarkRunResult, ...] --> One disruption/profile   |
|  + export_metrics(result): Path             --> Long-form metrics              |
+-------------------------------------------------------------------------------+
```

```text
+-------------------------------------------------------------------------------+
|                           BenchmarkSplitManager                                 |
+-------------------------------------------------------------------------------+
|  - No persistent internal state                                                |
+-------------------------------------------------------------------------------+
|  + assign_family(case): str                --> Parent/disruption family key    |
|  + split(cases): BenchmarkSplit            --> Train/dev/test split            |
|  + assert_no_leakage(split): None          --> Hard split validation           |
+-------------------------------------------------------------------------------+
```

```text
+-------------------------------------------------------------------------------+
|                              BenchmarkMetrics                                   |
+-------------------------------------------------------------------------------+
|  - No persistent internal state                                                |
+-------------------------------------------------------------------------------+
|  + compute(result): dict[str, float | str]  --> Required metric row            |
|  + preservation(diff): dict[str, float]     --> Locked/booked/locality         |
|  + route_metrics(route): dict[str, float]   --> Road-valid travel/budget       |
+-------------------------------------------------------------------------------+
```

## Class Diagram Rules

1. Diagram generator, ownership factory, method adapter, runner, split manager, and metrics.
2. Benchmark runners must consume artifacts, not mutable solver dictionaries.
3. Baselines must expose requested/executed method and fallback lineage.
4. Ineligible plans can be reported but not ranked as final comparison winners.
5. LLM-only baseline is optional and disabled unless frozen input and evaluation exist.
6. Split manager is stateless and must fail closed on leakage.

## Data Model

```python
class OwnershipProfile(StrEnum):
    FLEXIBLE = "flexible"
    MIXED = "mixed"
    BOOKING_CONSTRAINED = "booking_constrained"
    SCENIC_ROUTE_PROTECTIVE = "scenic_route_protective"
    HIGH_LOCK = "high_lock"

class BenchmarkMethod(StrEnum):
    LLM_ONLY_REWRITE = "llm_only_rewrite"
    FULL_REOPTIMIZATION = "full_reoptimization"
    CURRENT_HYBRID_PIPELINE = "current_hybrid_pipeline"
    WEIGHTED_SUM_REPAIR = "weighted_sum_repair"
    FIXED_RADIUS_REPAIR = "fixed_radius_repair"
    PROGRESSIVE_SEQUENTIAL_LEXICOGRAPHIC_REPAIR = "progressive_sequential_lexicographic_repair"

@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    parent_plan_id: str
    ownership_profile: OwnershipProfile
    disruption_family: DisruptionType
    repair_request_id: str
    catalog_snapshot_id: str
    context_snapshot_id: str
    split_family_id: str

@dataclass(frozen=True)
class BenchmarkRunResult:
    case_id: str
    method_id: BenchmarkMethod
    planner_run_ids: tuple[str, ...]
    child_plan_id: str | None
    diff_id: str | None
    evaluation_id: str | None
    explanation_ids: tuple[str, ...]
    metrics: dict[str, float | str]
```

Required metrics:

- `LockedPreservation`
- `BookedPreservation`
- `UnaffectedDayPreservation`
- `WeightedEditCost`
- `UtilityRetained`
- `UtilityRegret`
- `WeatherRiskReduction`
- `NatureExposureReduction`
- `RepairRadius`
- `CertificateCoverage`
- `ExplanationEvidenceCoverage`
- `FallbackRate`
- runtime
- solver gap/status
- route-valid travel time
- budget delta

## Storage / State

Persistent state:

- `data/benchmark/disruptions/*.json`
- `data/benchmark/ownership_profiles/*.json`
- `runs/<run_id>/benchmark/cases.jsonl`
- `runs/<run_id>/benchmark/results.jsonl`
- `runs/<run_id>/benchmark/metrics/benchmark_long_form.csv`
- `runs/<run_id>/benchmark/splits.json`
- `runs/<run_id>/benchmark/manifest.json`

Temporary state:

- Per-method solver objects and generated prompt payloads are temporary.
- LLM baseline prompt/output snapshots, if used, become artifacts with redaction and evaluator results.

## Required Methods

```python
def generate_six_disruptions(parent: PlanArtifactV2, bundle: DatasetBundle) -> tuple[RepairRequest, ...]
def apply_ownership_profile(parent: PlanArtifactV2, profile: OwnershipProfile) -> PlanArtifactV2
def run_benchmark_case(case: BenchmarkCase, methods: tuple[BenchmarkMethodAdapter, ...]) -> tuple[BenchmarkRunResult, ...]
def compute_benchmark_metrics(result: BenchmarkRunResult, artifacts: ArtifactStore) -> dict[str, float | str]
def split_by_parent_disruption_family(cases: tuple[BenchmarkCase, ...]) -> BenchmarkSplit
def assert_no_parent_family_leakage(split: BenchmarkSplit) -> None
```

## Validation Rules

1. All methods in one benchmark case use identical catalog snapshot, context snapshot, parent plan, ownership profile, disruption snapshot, and route matrix.
2. Benchmark case IDs are deterministic and content-hashed.
3. Rows from the same parent/disruption family cannot cross train/dev/test.
4. LLM-only rewrite is disabled unless frozen input, frozen output, and evaluator are available.
5. Route-valid travel metrics are blank/ineligible when route source is fallback.
6. Evaluator-ineligible plans cannot count as final comparison successes.
7. Fallback rate is reported separately.
8. Utility fusion is missing-source-aware; missing Yelp does not mean low Yelp.
9. Certificate and explanation coverage metrics reference actual artifact IDs.

## UI / API Integration

Internal and report integration:

- Benchmark artifacts feed later dashboard reports but no interactive UI is required in this gate.
- Existing method comparison exports can be wrapped by `CURRENT_HYBRID_PIPELINE` adapter with clear fallback labels.
- Benchmark result tables should be consumable by notebooks after G6.

## Workflow

1. Load canonical parent plans.
2. Apply each ownership profile.
3. Generate six disruptions per parent/profile.
4. Create benchmark cases and split by parent/disruption family.
5. For each case, run permitted method adapters.
6. Evaluate every produced plan independently.
7. Generate diffs and explanation evidence where available.
8. Compute metrics.
9. Export long-form results.
10. Run leakage and eligibility validation.

## Files to Create

- `src/itinerary_system/benchmark/__init__.py`
- `src/itinerary_system/benchmark/disruptions.py`
- `src/itinerary_system/benchmark/ownership_profiles.py`
- `src/itinerary_system/benchmark/methods.py`
- `src/itinerary_system/benchmark/runner.py`
- `src/itinerary_system/benchmark/splits.py`
- `src/itinerary_system/benchmark/metrics.py`
- `src/itinerary_system/benchmark/errors.py`
- `data/benchmark/disruptions/`
- `data/benchmark/ownership_profiles/`
- `tests/benchmark/test_disruptions.py`
- `tests/benchmark/test_ownership_profiles.py`
- `tests/benchmark/test_methods.py`
- `tests/benchmark/test_metrics.py`
- `tests/benchmark/test_no_leakage.py`

## Files to Modify

- `src/itinerary_system/experiment_runner.py` only through adapter wrappers or stable public functions.
- `src/itinerary_system/repair/progressive.py`
- `src/itinerary_system/evaluation/plan_evaluator.py`
- `src/itinerary_system/explanation/evidence.py`
- `src/itinerary_system/__init__.py`
- `README.md` after implementation only.

## Error Handling

- Missing parent fixture: benchmark setup error.
- Missing route-valid matrix: block publication benchmark or mark non-publication demo.
- Method adapter unsupported for a case: record skipped method with reason.
- LLM baseline missing frozen output: skip and report unavailable, not failed competitor.
- Split leakage: hard failure.
- Ineligible plan: metric row records ineligible status and excludes final rank.
- Missing explanation evidence: explanation coverage less than one, not fabricated text.

## Testing Checklist

- [ ] All six disruption families generate deterministic typed `RepairRequest`s.
- [ ] Five ownership profiles produce expected constraint strengths.
- [ ] Paired method runs share identical input hashes.
- [ ] Benchmark split has no parent/disruption leakage.
- [ ] LLM-only method is disabled without frozen evaluation.
- [ ] Full reoptimization baseline has no preservation objective unless explicitly modeled.
- [ ] Weighted-sum baseline differs from sequential lexicographic method.
- [ ] Fixed-radius baseline uses one configured radius.
- [ ] Progressive method reports accepted radius.
- [ ] Route fallback gates route-valid metrics.
- [ ] Certificate and explanation coverage use artifact refs.
- [ ] `python -m ruff check src tests scripts`
- [ ] `python -m pytest`

## Roadmap / Full Pipeline Update

After implementation and benchmark validation pass, update `BENCH-001` and `BENCH-002` in the technical specification. This gate connects to G6 by producing artifacts the authoritative pipeline and notebook/dashboard can reproduce. Do not claim IUI/CHI readiness until G7/G8.

## Acceptance Criteria

1. Six disruptions and five ownership profiles are generated deterministically.
2. Baselines run through a common artifact/evaluator contract.
3. Metrics include preservation before utility.
4. No split leakage occurs.
5. No ineligible or geodesic-fallback plan is reported as road-valid final evidence.
6. Benchmark package can support computational repair claims only after strict route validation passes.

# Phase G6.0 Implementation Plan: Authoritative Pipeline Runner And Interactive Prototype

## Goal

Implement **authoritative pipeline runner and interactive prototype**.

This gate should allow the system to:

1. Run generation and repair from one package-level entry point.
2. Emit immutable run directories with requests, planner runs, plans, diffs, routing, evaluations, explanations, metrics, and dashboard artifacts.
3. Make notebooks thin display wrappers and expose an interactive artifact-backed prototype.

Keep business logic in package modules, not notebooks.

## Current State

Repository evidence found:

- `src/itinerary_system/experiment_runner.py`: large production pipeline helper with method comparisons and dashboard output preparation.
- `notebook/production_system_blueprint.ipynb`: current notebook execution path.
- `README.md`: documents nbconvert execution and dashboard serving.
- `scripts/run_phase0_evidence_pipeline.py`: Phase 0 evidence command, but not full research pipeline runner.
- `scripts/serve_dashboard.py`: serves generated dashboard artifacts.
- `docs/planning/travel_itinerary_repair_technical_specification.md`: specifies `PipelineRun` and `run_research_pipeline()`.

Missing pieces:

- No `src/itinerary_system/pipeline_runner.py`.
- No immutable `runs/<run_id>/` output layout.
- No redacted resolved config.
- No generation/repair mode under one typed return object.
- No notebook-to-run consistency tests.
- No interactive backend or artifact-backed repair prototype.

What should not be claimed or implemented yet:

- Do not let notebook cells call APIs, build Gurobi models, mutate plan dictionaries, create snapshots, or compute canonical evaluation metrics.
- Do not start user-study claims until explanation and event logging are G7-ready.
- Do not overwrite run directories.

## New Components to Add

### Component 1

`PipelineRun`

Responsibility:

Typed result object for one generation or repair pipeline run.

### Component 2

`ResearchPipelineRunner`

Responsibility:

Resolve config, create immutable run directory, load snapshots, resolve providers, load/generate parent, run generation or repair, route, evaluate, diff, explain, export, and return `PipelineRun`.

### Component 3

`RunArtifactStore`

Responsibility:

Write and read immutable run artifacts using stable JSON/JSONL/CSV layout.

### Component 4

`ThinNotebookAdapter`

Responsibility:

Expose notebook-safe calls to `run_research_pipeline()` and completed-run display helpers.

### Component 5

`InteractiveRepairPrototypeController`

Responsibility:

Serve artifact-backed parent/child/diff/certificate/explanation views and accept typed edit requests without putting business logic in browser-only previews.

## Class / Registry Diagrams

```text
+-------------------------------------------------------------------------------+
|                                 PipelineRun                                     |
+-------------------------------------------------------------------------------+
|  - run_id: str                                                                 |
|  - data_bundle: DatasetBundle                                                  |
|  - parent_plan: PlanArtifactV2 | None                                          |
|  - planner_runs: tuple[PlannerRun, ...]                                        |
|  - output_plans: tuple[PlanArtifactV2, ...]                                    |
|  - evaluations: tuple[ResearchEvaluationReport, ...]                          |
|  - output_dir: Path                                                            |
+-------------------------------------------------------------------------------+
|  + manifest_record(): dict[str, Any]       --> Manifest payload                |
|  + strict_success(): bool                  --> All required plans eligible     |
+-------------------------------------------------------------------------------+
```

```text
+-------------------------------------------------------------------------------+
|                         ResearchPipelineRunner                                  |
+-------------------------------------------------------------------------------+
|  - output_root: Path                                                           |
|  - clock: Callable[[], datetime]                                               |
|  - artifact_store_factory: Callable                                            |
+-------------------------------------------------------------------------------+
|  + run_research_pipeline(...): PipelineRun  --> Authoritative entry point      |
|  + resolve_config(...): TripConfig          --> Config + overrides             |
|  + run_generation(...): tuple[PlanArtifactV2, ...] --> Parent/generation mode  |
|  + run_repair(...): ProgressiveRepairOutcome --> Repair mode                   |
+-------------------------------------------------------------------------------+
```

```text
+-------------------------------------------------------------------------------+
|                    RunArtifactStore (Persistent Registry / Manager)             |
+-------------------------------------------------------------------------------+
|  - run_dir: Path                                                               |
|  - manifest_path: Path                                                         |
+-------------------------------------------------------------------------------+
|  + create(): None                          --> Refuse existing run dir         |
|  + write_json(name, payload): Path         --> Stable artifact write           |
|  + append_jsonl(name, payload): Path       --> Run records                     |
|  + write_manifest(run): Path               --> Top-level manifest              |
|  + load_run(run_dir): PipelineRunView      --> Read completed outputs          |
+-------------------------------------------------------------------------------+
```

```text
+-------------------------------------------------------------------------------+
|                         ThinNotebookAdapter                                     |
+-------------------------------------------------------------------------------+
|  - No persistent internal state                                                |
+-------------------------------------------------------------------------------+
|  + run_from_config(config_path): PipelineRun --> Calls package runner          |
|  + load_completed_run(run_dir): dict        --> Display-only payloads          |
|  + dashboard_links(run_dir): dict[str, Path] --> Notebook links                |
+-------------------------------------------------------------------------------+
```

```text
+-------------------------------------------------------------------------------+
|                 InteractiveRepairPrototypeController (UI Controller)            |
+-------------------------------------------------------------------------------+
|  - artifact_store: RunArtifactStore                                            |
|  - pipeline_runner: ResearchPipelineRunner                                     |
+-------------------------------------------------------------------------------+
|  + list_parent_plans(): tuple[PlanSummary, ...] --> Parent selection           |
|  + submit_edit(request): RepairOutcome      --> Backend typed repair           |
|  + diff_view(parent, child): PlanDiff       --> Artifact-backed diff           |
|  + certificate_view(plan): EvaluationCertificate --> Evidence badge            |
+-------------------------------------------------------------------------------+
```

## Class Diagram Rules

1. Diagram pipeline run, runner, artifact store, notebook adapter, and UI controller.
2. Pipeline orchestration belongs in package code.
3. Notebook adapter and UI controller must call package services, not duplicate logic.
4. Run directories are immutable.
5. Resolved configs must be redacted before serialization.
6. Generated artifacts stay under `runs/<run_id>/` or ignored output directories.

## Data Model

```python
class RefreshPolicy(StrEnum):
    NEVER = "never"
    CACHE_ONLY = "cache_only"
    APPROVED_REFRESH = "approved_refresh"

@dataclass(frozen=True)
class PipelineRun:
    run_id: str
    data_bundle: DatasetBundle
    parent_plan: PlanArtifactV2 | None
    planner_runs: tuple[PlannerRun, ...]
    output_plans: tuple[PlanArtifactV2, ...]
    evaluations: tuple[ResearchEvaluationReport, ...]
    output_dir: Path

def run_research_pipeline(
    *,
    config_path: Path,
    catalog_snapshot_id: str,
    context_snapshot_id: str,
    parent_plan_id: str | None = None,
    repair_request_id: str | None = None,
    refresh_policy: RefreshPolicy = RefreshPolicy.NEVER,
    run_id: str | None = None,
) -> PipelineRun:
    ...
```

Output layout:

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

## Storage / State

Persistent state:

- Immutable `runs/<run_id>/` directories.
- Existing `results/outputs` and dashboard paths remain as compatibility outputs until notebook migration completes.

Temporary state:

- Solver models, provider clients, and notebook display objects are temporary.

## Required Methods

```python
def create_run_dir(output_root: Path, run_id: str) -> Path
def write_resolved_config_redacted(config: TripConfig, path: Path) -> Path
def run_generation_mode(config: TripConfig, bundle: DatasetBundle, store: RunArtifactStore) -> PipelineRun
def run_repair_mode(config: TripConfig, parent_plan_id: str, repair_request_id: str, store: RunArtifactStore) -> PipelineRun
def export_dashboard_artifacts(run: PipelineRun) -> Path
def load_completed_run(run_dir: Path) -> PipelineRunView
```

Notebook rule checks:

```python
def assert_notebook_is_thin(notebook_path: Path) -> tuple[Finding, ...]
def compare_notebook_to_run_outputs(notebook_path: Path, run_dir: Path) -> tuple[Finding, ...]
```

## Validation Rules

1. Run directory cannot already exist.
2. `RefreshPolicy.NEVER` prevents live API calls.
3. Resolved config serialization must redact secrets and environment variable values.
4. Every output plan has source run, parent lineage if applicable, diff if child, routing, evaluation, and certificate state.
5. Strict mode blocks ineligible final plans after writing diagnostics.
6. Notebook must not call APIs directly.
7. Notebook must not construct Gurobi models.
8. Notebook must not mutate plan dictionaries.
9. Notebook must not create canonical snapshots.
10. Notebook must not compute canonical evaluation metrics independently.

## UI / API Integration

Package/API integration:

- `run_research_pipeline()` becomes the authoritative entry point.
- Notebook cells resolve config, call the runner, display outputs, and link dashboard artifacts.
- Interactive prototype submits typed edit requests to backend package services.
- Existing static dashboard can be migrated to load parent, child, diff, certificate, and explanation artifacts.

## Workflow

1. Resolve config and run ID.
2. Create immutable run directory.
3. Write redacted resolved config.
4. Load and validate catalog/context snapshots.
5. Resolve providers according to `RefreshPolicy`.
6. Load/generate parent plan.
7. Run generation or repair.
8. Build route records from route matrix.
9. Evaluate and certificate final plans.
10. Compute diff and explanations.
11. Export metrics and dashboard assets.
12. Return `PipelineRun`.
13. Notebook displays completed run only.

## Files to Create

- `src/itinerary_system/pipeline_runner.py`
- `src/itinerary_system/run_artifacts.py`
- `src/itinerary_system/notebook_adapter.py`
- `src/itinerary_system/prototype/__init__.py`
- `src/itinerary_system/prototype/controller.py`
- `tests/test_pipeline_runner.py`
- `tests/test_run_artifacts.py`
- `tests/test_notebook_adapter.py`
- `tests/prototype/test_controller.py`

## Files to Modify

- `src/itinerary_system/experiment_runner.py`
- `src/itinerary_system/map_exporter.py`
- `src/itinerary_system/map_renderer.py`
- `src/itinerary_system/__init__.py`
- `notebook/production_system_blueprint.ipynb`
- `README.md`
- `docs/reference/code_quality_workflow.md`
- `tests/test_configurable_itinerary_system.py`

## Error Handling

- Existing run directory: `RunDirectoryExists`.
- Missing snapshot: typed snapshot error from existing data layer.
- Provider disabled or missing credentials: record unavailable status and send no normal request.
- Strict ineligible plan: pipeline returns/writes diagnostics and raises `PipelineIneligiblePlanError` for strict command contexts.
- Notebook still contains business logic: thin-notebook validation failure.
- Dashboard export missing artifact: dashboard validation failure.

## Testing Checklist

- [ ] Pipeline creates immutable run directory.
- [ ] Pipeline refuses existing run directory.
- [ ] Resolved config is redacted.
- [ ] Refresh policy `never` sends no live API requests.
- [ ] Generation mode emits run, plan, routing, evaluation, manifest.
- [ ] Repair mode emits parent, child, diff, routing, evaluation, explanation, manifest.
- [ ] Strict mode blocks ineligible final plan.
- [ ] Notebook calls package runner.
- [ ] Notebook does not construct models or call APIs.
- [ ] Notebook-to-run consistency passes.
- [ ] Interactive prototype uses backend repair, not browser-only mutation.
- [ ] `python -m ruff check src tests scripts`
- [ ] `python -m pytest`

## Roadmap / Full Pipeline Update

After implementation and validation pass, update `PIPE-001` and `NOTEBOOK-001` in the technical specification. This gate connects to G7 by making explanations, study logs, and UI replay possible from stable artifacts. Do not claim user-study readiness until explanation evidence and logging are complete.

## Acceptance Criteria

1. `run_research_pipeline()` is the authoritative entry point.
2. Run artifacts are immutable, complete, and redacted.
3. Notebook is thin and display-only.
4. Interactive prototype is artifact-backed.
5. Strict mode prevents ineligible plans from being published or ranked.

# Phase G6.5 Implementation Plan: Evidence-Bounded LLM Preference-To-Commitment Layer

## Goal

Implement **evidence-bounded LLM preference-to-commitment support**.

This work package should allow the system to:

1. Parse natural-language repair language into typed, inactive edit or counterfactual requests.
2. Infer inspectable taste hypotheses from the accepted parent itinerary without treating those hypotheses as user truth.
3. Suggest inactive protection labels for existing parent-itinerary elements.
4. Validate every LLM suggestion deterministically before showing it.
5. Show taste and commitment suggestions on the map for user confirmation, rejection, downgrade, or upgrade.
6. Compile only confirmed decisions into active `OwnedConstraint` records for downstream repair.

Keep the LLM out of itinerary generation, feasibility judgment, route validation, hotel availability, objective construction, solver certification, and independent evaluation.

## Current State

Repository evidence found:

- `src/itinerary_system/request_schema.py`: has structured trip-planning request objects, but no LLM-safe repair/edit/counterfactual parser contract.
- `src/itinerary_system/repair_planner.py`: has deterministic repair intent and operation scaffolds, but no parent-plan taste evidence pack, prompt-run artifact, or commitment suggestion validator.
- G1 in this plan defines `OwnedConstraint`, `ConstraintOrigin.LLM_INTERPRETATION`, active/inactive rules, parent-child lineage, and typed diffs.
- G3/G4 in this plan define ownership-aware repair and progressive neighborhoods, but require confirmed constraints as inputs.
- G6 in this plan defines an artifact-backed pipeline and prototype controller that can host map confirmation cards after run artifacts are stable.
- G7 in this plan defines deterministic explanations and optional LLM verbalization bounded by evidence refs.
- `docs/literature/repair_gap_review.md` and `docs/literature/literature_matrix_repair_gap.md` make broad "LLM travel planner" claims unsafe but support the narrower repair-commitment contribution.
- `docs/literature/evidence_matrix.md` marks LLM-to-symbolic interpretation as unsupported/planned and warns that user-understanding claims require study evidence.

Missing pieces:

- No `src/itinerary_system/llm_taste/` package.
- No `TasteEvidencePack` built from immutable parent artifacts.
- No controlled `TasteDimension` vocabulary.
- No prompt registry, prompt versioning, or `PromptRun` record.
- No `TasteSignalArtifact`, `TasteProfileArtifact`, or `SuggestedCommitmentArtifact`.
- No deterministic validator for LLM target IDs, evidence refs, strength rules, booking claims, sensitive inferences, or hidden solver directives.
- No map-card contract for suggested versus validated versus confirmed versus active states.
- No `UserCommitmentDecision` compiler into confirmed `OwnedConstraint` records.
- No counterfactual-question parser that produces sandbox `RepairRequest`s instead of free-form answers.

What should not be claimed or implemented yet:

- Do not claim that the LLM knows the user's true taste.
- Do not claim that LLM taste inference is calibrated uncertainty.
- Do not let inferred taste create `LOCKED` or `BOOKED` constraints.
- Do not activate LLM-interpreted constraints without explicit user confirmation.
- Do not let the LLM generate final itineraries for publication comparison.
- Do not let the LLM decide road validity, weather feasibility, hotel availability, or final eligibility.
- Do not store raw LLM conversations, secrets, or unredacted personal itineraries by default.

## New Components to Add

### Component 1

`TasteEvidencePackBuilder`

Responsibility:

Build a bounded, artifact-backed evidence packet from an immutable parent `PlanArtifactV2`, route/context records, catalog records, and existing constraints. The pack is the only input the taste prompts can inspect.

### Component 2

`PromptRegistry`

Responsibility:

Store prompt names, versions, allowed vocabularies, schema IDs, and runtime policies for taste extraction, profile compression, commitment generation, critic review, clarification, correction parsing, and counterfactual translation.

### Component 3

`LlmTasteSignalExtractor`

Responsibility:

Run narrow, structured prompt stages that produce taste signals and taste profile artifacts from `TasteEvidencePack` only.

### Component 4

`SuggestedCommitmentGenerator`

Responsibility:

Generate inactive commitment suggestions over existing parent-plan targets, with evidence refs, taste refs, confidence support scores, and repair tradeoff text.

### Component 5

`CommitmentSuggestionValidator`

Responsibility:

Deterministically validate or downgrade LLM suggestions before they appear in the UI or compiler.

### Component 6

`ClarificationQuestionPlanner`

Responsibility:

Ask at most repair-relevant clarification questions when uncertainty would materially affect active constraints.

### Component 7

`UserCommitmentCompiler`

Responsibility:

Convert confirmed user decisions into active or inactive `OwnedConstraint` records according to ownership rules.

### Component 8

`MapCommitmentCardBuilder`

Responsibility:

Create artifact-backed map/sidebar card payloads that separate taste hypotheses, suggested commitments, validated suggestions, confirmed decisions, and solver-active constraints.

### Component 9

`CounterfactualQuestionInterpreter`

Responsibility:

Translate user "why not" and "what if" language into structured sandbox repair/counterfactual requests. It must not answer from intuition; answers come only after solver/evaluator/counterfactual artifacts exist.

### Component 10

`PromptRunRepository`

Responsibility:

Persist prompt input hashes, output hashes, prompt versions, validation status, and redacted metadata under the run directory.

## Class / Registry Diagrams

```text
+-------------------------------------------------------------------------------+
|                TasteEvidencePackBuilder (Stateless Utility Module)             |
+-------------------------------------------------------------------------------+
|  - No persistent internal state                                                |
+-------------------------------------------------------------------------------+
|  + build(parent, dataset, routes): TasteEvidencePack --> Bounded LLM input     |
|  + candidate_entities(parent): tuple[TasteCandidateEntity, ...] --> Targets    |
|  + evidence_refs(parent): tuple[str, ...]       --> Valid source refs          |
+-------------------------------------------------------------------------------+
```

```text
+-------------------------------------------------------------------------------+
|                 PromptRegistry (Persistent Registry / Manager)                 |
+-------------------------------------------------------------------------------+
|  - registry_path: Path                                                         |
|  - prompt_versions: dict[str, PromptSpec]                                      |
+-------------------------------------------------------------------------------+
|  + get(name, version): PromptSpec          --> Versioned prompt contract       |
|  + allowed_vocabularies(): dict[str, tuple[str, ...]] --> Enum projection      |
|  + validate_prompt_spec(spec): tuple[Finding, ...] --> Schema guardrail        |
+-------------------------------------------------------------------------------+
```

```text
+-------------------------------------------------------------------------------+
|                         LlmTasteSignalExtractor (Service Class)                |
+-------------------------------------------------------------------------------+
|  - prompt_registry: PromptRegistry                                             |
|  - llm_client: LlmClient | None                                                |
|  - prompt_runs: PromptRunRepository                                            |
+-------------------------------------------------------------------------------+
|  + extract_signals(pack): TasteSignalArtifact --> Evidence-backed signals      |
|  + compress_profile(signals): TasteProfileArtifact --> User-editable summary   |
|  + run_prompt(spec, payload): PromptRun      --> Stored prompt record          |
+-------------------------------------------------------------------------------+
```

```text
+-------------------------------------------------------------------------------+
|                       SuggestedCommitmentGenerator (Service Class)             |
+-------------------------------------------------------------------------------+
|  - prompt_registry: PromptRegistry                                             |
|  - llm_client: LlmClient | None                                                |
|  - validator: CommitmentSuggestionValidator                                    |
+-------------------------------------------------------------------------------+
|  + suggest(pack, profile): SuggestedCommitmentArtifact --> Inactive labels     |
|  + critic_review(suggestions): PromptCriticReport --> Optional LLM critic      |
|  + validate_and_filter(...): CommitmentValidationReport --> Deterministic gate |
+-------------------------------------------------------------------------------+
```

```text
+-------------------------------------------------------------------------------+
|              CommitmentSuggestionValidator (Stateless Utility Module)          |
+-------------------------------------------------------------------------------+
|  - No persistent internal state                                                |
+-------------------------------------------------------------------------------+
|  + validate(suggestions, pack): CommitmentValidationReport --> Accept/downgrade|
|  + validate_target_id(target_id): Finding | None --> Parent/candidate target   |
|  + validate_strength(suggestion): Finding | None --> Lock/book/strong rules    |
|  + reject_sensitive_inference(text): Finding | None --> Safety guardrail       |
+-------------------------------------------------------------------------------+
```

```text
+-------------------------------------------------------------------------------+
|                    ClarificationQuestionPlanner (Service Class)                |
+-------------------------------------------------------------------------------+
|  - max_questions: int                                                          |
|  - prompt_registry: PromptRegistry                                             |
+-------------------------------------------------------------------------------+
|  + build_questions(profile, suggestions): tuple[ClarificationQuestion, ...]    |
|  + materiality_filter(question): bool       --> Ask only repair-relevant items |
+-------------------------------------------------------------------------------+
```

```text
+-------------------------------------------------------------------------------+
|                 UserCommitmentCompiler (Stateless Utility Module)              |
+-------------------------------------------------------------------------------+
|  - No persistent internal state                                                |
+-------------------------------------------------------------------------------+
|  + parse_correction(text, candidates): CandidateCommitmentUpdate               |
|  + compile(decisions, suggestions): tuple[OwnedConstraint, ...]                |
|  + validate_confirmation(decision): tuple[Finding, ...] --> Required user act  |
+-------------------------------------------------------------------------------+
```

```text
+-------------------------------------------------------------------------------+
|                    MapCommitmentCardBuilder (UI Controller)                    |
+-------------------------------------------------------------------------------+
|  - No persistent internal state                                                |
+-------------------------------------------------------------------------------+
|  + build_taste_cards(profile): tuple[MapCommitmentCard, ...]                  |
|  + build_suggestion_cards(report): tuple[MapCommitmentCard, ...]              |
|  + build_confirmed_cards(decisions): tuple[MapCommitmentCard, ...]            |
+-------------------------------------------------------------------------------+
```

```text
+-------------------------------------------------------------------------------+
|                  CounterfactualQuestionInterpreter (Service Class)             |
+-------------------------------------------------------------------------------+
|  - prompt_registry: PromptRegistry                                             |
|  - validator: CommitmentSuggestionValidator                                    |
+-------------------------------------------------------------------------------+
|  + interpret(question, targets): CounterfactualRequestDraft                    |
|  + validate_draft(draft): tuple[Finding, ...] --> No unsupported answer        |
+-------------------------------------------------------------------------------+
```

```text
+-------------------------------------------------------------------------------+
|                 PromptRunRepository (Persistent Registry / Manager)            |
+-------------------------------------------------------------------------------+
|  - run_dir: Path                                                               |
|  - prompt_runs_path: Path                                                      |
+-------------------------------------------------------------------------------+
|  + append(run): Path                       --> Redacted prompt metadata        |
|  + get(prompt_run_id): PromptRun | None    --> Replay prompt metadata          |
|  + verify_hashes(run): tuple[Finding, ...] --> Input/output hash check         |
+-------------------------------------------------------------------------------+
```

## Class Diagram Rules

1. Diagram every LLM-facing service, deterministic validator, compiler, registry, and UI card builder.
2. LLM service classes may call a configured `LlmClient`; stateless validators and compilers must not.
3. Every class that stores prompt metadata must redact secrets and avoid raw personal text by default.
4. Every LLM output class must include evidence refs and a validation status.
5. UI controllers display artifact-backed suggestions only; they cannot activate constraints by themselves.
6. Counterfactual interpretation returns structured requests, not final explanations.
7. If no LLM provider is configured, use frozen fixture outputs for benchmark/dev tests or mark the feature unavailable.

## Data Model

Controlled vocabularies:

```python
class TasteDimension(StrEnum):
    SCENIC_PHOTO = "scenic_photo"
    NATURE_OUTDOOR = "nature_outdoor"
    ARCHITECTURE_HISTORY = "architecture_history"
    MUSEUM_CULTURE = "museum_culture"
    FOOD_LOCAL_EXPERIENCE = "food_local_experience"
    SOCIAL_MUST_GO = "social_must_go"
    RELAXED_PACE = "relaxed_pace"
    DENSE_EXPLORER_PACE = "dense_explorer_pace"
    LOW_HOTEL_SWITCHING = "low_hotel_switching"
    SCENIC_DRIVING = "scenic_driving"
    LOW_DRIVING_BURDEN = "low_driving_burden"
    WEATHER_CAUTIOUS = "weather_cautious"
    BUDGET_SENSITIVE = "budget_sensitive"
    ACCESSIBILITY_SENSITIVE = "accessibility_sensitive"
    INDOOR_BACKUP_PREFERENCE = "indoor_backup_preference"

class TasteLevel(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    UNCERTAIN = "uncertain"

class SuggestionStatus(StrEnum):
    SUGGESTED_NOT_ACTIVE = "suggested_not_active"
    VALIDATED_NOT_ACTIVE = "validated_not_active"
    REJECTED = "rejected"
    CONFIRMED = "confirmed"
    NEEDS_CLARIFICATION = "needs_clarification"
```

Evidence packet:

```python
@dataclass(frozen=True)
class TasteEvidencePack:
    pack_id: str
    parent_plan_id: str
    plan_content_hash: str
    catalog_snapshot_id: str
    context_snapshot_id: str
    trip_summary: TripTasteSummary
    candidate_entities: tuple[TasteCandidateEntity, ...]
    route_segments: tuple[TasteRouteSegment, ...]
    lodging_entities: tuple[TasteLodgingEntity, ...]
    existing_constraints: tuple[OwnedConstraint, ...]
    evidence_refs: tuple[str, ...]
    created_at: str
    schema_version: str = "taste-evidence-pack-v1"

@dataclass(frozen=True)
class TasteCandidateEntity:
    entity_id: str
    entity_type: Literal["poi", "road_segment", "lodging", "day_structure", "time_window"]
    display_name: str
    day_ids: tuple[int, ...]
    categories: tuple[str, ...]
    tags: tuple[str, ...]
    route_role: str | None
    weather_sensitivity: str | None
    source_refs: tuple[str, ...]
    evidence_refs: tuple[str, ...]
```

Taste and commitment artifacts:

```python
@dataclass(frozen=True)
class TasteSignal:
    signal_id: str
    dimension: TasteDimension
    level: TasteLevel
    confidence: float
    evidence_refs: tuple[str, ...]
    reason: str
    uncertainty: str | None

@dataclass(frozen=True)
class TasteProfileItem:
    profile_item_id: str
    dimension: TasteDimension
    level: TasteLevel
    confidence: float
    display_label: str
    display_summary: str
    evidence_refs: tuple[str, ...]
    status: Literal["hypothesis_not_confirmed", "confirmed", "rejected", "edited"]

@dataclass(frozen=True)
class SuggestedCommitment:
    suggestion_id: str
    target_id: str
    target_type: ConstraintScope
    suggested_strength: ConstraintStrength
    confidence: float
    map_badge: str
    short_reason: str
    detailed_reason: str
    evidence_refs: tuple[str, ...]
    taste_profile_refs: tuple[str, ...]
    possible_repair_tradeoff: str
    requires_confirmation: bool
    status: SuggestionStatus

@dataclass(frozen=True)
class UserCommitmentDecision:
    decision_id: str
    suggestion_id: str
    target_id: str
    action: Literal["accept", "reject", "downgrade", "upgrade", "keep_flexible", "clarify"]
    selected_strength: ConstraintStrength | None
    user_note: str | None
    created_at: str
```

Prompt metadata:

```python
@dataclass(frozen=True)
class PromptRun:
    prompt_run_id: str
    parent_artifact_id: str
    prompt_name: str
    prompt_version: str
    model_name: str
    input_hash: str
    output_hash: str
    created_at: str
    validation_status: Literal["valid", "invalid", "needs_review"]
    schema_version: str = "prompt-run-v1"
```

Prompt contracts:

```text
Taste signal extraction:
- Input: TasteEvidencePack.
- Output: tuple[TasteSignal, ...] plus unsupported_inferences.
- Rule: every signal cites evidence_refs and uses only TasteDimension/TasteLevel.

Taste profile compression:
- Input: TasteSignalArtifact.
- Output: tuple[TasteProfileItem, ...] plus profile_uncertainties.
- Rule: user-facing phrasing says "the plan suggests", not "the user is".

Suggested commitment generation:
- Input: TasteEvidencePack plus TasteProfileArtifact.
- Output: tuple[SuggestedCommitment, ...] plus not_suggested.
- Rule: inferred taste maps only to STRONG, WEAK, or TEST_ONLY unless explicit lock/booking evidence exists.

Clarification:
- Input: uncertain profile items plus validated suggestions.
- Output: at most two repair-relevant questions.
- Rule: options map to allowed strengths or keep-flexible actions.

Correction parsing:
- Input: user correction text plus candidate targets and current suggestions.
- Output: candidate decision updates requiring confirmation.
- Rule: ambiguous targets require clarification.

Counterfactual translation:
- Input: why-not or what-if user question plus target inventory.
- Output: structured sandbox RepairRequest or CounterfactualRequestDraft.
- Rule: no free-form causal answer until counterfactual artifacts exist.
```

Confidence values are support scores only. They must not be documented as calibrated probability.

## Storage / State

Persistent state:

```text
runs/<run_id>/llm_taste/
|-- evidence_packs/
|-- prompt_runs.jsonl
|-- taste_signals/
|-- taste_profiles/
|-- suggested_commitments/
|-- validation_reports/
|-- clarification_questions/
|-- user_commitment_decisions.jsonl
|-- map_cards/
`-- counterfactual_drafts/
```

Persistent prompt metadata:

- Prompt name and version.
- Model/provider identifier.
- Input hash and output hash.
- Validation status.
- Redacted token/count/runtime metadata if available.

Temporary state:

- LLM clients are runtime-only.
- Raw provider responses are temporary unless explicitly saved in redacted development mode.
- Map UI selection state is temporary until converted to `UserCommitmentDecision`.

## Required Methods

Evidence and prompt execution:

```python
def build_taste_evidence_pack(
    *,
    parent_plan: PlanArtifactV2,
    dataset: DatasetBundle,
    route_matrix: RouteMatrix,
    existing_constraints: tuple[OwnedConstraint, ...],
) -> TasteEvidencePack

def extract_taste_signals(
    pack: TasteEvidencePack,
    *,
    prompt_registry: PromptRegistry,
    llm_client: LlmClient | None,
) -> TasteSignalArtifact

def compress_taste_profile(
    signals: TasteSignalArtifact,
    *,
    prompt_registry: PromptRegistry,
    llm_client: LlmClient | None,
) -> TasteProfileArtifact
```

Commitments:

```python
def generate_suggested_commitments(
    *,
    pack: TasteEvidencePack,
    profile: TasteProfileArtifact,
    prompt_registry: PromptRegistry,
    llm_client: LlmClient | None,
) -> SuggestedCommitmentArtifact

def validate_suggested_commitments(
    *,
    suggestions: SuggestedCommitmentArtifact,
    pack: TasteEvidencePack,
    profile: TasteProfileArtifact,
) -> CommitmentValidationReport

def build_clarifying_questions(
    *,
    profile: TasteProfileArtifact,
    validation_report: CommitmentValidationReport,
    max_questions: int = 2,
) -> tuple[ClarificationQuestion, ...]

def compile_confirmed_decisions_to_owned_constraints(
    *,
    decisions: tuple[UserCommitmentDecision, ...],
    validation_report: CommitmentValidationReport,
    parent_plan: PlanArtifactV2,
) -> tuple[OwnedConstraint, ...]
```

User language:

```python
def parse_user_commitment_correction(
    *,
    user_text: str,
    candidate_entities: tuple[TasteCandidateEntity, ...],
    current_suggestions: tuple[SuggestedCommitment, ...],
    prompt_registry: PromptRegistry,
    llm_client: LlmClient | None,
) -> tuple[CandidateCommitmentUpdate, ...]

def translate_counterfactual_question(
    *,
    user_question: str,
    candidate_entities: tuple[TasteCandidateEntity, ...],
    parent_plan: PlanArtifactV2,
    prompt_registry: PromptRegistry,
    llm_client: LlmClient | None,
) -> CounterfactualRequestDraft
```

Map cards:

```python
def build_map_commitment_cards(
    *,
    profile: TasteProfileArtifact,
    validation_report: CommitmentValidationReport,
    decisions: tuple[UserCommitmentDecision, ...],
) -> tuple[MapCommitmentCard, ...]
```

## Validation Rules

1. `TasteEvidencePack.parent_plan_id` and `plan_content_hash` must match the immutable parent artifact.
2. Candidate target IDs must exist in the parent plan or approved candidate list.
3. Candidate target type must match `ConstraintScope`.
4. Every taste signal must cite at least one valid evidence ref.
5. Every suggested commitment must cite entity evidence and taste-profile evidence.
6. `LOCKED` suggestions require explicit user-lock evidence.
7. `BOOKED` suggestions require booking evidence.
8. Inferred taste alone can produce only `STRONG`, `WEAK`, or `TEST_ONLY`.
9. `TEST_ONLY` suggestions cannot enter an accepted plan unless separately confirmed and converted to an allowed active strength.
10. `ConstraintOrigin.LLM_INTERPRETATION` constraints remain inactive unless `confirmed=True`.
11. No invented POIs, road segments, lodging, weather facts, closure facts, prices, bookings, safety facts, or dates.
12. No sensitive personal inference about demographics, health, income, religion, nationality, identity, or private traits.
13. No hidden objective coefficients, Gurobi code, solver constraints, or feasibility conclusions in LLM output.
14. Map badges must distinguish `suggested_not_active`, `validated_not_active`, `confirmed`, and solver-active states.
15. User-facing prose must say "the accepted itinerary suggests" rather than "you definitely prefer".
16. Counterfactual drafts must require solver/evaluator execution before explanation.

## UI / API Integration

Pipeline integration:

- G6 `run_research_pipeline()` may optionally create taste artifacts after parent-plan artifacts are loaded and before repair constraints are compiled.
- Benchmark/publication runs should use frozen prompt outputs or explicitly configured LLM providers; missing provider means unavailable, not implicit live call.
- `RefreshPolicy.NEVER` must block live LLM requests unless the run uses stored/frozen prompt outputs.

Prototype/map integration:

- Taste hypotheses appear as editable profile cards.
- Suggested commitments appear as inactive map/sidebar cards.
- Card actions: accept, reject, downgrade, upgrade, keep flexible, why, clarify.
- Visible status labels must include: taste hypothesis, suggested not active, validated not active, confirmed, active in solver, rejected, needs clarification.
- A suggestion never appears as active until a `UserCommitmentDecision` compiles into an `OwnedConstraint`.

Repair integration:

- Confirmed constraints feed G3/G4 repair through the same ownership policy layer as explicit user constraints.
- The solver receives only typed `OwnedConstraint` records, never raw prompt text.

Explanation integration:

- G7 deterministic explanation consumes prompt artifacts only as evidence refs for confirmed decisions or rejected suggestions.
- Optional LLM verbalization still receives structured evidence and must return claim-to-evidence mappings.

## Workflow

1. Load immutable parent plan, dataset bundle, route matrix, and existing constraints.
2. Build `TasteEvidencePack` from artifact IDs and evidence refs.
3. Run taste signal extraction or load frozen fixture output.
4. Validate taste signals against allowed vocabulary and evidence refs.
5. Compress taste signals into user-editable `TasteProfileArtifact`.
6. Generate inactive suggested commitments for existing parent-plan elements.
7. Run deterministic validation and downgrade/reject invalid suggestions.
8. Optionally run LLM critic review as a secondary check; deterministic validation remains authoritative.
9. Build map cards and at most two repair-relevant clarification questions.
10. User accepts, rejects, downgrades, upgrades, or keeps flexible.
11. Compile confirmed decisions into `OwnedConstraint` records.
12. Send only confirmed constraints to repair request construction.
13. Store prompt runs, validation reports, map cards, and user decisions under `runs/<run_id>/llm_taste/`.

## Files to Create

- `src/itinerary_system/llm_taste/__init__.py`
- `src/itinerary_system/llm_taste/models.py`
- `src/itinerary_system/llm_taste/evidence_pack.py`
- `src/itinerary_system/llm_taste/prompts.py`
- `src/itinerary_system/llm_taste/prompt_registry.py`
- `src/itinerary_system/llm_taste/prompt_runs.py`
- `src/itinerary_system/llm_taste/profile.py`
- `src/itinerary_system/llm_taste/commitments.py`
- `src/itinerary_system/llm_taste/validator.py`
- `src/itinerary_system/llm_taste/clarification.py`
- `src/itinerary_system/llm_taste/map_cards.py`
- `src/itinerary_system/llm_taste/counterfactual.py`
- `tests/llm_taste/test_evidence_pack.py`
- `tests/llm_taste/test_prompt_contracts.py`
- `tests/llm_taste/test_commitment_validator.py`
- `tests/llm_taste/test_user_commitment_compiler.py`
- `tests/llm_taste/test_map_cards.py`
- `tests/llm_taste/test_counterfactual_parser.py`

## Files to Modify

- `src/itinerary_system/pipeline_runner.py`
- `src/itinerary_system/prototype/controller.py`
- `src/itinerary_system/repair/requests.py`
- `src/itinerary_system/ownership/models.py`
- `src/itinerary_system/explanation/evidence.py`
- `src/itinerary_system/study/events.py`
- `docs/user_study_protocol.md`
- `docs/planning/travel_itinerary_repair_technical_specification.md`
- `README.md` only after the prototype surface is stable.

## Error Handling

- LLM provider disabled: return `LlmTasteUnavailable` and allow manual constraints or frozen fixtures.
- Prompt output invalid JSON: store invalid `PromptRun`, reject output, and show no active suggestion.
- Unknown target ID: reject suggestion.
- Missing evidence refs: reject suggestion.
- `LOCKED` without explicit user evidence: downgrade to `STRONG` or reject according to policy.
- `BOOKED` without booking evidence: reject or downgrade to `STRONG`; never keep booked label.
- Sensitive inference: reject the affected taste signal or suggestion.
- Hidden solver directive or objective coefficient: reject output as policy violation.
- User correction target ambiguous: produce clarification request, not a constraint.
- Map card stale relative to parent content hash: block confirmation.
- Counterfactual draft cannot map to known target: require clarification.

## Testing Checklist

- [ ] `TasteEvidencePack` is generated from immutable parent artifacts and contains valid snapshot IDs.
- [ ] Taste prompts use only allowed `TasteDimension` and `TasteLevel` values.
- [ ] Taste signal without evidence refs is rejected.
- [ ] Suggested commitment for unknown target is rejected.
- [ ] `LOCKED` suggestion from inferred taste is downgraded or rejected.
- [ ] `BOOKED` suggestion without booking evidence is rejected.
- [ ] Sensitive personal inference is rejected.
- [ ] Hidden objective coefficient or solver directive is rejected.
- [ ] Unconfirmed LLM-interpreted constraint is inactive.
- [ ] User-confirmed decision compiles to `OwnedConstraint` with correct origin, strength, scope, evidence refs, and confirmation flag.
- [ ] `TEST_ONLY` suggestion cannot enter accepted repair constraints directly.
- [ ] Map cards show suggested/validated/confirmed/active statuses distinctly.
- [ ] Clarification planner asks no more than two repair-relevant questions.
- [ ] Counterfactual question parser creates a sandbox request and does not answer from intuition.
- [ ] Prompt run metadata stores prompt version, model name, input hash, output hash, and validation status.
- [ ] Raw prompt text and raw personal itinerary text are not serialized by default.
- [ ] `RefreshPolicy.NEVER` blocks live LLM requests.
- [ ] Frozen prompt outputs can be replayed for benchmark tests.
- [ ] `python -m ruff check src tests scripts`
- [ ] `python -m pytest`

## Roadmap / Full Pipeline Update

After implementation and validation pass, add a `LLM-TASTE-001` or equivalent tracker item to the technical specification. This work package depends on G1 parent/child artifacts, G3/G4 ownership-aware repair inputs, and G6 artifact-backed pipeline/prototype surfaces. It connects to G7 by giving explanations evidence refs for taste hypotheses, commitment suggestions, user confirmations, rejected suggestions, and counterfactual drafts.

For this planning task, no implementation checklist item should be marked complete.

## Acceptance Criteria

1. The LLM taste layer consumes only bounded parent-plan evidence.
2. Taste hypotheses are stored as inspectable, editable artifacts.
3. Commitment suggestions are inactive until deterministic validation and user confirmation.
4. Inferred taste cannot create `LOCKED` or `BOOKED` constraints without required evidence.
5. Confirmed decisions compile into typed `OwnedConstraint` records.
6. The repair solver receives only validated structured constraints, not prompt text.
7. Prompt runs are versioned, hash-linked, and replayable.
8. Map UI distinguishes hypothesis, suggestion, validation, confirmation, and solver-active states.
9. Counterfactual language becomes structured sandbox repair requests.
10. The module supports IUI/CHI evaluation without letting the LLM silently control optimization or certification.

# Phase G7.0 Implementation Plan: Explanation And Study Readiness

## Goal

Implement **explanation and study readiness**.

This gate should allow the system to:

1. Produce structured why, why-not, what-if, and contrastive explanation evidence.
2. Require every numerical or causal explanation claim to cite a valid artifact.
3. Prepare study logging and materials around replayable artifacts without making unsupported human-outcome claims.

Keep deterministic template explanations as the benchmark default.

## Current State

Repository evidence found:

- `src/itinerary_system/repair_planner.py`: has `CounterfactualExplanation` for simple repair scaffold.
- `docs/literature/repair_gap_review.md`: states explanations should be grounded in typed diff, ownership objectives, failed neighborhoods, weather/route/hotel evidence, and certificates.
- `docs/literature/evidence_matrix.md`: marks why-selected as partial, why-skipped as partial, what-would-change as planned, and user outcome claims as needing more evidence.
- `docs/literature/core_paper_reading_cards.md`: emphasizes explanation aim, scope, method, format, and the need for user control.
- `docs/planning/travel_itinerary_repair_technical_specification.md`: specifies `WhyEvidence`, `WhyNotEvidence`, `WhatIfEvidence`, `ContrastiveEvidence`, and `ExplanationClaim`.

Missing pieces:

- No `src/itinerary_system/explanation/` package.
- No general `ExplanationClaim` with claim-to-evidence mapping.
- No counterfactual runner tied to repair pipeline.
- No deterministic template verbalizer.
- No LLM verbalizer boundary.
- No explanation bridge from G6.5 taste hypotheses, suggested commitments, validation reports, and user confirmation decisions.
- No counterfactual-question interface that guarantees "why not" and "what if" answers come from sandbox artifacts.
- No study event logging package.
- No user-study protocol document.

What should not be claimed or implemented yet:

- Do not claim explanations improve trust, understanding, or decision quality without a study.
- Do not let LLM verbalization invent claims.
- Do not expose unsupported explanation prose in publication outputs.

## New Components to Add

### Component 1

`ExplanationClaim`

Responsibility:

Represent a numerical or causal explanation claim with required evidence refs and support status.

### Component 2

`WhyEvidence`

Responsibility:

Explain why a stop, lodging choice, route, or child plan was selected/kept.

### Component 3

`WhyNotEvidence`

Responsibility:

Explain why a requested or candidate option was omitted, failed, or deferred.

### Component 4

`WhatIfEvidence`

Responsibility:

Represent sandbox counterfactual attempts and required changes.

### Component 5

`ContrastiveEvidence`

Responsibility:

Explain parent versus child or method A versus method B differences from diff/evaluator/route/solver evidence.

### Component 6

`DeterministicTemplateVerbalizer`

Responsibility:

Turn structured evidence into stable text for benchmark and study prototypes.

### Component 7

`StudyEventLogger`

Responsibility:

Log redacted participant/prototype interactions tied to plan, repair, diff, and explanation artifact IDs.

### Component 8

`PreferenceCommitmentExplanationBridge`

Responsibility:

Convert G6.5 `TasteProfileArtifact`, `SuggestedCommitmentArtifact`, `CommitmentValidationReport`, `UserCommitmentDecision`, and compiled `OwnedConstraint` records into explanation evidence refs. It explains what the LLM suggested, what the validator changed, what the user confirmed, and how confirmed commitments affected repair.

### Component 9

`CounterfactualQuestionInterface`

Responsibility:

Take G6.5 `CounterfactualRequestDraft` objects, run or retrieve sandbox counterfactual artifacts, and produce why-not/what-if evidence. It must not answer counterfactual questions from prompt text alone.

## Class / Registry Diagrams

```text
+-------------------------------------------------------------------------------+
|                               ExplanationClaim                                  |
+-------------------------------------------------------------------------------+
|  - claim_id: str                                                               |
|  - claim_type: str                                                             |
|  - text_template: str                                                          |
|  - evidence_refs: tuple[str, ...]                                              |
|  - unsupported: bool                                                           |
+-------------------------------------------------------------------------------+
|  + validate(ref_index): tuple[Finding, ...] --> Evidence coverage check        |
|  + render(values): str                      --> Deterministic text             |
+-------------------------------------------------------------------------------+
```

```text
+-------------------------------------------------------------------------------+
|                                WhyEvidence                                      |
+-------------------------------------------------------------------------------+
|  - evidence_id: str                                                            |
|  - plan_id: str                                                                |
|  - target_id: str                                                              |
|  - claims: tuple[ExplanationClaim, ...]                                        |
+-------------------------------------------------------------------------------+
|  + from_plan(...): WhyEvidence             --> Selected/kept rationale         |
|  + evidence_refs(): tuple[str, ...]        --> Artifact refs                   |
+-------------------------------------------------------------------------------+
```

```text
+-------------------------------------------------------------------------------+
|                               WhyNotEvidence                                    |
+-------------------------------------------------------------------------------+
|  - evidence_id: str                                                            |
|  - target_id: str                                                              |
|  - failed_attempt_ids: tuple[str, ...]                                         |
|  - claims: tuple[ExplanationClaim, ...]                                        |
+-------------------------------------------------------------------------------+
|  + from_failed_attempts(...): WhyNotEvidence --> Omission/failure rationale    |
|  + minimal_change(): str                  --> Required change summary          |
+-------------------------------------------------------------------------------+
```

```text
+-------------------------------------------------------------------------------+
|                               WhatIfEvidence                                    |
+-------------------------------------------------------------------------------+
|  - evidence_id: str                                                            |
|  - sandbox_request_id: str                                                     |
|  - counterfactual_run_ids: tuple[str, ...]                                     |
|  - outcome_status: str                                                         |
+-------------------------------------------------------------------------------+
|  + from_counterfactual(...): WhatIfEvidence --> What changed under sandbox     |
|  + claim_map(): dict[str, tuple[str, ...]] --> Claim-to-evidence mapping       |
+-------------------------------------------------------------------------------+
```

```text
+-------------------------------------------------------------------------------+
|                             ContrastiveEvidence                                 |
+-------------------------------------------------------------------------------+
|  - evidence_id: str                                                            |
|  - left_plan_id: str                                                           |
|  - right_plan_id: str                                                          |
|  - diff_id: str                                                                |
|  - claims: tuple[ExplanationClaim, ...]                                        |
+-------------------------------------------------------------------------------+
|  + from_diff(...): ContrastiveEvidence      --> Parent-child explanation       |
|  + metric_deltas(): dict[str, float]        --> Evidence-grounded deltas       |
+-------------------------------------------------------------------------------+
```

```text
+-------------------------------------------------------------------------------+
|                    DeterministicTemplateVerbalizer                              |
+-------------------------------------------------------------------------------+
|  - No persistent internal state                                                |
+-------------------------------------------------------------------------------+
|  + verbalize(evidence): tuple[str, ...]     --> Stable explanation text        |
|  + validate_claim_refs(text): None          --> Fail unsupported output         |
+-------------------------------------------------------------------------------+
```

```text
+-------------------------------------------------------------------------------+
|                    StudyEventLogger (Persistent Registry / Manager)             |
+-------------------------------------------------------------------------------+
|  - log_path: Path                                                              |
|  - redaction_policy: RedactionPolicy                                           |
+-------------------------------------------------------------------------------+
|  + append(event): None                      --> Redacted event write           |
|  + validate_event(event): tuple[Finding, ...] --> Required study fields        |
|  + export_summary(): Path                  --> Study readiness summary         |
+-------------------------------------------------------------------------------+
```

```text
+-------------------------------------------------------------------------------+
|              PreferenceCommitmentExplanationBridge (Service Class)             |
+-------------------------------------------------------------------------------+
|  - artifact_index: ArtifactIndex                                               |
+-------------------------------------------------------------------------------+
|  + build_suggestion_evidence(...): WhyEvidence --> Why suggestion appeared     |
|  + build_confirmation_evidence(...): ContrastiveEvidence --> User decision     |
|  + build_repair_effect_evidence(...): ContrastiveEvidence --> Constraint effect|
+-------------------------------------------------------------------------------+
```

```text
+-------------------------------------------------------------------------------+
|                    CounterfactualQuestionInterface (Service Class)             |
+-------------------------------------------------------------------------------+
|  - repair_runner: ProgressiveRepairController                                  |
|  - evaluator: IndependentPlanEvaluator                                         |
+-------------------------------------------------------------------------------+
|  + run_from_draft(draft): WhatIfEvidence    --> Sandbox counterfactual result  |
|  + explain_failed_draft(draft): WhyNotEvidence --> Invalid target/constraint   |
+-------------------------------------------------------------------------------+
```

## Class Diagram Rules

1. Diagram every evidence class, verbalizer, and study logger.
2. Explanation evidence must reference artifacts; prose is not evidence.
3. Deterministic templates are required before optional LLM verbalization.
4. LLM verbalizer may receive only structured evidence and must return claim-to-evidence mappings.
5. Study logging must avoid secrets, raw personal itineraries, and raw LLM conversation by default.
6. Unsupported claims fail closed.
7. G6.5 prompt artifacts can explain suggestions and confirmations, but cannot prove true user taste.

## Data Model

```python
@dataclass(frozen=True)
class ExplanationClaim:
    claim_id: str
    claim_type: str
    text_template: str
    evidence_refs: tuple[str, ...]
    confidence: str
    unsupported: bool = False
    schema_version: str = "explanation-claim-v1"

@dataclass(frozen=True)
class WhyEvidence:
    evidence_id: str
    plan_id: str
    target_id: str
    claims: tuple[ExplanationClaim, ...]

@dataclass(frozen=True)
class WhyNotEvidence:
    evidence_id: str
    repair_request_id: str
    target_id: str
    failed_attempt_ids: tuple[str, ...]
    claims: tuple[ExplanationClaim, ...]

@dataclass(frozen=True)
class WhatIfEvidence:
    evidence_id: str
    parent_plan_id: str
    sandbox_request_id: str
    counterfactual_run_ids: tuple[str, ...]
    outcome_status: str
    claims: tuple[ExplanationClaim, ...]

@dataclass(frozen=True)
class ContrastiveEvidence:
    evidence_id: str
    left_plan_id: str
    right_plan_id: str
    diff_id: str
    claims: tuple[ExplanationClaim, ...]
```

Every numerical or causal explanation claim must reference at least one valid artifact:

- owned constraint;
- plan diff component;
- route record;
- context observation;
- solver stage;
- evaluator metric;
- counterfactual run.
- prompt run;
- taste profile item;
- suggested commitment;
- commitment validation finding;
- user commitment decision.

Study event required fields:

```text
participant_id_hash
condition
task_id
event_type
target_id
strength_selected
plan_id
repair_request_id
timestamp
accepted
```

## Storage / State

Persistent state:

- `runs/<run_id>/explanations/why_*.json`
- `runs/<run_id>/explanations/why_not_*.json`
- `runs/<run_id>/explanations/what_if_*.json`
- `runs/<run_id>/explanations/contrastive_*.json`
- `runs/<run_id>/study/events.jsonl`
- `docs/user_study_protocol.md`

Temporary state:

- Counterfactual solver models exist only during sandbox runs.
- LLM verbalization output is not canonical unless validated against claim refs.

## Required Methods

```python
def build_why_evidence(plan: PlanArtifactV2, evaluation: ResearchEvaluationReport) -> tuple[WhyEvidence, ...]
def build_why_not_evidence(request: RepairRequest, attempts: tuple[RepairAttempt, ...]) -> tuple[WhyNotEvidence, ...]
def build_contrastive_evidence(diff: PlanDiff, evaluation: ResearchEvaluationReport) -> ContrastiveEvidence
def run_what_if_counterfactual(parent: PlanArtifactV2, edit: EditRequest) -> WhatIfEvidence
def validate_explanation_claims(claims: tuple[ExplanationClaim, ...], artifact_index: ArtifactIndex) -> tuple[Finding, ...]
def verbalize_deterministically(evidence: ExplanationEvidence) -> tuple[str, ...]
def build_preference_commitment_evidence(
    *,
    taste_profile: TasteProfileArtifact,
    suggestions: SuggestedCommitmentArtifact,
    validation_report: CommitmentValidationReport,
    decisions: tuple[UserCommitmentDecision, ...],
    compiled_constraints: tuple[OwnedConstraint, ...],
) -> tuple[WhyEvidence | ContrastiveEvidence, ...]
def answer_counterfactual_from_artifacts(
    *,
    draft: CounterfactualRequestDraft,
    run_artifacts: tuple[PlannerRun, ResearchEvaluationReport, PlanDiff],
) -> WhatIfEvidence | WhyNotEvidence
```

LLM boundary:

```python
def verbalize_with_llm(
    evidence: ExplanationEvidence,
    llm_client: LlmClient,
) -> tuple[ExplanationClaim, ...]:
    """LLM receives structured evidence only and must return claim-to-evidence mappings."""
```

## Validation Rules

1. Every claim must have at least one valid evidence ref.
2. Numerical claims must reference evaluator metric, route record, diff component, or solver stage.
3. Causal claims must reference disruption, constraint, diff, failed attempt, route record, or counterfactual run.
4. Unsupported claims are excluded from user-facing publication outputs.
5. Deterministic template output must be stable.
6. Counterfactual runs must not mutate parent plans.
7. LLM output without claim-to-evidence mapping fails validation.
8. Study events must be redacted and artifact-linked.
9. Claims about taste must be phrased as parent-plan evidence or user-confirmed decisions, not hidden user facts.
10. Claims about why a suggestion affected repair must reference the compiled `OwnedConstraint`, solver stage, diff, and evaluator certificate.
11. A "why not" or "what if" answer must reference a counterfactual run or a validation failure, not just LLM reasoning.

## UI / API Integration

Prototype integration:

- Interactive prototype displays certificate badge, diff view, and explanation evidence panel.
- Explanations should answer "why", "why not", "what if", and "what changed".
- Explanation panels should also answer "why was this suggested", "why did this become active", and "what did my confirmation change" using G6.5 artifacts.
- User-study mode logs interactions but does not claim outcomes until analyzed.

## Workflow

1. Build evidence index from run artifacts.
2. Generate why evidence for selected/kept elements.
3. Generate contrastive evidence from parent-child diff.
4. Generate why-not evidence from failed attempts and omitted requested options.
5. Run what-if counterfactual only as sandbox child requests.
6. Validate every claim.
7. Verbalize deterministically.
8. Store evidence and rendered text.
9. Link taste/commitment prompt artifacts only as evidence for suggestions, validation, and user confirmations.
10. Log study/prototype events if in study mode.

## Files to Create

- `src/itinerary_system/explanation/__init__.py`
- `src/itinerary_system/explanation/evidence.py`
- `src/itinerary_system/explanation/counterfactual.py`
- `src/itinerary_system/explanation/preference_commitments.py`
- `src/itinerary_system/explanation/verbalizer.py`
- `src/itinerary_system/explanation/validation.py`
- `src/itinerary_system/study/__init__.py`
- `src/itinerary_system/study/events.py`
- `src/itinerary_system/study/logger.py`
- `docs/user_study_protocol.md`
- `tests/explanation/test_evidence.py`
- `tests/explanation/test_counterfactual.py`
- `tests/explanation/test_preference_commitments.py`
- `tests/explanation/test_verbalizer.py`
- `tests/study/test_event_redaction.py`

## Files to Modify

- `src/itinerary_system/repair/progressive.py`
- `src/itinerary_system/pipeline_runner.py`
- `src/itinerary_system/prototype/controller.py`
- `src/itinerary_system/llm_taste/models.py`
- `src/itinerary_system/llm_taste/commitments.py`
- `src/itinerary_system/map_exporter.py`
- `src/itinerary_system/map_renderer.py`
- `tests/test_pipeline_runner.py`

## Error Handling

- Missing artifact ref: explanation validation failure.
- Unsupported numerical/causal claim: mark unsupported and exclude from final UI text.
- Counterfactual solver failure: store failure evidence, do not fabricate "why not".
- LLM returns unreferenced claim: reject verbalization.
- LLM explains true taste instead of parent-plan evidence or confirmed decision: reject verbalization.
- Counterfactual question has no valid sandbox artifact: return "counterfactual not evaluated" evidence, not a causal answer.
- Study event contains raw secret or unredacted text: reject event write.

## Testing Checklist

- [ ] Explanation claim without evidence refs fails closed.
- [ ] Numerical claim must cite metric/route/diff/stage evidence.
- [ ] Causal claim must cite constraint/disruption/diff/counterfactual evidence.
- [ ] Deterministic verbalizer output is stable.
- [ ] Why evidence references selected plan and evaluation.
- [ ] Why-not evidence references failed attempts.
- [ ] What-if evidence creates sandbox child request and does not mutate parent.
- [ ] LLM verbalization boundary rejects unmapped claims.
- [ ] Taste/commitment explanation says "suggested by evidence" or "confirmed by user", not "true taste".
- [ ] Suggestion-effect explanation cites suggestion, validation, user decision, compiled constraint, solver stage, diff, and certificate.
- [ ] Counterfactual answer without sandbox artifacts fails closed.
- [ ] Study events are redacted and artifact-linked.
- [ ] Explanation evidence coverage metric is computed.
- [ ] `python -m ruff check src tests scripts`
- [ ] `python -m pytest`

## Roadmap / Full Pipeline Update

After implementation and validation pass, update `EXPLAIN-001`, `EXPLAIN-002`, and `STUDY-001` only for implemented evidence/logging pieces. This gate connects to G8 by creating the artifact-backed explanation and logging foundation needed for IUI/CHI work. Do not claim study outcomes.

## Acceptance Criteria

1. Explanation evidence covers why, why-not, what-if, and contrastive cases.
2. Every explanation claim maps to valid artifacts.
3. Deterministic explanations work without an LLM.
4. Optional LLM verbalization cannot invent unsupported claims.
5. Study logging is redacted, artifact-linked, and replayable.

# Phase G8.0 Implementation Plan: IUI/CHI Extension Readiness

## Goal

Implement **IUI/CHI extension readiness**.

This gate should allow the system to:

1. Choose an interaction-study path only after computational artifacts are stable.
2. Prepare a controlled prototype, tasks, event logs, and analysis plan for inspectable itinerary repair.
3. Keep claims aligned with the evidence and participant population.

This is readiness planning, not a claim that a user study has been run.

## Current State

Repository evidence found:

- `docs/literature/literature_onboarding_guide.md`: suggests a study-ready dashboard mode with original versus adjusted route, explanations, preservation of intent, and user tasks.
- `docs/literature/core_paper_reading_cards.md`: frames explanation/user-control literature and warns that outcome claims require a study.
- `docs/literature/evidence_matrix.md`: marks trust calibration, user understanding, decision quality, and durable user modification as planned/needs-evidence.
- Existing dashboard exports and `scripts/serve_dashboard.py` provide a static artifact display base.

Missing pieces:

- No ethics-ready protocol beyond planned docs.
- No controlled task corpus.
- No condition assignment.
- No replayable event logs tied to repair artifacts.
- No analysis scripts.
- No pilot validation.
- No evidence for user outcomes.
- No study metrics for LLM taste agreement, suggested-commitment acceptance, false suggestion rate, or repair cost caused by confirmed commitments.
- No controlled comparison between manual constraints, LLM taste profile only, LLM taste plus commitments, and direct LLM rewrite baseline.

What should not be claimed or implemented yet:

- Do not claim improved trust, control, understanding, or decision quality without study data.
- Do not add broad autonomous agent/booking features.
- Do not log raw personal itineraries or raw LLM conversations by default.
- Do not use unstable pipeline outputs in a user study.

## New Components to Add

### Component 1

`StudyTaskCorpus`

Responsibility:

Store controlled tasks around weather deterioration, road closure, hotel issue, attraction closure, fatigue, and new must-visit disruptions.

### Component 2

`StudyConditionConfig`

Responsibility:

Define baseline and treatment UI conditions with identical backend artifacts.

### Component 3

`StudyReplayBundle`

Responsibility:

Package parent, child, diff, route, evaluation, certificate, explanation, and event schema artifacts for each task.

### Component 4

`StudyAnalysisPlan`

Responsibility:

Define measures, exclusion rules, hypotheses, qualitative coding plan, and limits.

### Component 5

`PrototypeReadinessValidator`

Responsibility:

Check that the UI/prototype uses stable artifacts, has no unsupported claims, and logs required events.

### Component 6

`PreferenceCommitmentStudyMetrics`

Responsibility:

Compute LLM taste and commitment metrics from G6.5 artifacts, G7 explanations, and study events without treating unconfirmed hypotheses as ground truth.

## Class / Registry Diagrams

```text
+-------------------------------------------------------------------------------+
|                              StudyTaskCorpus                                    |
+-------------------------------------------------------------------------------+
|  - tasks_path: Path                                                            |
|  - task_ids: tuple[str, ...]                                                   |
+-------------------------------------------------------------------------------+
|  + load(): tuple[StudyTask, ...]            --> Controlled tasks               |
|  + validate_against_runs(): tuple[Finding, ...] --> Artifact consistency       |
|  + export_task_sheet(): Path                --> Study material table           |
+-------------------------------------------------------------------------------+
```

```text
+-------------------------------------------------------------------------------+
|                             StudyConditionConfig                                |
+-------------------------------------------------------------------------------+
|  - condition_id: str                                                           |
|  - visible_components: tuple[str, ...]                                         |
|  - artifact_policy: str                                                        |
+-------------------------------------------------------------------------------+
|  + baseline(): StudyConditionConfig        --> Basic map/plan view             |
|  + treatment(): StudyConditionConfig       --> Diff/evidence/certificate view  |
|  + validate_parity(other): tuple[Finding, ...] --> Same backend artifacts      |
+-------------------------------------------------------------------------------+
```

```text
+-------------------------------------------------------------------------------+
|                             StudyReplayBundle                                   |
+-------------------------------------------------------------------------------+
|  - bundle_id: str                                                              |
|  - task_id: str                                                                |
|  - parent_plan_id: str                                                         |
|  - child_plan_id: str                                                          |
|  - evidence_refs: tuple[str, ...]                                              |
+-------------------------------------------------------------------------------+
|  + materialize(output_dir): Path            --> Study-ready artifact package   |
|  + validate(): tuple[Finding, ...]          --> Complete replay check          |
+-------------------------------------------------------------------------------+
```

```text
+-------------------------------------------------------------------------------+
|                             StudyAnalysisPlan                                   |
+-------------------------------------------------------------------------------+
|  - protocol_path: Path                                                         |
|  - measures: tuple[str, ...]                                                   |
|  - exclusion_rules: tuple[str, ...]                                            |
+-------------------------------------------------------------------------------+
|  + validate_measures(): tuple[Finding, ...] --> No unsupported outcome claims  |
|  + export_preregistration_stub(): Path      --> Analysis handoff               |
+-------------------------------------------------------------------------------+
```

```text
+-------------------------------------------------------------------------------+
|                        PrototypeReadinessValidator                              |
+-------------------------------------------------------------------------------+
|  - No persistent internal state                                                |
+-------------------------------------------------------------------------------+
|  + validate_artifacts(bundle): tuple[Finding, ...] --> Stable backend checks   |
|  + validate_ui_claims(view): tuple[Finding, ...] --> Claim guardrails          |
|  + validate_logging(events): tuple[Finding, ...] --> Required event schema     |
+-------------------------------------------------------------------------------+
```

```text
+-------------------------------------------------------------------------------+
|              PreferenceCommitmentStudyMetrics (Stateless Utility Module)       |
+-------------------------------------------------------------------------------+
|  - No persistent internal state                                                |
+-------------------------------------------------------------------------------+
|  + taste_agreement(events): StudyMetric    --> User confirms/edits hypotheses |
|  + suggestion_acceptance(events): StudyMetric --> Confirm/reject/downgrade     |
|  + repair_cost(run_pair): StudyMetric      --> Cost of confirmed commitments   |
|  + unsupported_claim_rate(reports): StudyMetric --> Validator/critic failures  |
+-------------------------------------------------------------------------------+
```

## Class Diagram Rules

1. Diagram study task, condition, replay, analysis, and readiness validation components.
2. Study components must consume G6/G7 artifacts, not run solvers in the UI.
3. Baseline and treatment conditions must use identical backend plans.
4. Logging must be redacted and schema-validated.
5. Outcome claims must be listed as planned measures until analyzed.
6. Skip this phase if G0-G7 are not stable enough for replay.
7. LLM conditions must differ only in the interaction layer; backend artifacts must remain comparable.

## Data Model

```python
@dataclass(frozen=True)
class StudyTask:
    task_id: str
    disruption_family: DisruptionType
    parent_plan_id: str
    child_plan_id: str
    prompt_text: str
    comprehension_questions: tuple[str, ...]
    expected_evidence_refs: tuple[str, ...]

@dataclass(frozen=True)
class StudyReplayBundle:
    bundle_id: str
    task_id: str
    run_id: str
    parent_plan_id: str
    child_plan_id: str
    diff_id: str
    certificate_id: str
    explanation_ids: tuple[str, ...]
    route_artifact_ids: tuple[str, ...]

@dataclass(frozen=True)
class StudyMeasure:
    measure_id: str
    construct: str
    source: str
    scale: str
    required_artifact_refs: tuple[str, ...]
```

Candidate constructs:

- constraint comprehension;
- error detection;
- correction quality;
- perceived control;
- workload;
- calibrated reliance;
- explanation usefulness;
- revision quality;
- time on task.
- taste profile agreement;
- taste profile correction rate;
- suggested commitment acceptance rate;
- false or over-strong suggestion rate;
- repair cost of confirmed commitments;
- preservation gain from confirmed suggestions;
- explanation evidence coverage for LLM-generated prose.

LLM-specific metrics:

```text
TasteProfileAgreement
TasteProfileCorrectionRate
UnsupportedTasteClaimRate
SensitiveInferenceRate
EvidenceCoverage
SuggestedCommitmentAcceptanceRate
SuggestedCommitmentRejectionRate
DowngradeRate
UpgradeRate
FalseLockSuggestionRate
OverStrongSuggestionRate
BookingMisclassificationRate
ClarificationUsefulness
PreservationGainFromConfirmedSuggestions
RepairCostOfConfirmedCommitments
CertificateChangeRate
ExplanationComprehension
UserControlRating
```

Experimental conditions:

```text
Condition A: No LLM taste suggestions.
Condition B: LLM taste profile only.
Condition C: LLM taste profile plus suggested commitments.
Condition D: Direct LLM planner baseline, only if frozen inputs, frozen outputs, and independent evaluation are available.
```

## Storage / State

Persistent state:

- `docs/user_study_protocol.md`
- `docs/study/analysis_plan.md`
- `data/study/tasks/*.json`
- `runs/<run_id>/study/replay_bundles/*.json`
- `runs/<run_id>/study/events.jsonl`
- `runs/<run_id>/study/readiness_report.md`

Temporary state:

- UI session state is temporary and must be replayable from logged event IDs.

## Required Methods

```python
def build_study_task_corpus(run_ids: tuple[str, ...]) -> tuple[StudyTask, ...]
def validate_study_replay_bundle(bundle: StudyReplayBundle, store: RunArtifactStore) -> tuple[Finding, ...]
def assign_study_condition(participant_id_hash: str, seed: int) -> StudyConditionConfig
def validate_prototype_readiness(run_id: str) -> StudyReadinessReport
def export_analysis_plan(tasks: tuple[StudyTask, ...], measures: tuple[StudyMeasure, ...]) -> Path
def compute_preference_commitment_metrics(
    *,
    events: tuple[StudyEvent, ...],
    taste_artifacts: tuple[TasteProfileArtifact, ...],
    commitment_artifacts: tuple[SuggestedCommitmentArtifact, ...],
    repair_runs: tuple[PipelineRun, ...],
) -> tuple[StudyMetric, ...]
```

## Validation Rules

1. No task may use an ineligible child plan.
2. Every task must have parent, child, diff, route, certificate, and explanation evidence.
3. Baseline and treatment must use identical backend artifacts.
4. UI text must not claim road-valid evidence when route source is fallback.
5. Event logs must use hashed participant IDs.
6. No secrets, raw personal itineraries, or raw LLM conversations are logged by default.
7. Outcome claims are not written as findings until study data are analyzed.
8. Ethics/IRB requirements must be handled by the human team before participant work.
9. Direct LLM rewrite baseline is disabled unless frozen outputs and independent evaluator results exist.
10. Study metrics must not treat LLM confidence as calibrated probability.
11. Taste agreement can be measured only from explicit user accept/reject/edit decisions.

## UI / API Integration

Prototype integration:

- Baseline condition: simple itinerary/map/metrics display.
- Treatment condition: parent-child diff, certificate, route/context evidence, why/why-not/what-if explanations, and correction controls.
- LLM taste condition: treatment view plus taste profile cards, inactive suggested commitments, clarification questions, and confirmation controls.
- Both conditions must be loaded from the same run artifacts.
- The UI can support task flow and logging, but cannot mutate canonical plan artifacts directly.

## Workflow

1. Select stable G6/G7 runs.
2. Build study replay bundles.
3. Validate every task artifact.
4. Define baseline/treatment conditions.
5. Implement or configure prototype views.
6. Validate UI claims and event logging.
7. Configure LLM interaction conditions and ensure backend artifact parity.
8. Export protocol and analysis plan.
9. Pilot internally.
10. Human team handles ethics and participant deployment.

## Files to Create

- `docs/study/analysis_plan.md`
- `data/study/tasks/`
- `src/itinerary_system/study/tasks.py`
- `src/itinerary_system/study/conditions.py`
- `src/itinerary_system/study/replay.py`
- `src/itinerary_system/study/readiness.py`
- `src/itinerary_system/study/preference_metrics.py`
- `tests/study/test_tasks.py`
- `tests/study/test_conditions.py`
- `tests/study/test_replay.py`
- `tests/study/test_readiness.py`
- `tests/study/test_preference_metrics.py`

## Files to Modify

- `docs/user_study_protocol.md`
- `src/itinerary_system/prototype/controller.py`
- `src/itinerary_system/study/logger.py`
- `README.md` only after prototype is stable.

## Error Handling

- Missing eligible artifact: block readiness.
- Baseline/treatment artifact mismatch: block readiness.
- Unsupported UI claim: readiness failure.
- Unredacted event: reject log entry.
- LLM condition uses different backend artifacts than baseline: readiness failure.
- Direct LLM baseline lacks frozen output or evaluator result: mark unavailable, not study-ready.
- Metric calculation uses unconfirmed hypothesis as user preference: metric validation failure.
- Missing ethics approval: record human action blocker, not a code failure.

## Testing Checklist

- [ ] Study task corpus loads and validates.
- [ ] Every replay bundle has parent/child/diff/route/evaluation/certificate/explanation refs.
- [ ] Baseline and treatment conditions use identical backend artifacts.
- [ ] UI claim validator blocks unsupported road-valid, booking, or safety claims.
- [ ] Event logs hash participant IDs and avoid raw sensitive content.
- [ ] Analysis plan lists measures without claiming results.
- [ ] LLM taste/profile conditions use identical backend artifacts where comparison requires parity.
- [ ] Taste agreement is computed from explicit user decisions only.
- [ ] False lock, over-strong, booking-misclassification, and unsupported-claim rates are computed from validation reports.
- [ ] Repair cost of confirmed commitments compares against matched no-LLM/manual-constraint runs.
- [ ] Direct LLM rewrite baseline is unavailable without frozen output and evaluation.
- [ ] Prototype readiness report blocks unstable artifacts.
- [ ] `python -m ruff check src tests scripts`
- [ ] `python -m pytest`

## Roadmap / Full Pipeline Update

After implementation and pilot validation pass, update `STUDY-001` and any UI readiness tracker. This is the extension gate after computational repair and explanation evidence are stable. It supports IUI/CHI preparation, not completed IUI/CHI results.

## Acceptance Criteria

1. A study-ready artifact corpus exists for stable repair tasks.
2. Prototype views replay from artifacts only.
3. Logging is redacted and schema-valid.
4. Baseline/treatment conditions are comparable.
5. Human-facing claims remain evidence-aligned.
6. Ethics and participant deployment remain explicit human actions.
7. LLM preference-to-commitment measures are defined, artifact-backed, and not reported as outcomes until study data exist.

## Final Definition Of Done

The complete system is done only when all of the following are true:

- A clean clone can load the benchmark catalog and context.
- Phase 0 strict validation passes with local or pinned road-valid route evidence.
- `plan_demo_current` and `plan_california_repair_v1` are immutable and validated.
- Parent plans are never mutated; every change creates a child plan with a new ID and content hash.
- All six disruptions produce typed `RepairRequest`s.
- Owned constraints distinguish locked, booked, strong, weak, and test-only commitments.
- LLM taste hypotheses are generated only from bounded parent-plan evidence packs.
- LLM suggested commitments remain inactive until deterministic validation and user confirmation.
- Confirmed LLM-assisted decisions compile into `OwnedConstraint` records and raw prompt text never reaches the solver.
- The solver minimizes hard/booked relaxations and typed owned changes before utility.
- The controller stops at the smallest independently eligible repair radius.
- Every travel duration used by publication comparisons is road validated.
- Every displayed plan has a source run, parent lineage, content hash, diff, evaluation, and certificate state.
- Post-solve mutation automatically invalidates certification until reevaluated.
- Why, why-not, what-if, and contrastive claims have valid evidence references.
- LLM verbalizations and counterfactual answers map every claim to structured evidence or fail closed.
- Prompt runs are versioned, hash-linked, redacted, and replayable for benchmark or study use.
- Benchmark splits do not leak parent-plan/disruption families.
- Provider adapters remain inactive without valid credentials.
- Secrets never enter artifacts.
- Missing Yelp or provider observations are handled as missing evidence, not low value.
- Notebook and package CLI use the same pipeline runner.
- Dashboard/prototype views load canonical artifacts, not mutable dictionaries.
- Map UI distinguishes taste hypothesis, suggested commitment, validated inactive suggestion, confirmed decision, and solver-active constraint states.
- IUI/CHI metrics for taste agreement, suggestion acceptance, false suggestions, preservation gain, and repair cost are computed from explicit events and artifacts only.
- Full checks pass:

```bash
python -m ruff check src tests scripts
python -m pytest
```

- Human-study or IUI/CHI claims are made only after ethics, stable prototype, event logs, analysis, and appropriate participant evidence exist.
