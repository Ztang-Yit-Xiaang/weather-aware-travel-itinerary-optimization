# Context-Aware, Inspectable Itinerary Repair: Integrated Implementation Plan

Prepared: 2026-07-08
Repository: `Ztang-Yit-Xiaang/weather-aware-travel-itinerary-optimization`
Status: planning only. This document does not implement code and does not mark any implementation checklist complete.

## 0. Executive Summary

The project identity is **Context-Aware, Inspectable Itinerary Repair**.

The project is not named TripTide. TripTide is a related disruption benchmark and threat paper in the literature, not this system.

Core thesis:

> Repair, do not regenerate. Treat the accepted itinerary as a persistent parent artifact. After a disruption, user edit, or context update, produce a child itinerary that preserves locked, booked, strong-preference, weak-preference, and flexible commitments as much as possible, expands the repair neighborhood only when needed, solves preservation before utility, independently evaluates the final plan, and explains the repair using evidence from plan diffs, constraints, route records, context snapshots, solver stages, and counterfactual attempts.

Safe contribution claim:

> This project combines ownership-labeled commitments, progressive repair neighborhoods, lexicographic preservation-before-utility objectives, independent validation, and evidence-grounded explanations for user-owned itinerary repair under localized disruptions.

Current repository truth on 2026-07-08:

1. The repository has moved beyond the earlier Phase 0-only state. `PlanArtifactV2`, ownership models, append-only `PlanRepository`, `PlanDiff`, `RouteMatrix`, source-missingness masks, repair neighborhoods, a repair master scaffold, lexicographic candidate selection, a route-matrix-backed day-route subproblem, and a progressive repair controller now exist.
2. The independent final-plan evaluator and certificate package is still missing. `repair_progressively()` accepts an evaluator hook, and tests use a fixture evaluator, but there is no production `src/itinerary_system/evaluation/` package.
3. The canonical package-level pipeline runner is still missing. Production remains tied to `experiment_runner.py`, Phase 0 scripts, notebooks, and flat outputs.
4. Explanation, benchmark, LLM taste-to-commitment, live/event-triggered repair, and study-readiness packages are still missing.
5. `multi_objective_route.py`, `hierarchical_gurobi.py`, and `route_gurobi_oracle.py` still retain geodesic fallback paths. Publication mode must use `RouteMatrix` or fail closed.
6. `docs/current/current_problem_manifest.md` is the most current problem-status document. It marks CP-000 through CP-005 as implemented or partially implemented and CP-006 through CP-007 as major remaining gaps.

Integrated gate order:

1. G0 - Truthful foundation and current problem closure.
2. G1 - Canonical parent/child plans, ownership, and diff.
3. G2 - Road-valid routing and context snapshot boundary.
4. G3 - Independent evaluator and certificate skeleton.
5. G4 - Single-day ownership-aware repair.
6. G5 - Progressive multi-day repair.
7. G6 - Benchmark package.
8. G7 - Authoritative pipeline runner and thin notebook.
9. G8 - Evidence-bounded LLM preference-to-commitment layer.
10. G9 - Explanation, counterfactual, and study readiness.
11. G10 - Live/event-triggered repair extension.
12. G11 - IUI/CHI study and venue readiness.

Implementation lock:

Do not build claims upward from demo output. Build claims from immutable artifacts, route-validated evidence, typed diffs, independent certificates, and replayable pipeline runs.

## 1. Repository-Grounded Current State

Documents inspected:

- `docs/planning/research_stabilization_and_publication_plan.md`: research stabilization plan, claim boundaries, venue decision logic, evaluator need, and LLM/compiler boundary.
- `docs/planning/current_problem_fix_phase_plans.md`: earlier phase plans for validation harness, artifact lineage, route matrix, utility missingness, parent/diff, progressive repair, evaluator/explanations, and pipeline/benchmark migration.
- `docs/planning/context_aware_itinerary_repair_detailed_phase_plan.md`: detailed G0-G8 plan including G6.5 LLM preference-to-commitment work package.
- `docs/planning/travel_itinerary_repair_technical_specification.md`: technical implementation contract. It is useful, but its "current repository truth" section is older than the current `plans/`, `routing/matrix.py`, and `repair/` implementations.
- `docs/current/current_score_audit.md`: current score caveats, utility limitations, geodesic-route warnings, and overclaim risks.
- `docs/current/current_problem_manifest.md`: current CP-000 through CP-007 status and validation commands.
- `docs/literature/repair_gap_review.md`: safe repair-gap claim and claims to avoid.
- `docs/literature/literature_matrix_repair_gap.md`: closest-work matrix around OP/TTDP, dynamic routing, LLM travel planning, TripTide, TravelEval, TripScore, and explainable optimization.
- `docs/literature/literature_onboarding_guide.md`: onboarding roadmap and narrative positioning.
- `docs/literature/evidence_matrix.md`: implementation-versus-claim evidence status.
- `docs/literature/core_paper_reading_cards.md`: related work cards for TTDP, LLM travel, evaluation, TripTide, and explanations.
- `docs/literature/literature_deep_read_study_report.md`: detailed evidence bank and recent paper notes.
- `docs/reference/data_dictionary.md`: catalog/context separation, routing eligibility, road-route-cache semantics, and Phase 0 evidence roles.
- `docs/reference/code_quality_workflow.md`: Ruff, pytest, coverage, dead-code, and dashboard validation workflow.
- `README.md`: current demo, pipeline commands, project structure, limitations, and docs index.
- `docs/README.md`: documentation index.
- `CODEX_EDIT_LOG.md`: work history including Phase 0, route matrix, utility missingness, parent/diff, repair, and planning updates.

Path variants checked:

- `doc/`: not present.
- `doc/planning/`: not present.
- `doc/literature/`: not present.
- `doc/current/`: not present.

Source modules inspected:

- `src/itinerary_system/repository_state.py`: `RepositoryState`, `RepositoryStateUnavailable`, `capture_repository_state()`.
- `src/itinerary_system/research_artifacts.py`: `PlannerRun`, `PlanArtifact`, `PlanArtifactV2`, `MutationReport`, child-plan helpers, certificate invalidation helpers, `ResearchEvaluationReport`, and `evaluate_phase0_plan()`.
- `src/itinerary_system/phase0_exporter.py`: Phase 0 artifact export, route audit, fallback labeling, and V2 plan emission.
- `src/itinerary_system/repair_planner.py`: deterministic repair scaffold, parsed intent, evidence conflict detection, simple alternatives, and lineaged repair scaffold output.
- `src/itinerary_system/multi_objective_route.py`: route solving with optional injected `RouteMatrix`; geodesic travel remains for demo/non-publication paths.
- `src/itinerary_system/hierarchical_gurobi.py`: hierarchical planning with optional route matrix; geodesic fallback remains for demo/non-publication paths.
- `src/itinerary_system/route_gurobi_oracle.py`: small route oracle wrapper; old geodesic helper code remains.
- `src/itinerary_system/utility_model.py`: `SourceSignalMask`, explicit source masks, masked utility scoring, source ablation, and separated coverage/uncertainty fields.
- `src/itinerary_system/request_schema.py`: `TripPlanningRequest`, interest weights, and config override conversion. No complete repair request schema yet.
- `src/itinerary_system/data/`: catalog/context/dataset bundle and validation scaffolds.
- `src/itinerary_system/routing/`: `RouteLegResult`, `RouteResult`, `RoadRouteCache`, `RouteMatrix`, `RouteMatrixCell`, provider protocols, OSRM route-cache builder, strict route-source checks.
- `src/itinerary_system/plans/`: ownership models, `PlanDiff`, and append-only repository.
- `src/itinerary_system/repair/`: repair variables, neighborhoods, master-model scaffold, day-route subproblem, lexicographic solver, and progressive controller.
- `src/itinerary_system/evaluation/`: missing.
- `src/itinerary_system/explanation/`: missing.
- `src/itinerary_system/benchmark/`: missing.
- `src/itinerary_system/llm_taste/`: missing.
- `src/itinerary_system/live/`: missing.

Tests inspected:

- `tests/test_repository_state.py`: repository identity and dirty state behavior.
- `tests/test_artifact_lineage.py`: V1-to-V2 migration, content hash mutation, child-plan lineage, and certificate invalidation.
- `tests/test_project_checks.py`: managed Windows temp/cache behavior and validation classification.
- `tests/test_utility_missingness.py`: missing Yelp is not low Yelp, source masks, fallback utility, and source ablation determinism.
- `tests/routing/test_route_matrix.py`: route matrix strictness, geodesic fallback gating, publication-mode route solving, and hierarchical planner matrix injection.
- `tests/plans/test_models.py`: owned constraints, inactive unconfirmed LLM constraints, locked/booked relaxation policy, and Phase 0-to-V2 plan construction.
- `tests/plans/test_diff.py`: typed diff changes and weighted edit cost.
- `tests/plans/test_repository.py`: append-only `PlanRepository`.
- `tests/repair/test_neighborhood.py`: progressive radii, freezing outside scope, booked lodging rules, and full reoptimization last.
- `tests/repair/test_master_model.py`: selection/day/lodging/relaxation/change variables, locked/booked rules, objective components, and child lineage.
- `tests/repair/test_lexicographic.py`: sequential lexicographic priority, tolerance, status/bound/gap persistence, infeasible stage records, and Gurobi reference when available.
- `tests/repair/test_day_route_solver.py`: route-matrix-backed day route candidates, opening-window and visit-duration checks, and strict fallback rejection.
- `tests/repair/test_progressive.py`: smallest eligible radius, full reoptimization final fallback, and no-success diagnosis.

Current completed or partially completed capabilities:

- CP-000 validation harness is present through `scripts/run_project_checks.py`, `tests/test_project_checks.py`, and temp/cache handling.
- CP-001 artifact lineage is partially implemented through `PlanArtifactV2`, mutation reports, child plan helpers, and certificate invalidation.
- CP-002 route matrix boundary is partially implemented through `RouteMatrix`, provider protocols, strict publication mode, and tests.
- CP-003 utility missingness is partially implemented through source masks, masked utility, source ablation, and coverage/uncertainty separation.
- CP-004 parent/diff foundation is partially implemented through `OwnedConstraint`, `OwnershipPolicy`, `PlanRepository`, and `PlanDiff`.
- CP-005 progressive repair solver scaffold is partially implemented through neighborhoods, master/change variables, lexicographic candidate selection, day-route subproblem, and progressive controller.

Current missing or incomplete capabilities:

- Complete repair request schema with typed edit/disruption classes.
- Production independent evaluator and `EvaluationCertificate`.
- Publication-ready local or pinned OSRM Table/Route/Nearest route evidence for all benchmark contexts.
- Benchmark package and split-safe scenario generation.
- Authoritative `run_research_pipeline()`.
- Immutable `runs/<run_id>/` layout.
- Thin notebook migration.
- Explanation package and counterfactual runner.
- LLM taste-to-commitment package.
- Event-triggered live repair extension.
- Study task corpus, event logging, and venue-readiness matrix.

## 2. Literature-Grounded Gap And Claim Boundaries

Safe contribution:

This project combines ownership-labeled commitments, progressive repair neighborhoods, lexicographic preservation-before-utility objectives, independent validation, and evidence-grounded explanations for user-owned itinerary repair under localized disruptions.

Do not claim:

- First disruption-aware itinerary system.
- First itinerary modification system.
- First LLM travel planner.
- First weather-aware route planner.
- First explainable optimizer.
- First scenic route planner.
- Real-time hotel booking or availability.
- Road-valid evaluation when geodesic fallback is used.
- Online bandit personalization without real sequential feedback.
- Calibrated uncertainty when a field is only source coverage.
- Accurate real-world congestion when demand is simulated or proxied.
- That an LLM knows the user's true taste.
- That LLM confidence is calibrated probability.
- That a generated fluent itinerary is feasible without independent evaluation.

Related-work interpretation:

- OP/TTDP and orienteering literature cover constrained route utility, but not persistent parent-child repair with ownership-labeled commitments and typed preservation costs.
- Dynamic routing and disruption-aware routing cover event response, but not user-owned itinerary lineage, plan diffs, and preservation-before-utility objectives.
- TripTide and itinerary modification benchmarks make broad "first disruption" or "first modification" claims unsafe. They are threats and benchmark inspiration.
- LLM travel planning, TTG, TRIP-PAL, LLMAP, and Logic-LM show that LLM-to-symbolic and LLM-plus-solver architectures are already known. The LLM contribution here must be narrower: preference-to-commitment elicitation for repair ownership.
- TravelEval and TripScore motivate independent evaluation and multi-component quality checks, but they do not define this repair solver.
- Explainable optimization supports infeasibility, sensitivity, and counterfactual explanation, but this project must ground every explanation claim in plan diffs, constraints, route records, context snapshots, solver stages, evaluator metrics, and counterfactual runs.

Manuscript paths:

- Transportation-first path: road-valid routing evidence, benchmark scenarios, preservation metrics, and evaluator certificates are central. LLM taste is optional or secondary.
- IUI path: bounded LLM interpretation, taste-to-commitment suggestions, map-based confirmation, solver boundary, and interaction metrics are central.
- CHI path: requires ethics-approved participant evidence about agency, understanding, reliance, workload, and revision quality. It cannot be claimed from artifact benchmarks alone.

## 3. Integrated System Architecture

Master architecture:

```text
+---------------------+        +----------------------+
| RepositoryState     |        | DatasetBundle        |
| PlannerRun lineage  |------->| Catalog + Context    |
+----------+----------+        +----------+-----------+
           |                              |
           v                              v
+---------------------+        +----------------------+
| Parent PlanArtifact |<------>| PlanRepository       |
| OwnedConstraints    |        | append-only storage  |
+----------+----------+        +----------+-----------+
           |                              |
           v                              v
+---------------------+        +----------------------+
| RepairRequest       |------->| RouteMatrix          |
| DisruptionSnapshot  |        | road-valid cells     |
+----------+----------+        +----------+-----------+
           |                              |
           v                              v
+---------------------+        +----------------------+
| RepairNeighborhood  |------->| LexicographicSolver  |
| affected scope      |        | Stage 0-3 traces     |
+----------+----------+        +----------+-----------+
           |                              |
           v                              v
+---------------------+        +----------------------+
| Child PlanArtifact  |------->| PlanDiff             |
| immutable hash      |        | typed edit cost      |
+----------+----------+        +----------+-----------+
           |                              |
           v                              v
+---------------------+        +----------------------+
| IndependentEvaluator|------->| EvaluationCertificate|
| recompute final     |        | hard gate + metrics  |
+----------+----------+        +----------+-----------+
           |                              |
           v                              v
+---------------------+        +----------------------+
| ExplanationEvidence |------->| Pipeline/UI/Study    |
| why/why-not/what-if |        | artifacts only       |
+---------------------+        +----------------------+
```

LLM integration:

```text
Parent PlanArtifact
    |
    v
TasteEvidencePack
    |
    v
LLM Taste Signal Extraction
    |
    v
TasteProfileArtifact
    |
    v
LLM SuggestedCommitmentArtifact
    |
    v
Deterministic Validator
    |
    v
Map Confirmation
    |
    v
UserCommitmentDecision
    |
    v
Confirmed OwnedConstraint
    |
    v
RepairRequest
```

Live/event-triggered extension:

```text
ContextSnapshot update
    |
    v
ContextDelta
    |
    v
ImpactReport
    |
    v
TriggerDecision
    |
    v
LiveRepairRequest
    |
    v
RepairProposal
    |
    v
EvaluationCertificate
    |
    v
UserAdoptionDecision
    |
    v
Accepted child plan or rejected proposal
```

Core implementation rule:

The optimizer, evaluator, renderer, explanation layer, benchmark runner, and UI must consume the same plan, route, context, and evaluation artifacts. No layer may recompute a hidden truth table from notebook state or mutable dictionaries.

Core data models:

```python
@dataclass(frozen=True)
class RepositoryState:
    repo_root: str
    git_commit: str
    git_branch: str
    is_dirty: bool
    package_version: str
    captured_at: str

@dataclass(frozen=True)
class PlannerRun:
    run_id: str
    planning_request_id: str
    catalog_snapshot_id: str
    context_snapshot_id: str
    planner_specification_id: str
    method_requested: str
    method_executed: str
    execution_status: str
    solver_certification: str
    parent_run_id: str | None
    fallback_reason: str
    result_plan_id: str | None

@dataclass(frozen=True)
class PlanArtifactV2:
    plan_id: str
    parent_plan_id: str | None
    source_run_id: str
    planning_request_id: str
    catalog_snapshot_id: str
    context_snapshot_id: str
    selected_stops: tuple[dict[str, Any], ...]
    sequence: tuple[str, ...]
    ordered_days: tuple[dict[str, Any], ...]
    route_ids_by_day: dict[int, str]
    owned_constraints: tuple[dict[str, Any], ...]
    change_components: dict[str, float]
    certificate_id: str | None
    artifact_uri: str
    schema_version: str = "plan-artifact-v2"

@dataclass(frozen=True)
class PlanRepositoryRecord:
    plan_id: str
    content_hash: str
    path: Path
    parent_plan_id: str | None
    created_at: str
```

Current ownership vocabulary in `src/itinerary_system/plans/models.py` already extends the requested vocabulary. The integrated plan should preserve current aliases while normalizing public documentation around the core terms.

```python
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

class RelaxationPolicy(StrEnum):
    NEVER = "never"
    EXPLICIT_ONLY = "explicit_only"
    AUTO_WITH_PENALTY = "auto_with_penalty"
    ALWAYS = "always"
```

```python
@dataclass(frozen=True)
class OwnedConstraint:
    constraint_id: str
    origin: ConstraintOrigin
    strength: ConstraintStrength
    scope: ConstraintScope
    target_id: str
    relation: str
    value: Any
    confirmed: bool
    relaxation_policy: RelaxationPolicy
    evidence_refs: tuple[str, ...]

@dataclass(frozen=True)
class OwnershipPolicy:
    locked_change_cost: float
    booked_change_cost: float
    strong_change_cost: float
    weak_change_cost: float
    flexible_change_cost: float

@dataclass(frozen=True)
class ParentPlanIndex:
    plan_id: str
    stop_ids_by_day: dict[int, tuple[str, ...]]
    day_by_stop_id: dict[str, int]
    lodging_by_day: dict[int, str]
    route_id_by_day: dict[int, str]
    active_constraints: tuple[OwnedConstraint, ...]

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
    explicit_relaxations: tuple[str, ...]
    unchanged_days: tuple[int, ...]
    weighted_edit_cost: float

@dataclass(frozen=True)
class StopChange:
    stop_id: str
    change_type: Literal["add", "delete", "replace"]
    day: int | None
    owner_strength: str
    cost: float
    evidence_refs: tuple[str, ...]

@dataclass(frozen=True)
class DayMove:
    stop_id: str
    from_day: int
    to_day: int
    owner_strength: str
    cost: float

@dataclass(frozen=True)
class TimeShift:
    stop_id: str
    from_time: str | None
    to_time: str | None
    delta_minutes: float
    cost: float

@dataclass(frozen=True)
class OrderChange:
    stop_id: str
    day: int
    from_position: int
    to_position: int
    cost: float

@dataclass(frozen=True)
class LodgingChange:
    day: int
    from_lodging_id: str | None
    to_lodging_id: str | None
    owner_strength: str
    cost: float

@dataclass(frozen=True)
class RoadChange:
    day: int
    from_route_id: str | None
    to_route_id: str | None
    corridor_id: str | None
    owner_strength: str
    cost: float
```

Repair models:

```python
@dataclass(frozen=True)
class EditRequest:
    edit_request_id: str
    edit_type: str
    target_id: str | None
    target_day: int | None
    value: Any
    user_text: str
    evidence_refs: tuple[str, ...]

@dataclass(frozen=True)
class DisruptionSnapshot:
    disruption_id: str
    disruption_type: str
    context_snapshot_id: str
    affected_target_ids: tuple[str, ...]
    affected_days: tuple[int, ...]
    observations: tuple[dict[str, Any], ...]

@dataclass(frozen=True)
class RepairRequest:
    repair_request_id: str
    parent_plan_id: str
    edit_requests: tuple[EditRequest, ...]
    disruption_snapshot: DisruptionSnapshot | None
    allowed_radii: tuple[RepairRadius, ...]
    confirmed_constraints: tuple[OwnedConstraint, ...]
    allow_booked_relaxation: bool

@dataclass(frozen=True)
class RepairNeighborhood:
    radius: RepairRadius
    editable_days: tuple[int, ...]
    editable_stop_ids: tuple[str, ...]
    frozen_stop_ids: tuple[str, ...]
    editable_lodging_nights: tuple[int, ...]
    boundary_route_days: tuple[int, ...]
    frozen_constraint_ids: tuple[str, ...]
    reason_codes: tuple[str, ...]

@dataclass(frozen=True)
class RepairOutcome:
    repair_outcome_id: str
    parent_plan_id: str
    repair_request_id: str
    status: str
    accepted_radius: RepairRadius | None
    attempts: tuple[RepairAttempt, ...]
    child_plan: PlanArtifactV2 | None
    diff_record: dict[str, Any] | None
    evaluation_record: dict[str, Any] | None
```

Routing and evaluation:

```python
@dataclass(frozen=True)
class RouteMatrixCell:
    origin_id: str
    destination_id: str
    duration_s: float
    distance_m: float
    provider: str
    road_validated: bool
    fallback_used: bool
    geometry: tuple[tuple[float, float], ...]
    context_snapshot_id: str
    evidence_id: str

@dataclass(frozen=True)
class RouteMatrix:
    matrix_id: str
    context_snapshot_id: str
    entity_ids: tuple[str, ...]
    cells: Mapping[tuple[str, str], RouteMatrixCell]

class RoutingProvider(Protocol):
    def activation(self) -> RoutingActivation: ...
    def table(self, request: RouteMatrixRequest) -> RouteMatrix: ...
    def route(self, request: RouteLegRequest) -> RouteMatrixCell: ...
    def nearest(self, lat: float, lon: float) -> SnapResult: ...

@dataclass(frozen=True)
class EvaluationCertificate:
    certificate_id: str
    plan_id: str
    parent_plan_id: str | None
    source_run_id: str
    eligible: bool
    hard_failures: tuple[str, ...]
    warnings: tuple[str, ...]
    metric_refs: tuple[str, ...]
    content_hash: str
```

LLM, pipeline, benchmark, and live extension:

```python
@dataclass(frozen=True)
class TasteEvidencePack:
    pack_id: str
    parent_plan_id: str
    plan_content_hash: str
    catalog_snapshot_id: str
    context_snapshot_id: str
    candidate_entities: tuple[dict[str, Any], ...]
    route_segments: tuple[dict[str, Any], ...]
    lodging_entities: tuple[dict[str, Any], ...]
    existing_constraints: tuple[OwnedConstraint, ...]
    evidence_refs: tuple[str, ...]

@dataclass(frozen=True)
class TasteProfileArtifact:
    profile_id: str
    parent_plan_id: str
    prompt_run_id: str
    items: tuple[dict[str, Any], ...]
    evidence_refs: tuple[str, ...]

@dataclass(frozen=True)
class SuggestedCommitmentArtifact:
    artifact_id: str
    parent_plan_id: str
    prompt_run_id: str
    suggestions: tuple[dict[str, Any], ...]
    status: str

@dataclass(frozen=True)
class UserCommitmentDecision:
    decision_id: str
    suggestion_id: str
    target_id: str
    action: str
    selected_strength: ConstraintStrength | None
    user_note: str | None
    created_at: str

@dataclass(frozen=True)
class PromptRun:
    prompt_run_id: str
    prompt_name: str
    prompt_version: str
    model_name: str
    input_hash: str
    output_hash: str
    validation_status: str

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
class BenchmarkScenario:
    scenario_id: str
    parent_plan_id: str
    disruption_family: str
    ownership_profile: str
    repair_request_id: str

@dataclass(frozen=True)
class BenchmarkResult:
    result_id: str
    scenario_id: str
    method_id: str
    plan_id: str | None
    certificate_id: str | None
    metrics: dict[str, float | str]

@dataclass(frozen=True)
class ContextDelta:
    delta_id: str
    previous_context_snapshot_id: str
    new_context_snapshot_id: str
    changed_observations: tuple[dict[str, Any], ...]

@dataclass(frozen=True)
class ImpactReport:
    impact_report_id: str
    parent_plan_id: str
    context_delta_id: str
    affected_target_ids: tuple[str, ...]
    affected_days: tuple[int, ...]
    severity: str
    trigger_recommendation: str
    evidence_refs: tuple[str, ...]

@dataclass(frozen=True)
class LiveEvent:
    event_id: str
    event_type: str
    context_delta_id: str
    detected_at: str
    source: str

class TriggerDecision(StrEnum):
    NO_ACTION = "no_action"
    PASSIVE_BADGE = "passive_badge"
    WARNING = "warning"
    REPAIR_PREVIEW = "repair_preview"
    CONFIRMATION_REQUIRED = "confirmation_required"
    DEFER_UNSUPPORTED = "defer_unsupported"

@dataclass(frozen=True)
class LiveRepairRequest:
    request_id: str
    parent_plan_id: str
    event_id: str
    trigger_decision: TriggerDecision
    allowed_radii: tuple[RepairRadius, ...]

@dataclass(frozen=True)
class RepairProposal:
    proposal_id: str
    live_request_id: str
    child_plan_id: str | None
    certificate_id: str | None
    status: str

@dataclass(frozen=True)
class NotificationRecord:
    notification_id: str
    proposal_id: str
    message_template_id: str
    displayed_at: str

@dataclass(frozen=True)
class UserAdoptionDecision:
    decision_id: str
    proposal_id: str
    action: str
    accepted_child_plan_id: str | None
    created_at: str

@dataclass(frozen=True)
class LiveRepairLineage:
    lineage_id: str
    parent_plan_id: str
    live_event_id: str
    impact_report_id: str
    trigger_decision: TriggerDecision
    live_repair_request_id: str | None
    repair_proposal_id: str | None
    user_adoption_decision_id: str | None
    accepted_child_plan_id: str | None
```

Core public method signatures:

```python
def load_parent_plan(plan_id: str) -> PlanArtifactV2: ...
def save_child_plan(parent: PlanArtifactV2, child: PlanArtifactV2) -> PlanArtifactV2: ...
def compute_plan_diff(parent: PlanArtifactV2, child: PlanArtifactV2, policy: OwnershipPolicy) -> PlanDiff: ...
def build_parent_plan_index(plan: PlanArtifactV2) -> ParentPlanIndex: ...

def build_repair_neighborhood(
    parent: PlanArtifactV2,
    request: RepairRequest,
    radius: RepairRadius,
) -> RepairNeighborhood: ...

def solve_lexicographically(
    model: RepairModel,
    tolerances: ObjectiveTolerances,
) -> LexicographicResult: ...

def repair_progressively(
    request: RepairRequest,
    plan_repository: PlanRepository,
    data_bundle: DatasetBundle,
    routing_context: RoutingContext,
    solver: RepairSolver,
    evaluator: PlanEvaluator,
) -> RepairOutcome: ...

def evaluate_plan(
    plan: PlanArtifactV2,
    data_bundle: DatasetBundle,
    routing_context: RoutingContext,
) -> EvaluationCertificate: ...

def build_taste_evidence_pack(
    parent: PlanArtifactV2,
    data_bundle: DatasetBundle,
    route_matrix: RouteMatrix,
) -> TasteEvidencePack: ...

def validate_suggested_commitments(
    suggestions: tuple[SuggestedCommitmentArtifact, ...],
    evidence_pack: TasteEvidencePack,
) -> CommitmentValidationReport: ...

def compile_confirmed_commitments(
    decisions: tuple[UserCommitmentDecision, ...],
) -> tuple[OwnedConstraint, ...]: ...

def run_research_pipeline(
    *,
    config_path: Path,
    catalog_snapshot_id: str,
    context_snapshot_id: str,
    parent_plan_id: str | None = None,
    repair_request_id: str | None = None,
    refresh_policy: RefreshPolicy = RefreshPolicy.NEVER,
    run_id: str | None = None,
) -> PipelineRun: ...

def analyze_context_delta(parent: PlanArtifactV2, delta: ContextDelta) -> ImpactReport: ...
def decide_live_trigger(impact: ImpactReport, policy: TriggerPolicy) -> TriggerDecision: ...
```

## 4. Master Dependency Graph

```text
G0 truthful foundation
    |
    +--> G1 parent/child plans, ownership, diff
    |       |
    |       +--> G4 single-day repair
    |       |       |
    |       |       +--> G5 progressive multi-day repair
    |       |
    |       +--> G8 LLM preference-to-commitment
    |
    +--> G2 road-valid routing/context
    |       |
    |       +--> G4/G5 repair and G6 benchmark
    |
    +--> G3 independent evaluator/certificate
            |
            +--> G4/G5 accepted repairs
            +--> G6 benchmark metrics
            +--> G7 pipeline outputs
            +--> G9 explanations

G6 benchmark depends on G1-G5.
G7 pipeline depends on G1-G6.
G8 LLM layer depends on G1, G3, G5, and G7 prototype artifacts.
G9 explanations depend on G1-G8 artifacts.
G10 live repair depends on G3, G5, G7, and G9.
G11 venue readiness depends on completed evidence from G6-G10.
```

Implementation-order locks:

1. Do not claim benchmark, transportation, or publication results until road-valid route evidence passes strict validation.
2. Do not rank or display a plan as final eligible until the independent evaluator returns an eligible certificate.
3. Do not let a post-solve mutation retain prior solver certification.
4. Do not let geodesic fallback count as road-valid.
5. Do not let notebooks create canonical evidence or mutate plan dictionaries.
6. Do not let an LLM activate constraints, set solver weights, generate final publication itineraries, or certify feasibility.
7. Do not start live/event-triggered claims until frozen replay streams validate the trigger pipeline.

## 5. Phase And Gate Overview

| Gate | Name | Current repository status | Exit requirement |
|---|---|---|---|
| G0 | Truthful foundation and current problem closure | Partially implemented; manifest exists | Current checks pass and truth blockers are explicit |
| G1 | Parent/child plans, ownership, diff | Partially implemented | V2 artifacts, repository, constraints, diff migration verified |
| G2 | Road-valid routing/context boundary | Partially implemented | RouteMatrix is complete and strict for publication contexts |
| G3 | Independent evaluator/certificate | Missing | Production evaluator package returns certificates |
| G4 | Single-day ownership-aware repair | Partially implemented | One-day repair is independently certified |
| G5 | Progressive multi-day repair | Partially implemented | Smallest-radius eligible repair works with production evaluator |
| G6 | Benchmark package | Missing | Six-disruption benchmark emits split-safe metrics |
| G7 | Pipeline runner/thin notebook | Missing | `run_research_pipeline()` emits immutable run directory |
| G8 | LLM preference-to-commitment | Missing | LLM suggestions are inactive, validated, and user-confirmed |
| G9 | Explanation/counterfactual/study readiness | Missing | Claims map to artifacts; event logs are redacted |
| G10 | Live/event-triggered repair | Missing | Frozen replay stream creates eligible proposals only |
| G11 | Venue readiness | Missing | TRB/IUI/CHI path chosen from completed evidence |

## 6. Detailed Gate Plans

# Phase G0.0 Implementation Plan: Truthful Foundation And Current Problem Closure

## Goal

Implement **truthful foundation and current problem closure**.

This gate should allow the system to:

1. Keep repository state, run lineage, route source, fallback lineage, and current blockers explicit.
2. Preserve compatibility with existing Phase 0 script names and artifact names.
3. Prevent stale solver certification, false road-valid claims, and untracked fallback behavior.

Do not build new repair master logic in this gate.

## Current State

Implemented or partially implemented:

- `docs/current/current_problem_manifest.md`: CP-000 through CP-007 status and validation commands.
- `scripts/run_project_checks.py`: project check wrapper and environment classification.
- `tests/test_project_checks.py`: validates temp/cache behavior and failure classification.
- `src/itinerary_system/repository_state.py`: captures repository state.
- `src/itinerary_system/research_artifacts.py`: planner runs, V1/V2 plan artifacts, mutation reports, certificate invalidation, Phase 0 report.
- `src/itinerary_system/phase0_exporter.py`: writes Phase 0 production evidence tables.
- `scripts/run_phase0_evidence_pipeline.py`, `scripts/validate_phase0_artifacts.py`, `scripts/summarize_phase0_readiness.py`, `scripts/check_route_source.py`: Phase 0 commands remain present.

Missing or incomplete:

- Current manifest is not necessarily regenerated automatically after every relevant implementation.
- Phase 0 strict readiness still depends on road-valid artifacts being present.
- Broader production call sites can still create artifacts outside the future `runs/<run_id>/` layout.

What not to claim yet:

- Do not claim final publication readiness.
- Do not claim strict route readiness if road cache/matrix is incomplete.
- Do not mark CP-006 or CP-007 complete.

## New Components to Add

### Component 1

`CurrentProblemManifestUpdater`

Responsibility:

Generate or validate `docs/current/current_problem_manifest.md` from known checks, source module presence, and artifact validation results.

### Component 2

`Phase0CompatibilityValidator`

Responsibility:

Confirm existing Phase 0 artifact names, CSV/JSONL shapes, and strict validation semantics remain compatible while V2 records are introduced.

### Component 3

`PostSolveMutationGate`

Responsibility:

Centralize checks that invalidate stale solver certificates and require child-plan lineage after material plan edits.

## Class / Registry Diagrams

```text
+-------------------------------------------------------------------------------+
|              CurrentProblemManifestUpdater (Stateless Utility Module)          |
+-------------------------------------------------------------------------------+
|  - No persistent internal state                                                |
+-------------------------------------------------------------------------------+
|  + collect_status(repo_root): tuple[ProblemStatus, ...] --> Current CP status  |
|  + write_manifest(path, statuses): Path   --> Markdown manifest                |
|  + validate_manifest(path): tuple[Finding, ...] --> Freshness/schema checks    |
+-------------------------------------------------------------------------------+
```

```text
+-------------------------------------------------------------------------------+
|                Phase0CompatibilityValidator (Stateless Utility Module)         |
+-------------------------------------------------------------------------------+
|  - No persistent internal state                                                |
+-------------------------------------------------------------------------------+
|  + validate_output_dir(path): tuple[Finding, ...] --> Phase 0 artifact checks  |
|  + validate_strict_ready(path): tuple[Finding, ...] --> Publication gate       |
+-------------------------------------------------------------------------------+
```

```text
+-------------------------------------------------------------------------------+
|                  PostSolveMutationGate (Stateless Utility Module)              |
+-------------------------------------------------------------------------------+
|  - No persistent internal state                                                |
+-------------------------------------------------------------------------------+
|  + detect(before, after): MutationReport  --> Material change report           |
|  + invalidate(plan): PlanArtifactV2       --> Clear certificate                |
|  + require_child(parent, changed, run): PlanArtifactV2 --> New child lineage   |
+-------------------------------------------------------------------------------+
```

## Class Diagram Rules

1. Keep all G0 modules stateless unless they write an explicit manifest.
2. Do not duplicate existing `RepositoryState`, `PlannerRun`, or `PlanArtifactV2` contracts.
3. Preserve existing script names as wrappers if implementation moves.
4. Make fallback and mutation status visible, not hidden behind success booleans.

## Data Model

```python
@dataclass(frozen=True)
class ProblemStatus:
    problem_id: str
    severity: str
    evidence_path: str
    owning_gate: str
    current_status: str
    acceptance_check: str

@dataclass(frozen=True)
class Phase0ReadinessReport:
    report_id: str
    output_dir: Path
    artifact_files_present: tuple[str, ...]
    missing_files: tuple[str, ...]
    strict_final_comparison_ready: bool
    failures: tuple[str, ...]
```

## Storage / State

Persistent:

- `docs/current/current_problem_manifest.md`
- Existing `production_phase0_*` artifacts.
- `results/quality/project_check_summary.json`

Temporary:

- Check process output and validation command logs.

## Required Methods

```python
def collect_current_problem_status(repo_root: Path) -> tuple[ProblemStatus, ...]: ...
def validate_phase0_compatibility(output_dir: Path) -> Phase0ReadinessReport: ...
def detect_and_gate_post_solve_mutation(before: PlanArtifactV2, after: PlanArtifactV2, run: PlannerRun) -> PlanArtifactV2: ...
```

## Validation Rules

1. Existing Phase 0 script names continue to run or call compatibility wrappers.
2. Missing road validation is recorded as ineligible, not treated as success.
3. Mutated plan content invalidates solver certification.
4. Requested and executed methods are both stored.
5. Fallback reason is explicit.
6. Repository dirty state is captured, not hidden.

## UI / API Integration

No user-facing UI required. Scripts and future pipeline runner consume the reports.

## Workflow

1. Capture repository state.
2. Run focused validation harness checks.
3. Validate Phase 0 outputs if present.
4. Update or validate current problem manifest.
5. Emit readiness summary.
6. Stop if strict route or mutation gates fail.

## Files to Create

- `src/itinerary_system/current_problems.py`
- `tests/test_current_problem_manifest.py`

## Files to Modify

- `docs/current/current_problem_manifest.md`
- `scripts/run_project_checks.py`
- `scripts/validate_phase0_artifacts.py`
- `scripts/summarize_phase0_readiness.py`

## Error Handling

- Missing Phase 0 artifact: report missing file with expected path.
- Mutated plan with stale certificate: fail strict readiness.
- Validation environment failure: classify as environment when evidence supports it.
- Unknown CP status: mark as `needs_inspection`.

## Testing Checklist

- [ ] Repository state capture includes commit, branch, dirty flag, and package version.
- [ ] Current problem manifest validates.
- [ ] Existing Phase 0 artifact names remain compatible.
- [ ] Post-solve mutation invalidates certification.
- [ ] Fallback lineage is recorded.
- [ ] `python -m ruff check src tests scripts`
- [ ] `python -m pytest`

## Roadmap / Full Pipeline Update

Update manifest text only after checks run. Do not mark CP items complete without passing tests and artifact evidence.

## Acceptance Criteria

1. Current problem state is reproducible and explicit.
2. Phase 0 scripts remain compatible.
3. Strict readiness rejects false publication evidence.
4. Post-solve mutation cannot retain stale certification.

# Phase G1.0 Implementation Plan: Canonical Parent/Child Plans, Ownership, And Diff

## Goal

Implement **canonical parent/child plans, ownership, and typed diff**.

This gate should allow the system to:

1. Treat an accepted itinerary as an immutable parent artifact.
2. Store ownership-labeled commitments.
3. Measure every child change with typed diff components and weighted edit cost.

Do not rely on repair solver claims before this gate exits.

## Current State

Implemented or partially implemented:

- `src/itinerary_system/research_artifacts.py`: `PlanArtifactV2`, V1 migration, mutation helpers.
- `src/itinerary_system/plans/models.py`: `ConstraintOrigin`, `ConstraintStrength`, `ConstraintScope`, `RelaxationPolicy`, `OwnedConstraint`, `OwnershipPolicy`, typed diff dataclasses, constraint validation helpers.
- `src/itinerary_system/plans/repository.py`: append-only `PlanRepository`, conflict checking, hash verification.
- `src/itinerary_system/plans/diff.py`: `PlanDiffBuilder` and `compute_plan_diff()`.
- `tests/plans/` and `tests/test_artifact_lineage.py`: focused coverage exists.

Missing or incomplete:

- Canonical named parent plans are not packaged as benchmark assets.
- `EditRequest` and `RepairRequest` are not yet full typed models.
- V2 plan storage has not been integrated into immutable `runs/<run_id>/`.
- Ownership model contains extra aliases (`HARD`, `SOFT`, `PREFERENCE`, `STOP`, `ROUTE`, `PLAN`, `USER`) that need documented normalization.

What not to claim yet:

- Do not claim a benchmark parent set exists.
- Do not claim final evaluation/certification exists.

## New Components to Add

### Component 1

`CanonicalParentPlanRegistry`

Responsibility:

Register `plan_demo_current`, `plan_california_repair_v1`, and future parent plans with stable IDs, content hashes, and dataset snapshot linkage.

### Component 2

`ParentPlanIndex`

Responsibility:

Expose fast lookup of stops, days, lodging, routes, and active constraints for repair, diff, evaluator, and explanation layers.

### Component 3

`ConstraintVocabularyNormalizer`

Responsibility:

Map current extended enum aliases to the public core vocabulary for reports, metrics, and prompts.

## Class / Registry Diagrams

```text
+-------------------------------------------------------------------------------+
|             CanonicalParentPlanRegistry (Persistent Registry / Manager)        |
+-------------------------------------------------------------------------------+
|  - registry_path: Path                                                         |
|  - plan_repository: PlanRepository                                             |
+-------------------------------------------------------------------------------+
|  + register(plan): PlanRepositoryRecord    --> Append-only parent record       |
|  + load(plan_id): PlanArtifactV2           --> Immutable parent                |
|  + validate_all(): tuple[Finding, ...]     --> Hash and snapshot checks        |
+-------------------------------------------------------------------------------+
```

```text
+-------------------------------------------------------------------------------+
|                              ParentPlanIndex                                   |
+-------------------------------------------------------------------------------+
|  - plan_id: str                                                                |
|  - stop_ids_by_day: dict[int, tuple[str, ...]]                                 |
|  - day_by_stop_id: dict[str, int]                                              |
|  - lodging_by_day: dict[int, str]                                              |
|  - route_id_by_day: dict[int, str]                                             |
|  - active_constraints: tuple[OwnedConstraint, ...]                             |
+-------------------------------------------------------------------------------+
|  + stop_day(stop_id): int | None       --> Day lookup                          |
|  + active_for(target_id): tuple[OwnedConstraint, ...] --> Constraint lookup    |
|  + route_days(): tuple[int, ...]       --> Route boundary lookup               |
+-------------------------------------------------------------------------------+
```

```text
+-------------------------------------------------------------------------------+
|              ConstraintVocabularyNormalizer (Stateless Utility Module)         |
+-------------------------------------------------------------------------------+
|  - No persistent internal state                                                |
+-------------------------------------------------------------------------------+
|  + normalize_strength(value): ConstraintStrength --> Public strength           |
|  + normalize_scope(value): ConstraintScope       --> Public scope              |
|  + report_aliases(): dict[str, str]           --> Audit mapping                |
+-------------------------------------------------------------------------------+
```

## Class Diagram Rules

1. Use existing `PlanArtifactV2`, `PlanRepository`, `OwnedConstraint`, and `PlanDiff` instead of creating duplicates.
2. Keep parent plan storage append-only.
3. Keep constraints inactive when unconfirmed.
4. Store alias normalization as explicit evidence, not silent coercion.

## Data Model

```python
@dataclass(frozen=True)
class CanonicalParentPlanRecord:
    parent_plan_id: str
    content_hash: str
    catalog_snapshot_id: str
    context_snapshot_id: str
    path: Path
    role: str
    status: str
```

Typed change components:

- `StopChange`
- `DayMove`
- `TimeShift`
- `OrderChange`
- `LodgingChange`
- `RoadChange`
- explicit relaxation record

## Storage / State

Persistent:

- Append-only plan JSON records under a canonical plan repository.
- Future `runs/<run_id>/plans/`.
- Benchmark parent registry JSON.

Temporary:

- `ParentPlanIndex` can be built in memory from `PlanArtifactV2`.

## Required Methods

```python
def build_parent_plan_index(plan: PlanArtifactV2) -> ParentPlanIndex: ...
def register_canonical_parent_plan(plan: PlanArtifactV2, registry: CanonicalParentPlanRegistry) -> CanonicalParentPlanRecord: ...
def normalize_constraint_record(record: dict[str, Any]) -> OwnedConstraint: ...
def compute_plan_diff(parent: PlanArtifactV2, child: PlanArtifactV2, policy: OwnershipPolicy) -> PlanDiff: ...
```

## Validation Rules

1. Parent plans are immutable.
2. Duplicate plan ID with different content fails.
3. Child plan references parent ID.
4. Content hash changes when meaningful content changes.
5. `LOCKED` cannot auto-relax.
6. `BOOKED` requires evidence and explicit relaxation permission.
7. Unconfirmed `LLM_INTERPRETATION` constraints are inactive.
8. `TEST_ONLY` cannot enter accepted plan constraints.

## UI / API Integration

Internal API only. Future UI reads plan summaries, ownership badges, and diff components from artifacts.

## Workflow

1. Load or migrate V1 Phase 0 artifact to V2.
2. Validate owned constraints.
3. Save parent append-only.
4. Build parent index.
5. Generate child only through repair or explicit mutation helper.
6. Compute diff.
7. Store diff with child artifact.

## Files to Create

- `src/itinerary_system/plans/registry.py`
- `tests/plans/test_registry.py`

## Files to Modify

- `src/itinerary_system/plans/models.py`
- `src/itinerary_system/plans/__init__.py`
- `src/itinerary_system/research_artifacts.py`
- `docs/current/current_problem_manifest.md`

## Error Handling

- Duplicate ID/content mismatch: `PlanRepositoryConflict`.
- Missing parent: `PlanNotFound`.
- Invalid constraint enum: typed validation error.
- Missing content hash: regenerate and report.

## Testing Checklist

- [ ] Parent save is append-only.
- [ ] Same ID different content fails.
- [ ] V1-to-V2 migration preserves Phase 0 compatibility.
- [ ] Active constraints exclude unconfirmed LLM and test-only records.
- [ ] Diff detects add/delete/replace/move/reorder/time/lodging/road/relaxation.
- [ ] Weighted edit cost follows ownership policy.
- [ ] `python -m ruff check src tests scripts`
- [ ] `python -m pytest`

## Roadmap / Full Pipeline Update

Update CP-004 only after canonical registry and benchmark parent artifacts are validated.

## Acceptance Criteria

1. Parent and child plans are immutable and hash-stable.
2. Ownership constraints are validated and auditable.
3. Typed diffs produce weighted edit cost.
4. Canonical parent plans are registered and loadable.

# Phase G2.0 Implementation Plan: Road-Valid Routing And Context Snapshot Boundary

## Goal

Implement **road-valid routing and context snapshot boundary**.

This gate should allow the system to:

1. Ensure transportation comparisons use explicit route evidence.
2. Keep optimizer, evaluator, renderer, benchmark, and explanations on the same route/context snapshot.
3. Reject geodesic fallback in publication mode.

## Current State

Implemented or partially implemented:

- `src/itinerary_system/routing/matrix.py`: `RouteMatrix`, `RouteMatrixCell`, validation reports, strict publication checks, context loading, geodesic demo matrix.
- `src/itinerary_system/routing/provider.py`: `RoutingProvider`, `RouteMatrixRequest`, `RouteLegRequest`, `SnapResult`, `RoutingActivation`.
- `scripts/build_validated_route_matrix.py`: route-matrix artifact builder.
- `scripts/build_road_route_cache.py`: OSRM route-cache builder.
- `src/itinerary_system/routing/road_cache_builder.py`: public OSRM policy gate and route cache rows.
- `tests/routing/test_route_matrix.py`: geodesic fallback gating and solver matrix injection tests.
- `src/itinerary_system/utility_model.py`: source masks and missing-source-aware utility.

Missing or incomplete:

- No unified OSRM Table/Route/Nearest client package.
- No local OSRM Docker/runbook assets.
- No pinned route evidence package for full benchmark scenarios.
- No evaluator package to enforce route validity on final plans.
- Some solver paths still retain geodesic fallback for demo mode.

## New Components to Add

### Component 1

`OsrmRoutingProvider`

Responsibility:

Implement `RoutingProvider` using local OSRM Table, Route, and Nearest/snap diagnostics.

### Component 2

`PinnedRouteEvidenceStore`

Responsibility:

Load publication route matrices from frozen route evidence when live/local OSRM is unavailable.

### Component 3

`RoutingContext`

Responsibility:

Bundle `RouteMatrix`, context snapshot ID, route source policy, validation report, and provider activation record.

## Class / Registry Diagrams

```text
+-------------------------------------------------------------------------------+
|                         OsrmRoutingProvider (Service Class)                    |
+-------------------------------------------------------------------------------+
|  - base_url: str                                                               |
|  - activation: RoutingActivation                                               |
+-------------------------------------------------------------------------------+
|  + table(request): RouteMatrix             --> OSRM Table matrix               |
|  + route(request): RouteMatrixCell         --> OSRM Route leg                  |
|  + nearest(lat, lon): SnapResult           --> Snap diagnostics                |
+-------------------------------------------------------------------------------+
```

```text
+-------------------------------------------------------------------------------+
|             PinnedRouteEvidenceStore (Persistent Registry / Manager)           |
+-------------------------------------------------------------------------------+
|  - evidence_root: Path                                                         |
+-------------------------------------------------------------------------------+
|  + load(matrix_id): RouteMatrix            --> Frozen route evidence           |
|  + validate(matrix): RouteMatrixValidationReport --> Strict route report       |
|  + list_available(): tuple[str, ...]       --> Matrix IDs                      |
+-------------------------------------------------------------------------------+
```

```text
+-------------------------------------------------------------------------------+
|                              RoutingContext                                    |
+-------------------------------------------------------------------------------+
|  - route_matrix: RouteMatrix                                                   |
|  - validation_report: RouteMatrixValidationReport                              |
|  - source_policy: str                                                          |
|  - provider_activation: RoutingActivation                                      |
+-------------------------------------------------------------------------------+
|  + require_publication_ready(): None       --> Fail if any fallback/unvalidated|
|  + route_result(sequence): RouteResult     --> Shared route evidence           |
+-------------------------------------------------------------------------------+
```

## Class Diagram Rules

1. Optimizer, evaluator, renderer, and explanations consume `RoutingContext`.
2. Geodesic fallback exists only as explicit demo mode.
3. Public OSRM remains opt-in for non-publication runs only.
4. Missing route evidence is an error in publication mode, not an implicit fallback.

## Data Model

Use existing `RouteMatrix`, `RouteMatrixCell`, `RouteMatrixValidationReport`, `RoutingProvider`, `RoutingActivation`, and `SnapResult`.

Additional:

```python
@dataclass(frozen=True)
class RoutingContext:
    routing_context_id: str
    context_snapshot_id: str
    route_matrix: RouteMatrix
    validation_report: RouteMatrixValidationReport
    source_policy: str
    evidence_refs: tuple[str, ...]
```

## Storage / State

Persistent:

- `data/routing/pinned/*.json` or CSV/JSON route matrix artifacts.
- `runs/<run_id>/routing/route_matrix.csv`
- `runs/<run_id>/routing/route_matrix_validation.json`
- OSRM cache artifacts under configured output directories.

Temporary:

- OSRM client connection and snap diagnostics while building matrices.

## Required Methods

```python
def build_routing_context(config: TripConfig, bundle: DatasetBundle, refresh_policy: RefreshPolicy) -> RoutingContext: ...
def validate_route_source_strict(context: RoutingContext, required_pairs: tuple[tuple[str, str], ...]) -> None: ...
def build_osrm_table_matrix(provider: RoutingProvider, request: RouteMatrixRequest) -> RouteMatrix: ...
def load_pinned_route_evidence(matrix_id: str) -> RouteMatrix: ...
```

## Validation Rules

1. Every required pair has a route cell.
2. Every publication route cell is road-validated and not fallback.
3. Geodesic fallback has `road_validated=False`.
4. Public OSRM is opt-in and non-publication by default.
5. Route matrix context snapshot matches plan/evaluation context.
6. Missing source is reported explicitly.
7. Utility missingness keeps missing source separate from low value.

## UI / API Integration

- Dashboard shows route source, provider, validation state, fallback state, and route matrix ID.
- Notebook and pipeline cannot silently switch route sources.

## Workflow

1. Load catalog/context bundle.
2. Resolve routing provider according to refresh policy.
3. Load pinned route evidence or build local OSRM matrix.
4. Validate required pairs.
5. Inject matrix into solver.
6. Pass the same routing context to evaluator, renderer, benchmark, and explanation.

## Files to Create

- `src/itinerary_system/routing/osrm_provider.py`
- `src/itinerary_system/routing/context.py`
- `src/itinerary_system/routing/pinned.py`
- `docs/reference/local_osrm_runbook.md`
- `tests/routing/test_osrm_provider.py`
- `tests/routing/test_routing_context.py`

## Files to Modify

- `src/itinerary_system/routing/__init__.py`
- `scripts/build_validated_route_matrix.py`
- `scripts/check_route_source.py`
- `src/itinerary_system/multi_objective_route.py`
- `src/itinerary_system/hierarchical_gurobi.py`

## Error Handling

- Missing matrix cell: `RouteMatrixCellMissing`.
- Empty matrix: `RouteMatrixMissing`.
- Fallback in publication mode: `RouteMatrixNotPublicationEligible`.
- Public OSRM without approval: route policy error.
- Snap mismatch: validation finding with affected coordinate.

## Testing Checklist

- [ ] OSRM provider fake Table response builds matrix.
- [ ] OSRM Route response creates road-valid leg.
- [ ] OSRM Nearest/snap diagnostics are stored.
- [ ] Geodesic fallback is demo-only.
- [ ] Publication solver rejects missing matrix.
- [ ] Optimizer/evaluator/renderer share route matrix ID.
- [ ] Missing Yelp/source is not scored as low Yelp/source.
- [ ] `python -m ruff check src tests scripts`
- [ ] `python -m pytest`

## Roadmap / Full Pipeline Update

Update CP-002 only when benchmark route matrices are complete and strict validation passes.

## Acceptance Criteria

1. Publication comparisons use road-valid route evidence.
2. All route consumers share the same context.
3. Geodesic fallback is impossible to mistake for road validation.
4. Local or pinned route evidence supports benchmark scenarios.

# Phase G3.0 Implementation Plan: Independent Evaluator And Certificate Skeleton

## Goal

Implement **independent evaluator and certificate skeleton**.

This gate should allow the system to:

1. Recompute final displayed plan validity from artifacts, not solver booleans.
2. Return hard failures, warnings, component metrics, and certificate IDs.
3. Make repair outputs count only after independent eligibility.

## Current State

Implemented or partially implemented:

- `ResearchEvaluationReport` and `evaluate_phase0_plan()` exist in `research_artifacts.py`.
- `repair_progressively()` accepts an evaluator hook and stores evaluation records.
- `tests/repair/test_progressive.py` uses `SequenceGateEvaluator` as a fixture.

Missing:

- `src/itinerary_system/evaluation/` package.
- `PlanEvaluator` production service.
- `EvaluationCertificate`.
- Full hard gate over route, lodging, ownership, time, opening windows, weather, closure, duplicate visits, budget, and post-solve mutation.
- Component metrics before aggregate score.

## New Components to Add

### Component 1

`PlanEvaluator`

Responsibility:

Independently evaluate a `PlanArtifactV2` using `DatasetBundle`, `RoutingContext`, ownership policy, and source artifacts.

### Component 2

`EvaluationCertificate`

Responsibility:

Persist evaluator eligibility, hard failures, warnings, metric references, and content hash.

### Component 3

`EvaluationMetricRegistry`

Responsibility:

Define component metrics and keep aggregate score from masking hard failures.

## Class / Registry Diagrams

```text
+-------------------------------------------------------------------------------+
|                              PlanEvaluator (Service Class)                     |
+-------------------------------------------------------------------------------+
|  - metric_registry: EvaluationMetricRegistry                                   |
|  - ownership_policy: OwnershipPolicy                                           |
+-------------------------------------------------------------------------------+
|  + evaluate(plan, data, routing): EvaluationCertificate --> Independent check  |
|  + hard_failures(plan, data, routing): tuple[Finding, ...] --> Blocking issues |
|  + warnings(plan, data, routing): tuple[Finding, ...] --> Non-blocking issues  |
+-------------------------------------------------------------------------------+
```

```text
+-------------------------------------------------------------------------------+
|                              EvaluationCertificate                             |
+-------------------------------------------------------------------------------+
|  - certificate_id: str                                                         |
|  - plan_id: str                                                                |
|  - content_hash: str                                                           |
|  - eligible: bool                                                              |
|  - hard_failures: tuple[str, ...]                                              |
|  - warnings: tuple[str, ...]                                                   |
|  - metric_refs: tuple[str, ...]                                                |
+-------------------------------------------------------------------------------+
|  + to_record(): dict[str, Any]             --> Stable artifact                 |
|  + explanation_refs(): tuple[str, ...]     --> Evidence for explanations       |
+-------------------------------------------------------------------------------+
```

```text
+-------------------------------------------------------------------------------+
|                 EvaluationMetricRegistry (Persistent Registry / Manager)       |
+-------------------------------------------------------------------------------+
|  - metrics: dict[str, MetricSpec]                                              |
+-------------------------------------------------------------------------------+
|  + get(metric_id): MetricSpec              --> Metric definition               |
|  + validate(result): tuple[Finding, ...]   --> Metric schema checks            |
+-------------------------------------------------------------------------------+
```

## Class Diagram Rules

1. Evaluator owns comparison metrics.
2. Solver booleans are evidence, not final truth.
3. Hard failures cannot be hidden by high utility.
4. Certificates bind to plan content hash.

## Data Model

```python
@dataclass(frozen=True)
class EvaluationFinding:
    finding_id: str
    severity: Literal["hard_failure", "warning", "info"]
    code: str
    target_id: str | None
    evidence_refs: tuple[str, ...]
    message: str

@dataclass(frozen=True)
class EvaluationMetric:
    metric_id: str
    name: str
    value: float | str | bool
    unit: str
    evidence_refs: tuple[str, ...]
```

## Storage / State

Persistent:

- `runs/<run_id>/evaluations/*.json`
- `runs/<run_id>/evaluations/certificates/*.json`
- Metric registry definitions under package data or code.

Temporary:

- Recomputed route/time/lodging/constraint checks.

## Required Methods

```python
def evaluate_plan(plan: PlanArtifactV2, data_bundle: DatasetBundle, routing_context: RoutingContext) -> EvaluationCertificate: ...
def validate_owned_constraints_in_plan(plan: PlanArtifactV2) -> tuple[EvaluationFinding, ...]: ...
def validate_route_connectivity(plan: PlanArtifactV2, routing_context: RoutingContext) -> tuple[EvaluationFinding, ...]: ...
def validate_daily_time_feasibility(plan: PlanArtifactV2, routing_context: RoutingContext) -> tuple[EvaluationFinding, ...]: ...
def invalidate_certificate_if_content_changed(plan: PlanArtifactV2, certificate: EvaluationCertificate) -> EvaluationCertificate: ...
```

## Validation Rules

1. Plan source run and artifact links exist.
2. Content hash matches certificate.
3. Parent-child lineage is valid.
4. Hard owned constraints are preserved or explicitly permitted.
5. Route connectivity is complete.
6. Road validation failure makes plan ineligible.
7. Daily time feasibility is recomputed.
8. Opening windows are enforced when available.
9. Lodging continuity and booked lodging are checked.
10. Budget feasibility is checked.
11. Weather and closure feasibility are checked.
12. Duplicate visits are checked.
13. Post-solve mutation invalidates certificate.

## UI / API Integration

- Pipeline, benchmark, and prototype display certificate state.
- Repair controller must use this evaluator before accepting a child plan.

## Workflow

1. Load plan and artifacts.
2. Verify linkage and content hash.
3. Recompute hard checks.
4. Compute component metrics.
5. Assign eligible/ineligible.
6. Write certificate.
7. Return explanation-ready evidence refs.

## Files to Create

- `src/itinerary_system/evaluation/__init__.py`
- `src/itinerary_system/evaluation/certificate.py`
- `src/itinerary_system/evaluation/evaluator.py`
- `src/itinerary_system/evaluation/metrics.py`
- `src/itinerary_system/evaluation/validation.py`
- `tests/evaluation/test_evaluator.py`
- `tests/evaluation/test_certificate.py`

## Files to Modify

- `src/itinerary_system/repair/progressive.py`
- `src/itinerary_system/research_artifacts.py`
- `docs/current/current_problem_manifest.md`

## Error Handling

- Missing artifact: hard failure.
- Stale certificate: hard failure.
- Missing route cell: hard failure.
- Fallback route in publication mode: hard failure.
- Missing optional opening hours: warning, unless scenario requires closure evidence.

## Testing Checklist

- [ ] Eligible plan receives certificate.
- [ ] Route fallback makes plan ineligible.
- [ ] Content hash mismatch invalidates certificate.
- [ ] Locked/booked violation is hard failure.
- [ ] Warnings are separate from failures.
- [ ] Component metrics are visible.
- [ ] Repair controller uses evaluator certificate.
- [ ] `python -m ruff check src tests scripts`
- [ ] `python -m pytest`

## Roadmap / Full Pipeline Update

Update CP-006 only after evaluator package and certificate tests pass.

## Acceptance Criteria

1. Final plan eligibility is evaluator-owned.
2. Every accepted repair has a certificate.
3. Hard failures cannot be ranked as successful plans.
4. Certificate evidence can feed explanations.

# Phase G4.0 Implementation Plan: Single-Day Ownership-Aware Repair

## Goal

Implement **single-day ownership-aware repair**.

This gate should allow the system to:

1. Accept typed user edits or disruptions for one affected day.
2. Build an ownership-aware repair model using confirmed constraints and road-valid route matrix cells.
3. Produce a child plan, typed diff, solver stage trace, and independent evaluator certificate.

## Current State

Implemented or partially implemented:

- `src/itinerary_system/repair/change_variables.py`: decision variables, typed change variables, objective terms/components.
- `src/itinerary_system/repair/master_model.py`: repair master scaffold, variables, locked/booked constraints, objective components.
- `src/itinerary_system/repair/day_route_solver.py`: route-matrix-backed day-route subproblem.
- `src/itinerary_system/repair/lexicographic.py`: sequential lexicographic candidate selection.
- Tests exist for variables, locked/booked handling, route matrix injection, opening windows, and lexicographic priority.

Missing or incomplete:

- Full typed `EditRequest`, `DisruptionSnapshot`, and `RepairRequest` modules.
- Production evaluator certificate integration.
- Full solver backend beyond enumeration/Gurobi reference for small candidates.
- Contextual burden still proxy-level.

## New Components to Add

### Component 1

`RepairRequestSchema`

Responsibility:

Define typed edit/disruption inputs and validate target IDs before model construction.

### Component 2

`SingleDayRepairService`

Responsibility:

Coordinate request validation, neighborhood build, master model, day-route subproblem, lexicographic solve, child extraction, diff, and evaluator.

### Component 3

`SolverStageTraceStore`

Responsibility:

Persist objective values, bounds, gaps, fixed limits, infeasibility evidence, and selected components.

## Class / Registry Diagrams

```text
+-------------------------------------------------------------------------------+
|                         RepairRequestSchema (Stateless Utility Module)         |
+-------------------------------------------------------------------------------+
|  - No persistent internal state                                                |
+-------------------------------------------------------------------------------+
|  + validate(request): tuple[Finding, ...]  --> Typed request checks            |
|  + normalize(raw): RepairRequest           --> Structured repair request       |
+-------------------------------------------------------------------------------+
```

```text
+-------------------------------------------------------------------------------+
|                         SingleDayRepairService (Service Class)                 |
+-------------------------------------------------------------------------------+
|  - route_matrix: RouteMatrix                                                   |
|  - evaluator: PlanEvaluator                                                    |
|  - ownership_policy: OwnershipPolicy                                           |
+-------------------------------------------------------------------------------+
|  + repair(request): RepairOutcome          --> One-day certified child attempt |
|  + build_model(...): RepairModel           --> Ownership-aware model           |
|  + certify(child): EvaluationCertificate   --> Independent evaluation          |
+-------------------------------------------------------------------------------+
```

```text
+-------------------------------------------------------------------------------+
|                  SolverStageTraceStore (Persistent Registry / Manager)         |
+-------------------------------------------------------------------------------+
|  - trace_dir: Path                                                             |
+-------------------------------------------------------------------------------+
|  + write(result): Path                     --> Stage trace artifact            |
|  + load(trace_id): LexicographicResult     --> Replay selected result          |
+-------------------------------------------------------------------------------+
```

## Class Diagram Rules

1. Request parsing is separate from model construction.
2. No live API calls inside model builders.
3. Route matrix is injected.
4. Sequential lexicographic stages must not collapse into one weighted sum.
5. Child plan is not accepted until independently evaluated.

## Data Model

Edit types:

- add stop
- delete stop
- replace stop
- move stop to another day
- reorder stops
- time shift
- lodging change
- road/corridor change
- constraint relaxation

Objective stages:

```text
Stage 0: minimize hard/booked relaxations
Stage 1: minimize ownership-weighted typed changes
Stage 2: minimize contextual burden
Stage 3: maximize utility/diversity/scenic value
```

## Storage / State

Persistent:

- `runs/<run_id>/requests/*.json`
- `runs/<run_id>/planner_runs.jsonl`
- `runs/<run_id>/plans/*.json`
- `runs/<run_id>/diffs/*.json`
- `runs/<run_id>/evaluations/*.json`
- `runs/<run_id>/solver/stage_traces/*.json`

Temporary:

- Repair model and candidate solutions.

## Required Methods

```python
def normalize_repair_request(raw: dict[str, Any]) -> RepairRequest: ...
def solve_single_day_repair(parent: PlanArtifactV2, request: RepairRequest, routing: RoutingContext, evaluator: PlanEvaluator) -> RepairOutcome: ...
def build_repair_master_model(parent: PlanArtifactV2, request: RepairRequest, neighborhood: RepairNeighborhood, matrix: RouteMatrix) -> RepairModel: ...
def solve_lexicographically(model: RepairModel, tolerances: ObjectiveTolerances) -> LexicographicResult: ...
```

## Validation Rules

1. Request parent exists.
2. Target IDs exist or are approved candidate additions.
3. Unconfirmed LLM constraints are inactive.
4. Locked constraints cannot be relaxed.
5. Booked relaxation requires explicit permission.
6. Route matrix is required in publication mode.
7. Prior lexicographic optima are fixed within tolerance before later stages.
8. Infeasible stages store evidence.

## UI / API Integration

Prototype can submit typed requests after G7. Until then, tests and pipeline scripts call the service.

## Workflow

1. Validate repair request.
2. Build parent index.
3. Build same-day neighborhood.
4. Build master model.
5. Generate day-route candidates.
6. Solve lexicographically.
7. Extract child plan.
8. Compute diff.
9. Evaluate independently.
10. Accept only if eligible.

## Files to Create

- `src/itinerary_system/repair/requests.py`
- `src/itinerary_system/repair/single_day.py`
- `src/itinerary_system/repair/stage_trace.py`
- `tests/repair/test_requests.py`
- `tests/repair/test_single_day.py`

## Files to Modify

- `src/itinerary_system/repair/master_model.py`
- `src/itinerary_system/repair/day_route_solver.py`
- `src/itinerary_system/repair/lexicographic.py`
- `src/itinerary_system/repair/__init__.py`

## Error Handling

- Unknown target: request validation failure.
- Missing route matrix: route matrix error.
- No feasible candidates: infeasible repair outcome with diagnosis.
- Evaluator rejects child: no accepted child.

## Testing Checklist

- [ ] Typed request schema rejects invalid target.
- [ ] Single-day repair preserves locked/booked constraints.
- [ ] Route matrix drives travel time.
- [ ] Stage 0 cannot degrade in Stage 1.
- [ ] Infeasible stage stores failure evidence.
- [ ] Child plan has parent ID and new content hash.
- [ ] Evaluator certificate gates acceptance.
- [ ] `python -m ruff check src tests scripts`
- [ ] `python -m pytest`

## Roadmap / Full Pipeline Update

Do not mark single-day repair complete until production evaluator integration exists.

## Acceptance Criteria

1. One-day repair produces independently eligible child plans.
2. Lexicographic preservation-before-utility is enforced.
3. Diff and stage evidence are stored.

# Phase G5.0 Implementation Plan: Progressive Multi-Day Repair

## Goal

Implement **progressive multi-day repair**.

This gate should allow the system to:

1. Try the smallest repair neighborhood first.
2. Stop at the smallest independently eligible child plan.
3. Store every failed attempt and provide no-success diagnosis.

## Current State

Implemented or partially implemented:

- `src/itinerary_system/repair/neighborhood.py`: current radii are `same_stop_time_shift`, `same_day_replacement`, `adjacent_day_move`, `hotel_preserving_reroute`, `hotel_changing_repair`, `full_reoptimization`.
- `src/itinerary_system/repair/progressive.py`: progressive controller, attempts, candidate evaluations, diagnosis.
- `tests/repair/test_progressive.py`: smallest eligible radius and final fallback behavior.

Missing or incomplete:

- Production evaluator integration.
- Prompt-specified R0-R5 labels are not exactly the current code names.
- Base city/lodging and corridor/regional radii need explicit mapping.
- No benchmark-scale failure diagnosis artifacts.

## New Components to Add

### Component 1

`RepairRadiusPolicy`

Responsibility:

Map public R0-R5 radius semantics to current `RepairRadius` enum values and future expanded radii.

### Component 2

`ProgressiveAttemptStore`

Responsibility:

Persist attempts, failed radius evidence, candidate evaluations, stage traces, and diagnosis.

### Component 3

`SmallestRelaxationDiagnoser`

Responsibility:

Summarize why no radius succeeded and what relaxation would be smallest.

## Class / Registry Diagrams

```text
+-------------------------------------------------------------------------------+
|                              RepairRadiusPolicy                                |
+-------------------------------------------------------------------------------+
|  - public_order: tuple[str, ...]                                               |
|  - implementation_order: tuple[RepairRadius, ...]                              |
+-------------------------------------------------------------------------------+
|  + expand(request): tuple[RepairRadius, ...] --> Ordered allowed radii         |
|  + public_label(radius): str              --> R0-R5 reporting label            |
+-------------------------------------------------------------------------------+
```

```text
+-------------------------------------------------------------------------------+
|                ProgressiveAttemptStore (Persistent Registry / Manager)         |
+-------------------------------------------------------------------------------+
|  - attempts_dir: Path                                                          |
+-------------------------------------------------------------------------------+
|  + write_attempt(attempt): Path           --> Attempt artifact                 |
|  + write_outcome(outcome): Path           --> Outcome artifact                 |
+-------------------------------------------------------------------------------+
```

```text
+-------------------------------------------------------------------------------+
|                SmallestRelaxationDiagnoser (Stateless Utility Module)          |
+-------------------------------------------------------------------------------+
|  - No persistent internal state                                                |
+-------------------------------------------------------------------------------+
|  + diagnose(attempts): RepairDiagnosis    --> No-success explanation          |
+-------------------------------------------------------------------------------+
```

## Class Diagram Rules

1. Full reoptimization is always final fallback.
2. Outside-neighborhood entities are frozen.
3. All failed radii are preserved as evidence.
4. Accepted radius must be the smallest eligible radius in the configured order.

## Data Model

Public radius labels:

```text
R0 direct disrupted item/edit only
R1 same day and adjacent route legs
R2 adjacent days and affected lodging boundary
R3 base city/lodging neighborhood
R4 corridor or regional neighborhood
R5 full editable scope as final fallback
```

Current code mapping:

```text
same_stop_time_shift -> R0
same_day_replacement -> R1
adjacent_day_move -> R2
hotel_preserving_reroute -> R2/R3 depending on lodging boundary
hotel_changing_repair -> R3
full_reoptimization -> R5
```

## Storage / State

Persistent:

- `runs/<run_id>/repair/attempts/*.json`
- `runs/<run_id>/repair/outcome.json`
- `runs/<run_id>/repair/diagnosis.json`

Temporary:

- Candidate pools and model objects.

## Required Methods

```python
def build_radius_sequence(request: RepairRequest, policy: RepairRadiusPolicy) -> tuple[RepairRadius, ...]: ...
def repair_progressively(... ) -> RepairOutcome: ...
def diagnose_failure(attempts: tuple[RepairAttempt, ...]) -> RepairDiagnosis: ...
```

## Validation Rules

1. Radii are deduplicated and sorted by policy.
2. Full reoptimization moves to final position if present.
3. Booked lodging changes require explicit permission.
4. Locked items remain frozen across radii.
5. Every attempted radius stores candidate/evaluation evidence.
6. Accepted child is evaluator-eligible.

## UI / API Integration

The prototype displays attempted radii, accepted radius, and why smaller radii failed.

## Workflow

1. Resolve allowed radii.
2. For each radius, build neighborhood.
3. Generate candidates and stage traces.
4. Evaluate candidate children.
5. Stop at first eligible child.
6. Save diagnosis if none succeeds.

## Files to Create

- `src/itinerary_system/repair/radius_policy.py`
- `src/itinerary_system/repair/attempt_store.py`
- `tests/repair/test_radius_policy.py`

## Files to Modify

- `src/itinerary_system/repair/neighborhood.py`
- `src/itinerary_system/repair/progressive.py`
- `src/itinerary_system/repair/__init__.py`

## Error Handling

- Unknown radius: validation failure.
- Empty radius sequence: request validation failure.
- No eligible child: infeasible outcome with diagnosis.
- Store write failure: pipeline failure, not silent loss.

## Testing Checklist

- [ ] Radius policy maps R0-R5 labels.
- [ ] Outside-neighborhood stops freeze.
- [ ] Full reoptimization is attempted only last.
- [ ] Smallest eligible radius is selected.
- [ ] No-success diagnosis includes best weighted edit cost and failure reasons.
- [ ] `python -m ruff check src tests scripts`
- [ ] `python -m pytest`

## Roadmap / Full Pipeline Update

Update CP-005 only when production evaluator-backed progressive repair passes.

## Acceptance Criteria

1. Progressive repair stops at smallest eligible radius.
2. Failed radii are replayable.
3. No-success diagnosis is explanation-ready.

# Phase G6.0 Implementation Plan: Benchmark Package

## Goal

Implement **benchmark package**.

This gate should allow the system to:

1. Generate split-safe benchmark scenarios from canonical parent plans.
2. Compare repair methods and baselines on frozen route/context evidence.
3. Emit defensible metrics without row-family leakage.

## Current State

Implemented:

- Some plans, routing, and repair primitives exist.

Missing:

- `src/itinerary_system/benchmark/` package.
- Canonical benchmark parent registry.
- Six disruption generator.
- Ownership-profile generator.
- Baseline adapters.
- Split-by-family logic.
- Benchmark metrics package.

## New Components to Add

### Component 1

`BenchmarkScenarioFactory`

Responsibility:

Build six disruption families over canonical parent plans and ownership profiles.

### Component 2

`BenchmarkMethodRegistry`

Responsibility:

Register full reoptimization, current hybrid pipeline, weighted-sum repair, fixed-radius repair, progressive lexicographic repair, and optional LLM-only rewrite.

### Component 3

`BenchmarkMetricComputer`

Responsibility:

Compute feasibility, preservation, utility, route, budget, runtime, certificate, fallback, and explanation metrics.

## Class / Registry Diagrams

```text
+-------------------------------------------------------------------------------+
|                       BenchmarkScenarioFactory (Service Class)                 |
+-------------------------------------------------------------------------------+
|  - parent_registry: CanonicalParentPlanRegistry                                |
+-------------------------------------------------------------------------------+
|  + build_scenarios(): tuple[BenchmarkScenario, ...] --> Frozen scenarios       |
|  + validate_splits(...): tuple[Finding, ...] --> No family leakage             |
+-------------------------------------------------------------------------------+
```

```text
+-------------------------------------------------------------------------------+
|               BenchmarkMethodRegistry (Persistent Registry / Manager)          |
+-------------------------------------------------------------------------------+
|  - methods: dict[str, BenchmarkMethod]                                         |
+-------------------------------------------------------------------------------+
|  + register(method): None                 --> Add method                       |
|  + run(method_id, scenario): BenchmarkResult --> Execute method                |
+-------------------------------------------------------------------------------+
```

```text
+-------------------------------------------------------------------------------+
|                   BenchmarkMetricComputer (Stateless Utility Module)           |
+-------------------------------------------------------------------------------+
|  - No persistent internal state                                                |
+-------------------------------------------------------------------------------+
|  + compute(result, scenario): dict[str, float | str] --> Metrics               |
|  + validate(metrics): tuple[Finding, ...] --> Metric schema checks             |
+-------------------------------------------------------------------------------+
```

## Class Diagram Rules

1. Every scenario references a full parent-plan/disruption family.
2. Never split rows from the same family across train/dev/test.
3. Optional LLM-only baseline needs frozen input, frozen output, and independent evaluation.
4. Metrics consume evaluator certificates, not solver booleans.

## Data Model

Disruption families:

1. weather deterioration
2. road closure
3. hotel unavailability
4. attraction closure/opening-window conflict
5. reduced driving tolerance/fatigue
6. newly added user must-visit

Ownership profiles:

- flexible
- mixed
- booking-constrained
- scenic-route-protective
- high-lock

Metrics:

- feasibility
- hard failure counts
- locked preservation
- booked preservation
- unaffected-day preservation
- weighted edit cost
- utility retained
- utility regret
- weather-risk reduction
- nature-exposure reduction
- route-valid travel
- budget delta
- repair radius
- runtime
- solver status/gap
- certificate coverage
- fallback rate
- explanation evidence coverage

## Storage / State

Persistent:

- `data/benchmark/scenarios/*.json`
- `runs/<run_id>/benchmark/results.jsonl`
- `runs/<run_id>/metrics/benchmark_summary.csv`

Temporary:

- Method runner objects.

## Required Methods

```python
def build_benchmark_scenarios(parent_plan_ids: tuple[str, ...]) -> tuple[BenchmarkScenario, ...]: ...
def split_benchmark_families(scenarios: tuple[BenchmarkScenario, ...], seed: int) -> BenchmarkSplit: ...
def run_benchmark_method(method_id: str, scenario: BenchmarkScenario, pipeline: ResearchPipelineRunner) -> BenchmarkResult: ...
def compute_benchmark_metrics(result: BenchmarkResult, scenario: BenchmarkScenario) -> dict[str, float | str]: ...
```

## Validation Rules

1. Scenario parent plan exists.
2. Disruption family is supported.
3. Ownership profile has valid constraints.
4. Route evidence is publication-ready for publication methods.
5. Same family cannot cross splits.
6. LLM baseline disabled without frozen/evaluated output.

## UI / API Integration

Dashboard can display benchmark summaries after G7, but benchmark package is primarily CLI/pipeline-facing.

## Workflow

1. Load canonical parents.
2. Generate ownership profiles.
3. Generate disruptions.
4. Validate route/context evidence.
5. Run methods.
6. Evaluate and certificate outputs.
7. Compute metrics.
8. Export summary.

## Files to Create

- `src/itinerary_system/benchmark/__init__.py`
- `src/itinerary_system/benchmark/scenarios.py`
- `src/itinerary_system/benchmark/methods.py`
- `src/itinerary_system/benchmark/metrics.py`
- `src/itinerary_system/benchmark/splits.py`
- `tests/benchmark/test_scenarios.py`
- `tests/benchmark/test_metrics.py`
- `tests/benchmark/test_splits.py`

## Files to Modify

- `docs/current/current_problem_manifest.md`
- `docs/reference/data_dictionary.md`

## Error Handling

- Missing canonical parent: benchmark setup failure.
- Route not publication-ready: scenario/method ineligible.
- Split leakage: hard validation failure.
- LLM baseline unavailable: mark skipped, not failed competitor.

## Testing Checklist

- [ ] Six disruption families generate.
- [ ] Five ownership profiles generate.
- [ ] Split leakage test passes.
- [ ] LLM-only baseline disabled without frozen evaluation.
- [ ] Metrics use certificates.
- [ ] Fallback rate computed.
- [ ] `python -m ruff check src tests scripts`
- [ ] `python -m pytest`

## Roadmap / Full Pipeline Update

Do not report benchmark results until G3 certificates and G2 route evidence are ready.

## Acceptance Criteria

1. Benchmark scenarios are frozen and split-safe.
2. Methods run through common evaluation.
3. Metrics are reproducible and artifact-linked.

# Phase G7.0 Implementation Plan: Authoritative Pipeline Runner And Thin Notebook

## Goal

Implement **authoritative pipeline runner and thin notebook**.

This gate should allow the system to:

1. Run generation, repair, evaluation, benchmark, explanation, and dashboard export from one package-level entry point.
2. Emit immutable `runs/<run_id>/` artifacts.
3. Make notebooks display-only.

## Current State

Implemented or partially implemented:

- `src/itinerary_system/experiment_runner.py`: large current pipeline helper.
- `notebook/production_system_blueprint.ipynb`: current notebook path.
- `scripts/run_phase0_evidence_pipeline.py`: Phase 0 command.
- `scripts/serve_dashboard.py`: dashboard server.

Missing:

- `src/itinerary_system/pipeline_runner.py`.
- `src/itinerary_system/run_artifacts.py`.
- Immutable run directory layout.
- Redacted config serialization.
- CLI entry point.
- Notebook-to-run consistency checks.

## New Components to Add

### Component 1

`ResearchPipelineRunner`

Responsibility:

Coordinate config, dataset loading, routing context, parent plan, repair/generation mode, evaluator, diffs, explanations, metrics, and dashboard export.

### Component 2

`RunArtifactStore`

Responsibility:

Create immutable run directories and write/read stable artifacts.

### Component 3

`ThinNotebookAdapter`

Responsibility:

Expose notebook-safe calls that call package services and display artifacts.

## Class / Registry Diagrams

```text
+-------------------------------------------------------------------------------+
|                         ResearchPipelineRunner (Service Class)                 |
+-------------------------------------------------------------------------------+
|  - output_root: Path                                                           |
|  - artifact_store_factory: Callable                                           |
+-------------------------------------------------------------------------------+
|  + run_research_pipeline(...): PipelineRun --> Authoritative entry point       |
|  + run_generation(...): PipelineRun       --> Parent/generation mode           |
|  + run_repair(...): PipelineRun           --> Repair mode                      |
|  + run_benchmark(...): PipelineRun        --> Benchmark mode                   |
+-------------------------------------------------------------------------------+
```

```text
+-------------------------------------------------------------------------------+
|                    RunArtifactStore (Persistent Registry / Manager)            |
+-------------------------------------------------------------------------------+
|  - run_dir: Path                                                               |
+-------------------------------------------------------------------------------+
|  + create(): None                         --> Refuse existing run dir          |
|  + write_json(name, payload): Path        --> Stable artifact                  |
|  + append_jsonl(name, payload): Path      --> Records                          |
|  + write_manifest(run): Path              --> Top-level manifest               |
+-------------------------------------------------------------------------------+
```

```text
+-------------------------------------------------------------------------------+
|                         ThinNotebookAdapter                                    |
+-------------------------------------------------------------------------------+
|  - No persistent internal state                                                |
+-------------------------------------------------------------------------------+
|  + run_from_config(config_path): PipelineRun --> Calls package runner          |
|  + load_completed_run(run_dir): dict       --> Display-only payload            |
+-------------------------------------------------------------------------------+
```

## Class Diagram Rules

1. Pipeline owns canonical artifact generation.
2. Notebook must not call APIs directly.
3. Notebook must not construct Gurobi models.
4. Notebook must not mutate plan dictionaries.
5. Run directories are immutable.

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
```

Run layout:

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

Persistent:

- Immutable `runs/<run_id>/`.

Temporary:

- Provider clients and solver models.

## Required Methods

```python
def run_research_pipeline(... ) -> PipelineRun: ...
def create_run_dir(output_root: Path, run_id: str) -> Path: ...
def write_resolved_config_redacted(config: TripConfig, path: Path) -> Path: ...
def assert_notebook_is_thin(notebook_path: Path) -> tuple[Finding, ...]: ...
```

## Validation Rules

1. Existing run directory cannot be overwritten.
2. Resolved config redacts secrets.
3. Refresh policy controls live provider calls.
4. Final plans have source run, route, diff, evaluation, and certificate artifacts.
5. Notebook contains no business logic.

## UI / API Integration

- CLI calls `run_research_pipeline()`.
- Notebook calls adapter only.
- Dashboard loads run artifacts.

## Workflow

1. Resolve config.
2. Create run directory.
3. Load dataset and routing context.
4. Load or generate parent.
5. Run repair/generation/benchmark.
6. Evaluate and certificate.
7. Write diffs, explanations, metrics, dashboard.
8. Return `PipelineRun`.

## Files to Create

- `src/itinerary_system/pipeline_runner.py`
- `src/itinerary_system/run_artifacts.py`
- `src/itinerary_system/notebook_adapter.py`
- `tests/test_pipeline_runner.py`
- `tests/test_run_artifacts.py`
- `tests/test_notebook_adapter.py`

## Files to Modify

- `src/itinerary_system/experiment_runner.py`
- `src/itinerary_system/map_exporter.py`
- `notebook/production_system_blueprint.ipynb`
- `README.md`

## Error Handling

- Existing run directory: fail.
- Provider unavailable: record unavailable status.
- Ineligible strict plan: write diagnostics and fail strict command.
- Notebook business logic: validation failure.

## Testing Checklist

- [ ] Run directory is immutable.
- [ ] Redacted config contains no secrets.
- [ ] Repair mode emits complete artifacts.
- [ ] Benchmark mode emits metrics.
- [ ] Notebook-to-run consistency passes.
- [ ] `python -m ruff check src tests scripts`
- [ ] `python -m pytest`

## Roadmap / Full Pipeline Update

Update CP-007 only after pipeline runner and thin notebook validation pass.

## Acceptance Criteria

1. `run_research_pipeline()` is authoritative.
2. Run artifacts are immutable and complete.
3. Notebook is display-only.

# Phase G8.0 Implementation Plan: Evidence-Bounded LLM Preference-To-Commitment Layer

## Goal

Implement **evidence-bounded LLM preference-to-commitment layer**.

This gate should allow the system to:

1. Parse repair language into typed edit candidates.
2. Infer inspectable taste hypotheses from an accepted parent plan.
3. Suggest inactive commitments for stops, routes, lodging, and day structures.
4. Activate only deterministic-valid and user-confirmed commitments.

This gate is inactive for solver control until validation and user confirmation.

## Current State

Implemented:

- `OwnedConstraint` supports `ConstraintOrigin.LLM_INTERPRETATION`; tests prove unconfirmed LLM constraints are inactive.
- `repair_planner.py` says LLM output is represented only as confirmed `ParsedRepairIntent`.

Missing:

- `src/itinerary_system/llm_taste/`.
- `TasteEvidencePack`, `TasteSignalArtifact`, `TasteProfileArtifact`, `SuggestedCommitmentArtifact`, `CommitmentValidationReport`, `PromptRun`, `PromptCriticReport`, `MapExplanationCard`, `UserCommitmentDecision`.
- Prompt registry and prompt validation.
- Map confirmation workflow.

## New Components to Add

### Component 1

`TasteEvidencePackBuilder`

Responsibility:

Build bounded parent-plan evidence for the LLM.

### Component 2

`LlmTastePromptRunner`

Responsibility:

Run strict JSON prompt calls or load frozen prompt outputs.

### Component 3

`CommitmentValidationService`

Responsibility:

Reject unsupported IDs, evidence, strengths, booking claims, sensitive inferences, and hidden solver directives.

### Component 4

`UserCommitmentCompiler`

Responsibility:

Compile confirmed decisions into `OwnedConstraint` records.

## Class / Registry Diagrams

```text
+-------------------------------------------------------------------------------+
|                TasteEvidencePackBuilder (Stateless Utility Module)             |
+-------------------------------------------------------------------------------+
|  - No persistent internal state                                                |
+-------------------------------------------------------------------------------+
|  + build(parent, data, route_matrix): TasteEvidencePack --> Bounded input      |
+-------------------------------------------------------------------------------+
```

```text
+-------------------------------------------------------------------------------+
|                         LlmTastePromptRunner (Service Class)                   |
+-------------------------------------------------------------------------------+
|  - prompt_registry: PromptRegistry                                             |
|  - llm_client: LlmClient | None                                                |
+-------------------------------------------------------------------------------+
|  + extract_signals(pack): TasteSignalArtifact --> Taste hypotheses             |
|  + compress_profile(signals): TasteProfileArtifact --> User-facing profile     |
|  + suggest_commitments(profile): SuggestedCommitmentArtifact --> Inactive      |
+-------------------------------------------------------------------------------+
```

```text
+-------------------------------------------------------------------------------+
|              CommitmentValidationService (Stateless Utility Module)            |
+-------------------------------------------------------------------------------+
|  - No persistent internal state                                                |
+-------------------------------------------------------------------------------+
|  + validate(suggestions, pack): CommitmentValidationReport --> Deterministic   |
|  + downgrade_or_reject(item): Finding | None --> Strength policy               |
+-------------------------------------------------------------------------------+
```

```text
+-------------------------------------------------------------------------------+
|                 UserCommitmentCompiler (Stateless Utility Module)              |
+-------------------------------------------------------------------------------+
|  - No persistent internal state                                                |
+-------------------------------------------------------------------------------+
|  + compile(decisions): tuple[OwnedConstraint, ...] --> Confirmed constraints   |
+-------------------------------------------------------------------------------+
```

## Class Diagram Rules

1. LLM never generates final itinerary for publication comparison.
2. LLM never decides feasibility, route validity, hotel availability, or certification.
3. LLM never emits Gurobi code or arbitrary mathematical constraints.
4. Every suggestion starts inactive.
5. Deterministic validator is authoritative.

## Data Model

Taste dimensions:

```text
scenic_photo
nature_outdoor
architecture_history
museum_culture
food_local_experience
social_must_go
relaxed_pace
dense_explorer_pace
low_hotel_switching
scenic_driving
low_driving_burden
weather_cautious
budget_sensitive
accessibility_sensitive
indoor_backup_preference
```

Prompt artifacts:

- `TasteEvidencePack`
- `TasteSignalArtifact`
- `TasteProfileArtifact`
- `SuggestedCommitmentArtifact`
- `CommitmentValidationReport`
- `PromptRun`
- `PromptCriticReport`
- `MapExplanationCard`
- `UserCommitmentDecision`

## Storage / State

Persistent:

- `runs/<run_id>/llm_taste/evidence_packs/`
- `runs/<run_id>/llm_taste/prompt_runs.jsonl`
- `runs/<run_id>/llm_taste/taste_profiles/`
- `runs/<run_id>/llm_taste/suggested_commitments/`
- `runs/<run_id>/llm_taste/user_commitment_decisions.jsonl`

Temporary:

- LLM client and raw provider response unless explicitly saved in redacted dev mode.

## Required Methods

```python
def build_taste_evidence_pack(parent: PlanArtifactV2, data_bundle: DatasetBundle, route_matrix: RouteMatrix) -> TasteEvidencePack: ...
def validate_suggested_commitments(suggestions: tuple[SuggestedCommitmentArtifact, ...], evidence_pack: TasteEvidencePack) -> CommitmentValidationReport: ...
def compile_confirmed_commitments(decisions: tuple[UserCommitmentDecision, ...]) -> tuple[OwnedConstraint, ...]: ...
def parse_user_commitment_correction(user_text: str, candidates: tuple[dict[str, Any], ...]) -> tuple[UserCommitmentDecision, ...]: ...
```

## Validation Rules

1. Prompt output is strict JSON.
2. Target IDs exist.
3. Evidence refs exist.
4. No invented places, routes, hotels, weather, prices, bookings, or closures.
5. No sensitive personal inferences.
6. Inferred taste cannot create `LOCKED` or `BOOKED`.
7. `LOCKED` requires explicit user lock evidence.
8. `BOOKED` requires booking evidence.
9. Suggested commitments are inactive.
10. Only confirmed choices become `OwnedConstraint`.
11. Every explanation claim maps to evidence.

## UI / API Integration

Map behavior:

- suggested badge
- confirmed badge
- rejected badge
- evidence panel
- tradeoff note
- accept/reject/downgrade/upgrade actions

## Workflow

1. Build evidence pack.
2. Extract taste signals.
3. Compress taste profile.
4. Generate suggested commitments.
5. Validate deterministically.
6. Optionally run LLM critic.
7. Generate clarifying questions.
8. Show map cards.
9. Parse user corrections.
10. Compile confirmed commitments.

## Files to Create

- `src/itinerary_system/llm_taste/__init__.py`
- `src/itinerary_system/llm_taste/models.py`
- `src/itinerary_system/llm_taste/evidence_pack.py`
- `src/itinerary_system/llm_taste/prompts.py`
- `src/itinerary_system/llm_taste/validator.py`
- `src/itinerary_system/llm_taste/map_cards.py`
- `tests/llm_taste/test_prompt_schema.py`
- `tests/llm_taste/test_commitment_validation.py`

## Files to Modify

- `src/itinerary_system/prototype/controller.py` after prototype exists.
- `src/itinerary_system/repair/requests.py`
- `src/itinerary_system/plans/models.py` only if enum aliases need public normalization.

## Error Handling

- Provider disabled: use frozen fixture or mark unavailable.
- Invalid JSON: reject prompt output.
- Unsupported booking/lock claim: reject or downgrade.
- Ambiguous user correction: ask clarification.
- Stale parent hash: block confirmation.

## Testing Checklist

- [ ] Prompt schema validation rejects invalid JSON.
- [ ] Unknown target ID rejected.
- [ ] Missing evidence rejected.
- [ ] Inferred `LOCKED` downgraded or rejected.
- [ ] Inferred `BOOKED` rejected without booking evidence.
- [ ] User confirmation compiles to active constraint.
- [ ] Unconfirmed suggestion remains inactive.
- [ ] Map card schema validates.
- [ ] `python -m ruff check src tests scripts`
- [ ] `python -m pytest`

## Roadmap / Full Pipeline Update

Add `LLM-TASTE-001` after implementation. Do not start this before G1, G3, G5, and G7 are stable enough for artifact replay.

## Acceptance Criteria

1. LLM acts as bounded mixed-initiative assistant, not planner.
2. Taste hypotheses are inspectable and editable.
3. Only confirmed constraints enter solver.
4. Prompt runs are versioned and hash-linked.

# Phase G9.0 Implementation Plan: Explanation, Counterfactual, And Study Readiness

## Goal

Implement **explanation, counterfactual, and study readiness**.

This gate should allow the system to:

1. Ground every explanation in artifacts.
2. Support why, why-not, what-if, and contrastive explanations.
3. Prepare redacted event logging for later IUI/CHI studies.

## Current State

Missing:

- `src/itinerary_system/explanation/`.
- General `ExplanationEvidence`, `WhyEvidence`, `WhyNotEvidence`, `WhatIfEvidence`, `ContrastiveEvidence`, `ExplanationClaim`.
- Counterfactual runner.
- Study event logger.

Partial:

- `repair_planner.py` has a simple `CounterfactualExplanation`.
- G5 progressive attempts and diagnosis can become explanation evidence.

## New Components to Add

### Component 1

`ExplanationEvidenceBuilder`

Responsibility:

Build evidence records from diff, constraints, routes, context, solver stages, evaluator metrics, and counterfactual runs.

### Component 2

`CounterfactualRunner`

Responsibility:

Run sandbox repair requests for why-not and what-if questions.

### Component 3

`DeterministicTemplateVerbalizer`

Responsibility:

Generate stable template explanations.

### Component 4

`StudyEventLogger`

Responsibility:

Log redacted, artifact-linked participant/prototype events.

## Class / Registry Diagrams

```text
+-------------------------------------------------------------------------------+
|                    ExplanationEvidenceBuilder (Service Class)                  |
+-------------------------------------------------------------------------------+
|  - artifact_index: ArtifactIndex                                               |
+-------------------------------------------------------------------------------+
|  + build_why(...): WhyEvidence             --> Why selected/kept               |
|  + build_why_not(...): WhyNotEvidence      --> Why omitted/failed              |
|  + build_what_if(...): WhatIfEvidence      --> Counterfactual result           |
|  + build_contrastive(...): ContrastiveEvidence --> Parent-child difference     |
+-------------------------------------------------------------------------------+
```

```text
+-------------------------------------------------------------------------------+
|                         CounterfactualRunner (Service Class)                   |
+-------------------------------------------------------------------------------+
|  - repair_controller: ProgressiveRepairController                              |
|  - evaluator: PlanEvaluator                                                    |
+-------------------------------------------------------------------------------+
|  + run(request): WhatIfEvidence           --> Sandbox counterfactual evidence  |
+-------------------------------------------------------------------------------+
```

```text
+-------------------------------------------------------------------------------+
|                 DeterministicTemplateVerbalizer (Stateless Utility Module)     |
+-------------------------------------------------------------------------------+
|  - No persistent internal state                                                |
+-------------------------------------------------------------------------------+
|  + verbalize(evidence): tuple[ExplanationClaim, ...] --> Stable claims         |
|  + validate_claims(claims): tuple[Finding, ...] --> Evidence mapping           |
+-------------------------------------------------------------------------------+
```

```text
+-------------------------------------------------------------------------------+
|                   StudyEventLogger (Persistent Registry / Manager)             |
+-------------------------------------------------------------------------------+
|  - log_path: Path                                                              |
|  - redaction_policy: RedactionPolicy                                           |
+-------------------------------------------------------------------------------+
|  + append(event): None                    --> Redacted event write             |
|  + validate(event): tuple[Finding, ...]   --> Required event schema            |
+-------------------------------------------------------------------------------+
```

## Class Diagram Rules

1. Prose is not evidence.
2. Every numerical or causal claim maps to artifact refs.
3. Optional LLM verbalizer receives structured evidence only.
4. Unsupported claims fail closed.
5. No raw personal itinerary or raw LLM conversation is logged by default.

## Data Model

```python
@dataclass(frozen=True)
class ExplanationClaim:
    claim_id: str
    claim_type: str
    text_template: str
    evidence_refs: tuple[str, ...]
    unsupported: bool = False

@dataclass(frozen=True)
class ExplanationEvidence:
    evidence_id: str
    evidence_type: str
    plan_id: str
    claims: tuple[ExplanationClaim, ...]
```

Counterfactuals:

- force omitted stop
- relax constraint
- add travel time
- change weather tolerance
- preserve route segment
- allow hotel switch

Study event fields:

- participant hash
- condition
- task
- event type
- target
- selected strength
- plan ID
- repair request ID
- timestamp
- accepted/rejected

## Storage / State

Persistent:

- `runs/<run_id>/explanations/*.json`
- `runs/<run_id>/study/events.jsonl`

Temporary:

- Sandbox counterfactual candidate models.

## Required Methods

```python
def build_why_evidence(plan: PlanArtifactV2, evaluation: EvaluationCertificate) -> tuple[WhyEvidence, ...]: ...
def build_why_not_evidence(request: RepairRequest, attempts: tuple[RepairAttempt, ...]) -> tuple[WhyNotEvidence, ...]: ...
def run_counterfactual(parent: PlanArtifactV2, request: RepairRequest) -> WhatIfEvidence: ...
def validate_explanation_claims(claims: tuple[ExplanationClaim, ...], artifact_index: ArtifactIndex) -> tuple[Finding, ...]: ...
```

## Validation Rules

1. Every claim has at least one evidence ref.
2. Numerical claims cite metric/route/diff/stage evidence.
3. Causal claims cite disruption/constraint/diff/counterfactual evidence.
4. Counterfactual answers require a run or validation failure.
5. Study events are redacted.

## UI / API Integration

Prototype displays:

- certificate badge
- diff panel
- route/context evidence
- why
- why not
- what if
- what changed
- why suggested for LLM commitment cards

## Workflow

1. Build artifact index.
2. Generate evidence records.
3. Run counterfactuals when requested.
4. Validate claims.
5. Render deterministic prose.
6. Store explanations.
7. Log events in study mode.

## Files to Create

- `src/itinerary_system/explanation/__init__.py`
- `src/itinerary_system/explanation/evidence.py`
- `src/itinerary_system/explanation/counterfactual.py`
- `src/itinerary_system/explanation/verbalizer.py`
- `src/itinerary_system/explanation/validation.py`
- `src/itinerary_system/study/events.py`
- `src/itinerary_system/study/logger.py`
- `tests/explanation/test_evidence.py`
- `tests/explanation/test_counterfactual.py`
- `tests/study/test_event_redaction.py`

## Files to Modify

- `src/itinerary_system/repair/progressive.py`
- `src/itinerary_system/pipeline_runner.py`
- `src/itinerary_system/llm_taste/models.py` after G8.

## Error Handling

- Missing evidence ref: reject claim.
- LLM unreferenced claim: reject verbalization.
- Counterfactual infeasible: store why-not evidence.
- Raw sensitive log content: reject event.

## Testing Checklist

- [ ] Claim without evidence fails.
- [ ] Why-not evidence cites failed attempts.
- [ ] What-if evidence cites counterfactual run.
- [ ] LLM verbalizer rejects unmapped claims.
- [ ] Study event redaction works.
- [ ] `python -m ruff check src tests scripts`
- [ ] `python -m pytest`

## Roadmap / Full Pipeline Update

Do not claim user trust, understanding, or decision quality until G11 study evidence exists.

## Acceptance Criteria

1. Explanations are artifact-grounded.
2. Unsupported claims fail closed.
3. Study logs are redacted and replayable.

# Phase G10.0 Implementation Plan: Live/Event-Triggered Repair Extension

## Goal

Implement **event-triggered ownership-aware itinerary repair**.

This later gate should allow the system to:

1. Detect context deltas.
2. Decide whether repair preview is warranted.
3. Create eligible repair proposals without mutating accepted plans.
4. Require user authorization before adoption.

## Current State

Missing:

- `src/itinerary_system/live/`.
- `ContextDelta`, `LiveEvent`, `ImpactReport`, `TriggerDecision`, `LiveRepairRequest`, `RepairProposal`, `NotificationRecord`, `UserAdoptionDecision`, `LiveRepairLineage`.
- Frozen replay stream for live events.

## New Components to Add

### Component 1

`ContextDeltaAnalyzer`

Responsibility:

Compare context snapshots and produce impact reports for accepted parent plans.

### Component 2

`TriggerPolicyEngine`

Responsibility:

Map impact reports to trigger decisions.

### Component 3

`LiveRepairProposalService`

Responsibility:

Create repair previews, evaluate them, and store proposal lineage.

## Class / Registry Diagrams

```text
+-------------------------------------------------------------------------------+
|                         ContextDeltaAnalyzer (Service Class)                   |
+-------------------------------------------------------------------------------+
|  - No persistent internal state                                                |
+-------------------------------------------------------------------------------+
|  + analyze(parent, delta): ImpactReport  --> Affected days/targets             |
+-------------------------------------------------------------------------------+
```

```text
+-------------------------------------------------------------------------------+
|                         TriggerPolicyEngine (Service Class)                    |
+-------------------------------------------------------------------------------+
|  - policy: TriggerPolicy                                                       |
+-------------------------------------------------------------------------------+
|  + decide(impact): TriggerDecision       --> No action/warning/preview         |
+-------------------------------------------------------------------------------+
```

```text
+-------------------------------------------------------------------------------+
|                       LiveRepairProposalService (Service Class)                |
+-------------------------------------------------------------------------------+
|  - repair_controller: ProgressiveRepairController                              |
|  - evaluator: PlanEvaluator                                                    |
+-------------------------------------------------------------------------------+
|  + propose(request): RepairProposal      --> Eligible or rejected proposal     |
|  + adopt(decision): PlanArtifactV2 | None --> User-authorized child            |
+-------------------------------------------------------------------------------+
```

## Class Diagram Rules

1. Live repair never mutates accepted plan in place.
2. Unsupported source leads to `DEFER_UNSUPPORTED`.
3. Proposed child requires independent evaluation before display as valid alternative.
4. User adoption is explicit.

## Data Model

Trigger decisions:

```text
NO_ACTION
PASSIVE_BADGE
WARNING
REPAIR_PREVIEW
CONFIRMATION_REQUIRED
DEFER_UNSUPPORTED
```

Replay stream:

```text
t0 parent accepted
t1 weather worsens
t2 road closure appears
t3 user reports fatigue
t4 attraction closure appears
t5 weather improves
```

## Storage / State

Persistent:

- `runs/<run_id>/live/events.jsonl`
- `runs/<run_id>/live/impact_reports/`
- `runs/<run_id>/live/proposals/`
- `runs/<run_id>/live/adoption_decisions.jsonl`

Temporary:

- Context watcher state.

## Required Methods

```python
def analyze_context_delta(parent: PlanArtifactV2, delta: ContextDelta) -> ImpactReport: ...
def decide_live_trigger(impact: ImpactReport, policy: TriggerPolicy) -> TriggerDecision: ...
def create_live_repair_request(parent: PlanArtifactV2, event: LiveEvent, decision: TriggerDecision) -> LiveRepairRequest: ...
def propose_live_repair(request: LiveRepairRequest) -> RepairProposal: ...
def adopt_repair_proposal(decision: UserAdoptionDecision) -> PlanArtifactV2 | None: ...
```

## Validation Rules

1. Context deltas are versioned.
2. Stale evidence is labeled.
3. No plan mutates in place.
4. Unsupported live source defers.
5. User authorization required before adoption.

## UI / API Integration

Use passive badges, warnings, repair previews, and confirmation-required cards. Do not claim production live monitoring until validated.

## Workflow

1. Receive context update.
2. Create `ContextDelta`.
3. Analyze impact.
4. Decide trigger.
5. Create repair preview if warranted.
6. Evaluate proposal.
7. Ask user to adopt/reject.
8. Save lineage.

## Files to Create

- `src/itinerary_system/live/__init__.py`
- `src/itinerary_system/live/models.py`
- `src/itinerary_system/live/delta.py`
- `src/itinerary_system/live/triggers.py`
- `src/itinerary_system/live/proposals.py`
- `tests/live/test_delta.py`
- `tests/live/test_triggers.py`
- `tests/live/test_proposals.py`

## Files to Modify

- `src/itinerary_system/pipeline_runner.py`
- `src/itinerary_system/prototype/controller.py`
- `docs/study/analysis_plan.md` after G11.

## Error Handling

- Unsupported source: `DEFER_UNSUPPORTED`.
- Stale evidence: warning or defer.
- Ineligible proposal: show diagnostic, not valid alternative.
- User rejects: preserve parent.

## Testing Checklist

- [ ] Frozen context replay produces expected triggers.
- [ ] Unsupported live source defers.
- [ ] Ineligible repair proposal is not displayed as valid.
- [ ] Adoption creates child lineage.
- [ ] Rejection preserves parent.
- [ ] `python -m ruff check src tests scripts`
- [ ] `python -m pytest`

## Roadmap / Full Pipeline Update

This is a late extension. Do not start until G3, G5, G7, and G9 are stable.

## Acceptance Criteria

1. Event-triggered repair works on frozen replay streams.
2. Proposals are independently evaluated.
3. Adoption is explicit and lineaged.

# Phase G11.0 Implementation Plan: IUI/CHI Study And Venue Readiness

## Goal

Implement **IUI/CHI study and venue readiness**.

This gate should allow the project team to:

1. Decide whether evidence supports TRB, IUI, or CHI.
2. Prepare controlled tasks, conditions, event logs, and analysis plan.
3. Keep venue claims aligned with completed evidence.

## Current State

Missing:

- `docs/study/analysis_plan.md`.
- Study task corpus.
- Condition assignment.
- Venue readiness matrix.
- Ethics-ready participant protocol.
- User-outcome evidence.

## New Components to Add

### Component 1

`VenueReadinessMatrix`

Responsibility:

Map completed evidence to TRB/IUI/CHI readiness.

### Component 2

`StudyTaskCorpus`

Responsibility:

Package controlled repair tasks from benchmark scenarios and artifact replay bundles.

### Component 3

`PreferenceCommitmentStudyMetrics`

Responsibility:

Compute LLM taste agreement, suggestion acceptance, false suggestion, preservation gain, repair cost, and user-control metrics.

## Class / Registry Diagrams

```text
+-------------------------------------------------------------------------------+
|                         VenueReadinessMatrix                                   |
+-------------------------------------------------------------------------------+
|  - venue_rows: tuple[VenueReadinessRow, ...]                                   |
+-------------------------------------------------------------------------------+
|  + evaluate(evidence): VenueRecommendation --> TRB/IUI/CHI readiness           |
+-------------------------------------------------------------------------------+
```

```text
+-------------------------------------------------------------------------------+
|                         StudyTaskCorpus (Persistent Registry / Manager)        |
+-------------------------------------------------------------------------------+
|  - tasks_path: Path                                                            |
+-------------------------------------------------------------------------------+
|  + load(): tuple[StudyTask, ...]          --> Controlled tasks                 |
|  + validate_against_runs(): tuple[Finding, ...] --> Artifact consistency       |
+-------------------------------------------------------------------------------+
```

```text
+-------------------------------------------------------------------------------+
|             PreferenceCommitmentStudyMetrics (Stateless Utility Module)        |
+-------------------------------------------------------------------------------+
|  - No persistent internal state                                                |
+-------------------------------------------------------------------------------+
|  + compute(events, artifacts): tuple[StudyMetric, ...] --> LLM interaction     |
+-------------------------------------------------------------------------------+
```

## Class Diagram Rules

1. TRB path needs computational benchmark and road-valid evidence.
2. IUI path needs parser/taste/commitment interface evidence.
3. CHI path needs ethics-approved participant evidence.
4. Deadlines are marked `REQUIRES_REVERIFICATION`.
5. No venue is chosen until evidence is complete.

## Data Model

```python
@dataclass(frozen=True)
class VenueReadinessRow:
    venue: str
    required_evidence: tuple[str, ...]
    current_status: str
    blockers: tuple[str, ...]
    deadline_status: str

@dataclass(frozen=True)
class StudyTask:
    task_id: str
    disruption_family: str
    parent_plan_id: str
    child_plan_id: str
    condition_ids: tuple[str, ...]
    expected_evidence_refs: tuple[str, ...]
```

Metrics:

- TasteProfileAgreement
- TasteProfileCorrectionRate
- SuggestedCommitmentAcceptanceRate
- SuggestedCommitmentRejectionRate
- DowngradeRate
- UpgradeRate
- FalseLockSuggestionRate
- OverStrongSuggestionRate
- EvidenceCoverage
- UnsupportedTasteClaimRate
- ClarificationUsefulness
- PreservationGainFromConfirmedSuggestions
- RepairCostOfConfirmedCommitments
- UserControlRating

## Storage / State

Persistent:

- `docs/study/analysis_plan.md`
- `data/study/tasks/*.json`
- `runs/<run_id>/study/replay_bundles/*.json`
- `runs/<run_id>/study/events.jsonl`

Temporary:

- UI session state; replayable from events.

## Required Methods

```python
def build_study_task_corpus(run_ids: tuple[str, ...]) -> tuple[StudyTask, ...]: ...
def assign_study_condition(participant_id_hash: str, seed: int) -> StudyConditionConfig: ...
def validate_venue_readiness(evidence_index: EvidenceIndex) -> VenueReadinessMatrix: ...
def compute_preference_commitment_metrics(events: tuple[StudyEvent, ...]) -> tuple[StudyMetric, ...]: ...
```

## Validation Rules

1. No task uses ineligible child plan.
2. Baseline and treatment use identical backend artifacts where required.
3. Event logs hash participant IDs.
4. No raw personal itinerary or raw LLM conversation logged by default.
5. Outcome claims wait for analyzed study data.
6. Deadlines require re-verification.

## UI / API Integration

- Baseline: itinerary/map/metrics display.
- Treatment: diff/certificate/evidence/explanation/correction controls.
- LLM condition: taste cards and inactive commitment suggestions.

## Workflow

1. Select stable runs.
2. Build replay bundles.
3. Define conditions.
4. Validate UI claims and logs.
5. Export analysis plan.
6. Pilot internally.
7. Human team handles ethics and participant deployment.

## Files to Create

- `docs/study/analysis_plan.md`
- `src/itinerary_system/study/tasks.py`
- `src/itinerary_system/study/conditions.py`
- `src/itinerary_system/study/readiness.py`
- `src/itinerary_system/study/preference_metrics.py`
- `tests/study/test_tasks.py`
- `tests/study/test_readiness.py`

## Files to Modify

- `docs/user_study_protocol.md` if present or create in G9.
- `src/itinerary_system/prototype/controller.py`
- `README.md` only after prototype is stable.

## Error Handling

- Missing artifact: readiness failure.
- Unsupported UI claim: readiness failure.
- Direct LLM baseline missing frozen output/evaluation: mark unavailable.
- Ethics missing: human-action blocker.

## Testing Checklist

- [ ] Study tasks replay from artifacts.
- [ ] Baseline/treatment artifact parity holds.
- [ ] Event redaction works.
- [ ] LLM metrics are computed from explicit decisions.
- [ ] Venue readiness matrix marks incomplete evidence.
- [ ] `python -m ruff check src tests scripts`
- [ ] `python -m pytest`

## Roadmap / Full Pipeline Update

Do not claim TRB/IUI/CHI readiness until evidence exists. Mark deadlines `REQUIRES_REVERIFICATION`.

## Acceptance Criteria

1. Venue path is evidence-driven.
2. Study tasks are artifact-backed.
3. User-outcome claims remain unmade until data exists.

## 7. LLM Integration Plan: Evidence-Bounded Preference-to-Commitment

Protocol name:

# Evidence-Bounded Preference-to-Commitment Prompting

Definition:

> A domain-specific LLM prompting protocol that extracts taste hypotheses from a parent itinerary, proposes inactive protection labels for existing itinerary elements, validates those suggestions against evidence, and activates them only after user confirmation for downstream ownership-aware repair.

Allowed LLM roles:

- Request interpreter.
- Taste hypothesis generator.
- Commitment recommender.
- Clarification assistant.
- Evidence verbalizer.
- Counterfactual interface.

Prohibited LLM roles:

- Final itinerary generator for publication comparison.
- Optimizer.
- Feasibility judge.
- Road-validity judge.
- Hotel-availability judge.
- Evaluator/certifier.
- Solver coefficient author.
- Source of unsupported explanations.

Prompt Call 1 - Taste Signal Extraction:

Input:

```text
TasteEvidencePack
```

Output:

```json
{
  "taste_signals": [
    {
      "dimension": "scenic_photo | nature_outdoor | architecture_history | museum_culture | food_local_experience | social_must_go | relaxed_pace | dense_explorer_pace | low_hotel_switching | scenic_driving | low_driving_burden | weather_cautious | budget_sensitive",
      "level": "low | medium | high | uncertain",
      "confidence": 0.0,
      "evidence_refs": ["string"],
      "reason": "string",
      "uncertainty": "string | null"
    }
  ],
  "unsupported_inferences": []
}
```

Prompt Call 2 - Taste Profile Compression:

```json
{
  "taste_profile": [
    {
      "profile_item_id": "string",
      "dimension": "string",
      "level": "low | medium | high | uncertain",
      "confidence": 0.0,
      "display_label": "string",
      "display_summary": "string",
      "evidence_refs": ["string"],
      "status": "hypothesis_not_confirmed"
    }
  ],
  "profile_uncertainties": []
}
```

Prompt Call 3 - Suggested Commitment Generation:

```json
{
  "suggested_commitments": [
    {
      "suggestion_id": "string",
      "target_id": "string",
      "target_type": "poi | road_segment | lodging | day_structure | time_window",
      "suggested_strength": "LOCKED | BOOKED | STRONG | WEAK | TEST_ONLY",
      "confidence": 0.0,
      "map_badge": "suggested_lock | suggested_strong | suggested_weak | needs_confirmation",
      "short_reason": "string",
      "detailed_reason": "string",
      "evidence_refs": ["string"],
      "taste_profile_refs": ["string"],
      "possible_repair_tradeoff": "string",
      "requires_confirmation": true,
      "default_user_action": "ask_user"
    }
  ],
  "not_suggested": []
}
```

Prompt Call 4 - Optional Critic:

Checks:

- invented entity
- missing evidence
- over-strong lock
- unsupported booking claim
- unsupported route/weather/hotel claim
- sensitive personal inference
- missing confirmation
- vague map reason

Prompt Call 5 - Clarifying Question Generator:

Ask at most two questions, only if the answer materially changes repair constraints.

Prompt Call 6 - Map Card Verbalizer:

Create concise map badges and cards with evidence refs and repair tradeoff notes.

Prompt Call 7 - User Correction Parser:

Parse user corrections into candidate commitment updates. Do not activate directly.

Evaluation metrics:

- TasteProfileAgreement
- TasteProfileCorrectionRate
- SuggestedCommitmentAcceptanceRate
- SuggestedCommitmentRejectionRate
- DowngradeRate
- UpgradeRate
- FalseLockSuggestionRate
- OverStrongSuggestionRate
- EvidenceCoverage
- UnsupportedTasteClaimRate
- ClarificationUsefulness
- PreservationGainFromConfirmedSuggestions
- RepairCostOfConfirmedCommitments
- UserControlRating

## 8. Live/Event-Triggered Repair Plan

Frame:

> Event-triggered ownership-aware itinerary repair.

This is a later extension. It should be tested first on frozen replay streams, not uncontrolled live APIs.

Architecture:

```text
Context watcher
    |
    v
ContextDelta
    |
    v
Impact analyzer
    |
    v
Trigger policy
    |
    v
Repair preview
    |
    v
Independent evaluator
    |
    v
User authorization
    |
    v
accepted child plan or rejected proposal
```

Metrics:

- DetectionLatency
- RepairLatency
- NotificationPrecision
- UnnecessaryInterruptionRate
- EligibleRepairRate
- SmallestRadiusSuccessRate
- WeightedEditCost
- CommitmentPreservation
- ChurnRate
- RepairRegret
- UserAdoptionRate
- CounterfactualUseRate
- StaleEvidenceRate
- FallbackRate

Do not claim:

- production live monitoring
- safety routing
- booking or cancellation
- real-time hotel inventory
- guaranteed real-world availability

## 9. Benchmark And Evaluation Plan

Benchmark must use:

- canonical parent plans
- frozen disruptions
- frozen ownership profiles
- frozen route/context evidence
- independent certificates
- split-by-family logic

Primary comparison:

1. full reoptimization
2. current hybrid pipeline
3. weighted-sum repair
4. fixed-radius repair
5. progressive sequential lexicographic repair
6. LLM-only rewrite only if frozen and independently evaluated

Evaluator gates:

- artifact/source/run linkage
- content hash
- lineage
- owned constraints
- route connectivity
- road validation
- daily time feasibility
- opening windows
- lodging continuity
- budget
- weather/closure feasibility
- duplicate visits
- post-solve mutation

Outputs:

- `BenchmarkScenario`
- `BenchmarkResult`
- `EvaluationCertificate`
- `PlanDiff`
- `ExplanationEvidence`
- metrics CSV/JSON
- dashboard summary

## 10. UI, Dashboard, And Study Readiness Plan

Prototype views:

- parent plan
- child plan
- diff view
- ownership badges
- route evidence
- certificate badge
- explanation panel
- LLM taste/commitment cards
- counterfactual controls
- live repair preview cards

Rules:

1. UI consumes artifacts only.
2. UI cannot mutate canonical plan artifacts.
3. UI changes become typed requests or user decisions.
4. UI labels route fallback clearly.
5. UI labels LLM suggestions as inactive until confirmed.
6. UI logs study events with hashed participant IDs.

Study conditions:

- baseline map/itinerary view
- artifact/evidence/certificate view
- LLM taste profile only
- LLM taste profile plus suggested commitments
- direct LLM rewrite baseline only if frozen/evaluated

## 11. Documentation And Roadmap Update Policy

This document is a planning deliverable. It does not complete implementation gates. No checklist item should be marked complete unless code, tests, and required evidence already pass.

Rules:

1. Do not rewrite unrelated roadmap sections.
2. Do not mark `[ ]` as `[x]` for planned work.
3. Update `CODEX_EDIT_LOG.md` after file changes.
4. Keep human reflection separate from required human action.
5. Record tests run and tests not run with reasons.
6. Prefer new integrated plan as the forward-looking roadmap, while preserving older plans as supporting references.

## 12. Definition Of Done

The full project is done only when:

- A clean clone can load catalog and context snapshots.
- Current problem manifest is up to date.
- Phase 0 strict validation passes.
- Parent plans are immutable.
- Child plans have parent IDs and content hashes.
- Owned constraints are explicit and validated.
- Route matrices are road-valid for publication scenarios.
- Geodesic fallback cannot pass publication gates.
- Independent evaluator issues certificates.
- Single-day and progressive repairs are evaluator-certified.
- Benchmark scenarios cover six disruptions and five ownership profiles.
- Split leakage is blocked.
- Pipeline runner emits immutable `runs/<run_id>/`.
- Notebook is display-only.
- LLM suggestions are inactive until validated and confirmed.
- Explanation claims map to evidence.
- Live repair uses frozen replay validation before any live claim.
- Study logs are redacted and artifact-linked.
- Global checks pass:

```bash
python -m ruff check src tests scripts
python -m pytest
```

- Publication and venue claims match completed evidence.

## 13. Immediate Next Codex Tasks

Recommended next implementation order:

1. Re-run repository inspection and validation harness.
2. Confirm `docs/current/current_problem_manifest.md` reflects the current code after recent repair work.
3. Close G0 blockers and keep Phase 0 strict validation honest.
4. Finish/verify `PlanArtifactV2`, `PlanRepository`, `OwnedConstraint`, and `PlanDiff` integration into canonical parent artifacts.
5. Verify road-valid `RouteMatrix` integration for the first benchmark contexts.
6. Implement independent `PlanEvaluator` and `EvaluationCertificate`.
7. Wire the evaluator into single-day repair and progressive repair.
8. Only then harden benchmark scenarios.
9. Then implement `run_research_pipeline()` and immutable run directories.
10. Defer LLM taste-to-commitment until canonical artifacts and UI/pipeline outputs exist.
11. Defer live/event-triggered repair until benchmark and pipeline gates are stable.
