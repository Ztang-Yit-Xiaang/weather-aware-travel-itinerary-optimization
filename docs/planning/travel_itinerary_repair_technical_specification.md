# Travel Itinerary Repair Technical Specification for Codex

**Status:** Draft implementation contract  
**Repository:** `Ztang-Yit-Xiaang/weather-aware-travel-itinerary-optimization`  
**Author:** Yit Xiaang Ztang  
**Primary research framing:** IUI-first, CHI-compatible  
**Implementation target:** Extend the current `main` branch and Phase 0 foundation; do not create a parallel system.

---

## 0. How Codex must use this specification

This file is an implementation contract, not a prompt to rewrite the entire repository. Complete work packages in dependency order. For every package:

1. inspect the current files named in **Current repository evidence**;
2. make the smallest coherent change that satisfies the public interface and invariants;
3. do not modify files listed as out of scope;
4. add or update focused tests;
5. run the required checks;
6. write generated evidence only under the specified run or test directory;
7. update `CODEX_EDIT_LOG.md` with the task ID, files changed, tests run, and unresolved limitations.

Do not claim a task is complete merely because a class or placeholder file exists. Acceptance criteria and tests define completion.

### 0.1 Global coding rules

- Python 3.12.
- Type hints on all public functions, methods, and dataclass fields.
- Prefer `@dataclass(frozen=True)` for immutable research records.
- Use `Enum`/`StrEnum` for closed vocabularies. Do not spread unvalidated strings across modules.
- Do not make API calls inside optimization model builders.
- Do not put business logic in notebooks.
- Do not use hidden mutable global state.
- Inject provider, routing, repository, clock, and random-number dependencies.
- Parent plans are immutable. A change creates a child plan with a new ID and content hash.
- Do not serialize secrets in resolved config, logs, cache manifests, dashboards, or exception messages.
- Missing credentials must disable a provider before any normal data request is sent.
- Geodesic or straight-line fallback is never `road_validated=True`.
- A plan modified after solve is not solver-certified until a new independent evaluation.
- Use deterministic seeds and stable content hashes.
- Functions should normally have one responsibility and readable names. Avoid functions longer than roughly 60 lines unless a model-construction block genuinely benefits from locality.
- Public docstrings must state inputs, outputs, assumptions, side effects, and failure behavior.
- Use structured exceptions. Do not hide errors behind an empty DataFrame unless the caller explicitly requests permissive mode.
- Keep compatibility wrappers temporarily, mark them as deprecated, and add tests proving equivalence.

### 0.2 Required checks after every task

```bash
python -m ruff check src tests scripts
python -m pytest
```

Run focused tests first during development, then the full suite before marking a task complete.

---

## 1. Current repository truth

The June 30, 2026 `main` branch already contains a Phase 0 research foundation. Do not re-create these components under different names.

### 1.1 Already implemented or partially implemented

- `data/snapshots/california_v1/`
- `src/itinerary_system/data/schemas.py`
  - `DatasetBundle`
  - `DatasetValidationReport`
- `src/itinerary_system/data/snapshot.py`
  - `load_dataset_bundle()`
  - `validate_dataset_bundle()`
- `src/itinerary_system/routing/models.py`
  - `RouteLegResult`
  - `RouteResult`
- `src/itinerary_system/routing/cache.py`
  - `RoadRouteCache`
- `src/itinerary_system/research_artifacts.py`
  - `PlannerRun`
  - `PlanArtifact` v1
  - `ResearchEvaluationReport`
  - `evaluate_phase0_plan()`
- Phase 0 exporter, route-cache builder, route-source checks, readiness summary, and strict validation scripts.
- Config keys for `catalog_snapshot_id`, `context_snapshot_id`, `refresh_policy`, and `run_id`.

### 1.2 Not yet implemented as a complete research system

- parent-plan-aware repair optimization;
- typed constraint ownership;
- confirmed edit semantics;
- progressive repair neighborhoods;
- sequential lexicographic solve stages;
- generalized plan diff;
- final-plan evaluator covering all hard constraints;
- counterfactual explanation evidence;
- authoritative package pipeline runner;
- provider-independent lodging taxonomy and provider protocol;
- local OSRM Docker assets and a unified OSRM client;
- interactive service endpoint and durable user edits.

### 1.3 Important current-code issues

1. `multi_objective_route.py` solves a candidate selection/routing problem but has no parent plan or explicit change variables.
2. It calculates geodesic proxy travel internally. This must be replaced by an injected route matrix for publication comparisons.
3. `route_gurobi_oracle.py` returns from `solve_enriched_route_with_gurobi()` before a large legacy implementation block. That block is unreachable and should be removed after regression tests.
4. `experiment_runner.py` remains notebook-oriented and reads/writes flat production files.
5. `map_renderer.py` must sanitize stale notebook state. The target architecture should eliminate this reliance on mutable notebook dictionaries.
6. `data_enrichment.py` now exposes `source_coverage_score` but still retains legacy `data_confidence` aliases and writes one wide canonical CSV.
7. `utility_model.py` still treats absent Yelp as numeric zero in several paths. Missing-source masks are required.

---

## 2. Target architecture

```text
stable catalog snapshot
        +
time-sensitive context snapshot
        +
immutable parent plan
        +
confirmed edit/disruption
        ↓
repair master
        ↓
affected-day route subproblems
        ↓
road-valid routing records
        ↓
independent final-plan evaluator
        ↓
child plan + plan diff + certificate + explanation evidence
```

### 2.1 Mandatory boundaries

- **Catalog snapshot:** entity identity, source observations, curated annotations, feature definitions.
- **Context snapshot:** weather valid time, hotel rate query, closures, route matrix/geometry.
- **Run artifact:** config, random seed, parent plan, request, solver stages, child plan, diff, evaluation.
- **No live API request** during a frozen benchmark run.
- **No optimizer view** is treated as a source-of-truth table; it is a materialized read-only join.

---

## 3. Mathematical contract and math-to-code mapping

### 3.1 Sets and parameters

| Math | Code |
|---|---|
| `D` days | `tuple[DayId, ...]` |
| `I` POIs | `tuple[PoiId, ...]` |
| `H` lodging entities | `tuple[LodgingId, ...]` |
| `A_d` day arcs | `RouteMatrix` and day candidate graph |
| `x0[i,d]` parent assignment | `ParentPlanIndex.poi_day` |
| `z0[h,d]` parent lodging | `ParentPlanIndex.lodging_by_night` |
| `q0[i,j,d]` parent road arc | `ParentPlanIndex.route_arcs` |
| `F[i,d,omega]` disruption feasibility | `DisruptionFeasibilityIndex` |
| `rho` ownership weight | `OwnershipPolicy.change_cost(...)` |
| `tau[i,j]` road-valid time | `RouteMatrix.duration_s` |

### 3.2 Decision variables

| Variable | Meaning | Builder module |
|---|---|---|
| `x[i,d]` | assign POI to day | `repair/master_model.py` |
| `v[i]` | include POI | `repair/master_model.py` |
| `z[h,d]` | lodging after day | `repair/master_model.py` |
| `delta_minus[i]` | delete parent POI | `repair/change_variables.py` |
| `delta_plus[i]` | add POI | `repair/change_variables.py` |
| `move[i,d]` | move POI to different day | `repair/change_variables.py` |
| `hotel_change[h,d]` | change lodging | `repair/change_variables.py` |
| `road_change[i,j,d]` | change protected route arc/corridor | master or route option layer |
| `relax[c]` | relax owned constraint | `repair/relaxation.py` |
| `y[i,j,d]` | within-day route arc | `repair/day_route_solver.py` |
| `T[i,d]` | arrival time | `repair/day_route_solver.py` |

### 3.3 Sequential objective stages

```text
Stage 0: minimize hard/booked relaxations
Stage 1: minimize ownership-weighted typed plan changes
Stage 2: minimize weather/closure risk, road-valid travel, hotel switches, cost
Stage 3: maximize utility, diversity, and preliminary scenic fit
```

After every stage, add a constraint fixing that objective to its optimum within the configured tolerance before optimizing the next stage.

```python
def solve_lexicographically(model: RepairModel, tolerances: ObjectiveTolerances) -> LexicographicResult:
    stage0 = model.solve(ObjectiveStage.HARD_RELAXATION)
    model.fix_stage(ObjectiveStage.HARD_RELAXATION, stage0.value, tolerances.hard)

    stage1 = model.solve(ObjectiveStage.PLAN_CHANGE)
    model.fix_stage(ObjectiveStage.PLAN_CHANGE, stage1.value, tolerances.change)

    stage2 = model.solve(ObjectiveStage.CONTEXTUAL_BURDEN)
    model.fix_stage(ObjectiveStage.CONTEXTUAL_BURDEN, stage2.value, tolerances.context)

    stage3 = model.solve(ObjectiveStage.ROUTE_UTILITY)
    return LexicographicResult.from_stages(stage0, stage1, stage2, stage3)
```

### 3.4 Progressive repair

```python
def repair_progressively(
    request: RepairRequest,
    plan_repository: PlanRepository,
    data_bundle: DatasetBundle,
    routing_context: RoutingContext,
    solver: RepairSolver,
    evaluator: PlanEvaluator,
) -> RepairOutcome:
    parent = plan_repository.load(request.parent_plan_id)
    attempts: list[PlannerRun] = []

    for radius in request.allowed_radii:
        neighborhood = build_repair_neighborhood(parent, request, radius)
        candidate = solver.solve(parent, request, neighborhood, data_bundle, routing_context)
        attempts.extend(candidate.runs)
        if candidate.plan is None:
            continue
        evaluation = evaluator.evaluate(candidate.plan, routing_context, data_bundle)
        if evaluation.comparison_eligibility is ComparisonEligibility.ELIGIBLE:
            return RepairOutcome.success(candidate.plan, evaluation, attempts, radius)

    diagnosis = compute_smallest_relaxation(parent, request, attempts)
    return RepairOutcome.infeasible(attempts, diagnosis)
```

---

## 4. Public data models

### 4.1 Constraint vocabularies

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
```

### 4.2 OwnedConstraint

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
```

**Invariants**

- `LLM_INTERPRETATION` with `confirmed=False` is inactive by default.
- `LOCKED` has no automatic relaxation variable.
- `BOOKED` requires explicit `allow_booked_relaxation=True` in the request.
- `TEST_ONLY` cannot become part of an accepted plan unless separately confirmed.

### 4.3 PlanArtifact v2

Do not delete `research_artifacts.PlanArtifact` immediately. Introduce a migration-compatible v2 or extend it with backward-compatible defaults.

Required fields:

```text
plan_id
parent_plan_id
source_run_id
planning_request_id
catalog_snapshot_id
context_snapshot_id
selected_stops
ordered_days
lodging_assignments
route_ids_by_day
owned_constraints
modeled_metrics
change_components
certificate_id
content_hash
created_at
schema_version
```

### 4.4 RepairRequest

```python
@dataclass(frozen=True)
class RepairRequest:
    repair_request_id: str
    parent_plan_id: str
    disruption_snapshot_id: str
    edits: tuple[EditRequest, ...]
    allowed_radii: tuple[RepairRadius, ...]
    allow_booked_relaxation: bool = False
    test_only: bool = False
```

### 4.5 PlanDiff

```python
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

---

## 5. Lodging architecture

### 5.1 Provider-independent lodging categories

```python
class LodgingCategory(StrEnum):
    HOTEL = "hotel"
    MOTEL = "motel"
    HOSTEL = "hostel"
    GUEST_HOUSE = "guest_house"
    BED_AND_BREAKFAST = "bed_and_breakfast"
    INN = "inn"
    RESORT = "resort"
    EXTENDED_STAY = "extended_stay"
    APARTMENT = "apartment"
    VACATION_RENTAL = "vacation_rental"
    PRIVATE_ROOM = "private_room"
    CAMPGROUND = "campground"
    CAMPING_CABIN = "camping_cabin"
    RV_PARK = "rv_park"
    OTHER = "other"
```

Do not use `AIRBNB` as a lodging category. Airbnb is a provider/brand. A listing is `vacation_rental` or `private_room` unless official provider data says otherwise.

### 5.2 Provider protocol

```python
class LodgingProvider(Protocol):
    provider_id: str

    def probe_credentials(self) -> ProviderActivation: ...
    def capabilities(self) -> frozenset[LodgingCapability]: ...
    def search_entities(self, query: LodgingSearchQuery) -> tuple[LodgingObservation, ...]: ...
    def fetch_rates(self, query: RateQuery) -> tuple[HotelRateSnapshot, ...]: ...
```

### 5.3 Activation policy

1. Provider is disabled unless config says `enabled: true`.
2. Resolve only the environment-variable name from config.
3. If the environment variable is missing, return `ProviderActivation.unavailable("missing_credentials")`.
4. Perform a low-cost documented probe endpoint or sandbox request.
5. Invalid/unauthorized key disables the provider and records a redacted reason.
6. Do not fall through to repeated network calls.
7. Never write the credential or full authorization header.
8. Fallback to stable OSM/curated entities and explicit estimated planning cost.

### 5.4 Provider priority

```text
OSM identity (required benchmark)
→ Geoapify optional category/identity enrichment
→ LiteAPI optional date-specific rates/availability
→ Google Places optional rating/status/photos
→ Booking.com / Expedia dormant partner adapters
→ estimated planning-cost fallback
```

### 5.5 API-specific policies

- **Geoapify:** optional; store query/response provenance and attribution; rate-limit according to active plan.
- **LiteAPI:** sandbox responses must set `environment="sandbox"` and `production_booking_evidence=False`.
- **Google Places:** request the smallest field mask; never make it a reproducibility dependency.
- **Booking.com Demand / Expedia Rapid:** adapter may compile and be tested with fakes, but production activation requires partner credentials.
- **Airbnb:** no undocumented API, scraping, or brand inference. Add only after official partner access.

---

## 6. Routing architecture

### 6.1 Existing contracts to preserve

- `RouteLegResult`
- `RouteResult`
- `RoadRouteCache`

Extend rather than replace them.

### 6.2 New interfaces

```python
class RoutingProvider(Protocol):
    provider_id: str

    def probe(self) -> RoutingActivation: ...
    def table(self, request: RouteMatrixRequest) -> RouteMatrix: ...
    def route(self, request: RouteRequest) -> RouteLegResult: ...
    def nearest(self, request: NearestRequest) -> SnapResult: ...
```

### 6.3 Use of OSRM services

- **Table:** optimizer distance and duration matrix.
- **Route:** geometry/distance/duration for selected legs.
- **Nearest:** snapping quality and snap-distance diagnostics.
- Any OSRM `fallback_speed` cell is approximate and `road_validated=False`.

### 6.4 Provider order

```text
validated pinned context cache
→ local OSRM Docker
→ explicitly approved remote OSRM
→ geodesic fallback (demo/approximate only)
```

### 6.5 Local OSRM Docker files

```text
docker/osrm/
├── docker-compose.yml
├── .env.example
├── README.md
├── profiles/car.lua                # pinned or documented upstream profile
└── scripts/
    ├── download_extract.sh
    ├── preprocess.sh
    └── healthcheck.sh
```

Do not commit large `.osm.pbf`, `.osrm*`, or extracted graph files.

### 6.6 Route matrix invariants

- Matrix rows/columns identify stable entity IDs.
- Every cell has source, query hash, and validation state.
- Missing cells are explicit; do not silently substitute zero.
- The optimizer, map, and evaluator use the same matrix/context snapshot.
- Selected route geometry must match the plan sequence and route IDs.

---

## 7. Pipeline runner

### 7.1 Authoritative interface

```python
@dataclass(frozen=True)
class PipelineRun:
    run_id: str
    data_bundle: DatasetBundle
    parent_plan: PlanArtifact | None
    planner_runs: tuple[PlannerRun, ...]
    output_plans: tuple[PlanArtifact, ...]
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

### 7.2 Required order

1. resolve and validate config;
2. create immutable run directory;
3. load/validate catalog and context;
4. resolve providers but do not activate unconfigured providers;
5. load/generate parent plan;
6. run generation or repair;
7. route selected legs;
8. independent evaluation;
9. plan diff and explanation evidence;
10. export run manifest and dashboard inputs;
11. never overwrite a previous run directory.

### 7.3 Output layout

```text
runs/<run_id>/
├── manifest.json
├── resolved_config.redacted.json
├── dataset_manifest.json
├── requests/
├── planner_runs.jsonl
├── plans/
├── diffs/
├── routing/
├── evaluations/
├── explanations/
├── metrics/
└── dashboard/
```

---

## 8. Notebook migration

### Stage A

Keep the current nbconvert command working. Replace notebook business logic with calls to package functions. Notebook cells may:

- resolve paths/config;
- call `run_research_pipeline()`;
- display DataFrames/figures;
- link exported dashboard artifacts.

Notebook cells may not:

- call external APIs directly;
- construct Gurobi models;
- merge source entities;
- synthesize demand proxies;
- mutate plan dictionaries;
- write canonical source snapshots.

### Stage B

Add a CLI:

```bash
python -m itinerary_system.cli run \
  --config configs/nature_trip_config.yaml \
  --catalog california_v1 \
  --context context_2026_07_road_validated_v1
```

The notebook then loads a completed run.

---

## 9. Benchmark specification

### 9.1 Canonical plans

- `plan_demo_current.json`: serialized current 7-day route.
- `plan_california_repair_v1.json`: reviewed research parent.

### 9.2 Six disruptions

1. weather deterioration;
2. road closure;
3. hotel unavailability;
4. attraction closure/opening-window conflict;
5. reduced driving tolerance/fatigue;
6. newly added user must-visit.

### 9.3 Ownership profiles

- flexible;
- mixed;
- booking-constrained;
- scenic-route-protective;
- high-lock.

### 9.4 Baselines

- LLM-only rewrite;
- full reoptimization;
- current hybrid pipeline;
- weighted-sum repair;
- fixed-radius repair;
- progressive sequential lexicographic repair.

### 9.5 Split rule

Split by complete parent-plan/disruption family. Never split POI rows from the same plan across train/development/test.

---

## 10. Execution Phases and Handoff Gates

This section is the authoritative phase overlay for executing the work packages below. The package IDs remain the implementation backlog; the gates here define what must be true before later research, benchmark, UI, or study claims can be made.

### 10.1 Current status as of 2026-06-30

**Completed or substantially present**

- Clean data snapshot flow and `DatasetBundle` validation exist for the current California-oriented corpus.
- Phase 0 artifact export, strict validators, readiness summaries, and route-source prechecks exist.
- Route request manifests and legacy route-cache audits exist, so missing road-routing evidence is inspectable rather than silent.
- The current code already has route artifact contracts, planner-run artifacts, map/evaluation outputs, Gurobi and heuristic planning entry points, and a research-foundation test suite.

**Blocked or not yet publication-valid**

- Strict final-plan eligibility is blocked until local OSRM or pinned road-route responses exist for all required benchmark route legs.
- Existing solver-facing flows still include geodesic or fallback travel proxies in places that cannot support transportation-optimization claims.
- The repository does not yet have parent-plan-aware repair, typed ownership semantics, progressive repair neighborhoods, generalized final-plan diffing, or evidence-grounded counterfactual explanations.
- The package-level pipeline runner is not yet the only authoritative entry point; notebook-oriented orchestration still exists.

**Next milestone**

- Produce road-valid route evidence, rerun strict Phase 0 validation, then implement canonical parent/child plans and typed plan diff before implementing the repair master.

### 10.2 Gate table

| Gate | Research purpose | Required package IDs | Start condition | Exit evidence | Blocking risks |
| --- | --- | --- | --- | --- | --- |
| G0 - Repository truth and Phase 0 closeout | Freeze the current system state so later claims are grounded in reproducible evidence. | `FOUND-001`, `QA-001`, current Phase 0 scripts | Clean checkout can run the current foundation tests and artifact exporters. | Strict Phase 0 validation passes, or the only remaining blocker is explicitly documented missing approved OSRM route evidence. | Hidden data drift, stale manifests, untracked notebook outputs, route fallback treated as validated. |
| G1 - Canonical plans and plan diff | Establish immutable parent/child plans and measurable repair changes. | `PLAN-001`, `PLAN-002`, `DIFF-001`, `VERIFY-001` | G0 evidence is available and current artifact contracts are preserved. | Frozen parent and child plans load, hash, diff, and evaluate independently. | Repair work starts before ownership and diff semantics are stable. |
| G2 - Road-valid routing context | Ensure every transportation comparison uses validated road evidence. | `ROUTE-001`, `ROUTE-002`, `ROUTE-003`, `ROUTE-004` | Route request manifest exists and local/pinned OSRM source is available. | Optimizer, evaluator, and renderer consume the same validated route records; strict route-source checks pass. | Public OSRM dependency, geodesic fallback leakage, partial cache coverage, mismatched geometry/duration sources. |
| G3 - Single-day ownership-aware repair | Prove the core mixed-initiative repair idea on one inspectable disruption. | `REPAIR-001`, `REPAIR-002`, `REPAIR-003`, `REPAIR-004`, `VERIFY-001` | G1 and G2 pass for the selected parent plan and day. | Locked/booked/user-strong commitments are preserved unless explicitly relaxed; output has typed diff, evaluator report, and explanation-ready evidence. | Weighted-sum shortcuts, untyped deletes/adds, hidden post-solve mutation, route-invalid subproblem costs. |
| G4 - Progressive multi-day repair | Expand from one day to smallest-radius certified multi-day repair. | `REPAIR-005`, `LODGE-001` through `LODGE-007`, `ROUTE-004` | G3 succeeds on at least one canonical disruption. | Neighborhood expansion is deterministic, explainable, and stops at the smallest certified feasible child plan. | Over-broad reoptimization, lodging provider ambiguity, corridor changes not connected to route evidence. |
| G5 - Benchmark package | Convert the system into a defensible computational paper package. | `BENCH-001`, `BENCH-002`, completed G0-G4 dependencies | Canonical plans, route evidence, diff, repair, and evaluator are stable. | Six-disruption benchmark reports feasibility, preservation, diff size, travel, risk, utility, and runtime against baselines. | Split leakage, missing-source utility bias, unvalidated route evidence, overclaiming beyond benchmark scope. |
| G6 - Interactive prototype | Make the repair system usable from the same pipeline artifacts. | `PIPE-001`, `NOTEBOOK-001`, `UI-001` | G5 artifact contracts are stable enough to render without notebook-only logic. | One command runs the configured case and produces artifacts consumable by notebook and UI. | Duplicate orchestration, notebook state mutation, UI consuming noncanonical dictionaries. |
| G7 - Explanation and study readiness | Ground user-facing claims in inspectable evidence links. | `EXPLAIN-001`, `EXPLAIN-002`, `STUDY-001` | G6 prototype can show original/repaired plans and certified diffs. | Every explanation links to diff components, owned constraints, route evidence, and evaluator metrics. | Explanations become template text without evidence IDs, study logs cannot reconstruct decisions. |
| G8 - CHI/IUI extension | Extend the backend repair result into interaction and design claims. | Completed G0-G7, `STUDY-001` refinements | Pilot evidence identifies stable user tasks and explanation formats. | Stable prototype, logged interactions, study materials, and defensible claims about user control and plan preservation. | Premature study before backend claims are valid, interaction polish masking uncertified solver behavior. |

### 10.3 Implementation order lock

- Do not implement repair master logic before canonical `PlanArtifact`, `PlanDiff`, and independent evaluator contracts exist.
- Do not make benchmark, transportation, or publication claims until local or pinned road-valid routing evidence passes strict validation.
- Do not start UI or user-study work until pipeline, diff, repair, evaluator, and explanation contracts are stable enough to replay from artifacts.
- Preserve existing Phase 0 script names and artifact names unless a compatibility wrapper is provided.
- Treat public OSRM as disabled by default. Use local OSRM or pinned route evidence unless a human explicitly approves another routing source for a non-publication run.

### 10.4 Gate-by-gate execution detail

#### G0 - Repository truth and Phase 0 closeout

1. Run the current research-foundation tests.
2. Run the route-source precheck with probe mode against the configured routing source.
3. Run the Phase 0 evidence pipeline without allowing silent fallback routes.
4. Record whether strict final eligibility passes or is blocked only by missing approved OSRM evidence.
5. Do not proceed to transportation claims if any route source is geodesic, synthetic, stale, or untraceable.

Required handoff artifacts:

- dataset snapshot manifest;
- route request manifest;
- route-source check output;
- Phase 0 validation report;
- readiness summary naming any remaining blocker.

#### G1 - Canonical plans and plan diff

1. Add canonical parent/child plan schemas while preserving compatibility with existing artifact readers.
2. Represent commitment ownership explicitly with typed strengths and relaxation policy.
3. Store immutable parent IDs, child IDs, content hashes, run IDs, and artifact lineage.
4. Implement generalized diff components for delete, add, day movement, time shift, order change, lodging change, road/corridor change, and explicit relaxation.
5. Add independent evaluation hooks so a plan can be certified after diffing.

Required handoff artifacts:

- `plan_demo_current` parent artifact;
- `plan_california_repair_v1` parent artifact;
- one child repair artifact fixture;
- typed `PlanDiff` JSON;
- evaluator report tied to parent and child hashes.

#### G2 - Road-valid routing context

1. Add local OSRM Docker/runbook assets and document the map extract/version used.
2. Implement OSRM Table for matrices and OSRM Route for selected geometry.
3. Keep Nearest/snap evidence separate from route duration evidence.
4. Make route cache entries source-aware and fail strict validation when required route legs are missing.
5. Replace solver-facing geodesic travel costs with validated route matrix lookups or explicit non-publication fallback flags.

Required handoff artifacts:

- local OSRM run instructions;
- complete route matrix/cache for the benchmark slice;
- route-source policy report;
- strict validation proving evaluator and optimizer use the same road evidence.

#### G3 - Single-day ownership-aware repair

1. Implement `EditRequest`, `DisruptionSnapshot`, `RepairNeighborhood`, `RepairRequest`, and `RepairOutcome` for one canonical single-day disruption.
2. Convert the current route solver into a day-route subproblem that receives parent-plan commitments and validated route matrix costs.
3. Implement sequential lexicographic solve stages in this order: hard relaxations, ownership-weighted changes, risk/travel/cost, then utility/diversity/scenic value.
4. Emit typed change metrics instead of only aggregate objective values.
5. Re-evaluate the final child plan independently after solving and invalidate certification after any post-solve mutation.

Required handoff artifacts:

- repair request fixture;
- repair outcome fixture;
- objective-stage trace;
- typed plan diff;
- independent evaluator report;
- explanation evidence references.

#### G4 - Progressive multi-day repair

1. Add deterministic neighborhood expansion across day, adjacent days, lodging/base, and route corridor scopes.
2. Freeze all entities outside the active neighborhood unless a documented relaxation opens them.
3. Stop at the smallest neighborhood that produces a certified feasible child plan.
4. Record every expansion attempt, infeasibility reason, and accepted radius.

#### G5 - Benchmark package

1. Generate the six disruption families from canonical parent plans.
2. Evaluate flexible, mixed, booking-constrained, scenic-route-protective, and high-lock ownership profiles.
3. Compare against LLM-only rewrite, full reoptimization, current hybrid pipeline, weighted-sum repair, fixed-radius repair, and progressive sequential lexicographic repair.
4. Report feasibility, relaxation count, ownership-weighted diff, route-valid travel, weather/risk score, utility, scenic/diversity metrics, runtime, and explanation coverage.
5. Split by complete parent-plan/disruption family; never split rows from the same plan family across train, development, and test.

#### G6 - Interactive prototype

1. Implement `pipeline_runner.py` as the authoritative execution entry point.
2. Move notebook logic into configure, execute, display, and export cells only.
3. Render original/repaired routes from canonical artifacts, not mutable notebook dictionaries.
4. Ensure one command can reproduce the artifacts shown in the notebook or UI.

#### G7 - Explanation and study readiness

1. Create structured explanation evidence that references constraints, diffs, route records, objective stages, and evaluator metrics.
2. Add why, why-not, and what-if explanation runners only when they can cite stored evidence.
3. Export study tasks and event logs that can reconstruct parent plan, edit, disruption, repair output, and user-facing explanation.

#### G8 - CHI/IUI extension

1. Use the G5 benchmark and G7 pilot evidence to decide whether the first submission is transportation-optimization-first, IUI-first, or CHI-extension-first.
2. Keep claims limited to certified repair behavior, visible ownership semantics, consequence preview, and calibrated reliance.
3. Do not claim production-grade booking, global travel planning, or real-time universal weather/traffic reliability.

---
## 11. Work packages

Each package below is independently reviewable. Do not merge unrelated packages into one large change.

---

### FOUND-001 — Baseline and repository truth

**Research motivation**  
Ensure later experiments are grounded in an exact code/data state.

**Current repository evidence**  
Phase 0 files are present, but the current head SHA is not written into every manifest and several flat output paths remain.

**Files to create**

- `src/itinerary_system/repository_state.py`
- `tests/test_repository_state.py`

**Files to modify**

- `src/itinerary_system/artifact_metadata.py`
- `src/itinerary_system/config.py`

**Public interfaces**

```python
@dataclass(frozen=True)
class RepositoryState:
    commit_sha: str
    dirty: bool
    package_version: str
    captured_at: str
```

**Mathematical requirement**  
None.

**Algorithm**

- Resolve Git commit when `.git` exists.
- Allow an explicit environment override for packaged/CI runs.
- Mark unknown rather than inventing a SHA.

**Inputs / outputs**

- Input: repository root.
- Output: repository state embedded in run manifest.

**Invariants**

- Unknown commit is explicit.
- No shell failure aborts a demonstration run unless strict mode is enabled.

**Failure behavior**

- `strict=True` raises `RepositoryStateUnavailable`.
- permissive mode returns `commit_sha="unknown"`.

**Tests**

- known mocked commit;
- dirty flag;
- unknown repository;
- artifact metadata serialization.

**Acceptance criteria**

- Every new run manifest contains repository state.
- No fabricated commit identifier.

**Artifacts produced**

- `repository_state` section in `manifest.json`.

**Out of scope**

- GitHub API calls.

**Dependencies**

- none.

---

### DATA-001 — Complete catalog/context snapshot separation

**Research motivation**  
Stable entities and time-sensitive context must not share one directory contract.

**Current repository evidence**  
`data/snapshot.py` loads stable and context CSVs from the same `california_v1` directory.

**Files to create**

- `data/contexts/context_static_demo_2026_06/manifest.json`
- `src/itinerary_system/data/context.py`
- `tests/data/test_context_snapshot.py`

**Files to modify**

- `src/itinerary_system/data/schemas.py`
- `src/itinerary_system/data/snapshot.py`
- `data/snapshots/california_v1/manifest.json`

**Public interfaces**

```python
@dataclass(frozen=True)
class CatalogBundle: ...

@dataclass(frozen=True)
class ContextBundle: ...

@dataclass(frozen=True)
class DatasetBundle:
    catalog: CatalogBundle
    context: ContextBundle
```

**Mathematical requirement**  
All compared methods use identical catalog and context identifiers.

**Algorithm**

- Load catalog manifest from `data/snapshots/<id>`.
- Load context manifest from `data/contexts/<id>`.
- Validate both separately.
- Compose a DatasetBundle without copying/mutating frames.

**Invariants**

- Context rows all match `context_snapshot_id`.
- Catalog tables contain no rate quote or forecast that changes by trip date.

**Failure behavior**

- Missing table raises typed snapshot error.
- Schema mismatch returns validation errors and blocks strict optimization.

**Tests**

- clean clone;
- mismatched context ID;
- missing table;
- invalid hash;
- backward-compatible old snapshot load with warning.

**Acceptance criteria**

- New runs reference two manifests.
- Legacy combined snapshot remains loadable during migration.

**Artifacts produced**

- catalog and context validation reports.

**Out of scope**

- API refresh.

**Dependencies**

- FOUND-001.

---

### DATA-002 — Evidence/provenance contract

**Research motivation**  
Separate observed, curated, imputed, simulated, derived, and optimizer output values.

**Files to create**

- `src/itinerary_system/data/provenance.py`
- `tests/data/test_feature_provenance.py`

**Files to modify**

- `src/itinerary_system/data/schemas.py`
- `src/itinerary_system/data_enrichment.py`

**Public interfaces**

```python
class EvidenceKind(StrEnum):
    OBSERVED = "observed"
    CURATED = "curated"
    IMPUTED = "imputed"
    SIMULATED = "simulated"
    DERIVED = "derived"
    OPTIMIZER_OUTPUT = "optimizer_output"

@dataclass(frozen=True)
class FeatureProvenance: ...
```

**Mathematical requirement**  
Every feature entering an objective or hard constraint has a provenance row and method version.

**Invariants**

- `source_coverage_score` is not labeled calibrated uncertainty.
- `simulated_demand_proxy` has `EvidenceKind.SIMULATED`.

**Tests**

- all optimizer columns covered;
- no optimizer output used as source observation;
- legacy aliases generate warnings.

**Acceptance criteria**

- strict data validation fails on missing provenance.

**Dependencies**

- DATA-001.

---

### DATA-003 — Missing-source-aware utility fusion

**Research motivation**  
Missing Yelp must not equal a low Yelp score.

**Files to create**

- `tests/data/test_missing_source_fusion.py`

**Files to modify**

- `src/itinerary_system/utility_model.py`

**Public interfaces**

```python
@dataclass(frozen=True)
class SourceSignal:
    value: float
    available: bool
    reliability: float
```

**Mathematical requirement**

```text
u_i = sum_s w_s m_is z_is / sum_s w_s m_is
coverage_i = sum_s w_s m_is / sum_s w_s
```

**Invariants**

- unavailable source is excluded from numerator and denominator;
- source coverage is reported separately from utility;
- within-source normalization uses only available rows.

**Tests**

- identical non-Yelp POIs do not differ because one city lacks Yelp;
- all missing source returns explicit fallback;
- source ablation outputs deterministic results.

**Acceptance criteria**

- utility model exports source masks and ablation-ready signals.

**Dependencies**

- DATA-002.

---

### LODGE-001 — Canonical lodging taxonomy and entities

**Research motivation**  
Travelers choose lodging type as well as price and location.

**Files to create**

- `src/itinerary_system/lodging/models.py`
- `src/itinerary_system/lodging/taxonomy.py`
- `data/curated/lodging_category_mapping.csv`
- `tests/lodging/test_taxonomy.py`

**Files to modify**

- `src/itinerary_system/data_enrichment.py`

**Public interfaces**

- `LodgingCategory`
- `LodgingEntity`
- `LodgingObservation`
- `HotelRateSnapshot`

**Mathematical requirement**  
Master model can filter or penalize lodging category independently of provider.

**Invariants**

- Airbnb is a provider, not a category.
- unknown types map to `OTHER` with review flag.

**Tests**

- OSM hotel/motel/hostel/guest_house/apartment mapping;
- vacation-rental and private-room mappings;
- unmapped type audit.

**Acceptance criteria**

- stable hotel entities include canonical category and raw source type.

**Dependencies**

- DATA-001.

---

### LODGE-002 — Provider protocol and activation gate

**Files to create**

- `src/itinerary_system/lodging/provider.py`
- `src/itinerary_system/lodging/errors.py`
- `tests/lodging/test_provider_activation.py`

**Files to modify**

- `src/itinerary_system/config.py`

**Public interfaces**

- `LodgingProvider`
- `ProviderActivation`
- `LodgingCapability`
- `ProviderRegistry`

**Invariants**

- disabled or missing-key provider sends no normal API request;
- exceptions never contain secret values;
- activation result is cacheable for one run.

**Tests**

- disabled;
- missing key;
- unauthorized probe;
- successful sandbox probe with fake client;
- redaction.

**Acceptance criteria**

- all later commercial adapters implement the same protocol.

**Dependencies**

- LODGE-001.

---

### LODGE-003 — OSM lodging adapter

**Files to create**

- `src/itinerary_system/lodging/osm_adapter.py`
- `tests/lodging/test_osm_adapter.py`

**Files to modify**

- `src/itinerary_system/data_enrichment.py`

**Requirement**

Move OSM hotel parsing from the monolithic enrichment file into the adapter while preserving compatibility output.

**Acceptance criteria**

- same or explicitly migrated rows;
- provider-neutral category;
- OSM ID and raw tags preserved;
- no price presented as observed.

**Dependencies**

- LODGE-001, LODGE-002.

---

### LODGE-004 — Geoapify optional enrichment adapter

**Files to create**

- `src/itinerary_system/lodging/geoapify_adapter.py`
- `tests/lodging/test_geoapify_adapter.py`

**Requirement**

Implement with an injected HTTP client and fake-response tests. Do not require a real key in CI.

**Acceptance criteria**

- adapter activates only with valid configured key;
- category/address/identity observations retain provider IDs and retrieval timestamp;
- attribution and query hash are written to cache manifest.

**Dependencies**

- LODGE-002.

---

### LODGE-005 — LiteAPI sandbox/production rate adapter

**Files to create**

- `src/itinerary_system/lodging/liteapi_adapter.py`
- `tests/lodging/test_liteapi_adapter.py`

**Public behavior**

- supports sandbox and production base URLs;
- probes key before rate calls;
- parses rate, currency, dates, occupancy, cancellation, meal plan, and availability;
- marks sandbox observations as non-production evidence.

**Failure behavior**

- invalid key -> provider unavailable, no hidden fallback claim;
- no match -> empty tuple plus audit record;
- rate response never mutates stable entity.

**Acceptance criteria**

- fake sandbox response fully parsed;
- no secret in snapshots;
- API tests opt-in via marker and environment variable.

**Dependencies**

- LODGE-001, LODGE-002.

---

### LODGE-006 — Google Places optional metadata adapter

Implement field-minimized Place Details/Text Search with fake tests. Store rating/status/address/website/photos as source observations. Do not use date-specific price claims.

**Dependencies:** LODGE-002.

---

### LODGE-007 — Dormant Booking.com and Expedia adapters

Implement protocols, configuration, response models, and fake contract tests. Do not activate or send requests without approved credentials/partner access.

**Dependencies:** LODGE-002.

---

### ROUTE-001 — Unified routing provider and RouteMatrix

**Phase 0.2 status:** Core `RouteMatrix`, provider protocol types, CSV/context loaders, explicit geodesic fallback matrix, and solver adapter are implemented. Validated provider completion and full benchmark matrix generation remain in ROUTE-002/ROUTE-003/Phase 4.

**Files to create**

- `src/itinerary_system/routing/provider.py`
- `src/itinerary_system/routing/matrix.py`
- `tests/routing/test_route_matrix.py`

**Files to modify**

- `src/itinerary_system/routing/models.py`

**Mathematical requirement**  
All `tau_ij` and distance values used by a solver come from a RouteMatrix with explicit cell provenance.

**Acceptance criteria**

- no zero for missing cell;
- fallback cells are not road validated;
- stable entity IDs index rows/columns.

**Dependencies**

- DATA-001.

---

### ROUTE-002 — OSRM client

**Files to create**

- `src/itinerary_system/routing/osrm_client.py`
- `tests/routing/test_osrm_client.py`

**Requirement**

Implement Table, Route, and Nearest calls with injected HTTP client, timeout, retry policy, query hash, and typed parsing.

**Invariants**

- `fallback_speed_cells` become nonvalidated;
- route and table provider/version recorded;
- public endpoint requires explicit policy flag.

**Dependencies**

- ROUTE-001.

---

### ROUTE-003 — Local OSRM Docker

**Files to create**

- `docker/osrm/docker-compose.yml`
- `docker/osrm/README.md`
- `docker/osrm/.env.example`
- preprocessing/healthcheck scripts

**Acceptance criteria**

- documented clean setup;
- pinned image tag;
- extract checksum recorded;
- healthcheck and one test route;
- large graph files ignored.

**Dependencies**

- ROUTE-002.

---

### ROUTE-004 — Replace geodesic travel inside solvers

**Phase 0.2 status:** `multi_objective_route.py`, `hierarchical_gurobi.py`, and the route oracle wrapper accept injected route matrices/adapters. Publication mode now fails closed on missing, fallback, or non-road-validated cells; demo mode keeps explicit geodesic fallback behavior. Full pipeline calls still need validated matrix provisioning before publication comparisons.

**Files to modify**

- `src/itinerary_system/multi_objective_route.py`
- `src/itinerary_system/hierarchical_gurobi.py`
- `src/itinerary_system/route_gurobi_oracle.py`

**Requirement**

Inject RouteMatrix. Keep geodesic implementation only in explicit approximate mode. Remove unreachable code from `route_gurobi_oracle.py` after equivalence tests.

**Acceptance criteria**

- publication mode refuses missing road-valid matrix;
- demo mode marks fallback;
- optimizer/map/evaluator totals match route records within tolerance.

**Dependencies**

- ROUTE-001, ROUTE-002.

---

### PLAN-001 — Plan and ownership schemas

**Implementation status**

Phase 1.0 substrate implemented: `OwnedConstraint`, closed ownership/relaxation vocabularies, v2 plan migration compatibility, and active/inactive constraint filtering are available in `src/itinerary_system/plans/` and `src/itinerary_system/research_artifacts.py`.

**Files to create**

- `src/itinerary_system/plans/models.py`
- `tests/plans/test_models.py`

**Files to modify**

- `src/itinerary_system/research_artifacts.py`
- `src/itinerary_system/schemas.py`

**Requirement**

Introduce v2 types without breaking Phase 0 readers.

**Acceptance criteria**

- content hash stable;
- parent plan immutable;
- unconfirmed LLM constraints inactive;
- JSON round trip.

**Dependencies**

- FOUND-001, DATA-001.

---

### PLAN-002 — Immutable plan repository and canonical parents

**Implementation status**

Phase 1.0 substrate implemented for append-only JSON storage and the current demo parent: `PlanRepository`, `load_plan()`, `save_plan_append_only()`, and `data/benchmark/parent_plans/plan_demo_current.json` exist. Broader reviewed benchmark parent publication remains a later pipeline step.

**Files to create**

- `src/itinerary_system/plans/repository.py`
- `data/benchmark/parent_plans/plan_demo_current.json`
- `data/benchmark/parent_plans/plan_california_repair_v1.json`
- `tests/plans/test_repository.py`

**Acceptance criteria**

- save is append-only;
- duplicate ID with different content fails;
- current demo and reviewed plan validate;
- no generated plan overwrites parent.

**Dependencies**

- PLAN-001.

---

### DIFF-001 — Generalized plan diff

**Implementation status**

Phase 1.0 substrate implemented: `compute_plan_diff()` and `PlanDiffBuilder` emit typed stop, day, time, order, lodging, road, unchanged-day, and deterministic weighted-cost components. Solver consumption of these components remains Phase 2 work.

**Files to create**

- `src/itinerary_system/plans/diff.py`
- `tests/plans/test_diff.py`

**Mathematical requirement**

Compute deletion, addition, day move, time shift, reorder, lodging, road, and unchanged-day components.

**Acceptance criteria**

- symmetric identity cases;
- known hand-built examples;
- weighted cost agrees with ownership policy;
- Kendall/order metric handles partially shared sequences.

**Dependencies**

- PLAN-001.

---

### REPAIR-001 — Repair neighborhood

**Files to create**

- `src/itinerary_system/repair/neighborhood.py`
- `tests/repair/test_neighborhood.py`

**Requirement**

Build radius 0, 1, 2, and full editable sets from disruption and parent plan.

**Acceptance criteria**

- unaffected days frozen;
- adjacent incoming/outgoing legs included;
- hotel nights included when a day boundary changes.

**Dependencies**

- PLAN-001.

---

### REPAIR-002 — Ownership-aware repair master

**Files to create**

- `src/itinerary_system/repair/master_model.py`
- `src/itinerary_system/repair/change_variables.py`
- `tests/repair/test_master_model.py`

**Mathematical requirement**

Implement selection/day/lodging variables and typed change variables relative to `pi0`.

**Acceptance criteria**

- locked POI cannot be deleted;
- booked hotel requires explicit relaxation permission;
- outside-neighborhood assignments fixed;
- objective expressions export component values.

**Dependencies**

- PLAN-002, DIFF-001, REPAIR-001, LODGE-001.

---

### REPAIR-003 — Sequential lexicographic solver

**Files to create**

- `src/itinerary_system/repair/lexicographic.py`
- `tests/repair/test_lexicographic.py`

**Acceptance criteria**

- later stage cannot degrade prior optimum beyond tolerance;
- all stage statuses and bounds stored;
- infeasible stage creates a failed PlannerRun;
- equivalent small instance matches Gurobi multiobjective implementation.

**Dependencies**

- REPAIR-002.

---

### REPAIR-004 — Day-route subproblem

Refactor current `multi_objective_route.py` into a typed `DayRouteSolver` using RouteMatrix, opening windows, visit duration, and fixed day assignment.

**Dependencies:** ROUTE-004, REPAIR-002.

---

### REPAIR-005 — Progressive repair controller

**Files to create**

- `src/itinerary_system/repair/progressive.py`
- `tests/repair/test_progressive.py`

**Acceptance criteria**

- stops at smallest independently eligible radius;
- stores every attempt;
- full replan only final configured fallback;
- returns smallest-relaxation diagnosis if no radius succeeds.

**Dependencies**

- REPAIR-003, REPAIR-004, VERIFY-001.

---

### VERIFY-001 — Independent final-plan evaluator

**Files to create**

- `src/itinerary_system/evaluation/plan_evaluator.py`
- `src/itinerary_system/evaluation/certificate.py`
- `tests/evaluation/test_plan_evaluator.py`

**Requirement**

Recompute final displayed plan constraints from PlanArtifact and routing/context records. Do not reuse solver feasibility booleans as proof.

**Checks**

- artifact/source/run linkage;
- hard owned constraints;
- route connectivity and road validation;
- daily time and opening windows;
- lodging continuity;
- budget;
- weather/closure feasibility;
- duplicate visits;
- post-solve mutation/content hash.

**Acceptance criteria**

- mutation invalidates certificate;
- unvalidated route blocks comparison;
- warnings separated from failures.

**Dependencies**

- PLAN-001, ROUTE-001, DATA-001.

---

### EXPLAIN-001 — Structured explanation evidence

**Files to create**

- `src/itinerary_system/explanation/evidence.py`
- `tests/explanation/test_evidence.py`

**Public objects**

- `WhyEvidence`
- `WhyNotEvidence`
- `WhatIfEvidence`
- `ContrastiveEvidence`
- `ExplanationClaim`

**Invariant**

Every numerical or causal claim has one or more valid references to a constraint, route record, plan diff, evaluation, or counterfactual run.

**Dependencies**

- DIFF-001, VERIFY-001.

---

### EXPLAIN-002 — Counterfactual runner and verbalizer

**Files to create**

- `src/itinerary_system/explanation/counterfactual.py`
- `src/itinerary_system/explanation/verbalizer.py`
- `tests/explanation/test_counterfactual.py`

**Requirement**

- Why-not forces or strengthens the omitted option and re-solves.
- What-if creates a sandbox child request.
- LLM verbalizer receives only structured evidence and returns claim-to-evidence mappings.
- Deterministic template verbalizer is the benchmark default.

**Dependencies**

- REPAIR-005, EXPLAIN-001.

---

### PIPE-001 — Authoritative pipeline runner

**Files to create**

- `src/itinerary_system/pipeline_runner.py`
- `tests/test_pipeline_runner.py`

**Files to modify**

- `src/itinerary_system/experiment_runner.py`
- `src/itinerary_system/__init__.py`

**Acceptance criteria**

- generation and repair modes;
- immutable run directory;
- no notebook dependency;
- returns typed PipelineRun;
- strict mode blocks ineligible plan;
- permissive demo mode reports warnings.

**Dependencies**

- DATA-001, PLAN-002, ROUTE-004, VERIFY-001; repair mode also requires REPAIR-005.

---

### NOTEBOOK-001 — Thin notebook migration

**Files to modify**

- `notebook/production_system_blueprint.ipynb`
- README execution instructions

**Acceptance criteria**

- documented nbconvert command still runs;
- notebook invokes pipeline runner;
- no direct API/model construction/simulation logic remains;
- outputs point to one run directory.

**Dependencies**

- PIPE-001.

---

### BENCH-001 — Six disruption generators

**Files to create**

- `src/itinerary_system/benchmark/disruptions.py`
- `data/benchmark/disruptions/*.json`
- `tests/benchmark/test_disruptions.py`

**Acceptance criteria**

- all six disruption families;
- deterministic IDs;
- explicit observed/synthetic status;
- no mutation of catalog snapshot.

**Dependencies**

- PLAN-002, DATA-001.

---

### BENCH-002 — Benchmark runner, baselines, and splits

**Files to create**

- `src/itinerary_system/benchmark/runner.py`
- `src/itinerary_system/benchmark/splits.py`
- `src/itinerary_system/benchmark/metrics.py`
- `tests/benchmark/test_no_leakage.py`

**Acceptance criteria**

- paired method runs on identical snapshots;
- split by parent/disruption family;
- baseline adapters;
- preservation, quality, computation, certificate, and explanation metrics;
- export long-form result table and run manifest.

**Dependencies**

- REPAIR-005, EXPLAIN-002, BENCH-001.

---

### UI-001 — Original/repaired route comparison

**Files to modify**

- dashboard exporter and renderer modules after canonical artifact migration.

**Requirement**

Display unchanged, added, removed, moved, locked, booked, disruption-caused, and solver-caused changes. Add certificate badge and evidence panel. Browser preview must be labeled preview unless it triggers a real backend repair run.

**Dependencies**

- PIPE-001, DIFF-001, EXPLAIN-001.

---

### STUDY-001 — User-study event logging

**Files to create**

- `src/itinerary_system/study/events.py`
- `src/itinerary_system/study/logger.py`
- `docs/user_study_protocol.md`
- `tests/study/test_event_redaction.py`

**Required fields**

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

Do not log secrets, raw personal itineraries, or raw LLM conversation by default.

**Dependencies**

- UI-001.

---

### QA-001 — Full research integrity test suite

Add tests for:

- clean-clone snapshot;
- manifest hashes;
- provenance completeness;
- missing-source fusion;
- provider activation and secret redaction;
- road validation and fallback gating;
- parent immutability;
- ownership constraints;
- sequential objective priority;
- neighborhood freezing;
- plan diff correctness;
- certificate invalidation;
- explanation evidence coverage;
- benchmark split leakage;
- notebook-to-run consistency.

**Dependencies:** all prior packages.

---

## 12. Dependency gates and venue-oriented exits

Section 10 is the authoritative engineering execution order. This section compresses those gates into venue-oriented exits for paper planning and should not be used to skip prerequisites.

### Gate A - Transportation-ready evidence

Corresponds to G0 through G5.

Requires:

- repository truth, clean data/context boundaries, and strict Phase 0 evidence;
- road-valid routing context with local or pinned OSRM evidence;
- canonical parent/child plans, typed ownership, generalized diff, and independent evaluator;
- ownership-aware repair through progressive multi-day neighborhoods;
- six-disruption benchmark with baselines, preservation/quality/runtime analysis, and honest limitations.

Exit condition: a road-validated computational benchmark package that can support transportation optimization claims without relying on geodesic or unverified fallback routing.

### Gate B - IUI-ready system

Corresponds to G0 through G7.

Adds:

- authoritative package-level pipeline runner;
- thin notebook migration;
- original/repaired route comparison UI;
- structured explanation evidence and counterfactual runners;
- controlled study prototype, pilot tasks, and replayable event logs.

Exit condition: a working mixed-initiative repair loop with computational evidence, evidence-grounded explanations, and pilot-ready user interaction traces.

### Gate C - CHI-ready system

Corresponds to G0 through G8.

Adds:

- refined user study design and analysis plan;
- interaction polish around plan ownership, consequence preview, and calibrated reliance;
- qualitative analysis that connects interface behavior to certified backend repair behavior.

Exit condition: a stable prototype, logged interactions, study materials, and defensible claims about user control and plan preservation.

---
## 13. Non-goals for the first paper

- nationwide or worldwide route planning;
- production booking transactions;
- scraping Airbnb or unofficial hotel APIs;
- claiming accurate real-world queue prediction;
- learned edge-level scenic perception;
- full conversational autonomous agent;
- online bandit personalization without real user feedback;
- replacing Gurobi with a custom solver;
- claiming global optimality after heuristic fallback or post-processing.

---

## 14. Definition of done for the complete implementation

The system is paper-ready only when all statements below are true:

1. A clean clone can load the benchmark catalog and context.
2. `plan_demo_current` and `plan_california_repair_v1` are immutable and validated.
3. All six disruptions produce typed RepairRequests.
4. The solver minimizes hard relaxations and typed owned changes before utility.
5. The controller stops at the smallest certified repair radius.
6. Every travel duration used by publication comparisons is road validated.
7. Every displayed plan has a source run, parent, content hash, diff, evaluation, and certificate state.
8. Post-solve mutation automatically invalidates certification.
9. Why/why-not/what-if claims have evidence references.
10. Benchmark splits do not leak parent-plan/disruption families.
11. Provider adapters remain inactive without valid credentials.
12. Secrets never enter artifacts.
13. The notebook and CLI use the same pipeline runner.
14. Full tests and strict evidence validation pass.
