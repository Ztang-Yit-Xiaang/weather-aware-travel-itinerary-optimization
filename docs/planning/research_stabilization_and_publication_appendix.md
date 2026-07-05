# Context-Aware, Inspectable Travel Optimization

## Research Stabilization and Publication Appendix

**Status:** Normative technical companion to the [main publication plan](research_stabilization_and_publication_plan.md)<br>
**Revised:** June 24, 2026<br>
**Scope:** Contracts, validation, compiler boundaries, planning and repair, evaluation, explanation, experiments, tests, artifacts, and staged implementation

This appendix defines the technical interpretation of the roadmap. The main plan controls research scope, contribution, venue, and claim decisions. A conflict between the documents must be resolved explicitly; neither document silently overrides the other.

---

## Contents

- [A. Repository Audit and Problem Register](#a-repository-audit-and-problem-register)
- [B. Context, Provider, and Geographic Scope](#b-context-provider-and-geographic-scope)
- [C. Canonical Contracts](#c-canonical-contracts)
- [D. Status and Eligibility Architecture](#d-status-and-eligibility-architecture)
- [E. Validation, Compilation, and Solver Boundaries](#e-validation-compilation-and-solver-boundaries)
- [F. Branch-and-Check and Restricted Repair](#f-branch-and-check-and-restricted-repair)
- [G. Independent Layered Evaluator](#g-independent-layered-evaluator)
- [H. Explanation and Correction Requirements](#h-explanation-and-correction-requirements)
- [I. Experiment and Study Designs](#i-experiment-and-study-designs)
- [J. Staged Implementation Backlog](#j-staged-implementation-backlog)
- [K. Test, Artifact, and Reproducibility Policy](#k-test-artifact-and-reproducibility-policy)
- [L. Scope, Literature Traceability, and Acceptance](#l-scope-literature-traceability-and-acceptance)

---

## A. Repository Audit and Problem Register

### A.1 Current implementation classification

| Capability | Status | Technical note |
|---|---|---|
| Structured trip and interest request | PARTIALLY_IMPLEMENTED | `TripPlanningRequest` maps explicit fields and normalized interests to config overrides; it does not represent parsed language, ambiguity, dependencies, or confirmation |
| Weather-aware utility and constraints | PARTIALLY_IMPLEMENTED | Weather risk affects scoring and epsilon constraints; no frozen disruption or repair contract exists |
| Multi-day planning, hotels, and base cities | IMPLEMENTED_BUT_NOT_VALIDATED | Produced through current experiment and hierarchical pipelines; no independent complete-plan evaluation |
| Gurobi and heuristic/fallback route generation | IMPLEMENTED_BUT_NOT_VALIDATED | Status strings and gaps exist but method lineage and fallback semantics are not canonical |
| Route alternatives and dashboard | IMPLEMENTED_BUT_NOT_VALIDATED | Multiple route families, weather layers, source confidence, and evaluation views exist; browser previews do not rerun the solver |
| Existing provider access | PARTIALLY_IMPLEMENTED | Open-Meteo, NPS, OSM/Overpass, Wikidata/Wikipedia, and OSRM are present with provider-specific caching/fallback behavior |
| Artifact metadata | PARTIALLY_IMPLEMENTED | Config hash, timestamp, and file list exist; canonical content hashes and immutable lineage do not |
| Why-selected and why-not explanation | PARTIALLY_IMPLEMENTED | Basic strings and nature-region reason codes exist; broad contrastive and counterfactual evidence does not |
| Natural-language parser | PROPOSED | No parser model, parser benchmark, or semantic confirmation flow exists |
| Request compiler and solver-update allow list | PROPOSED | No canonical compiler prevents arbitrary semantic-to-math mappings |
| Independent evaluator and hard eligibility gate | PROPOSED | Dashboard export validation checks assets, not complete itinerary validity |
| Minimal-change repair | PROPOSED | Alternatives exist but no declared parent/child repair objective and intent metric contract exists |
| Branch-and-check/restricted repair | PROPOSED | No master/checker or conflict-set implementation exists |
| User correction or inspectability study | PROPOSED | No ethics-approved study or participant logging exists |

### A.2 P0 publication blockers

| ID | Problem | Consequence | Required correction |
|---|---|---|---|
| P0-01 | Required anchors can be inserted or substituted after route generation | Displayed content can differ from the solved model | Move anchor behavior into planning or create a new child run, plan, and evaluation |
| P0-02 | Requested and executed method are not canonical separate fields | Fallback can be presented as solver-backed | Add truthful method and fallback lineage |
| P0-03 | Planner attempts and plans share ad hoc tables | Failed/no-plan runs cannot be represented consistently | Introduce `PlannerRun`; create `PlanArtifact` only when a route exists |
| P0-04 | No independent complete-plan evaluator | Planner or export summaries can be mistaken for validity | Recompute final checks independently and gate eligibility |
| P0-05 | Plans can change without canonical content lineage | Status may refer to older content | Content-hash immutable records and parent-child derivation |
| P0-06 | LLM or unstructured interpretation boundary is unspecified | Future language output could control solver behavior | Add typed parser, validation, confirmation, and deterministic compiler |
| P0-07 | Hard/soft preference semantics are not canonical | Vague language can create false precision or wrong constraints | Use explicit levels, consequence rules, and user confirmation |
| P0-08 | Explanations are not universally evidence-linked | Plausible text can be unfaithful | Require IDs, reason codes, evaluator findings, and counterfactual runs |
| P0-09 | Solver feasibility, evaluation, and real-world validity can be conflated | Claims exceed evidence | Separate status dimensions and use conditional guarantee language |

### A.3 Parser and semantic risks

| ID | Failure | Class | Required behavior |
|---|---|---|---|
| PAR-01 | Invalid JSON/schema/type/unit | Syntactic/executable | Reject or deterministically repair when low consequence; log lineage |
| PAR-02 | Unknown or nonexistent POI | Syntactic/grounding | Reject; do not create a candidate or constraint |
| PAR-03 | Unsupported effect or mathematical expression | Syntactic/authorization | Reject at compiler boundary |
| PAR-04 | Reversed ordering dependency | Semantic | Require confirmation; report consequence |
| PAR-05 | Soft preference inferred as hard constraint | Semantic/outcome-sensitive | Require confirmation; never auto-apply |
| PAR-06 | Sentence order treated as visit order | Semantic | Do not infer dependency without linguistic evidence or confirmation |
| PAR-07 | Vague phrase mapped to precise coefficient | Semantic/false precision | Map to visible ordinal level and range; permit manual adjustment |
| PAR-08 | Wrong date, location, entity, or scope | Semantic | Require correction/confirmation or defer |
| PAR-09 | Correct syntax but misunderstood must-go/avoid | Semantic/high consequence | Block planning until confirmed |
| PAR-10 | Solver makes malformed formulation executable | Semantic | Executability does not establish correctness; preserve semantic review |

### A.4 Preserved safety and research safeguards

- no raw text or embeddings directly control feasibility;
- no arbitrary LLM-generated Gurobi code or expressions;
- no unsupported safety labels or incident-derived hard exclusion;
- missing context is unknown, not favorable;
- provider provenance, freshness, coverage, license, and failure are explicit;
- fallback is a separate run or child run, not a rewritten status;
- a feasible plan cannot rank before independent evaluation;
- solver certification cannot override evaluation failure;
- paid providers require explicit human approval;
- participant evidence requires the applicable ethics determination.

---

## B. Context, Provider, and Geographic Scope

### B.1 First-slice context contract

| Context | First-slice role | Allowed effects | Required evidence |
|---|---|---|---|
| Weather | Primary differentiating context | Soft exposure/burden; authoritative closure candidate only when separately confirmed | Forecast timestamp, interval, geography, units, confidence/uncertainty, source |
| Operational closure | Hard feasibility context | Remove availability or update time window only from authoritative evidence | Entity match, authority, effective interval, retrieval time |
| Mobility/travel-time disruption | Route context | Travel-time range, edge/mode availability, warning | Edge/mode match, interval, authority/confidence, frozen baseline |
| User-approved pace/accessibility | Preference or hard limit according to explicit choice | Daily time, effort, mode, service-time, repair-scope settings | Direct user confirmation and versioned mapping |
| Nature | Intent and utility context | Interest, substitution ontology, exposure sensitivity, must-go policy | Candidate provenance and declared traveler preference |

Other families remain in the schema and registry as `STAGED_EXTENSION`. They do not enter first-paper claims without operational adapters, tests, and experiments.

### B.2 Provider selection hierarchy

1. open government/public source;
2. open community source with acceptable provenance/license;
3. free commercial tier with acceptable terms;
4. paid provider only after written budget and license approval.

Reject or defer when provenance, time/geography, missingness, license, cost, or snapshot behavior cannot be represented truthfully.

### B.3 `ProviderRecord`

```text
ProviderRecord:
    provider_id
    provider_version
    display_name
    context_families
    subtypes
    evidence_class
    geographic_coverage
    temporal_coverage
    expected_freshness
    authentication_class
    rate_limit_policy
    cost_policy
    source_license
    redistribution_policy
    retention_policy
    privacy_classification
    known_biases
    missingness_semantics
    normalization_adapter
    failure_policy
    fallback_provider_ids
    snapshot_policy
    approval_status
    schema_version
    content_hash
```

### B.4 Evidence and coverage states

```text
EvidenceClass:
    AUTHORITATIVE_ORDER
    AUTHORITATIVE_ADVISORY
    OBSERVED_MEASUREMENT
    FORECAST
    COMMERCIAL_INVENTORY
    ESTIMATE
    HISTORICAL_PROXY
    USER_CONFIRMED
    MODEL_INFERRED
    UNKNOWN

CoverageState:
    SUPPORTED_PRIMARY
    SUPPORTED_REGIONAL
    SUPPORTED_PROXY
    UNSUPPORTED
    UNKNOWN_PENDING_AUDIT
```

`SUPPORTED_PROXY` is never displayed as primary or real-time support. Unknown coverage cannot participate in ranking.

### B.5 Geographic maturity levels

1. **Schema-level capability:** fields and status behavior exist.
2. **Adapter availability:** a provider path exists and passes contract tests.
3. **Audited provider coverage:** jurisdiction and subtype coverage have been sampled and recorded.
4. **Experimentally validated scope:** frozen scenarios and results exist for a declared corridor.

The first slice requires California and optionally one additional reproducible corridor. A 50-state plus D.C. audit and three-corridor benchmark remain long-term milestones. The phrase **nationwide-capable architecture with audited gaps** is permitted only after schema and registry behavior exist; **nationwide support** is prohibited without coverage evidence.

### B.6 Snapshot requirements

Every acquisition records provider/adapter version, request parameters without secrets, provider and retrieval time, timezone, coordinate reference system, raw and normalized hashes, evidence/coverage class, license/retention, failure/fallback state, conflict references, and quality warnings.

---

## C. Canonical Contracts

### C.1 `ParsedTravelRequest`

```text
ParsedTravelRequest:
    request_id
    original_text_hash
    entities
    trip_dates
    start_location
    end_location
    hard_constraints
    soft_preferences
    conditional_preferences
    must_visit
    avoid
    ordering_dependencies
    time_limits
    pace_level
    physical_effort_level
    driving_tolerance_level
    weather_tolerance_level
    budget_level
    ambiguity_flags
    confidence_by_field
    requires_confirmation
    user_confirmation_status
    parser_model_version
    parser_prompt_version
    schema_version
    parent_request_id
    content_hash
```

Invariants:

- hard constraints identify their explicit linguistic or user-confirmed source;
- sentence order alone never creates `ordering_dependencies`;
- ordinal levels are stored instead of invented floating-point weights;
- unknown entities remain unresolved rather than silently created;
- original text is retained only according to privacy policy; the canonical contract always stores its hash.

### C.2 `ParserRun`

```text
ParserRun:
    parser_run_id
    request_id
    original_text_hash
    parser_model_id
    parser_model_version
    prompt_version
    decoding_parameters
    raw_output_hash
    parsed_request_id
    parser_execution_status
    schema_validation_status
    semantic_validation_status
    ambiguity_count
    unsupported_fact_count
    started_at
    completed_at
    schema_version
    content_hash
```

### C.3 `ParserCorrection`

```text
ParserCorrection:
    correction_id
    parser_run_id
    original_parsed_request_id
    validator_feedback_codes
    validator_feedback_payload_hash
    corrected_parsed_request_id
    correction_type
    consequence_level
    automatic_repair_allowed
    approval_status
    approved_by_role
    created_at
    schema_version
    content_hash
```

`correction_type` is `SYNTACTIC`, `GROUNDING`, `SEMANTIC`, or `OUTCOME_SENSITIVE`. Only deterministic low-consequence syntactic/grounding corrections may be automatic.

### C.4 `RequestValidationReport`

```text
RequestValidationReport:
    validation_id
    parsed_request_id
    schema_errors
    type_errors
    unit_errors
    entity_grounding_results
    temporal_scope_results
    spatial_scope_results
    hard_soft_classification_results
    dependency_direction_results
    false_precision_flags
    unsupported_effect_flags
    semantic_conflicts
    consequence_by_field
    requires_confirmation
    confirmation_reasons
    validation_status
    validator_version
    schema_version
    content_hash
```

### C.5 `ContextObservation`

```text
ContextObservation:
    observation_id
    context_family
    context_subtype
    entity_ids
    value
    units
    spatial_scope
    temporal_scope
    evidence_class
    provider_id
    source_record_id
    event_time
    retrieval_time
    freshness_seconds
    severity
    confidence
    uncertainty
    coverage_state
    license_class
    raw_data_hash
    normalized_data_hash
    warnings
    schema_version
    content_hash
```

Missing is distinct from zero, false, open, or unavailable. Geography, time, and units are never implicit.

### C.6 `ContextInterpretation`

```text
ContextInterpretation:
    interpretation_id
    observation_ids
    source_text_hashes
    interpreter_type
    model_id
    model_version
    prompt_version
    extracted_entities
    interpreted_family
    interpreted_subtype
    interpreted_spatial_scope
    interpreted_temporal_scope
    proposed_effects
    confidence_by_field
    ambiguity_flags
    conflict_flags
    grounding_results
    schema_validation_results
    semantic_validation_results
    requires_confirmation
    confirmation_status
    schema_version
    content_hash
```

### C.7 `SolverUpdate`

```text
SolverUpdate:
    update_id
    interpretation_id
    planning_request_id
    effect_type
    target_entities
    parameter_name
    previous_value
    proposed_value
    units
    authority_rule
    allow_list_rule_id
    validation_results
    consequence_level
    confirmation_status
    applied
    rejection_reason
    compiler_version
    schema_version
    content_hash
```

### C.8 `CompilationRecord`

```text
CompilationRecord:
    compilation_id
    parsed_request_id
    request_validation_id
    compiler_version
    preference_mapping_version
    planning_request_id
    planner_specification_id
    generated_update_ids
    rejected_field_paths
    unsupported_field_paths
    compilation_status
    warnings
    schema_version
    content_hash
```

Only this deterministic compiler maps semantic request fields to planning fields, parameter ranges, objective terms, or constraints.

### C.9 `PlanningRequest`

```text
PlanningRequest:
    planning_request_id
    parsed_request_id
    compilation_id
    scenario_id
    trip_dates
    start_location
    end_location
    trip_days
    traveler_profile
    interest_levels
    must_visit
    avoid
    ordering_dependencies
    total_budget
    daily_time_limits
    driving_tolerance_level
    weather_tolerance_level
    pace_level
    physical_effort_level
    preference_flexibility
    candidate_snapshot_id
    context_snapshot_id
    user_confirmation_status
    schema_version
    content_hash
```

### C.10 `CandidateSnapshot` and `ContextSnapshot`

```text
CandidateSnapshot:
    candidate_snapshot_id
    poi_records
    lodging_records
    travel_time_matrix
    duration_records
    cost_records
    opening_hour_records
    source_records
    unsupported_fields
    generated_at
    frozen_at
    schema_version
    content_hash

ContextSnapshot:
    context_snapshot_id
    scenario_id
    provider_registry_version
    coverage_audit_version
    observation_ids
    interpretation_ids
    approved_update_ids
    rejected_update_ids
    conflict_records
    unsupported_families
    created_at
    frozen_at
    schema_version
    content_hash
```

### C.11 `PlannerSpecification`

```text
PlannerSpecification:
    planner_specification_id
    planner_id
    planner_version
    objective_terms
    hard_constraints
    soft_constraints
    parameter_ranges
    preference_mapping_version
    solver_update_allow_list_version
    time_limit_seconds
    mip_gap_target
    random_seed
    fallback_policy_id
    schema_version
    content_hash
```

### C.12 `PlannerRun`

```text
PlannerRun:
    run_id
    parent_run_id
    planning_request_id
    candidate_snapshot_id
    context_snapshot_id
    planner_specification_id
    method_requested
    method_executed
    execution_status
    solver_certification
    solver_backend
    solver_version
    model_fingerprint
    applied_update_ids
    rejected_update_ids
    fallback_reason
    solver_status_raw
    objective_value
    best_bound
    optimality_gap
    runtime_seconds
    result_plan_id
    conflict_set_ids
    error_class
    error_summary
    run_log_hash
    started_at
    completed_at
    schema_version
    content_hash
```

No-plan outcomes have no `result_plan_id`. Fallback creates a child run or explicit linked run.

### C.13 `PlanArtifact`

```text
PlanArtifact:
    plan_id
    parent_plan_id
    source_run_id
    planning_request_id
    candidate_snapshot_id
    context_snapshot_id
    selected_stops
    day_assignments
    sequence
    lodging_assignments
    cost_components
    modeled_metrics
    context_exposure_components
    change_components
    created_at
    schema_version
    content_hash
```

Post-processing may not alter a frozen plan. A material change creates a new run and plan.

### C.14 `EvaluationReport`

```text
EvaluationReport:
    evaluation_id
    evaluator_version
    plan_id
    source_run_id
    artifact_grounding_status
    hard_feasibility_status
    soft_quality_status
    preference_fit_status
    robustness_status
    evaluation_status
    blocking_warning_count
    nonblocking_warning_count
    comparison_eligibility
    artifact_metrics
    hard_metrics
    soft_metrics
    preference_metrics
    robustness_metrics
    component_uncertainty
    conditional_reward
    reward_eligibility
    rule_metric_ids
    llm_judged_metric_ids
    sensitivity_references
    evaluated_at
    schema_version
    content_hash
```

### C.15 `ExplanationEvidence`

```text
ExplanationEvidence:
    explanation_id
    plan_id
    explanation_type
    explanation_aim
    explanation_scope
    explanation_method
    target_entity
    parsed_request_field_paths
    observation_ids
    interpretation_ids
    solver_update_ids
    reason_codes
    constraint_slack_fields
    evaluator_findings
    compared_alternative_id
    counterfactual_run_id
    deferral_reason_codes
    rendering_template_id
    grounding_status
    schema_version
    content_hash
```

### C.16 `StudyEvent`

```text
StudyEvent:
    event_id
    participant_pseudonym
    session_id
    condition
    scenario_id
    task_id
    event_type
    target_id
    parsed_request_id
    plan_id
    parent_plan_id
    run_id
    correction_id
    client_timestamp
    server_timestamp
    sequence_number
    interface_version
    minimized_payload
```

### C.17 Canonical hashing

Use UTF-8, deterministic key order, stable numeric/datetime representations, explicit nulls, a versioned hash algorithm, and parent-child cycle checks. Any semantic or plan-content change must change the hash.

---

## D. Status and Eligibility Architecture

### D.1 Parser and semantic statuses

```text
ParserExecutionStatus:
    PARSER_SUCCEEDED
    PARSER_FAILED
    PARSER_TIMEOUT
    PARSER_NOT_RUN

SchemaValidationStatus:
    SCHEMA_VALID
    SCHEMA_REPAIRED_AUTOMATICALLY
    SCHEMA_INVALID

SemanticValidationStatus:
    SEMANTIC_CONFIRMED
    SEMANTIC_REQUIRES_CONFIRMATION
    SEMANTIC_CONFLICT
    SEMANTIC_REJECTED
    SEMANTIC_NOT_EVALUATED

ConfirmationStatus:
    NOT_REQUIRED
    PENDING_USER
    CONFIRMED_BY_USER
    CORRECTED_AND_CONFIRMED
    REJECTED_BY_USER
    DEFERRED

CompilationStatus:
    COMPILED
    COMPILED_WITH_WARNINGS
    COMPILATION_BLOCKED
    UNSUPPORTED_REQUEST
```

### D.2 Planner statuses

```text
ExecutionStatus:
    OPTIMAL
    FEASIBLE_GAP
    TIME_LIMIT_WITH_INCUMBENT
    INFEASIBLE
    NO_SOLUTION
    SOLVER_ERROR
    HEURISTIC
    HEURISTIC_AFTER_SOLVER_FAILURE
    PREVIEW_ONLY

SolverCertification:
    OPTIMALITY_CERTIFIED
    FEASIBILITY_CERTIFIED
    INCUMBENT_FEASIBILITY_CERTIFIED
    INFEASIBILITY_CERTIFIED
    NO_SOLVER_CERTIFICATE
    CERTIFICATE_UNAVAILABLE
    NOT_APPLICABLE
```

### D.3 Evaluation and eligibility

```text
EvaluationStatus:
    NOT_EVALUATED
    EVALUATION_PASSED
    EVALUATION_PASSED_WITH_WARNINGS
    EVALUATION_FAILED
    EVALUATION_ERROR

ComparisonEligibility:
    ELIGIBLE_CERTIFIED
    ELIGIBLE_EVALUATED_UNCERTIFIED
    INELIGIBLE_PARSER_OR_CONFIRMATION
    INELIGIBLE_COMPILATION
    INELIGIBLE_GROUNDING
    INELIGIBLE_HARD_FEASIBILITY
    INELIGIBLE_EVALUATION_ERROR
    INELIGIBLE_BLOCKING_WARNING
    INELIGIBLE_NOT_EVALUATED
    INELIGIBLE_NO_PLAN
    PREVIEW_NOT_RANKABLE
```

### D.4 Eligibility derivation

```python
if preview_only:
    eligibility = "PREVIEW_NOT_RANKABLE"
elif confirmation_status not in {"NOT_REQUIRED", "CONFIRMED_BY_USER", "CORRECTED_AND_CONFIRMED"}:
    eligibility = "INELIGIBLE_PARSER_OR_CONFIRMATION"
elif compilation_status not in {"COMPILED", "COMPILED_WITH_WARNINGS"}:
    eligibility = "INELIGIBLE_COMPILATION"
elif no_plan_artifact:
    eligibility = "INELIGIBLE_NO_PLAN"
elif evaluation_status == "EVALUATION_ERROR":
    eligibility = "INELIGIBLE_EVALUATION_ERROR"
elif artifact_grounding_status != "PASSED":
    eligibility = "INELIGIBLE_GROUNDING"
elif hard_feasibility_status != "PASSED":
    eligibility = "INELIGIBLE_HARD_FEASIBILITY"
elif blocking_warning_count > 0:
    eligibility = "INELIGIBLE_BLOCKING_WARNING"
elif evaluation_status not in {"EVALUATION_PASSED", "EVALUATION_PASSED_WITH_WARNINGS"}:
    eligibility = "INELIGIBLE_NOT_EVALUATED"
elif solver_certification in {"OPTIMALITY_CERTIFIED", "FEASIBILITY_CERTIFIED", "INCUMBENT_FEASIBILITY_CERTIFIED"}:
    eligibility = "ELIGIBLE_CERTIFIED"
else:
    eligibility = "ELIGIBLE_EVALUATED_UNCERTIFIED"
```

### D.5 Cross-field invariants

- valid JSON does not imply semantic confirmation;
- semantic confirmation does not imply compilation success;
- compilation success does not imply a feasible plan;
- solver certification does not imply evaluation pass or real-world validity;
- soft/preference scores are absent or `NOT_APPLICABLE` for ineligible hard-invalid plans;
- a unified reward is absent unless `reward_eligibility` passes;
- a fallback never inherits a failed solver certificate;
- a modified plan never inherits an earlier evaluation;
- preview artifacts never enter experiments or studies.

---

## E. Validation, Compilation, and Solver Boundaries

### E.1 Validation sequence

1. Validate JSON/schema and field types.
2. Normalize units, dates, coordinates, and timezone.
3. Resolve entities against frozen candidates and source records.
4. Validate temporal and spatial scope.
5. Classify hard, soft, and conditional fields.
6. Validate dependency direction and evidence.
7. Detect false precision and unsupported facts/effects.
8. assign consequence level by field.
9. determine automatic repair, confirmation, rejection, or deferral.
10. freeze the confirmed `ParsedTravelRequest`.
11. compile through the versioned allow list.
12. record accepted/rejected fields and generated updates.

### E.2 Automatic repair policy

Automatic repair is allowed only when deterministic, low consequence, and meaning-preserving, for example canonical enum casing, an unambiguous unit conversion, or an entity ID resolved from an exact alias. The original output, validator feedback, correction, and status remain stored.

Never automatically repair hard/soft classification, dependency direction, must-go/avoid, date/location scope, user tolerance, broad repair scope, or any change that materially alters feasible plans.

### E.3 Preference-level mapping

```text
NatureImportance: LOW | MEDIUM | HIGH
DrivingSensitivity: LOW | MEDIUM | HIGH
WeatherTolerance: CAUTIOUS | BALANCED | FLEXIBLE
PaceLevel: RELAXED | MODERATE | COMPACT
PhysicalEffortLevel: LOW | MODERATE | HIGH
BudgetLevel: BUDGET | STANDARD | PREMIUM
```

Each level maps to a versioned parameter interval rather than one falsely precise value. The compiler chooses the declared default inside the interval for a run; sensitivity analysis varies the interval. The mapping and chosen value are visible and manually adjustable.

### E.4 Compiler allow list

Permitted effect families include candidate inclusion/exclusion when explicitly confirmed, availability/time-window updates, travel-time ranges, network edge/mode availability, service-duration ranges, cost ranges, soft exposure/burden terms, user-approved tolerance, repair scope, warning, confirmation, and deferral.

The compiler rejects arbitrary source code, variable definitions, constraint expressions, unknown effect types, ungrounded entities, out-of-range values, missing units, and hidden objective weights.

### E.5 Direct LLM-to-solver prohibition

Architecture tests must prove:

- the parser package cannot import or call Gurobi/planner entry points;
- parser output cannot instantiate `PlanningRequest` directly;
- only a confirmed request and successful `CompilationRecord` can create a planner input;
- provider adapters cannot apply solver updates;
- LLM-generated explanations cannot modify plans or statuses;
- solver/evaluator feedback passes through typed correction records.

### E.6 Context effect policy

| Context | Hard effects | Soft effects | Confirmation |
|---|---|---|---|
| Weather | Authoritative separately confirmed closure only | Exposure/burden | Must-go or broad-scope effect |
| Operational closure | Confirmed unavailability/time window | Uncertainty warning | Conflicting or unstructured evidence |
| Mobility | Confirmed edge/mode closure | Delay/travel-time range | Broad route-scope expansion |
| Pace/accessibility | Explicit user-confirmed limit | User-confirmed preference | Always for newly interpreted consequential state |
| Reported incidents | Official legal closure only, never incident-derived | Optional warning/soft term | Always for material ranking effect |

Reported incidents remain disabled by default for the first slice and never support safe/unsafe labels.

---

## F. Branch-and-Check and Restricted Repair

### F.1 Conceptual decomposition

**Master selection problem:** choose bases, cities, days, hotels, regions, or POI bundles under budget, category, must-go, exclusion, and high-level preference constraints.

**Detailed checker/restricted solver:** schedule selected POIs under travel time, visit duration, opening hours, daily limits, dependencies, and return requirements; optionally recover the best feasible subset.

### F.2 `FeasibilityCheckReport`

```text
FeasibilityCheckReport:
    check_id
    source_run_id
    selected_bundle_ids
    checker_version
    check_status
    temporal_feasibility
    travel_feasibility
    opening_hour_feasibility
    dependency_feasibility
    daily_limit_feasibility
    return_requirement_feasibility
    conflict_set_ids
    lower_bounds
    maximum_feasible_cardinality_bound
    runtime_seconds
    schema_version
    content_hash
```

### F.3 `ConflictSet`

```text
ConflictSet:
    conflict_set_id
    check_id
    member_entities
    constraint_ids
    conflict_type
    minimality_status
    evidence_fields
    suggested_relaxation_types
    schema_version
    content_hash
```

### F.4 `RepairProposal`

```text
RepairProposal:
    repair_proposal_id
    parent_plan_id
    source_conflict_set_ids
    repair_scope
    removed_entities
    moved_entities
    substituted_entities
    changed_constraints
    required_user_relaxations
    expected_metric_changes
    restricted_run_id
    evaluation_id
    confirmation_status
    schema_version
    content_hash
```

### F.5 Staged mechanism

1. **Baseline check:** complete-bundle feasibility and simple infeasible-set cut.
2. **Enhanced check:** pairwise incompatibility, duration lower bounds, smaller conflicts, stronger cardinality bounds, and infeasible partial-selection pruning.
3. **Branch-and-solve/restricted repair:** recover the best feasible subset, create a truthful child run, and independently evaluate it.
4. **Human-facing evidence:** explain which conflict caused removal and show minimal repairs such as more time, changed base, removed POI, or relaxed soft preference.

The first slice may stop after stages 1 and 3. It must not claim enhanced branch-and-check if only a complete-bundle checker exists.

### F.6 Branch/check experiments

- baseline versus enhanced conflict detection;
- number and size of cuts/conflict sets;
- checker runtime and master nodes pruned;
- feasible-subset quality and cardinality;
- comparison with full replanning and greedy removal;
- explanation correctness from conflict evidence;
- sensitivity to opening-hour and travel-time uncertainty.

---

## G. Independent Layered Evaluator

### G.1 Independence rule

The evaluator may share contracts, constants, and frozen data with the planner. It may not import planner objectives, planner totals, selection logic, fallback choices, or explanation prose. It recomputes from the final artifact.

### G.2 Group 1 - Artifact and grounding validity

- schema and ID validity;
- all POIs, hotels, modes, times, and sources resolve;
- names and descriptions match source records;
- no hallucinated entity or unsupported field;
- parser, compiler, update, run, plan, and source lineage resolve;
- provenance, freshness, coverage, license, and snapshot hashes pass.

Failure blocks all soft/preference credit.

### G.3 Group 2 - Hard and commonsense feasibility

- chronological order and no overlap;
- location and route continuity;
- opening/availability windows when required;
- travel transitions and no activity during transportation;
- complete overnight lodging and intercity transportation;
- budget and daily-time limits;
- dependency direction and required waypoint satisfaction;
- closure and mode availability;
- complete required travel-time edges.

Failure makes the plan ineligible.

### G.4 Group 3 - Soft itinerary quality

- schedule density and daytime utilization;
- geographic clustering and unnecessary detour;
- hotel/base consistency;
- attraction uniqueness, diversity, and high-priority coverage;
- context burden by family;
- retained utility;
- minimal-change repair quality;
- route coherence and pacing.

### G.5 Group 4 - Personal preference fit

- pace and physical effort;
- budget style;
- category interests and nature intent;
- driving and weather tolerance;
- must-go retention and avoid compliance;
- conditional preference satisfaction;
- free-form request fulfillment against confirmed fields.

### G.6 Hard gating and conditional reward

```python
if artifact_grounding_status != "PASSED":
    eligible = False
elif hard_feasibility_status != "PASSED":
    eligible = False
elif blocking_warning_count > 0:
    eligible = False
else:
    eligible = True

conditional_reward = aggregate(soft_metrics, preference_metrics) if eligible else None
```

The conditional reward is versioned, transparent, and secondary to component metrics. It may support ranking, sensitivity analysis, or later training only after eligibility. Reinforcement learning is not a first-paper deliverable.

### G.7 Rule versus LLM judgment

Every metric declares `RULE_BASED`, `LLM_JUDGED`, `EXPERT_ANNOTATED`, or `HYBRID`. Rule checks are preferred for identity, chronology, arithmetic, continuity, and declared constraints. LLM judgment is limited to semantic quality where a deterministic rule is inadequate, and stores model/prompt/version, uncertainty, and calibration results.

### G.8 Evaluator validation

- expert pairwise plan comparisons;
- inter-rater and evaluator-expert agreement;
- confidence intervals and disagreement analysis;
- holdout scenarios and perturbation tests;
- rank stability under weight changes;
- adversarial plans that maximize reward while violating intent;
- ablation of LLM-judged and soft components;
- checks that ineligible plans never outrank eligible plans.

---

## H. Explanation and Correction Requirements

### H.1 Explanation design contract

Every explanation specifies:

- **aim:** transparency, scrutability, decision support, correction, or uncertainty communication;
- **scope:** input, process, output, comparison, or counterfactual;
- **method:** direct rule/constraint evidence, evaluator evidence, feature contribution, conflict set, or generated summary;
- **format:** text, map, timeline, bar/contribution chart, constraint panel, evidence view, diff, or dependency graph.

### H.2 Input explanation

Show what the system believes about the traveler, which fields are hard/soft/conditional, unresolved ambiguity, confidence by field, level-to-parameter mapping, and confirmation state.

### H.3 Process explanation

Show context observations that affected the model, accepted/rejected updates, compiler mapping, objective/constraint influence, solver and fallback status, evaluator status, and deferral decisions.

### H.4 Output explanation types

| Type | Required evidence |
|---|---|
| Why selected | Preference field, utility contribution, feasibility placement, compared eligible alternative |
| Why skipped | Validated reason code, conflict/constraint evidence, alternative status |
| Why changed | Parent/child diff, disruption evidence, repair scope, metric change |
| What would change | Reproducible counterfactual run and evaluation, not speculative prose |
| Why deferred | Ambiguity/conflict/consequence rule and missing confirmation/evidence |

### H.5 Layered presentation

- Level 1: concise plain-language summary.
- Level 2: major tradeoffs, evidence, constraints, slack, and uncertainty.
- Level 3: full source, parser, correction, compiler, update, run, evaluation, and counterfactual lineage.

### H.6 Visualization mapping

- map: spatial sequence, detour, alternatives;
- timeline: chronology, opening windows, travel/visit allocation;
- bars/contribution chart: quantitative tradeoffs;
- constraint panel: limits, usage, slack, and binding status;
- linked evidence view: source-to-update-to-plan lineage;
- side-by-side diff: original versus repaired itinerary;
- node-link view: ordering/dependency or conflict relationships only;
- badges: stale, proxy, conflicting, unsupported, unconfirmed, fallback.

### H.7 Explanation evaluation

Do not claim that more explanation increases trust. Measure detection, correction, constraint comprehension, counterfactual accuracy, evidence use, revision quality, workload, and appropriate/calibrated reliance.

---

## I. Experiment and Study Designs

### I.1 Parser benchmark

Include clear requests, vague deadlines, ambiguous modifiers, nonexistent POIs, reversed dependencies, sentence-order traps, hard/soft ambiguity, conditional preferences, correction turns, and requests that should defer.

Metrics:

- schema-valid output rate;
- field-level exact match or typed tolerance;
- entity extraction F1;
- hard-versus-soft classification accuracy;
- dependency-direction accuracy;
- temporal/spatial scope accuracy;
- unsupported-fact rate;
- consequence-weighted interpretation error;
- confirmation precision/recall;
- correction locality: unaffected fields unchanged;
- end-to-end plan degradation from parser errors.

### I.2 Gold-parser versus predicted-parser planning

Run identical scenarios with gold confirmed requests and predicted requests. Attribute outcome degradation to entity, constraint class, dependency, scope, or level-mapping errors. A feasible predicted-parser plan is still counted as an interpretation failure when it solves the wrong request.

### I.3 LLM-only versus solver-backed planning

Only when the language path is implemented, compare direct LLM route generation with parser-plus-solver and gold-plus-solver using identical candidates and context. Report hard feasibility, hallucinations, utility, travel, preference fit, runtime, and failure causes.

### I.4 Context-blind versus context-aware repair

Methods:

1. context-blind solver;
2. deterministic context-aware heuristic;
3. solver-backed minimal-change repair;
4. full replanning.

Scenarios cover weather burden, authoritative closure, mobility delay, and user-approved pace/accessibility. Compare feasibility, exposure, utility, travel, cost, preserved stops, sequence edit distance, lodging/base changes, runtime, and repair scope.

### I.5 Baseline versus enhanced checker

Compare complete-bundle check, enhanced conflict extraction, restricted repair, greedy removal, and full replanning. Report detection, conflict minimality/size, runtime, quality of recovered subset, and explanation correctness.

### I.6 Original versus repaired itinerary

Each pair stores parent/child lineage and reports retained stops, weighted intent, must-go retention, semantic substitution, day/time/order changes, travel/cost delta, context burden reduction, and eligibility.

### I.7 Evaluator validation

Construct eligible high-quality, eligible poor-quality, ineligible attractive, hallucinated, preference-misaligned, and reward-gaming plans. Obtain expert pairwise judgments, report agreement and uncertainty, and test rank stability.

### I.8 Explanation and correction study

Use identical precomputed plans across basic and inspectable conditions. Primary outcomes are interpretation/repair error detection and correction accuracy. Secondary outcomes include constraint comprehension, appropriate acceptance/rejection, revision quality, decision time, workload, perceived control, and calibrated reliance.

### I.9 Robustness and sensitivity

Vary weather/context values, freshness/confidence, travel times, visit durations, level mappings, change penalties, solver time/gap, evaluator weights, missingness, and conflict policies. Report eligibility changes, rank reversals, and failure regions.

---

## J. Staged Implementation Backlog

### J.1 Phase 0 - Truthful research foundation

| Task | Priority | Output |
|---|---:|---|
| Remove post-solve mutation from publication path | P0 | Planner-owned behavior or child run/evaluation |
| Add `PlannerRun` and immutable `PlanArtifact` | P0 | Truthful artifacts |
| Separate requested/executed method and fallback | P0 | Canonical lineage |
| Freeze candidates, context, travel, and specifications | P0 | Snapshot contracts |
| Build independent evaluator skeleton and eligibility | P0 | Hard-gated report |
| Add canonical hashing and artifact roles | P0 | Reproducible evidence |

### J.2 Phase 1 - Structured request and context contracts

Depends on Phase 0 contract acceptance.

- implement `ParsedTravelRequest`, `ParserRun`, correction, and validation report;
- implement observations, interpretations, updates, and snapshots;
- implement ordinal preference mappings;
- implement deterministic compiler and allow list;
- implement confirmation/deferral statuses;
- add direct LLM-to-solver architecture tests.

### J.3 Phase 2 - Narrow context-aware planning slice

Depends on Phases 0-1.

- freeze weather, closure, mobility, and user-approved pace/accessibility scenarios;
- implement context-blind solver baseline;
- implement deterministic heuristic baseline;
- implement allow-listed context-aware solver updates;
- implement local/day/remainder repair scope and minimal-change terms;
- produce parent/child plan diffs.

### J.4 Phase 3 - Solver-backed conflict and repair

Depends on Phase 2.

- complete-bundle checker and simple infeasible cut;
- conflict-set contract and extraction;
- restricted best-feasible-subset solver;
- enhanced pairwise/lower-bound/cardinality cuts if schedule permits;
- human-readable repair alternatives.

### J.5 Phase 4 - Evaluation and explainability

Depends on Phases 0-3.

- complete four evaluator groups and hard gate;
- implement conditional reward only for eligible plans;
- add original-versus-repaired view;
- add why-selected/skipped/changed/deferred evidence;
- implement counterfactual runs;
- version explanation templates and evaluator metrics.

### J.6 Phase 5 - Experiments and venue decision

Depends on Phases 0-4.

- parser benchmark if parser is in scope;
- planning and repair baselines;
- checker comparison;
- robustness/sensitivity and Goodhart tests;
- expert pairwise evaluator validation;
- expert transportation inspection or ethics-approved interface study;
- evidence-driven TRB versus IUI/CHI decision.

### J.7 Phase 6 - IUI or CHI extension

- mixed-initiative interpretation and correction;
- confirmation and deferral interface;
- controlled explanation/correction study;
- calibrated-reliance analysis;
- participant logging and reproducibility.

### J.8 Phase 7 - Long-term agent

- continuous monitoring;
- disruption detection;
- proactive but non-coercive notification;
- repair proposal;
- user-authorized adoption;
- no automatic booking, purchasing, cancellation, or financial action without separate safeguards.

---

## K. Test, Artifact, and Reproducibility Policy

### K.1 Parser and correction tests

- malformed schema/type/unit;
- vague deadline and modifier;
- nonexistent/ambiguous POI;
- reversed dependency;
- sentence order mistaken for visit order;
- hard/soft confusion;
- valid JSON with incorrect meaning;
- false precision rejected;
- correction changes only intended fields;
- semantic change requires confirmation;
- automatic repair limited to deterministic low-consequence cases.

### K.2 Compiler and architecture tests

- parser cannot import/call solver;
- only confirmed request compiles;
- arbitrary expression/code rejected;
- unsupported field/effect fails closed;
- parameter level mapping is versioned and bounded;
- provider cannot apply update;
- rejected update has no plan effect;
- no LLM output bypasses validation/compiler boundary.

### K.3 Planner, conflict, and repair tests

- solver success, time limit, infeasibility, error, and fallback lineage;
- post-solve mutation creates new run/plan/evaluation;
- narrow repair preserves unaffected days;
- broad repair requires authorization;
- complete-bundle infeasibility yields conflict evidence;
- restricted repair returns evaluated child plan;
- conflict explanation resolves to constraints and entities;
- heuristic route carries no solver certificate.

### K.4 Evaluator tests

- hallucinated entity fails grounding;
- chronology, travel overlap, opening hour, route continuity, lodging, budget, and time failures block eligibility;
- ineligible plan receives no soft/preference reward;
- rule and LLM-judged metrics remain separated;
- conditional reward cannot replace components;
- weight changes and adversarial Goodhart cases are detected;
- identical plans receive identical evaluator metrics across planners;
- feasible solver result can fail independent evaluation.

### K.5 Explanation and deferral tests

- every explanation resolves to evidence IDs;
- why-selected compares an eligible alternative;
- why-skipped uses a conflict or validated reason;
- why-changed resolves to parent/child diff;
- what-would-change links a counterfactual run;
- why-deferred cites ambiguity/conflict/consequence policy;
- unsupported causal language fails;
- stale/proxy/conflicting evidence remains visible.

### K.6 Artifact roles and directory layout

```text
data/snapshots/<snapshot_id>/
results/previews/
results/demonstrations/
results/experiments/<experiment_id>/
results/study_evidence/<study_version>/
results/debug/
results/legacy/
```

Participant studies use frozen `STUDY_EVIDENCE`; publication results use eligible `RESEARCH_EXPERIMENT` or `STUDY_EVIDENCE` artifacts only.

### K.7 Experiment folder

```text
manifest.json
environment_manifest.json
provider_registry.json
parsed_travel_request.json
parser_run.json
parser_corrections.jsonl
request_validation_report.json
compilation_record.json
candidate_snapshot_manifest.json
context_snapshot_manifest.json
context_observations.jsonl
context_interpretations.jsonl
solver_updates.jsonl
planning_request.json
planner_specification.json
planner_run.json
plan_artifact.json
feasibility_check_report.json
conflict_sets.jsonl
repair_proposals.jsonl
evaluation_report.json
plan_diff.json
explanation_evidence.json
sensitivity_manifest.json
source_provenance.json
interface_build_manifest.json
```

No-plan runs omit `plan_artifact.json` and record the expected absence.

### K.8 Reproducibility manifest

Record repository commit/dirty state, dependency/runtime hashes, hardware/OS/timezone, random seeds, commands/configs, solver version/parameters/status/bound/gap/log, parser model/prompt/schema/decoding, compiler and preference mapping versions, provider registry, snapshots and licenses, all artifact hashes, warnings/failures/exclusions, and review/freeze timestamps.

---

## L. Scope, Literature Traceability, and Acceptance

### L.1 First slice versus long-term mapping

| Requirement | First slice | Long-term |
|---|---|---|
| Immutable artifacts and evaluator | Required | Maintained |
| Typed confirmed requests and compiler | Required | Extended with LLM parsing |
| Weather, closures, mobility, pace/accessibility | Required as data permits | Broader dynamic contexts |
| California | Required | Nationwide-capable architecture |
| One additional corridor | Optional if reproducible | Three or more corridors |
| 50-state plus D.C. audit | Deferred | Required before national coverage claims |
| LLM parser benchmark | Optional for TRB; required for parser claims | Core IUI/CHI path |
| Baseline feasibility checker | Desirable | Required technical foundation |
| Enhanced branch-and-check | Deferred if schedule constrained | Technical extension |
| Expert route/evaluator validation | Required for technical claims | Expanded |
| Controlled participant study | Only for IUI/CHI claims | Human-centered program |
| Reinforcement learning | Deferred | Only after evaluator validation |
| Autonomous monitoring/booking | Prohibited | Separate safeguarded research |

### L.2 Literature-to-plan traceability

| Literature insight | Plan mechanism | Contracts/tests | Evidence |
|---|---|---|---|
| Logic-LM solver-guided refinement | Syntactic feedback, semantic confirmation, correction lineage | `ParserCorrection`, parser tests | Parser/correction benchmark |
| TTG constrained structured generation | Typed request and deterministic compiler | `ParsedTravelRequest`, `CompilationRecord` | Exact/field match and end-to-end degradation |
| LLMAP LLM-as-parser | Parse POI categories, time, dependencies, preferences; formal route method solves | Parser/compiler boundary tests | Gold versus predicted parser and LLM-only comparison |
| Rich TTDP branch-and-check | Master selection, detailed checker, conflict sets, restricted repair | `FeasibilityCheckReport`, `ConflictSet`, repair tests | Baseline/enhanced/checker experiments |
| TripScore layered evaluation | Artifact, hard, soft, preference groups; conditional reward | `EvaluationReport`, gating tests | Expert pairwise agreement and Goodhart tests |
| TravelEval complete-plan evaluation | Independent temporal/spatial/economic/utility checks | Evaluator independence tests | Full-route benchmark |
| TripTide intent-preserving adaptation | Scoped repair and minimal change | `RepairProposal`, plan diff | Original-versus-repaired metrics |
| Visual explanation survey | Aim/scope/method/format and task-matched visuals | `ExplanationEvidence`, explanation tests | Correction/inspectability study |
| CityHood explainable travel UI | Progressive geographic inspection | Layered dashboard views | User/expert task evidence |

### L.3 Terminology

Use:

- modeled feasibility under the frozen snapshot;
- solver-backed;
- user-confirmed interpretation;
- independent evaluation;
- context-aware minimal-change repair;
- conditional solver guarantee;
- nationwide-capable architecture with audited gaps;
- appropriate or calibrated reliance.

Avoid:

- the LLM solves the route;
- guaranteed real-world itinerary;
- optimal user itinerary;
- safe route or neighborhood;
- nationwide support without coverage evidence;
- explainable without specifying aim, evidence, format, and evaluation;
- fully autonomous agent;
- IUI alone when discussing the interactive venue path: use IUI or CHI.

### L.4 Main-plan/appendix consistency rules

- The first slice contains narrow validated contexts, not all families.
- TRB is an evidence option, not the assumed first target.
- Parser work is required only for parser or IUI/CHI claims.
- Enhanced branch-and-check is staged after truthful artifacts and baseline repair.
- Independent evaluation is mandatory before any plan comparison.
- A feasible plan is not eligible until grounding and hard checks pass.
- Agent monitoring and external side effects remain deferred.

### L.5 Acceptance checklist

- main plan and appendix links resolve;
- heading hierarchy and code fences are valid;
- every main-plan contract exists here;
- direct LLM-to-solver control is prohibited and tested;
- syntactic and semantic failures have distinct statuses;
- consequential semantic changes require confirmation;
- preference levels avoid false precision;
- branch/checker claims match implemented stage;
- evaluator hard gating precedes soft/preference scoring;
- component metrics remain visible beside any reward;
- explanations resolve to evidence and counterfactuals;
- no incident-derived hard exclusion or unsupported safety label exists;
- geographic claims distinguish schema, adapter, audit, and experiment scope;
- venue dates are reverified before use;
- no deadline overrides validity, ethics, or reproducibility.
