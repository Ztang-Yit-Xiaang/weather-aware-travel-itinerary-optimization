# Context-Aware, Inspectable Travel Optimization

## Research Stabilization and Publication Plan

**Repository:** Weather-Aware Travel Itinerary Optimization<br>
**Working research name:** Context-Aware, Inspectable Travel Optimization<br>
**Status:** Revised roadmap for research-team and supervisor review<br>
**Revised:** June 24, 2026<br>
**Venue decision:** Evidence-driven TRB or IUI/CHI path; no venue is assumed in advance<br>
**Technical appendix:** [Research Stabilization and Publication Appendix](research_stabilization_and_publication_appendix.md)

---

## Contents

1. [Executive Decision Summary](#1-executive-decision-summary)
2. [Contribution and Research Questions](#2-contribution-and-research-questions)
3. [System Architecture and Research Mechanism](#3-system-architecture-and-research-mechanism)
4. [Technical Validity and Evaluation](#4-technical-validity-and-evaluation)
5. [Evidence-Driven Venue Strategy](#5-evidence-driven-venue-strategy)
6. [Implementation Phases and Readiness Gates](#6-implementation-phases-and-readiness-gates)
7. [Risks, Stop Conditions, and Deferred Scope](#7-risks-stop-conditions-and-deferred-scope)
8. [Claim Matrix and Immediate Execution](#8-claim-matrix-and-immediate-execution)

---

## 1. Executive Decision Summary

### 1.1 Revised core thesis

The first defensible research thesis is:

> We design and evaluate an inspectable, solver-backed workflow that converts user-confirmed travel requests and frozen context evidence into minimal-change itinerary repairs, independently evaluates the resulting plans, and exposes interpretation, feasibility, tradeoff, and counterfactual evidence for review.

The project is not novel merely because it combines an LLM with a solver. LOGIC-LM, TTG, TRIP-PAL, and LLMAP already establish language-to-symbolic and language-to-route-solver patterns. The plausible contribution is the combination of:

- user-visible and correctable structured interpretation;
- context-aware minimal-change repair under a declared model;
- independent hard, soft, preference, and robustness evaluation;
- evidence-linked why, why-not, changed, and what-would-change explanations;
- truthful solver, fallback, and artifact lineage.

Weather and nature remain strong differentiating contexts. The first paper should also include authoritative operational closures, mobility or travel-time disruption, and traveler-approved pace or accessibility constraints when the required data and tests are available.

### 1.2 Repository-grounded capability audit

| Capability | Status | Repository evidence and boundary |
|---|---|---|
| Multi-day California route generation | IMPLEMENTED_BUT_NOT_VALIDATED | `experiment_runner.py`, route artifacts, hotels/base cities, and dashboard exports exist; no canonical independent complete-plan evaluator exists |
| Interest profiles and structured request overrides | PARTIALLY_IMPLEMENTED | `TripPlanningRequest` and interest normalization exist; no canonical `ParsedTravelRequest`, semantic confirmation, or parser lineage exists |
| Weather-aware scoring and constraints | PARTIALLY_IMPLEMENTED | Weather risk affects utility and epsilon constraints; this is not live disruption monitoring or validated repair |
| Gurobi and heuristic/fallback route paths | IMPLEMENTED_BUT_NOT_VALIDATED | Solver status and gap fields exist in several outputs; requested/executed method lineage and fallback ownership are not canonical |
| Hotels, base cities, nature regions, and alternatives | IMPLEMENTED_BUT_NOT_VALIDATED | Generated artifacts and dashboard selectors exist; common post-planning feasibility evaluation is missing |
| Provider adapters | PARTIALLY_IMPLEMENTED | Open-Meteo, NPS, OSM/Overpass, Wikidata/Wikipedia, and OSRM paths exist; common provider contracts, licensing gates, coverage audits, and immutable snapshots do not |
| Provenance and explanation display | PARTIALLY_IMPLEMENTED | Source fields, confidence, basic why-selected text, and some nature why-not codes exist; generalized faithful and counterfactual explanations do not |
| Immutable plan and run artifacts | PROPOSED | Current artifact metadata includes a config hash and timestamps but not the required canonical run/plan/evaluation lineage |
| Independent post-planning evaluator | PROPOSED | Dashboard export validation is not independent itinerary evaluation |
| Natural-language parser and semantic confirmation | PROPOSED | No LLM parser or user-confirmed semantic interpretation exists |
| Parser-to-planner compiler and allow-listed updates | PROPOSED | Structured overrides exist, but no versioned compiler or solver-update registry enforces the boundary |
| Minimal-change repair | PROPOSED | Route alternatives exist; no declared repair objective or original-versus-repaired contract exists |
| Branch-and-check or restricted repair | PROPOSED | Current Gurobi models do not implement the cited master/checker architecture |
| Controlled inspectability or correction study | PROPOSED | No ethics-approved participant evidence exists |

One current blocker is explicit: `experiment_runner.py` can insert or replace required anchors after local route generation. Publication artifacts cannot retain solver certification after such mutation unless the changed route receives a new run lineage and complete independent evaluation.

### 1.3 Formulate-validate-solve-evaluate-explain-correct architecture

```text
Natural-language request or structured user input
    -> request formulation
    -> typed schema validation
    -> semantic and consequence validation
    -> user confirmation when required
    -> deterministic configuration compiler
    -> PlanningRequest + PlannerSpecification + allow-listed SolverUpdates
    -> symbolic optimizer or route solver
    -> immutable PlannerRun and optional PlanArtifact
    -> independent EvaluationReport and eligibility gate
    -> grounded natural-language and visual ExplanationEvidence
    -> user confirmation, correction, counterfactual, or scoped replanning
```

Responsibilities are deliberately separated:

- **LLM:** understand language; propose entities, constraints, dependencies, and preferences; detect ambiguity; explain stored solver/evaluator evidence; support correction.
- **Compiler:** map validated semantic fields to allow-listed configuration fields, variables, bounds, objectives, and constraints. The LLM may not emit Gurobi code or arbitrary mathematical expressions.
- **Solver:** enforce encoded feasibility; optimize under the declared model; return status, bound, gap, conflict, and infeasibility evidence.
- **Evaluator:** independently recompute grounding, hard feasibility, soft quality, preference fit, robustness, and explanation support; determine comparison eligibility.
- **User:** approve ambiguous or consequential interpretations, correct preferences, choose among eligible alternatives, and authorize broader repair scope.

### 1.4 Conditional guarantee statement

Use this statement, and no stronger one:

> Given the user-confirmed structured interpretation, frozen candidate and context data, and declared mathematical model, the solver can certify modeled feasibility or optimality for that model when the backend returns the corresponding certificate.

This does not guarantee true traveler intent, data completeness, future availability, safety, real-world validity, or a globally optimal real-world trip.

### 1.5 Scope reduction decision

The former first-paper scope required operational providers for every context family, a 50-state plus D.C. audit, three regional corridors, an LLM parser, branch-and-check repair, independent evaluation, an expert study, and a nearly complete paper within weeks. That combination is unrealistic and would encourage shallow integration or unsupported claims.

The roadmap is now split into two scopes.

**First publishable research slice:**

- truthful immutable run, plan, evaluation, and explanation artifacts;
- user-confirmed typed requests, with deterministic or gold interpretation acceptable for the first technical experiment;
- weather, authoritative closures, mobility/travel-time disruption, and user-approved pace/accessibility;
- California plus one additional reproducible corridor only if provider and snapshot evidence is ready;
- context-blind solver, deterministic heuristic, and context-aware solver-backed repair;
- original-versus-repaired comparison and minimal-change metrics;
- independent gated evaluation;
- expert review for a transportation path or a controlled correction study for an interactive path.

**Longer-term full system roadmap:**

- validated LLM request parser and conversational correction;
- branch-and-check enhancements and restricted subset recovery;
- common schema for all context families;
- nationwide provider registry and 50-state plus D.C. audit;
- three or more corridors;
- proactive monitoring and user-authorized repair proposals;
- broader IUI or CHI interaction evidence.

Other context families remain schema-level capabilities and staged adapters. They are excluded from first-paper empirical claims until their provider coverage and operational semantics are validated.

---

## 2. Contribution and Research Questions

### 2.1 Contribution matrix

| Area | Literature precedent | Current capability | Remaining gap | Proposed contribution |
|---|---|---|---|---|
| Language-to-symbolic formulation | Logic-LM, TTG, TRIP-PAL, LLMAP | Structured request dataclass and config overrides | No parser, semantic confirmation, correction lineage, or compiler | User-auditable typed interpretation with consequence-aware confirmation and deterministic compilation |
| Constrained planning | TTDP, rich branch-and-check, TRIP-PAL, LLMAP | Gurobi and heuristic routes with weather, budget, detour, diversity, hotels, and base cities | Post-solve mutation, distributed fallback, incomplete status lineage | Truthful solver-backed planning plus staged feasibility conflict and restricted repair |
| Context-aware repair | TripTide and contextual recommendation | Weather-sensitive route alternatives | No explicit disruption contract, repair objective, or intent-preservation evidence | Minimal-change repair under frozen weather, closure, mobility, and traveler-approved context |
| Plan evaluation | TravelEval and TripScore | Planner summaries and export validation | No independent gated evaluator or expert-calibrated comparison | Layered artifact, feasibility, quality, preference, and robustness evaluation with visible components |
| Explanation and interaction | Visual explanation survey, CityHood, mixed-initiative work | Maps, alternatives, source confidence, basic reasons | No faithful why-not/counterfactual evidence or correction study | Layered evidence, counterfactuals, and user-visible interpretation/repair correction |

The novelty claim must concern the tested mechanism and evidence, not the component list.

### 2.2 Revised research questions and evidence map

| Research question | Components | Data or scenarios | Metrics | Baselines | Allowable claim |
|---|---|---|---|---|---|
| **RQ1 - Interpretation:** How accurately and safely can natural-language requests and unstructured context be translated into validated, user-auditable planning inputs? | `ParsedTravelRequest`, parser run, semantic validator, confirmation, compiler | Gold request/context corpus with ambiguity, dependencies, nonexistent entities, vague deadlines, and corrections | Schema-valid rate, field exact match, entity F1, hard/soft accuracy, dependency direction, scope accuracy, unsupported-fact rate, consequence-weighted error, confirmation precision/recall | Deterministic forms, direct LLM JSON, gold parser | Parser claims only if this benchmark is completed; otherwise use user-confirmed structured inputs and defer RQ1 |
| **RQ2 - Planning and repair:** How do solver-backed context-aware planning and minimal-change repair compare with context-blind, heuristic, and LLM-only alternatives? | Compiler, solver, repair scope, optional branch/checker | Frozen weather, closure, delay, and pace/accessibility scenarios | Feasibility, utility, risk/burden, travel, cost, retained intent, edit distance, runtime, bound/gap, conflict detection | Context-blind solver, deterministic heuristic, full replanning; LLM-only only when a parser/agent path is evaluated | Modeled outcome and repair claims under frozen snapshots |
| **RQ3 - Evaluation:** Do independent hard, soft, preference, and robustness metrics distinguish feasible but poor itineraries from genuinely useful plans? | Independent evaluator, hard gate, component report, conditional reward | Strong, weak, infeasible, ungrounded, and preference-misaligned plans | Eligibility, violation detection, pairwise agreement, uncertainty, rank stability, Goodhart tests | Planner totals, ungated aggregate, expert pairwise judgment | Evaluator discrimination and agreement, not universal trip quality |
| **RQ4 - Interaction:** Can users or experts detect and correct inappropriate interpretations or repairs more effectively with layered evidence and counterfactual explanations? | Interpretation view, provenance, diff, why/why-not, counterfactual, correction log | Balanced strong and questionable cases with identical plans across conditions | Detection, correction accuracy, revision quality, time, workload, control, calibrated reliance | Basic interface versus inspectable interface | Population- and study-specific interaction effects only |
| **RQ5 - Action and deferral:** When should the system automatically repair a low-consequence syntactic issue, request confirmation, broaden repair scope, or defer? | Validator, consequence rules, confirmation statuses, repair-scope policy | Syntactic errors, semantic errors, conflicts, must-go effects, broad-scope changes | Auto-repair precision, unsafe-auto-repair rate, confirmation/deferral precision/recall, unnecessary escalation | Always act, always confirm, rule-based consequence policy | Decision-policy performance under declared scenarios |

### 2.3 Syntactic validity is not semantic correctness

Two failure classes must remain separate:

1. **Syntactic or executable failure:** invalid schema/type, missing field, unknown entity, unsupported effect, impossible unit, malformed constraint.
2. **Semantic failure:** valid structure with a wrong interpretation, reversed dependency, hard/soft confusion, wrong date/location/entity/scope, false precision, or misunderstood must-go/avoid intent.

Machine feedback may repair a low-consequence syntactic error automatically when the correction is deterministic and logged. Semantic or outcome-sensitive changes require explicit user or authorized reviewer confirmation. Valid JSON or a feasible route never proves correct interpretation.

### 2.4 Working paper narratives

**Transportation optimization narrative:** context-aware minimal-change repair with independent evaluation and operational evidence.

**Interactive intelligence narrative:** user-visible request formulation, semantic correction, confirmation/deferral, and grounded route explanations.

Do not combine both narratives into one manuscript unless both technical and human evidence are complete.

---

## 3. System Architecture and Research Mechanism

### 3.1 Canonical evidence chain

```text
ParsedTravelRequest
    -> ParserRun and optional ParserCorrection records
    -> RequestValidationReport and user confirmation
    -> CompilationRecord
    -> PlanningRequest + CandidateSnapshot + ContextSnapshot + PlannerSpecification
    -> PlannerRun
    -> optional immutable PlanArtifact
    -> EvaluationReport and comparison eligibility
    -> ExplanationEvidence
    -> user/expert decision and correction log
```

Every record is versioned and content-hashed. A modification creates a new record with parent lineage rather than rewriting the prior artifact.

### 3.2 Request formulation and compiler boundary

`ParsedTravelRequest` stores entities, dates, start/end locations, hard constraints, soft and conditional preferences, must-visit/avoid lists, ordering dependencies, time limits, interpretable tolerance levels, ambiguity flags, field confidence, confirmation status, parser version, schema version, and content hash.

Preference language maps to visible levels, not false numerical precision:

- nature importance: low, medium, high;
- driving sensitivity: low, medium, high;
- weather tolerance: cautious, balanced, flexible;
- pace and physical effort: declared ordinal levels.

A versioned compiler maps each level to sensitivity-tested parameter ranges. Users and researchers can inspect and manually change the level. The LLM cannot set arbitrary objective coefficients.

### 3.3 Solver-guided refinement and confirmation

The validator may return machine-readable errors. Deterministic repairs may correct casing, canonical units, missing defaults with declared semantics, or an unambiguous entity ID. Every repair stores original interpretation, feedback, revised interpretation, and approval status.

Reversed dependencies, hard/soft reclassification, changed dates or locations, new must-go/avoid requirements, broad repair scope, or material route effects require confirmation. Unresolved ambiguity defers planning.

### 3.4 Narrow context scope

The first slice prioritizes:

1. weather exposure and weather-sensitive activity suitability;
2. authoritative operational closures;
3. mobility or travel-time disruption;
4. traveler-approved pace or accessibility limits.

Nature is retained as a strong application context and intent category. Other environmental, seasonal, crowding, event, lodging-inventory, advisory, and reported-incident families remain in the common schema and provider registry but are staged extensions. Reported incidents never create hard exclusions or safety labels.

Geographic scope must distinguish:

- **schema-level capability**;
- **adapter availability**;
- **audited provider coverage**;
- **experimentally validated corridor scope**.

The architecture remains nationwide-capable with audited gaps, but the first paper should not claim nationwide support.

### 3.5 Staged branch-and-check and restricted repair

The current repository does not implement the cited branch-and-check method. The future technical path is staged:

1. **Baseline check:** test whether a selected bundle is schedulable; return a simple infeasible-set cut.
2. **Enhanced check:** identify pairwise incompatibilities, duration lower bounds, smaller conflicts, and stronger cardinality bounds.
3. **Restricted repair:** recover the best feasible subset and preserve a truthful child run.
4. **Human-facing conflict explanation:** translate the conflict into options such as more time, a changed base, one removed POI, or a relaxed preference.

This path is not a first-paper dependency unless the baseline checker is stable early. Enhanced cuts and branch-and-solve remain longer-term technical contributions.

### 3.6 Layered explanation

Each explanation declares its aim, scope, method, and format.

- **Level 1:** short plain-language summary.
- **Level 2:** major tradeoffs, evidence, binding constraints, and uncertainty.
- **Level 3:** source-to-observation-to-interpretation-to-update-to-plan lineage, evaluator checks, and counterfactual run.

Use maps for sequence/detour, timelines for temporal feasibility, bars for quantitative tradeoffs, constraint panels for limits/slack, linked evidence for lineage, and side-by-side diffs for repair. Node-link views are reserved for dependencies. Provenance badges identify stale, proxy, conflicting, or unsupported evidence.

---

## 4. Technical Validity and Evaluation

### 4.1 Preserved non-negotiable safeguards

- immutable content-hashed artifacts;
- separate parser, planner-run, plan, evaluation, and explanation records;
- truthful requested-versus-executed method lineage;
- centralized fallback with parent-child run records;
- frozen candidate, context, travel-time, and planner snapshots;
- allow-listed solver updates and deterministic compilation;
- no unvalidated LLM output directly controlling the solver;
- explicit provenance, freshness, coverage, license, and missingness;
- no unsupported safety label or incident-derived hard exclusion;
- no inference from solver feasibility to real-world validity.

### 4.2 TripScore-style independent evaluator

The evaluator has four groups:

1. **Artifact and grounding validity:** schemas, IDs, names, no hallucinated entities/times/sources, complete lineage and licensing.
2. **Hard and commonsense feasibility:** chronology, location consistency, opening hours when available, route continuity, no activity during transport, lodging/transport completeness, budget/time limits, closures, and mode availability.
3. **Soft itinerary quality:** density, clustering, hotel consistency, daytime use, uniqueness, diversity, high-priority coverage, detour, context burden, and minimal-change quality.
4. **Personal preference fit:** pace, budget style, interests, effort, driving and weather tolerance, must-go retention, and conditional/free-form request fulfillment.

Hard gating applies. Invalid or ungrounded artifacts receive no soft-quality credit. Hard-infeasible plans cannot rank highly because they are attractive. Soft and preference metrics contribute only after eligibility. A conditional aggregate may support ranking or sensitivity analysis, but component metrics remain visible.

Rule-based and LLM-judged metrics are stored separately. Evaluator versions, uncertainty, expert pairwise agreement, rank stability, and Goodhart-resistance tests are required. Reinforcement learning is deferred until the evaluator is stable and validated.

### 4.3 Publication blockers

Before result generation:

- eliminate or re-run post-solve anchor mutation;
- implement truthful requested/executed method and fallback lineage;
- separate `PlannerRun` from immutable `PlanArtifact`;
- implement independent evaluation and hard eligibility gating;
- freeze snapshots and canonical hashes;
- reject direct LLM-to-solver access;
- ensure comparison metrics are evaluator-owned;
- distinguish preview, demonstration, experiment, and study artifacts.

### 4.4 Baselines and experiments

Required technical baselines are context-blind solver, deterministic context-aware heuristic, proposed solver-backed repair, and full replanning. LLM-only route generation is included only for a parser/agent experiment and must use identical frozen inputs.

Key comparisons include gold versus predicted parser, context-blind versus context-aware repair, baseline versus enhanced feasibility checking, original versus repaired route, and basic versus layered explanation/correction.

---

## 5. Evidence-Driven Venue Strategy

### 5.1 Venue decision rule

Do not assume one manuscript can be relabeled. Select a venue only after comparing implemented components, completed experiments, ethics/participant readiness, and the strongest evidence.

### 5.2 TRB path

Choose TRB when the strongest result is transportation disruption and route repair. Required evidence includes frozen context scenarios, solver/heuristic/context-blind comparisons, robustness, ablation, minimal-change metrics, operational error analysis, and expert review. Inspectability supports validation but cannot carry broad user-benefit claims.

The first TRB slice does not require every context family, a 51-jurisdiction audit, three corridors, or an LLM parser. It requires a coherent narrow context set and one or two reproducible corridors.

The existing plan lists an August 1, 2026 submission end date and TRB formatting requirements. **REQUIRES_REVERIFICATION:** confirm all deadline and format facts from the official call immediately before scheduling submission work.

### 5.3 IUI or CHI path

Choose IUI or CHI when the strongest result is mixed-initiative interpretation, correction, explanation, or calibrated reliance.

- **IUI path:** parser benchmark, structured interpretation display, correction, confirmation/deferral, solver boundary, and an intelligent-interface evaluation.
- **CHI path:** ethics-approved controlled evidence about understanding, error detection, agency, revision, workload, or calibrated reliance using identical plan artifacts across conditions.

The existing plan lists August 2026 IUI dates and a September 10, 2026 CHI date. **REQUIRES_REVERIFICATION:** do not rely on these dates until checked against the official calls.

### 5.4 Venue readiness comparison

| Decision factor | TRB | IUI or CHI |
|---|---|---|
| Core implemented result | Context-aware route repair | Validated interpretation/correction interface |
| Primary evidence | Route, robustness, operational metrics | Controlled interaction and behavior evidence |
| Human evidence | Expert review can support validity | Ethics-approved participant study required for user claims |
| Parser requirement | Optional | Required for interpretation claims |
| Branch-and-check requirement | Optional enhancement | Not central unless it supports explanations |
| Current readiness | NEEDS_REPOSITORY_VERIFICATION after Phase 0 | PROPOSED; parser and study infrastructure absent |

---

## 6. Implementation Phases and Readiness Gates

### 6.1 Dependency order

| Phase | Deliverable | Depends on | First-slice status |
|---|---|---|---|
| 0 - Truthful foundation | Immutable plans, truthful runs, requested/executed method, centralized fallback, independent evaluator skeleton and eligibility gate, frozen snapshots | None | Mandatory blocker |
| 1 - Structured contracts | `ParsedTravelRequest`, observations, interpretations, updates, validation, compiler, confirmation | Phase 0 contracts | Mandatory; LLM parser optional for TRB |
| 2 - Narrow context planning | Weather, closures, mobility disruption, approved pace/accessibility, minimal-change repair | Phases 0-1 | Mandatory |
| 3 - Conflict and restricted repair | Feasibility checker, conflict extraction, enhanced cuts, best feasible subset, repair alternatives | Phase 2 | Baseline checker desirable; enhancements staged |
| 4 - Evaluation and explanation | Hard/soft/preference evaluator, diff, evidence-linked why-not/counterfactuals | Phases 0-3 | Mandatory evaluator; explanation depth venue-dependent |
| 5 - Experiments | Parser if applicable, planning baselines, robustness, sensitivity, expert inspection | Phases 0-4 | Mandatory before venue decision |
| 6 - IUI or CHI extension | Mixed-initiative interface and controlled correction/inspectability study | Ethics and Phase 5 | Later fork |
| 7 - Long-term agent | Monitoring, detection, notification, repair proposal, user-authorized adoption | Stable prior phases | Deferred |

### 6.2 Revised near-term calendar

| Period | Required delivery | Gate consequence |
|---|---|---|
| June 24-30 | Freeze first-slice scope; accept contracts/statuses; remove mutation and lineage blockers; define evaluator gates | No research result generation without Phase 0 design acceptance |
| July 1-7 | Implement truthful artifacts, compiler boundary, frozen snapshots, and evaluator skeleton; create gold scenarios | TRB at risk if Phase 0 is incomplete |
| July 8-17 | Implement narrow context repair, baselines, diffs, and core metrics; run deterministic scenarios | July 18 evidence audit, not an automatic submission decision |
| July 18 | Technical readiness review | Continue toward TRB only if reproducible route results and a coherent transportation claim exist |
| July 19-26 | Robustness, sensitivity, expert review, figures, and full limitations | Stop if evidence remains integration-only |
| July 27-August 1 | Submission work only after a passed gate and reverified deadline | Otherwise defer or select a later path |
| August onward | Parser/correction benchmark and IUI or CHI study preparation | Requires ethics and venue-specific evidence |

### 6.3 Readiness gates

**Technical slice gate:** no post-solve mutation, complete lineage, independent evaluation, frozen scenarios, eligible plans only, baseline comparison, and reconstructable results.

**TRB gate:** technically complete transportation story, robustness and operational analysis, expert feedback, venue-formatted draft, and reverified deadline.

**IUI/CHI gate:** validated parser/correction workflow, ethics readiness, frozen conditions, study evidence, and claims aligned with the participant population.

Missing a deadline does not make another venue appropriate. Defer when the evidence does not fit.

---

## 7. Risks, Stop Conditions, and Deferred Scope

### 7.1 Main risks

| Risk | Impact | Required response |
|---|---:|---|
| First slice expands back to all context families | Critical | Enforce scope freeze; move adapters to long-term registry |
| Valid JSON contains wrong meaning | Critical | Separate semantic validation and require consequential confirmation |
| Parser creates false numerical precision | High | Use visible ordinal levels and versioned sensitivity-tested mappings |
| Solver feedback repairs syntax but preserves semantic error | Critical | Never treat executability as semantic proof; retain original/revision/approval lineage |
| Branch-and-check complexity exceeds schedule | High | Stop at baseline checker and restricted repair; defer enhanced cuts |
| Unified reward hides hard failures | Critical | Apply hard gating and publish component metrics |
| Reward or judge is optimized rather than trip quality | High | Pairwise expert validation, uncertainty, holdout tests, and Goodhart probes |
| Provider/licensing coverage fails | High | Record unsupported state; do not add a weak substitute merely to preserve scope |
| Solver and evaluator disagree | High | Make plan ineligible and investigate |
| Explanation is plausible but unsupported | Critical | Require evidence IDs and reproducible counterfactual runs |
| Human evidence is unavailable | High | Restrict to technical claims and defer IUI/CHI path |

### 7.2 Stop conditions

Pause publication result generation when artifacts mutate without lineage, displayed and executed methods differ, an LLM bypasses validation/compiler boundaries, semantic confirmation is missing, a hard constraint relies on weak evidence, evaluator independence is compromised, a plan is ranked before eligibility, source terms prohibit use, ethics approval is absent, or frozen runs cannot be reproduced.

### 7.3 Deliberately deferred

- full operational integration for every context family;
- nationwide empirical validation and uniform coverage claims;
- third-corridor experiments unless the first two are stable;
- enhanced branch-and-check cuts when baseline checking is incomplete;
- reinforcement learning on TripScore-style reward;
- continuous monitoring and unsolicited notification;
- autonomous plan adoption, booking, cancellation, purchasing, or payment;
- long-term behavioral preference learning and group negotiation;
- incident-derived hard exclusions or safety rankings.

---

## 8. Claim Matrix and Immediate Execution

### 8.1 Updated claim matrix

| Claim | Current status | Evidence required before use |
|---|---|---|
| The prototype generates multi-day California route artifacts | IMPLEMENTED_BUT_NOT_VALIDATED | Frozen reproducible run and independent evaluation |
| Weather affects utility and modeled route constraints | PARTIALLY_IMPLEMENTED | Scenario tests, provenance, sensitivity, and evaluator recomputation |
| Hotels/base cities and route alternatives are integrated | IMPLEMENTED_BUT_NOT_VALIDATED | Frozen artifacts and common evaluation |
| Plans and runs are immutable and truthfully separated | PROPOSED | Canonical contracts, hashes, and mutation tests |
| Requested and executed methods are always truthful | PROPOSED | Centralized fallback and lineage tests |
| Natural-language requests are parsed accurately | UNSUPPORTED | Parser benchmark and semantic-error analysis |
| Parser output cannot bypass the compiler | PROPOSED | Architecture and adversarial tests |
| Solver feasibility is conditionally certified | PARTIALLY_IMPLEMENTED | Canonical statuses, frozen model, bound/gap evidence, and no post-solve mutation |
| Context-aware repair improves outcomes | UNSUPPORTED | Frozen baseline comparisons |
| Repairs preserve traveler intent | UNSUPPORTED | Independent preservation metrics and annotations |
| Branch-and-check conflict evidence is available | UNSUPPORTED | Baseline checker, conflict tests, and certificates |
| Final plans pass independent layered evaluation | UNSUPPORTED | Versioned evaluator and eligibility reports |
| Why/why-not/counterfactual explanations are grounded | PARTIALLY_IMPLEMENTED | Evidence links and reproducible counterfactuals |
| The architecture is nationwide-capable with audited gaps | PROPOSED | Provider registry and later coverage audit |
| Users or experts benefit from inspectability | UNSUPPORTED | Appropriate expert or controlled participant evidence |
| The system guarantees real-world validity or true intent | PROHIBITED | Do not claim |

### 8.2 First publishable slice is done when

- Phase 0 safeguards pass;
- one typed request and compiler path is canonical;
- narrow contexts have frozen, legally usable evidence;
- context-blind, heuristic, repair, and full-replanning baselines run on identical inputs;
- original and repaired plans have independent component metrics;
- every ranked plan is eligible;
- unsupported explanations fail closed;
- results include robustness, sensitivity, and error analysis;
- the venue claim follows completed evidence.

### 8.3 Recommended next seven days

1. Freeze the first-slice contexts and corridor; mark every other family staged.
2. Define `ParsedTravelRequest`, parser/confirmation statuses, compiler output, and artifact hashes in the appendix.
3. Remove post-solve anchor mutation from publication paths or require a new child run and evaluation.
4. Implement the `PlannerRun`/`PlanArtifact` separation and requested/executed method lineage design.
5. Build the independent evaluator skeleton with artifact and hard-feasibility gates before soft scoring.
6. Create four frozen golden scenarios: weather burden, authoritative closure, mobility delay, and user-approved pace/accessibility.
7. Run an end-of-week evidence review and choose the smallest implementation slice that can produce a valid baseline comparison.

### 8.4 Human decisions required

- approve the first corridor and optional second corridor;
- decide whether RQ1 parser evaluation belongs in the first paper or the IUI/CHI extension;
- approve any paid-provider budget or restricted-license source;
- confirm ethics timing and participant access;
- reverify venue deadlines and select a target only after the technical gate.

Detailed schemas, status rules, compiler contracts, branch/checker stages, evaluator gates, experiments, tests, and literature traceability are maintained in the [technical appendix](research_stabilization_and_publication_appendix.md).
