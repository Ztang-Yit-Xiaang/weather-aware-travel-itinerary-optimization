# Research Pipeline and Evidence Gate Map

**Status:** Active master roadmap  
**Effective date:** 2026-07-29  
**Near-term execution authority:** `docs/planning/current_execution_plan.md`  
**Implementation-status authority:** `docs/current/current_problem_manifest.md`  
**Canonical technical contract:** `docs/planning/travel_itinerary_repair_technical_specification.md`

## Objective

Produce two evidence-bounded research slices without allowing the longer-term
LLM and agent architecture to outrun the technical foundation:

1. a technical repair slice that tests whether ownership-aware, smallest-radius
   repair preserves more of an accepted itinerary while remaining independently
   eligible; and
2. an interaction slice that tests whether permission-aware counterfactual
   probes improve clarification and user control over ambiguous itinerary
   repairs.

The accepted itinerary remains an immutable, user-owned parent artifact. Every
reported repair must be a traceable child with a typed `PlanDiff`, route-valid
evidence, an independent evaluator result, and reconstructable artifacts.

## Authority and Conflict Resolution

Use planning sources in this order:

1. Current code, tests, and immutable run artifacts establish what exists.
2. `docs/current/current_problem_manifest.md` establishes implementation status.
3. `docs/planning/current_execution_plan.md` establishes the immediate queue and
   stop conditions.
4. `docs/planning/travel_itinerary_repair_technical_specification.md` establishes
   the canonical implementation contract.
5. This file establishes the cross-slice dependency order and gate meanings.
6. Dedicated current E3 phase plans define implementation-ready phase details
   but do not override higher-authority status evidence.
7. Older integrated and detailed phase plans are design history and acceptance
   criteria. Their unchecked boxes are not current status.
8. `Travel_Planner_Novelty_Report.md` is an external synthesis used to refine
   scope and claim boundaries; it is not implementation evidence.

When these sources conflict, prefer the highest current source above. Record a
material unresolved conflict as a gate decision rather than silently choosing a
convenient interpretation.

## Planning State Model

- `planned`: accepted future work whose entry gate is not yet satisfied.
- `ready`: dependencies and material decisions for the phase are resolved.
- `in-progress`: implementation or evidence generation has started.
- `blocked`: a named dependency prevents progress.
- `implemented`: code is complete but required verification remains.
- `verified`: the phase exit evidence has passed.
- `deferred`: intentionally postponed with a recorded reason.

A planning document does not move a phase to `implemented` or `verified`.

## Verified Current State

- E1 is verified locally: the package entry point is authoritative and the
  production notebook boundary is tested.
- E2 is verified locally for the reviewed base bundle. The later E3 diagnostic
  uses expanded bundle `route_bundle_a60c80047098a3b6` and route matrix
  `route_matrix_68ab535465b06808`, with 223/223 cells reported road- and
  endpoint-snap-valid and no fallback.
- The real E3 v14 diagnostic completed all 24 scenario-method cells. Its
  evidence, provenance, and shared-route-input contract passed.
- Eight E3 rows are independently ranking-eligible. The two exact methods
  refused all 12 exact rows under diagnostic
  `max_complete_candidates=50000`.
  Road-closure and reduced-driving-tolerance scenarios remain infeasible for
  both non-exact methods.
- The v14 diagnostic is evidence-complete, but it is not a complete four-method
  performance comparison and does not support a superiority claim.
- E3.M metric/UI integrity and E3.2 non-exact failure policy are verified. v14
  predates E3.M and must not be relabeled as generated under the hardened
  contract.
- E3.C1-E3.C4 are verified. The E3.C4 selector-containment fix preserves
  desktop geometry and all rendered-data/Leaflet contracts; both legacy UIs
  pass the 1440/768/520/390 browser matrix.
- The legacy Folium and modular dashboards are frozen research/parity
  artifacts. E3.UX is a verified additive product-dashboard path, not an in-place
  redesign.
- A disabled-by-default deterministic interaction scaffold exists under
  `src/itinerary_system/interaction/`. The overall interaction phase remains
  deferred and is excluded from E3/E4 publication runs.

The current evidence is recorded in:

- `docs/current/e1_authoritative_entrypoint_closeout.md`
- `docs/current/e2_route_bundle_preflight.json`
- `docs/current/e3_publication_benchmark_contract.md`
- `docs/current/current_problem_manifest.md`
- `tmp_test/research_pipeline_raw/e3-real-production-20260725-optimized-v14-cap50000`
- `docs/current/current_repository_truth_2026_07.md`
- `docs/current/ui_skill_application_record.md`

## Scope

### Technical slice

- One reproducible California corridor.
- Weather burden, authoritative closure, mobility/travel-time disruption, and
  user-approved pace/accessibility contexts.
- Four locked methods: context-blind exact repair, deterministic context-aware
  heuristic repair, ownership-aware progressive repair, and exact full
  reoptimization.
- Immutable inputs and outputs, retained failures, evaluator-owned eligibility,
  component metrics, runtime, failure diagnosis, and evidence-linked
  explanation.

### Interaction slice

- Ambiguous natural-language edits represented by multiple typed semantic
  hypotheses.
- Allow-listed repair programs and bounded hypothetical solver probes.
- Clarification based on material consequence and permission differences.
- Explicit autonomy ceilings that cannot be raised by inferred preferences.
- A staged anchor program, certificate-first runtime, and later human
  evaluation only after the technical slice is reproducible.

## Non-Goals

- No first-of-kind claim for disruption-aware planning, itinerary modification,
  LLM travel planning, LLM-plus-solver systems, explainable optimization, or
  multi-agent travel planning.
- No live provider, nationwide, participant-study, learned-autonomy, or
  longitudinal-personalization claim in the technical slice.
- No blind increase of exact-search caps and no optimality claim from an
  incomplete candidate universe.
- No ranking of hard-ineligible rows and no planner-owned score treated as
  independent evaluation.
- No use of preferences, memory, an LLM, or an agent to grant permission.

## Workstreams

1. **T - Technical evidence:** close E3, run E4, and reconstruct claim-ready
   artifacts.
2. **L - Closest-work verification:** maintain a reproducible literature
   protocol and narrow the contribution statement.
3. **I - Interaction foundation:** formalize typed hypotheses, repair programs,
   permission, consequences, and certificates after the technical gate.
4. **A - Anchor and benchmark data:** author, validate, and version dialogue
   trajectories and semantic-consequence evidence.
5. **P - Autonomy and personalization:** add table-gated autonomy first, then
   calibrated low-risk learning; advanced graphs and multi-agent topology are
   later extensions.
6. **UX - Artifact-grounded product presentation:** after E3.C4 verification,
   render canonical immutable artifacts through a separate customer/research
   product path without changing technical evidence or interaction authority.
7. **CP - Corrective local product:** extend the verified product foundation with
   typed direct map interaction, complete required-leg display, exact Compare and
   Evidence resolution, provider verification, and later W5 decision integrity.
   This track has no status-propagation edge to the technical research track.

## Dependency Map

```text
+---------------------+      +---------------------+
| E1 entry point      | ---> | E2 route evidence   |
| verified            |      | verified            |
+---------------------+      +---------------------+
                                  |
                                  v
                       +-------------------------+
                       | E3.0 real diagnostic    |
                       | verified evidence gate  |
                       | E3.M verified           |
                       +-------------------------+
                                  |
                    +-------------+-------------------------+
                    |                                       |
                    v                                       v
          +---------------------+     +---------------------+
          | E3.1 exact-baseline |     | E3.2 non-exact     |
          | tractability        |     | failure policy      |
          | blocked             |     | verified            |
          | blocker: D1         |     |                     |
          +---------------------+     +---------------------+
                    |                           |
                    +-------------+-------------+
                                  |
                                  v
                       +-------------------------+
                       | E3.3 immutable rerun    |
                       | and comparison closeout |
                       | blocked                 |
                       +-------------------------+
                                  |
                                  v
                       +-------------------------+
                       | E4 robustness, ablation |
                       | and paper evidence      |
                       | blocked                 |
                       +-------------------------+
                                  |
                 +----------------+----------------+
                 |                                 |
                 v                                 v
      +-------------------------+       +-------------------------+
      | Technical claim package |       | E5 interaction minimum  |
      | and venue decision      |       | viable architecture     |
      +-------------------------+       | deferred                |
                                        +-------------------------+
                                                   |
                                                   v
                                        +-------------------------+
                                        | E6 anchors and semantic |
                                        | benchmark               |
                                        +-------------------------+
                                                   |
                                                   v
                                        +-------------------------+
                                        | E7 certificate-first    |
                                        | autonomy V1             |
                                        +-------------------------+
                                                   |
                                                   v
                                        +-------------------------+
                                        | E8 interaction study    |
                                        | and publication evidence|
                                        +-------------------------+

Independent additive presentation branch:

+-------------------------+
| E3.C4 mobile/parity     |
| implemented             |
| mobile gate pending     |
+-------------------------+
             |
             v
+-------------------------+
| E3.UX0 -> UX1 -> UX2    |
| -> UX3 -> UX4           |
| planned                 |
+-------------------------+

E3.UX0-E3.UX4 may run in parallel with D1/E3.1 after E3.C4 is verified.
E3.UX5 is not on this branch: it depends on verified E5 and explicit
experimental enablement.

Corrective local-product branch (non-publication):

W4 implemented offline -> W4R in-progress -> G4R planned -> G4 blocked
                       -> W5 planned -> G5 -> W6 -> W7 -> W8

W4R addresses direct interaction, required route-leg display or explicit gaps,
and exact Compare/Evidence selection. G4R requires independent interaction
verification. G4 remains the separate live-provider gate. W5 remains closed
until G4 passes.

L1 closest-work review may proceed in parallel, but its verified matrix is
required before either publication slice promotes a novelty hypothesis to a
contribution claim.
```

## Phase Sequence and Gates

| Phase | Objective | Status | Entry gate | Exit gate and required evidence |
|---|---|---|---|---|
| E1 | Establish one authoritative package runner and thin notebook boundary. | `verified` | Repository validation harness available. | Focused tests and frozen-input smoke run emit an immutable run and manifest. |
| E2 | Freeze publication-eligible route evidence and reject nonvalidated cells. | `verified` | E1 artifact path is authoritative. | Bundle and matrix identity, provider/cache/snap/fallback audit, and strict evaluation pass. |
| E3.0 | Exercise all four methods on the real paired design and verify evidence contracts. | `verified` | Expanded E2 route universe exists. | Twenty-four unique scenario-method rows, retained failures, complete method provenance, and shared route lineage are recorded. |
| E3.M | Enforce metric ownership, finite values, fail-closed eligibility, and truthful evaluation UI. | `verified` | E3.0 exposed reportability risks and v14 remains preserved. | Ruff, 265/265 pytest, project checks, and real dashboard export validation pass; future E3 artifacts use the new contract. |
| E3.1 | Resolve exact-baseline tractability without false certification. | `blocked` | Blocker D1: v14 confirms cap refusal at 50,000 candidates; frozen inputs remain available. | D1-A complete solver/search or D1-B formally approved common universe; both exact methods then complete or prove infeasibility for all six scenarios. Candidate-limit refusal is not completion. |
| E3.2 | Classify road-closure and reduced-driving-tolerance non-exact failures. | `verified` | v14 diagnoses are retained. | Adapter-level regressions run both non-exact methods, retain `context_closed_route_selected:route_parent` and `day_time_exceeded:1`, and keep all four rows unranked. |
| E3.3 | Rerun and close the four-method comparison. | `blocked` | Blocker E3.1: metric integrity and failure policy are verified, but the exact-search strategy is not. | A new immutable 24-cell run passes provenance and route lineage, exact rows no longer fail from diagnostic caps, and every ranking/table gates on independent eligibility. |
| E3.C | Reduce oversized dashboard/notebook modules without semantic changes. | `verified` | E3.C1-E3.C4 pass; the 10-line orchestrator, 2-line wrapper, reviewed E3.C4 signature, 289 tests, validator, project checks, and four-width browser matrix are verified. | Preserve the reviewed E3.C4 and modular-dashboard oracles during additive product work. |
| E3.C1 | Decompose the package dashboard exporter. | `verified` | Five extracted responsibilities, exact asset contracts, and a 138-line writer. | Preserve 273-test, validator, and desktop/mobile interaction evidence. |
| E3.C2 | Move live non-renderer blueprint helpers into package ownership. | `verified` | The 29-function/9-constant core is package-owned; experiment runner has no legacy import; notebook aliases are identity-checked; parity and 273 tests pass. | Preserve the core contract through E3.C4. |
| E3.C3 | Isolate the legacy Folium renderer. | `verified` | Five package modules own the measured closure; `map_renderer.py` has no legacy blueprint import; the notebook re-exports identical package objects with no duplicates. | Normalized HTML/day-plan/route-debug signatures, edge contracts, 280 tests, validator, standalone import, and desktop/mobile browser checks pass. |
| E3.C4 | Decompose the renderer orchestrator and route-debug UI controller. | `verified` | Both size targets pass; the 390px selector is contained; exact data/Leaflet contracts, 23 focused tests, 81 regressions, 289 full tests, validator, project checks, and browser matrix pass. | Preserve the reviewed E3.C4 full-HTML signature and every unchanged legacy semantic contract. |
| E3.UX | Reframe canonical repair artifacts into a separately versioned customer/research product dashboard. | `verified` | E3.UX0–E3.UX4 pass on `runs/e3ux-weather-repair-demo-v6/dashboard_product/`; legacy parity remains frozen. | Preserve the additive artifact and reports; do not treat it as E3.3, E4, E5, or user-study evidence. |
| E3.UX0 | Audit current UIs, artifact combinations, state ownership, and compatibility boundaries. | `verified` | Canonical source matrix, truth-state matrix, frozen legacy hashes/interactions, additive-path decision, and rollback oracle are recorded in `docs/current/e3ux0_artifact_and_ui_audit.md`. | Preserve the source/legacy boundary. |
| E3.UX1 | Define product information architecture, design tokens, responsive/accessibility contracts, and low-fidelity wireframes. | `verified` | Customer/research IA, desktop/tablet/mobile wireframes, semantic tokens, and accessibility/component contracts are recorded and implemented. | Preserve the versioned design contract. |
| E3.UX2 | Implement a read-only artifact-grounded customer dashboard. | `verified` | Timeline, issue/result, changed/unchanged summary, comparison, evidence, contextual map, and customer truth states render from validated copied artifacts. | No mutation or UX5 controls. |
| E3.UX3 | Add research/evidence mode and complete comparison/failure views. | `verified` | Requested/executed methods, failure reasons, exact-cap state, lineage/hashes, route identity, certificate, `PlanDiff`, evidence, and eligibility are visible without truth recomputation. | Keep ineligible rows unranked and failures visible. |
| E3.UX4 | Close responsive, accessibility, integrity, security, browser, and read-only end-to-end gates. | `verified` | 26 product tests, 23 legacy focused tests, 81 regressions, Ruff, project checks, 315 full tests, both validators, artifact readback, accessibility/integrity reports, screenshot hashes, and the six-width browser matrix pass. | Preserve evidence and report the keyboard-backend limitation without inflating it into a user-study claim. |
| E3.UX5 | Prototype feature-gated repair/permission actions through existing interaction boundaries. | `deferred` | Deferred until E5 is verified and explicit experimental enablement is approved. | Default-disabled controls call existing interaction/pipeline boundaries, preserve session-scoped permission, create immutable continuation runs, and never enter E3/E4 evidence. |
| E4 | Establish robustness, ablations, failure taxonomy, and reproducibility. | `blocked` | Blocker E3.3: the comparison closeout is not verified. | Prespecified sensitivity and ablation runs finish; failures and noncoverage are reported; every table/figure is rebuilt from immutable manifests. |
| L1 | Verify the closest-work gap and safe novelty statement. | `ready` | Ready in parallel: search protocol and comparison fields are frozen. | Reproducible screening log, closest-work matrix, and evidence-bounded claim language are reviewed. |
| E5 | Freeze the interaction minimum viable architecture and integrate the deterministic scaffold behind explicit boundaries. | `deferred` | Deferred until E4; technical artifacts and evaluator contracts must be reproducible and the ethics boundary defined. | Typed semantic hypotheses, allow-listed programs, permission ceiling, consequence comparison, probe lineage, and execution/autonomy certificates pass frozen-fixture tests without entering E3/E4. |
| E6 | Pilot and validate anchor trajectories and the semantic benchmark. | `deferred` | Deferred until E5 schemas and authoring workflow are stable. | Ten pilot trajectories pass semantic, optimization, and permission review; scale-up coverage criteria and versioning rules are approved before 50+ authoring. |
| E7 | Implement certificate-first runtime and table-gated autonomy V1. | `deferred` | Deferred until E6; validated anchors and consequence labels must exist. | Commit/ask/no-material-difference certificates, hard compute caps, conservative fallback, and permission boundary tests pass. No learned model raises authority. |
| E8 | Evaluate interaction quality and prepare the second publication slice. | `deferred` | Deferred until E7 and ethics approval; technical and interaction benchmarks must be frozen. | Clarification, control, interruption, regret, acceptance, rollback, latency, and traceability evidence are reproducible. |
| E9 | Add advanced personalization, calibration, sentinels, and selective multi-agent communication. | `deferred` | E8 shows a measured need and supplies calibration data. | Each extension beats a simpler baseline under held-out evidence and preserves explicit permission ceilings. |

### Corrective local-product phases and gates

| Node | Objective | Status | Exit boundary |
|---|---|---|---|
| W4 | Provider, prompt, transcript, and deterministic evaluation implementation. | `implemented` offline | Preserve the implementation evidence; this status is not a G4 provider-verification verdict. |
| W4R | Correct direct map interaction, required-leg display, and exact Compare/Evidence behavior. | `in-progress` | Typed server-owned edits, validated geometry or explicit gaps, exact option resolution, deterministic checks, and no placeholder decision behavior. |
| G4R | Independently verify W4R user journeys and integrity. | `planned`, not verified | Map journey, route artifact, accessibility, UI/UX, content, visual, security/privacy, phase/status, and black-box audits all pass. |
| G4 | Verify the live provider after G4R. | `blocked` | Newly authorized post-fix smoke, fixed-24 evaluation, prescribed reasoning comparison, sanitized evidence, deterministic checks, and independent provider audits pass. |
| W5 | Implement Compare/Evidence decision integrity and durable Accept/Keep transactions. | `planned` | Entry requires verified G4; repository, journal, pointer CAS, restart, concurrency, and exact-evidence gates pass. |

The independent research sequence remains `D1 -> E3.1 -> E3.3 -> E4` with
its current statuses unchanged.

## Parallel Work and Contention

- E3.M and E3.2 are verified and must remain unchanged while D1 is resolved.
- E3.C is `verified`: E3.C1-E3.C4 pass and the reviewed output oracles are
  frozen. Later work must not alter exact-search, evaluator, or benchmark
  semantics.
- E3.UX0-E3.UX4 are `verified`. Their additive v6 artifact remains independent
  of D1/E3.1 and uses a separate `dashboard_product/` artifact category.
- E3.UX must not edit the exact-baseline implementation, benchmark method
  adapters, evaluator metrics, ranking rules, immutable E3 roots, legacy
  dashboard assets, or normalized renderer oracles. Shared contention is
  limited to additive pipeline-manifest/layout changes and must be protected by
  legacy regression tests.
- E3.UX5 is E5-dependent and remains disabled by default. Technical evidence UI
  (E3.UX0-E3.UX4) and interaction research (E3.UX5/E5+) are separate work.
- L1 may proceed alongside E3/E4. Literature results may narrow claim language but must not change frozen E3.3 methods or inputs.
- E5 design documentation may improve, but publication-mode interaction implementation and claims remain deferred until E4.
- E6–E9 must not change parent immutability, route validation, independent evaluation, or evidence lineage without a compatibility gate.

## Critical Path

The earliest technical publication path is:

```text
D1/E3.1 -> E3.3 -> E4 -> claim reconstruction -> venue decision
```

E3.1 remains the highest-risk node: v14 refused all exact cells at 50,000
candidates. The next critical-path action is the D1-A versus D1-B research
decision. E3.C4 may close in parallel without changing benchmark semantics;
after E3.C4 is verified, E3.UX0-E3.UX4 may also proceed in parallel and remain
off the technical critical path.

The separate corrective local-product path is:

```text
W4 implemented -> W4R -> G4R -> G4 -> W5 -> G5 -> W6 -> W7 -> W8
```

It may progress alongside D1, but it cannot change E3.1, E3.3, E4, E3.UX5, or
E5 status and is not publication evidence.

The earliest interaction publication path is:

```text
verified E4 + verified L1 -> E5 -> E6 -> E7 -> E8
```

E9 is optional follow-up work, not a prerequisite for the first interaction
evaluation.

## Milestones and Verification Gates

### M1 - Technical comparison closeout

- E3.M, E3.1, E3.2, and E3.3 are `verified`.
- All four requested method identities are present.
- Exact methods do not claim optimality after cap refusal.
- Failed and ineligible rows remain visible.
- Only evaluator-owned eligible rows enter rankings.

### M2 - Technical evidence package

- E4 is `verified`.
- Tables and figures are reconstructed from manifests.
- Sensitivity, ablation, failure, runtime, and reproducibility evidence are
  included.
- The technical claim is no broader than the completed method/scenario
  coverage.

### M2-UX - Product presentation closeout

- E3.C4 and E3.UX0-E3.UX4 are `verified`.
- Legacy research dashboards and all frozen parity contracts remain unchanged.
- The product path has independent version/hash/test/screenshot boundaries.
- Customer and research modes expose the same canonical truth with different
  disclosure depth.
- No UI or user-study claim is added to the technical evidence package.

### M3 - Interaction foundation

- E5 is `verified`.
- Hypothetical probes cannot mutate or authorize a real repair.
- Preference inference cannot raise the autonomy ceiling.
- Post-solve changes always produce a new child and new evaluation.

### M4 - Interaction evidence

- E6-E8 are `verified`.
- Anchor and human evidence sources remain distinct from synthetic generation.
- Clarification and autonomy claims are supported by held-out and study evidence,
  not architecture diagrams.

## Risks and Mitigations

| Risk | Mitigation and stop rule |
|---|---|
| Exact search remains intractable. | Profile one scenario/method first; use a complete solver-backed strategy or formally narrow the common declared universe. Never raise caps blindly. |
| Oversized dashboard/notebook code hides contract drift. | Extract one responsibility at a time behind UI/export snapshots; do not combine refactoring with benchmark semantic changes. |
| Product redesign overwrites or weakens a research parity artifact. | Use a separate `dashboard_product/` category, freeze legacy hashes, and require rollback/compatibility tests before product verification. |
| Product presentation invents truth from incomplete artifacts. | Validate manifest-declared paths, hashes, lineage, provenance, numbers, and certificate ownership before building view models; render explicit missing/stale/mismatch states. |
| Product polish is mistaken for interaction or publication evidence. | Keep E3.UX0-E3.UX4 read-only and off the critical path; keep E3.UX5 E5-dependent, disabled, and excluded from E3/E4. |
| Non-exact failures are hidden to improve rankings. | Retain every row and require an explicit expected-infeasibility, contract-mismatch, or defect classification. |
| Evidence-ready is confused with performance-complete. | Keep E3.0 and E3.3 as separate gates and use distinct labels in manifests and prose. |
| Rapid literature growth erodes novelty. | Maintain L1 as a living, reproducible closest-work matrix and use narrow mechanism-level claims. |
| Interaction scope delays technical evidence. | Keep E5-E9 deferred until E4 and exclude interaction entry points from publication runs. |
| Inferred preferences leak into permission. | Store and evaluate permission separately; explicit contract is always the maximum action ceiling. |
| Post-solve mutation invalidates certification. | Require new child lineage and independent re-evaluation after every content change. |
| Multi-agent complexity adds noise without benefit. | Keep E9 optional and require comparison against a single-agent/central-only baseline. |

## Tracking Update Policy

- Planning edits may clarify scope, dependencies, gates, and evidence paths.
- A phase status may move only when the destination state's evidence exists.
- Legacy checklist boxes remain unchanged during planning-only work.
- `docs/current/current_problem_manifest.md` changes only when implementation
  evidence changes.
- `docs/planning/current_execution_plan.md` changes when the immediate queue,
  blockers, or gate status changes.
- This master roadmap changes when phase order, scope, or publication-slice
  boundaries change.
- The E3.UX umbrella and E3.UX0–E3.UX4 are `verified`; CP-009 is resolved by
  same-worktree evidence. E3.UX5 remains `deferred`.

## Recommended Next Phase

The next technical action is Gate D1 using the v14 cap-refusal evidence and
`docs/planning/e3_exact_baseline_scalability_phase_plan.md`. Preserve the
verified E3.C and E3.UX artifacts while making that decision.

For D1:

1. record D1-A or D1-B in `docs/current/e3_exact_baseline_strategy_decision.md`;
2. preserve the verified E3.M metric/UI and E3.2 failure-classification contracts;
3. implement the chosen exact-universe strategy and prove completion or infeasibility for all 12 exact cells;
4. start E3.3 only after E3.1 is verified.

Do not reopen E3.C or E3.UX without a scoped compatibility gate, and do not
mark E3.1 verified after a cap refusal, partial search, or method-specific
narrowed universe.

## Parallel Product Track

The Itinerary Repair Copilot is a separate local product prototype. It consumes
validated immutable artifacts and may close its own AUD-0/W0-W8 implementation
gates, but it does not provide publication, participant-study, E3.UX5, or E5
evidence. Mutable application state remains outside the research run.

```text
AUD-0 -> W0/G0 -> W1 -> W1M/G1 -> W2/G2 -> W3/G3 -> W4/G4
       -> W5/G5 -> W6/G6 -> W7/G7 -> W8/G8 + user sign-off
```

Current product state: AUD-0, W0/G0, W1M/G1, corrected-v2 W2/G2, and
corrected-v2 W3/G3 are `verified`; historical W1 is `implemented`. The closed
local MapLibre/PMTiles package, continuous three-plan v2 workspace, recalculated
metrics/certificates, persistent typed previews, live desktop/mobile journeys,
security/state, accessibility/content, and status truth passed independent
revalidation. Historical v1 reports remain historical evidence only. Atlas remains
a deferred explicit backup. W4 is `implemented`; G4 is `blocked` on authorized live provider evidence, and W5-W8 remain `planned`. CP-010
remains `in-progress` until G8. Audit and implementation agents are independent,
and a blocking specialist finding stops the relevant product gate.

There is no status-propagation edge from any product gate to E3.1, E3.3, E4,
E3.UX5, or E5. The next research action remains D1 for E3.1; the next product
product action is the credential-gated G4 smoke and fixed-24 evaluation.
Each track requires its own evidence and named external dependencies.
