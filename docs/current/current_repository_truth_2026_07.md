# Current Repository Truth — July 2026

**Evidence date:** 2026-07-29  
**Purpose:** repository-grounded implementation and gate snapshot  
**Scope:** current source, current tests, immutable E3 artifacts, and active
planning documents

## Authority Used

This report applies the repository authority order without treating older plans
as current status:

1. current source, same-worktree tests, and immutable run artifacts;
2. `docs/current/current_problem_manifest.md`;
3. `docs/planning/current_execution_plan.md`;
4. `docs/planning/travel_itinerary_repair_technical_specification.md`;
5. `docs/planning/research_pipeline_and_gate_map.md`;
6. dedicated current E3 phase plans; and
7. older integrated, stabilization, detailed, and Phase 0 plans as historical
   acceptance-criteria sources only.

## Same-Worktree Verification

The worktree contains extensive pre-existing modified and untracked research
work. This report preserves that work and does not interpret a clean Git status
as a correctness requirement.

The first focused test run encountered `WinError 5` while pytest tried to scan
the configured shared root
`.codex_tmp_pytest/pytest/pytest-of-Ztang_Yit_Xiaang`. Tests that did not require
`tmp_path` reached `102 passed`; 31 tests stopped during fixture setup. The same
matrix was repeated with an isolated repository-local `--basetemp` and passed:

```text
133 passed in 16.23s
```

The matrix covered the authoritative pipeline entry point, pipeline runner,
artifact lineage, plan models/repository/diff, route matrix, progressive and
exact repair, independent evaluation, explanations, publication contracts,
six disruptions, the interaction scaffold, modular dashboard contracts, and
the package-owned Folium renderer/selector/core compatibility boundaries.

This focused same-worktree result supplements rather than replaces the
historical full-suite evidence recorded in the owning gate documents. A new
full-suite result is still required for later implementation closeout.

## Current Gate Truth

| Gate | Current state | Evidence basis | Remaining condition |
|---|---|---|---|
| E1 — authoritative entry point | `verified` | `scripts/run_research_pipeline.py`, `run_research_pipeline()`, thin-notebook boundary tests, and `docs/current/e1_authoritative_entrypoint_closeout.md` | Preserve the package runner and thin compatibility boundaries. |
| E2 — road-valid route bundle | `verified` | Base bundle preflight plus immutable v14 route lineage: `route_bundle_a60c80047098a3b6`, `route_matrix_68ab535465b06808`, and complete route/snap coverage with no fallback | Future E3 runs must reuse or explicitly supersede the frozen route lineage. |
| E3.0 — evidence-complete diagnostic | `verified` | v14 closeout contains 24 unique method-scenario rows, all four method IDs, retained failures, complete method provenance, and shared route inputs | Do not interpret evidence completeness as method completion. |
| E3.M — metric and UI integrity | `verified` | Evaluator-owned quality, `PlanDiff`-owned preservation, finite-number rejection, fail-closed ranking, direction-correct bars, null handling, and escaped text are covered by source/tests and the recorded 265-test closeout | v14 predates E3.M; a future E3.3 root must be regenerated under the hardened contract. |
| E3.1 — exact-baseline completion | `blocked` | All 12 exact v14 cells report `complete_candidate_limit_exceeded:50000` and have no child, ranking eligibility, or optimality certificate | D1-A complete search or D1-B a formally approved method-independent common universe, followed by complete/infeasible evidence for all 12 exact cells. |
| E3.2 — non-exact failure policy | `verified` | Adapter regressions retain physical causes for road closure and reduced driving tolerance and keep all four non-exact rows unranked | Preserve the failure diagnoses in E3.3. |
| E3.3 — four-method closeout | `blocked` | v14 has 8 ranking-eligible rows and 16 retained failures; neither exact method completed | E3.1 must be `verified`, then a new immutable 24-cell run must pass the current contracts. |
| E3.C1 | `verified` | Package-owned dashboard evaluation/assets/data-loader/map-controls/UI modules and frozen asset contracts | Preserve the legacy modular dashboard. |
| E3.C2 | `verified` | `blueprint_core.py` owns the migrated core and the notebook re-exports compatibility aliases | Preserve identity and parity tests. |
| E3.C3 | `verified` | Package-owned renderer modules, thin notebook facade, frozen HTML/day-plan/route-debug signatures, and renderer parity tests | Preserve the legacy Folium artifact. |
| E3.C4 | `verified` | Renderer and selector responsibilities are decomposed; the open selector is contained at 390px; focused, regression, full-suite, validator, project-check, and four-width browser gates pass | Freeze the reviewed E3.C4 signature and legacy semantics during additive product work. |
| E3.UX | `verified` | E3.UX0–E3.UX4 pass on the separately versioned v6 product artifact with canonical source/asset/screenshot hashes, 26 product tests, 315 full tests, both validators, project checks, accessibility/integrity reports, and a six-width browser matrix | Preserve v6 and legacy oracles. E3.UX5 stays deferred; D1/E3.1/E3.3/E4 are unchanged. |
| E4 | `blocked` | No claim-ready four-method E3.3 closeout exists | E3.3 must be `verified`. |
| E5 | `deferred` | A deterministic permission-aware scaffold exists behind an explicit entry point, but no interaction-study gate is satisfied | Keep default interaction disabled; E3.UX5 remains E5-dependent and experimental. |

## Major Subsystem Inventory

| Subsystem | Implemented status | Authoritative module(s) | Authoritative tests | Generated artifacts | Known limitations | Owning gate |
|---|---|---|---|---|---|---|
| Immutable plan and run lineage | `verified` foundation | `src/itinerary_system/research_artifacts.py`; `src/itinerary_system/pipeline_runner.py`; `src/itinerary_system/product_dashboard_renderer.py` | `tests/test_artifact_lineage.py`; `tests/test_pipeline_runner.py`; `tests/test_research_pipeline_entrypoint.py`; `tests/product_dashboard/` | `runs/<run_id>/manifest.json`; redacted config; dataset manifest; canonical artifacts; legacy `dashboard/`; additive `dashboard_product/` | The product exporter creates a derived non-overwritable snapshot; it does not turn product output into canonical optimizer/evaluator evidence. | E1 / E3 / E3.UX |
| `PlanArtifactV2` | `verified` foundation | `src/itinerary_system/research_artifacts.py` | `tests/test_artifact_lineage.py`; `tests/plans/test_models.py` | `runs/<run_id>/plans/<plan_id>.json` | Compatibility with older plan records remains intentional; content changes require a new hash and child lineage. | E1 / E3 |
| `OwnedConstraint` and `OwnershipPolicy` | `verified` foundation | `src/itinerary_system/plans/models.py` | `tests/plans/test_models.py`; repair and evaluator tests | Owned constraints inside plan artifacts; permission evidence in interaction artifacts | Permission cannot be inferred from display state or hypothetical probes. | E3 / E5 |
| `PlanRepository` | `verified` foundation | `src/itinerary_system/plans/repository.py` | `tests/plans/test_repository.py` | Append-only plan JSON and repository index under a run's `rw/` boundary when used | It is a plan store, not an optimizer or UI state store. | E3 |
| `PlanDiff` | `verified` foundation | `src/itinerary_system/plans/models.py`; `src/itinerary_system/plans/diff.py` | `tests/plans/test_diff.py`; pipeline and interaction integration tests | `runs/<run_id>/diffs/<diff_id>.json` | Missing or mismatched parent/child evidence must remain a distinct UI failure state; it cannot be recomputed from incomplete display data. | E3.M / E3.UX |
| `RouteMatrix` and road evidence | `verified` for the frozen slice | `src/itinerary_system/routing/matrix.py`; `src/itinerary_system/routing/evidence_bundle.py` | `tests/routing/test_route_matrix.py`; `tests/routing/test_evidence_bundle.py`; `tests/benchmark/test_route_coverage.py` | Route bundle manifests, validated cache, content-addressed matrix JSON/CSV, and per-run routing records | Fallback or missing cells are not road-valid; route evidence is scoped to the frozen corridor/universe. | E2 |
| Progressive ownership-aware repair | `verified` implementation foundation | `src/itinerary_system/repair/neighborhood.py`; `master_model.py`; `lexicographic.py`; `day_route_solver.py`; `progressive.py` | `tests/repair/test_neighborhood.py`; `test_master_model.py`; `test_lexicographic.py`; `test_day_route_solver.py`; `test_progressive.py` | Planner attempts, accepted child, diff, certificate, explanation, and metrics inside an immutable run | Publication comparison remains blocked by the exact baselines, not by absence of progressive repair. | E3 |
| Exact baselines | `implemented`; completion gate `blocked` | `src/itinerary_system/repair/exact_baselines.py` | `tests/repair/test_exact_baselines.py`; `test_exact_baseline_pipeline.py`; `test_exact_baseline_scalability_gate.py` | Failed exact planner rows and cap-refusal diagnostics in v14 | Raw candidate lower bounds far exceed 50,000; cap refusal is incomplete search and supplies no certificate. | E3.1 |
| Independent evaluator and certificate | `verified` | `src/itinerary_system/evaluation/plan_evaluator.py`; `certificate.py` | `tests/evaluation/test_plan_evaluator.py`; `tests/benchmark/test_no_leakage.py` | `runs/<run_id>/evaluations/<certificate_id>.json`; evaluator-owned metrics | Missing, stale, mismatched, nonfinite, route-invalid, or planner-invalid evidence fails closed. | E3.M |
| Evidence-linked explanations | `verified` foundation | `src/itinerary_system/explanation/evidence.py`; `counterfactual.py`; `verbalizer.py` | `tests/explanation/test_evidence.py`; `test_evidence_builder.py`; `test_counterfactual.py` | `runs/<run_id>/explanations/<evidence_id>.json` | Unsupported numerical or causal claims are omitted; no UI may synthesize missing evidence. | E3 / E3.UX |
| Authoritative research pipeline | `verified` | `src/itinerary_system/pipeline_runner.py`; `scripts/run_research_pipeline.py` | `tests/test_pipeline_runner.py`; `test_research_pipeline_entrypoint.py`; `test_research_pipeline_cli_summary.py` | Immutable run directory and manifest with content paths | The notebook/legacy experiment surfaces remain compatibility consumers, not the authority. | E1 |
| Benchmark adapters and six disruptions | `verified` contract; comparison closeout `blocked` | `src/itinerary_system/benchmark/disruptions.py`; `methods.py`; `runner.py`; `metrics.py`; `publication.py` | `tests/benchmark/test_disruptions.py`; `test_method_adapters.py`; `test_publication_method_factory.py`; `test_publication_contract.py`; `test_no_leakage.py` | `benchmark/manifest.json`; `metrics/benchmark_metrics.jsonl`; E3 closeout | `publication_ready: true` describes evidence completeness. Only independently eligible rows may be ranked. | E3.0 / E3.1 / E3.2 / E3.3 |
| Permission-aware interaction scaffold | `implemented`; overall phase `deferred` | `src/itinerary_system/interaction/`; `scripts/run_permission_aware_repair.py` | `tests/interaction/` | Paused interaction artifacts and authorized continuation runs under an explicit entry point | Default mode is disabled; a probe is hypothetical and cannot enter accepted plans or receive execution authority. | E5 |
| Legacy modular dashboard | `verified` parity artifact | `src/itinerary_system/map_exporter.py`; `dashboard_evaluation.py`; `dashboard_assets.py`; `dashboard_data_loader.py`; `dashboard_map_controls.py`; `dashboard_ui.py` | `tests/test_evaluation_dashboard_contract.py`; `tests/test_configurable_itinerary_system.py`; dashboard validator | `results/figures/full_interactive_dashboard/`; `evaluation.html`; versioned CSS/JS/JSON artifacts | Information architecture is layer/research oriented and does not provide the full parent → disruption → child → diff → certificate → evidence review flow. Frozen hashes must not be reused as product-UI acceptance. | E3.C1 / E3.M |
| Legacy Folium renderer and selector | E3.C1–C4 `verified` | `blueprint_core.py`; `blueprint_day_plans.py`; `blueprint_render_primitives.py`; `blueprint_render_layers.py`; `blueprint_render_panels.py`; `blueprint_renderer.py`; `blueprint_renderer_sections.py`; `blueprint_route_selector.py`; `map_renderer.py` | `tests/test_blueprint_core_parity.py`; `test_blueprint_renderer_parity.py`; `test_blueprint_route_selector.py`; `test_legacy_blueprint_boundary.py` | Folium HTML, day-plan CSV, route-debug CSV, and frozen normalized signatures | Desktop remains 430px at left 74; below 520px the open selector uses a 12px gutter and viewport-derived width. The reviewed E3.C4 normalized HTML signature is frozen. | E3.C4 |
| Notebook compatibility facade | `verified` compatibility boundary | `notebook/blueprint_trip_map.py`; `notebook/production_system_blueprint.ipynb` | `tests/test_production_notebook_boundary.py`; `tests/test_legacy_blueprint_boundary.py` | Notebook display references and legacy exports | Compatibility-only; new business logic and product dashboard logic must remain package-owned. | E1 / E3.C |

## Immutable E3 Artifact Readback

The latest immutable diagnostic is:

`tmp_test/research_pipeline_raw/e3-real-production-20260725-optimized-v14-cap50000`

Its readback establishes:

- four locked methods and six frozen scenarios;
- 24 unique rows;
- 8 independently ranking-eligible rows;
- 16 retained failures;
- complete method provenance and shared route lineage;
- route bundle `route_bundle_a60c80047098a3b6`;
- route matrix `route_matrix_68ab535465b06808`;
- `publication_ready: true` under the evidence-completeness contract; and
- 12 exact cap refusals at `complete_candidate_limit_exceeded:50000`.

The v14 artifacts predate E3.M hardening. They are immutable historical
diagnostic evidence and must not be rewritten, copied into a new UI as if
current, or treated as E3.3 completion evidence.

## Planning Conflicts and Decisions

| Conflict | Higher-authority evidence | Decision |
|---|---|---|
| The dedicated E3.1 plan said `ready`, cited v13, and treated E3.2 as unresolved. | Current manifest, current execution plan, v14 closeout, current benchmark contract, and the exact-baseline audit say E3.1 is blocked on D1 and E3.2 is verified. | Reconciled the dedicated E3.1 plan to `blocked`, v14, and the verified E3.2 handoff. |
| The roadmap's “Verified Current State” still named v13 and `max_complete_candidates=1`. | The immutable v14 closeout is newer and uses 50,000 while preserving the same incompleteness conclusion. | Updated the roadmap evidence references without changing E3.0/E3.1/E3.3 semantics. |
| Technical-specification sections 1 and 10 describe components as missing based on the 2026-06-30 baseline. | Current source and the 133-test matrix prove plans, diff, progressive repair, evaluator, explanation, benchmark, pipeline, and interaction packages now exist. | Treat those dated sections as implementation-baseline history; use this report and the current manifest for current status. Preserve the technical specification's contracts and invariants. |
| Existing E3.C UIs are frozen parity artifacts, but the requested experience is a new product information architecture. | Frozen hashes/tests and the current layer-oriented exporter/renderer establish both the compatibility obligation and the UX gap. | Keep E3.C4 parity work separate. Plan E3.UX as a new `dashboard_product/` artifact category with its own manifest, assets, tests, screenshots, and rollback. |

## Current Limitations and Stop Conditions

- E3.C4 is verified by same-worktree mobile selector containment and the final
  browser assertions for both legacy UIs.
- E3.UX0–E3.UX4 are verified through the separate v6 product artifact
  boundary; E3.UX5 remains deferred.
- E3.1/E3.3/E4 remain blocked regardless of UI work.
- No current artifact supports a user-study, interaction-effect, exact-method
  completion, four-method superiority, or universal road-validity claim.
- A product adapter must stop on parent mutation, lineage/hash mismatch,
  method-identity loss, unsupported route-validity claims, planner/evaluator
  ownership confusion, nonfinite numbers, ranked ineligible rows, incomplete
  exact search described as complete, ungrounded explanation claims, or a
  hypothetical probe presented as executable.
