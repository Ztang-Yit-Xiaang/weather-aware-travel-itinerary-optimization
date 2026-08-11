# Current Execution Plan

**Status:** Active implementation plan  
**Effective date:** 2026-07-29  
**Authority:** Use this document for near-term execution. The technical specification remains the canonical contract; `docs/current/current_problem_manifest.md` remains the implementation-status source.

## Decision

Keep the first publication slice focused on a reproducible, road-valid comparison of minimal-change itinerary repair. Do not promote an evidence-complete diagnostic into a complete performance comparison. LLM preference inference, participant studies, nationwide coverage, and autonomous monitoring remain deferred.

## Gate Status

| Gate | Status | Current evidence | Exit condition |
|---|---|---|---|
| E1 — authoritative entry point | `verified` | Package CLI, thin notebook boundary, immutable smoke evidence | No change. |
| E2 — road-valid benchmark bundle | `verified` | Expanded E3 matrix `route_matrix_68ab535465b06808`; 223/223 road-and-snap-validated cells, no fallback | No change. |
| E3.0 — real paired diagnostic | `verified` | v14 contains all 24 method-scenario rows, shared route lineage, retained failures, and `publication_ready: true`; this is an evidence gate, not performance completion | Evidence completeness must remain distinct from performance completeness. |
| E3.M — metric and UI integrity | `verified` | Evaluator-only quality metrics, finite-number checks, fail-closed ranking, sequence-grounded utility, honest interaction metrics, canonical UI methods, direction-correct bars, and escaped UI text; focused matrix 45/45; Ruff, 265/265 full pytest, project checks, and real dashboard export validation passed | Source and generated dashboard artifacts satisfy the current integrity contract; future benchmark artifacts must be regenerated under it. |
| E3.1 — exact-baseline completion | `blocked` | Blocker D1-S0: v14 dropped source lodging assignments; its synthetic hotel replacement lacks catalog/access/route evidence and existing lodging decisions are not coupled to route anchors. Separately, all 12 exact cells refused at the 50,000 cap; `gurobi_exact_v2.py` is currently a non-solving representation/index scaffold. | First pass a prospective D1-S0 semantic-validity gate with source-backed lodging alternatives, lodging-dependent routing, evaluator enforcement, and real diff/lineage. Then approve D1-A complete search or D1-B a frozen method-independent universe and complete or prove infeasibility for all 12 exact cells. |
| E3.2 — non-exact failure policy | `verified` | Adapter-level regression runs both non-exact methods on road closure and reduced driving tolerance; all four rows fail, remain unranked, and retain the physical-cause tokens | No change; preserve these diagnoses in E3.3. |
| E3.3 — four-method closeout | `blocked` | Blocker E3.1: v14 is not a complete four-method comparison | New immutable 24-cell run uses the approved universe, current metric contract, complete provenance/route lineage, and eligibility-gated tables. |
| E3.C - code maintainability | `verified` | E3.C1-E3.C4 pass. The open Folium selector is contained at 390px without changing desktop geometry, route order/defaults, data hashes, or Leaflet counts. Twenty-three focused tests, 81 regressions, Ruff, validator, project checks, 289 full tests, and 1440/768/520/390 browser checks pass. | Freeze the reviewed E3.C4 normalized HTML baseline `a06583549a135688e62d663ff5c6197074e96f3a0bad57d5cb791f37273fc2bb` and all other legacy contracts during E3.UX. |
| E3.UX — artifact-grounded product dashboard reframe | `verified` | E3.UX0–E3.UX4 produced `runs/e3ux-weather-repair-demo-v6/dashboard_product/`: product manifest/version, canonical source and asset hashes, customer/research modes, 26 focused tests, 23 legacy focused tests, 81 regressions, Ruff, both validators, project checks, 315 full tests, accessibility/integrity reports, and the 1440/1024/768/430/390/360 browser matrix pass. | Preserve the additive v6 artifact and legacy oracles. E3.UX5 stays deferred and E5-dependent. |
| E4 — robustness and paper evidence | `blocked` | Blocker E3.3: no claim-ready four-method closeout | Prespecified sensitivity, ablation, failure, runtime, and reproducibility evidence is rebuilt from immutable manifests. |
| E5 — interaction extension | `deferred` | Deferred until E4; deterministic scaffold exists behind an explicit entry point | No publication-mode or participant claims before E4 and ethics gates. |

## Current Truth

The repository has immutable plan/run lineage, typed diffs, ownership-aware progressive repair, three standalone baselines, independent evaluation, evidence-linked explanations, an authoritative runner, six deterministic disruption families, and two evidence-complete 24-row diagnostics.

The current immutable diagnostic is:

`tmp_test/research_pipeline_raw/e3-real-production-20260725-optimized-v14-cap50000`

Its closeout records 8 ranking-eligible rows and 16 retained failures. The deterministic heuristic and progressive repair each pass weather deterioration, hotel unavailability, attraction closure, and new must-visit. Road closure and reduced driving tolerance remain non-exact failures. Both exact methods refuse all six scenarios at the 50,000-candidate safety cap. Therefore:

- `publication_ready: true` means the evidence bundle is structurally complete;
- it does not mean every method completed;
- the exact-baseline tractability gate is not resolved;
- no four-method superiority claim is permitted;
- v14 predates E3.M hardening and must not be reused as if it had been generated under the new metric/UI contract.

The existing Folium and modular dashboards are frozen E3.C/E3.M research
artifacts. E3.C4 intentionally versioned the normalized Folium HTML signature
for one reviewed mobile-only selector rule; its new exact hash and the unchanged
modular CSS/JS, `evaluation.html`/`evaluation_metrics.json`,
day-plan/route-debug, Leaflet-count, and browser contracts are now frozen.
E3.UX is additive: it uses a new
`runs/<run_id>/dashboard_product/` artifact category and does not reinterpret
the existing `dashboard/` category.

## Locked Scope

- **Question:** Does ownership-aware, smallest-radius repair preserve more of an accepted itinerary than context-blind repair or full replanning while remaining independently eligible?
- **Contexts:** weather burden, authoritative closure, mobility/travel-time disruption, and user-approved pace/accessibility.
- **Geography:** one reproducible California corridor; a second corridor is optional only after the first passes all gates.
- **Methods:** `context_blind_solver`, `deterministic_context_aware_heuristic`, `progressive_sequential_lexicographic_repair`, and `full_reoptimization`.
- **Outputs:** immutable inputs/runs, typed diff, route evidence, independent certificate, evaluator-owned component metrics, runtime, failure diagnosis, and evidence-linked explanation.

## Test Strategy

Use a layered, fail-closed strategy:

1. **Contract/unit tests:** numeric finiteness, metric ownership, status allowlists, utility denominator/unit consistency, duplicate/off-sequence records, permission provenance, and missing diff evidence.
2. **Integration tests:** exact and non-exact adapters through the package runner, independent evaluator, immutable artifact lineage, and interaction continuation boundaries.
3. **UI contract tests:** canonical method IDs, honest no-data state, correct higher/lower-is-better normalization, null rendering, and HTML escaping.
4. **Product UI contract tests:** manifest-declared canonical inputs, path/hash/lineage validation, explicit truth states, customer/research labels, no hidden evaluator/optimizer recomputation, semantic HTML, keyboard/focus behavior, responsive containment, and unchanged legacy hashes.
5. **Regression tests:** dedicated frozen cases for road closure, reduced driving tolerance, exact-cap refusal, and post-solve certification invalidation.
6. **Repository validation:** Ruff, complete pytest suite, project checks, legacy dashboard export validation, and the separate product-dashboard validator once implemented.
7. **Evidence validation:** read back immutable manifests after process exit; verify 24 unique cells, route/source hashes, method identity, failed-row retention, eligibility-gated ranking, and product asset/source hashes.

A written plan or passing unit suite is not publication evidence. E3.3 exits only after the immutable rerun and readback pass.

## Immediate Queue

1. `[VERIFIED]` E3.M passed Ruff, 265/265 full pytest, project checks, and real dashboard export validation; the generated dashboard was refreshed under the new contract.
2. `[VERIFIED]` E3.2 adapter-level regressions retain road-closure and reduced-driving-tolerance physical causes through both non-exact methods; all four rows fail closed and stay unranked.
3. `[BLOCKED]` Pass D1-S0 benchmark semantic validity before any representative exact solve: restore source-backed parent lodging assignments, add a distinct located replacement, couple lodging to day-route anchors, expand route evidence, require lodging in independent evaluation, and preserve v14 unchanged.
4. `[DECISION REQUIRED]` After D1-S0, resolve D1:
   - **D1-A:** replace enumeration with a complete solver/search for the existing universe; or
   - **D1-B:** freeze a defensible, method-independent common universe and apply it to all four methods.
5. `[BLOCKED]` Do not run E3.3 until E3.1 is verified; preserve the verified E3.M and E3.2 contracts.
6. `[VERIFIED]` E3.C1-E3.C4. The 390px selector is contained, both legacy UIs pass the four-width browser matrix, and exact semantic/data contracts remain frozen under the reviewed E3.C4 HTML signature.
7. `[VERIFIED]` E3.UX0–E3.UX4 pass on the separately versioned v6 product snapshot. Preserve its product/source/screenshot hashes and the reviewed E3.C4/modular oracles; this result does not change D1/E3.1/E3.3/E4.
8. `[DEFERRED]` E3.UX5 belongs to the E5 interaction boundary. Keep it disabled by default and do not treat product UI implementation as E5 evidence.
9. `[BLOCKED]` Begin E4 only after the E3.3 closeout is verified.

## E3.C Extraction Evidence — 2026-07-28

- Extracted canonical evaluation payload construction and standalone page rendering into `src/itinerary_system/dashboard_evaluation.py`.
- Retained thin `_evaluation_metrics()` and `_write_evaluation_page()` wrappers so exporter call sites and private compatibility paths remain unchanged.
- Proved the generated `evaluation.html` byte-for-byte identical and `evaluation_metrics.json` structure-equivalent to the current artifacts.
- Passed focused dashboard tests, the map-export integration test, Ruff, full pytest (273/273), project checks, dashboard export validation, and Chrome checks at 1440 px and 390 px with no errors or horizontal overflow.
- Extracted the 18,996-byte stylesheet into `src/itinerary_system/dashboard_assets.py`; the integration test requires byte equality and SHA-256 `82e968c2b88007b7aa8be6cf8d2e4c1413fde68fdcfb23db2e357afef01d2b1f`.
- Extracted the 7,459-byte loader into `src/itinerary_system/dashboard_data_loader.py`; the integration test requires byte equality and SHA-256 `aafebd304c2e08d5cbf6df5b78196cbe787cbc246ee2ff17f704f23bcaea0409`, while Chrome confirms its loader globals register in both viewports.
- Extracted the 47,461-byte map-controls runtime into `src/itinerary_system/dashboard_map_controls.py`; the integration test requires byte equality and SHA-256 `a6f1be3515033e832852a6e7b2cd10b642c2d72d68071c65b4927d4e23997bde`, while Chrome exercises route selection, zoom, and playback activation in both viewports.
- Extracted the 43,915-byte UI runtime into `src/itinerary_system/dashboard_ui.py`; the integration test requires byte equality and SHA-256 `3506e86ef671e467c3a61634c76e0ea08ced4d41c1623a54fd9bfb734b7fce74`, while Chrome verifies mode switching, collapse/expand, registered render APIs, and representative rendering in both viewports.
- E3.C2 moved the measured 29-function/9-constant closure into the 927-line `src/itinerary_system/blueprint_core.py`; `notebook/blueprint_trip_map.py` fell from 6,610 to 5,788 lines, re-exports identical core objects, retains its renderer, and passes six frozen parity groups plus standalone import.
- E3.C3 moved 57 renderer functions and 12 constants exactly once into `blueprint_day_plans.py` (340 lines), `blueprint_render_primitives.py` (838), `blueprint_render_layers.py` (1,523), `blueprint_render_panels.py` (1,628), and `blueprint_renderer.py` (976). The notebook facade is 919 lines/12 functions and `map_renderer.py` has no legacy blueprint import.
- The E3.C3 entry baseline was normalized HTML SHA-256 `a2fbbb85c56019ccd5f64315cbe536965a14c7858c9a2bcc971b77d96e27c320`. E3.C4 intentionally versions only that full-HTML signature to `a06583549a135688e62d663ff5c6197074e96f3a0bad57d5cb791f37273fc2bb` for the reviewed mobile media rule. Day-plan SHA-256 `6bb4a3a40d76a07ba62e02bf055fd40fece8853b09fa76be07cffe16b7f88e27`, normalized route-debug SHA-256 `b723926ceb77887660a7730104e3e2ebc891ca668ab33f4b43fe9d8f9444ff7e`, 7 day-plan rows, 335 debug rows, and exact Leaflet object counts remain unchanged.
- The current gate passed Ruff, 5 context tests, 280 full tests, dashboard validation, standalone notebook import, and fallback-browser checks at 1440px/390px for both UIs with no errors, pending resources, or horizontal overflow. Route filter/zoom, customer/research switching, collapse/expand, and next/play/pause interactions work.
- E3.C4 composes nine renderer sections through request-scoped state and
  separates selector validation/model, markup/CSS, and client runtime. A
  mobile-only rule keeps the desktop 430px/74px geometry while containing the
  expanded selector at 390px. The focused 23-test matrix, 81 regressions, Ruff,
  5 context tests, 289 full tests, dashboard validation, project checks, and
  1440/768/520/390 interactions for both legacy UIs pass.
- The minimal-implementation audit removed two write-only state fields and
  duplicate scaffolding. The old 390px artifact measured left 74/right 504;
  the same-worktree render measures left 12/right 378.4 with no clipping.
### E3.C subgates

| Subgate | Status | Exit evidence |
|---|---|---|
| E3.C1 — dashboard exporter decomposition | `verified` | `_write_full_dashboard()` is 138 lines; evaluation, CSS, loader, map-controls, and UI modules emit byte-equivalent artifacts under frozen hashes; the current 273-test suite and interactive browser checks pass. |
| E3.C2 — legacy blueprint core migration | `verified` | `blueprint_core.py` owns 29 functions/9 constants; six parity groups and frozen contracts pass; the notebook re-exports identical objects with no duplicates; `experiment_runner.py` has no legacy blueprint import; standalone import and 273 tests pass. |
| E3.C3 - legacy renderer extraction | `verified` | Five package modules own 57 functions/12 constants exactly once; the notebook is a 919-line compatibility facade; `map_renderer.py` calls the package renderer; full render signatures, 280 tests, validator, standalone import, and desktop/mobile browser interactions pass. |
| E3.C4 - renderer/UI controller decomposition | `verified` | The renderer is a 10-line orchestrator over nine request-scoped sections; the route-debug wrapper is 2 lines; the 390px selector is contained; 23 focused tests, 81 regressions, 289 full tests, Ruff, validator, project checks, and the four-width browser matrix pass. |
| E3.UX - artifact-grounded product dashboard reframe | `verified` | E3.UX0–E3.UX4 pass through the separate v6 `dashboard_product/` manifest/assets/test/browser boundary. Customer/research views consume canonical artifacts; 26 product tests, 315 full tests, both validators, project checks, accessibility/integrity reports, versioned screenshots, and the six-width browser matrix pass. E3.UX5 remains disabled and E5-dependent. |

## E3.UX Closeout Evidence — 2026-07-29

- Source artifacts were copied from immutable run `benchmark_158cf6d48be8`
  into non-overwritable derived run `e3ux_weather_repair_demo_v6`; the source
  run was not edited.
- `product-dashboard-manifest-v1`, product version `1.0.0`, full source/asset
  hashes, screenshot hashes, and the read-only compatibility boundary validate.
- Customer and research modes share one validated view model. Exact-cap
  failures remain incomplete/unranked; null remains null; certificate,
  `PlanDiff`, route lineage, evidence, and requested/executed methods remain
  source-owned.
- Twenty-six product tests, 23 legacy focused tests, 81 legacy regressions,
  Ruff, the legacy and product validators, project checks, and 315 full tests
  pass in the same worktree.
- Browser checks pass at 1440, 1024, 768, 430, 390, and 360px with no document
  overflow, clipped controls, product console issues, incomplete images, or
  map initialization failure.
- E3.UX5 was not started. E5 remains `deferred`.
## Stop Conditions

Stop result generation if any of the following occurs:

- a plan mutates without child lineage;
- requested and executed methods differ without a fallback record;
- a route cell is missing or nonvalidated;
- planner-owned values are reported as evaluator-owned quality or preservation metrics;
- NaN or infinity enters ranking, consequence, or UI metric paths;
- a failed, unknown-status, or hard-ineligible plan is ranked;
- a selected stop is omitted from the displayed sequence;
- a permission-gated interpretation loses its provenance;
- an explanation claim lacks artifact evidence;
- a capped or partial exact search is labeled complete or optimal.
- a product UI mutates or overwrites a canonical artifact, changes a legacy
  parity oracle, recomputes evaluator/optimizer truth, ranks an ineligible row,
  converts null to zero, exposes an unsupported raw path, or presents a
  hypothetical probe as executable.

## Parallel Copilot Track

The corrective local-product path is `AUD-0 -> W0 -> W1 -> W1M -> W2 -> W3 -> W4 ->
W4R -> G4R -> G4 -> W5 -> W6 -> W7 -> W8`, with evidence gates defined in
`itinerary_repair_copilot_implementation_plan.md`. AUD-0 and W0/G0 are
`verified`: three broad and nine specialist audits, detailed traceability,
final evidence metadata, and independent Content/Web/Phase-Gate sign-offs pass.
W1 is `implemented`; Docker Linux/WSL2 readiness is verified. MAP-DEC-002 makes local MapLibre GL JS plus PMTiles the primary runtime and Atlas a deferred explicit backup. W1M/G1 are `verified`: the closed local package, provenance, coverage, live range/CORS/security behavior, no-egress replay, product recovery, browser shell, 133 focused tests, 444 full tests, and six independent audits pass. Corrected-v2 W2/G2 and W3/G3 are `verified`: the immutable three-plan package, continuous 16/16 road-validated routes, recalculated metrics/certificates, persistent preview, desktop/mobile journeys, security/state, accessibility/content, and phase/status truth passed independent revalidation. W4 is `implemented` offline. W4R is `in-progress` because the approved truth/contract correction has begun; its unresolved scope is direct map interaction, required-leg display or explicit gaps, exact Compare selection/evidence, and clear W5-gated decision controls. G4R is `planned` and not verified. G4 remains `blocked` on a newly authorized post-fix live smoke, fixed-24 evaluation, and low-versus-medium evidence; W5-W8 remain `planned`. Later waves may start only after their predecessor gate passes and any named external dependency is available.

The product sequence above is independent of the research critical path
`D1 -> E3.1 -> E3.3 -> E4`. No W4R or G4R result changes a research status.

The corrective evidence and immutable v2 identities are recorded in
[`../audits/w2_route_continuity_correction_report.md`](../audits/w2_route_continuity_correction_report.md)
and the final gate decision in
[`../audits/w2_v2_g2_g3_revalidation_report.md`](../audits/w2_v2_g2_g3_revalidation_report.md).

CP-010 is `in-progress`, not verified. Product implementation or verification
cannot advance E3.1, E3.3, E4, E3.UX5, or E5. A phase advances only when its
specified behavior evidence and independent audit pass; code presence or HTTP
200 is insufficient.

| Product gate | Status | Next evidence |
|---|---|---|
| AUD-0 | `verified` | Three broad audits, nine attributed specialist reports, and the evidence record pass |
| W0 / G0 | `verified` | Truth/status authority, detailed traceability, and independent Content/Web/Phase-Gate sign-offs pass; no research-status change |
| W1 / W1M / G1 | W1 `implemented`; W1M/G1 `verified` | Preserve the closed package, live evidence, and six independent PASS verdicts |
| W2 / G2 | `verified` | Preserve immutable v2 continuity, recalculated evidence, and independent audit record |
| W3 / G3 | `verified` | Preserve restart-safe preview, exact evaluated geography/evidence, and W5 fail-closed boundary |
| W4 / G4 | W4 `implemented`; G4 `blocked` | Preserve provider/prompt/transcript implementation evidence without treating it as a G4 pass; after G4R, obtain a new explicit authorization and run the bounded post-fix live smoke, fixed-24 evaluation, and low-versus-medium comparison; keep W5 closed |
| W4R / G4R | W4R `in-progress`; G4R `planned` and not verified | Implement typed direct interaction, required-leg coverage/gaps, and exact Compare behavior; then obtain independent map, route, accessibility, UI/UX, content, visual, security, status, and black-box sign-offs |
| W5-W8 / G5-G8 | `planned` | Decision, mobile/PWA, launch, and replacement evidence |

The first publishable slice is complete only when E1–E4 pass and every reported number is reconstructable from immutable artifacts.
