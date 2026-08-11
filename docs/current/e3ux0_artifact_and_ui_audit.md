# E3.UX0 Artifact and UI Audit

**Status:** `verified`  
**Evidence date:** 2026-07-29  
**Product source run:** `benchmark_158cf6d48be8`  
**Product snapshot:** `runs/e3ux-weather-repair-demo-v6`

## Decision

The repository already has a static, package-owned dashboard architecture, so
E3.UX uses an additive `runs/<run_id>/dashboard_product/` artifact. It does not
replace the frozen Folium renderer, modular research dashboard,
`evaluation.html`, or any E3 benchmark artifact.

The selected source run is the independently eligible progressive repair for
the synthetic weather-deterioration scenario. The product exporter copies
declared canonical artifacts into a new non-overwritable run snapshot, records
the source run ID and source-manifest SHA-256, and then renders only from the
copied run-relative files.

## Canonical artifact matrix

| Product question | Canonical source | Truth owner | Product behavior |
|---|---|---|---|
| What was accepted? | parent `PlanArtifactV2` | plan repository / plan artifact | Show original timeline and route; never mutate it. |
| What happened? | repair request and evidence records | request artifact | Label the scenario synthetic because `observed=false`. |
| What repair was produced? | child `PlanArtifactV2` and planner runs | repair pipeline | Show child only when lineage and content hashes validate. |
| What changed? | `PlanDiff` | diff artifact | Render added/deleted/road/ownership changes and unchanged days. |
| Is it eligible? | `EvaluationCertificate` | independent evaluator | Render certificate status; never infer eligibility from a planner score. |
| Why? | explanation evidence | evidence-linked explanation artifact | Render supported claims and evidence references only. |
| Is routing defensible? | `RouteMatrix` and certificate route validation | routing artifact plus evaluator | Keep missing/fallback/road-validated states distinct. |
| How did other methods behave? | benchmark metrics plus copied planner-run diagnostics | benchmark adapter / planner run | Keep failures visible; label capped exact searches incomplete; rank only eligible rows. |
| Can the user execute a change? | explicit interaction boundary | E5, currently deferred | No. E3.UX0–E3.UX4 are read-only and expose no UX5 action. |

## Audited source lineage

- Parent: `plan_e1c4f803691e3188`, content hash `20b540fdc5ed5cc9`.
- Child: `plan_7a706ef44466f240`, content hash `a7e7a0d888278daf`.
- Diff: `diff_1923c674931ec053`.
- Certificate: `cert_e68c2d7c0d169a3e`.
- Explanation: `repair_explanation_b54c99988319083e`.
- Route matrix: `route_matrix_68ab535465b06808`.
- Route evidence bundle: `route_bundle_a60c80047098a3b6`.
- Scenario: day 7 weather deterioration; Golden Gate Bridge is replaced by
  Bixby Creek Bridge Viewpoint; days 1–6 are unchanged.
- Exact baselines: context-blind and full reoptimization are recorded as
  `complete_candidate_limit_exceeded:50000`, so they are incomplete, not
  infeasible and not ranking-eligible.

## Existing UI compatibility oracle

The legacy UIs remain authoritative E3.C evidence:

- Folium normalized HTML SHA-256:
  `a06583549a135688e62d663ff5c6197074e96f3a0bad57d5cb791f37273fc2bb`.
- Day-plan SHA-256:
  `6bb4a3a40d76a07ba62e02bf055fd40fece8853b09fa76be07cffe16b7f88e27`.
- Normalized route-debug SHA-256:
  `b723926ceb77887660a7730104e3e2ebc891ca668ab33f4b43fe9d8f9444ff7e`.
- Modular dashboard validation remains owned by
  `scripts/validate_dashboard_export.py`.

Rollback is deletion or non-selection of the additive product run. No legacy
asset needs to be regenerated or reverted.

## UI audit

| Surface | Current strength | Gap relative to product review | E3.UX disposition |
|---|---|---|---|
| Folium renderer | Strong map and route-family inspection; frozen parity | Layer-oriented; does not present the complete decision flow | Preserve unchanged. |
| Modular dashboard | Rich research data and playback | Optimizer/dataset oriented; not repair-review first | Preserve unchanged. |
| Evaluation page | Existing metric contract | Not a parent/child repair narrative | Preserve unchanged. |
| Product dashboard | New artifact-grounded path | None within E3.UX0–E3.UX4 acceptance scope | Version and validate independently. |

## Truth-state contract

The product data contains a stable catalog for all required states:

`eligible_repair`, `ineligible_repair`, `failed_method`,
`exact_search_incomplete`, `complete_infeasibility`,
`missing_route_evidence`, `fallback_route`, `stale_artifact`,
`missing_certificate`, `certificate_mismatch`, `no_child_plan`,
`no_material_change`, `unchanged_parent`, `null_metric`,
`unavailable_metric`, `permission_required`, `locked_change_blocked`,
`hypothetical_probe`, `interaction_mode_disabled`, `empty_data`, `loading`,
`malformed_artifact`, and `partial_run`.

Active states in the v6 snapshot are interaction disabled, eligible repair,
failed method, exact search incomplete, unavailable metric, and null metric.
The first four are source-manifest states; the latter two arise from explicit
missing comparison values and remain null rather than becoming zero.

## Stop-condition audit

No parent was mutated; parent/child/diff/certificate lineage validates; every
copied source file has a full SHA-256; no non-finite JSON reaches the UI; no
ineligible alternative receives a rank; exact-cap failures retain their
diagnostic; JavaScript does not recompute optimizer or evaluator truth; and UX5
controls are absent.
