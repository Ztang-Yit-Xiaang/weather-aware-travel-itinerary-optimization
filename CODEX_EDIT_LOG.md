# Code Edit Log

Entries record Codex-assisted work sessions, findings, validation, conclusions, autonomous next work, human reflection, and required human action.

## Integrated core literature review reorganization

- Status: completed
- Start local time: 2026-06-21 22:35:25 CDT
- End local time: 2026-06-21 22:47:21 CDT-0500
- Duration: 11m 39s

### Goal

- Replace the compact eight-paper cards with a detailed integrated learning guide organized around preference, constrained planning, language interaction, evaluation and adaptation, and explanation and control.

### What changed

- docs/core_paper_reading_cards.md: replaced the card layout with a 5,646-word integrated review covering five layers, eight verified paper chapters, synthesis, architecture horizons, gaps, claims, and prioritized upgrades.
- README.md: renamed and expanded the literature navigation description for the integrated guide.
- docs/literature_onboarding_guide.md: made the integrated guide the recommended second read.
- docs/literature_deep_read_study_report.md: updated the evidence-bank reader path to reference the integrated guide.
- docs/recent_papers_2023_2026_addendum.md: updated the quick-index reading path.
- `git status`: M README.md
- `git status`: M docs/core_paper_reading_cards.md
- `git status`: M docs/literature_deep_read_study_report.md
- `git status`: M docs/literature_onboarding_guide.md
- `git status`: M docs/recent_papers_2023_2026_addendum.md

### What was found

- TRIP-PAL explicitly uses 15-minute units, popularity-based utility, and randomly assigned 15-60 minute travel times in the reported experiment.
- TripTide is arXiv:2510.21329v1 and contains placeholder conference metadata; it remains labeled as a preprint.
- TravelEval names accuracy, compliance, temporality, spatiality, economy, and utility as its six evaluation dimensions.
- CityHood uses LightGBM, LIME, LLM-generated descriptions, and review volume as an interest proxy.
- No inconsistency requiring structural changes was found in the evidence matrix or related-work outline.
- Three reference PDFs were already marked modified in the worktree when the final repository-wide status was captured. This task read but did not edit or revert them.

### Validation

- Integrated guide word count: 5,646 words, within the requested 5,000-6,500 range.
- Structure check: five learning layers, three architecture horizons, eight paper chapters, and all 16 required fields in every paper chapter.
- Local evidence check: all eight PDF filenames resolve; targeted claims were verified with pypdf extraction.
- Markdown link check: no missing local links across README and the four updated literature documents.
- ASCII check: all updated literature documents are ASCII-only; README retains only its pre-existing tree characters.
- Unsupported-overclaim search: matches were cautionary statements that explicitly deny unsupported capabilities.
- git diff --check: passed.
- python scripts\\validate_dashboard_export.py: passed.

### Conclusion

- The core reading package now follows the preferred integrated teaching structure while keeping agent-assisted planning clearly future work and preserving conservative publication claims.

### Next steps

**Codex can proceed:**

- Convert the prioritized near-term upgrades into a scoped implementation and evaluation backlog.

**Human reflection:**

- The primary paper will be stronger if it focuses on solver-backed weather repair and inspectable route differences rather than expanding immediately into autonomous agents.

### Human action

- Review the three architecture horizons and confirm that the next research stage matches the intended course or publication scope.

## Context-aware publication plan revision

- Status: completed
- Start local time: 2026-06-23 20:41:01 CDT
- End local time: 2026-06-23 20:58:05 CDT-0500
- Duration: 16m 44s

### Goal

- Rewrite the research stabilization and publication plan around a context-aware, inspectable solver-backed contribution with a TRB-first evidence gate and separate technical appendix.

### What changed

- docs/research_stabilization_and_publication_plan.md: replaced the 1,758-line weather-centered roadmap with a 650-line context-aware decision plan, venue-specific claims, readiness gates, ethics, risks, and definitions of done.
- docs/research_stabilization_and_publication_appendix.md: created the 1,521-line normative companion containing the full problem register, provider and coverage policies, contracts, statuses, solver-update allow list, evaluator, experiments, backlog, tests, manifests, and migration map.
- `git status`: M README.md
- `git status`: M docs/core_paper_reading_cards.md
- `git status`: M docs/literature_deep_read_study_report.md
- `git status`: M docs/literature_onboarding_guide.md
- `git status`: M docs/recent_papers_2023_2026_addendum.md
- `git status`: M reference/1-s2.0-S0305054821002963-main.pdf
- `git status`: M reference/2305.11755v3.pdf
- `git status`: M reference/2507.18778v1.pdf
- `git status`: ?? CODEX_EDIT_LOG.md
- `git status`: ?? docs/research_stabilization_and_publication_appendix.md
- `git status`: ?? docs/research_stabilization_and_publication_plan.md
- `git status`: ?? reference/2305.12295v2.pdf
- `git status`: ?? reference/2510.09011v3.pdf

### What was found

- The repository already integrates Open-Meteo, NPS, OpenStreetMap/Overpass, Wikidata, and OSRM, but the remaining context families require new provider audits and adapters.
- The original plan was untracked and 1,758 lines; unrelated modified and untracked literature/reference files were present and were preserved without edits.
- The venue strategy requires distinct evidence: TRB for transportation results, IUI for validated intelligent interaction, and CHI for ethics-approved controlled-study outcomes.

### Validation

- Document structure: main plan 650 lines; appendix 1,521 lines; heading hierarchy scan passed.
- Markdown integrity: main and appendix code fences are balanced; local file links resolve.
- Scope and terminology search: obsolete nationwide-out-of-scope, incident-deferred, weather-primary, and IUI/CHI-primary statements are absent.
- Boundary coverage: every context family appears in the appendix; PlannerRun, infeasibility certification, blocking warnings, raw-embedding prohibition, and incident hard-exclusion prohibition are present.
- git diff --no-index --check: passed for both new documentation files.

### Conclusion

- The publication roadmap now reflects the professor framing, preserves a concise main narrative, supplies implementation-ready technical detail, and uses evidence-specific TRB, IUI, and CHI gates.

### Next steps

**Codex can proceed:**

- Translate Appendix H milestones M0 and M1 into repository issues or an implementation checklist.

**Human reflection:**

- Nationwide real-data integration across every context family is intentionally ambitious and may prevent the July 18 TRB gate from passing; the plan now treats that as a quality gate rather than a promise.

### Human action

- Review and approve the context taxonomy, paid-provider budget boundary, incident soft-objective policy, and July 18 TRB readiness criteria before implementation begins.

## Professor-recommended literature integration

- Status: completed
- Start local time: 2026-06-23 21:19:07 CDT
- End local time: 2026-06-23 21:27:46 CDT-0500
- Duration: 8m 18s

### Goal

- Inspect four professor-recommended PDFs, deduplicate TTG, and integrate the three unique papers into the literature learning and publication package.

### What changed

- docs/literature_deep_read_study_report.md: added complete entries 41-43 for LLMAP, Logic-LM, and TripScore; marked the arXiv TTG file as a duplicate; updated citation backbones and project directions.
- docs/core_paper_reading_cards.md: added a professor-recommended companion-reading section and updated the evidence-bank count to 43 unique papers.
- docs/literature_onboarding_guide.md: added the four-file companion reading pass with TTG deduplication.
- docs/recent_papers_2023_2026_addendum.md: added professor-recommended indexing and updated modern paper clusters.
- docs/related_work_outline.md: added LLMAP and TripScore as essential citations and Logic-LM as neuro-symbolic support.
- docs/project_literature_evidence_matrix.md: added LLMAP/Logic-LM support and a complete-plan evaluation row using TravelEval and TripScore.
- README.md: updated literature navigation to the 43-paper evidence bank and companion readings.

### What was found

- 2509.12273v1.pdf is LLMAP, an LLM-as-Parser plus MSGS multi-objective route-planning preprint evaluated on 1,000 prompts across 27 cities and 14 countries.
- 2305.12295v2.pdf is Logic-LM, a general neuro-symbolic reasoning preprint using solver-error-guided self-refinement; it is not travel-specific.
- 2510.09011v3.pdf is TripScore, a fine-grained travel evaluation preprint with 4,870 queries, 219 real-world requests, and 60.75% expert agreement.
- 2410.16456v1.pdf is the arXiv copy of TTG already represented by 2024.emnlp-demo.25.pdf and is not a separate paper.

### Validation

- Deep-report structure: 43 sequential unique entries, 43 Project Action Takeaways, and all required action fields present.
- Local corpus: 44 PDFs on disk representing 43 report entries because TTG has two copies; all primary PDF references resolve.
- Core guide: 6,198 words, retaining the requested approximate range after adding companion readings.
- Markdown links: no missing local links across the seven updated documentation files.
- ASCII: all updated literature documents are ASCII-only; README retains only pre-existing tree characters.
- Implementation status vocabulary: all evidence-matrix statuses are valid.
- Unsupported-overclaim search: matches are cautionary statements only.
- git diff --check: passed.
- python scripts\\validate_dashboard_export.py: passed.

### Conclusion

- The professor-recommended papers are now inspected, deduplicated, and connected to the project framing, evidence matrix, reading order, and publication citation backbone.

### Next steps

**Codex can proceed:**

- Add the selected references to a formal bibliography or draft related-work prose when the paper format is chosen.

**Human reflection:**

- LLMAP and TripScore materially strengthen the future AI/solver and evaluation paths; Logic-LM is best kept as architectural support rather than a central travel citation.

### Human action

- Confirm whether the professor expects all three unique papers in the final submitted bibliography or only the most relevant citations for the chosen contribution.

## Context-aware research roadmap literature revision

- Status: completed
- Start local time: 2026-06-24 11:23:24 CDT
- End local time: 2026-06-24 11:40:55 Central Daylight Time-0500
- Duration: 17m 09s

### Goal

- Revise the stabilization plan and technical appendix around an inspectable neuro-symbolic architecture, realistic first publication slice, independent evaluation, and evidence-driven venue strategy.

### What changed

- docs/research_stabilization_and_publication_plan.md: replaced the broad roadmap with a repository-grounded decision plan, RQ1-RQ5 evidence map, conditional solver guarantee, reduced scope, venue gates, claim matrix, and seven-day order.
- docs/research_stabilization_and_publication_appendix.md: replaced the appendix with canonical parser/planner/evaluator/explanation contracts, status machines, compiler boundary, branch-and-check stages, evaluator gating, experiments, tests, backlog, and literature traceability.

### What was found

- The current prototype supports structured request overrides, interest scoring, weather-sensitive route logic, hotels/base cities, alternatives, Gurobi and heuristic paths, dashboard exports, and partial provenance, but these capabilities are not backed by a canonical independent complete-plan evaluator.
- experiment_runner.py can insert or replace required anchors after local route generation, so an inherited solver certificate is not publication-safe unless the changed route receives new run lineage and independent evaluation.
- Natural-language parsing, semantic confirmation, deterministic parser-to-planner compilation, minimal-change repair, branch-and-check, complete contrastive explanations, calibrated-reliance studies, and live disruption response are proposed rather than implemented.

### Validation

- Read-only document validation: passed for local links, heading hierarchy, balanced fences, table widths, ASCII, 20 referenced contracts, eight phases, five research questions, direct-control boundary, evaluator hard gating, and venue-date reverification markers.
- python scripts\validate_dashboard_export.py: passed.

### Conclusion

- The two roadmap documents now integrate the cited literature into architecture, scope, research questions, experiments, risks, claims, and venue decisions without presenting proposed capabilities as implemented.

### Next steps

**Codex can proceed:**

- Translate Phase 0 into implementation issues and begin immutable PlannerRun/PlanArtifact lineage plus the independent evaluator skeleton.

**Human reflection:**

- Choose the first venue from completed evidence: a narrow transportation-repair result favors TRB, while interpretation and correction claims require an IUI or CHI study.

### Human action

- Approve the first corridor and optional second corridor, decide whether RQ1 belongs in the first paper, reverify venue deadlines, and confirm provider budget, ethics timing, and participant access.

## Prof Choi literature review slide update

- Status: completed
- Start local time: 2026-06-25 10:36:19 -05:00
- End local time: 2026-06-25 10:45:47 CDT-0500
- Duration: 9m 10s

### Goal

- Update the attached literature review Beamer deck with the professor-recommended references and replace the final Prof. Choi question slide with broader possible discussion questions while keeping changes small.

### What changed

- docs/literature_review_slides.tex: added the pasted deck as the maintained TeX source, folded LOGIC-LM, LLMAP, TripScore, and TTG into the existing LLM/evaluation framing, tightened validation-boundary wording, and replaced the final slide with possible discussion questions.
- docs/literature_review_slides.pdf: regenerated the compiled 20-slide Beamer PDF from the updated source.

### What was found

- The pasted deck was not already present as a tracked TeX source; report/* is gitignored, so the maintained source was placed under docs/.
- The four professor-recommended PDFs identify LOGIC-LM, LLMAP, TripScore, and TTG; these fit naturally in the LLM-to-symbolic, LLM-as-parser, solver-backed planning, and fine-grained evaluation slides.
- TinyTeX is installed but missing beamer.cls; bundled Tectonic compiled the deck successfully after escalation for its external TeX/package cache.

### Validation

- Extracted first-page text from the four local PDFs to verify titles and roles.
- Checked docs/literature_review_slides.tex: ASCII-only, 20 Beamer frames, old 'Questions for Prof. Choi' title removed, added LOGIC-LM/LLMAP/TripScore/TTG references.
- Compiled with bundled Tectonic: success; docs/literature_review_slides.pdf produced with 20 pages.
- Rendered representative slides 1, 8, 12, 18, and 20 with PyMuPDF and visually inspected them for readability.

### Conclusion

- The literature review deck is updated with the necessary recent reference/framing additions, the final slide now contains possible discussion questions, and the compiled PDF is available.

### Next steps

**Codex can proceed:**

- If requested, convert this deck into a shorter 10-minute meeting version or add speaker notes for each slide.

**Human reflection:**

- The final deck keeps agent-assisted planning as future work and centers the defensible contribution on inspectable solver-backed adaptation and independent evaluation.

### Human action

- Review the final discussion questions and decide which venue path, TRB or IUI/CHI, you want the meeting to emphasize.

## Research foundation and Phase 0 seed

- Status: completed
- Start local time: 2026-06-29T21:37:12.3227510-05:00
- End local time: 2026-06-29 21:58:46 CDT-0500
- Duration: 20m 46s

### Goal

- Discern the current research problem from the roadmap documents, fix the project foundation first, and seed Phase 0 with truthful data, routing, and evaluation contracts.

### What changed

- configs/default_trip_config.yaml: added data snapshot and run identity defaults.
- data/README.md, data/registry/sources.yaml, data/snapshots/california_v1/*: added clean-clone California fallback catalog, context seed tables, source audit, and manifest hashes.
- src/itinerary_system/data/*: added dataset bundle schemas, snapshot loading, manifest hash exposure, and Phase 0 validation gates.
- src/itinerary_system/routing/*: added route-leg and route-result contracts with explicit road validation and fallback eligibility fields.
- src/itinerary_system/research_artifacts.py: added PlannerRun, PlanArtifact, ResearchEvaluationReport, stable hashes, and Phase 0 evaluation gating.
- src/itinerary_system/config.py and src/itinerary_system/artifact_metadata.py: surfaced catalog/context/run identifiers in resolved config and artifact freshness metadata.
- src/itinerary_system/data_enrichment.py and src/itinerary_system/utility_model.py: separated source_coverage_score from model/data uncertainty while keeping data_confidence compatibility.
- src/itinerary_system/__init__.py: exported the new data, routing, and research artifact contracts.
- docs/data_dictionary.md and docs/research_question_and_phase0_execution.md: documented the current problem, data contracts, and Phase 0 execution direction.
- tests/test_research_foundation.py: added focused coverage for snapshot loading, hashes, metadata identity, route eligibility, evaluator gating, and coverage/uncertainty separation.

### What was found

- The central project problem is evidence architecture rather than route generation alone: the repo needs stable catalog IDs, time-sensitive context IDs, run lineage, route provenance, and independent evaluation eligibility before publication claims.
- The fallback snapshot is suitable for clean-clone optimization demos but not final comparison metrics because current route_options are explicitly non-road-validated.
- The first focused test run exposed that the loader did not expose manifest hashes for non-table files such as README.md; snapshot loading now computes hashes for all manifest-listed files that exist.
- Unrelated pre-existing dirty files remain in the worktree, including the slide PDF/reference changes and untracked LaTeX/reference artifacts; they were not reverted.

### Validation

- python -m pytest -p no:cacheprovider tests/test_research_foundation.py: 7 passed.
- python -m pytest -p no:cacheprovider tests/test_configurable_itinerary_system.py::ConfigurableItinerarySystemTests::test_artifact_metadata_mismatch_marks_dashboard_stale tests/test_configurable_itinerary_system.py::ConfigurableItinerarySystemTests::test_utility_model_adds_bayesian_ucb_without_sparse_penalty tests/test_repair_planner.py: 8 passed.

### Conclusion

- The project now has a defensible Phase 0 foundation for clean-clone data, artifact identity, route validation gating, and independent evaluation eligibility; the next work is wiring these contracts into the live planner/export pipeline.

### Next steps

**Codex can proceed:**

- Wire PlannerRun, PlanArtifact, RouteResult, and ResearchEvaluationReport records into experiment_runner.py outputs.
- Replace fallback route_options with OSRM or local road-graph validated route cache rows and mark final-comparison eligibility only after validation passes.
- Add a Phase 0 export/evaluation script that emits planner_runs, plan_artifacts, route_audit, and evaluation_reports tables for paper evidence.

**Human reflection:**

- Treat soft utility and preference scores as conditional on artifact grounding and hard feasibility, not as proof that a displayed route is valid.
- The first paper can be narrower and stronger if it claims itinerary repair under validated disruptions, while keeping conversational and live-provider claims as later phases.

### Human action

- Review docs/research_question_and_phase0_execution.md and docs/data_dictionary.md for whether the Phase 0 scope matches your intended publication direction.
- Rotate any external API credentials if they were exposed outside this repository or logs; no credential value was copied into the report.

## Phase 0 evidence export integration

- Status: completed
- Start local time: 2026-06-29T22:00:22.9502356-05:00
- End local time: 2026-06-29 22:10:54 CDT-0500
- Duration: 10m 7s

### Goal

- Dive into Phase 0 by wiring the new research artifact contracts into executable production outputs.

### What changed

- src/itinerary_system/phase0_exporter.py: added the Phase 0 exporter that writes dataset validation, planner runs, immutable plan artifacts, route audit rows, evaluation reports, and an evidence summary.
- src/itinerary_system/experiment_runner.py: integrated Phase 0 artifact generation into method comparison, dashboard preparation, and full configurable pipeline outputs; included Phase 0 files in production artifact metadata.
- src/itinerary_system/__init__.py: exported Phase 0 artifact helpers and filenames.
- tests/test_research_foundation.py: added coverage for Phase 0 file generation, metadata inclusion, fallback route gating, and dataset validation output.
- docs/research_question_and_phase0_execution.md: documented the production_phase0_* evidence files and the explicit road-validation gate.

### What was found

- The existing runner computes route distances from coordinate sequences, so Phase 0 must label those legs as geodesic fallback unless route rows explicitly carry road_validated=true evidence.
- The clean fallback snapshot remains useful for optimization demos but current generated route evidence is intentionally ineligible for publication comparison until road validation is present.
- Production metadata can now list Phase 0 files, so stale/missing evidence tables become visible through the existing artifact-freshness pathway.

### Validation

- python -m pytest -p no:cacheprovider tests/test_research_foundation.py: 9 passed.
- python -m pytest -p no:cacheprovider tests/test_configurable_itinerary_system.py::ConfigurableItinerarySystemTests::test_artifact_metadata_mismatch_marks_dashboard_stale tests/test_configurable_itinerary_system.py::ConfigurableItinerarySystemTests::test_utility_model_adds_bayesian_ucb_without_sparse_penalty tests/test_repair_planner.py: 8 passed.
- $env:PYTHONDONTWRITEBYTECODE=1; $env:PYTHONPATH=src; python -c phase0 import smoke: passed.
- python -m compileall src/itinerary_system/phase0_exporter.py src/itinerary_system/experiment_runner.py: not used as validation because local __pycache__ writes were denied; bytecode-disabled import smoke covered syntax/imports.

### Conclusion

- Phase 0 is now connected to executable outputs: each production comparison refresh can emit planner, plan, route, dataset, and evaluation evidence tables with final-comparison eligibility gated by road validation.

### Next steps

**Codex can proceed:**

- Add a road-route cache adapter that converts validated OSRM/local graph responses into road_validated route rows.
- Add a strict Phase 0 validator script that fails when publication-comparison outputs include ineligible routes.
- Thread parent_run_id and parent_plan_id through actual repair operations once post-solve edits are promoted to child runs.

**Human reflection:**

- The project can now show demos honestly while reserving publication metrics for road-validated and independently evaluated plans.
- The first paper should report conditional rewards only after artifact grounding and hard feasibility pass.

### Human action

- Review the new production_phase0_* artifact schema names before building downstream paper tables around them.
- Decide whether Phase 0 should require OSRM validation for all methods or allow a smaller manually verified road-cache seed for the first study.

## Phase 0 artifact validator

- Status: completed
- Start local time: 2026-06-29T22:11:23.1886080-05:00
- End local time: 2026-06-29 22:15:30 CDT-0500
- Duration: 3m 46s

### Goal

- Add a concrete validator for the Phase 0 evidence files so generated artifacts can be checked before paper use.

### What changed

- scripts/validate_phase0_artifacts.py: added a Phase 0 artifact validator with default consistency checks and optional --require-final-eligible strict mode.
- tests/test_research_foundation.py: added subprocess coverage proving the validator accepts consistently ineligible artifacts and rejects a false eligible claim over fallback route rows.
- docs/research_question_and_phase0_execution.md: documented the validator command and strict final-comparison mode.

### What was found

- The validator should not fail merely because the current demo route is ineligible; it should fail when an artifact claims eligibility while route audit rows show fallback or unvalidated legs.
- Strict final-comparison mode is useful for paper tables, but default mode is better for day-to-day Phase 0 artifact development because it verifies honesty rather than requiring finished road validation.

### Validation

- python -m pytest -p no:cacheprovider tests/test_research_foundation.py: 10 passed.
- python -m pytest -p no:cacheprovider tests/test_configurable_itinerary_system.py::ConfigurableItinerarySystemTests::test_artifact_metadata_mismatch_marks_dashboard_stale tests/test_configurable_itinerary_system.py::ConfigurableItinerarySystemTests::test_utility_model_adds_bayesian_ucb_without_sparse_penalty tests/test_repair_planner.py: 8 passed.
- No extra test rerun after the final docs-only command note; code behavior was unchanged.

### Conclusion

- Phase 0 now has an executable validator that checks evidence table presence, ID linkage, plan/evaluation references, and false eligibility claims over fallback route audits.

### Next steps

**Codex can proceed:**

- Generate validated OSRM/local route rows and rerun the validator with --require-final-eligible.
- Add the validator to the notebook or CI quality path once Phase 0 artifacts are generated consistently in the normal pipeline.

**Human reflection:**

- The validator creates a clean distinction between honest demo artifacts and publication-ready comparison artifacts.

### Human action

- Decide when to turn --require-final-eligible into a required gate for paper figures and tables.

## Road-route cache adapter for Phase 0

- Status: completed
- Start local time: 2026-06-29T22:16:43.1619046-05:00
- End local time: 2026-06-29 22:27:10 CDT-0500
- Duration: 9m 52s

### Goal

- Add a Phase 0 road-route cache adapter so strict final-comparison eligibility can be proven from explicit validated route evidence instead of geodesic fallbacks.

### What changed

- src/itinerary_system/routing/cache.py: added RoadRouteCache, route anchor normalization, CSV loading, and validated leg lookup from production_road_route_cache.csv.
- src/itinerary_system/routing/__init__.py and src/itinerary_system/__init__.py: exported the cache helper, filename constant, and loader.
- src/itinerary_system/phase0_exporter.py: made Phase 0 route construction prefer validated cache legs before falling back to geodesic proxy legs; added cache path/row counts to the evidence summary.
- configs/default_trip_config.yaml and src/itinerary_system/config.py: added routing.road_route_cache_path with production_road_route_cache.csv as the default output-adjacent cache.
- tests/test_research_foundation.py: added cache-backed strict eligibility coverage and pinned the new routing config key.
- docs/data_dictionary.md and docs/research_question_and_phase0_execution.md: documented production_road_route_cache.csv, expected columns, and how it makes strict Phase 0 validation pass.

### What was found

- The existing route-stop artifacts provide labels and coordinates but not road-derived geometry by default, so the adapter must require an explicit road_validated cache row rather than infer validity from coordinates.
- A complete cache covering SFO to Golden Gate Bridge to Ferry Building to the selected hotel makes the Phase 0 evaluator and strict validator pass; without that cache, the same plan remains honestly ineligible.
- The loader now checks both output-dir-relative and repo-relative cache paths so experiments can use generated caches or checked-in research fixtures.

### Validation

- python -m pytest -p no:cacheprovider tests/test_research_foundation.py: 11 passed.
- python -m pytest -p no:cacheprovider tests/test_configurable_itinerary_system.py::ConfigurableItinerarySystemTests::test_artifact_metadata_mismatch_marks_dashboard_stale tests/test_configurable_itinerary_system.py::ConfigurableItinerarySystemTests::test_utility_model_adds_bayesian_ucb_without_sparse_penalty tests/test_repair_planner.py: 8 passed before the final path-resolution hardening.
- $env:PYTHONDONTWRITEBYTECODE=1; $env:PYTHONPATH=src; python -c routing/phase0 import smoke: passed before the final path-resolution hardening.

### Conclusion

- Phase 0 can now distinguish three states in executable artifacts: missing cache means fallback/ineligible, complete validated cache means strict eligible, and false eligible claims over fallback rows are rejected by the validator.

### Next steps

**Codex can proceed:**

- Build the actual OSRM/local-road-graph cache generator that fills production_road_route_cache.csv for selected production routes.
- Add provenance fields such as validator_version, retrieved_at, and route-cache hash to the strict validator once real cache generation is in place.

**Human reflection:**

- The project is now ready to collect road-route evidence without weakening the honesty gate for current demo artifacts.

### Human action

- Choose whether the first Phase 0 route cache should come from live OSRM, a local OSRM extract, or a manually reviewed small cache for the paper route.

## OSRM cache builder for Phase 0 road evidence

- Status: completed
- Start local time: 2026-06-29T22:28:20.4818474-05:00
- End local time: 2026-06-29 22:45:09 CDT-0500
- Duration: 16m 20s

### Goal

- Create an offline-safe generator for production_road_route_cache.csv so Phase 0 can move from cache consumption to auditable route-evidence collection.

### What changed

- src/itinerary_system/routing/road_cache_builder.py: added route-leg request extraction, OSRM cache key matching, cached OSRM JSON conversion, validated cache CSV writing, and missing/invalid evidence audit writing.
- scripts/build_road_route_cache.py: added the command-line builder with --output-dir, --cache-dir, --route-stops, and --require-complete.
- src/itinerary_system/routing/__init__.py and src/itinerary_system/__init__.py: exported the cache builder, audit filename, and OSRM cache key helper.
- tests/test_research_foundation.py: added OSRM fixture cache generation, missing-cache audit tests, cached-OSRM strict Phase 0 tests, and CLI wrapper coverage.
- docs/research_question_and_phase0_execution.md and docs/data_dictionary.md: documented the builder command, cache audit file, and missing-evidence semantics.
- .gitignore and tests/test_configurable_itinerary_system.py: ignored generated results/outputs/*.jsonl Phase 0 plan artifacts and covered the pattern in the ignore test.

### What was found

- The existing OSRM cache convention is open_osrm_route_<sha1>.json with latlon_geometry and raw OSRM payload fields; the builder reuses that key convention instead of inventing a parallel cache.
- Running the builder on current results/outputs requested 21 legs and validated 0 because the project cache has no matching OSRM responses for the current production route stops.
- Default Phase 0 validation now passes on current outputs as honest ineligible evidence; strict validation correctly fails because 3 evaluation reports are not final-comparison eligible.

### Validation

- python -m pytest -p no:cacheprovider tests/test_research_foundation.py: 14 passed.
- python -m pytest -p no:cacheprovider tests/test_configurable_itinerary_system.py::ConfigurableItinerarySystemTests::test_artifact_metadata_mismatch_marks_dashboard_stale tests/test_configurable_itinerary_system.py::ConfigurableItinerarySystemTests::test_utility_model_adds_bayesian_ucb_without_sparse_penalty tests/test_repair_planner.py: 8 passed.
- python -m pytest -p no:cacheprovider tests/test_configurable_itinerary_system.py::ConfigurableItinerarySystemTests::test_generated_large_outputs_are_ignored: 1 passed.
- python scripts/build_road_route_cache.py --output-dir results/outputs: completed with 0/21 validated legs and wrote production_road_route_cache_audit.csv.
- python scripts/validate_phase0_artifacts.py --output-dir results/outputs: passed with final_comparison_eligible=false warning.
- python scripts/validate_phase0_artifacts.py --output-dir results/outputs --require-final-eligible: expected failure, 3 evaluation reports not final-comparison eligible and no eligible reports found.

### Conclusion

- Phase 0 now has an offline-safe route-evidence collection command: it can convert real cached OSRM responses into validated route rows, audit missing evidence, and keep current publication comparison gates closed until route coverage exists.

### Next steps

**Codex can proceed:**

- Add an optional live OSRM/local-OSRM fetch step that writes the expected open_osrm_route_<key>.json files before the cache builder runs.
- Add route-cache hash and audit coverage metrics into production_phase0_dataset_validation.json or the evidence summary.

**Human reflection:**

- The current project state is honest but not publication-comparison ready: it has 21 current route legs to validate before strict Phase 0 can pass.

### Human action

- Decide whether Codex should use live public OSRM, a local OSRM extract, or a manually reviewed small cache to fill the 21 current missing route legs.

## Opt-in OSRM fetch path for Phase 0 route cache

- Status: completed
- Start local time: 2026-06-30T08:31:42.3387840-05:00
- End local time: 2026-06-30 08:38:31 CDT-0500
- Duration: 6m 12s

### Goal

- Add an explicit network mode that can fill missing OSRM cache responses before building the Phase 0 road-route cache, while preserving offline audit-only defaults.

### What changed

- src/itinerary_system/routing/road_cache_builder.py: added fetch_osrm_payload, OSRM_PUBLIC_BASE_URL, --fetch-missing support hooks, cached response writing, fetch status audit fields, and injectable fetcher support for tests.
- src/itinerary_system/routing/__init__.py: exported the OSRM base URL and fetch helper.
- scripts/build_road_route_cache.py: added --fetch-missing, --osrm-base-url, and --timeout-seconds CLI options while keeping default behavior offline.
- tests/test_research_foundation.py: added deterministic fake-fetcher coverage proving missing OSRM cache entries can be fetched, written, audited, and converted into validated road-route rows without real network access.
- docs/research_question_and_phase0_execution.md and docs/data_dictionary.md: documented default offline mode, explicit network mode, and the recommendation to use a local or pinned OSRM endpoint for paper evidence.

### What was found

- The builder can now fill open_osrm_route_<key>.json files when explicitly requested, but default command behavior remains audit-only and does not touch the network.
- Current results/outputs still has 21 missing OSRM cache responses and 0 validated route legs when run offline, so strict Phase 0 publication validation remains correctly closed.

### Validation

- python -m pytest -p no:cacheprovider tests/test_research_foundation.py: 15 passed.
- python -m pytest -p no:cacheprovider tests/test_configurable_itinerary_system.py::ConfigurableItinerarySystemTests::test_artifact_metadata_mismatch_marks_dashboard_stale tests/test_configurable_itinerary_system.py::ConfigurableItinerarySystemTests::test_utility_model_adds_bayesian_ucb_without_sparse_penalty tests/test_configurable_itinerary_system.py::ConfigurableItinerarySystemTests::test_generated_large_outputs_are_ignored tests/test_repair_planner.py: 9 passed.
- python scripts/build_road_route_cache.py --help: passed and shows --fetch-missing, --osrm-base-url, and --timeout-seconds.
- $env:PYTHONDONTWRITEBYTECODE=1; $env:PYTHONPATH=src; python -c fetch import smoke: passed.
- python scripts/build_road_route_cache.py --output-dir results/outputs: completed offline with 0/21 validated legs and missing_osrm_cache audit rows.
- python scripts/validate_phase0_artifacts.py --output-dir results/outputs: passed with final_comparison_eligible=false warning.

### Conclusion

- Phase 0 now has both sides of route-evidence collection: an offline audit/conversion path and an explicit opt-in OSRM fetch path that can populate the cache before strict validation.

### Next steps

**Codex can proceed:**

- Run the builder with --fetch-missing against an approved OSRM endpoint, then rerun strict Phase 0 validation.
- Add route-cache coverage and hash summaries into the Phase 0 evidence summary for easier paper-table reporting.

**Human reflection:**

- For publication, live public OSRM should be treated as a data collection source; a local or pinned OSRM endpoint gives stronger reproducibility.

### Human action

- Approve whether to use public OSRM, provide a local OSRM endpoint, or keep the current audit-only state until route evidence is reviewed manually.

## Route-cache coverage metrics for Phase 0 evidence

- Status: completed
- Start local time: 2026-06-30T08:39:45.9237462-05:00
- End local time: 2026-06-30 08:45:38 CDT-0500
- Duration: 5m 18s

### Goal

- Expose route-cache coverage, hashes, and per-plan road-validation ratios in Phase 0 artifacts so route evidence readiness can be read directly from generated outputs.

### What changed

- src/itinerary_system/phase0_exporter.py: added route-cache coverage aggregation, cache/audit SHA-256 hashes, per-plan validated/fallback leg counts, and road-validation coverage ratios in dataset validation and evidence summary outputs.
- scripts/validate_phase0_artifacts.py: required the new coverage columns and added warnings/errors for incomplete road-route cache coverage, especially under --require-final-eligible.
- tests/test_research_foundation.py: pinned coverage metrics for fallback, validated cache, and strict validator cases.
- docs/data_dictionary.md and docs/research_question_and_phase0_execution.md: documented route_cache_coverage and evidence-summary coverage fields.

### What was found

- Regenerated current results/outputs now reports road_route_requested_leg_count=21, road_route_validated_leg_count=0, road_route_missing_leg_count=21, and road_route_validation_coverage=0.0.
- Per-method evidence summaries now show 0.0 route_road_validation_coverage for hierarchical_gurobi_pipeline, hierarchical_greedy_baseline, and hierarchical_bandit_gurobi_repair.

### Validation

- python -m pytest -p no:cacheprovider tests/test_research_foundation.py: 15 passed.
- python -m pytest -p no:cacheprovider tests/test_configurable_itinerary_system.py::ConfigurableItinerarySystemTests::test_artifact_metadata_mismatch_marks_dashboard_stale tests/test_configurable_itinerary_system.py::ConfigurableItinerarySystemTests::test_utility_model_adds_bayesian_ucb_without_sparse_penalty tests/test_configurable_itinerary_system.py::ConfigurableItinerarySystemTests::test_generated_large_outputs_are_ignored tests/test_repair_planner.py: 9 passed.
- python scripts/build_road_route_cache.py --output-dir results/outputs: completed offline with 0/21 validated legs.
- Regenerated production_phase0_* artifacts for results/outputs after rebuilding the cache audit.
- python scripts/validate_phase0_artifacts.py --output-dir results/outputs: passed with final_comparison_eligible=false and road-route coverage incomplete warnings.
- python scripts/validate_phase0_artifacts.py --output-dir results/outputs --require-final-eligible: expected failure for 0.000 cache coverage, 3 ineligible evaluations, and incomplete road-validation coverage.
- python scripts/build_road_route_cache.py --help: passed.

### Conclusion

- Phase 0 outputs now make route evidence readiness explicit: the current project state is validated as honest demo evidence, while strict publication comparison is blocked by 0/21 road-route coverage.

### Next steps

**Codex can proceed:**

- After an approved route source is chosen, run --fetch-missing or use a local OSRM cache, regenerate Phase 0 artifacts, and verify road_route_validation_coverage reaches 1.0.
- Add these coverage fields to paper-table extraction or notebook summaries so route-readiness is visible in reports.

**Human reflection:**

- The project now has a clean quantitative gate instead of a vague route-validation TODO.

### Human action

- Approve the route source policy for filling the 21 missing road legs: public OSRM, local OSRM, or manually reviewed cache.

## Phase 0 readiness summary tables

- Status: completed
- Start local time: 2026-06-30T08:46:53.5499222-05:00
- End local time: 2026-06-30 08:52:02 CDT-0500
- Duration: 4m 35s

### Goal

- Add a readable Phase 0 readiness summary command so route evidence coverage and method eligibility can be inspected without manually joining artifact files.

### What changed

- scripts/summarize_phase0_readiness.py: added Phase 0 readiness loading, method-readiness table construction, Markdown/JSON output, and optional summary file writing.
- tests/test_research_foundation.py: added blocked-vs-ready summary coverage using fallback artifacts and validated route-cache artifacts.
- docs/research_question_and_phase0_execution.md: documented the readiness summary command and results/quality outputs.

### What was found

- The current readiness summary reports catalog california_v1, context context_static_demo_2026_06, 0/21 road-route cache coverage, 0/3 eligible evaluations, and strict_comparison_ready=false.
- Method-level blocking reasons are now explicit: each current production method is blocked by route_not_road_validated, comparison_ineligible, and hard_feasibility_failed.

### Validation

- python -m pytest -p no:cacheprovider tests/test_research_foundation.py: 16 passed.
- python scripts/summarize_phase0_readiness.py --output-dir results/outputs --write-dir results/quality: passed and wrote phase0_readiness_summary.md plus phase0_method_readiness.csv.
- python scripts/summarize_phase0_readiness.py --output-dir results/outputs --format json: passed and reported strict_comparison_ready=false, road_route_validation_coverage=0.0, eligible_evaluation_count=0.
- python scripts/validate_phase0_artifacts.py --output-dir results/outputs: passed with final_comparison_eligible=false and road-route coverage warnings.
- python -m pytest -p no:cacheprovider tests/test_configurable_itinerary_system.py::ConfigurableItinerarySystemTests::test_artifact_metadata_mismatch_marks_dashboard_stale tests/test_configurable_itinerary_system.py::ConfigurableItinerarySystemTests::test_utility_model_adds_bayesian_ucb_without_sparse_penalty tests/test_configurable_itinerary_system.py::ConfigurableItinerarySystemTests::test_generated_large_outputs_are_ignored tests/test_repair_planner.py: 9 passed.

### Conclusion

- Phase 0 now has a human-readable readiness table: the project can quickly state which methods are blocked, why strict comparison is closed, and what route-cache coverage is missing.

### Next steps

**Codex can proceed:**

- Use the readiness summary in notebook/report extraction so Phase 0 status appears in the research narrative automatically.
- After route-cache coverage is filled, rerun the summary to produce the final strict-ready table.

**Human reflection:**

- The summary turns the current blocker into a compact table rather than an implicit validator failure.

### Human action

- Review whether the blocking reason labels are the terms you want to carry into paper-facing tables.

## Phase 0 evidence pipeline command

- Status: completed
- Start local time: 2026-06-30 08:54:50 America/Chicago
- End local time: 2026-06-30 08:59:42 Central Daylight Time-0500
- Duration: 4m 23s

### Goal

- Add one repeatable command that diagnoses whether the current project artifacts are internally honest, road-route validated, and ready for strict Phase 0 comparison.

### What changed

- scripts/run_phase0_evidence_pipeline.py: added an orchestration command for road-route cache build, Phase 0 artifact export, artifact validation, and readiness summary writing.
- tests/test_research_foundation.py: added subprocess coverage for the pipeline blocked state and strict-ready state with cached OSRM evidence.
- docs/research_question_and_phase0_execution.md: documented the one-command diagnosis, strict comparison gate, and explicit OSRM fetch boundary.
- `git status`: M .gitignore
- `git status`: M CODEX_EDIT_LOG.md
- `git status`: M configs/default_trip_config.yaml
- `git status`: M docs/literature_review_slides.pdf
- `git status`: M reference/2305.12295v2.pdf
- `git status`: M reference/2509.12273v1.pdf
- `git status`: M src/itinerary_system/__init__.py
- `git status`: M src/itinerary_system/artifact_metadata.py
- `git status`: M src/itinerary_system/config.py
- `git status`: M src/itinerary_system/data_enrichment.py
- `git status`: M src/itinerary_system/experiment_runner.py
- `git status`: M src/itinerary_system/utility_model.py
- `git status`: M tests/test_configurable_itinerary_system.py
- `git status`: ?? data/
- `git status`: ?? docs/data_dictionary.md
- `git status`: ?? docs/literature_review_slides.aux
- `git status`: ?? docs/literature_review_slides.fdb_latexmk
- `git status`: ?? docs/literature_review_slides.fls
- `git status`: ?? docs/literature_review_slides.log
- `git status`: ?? docs/literature_review_slides.tex
- `git status`: ?? docs/pdflatex8088.fls
- `git status`: ?? docs/research_question_and_phase0_execution.md
- `git status`: ?? reference/TripScore_2510.09011_summary.md
- `git status`: ?? scripts/build_road_route_cache.py
- `git status`: ?? scripts/run_phase0_evidence_pipeline.py
- `git status`: ?? scripts/summarize_phase0_readiness.py
- `git status`: ?? scripts/validate_phase0_artifacts.py
- `git status`: ?? src/itinerary_system/data/
- `git status`: ?? src/itinerary_system/phase0_exporter.py
- `git status`: ?? src/itinerary_system/research_artifacts.py
- `git status`: ?? src/itinerary_system/routing/
- `git status`: ?? tests/test_research_foundation.py

### What was found

- The current generated outputs are internally consistent in non-strict mode but still have 0/21 road-validated route legs and 0/3 final-comparison eligible evaluations.
- Strict mode fails by design when route evidence is missing, with explicit errors for incomplete road-route coverage and ineligible evaluations.

### Validation

- python -m pytest -p no:cacheprovider tests/test_research_foundation.py: passed, 18 tests.
- python -m pytest -p no:cacheprovider tests/test_configurable_itinerary_system.py::ConfigurableItinerarySystemTests::test_artifact_metadata_mismatch_marks_dashboard_stale tests/test_configurable_itinerary_system.py::ConfigurableItinerarySystemTests::test_utility_model_adds_bayesian_ucb_without_sparse_penalty tests/test_configurable_itinerary_system.py::ConfigurableItinerarySystemTests::test_generated_large_outputs_are_ignored tests/test_repair_planner.py: passed, 9 tests.
- python scripts/run_phase0_evidence_pipeline.py --output-dir results/outputs --quality-dir results/quality: passed non-strict, wrote readiness summary, reported 0/21 road-route cache coverage and strict_comparison_ready=false.
- python scripts/run_phase0_evidence_pipeline.py --output-dir results/outputs --quality-dir results/quality --require-final-eligible: expected failure, reported incomplete route coverage, 3 ineligible evaluations, no eligible Phase 0 evaluations, and incomplete route validation in the evidence summary.

### Conclusion

- The project now has a single Phase 0 diagnostic command. The current problem is confirmed as missing road-validated route evidence, not a missing optimization/demo pipeline.

### Next steps

**Codex can proceed:**

- Add or fetch OSRM route-response cache files for the 21 requested legs, then rerun the pipeline in strict mode.

**Human reflection:**

- Decide whether publication experiments should use public OSRM, a local pinned OSRM service, or another reproducible road-routing source.

### Human action

- Review the readiness summary in results/quality and choose the route-evidence source policy before treating any comparison metrics as publication-grade.

## Offline route evidence request manifest

- Status: completed
- Start local time: 2026-06-30 09:00:51 America/Chicago
- End local time: 2026-06-30 09:06:44 Central Daylight Time-0500
- Duration: 5m 16s

### Goal

- Advance Phase 0 by turning the missing road-route evidence blocker into an auditable offline request manifest after public OSRM fetch was not approved.

### What changed

- src/itinerary_system/routing/road_cache_builder.py: added production_road_route_requests.csv generation with route labels, coordinates, cache keys, cache paths, and OSRM URL shape for every requested leg.
- src/itinerary_system/routing/__init__.py: exported the route request manifest filename and OSRM URL helper.
- scripts/build_road_route_cache.py: prints the route request manifest path whenever the cache/audit builder runs.
- scripts/run_phase0_evidence_pipeline.py: prints the route request manifest path as part of the Phase 0 diagnostic pipeline.
- tests/test_research_foundation.py: verifies the manifest is written by the builder, fetch path, script, and pipeline blocked-state flow.
- docs/data_dictionary.md: documents production_road_route_requests.csv as the offline handoff manifest for reviewed route-evidence collection.
- docs/research_question_and_phase0_execution.md: documents how the request manifest should be used with an approved local or external OSRM endpoint.
- results/outputs/production_road_route_requests.csv: generated 21 current route-evidence requests for the active production route-stop artifacts.
- results/quality/phase0_readiness_summary.md and results/quality/phase0_method_readiness.csv: refreshed readiness outputs after the offline pipeline run.
- `git status`: M .gitignore
- `git status`: M CODEX_EDIT_LOG.md
- `git status`: M configs/default_trip_config.yaml
- `git status`: M docs/literature_review_slides.pdf
- `git status`: M reference/2305.12295v2.pdf
- `git status`: M reference/2509.12273v1.pdf
- `git status`: M src/itinerary_system/__init__.py
- `git status`: M src/itinerary_system/artifact_metadata.py
- `git status`: M src/itinerary_system/config.py
- `git status`: M src/itinerary_system/data_enrichment.py
- `git status`: M src/itinerary_system/experiment_runner.py
- `git status`: M src/itinerary_system/utility_model.py
- `git status`: M tests/test_configurable_itinerary_system.py
- `git status`: ?? data/
- `git status`: ?? docs/data_dictionary.md
- `git status`: ?? docs/literature_review_slides.aux
- `git status`: ?? docs/literature_review_slides.fdb_latexmk
- `git status`: ?? docs/literature_review_slides.fls
- `git status`: ?? docs/literature_review_slides.log
- `git status`: ?? docs/literature_review_slides.tex
- `git status`: ?? docs/pdflatex8088.fls
- `git status`: ?? docs/research_question_and_phase0_execution.md
- `git status`: ?? reference/TripScore_2510.09011_summary.md
- `git status`: ?? scripts/build_road_route_cache.py
- `git status`: ?? scripts/run_phase0_evidence_pipeline.py
- `git status`: ?? scripts/summarize_phase0_readiness.py
- `git status`: ?? scripts/validate_phase0_artifacts.py
- `git status`: ?? src/itinerary_system/data/
- `git status`: ?? src/itinerary_system/phase0_exporter.py
- `git status`: ?? src/itinerary_system/research_artifacts.py
- `git status`: ?? src/itinerary_system/routing/
- `git status`: ?? tests/test_research_foundation.py

### What was found

- A public OSRM fetch attempt was rejected because route-source choice and sending route coordinates to an external service require explicit human approval.
- The current outputs now expose the 21 missing road legs as concrete cache requests instead of only reporting missing evidence.
- Strict Phase 0 remains correctly blocked: 0/21 validated road legs and 0/3 final-comparison eligible evaluations.

### Validation

- python -m pytest -p no:cacheprovider tests/test_research_foundation.py: passed, 18 tests.
- python scripts/run_phase0_evidence_pipeline.py --output-dir results/outputs --quality-dir results/quality: passed non-strict, wrote production_road_route_requests.csv, reported 0/21 route coverage and strict_comparison_ready=false.
- python -m pytest -p no:cacheprovider tests/test_configurable_itinerary_system.py::ConfigurableItinerarySystemTests::test_artifact_metadata_mismatch_marks_dashboard_stale tests/test_configurable_itinerary_system.py::ConfigurableItinerarySystemTests::test_utility_model_adds_bayesian_ucb_without_sparse_penalty tests/test_configurable_itinerary_system.py::ConfigurableItinerarySystemTests::test_generated_large_outputs_are_ignored tests/test_repair_planner.py: passed, 9 tests.
- python scripts/run_phase0_evidence_pipeline.py --output-dir results/outputs --quality-dir results/quality --require-final-eligible: expected failure, still rejects final comparison because route coverage is 0/21 and all 3 evaluations are ineligible.
- python scripts/validate_phase0_artifacts.py --output-dir results/outputs: passed non-strict with warnings for final_comparison_eligible=false and incomplete road-route coverage.

### Conclusion

- The project now has an offline-safe handoff from diagnosis to route evidence collection; Phase 0 cannot honestly proceed to strict comparison until the 21 request rows are resolved through an approved route source.

### Next steps

**Codex can proceed:**

- After the user approves a route source or provides a local OSRM endpoint, run the pipeline with --fetch-missing and --require-final-eligible, then inspect any invalid route responses.

**Human reflection:**

- For publication evidence, a local pinned OSRM service is preferable to public OSRM because it makes the route source reproducible and avoids sending route coordinates to a public endpoint.

### Human action

- Choose or approve the route-evidence source for the 21 rows in results/outputs/production_road_route_requests.csv before using strict Phase 0 comparison metrics.

## Legacy route cache compatibility audit

- Status: completed
- Start local time: 2026-06-30 09:08:02 America/Chicago
- End local time: 2026-06-30 09:12:56 Central Daylight Time-0500
- Duration: 4m 25s

### Goal

- Check whether existing legacy road-path cache evidence can safely reduce the 21-leg Phase 0 route-evidence blocker.

### What changed

- src/itinerary_system/routing/legacy_cache_audit.py: added offline audit logic comparing Phase 0 route requests with the legacy production_road_route_cache.json path cache.
- src/itinerary_system/routing/__init__.py: exported the legacy route-cache audit helper and output filename.
- scripts/audit_legacy_route_cache.py: added a CLI that writes production_legacy_route_cache_audit.csv and prints match/conversion counts.
- tests/test_research_foundation.py: added coverage that legacy geometry without duration provenance is not treated as strict-valid route evidence.
- docs/data_dictionary.md: documented the legacy route-cache audit as diagnostic-only evidence.
- docs/research_question_and_phase0_execution.md: documented how to run the legacy cache audit before route-source collection.
- results/outputs/production_legacy_route_cache_audit.csv: generated current legacy-cache compatibility audit for the 21 Phase 0 route requests.
- results/outputs/production_road_route_requests.csv and Phase 0/readiness outputs: refreshed by the offline pipeline run.
- `git status`: M .gitignore
- `git status`: M CODEX_EDIT_LOG.md
- `git status`: M configs/default_trip_config.yaml
- `git status`: M docs/literature_review_slides.pdf
- `git status`: M reference/2305.12295v2.pdf
- `git status`: M reference/2509.12273v1.pdf
- `git status`: M src/itinerary_system/__init__.py
- `git status`: M src/itinerary_system/artifact_metadata.py
- `git status`: M src/itinerary_system/config.py
- `git status`: M src/itinerary_system/data_enrichment.py
- `git status`: M src/itinerary_system/experiment_runner.py
- `git status`: M src/itinerary_system/utility_model.py
- `git status`: M tests/test_configurable_itinerary_system.py
- `git status`: ?? data/
- `git status`: ?? docs/data_dictionary.md
- `git status`: ?? docs/literature_review_slides.aux
- `git status`: ?? docs/literature_review_slides.fdb_latexmk
- `git status`: ?? docs/literature_review_slides.fls
- `git status`: ?? docs/literature_review_slides.log
- `git status`: ?? docs/literature_review_slides.tex
- `git status`: ?? docs/pdflatex8088.fls
- `git status`: ?? docs/research_question_and_phase0_execution.md
- `git status`: ?? reference/TripScore_2510.09011_summary.md
- `git status`: ?? scripts/audit_legacy_route_cache.py
- `git status`: ?? scripts/build_road_route_cache.py
- `git status`: ?? scripts/run_phase0_evidence_pipeline.py
- `git status`: ?? scripts/summarize_phase0_readiness.py
- `git status`: ?? scripts/validate_phase0_artifacts.py
- `git status`: ?? src/itinerary_system/data/
- `git status`: ?? src/itinerary_system/phase0_exporter.py
- `git status`: ?? src/itinerary_system/research_artifacts.py
- `git status`: ?? src/itinerary_system/routing/
- `git status`: ?? tests/test_research_foundation.py

### What was found

- The legacy cache contains 132 path entries but only 2 of the 21 current Phase 0 route requests match exactly.
- The two matching legacy rows provide path geometry but no duration field, so 0/21 rows are conversion-eligible for strict Phase 0 route evidence.
- Strict Phase 0 remains correctly blocked by 0/21 validated route legs and 0/3 eligible evaluations.

### Validation

- python -m pytest -p no:cacheprovider tests/test_research_foundation.py: passed, 19 tests.
- python scripts/audit_legacy_route_cache.py --output-dir results/outputs: passed, wrote production_legacy_route_cache_audit.csv, reported 2/21 legacy geometry matches and 0/21 conversion-eligible rows.
- python scripts/run_phase0_evidence_pipeline.py --output-dir results/outputs --quality-dir results/quality: passed non-strict, still reported 0/21 road-route coverage and strict_comparison_ready=false.
- python -m pytest -p no:cacheprovider tests/test_configurable_itinerary_system.py::ConfigurableItinerarySystemTests::test_artifact_metadata_mismatch_marks_dashboard_stale tests/test_configurable_itinerary_system.py::ConfigurableItinerarySystemTests::test_utility_model_adds_bayesian_ucb_without_sparse_penalty tests/test_configurable_itinerary_system.py::ConfigurableItinerarySystemTests::test_generated_large_outputs_are_ignored tests/test_repair_planner.py: passed, 9 tests.
- python scripts/run_phase0_evidence_pipeline.py --output-dir results/outputs --quality-dir results/quality --require-final-eligible: expected failure, strict gate still rejects missing route evidence.
- python scripts/validate_phase0_artifacts.py --output-dir results/outputs: passed non-strict with expected warnings.

### Conclusion

- Existing legacy route geometry cannot honestly unblock strict Phase 0; the project now proves that conclusion with an auditable CSV instead of relying on manual inspection.

### Next steps

**Codex can proceed:**

- Once a route source is approved, fetch or ingest OSRM responses for the 21 request rows and rerun strict Phase 0.

**Human reflection:**

- The legacy cache may still be useful for debugging route shapes, but using it for paper metrics would require complete duration provenance or a documented duration model that the validator treats separately.

### Human action

- Approve public OSRM, provide a local OSRM endpoint, or supply route-response JSON files matching results/outputs/production_road_route_requests.csv.

## Route fetch source-policy guard

- Status: completed
- Start local time: 2026-06-30 09:14:03 America/Chicago
- End local time: 2026-06-30 09:19:24 Central Daylight Time-0500
- Duration: 4m 50s

### Goal

- Prevent accidental public OSRM use while preparing strict Phase 0 route-evidence collection.

### What changed

- src/itinerary_system/routing/road_cache_builder.py: added local OSRM default, public OSRM detection, fetch-policy validation, and allow_public_osrm plumbing for route cache construction.
- src/itinerary_system/routing/__init__.py: exported local/public OSRM constants and fetch-policy helpers.
- scripts/build_road_route_cache.py: defaulted fetch mode to local OSRM and added --allow-public-osrm with a pre-network failure path.
- scripts/run_phase0_evidence_pipeline.py: defaulted fetch mode to local OSRM and added --allow-public-osrm for explicit public endpoint approval.
- tests/test_research_foundation.py: added coverage that public OSRM fetch is blocked without explicit approval and that request manifests default to localhost URLs.
- docs/data_dictionary.md: documented local OSRM as the default fetch endpoint and public OSRM as an explicit approval path.
- docs/research_question_and_phase0_execution.md: updated Phase 0 execution instructions with the local endpoint default and public OSRM guard.
- results/outputs/production_road_route_requests.csv and Phase 0/readiness outputs: refreshed offline so the 21 route request URLs now point at http://127.0.0.1:5000 by default.
- `git status`: M .gitignore
- `git status`: M CODEX_EDIT_LOG.md
- `git status`: M configs/default_trip_config.yaml
- `git status`: M docs/literature_review_slides.pdf
- `git status`: M reference/2305.12295v2.pdf
- `git status`: M reference/2509.12273v1.pdf
- `git status`: M src/itinerary_system/__init__.py
- `git status`: M src/itinerary_system/artifact_metadata.py
- `git status`: M src/itinerary_system/config.py
- `git status`: M src/itinerary_system/data_enrichment.py
- `git status`: M src/itinerary_system/experiment_runner.py
- `git status`: M src/itinerary_system/utility_model.py
- `git status`: M tests/test_configurable_itinerary_system.py
- `git status`: ?? data/
- `git status`: ?? docs/data_dictionary.md
- `git status`: ?? docs/literature_review_slides.aux
- `git status`: ?? docs/literature_review_slides.fdb_latexmk
- `git status`: ?? docs/literature_review_slides.fls
- `git status`: ?? docs/literature_review_slides.log
- `git status`: ?? docs/literature_review_slides.tex
- `git status`: ?? docs/pdflatex8088.fls
- `git status`: ?? docs/research_question_and_phase0_execution.md
- `git status`: ?? reference/TripScore_2510.09011_summary.md
- `git status`: ?? scripts/audit_legacy_route_cache.py
- `git status`: ?? scripts/build_road_route_cache.py
- `git status`: ?? scripts/run_phase0_evidence_pipeline.py
- `git status`: ?? scripts/summarize_phase0_readiness.py
- `git status`: ?? scripts/validate_phase0_artifacts.py
- `git status`: ?? src/itinerary_system/data/
- `git status`: ?? src/itinerary_system/phase0_exporter.py
- `git status`: ?? src/itinerary_system/research_artifacts.py
- `git status`: ?? src/itinerary_system/routing/
- `git status`: ?? tests/test_research_foundation.py

### What was found

- The fetch-capable commands previously defaulted URL generation to public OSRM, which made accidental external route-coordinate submission too easy.
- After the change, non-fetch diagnostics remain offline, fetch mode defaults to local OSRM, and public OSRM fails before network access unless explicitly allowed.
- Strict Phase 0 still correctly fails because no validated route responses are available yet.

### Validation

- python -m pytest -p no:cacheprovider tests/test_research_foundation.py: passed, 20 tests.
- python scripts/run_phase0_evidence_pipeline.py --output-dir results/outputs --quality-dir results/quality: passed non-strict, refreshed request manifest with localhost OSRM URLs, still reported 0/21 route coverage.
- python scripts/audit_legacy_route_cache.py --output-dir results/outputs: passed, reported 2/21 legacy geometry matches and 0/21 conversion-eligible rows.
- Sequential inspection of results/outputs/production_road_route_requests.csv: confirmed all osrm_route_url values start with http://127.0.0.1:5000.
- python -m pytest -p no:cacheprovider tests/test_configurable_itinerary_system.py::ConfigurableItinerarySystemTests::test_artifact_metadata_mismatch_marks_dashboard_stale tests/test_configurable_itinerary_system.py::ConfigurableItinerarySystemTests::test_utility_model_adds_bayesian_ucb_without_sparse_penalty tests/test_configurable_itinerary_system.py::ConfigurableItinerarySystemTests::test_generated_large_outputs_are_ignored tests/test_repair_planner.py: passed, 9 tests.
- python scripts/validate_phase0_artifacts.py --output-dir results/outputs: passed non-strict with expected warnings.
- python scripts/run_phase0_evidence_pipeline.py --output-dir results/outputs --quality-dir results/quality --require-final-eligible: expected failure, strict gate still rejects missing route evidence.
- python scripts/build_road_route_cache.py --output-dir results/outputs --fetch-missing --osrm-base-url https://router.project-osrm.org: failed before network with the public OSRM approval error, as intended.

### Conclusion

- Route evidence collection is now safer and more reproducible by default; the project is prepared to use a local/pinned OSRM endpoint without accidental public fetches.

### Next steps

**Codex can proceed:**

- When a local OSRM service is available, run the Phase 0 pipeline with --fetch-missing --require-final-eligible and inspect any invalid route responses.

**Human reflection:**

- This keeps the Phase 0 evidence pathway aligned with publication reproducibility rather than merely making the current validator pass.

### Human action

- Start or provide a local/pinned OSRM endpoint, or explicitly approve public OSRM with --allow-public-osrm if that tradeoff is acceptable.

## Route-source readiness precheck

- Status: completed
- Start local time: 2026-06-30 09:20:34 America/Chicago
- End local time: 2026-06-30 09:24:44 Central Daylight Time-0500
- Duration: 3m 31s

### Goal

- Add an offline/local route-source readiness check before strict Phase 0 fetch attempts.

### What changed

- scripts/check_route_source.py: added manifest/policy precheck and optional one-leg OSRM probe for Phase 0 route evidence collection.
- tests/test_research_foundation.py: added coverage for missing manifests, local precheck success, public OSRM policy rejection, and the CLI precheck.
- docs/research_question_and_phase0_execution.md: documented the route-source precheck and optional local endpoint probe before fetch-missing runs.
- docs/data_dictionary.md: documented check_route_source.py as the no-network route manifest and source-policy precheck.
- results/outputs/production_road_route_requests.csv, production_road_route_cache_audit.csv, Phase 0 outputs, and readiness files: refreshed through the offline evidence pipeline.
- results/outputs/production_legacy_route_cache_audit.csv: refreshed through the legacy route-cache audit.
- `git status`: M .gitignore
- `git status`: M CODEX_EDIT_LOG.md
- `git status`: M configs/default_trip_config.yaml
- `git status`: M docs/literature_review_slides.pdf
- `git status`: M reference/2305.12295v2.pdf
- `git status`: M reference/2509.12273v1.pdf
- `git status`: M src/itinerary_system/__init__.py
- `git status`: M src/itinerary_system/artifact_metadata.py
- `git status`: M src/itinerary_system/config.py
- `git status`: M src/itinerary_system/data_enrichment.py
- `git status`: M src/itinerary_system/experiment_runner.py
- `git status`: M src/itinerary_system/utility_model.py
- `git status`: M tests/test_configurable_itinerary_system.py
- `git status`: ?? data/
- `git status`: ?? docs/data_dictionary.md
- `git status`: ?? docs/literature_review_slides.aux
- `git status`: ?? docs/literature_review_slides.fdb_latexmk
- `git status`: ?? docs/literature_review_slides.fls
- `git status`: ?? docs/literature_review_slides.log
- `git status`: ?? docs/literature_review_slides.tex
- `git status`: ?? docs/pdflatex8088.fls
- `git status`: ?? docs/research_question_and_phase0_execution.md
- `git status`: ?? reference/TripScore_2510.09011_summary.md
- `git status`: ?? scripts/audit_legacy_route_cache.py
- `git status`: ?? scripts/build_road_route_cache.py
- `git status`: ?? scripts/check_route_source.py
- `git status`: ?? scripts/run_phase0_evidence_pipeline.py
- `git status`: ?? scripts/summarize_phase0_readiness.py
- `git status`: ?? scripts/validate_phase0_artifacts.py
- `git status`: ?? src/itinerary_system/data/
- `git status`: ?? src/itinerary_system/phase0_exporter.py
- `git status`: ?? src/itinerary_system/research_artifacts.py
- `git status`: ?? src/itinerary_system/routing/
- `git status`: ?? tests/test_research_foundation.py

### What was found

- The current 21-leg route request manifest passes the local endpoint policy precheck and still has 0 validated route legs.
- Public OSRM remains blocked without explicit approval at the route-source precheck layer.
- Strict Phase 0 remains correctly blocked until a local/pinned endpoint or approved response files provide validated geometry, distance, and duration.

### Validation

- python -m pytest -p no:cacheprovider tests/test_research_foundation.py: passed, 21 tests.
- python scripts/run_phase0_evidence_pipeline.py --output-dir results/outputs --quality-dir results/quality: passed non-strict, still reported 0/21 route coverage.
- python scripts/check_route_source.py --output-dir results/outputs: passed route-source readiness precheck for the 21-leg manifest and local OSRM endpoint policy.
- python scripts/check_route_source.py --output-dir results/outputs --osrm-base-url https://router.project-osrm.org: failed with public OSRM policy error, as intended.
- python scripts/audit_legacy_route_cache.py --output-dir results/outputs: passed, still reported 2/21 legacy geometry matches and 0/21 conversion-eligible rows.
- python -m pytest -p no:cacheprovider tests/test_configurable_itinerary_system.py::ConfigurableItinerarySystemTests::test_artifact_metadata_mismatch_marks_dashboard_stale tests/test_configurable_itinerary_system.py::ConfigurableItinerarySystemTests::test_utility_model_adds_bayesian_ucb_without_sparse_penalty tests/test_configurable_itinerary_system.py::ConfigurableItinerarySystemTests::test_generated_large_outputs_are_ignored tests/test_repair_planner.py: passed, 9 tests.
- python scripts/validate_phase0_artifacts.py --output-dir results/outputs: passed non-strict with expected warnings.
- python scripts/run_phase0_evidence_pipeline.py --output-dir results/outputs --quality-dir results/quality --require-final-eligible: expected failure, strict gate still rejects missing route evidence.

### Conclusion

- The project now has a clear preflight step between route-source selection and strict Phase 0 fetching, reducing the chance of accidental external calls or confusing endpoint failures.

### Next steps

**Codex can proceed:**

- When a local OSRM endpoint is running, execute check_route_source.py --probe, then run the Phase 0 pipeline with --fetch-missing --require-final-eligible.

**Human reflection:**

- The remaining blocker is external route-source availability, not project structure or validator ambiguity.

### Human action

- Start or provide the local/pinned OSRM endpoint, or supply matching OSRM response JSON files for the 21 request rows.

