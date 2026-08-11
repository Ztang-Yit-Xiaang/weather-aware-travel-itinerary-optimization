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

## Roadmap phase integration in repair specification

- Status: completed
- Start local time: 2026-06-30 16:29:48 CDT
- End local time: 2026-06-30 16:35:38 Central Daylight Time-0500
- Duration: about 6 minutes

### Goal

- Integrate the G0-G8 roadmap execution phases into the travel itinerary repair technical specification.

### What changed

- D:\UMN Courses\IE 5533\Project\Travel_Itinerary_Repair_Technical_Specification_for_Codex.md: inserted the Execution Phases and Handoff Gates section, renumbered later sections, and rewrote venue-oriented gates to reference Section 10.

### What was found

- The specification previously jumped from Benchmark specification directly to Work packages, with only shorter A/B/C venue gates near the end.
- The current roadmap blocker is strict road-valid evidence: local or pinned OSRM route responses are still required before transportation claims.

### Validation

- PowerShell content checks: Section 10 appears before Section 11, G0-G8 detail headings are present, the gate table columns are present, the OSRM blocker is named, venue gates point to Section 10, and old Work packages are renumbered.
- No Python test suite was run because this change only updates the external markdown planning/specification document.

### Conclusion

- The markdown specification now contains the concrete G0-G8 execution overlay and uses it as the authoritative order for work-package execution.

### Next steps

**Codex can proceed:**

- Implement G0/G2 route-evidence closure by adding or running local/pinned OSRM evidence generation and strict Phase 0 validation.
- Implement G1 canonical plan, ownership, repository, diff, and evaluator contracts after route evidence is unblocked.

**Human reflection:**

- The document now prioritizes route-valid evidence before repair-system claims, which may shift effort away from UI work until the routing source is settled.

### Human action

- Review or provide the intended local/pinned OSRM route source so the next implementation phase can move past the current evidence blocker.

## Roadmap phase integration in repair specification

- Status: completed
- Start local time: 2026-06-30 16:29:48 CDT
- End local time: 2026-06-30 16:35:38 Central Daylight Time-0500
- Duration: about 6 minutes

### Goal

- Integrate the G0-G8 roadmap execution phases into the travel itinerary repair technical specification.

### What changed

- D:\UMN Courses\IE 5533\Project\Travel_Itinerary_Repair_Technical_Specification_for_Codex.md: inserted the Execution Phases and Handoff Gates section, renumbered later sections, and rewrote venue-oriented gates to reference Section 10.

### What was found

- The specification previously jumped from Benchmark specification directly to Work packages, with only shorter A/B/C venue gates near the end.
- The current roadmap blocker is strict road-valid evidence: local or pinned OSRM route responses are still required before transportation claims.

### Validation

- PowerShell content checks: Section 10 appears before Section 11, G0-G8 detail headings are present, the gate table columns are present, the OSRM blocker is named, venue gates point to Section 10, and old Work packages are renumbered.
- No Python test suite was run because this change only updates the external markdown planning/specification document.

### Conclusion

- The markdown specification now contains the concrete G0-G8 execution overlay and uses it as the authoritative order for work-package execution.

### Next steps

**Codex can proceed:**

- Implement G0/G2 route-evidence closure by adding or running local/pinned OSRM evidence generation and strict Phase 0 validation.
- Implement G1 canonical plan, ownership, repository, diff, and evaluator contracts after route evidence is unblocked.

**Human reflection:**

- The document now prioritizes route-valid evidence before repair-system claims, which may shift effort away from UI work until the routing source is settled.

### Human action

- Review or provide the intended local/pinned OSRM route source so the next implementation phase can move past the current evidence blocker.

## Roadmap phase integration in repair specification

- Status: completed
- Start local time: 2026-06-30 16:29:48 CDT
- End local time: 2026-06-30 16:35:38 Central Daylight Time-0500
- Duration: about 6 minutes

### Goal

- Integrate the G0-G8 roadmap execution phases into the travel itinerary repair technical specification.

### What changed

- D:\UMN Courses\IE 5533\Project\Travel_Itinerary_Repair_Technical_Specification_for_Codex.md: inserted the Execution Phases and Handoff Gates section, renumbered later sections, and rewrote venue-oriented gates to reference Section 10.

### What was found

- The specification previously jumped from Benchmark specification directly to Work packages, with only shorter A/B/C venue gates near the end.
- The current roadmap blocker is strict road-valid evidence: local or pinned OSRM route responses are still required before transportation claims.

### Validation

- PowerShell content checks: Section 10 appears before Section 11, G0-G8 detail headings are present, the gate table columns are present, the OSRM blocker is named, venue gates point to Section 10, and old Work packages are renumbered.
- No Python test suite was run because this change only updates the external markdown planning/specification document.

### Conclusion

- The markdown specification now contains the concrete G0-G8 execution overlay and uses it as the authoritative order for work-package execution.

### Next steps

**Codex can proceed:**

- Implement G0/G2 route-evidence closure by adding or running local/pinned OSRM evidence generation and strict Phase 0 validation.
- Implement G1 canonical plan, ownership, repository, diff, and evaluator contracts after route evidence is unblocked.

**Human reflection:**

- The document now prioritizes route-valid evidence before repair-system claims, which may shift effort away from UI work until the routing source is settled.

### Human action

- Review or provide the intended local/pinned OSRM route source so the next implementation phase can move past the current evidence blocker.

## FOUND-001 repository truth implementation

- Status: completed
- Local date: 2026-06-30 CDT

### Goal

- Add explicit repository-state capture to new research artifacts so Phase 0 and production metadata can report the exact code identity used for a run.

### What changed

- `src/itinerary_system/repository_state.py`: added `RepositoryState`, `RepositoryStateUnavailable`, and `capture_repository_state()` with Git capture, environment overrides, strict/permissive behavior, package-version capture, and `GIT_OPTIONAL_LOCKS=0`.
- `src/itinerary_system/config.py`: added `run.repository_state_strict: false`.
- `src/itinerary_system/artifact_metadata.py`: embedded `repository_state` in `production_artifact_metadata.json` and made freshness checks compare commit, dirty flag, and package version while ignoring `captured_at`.
- `src/itinerary_system/phase0_exporter.py`: added `repository_state` to `production_phase0_dataset_validation.json`.
- `src/itinerary_system/__init__.py`: exported the repository-state public helpers.
- `tests/test_repository_state.py`: added unit tests for clean Git capture, dirty Git capture, env override, permissive unknown repo, strict failure, and unknown package-version fallback.
- `tests/test_research_foundation.py`: asserted repository-state metadata/evidence presence and moved one temp-output existence check inside its temporary-directory context.

### Validation

- `python -m ruff check src/itinerary_system/repository_state.py src/itinerary_system/artifact_metadata.py src/itinerary_system/config.py src/itinerary_system/phase0_exporter.py src/itinerary_system/__init__.py tests/test_repository_state.py tests/test_research_foundation.py`: passed.
- `python -m unittest discover -s tests -p 'test_repository_state.py'`: passed, 6 tests.
- `python -m unittest discover -s tests -p 'test_research_foundation.py'`: passed, 21 tests.
- `python -m ruff check src tests scripts`: failed on pre-existing unrelated lint issues in scripts, `nature_catalog.py`, `nature_site_routes.py`, routing return annotations, and `tests/test_configurable_itinerary_system.py`.
- `PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider tests/test_repository_state.py tests/test_research_foundation.py`: exited 139 before collection.
- `PYTHONDONTWRITEBYTECODE=1 python -X faulthandler -m pytest -q -p no:cacheprovider tests/test_repository_state.py tests/test_research_foundation.py`: confirmed a segmentation fault during pytest startup in `_pytest/capture.py`, before project tests collected.

### Conclusion

- FOUND-001 is implemented for the current artifact surfaces without starting DATA-001. Repository state is now present in new production metadata and Phase 0 evidence payloads, and targeted lint plus unittest coverage pass.

### Unresolved limitations

- Full-repo Ruff remains blocked by unrelated existing lint debt.
- Pytest remains blocked by an environment/native-library startup segmentation fault, so unittest was used as the successful test runner for the touched suites.
- Existing dirty files in `data/snapshots/california_v1/` and `notebook/production_system_blueprint.ipynb` were preserved and not reverted.

## DATA-001 catalog/context snapshot separation

- Status: completed
- Local date: 2026-07-01 CDT

### Goal

- Split stable catalog data from time-sensitive context data while preserving the existing `load_dataset_bundle()` compatibility surface for Phase 0 and notebook-oriented callers.

### What changed

- `src/itinerary_system/data/schemas.py`: added `CatalogBundle` and `ContextBundle`; changed `DatasetBundle` to compose both while preserving compatibility properties such as `catalog_snapshot_id`, `context_snapshot_id`, `snapshot_dir`, `manifest`, `tables`, `file_hashes`, and `table()`.
- `src/itinerary_system/data/context.py`: added context snapshot loading, typed `SnapshotLoadError`/`SnapshotTableMissing`, manifest reading, table loading, hashing, and legacy combined-snapshot fallback.
- `src/itinerary_system/data/snapshot.py`: refactored `load_dataset_bundle()` to load catalog and context separately, added `load_catalog_bundle()`, added manifest hash validation, and emits a legacy-context warning when context tables are loaded from an old combined snapshot.
- `src/itinerary_system/data/__init__.py`: exported the new data bundle types and loading helpers.
- `data/contexts/context_static_demo_2026_06/`: added a separated context manifest plus `weather_scenarios.csv` and `route_options.csv`.
- `data/snapshots/california_v1/manifest.json`: changed the snapshot manifest into a catalog manifest with `default_context_snapshot_id` and removed context tables/files from the catalog-owned file list.
- `src/itinerary_system/phase0_exporter.py`: added catalog/context snapshot directory and manifest details to `production_phase0_dataset_validation.json`.
- `tests/data/test_context_snapshot.py`: added coverage for clean separated loading, mismatched context IDs, missing context tables, invalid context hashes, and old combined-snapshot fallback.

### Validation

- `python -m ruff check src/itinerary_system/data src/itinerary_system/phase0_exporter.py tests/data/test_context_snapshot.py tests/test_research_foundation.py`: passed.
- `python -m ruff check src/itinerary_system/data src/itinerary_system/repository_state.py src/itinerary_system/artifact_metadata.py src/itinerary_system/config.py src/itinerary_system/phase0_exporter.py src/itinerary_system/__init__.py tests/data/test_context_snapshot.py tests/test_repository_state.py tests/test_research_foundation.py`: passed.
- `python -m unittest discover -s tests/data -p 'test_context_snapshot.py'`: passed, 5 tests.
- `python -m unittest discover -s tests -p 'test_repository_state.py'`: passed, 6 tests.
- `python -m unittest discover -s tests -p 'test_research_foundation.py'`: passed, 21 tests.
- `GIT_OPTIONAL_LOCKS=0 git -c filter.lfs.process= -c filter.lfs.clean= -c filter.lfs.smudge= -c filter.lfs.required=false diff --check -- src/itinerary_system/data src/itinerary_system/phase0_exporter.py data/snapshots/california_v1/manifest.json data/contexts tests/data/test_context_snapshot.py CODEX_EDIT_LOG.md`: passed.
- `python -m ruff check src tests scripts`: failed on pre-existing unrelated lint issues in scripts, `nature_catalog.py`, `nature_site_routes.py`, routing return annotations, and `tests/test_configurable_itinerary_system.py`.
- `PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider tests/test_repository_state.py tests/test_research_foundation.py tests/data/test_context_snapshot.py`: exited 139 before collection.
- `PYTHONDONTWRITEBYTECODE=1 python -X faulthandler -m pytest -q -p no:cacheprovider tests/test_repository_state.py tests/test_research_foundation.py tests/data/test_context_snapshot.py`: confirmed the same segmentation fault during pytest startup in `_pytest/capture.py`, before project tests collected.

### Conclusion

- DATA-001 is implemented with a separated catalog/context loading contract and backward-compatible legacy snapshot path. Phase 0 evidence generation still works against the compatibility API while now exporting catalog and context manifest details.

### Unresolved limitations

- The old context CSV copies still remain under `data/snapshots/california_v1/` for migration compatibility but are no longer listed as catalog-owned manifest files.
- Full-repo Ruff and pytest remain blocked for the same unrelated reasons recorded in FOUND-001.
- Existing dirty files in `data/snapshots/california_v1/feature_provenance.csv`, `route_options.csv`, `source_audit.csv`, and `notebook/production_system_blueprint.ipynb` were preserved and not reverted.
## Literature repair gap documentation

- Status: completed
- Start local time: 2026-07-04 09:30:30 -05:00
- End local time: 2026-07-04 09:48:34 CDT-0500
- Duration: 17m 36s

### Goal

- Create documentation-only literature review deliverables for ownership-aware minimal itinerary repair under travel disruptions.

### What changed

- docs/literature_review_repair_gap.md: added scoping synthesis, RQ-Lit answers, novelty framing, metrics, claims-to-avoid box, and verified references.
- docs/literature_search_log.md: recorded search surfaces, queries, screening counts, inclusion/exclusion criteria, and verification limitations.
- docs/literature_matrix_repair_gap.md: added 26-row comparison matrix across OP/TTDP, dynamic/RL routing, LLM travel, explainability, baseline, and proposed method.
- docs/current_score_audit.md: documented current utility, weather, nature, hotel, route, and provenance scoring formulas and limitations.
- docs/figures/repair_literature_gap_map.md: added Mermaid gap map for the repair contribution.
- docs/literature_review_repair_gap_citation_report.json: generated citation verification report for DOI-backed references.
- `git status`: ?? docs/current_score_audit.md
- `git status`: ?? docs/figures/repair_literature_gap_map.md
- `git status`: ?? docs/literature_matrix_repair_gap.md
- `git status`: ?? docs/literature_review_repair_gap.md
- `git status`: ?? docs/literature_review_repair_gap_citation_report.json
- `git status`: ?? docs/literature_search_log.md

### What was found

- The strongest defensible novelty is the combination of ownership labels, typed parent-child diffs, progressive neighborhoods, lexicographic preservation-before-utility objectives, independent certification, and evidence-grounded explanations.
- Recent iTIMO and TripTide work makes broad first itinerary-modification or first disruption-benchmark claims unsafe.
- The current repository score is a heuristic utility proxy; route travel uses geodesic fallback in key paths and hotel values rely on priors/fallbacks rather than live availability.

### Validation

- git diff --check: passed with no whitespace errors.
- ASCII check on new Markdown deliverables: passed after normalizing one author name in the search log.
- python C:\Users\1\.codex\skills\literature-evidence-synthesis\scripts\verify_citations.py docs\literature_review_repair_gap.md: 12 DOI records verified, 0 failed.
- Local deliverable checks: created all requested Markdown docs plus optional Mermaid map and citation report; matrix has 26 rows; review has about 2,810 words.

### Conclusion

- Documentation deliverables now support the shifted research framing and avoid overclaiming against recent LLM travel-modification literature.

### Next steps

**Codex can proceed:**

- Turn the repair framing into an implementation phase plan with ParentPlan/ChildPlan lineage, typed edits, progressive neighborhoods, certificates, and explanation evidence IDs.

**Human reflection:**

- The riskiest scholarly claim is any broad first claim around itinerary modification or disruption-aware travel planning; the safer claim is the specific ownership-aware repair architecture.

### Human action

- Review the novelty framing and decide whether to keep the generated citation JSON artifact in version control.
## Fix GitHub quality workflow

- Status: completed
- Start local time: 2026-07-05 20:19:54 CDT
- End local time: 2026-07-05 21:04:22 CDT-0500
- Duration: 43m 43s

### Goal

- Find why the GitHub quality workflow failed and fix the repository so formatting, lint, tests, coverage report, and dead-code report can pass.

### What changed

- scripts/audit_legacy_route_cache.py, scripts/build_road_route_cache.py, scripts/check_route_source.py, scripts/run_phase0_evidence_pipeline.py, scripts/validate_nature_route_pipeline.py: annotated intentional script import bootstraps with Ruff E402 noqa comments while preserving direct script execution.
- scripts/generate_paper_summaries.py: removed unused import and unused discussion assignment; Ruff formatted generated-summary helpers.
- scripts/render_literature_review_audit_pdf.py: made Markdown table zip use strict=False and Ruff formatted imports/spacing.
- scripts/summarize_phase0_readiness.py, scripts/validate_dashboard_export.py, scripts/validate_phase0_artifacts.py: applied Ruff formatting and safe lint autofixes.
- src/itinerary_system/artifact_metadata.py, data_enrichment.py, experiment_runner.py, map_exporter.py, multi_objective_route.py, phase0_exporter.py, region_scenarios.py, repair_planner.py, research_artifacts.py, utility_model.py: applied Ruff formatting and safe lint autofixes.
- src/itinerary_system/nature_catalog.py: removed unused visit-pattern work, replaced a late-bound lambda with a regex mask, and refactored profile route picking to pass segment state explicitly.
- src/itinerary_system/nature_site_routes.py: removed unused hashlib import and applied Ruff formatting.
- src/itinerary_system/routing/cache.py, src/itinerary_system/routing/legacy_cache_audit.py, src/itinerary_system/routing/models.py: applied Ruff formatting and safe annotation/import lint fixes.
- tests/test_configurable_itinerary_system.py, tests/test_research_foundation.py: applied Ruff formatting and import cleanup.
- `git status`: M scripts/audit_legacy_route_cache.py
- `git status`: M scripts/build_road_route_cache.py
- `git status`: M scripts/check_route_source.py
- `git status`: M scripts/generate_paper_summaries.py
- `git status`: M scripts/render_literature_review_audit_pdf.py
- `git status`: M scripts/run_phase0_evidence_pipeline.py
- `git status`: M scripts/summarize_phase0_readiness.py
- `git status`: M scripts/validate_dashboard_export.py
- `git status`: M scripts/validate_nature_route_pipeline.py
- `git status`: M scripts/validate_phase0_artifacts.py
- `git status`: M src/itinerary_system/artifact_metadata.py
- `git status`: M src/itinerary_system/data_enrichment.py
- `git status`: M src/itinerary_system/experiment_runner.py
- `git status`: M src/itinerary_system/map_exporter.py
- `git status`: M src/itinerary_system/multi_objective_route.py
- `git status`: M src/itinerary_system/nature_catalog.py
- `git status`: M src/itinerary_system/nature_site_routes.py
- `git status`: M src/itinerary_system/phase0_exporter.py
- `git status`: M src/itinerary_system/region_scenarios.py
- `git status`: M src/itinerary_system/repair_planner.py
- `git status`: M src/itinerary_system/research_artifacts.py
- `git status`: M src/itinerary_system/routing/cache.py
- `git status`: M src/itinerary_system/routing/legacy_cache_audit.py
- `git status`: M src/itinerary_system/routing/models.py
- `git status`: M src/itinerary_system/utility_model.py
- `git status`: M tests/test_configurable_itinerary_system.py
- `git status`: M tests/test_research_foundation.py

### What was found

- The attached GitHub Actions screenshot showed the quality job failing at Ruff format check because 22 Python files would be reformatted.
- After formatting, Ruff lint exposed additional issues: intentional script E402 imports, unused variables/imports, a missing zip strict parameter, quoted forward annotations, constant getattr, and a late-bound closure in nature_catalog.py.
- The local one-shot coverage run exceeded the initial 5-minute wrapper timeout because the full suite takes about 16 minutes in this environment; running the same tests by file group produced clean results.

### Validation

- ruff format --check .: passed, 51 files already formatted.
- ruff check .: passed, all checks passed.
- git diff --check: passed.
- coverage run --append -m pytest tests/data/test_context_snapshot.py tests/test_repair_planner.py tests/test_repository_state.py: passed, 17 tests.
- coverage run --append -m pytest tests/test_adaptive_bandit_planner.py: passed, 6 tests.
- coverage run --append -m pytest tests/test_research_foundation.py: passed, 21 tests on clean rerun.
- coverage run --append -m pytest tests/test_configurable_itinerary_system.py: passed, 37 tests.
- coverage report -m: passed, total coverage reported at 58%.
- python scripts/find_dead_code.py: passed and wrote the ignored Vulture report.

### Conclusion

- The GitHub quality workflow blocker was fixed. Formatting and lint gates now pass, and all 81 tests passed locally in coverage-backed batches.

### Next steps

**Codex can proceed:**

- Commit the changes, push the branch, and recheck GitHub Actions if the user wants repository publication handled next.

**Human reflection:**

- The fix touches many files because the primary CI blocker was repository-wide Ruff formatting drift; the behavioral edits were kept to lint-required cleanup.

### Human action

- Review the broad formatting diff, then push or rerun GitHub Actions to confirm the remote quality workflow.

## Fetch LFS data in quality workflow

- Status: completed
- Start local time: 2026-07-05 21:41:56 CDT
- End local time: 2026-07-05 21:43:16 CDT-0500
- Duration: 1m 05s

### Goal

- Fix the GitHub-only test failures caused by LFS-backed CSV snapshots not being present in Actions checkout.

### What changed

- .github/workflows/quality.yml: enabled lfs: true for actions/checkout so GitHub Actions receives real CSV snapshot files instead of Git LFS pointer files.
- `git status`: M .github/workflows/quality.yml

### What was found

- The pushed quality run passed Ruff format and lint, then failed tests with catalog/context manifest hash mismatches and missing CSV columns.
- .gitattributes stores *.csv through Git LFS, and actions/checkout defaults to not downloading LFS objects; CI was testing pointer files rather than actual snapshot CSVs.

### Validation

- python -m pytest tests/data/test_context_snapshot.py tests/test_research_foundation.py::ResearchFoundationTests::test_snapshot_manifest_hashes_match_files tests/test_research_foundation.py::ResearchFoundationTests::test_clean_clone_snapshot_loads_and_gates_final_comparison: passed, 7 tests.
- git lfs ls-files: confirmed snapshot CSVs are LFS-managed.

### Conclusion

- The quality workflow now fetches LFS data before installing and testing the package.

### Next steps

**Codex can proceed:**

- Commit and push the workflow fix, then watch the next GitHub Actions quality run.

**Human reflection:**

- The repository relies on LFS-managed CSV fixtures, so future workflows that read snapshot data should also opt into LFS checkout.

### Human action

- Review the GitHub Actions run after the pushed workflow fix completes.

## Preserve current repair-gap artifacts

- Status: completed
- Start local time: 2026-07-05 23:27:43 CDT
- End local time: 2026-07-05 23:31:39 CDT-0500
- Duration: 3m

### Goal

- Keep the compact repair-gap matrix and current score audit as standalone current-project artifacts while preserving the consolidated literature documents.

### What changed

- docs/literature/current/README.md: added a small index for obvious and latent current-project problems.
- docs/literature/current/literature_matrix_repair_gap.md: restored the compact repair-gap matrix in a new current folder and added a column guide plus full reference key.
- docs/literature/current/current_score_audit.md: restored the current project score audit as a standalone artifact.
- `git status`: M .codex/code-edit-log.md
- `git status`: M README.md
- `git status`: M docs/README.md
- `git status`: D docs/literature/chi_oriented_literature_review.md
- `git status`: M docs/literature/literature_deep_read_study_report.md
- `git status`: M docs/literature/literature_onboarding_guide.md
- `git status`: D docs/literature/literature_review_update_audit.md
- `git status`: D docs/literature/project_literature_evidence_matrix.md
- `git status`: D docs/literature/recent_papers_2023_2026_addendum.md
- `git status`: D docs/literature/related_work_outline.md
- `git status`: D docs/literature/repair_gap/current_score_audit.md
- `git status`: D docs/literature/repair_gap/literature_matrix_repair_gap.md
- `git status`: D docs/literature/repair_gap/literature_review_repair_gap.md
- `git status`: D docs/literature/repair_gap/literature_review_repair_gap_citation_report.json
- `git status`: D docs/literature/repair_gap/literature_search_log.md
- `git status`: D docs/literature/repair_gap/novelty_claim_verification.md
- `git status`: D docs/literature/repair_gap/repair_literature_gap_map.md
- `git status`: D docs/methods/limitation_driven_itinerary_repair_method.md
- `git status`: D docs/methods/research_question_and_phase0_execution.md
- `git status`: D docs/planning/publication_oriented_research_and_system_design_roadmap.md
- `git status`: D docs/planning/publication_oriented_research_and_system_design_roadmap.pdf
- `git status`: D docs/planning/research_stabilization_and_publication_appendix.md
- `git status`: M docs/planning/research_stabilization_and_publication_plan.md
- `git status`: M docs/planning/travel_itinerary_repair_technical_specification.md
- `git status`: D docs/planning/travel_itinerary_repair_technical_specification_extended.md
- `git status`: D docs/slides/build/literature_review_slides.aux
- `git status`: D docs/slides/build/literature_review_slides.fdb_latexmk
- `git status`: D docs/slides/build/literature_review_slides.fls
- `git status`: D docs/slides/build/literature_review_slides.log
- `git status`: D docs/slides/build/pdflatex8088.fls
- `git status`: ?? docs/literature/current/
- `git status`: ?? docs/literature/evidence_matrix.md
- `git status`: ?? docs/literature/repair_gap_review.md
- `git status`: ?? docs/methods/repair_method.md

### What was found

- The original repair-gap matrix and score-audit files are deleted in the current working tree as part of the broader document consolidation, but their last tracked contents are recoverable from HEAD.
- The compact matrix should remain inspectable because it makes the gap evidence easy to scan, while the consolidated review serves a different narrative purpose.

### Validation

- git diff --check -- docs/literature/current: passed.
- Manual file inspection: confirmed the new current folder contains README.md, literature_matrix_repair_gap.md, and current_score_audit.md.

### Conclusion

- The matrix and audit are preserved under docs/literature/current without restoring or disturbing the old deleted paths.

### Next steps

**Codex can proceed:**

- Link docs/literature/current from docs/literature/repair_gap_review.md or docs/README.md if the user wants these artifacts discoverable from the consolidated docs.

**Human reflection:**

- The matrix now has enough column/reference explanation for a reader while keeping the compact style intact.

### Human action

- Review whether docs/literature/current is the preferred folder name and whether the old deleted repair_gap paths should remain deleted.

## Reorganize current matrix and audit docs

- Status: completed
- Start local time: 2026-07-06 00:01:40 CDT
- End local time: 2026-07-06 00:03:30 CDT-0500
- Duration: 1m 28s

### Goal

- Move the repair-gap matrix into the literature directory and the current implementation audit into a docs/current folder.

### What changed

- docs/literature/literature_matrix_repair_gap.md: moved the compact repair-gap matrix out of docs/literature/current and into the literature directory.
- docs/current/current_score_audit.md: moved the current score audit into a current-project documentation folder.
- docs/current/README.md: added a short index explaining why implementation audits live under docs/current and the literature matrix lives under docs/literature.
- docs/literature/current/: removed the temporary folder created in the previous organization pass.
- `git status`: M .codex/code-edit-log.md
- `git status`: M CODEX_EDIT_LOG.md
- `git status`: M README.md
- `git status`: M docs/README.md
- `git status`: D docs/literature/chi_oriented_literature_review.md
- `git status`: M docs/literature/literature_deep_read_study_report.md
- `git status`: M docs/literature/literature_onboarding_guide.md
- `git status`: D docs/literature/literature_review_update_audit.md
- `git status`: D docs/literature/project_literature_evidence_matrix.md
- `git status`: D docs/literature/recent_papers_2023_2026_addendum.md
- `git status`: D docs/literature/related_work_outline.md
- `git status`: D docs/literature/repair_gap/current_score_audit.md
- `git status`: D docs/literature/repair_gap/literature_matrix_repair_gap.md
- `git status`: D docs/literature/repair_gap/literature_review_repair_gap.md
- `git status`: D docs/literature/repair_gap/literature_review_repair_gap_citation_report.json
- `git status`: D docs/literature/repair_gap/literature_search_log.md
- `git status`: D docs/literature/repair_gap/novelty_claim_verification.md
- `git status`: D docs/literature/repair_gap/repair_literature_gap_map.md
- `git status`: D docs/methods/limitation_driven_itinerary_repair_method.md
- `git status`: D docs/methods/research_question_and_phase0_execution.md
- `git status`: D docs/planning/publication_oriented_research_and_system_design_roadmap.md
- `git status`: D docs/planning/publication_oriented_research_and_system_design_roadmap.pdf
- `git status`: D docs/planning/research_stabilization_and_publication_appendix.md
- `git status`: M docs/planning/research_stabilization_and_publication_plan.md
- `git status`: M docs/planning/travel_itinerary_repair_technical_specification.md
- `git status`: D docs/planning/travel_itinerary_repair_technical_specification_extended.md
- `git status`: D docs/slides/build/literature_review_slides.aux
- `git status`: D docs/slides/build/literature_review_slides.fdb_latexmk
- `git status`: D docs/slides/build/literature_review_slides.fls
- `git status`: D docs/slides/build/literature_review_slides.log
- `git status`: D docs/slides/build/pdflatex8088.fls
- `git status`: ?? docs/current/
- `git status`: ?? docs/literature/evidence_matrix.md
- `git status`: ?? docs/literature/literature_matrix_repair_gap.md
- `git status`: ?? docs/literature/repair_gap_review.md
- `git status`: ?? docs/methods/repair_method.md

### What was found

- The user's preferred structure is matrix with literature artifacts and audit with current-project notes, rather than both under docs/literature/current.

### Validation

- git diff --check -- docs/current docs/literature/literature_matrix_repair_gap.md: passed.
- Manual inspection: confirmed docs/current contains README.md and current_score_audit.md, and docs/literature/literature_matrix_repair_gap.md starts with the preserved matrix and column guide.

### Conclusion

- The matrix and audit now live in the requested locations without touching the broader in-progress document consolidation.

### Next steps

**Codex can proceed:**

- Update docs/README.md or docs/literature/repair_gap_review.md with links to the moved files if desired.

**Human reflection:**

- This split better separates literature evidence from implementation claim-control notes.

### Human action

- Review whether docs/current is the final preferred folder name.

## Current problem phase plans

- Status: completed
- Start local time: 2026-07-06 00:11:03 CDT
- End local time: 2026-07-06 00:27:08 Central Daylight Time-0500
- Duration: 15m 19s

### Goal

- Generate Codex phase plans from the current score audit and current project docs to fix obvious and latent repository problems.

### What changed

- docs/planning/current_problem_fix_phase_plans.md: added eight implementation-ready phase plans covering validation harness, artifact lineage, route matrix, utility missingness, parent/diff foundation, progressive repair, evaluator/explanations, and pipeline/benchmark migration.

### What was found

- docs/current/current_score_audit.md identifies heuristic utility, geodesic routing, non-parent-aware repair, unreachable route-oracle code, and source-coverage claim risks.
- Source inspection confirmed Phase 0 scaffolds exist in research_artifacts.py, routing/models.py, and repair_planner.py, while optimizer paths still depend on geodesic travel and lack parent-aware edit variables.
- The worktree already had substantial unrelated documentation moves and edits; those were left untouched.

### Validation

- python -m ruff check src tests scripts: passed.
- python -m pytest: collected 81 tests and reached 77 passed / 4 failed before timeout status; failures were PermissionError cases in tests/data/test_context_snapshot.py while writing under C:\Users\1\AppData\Local\Temp, with pytest cache permission warnings.
- Focused rerun of tests/data/test_context_snapshot.py with TEMP/TMP/TMPDIR pointed at .codex_tmp_pytest: still failed 4 tests because Python tempfile resolved to C:\Users\1\AppData\Local\Temp in this managed shell.
- Manual inspection: confirmed docs/planning/current_problem_fix_phase_plans.md has eight Phase X.X implementation plans with required sections and planning-only roadmap notes.

### Conclusion

- Created a current-problem phase plan document grounded in current docs, source files, and live validation results; planning goal is satisfied.

### Next steps

**Codex can proceed:**

- Implement Phase 0.0 to stabilize pytest temp/cache behavior and produce a current problem manifest.
- Then implement Phase 0.1 artifact lineage and Phase 0.2 route-matrix boundaries before claiming publication-grade repair evidence.

**Human reflection:**

- The strongest near-term path is to fix evidence truthfulness and validation reproducibility before expanding solver functionality.

### Human action

- Review the phase ordering and decide whether Phase 0.0 validation stabilization should be implemented first.

## Context-aware itinerary repair detailed phase plan

- Status: completed
- Start local time: 2026-07-06 00:36:04 CDT
- End local time: 2026-07-06 00:53:16 CDT-0500
- Duration: Not recorded

### Goal

- Create a repository-grounded, implementation-ready G0-G8 phase plan for Context-Aware, Inspectable Itinerary Repair without implementing code.

### What changed

- docs/planning/context_aware_itinerary_repair_detailed_phase_plan.md: added the detailed planning-only master plan with G0-G8 gate sections, diagrams, data models, method signatures, validation rules, tests, acceptance criteria, literature guardrails, and definition of done.
- CODEX_EDIT_LOG.md: appended this completed planning report.

### What was found

- Files inspected: README.md; docs/README.md; docs/current/current_score_audit.md; docs/current/README.md; docs/reference/data_dictionary.md; docs/reference/code_quality_workflow.md; docs/planning/research_stabilization_and_publication_plan.md; docs/planning/travel_itinerary_repair_technical_specification.md; docs/planning/current_problem_fix_phase_plans.md; docs/literature/core_paper_reading_cards.md; docs/literature/evidence_matrix.md; docs/literature/literature_deep_read_study_report.md; docs/literature/literature_matrix_repair_gap.md; docs/literature/literature_onboarding_guide.md; docs/literature/repair_gap_review.md; relevant src, scripts, and tests for artifacts, routing, repair scaffold, utilities, Phase 0 scripts, and current validation contracts.
- Repository evidence shows Phase 0 artifacts, route cache/source checks, catalog/context snapshots, and a deterministic repair scaffold exist, but canonical parent-child PlanArtifact v2, PlanDiff, independent certificates, ownership-aware repair, progressive repair, authoritative pipeline runner, and explanation evidence contracts remain planned work.
- Literature files support the safe claim that the contribution combines ownership-labeled commitments, progressive repair neighborhoods, lexicographic preservation-before-utility objectives, independent validation, and evidence-grounded explanations for user-owned itinerary repair under localized disruptions.
- Limitations: this task intentionally did not implement code, did not mark implementation checkboxes complete, did not resolve pre-existing dirty worktree changes, and did not run full source tests because the requested deliverable was a planning document.

### Validation

- Structural heading check for G0-G8: passed; every phase contains the required @codex-phase-plan headings in order.
- Concept coverage spot check with rg for TripTide guardrails, ConstraintOrigin, RepairRadius, RoutingProvider, LodgingCategory, PipelineRun, run_research_pipeline, road_validated, ruff, and pytest: passed.
- python -m ruff check src tests scripts: not run; planning-only documentation task with no code implementation.
- python -m pytest: not run; planning-only documentation task with no code implementation.

### Conclusion

- The requested implementation-ready planning artifact now exists under docs/planning and is grounded in the repository and literature documents rather than invented structure.

### Next steps

**Codex can proceed:**

- Implement G0 first: repository truth, Phase 0 closeout, strict route-source validation, artifact manifest checks, and no public-route or geodesic publication claims.
- After G0 passes, implement G1: PlanArtifact v2 migration, PlanRepository, typed PlanDiff, and EvaluationCertificate contracts before repair master logic.

**Human reflection:**

- The most important sequencing choice is to stabilize evidence truthfulness and parent-child artifact contracts before expanding solver behavior or UI work.

### Human action

- Review docs/planning/context_aware_itinerary_repair_detailed_phase_plan.md and approve G0 as the next implementation gate if the ordering matches your research priorities.

## Incorporate LLM preference-to-commitment plan

- Status: completed
- Start local time: 2026-07-06 14:41:14 CDT
- End local time: 2026-07-06 14:47:40 CDT-0500
- Duration: Not recorded

### Goal

- Incorporate the LLM preference-to-commitment architecture into the Context-Aware, Inspectable Itinerary Repair phase plan without implementing code.

### What changed

- docs/planning/context_aware_itinerary_repair_detailed_phase_plan.md: added G6.5 evidence-bounded LLM preference-to-commitment work package and updated the executive summary, implementation locks, artifact flow, G7 explanation bridge, G8 study metrics, and final definition of done.
- CODEX_EDIT_LOG.md: appended this completed planning report.

### What was found

- The existing plan already bounded LLM verbalization in G7 and inactive LLM constraints in G1, but lacked a detailed pre-repair taste hypothesis and suggested-commitment layer.
- The added architecture keeps the LLM out of generation, optimization, feasibility, routing validity, hotel availability, and final evaluation; LLM outputs remain inactive until deterministic validation and user confirmation.
- The plan now treats taste as an evidence-backed hypothesis from parent-plan artifacts, not ground-truth user preference, and defines study metrics for taste agreement, commitment acceptance, false suggestions, preservation gain, and repair cost.
- Limitations: this was a documentation-only planning update; no code, prompt templates, UI, tests, or live LLM integrations were implemented.

### Validation

- Phase heading structure check: passed for 10 phase/work-package sections, including G6.5, with all required @codex-phase-plan headings in order.
- Targeted concept coverage check: passed for TasteEvidencePack, TasteDimension, SuggestedCommitment, UserCommitmentDecision, PromptRun, CounterfactualQuestionInterpreter, PreferenceCommitmentExplanationBridge, PreferenceCommitmentStudyMetrics, and LLM study conditions.
- Trailing whitespace check on touched files: passed.
- ASCII check on the plan document: passed.
- python -m ruff check src tests scripts: not run; planning-only documentation update with no code implementation.
- python -m pytest: not run; planning-only documentation update with no code implementation.

### Conclusion

- The detailed phase plan now includes the LLM as a bounded mixed-initiative preference-to-commitment assistant while preserving the solver/evaluator boundaries.

### Next steps

**Codex can proceed:**

- Implement G0 and G1 before any LLM work, then consider G6.5 only after parent artifacts, diff, repair inputs, and pipeline replay are stable.

**Human reflection:**

- The strongest IUI/CHI framing is not LLM planning, but making implicit itinerary taste visible, editable, and operational through confirmed repair commitments.

### Human action

- Review the new G6.5 work package and decide whether the LLM taste module should remain between G6 and G7 or be split further into prototype and study subpackages.

## Phase 0.0 validation harness

- Status: completed
- Start local time: 2026-07-06 18:24:34 CDT
- End local time: 2026-07-06 18:48:18 Central Daylight Time-0500
- Duration: 23m 20s

### Goal

- Implement Phase 0.0 by stabilizing pytest temp/cache behavior, adding a project check wrapper, and recording the current problem manifest.

### What changed

- tests/conftest.py: added pytest startup configuration that forces temp paths into .codex_tmp_pytest/pytest and replaces tempfile.TemporaryDirectory with a workspace-safe context manager.
- pyproject.toml: disabled pytest cacheprovider through pytest addopts to avoid managed-workspace cache permission warnings.
- scripts/run_project_checks.py: added a validation wrapper with workspace temp configuration, command execution, failure classification, and JSON summary output.
- tests/test_project_checks.py: added focused tests for environment, product-code, and timeout failure classification.
- tests/data/test_context_snapshot.py: changed the missing-table fixture to copy contexts while excluding route_options.csv instead of deleting it.
- docs/current/current_problem_manifest.md: added the active problem manifest linking current blockers to phases and acceptance checks.

### What was found

- Python tempfile respected the workspace temp path after conftest setup, but directories created through tempfile/mkdtemp were not writable for nested data folders in this managed Windows workspace.
- The workspace permits writing files under .codex_tmp_pytest but denies delete and rename operations, so tests must avoid delete-based fixture setup and temp cleanup must be best-effort.
- Ruff passes, though its cache write still reports access-denied warnings; pytest cache warnings are removed by disabling cacheprovider.

### Validation

- python -m ruff check src tests scripts: passed; ruff emitted non-fatal cache write warnings.
- python -m pytest tests/data/test_context_snapshot.py tests/test_project_checks.py: 8 passed.
- python -m pytest: 84 passed in 461.41s.
- python scripts/run_project_checks.py: passed ruff, context snapshot pytest, and full pytest; wrote results/quality/project_check_summary.json.

### Conclusion

- Phase 0.0 is implemented: validation no longer fails on the prior temp-permission issue, the check wrapper works, and the current problem manifest exists.

### Next steps

**Codex can proceed:**

- Implement Phase 0.1 artifact lineage and post-solve mutation gate.

**Human reflection:**

- The managed workspace denies deletion/rename even for generated temp files, so future tests should prefer construction-only fixtures over copy-then-delete setup patterns.

### Human action

- Review the new current problem manifest and confirm Phase 0.1 should be next.

## Phase 0.1 artifact lineage

- Status: completed
- Start local time: 2026-07-06 20:01:35 CDT
- End local time: 2026-07-06 20:21:32 Central Daylight Time-0500
- Duration: 19m 7s

### Goal

- Implement Phase 0.1 artifact lineage and the post-solve mutation gate.

### What changed

- src/itinerary_system/research_artifacts.py: added PlanArtifactV2, MutationReport, v1 migration, post-solve mutation detection, child-plan creation, and certificate/run invalidation helpers.
- src/itinerary_system/phase0_exporter.py: emits v2 plan records with ordered days and route ids, and invalidates solver certification for known required-anchor or placeholder post-solve edits.
- tests/test_artifact_lineage.py: added focused lineage, mutation, child-plan, and certificate invalidation tests.
- tests/test_research_foundation.py: asserts v2 Phase 0 plan artifacts and exporter invalidation when anchor mutation evidence is present.
- docs/current/current_problem_manifest.md: updated CP-001 artifact-lineage status after implementation.

### What was found

- Phase 0 export could adopt v2 plan records additively because the validator already accepts flexible JSONL records with required lineage fields.
- Known required-anchor insertion evidence is available in exported route status and method notes, so Phase 0.1 can invalidate certification without changing planner or solver behavior.

### Validation

- python -m ruff check src tests scripts: passed; ruff emitted non-fatal .ruff_cache access-denied warnings.
- python -m pytest tests\\test_artifact_lineage.py tests\\test_research_foundation.py: passed, 28 tests.
- python scripts\\run_project_checks.py: passed; ruff, context snapshot pytest, and full pytest all passed; full pytest collected 91 tests.
- git diff --check on tracked Phase 0.1 edits: passed.

### Conclusion

- Phase 0.1 is implemented: plan artifacts now carry v2 lineage fields and known post-solve mutation evidence invalidates certification at Phase 0 export.

### Next steps

**Codex can proceed:**

- Proceed to Phase 0.2 route matrix boundary and acceptance tests.

**Human reflection:**

- Broader production ownership and repair-solver integration remains intentionally deferred to later phases in the current problem plan.

### Human action

- None

## Ruff validation cache correction

- Status: completed
- Start local time: 2026-07-06 20:42:32 CDT
- End local time: 2026-07-06 21:05:22 Central Daylight Time-0500
- Duration: 12m 3s

### Goal

- Audit and correct the Ruff validation path before Phase 0.2.

### What changed

- scripts/run_project_checks.py: disables Ruff caching with RUFF_NO_CACHE and runs ruff check with --no-cache in the standard validation wrapper.
- tests/test_project_checks.py: added a regression check that workspace temp configuration disables Ruff cache use.
- docs/current/current_problem_manifest.md: updated the current validation command to use python -m ruff check --no-cache src tests scripts.

### What was found

- Plain python -m ruff check src tests scripts exits 0 but emits .ruff_cache access-denied warnings in this managed workspace.
- python -m ruff check --no-cache src tests scripts exits 0 with clean stdout and empty stderr.

### Validation

- python -m ruff check --no-cache src tests scripts: passed with no stderr warnings.
- python -m pytest tests\\test_project_checks.py: passed, 4 tests.
- python scripts\\run_project_checks.py: passed; summary shows Ruff command includes --no-cache and stderr_excerpt is empty; full pytest collected 92 tests and passed.
- git diff --check on tracked Ruff validation edits: passed.

### Conclusion

- The standard validation path now treats Ruff as a clean no-cache check rather than a warning-tolerant cached check.

### Next steps

**Codex can proceed:**

- Proceed to Phase 0.2 after this corrected validation baseline.

**Human reflection:**

- The raw cached Ruff command still warns in this managed workspace, so future local instructions should prefer the no-cache form.

### Human action

- None

## Phase 0.2 route matrix boundary

- Status: completed
- Start local time: 2026-07-07 14:40:01 CDT
- End local time: 2026-07-07 15:17:06 中部夏令时-0500
- Duration: 36m 22s

### Goal

- Implement Phase 0.2 route matrix and road-validation boundary.

### What changed

- src/itinerary_system/routing/matrix.py: added RouteMatrixCell, RouteMatrix, strict publication errors, CSV/context loaders, explicit geodesic fallback matrix, route-result construction, and SolverRouteMatrixAdapter.
- src/itinerary_system/routing/provider.py: added provider protocol and request/activation/snap dataclasses for future OSRM/cache providers.
- src/itinerary_system/routing/__init__.py: exported route matrix, provider, adapter, and strict error types.
- src/itinerary_system/multi_objective_route.py: added optional route matrix/adapter inputs, publication/demo routing modes, stable node ID mapping, matrix-backed travel tables, and matrix-backed route totals.
- src/itinerary_system/hierarchical_gurobi.py: added optional route matrix/adapter inputs for intercity and strict nature-region detour scoring, with publication-mode missing-matrix rejection.
- src/itinerary_system/route_gurobi_oracle.py: forwards route matrix controls to the active day-route solver and marks the legacy block as quarantined.
- tests/routing/test_route_matrix.py: added RouteMatrix, fallback gating, cache loading, solver strict-mode, and hierarchical adapter tests.
- docs/current/current_problem_manifest.md: updated CP-002 route matrix boundary status.
- docs/planning/travel_itinerary_repair_technical_specification.md: added Phase 0.2 status notes for ROUTE-001 and ROUTE-004.

### What was found

- The project already had RouteResult provenance and road cache rows; the missing boundary was solver-facing RouteMatrix injection and strict-mode refusal.
- The managed shell resolved bare python to a Windows app-alias stub during this turn, so validation used the concrete Python 3.12 path recorded by the project-check wrapper.
- Route oracle legacy optimization code is unreachable after the wrapper return, so Phase 0.2 quarantined it rather than deleting it without a separate equivalence pass.

### Validation

- C:\\Users\\1\\AppData\\Local\\Programs\\Python\\Python312\\python.exe -m ruff check --no-cache src tests scripts: passed with empty stderr.
- C:\\Users\\1\\AppData\\Local\\Programs\\Python\\Python312\\python.exe -m pytest tests\\routing\\test_route_matrix.py: passed, 8 tests.
- Route-focused existing pytest selection: passed, 12 tests including route matrix, open-path route solver, hierarchical allocation/pass-through, and route oracle wrapper tests.
- C:\\Users\\1\\AppData\\Local\\Programs\\Python\\Python312\\python.exe scripts\\run_project_checks.py: passed; full pytest collected 100 tests and all passed; summary shows Ruff --no-cache with empty stderr.
- git diff --check on tracked Phase 0.2 edits: passed; Git emitted only CRLF normalization warnings.

### Conclusion

- Phase 0.2 is implemented: publication-mode solver paths now require RouteMatrix evidence and reject missing, fallback, or non-road-validated cells; demo geodesic fallback remains explicit and auditable.

### Next steps

**Codex can proceed:**

- Proceed to Phase 0.3 utility source-missingness and claim-safe scores.

**Human reflection:**

- Complete publication claims still depend on later provider/pipeline work that generates complete validated matrices for benchmark contexts.

### Human action

- None

## Validated route matrix and Phase 0.3 utility missingness

- Status: completed
- Start local time: 2026-07-07 15:53:16 CDT
- End local time: 2026-07-07 16:13:19 Central Daylight Time-0500
- Duration: 19m 40s

### Goal

- Build validated route matrix validation artifacts and implement Phase 0.3 utility source-missingness behavior.

### What changed

- src/itinerary_system/routing/matrix.py: added RouteMatrixValidationReport, required-pair extraction, publication-readiness validation, matrix DataFrame serialization, artifact writing, and validated-matrix build helper.
- scripts/build_validated_route_matrix.py: added CLI to build matrix artifacts from route_options or road-cache CSV and fail when required cells are not publication-ready.
- src/itinerary_system/routing/__init__.py: exported route matrix validation/report APIs.
- src/itinerary_system/utility_model.py: added source masks, masked source normalization, masked MCDA scoring, TOPSIS missing-source neutrality, source-ablation audit, and utility output fields.
- src/itinerary_system/data_enrichment.py: emits source availability masks, source coverage, and legacy data_confidence alias from the shared mask builder.
- src/itinerary_system/schemas.py: added source coverage, uncertainty, and missing-source fields to EnrichedPOI.
- tests/routing/test_route_matrix.py: added matrix validation/report and CLI failure tests, and corrected the default config path.
- tests/test_utility_missingness.py: added Phase 0.3 tests for missing-vs-poor Yelp, equal non-Yelp POIs, all-source-missing fallback, coverage/uncertainty separation, and deterministic ablation output.
- docs/current/current_score_audit.md: updated utility-source wording to describe masked missingness instead of zero penalties.
- docs/reference/data_dictionary.md: documented source mask fields, ablation audit, and validated route matrix artifacts.
- docs/current/current_problem_manifest.md: updated CP-002 and CP-003 statuses.

### What was found

- The project already built road-route caches, but needed a RouteMatrix-level publication-readiness report for solver-required sequences.
- Missing Yelp and present-but-poor Yelp now have distinct utility semantics: missing is excluded from the denominator, while poor available evidence can lower score.
- Git now detects dubious ownership under the current Windows user, so workspace git inspections used a command-local safe.directory override rather than changing global Git config.

### Validation

- C:\\Users\\1\\AppData\\Local\\Programs\\Python\\Python312\\python.exe -m ruff check --no-cache src tests scripts: passed with empty stderr.
- C:\\Users\\1\\AppData\\Local\\Programs\\Python\\Python312\\python.exe -m pytest tests\\routing\\test_route_matrix.py tests\\test_utility_missingness.py: passed, 16 tests.
- C:\\Users\\1\\AppData\\Local\\Programs\\Python\\Python312\\python.exe -m pytest tests\\test_research_foundation.py: passed, 22 tests.
- C:\\Users\\1\\AppData\\Local\\Programs\\Python\\Python312\\python.exe scripts\\run_project_checks.py: passed; full pytest collected 108 tests and all passed; summary shows Ruff --no-cache with empty stderr.
- git -c safe.directory=... diff --check on tracked edited files: passed; Git emitted only CRLF normalization warnings.

### Conclusion

- Validated route matrix artifacts can now be built and checked, and Phase 0.3 missing-source utility behavior is implemented with coverage, uncertainty, and utility kept separate.

### Next steps

**Codex can proceed:**

- Proceed to Phase 1.0 parent plan, ownership, and plan diff foundation.

**Human reflection:**

- Live or local routing provider work is still needed to populate complete road-valid matrices for large benchmark contexts, but the validation/report gate now exists.

### Human action

- None

## Phase 1.0 parent plan and diff foundation

- Status: completed
- Start local time: 2026-07-07 19:04:15 CDT
- End local time: 2026-07-07 20:05:39 Central Daylight Time-0500
- Duration: 60m 40s

### Goal

- Implement canonical parent/child plan ownership, append-only storage, typed plan diffs, and repair scaffold lineage metadata.

### What changed

- src/itinerary_system/plans/models.py: added closed ownership vocabularies, OwnedConstraint, OwnershipPolicy, PlanDiff dataclasses, validation helpers, and Phase 0 route-stops plan construction.
- src/itinerary_system/plans/repository.py: added append-only JSON PlanRepository with load/save helpers, conflict detection, hash verification, and index writing.
- src/itinerary_system/plans/diff.py: added PlanDiffBuilder and compute_plan_diff for stop additions/deletions, day moves, time shifts, order, lodging, road changes, unchanged days, and weighted edit cost.
- src/itinerary_system/plans/__init__.py: exported the canonical plan APIs.
- src/itinerary_system/research_artifacts.py: normalized ordered-day serialization and added owned_constraint_records() compatibility helper.
- src/itinerary_system/repair_planner.py: added parent_plan_id, child_plan_id, and plan_diff metadata for existing deterministic repair outputs.
- src/itinerary_system/schemas.py: added additive plan lineage, route-id, and owned-constraint fields to HierarchicalGurobiPlan.
- data/benchmark/parent_plans/plan_demo_current.json: added current demo parent plan artifact with ownership records and content hash.
- tests/plans/test_models.py: added ownership, relaxation, validation, and Phase 0 route-stops construction tests.
- tests/plans/test_repository.py: added append-only save/load/conflict/missing-plan tests.
- tests/plans/test_diff.py: added identity, typed-change, weighted-cost, and incompatible-catalog tests.
- tests/test_repair_planner.py: asserted repair scaffold emits parent/child plan IDs and diff metadata.
- scripts/run_project_checks.py: raised default check timeout to 900 seconds so the current full suite can complete in the wrapper.
- docs/current/current_problem_manifest.md: updated CP-004 to reflect the implemented Phase 1.0 substrate.
- docs/planning/travel_itinerary_repair_technical_specification.md: documented implementation status for PLAN-001, PLAN-002, and DIFF-001.

### What was found

- The existing repair planner could be annotated additively; no repair decision behavior had to change for Phase 1.0.
- JSON round trips convert nested ordered-day stop tuples to lists, so PlanArtifactV2 serialization needed normalization for append-only idempotence.
- The current full pytest suite collects 119 tests and takes about 10-11 minutes on this Windows workspace, exceeding the old 600-second wrapper timeout.

### Validation

- C:\Users\1\AppData\Local\Programs\Python\Python312\python.exe -m ruff check --no-cache src tests scripts: passed.
- C:\Users\1\AppData\Local\Programs\Python\Python312\python.exe -m pytest tests/plans tests/test_repair_planner.py tests/test_artifact_lineage.py: passed, 23 tests.
- C:\Users\1\AppData\Local\Programs\Python\Python312\python.exe -m pytest: passed, 119 tests in 676.68s.
- C:\Users\1\AppData\Local\Programs\Python\Python312\python.exe scripts/run_project_checks.py: passed; summary shows Ruff, context snapshot pytest, and full pytest all passed.

### Conclusion

- Phase 1.0 is implemented: canonical ownership records, append-only parent storage, typed plan diffs, and repair scaffold lineage metadata are available and validated.

### Next steps

**Codex can proceed:**

- Proceed to Phase 2.0 progressive ownership-aware repair solver, starting with repair neighborhoods that consume PlanArtifactV2, OwnedConstraint, RouteMatrix, and PlanDiff.

**Human reflection:**

- The standard full-suite validation now needs a generous timeout because the Windows workspace runs the 119-test suite near the 11-minute mark.
- Phase 1.0 makes repair changes measurable, but it does not yet improve repair decisions; that claim depends on the Phase 2 solver.

### Human action

- None

## Phase 2.0 REPAIR-001 repair neighborhoods

- Status: completed
- Start local time: 2026-07-07 20:45:02 CDT
- End local time: 2026-07-07 21:02:12 Central Daylight Time-0500
- Duration: 16m 35s

### Goal

- Implement the first Phase 2.0 slice: parent-plan-aware repair neighborhoods that freeze unaffected assignments and protected constraints.

### What changed

- src/itinerary_system/repair/neighborhood.py: added RepairRadius order, ParentPlanIndex, RepairNeighborhood, RepairNeighborhoodBuilder, affected-day inference, editable-set construction, and constraint-freezing rules.
- src/itinerary_system/repair/__init__.py: exported repair neighborhood APIs.
- src/itinerary_system/__init__.py: exported the new repair neighborhood types and helpers at package root.
- tests/repair/test_neighborhood.py: added REPAIR-001 tests for unaffected-day freezing, adjacent-day boundary inclusion, booked-lodging permission, affected-day inference, and full-reoptimization ordering.
- docs/current/current_problem_manifest.md: updated CP-005 to show REPAIR-001 is implemented while later Phase 2 solver work remains open.
- docs/planning/travel_itinerary_repair_technical_specification.md: added implementation status for REPAIR-001.

### What was found

- Phase 2.0 is broader than one safe step, so this pass implements REPAIR-001 only and leaves master-model, lexicographic solver, day-route solver, progressive controller, and evaluator work open.
- Existing repair request records can be consumed by the neighborhood builder through confirmed_constraints without changing deterministic repair scaffold behavior.
- Route constraints keyed by route ID need to be tied back to route days before deciding whether a neighborhood can edit them.

### Validation

- C:\Users\1\AppData\Local\Programs\Python\Python312\python.exe -m ruff check --no-cache src tests scripts: passed.
- C:\Users\1\AppData\Local\Programs\Python\Python312\python.exe -m pytest tests/repair/test_neighborhood.py: passed, 5 tests.
- C:\Users\1\AppData\Local\Programs\Python\Python312\python.exe scripts/run_project_checks.py: passed; Ruff, context snapshot pytest, and full pytest passed with 124 collected tests.

### Conclusion

- REPAIR-001 is implemented and validated: the system can now build progressive editable neighborhoods around immutable parent plans while freezing locked, booked, and outside-scope commitments.

### Next steps

**Codex can proceed:**

- Proceed to REPAIR-002 ownership-aware repair master/change-variable scaffold that consumes RepairNeighborhood, PlanArtifactV2, OwnedConstraint, RouteMatrix, and PlanDiff.

**Human reflection:**

- This layer says what may change, but it still does not choose an optimized repair; improvement claims remain blocked until the master model, lexicographic solve, evaluator, and progressive controller are implemented.

### Human action

- None

## Phase 2.0 REPAIR-002 ownership-aware repair master

- Status: completed
- Start local time: 2026-07-07 21:11:19 CDT
- End local time: 2026-07-07 21:40:33 Central Daylight Time-0500
- Duration: 28m 46s

### Goal

- Implement REPAIR-002: ownership-aware repair master and typed change-variable scaffold relative to immutable parent plans.

### What changed

- src/itinerary_system/repair/change_variables.py: added decision-variable domains, repair variable kinds, typed change-variable kinds, objective terms/components, RepairVariableSet, and deterministic change weights.
- src/itinerary_system/repair/master_model.py: added RepairConstraint, RepairSolution, RepairModel, RepairMasterModel, build_repair_master_model(), fixed-assignment validation, objective component export, route-pair requirements, and child-plan extraction with diff cost.
- src/itinerary_system/repair/__init__.py: exported change-variable and master-model APIs.
- src/itinerary_system/__init__.py: exported REPAIR-002 APIs at the package root.
- tests/repair/test_master_model.py: added acceptance tests for variable declarations, locked POI protection, booked lodging permission, objective component export, and child plan extraction.
- docs/current/current_problem_manifest.md: updated CP-005 to record REPAIR-002 implementation while later solver/controller phases remain open.
- docs/planning/travel_itinerary_repair_technical_specification.md: added implementation status for REPAIR-002.

### What was found

- REPAIR-002 can be completed as a solver-neutral master scaffold: it creates variables and constraints and evaluates candidate assignments, while REPAIR-003/004/005 remain responsible for actual lexicographic solve execution and progressive orchestration.
- Deleted parent stops should not also count as day moves just because their day assignment is absent; the change indicator evaluation now treats deletion and day movement separately.
- Booked lodging needs both a lodging-edit radius and explicit booked-relaxation permission before the model exposes lodging/relaxation variables as editable.

### Validation

- C:\Users\1\AppData\Local\Programs\Python\Python312\python.exe -m ruff check --no-cache src tests scripts: passed.
- C:\Users\1\AppData\Local\Programs\Python\Python312\python.exe -m pytest tests/repair: passed, 10 tests.
- C:\Users\1\AppData\Local\Programs\Python\Python312\python.exe scripts/run_project_checks.py: passed; Ruff, context snapshot pytest, and full pytest passed with 129 collected tests.

### Conclusion

- REPAIR-002 is implemented and validated: repair models now expose parent-relative selection, day, lodging, relaxation, and typed change variables with locked/booked/frozen safeguards and objective component exports.

### Next steps

**Codex can proceed:**

- Proceed to REPAIR-003 sequential lexicographic solver that consumes RepairModel objective components and records stage results.

**Human reflection:**

- The model can evaluate candidate assignments and extract child plans, but it intentionally does not optimize them yet; repair quality claims remain blocked until REPAIR-003 through REPAIR-005 are implemented.

### Human action

- None

## Phase 2.0 REPAIR-003 sequential lexicographic solver

- Status: completed
- Start local time: 2026-07-07 21:45:25 CDT
- End local time: 2026-07-07 22:01:26 Central Daylight Time-0500
- Duration: 15m 41s

### Goal

- Implement REPAIR-003: sequential lexicographic solver over repair candidate solutions with stage preservation and persisted results.

### What changed

- src/itinerary_system/repair/lexicographic.py: added ObjectiveTolerances, LexicographicStageResult, LexicographicResult, LexicographicRepairSolver, solve_lexicographically(), child-plan extraction, PlannerRun conversion, and small candidate-choice Gurobi reference solver.
- src/itinerary_system/repair/__init__.py: exported REPAIR-003 solver APIs.
- src/itinerary_system/__init__.py: exported REPAIR-003 APIs at package root.
- tests/repair/test_lexicographic.py: added tests for prior-stage preservation, stage tolerance tradeoffs, persisted status/bound/gap fields, failed PlannerRun emission, and Gurobi-reference agreement.
- docs/current/current_problem_manifest.md: updated CP-005 to record REPAIR-003 implementation while day-route/progressive/evaluator work remains open.
- docs/planning/travel_itinerary_repair_technical_specification.md: added implementation status for REPAIR-003.

### What was found

- The current safest REPAIR-003 boundary is an exact candidate selector: REPAIR-004 and REPAIR-005 should generate and orchestrate candidates, while REPAIR-003 enforces sequential objective semantics.
- Explicit candidate lists should be solved as provided; the parent baseline is only the default when no candidates are supplied, so it does not silently dominate generated repair candidates.
- The local Gurobi installation can solve tiny binary reference models under its restricted license, allowing a concrete small-instance equivalence test.

### Validation

- C:\Users\1\AppData\Local\Programs\Python\Python312\python.exe -m ruff check --no-cache src tests scripts: passed.
- C:\Users\1\AppData\Local\Programs\Python\Python312\python.exe -m pytest tests/repair: passed, 15 tests.
- C:\Users\1\AppData\Local\Programs\Python\Python312\python.exe scripts/run_project_checks.py: passed; Ruff, context snapshot pytest, and full pytest passed with 134 collected tests.

### Conclusion

- REPAIR-003 is implemented and validated: repair candidate solutions now pass through sequential lexicographic stage filtering, produce persisted stage results, and emit PlannerRun records for failed stages.

### Next steps

**Codex can proceed:**

- Proceed to REPAIR-004 day-route subproblem, generating feasible candidate RepairSolution records from RouteMatrix-backed day routing.

**Human reflection:**

- The solver now chooses among explicit candidates but does not generate candidates itself; repair quality claims still need REPAIR-004 candidate generation, REPAIR-005 progressive orchestration, and VERIFY-001 independent evaluation.

### Human action

- None

## Phase 2.0 REPAIR-004 day-route subproblem

- Status: completed
- Start local time: 2026-07-07 22:12:31 CDT
- End local time: 2026-07-07 22:27:13 Central Daylight Time-0500
- Duration: 14m 18s

### Goal

- Implement REPAIR-004: RouteMatrix-backed typed day-route subproblem that generates RepairSolution candidates with windows, visit durations, and fixed day assignments.

### What changed

- src/itinerary_system/repair/day_route_solver.py: added DayRouteSolverConfig, DayRouteCandidate, DayRouteSubproblemResult, DayRouteSolver, and solve_day_route_subproblem() using RouteMatrix legs, anchors, opening windows, visit duration, max-day time, and fixed assignment validation.
- src/itinerary_system/repair/master_model.py: child-plan extraction now honors day-route sequence metadata produced by RepairSolution.
- src/itinerary_system/repair/__init__.py: exported day-route solver APIs.
- src/itinerary_system/__init__.py: exported day-route solver APIs at package root.
- src/itinerary_system/routing/__init__.py: exported RouteMatrixError for uniform repair-side matrix failure handling.
- tests/repair/test_day_route_solver.py: added REPAIR-004 tests for matrix-backed travel durations, generated same-day replacement candidates, opening-window and fixed-day violations, and strict publication rejection of fallback cells.
- docs/current/current_problem_manifest.md: updated CP-005 to record REPAIR-004 implementation while controller/evaluator work remains open.
- docs/planning/travel_itinerary_repair_technical_specification.md: added implementation status for REPAIR-004.

### What was found

- The day-route layer should generate/evaluate full RepairSolution candidates, while REPAIR-003 remains responsible for lexicographic choice among those candidates.
- Service duration must be applied to the first stop even when no start anchor is present; route travel still comes only from RouteMatrix legs.
- Child-plan extraction needed to honor day-route sequence metadata so replacement candidates preserve the chosen within-day order.

### Validation

- C:\Users\1\AppData\Local\Programs\Python\Python312\python.exe -m ruff check --no-cache src tests scripts: passed.
- C:\Users\1\AppData\Local\Programs\Python\Python312\python.exe -m pytest tests/repair: passed, 19 tests.
- C:\Users\1\AppData\Local\Programs\Python\Python312\python.exe scripts/run_project_checks.py: passed; Ruff, context snapshot pytest, and full pytest passed with 138 collected tests.

### Conclusion

- REPAIR-004 is implemented and validated: the repair stack can now produce RouteMatrix-backed single-day candidate solutions with route evidence, time windows, visit duration, fixed-day validation, and strict publication routing gates.

### Next steps

**Codex can proceed:**

- Proceed to REPAIR-005 progressive repair controller that builds neighborhoods, generates day-route candidates, solves lexicographically, records attempts, and stops at the smallest eligible radius.

**Human reflection:**

- The system can generate and choose day-route candidates now, but production accepted repair still needs REPAIR-005 orchestration and VERIFY-001 independent final-plan evaluation before improvement claims are safe.

### Human action

- None

## REPAIR-003 validation audit

- Status: completed
- Start local time: 2026-07-08 09:19:05 -05:00
- End local time: 2026-07-08 09:23:27 Central Daylight Time-0500
- Duration: 4m 21s

### Goal

- Proceed to REPAIR-003 by auditing the current sequential lexicographic solver implementation and validating its targeted tests.

### What changed

- CODEX_EDIT_LOG.md: recorded the REPAIR-003 validation audit for this proceed request; no solver source or tests were changed.

### What was found

- REPAIR-003 was already implemented in src/itinerary_system/repair/lexicographic.py with matching tests and an existing completed implementation log entry.
- The current workspace also contains later repair work, so this pass intentionally scoped validation to the REPAIR-003 solver surface requested by the user.

### Validation

- C:\Users\1\AppData\Local\Programs\Python\Python312\python.exe -m pytest tests/repair/test_lexicographic.py: passed, 5 tests.
- C:\Users\1\AppData\Local\Programs\Python\Python312\python.exe -m ruff check src/itinerary_system/repair/lexicographic.py tests/repair/test_lexicographic.py: passed.

### Conclusion

- REPAIR-003 remains implemented and targeted validation passes; no code patch was required.

### Next steps

**Codex can proceed:**

- Proceed from the current repo state to the next unclosed repair orchestration work after confirming whether to continue with REPAIR-005 or rewind to any earlier phase.

**Human reflection:**

- REPAIR-003 only chooses among explicit candidate RepairSolution records; candidate generation and accepted repair orchestration are later-phase responsibilities.

### Human action

- None

## Phase 2.0 REPAIR-005 progressive repair controller

- Status: completed
- Start local time: 2026-07-08 09:24:29 -05:00
- End local time: 2026-07-08 09:58:43 Central Daylight Time-0500
- Duration: 0h 34m 13s

### Goal

- Implement REPAIR-005: progressive repair orchestration that records attempts, keeps full reoptimization as the final fallback, and accepts only the smallest evaluator-eligible child plan.

### What changed

- src/itinerary_system/repair/progressive.py: completed progressive controller behavior, public outcome/attempt/evaluation/diagnosis records, evaluator-gated candidate loop, enum radius handling, editable-day candidate generation, and component-backed no-success diagnosis.
- src/itinerary_system/repair/__init__.py: exported REPAIR-005 controller and record APIs from the repair package.
- src/itinerary_system/__init__.py: exposed REPAIR-005 APIs at the package root following the existing repair export pattern.
- tests/repair/test_progressive.py: added acceptance tests for smallest eligible radius, repository save, full-reoptimization fallback ordering, stored attempts, and smallest-relaxation diagnosis metrics.
- docs/current/current_problem_manifest.md: updated CP-005 to record REPAIR-005 implementation while leaving VERIFY-001 certification open.
- docs/planning/travel_itinerary_repair_technical_specification.md: added REPAIR-005 implementation status with explicit evaluator-package caveat.

### What was found

- The worktree already contained an unexported progressive.py draft; the first focused test failed at collection because ProgressiveRepairController was not public.
- The draft diagnosis path only noted that candidates were evaluated; REPAIR-005 needs component evidence, so CandidateEvaluationRecord now stores hard/booked relaxation and weighted edit metrics.
- scripts/run_project_checks.py needs a long tool timeout in this workspace because full pytest currently takes about 9 minutes after adding REPAIR-005 tests.

### Validation

- RED check: C:\Users\1\AppData\Local\Programs\Python\Python312\python.exe -m pytest tests/repair/test_progressive.py failed during collection with ImportError for ProgressiveRepairController before exports were patched.
- C:\Users\1\AppData\Local\Programs\Python\Python312\python.exe -m pytest tests/repair/test_progressive.py: passed, 3 tests.
- C:\Users\1\AppData\Local\Programs\Python\Python312\python.exe -m ruff check src tests scripts: passed.
- C:\Users\1\AppData\Local\Programs\Python\Python312\python.exe -m pytest tests/repair: passed, 22 tests.
- C:\Users\1\AppData\Local\Programs\Python\Python312\python.exe -m pytest: passed, 141 tests.
- C:\Users\1\AppData\Local\Programs\Python\Python312\python.exe scripts/run_project_checks.py: passed; Ruff, context snapshot pytest, and full pytest passed, and results/quality/project_check_summary.json reports 141 collected tests.

### Conclusion

- REPAIR-005 is implemented and validated: progressive repair now builds radii in order, solves/evaluates candidates, saves only the first eligible child, records all attempts made, and returns diagnosis metrics when no radius succeeds.

### Next steps

**Codex can proceed:**

- Proceed to VERIFY-001 independent final-plan evaluator and certificate package so the progressive controller can use a production-grade evaluator instead of caller-provided test hooks.

**Human reflection:**

- The controller can orchestrate and gate repairs now, but accepted production-repair claims still depend on the independent final-plan evaluator and complete validated route evidence for real benchmark contexts.

### Human action

- None

## Integrated implementation phase-gate plan

- Status: completed
- Start local time: 2026-07-08 10:59:43 CDT
- End local time: 2026-07-08 11:10:00 CDT
- Duration: Not recorded

### Goal

- Create a new repository-grounded integrated implementation roadmap and phase-gate plan for Context-Aware, Inspectable Itinerary Repair without implementing code.

### What changed

- docs/planning/context_aware_itinerary_repair_integrated_implementation_plan.md: created the integrated implementation plan with sections 0-13, G0-G11 gate plans, architecture diagrams, data models, method signatures, validation rules, LLM prompt protocol, live/event-triggered repair plan, benchmark/evaluation plan, UI/study plan, definition of done, and immediate next tasks.
- CODEX_EDIT_LOG.md: appended this completed planning report.

### What was found

- Files inspected: README.md; docs/README.md; CODEX_EDIT_LOG.md; docs/planning/research_stabilization_and_publication_plan.md; docs/planning/current_problem_fix_phase_plans.md; docs/planning/context_aware_itinerary_repair_detailed_phase_plan.md; docs/planning/travel_itinerary_repair_technical_specification.md; docs/current/current_score_audit.md; docs/current/current_problem_manifest.md; docs/literature/repair_gap_review.md; docs/literature/literature_matrix_repair_gap.md; docs/literature/literature_onboarding_guide.md; docs/literature/evidence_matrix.md; docs/literature/core_paper_reading_cards.md; docs/literature/literature_deep_read_study_report.md; docs/reference/data_dictionary.md; docs/reference/code_quality_workflow.md; required source directories under src/itinerary_system; required tests/scripts/notebook paths.
- Repository truth changed since the older detailed plan: PlanArtifactV2, ownership models, PlanRepository, PlanDiff, RouteMatrix, source missingness masks, repair neighborhoods, repair master scaffold, lexicographic solver, day-route subproblem, and progressive repair controller now exist with tests.
- Still missing: production evaluation package/certificates, benchmark package, canonical pipeline runner, explanation package, LLM taste package, live/event-triggered package, and venue/study readiness package.
- Limitations: this was documentation-only; no code, tests, prompt templates, pipeline runner, evaluator, benchmark, UI, LLM integration, or live repair implementation was added.

### Validation

- Phase heading structure check: passed for 12 G0-G11 phase sections, with all required @codex-phase-plan headings in order.
- ASCII check on the integrated plan: passed.
- Trailing whitespace check on the integrated plan: passed.
- Targeted coverage check: passed for RelaxationPolicy, StopChange, DayMove, TimeShift, OrderChange, LodgingChange, RoadChange, ImpactReport, LiveRepairLineage, LLM prompt calls, TriggerDecision values, and roadmap policy.
- python C:\Users\1\.codex\skills\code-edit-report\scripts\append_code_edit_log.py ...: not run successfully because this Windows environment only exposes the Microsoft Store python shim and `py` reports no installed Python.
- python -m ruff check src tests scripts: not run; planning-only documentation task with no code implementation.
- python -m pytest: not run; planning-only documentation task with no code implementation.

### Conclusion

- The requested integrated roadmap now exists at docs/planning/context_aware_itinerary_repair_integrated_implementation_plan.md and is grounded in the current repository rather than the older Phase 0-only snapshot.

### Next steps

**Codex can proceed:**

- Start with G0 validation/current-manifest refresh, then G3 independent evaluator and EvaluationCertificate, because current progressive repair already has an evaluator hook but lacks the production evaluator.

**Human reflection:**

- The integrated plan should now supersede older planning documents for sequencing, while older docs remain useful evidence and specification references.

### Human action

- Review the integrated plan and decide whether the next implementation session should begin with G0 manifest validation or jump directly to G3 evaluator/certificate implementation.

## Phase 3.0 VERIFY-001 independent final-plan evaluator

- Status: completed
- Start local time: 2026-07-08 11:07:12 -05:00
- End local time: 2026-07-08 11:35:24 Central Daylight Time-0500
- Duration: 0h 28m 10s

### Goal

- Implement VERIFY-001: independent final-plan evaluator and certificate records that recompute final plan validity from artifacts and route/context evidence.

### What changed

- src/itinerary_system/evaluation/certificate.py: added EvaluationFinding and PlanEvaluationCertificate with content-hash binding, warning/failure separation, eligibility fields, and to_record compatibility for REPAIR-005 evaluator hooks.
- src/itinerary_system/evaluation/plan_evaluator.py: added PlanEvaluatorConfig and PlanEvaluator for artifact/run linkage, stale certificate checks, hard owned constraints, route matrix publication readiness, schedule/windows, lodging consistency, budget/weather/closure checks, duplicate visits, and final eligibility status.
- src/itinerary_system/evaluation/__init__.py: exported the independent evaluation APIs.
- tests/evaluation/test_plan_evaluator.py: added VERIFY-001 tests for eligible certificates with separated warnings, mutation/content-hash invalidation, unvalidated route blocking, and locked-stop recomputation from plan artifacts.
- src/itinerary_system/__init__.py: exported PlanEvaluator, PlanEvaluatorConfig, PlanEvaluationCertificate, and EvaluationFinding at the package root.
- docs/current/current_problem_manifest.md: updated CP-006 to record VERIFY-001 implementation while keeping explanation evidence work open.
- docs/planning/travel_itinerary_repair_technical_specification.md: added VERIFY-001 implementation status and explicit boundary before EXPLAIN-001/002.

### What was found

- VERIFY-001 did not exist as a package; the first focused test failed during collection with ModuleNotFoundError for itinerary_system.evaluation.
- PlanArtifactV2 already exposes stable content hashes and certificate_id fields, so the certificate can bind to plan_content_hash without altering artifact storage.
- RouteMatrix.validate_route_matrix already reports publication readiness; the evaluator reuses that rather than trusting solver feasibility flags.

### Validation

- RED check: C:\Users\1\AppData\Local\Programs\Python\Python312\python.exe -m pytest tests/evaluation/test_plan_evaluator.py failed during collection with ModuleNotFoundError before the evaluation package existed.
- C:\Users\1\AppData\Local\Programs\Python\Python312\python.exe -m pytest tests/evaluation/test_plan_evaluator.py: passed, 4 tests.
- C:\Users\1\AppData\Local\Programs\Python\Python312\python.exe -m ruff check src tests scripts: passed.
- C:\Users\1\AppData\Local\Programs\Python\Python312\python.exe -m pytest tests/evaluation tests/repair: passed, 26 tests.
- C:\Users\1\AppData\Local\Programs\Python\Python312\python.exe -m pytest: passed, 145 tests.
- C:\Users\1\AppData\Local\Programs\Python\Python312\python.exe scripts/run_project_checks.py: passed; Ruff, context snapshot pytest, and full pytest passed, and results/quality/project_check_summary.json reports 145 collected tests.

### Conclusion

- VERIFY-001 is implemented and validated: final plans can now receive independent certificates that fail closed for stale content hashes and unvalidated route evidence while preserving nonblocking warnings separately.

### Next steps

**Codex can proceed:**

- Proceed to EXPLAIN-001 structured explanation evidence so numerical and causal explanation claims can reference plan diff, route, constraint, evaluation, and certificate evidence.

**Human reflection:**

- The evaluator now certifies final plans independently, but production benchmark claims still need complete validated route matrices for benchmark contexts and explanation evidence layers.

### Human action

- None

## EXPLAIN-001 structured explanation evidence

- Status: completed
- Start local time: 2026-07-08 23:17:20 -05:00
- End local time: 2026-07-08 23:38:20 Central Daylight Time-0500
- Duration: 20m 37s

### Goal

- Implement EXPLAIN-001 structured explanation evidence so numerical and causal explanation claims cite valid artifacts or fail closed.

### What changed

- src/itinerary_system/explanation/evidence.py - Added EvidenceRecord, ExplanationClaim, evidence containers, publication filtering, and evidence-reference validation.
- src/itinerary_system/explanation/__init__.py - Exported the explanation evidence API.
- src/itinerary_system/__init__.py - Added root package exports for the explanation evidence API.
- tests/explanation/test_evidence.py - Added focused EXPLAIN-001 coverage for missing references, invalid evidence type, serialization, and publication hiding.
- docs/current/current_problem_manifest.md - Updated CP-005/CP-006 status to reflect VERIFY-001 and EXPLAIN-001 while leaving EXPLAIN-002 open.
- docs/planning/travel_itinerary_repair_technical_specification.md - Added EXPLAIN-001 implementation status and clarified the Phase 3.0 remaining explanation work.

### What was found

- Initial focused test was RED with ModuleNotFoundError for itinerary_system.explanation, as expected before implementation.
- Dataclass inheritance needed keyword-only evidence containers because specialized classes set evidence_type defaults.
- validate_explanation_claims now accepts either a single claim/record or an iterable while keeping strict evidence-type validation.

### Validation

- python -m ruff check src tests scripts - passed.
- python -m pytest tests/explanation/test_evidence.py - 5 passed.
- python -m pytest tests/explanation tests/evaluation tests/repair - 31 passed.
- python -m pytest - 150 passed.
- python scripts/run_project_checks.py - passed and wrote results/quality/project_check_summary.json.

### Conclusion

- EXPLAIN-001 is implemented: unsupported numerical/causal claims fail closed and are omitted from publication records unless backed by allowed evidence records.

### Next steps

**Codex can proceed:**

- Proceed to EXPLAIN-002 by adding counterfactual runners and deterministic verbalizers that consume only structured evidence bundles.
- Wire explanation bundles into repair/evaluator pipeline artifacts once PIPE-001 begins.

**Human reflection:**

- Descriptive claims are currently allowed without evidence; decide later whether public-facing descriptive text should also require evidence IDs.

### Human action

- None.

## EXPLAIN-002 counterfactual runner and verbalizer

- Status: completed
- Start local time: 2026-07-08 23:39:31 -05:00
- End local time: 2026-07-08 23:56:41 Central Daylight Time-0500
- Duration: 16m 51s

### Goal

- Implement EXPLAIN-002 so why-not and what-if explanations use sandbox counterfactual run evidence and deterministic claim-to-evidence verbalization.

### What changed

- src/itinerary_system/explanation/counterfactual.py - Added sandbox CounterfactualRequest, CounterfactualRunRecord, dependency-injected CounterfactualRunner, and why-not/what-if evidence builders.
- src/itinerary_system/explanation/verbalizer.py - Added deterministic template verbalizer and claim-to-evidence mapping validator.
- src/itinerary_system/explanation/evidence.py - Added counterfactual run IDs and outcome status to WhyNotEvidence records.
- src/itinerary_system/explanation/__init__.py - Exported EXPLAIN-002 public objects.
- src/itinerary_system/__init__.py - Added root exports for EXPLAIN-002 objects.
- tests/explanation/test_counterfactual.py - Added focused tests for sandbox requests, forced why-not constraints, what-if overrides, failure evidence, parent mutation detection, and deterministic verbalization.
- docs/current/current_problem_manifest.md - Updated CP-006 to mark EXPLAIN-002 implemented and keep Phase 4 pipeline export open.
- docs/planning/travel_itinerary_repair_technical_specification.md - Added EXPLAIN-002 implementation status and clarified remaining pipeline integration.
- docs/planning/current_problem_fix_phase_plans.md - Added Phase 3.0 implementation status note.

### What was found

- The new tests were RED with ImportError for CounterfactualRunner before implementation.
- The counterfactual runner is dependency-injected: callers can pass the progressive repair controller or a test executor, and missing executor returns not_evaluated evidence instead of fabricated causal text.
- Parent-plan mutation is detected by comparing content hashes before and after executor calls.

### Validation

- python -m pytest tests/explanation/test_counterfactual.py - 6 passed.
- python -m ruff check src tests scripts - passed.
- python -m pytest tests/explanation - 11 passed.
- python -m pytest tests/explanation tests/evaluation tests/repair - 37 passed.
- python -m pytest - 156 passed.
- python scripts/run_project_checks.py - passed and wrote results/quality/project_check_summary.json.
- git diff --check - passed; only line-ending warnings were reported.

### Conclusion

- EXPLAIN-002 is implemented as a bounded counterfactual/verbalization contract: why-not and what-if answers cite stored counterfactual run records or failure evidence, and deterministic verbalization hides unsupported claims.

### Next steps

**Codex can proceed:**

- Proceed to Phase 4.0/PIPE-001 by wiring plans, diffs, route matrices, evaluations, certificates, and explanation bundles into immutable run directories.
- Add pipeline-level explanation export once PIPE-001 creates canonical run layout.

**Human reflection:**

- The counterfactual runner intentionally does not create route matrices or solver dependencies; production re-solve behavior depends on injecting the existing repair controller from the future pipeline.

### Human action

- None.

## PIPE-001 immutable pipeline run directory

- Status: completed
- Start local time: 2026-07-08 23:57:46 -05:00
- End local time: 2026-07-09 00:15:15 Central Daylight Time-0500
- Duration: 17m 10s

### Goal

- Implement the first PIPE-001 slice: a package-level pipeline runner that writes immutable, redacted, canonical run artifacts from injected generation or repair executors.

### What changed

- src/itinerary_system/pipeline_runner.py - Added RefreshPolicy, PipelineRunContext, PipelineExecutionResult, PipelineRun, immutable run-directory creation, artifact writers, config redaction, refresh-policy live-API disabling, overwrite protection, and strict eligibility blocking.
- src/itinerary_system/__init__.py - Exported the PIPE-001 runner API.
- tests/test_pipeline_runner.py - Added focused tests for generation layout, redaction, overwrite refusal, repair parent/child/diff/explanation export, strict blocking, and permissive diagnostics.
- docs/current/current_problem_manifest.md - Updated CP-007 to record the PIPE-001 first slice and leave benchmark/notebook work open.
- docs/planning/travel_itinerary_repair_technical_specification.md - Added PIPE-001 implementation status without claiming real executor or benchmark completion.
- docs/planning/current_problem_fix_phase_plans.md - Added Phase 4.0 implementation status for the pipeline-runner slice.

### What was found

- The focused test was RED with ModuleNotFoundError for itinerary_system.pipeline_runner before implementation.
- The existing experiment_runner.py remains notebook-era and very large, so the first safe Phase 4 step is a typed injected-executor run boundary rather than editing notebook business logic directly.
- RefreshPolicy.NEVER now disables live API flags in the resolved config passed to executors and written as redacted config.

### Validation

- python -m pytest tests/test_pipeline_runner.py - 5 passed.
- python -m ruff check src tests scripts - passed.
- python -m pytest tests/test_pipeline_runner.py tests/explanation tests/evaluation tests/repair - 42 passed.
- python -m pytest - 161 passed.
- python scripts/run_project_checks.py - passed and wrote results/quality/project_check_summary.json with 161 collected tests.
- git diff --check - passed; only Windows line-ending warnings were reported.

### Conclusion

- PIPE-001 first slice is implemented and validated: package-level runs now have an immutable, redacted artifact directory contract, while real executor wiring, benchmark suites, and notebook migration remain open.

### Next steps

**Codex can proceed:**

- Proceed to PIPE-001 executor wiring by adapting the existing Phase 0/generation path to return PipelineExecutionResult without notebook dependency.
- Proceed to BENCH-001 after executor wiring by adding deterministic six-family disruption request generation.

**Human reflection:**

- The new pipeline runner intentionally uses injected executors so it does not accidentally activate live providers or rewrite the notebook-era runner in one risky step.

### Human action

- None.

## PIPE-001 Phase 0 executor adapter

- Status: completed
- Start local time: 2026-07-09 00:16:16 -05:00
- End local time: 2026-07-09 00:38:50 CDT-0500
- Duration: Not recorded

### Goal

- Adapt the existing Phase 0 generation evidence path into the canonical package pipeline runner after EXPLAIN-001 was confirmed implemented.

### What changed

- src/itinerary_system/pipeline_runner.py: added run_phase0_generation_executor(), build_phase0_generation_executor(), Phase 0 dataframe/JSONL conversion helpers, grouped route audit records, request records, metrics, and dashboard summaries.
- src/itinerary_system/__init__.py: exported the Phase 0 executor adapter APIs.
- tests/test_pipeline_runner.py: added TDD coverage for Phase 0 legacy artifact export, canonical pipeline artifact export, and strict-mode diagnostic blocking.
- docs/current/current_problem_manifest.md: updated CP-006/CP-007 status for executor-provided explanation export and Phase 0 pipeline adapter completion.
- docs/planning/current_problem_fix_phase_plans.md: updated Phase 4.0 status and remaining missing work.
- docs/planning/travel_itinerary_repair_technical_specification.md: updated VERIFY/EXPLAIN/PIPE status language to distinguish completed adapter/export support from remaining production executor work.
- `git status`: M .codex/code-edit-log.md
- `git status`: M CODEX_EDIT_LOG.md
- `git status`: M docs/current/current_problem_manifest.md
- `git status`: M docs/planning/current_problem_fix_phase_plans.md
- `git status`: M docs/planning/travel_itinerary_repair_technical_specification.md
- `git status`: M src/itinerary_system/__init__.py
- `git status`: M src/itinerary_system/routing/__init__.py
- `git status`: ?? docs/planning/context_aware_itinerary_repair_integrated_implementation_plan.md
- `git status`: ?? docs/slides/context_aware_itinerary_repair_professor_update.pptx
- `git status`: ?? docs/slides/context_aware_itinerary_repair_professor_update_polished.pptx
- `git status`: ?? src/itinerary_system/evaluation/
- `git status`: ?? src/itinerary_system/explanation/
- `git status`: ?? src/itinerary_system/pipeline_runner.py
- `git status`: ?? src/itinerary_system/repair/
- `git status`: ?? tests/evaluation/
- `git status`: ?? tests/explanation/
- `git status`: ?? tests/repair/
- `git status`: ?? tests/test_pipeline_runner.py

### What was found

- EXPLAIN-001 and EXPLAIN-002 were already implemented in the current workspace; the open downstream issue was canonical pipeline export and executor wiring.
- write_phase0_research_artifacts() already writes the stable legacy Phase 0 files and returns dataframes, making it the safest package-level adapter target.
- The first RED test failed at import because build_phase0_generation_executor did not exist, then passed after the adapter implementation.

### Validation

- python -m pytest tests/test_pipeline_runner.py -k phase0_generation_executor: RED first with ImportError for build_phase0_generation_executor, then 2 passed after implementation.
- python -m ruff check src/itinerary_system/pipeline_runner.py tests/test_pipeline_runner.py src/itinerary_system/__init__.py: passed after fixing import order.
- python -m ruff check src tests scripts: passed.
- python -m pytest tests/test_pipeline_runner.py: 7 passed.
- python -m pytest tests/test_pipeline_runner.py tests/explanation tests/evaluation tests/repair: 44 passed.
- python -m pytest: first parallel attempt timed out at 244s; rerun alone passed with 163 passed in 242.71s.
- python scripts/run_project_checks.py: passed; ruff, context snapshot pytest, and full pytest succeeded and refreshed results/quality/project_check_summary.json.
- git diff --check: passed with CRLF normalization warnings only.

### Conclusion

- Phase 0 generation can now be run through the package pipeline boundary without notebook dependency, emitting legacy Phase 0 files plus canonical run artifacts; real optimizer/repair executor wiring, benchmarks, and notebook migration remain open.

### Next steps

**Codex can proceed:**

- Proceed to wire the production repair/progressive controller executor into PipelineExecutionResult, including independent evaluator certificates and generated explanation records.
- After production executors are wired, proceed to BENCH-001 deterministic disruption generation.

**Human reflection:**

- The Phase 0 adapter intentionally preserves legacy artifact names while adding canonical run artifacts, which reduces risk during notebook migration.
- Strict Phase 0 runs still block without validated route evidence; that is expected and useful until benchmark contexts have complete validated matrices.

### Human action

- Review the new PIPE-001 adapter API names if you want a different public naming convention before later callers depend on them.

## PIPE-001 progressive repair executor adapter

- Status: completed
- Start local time: 2026-07-09 00:40:04 -05:00
- End local time: 2026-07-09 00:57:50 CDT-0500
- Duration: Not recorded

### Goal

- Wire the existing REPAIR-005 progressive repair controller into the canonical package pipeline result contract.

### What changed

- src/itinerary_system/pipeline_runner.py: added build_progressive_repair_executor(), run_progressive_repair_executor(), default independent certificate evaluator wrapper, repair request/metrics conversion, and accepted/failed repair explanation evidence builders.
- src/itinerary_system/__init__.py: exported the progressive repair executor adapter APIs.
- tests/test_pipeline_runner.py: added progressive repair pipeline fixture and test proving child plan, diff, certificate, explanation, route, metrics, and strict-success export.
- docs/current/current_problem_manifest.md: updated CP-005/CP-006/CP-007 to record progressive repair executor wiring while leaving optimizer, benchmark, and notebook work open.
- docs/planning/current_problem_fix_phase_plans.md: updated Phase 4.0 status and missing-work list.
- docs/planning/travel_itinerary_repair_technical_specification.md: updated VERIFY/EXPLAIN/PIPE implementation status for progressive repair pipeline exports.
- `git status`: M .codex/code-edit-log.md
- `git status`: M CODEX_EDIT_LOG.md
- `git status`: M docs/current/current_problem_manifest.md
- `git status`: M docs/planning/current_problem_fix_phase_plans.md
- `git status`: M docs/planning/travel_itinerary_repair_technical_specification.md
- `git status`: M src/itinerary_system/__init__.py
- `git status`: M src/itinerary_system/routing/__init__.py
- `git status`: ?? docs/planning/context_aware_itinerary_repair_integrated_implementation_plan.md
- `git status`: ?? docs/slides/context_aware_itinerary_repair_professor_update.pptx
- `git status`: ?? docs/slides/context_aware_itinerary_repair_professor_update_polished.pptx
- `git status`: ?? src/itinerary_system/evaluation/
- `git status`: ?? src/itinerary_system/explanation/
- `git status`: ?? src/itinerary_system/pipeline_runner.py
- `git status`: ?? src/itinerary_system/repair/
- `git status`: ?? tests/evaluation/
- `git status`: ?? tests/explanation/
- `git status`: ?? tests/repair/
- `git status`: ?? tests/test_pipeline_runner.py

### What was found

- The REPAIR-005 controller already returned child plan, diff, evaluation record, attempts, planner runs, and diagnosis; pipeline work only needed translation into PipelineExecutionResult.
- Default independent evaluation needed planner-run evidence matching the generated child plan source_run_id, so the adapter creates a synthetic planner-run record for the accepted child before exporting the certificate.
- The first RED test failed with ImportError because build_progressive_repair_executor did not exist, then passed after implementation.

### Validation

- python -m pytest tests/test_pipeline_runner.py -k progressive_repair_executor: RED first with ImportError, then 1 passed after implementation.
- python -m pytest tests/test_pipeline_runner.py: 8 passed.
- python -m ruff check src/itinerary_system/pipeline_runner.py tests/test_pipeline_runner.py src/itinerary_system/__init__.py: passed.
- python -m ruff check src tests scripts: passed.
- python -m pytest tests/test_pipeline_runner.py tests/explanation tests/evaluation tests/repair: 45 passed.
- python -m pytest: 164 passed in 242.86s.
- python scripts/run_project_checks.py: passed; ruff, context snapshot pytest, and full pytest succeeded and refreshed results/quality/project_check_summary.json.
- git diff --check: passed with CRLF normalization warnings only.

### Conclusion

- Progressive repair can now be executed through run_research_pipeline() and exported as canonical run artifacts with child plan, diff, independent certificate, route matrix, grounded explanation, metrics, and dashboard record.

### Next steps

**Codex can proceed:**

- Proceed to real production optimizer executor wiring, then BENCH-001 deterministic disruption generation.
- Add an infeasible-repair pipeline test and strict failure behavior if the benchmark runner needs no-child repair attempts to fail pipeline runs.

**Human reflection:**

- The adapter preserves the existing controller boundary and avoids changing repair solver behavior; it only packages the controller outcome for canonical runs.
- The default certificate wrapper creates accepted-child planner-run evidence because current child plans use the repair master source_run_id rather than the lexicographic stage run IDs.

### Human action

- Review whether no-child repair outcomes should make strict pipeline runs fail immediately or remain diagnostic artifacts for benchmark analysis.

## EXPLAIN-001 evidence builder

- Status: completed
- Start local time: 2026-07-09 09:58:52 -05:00
- End local time: 2026-07-09 10:25:36 Central Daylight Time-0500
- Duration: 26m 03s

### Goal

- Proceed on EXPLAIN-001 by adding a deterministic builder that converts plan diff, certificate, and route-validation artifacts into structured explanation evidence.

### What changed

- src/itinerary_system/explanation/evidence.py: added ExplanationEvidenceBuilder, build_explanation_evidence, artifact normalization, and route-validation evidence refs
- src/itinerary_system/explanation/__init__.py: exported the EXPLAIN-001 builder API
- src/itinerary_system/__init__.py: re-exported the EXPLAIN-001 builder API from the package root
- tests/explanation/test_evidence_builder.py: added focused tests for builder-derived why and contrastive evidence
- docs/current/current_problem_manifest.md: updated CP-006 status for builder-backed EXPLAIN-001 evidence
- docs/planning/current_problem_fix_phase_plans.md: recorded builder-backed EXPLAIN-001 status and test coverage
- docs/planning/travel_itinerary_repair_technical_specification.md: updated EXPLAIN-001 implementation status, files, and public objects
- `git status`: M .codex/code-edit-log.md
- `git status`: M CODEX_EDIT_LOG.md
- `git status`: M docs/current/current_problem_manifest.md
- `git status`: M docs/planning/current_problem_fix_phase_plans.md
- `git status`: M docs/planning/travel_itinerary_repair_technical_specification.md
- `git status`: M src/itinerary_system/__init__.py
- `git status`: M src/itinerary_system/routing/__init__.py
- `git status`: ?? docs/planning/context_aware_itinerary_repair_integrated_implementation_plan.md
- `git status`: ?? docs/slides/context_aware_itinerary_repair_professor_update.pptx
- `git status`: ?? docs/slides/context_aware_itinerary_repair_professor_update_polished.pptx
- `git status`: ?? src/itinerary_system/evaluation/
- `git status`: ?? src/itinerary_system/explanation/
- `git status`: ?? src/itinerary_system/pipeline_runner.py
- `git status`: ?? src/itinerary_system/repair/
- `git status`: ?? tests/evaluation/
- `git status`: ?? tests/explanation/
- `git status`: ?? tests/repair/
- `git status`: ?? tests/test_pipeline_runner.py

### What was found

- EXPLAIN-001 evidence containers and validators already existed, but the documented ExplanationEvidenceBuilder surface was missing.
- Builder-derived contrastive evidence needed to cite route-validation records in addition to diff and certificate eligibility records.

### Validation

- Red: python -m pytest tests/explanation/test_evidence_builder.py failed with missing ExplanationEvidenceBuilder import.
- Red: added route evidence assertion and confirmed it failed before route-validation record support.
- Green: python -m pytest tests/explanation/test_evidence_builder.py passed, 2 tests.
- Focused: python -m pytest tests/explanation tests/evaluation passed, 17 tests.
- Pipeline: python -m pytest tests/test_pipeline_runner.py passed, 8 tests.
- Lint: python -m ruff check src tests scripts passed.
- Full: python -m pytest passed, 166 tests.
- Wrapper: python scripts/run_project_checks.py passed and wrote results/quality/project_check_summary.json.
- Whitespace: git diff --check exited 0 with CRLF normalization warnings only.

### Conclusion

- EXPLAIN-001 now includes a deterministic builder for artifact-backed why and contrastive evidence while preserving fail-closed claim validation.

### Next steps

**Codex can proceed:**

- Proceed with remaining Phase 4.0 work, especially real production optimizer executor wiring or BENCH-001 benchmark disruption generation.

**Human reflection:**

- This strengthens explanation evidence generation, but publication claims still depend on complete pipeline and benchmark artifacts rather than the explanation builder alone.

### Human action

- None

## PIPE-001 production optimizer executor adapter

- Status: completed
- Start local time: 2026-07-09 10:27:13 -05:00
- End local time: 2026-07-09 10:44:54 Central Daylight Time-0500
- Duration: 17m 19s

### Goal

- Continue Phase 4.0 by wiring the existing production optimizer callable into the canonical pipeline runner as a PipelineExecutionResult-producing generation executor.

### What changed

- src/itinerary_system/pipeline_runner.py: added build_production_generation_executor() and run_production_generation_executor() plus helpers for production output adaptation
- src/itinerary_system/__init__.py: exported the production generation executor API from the package root
- tests/test_pipeline_runner.py: added a red-green test proving production optimizer outputs become canonical pipeline artifacts
- docs/current/current_problem_manifest.md: updated CP-007 to show production optimizer executor wiring is implemented
- docs/planning/current_problem_fix_phase_plans.md: updated Phase 4.0 status and remaining open items
- docs/planning/travel_itinerary_repair_technical_specification.md: updated PIPE-001 implementation status and remaining gaps
- `git status`: M .codex/code-edit-log.md
- `git status`: M CODEX_EDIT_LOG.md
- `git status`: M docs/current/current_problem_manifest.md
- `git status`: M docs/planning/current_problem_fix_phase_plans.md
- `git status`: M docs/planning/travel_itinerary_repair_technical_specification.md
- `git status`: M src/itinerary_system/__init__.py
- `git status`: M src/itinerary_system/routing/__init__.py
- `git status`: ?? docs/planning/context_aware_itinerary_repair_integrated_implementation_plan.md
- `git status`: ?? docs/slides/context_aware_itinerary_repair_professor_update.pptx
- `git status`: ?? docs/slides/context_aware_itinerary_repair_professor_update_polished.pptx
- `git status`: ?? src/itinerary_system/evaluation/
- `git status`: ?? src/itinerary_system/explanation/
- `git status`: ?? src/itinerary_system/pipeline_runner.py
- `git status`: ?? src/itinerary_system/repair/
- `git status`: ?? tests/evaluation/
- `git status`: ?? tests/explanation/
- `git status`: ?? tests/repair/
- `git status`: ?? tests/test_pipeline_runner.py

### What was found

- Phase 4.0 still listed real production optimizer executor wiring as open, and experiment_runner.py already exposed run_configurable_blueprint_pipeline() as a callable production path.
- The production callable writes legacy method comparison and route stop outputs, so the narrow aligned adapter is to run it inside the immutable run directory and feed those outputs through the existing Phase 0 artifact translator.
- A plain Python import smoke test needs PYTHONPATH=src in this workspace; pytest config already supplies src on pythonpath.

### Validation

- Red: python -m pytest tests/test_pipeline_runner.py -k production_generation_executor failed with missing build_production_generation_executor import.
- Green: python -m pytest tests/test_pipeline_runner.py -k production_generation_executor passed, 1 selected test.
- Focused: python -m pytest tests/test_pipeline_runner.py passed, 9 tests.
- Lint: python -m ruff check src tests scripts passed.
- Smoke: PYTHONPATH=src python import of build_production_generation_executor and run_production_generation_executor passed.
- Full: python -m pytest passed, 167 tests.
- Wrapper: python scripts/run_project_checks.py passed and wrote results/quality/project_check_summary.json.
- Whitespace: git diff --check exited 0 with CRLF normalization warnings only.

### Conclusion

- The canonical pipeline can now wrap the existing production optimizer output path and export its Phase 0 artifacts under an immutable run directory.

### Next steps

**Codex can proceed:**

- Proceed to BENCH-001 deterministic six-disruption generators, then BENCH-002 paired benchmark runner and notebook migration.

**Human reflection:**

- This adapter validates the orchestration boundary with injected production data; a future live production run should still be exercised with real catalog/hotel inputs before claiming notebook replacement.

### Human action

- None

## BENCH-001 deterministic disruption generators

- Status: completed
- Start local time: 2026-07-09 10:46:20 -05:00
- End local time: 2026-07-09 11:03:46 Central Daylight Time-0500
- Duration: 17m 03s

### Goal

- Continue Phase 4.0 by implementing deterministic six-family benchmark disruption generation for repair requests.

### What changed

- src/itinerary_system/benchmark/__init__.py: added benchmark package exports for disruption generation
- src/itinerary_system/benchmark/disruptions.py: added DisruptionFamily, DisruptionScenario, DisruptionGenerator, generate_disruption_scenarios(), and generate_disruption_requests()
- src/itinerary_system/__init__.py: exported the BENCH-001 disruption generation API from the package root
- tests/benchmark/test_disruptions.py: added red-green tests for six families, deterministic IDs, observed/synthetic status, parent immutability, repair-ready constraints, and static manifest alignment
- data/benchmark/disruptions/bench_001_families.json: added static six-family benchmark disruption manifest
- docs/current/current_problem_manifest.md: updated CP-007 status for BENCH-001 implementation
- docs/planning/current_problem_fix_phase_plans.md: updated Phase 4.0 status and remaining open benchmark runner/notebook work
- docs/planning/travel_itinerary_repair_technical_specification.md: updated BENCH-001 implementation status
- `git status`: M .codex/code-edit-log.md
- `git status`: M CODEX_EDIT_LOG.md
- `git status`: M docs/current/current_problem_manifest.md
- `git status`: M docs/planning/current_problem_fix_phase_plans.md
- `git status`: M docs/planning/travel_itinerary_repair_technical_specification.md
- `git status`: M src/itinerary_system/__init__.py
- `git status`: M src/itinerary_system/routing/__init__.py
- `git status`: ?? data/benchmark/disruptions/
- `git status`: ?? docs/planning/context_aware_itinerary_repair_integrated_implementation_plan.md
- `git status`: ?? docs/slides/context_aware_itinerary_repair_professor_update.pptx
- `git status`: ?? docs/slides/context_aware_itinerary_repair_professor_update_polished.pptx
- `git status`: ?? src/itinerary_system/benchmark/
- `git status`: ?? src/itinerary_system/evaluation/
- `git status`: ?? src/itinerary_system/explanation/
- `git status`: ?? src/itinerary_system/pipeline_runner.py
- `git status`: ?? src/itinerary_system/repair/
- `git status`: ?? tests/benchmark/
- `git status`: ?? tests/evaluation/
- `git status`: ?? tests/explanation/
- `git status`: ?? tests/repair/
- `git status`: ?? tests/test_pipeline_runner.py

### What was found

- BENCH-001 was the next open Phase 4.0 slice after production executor wiring; docs require six disruption families, deterministic IDs, explicit observed/synthetic status, and no catalog snapshot mutation.
- Existing repair code accepts RepairRequest-like objects with request_id, confirmed_constraints, candidate_pois, and baseline_route, so the generator can emit the existing RepairRequest dataclass rather than a parallel request type.
- A static data/benchmark/disruptions manifest was missing even though the technical spec listed data/benchmark/disruptions/*.json.

### Validation

- Red: python -m pytest tests/benchmark/test_disruptions.py failed with ModuleNotFoundError for itinerary_system.benchmark.
- Red: after adding generator tests, manifest alignment test failed with FileNotFoundError for data/benchmark/disruptions/bench_001_families.json.
- Green: python -m pytest tests/benchmark/test_disruptions.py passed, 4 tests.
- Lint: python -m ruff check src tests scripts passed.
- Smoke: PYTHONPATH=src python import of DisruptionFamily, generate_disruption_requests, and generate_disruption_scenarios passed.
- Full: python -m pytest passed, 171 tests.
- Wrapper: python scripts/run_project_checks.py passed and wrote results/quality/project_check_summary.json.
- Whitespace: git diff --check exited 0 with CRLF normalization warnings only.

### Conclusion

- BENCH-001 now has deterministic six-family disruption generation into existing RepairRequest objects, plus a static family manifest and focused tests.

### Next steps

**Codex can proceed:**

- Proceed to BENCH-002 paired benchmark runner, no-leakage split checks, and long-form metrics export.

**Human reflection:**

- Generated disruptions are benchmark-ready request inputs, not completed benchmark results; publication claims still require paired runner execution, route validation, and metrics.

### Human action

- None

## BENCH-002 paired benchmark runner

- Status: completed
- Start local time: 2026-07-09 11:10:39 -05:00
- End local time: 2026-07-09 11:38:01 CDT-0500
- Duration: Not recorded

### Goal

- Proceed with the next future-plan slice by implementing BENCH-002 paired benchmark execution, split leakage validation, and long-form metric export without changing solver behavior.

### What changed

- src/itinerary_system/benchmark/runner.py: added BenchmarkMethodAdapter, BenchmarkRunRecord, BenchmarkResult, and run_benchmark_suite() for paired injected method execution over frozen disruption scenarios.
- src/itinerary_system/benchmark/splits.py: added parent-plan/disruption-family split keys, deterministic split assignment, and hard leakage rejection.
- src/itinerary_system/benchmark/metrics.py: added preservation, quality, computation, certificate, and explanation metric extraction from method results.
- src/itinerary_system/benchmark/__init__.py: exported BENCH-002 runner, split, and metric APIs.
- src/itinerary_system/__init__.py: surfaced BENCH-002 public APIs at the root package level.
- tests/benchmark/test_no_leakage.py: added red-first coverage for split leakage rejection, deterministic grouping, paired identical inputs, metrics JSONL, and manifest export.
- docs/current/current_problem_manifest.md: updated CP-007 status for BENCH-002 while keeping notebook and full matrix work open.
- docs/planning/current_problem_fix_phase_plans.md: updated Phase 4.0 status, run_benchmark_suite signature, and BENCH-002 validation checklist.
- docs/planning/travel_itinerary_repair_technical_specification.md: added BENCH-002 implementation status and limited publication-readiness caveat.
- `git status`: M .codex/code-edit-log.md
- `git status`: M CODEX_EDIT_LOG.md
- `git status`: M docs/current/current_problem_manifest.md
- `git status`: M docs/planning/current_problem_fix_phase_plans.md
- `git status`: M docs/planning/travel_itinerary_repair_technical_specification.md
- `git status`: M src/itinerary_system/__init__.py
- `git status`: M src/itinerary_system/routing/__init__.py
- `git status`: ?? data/benchmark/disruptions/
- `git status`: ?? docs/planning/context_aware_itinerary_repair_integrated_implementation_plan.md
- `git status`: ?? docs/slides/context_aware_itinerary_repair_professor_update.pptx
- `git status`: ?? docs/slides/context_aware_itinerary_repair_professor_update_polished.pptx
- `git status`: ?? src/itinerary_system/benchmark/
- `git status`: ?? src/itinerary_system/evaluation/
- `git status`: ?? src/itinerary_system/explanation/
- `git status`: ?? src/itinerary_system/pipeline_runner.py
- `git status`: ?? src/itinerary_system/repair/
- `git status`: ?? tests/benchmark/
- `git status`: ?? tests/evaluation/
- `git status`: ?? tests/explanation/
- `git status`: ?? tests/repair/
- `git status`: ?? tests/test_pipeline_runner.py

### What was found

- EXPLAIN-001 was already implemented and verified before this continuation; the next aligned work was BENCH-002.
- Existing benchmark package only contained BENCH-001 disruption generation, so the red test failed on the missing BenchmarkLeakageError public API before implementation.
- Full pytest needs more than five minutes in this workspace; the first 300-second run timed out, and the rerun with a longer timeout passed.

### Validation

- python -m pytest tests/benchmark/test_no_leakage.py: red run failed on missing BenchmarkLeakageError import, then passed 3 tests after implementation.
- python -m pytest tests/benchmark: passed 7 tests.
- python -m ruff check src/itinerary_system/benchmark tests/benchmark: passed after removing one unused import.
- python -m ruff check src tests scripts: passed.
- python -m pytest: passed 174 tests in 386.99 seconds after the initial 300-second timeout.
- python scripts/run_project_checks.py: passed ruff, context snapshot pytest, and full pytest; wrote results/quality/project_check_summary.json.
- git diff --check: exit 0 with only LF-to-CRLF working-copy warnings.

### Conclusion

- BENCH-002 now has a deterministic paired benchmark harness that enforces split isolation and exports long-form result rows plus a manifest; notebook migration and complete provider-backed benchmark matrices remain future work.

### Next steps

**Codex can proceed:**

- Proceed to the next Phase 4.0 slice: connect the BENCH-002 runner to concrete pipeline/provider method adapters or begin notebook migration, depending on which current doc gate is selected next.

**Human reflection:**

- The runner intentionally accepts injected method adapters so it can compare baselines without changing solver internals; complete benchmark publication claims still require real method outputs over canonical parent/profile matrices.

### Human action

- Review the BENCH-002 API shape, especially whether benchmark output should remain JSONL-only for now or add CSV export before notebook migration.

## BENCH-002 pipeline method adapter

- Status: completed
- Start local time: 2026-07-09 11:39:29 -05:00
- End local time: 2026-07-09 11:57:33 CDT-0500
- Duration: Not recorded

### Goal

- Continue Phase 4.0 future-plan work by adding a pipeline-backed benchmark method adapter so BENCH-002 can execute methods through run_research_pipeline and reload emitted artifacts for metrics.

### What changed

- src/itinerary_system/benchmark/methods.py: added build_pipeline_benchmark_method_adapter() and pipeline_run_to_benchmark_result() to run benchmark methods through run_research_pipeline and load manifest-listed artifacts.
- src/itinerary_system/benchmark/runner.py: added per-method output_dir, manifest_path, and metrics_path fields to BenchmarkRunRecord so benchmark rows can point back to pipeline evidence.
- src/itinerary_system/benchmark/__init__.py: exported pipeline benchmark method adapter APIs.
- src/itinerary_system/__init__.py: surfaced pipeline benchmark adapter APIs from the root package.
- tests/benchmark/test_method_adapters.py: added red-first coverage for scenario-bound pipeline execution, artifact reload, benchmark metric extraction, and manifest/metrics path preservation.
- docs/current/current_problem_manifest.md: updated CP-007 to include the pipeline benchmark adapter while keeping notebook and full matrix work open.
- docs/planning/current_problem_fix_phase_plans.md: updated Phase 4.0 status and file list for the adapter slice.
- docs/planning/travel_itinerary_repair_technical_specification.md: updated BENCH-002 status and file list for methods.py and the adapter test.
- `git status`: M .codex/code-edit-log.md
- `git status`: M CODEX_EDIT_LOG.md
- `git status`: M docs/current/current_problem_manifest.md
- `git status`: M docs/planning/current_problem_fix_phase_plans.md
- `git status`: M docs/planning/travel_itinerary_repair_technical_specification.md
- `git status`: M src/itinerary_system/__init__.py
- `git status`: M src/itinerary_system/routing/__init__.py
- `git status`: ?? data/benchmark/disruptions/
- `git status`: ?? docs/planning/context_aware_itinerary_repair_integrated_implementation_plan.md
- `git status`: ?? docs/slides/context_aware_itinerary_repair_professor_update.pptx
- `git status`: ?? docs/slides/context_aware_itinerary_repair_professor_update_polished.pptx
- `git status`: ?? src/itinerary_system/benchmark/
- `git status`: ?? src/itinerary_system/evaluation/
- `git status`: ?? src/itinerary_system/explanation/
- `git status`: ?? src/itinerary_system/pipeline_runner.py
- `git status`: ?? src/itinerary_system/repair/
- `git status`: ?? tests/benchmark/
- `git status`: ?? tests/evaluation/
- `git status`: ?? tests/explanation/
- `git status`: ?? tests/repair/
- `git status`: ?? tests/test_pipeline_runner.py

### What was found

- BENCH-002 had a generic injected-method runner but no reusable bridge from benchmark scenarios to run_research_pipeline outputs.
- The first adapter test failed on missing build_pipeline_benchmark_method_adapter, then revealed a Windows path-length risk from verbose benchmark run IDs; the implementation now uses compact content-hashed run IDs.
- Per-method pipeline artifact paths were not preserved in BenchmarkRunRecord, so manifest_path and metrics_path were added for traceability.

### Validation

- python -m pytest tests/benchmark/test_method_adapters.py: red import failure first, then passed 1 test after implementation.
- python -m pytest tests/benchmark: passed 8 tests.
- python -m ruff check src/itinerary_system/benchmark tests/benchmark: passed after import ordering cleanup.
- python -m ruff check src tests scripts: passed.
- python -m pytest: passed 175 tests in 258.33 seconds.
- python scripts/run_project_checks.py: passed ruff, context snapshot pytest, and full pytest; refreshed results/quality/project_check_summary.json.
- git diff --check: exit 0 with only LF-to-CRLF working-copy warnings.

### Conclusion

- Benchmark methods can now execute through the authoritative package pipeline and feed emitted plans, diffs, evaluations, explanations, and metrics back into the BENCH-002 long-form result table.

### Next steps

**Codex can proceed:**

- Proceed to the next Phase 4.0 slice: either add concrete configured benchmark method factories for the existing production/progressive executors, or start NOTEBOOK-001 thin notebook migration after inspecting the notebook structure.

**Human reflection:**

- The adapter keeps benchmark comparison injectable and does not change solver behavior; publication claims still depend on provider-backed full matrices across canonical parents, profiles, and methods.

### Human action

- Review whether the next step should prioritize concrete benchmark method factory presets or notebook migration.


## Permission-aware counterfactual clarification extension

### Task

Add a disabled-by-default, deterministic research extension for permission-aware counterfactual repair and consequence-grounded clarification without changing the E1-E4 publication pipeline.

### Files changed

- Added `src/itinerary_system/interaction/` with typed interaction records, frozen/rule-based semantic providers, allow-listed patch compilation, permission policy, isolated and frozen probe executors, consequence analysis, critical-tradeoff selection, deterministic clarification policy, controller, immutable interaction pipeline entry point, and CLI.
- Added `scripts/run_permission_aware_repair.py` as the non-default, non-interactive CLI wrapper.
- Updated `src/itinerary_system/__init__.py` with public interaction exports.
- Added `tests/interaction/test_patch_compiler.py`, `test_permission_policy.py`, `test_probe_executor.py`, `test_consequence_and_clarification.py`, and `test_pipeline_integration.py`.
- Added `docs/planning/permission_aware_counterfactual_clarification_plan.md`.
- Added only a labeled design/scaffold note to the deferred E5 section of `docs/planning/current_execution_plan.md`; E3, E4, the immediate queue, and E5 completion status were not changed.

### Existing components reused

- Reused `OwnedConstraint`, ownership/relaxation enums, `PlanArtifactV2`, `PlanDiff`, `RepairRequest`, `RepairRadius`, `ProgressiveRepairController`, append-only `PlanRepository`, `RouteMatrix`, independent evaluation/certificate artifacts, explanation evidence, and `run_research_pipeline()` for authorized continuation runs.
- Did not add a second ownership model, plan model, diff model, repair solver, evaluator, certificate, route matrix, or accepted-plan repository.

### Tests run

- `python -m ruff check src tests scripts` -> passed (`All checks passed!`).
- `python -m pytest tests/benchmark/test_publication_method_factory.py -q` -> passed (`1 passed in 3.23s`).
- `python -m pytest` -> passed (`238 passed in 71.89s`).
- `python -m pytest tests/interaction -q` during focused development -> passed (`17 passed in 2.39s`).
- `python scripts/run_permission_aware_repair.py --help` -> passed and displayed frozen-fixture, continuation, permission-decision, route-cache, and question-budget options.
- `git diff --check` -> passed; only existing LF-to-CRLF warnings were reported.

### Artifacts generated

- Focused tests generated immutable interaction and authorized-continuation run fixtures only under pytest temporary directories.
- No real E3 result, production interaction run, participant-study artifact, or publication claim was generated.

### Default E3 behavior

- Unchanged. `InteractionOptions.clarification_mode` defaults to `disabled`, whose narrow entry point delegates directly to the existing pipeline without interaction directories or manifest fields.
- The current four required E3 method IDs and method provenance remain unchanged.
- The synthetic four-method publication-mode integration test passes.
- No E2 route matrix, frozen input hash, evaluator-owned ranking metric, or current benchmark factory was modified by this task.

### Unresolved limitations

- No live LLM client and no reinforcement-learning policy were implemented.
- The CLI uses frozen probe fixtures for deterministic replay. An actual authorized continuation requires a supplied frozen route cache and the existing progressive repair adapter.
- Probe time budgets are recorded and passed into typed requests; backend-specific hard interruption remains limited by current solver/controller support.
- Numerical burden fields remain unknown rather than zero when parent and hypothetical artifacts lack paired evidence.
- No participant study, calibrated semantic probability, novelty claim, or E5 completion claim exists.
- The real E3 paired benchmark remains incomplete and was not rerun; prior scalability limits on the full-reoptimization baseline remain outside this task.

### Planning truth found stale

- The request named `docs/current/current_execution_plan.md`, but the repository authority is `docs/planning/current_execution_plan.md`.
- Older detailed phase plans still contain proposed or missing-status language that does not match current implemented components; the current execution plan and problem manifest were treated as authoritative.

## E3 real evidence diagnostic and correctness fixes - 2026-07-22

### Outcome

- Expanded the conservative E3 route universe to immutable bundle `route_bundle_a60c80047098a3b6` and matrix `route_matrix_68ab535465b06808`; 223/223 cells are road-and-snap validated with no fallback.
- Completed immutable 24-row diagnostic `e3-real-production-20260722-expanded-v13-budget720-cap1`.
- Closeout: 8 ranking-eligible rows, 16 retained failures, complete method provenance and route lineage, `evidence_complete: true`, and `publication_ready: true`.
- This is not a complete four-method comparison: both exact methods refused every scenario at diagnostic cap 1, and road-closure plus reduced-driving-tolerance remain infeasible for both non-exact methods.

### Correctness changes

- Added exact candidate-space lower-bound preflight and fail-fast cap refusal.
- Added complete E3 route-coverage preflight, frozen catalog snap audit, and route-evidence expansion CLIs.
- Corrected label-keyed parent day assignments to canonical POI IDs without mutating frozen parent artifacts.
- Corrected empty lodging validation and made independent evaluator lodging requirements explicit and parent-derived.
- Preserved canonical progressive method provenance even when inner lexicographic runs exist.
- Corrected publication route-evidence semantics for failed runs without output plans while retaining output-certificate requirements for failed strict plans.
- Parameterized normal disruption travel budgets from the benchmark daily limit; the reduced-driving scenario retains its explicit 180-minute tolerance.

### Validation

- `python -m ruff check .` -> passed.
- `python -m pytest -q` -> 251 passed.
- Real v13 process exited cleanly; closeout and all 24 immutable pipeline run directories were read back.
- OSRM remained bound to `127.0.0.1:5000`; no container, route bundle, or prior immutable run was removed.

## 2026-07-25: Exact Baselines Optimization & E3 Benchmark Execution v14

### Summary

- Implemented zero-overhead preflight assignment duration pruning ($\sum \text{durations} > \text{max\_day\_minutes}$) and budget pruning in `src/itinerary_system/repair/exact_baselines.py`.
- Preserved strict candidate enumeration cap preflight checking (`complete_candidate_limit_exceeded`) without generating false optimality claims.
- Diagnosed physical infeasibility root causes for `road_closure` (no alternate unclosed intercity route in the frozen matrix for Day 4) and `reduced_driving_tolerance` (Day 7 travel time > 180 min threshold).
- Executed `scripts/run_e3_publication_benchmark.py` with candidate budget 50,000 (`e3_real_california_20260725_optimized_v14_cap50000`). Closeout reports `publication_ready: true`, 8 ranking-eligible PASSED rows, 16 retained failure rows.

### Validation

- `python scripts/run_project_checks.py` -> PASS ruff (0.9s), PASS context_snapshot_pytest (2.0s), PASS full_pytest (34.7s).
- `python -m ruff check .` -> All checks passed.
- `python -m pytest -q` -> 251 passed in 33.78s.

## 2026-07-29: Phase A Plan Reconciliation and E3.UX Definition

### Phase ID

- Phase A — repository and plan reconciliation.
- New planned phase: E3.UX — Artifact-Grounded Product Dashboard Reframe.

### Files changed

- Added `docs/current/current_repository_truth_2026_07.md`.
- Added `docs/current/ui_skill_application_record.md`.
- Added `docs/planning/e3_user_facing_dashboard_reframe_phase_plan.md`.
- Added `docs/reports/current_plan_reconciliation_report.md`.
- Updated `docs/planning/current_execution_plan.md`.
- Updated `docs/planning/research_pipeline_and_gate_map.md`.
- Reconciled `docs/planning/e3_exact_baseline_scalability_phase_plan.md` from
  stale `ready`/v13 wording to `blocked` on D1 with current v14 evidence and the
  verified E3.2 handoff.
- Added planned CP-009 to `docs/current/current_problem_manifest.md`; CP-009 is
  not resolved.

### Tests run

- Focused subsystem matrix using the configured shared pytest temp root:
  `102 passed`, then 31 setup errors caused by `WinError 5` while scanning
  `.codex_tmp_pytest/pytest/pytest-of-Ztang_Yit_Xiaang`.
- The same focused matrix with isolated
  `--basetemp tmp_test/codex_phasea_pytest_20260729_019faf6d`:
  `133 passed in 16.23s`.
- The matrix covered pipeline entry/run/artifact lineage, plans/diff/repository,
  route matrix, progressive and exact repair, independent evaluation,
  explanations, benchmark/disruption contracts, interaction, dashboard
  contracts, and Folium renderer/selector/core boundaries.

### Artifact paths inspected

- `tmp_test/research_pipeline_raw/e3-real-production-20260725-optimized-v14-cap50000/closeout.json`
- `tmp_test/research_pipeline_raw/e3-real-production-20260725-optimized-v14-cap50000/benchmark/manifest.json`
- `tmp_test/research_pipeline_raw/e3-real-production-20260725-optimized-v14-cap50000/benchmark/metrics/benchmark_metrics.jsonl`
- `tmp_test/research_pipeline_raw/e3-real-production-20260725-optimized-v14-cap50000/inputs/manifest.json`
- `tmp_test/research_pipeline_raw/e3-real-production-20260725-optimized-v14-cap50000/inputs/route_coverage_preflight.json`

The v14 root was read only and remains immutable.

### Browser widths

- No new browser run occurred in Phase A because no UI implementation changed.
- The E3.C4 closeout remains pending at 390px.
- E3.UX4 now requires 1440, 1024, 768, 430, 390, and 360px, plus the
  Folium selector's natural expanded width.

### Remaining limitations

- E3.C4 is `implemented`, not `verified`; the 430px open selector still exceeds
  the 390px viewport and the final mobile dashboard assertion remains pending.
- E3.1 is `blocked` on D1; all 12 exact v14 cells remain cap-refused at 50,000.
- E3.3 and E4 remain `blocked`.
- E3.UX and E3.UX0-E3.UX4 remain `planned`; no product dashboard code exists.
- E3.UX5 and E5 remain `deferred` and disabled by default.

### Exact gate status

- `verified`: E1, E2, E3.0, E3.M, E3.2, E3.C1, E3.C2, E3.C3.
- `in-progress`: E3.C.
- `implemented`: E3.C4.
- `planned`: E3.UX, E3.UX0, E3.UX1, E3.UX2, E3.UX3, E3.UX4, CP-009.
- `blocked`: E3.1, E3.3, E4.
- `deferred`: E3.UX5, E5.

## 2026-07-29: Phase B Code Mastery Documentation

### Phase ID

- Phase B — code mastery documentation.

### Files changed

- Added `docs/code_mastery/00_start_here.md` through
  `docs/code_mastery/15_glossary.md`.
- Added 12 repository-specific Mermaid diagrams under
  `docs/code_mastery/diagrams/`.
- Added `docs/reports/code_mastery_report.md`.

### Verification

- `python -m pytest --basetemp tmp_test\codex_phaseb_collect_20260729_019faf6d --collect-only -q`
  -> `288 tests collected in 3.58s`.
- The package references current modules/tests and preserves E3.1/E3.3/E4,
  E3.C4, E3.UX, and E5 status boundaries.
- No runtime source, generated dashboard, legacy parity oracle, or immutable
  E3 artifact was changed.

### Remaining limitation

- The testing guide records the current shared pytest temp-root `WinError 5`
  and the verified unique `--basetemp` workaround.
- Phase C browser evidence has not yet run; E3.C4 remains `implemented`.
## 2026-07-29 — E3.C4 verified mobile closeout

### Phase

- E3.C4 — renderer/UI controller decomposition

### Files changed

- `src/itinerary_system/blueprint_route_selector.py`: added the mobile-only
  selector containment rule.
- `tests/test_blueprint_route_selector.py`: added the responsive CSS contract.
- `tests/test_blueprint_renderer_parity.py`: froze the reviewed E3.C4
  full-HTML signature while retaining all data/Leaflet assertions.
- `scripts/run_project_checks.py`: isolated the full pytest subprocess under a
  unique repository-local base temp directory.
- Updated current truth, manifest, execution/gate plans, code-size audit, UI
  skill record, E3.C4 phase plan, and the closeout report.

### Tests and artifacts

- Focused E3.C4: 23 passed.
- Benchmark/evaluator/interaction regressions: 81 passed.
- Ruff: passed.
- Dashboard export validator: passed.
- Project checks: passed; full pytest 289 passed.
- Same-worktree Folium render:
  `tmp_test/codex_e3c4_focused_green_20260729/test_package_renderer_matches_0/figures/package_renderer.html`.
- Reviewed normalized Folium HTML SHA-256:
  `a06583549a135688e62d663ff5c6197074e96f3a0bad57d5cb791f37273fc2bb`.
- Browser widths: 1440, 768, 520, and 390px for Folium and modular dashboards.
- Screenshots:
  `docs/reports/e3c4_folium_mobile_before.png`,
  `docs/reports/e3c4_folium_mobile_after.png`,
  `docs/reports/e3c4_modular_mobile.png`, and
  `docs/reports/e3c4_modular_desktop.png`.

### Status and limitations

- E3.C4 is `verified`.
- E3.UX0–E3.UX4 are unblocked on the additive `dashboard_product/` path.
- The historical generated Folium file remains the before artifact; the next
  normal export will contain the verified source rule.
- D1/E3.1/E3.3/E4, E5, and E3.UX5 are unchanged.

## 2026-07-29 — E3.UX0–E3.UX4 verified product-dashboard closeout

### Phase IDs

- E3.UX0 — current UI and artifact audit.
- E3.UX1 — product information architecture and design system.
- E3.UX2 — read-only artifact-grounded customer dashboard.
- E3.UX3 — research/evidence and comparison views.
- E3.UX4 — responsive, accessibility, integrity, and browser verification.

### Code and tests

- Added package-owned product contracts, adapter, shared view models, semantic
  assets/controllers, and a non-overwritable renderer under
  `src/itinerary_system/product_dashboard_*.py`.
- Added export, screenshot-registration, and validation scripts.
- Added 26 focused tests under `tests/product_dashboard/` for canonical
  loading, lineage/hash/path/finite-value enforcement, exact-cap versus
  infeasibility truth, permission provenance, customer/research view models,
  rendering, security, host-path suppression, non-overwrite, screenshot
  hashes, and UX5 exclusion.

### Artifacts and reports

- Canonical source run:
  `tmp_test/research_pipeline_raw/e3-real-production-20260725-optimized-v14-cap50000/pipeline_runs/benchmark_158cf6d48be8`.
- Final derived product run:
  `runs/e3ux-weather-repair-demo-v6`.
- Product path:
  `runs/e3ux-weather-repair-demo-v6/dashboard_product/`.
- Manifest schema `product-dashboard-manifest-v1`; product version `1.0.0`.
- Added E3.UX0 audit, E3.UX1 design contract, desktop/tablet/mobile Mermaid
  wireframes, reframe/testing/browser/accessibility/artifact-integrity reports,
  and updated code-mastery/current planning documents.

### Verification

- Product focused: 26 passed.
- Legacy E3.C focused: 23 passed in 13.39s.
- Legacy regression matrix: 81 passed in 13.79s.
- Ruff across `src tests scripts`: passed.
- Legacy dashboard validator: passed.
- Product dashboard validator and source/asset/screenshot hash readback: passed.
- Project checks: Ruff, context snapshot, and full pytest passed.
- Full pytest: 315 passed in 46.95s.
- Markdown local-link and mojibake checks: passed.
- Browser widths: 1440, 1024, 768, 430, 390, and 360px.
- Browser result: no document overflow, clipped control, product console issue,
  incomplete image, or map initialization failure; customer/research, day/map,
  evidence, failure, certificate, hash, focus, touch-target, and contrast flows
  pass.
- Screenshots and SHA-256 values are registered in the product manifest.

### Corrections found during browser/accessibility verification

- Added product-owned essential Leaflet layout/control CSS so a missing remote
  Leaflet stylesheet cannot expand panes into normal document flow.
- Darkened the warning token from `#a65a10` to `#854306`; the repeated
  computed-style audit passed 196 visible leaf-text nodes with zero WCAG AA
  threshold failures.

### Exact gate status and limitations

- `verified`: E3.UX, E3.UX0, E3.UX1, E3.UX2, E3.UX3, E3.UX4, and CP-009.
- `blocked` and unchanged: D1/E3.1, E3.3, E4.
- `deferred` and unchanged: E3.UX5 and E5.
- The UI is read-only. It supplies no interaction, user-study, exact-completion,
  four-method superiority, or publication claim.
- Visible focus and native keyboard semantics were verified. The browser
  backend did not synthesize locator-level Enter activation; this limitation is
  disclosed in the accessibility report rather than overstated.

## 2026-07-29 — Product dashboard discoverability launcher

### Phase

- E3.UX4 follow-up — discoverability without legacy replacement.

### Changes

- Added `OPEN_PRODUCT_DASHBOARD.cmd` at the repository root. Double-clicking it
  validates the v6 entry path, reuses `scripts/serve_dashboard.py`, and opens
  the product page over localhost.
- Added a prominent README section that identifies the redesigned launcher and
  explicitly distinguishes the existing legacy-dashboard command.
- Updated the product reframe report with the launcher boundary.

### Compatibility

- The legacy server default, Folium map, modular dashboard, hashes, and
  optimizer/evaluator/benchmark semantics are unchanged.
- The authoritative product artifact remains
  `runs/e3ux-weather-repair-demo-v6/dashboard_product/`.

## 2026-08-05 — W4 Copilot provider and transcript implementation handoffs

### Phase and status boundary

- W4 — Copilot provider, prompt engineering, and local transcript persistence.
- W4 is `in-progress`; G4 has not been promoted or verified.
- CP-010 and every research status remain unchanged. W5 acceptance and Keep
  original transactions remain disabled.

### Multi-agent ownership and handoffs

- `EXECUTION_ORCHESTRATOR` froze the shared `models.py` contracts and retained
  ownership of cross-module integration, status integrity, and final review.
- `w4_copilot_contract` exclusively owned `product_app/copilot.py` and
  `tests/product_app/test_copilot_contracts.py`.
- `w4_prompt_engineering` exclusively owned `product_app/copilot_prompt.py`,
  `tests/product_app/test_copilot_prompt.py`, the 72-case prompt fixture, and
  `scripts/evaluate_copilot_prompt.py`.
- `w4_conversations` exclusively owned `product_app/conversations.py`, the
  conversation-only persistence additions, and
  `tests/product_app/test_conversations.py`.
- `w4_browser_journey` owns the independent browser-journey evidence boundary;
  `w4_privacy_tests` owns privacy/security regression evidence; and
  `w4_docs_handoff` owns only `README.md`, `docs/README.md`, and this log entry.
- Shared paths were not edited by workers without an orchestrator handoff;
  implementation agents do not provide their own independent G4 sign-off.

### Implementation evidence present at this documentation checkpoint

- The deterministic adapter remains the default. OpenAI requires explicit
  `PRODUCT_COPILOT_ADAPTER=openai`; there is no silent provider fallback.
- The repository-root `.env.local` loader accepts only an allow-list, does not
  execute shell syntax, does not overwrite process environment values, and
  reports stable secret-free errors. `.env.local` is Git-ignored.
- The provider boundary targets `gpt-5.6-terra`, uses strict typed Responses
  output with no tools and `store=false`, bounds timeout/history/output, and
  permits one retry only for transient failures.
- Local conversation persistence, 30-day expiry, authenticated read, delete
  current, and confirmed delete-all paths are present. Transcript data is kept
  out of health, logs, evidence downloads, and immutable research artifacts.
- The browser identifies Deterministic demo versus OpenAI Copilot, displays the
  outbound-data disclosure, restores local turns, exposes delete controls, and
  keeps provider proposals advisory.

These statements describe inspected code at an in-progress checkpoint. They
are not a G4 verdict. Focused, product-wide, browser, live-provider, and eight
independent audit gates still determine whether W4 may become verified.

## 2026-08-07 - W4R.0 status and corrective interaction contract start

### Scope and status

- Started W4R at `in-progress` with planning/status production edits only.
- Preserved W4 as `implemented` offline, G4R as `planned` and not verified,
  G4 as `blocked`, and W5-W8 as `planned`.
- Recorded the enforced product order
  `W4 -> W4R -> G4R -> G4 -> W5`.
- Preserved the independent research order
  `D1 -> E3.1 -> E3.3 -> E4`; no research status changed.

### Defects recorded

- Direct map editing lacks complete catalog-stop drag, custom-waypoint drag,
  empty-map confirmation, runtime snap, and route-segment editing workflows.
- Every ordered adjacent itinerary pair must render validated route geometry or
  an explicit unvalidated gap; a silently missing line is not accepted.
- Compare requires exact option selection, active-preview support, instance-owned
  map controllers, and exact Evidence resolution.
- Accept and Keep remain fail-closed until the W5 repository/ledger/pointer
  transaction exists; W4R must present that boundary truthfully.

### Files changed

- `docs/current/current_problem_manifest.md`
- `docs/planning/current_execution_plan.md`
- `docs/planning/research_pipeline_and_gate_map.md`
- `docs/planning/itinerary_repair_copilot_implementation_plan.md`
- `CODEX_EDIT_LOG.md`

No product source, test source, immutable run, frozen dashboard, or research
artifact was changed by this status-edit package.

### Verification

- Focused local Markdown link, balanced-fence, and whitespace scan: passed for
  the four edited authority files.
- Status consistency scan: passed; required W4R/G4R and independent research
  sequence are present, and no old direct W4-to-W5 sequence remains.
- `python -m pytest tests\plans\test_repository.py -q --basetemp <unique-temp>`
  -> `4 passed in 0.96s`.
- `git diff --check -- <five assigned paths>` -> passed; Git reported only the
  existing LF-to-CRLF working-copy warnings.

### Remaining boundary

- This documentation establishes W4R entry and scope only. It is not evidence
  that W4R code is implemented or that G4R/G4 passed.
- G4 still requires newly authorized post-fix live-provider evidence.
- W5 decision mutations remain disabled.

## 2026-08-08 - W4R route-integrity and direct-interaction checkpoint

### Status boundary

- W4R remains `in-progress`; G4R is not verified and G4 remains blocked.
- W5 acceptance and Keep-original mutations remain disabled.
- Product work did not advance D1/E3.1, E3.3, E4, E3.UX5, or E5.

### Implemented checkpoint

- Added `product-geography-v2`: every required itinerary connection is either
  a road-validated LineString or an explicit null-geometry gap. The California
  demo exposes three hash-distinct plans, each with 17 route-path nodes and
  16/16 validated legs.
- Added loopback-only OSRM readiness and authenticated, revision-checked,
  non-mutating snap preview with normal, warning/confirmation, and rejected
  distance states.
- Replaced the singleton map lifecycle with an instance-owned MapLibre
  controller and added Select/Edit modes, stop and exact route-leg selection,
  keyboard route alternatives, exploratory raw-point semantics, and explicit
  route-gap text.
- Corrected route selection to use immutable `route_leg_id` rather than the
  internal GeoJSON feature ID.
- Corrected evaluated preview geography. A fresh child may reuse registered
  route geometry only when its route-relevant sequence, stops, days,
  coordinates, and lodging projection exactly match the source child; lineage
  is rewritten to the new child ID/hash. Otherwise geography remains
  unavailable.
- Corrected route-feedback targeting and certificate messaging. The live
  route-feedback -> Preview -> refresh journey now retains one typed draft, an
  eligible/PASSED proposal, a current certificate, and a visible map canvas.

### Verification

- Product/PlanRepository regression: 463 passed, one known Starlette
  TestClient deprecation warning.
- Focused preview/controller/frontend regression after live fixes: 40 passed.
- Focused route-selection regression: 12 passed.
- JavaScript syntax, Ruff, and diff checks passed for the changed slice.
- Independent route-artifact comparison found 48/48 required legs, zero gaps,
  and zero plan/hash/matrix/geometry mismatches.
- Independent security audit passed authentication, revision, Host, Origin,
  schema/body-size, cache, CSP, and redaction boundaries.
- Root live browser regression confirmed one map canvas immediately after
  evaluated Preview and after refresh, current PASSED certificate text, and no
  console warnings/errors.

### Independent blocking findings

- G4R remains blocked: exploratory placement cannot append an executable
  custom waypoint; catalog/custom drag, POI discovery, richer stop/timeline
  editing, and route-policy editing are incomplete.
- Compare still requires in-place exact option selection and baseline/option
  maps; mobile/tablet workflows and accessibility sizing/hierarchy remain
  incomplete. Separate remediation batches are in progress.
- Accept/Keep remain intentionally closed until the W5 repository, decision
  journal, and atomic-pointer transaction exists.

### D1 research audit

- Frozen v14 remains unchanged. All 12 exact cells still refused at the
  50,000 cap.
- A new prerequisite blocker, D1-S0 benchmark semantic validity, was recorded:
  Phase 0 dropped source lodging assignments; synthetic hotel replacement IDs
  lack catalog/access/route evidence; passing non-exact hotel rows were
  zero-edit no-ops; and lodging decisions are not coupled to route anchors.
- `gurobi_exact_v2.py` is currently a representation/index scaffold and does
  not optimize. D1-S0 must pass before a representative exact solve, followed
  by D1-A or D1-B and a new immutable rerun. No research status was promoted.

## 2026-08-08 - W4R Compare, responsive, and affected-route remediation

### Implemented

- Added exact in-place Compare selection with separate baseline and option
  MapLibre controllers, exact plan/content-hash/evidence resolution, pane-local
  text and failure states, a mobile Baseline/Option toggle, and no silent first
  alternative fallback. Selecting an option no longer navigates to Evidence.
- Fixed the initial Compare baseline regression so the accepted map is created
  before an option is selected. A subsequent blank-pane audit led to deferred
  controller startup, explicit map sizing, post-ready resize, and camera
  synchronization only after both maps finish their initial fitted view.
- Improved responsive and accessibility contracts: 44px critical targets,
  readable support text, compact scrollable mobile inspector/draft sheets,
  reachable toolbar scrolling, wrapping technical IDs, and removal of the
  whole-workspace `aria-live` region.
- Extended authenticated snap preview for an exact accepted-plan route leg.
  The server resolves authoritative endpoints and obtains predecessor-to-snap
  and snap-to-successor road legs from loopback OSRM while leaving the session
  revision, draft, parent, and workspace unchanged.
- Hardened affected-route evidence after an independent adversarial failure.
  Preview legs now require the exact runtime-OSRM provenance contract, a
  64-character lowercase SHA-256 query hash, finite positive metrics, bounded
  finite geometry, timezone-aware retrieval time, and finite nonnegative snap
  diagnostics.

### Verification

- Product-app plus PlanRepository regression after Compare/mobile integration:
  `482 passed`, with one known Starlette TestClient/httpx deprecation warning.
- Final independent affected-route/provenance re-audit: `104 passed`, with the
  same known warning. Adversarial null-offset timestamps, untrusted/invalid
  snaps, malformed hashes, non-finite geometry, and source/status mismatches
  all failed closed.
- Compare focused contract regression: `23 passed`; integrated Compare,
  geographic frontend, responsive, controller, and frontend contracts also
  passed after stale assertions were migrated.
- Ruff passed for the entire product application, product tests, and launcher.
  JavaScript syntax and repository-wide `git diff --check` passed.
- `scripts/run_project_checks.py` passed Ruff, context-snapshot tests, and the
  full `816 passed` repository suite. A duplicate pytest module basename was
  resolved by renaming the product test to
  `tests/product_app/test_product_route_coverage.py`; full collection now
  accounts for all 816 tests without import ambiguity.
- Live affected-route request on the replacement service returned two
  road-validated, non-fallback GeoJSON legs with query evidence and expiry;
  revision remained 0 and the draft remained empty.
- Live Compare verified exact in-place selection, exact Evidence binding, and
  one baseline/two selected canvases before the blank-pane visual audit. The
  later camera/layout remediation has deterministic coverage but still awaits
  a post-fix independent visual re-audit.

### Gate boundary and remaining work

- W4R remains `in-progress`; G4R remains unverified. No status was promoted.
- Independent pre-remediation Compare audit failed because mounted canvases
  appeared blank despite ready status. The fix is implemented, but browser
  availability prevented a post-fix independent screenshot in this checkpoint.
- Snap preview remains deliberately `persisted:false`; confirmed custom
  waypoint append/revise/undo, catalog/custom drag, route-aware POI discovery,
  richer stop/timeline/constraint editing, and full mobile route equivalence
  remain required.
- G4 live provider evaluation and W5 acceptance/Keep transactions remain
  blocked by their existing gates. D1-S0 remains independent and blocked.

## 2026-08-08 - W4R confirmed custom-waypoint integrity slice

### Implemented

- Added a server-owned, revision-checked confirmation path for route-checked
  custom stops and route-only waypoints. The browser confirms only the preview
  ID and user-entered metadata; it cannot submit authoritative snapped
  coordinates, access points, route geometry, or route evidence.
- Added explicit `map-edit-capabilities-v1` states. Confirmed waypoint add and
  relocation are enabled as route-checked interactions, while evaluated repair
  remains truthfully unavailable until the draft compiler/evaluator supports
  these operations.
- Added stable waypoint identity, one-time preview consumption, warning
  acknowledgement, ordered draft persistence, refresh restoration, Undo, and
  custom-waypoint relocation. Catalog markers remain immutable and
  non-draggable.
- Added strict duration-mode and role validation, exact two-leg road-evidence
  validation, selected-access-point provenance, and stable insertion context.
- Hardened persisted-session trust after independent adversarial findings.
  Confirmed map operations are reduced and validated during restore, read,
  authentication, mutation, and waypoint resolution. Malformed or
  semantically inconsistent snapshots fail with
  `confirmed_map_draft_invalid` without rewriting stored bytes or appending a
  new operation.
- Replaced the permissive browser reducer with a fail-closed executable
  reducer. Invalid operations are ignored whole; no partial marker or route is
  displayed as route-checked.
- Added selected-marker-before-drag, Cancel/Escape restoration, accessible
  keyboard/text relocation controls, draft waypoint/route overlay rendering,
  and truthful route-checked-versus-evaluated copy.

### Verification

- Settled focused backend/frontend/geography/security regression:
  `142 passed`, with one known Starlette TestClient/httpx deprecation warning.
- Full product application plus `PlanRepository` regression: `543 passed`,
  with the same known warning.
- Independent confirmed-waypoint integrity audit: PASS; `110 passed` plus an
  executable adversarial JavaScript reducer check, syntax checks, Ruff,
  whitespace validation, and secret/raw-path scanning.
- A real API-confirmed operation was passed into the actual JavaScript reducer
  and resolved as exactly one waypoint with exactly two validated road legs.
- Tampered restart cases covering missing authoritative fields, changed
  metadata, and changed insertion context failed with HTTP 409; stored bytes
  remained unchanged.
- Independent route-waypoint relocation preserved stable identity, null visit
  duration, stored insertion context, exact-one append semantics, Undo, and the
  immutable parent hash.
- JavaScript syntax checks, product-wide Ruff, and repository-wide
  `git diff --check` passed. One test was made deterministic by choosing a
  known parent-plan stop instead of relying on set iteration order.

### Gate boundary and remaining work

- W4R remains `in-progress`; G4R and G4 remain unverified/blocked. This slice
  does not promote any product or research gate.
- The running product process must be restarted before live browser testing of
  the new endpoints and cache-busted frontend assets.
- Live pointer/drag, layout, and Compare post-fix visual verification remains
  pending. The independent audit for this slice was code/API focused.
- Confirmed custom map operations currently stop at truthful route-checked
  feedback. Compiler/evaluator support, route-aware POI replacement/discovery,
  richer stop editing, and W5 acceptance/Keep transactions remain later work.
- D1-S0 lodging semantic validity remains an independent research blocker;
  no frozen research artifact or status was changed.

## 2026-08-08 - W4R route-aware POI discovery and registered replacement slice

### Implemented

- Added a hash-pinned California Coast product POI catalog with two exact,
  source-bound registered replacements: Bixby Creek Bridge Viewpoint and the
  Santa Barbara Sea Center. Display coordinates remain separate from the
  router-selected access points. Access is labeled as a provider route point,
  not as a verified venue entrance.
- Added deterministic catalog generation and verification. The builder binds
  the product package, registered child artifacts, certificates, and frozen
  route matrix; it refuses output outside the configured catalog directory.
- Added authenticated, revision-checked, read-only route-aware candidate
  discovery for an exact day, route leg, and optional replacement target.
  Replacement burden uses the exact two-leg baseline:
  `T(i,c) + T(c,j) - T(i,target) - T(target,j)`.
- Added explicit candidate states and nullable evidence. Both current frozen
  candidates are independently evaluated feasible, but neither is ranked or
  labeled recommended. Missing opening hours, weather, parking, walking,
  waiting, visit duration, descriptions, and URLs remain `Unavailable`.
- Added bounded candidate and route-access layers, keyboard/touch candidate
  controls, strict exact-selection state, safe external-link handling, and an
  explicit registered-mapping-only `replace_nearby` action. Discovery never
  commits a draft automatically.
- Hardened the complete API-to-browser trust boundary after independent
  adversarial findings. Route-matrix schema, entity index, header provenance,
  raw route cells, access evidence, candidate burden, evaluator evidence,
  precheck state, registered mapping, route roles, and browser cross-field
  consistency now fail closed.

### Verification

- Settled backend catalog/burden/POI API suite: `85 passed`, with one known
  Starlette TestClient/httpx deprecation warning.
- Settled frontend API-to-actual-normalizer suite: `35 passed`, with the same
  warning. Nine forged cross-field payload families are rejected.
- Independent POI integrity audit: PASS WITH LIMITATIONS; `99 passed` plus
  Ruff, three JavaScript syntax checks, deterministic catalog regeneration,
  repository-wide whitespace validation, and secret/raw-path/placeholder
  scans.
- The independent exact journey discovered two candidates, appended exactly
  one `replace_nearby` draft, produced eligible child
  `plan_f5ee52459659dcb5`, and preserved both parent-plan hashes. Accept and
  Keep-original continued to fail closed with HTTP 409 until W5.
- Additional map/controller/security/W3 regression after cache-token migration:
  `77 passed`. Geographic/frontend normalization regression: `25 passed`.
- Full product application plus `PlanRepository` regression was completed as
  three isolated, non-overlapping shards after the monolithic command exceeded
  its time limit: `207 + 223 + 166 = 596 passed`, with the same known warning.
- The approved shell now cache-busts the POI JavaScript and CSS assets with
  `20260808-w4r-poi1`.

### Gate boundary and remaining work

- W4R remains `in-progress`; G4R and G4 remain unverified/blocked. No product
  or research status was promoted.
- This is a frozen two-candidate deterministic catalog, not a general POI
  provider. It does not claim opening-hours, weather, availability, semantic
  recommendation, or verified entrance evidence.
- Only registered candidates can currently compile into executable
  `replace_nearby` operations. General place ingestion and arbitrary candidate
  confirmation remain unavailable.
- The running product process must be restarted before live browser testing.
  Live visual, touch, mobile, and post-fix Compare audits are still pending.
- Rich role/time/duration/order/attribute-level stop editing, route-policy
  editing, W5 acceptance, and D1-S0 exact-research recovery remain later work.

## 2026-08-08 - W4R typed stop editor and route-issue integrity slice

### Implemented

- Added a closed `product-typed-edit-capabilities-v1` contract for role, day,
  order, duration, time-window, commitment, attribute-constraint, route
  preference, and route-issue actions. Each operation declares whether it is
  enabled, its feedback tier, whether evaluated Preview can execute it, and a
  stable blocking code.
- Added strict persistent typed drafts for stop role/order/duration/time-window,
  commitment, independent attribute constraints, and exact accepted-route-leg
  user issue reports. These survive restart and support Undo.
- Added authenticated `POST /api/sessions/{session_id}/draft/impact-preview`,
  bound to the exact session revision and immutable parent plan/hash. It
  reports changed attributes and draft/evaluated capability, and explicitly
  returns `not_certified`; it never emits eligibility or certificate claims.
- Enabled only `set_stop_day` through the existing deterministic repair
  pipeline. A real Griffith Observatory Day 3 to Day 4 request currently
  returns a truthful `no_feasible_evaluated_child` without fabricating a child,
  diff, or certificate. Same-day requests fail as `draft_no_effect`.
- Kept stop role/order/duration/time-window/commitment/attribute edits and route
  issue reports draft-only because the current evaluator does not model them.
  Route preference remains disabled because local OSRM/evaluator cannot execute
  it truthfully.
- Added shared conflict accounting across legacy and typed day/order/attribute
  operations, protected commitment enforcement, exact target revalidation on
  restore/read/mutation, and fail-closed handling for stale or tampered drafts.
- Replaced the limited stop/route inspector with human-readable accepted-plan
  details, seven typed stop-edit forms, an exact route-issue form, draft impact
  summaries, explicit feedback tiers, unavailable-state rendering, responsive
  mobile forms, and keyboard/touch-sized actions.
- Hardened the UI so child-plan features cannot open accepted-plan edit actions,
  impact rows are rebound to the exact ordered session draft and capabilities,
  raw target IDs do not dominate primary content, and null/string/nonpositive
  route metrics render `Unavailable` rather than zero.

### Verification

- Backend typed contract suite: `14 passed`; post-hardening
  waypoint/workspace/product trust regression: `35 passed`; broader impacted
  backend regression: `89 passed`.
- Frontend typed/geographic/POI/waypoint/Compare/responsive integration:
  `69 passed`, including an actual nonempty backend payload through the real
  JavaScript normalizers and executable forged-payload cases.
- Root cross-module regression: `159 passed`, with one known Starlette
  TestClient/httpx deprecation warning.
- Independent typed stop/route integrity audit: PASS for this deterministic
  slice; `133 passed` plus API forgery, restart/Undo, immutable-parent hash,
  executable JavaScript adversarial, Ruff, and whitespace checks.
- JavaScript syntax, executable typed-editor adversarial tests, product-wide
  Ruff, and repository-wide `git diff --check` passed.

### Gate boundary and remaining work

- W4R remains `in-progress`; G4R and G4 remain unverified/blocked. No product
  or research status was promoted.
- Draft-only operations intentionally cannot run evaluated repair. Their solver
  and independent evaluator semantics must be implemented before they can
  become decision-eligible.
- Protected `must_keep`/`booked` permission acquisition is not enabled; those
  strengths fail closed. Route preference remains disabled.
- The running product process must be restarted before live browser testing of
  `20260808-w4r-typed1`. Live visual, touch, mobile, and manual screen-reader
  audits remain required before G4R verification.
- W5 acceptance and Keep-original remain disabled. D1-S0 exact-research
  recovery remains separate and blocked.

## 2026-08-08 - W4R independently evaluated same-day stop reorder

### Implemented

- Promoted only `set_stop_order` from draft-only feedback to an independently
  evaluated direct edit. The operation is restricted to the stop's accepted
  day and uses a zero-based server contract while the UI presents human
  one-based positions.
- Added pre-append and restore validation against the immutable parent's exact
  in-day order. Wrong-day, Boolean/out-of-range, no-effect, stale, and
  conflicting requests fail before revision or draft mutation.
- Added deterministic direct-edit compilation that creates an immutable child,
  updates `selected_stops`, `ordered_days`, the global sequence, and
  content-addressed per-day route identities, then validates the complete
  frozen route matrix.
- Added exact reorder and road-change `PlanDiff` records, a fresh independent
  `PlanEvaluator` certificate, and `product-geography-v2` bound to the new
  child identity. The representative Day 4 reorder has 16/16 required
  road-validated legs and zero gaps.
- Added strict response identities for session revision, ordered draft
  operations, parent/child, diff, certificate, route evidence, and geography.
  The browser recomputes the draft hash and rejects identity, revision,
  route-chain, endpoint, distance, provenance, and geography forgeries before
  changing session state.
- Removed inherited aggregate travel, utility, contextual-risk, and edit
  metrics that the direct evaluator did not recompute. Only the exact selected
  attraction count is retained; all other missing metrics remain unavailable.
- Made impact and Preview agree for mixed drafts. A reorder combined with any
  non-order operation is non-executable and reports
  `draft_evaluated_operation_combination_unsupported` before Preview.
- Added evaluated-edit evidence UI, a fixed-day reorder form, responsive
  keyboard/touch controls, human-readable blocking text, and cache-busted
  `20260808-w4r-typed2` assets.

### Verification

- Root regression across reorder compiler/API/frontend, typed-edit contracts,
  W3 workspace/preview, geography, security, product API, `PlanDiff`, and
  `PlanEvaluator`: `123 passed`, with one known Starlette TestClient/httpx
  deprecation warning.
- Independent frozen-boundary audit: PASS; `18 passed` for backend plus actual
  API-to-JavaScript and forgery cases, and `5 passed` for certificate freshness
  and W5-deferred guardrails.
- Focused `PlanDiff`/`PlanEvaluator` regression: `15 passed`. Frontend
  integration: `14 passed`; actual reorder/legacy/mixed/forgery contract:
  `4 passed`.
- JavaScript syntax, executable typed-editor adversarial checks, scoped Ruff,
  and repository-wide `git diff --check` passed.

### Gate boundary and remaining work

- This is an implemented and independently verified feature slice, not W4R or
  G4R completion. CP-010/W4R remains `in-progress`; G4R remains unverified; G4
  remains blocked; W5 acceptance and Keep-original remain disabled.
- Role, duration, time-window, commitment, attribute-constraint, route-issue,
  and route-preference changes are not newly evaluator-enabled. Duration lacks
  a typed diff, departure-window semantics are not modeled losslessly, and
  unsupported route policies remain unavailable.
- Mixed-operation evaluated preview is intentionally unsupported. The browser
  validates route structure, endpoints, distance, and provenance, but the
  current evidence contract has no per-leg hash for every interior polyline
  coordinate.
- Stop-by-stop opening-window accounting remains unavailable. Live visual,
  accessibility, mobile/touch, and full black-box journey audits still require
  a restarted product process and are mandatory before G4R verification.
- The D1-S0 lodging-semantic repair and exact Gurobi recovery remain a separate
  blocked research workstream; no research status or frozen artifact changed.

## 2026-08-08 - W4R independently evaluated exact visit duration

### Implemented

- Promoted only canonical `set_stop_duration` `exact` mode to independently
  evaluated preview. Exact values must be integer minutes from 15 through 480
  and repeat the same value in preferred/minimum/maximum fields. Boolean,
  float, string, missing, unequal, and out-of-range values fail before draft
  mutation.
- Kept `preferred`, `minimum`, `maximum`, and `range` duration semantics
  draft-only with the stable `duration_mode_evaluation_not_supported` blocker;
  the scalar evaluator cannot enforce those modes or their objective penalties
  losslessly.
- Added a versioned `plan-diff-v2` only when typed `DurationChange` records are
  present. Historical and non-duration diffs retain the exact
  `plan-diff-v1` serialization. Duration changes preserve raw missingness,
  canonical before/after constraints, scalar sources, evaluator accounting
  sources, delta, ownership strength, and cost-policy evidence; they are not
  misreported as time shifts.
- Added deterministic exact-duration compilation to an immutable child with a
  new identity/hash. The child changes only the target duration constraint and
  `visit_duration_minutes`; route sequence, route identities, and all 16
  cached-OSRM legs remain unchanged and are revalidated with zero gaps and
  fallbacks.
- Added independently recomputed parent/child schedule accounting, fresh
  evaluator certificate evidence, and exact API identities for the diff,
  schedule, certificate warnings/failures, route matrix, child geography, and
  ordered draft. Schedule evidence exposes travel, visit, conditional opening
  wait, day limit, slack, and overrun.
- Made the schedule scope explicit: parking/drop-off, walking transfer, queue
  wait, and service buffers remain unavailable, and all fixture opening
  windows are missing. The resulting eligible example is therefore
  `PASSED_WITH_WARNINGS` with `incomplete_evidence`, not a claim of complete
  itinerary-time validation.
- Added a truthful infeasible journey: two 480-minute Day 4 edits produce a
  fresh `day_time_exceeded` failure with a positive modeled overrun, an
  ineligible proposal, full route evidence, and no acceptance path.
- Hardened the API-to-browser contract against ordinary and coherently
  self-rehashed diff, schedule, certificate, route, proposal, and cost-policy
  forgeries. The UI shows explicit-versus-fallback duration, modeled schedule
  components, exact slack/overrun, missing evidence, and W5-disabled state.
- Corrected evaluator missingness globally: `weather_risk_delta` is emitted
  only when parent and child have the same sequenced stop universe and complete
  finite weather-risk evidence. Missing or non-finite risk now remains absent,
  and the product renders `Unavailable` rather than fabricated zero.
- Duration preview decision eligibility follows independent hard evaluation,
  while `ranking_eligible` is always false because no evaluator-owned ranking
  criterion exists. `acceptance_eligible` remains false with the W5 blocker.
  The approved shell cache-busts duration assets as
  `20260808-w4r-duration1`.

### Verification

- Backend exact-duration/diff/evaluator/API regression: `78 passed`, with one
  known Starlette TestClient/httpx deprecation warning.
- Frontend eligible/ineligible API-to-JavaScript, coherent-rehash, reorder,
  and contract regression: `7 passed`; broader frontend integration:
  `31 passed`.
- Root current-files integration across duration, reorder, typed drafts, W3
  preview/workspace, geography, security, `PlanDiff`, and `PlanEvaluator`:
  `151 passed`, with the same known warning.
- Independent duration audit: PASS. Its branches covered `49` core
  duration/diff/evaluator tests, `44` route/browser regression tests, targeted
  missingness and API-to-browser checks, `6/6` raw zero/non-finite rejection
  probes, and `18` final PlanEvaluator/PlanDiff regressions.
- JavaScript syntax, executable typed-editor adversarial checks, Python
  compileall, scoped Ruff, cache-token assertions, and repository-wide
  `git diff --check` passed.

### Gate boundary and remaining work

- This is an implemented and independently verified feature slice, not W4R or
  G4R completion. CP-010/W4R remains `in-progress`; G4R remains unverified; G4
  remains blocked; W5 acceptance and Keep-original remain disabled.
- Non-exact duration modes remain draft-only. Time-window departure semantics,
  role changes, commitment/attribute constraints, and supported route-policy
  evaluation still require separate lossless diff/evaluator contracts.
- Schedule totals are partial modeled accounting, not complete real-world
  visit duration: opening windows, parking, walking, queueing, and service
  buffers remain unavailable for this fixture.
- No live visual, manual accessibility, mobile/touch, or complete black-box
  audit was performed for the new duration UI. The stale running product
  process must be restarted before those independent G4R audits.
- The D1-S0 lodging-semantic repair and exact Gurobi recovery remain separate
  blocked research work; no frozen research artifact or status changed.

## 2026-08-08 - W4R independently evaluated stop time windows

### Implemented

- Promoted only canonical `set_stop_time_window` edits to independently
  evaluated preview. Requests use the exact
  `{earliest_arrival: HH:MM|null, latest_departure: HH:MM|null}` contract,
  require at least one bound, reject non-zero-padded clocks, Boolean/extra
  fields, cross-midnight ranges, duplicates, stale revisions, wrong targets,
  no-effect writes, and unsupported mixed-operation drafts before unintended
  persistence.
- Added the trip-owned `stop-time-window-constraint-v1` field without
  rewriting place opening-hour or scheduled arrival/departure aliases.
  Earliest arrival is explicitly modeled as service admission: the raw road
  arrival remains unchanged and a separate required-window wait advances
  service start. Latest departure is checked after the modeled visit rather
  than as a latest-start shortcut.
- Added conditional `plan-diff-v3` serialization with typed
  `TimeWindowChange` records and ownership-weighted cost evidence. Existing
  non-time-window v1 and duration-only v2 diff shapes remain unchanged; time
  windows are never represented as generic time shifts.
- Added deterministic immutable-child compilation, complete revalidation of
  the unchanged 16-leg frozen road route, and a fresh independent evaluator
  certificate. Feasible earliest-only, latest-departure violation, and
  earliest-wait day-overrun cases produce distinct truthful outcomes with
  positive violation accounting where applicable.
- Added hash-bound parent/child schedule evidence with the registered demo's
  configured 09:00 day start, exact ordered incoming route leg and query hash,
  raw road arrival, opening-window value/source or explicit unavailability,
  opening wait, required-window wait, service start, visit value/source,
  departure, latest-departure status and overrun, blocking codes, and complete
  trace order for all nine accepted stops.
- Hardened both server and browser boundaries against coherently rehashed
  trace removal/reordering, shifted day start and arrivals, modified route
  durations, invented opening evidence, altered visit sources, bounds, waits,
  blockers, statuses, certificate evidence, and cost policy. Missing opening
  evidence remains missing and is never converted to zero or a checked claim.
- Added evaluated time-window UI evidence that distinguishes route arrival,
  place opening evidence, trip-required wait, service start, visit, and
  departure. Decision eligibility follows the independent certificate;
  `ranking_eligible` remains false and W5 acceptance remains disabled. Assets
  are cache-busted as `20260808-w4r-timewindow1`.

### Verification

- Backend time-window developer suite: `17 passed`; seven-file impacted
  backend regression: `104 passed`.
- Root current-file split runs: `17 passed` for the time-window backend,
  `23 passed` for `PlanDiff` and `PlanEvaluator`, and `52 passed` for the
  time-window/duration/reorder/typed/geographic frontend integration. Each
  completed with only the known Starlette TestClient/httpx deprecation
  warning where applicable.
- Independent time-window audit: PASS. Clean authoritative runs were
  `40 passed` for backend/diff/evaluator, `3 passed` for the actual
  API-to-JavaScript adversarial contract, and `14 passed` for typed
  capability/session regression.
- JavaScript syntax, executable typed-editor adversarial checks, scoped Ruff,
  Python compilation, and scoped whitespace checks passed. A larger combined
  pytest process reached 100% without reported failures but hung during
  Windows post-test cleanup; it was terminated and is deliberately excluded
  from pass counts.

### Gate boundary and remaining work

- This is an implemented and independently verified feature slice, not W4R
  or G4R completion. CP-010/W4R remains `in-progress`; G4R remains unverified;
  G4 remains blocked; W5 acceptance and Keep-original remain disabled.
- The browser verifier is intentionally pinned to this registered demo's
  09:00 day start. All nine demo stops lack source-backed opening windows and
  use the disclosed 45-minute configured visit fallback. A future runtime
  configuration or source-backed opening-window rollout requires separately
  validated workspace evaluator-configuration and source identities.
- No live browser, mobile/touch, visual, manual screen-reader, or complete
  black-box audit was performed for this slice because the running product
  process still needs a user-controlled restart. Those audits remain required
  before G4R verification.
- Role, commitment, attribute-constraint, and supported route-policy
  evaluation remain separate work. The D1-S0 lodging-semantic repair and exact
  Gurobi recovery remain a separate blocked research workstream; no frozen
  research artifact or status changed.

## 2026-08-08 - W4R independently evaluated itinerary-stop roles

### Implemented

- Promoted only the non-structural, visit-bearing `set_stop_role` values
  `attraction`, `activity`, `meal`, `rest_stop`, and `scenic_stop` to
  independently evaluated preview. Structural roles (`lodging`,
  `transport_hub`, `route_waypoint`, `origin`, and `destination`) remain
  draft-only with the stable `stop_role_evaluation_not_supported` blocker.
- Added a distinct nullable trip-owned `itinerary_role` plus
  `itinerary_role_source`; map/plan display role and place categories remain
  separate and unchanged. Missing parent role evidence stays explicitly
  unavailable, and the editor no longer silently defaults it to attraction.
- Added conditional `plan-diff-v4` serialization with typed `RoleChange`
  records, exact raw-source missingness, role-specific ownership strength, and
  disclosed cost. Existing v1 through v3 diff shapes remain unchanged.
- Added deterministic immutable-child compilation and fresh independent
  evaluation. Role edits preserve stop order, days, route IDs, all 16 validated
  cached-OSRM legs, and modeled schedule accounting. Unrecomputed utility,
  contextual-risk, recommendation, and ranking metrics remain unavailable.
- Added parent-bound, deterministic, sanitized role-constraint evidence.
  Existence locks do not block or weight role changes; role-scoped constraints
  are value-aware, explicit-permission mismatches fail closed, and only the
  documented weighted-mismatch policy may proceed with a warning and cost.
- Added explicit semantic scope: itinerary role is a trip-specific user
  assertion, does not change place identity or category, has no inferred
  semantic-fit/recommendation claim, and has no route or schedule effect for
  the evaluated subset.
- Hardened API restore and browser consumption. The full proposal is
  HMAC-SHA256 authenticated with the in-memory session mutation token; only a
  salted verifier is stored. Restore verifies the HMAC, recompiles, reconstructs
  the exact timestamp-bound certificate, and exact-compares role impact, diff,
  geography, route, schedule, constraint, and certificate evidence. The
  browser independently recomputes the HMAC before accepting the proposal.
- Duplicate, stale, mixed-operation, malformed-parent, wrong-role, structural,
  certificate timestamp/hash, ownership, route, schedule, geography, and
  coherently rehashed forgery cases fail closed. W5 acceptance remains disabled
  and `ranking_eligible` remains false.
- Cache-busted the evaluated-role frontend assets as
  `20260808-w4r-role2`.

### Verification

- Backend role suite: `23 passed`; impacted diff/evaluator/geography/typed
  regression: `50 passed`; prior evaluated order/duration/time-window/W3
  regression: `85 passed`.
- Frontend actual API-to-JavaScript/HMAC harness: `3 passed`; five-slice
  integration: `23 passed`; full eleven-file frontend boundary: `86 passed`.
- Root current-files role/API/frontend/PlanDiff/evaluator gate: `53 passed`.
- Independent role audit: PASS with `53 passed` for the role boundary and
  `73 passed` for prior evaluated-slice regression.
- JavaScript syntax, executable adversarial checks, scoped Ruff, Python
  compilation, and repository-wide `git diff --check` passed. TestClient runs
  retain only the known Starlette/httpx deprecation warning.

### Gate boundary and remaining work

- This is an implemented and independently verified feature slice, not W4R or
  G4R completion. CP-010/W4R remains `in-progress`; G4R remains unverified; G4
  remains blocked; W5 acceptance and Keep-original remain disabled.
- Historical PlanDiff v1 through v3 ownership-cost logic still collapses some
  non-role constraint relations by target. It did not affect the registered
  demo or this role-v4 slice, but a versioned relation-specific migration is
  required before claiming general attribute-level ownership compliance.
- The integrity boundary assumes the existing loopback single-user model.
  An adversary holding both state-file write access and the live bearer token
  is outside scope and is not claimed to be cryptographically prevented.
- Structural role semantics, commitment and attribute-constraint evaluation,
  supported route-policy evaluation, and live browser/mobile/accessibility
  audits remain separate work. The D1-S0 lodging-semantic repair and exact
  Gurobi recovery remain a separate blocked research workstream; no frozen
  research artifact or phase status changed.

## 2026-08-08 - PlanDiff v5 relation-specific ownership and cost evidence

### Implemented

- Replaced target-wide ownership weighting for newly owned non-role changes
  with an explicit relation-to-attribute mapping covering existence, day,
  start time, end time, exact duration, required time window, and one-based
  stop order. Existence locks no longer weight unrelated day, time, duration,
  window, or order edits.
- Added conditional `plan-diff-v5` serialization. V5 is emitted only when an
  actual changed attribute has relevant active accepted-parent ownership
  evidence; unconstrained historical v1 through v4 outputs retain their prior
  schemas. Role evidence remains separately versioned under v4.
- Added deterministic, parent-hash-bound ownership evidence with sanitized
  constraint rows, exact before/after values, satisfaction before/after,
  strongest applicable strength, policy weights, delta scale, and finite cost
  for every listed diff change. The evidence explicitly makes no permission,
  feasibility, ranking, or acceptance claim.
- Preserved raw missingness and strict schemas: booleans are not integers;
  exact durations require equal integer values from 15 through 480; time
  windows allow either one-sided bound but reject both-null and cross-midnight
  values; start/end aliases remain independent; order evidence is one-based
  and deliberately excludes editor `sequence_index` aliases.
- Bound serialized evidence back to the exact typed PlanDiff universe. Kind,
  target, changed attributes, before/after values, owner strength, delta,
  cost, satisfaction rows, selected constraints, and the complete consumed
  constraint-ID set must all match. Coherently rehashed substitutions and
  unused constraint rows fail closed.
- Required child owned constraints to exactly match the accepted parent before
  v5 cost derivation, preventing child-side injection, removal, or strength
  upgrade. Relevant malformed, duplicate, or contradictory constraints fail
  deterministically regardless of input order.
- Kept lodging and road relation ownership explicitly unmigrated. Their
  ordinary base-cost rows remain represented when v5 is active, but legacy
  lodging/road ownership cannot be mixed into v5 evidence.
- Kept the migration domain-only: the registered product compiler verifier
  rejects `plan-diff-v5` with `draft_pipeline_diff_invalid`. No browser,
  product capability, evaluator eligibility, W5 transaction, or status was
  activated by this slice.

### Verification

- Root focused v5 adversarial suite: `19 passed`; PlanDiff and PlanEvaluator
  compatibility: `27 passed`.
- Independent audit: PASS with `46 passed` for v5/diff/evaluator, `46 passed`
  for evaluated order/duration product regression, `40 passed` for evaluated
  time-window/role regression, and `8 passed` for model/repository
  compatibility: **140 clean authoritative passes** total.
- One combined 86-test run printed `86 passed` but the Windows wrapper timed
  out during cleanup; it was excluded and replaced by the two clean-exit
  product regression splits above.
- Python compilation, scoped Ruff, and scoped whitespace checks passed. Only
  existing LF-to-CRLF notices remain.

### Gate boundary and remaining work

- This is an implemented and independently verified domain-foundation slice,
  not product activation or W4R/G4R completion. CP-010/W4R remains
  `in-progress`; G4R remains unverified; G4 remains blocked; W5 acceptance and
  Keep-original remain disabled.
- A future product consumer must resolve the evidence parent content hash
  against the authoritative accepted parent artifact before enabling v5.
  Lodging/road relation migration and role-v4 unification remain separate.
- Live browser, mobile/touch, visual, accessibility, and complete black-box
  audits still require the user-controlled product-process restart. The D1-S0
  lodging-semantic repair and exact Gurobi recovery remain a separate blocked
  research workstream; no frozen research artifact or phase status changed.
