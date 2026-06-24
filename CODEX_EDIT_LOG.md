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

