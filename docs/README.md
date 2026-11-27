# Documentation Index

This directory is organized by the kind of question you are trying to answer.

## Folder Map

| Folder | Use It For |
| --- | --- |
| [literature/](literature/) | Literature onboarding, core reading cards, evidence matrix, deep evidence bank, and repair-gap review. |
| [methods/](methods/) | Canonical method notes for itinerary repair and nature-aware scoring. |
| [planning/](planning/) | Active planning index, research gates, technical specification, and Copilot implementation plan. |
| [audits/](audits/) | Current independent product audits, reproducible baseline evidence, and cross-audit synthesis. |
| [reference/](reference/) | Stable engineering references such as the data dictionary and quality workflow. |
| [deployment/](deployment/) | Railway, Supabase, ORS and GitHub Pages deployment instructions and cost limits. |
| [slides/](slides/) | Literature-review slide source and PDF output. |
| [assets/](assets/) | README/dashboard images. |
| [figures/](figures/) | Research figures referenced from Markdown or papers. |

## Reading Paths

| Need | Start Here |
| --- | --- |
| Low-cost interactive app hosting | [deployment/railway_supabase_ors.md](deployment/railway_supabase_ors.md) |
| Project and literature orientation | [literature/literature_onboarding_guide.md](literature/literature_onboarding_guide.md) |
| Core papers and publication positioning | [literature/core_paper_reading_cards.md](literature/core_paper_reading_cards.md) |
| Full evidence bank | [literature/literature_deep_read_study_report.md](literature/literature_deep_read_study_report.md) |
| Evidence matrix and scoring caveats | [literature/evidence_matrix.md](literature/evidence_matrix.md) |
| Repair gap, novelty framing, and related-work outline | [literature/repair_gap_review.md](literature/repair_gap_review.md) |
| Repair method and Phase 0 execution | [methods/repair_method.md](methods/repair_method.md) |
| Nature-aware scoring design | [methods/nature_aware_model_extension.md](methods/nature_aware_model_extension.md) |
| Planning authority and migration index | [planning/README.md](planning/README.md) |
| Itinerary Repair Copilot | [planning/itinerary_repair_copilot_implementation_plan.md](planning/itinerary_repair_copilot_implementation_plan.md) |
| W4 Copilot provider, prompt, transcript, and privacy contract | [planning/w4_copilot_provider_transcript_phase_plan.md](planning/w4_copilot_provider_transcript_phase_plan.md) |
| Local Copilot launch, provider selection, and privacy instructions | [Repository README](../README.md#open-the-itinerary-repair-copilot) |
| Current implementation status | [current/current_problem_manifest.md](current/current_problem_manifest.md) |
| Current runtime progress, remaining concerns, exact file/LLM I/O | [October 7 checkpoint](reports/runtime_coherence_status_20261007.md) |
| Product audit synthesis and baseline | [audits/product_audit_synthesis.md](audits/product_audit_synthesis.md) |
| Verified W1M/G1 live runtime closeout | [audits/w1m_live_g1_verification_report.md](audits/w1m_live_g1_verification_report.md) and [manifest](audits/w1m_live_g1_evidence_manifest.json) |
| Repair implementation contract | [planning/travel_itinerary_repair_technical_specification.md](planning/travel_itinerary_repair_technical_specification.md) |
| Data records and evidence roles | [reference/data_dictionary.md](reference/data_dictionary.md) |
| Formatting, tests, and artifact policy | [reference/code_quality_workflow.md](reference/code_quality_workflow.md) |
| Literature-review slides | [slides/literature_review_slides.pdf](slides/literature_review_slides.pdf) |

## Notes

- Process/audit notes, duplicate specs, and LaTeX build outputs were merged away so this index points only to durable docs.
- `planning/travel_itinerary_repair_technical_specification.md` is the single canonical repair implementation contract.
- W4 is implemented and its offline audits pass. The first authorized G4 live
  smoke and fixed-24 comparison passed aggregate thresholds but failed independent
  provenance audit. V4 closed recomputability; the authorized v5 run now preserves
  exact synthetic LLM input/output, complete file I/O, and matching 48-attempt SDK
  and HTTP transport evidence. Three independent audits passed with zero blockers,
  so G4 v5 verified its frozen source snapshot. The authorized 2026-08-13
  current-source `gpt-5.4-mini` v6 run passed all 48 cases with exact file/LLM
  input and output and matching SDK/HTTP attempt counts. Three independent
  read-only reviews pass after audit-binding remediation, so current-source G4 v6
  is verified. W5 is implemented with durable confirmed Accept/Keep/rollback;
  G5 remains pending and W6 remains closed.
