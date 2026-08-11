# Planning Authority Index

This directory now has one explicit authority chain. Historical plans were
merged by responsibility and removed on 2026-07-30 so a stale roadmap cannot
silently override current evidence.

## Active documents

| Document | Authority |
| --- | --- |
| [current_execution_plan.md](current_execution_plan.md) | Near-term queue and current gate status |
| [research_pipeline_and_gate_map.md](research_pipeline_and_gate_map.md) | Research dependencies, publication gates, and claim boundaries |
| [travel_itinerary_repair_technical_specification.md](travel_itinerary_repair_technical_specification.md) | Durable optimizer, evaluator, artifact, and product invariants |
| [e3_exact_baseline_scalability_phase_plan.md](e3_exact_baseline_scalability_phase_plan.md) | Implementation-ready E3.1 exact-baseline work |
| [itinerary_repair_copilot_implementation_plan.md](itinerary_repair_copilot_implementation_plan.md) | Corrective multi-agent local-product contract, AUD-0/W0-W8 phases, G0-G8 gates, ownership, and verification |
| [w4_copilot_provider_transcript_phase_plan.md](w4_copilot_provider_transcript_phase_plan.md) | Implementation-ready W4 provider, versioned prompt engineering/evaluation, typed interpretation, local transcript, synchronization, and G4 audit contract |
| [../current/map_runtime_substitution_decision.md](../current/map_runtime_substitution_decision.md) | MAP-DEC-002 active MapLibre-primary/Atlas-backup runtime decision and W1M/G1 transition rules |

`docs/current/current_problem_manifest.md` remains the implementation-status
source. A plan is not completion evidence.

Product audit evidence is recorded under `docs/audits/`. Three broad baseline
audits, the synthesis, and nine independent specialist reports support AUD-0/G0:
Design System, Map/Artifact Integrity, Copilot Privacy/Security,
Acceptance/Persistence, Mobile/PWA, Accessibility, Visual Fidelity, User
Journey/Black Box, and Phase/Gate Status. Later gate reports must remain
attributed, read-only reviews and cannot advance research gates.

Historical W1M deterministic evidence is recorded in
[`w1m_map_runtime_migration_report.md`](../audits/w1m_map_runtime_migration_report.md)
and its machine-readable
[`w1m_map_runtime_evidence_manifest.json`](../audits/w1m_map_runtime_evidence_manifest.json).
The verified live G1 closeout is recorded in
[`w1m_live_g1_verification_report.md`](../audits/w1m_live_g1_verification_report.md)
and
[`w1m_live_g1_evidence_manifest.json`](../audits/w1m_live_g1_evidence_manifest.json).
W1M/G1 and corrected-v2 W2/G2 and W3/G3 are verified. W4 is implemented; G4 is
blocked on authorized live provider evidence, and W5-W8 remain planned. Historical v1 draft, preview, browser, regression, and
independent-audit evidence is recorded in
[`w3_g3_verification_report.md`](../audits/w3_g3_verification_report.md) and its
[`w3_g3_evidence_manifest.json`](../audits/w3_g3_evidence_manifest.json). It does
not verify v2. Corrected-v2 evidence is recorded in
[`w2_v2_g2_g3_revalidation_report.md`](../audits/w2_v2_g2_g3_revalidation_report.md)
and its machine-readable evidence manifest.
The current W4/G4 boundary is recorded in
[`w4_g4_verification_report.md`](../audits/w4_g4_verification_report.md) and
[`w4_g4_evidence_manifest.json`](../audits/w4_g4_evidence_manifest.json).

## Migration table

| Removed document | Durable owner |
| --- | --- |
| `context_aware_itinerary_repair_detailed_phase_plan.md` | Technical specification and current execution plan |
| `context_aware_itinerary_repair_integrated_implementation_plan.md` | Technical specification and gate map |
| `current_problem_fix_phase_plans.md` | Current problem manifest and current execution plan |
| `research_stabilization_and_publication_plan.md` | Research pipeline and gate map |
| `e3_legacy_blueprint_core_migration_phase_plan.md` | Historical E3.C reports and Git history |
| `e3_legacy_renderer_extraction_phase_plan.md` | Historical E3.C reports and Git history |
| `e3_renderer_orchestrator_decomposition_phase_plan.md` | Historical E3.C reports and Git history |
| `e3_user_facing_dashboard_reframe_phase_plan.md` | Copilot plan for future product work; E3.UX reports for verified v6 history |
| `permission_aware_counterfactual_clarification_plan.md` | Technical specification and Copilot plan |

Verified historical evidence remains in `docs/reports/`, immutable run
manifests, and Git history. Deleting a superseded plan does not advance a gate.
