# W4R Stability and Coherence Implementation Report

Date: 2026-08-11

Status: **Historical W4R implementation report. The later remediation closed the recorded defects and G4R is now independently verified/passed.** Current status authority is `../current/current_problem_manifest.md`; preserve this file as implementation history, not as the live gate verdict.

## Scope completed

- Bounded desktop and mobile map workspaces with explicit map-readiness checks.
- Resize and route/module lifecycle handling for Trip, Compare, Evidence, Copilot, and viewport changes.
- Coherent California Coast product fixture across itinerary order, day assignments, routes, markers, repair context, metrics, and explanations.
- Shared selection behavior across the timeline, map, Compare, Evidence, and contextual panels.
- User-oriented Compare defaults, exact option binding, text-route evidence, and simplified navigation.
- Route-validated geography, explicit gaps, controlled snap/waypoint workflows, registered POI replacement, and typed stop/route edits.

## Root-run stability evidence

The detailed acceptance record is [the stability/coherence acceptance report](../../results/product_app/stability_coherence_20260811/ACCEPTANCE_REPORT.md).

Observed evidence includes:

- Fresh load, reload, Trip/Compare/Trip switching, Copilot open/close, and desktop resize checks.
- Full-size captures at 1440x900, 1280x800, 1024x768, and 390x844.
- Map-ready capture gating with at least 95% visible tile coverage.
- Compare opening with an exact selected recommendation and synchronized evidence.
- A visible, testable text-route panel.

## Automated evidence

- Coherent workflow contracts: 21 passed.
- Final product/UI slice: 51 passed; one known Starlette TestClient/httpx deprecation warning.
- Research-only POI coverage: all 45 cases covered by a 32-pass run plus a 13-pass targeted rerun.
- Coherent-fixture deterministic builder check: passed.
- Ruff, JavaScript syntax checks, Python compileall, and git diff checks: passed.

## Strict-verifier review

### Logical correctness

The final product fixture uses one itinerary identity and ordering across schedule, geography, affected stop, repair, comparison, and evidence. Map readiness is a measurable state rather than a screenshot delay, and stale or mismatched option/route identities fail closed.

### Adversarial review

The verification covered reload and module transitions, viewport changes, missing/stale selections, exact alternative resolution, route gaps, unsupported operations, and acceptance remaining disabled before W5. No root-run blocker remains for the W4R implementation state.

### Independence objection

This report is implementation evidence produced by the root workflow. It is not an independent G4R verdict. A fresh independent audit must recheck the final repository and live service across product behavior, map rendering, responsive accessibility, content/IA, security/privacy, and black-box journeys.

## Gate decision

- W4R: **implemented**.
- G4R: **failed / blocked pending remediation and fresh independent re-audit**.
- G4: **blocked** until G4R passes and a bounded external-provider evaluation is explicitly authorized.
- W5: **closed/planned**; Accept and Keep-original remain disabled.
- CP-010: remains **in progress** because its independent gate is not closed.
- Research-track statuses remain unchanged.
