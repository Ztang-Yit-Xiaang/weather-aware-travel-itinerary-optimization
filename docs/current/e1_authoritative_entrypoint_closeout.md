# E1 Authoritative Entry Point Closeout

**Closeout date:** 2026-07-11  
**Gate:** E1 in `docs/planning/current_execution_plan.md`

## Delivered

- `scripts/run_research_pipeline.py` is the package-backed generation entry point.
- It supports frozen-artifact adaptation and explicit raw-catalog production execution.
- Refresh defaults to `never`; strict evaluation is the default and permissive mode must be explicit.
- `notebook/production_system_blueprint.ipynb` is now a six-cell thin client with 28 code lines.
- The notebook declares parameters, invokes the authoritative command, and reads the emitted manifest and metrics.
- Data collection, model construction, solver construction, evaluation, and artifact export are absent from the notebook.
- Automated tests enforce the notebook boundary and CLI input/summary contracts.

## Verification Evidence

- Ruff: `python -m ruff check --no-cache src tests scripts` — passed.
- Full test suite: 179 tests passed on Python 3.12.10.
- Focused pipeline/routing/benchmark suite: 32 tests passed.
- Frozen-input smoke run: `e1-smoke-20260711-v2`.
- Smoke result: 3 planner runs, 3 plans, 3 evaluations, immutable manifest and metrics emitted.
- Smoke status: `completed_with_warnings`, with 3 strict failures retained because route evidence is not road validated.

## Gate Decision

E1 is closed. The warning status is the expected fail-closed transition into E2, not an E1 failure. E2 remains open until the route-evidence conditions in `route_evidence_readiness_audit.md` pass.
