# Current Problem Manifest

Generated for Phase 0.0 on 2026-07-06. This manifest records active repair work for the current repository state. It is not a completion tracker: do not tick roadmap or pipeline checkboxes until the relevant implementation and validation pass.

| Problem ID | Severity | Evidence path | Owning phase | Current status | Acceptance check |
|---|---|---|---|---|---|
| CP-000 | blocker | `tests/data/test_context_snapshot.py`; `pyproject.toml` | Phase 0.0 validation harness | Pytest temp and cache behavior needed stabilization in the managed Windows workspace. | `python -m pytest tests/data/test_context_snapshot.py` passes using `.codex_tmp_pytest/pytest`; `python scripts/run_project_checks.py` writes `results/quality/project_check_summary.json`. |
| CP-001 | high | `src/itinerary_system/research_artifacts.py`; `src/itinerary_system/phase0_exporter.py`; `src/itinerary_system/experiment_runner.py` | Phase 0.1 artifact lineage | Implemented v2 artifacts, mutation reporting, child-plan helpers, and Phase 0 invalidation for known required-anchor post-solve edits; broader production call sites still need deeper Phase 1/2 ownership integration. | Mutated plans cannot retain stale solver certification in strict validation. |
| CP-002 | medium | `src/itinerary_system/multi_objective_route.py`; `src/itinerary_system/hierarchical_gurobi.py`; `src/itinerary_system/routing/matrix.py`; `scripts/build_validated_route_matrix.py` | Phase 0.2 route matrix boundary | Implemented `RouteMatrix`, provider protocol types, solver adapter, strict publication-mode gates, demo-only geodesic fallback tests, and matrix-level validation/report artifacts. Remaining provider/pipeline phases must generate complete validated matrices for benchmark contexts before publication claims. | Publication-mode solvers consume `RouteMatrix` cells and reject unvalidated fallback cells. |
| CP-003 | medium | `src/itinerary_system/utility_model.py`; `src/itinerary_system/data_enrichment.py`; `docs/current/current_score_audit.md` | Phase 0.3 utility missingness | Implemented explicit source masks, masked MCDA/TOPSIS utility behavior, deterministic source-ablation audit, and legacy `data_confidence` coverage alias. | Missing source families are masked out of utility denominators, with utility, coverage, and uncertainty reported separately. |
| CP-004 | high | `src/itinerary_system/research_artifacts.py`; `src/itinerary_system/repair_planner.py`; `src/itinerary_system/plans/` | Phase 1.0 parent/diff foundation | Implemented canonical ownership records, append-only plan storage, typed plan diffs, current demo parent, and additive repair scaffold lineage metadata; downstream ownership-aware optimization remains Phase 2.0. | Parent plans are append-only, owned constraints are explicit, and `PlanDiff` reports typed weighted edits. |
| CP-005 | high | `src/itinerary_system/repair_planner.py`; `src/itinerary_system/multi_objective_route.py` | Phase 2.0 progressive repair solver | Current repair is deterministic scaffold logic, not ownership-aware optimization with progressive neighborhoods. | Controller returns the smallest independently eligible repair radius and stores every attempt. |
| CP-006 | high | `src/itinerary_system/research_artifacts.py`; `docs/planning/travel_itinerary_repair_technical_specification.md` | Phase 3.0 evaluator and explanations | Phase 0 evaluator is useful but not a complete final-plan evaluator with certificates and evidence-grounded explanations. | Final plans receive independent certificates, and unsupported explanation claims fail closed. |
| CP-007 | medium | `src/itinerary_system/experiment_runner.py`; `notebook/production_system_blueprint.ipynb`; `scripts/run_phase0_evidence_pipeline.py` | Phase 4.0 pipeline and benchmark | Production flow remains notebook-oriented and lacks a canonical immutable run directory and benchmark runner. | `run_research_pipeline()` emits complete run artifacts and deterministic benchmark metrics on frozen inputs. |

## Current Validation Commands

```powershell
python -m ruff check --no-cache src tests scripts
python -m pytest tests/data/test_context_snapshot.py
python -m pytest
python scripts/run_project_checks.py
```
