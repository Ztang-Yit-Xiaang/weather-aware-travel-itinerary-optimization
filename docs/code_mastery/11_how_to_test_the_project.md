# How to Test the Project

**Collected on 2026-07-29 after E3.UX4:** `315 tests collected in 1.60s`.

Do not hardcode 315 in scripts or acceptance logic. It is a current observation
and will change when tests are added.

## Windows Temp-Root Note

The configured shared pytest root currently produced `WinError 5` in this
workspace. Use a unique repository-local base when that happens:

```powershell
$phaseTestTemp = "tmp_test\manual_pytest_$(Get-Date -Format yyyyMMdd_HHmmss)"
New-Item -ItemType Directory -Path $phaseTestTemp | Out-Null
python -m pytest --basetemp $phaseTestTemp
```

Do not reuse an existing `--basetemp`; pytest may remove its contents.

## Test Layers

| Layer and verified command | What it checks | Runtime class / expected output | Common failure classification | Next debugging command |
|---|---|---|---|---|
| 1. `python -m pytest --basetemp tmp_test\codex_phasea_pytest_20260729_019faf6d tests/test_blueprint_route_selector.py -q` | One focused selector contract | seconds; pass/fail per selector test | product code unless setup/path fails | `python -m pytest ... -vv -x` with a new unique base |
| 2. `python -m pytest --basetemp <unique-dir> tests/interaction -q` | Permission, probe, clarification, and continuation subsystem | seconds; all interaction tests pass | permission/product boundary | rerun the failing file with `-vv -x` |
| 2. `python -m pytest --basetemp <unique-dir> tests/repair tests/evaluation tests/explanation -q` | Repair/evaluator/explanation contracts | short-to-medium; failures name the owning subsystem | product code, route fixture, or stale expectation | rerun the failing node and inspect its fixture |
| 3. `python -m pytest --basetemp <unique-dir>` | Entire collected suite | medium; current collection is 315 and the verified run passed in 46.95s | environment if setup/temp permission; otherwise product/contract | `python -m pytest --basetemp <new-dir> -x -vv` |
| Collection: `python -m pytest --basetemp <unique-dir> --collect-only -q` | Importability and actual current count | seconds; ends with collected count | import/environment/collection | rerun failing module import directly |
| 4. `python -m ruff check --no-cache src tests scripts` | Syntax/style/import and selected quality rules | seconds; `All checks passed!` | product/source formatting | `python -m ruff check --no-cache <path>` |
| 5. `python scripts/run_project_checks.py` | Ruff, context snapshot test, and full pytest with classification summary | medium; writes `results/quality/project_check_summary.json` | explicitly labels environment, product code, timeout | open the summary, then rerun the failed command |
| 6a. `python scripts/validate_dashboard_export.py` | Frozen modular dashboard files, references, data/assets, and integrity | seconds; `Dashboard export validation PASSED.` | stale artifact, renderer/export drift, missing generated file | rerun the dashboard-producing path, then validator |
| 6b. `python scripts/validate_product_dashboard.py runs/e3ux-weather-repair-demo-v6` | Product schemas, source/asset hashes, safe paths, semantic/security/truth-state contracts, and disabled interaction | seconds; `Product dashboard validation PASSED.` | stale/tampered product artifact or code/manifest contract | inspect `dashboard_product/manifest.json`, then rerun the focused product test |
| Product focus: `python -m pytest --basetemp <unique-dir> tests/product_dashboard -q` | Adapter, view-model, rendering, security, screenshot, and UX5-exclusion contracts | seconds; 26 pass in the verified worktree | product adapter/view/render contract | rerun failing node with `-vv -x` |
| 7. PowerShell readback below | Manifest paths, row counts, eligibility/failure split, route lineage | seconds; 24/8/16 for immutable v14 | stale/malformed artifact or wrong root | inspect `closeout.json`, benchmark manifest, and first failing run manifest |
| 8. Browser matrix | Runtime DOM, resources, console, overflow, controls, responsive behavior | manual/automated medium; evidence per width | renderer/UI/resource defect | inspect console/network and smallest failing width |
| 9. `python -m pytest --basetemp <unique-dir> tests/interaction -q` | Deterministic fixture replay and authorization boundary | seconds | interaction contract | failing integration node with `-vv -x` |
| 10. `python -m pytest --basetemp <unique-dir> tests/benchmark/test_publication_contract.py tests/benchmark/test_publication_method_factory.py tests/benchmark/test_route_coverage.py -q` | Method set, retained failures, route/source lineage, rankability, coverage | seconds | research-contract violation | inspect the exact assertion and immutable manifest |

## Immutable v14 Readback

```powershell
$e3Root = "tmp_test\research_pipeline_raw\e3-real-production-20260725-optimized-v14-cap50000"
$closeout = Get-Content -LiteralPath "$e3Root\closeout.json" -Raw | ConvertFrom-Json
$closeout.publication_readiness | Format-List run_count,ranking_eligible_run_count,failed_run_count,publication_ready,all_runs_ranking_eligible
```

Expected current diagnostic:

```text
run_count                  : 24
ranking_eligible_run_count : 8
failed_run_count           : 16
publication_ready          : True
all_runs_ranking_eligible  : False
```

Interpretation: structurally complete evidence, incomplete four-method
performance comparison.

## Browser Checks

E3.C4:

- 1440px desktop;
- 390px mobile;
- selector natural expanded width;
- one intermediate tablet width;
- both Folium and modular dashboards.

E3.UX4 uses 1440, 1024, 768, 430, 390, and 360px. Check the product timeline,
selected-day map synchronization, customer/research switch, certificate,
failed exact alternative, source hashes, primary evidence action, focus,
44px targets, labels, console, pending resources, clipping, and document
overflow. See `docs/reports/product_dashboard_browser_matrix.md`.

See [test hierarchy](diagrams/test_hierarchy.md).

> **Beginner note / 初学者提示:** A green unit test proves only its asserted
> contract. It does not prove a browser layout, immutable real run, or
> publication claim unless that scope is actually tested.
