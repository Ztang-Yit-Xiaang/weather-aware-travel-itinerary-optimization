# Debugging Playbook

## 1. Classify Before Editing

| Symptom | Likely class | First evidence |
|---|---|---|
| `PermissionError` under pytest temp | environment | path in traceback and `tests/test_project_checks.py` classification |
| Import/collection failure | environment or module boundary | `python -m pytest --collect-only -q` |
| Hash/parity mismatch | renderer or generated artifact drift | owning parity test and generated file hash |
| Missing/mismatched plan ID/hash | stale or malformed artifact | run manifest, plan JSON, repository index |
| Route marked invalid/missing | route data/evidence | route matrix validation report and source bundle |
| Ineligible certificate | product/research contract, not necessarily solver bug | certificate failures/warnings |
| Null metric rendered as zero | UI integrity defect | raw certificate/metrics JSON and dashboard contract test |
| Exact method cap refusal | known E3.1 research blocker | planner run failure reasons and candidate lower bound |
| Browser overflow/clipping | UI layout defect | smallest failing viewport and computed bounding box |

## 2. Narrow Reproduction

```powershell
$debugTemp = "tmp_test\debug_pytest_$(Get-Date -Format yyyyMMdd_HHmmss)"
New-Item -ItemType Directory -Path $debugTemp | Out-Null
python -m pytest --basetemp $debugTemp path\to\test_file.py::test_name -vv -x
```

Do not delete or reuse an unknown temp root.

## 3. Trace Artifact Truth

1. open `manifest.json`;
2. follow only manifest-declared relative paths;
3. compare plan IDs and content hashes;
4. inspect planner requested/executed method IDs;
5. inspect route matrix/source bundle IDs;
6. inspect certificate plan hash, status, failures, and comparison eligibility;
7. inspect `PlanDiff`;
8. inspect explanation evidence references;
9. only then inspect UI rendering.

## 4. Route Debugging

Run focused matrix/evidence tests:

```powershell
python -m pytest --basetemp <unique-dir> tests/routing/test_route_matrix.py tests/routing/test_evidence_bundle.py -q
```

Never “fix” a missing road route by labeling geodesic fallback as validated.

## 5. Renderer Debugging

```powershell
python -m pytest --basetemp <unique-dir> tests/test_blueprint_route_selector.py tests/test_blueprint_renderer_parity.py tests/test_legacy_blueprint_boundary.py tests/test_blueprint_core_parity.py -q
python scripts/validate_dashboard_export.py
```

If a frozen hash changes, identify the exact output difference. Do not update
the oracle merely because the new output looks plausible.

## 6. Benchmark Debugging

Distinguish:

- evidence completeness;
- method execution success;
- independent eligibility;
- ranking eligibility; and
- exact-search completion.

A failed row is normally retained evidence, not something to delete.

## 7. Interaction Debugging

Check `test_only`, session permission, parent hash, plan repository contents,
and whether an immutable continuation run exists. A probe without a certificate
must never appear in accepted plans.

## 8. When to Stop

Stop on parent mutation, lineage/hash loss, method-identity loss, unsupported
fallback validity, metric-owner confusion, nonfinite UI data, ranked ineligible
output, false exactness, ungrounded explanation, permission leakage, hidden
legacy parity drift, or overwritten immutable evidence.

