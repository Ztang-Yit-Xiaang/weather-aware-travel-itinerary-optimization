# Product Dashboard Artifact Integrity Report

**Status:** `verified`  
**Run:** `e3ux_weather_repair_demo_v6`  
**Manifest schema:** `product-dashboard-manifest-v1`  
**Product version:** `1.0.0`

## Boundary

The exporter created a new, non-overwritable snapshot from
`benchmark_158cf6d48be8`. The run manifest records only the source run ID and the
source-manifest SHA-256; it does not embed a host filesystem path. Source
artifacts are copied under run-relative paths and never written back.

The product manifest registers:

- canonical artifact references and full source SHA-256 values;
- parent and child IDs/content hashes;
- diff, certificate, explanation, route-matrix, and method identities;
- active truth states;
- HTML, CSS, data, UI, map, and screenshot SHA-256 values;
- the read-only compatibility boundary.

## Integrity checks

- Safe path validation rejects absolute paths and `..` traversal.
- Parent and child content hashes are recomputed and compared.
- Parent/child/diff/explanation lineage is checked before rendering.
- Certificate plan/diff mismatch becomes an explicit danger state.
- JSON root shape, JSONL records, and every numeric value are validated; NaN
  and positive/negative infinity are rejected.
- Embedded JSON escapes script-breaking characters.
- JavaScript uses `textContent` and `createElement`; forbidden raw insertion,
  storage, evaluation, and document-write APIs are rejected by the validator.
- Ineligible alternatives cannot receive a rank.
- Exact-cap diagnostics are copied from canonical planner-run records instead
  of guessed from empty benchmark summary fields.
- Product source/asset hash readback and the legacy dashboard validator pass.

## Final screenshot hashes

| Asset | SHA-256 |
|---|---|
| `screenshots/product_dashboard_desktop_1440.png` | `5a5f9475c823e8ecbe3ded78b8ca9e24898e47c79878bc0a1f5c7e423b7d5810` |
| `screenshots/product_dashboard_mobile_390.png` | `9662a2e3495b8696e6f2b00a9d4deaf3f1596f6ec9f216f6e183e902d0600a8a` |
| `screenshots/product_dashboard_evidence_390.png` | `a04c9b79f3c6e30c18d63a64d8181d7e34bedf03bf1e521711fd61fc1f1fbf97` |

## Rollback

Stop selecting or distribute-delete the derived v6 snapshot. The legacy
Folium, modular dashboard, evaluation export, optimizer, evaluator, benchmark,
route model, plan repository, and source run require no rollback because the
product exporter did not modify them.
