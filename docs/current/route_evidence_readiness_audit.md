# Route Evidence Readiness Audit

**Audit date:** 2026-07-20  
**Authoritative E2 run:** `tmp_test/research_pipeline_raw/e2-raw-production-20260711/production_legacy`  
**Frozen manifest:** `route_evidence_bundle_manifest.json`

## Result

The frozen real comparison is publication-ready for route-evidence use within its recorded California scope.

| Check | Frozen E2 result |
| --- | ---: |
| Bundle ID | `route_bundle_e910cf488994b7a2` |
| Request-set hash | `b5a66699c18d0b78` |
| Unique route requests | 42 |
| Road validated | 42 |
| Endpoint snap validated (≤100m) | 42 |
| Maximum endpoint snap distance | `85.99752963 m` |
| Missing route responses | 0 |
| Fallback route rows | 0 |
| Request/audit/cache-key alignment | passed |
| Cache freshness at freeze | passed |
| Reviewed provider provenance | passed |
| Publication ready | true |

The provider record binds local OSRM `v26.5.0`, the immutable GHCR image digest, the fixed dated Geofabrik California PBF and independently computed SHA-256, the bundled car profile/MLD flow, ODbL terms, localhost endpoint, reviewer, and review timestamp.

## Content-Addressed Matrix

The exact validated-cache SHA-256 `2bbd690b690cd75ef8708dc4256755b7c9662394b4fd1745e423d275ad3cda09` produces matrix `route_matrix_fde2c44a16a62ef3`. Its report checks all 42 cells and records:

- 42 present and road validated;
- zero missing, fallback, or invalid-value cells;
- source bundle `route_bundle_e910cf488994b7a2`;
- `publication_ready: true`.

Strict Phase 0 then passed with 42/42 road coverage and 3/3 independently eligible evaluations. The frozen raw plan inputs retained their pre-validation hashes.

## Snapshot Separation

`results/outputs` remains a different legacy/default artifact snapshot and must not be used to describe this E2 bundle. The context snapshot also retains five non-road fallback `route_options.csv` seed rows; they remain ineligible and are not the certified E2 source. All E2 and E3 route claims must resolve through the authoritative run, frozen manifest, and matrix hashes above.

## Gate Consequence

Execution gate E2 is closed. Route-time, distance, and feasibility evidence may now be consumed only through the frozen bundle and matrix lineage. This does not establish that one repair method is superior; that claim remains blocked until the real E3 paired benchmark passes its method-provenance, evaluator-eligibility, shared-input, and failure-retention checks.

## Reproduction Commands

```powershell
python scripts/freeze_route_evidence_bundle.py --output-dir "tmp_test/research_pipeline_raw/e2-raw-production-20260711/production_legacy" --provider-provenance "tmp_test/research_pipeline_raw/e2-raw-production-20260711/production_legacy/source-provenance.json" --expected-request-count 42 --require-publication-ready
python scripts/build_validated_route_matrix.py --input "tmp_test/research_pipeline_raw/e2-raw-production-20260711/production_legacy/production_road_route_cache.csv" --context-snapshot-id "context_static_demo_2026_06" --output-dir "tmp_test/research_pipeline_raw/e2-raw-production-20260711/production_legacy/routing" --route-evidence-manifest "tmp_test/research_pipeline_raw/e2-raw-production-20260711/production_legacy/route_evidence_bundle_manifest.json" --require-publication-ready
python scripts/run_phase0_evidence_pipeline.py --config "configs/default_trip_config.yaml" --output-dir "tmp_test/research_pipeline_raw/e2-raw-production-20260711/production_legacy" --quality-dir "tmp_test/research_pipeline_raw/e2-raw-production-20260711/production_legacy/quality" --skip-cache-build --require-final-eligible
```
