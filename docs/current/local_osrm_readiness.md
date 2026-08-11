# Local OSRM Readiness

**Status date:** 2026-07-20  
**Work package:** ROUTE-003 / E2  
**Status:** Installed, reviewed, healthy, and accepted for the frozen E2 bundle

## Deployment

- Engine: Project OSRM `v26.5.0`
- Image: `ghcr.io/project-osrm/osrm-backend@sha256:aa6a1de3a71dafffd0ba39340542524f66e6841fc19bf7874a0e6a7967837f56`
- Endpoint: `http://127.0.0.1:5000`
- Profile/algorithm: bundled `/opt/car.lua`, MLD
- Extract: `https://download.geofabrik.de/north-america/us/california-260713.osm.pbf`
- Extract size: `1,319,908,507 bytes`
- Extract SHA-256: `5a8313f8631a1ffceb837a6b3a5049f486bcbbdb0b786796ddb0df898565a85e`
- Preprocessed graph: 26 nonempty `california.osrm*` files
- Reviewer: `Ztang-Yit-Xiaang`
- Review timestamp: `2026-07-20T20:22:56.7797846Z`

Docker Compose binds only to localhost. The repository healthcheck returns OSRM `code: Ok`, and the container was running when the E2 evidence was frozen.

## Evidence Acceptance

The local provider resolved the complete 42-request set. All 42 rows pass road validation and the 100m endpoint-snap gate, with no missing or fallback rows. The reviewed provenance, requests, audit, and validated cache are frozen as bundle `route_bundle_e910cf488994b7a2`.

The resulting RouteMatrix `route_matrix_fde2c44a16a62ef3` is publication-ready for all 42 cells, and strict Phase 0 reports 3/3 eligible evaluations. ROUTE-003 and E2 are therefore closed for this exact frozen California bundle.

Large PBF and graph files remain ignored by Git. Reproduction requires the pinned image, fixed extract, checksum, provenance record, and content hashes; a healthy mutable or public endpoint alone is not equivalent evidence.
