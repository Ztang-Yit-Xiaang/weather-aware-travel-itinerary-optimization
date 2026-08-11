# Local OSRM Candidate Approval

**Prepared:** 2026-07-13  
**Reviewed and accepted:** 2026-07-20  
**Status:** Approved, installed, and bound to the frozen E2 route bundle  
**Reviewer:** `Ztang-Yit-Xiaang`

## Approved Reproducible Deployment

| Component | Approved value |
| --- | --- |
| Routing engine | Project OSRM `v26.5.0` |
| Immutable image | `ghcr.io/project-osrm/osrm-backend@sha256:aa6a1de3a71dafffd0ba39340542524f66e6841fc19bf7874a0e6a7967837f56` |
| Extract | `https://download.geofabrik.de/north-america/us/california-260713.osm.pbf` |
| Extract size | `1,319,908,507 bytes` |
| Extract SHA-256 | `5a8313f8631a1ffceb837a6b3a5049f486bcbbdb0b786796ddb0df898565a85e` |
| Data license | OpenStreetMap ODbL 1.0 |
| Endpoint | `http://127.0.0.1:5000` |
| Profile/algorithm | bundled `/opt/car.lua`, MLD |
| Review timestamp | `2026-07-20T20:22:56.7797846Z` |

Authoritative sources retained by the approval record:

- Project OSRM releases: https://github.com/Project-OSRM/osrm-backend/releases
- Project OSRM repository and Docker quick start: https://github.com/Project-OSRM/osrm-backend
- Geofabrik California extracts: https://download.geofabrik.de/north-america/us/california.html
- ODbL 1.0: https://opendatacommons.org/licenses/odbl/1-0/

## Completed Acceptance Sequence

1. Docker Desktop and WSL2 integration were enabled.
2. The immutable OSRM image digest was resolved and used by Compose.
3. The fixed California PBF was downloaded and independently hashed.
4. `osrm-extract`, `osrm-partition`, and `osrm-customize` completed.
5. Local `osrm-routed` passed the repository healthcheck.
6. The provider record was reviewed and signed.
7. All 42 requested routes passed road and 100m endpoint-snap validation.
8. Bundle `route_bundle_e910cf488994b7a2` froze with `publication_ready: true`.
9. Matrix `route_matrix_fde2c44a16a62ef3` passed 42/42 strict cell validation.
10. Strict Phase 0 passed with 3/3 eligible evaluations.

This approval applies only to the exact digest, extract checksum, provenance, route bundle, and matrix lineage recorded above.
