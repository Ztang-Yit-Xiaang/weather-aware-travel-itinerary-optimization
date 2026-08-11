# E2 Raw Comparison Status

**Audit date:** 2026-07-20  
**Run:** `e2-raw-production-20260711`

## Frozen Raw Comparison

The authoritative production command consumed the local Yelp business JSONL, the local hotel CSV, seven California cities, and `refresh_policy=never`. The run remains the authoritative raw parent; the later diagnostic regeneration was not adopted because current optimizer and workspace state had drifted.

The raw comparison preserves these truthful method outcomes:

| Method | Truthful status | Budget used | Total budget | Selected stops |
| --- | --- | ---: | ---: | ---: |
| Hierarchical Gurobi | `OPTIMAL` | 1890.97 | 2148.74 | 9 |
| Hierarchical Greedy | `HEURISTIC` | 2109.97 | 2148.74 | 13 |
| Bandit + Small Gurobi Repair | `HEURISTIC_FALLBACK` | 1918.97 | 2148.74 | 13 |

No selected route row contains `catalog pending` or `data_ingestion_needed`. The hybrid result intentionally retains its fallback status and must not be described as an optimal solve.

## Frozen Publication Identity

The 42-request route evidence is frozen against the authoritative run:

- Bundle ID: `route_bundle_e910cf488994b7a2`
- Request-set hash: `b5a66699c18d0b78`
- Request CSV SHA-256: `90a86924d2cdfa5093dd423240933c29be19f691cdb23525c2e65e95c8b0983c`
- Cache-audit SHA-256: `cb2f4576e09cb4334393853b113e1d220eeee10c458d7f34299875983e696b24`
- Validated-cache SHA-256: `2bbd690b690cd75ef8708dc4256755b7c9662394b4fd1745e423d275ad3cda09`
- Provider-provenance SHA-256: `fcb6328ad210261e2ff260cd337eeb44cf4b27e5a296b626abb7f7b33f13db65`
- Unique requests, road validated, and endpoint-snap validated: 42/42
- Maximum endpoint snap distance: `85.99752963 m` under the 100m gate
- Missing and fallback route rows: 0
- Requests/audit/cache-key alignment and freshness: passed
- Reviewed provider: local OSRM `v26.5.0`, reviewer `Ztang-Yit-Xiaang`

The frozen cache produces content-addressed matrix `route_matrix_fde2c44a16a62ef3`. Its validation report covers 42/42 cells with zero missing, fallback, or invalid-value cells.

## Strict Phase 0 Result

Phase 0 was rebuilt from the frozen raw parent while reusing the validated cache. It passed `--require-final-eligible`:

- Road-route cache coverage: 42/42
- Eligible evaluations: 3/3
- Strict comparison ready: `true`

The command also preserves a warning that the context snapshot's five seed `route_options.csv` rows are non-road fallback examples. They are not part of the certified E2 route source and must not be consumed by E3; E3 must bind to `route_matrix_fde2c44a16a62ef3`.

The frozen raw inputs were not regenerated or modified. Their SHA-256 values remained:

- `production_method_route_stops.csv`: `4e9ec01e57961c9bfb2703ae22cb95b05b86a15f8e030124949c8a13ba61fc62`
- `production_nature_route_stops.csv`: `8c3e89e23b66d38f3b2a2a20cf91ba6d410104242813e0b474903e9dfc07f434`
- `production_artifact_metadata.json`: `27e82527778b0b078c430856969239f9a64444fd7e59fd976eef30d59dee04ae`

## Gate Status

E2 is closed for this frozen California benchmark bundle. This certifies the route-evidence boundary, not method superiority. The next gate is the real E3 four-method paired run on identical parent, disruption, catalog, context, bundle, and matrix hashes.
