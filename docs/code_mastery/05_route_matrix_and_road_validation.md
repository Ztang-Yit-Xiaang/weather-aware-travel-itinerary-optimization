# Route Matrix and Road Validation

## `RouteMatrix`

1. **Why:** give optimizer, evaluator, and renderer one source for travel
   duration/distance/provenance.
2. **Category:** domain model with validation service behavior.
3. **Called by:** day-route solver, exact baselines, evaluator, benchmark
   coverage, pipeline.
4. **Inputs:** content-addressed `RouteMatrixCell` records keyed by stable entity
   IDs.
5. **Outputs:** strict cells/totals, validation reports, serialized record.
6. **State ownership:** immutable matrix object and run routing artifact.
7. **Invariants:** missing cells explicit; fallback is not publication eligible;
   identity travel may be valid zero.
8. **Failures:** typed missing/not-publication-eligible errors.
9. **Tests:** `tests/routing/test_route_matrix.py`.
10. **Gate:** E2.
11. **State:** current for the frozen corridor.
12. **Read next:** `routing/evidence_bundle.py`.

## Route Evidence Bundle

- **Why/category:** artifact writer/validator tying requests, provider
  provenance, endpoint snaps, cache rows, freshness, and hashes together.
- **Caller:** freeze/build scripts and strict matrix construction.
- **Inputs/outputs:** request/cache/provenance artifacts in; bundle manifest and
  publication-readiness report out.
- **State:** immutable bundle directory.
- **Invariants:** request/cache keys align, snaps exist, provider provenance is
  approved, cache is fresh, and no fallback is hidden.
- **Failures/tests:** incomplete/tampered/stale bundles fail closed;
  `tests/routing/test_evidence_bundle.py`.
- **Gate/state:** E2; current.

## Provider Order

```text
validated pinned context cache
-> local private OSRM
-> explicitly approved remote OSRM
-> geodesic fallback (demo only, road_validated = false)
```

## Frozen E3 Route Evidence

- bundle: `route_bundle_a60c80047098a3b6`
- matrix: `route_matrix_68ab535465b06808`
- reported cells: 223/223 road- and endpoint-snap-validated, no fallback

This proves coverage only for the frozen E3 universe, not every possible trip.

> **Beginner note / 初学者提示:** A route line drawn on a map is not proof that
> the optimizer used road travel. Provenance, matrix cell, and sequence must
> agree.

