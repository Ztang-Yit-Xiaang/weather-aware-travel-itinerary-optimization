# Itinerary Repair Copilot Map and Artifact-Integrity Audit

**Audit date:** 2026-08-03

**Audit role:** Independent read-only map/artifact specialist

**Scope:** Registered v6 parent/child plans, route matrix, diff, evaluation
certificate, and their use by the current `/app`.

**Gate verdict:** **BLOCKING — G2, G3, and G5 fail.**

## Method

The auditor cross-read `configs/product_app_registry.json`, both registered plan
JSON files, `routing/route_matrix_68ab535465b06808.json`, the diff, certificate,
and product service/frontend. IDs, hashes, coordinate order, route coverage, and
browser presentation were checked without modifying artifacts.

## Confirmed artifact foundation

- Parent and child each contain nine coordinate-bearing stops.
- The route matrix contains eight complete road-validated segments per plan.
- Parent/child geometry contains 961/967 points respectively, with no recorded
  fallback or missing segment in this fixture.
- Primary plan, diff, and certificate identifiers are mutually consistent in the
  inspected happy-path artifacts.

These facts show that geographic evidence exists; they do not prove `/app` uses
or validates it correctly.

## Findings

| ID | Severity | Gates | Evidence | Required closure |
|---|---|---|---|---|
| MAP-001 | Critical | G2 | `static/js/app.js` discards geographic evidence and draws an SVG schematic. | Serve validated GeoJSON and render it in the approved map runtime. |
| MAP-002 | Critical | G2 | Route points are stored as latitude/longitude (for example `[34.101632, -118.326901]`), while GeoJSON requires longitude/latitude. | Normalize explicitly at the backend boundary and test known California bounds. |
| MAP-003 | Critical | G2 | Stop features are omitted and original/repaired routes are presented from effectively identical schematic points. | Expose stops and distinct plan geometry with changed/selected states. |
| MAP-004 | Critical | G5 | Compare presents benchmark method rows as alternatives without proving distinct plan IDs and hashes. | Require immutable plan identity and hash uniqueness for every alternative. |
| MAP-005 | Critical | G5 | The Keep-original card's `Inspect option` path calls the durable keep action. | Separate inspection from decision mutation and add a no-side-effect test. |
| MAP-006 | High | G2/G5 | Runtime checks do not fully revalidate plan/diff/certificate/route lineage as one matrix. | Add cross-artifact schema, hash, lineage, and route-coverage validation. |
| MAP-007 | High | G2/G5 | Fallback, missing, sampled, and route-validation states are not comprehensively surfaced. | Fail closed and expose every route-evidence state in Map and Evidence. |
| MAP-008 | High | G5 | Missing preservation data can be rendered as `0%`, collapsing null into zero. | Preserve `Unavailable` end to end and test null rendering. |
| MAP-009 | High | G3/G5 | No validated segment/alternative selection contract drives map, Compare, and Evidence together. | Add typed selection IDs and revision-tested synchronization. |

## Closure evidence

G2 requires known-coordinate, route-coverage, WebGL/Atlas, and textual-fallback
tests. G3 requires selection-to-draft-to-preview integration. G5 requires three
distinct plan hashes, null-preserving metrics, artifact lineage verification,
and proof that inspection is non-mutating.
