# MAP-DEC-002 - Local Map Runtime Substitution

**Decision date:** 2026-08-04  
**Status:** accepted; W1M/G1 verified  
**Scope:** CP-010 product track only

## Decision

The primary V1 map runtime is locally hosted MapLibre GL JS plus the PMTiles browser protocol, a local style, glyphs, sprites, and a bounded California Coast PMTiles archive derived from a legally distributable OpenStreetMap/Protomaps build. Visible OpenStreetMap attribution and retained license/provenance records are mandatory.

Mapbox Atlas is a deferred, explicitly selected backup. It is disabled by default, never selected automatically, and is not a G1 or W2 dependency. If entitlement, `atlas:read`, licensed assets, and a runtime license later become available, Atlas must pass a separate provider preflight before use.

The public Mapbox token is not used, stored, logged, or returned by the MapLibre path.

## Status and gates

- Historical W1 remains `implemented`; its timestamped Atlas-oriented evidence is not rewritten.
- W1M and G1 are `verified`. The closed local package, provenance/license records, PMTiles coverage, live HTTP 206/CORS/security checks, Docker-internal no-egress replay, recovery behavior, browser shell, and six independent audits pass.
- W2/G2 are `verified`. Geographic MapLibre rendering, artifact-derived route/stop layers, WebGL checks, visible attribution, responsive browser evidence, regressions, and six independent audit categories pass. W3 is `ready`.
- CP-010 remains `in-progress`. E3.1, E3.3, E4, E3.UX5, and E5 do not change.

## Frozen runtime contract

- Default `PRODUCT_MAP_PROVIDER=maplibre_pmtiles`; optional `mapbox_atlas_v3` is explicit only.
- `PRODUCT_MAP_BASE_URL` must be an exact HTTP loopback URL on port 8080.
- Health schema is `product-health-v2` with non-core component `map`.
- Map configuration schema is `product-map-configuration-v2` and includes the PMTiles protocol script URL and provenance URL.
- The active provider alone is probed. The MapLibre probe requires local GL JS/CSS, PMTiles JS, a loopback-contained style, valid ODbL provenance, and HTTP 206 byte ranges.
- The textual route remains available when the map component is degraded.

## Verified G1 evidence

The durable closeout is recorded in
[`w1m_live_g1_verification_report.md`](../audits/w1m_live_g1_verification_report.md)
and
[`w1m_live_g1_evidence_manifest.json`](../audits/w1m_live_g1_evidence_manifest.json).
Do not cache `tile.openstreetmap.org`. Atlas remains unverified and may be used
only after its separate entitlement and provider preflight.
