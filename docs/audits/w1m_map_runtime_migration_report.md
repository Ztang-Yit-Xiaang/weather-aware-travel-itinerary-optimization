# W1M Local Map Runtime Migration Evidence

**Recorded:** 2026-08-04  
**Decision:** MAP-DEC-002  
**Implementation status:** implemented, not verified  
**G1 status:** blocked  
**W2 status:** planned, not ready

## Outcome

Option 2 is now the primary runtime contract: loopback-served MapLibre GL JS, the PMTiles browser protocol, a local style/glyph/sprite package, and a bounded OpenStreetMap-derived Protomaps archive. Mapbox Atlas remains an explicitly selected licensed backup and is never an automatic fallback. The MapLibre path does not use a Mapbox token or license.

The deterministic foundation is complete. It includes provider-neutral health/configuration v2, exact launcher-version matching, loopback and Origin/CSP/CORS controls, a digest-pinned read-only Nginx boundary, strict provenance and closed asset/license manifests, exact PMTiles CLI evidence, all-registered-workspace coverage, local glyph/font-stack enforcement, and truthful degraded recovery.

This report does not verify G1. The real external asset package has not been staged, so live HTTP 206, CORS, browser asset loading, and disconnected-internet operation have not been demonstrated. The geographic renderer remains W2 work.

## Deterministic verification

| Check | Result |
|---|---|
| Product application and `PlanRepository` suite | 119 passed; one known Starlette deprecation warning |
| Full repository pytest | 430 passed; one known Starlette deprecation warning |
| Ruff | Passed |
| Project checks | Passed |
| MapLibre Compose static configuration | Passed; Docker client-config access warning only |
| JavaScript syntax | Passed |
| Diff whitespace check | Passed; existing line-ending notices only |
| Adversarial PMTiles preflight | 11 passed, including missing glyphs, evidence mismatch, CLI failure, and exact-version rejection |
| Preserved legacy state | Pointer and three decision hashes unchanged |

Commands:

```powershell
python -m pytest tests\product_app tests\plans\test_repository.py -q --basetemp <disposable-root>
python -m ruff check src\itinerary_system\product_app tests\product_app scripts\run_product_app.py scripts\validate_local_map_assets.py
python scripts\run_project_checks.py
docker compose --env-file docker\maplibre\.env.example -f docker\maplibre\docker-compose.yml config --quiet
node --check src\itinerary_system\product_app\static\js\app.js
git diff --check
```

## Independent deterministic audits

| Auditor | Scope | Verdict |
|---|---|---|
| `maplibre_architecture_audit` | Runtime/API/launcher contracts and W2 boundary | PASS after stale-version repair |
| `maplibre_source_research` | Provenance, PMTiles validity, glyphs, licensing, coverage | PASS after exact CLI/evidence repairs |
| `maplibre_gate_audit` | Phase, gate, dependency, and research status | PASS |
| `maplibre_security_audit` | Loopback, Host/Origin/CSP/CORS, secrets, fail-closed behavior | PASS |
| `maplibre_journey_audit` | Launcher/recovery/frontend/textual-fallback journey | PASS |
| `maplibre_content_audit` | Attribution, documentation, and evidence truthfulness | PASS after closeout corrections |

All auditors were independent, read-only agents. Findings were not averaged; each blocking finding was repaired and returned for re-audit.

## Remaining G1 work

1. Stage the pinned external package outside Git using `docker/maplibre/README.md`.
2. Run the strict preflight with PMTiles CLI 1.30.0 and preserve its output hashes.
3. Confirm the pinned Nginx digest pulls on the active Linux platform.
4. Start the local service and verify health, assets, exact CORS, and PMTiles HTTP 206 range behavior.
5. Repeat the checks with internet disconnected.
6. Run live Map/Artifact, Security, Journey, and Phase/Gate audits.

Until those steps pass, G1 remains blocked and W2 must not start. CP-010 remains in progress. E3.1, E3.3, E4, E3.UX5, and E5 are unchanged.
