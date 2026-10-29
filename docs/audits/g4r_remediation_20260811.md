# G4R remediation evidence — 2026-08-11

Status: implementation checks passed and the independent re-audit subsequently passed. G4R is verified; G4 remains blocked. See `g4r_remediation_final_reaudit_20260811.md`.

## Corrected blockers

- Map readiness now fails closed unless MapLibre exposes tile-state APIs, every tile-bearing style source reports loaded, `areTilesLoaded()` is true, and canvas viewport coverage is at least 95%.
- Ready map containers expose `data-map-ready`, `data-map-ready-at`, and `data-map-tile-coverage` so captures can be ordered after readiness.
- Global navigation contains only Itinerary, Map, Repairs, Compare, and Evidence.
- Compare auto-selects the recommended eligible repair and provides an enabled `Choose option for Trip preview` action without enabling W5 acceptance.
- Text route output uses authoritative human stop names rather than `Stop unavailable` for known route endpoints.
- Copilot context shows the selected repair label rather than a raw plan ID.

## Verification

- Executable tile-readiness adversaries: passed.
- Focused frontend/map/Compare/accessibility contracts: 64 passed.
- JavaScript syntax checks: passed for app, map, map controller, Compare, and Copilot modules.
- Ruff and scoped diff checks: passed.
- Live Chromium verification: 12 scenarios passed with no console errors or failed network requests on the authoritative run.

The live run covered fresh load, reload, Compare and return to Trip, Copilot open/close, Evidence open/close, 1440→1024→1440 resizing, the text route panel, and Trip at 1440×900, 1280×800, 1024×768, and 390×844. All visible maps reported tile coverage `1.0`; map heights were 630px, 558px, 558px, and 592px respectively. Compare reported both map panes ready at coverage `1.0`. Screenshots were captured only after the corresponding `data-map-ready-at` timestamp.

Evidence and screenshot hashes are recorded in [live_evidence.json](../../results/product_app/g4r_remediation_20260811/live_evidence.json).

## Remaining gate conditions

- The Codex in-app Browser runtime could not initialize because its Windows sandbox helper failed before browser discovery. The final independent slices recorded this limitation and corroborated the local Chromium evidence with executable contracts, hashes, timestamps, and rendered-pixel checks.
- A broader cross-slice suite previously produced 73 passes and 5 coherent-fixture migration failures in evaluated-edit/POI API tests. Those failures are outside this focused UI remediation and remain unresolved; no contract was weakened to hide them.
- W5 Accept and Keep-original actions remain disabled. No G4/provider call was made.
