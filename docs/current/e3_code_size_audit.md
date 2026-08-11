# E3 Code-Size and Responsibility Audit

**Audit date:** 2026-07-29  
**Scope:** `src/itinerary_system/**/*.py` after E3.C4 implementation  
**Purpose:** identify excessive code concentration without confusing size with a
correctness failure.

## Review Heuristics

These are audit triggers, not automatic defects:

- review a package module above 2,000 physical lines;
- review a function above 300 physical lines;
- treat a large function as higher risk when it owns UI state, artifact writes,
  or many mutable local variables;
- do not split cohesive code merely to satisfy a line count; require a clear
  responsibility and a parity/rollback gate.

## Current Findings

| Path / symbol | Size | Finding | Gate |
|---|---:|---|---|
| `src/itinerary_system/experiment_runner.py` | 4,208 lines | Aggregate experiment orchestration remains the largest package module. Its largest function is 334 lines; the module needs a later responsibility audit, but changing it is outside renderer-only E3.C3. | Record as an E3.D candidate; do not mix with D1 or benchmark semantics. |
| `src/itinerary_system/map_exporter.py` | 2,429 lines | Above the module review threshold, but E3.C1 reduced `_write_full_dashboard()` to 138 lines and extracted the large CSS/JS/evaluation assets. Its largest remaining function is 173 lines. | Reassess after E3.C4; no new monolithic function is presently blocking. |
| `src/itinerary_system/blueprint_render_panels.py::_add_route_debug_controls()` | 2 lines | Compatibility composition wrapper only; model, HTML/CSS, and client runtime are package-owned by `blueprint_route_selector.py`. | E3.C4 size target met. |
| `src/itinerary_system/blueprint_renderer.py::build_production_trip_map()` | 10 lines | Ordered composition over request-scoped state and nine named sections. | E3.C4 size target met. |
| `src/itinerary_system/blueprint_renderer_sections.py` | 1,152 lines / 9 functions; maximum 172 lines | The module is below the module review trigger and every section has one current rendering responsibility. All 43 state fields have downstream reads; no module-global mutable state exists. | Retain; further grouping would add speculative nested state or string-keyed containers. |
| `src/itinerary_system/blueprint_route_selector.py` | 944 lines / 8 functions; maximum 139 lines | Most physical lines are the existing CSS/JavaScript payload. The six-line responsive addition is the reviewed E3.C4 containment boundary. Splitting those assets again would add files and loading indirection without reducing runtime behavior. | Retain under exact render and selector edge tests. |
| `src/itinerary_system/blueprint_render_layers.py::_add_route_matrix_layers()` | 481 lines | Above the function review threshold, but isolated behind exact rendered-output parity. | Inspect during E3.C4; split only if responsibility boundaries remain acyclic. |
| `notebook/blueprint_trip_map.py` | 919 lines / 12 functions | No longer owns the 57-function renderer closure. It is a compatibility facade plus 12 out-of-closure legacy helpers. | E3.C3 verified; retain until a separate dead-code/caller audit. |

## E3.C4 Implementation Evidence

- Five package modules own 57 renderer functions and 12 constants exactly once.
- `map_renderer.py` has no `blueprint_trip_map` import.
- The notebook facade has no duplicate migrated definition and imports
  standalone with an empty `PYTHONPATH`.
- The normalized Folium HTML hash, day-plan hash, normalized route-debug hash,
  row counts, schemas, and Leaflet object counts match the pre-migration
  baseline.
- Renderer edge tests cover empty/duplicate geometry, NaN/Inf, missing
  artifacts, HTML escaping, cached/offline routing, and the package integration
  call.
- The normalized HTML SHA-256 remains
  `a2fbbb85c56019ccd5f64315cbe536965a14c7858c9a2bcc971b77d96e27c320`;
  the day-plan and normalized route-debug hashes remain
  `6bb4a3a40d76a07ba62e02bf055fd40fece8853b09fa76be07cffe16b7f88e27`
  and `b723926ceb77887660a7730104e3e2ebc891ca668ab33f4b43fe9d8f9444ff7e`.
- Twenty-three focused selector, state, renderer, ownership, and core parity tests
  pass.
- Eighty-one evaluator/benchmark/interaction regressions pass.
- Ruff, 5 context tests, project checks, and 289 full tests pass.
- Dashboard validation passes.
- Browser interactions pass for the Folium map and modular dashboard at
  1440/768/520/390px. The old 390px selector measured left 74/right 504; the
  same-worktree render measures left 12/right 378.4 with no clipped control,
  document overflow, console warning/error, incomplete image, or pending load
  state.
- The `minimal-implementation` / `karpathy-guidelines` simplification pass
  removed two write-only renderer-state fields, a redundant docstring,
  duplicate test assertions, and two empty browser-command artifacts. Every
  remaining state field and extracted helper has a current caller or
  downstream consumer.

## Decision

E3.C4 is `verified`. Its code-size, ownership, semantic/data parity, edge-test,
project-check, responsive containment, and desktop/mobile interaction
requirements pass. The E3.C3 full-HTML signature is retained as historical
entry evidence; the reviewed E3.C4 signature is the new frozen Folium baseline.

The parent E3.C gate remains `in progress`. The `experiment_runner.py`,
`map_exporter.py`, and 481-line `_add_route_matrix_layers()` findings remain
visible follow-up candidates; they must not be silently folded into E3.C4 or
used to change D1/E3 benchmark semantics.
