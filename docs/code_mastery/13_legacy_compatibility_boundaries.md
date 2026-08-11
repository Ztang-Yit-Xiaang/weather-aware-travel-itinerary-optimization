# Legacy Compatibility Boundaries

## Current vs Compatibility-Only

| Boundary | Current owner | Compatibility surface | Rule |
|---|---|---|---|
| Pipeline execution | `pipeline_runner.py` | thin notebook/CLI wrappers | No orchestration/business logic in notebook |
| Blueprint core | `blueprint_core.py` | aliases in `notebook/blueprint_trip_map.py` | Alias identity and parity must hold |
| Folium renderer | package `blueprint_*` modules | notebook facade and `map_renderer.py` call boundary | No legacy notebook import in package renderer |
| Modular dashboard | `map_exporter.py` + `dashboard_*.py` | generated HTML/assets | Frozen bytes/hashes/interactions |
| Plan artifacts | `PlanArtifactV2` | v1 migration defaults | Preserve migration, but new work emits v2 |
| Legacy flat production outputs | canonical run adapters | `production_legacy` files | Compatibility inputs only; manifest remains authority |

## Frozen UI Contracts

Preserve:

- Folium HTML normalized hash;
- day-plan and route-debug signatures;
- route ordering/defaults/layer references;
- modular CSS/JS hashes;
- `evaluation.html` and evaluation metric contracts;
- existing browser interaction behavior.

The product dashboard must use a new `dashboard_product/` path. It must not
“modernize” these artifacts in place.

## What Legacy Does Not Mean

Legacy does not mean unused or safe to delete. It means a compatibility
boundary with current callers/tests whose behavior is frozen until an explicit
migration and rollback gate.

> **Beginner note / 初学者提示:** A thin wrapper is useful when it protects
> existing callers while moving ownership. A second implementation is harmful
> because the two sources can drift.

