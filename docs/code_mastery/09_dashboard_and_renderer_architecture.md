# Dashboard and Renderer Architecture

There are two frozen research UIs. They are not the future product dashboard.

## Modular Dashboard

### `map_exporter.py`

- **Why/category:** artifact writer/orchestrator for lightweight and modular map
  exports.
- **Caller:** production execution/map rendering.
- **Inputs/outputs:** materialized route/POI/config/evaluation artifacts in;
  static HTML, JSON/JS, route/POI GeoJSON, metrics, and asset report out.
- **State:** output directory owned by the export.
- **Invariant:** current generated contracts remain stable; selected sequence
  and artifact freshness are truthful.
- **Failure:** missing/stale artifacts render explicit diagnostics.
- **Tests/gate/state:** configurable-system and dashboard-contract tests,
  `validate_dashboard_export.py`; E3.C1/E3.M; current parity artifact.

### `dashboard_evaluation.py`

- **Why/category:** stateless evaluation payload/page component.
- **Caller:** `map_exporter` compatibility wrappers.
- **Inputs/outputs:** canonical evaluator evidence in; evaluation JSON/page out.
- **Invariant:** four canonical methods, correct directionality, escaped text,
  no invented rows.
- **Tests:** `tests/test_evaluation_dashboard_contract.py`.

### `dashboard_assets.py`

- **Why/category:** stateless exact CSS asset emitter.
- **Caller/output:** `map_exporter`; frozen stylesheet bytes.
- **State:** none.
- **Invariant:** legacy SHA-256 stays frozen.
- **Tests/gate:** configurable dashboard integration; E3.C1.

### `dashboard_data_loader.py`

- **Why/category:** UI controller asset that loads generated local JSON/JS.
- **Caller/output:** browser page; registered loader globals/state.
- **State:** one page.
- **Failure:** diagnostics rather than fabricated empty success.
- **Tests/gate:** browser/asset hash contracts; E3.C1.

### `dashboard_map_controls.py`

- **Why/category:** browser UI controller for Leaflet route selection, filters,
  zoom, and playback.
- **Inputs/outputs:** loaded route/POI records and DOM controls in; synchronized
  Leaflet layers out.
- **State:** one page; no optimizer truth.
- **Invariant:** route sequence and layer identity remain artifact-derived.
- **Tests/gate:** configurable-system/browser checks; E3.C1.

### `dashboard_ui.py`

- **Why/category:** browser UI controller for customer/research panels, mode
  switch, collapse/expand, metrics, details, hotels, and debug views.
- **Inputs/outputs:** loaded artifact state in; escaped DOM presentation out.
- **State:** one page.
- **Invariant:** null/failure/method/direction contracts stay truthful.
- **Tests/gate:** dashboard-contract and browser checks; E3.M/E3.C1.

See [dashboard loading](diagrams/dashboard_data_loading.md) and
[modular modules](diagrams/modular_dashboard_modules.md).

## Folium Renderer

### `blueprint_core.py`

- **Why/category:** package-owned compatibility/domain utility core migrated
  from the notebook.
- **Caller:** renderer/day-plan helpers and notebook aliases.
- **Inputs/outputs:** catalog/context/config artifacts in; normalized lookups and
  helper results out.
- **State:** stateless except explicit passed records.
- **Tests/gate:** six parity groups in `test_blueprint_core_parity.py`; E3.C2.

### `blueprint_day_plans.py`

- **Why/category:** service building profile/day-plan records.
- **Caller:** renderer sections.
- **Inputs/outputs:** context/profile inputs in; day-plan frames out.
- **State:** request-scoped.
- **Tests/gate:** renderer/day-plan frozen signature; E3.C3.

### `blueprint_render_primitives.py`

- **Why/category:** stateless low-level Folium/HTML/geometry utilities.
- **Caller:** layer/panel modules.
- **Invariant:** empty, duplicate, nonfinite, and escaping behavior stays
  compatible.
- **Tests/gate:** renderer edge/parity tests; E3.C3.

### `blueprint_render_layers.py`

- **Why/category:** UI rendering service for context, selected, comparison,
  overview, city, and route-matrix layers.
- **Caller:** renderer sections.
- **Inputs/outputs:** request state and route artifacts in; Folium layer objects
  and registry entries out.
- **State:** `RendererBuildState` owned by one render request.
- **Invariant:** Leaflet identities, ordering, default visibility, geometry, and
  debug rows match frozen contracts.
- **Tests/gate:** renderer parity; E3.C3.

### `blueprint_render_panels.py`

- **Why/category:** UI panel rendering/compatibility composition.
- **Caller:** renderer sections; contains a thin 2-line selector wrapper.
- **Outputs:** comparison/debug/day/summary panels.
- **Invariant:** selector responsibility stays in
  `blueprint_route_selector.py`.
- **Tests/gate:** legacy boundary and renderer parity; E3.C3/C4.

### `blueprint_route_selector.py`

- **Why/category:** UI controller asset split into validation/model,
  HTML/CSS, and client runtime.
- **Caller:** panel wrapper.
- **Inputs/outputs:** route registry and Leaflet layer names in; selector model,
  markup, and script out.
- **State:** browser page; no cross-render global Python state.
- **Invariants:** route order, control semantics, default checked/visible state,
  bounds, escaping, and layer references.
- **Failure:** duplicate IDs, missing layer variables, and mismatched defaults
  raise before broken controls emit.
- **Tests/gate:** `test_blueprint_route_selector.py`; E3.C4 verified.
- **Responsive rule:** desktop remains 430px at left 74; below 520px the open
  selector uses a 12px gutter and viewport-derived width.

### `blueprint_renderer_sections.py`

- **Why/category:** request-scoped runtime-state registry plus nine bounded
  composition sections.
- **Caller:** `blueprint_renderer.py`.
- **Inputs/outputs:** renderer context in; progressively populated
  `RendererBuildState` and final artifacts out.
- **State:** one render request; no module-global mutable renderer state.
- **Tests/gate:** state isolation and function-size tests; E3.C4.

### `blueprint_renderer.py`

- **Why/category:** 10-line workflow orchestrator.
- **Caller:** `map_renderer.py` and notebook compatibility facade.
- **Inputs/outputs:** context/output/routing flag in; final Folium map out.
- **Invariant:** exactly one ordered section pipeline.
- **Tests/gate:** renderer parity/orchestrator tests; E3.C4.

### `map_renderer.py`

- **Why/category:** package integration/compatibility boundary calling the
  package renderer without importing the legacy notebook module.
- **Tests/gate:** renderer parity and legacy boundary; E3.C3.

See [Folium modules](diagrams/folium_renderer_modules.md).

> **Beginner note / 初学者提示:** Frozen parity means “keep behavior exactly
> stable while reorganizing ownership.” It does not mean this is the preferred
> information architecture for new users.

## Additive Product Dashboard (E3.UX)

### `product_dashboard_models.py`

- **Why/category:** domain contracts for the product manifest, validated source
  bundle, versions, safe paths, finite JSON, and 23 truth states.
- **Caller:** adapter, renderer, validators, and tests.
- **Inputs/outputs:** run-relative paths and JSON in; immutable bundle/manifest
  records or validation errors out.
- **State/invariants:** stateless; rejects unsafe paths and non-finite values.
- **Tests/gate/status:** `tests/product_dashboard/`; E3.UX0/E3.UX4 verified;
  current. Read the adapter next.

### `product_dashboard_adapter.py`

- **Why/category:** service loading only manifest-declared canonical files.
- **Caller:** product renderer and focused tests.
- **Inputs/outputs:** derived run directory in; validated source bundle, source
  hashes, and truth states out.
- **State ownership:** copied artifacts own truth; the adapter owns no solver,
  evaluator, ranking, or plan state.
- **Invariants/failure:** checks content hashes and lineage, rejects missing
  required artifacts, preserves certificate mismatch as a UI state, and
  distinguishes exact-cap refusal from complete infeasibility.
- **Tests/gate/status:** adapter/security cases; E3.UX0/E3.UX4 verified;
  current. Read view models next.

### `product_dashboard_view_models.py`

- **Why/category:** stateless presentation utility shared by both modes.
- **Caller:** product renderer.
- **Inputs/outputs:** validated bundle in; timeline, repair, comparison,
  evidence, alternative, research, and map JSON out.
- **State/invariants:** null stays null, ineligible rows have no rank, exact
  failures keep reasons, metric owner/direction remains explicit, and map
  geometry is sampled only for display.
- **Tests/gate/status:** view-model tests; E3.UX2/E3.UX3 verified; current.
  Read assets next.

### `product_dashboard_assets.py`

- **Why/category:** semantic HTML/CSS plus read-only UI/map controllers.
- **Caller:** renderer writes its returned strings as versioned assets.
- **Inputs/outputs:** validated embedded data in; accessible customer/research
  DOM and synchronized selected-day map out.
- **State ownership:** browser-only selected day and disclosure mode.
- **Invariants/failure:** `textContent`/`createElement`, no optimizer/evaluator
  logic, no storage/eval/raw HTML, no UX5 actions, map text alternative, 44px
  targets, visible focus, and reduced motion.
- **Tests/gate/status:** render/security tests and six-width browser matrix;
  E3.UX2–E3.UX4 verified; current. Read renderer/export next.

### `product_dashboard_renderer.py` and product scripts

- **Why/category:** non-overwritable artifact writer and validator boundary.
- **Caller:** export, screenshot-registration, and validation scripts.
- **Inputs/outputs:** copied run snapshot in; `dashboard_product/` manifest,
  HTML/assets/screenshots, hashes, and top-level registration out.
- **State ownership:** derived run only; canonical source run is never edited.
- **Invariants/failure:** refuses overwrite, replaces manifests atomically, and
  validates hashes, security, lineage, and the read-only boundary.
- **Tests/gate/status:** 26 product tests, validators, readback, and browser
  evidence; E3.UX4 verified; current. Read the E3.UX0 audit next.

> **Beginner note / 初学者提示:** This dashboard is another view of existing
> evidence, not another solver. Python validates and prepares the view model;
> browser JavaScript only presents it.
