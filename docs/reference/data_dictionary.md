# Data Dictionary And Evidence Roles

This document records the first Phase 0 data vocabulary for the itinerary-repair
research path. It is intentionally narrow: enough to distinguish source data,
curated assumptions, simulated context, derived features, optimizer outputs, and
evaluation eligibility before broader adapter refactors.

## Research Framing

The current publication question is:

> How can a travel-planning system preserve a user-owned multi-day itinerary
> under weather, closure, mobility, or pace disruptions, while exposing truthful
> solver, routing, data, and evaluator evidence?

The project should therefore be evaluated as a user-owned itinerary repair
system, not as a generic tourism popularity predictor or an autonomous booking
agent.

## Core Identifiers

| Field | Role | Stable or time-sensitive | Notes |
| --- | --- | --- | --- |
| `catalog_snapshot_id` | Identifies stable POI, hotel, source, and annotation records | Stable | Example: `california_v1` |
| `context_snapshot_id` | Identifies weather, routing, closure, rate, or disruption context | Time-sensitive | Example: `context_static_demo_2026_06` |
| `run_id` | Identifies one resolved config, solver/evaluator run, and output set | Run-specific | Auto-generated from config hash when not provided |
| `plan_id` | Identifies immutable displayed itinerary content | Run-specific | Material post-processing must create a new plan |
| `source_run_id` | Links a plan to the planner run that produced it | Run-specific | Prevents claiming certification from an older run |

## Stable Catalog Tables

| Table | Required role |
| --- | --- |
| `poi_entities` | One row per stable real-world place identity |
| `poi_observations` | One row per source observation before fusion |
| `poi_features` | Derived planning features for a specific catalog snapshot |
| `feature_provenance` | Feature-level source and transformation traceability |
| `hotel_entities` | Stable lodging identities and planning priors |
| `source_audit` | Source role, license, redistribution, and limitation records |

## Context Tables

| Table | Required role |
| --- | --- |
| `weather_scenarios` | Weather observations, forecasts, or controlled scenarios with valid times |
| `route_options` | Route-leg or route-option records with explicit routing provenance |

## Coverage And Uncertainty

`source_coverage_score` replaces new uses of `data_confidence`. It measures how
complete the available sources are. It is not calibrated truth probability.

Use these fields separately:

| Field | Meaning |
| --- | --- |
| `identity_coverage` | Whether identity/source records are present |
| `location_coverage` | Whether coordinates are available and valid |
| `semantic_coverage` | Whether category/description semantics exist |
| `popularity_coverage` | Whether popularity/rating source coverage exists |
| `temporal_coverage` | Whether time-bound context exists |
| `routing_coverage` | Whether routing context exists |
| `model_uncertainty` | Model-level uncertainty when actually estimated |
| `annotation_uncertainty` | Human label disagreement or review uncertainty |
| `entity_match_confidence` | Entity-resolution confidence |

Legacy `data_confidence` may remain as a compatibility alias while older
dashboards and tests still use it. New research code should prefer
`source_coverage_score`.

## Routing Eligibility

Every movement should eventually be represented as a `RouteLegResult` with:

```text
routing_status
geometry_source
distance_source
duration_source
road_validated
fallback_used
fallback_reason
```

Final comparison eligibility requires solver feasibility, schedule feasibility,
snapshot validity, and road validation. A straight-line or geodesic fallback can
be displayed for debugging, but it is not a road-valid final evaluation route.

`production_road_route_cache.csv` is the Phase 0 handoff point for validated road
evidence. Rows may be generated from OSRM, a local road graph, or another
auditable routing backend, but they must not be marked valid unless the geometry,
distance, and duration are all road-derived.

Required or expected columns:

```text
origin_label
destination_label
geometry
distance_m
duration_s
provider
routing_profile
routing_status
geometry_source
distance_source
duration_source
road_validated
```

When this file is absent, Phase 0 exports keep route legs as fallback geodesic
proxies. When it is present and covers all route movements, strict Phase 0
validation can pass.

`production_road_route_cache_audit.csv` records every requested leg, its OSRM
cache key, cache path, status, and whether it was converted into a validated
road-route row. A missing or invalid cache response belongs in this audit, not
in the validated cache file.

`production_road_route_requests.csv` is the offline handoff manifest for route
evidence collection. It records each requested leg's labels, coordinates, cache
key, target cache file, and OSRM route URL. Use it to review route coordinates
and point an approved local or external routing service at the exact legs needed
for strict Phase 0 validation.

`production_legacy_route_cache_audit.csv` compares the current request manifest
against the older `production_road_route_cache.json` path cache, if present. It
is an audit only: legacy path geometry is useful for debugging, but it is not
promoted to strict Phase 0 evidence unless geometry, distance, and duration
provenance are all present.

The builder is offline by default and only reads existing
`open_osrm_route_<key>.json` files. `--fetch-missing` is an explicit network
mode that writes those JSON responses first. Fetch mode defaults to
`http://127.0.0.1:5000` so route coordinates stay inside a local or pinned
routing environment unless an endpoint is supplied explicitly. Public OSRM use
requires an explicit approval flag and is not the preferred publication path.
Use `scripts/check_route_source.py` for a no-network manifest/policy precheck,
or add `--probe` after a local or approved endpoint is running.

Phase 0 exports summarize route evidence in two places:

| Artifact | Coverage fields |
| --- | --- |
| `production_phase0_dataset_validation.json` | `route_cache_coverage.road_route_requested_leg_count`, `road_route_validated_leg_count`, `road_route_missing_leg_count`, `road_route_validation_coverage`, cache/audit hashes |
| `production_phase0_evidence_summary.csv` | `route_validated_leg_count`, `route_fallback_leg_count`, `route_road_validation_coverage`, plus the cache coverage fields above |

## Immediate Fix Scope

The first committed fix is `data/snapshots/california_v1`, a small fallback
snapshot that avoids private Yelp raw files and ignored `results/outputs` files.
It is sufficient for clean-clone validation and Phase 0 contract tests, but it is
not a comprehensive benchmark.
