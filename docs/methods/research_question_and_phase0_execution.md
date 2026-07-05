# Research Question And Phase 0 Execution

## Document Check Summary

The roadmap, appendix, dataset audit, routing audit, novelty memo, and repair
method note converge on the same direction:

> Repair, do not regenerate: preserve a user-owned multi-day itinerary under
> disruptions with truthful data, solver, route, evaluation, and explanation
> evidence.

The project already has useful route generation, weather/nature scoring,
Gurobi/heuristic paths, dashboard exports, and a repair scaffold. The current
research blocker is not route generation. The blocker is that generated demo
artifacts are not yet separated from publication-grade evidence.

## Current Problems To Fix First

1. Clean clones depend on ignored `dataset/` or generated `results/outputs`
   files.
2. Stable catalog data and time-sensitive context are not separately identified.
3. `data_confidence` is used like uncertainty even though it is source coverage.
4. Route geometry can fall back to straight/geodesic approximations without a
   route-leg provenance contract.
5. Solver feasibility, road validation, and final comparison eligibility are not
   separate gate states.
6. Planner attempts and displayed plans need canonical run, plan, and evaluation
   records before broader experiments.

## First Fix Implemented

The first fix is the clean fallback data foundation:

```text
data/snapshots/california_v1/
```

It contains stable catalog tables and static context tables with separate
`catalog_snapshot_id` and `context_snapshot_id` values. It excludes private Yelp
raw records and marks demonstration routing rows as non-road-validated.

## Phase 0 Direction

Phase 0 should now proceed in this order:

1. Use `catalog_snapshot_id`, `context_snapshot_id`, and `run_id` in metadata and
   outputs.
2. Use `RouteLegResult` and `RouteResult` to gate road validation.
3. Use `PlannerRun`, `PlanArtifact`, and `ResearchEvaluationReport` to separate
   solver attempts, immutable plans, and independent eligibility.
4. Treat post-solve mutation as a new child run/plan/evaluation before using it
   in research comparisons.
5. Add route-leg audit output and wire these contracts into the production
   pipeline incrementally.

## Phase 0 Evidence Outputs

The production comparison refresh now writes these Phase 0 evidence files next
to the existing `production_method_*` artifacts:

```text
production_phase0_dataset_validation.json
production_phase0_planner_runs.csv
production_phase0_plan_artifacts.jsonl
production_phase0_route_audit.csv
production_phase0_evaluation_reports.csv
production_phase0_evidence_summary.csv
```

These files intentionally keep current geodesic/straight-line route legs
ineligible for final comparison unless an artifact row carries explicit
`road_validated=true` evidence. This lets the project continue demo/dashboard
work while preventing publication metrics from silently inheriting unvalidated
route geometry.

For the current project diagnosis, use the one-command evidence pipeline:

```powershell
python scripts/run_phase0_evidence_pipeline.py --output-dir results/outputs --quality-dir results/quality
```

It rebuilds the road-route cache audit from available OSRM cache files, exports
the Phase 0 tables, runs the validator, and writes the readiness summary. A
passing non-strict run means the artifacts are internally honest; check
`Strict comparison ready` and the road-route coverage line to see whether the
project is ready for publication-grade comparison.

For final comparison tables, use strict mode:

```powershell
python scripts/run_phase0_evidence_pipeline.py --output-dir results/outputs --quality-dir results/quality --require-final-eligible
```

Strict mode should fail until every compared method has road-validated route
legs and an eligible evaluation report.

Use the validator before treating these artifacts as paper evidence:

```powershell
python scripts/validate_phase0_artifacts.py --output-dir results/outputs
```

For final comparison tables, add `--require-final-eligible`; this strict mode
should fail until all included plans have road-validated route audit rows.

To make strict mode pass, provide `production_road_route_cache.csv` beside the
method artifacts. The Phase 0 exporter will use matching road-validated cache
legs before falling back to geodesic proxies.

If OSRM response JSON files already exist in the project cache, build the road
cache with:

```powershell
python scripts/build_road_route_cache.py --output-dir results/outputs --require-complete
```

The command also writes `production_road_route_cache_audit.csv`, so missing or
invalid cached route responses are visible instead of silently becoming
validated route legs.

It also writes `production_road_route_requests.csv`, an offline manifest with
the exact route-leg coordinates, cache keys, target cache paths, and OSRM URL
shape. Use that file to review the 21 current missing legs and to route them
through an approved local or external OSRM endpoint.

To check whether the older route-path cache can help, run:

```powershell
python scripts/audit_legacy_route_cache.py --output-dir results/outputs
```

This writes `production_legacy_route_cache_audit.csv`. Treat it as diagnostic
only unless the legacy cache row has complete geometry, distance, and duration
provenance.

To fill missing cache responses from an OSRM service, use the explicit network
mode. The default endpoint is local OSRM at `http://127.0.0.1:5000`:

Before fetching, check that the request manifest and endpoint policy are ready:

```powershell
python scripts/check_route_source.py --output-dir results/outputs
```

When the local endpoint is running, add `--probe` to send one test request:

```powershell
python scripts/check_route_source.py --output-dir results/outputs --probe
```

```powershell
python scripts/build_road_route_cache.py --output-dir results/outputs --fetch-missing --require-complete
```

Or run the full evidence pipeline with the same explicit fetch boundary:

```powershell
python scripts/run_phase0_evidence_pipeline.py --output-dir results/outputs --quality-dir results/quality --fetch-missing --require-final-eligible
```

For reproducible paper evidence, prefer a local or pinned OSRM endpoint through
`--osrm-base-url`. The public OSRM demo service is blocked unless
`--allow-public-osrm` is also supplied, because that sends route coordinates to
an external endpoint and should be a deliberate source-policy decision.

After exporting Phase 0 artifacts, check `route_cache_coverage` in
`production_phase0_dataset_validation.json` or the
`road_route_validation_coverage` columns in
`production_phase0_evidence_summary.csv`. These fields are the quick-read
answer to whether the current route evidence is still demo-only or ready for
strict comparison.

For a paper/debug summary, run:

```powershell
python scripts/summarize_phase0_readiness.py --output-dir results/outputs --write-dir results/quality
```

This writes a Markdown readiness summary and a method-level readiness CSV.
