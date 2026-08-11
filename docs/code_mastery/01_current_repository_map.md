# Current Repository Map

## Package Areas

| Area | Category | Main responsibility | Current state |
|---|---|---|---|
| `data/` | domain/data boundary | Frozen catalog/context schemas and loading | current |
| `research_artifacts.py` | domain model + artifact writer helpers | Versioned plans, planner runs, hashes, lineage | current |
| `plans/` | domain model + repository + stateless diff | Ownership, append-only plan store, typed changes | current |
| `routing/` | domain model + service boundary | Validated routes, matrices, evidence bundles | current |
| `repair/` | service + orchestrator | Neighborhoods, master model, lexicographic stages, exact and heuristic methods | current |
| `evaluation/` | independent service | Recompute eligibility and quality certificate | current |
| `explanation/` | evidence service | Ground claims in diff/certificate/route/counterfactual records | current |
| `benchmark/` | orchestrator + adapters | Six disruptions, four locked methods, paired runs, metrics, closeout | current; E3.3 blocked |
| `interaction/` | optional controller/orchestrator | Permission-aware hypothetical probes and authorized continuation | implemented scaffold; E5 deferred |
| `pipeline_runner.py` | workflow orchestrator + artifact writer | Authoritative immutable execution boundary | current |
| `map_exporter.py` + `dashboard_*.py` | artifact writer + UI controllers | Frozen modular Leaflet dashboard | current parity artifact |
| `blueprint_*.py` | renderer + UI controller | Frozen Folium renderer and selector | current parity artifact |
| `notebook/` | compatibility boundary | Thin pipeline client and legacy import facade | compatibility-only |

## Module Categories

- **Domain model:** immutable records such as `PlanArtifactV2`, `PlanDiff`, and
  `PlanEvaluationCertificate`.
- **Stateless utility:** receives input and returns output without keeping
  cross-call state, such as `compute_plan_diff()`.
- **Service:** performs one bounded job, such as `PlanEvaluator`.
- **Orchestrator:** orders multiple services, such as
  `run_research_pipeline()`.
- **Compatibility boundary:** preserves an old import/output contract while
  package ownership moves elsewhere.
- **UI controller:** synchronizes browser state and controls.
- **Artifact writer:** serializes immutable evidence to a run directory.

## What Calls What

See [package entry and pipeline](diagrams/package_entry_and_pipeline.md) and
[immutable artifact flow](diagrams/immutable_artifact_flow.md).

> **Beginner note / 初学者提示:** Folder names are not automatically the source
> of truth. The current manifest, code callers, tests, and immutable artifacts
> decide which path is authoritative.

## Read Next

Read [authoritative entry points](02_authoritative_entry_points.md) before
opening a solver module. It shows how data reaches the code.

