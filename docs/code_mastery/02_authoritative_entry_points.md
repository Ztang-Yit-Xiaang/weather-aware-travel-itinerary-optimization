# Authoritative Entry Points

## `scripts/run_research_pipeline.py`

1. **Why:** command-line bridge into the package runner.
2. **Category:** compatibility-friendly CLI controller.
3. **Called by:** a human, automation, and the thin production notebook.
4. **Inputs:** config, catalog/context IDs, run ID/output root, execution mode,
   and explicit raw/frozen artifact inputs.
5. **Outputs:** process status plus a printed `PipelineRun` summary.
6. **State owner:** none; the created run directory owns persistence.
7. **Invariants:** refresh defaults to `never`; raw-catalog mode requires
   explicit inputs.
8. **Failures:** argparse/input errors or propagated typed pipeline failures.
9. **Tests:** `tests/test_research_pipeline_entrypoint.py`,
   `tests/test_research_pipeline_cli_summary.py`,
   `tests/test_research_pipeline_raw_catalog.py`.
10. **Gate:** E1.
11. **Lifecycle:** current authoritative CLI.
12. **Read next:** `src/itinerary_system/pipeline_runner.py`.

## `run_research_pipeline()`

1. **Why:** provide one package-owned execution and artifact boundary.
2. **Category:** workflow orchestrator and artifact writer.
3. **Called by:** the CLI, benchmark method adapters, and the permission-aware
   continuation pipeline.
4. **Inputs:** config/snapshot IDs, optional parent/request IDs, refresh policy,
   output root, injected executor, strict mode, overrides, optional data bundle.
5. **Outputs:** typed `PipelineRun`.
6. **State owner:** one new `runs/<run_id>/` directory.
7. **Invariants:** refuse overwrite, redact config, write diagnostics before
   strict failure, and use injected execution rather than notebook logic.
8. **Failures:** `RunDirectoryExists`, `PipelineExecutionMissing`, executor
   failure, or `PipelineStrictModeError`.
9. **Tests:** `tests/test_pipeline_runner.py`.
10. **Gate:** E1 and PIPE-001.
11. **Lifecycle:** current.
12. **Read next:** `_write_execution_artifacts()` and `_write_manifest()`.

## `scripts/run_e3_publication_benchmark.py`

1. **Why:** freeze E3 inputs and run the locked four-method paired matrix.
2. **Category:** research CLI/orchestrator.
3. **Called by:** a deliberate E3 evidence run, not normal UI use.
4. **Inputs:** frozen parent, route/cache/catalog artifacts, cap, seed, output
   root.
5. **Outputs:** input manifest, 24 run directories, benchmark metrics/manifest,
   and closeout.
6. **State owner:** a new immutable E3 root.
7. **Invariants:** reject an existing root; preserve method and route lineage.
8. **Failures:** preflight, cap refusal, method failure, or closeout validation;
   failed rows remain evidence.
9. **Tests:** publication factory/contract/route-coverage tests.
10. **Gate:** E3.0–E3.3.
11. **Lifecycle:** current, but E3.3 execution is blocked.
12. **Read next:** `benchmark/methods.py` and `benchmark/runner.py`.

## `scripts/run_permission_aware_repair.py`

This is an explicit optional entry point. It imports
`itinerary_system.interaction.cli.main`; default interaction mode remains
disabled. It does not replace the research CLI and cannot authorize a
hypothetical probe.

See [interaction boundary](10_interaction_scaffold_and_permission_boundary.md).

