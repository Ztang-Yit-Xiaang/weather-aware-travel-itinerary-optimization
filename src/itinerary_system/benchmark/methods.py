"""Pipeline-backed benchmark method adapters."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from ..pipeline_runner import PipelineExecutor, PipelineRun, RefreshPolicy, run_research_pipeline
from ..research_artifacts import stable_content_hash
from .disruptions import DisruptionScenario
from .runner import BenchmarkMethodAdapter

PipelineExecutorFactory = Callable[[DisruptionScenario], PipelineExecutor]


def build_pipeline_benchmark_method_adapter(
    *,
    method_id: str,
    config_path: str | Path,
    output_root: str | Path,
    executor_factory: PipelineExecutorFactory,
    method_family: str = "repair",
    baseline: bool = False,
    strict: bool = True,
    refresh_policy: RefreshPolicy | str = RefreshPolicy.NEVER,
    config_overrides: dict[str, Any] | None = None,
    pipeline_runner: Callable[..., PipelineRun] = run_research_pipeline,
    run_id_prefix: str = "benchmark",
) -> BenchmarkMethodAdapter:
    """Build a benchmark method that executes through the package pipeline runner."""

    def runner(scenario: DisruptionScenario) -> dict[str, Any]:
        run = pipeline_runner(
            config_path=config_path,
            catalog_snapshot_id=scenario.catalog_snapshot_id,
            context_snapshot_id=scenario.context_snapshot_id,
            parent_plan_id=scenario.parent_plan_id,
            repair_request_id=scenario.request.request_id,
            refresh_policy=refresh_policy,
            run_id=_pipeline_benchmark_run_id(run_id_prefix, method_id, scenario),
            output_root=output_root,
            executor=executor_factory(scenario),
            strict=strict,
            config_overrides=config_overrides,
        )
        return pipeline_run_to_benchmark_result(run)

    return BenchmarkMethodAdapter(
        method_id=method_id,
        runner=runner,
        method_family=method_family,
        baseline=baseline,
    )


def pipeline_run_to_benchmark_result(run: PipelineRun) -> dict[str, Any]:
    """Load a completed pipeline run directory into the BENCH-002 method-result shape."""

    manifest = _read_json(run.manifest_path)
    artifact_paths = _artifact_paths(run.output_dir, manifest)
    return {
        "run_id": run.run_id,
        "status": run.status,
        "output_dir": str(run.output_dir),
        "manifest_path": str(run.manifest_path),
        "metrics_path": str(run.metrics_path),
        "planner_runs": _read_jsonl_records(artifact_paths.get("planner_runs", ())),
        "output_plans": _read_json_records(artifact_paths.get("plans", ())),
        "diff_records": _read_json_records(artifact_paths.get("diffs", ())),
        "route_records": _read_json_records(artifact_paths.get("routing", ())),
        "evaluations": _read_json_records(artifact_paths.get("evaluations", ())),
        "explanation_records": _read_json_records(artifact_paths.get("explanations", ())),
        "request_records": _read_json_records(artifact_paths.get("requests", ())),
        "metrics": _read_json(run.metrics_path),
        "pipeline_manifest": manifest,
    }


def _pipeline_benchmark_run_id(prefix: str, method_id: str, scenario: DisruptionScenario) -> str:
    digest = stable_content_hash(
        {
            "method_id": method_id,
            "scenario_id": scenario.scenario_id,
            "catalog_snapshot_id": scenario.catalog_snapshot_id,
            "context_snapshot_id": scenario.context_snapshot_id,
            "parent_plan_id": scenario.parent_plan_id,
        }
    )
    return _safe_filename(f"{prefix}_{digest[:12]}")


def _artifact_paths(output_dir: Path, manifest: Mapping[str, Any]) -> dict[str, tuple[Path, ...]]:
    artifacts = manifest.get("artifacts", {})
    if not isinstance(artifacts, Mapping):
        return {}
    return {
        str(kind): tuple(output_dir / str(relative_path) for relative_path in paths)
        for kind, paths in artifacts.items()
        if isinstance(paths, list)
    }


def _read_json_records(paths: tuple[Path, ...]) -> tuple[dict[str, Any], ...]:
    return tuple(_read_json(path) for path in paths if path.exists() and path.suffix.lower() == ".json")


def _read_jsonl_records(paths: tuple[Path, ...]) -> tuple[dict[str, Any], ...]:
    records: list[dict[str, Any]] = []
    for path in paths:
        if not path.exists() or path.suffix.lower() != ".jsonl":
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
    return tuple(records)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_filename(value: str) -> str:
    return value.replace("/", "_").replace("\\", "_").replace(":", "_")
