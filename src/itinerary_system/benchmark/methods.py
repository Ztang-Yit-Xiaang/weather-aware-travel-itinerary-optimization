"""Pipeline-backed benchmark method adapters."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from ..pipeline_runner import (
    PipelineExecutor,
    PipelineRun,
    PipelineStrictModeError,
    RefreshPolicy,
    build_context_blind_solver_executor,
    build_deterministic_context_aware_heuristic_executor,
    build_full_reoptimization_executor,
    build_progressive_repair_executor,
    run_research_pipeline,
)
from ..repair.day_route_solver import DayRouteSolverConfig
from ..research_artifacts import PlanArtifactV2, stable_content_hash
from ..routing import RouteMatrix
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
        try:
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
        except PipelineStrictModeError as exc:
            if exc.pipeline_run is None:
                raise
            run = exc.pipeline_run
        return pipeline_run_to_benchmark_result(run)

    return BenchmarkMethodAdapter(
        method_id=method_id,
        runner=runner,
        method_family=method_family,
        baseline=baseline,
    )


def build_publication_benchmark_method_adapters(
    *,
    parent_plan: PlanArtifactV2,
    route_matrix: RouteMatrix,
    config_path: str | Path,
    output_root: str | Path,
    day_route_config: DayRouteSolverConfig | None = None,
    publication_mode: bool = True,
    strict: bool = True,
    max_complete_candidates: int = 50_000,
) -> tuple[BenchmarkMethodAdapter, ...]:
    """Build the four canonical, method-specific E3 pipeline adapters."""

    shared = {
        "config_path": config_path,
        "output_root": output_root,
        "strict": strict,
    }
    return (
        build_pipeline_benchmark_method_adapter(
            method_id="context_blind_solver",
            method_family="solver_baseline",
            baseline=True,
            executor_factory=lambda scenario: build_context_blind_solver_executor(
                parent_plan=parent_plan,
                repair_request=scenario.request,
                route_matrix=route_matrix,
                day_route_config=day_route_config,
                publication_mode=publication_mode,
                max_complete_candidates=max_complete_candidates,
            ),
            **shared,
        ),
        build_pipeline_benchmark_method_adapter(
            method_id="deterministic_context_aware_heuristic",
            method_family="heuristic_baseline",
            baseline=True,
            executor_factory=lambda scenario: build_deterministic_context_aware_heuristic_executor(
                parent_plan=parent_plan,
                repair_request=scenario.request,
                route_matrix=route_matrix,
                day_route_config=day_route_config,
                publication_mode=publication_mode,
            ),
            **shared,
        ),
        build_pipeline_benchmark_method_adapter(
            method_id="progressive_sequential_lexicographic_repair",
            method_family="proposed_repair",
            executor_factory=lambda scenario: build_progressive_repair_executor(
                parent_plan=parent_plan,
                repair_request=scenario.request,
                route_matrix=route_matrix,
                day_route_config=day_route_config,
                publication_mode=publication_mode,
                repository_subdir="rw",
            ),
            **shared,
        ),
        build_pipeline_benchmark_method_adapter(
            method_id="full_reoptimization",
            method_family="solver_baseline",
            baseline=True,
            executor_factory=lambda scenario: build_full_reoptimization_executor(
                parent_plan=parent_plan,
                repair_request=scenario.request,
                route_matrix=route_matrix,
                day_route_config=day_route_config,
                publication_mode=publication_mode,
                max_complete_candidates=max_complete_candidates,
            ),
            **shared,
        ),
    )


def pipeline_run_to_benchmark_result(run: PipelineRun) -> dict[str, Any]:
    """Load a completed pipeline run directory into the BENCH-002 method-result shape."""

    manifest = _read_json(run.manifest_path)
    artifact_paths = _artifact_paths(run.output_dir, manifest)
    plan_records = _read_json_records(artifact_paths.get("plans", ()))
    pipeline_manifest = _mapping(manifest.get("pipeline_run"))
    output_plan_ids = {str(plan_id) for plan_id in pipeline_manifest.get("output_plan_ids", ())}
    output_plans = tuple(
        record for record in plan_records if str(record.get("plan_id") or "") in output_plan_ids
    )
    return {
        "run_id": run.run_id,
        "status": run.status,
        "output_dir": str(run.output_dir),
        "manifest_path": str(run.manifest_path),
        "metrics_path": str(run.metrics_path),
        "planner_runs": _read_jsonl_records(artifact_paths.get("planner_runs", ())),
        "output_plans": output_plans,
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


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _safe_filename(value: str) -> str:
    return value.replace("/", "_").replace("\\", "_").replace(":", "_")
