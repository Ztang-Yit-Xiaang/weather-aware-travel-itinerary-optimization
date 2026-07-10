"""Paired BENCH-002 benchmark execution and export."""

from __future__ import annotations

import copy
import json
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ..research_artifacts import stable_content_hash
from .disruptions import DisruptionScenario
from .metrics import extract_benchmark_metrics
from .splits import assert_no_parent_family_leakage, split_by_parent_disruption_family


@dataclass(frozen=True)
class BenchmarkMethodAdapter:
    """Callable benchmark method wrapper with baseline metadata."""

    method_id: str
    runner: Callable[[DisruptionScenario], Any]
    method_family: str = "repair"
    baseline: bool = False


BenchmarkMethod = BenchmarkMethodAdapter


@dataclass(frozen=True)
class BenchmarkRunRecord:
    """One method result on one frozen disruption scenario."""

    benchmark_id: str
    scenario_id: str
    family: str
    parent_plan_id: str
    catalog_snapshot_id: str
    context_snapshot_id: str
    method_id: str
    method_family: str
    baseline: bool
    run_id: str
    status: str
    metrics: dict[str, Any]
    frozen_input_hash: str
    output_plan_id: str = ""
    output_dir: str = ""
    manifest_path: str = ""
    metrics_path: str = ""
    error_class: str = ""
    error_summary: str = ""
    schema_version: str = "benchmark-run-record-v1"

    def to_record(self) -> dict[str, Any]:
        return {**asdict(self), **self.metrics}


@dataclass(frozen=True)
class BenchmarkResult:
    """Exported BENCH-002 benchmark result manifest."""

    benchmark_id: str
    scenario_ids: tuple[str, ...]
    method_ids: tuple[str, ...]
    run_ids: tuple[str, ...]
    metrics_path: Path
    manifest_path: Path
    run_records: tuple[BenchmarkRunRecord, ...]
    schema_version: str = "benchmark-result-v1"

    def to_record(self) -> dict[str, Any]:
        return {
            "benchmark_id": self.benchmark_id,
            "scenario_ids": list(self.scenario_ids),
            "method_ids": list(self.method_ids),
            "run_ids": list(self.run_ids),
            "metrics_path": str(self.metrics_path),
            "manifest_path": str(self.manifest_path),
            "run_count": len(self.run_records),
            "schema_version": self.schema_version,
        }


def run_benchmark_suite(
    *,
    scenarios: Iterable[DisruptionScenario],
    methods: Iterable[BenchmarkMethodAdapter],
    output_dir: str | Path,
    benchmark_id: str | None = None,
    splits: Mapping[str, Iterable[DisruptionScenario]] | None = None,
) -> BenchmarkResult:
    """Run paired benchmark methods over identical frozen scenarios and export rows."""

    scenario_tuple = tuple(scenarios)
    method_tuple = tuple(methods)
    if not scenario_tuple:
        raise ValueError("scenarios must contain at least one disruption scenario")
    if not method_tuple:
        raise ValueError("methods must contain at least one benchmark method")

    resolved_benchmark_id = benchmark_id or _benchmark_id(scenario_tuple, method_tuple)
    resolved_splits = splits or split_by_parent_disruption_family(scenario_tuple)
    assert_no_parent_family_leakage(resolved_splits)

    output_root = Path(output_dir)
    metrics_path = output_root / "metrics" / "benchmark_metrics.jsonl"
    manifest_path = output_root / "manifest.json"
    records: list[BenchmarkRunRecord] = []

    for scenario in scenario_tuple:
        frozen_record = scenario.to_record()
        frozen_hash = stable_content_hash(frozen_record)
        for method in method_tuple:
            started = time.perf_counter()
            try:
                raw_result = method.runner(copy.deepcopy(scenario))
                runtime_seconds = time.perf_counter() - started
                record = _run_record(
                    benchmark_id=resolved_benchmark_id,
                    scenario=scenario,
                    method=method,
                    raw_result=raw_result,
                    runtime_seconds=runtime_seconds,
                    frozen_input_hash=frozen_hash,
                )
            except Exception as exc:  # pragma: no cover - behavior is data-export defensive.
                runtime_seconds = time.perf_counter() - started
                record = _failed_run_record(
                    benchmark_id=resolved_benchmark_id,
                    scenario=scenario,
                    method=method,
                    runtime_seconds=runtime_seconds,
                    frozen_input_hash=frozen_hash,
                    error=exc,
                )
            records.append(record)
            if scenario.to_record() != frozen_record:
                raise ValueError(f"benchmark scenario {scenario.scenario_id!r} was mutated during execution")

    _write_jsonl(metrics_path, (record.to_record() for record in records))
    result = BenchmarkResult(
        benchmark_id=resolved_benchmark_id,
        scenario_ids=tuple(scenario.scenario_id for scenario in scenario_tuple),
        method_ids=tuple(method.method_id for method in method_tuple),
        run_ids=tuple(record.run_id for record in records),
        metrics_path=metrics_path,
        manifest_path=manifest_path,
        run_records=tuple(records),
    )
    _write_json(manifest_path, _manifest(result, resolved_splits))
    return result


def _run_record(
    *,
    benchmark_id: str,
    scenario: DisruptionScenario,
    method: BenchmarkMethodAdapter,
    raw_result: Any,
    runtime_seconds: float,
    frozen_input_hash: str,
) -> BenchmarkRunRecord:
    metrics = extract_benchmark_metrics(raw_result, scenario=scenario, runtime_seconds=runtime_seconds)
    return BenchmarkRunRecord(
        benchmark_id=benchmark_id,
        scenario_id=scenario.scenario_id,
        family=scenario.family.value,
        parent_plan_id=scenario.parent_plan_id,
        catalog_snapshot_id=scenario.catalog_snapshot_id,
        context_snapshot_id=scenario.context_snapshot_id,
        method_id=method.method_id,
        method_family=method.method_family,
        baseline=method.baseline,
        run_id=_run_id(method, scenario, raw_result),
        status=str(_value(raw_result, "status") or "completed"),
        metrics=metrics,
        frozen_input_hash=frozen_input_hash,
        output_plan_id=_output_plan_id(raw_result),
        output_dir=str(_value(raw_result, "output_dir") or ""),
        manifest_path=str(_value(raw_result, "manifest_path") or ""),
        metrics_path=str(_value(raw_result, "metrics_path") or ""),
    )


def _failed_run_record(
    *,
    benchmark_id: str,
    scenario: DisruptionScenario,
    method: BenchmarkMethodAdapter,
    runtime_seconds: float,
    frozen_input_hash: str,
    error: Exception,
) -> BenchmarkRunRecord:
    metrics = extract_benchmark_metrics(
        {"status": "failed", "metrics": {"runtime_seconds": runtime_seconds}},
        scenario=scenario,
        runtime_seconds=runtime_seconds,
    )
    return BenchmarkRunRecord(
        benchmark_id=benchmark_id,
        scenario_id=scenario.scenario_id,
        family=scenario.family.value,
        parent_plan_id=scenario.parent_plan_id,
        catalog_snapshot_id=scenario.catalog_snapshot_id,
        context_snapshot_id=scenario.context_snapshot_id,
        method_id=method.method_id,
        method_family=method.method_family,
        baseline=method.baseline,
        run_id=f"{method.method_id}_{scenario.scenario_id}_failed",
        status="failed",
        metrics=metrics,
        frozen_input_hash=frozen_input_hash,
        error_class=type(error).__name__,
        error_summary=str(error),
    )


def _run_id(method: BenchmarkMethodAdapter, scenario: DisruptionScenario, raw_result: Any) -> str:
    run_id = _value(raw_result, "run_id")
    if run_id:
        return str(run_id)
    return f"{method.method_id}_{scenario.scenario_id}"


def _output_plan_id(raw_result: Any) -> str:
    output_plans = _records(_value(raw_result, "output_plans"))
    if not output_plans:
        return ""
    return str(output_plans[0].get("plan_id") or "")


def _benchmark_id(scenarios: tuple[DisruptionScenario, ...], methods: tuple[BenchmarkMethodAdapter, ...]) -> str:
    digest = stable_content_hash(
        {
            "scenario_ids": tuple(scenario.scenario_id for scenario in scenarios),
            "method_ids": tuple(method.method_id for method in methods),
        }
    )
    return f"benchmark_{digest[:12]}"


def _manifest(result: BenchmarkResult, splits: Mapping[str, Iterable[DisruptionScenario]]) -> dict[str, Any]:
    return {
        **result.to_record(),
        "metrics_path": str(result.metrics_path),
        "manifest_path": str(result.manifest_path),
        "splits": {
            split_name: [scenario.scenario_id for scenario in split_scenarios]
            for split_name, split_scenarios in splits.items()
        },
    }


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(_jsonable(record), sort_keys=True) + "\n")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _records(value: Any) -> tuple[dict[str, Any], ...]:
    if value is None:
        return ()
    if isinstance(value, Mapping):
        return (_mapping(value),)
    return tuple(_mapping(item) for item in value)


def _mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "to_record"):
        return dict(value.to_record())
    if hasattr(value, "__dict__"):
        return dict(vars(value))
    return {}


def _value(result: Any, key: str) -> Any:
    if isinstance(result, Mapping):
        return result.get(key)
    return getattr(result, key, None)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value
