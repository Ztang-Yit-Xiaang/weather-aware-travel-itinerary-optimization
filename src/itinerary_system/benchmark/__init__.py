"""Benchmark scenario generation APIs."""

from .disruptions import (
    DisruptionFamily,
    DisruptionGenerator,
    DisruptionScenario,
    generate_disruption_requests,
    generate_disruption_scenarios,
)
from .methods import (
    build_pipeline_benchmark_method_adapter,
    build_publication_benchmark_method_adapters,
    pipeline_run_to_benchmark_result,
)
from .metrics import extract_benchmark_metrics
from .route_coverage import BenchmarkRouteCoverageReport, build_benchmark_route_coverage
from .runner import (
    BenchmarkMethod,
    BenchmarkMethodAdapter,
    BenchmarkResult,
    BenchmarkRunRecord,
    run_benchmark_suite,
)
from .splits import (
    BenchmarkLeakageError,
    assert_no_parent_family_leakage,
    benchmark_split_key,
    split_by_parent_disruption_family,
)

__all__ = [
    "BenchmarkLeakageError",
    "BenchmarkMethod",
    "BenchmarkMethodAdapter",
    "BenchmarkResult",
    "BenchmarkRunRecord",
    "BenchmarkRouteCoverageReport",
    "DisruptionFamily",
    "DisruptionGenerator",
    "DisruptionScenario",
    "assert_no_parent_family_leakage",
    "benchmark_split_key",
    "build_pipeline_benchmark_method_adapter",
    "build_benchmark_route_coverage",
    "build_publication_benchmark_method_adapters",
    "extract_benchmark_metrics",
    "generate_disruption_requests",
    "generate_disruption_scenarios",
    "pipeline_run_to_benchmark_result",
    "run_benchmark_suite",
    "split_by_parent_disruption_family",
]
