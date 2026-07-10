"""Split validation for BENCH-002 benchmark scenarios."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from ..research_artifacts import stable_content_hash
from .disruptions import DisruptionFamily, DisruptionScenario

BenchmarkSplit = Mapping[str, Iterable[DisruptionScenario | Mapping[str, Any]]]


class BenchmarkLeakageError(ValueError):
    """Raised when one parent/disruption-family key crosses benchmark splits."""


def benchmark_split_key(scenario: DisruptionScenario | Mapping[str, Any]) -> str:
    """Return the no-leakage grouping key for a benchmark scenario."""

    parent_plan_id = _field(scenario, "parent_plan_id")
    family = _field(scenario, "family") or _field(scenario, "disruption_family")
    if isinstance(family, DisruptionFamily):
        family = family.value
    return f"{parent_plan_id}::{family}"


def assert_no_parent_family_leakage(splits: BenchmarkSplit) -> None:
    """Fail closed if the same parent/disruption family appears in multiple splits."""

    split_by_key: dict[str, str] = {}
    scenario_by_key: dict[str, str] = {}
    for split_name, scenarios in splits.items():
        for scenario in scenarios:
            key = benchmark_split_key(scenario)
            previous_split = split_by_key.get(key)
            if previous_split is not None and previous_split != split_name:
                scenario_id = _field(scenario, "scenario_id")
                first_scenario_id = scenario_by_key.get(key, "")
                raise BenchmarkLeakageError(
                    "Benchmark split leakage detected for "
                    f"{key}: {first_scenario_id!r} is in {previous_split!r}, "
                    f"but {scenario_id!r} is in {split_name!r}."
                )
            split_by_key[key] = split_name
            scenario_by_key.setdefault(key, _field(scenario, "scenario_id"))


def split_by_parent_disruption_family(
    scenarios: Iterable[DisruptionScenario],
    *,
    split_names: tuple[str, ...] = ("train", "development", "test"),
    seed: int = 0,
) -> dict[str, tuple[DisruptionScenario, ...]]:
    """Assign scenarios to deterministic splits while keeping split keys intact."""

    if not split_names:
        raise ValueError("split_names must contain at least one split")
    grouped: dict[str, list[DisruptionScenario]] = {}
    for scenario in scenarios:
        grouped.setdefault(benchmark_split_key(scenario), []).append(scenario)

    splits: dict[str, list[DisruptionScenario]] = {name: [] for name in split_names}
    for key in sorted(grouped):
        digest = stable_content_hash({"split_key": key, "seed": int(seed)})
        split_index = int(digest[:8], 16) % len(split_names)
        splits[split_names[split_index]].extend(grouped[key])

    frozen_splits = {name: tuple(items) for name, items in splits.items()}
    assert_no_parent_family_leakage(frozen_splits)
    return frozen_splits


def _field(scenario: DisruptionScenario | Mapping[str, Any], key: str) -> str:
    if isinstance(scenario, Mapping):
        value = scenario.get(key, "")
    else:
        value = getattr(scenario, key, "")
    return str(value)
