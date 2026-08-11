"""Create a new read-only run snapshot and export its product dashboard."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from itinerary_system.product_dashboard_models import file_sha256, json_text  # noqa: E402
from itinerary_system.product_dashboard_renderer import render_product_dashboard  # noqa: E402

COPIED_ARTIFACT_CATEGORIES = (
    "requests",
    "plans",
    "diffs",
    "evaluations",
    "explanations",
    "routing",
    "planner_runs",
    "metrics",
    "dashboard",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a separately versioned product dashboard from one canonical run."
    )
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output-run", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--benchmark-metrics", type=Path)
    return parser.parse_args()


def clone_source_run(
    *,
    source_run: Path,
    output_run: Path,
    run_id: str,
    benchmark_metrics: Path | None = None,
) -> Path:
    """Copy declared canonical artifacts into a new non-overwritable run."""

    source_run = source_run.resolve()
    output_run = output_run.resolve()
    if output_run.exists():
        raise FileExistsError(f"output run already exists: {output_run}")
    source_manifest_path = source_run / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    source_artifacts = source_manifest.get("artifacts") or {}

    output_run.mkdir(parents=True)
    copied_artifacts: dict[str, list[str]] = {}
    for category in COPIED_ARTIFACT_CATEGORIES:
        paths = source_artifacts.get(category) or []
        if not isinstance(paths, list):
            continue
        for path_text in paths:
            relative = Path(str(path_text))
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"unsafe source artifact path: {path_text}")
            source = source_run / relative
            if not source.is_file():
                raise FileNotFoundError(f"declared source artifact is missing: {source}")
            destination = output_run / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            copied_artifacts.setdefault(category, []).append(relative.as_posix())

    for filename in ("dataset_manifest.json", "resolved_config.redacted.json"):
        source = source_run / filename
        if source.is_file():
            shutil.copy2(source, output_run / filename)

    if benchmark_metrics is not None:
        benchmark_metrics = benchmark_metrics.resolve()
        destination = output_run / "benchmark" / "metrics" / "benchmark_metrics.jsonl"
        destination.parent.mkdir(parents=True)
        shutil.copy2(benchmark_metrics, destination)
        copied_artifacts["benchmark_metrics"] = [
            "benchmark/metrics/benchmark_metrics.jsonl"
        ]
        copied_artifacts["benchmark_planner_runs"] = _copy_benchmark_planner_runs(
            benchmark_metrics=benchmark_metrics,
            output_run=output_run,
        )

    derived_manifest = {
        key: value
        for key, value in source_manifest.items()
        if key not in {"run_id", "artifacts", "artifact_counts"}
    }
    derived_manifest.update(
        {
            "run_id": run_id,
            "artifacts": copied_artifacts,
            "artifact_counts": {
                category: len(paths) for category, paths in copied_artifacts.items()
            },
            "derived_from": {
                "derivation_kind": "read_only_product_snapshot",
                "source_run_id": source_manifest.get("run_id"),
                "source_manifest_sha256": file_sha256(source_manifest_path),
            },
        }
    )
    (output_run / "manifest.json").write_text(
        json_text(derived_manifest) + "\n", encoding="utf-8", newline="\n"
    )
    return output_run


def _copy_benchmark_planner_runs(
    *, benchmark_metrics: Path, output_run: Path
) -> list[str]:
    """Copy canonical failure diagnostics referenced by benchmark rows."""

    copied: list[str] = []
    for line in benchmark_metrics.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        manifest_text = str(row.get("manifest_path") or "")
        run_id = str(row.get("run_id") or "")
        if not manifest_text or not run_id:
            continue
        manifest_path = Path(manifest_text)
        if not manifest_path.is_absolute():
            manifest_path = REPO_ROOT / manifest_path
        planner_runs = manifest_path.parent / "planner_runs.jsonl"
        if not planner_runs.is_file():
            continue
        relative = Path("benchmark") / "planner_runs" / f"{run_id}.jsonl"
        destination = output_run / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(planner_runs, destination)
        copied.append(relative.as_posix())
    return copied


def main() -> int:
    args = parse_args()
    output_run = clone_source_run(
        source_run=args.source_run,
        output_run=args.output_run,
        run_id=args.run_id,
        benchmark_metrics=args.benchmark_metrics,
    )
    manifest = render_product_dashboard(output_run)
    print(f"Product dashboard: {output_run / 'dashboard_product' / 'index.html'}")
    print(f"Manifest schema: {manifest.schema_version}")
    print(f"Product version: {manifest.product_version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
