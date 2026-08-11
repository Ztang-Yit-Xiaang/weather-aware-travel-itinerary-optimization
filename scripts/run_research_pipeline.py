"""Run the authoritative generation pipeline from frozen inputs or raw catalog CSVs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from itinerary_system.config import load_trip_config  # noqa: E402
from itinerary_system.pipeline_runner import (  # noqa: E402
    RefreshPolicy,
    build_phase0_generation_executor,
    build_production_generation_executor,
    run_research_pipeline,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default_trip_config.yaml")
    parser.add_argument("--output-root", default="runs")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--catalog-snapshot-id", default=None)
    parser.add_argument("--context-snapshot-id", default=None)
    parser.add_argument("--refresh-policy", choices=[item.value for item in RefreshPolicy], default="never")
    parser.add_argument("--permissive", action="store_true", help="Write ineligible diagnostics without raising.")
    parser.add_argument(
        "--input-mode",
        choices=("frozen-artifacts", "raw-catalog"),
        default="frozen-artifacts",
        help="Adapt committed method artifacts, or rerun the production optimizer from raw catalog CSVs.",
    )
    parser.add_argument("--artifact-dir", default="results/outputs")
    parser.add_argument("--business-path", "--business-csv", dest="business_path", default=None)
    parser.add_argument("--hotels-csv", default=None)
    parser.add_argument("--city", action="append", dest="cities", default=[])
    parser.add_argument("--primary-city", default="Santa Barbara")
    return parser.parse_args()


def build_executor(args: argparse.Namespace):
    if args.input_mode == "frozen-artifacts":
        artifact_dir = Path(args.artifact_dir)
        method_df = pd.read_csv(artifact_dir / "production_method_comparison.csv")
        route_stops_df = pd.read_csv(artifact_dir / "production_method_route_stops.csv")
        return build_phase0_generation_executor(method_df=method_df, route_stops_df=route_stops_df)

    if not args.business_path or not args.hotels_csv or not args.cities:
        raise ValueError("raw-catalog mode requires --business-path, --hotels-csv, and at least one --city")
    business_path = Path(args.business_path)
    all_business_df = (
        pd.read_json(business_path, lines=True)
        if business_path.suffix.lower() in {".json", ".jsonl"}
        else pd.read_csv(business_path)
    )
    return build_production_generation_executor(
        all_business_df=all_business_df,
        hotels_df=pd.read_csv(args.hotels_csv),
        city_names=args.cities,
        primary_city=args.primary_city,
    )


def run_from_args(args: argparse.Namespace):
    config = load_trip_config(args.config)
    catalog_snapshot_id = args.catalog_snapshot_id or config.get("data", "catalog_snapshot_id")
    context_snapshot_id = args.context_snapshot_id or config.get("data", "context_snapshot_id")
    if not catalog_snapshot_id or not context_snapshot_id:
        raise ValueError("catalog and context snapshot IDs must be supplied by config or command line")
    return run_research_pipeline(
        config_path=args.config,
        catalog_snapshot_id=str(catalog_snapshot_id),
        context_snapshot_id=str(context_snapshot_id),
        refresh_policy=args.refresh_policy,
        run_id=args.run_id,
        output_root=args.output_root,
        executor=build_executor(args),
        strict=not args.permissive,
    )


def main() -> int:
    try:
        run = run_from_args(parse_args())
    except (FileNotFoundError, ValueError) as exc:
        print(f"FAILED research pipeline: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(run.to_record(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
