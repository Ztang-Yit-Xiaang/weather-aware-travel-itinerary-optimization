"""Run the Phase 0 evidence pipeline as one diagnostic command."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from itinerary_system.config import load_trip_config
from itinerary_system.phase0_exporter import PHASE0_ARTIFACT_FILES, write_phase0_research_artifacts
from itinerary_system.routing import OSRM_LOCAL_BASE_URL, build_road_route_cache_from_artifacts
from scripts.summarize_phase0_readiness import load_phase0_readiness, readiness_markdown, write_readiness_outputs
from scripts.validate_phase0_artifacts import validate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Phase 0 evidence pipeline and print readiness blockers.")
    parser.add_argument("--config", default="configs/default_trip_config.yaml", help="Trip config path.")
    parser.add_argument("--output-dir", default="results/outputs", help="Directory containing production artifacts.")
    parser.add_argument(
        "--quality-dir",
        default="results/quality",
        help="Directory for phase0_readiness_summary.md and phase0_method_readiness.csv.",
    )
    parser.add_argument(
        "--skip-cache-build",
        action="store_true",
        help="Use an existing production_road_route_cache.csv instead of rebuilding it first.",
    )
    parser.add_argument(
        "--cache-dir",
        default="",
        help="Optional OSRM JSON cache directory. Defaults to the project cache beside output-dir.",
    )
    parser.add_argument(
        "--fetch-missing",
        action="store_true",
        help="Fetch missing OSRM responses before building the validated route cache.",
    )
    parser.add_argument(
        "--osrm-base-url",
        default=OSRM_LOCAL_BASE_URL,
        help="OSRM service base URL used only with --fetch-missing. Defaults to local OSRM.",
    )
    parser.add_argument(
        "--allow-public-osrm",
        action="store_true",
        help="Permit the public OSRM demo endpoint if --osrm-base-url points at router.project-osrm.org.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=35,
        help="HTTP timeout for each OSRM request used only with --fetch-missing.",
    )
    parser.add_argument(
        "--require-final-eligible",
        action="store_true",
        help="Fail unless every Phase 0 evaluation is eligible for final comparison.",
    )
    return parser.parse_args()


def run_phase0_evidence_pipeline(
    *,
    config_path: str | Path,
    output_dir: str | Path,
    quality_dir: str | Path,
    skip_cache_build: bool = False,
    cache_dir: str | Path | None = None,
    fetch_missing: bool = False,
    osrm_base_url: str = OSRM_LOCAL_BASE_URL,
    timeout_seconds: int = 35,
    allow_public_osrm: bool = False,
    require_final_eligible: bool = False,
) -> dict[str, Any]:
    """Build route evidence, export Phase 0 artifacts, validate, and summarize."""

    output_path = Path(output_dir)
    config = load_trip_config(config_path)
    cache_result = None
    if not skip_cache_build:
        cache_result = build_road_route_cache_from_artifacts(
            output_dir=output_path,
            cache_dir=cache_dir,
            fetch_missing=bool(fetch_missing),
            osrm_base_url=osrm_base_url,
            request_timeout_seconds=int(timeout_seconds),
            allow_public_osrm=bool(allow_public_osrm),
        )
    export_result = write_phase0_research_artifacts(output_dir=output_path, config=config)
    errors, warnings = validate(output_path, require_final_eligible=bool(require_final_eligible))
    readiness = load_phase0_readiness(output_path)
    quality_paths = write_readiness_outputs(readiness, quality_dir)
    return {
        "config": config,
        "cache_result": cache_result,
        "export_result": export_result,
        "errors": errors,
        "warnings": warnings,
        "readiness": readiness,
        "quality_paths": quality_paths,
    }


def _print_result(result: dict[str, Any], *, skip_cache_build: bool) -> None:
    cache_result = result["cache_result"]
    if skip_cache_build:
        print("Skipped road-route cache build")
    elif cache_result is not None:
        validated = int(cache_result.cache_df["road_validated"].astype(bool).sum()) if not cache_result.cache_df.empty else 0
        requested = int(len(cache_result.audit_df))
        print(f"Wrote {cache_result.cache_path}")
        print(f"Wrote {cache_result.audit_path}")
        print(f"Wrote {cache_result.request_path}")
        print(f"Validated road legs: {validated}/{requested}")

    print("Exported Phase 0 artifacts:")
    for filename in PHASE0_ARTIFACT_FILES:
        print(f"- {filename}")

    for warning in result["warnings"]:
        print(f"WARNING: {warning}")
    if result["errors"]:
        print("FAILED Phase 0 evidence pipeline")
        for error in result["errors"]:
            print(f"- {error}")
    else:
        print("PASSED Phase 0 evidence pipeline")

    print()
    print(readiness_markdown(result["readiness"]), end="")
    quality_paths = result["quality_paths"]
    print(f"Wrote {quality_paths['markdown']}")
    print(f"Wrote {quality_paths['method_csv']}")


def main() -> int:
    args = parse_args()
    cache_dir = args.cache_dir or None
    try:
        result = run_phase0_evidence_pipeline(
            config_path=args.config,
            output_dir=args.output_dir,
            quality_dir=args.quality_dir,
            skip_cache_build=bool(args.skip_cache_build),
            cache_dir=cache_dir,
            fetch_missing=bool(args.fetch_missing),
            osrm_base_url=args.osrm_base_url,
            timeout_seconds=int(args.timeout_seconds),
            allow_public_osrm=bool(args.allow_public_osrm),
            require_final_eligible=bool(args.require_final_eligible),
        )
    except ValueError as exc:
        print(f"FAILED Phase 0 evidence pipeline: {exc}")
        return 2
    _print_result(result, skip_cache_build=bool(args.skip_cache_build))
    return 1 if result["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
