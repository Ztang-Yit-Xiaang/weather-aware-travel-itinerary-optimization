"""Audit legacy route-path cache coverage for Phase 0 route requests."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from itinerary_system.routing import audit_legacy_route_cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit legacy route cache coverage for Phase 0 requests.")
    parser.add_argument("--output-dir", default="results/outputs", help="Directory containing route request manifest.")
    parser.add_argument(
        "--legacy-cache",
        default="",
        help="Optional legacy JSON cache path. Defaults to results/cache/production_road_route_cache.json.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = audit_legacy_route_cache(
        output_dir=args.output_dir,
        legacy_cache_path=args.legacy_cache or None,
    )
    print(f"Wrote {result['audit_path']}")
    print(f"Legacy cache: {result['legacy_cache_path']}")
    print(f"Requested legs: {result['request_count']}")
    print(f"Legacy geometry matches: {result['geometry_available_count']}/{result['request_count']}")
    print(f"Conversion eligible rows: {result['conversion_eligible_count']}/{result['request_count']}")
    if result["request_count"] and result["conversion_eligible_count"] == result["request_count"]:
        print("PASSED legacy route cache audit")
        return 0
    print("COMPLETED legacy route cache audit with incomplete reusable evidence")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
