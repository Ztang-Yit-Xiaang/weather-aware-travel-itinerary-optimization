"""Freeze and audit an existing route-evidence bundle without network access."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from itinerary_system.routing import freeze_route_evidence_bundle  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, help="Directory containing route request/cache CSV artifacts.")
    parser.add_argument("--provider-provenance", default="", help="Reviewed local-OSRM provenance JSON.")
    parser.add_argument("--expected-request-count", type=int, default=None)
    parser.add_argument("--max-cache-age-days", type=float, default=30.0)
    parser.add_argument("--manifest-output", default="", help="Optional output manifest path.")
    parser.add_argument("--require-publication-ready", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = freeze_route_evidence_bundle(
        args.output_dir,
        provider_provenance_path=args.provider_provenance or None,
        expected_request_count=args.expected_request_count,
        max_cache_age_days=args.max_cache_age_days,
        manifest_path=args.manifest_output or None,
    )
    print(f"Bundle: {result.bundle_id}")
    print(f"Requests: {result.request_count} ({result.validated_route_count} road, {result.snap_validated_count} snap)")
    print(f"Request-set hash: {result.request_set_hash}")
    print(f"Publication ready: {result.publication_ready}")
    for error in result.errors:
        print(f"ERROR: {error}")
    if args.require_publication_ready and not result.publication_ready:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
