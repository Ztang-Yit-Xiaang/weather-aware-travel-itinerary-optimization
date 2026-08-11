"""Build and validate a RouteMatrix artifact from cached route rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from itinerary_system.routing import build_validated_route_matrix_from_cache  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a validated route matrix from route_options or road-cache CSV.")
    parser.add_argument("--input", required=True, help="CSV with route_options or production_road_route_cache rows.")
    parser.add_argument("--context-snapshot-id", required=True, help="Context snapshot ID for the route matrix.")
    parser.add_argument("--output-dir", default="results/quality", help="Directory for matrix and validation report.")
    parser.add_argument(
        "--route-evidence-manifest",
        default="",
        help="Frozen route-evidence manifest that owns the input cache.",
    )
    parser.add_argument(
        "--required-sequence",
        action="append",
        default=[],
        help="Comma-separated entity IDs for a required route sequence. Repeat for multiple routes.",
    )
    parser.add_argument(
        "--require-publication-ready",
        action="store_true",
        help="Exit nonzero unless all checked cells are road validated, non-fallback, and complete.",
    )
    return parser.parse_args()


def _parse_sequence(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in str(value).split(",") if part.strip())



def _bundle_lineage(args: argparse.Namespace) -> tuple[str, str]:
    if not args.route_evidence_manifest:
        if args.require_publication_ready:
            raise ValueError("strict matrix build requires --route-evidence-manifest")
        return "", ""
    manifest = json.loads(Path(args.route_evidence_manifest).read_text(encoding="utf-8"))
    bundle_id = str(manifest.get("bundle_id") or "").strip()
    expected_sha256 = str(manifest.get("artifact_hashes", {}).get("validated_cache") or "").strip()
    if not bundle_id or len(expected_sha256) != 64:
        raise ValueError("route-evidence manifest lacks bundle ID or validated-cache SHA-256")
    actual_sha256 = hashlib.sha256(Path(args.input).read_bytes()).hexdigest()
    if actual_sha256.lower() != expected_sha256.lower():
        raise ValueError("input cache SHA-256 does not match route-evidence manifest")
    if args.require_publication_ready and manifest.get("publication_ready") is not True:
        raise ValueError("strict matrix build requires a publication-ready route-evidence manifest")
    return bundle_id, expected_sha256

def main() -> int:
    args = parse_args()
    required_sequences = tuple(_parse_sequence(value) for value in args.required_sequence)
    try:
        source_bundle_id, expected_source_sha256 = _bundle_lineage(args)
        matrix, report = build_validated_route_matrix_from_cache(
            Path(args.input),
            str(args.context_snapshot_id),
            Path(args.output_dir),
            required_sequences=required_sequences,
            require_publication_ready=bool(args.require_publication_ready),
            source_bundle_id=source_bundle_id,
            expected_source_sha256=expected_source_sha256,
        )
    except Exception as exc:
        print(f"FAILED route matrix build: {exc}")
        return 1
    print(f"Built route matrix: {matrix.matrix_id}")
    print(f"Cells: {len(matrix.cells)}")
    print(f"Source bundle: {matrix.source_bundle_id or 'unbound'}")
    print(f"Source cache SHA-256: {matrix.source_content_sha256}")
    print(
        "Publication ready: "
        f"{report.publication_ready} ({report.road_validated_leg_count}/{report.required_leg_count} checked legs)"
    )
    if report.errors:
        print("Errors:")
        for error in report.errors:
            print(f"- {error}")
    if args.require_publication_ready and not report.publication_ready:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
