"""Build and validate a RouteMatrix artifact from cached route rows."""

from __future__ import annotations

import argparse
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


def main() -> int:
    args = parse_args()
    required_sequences = tuple(_parse_sequence(value) for value in args.required_sequence)
    try:
        matrix, report = build_validated_route_matrix_from_cache(
            Path(args.input),
            str(args.context_snapshot_id),
            Path(args.output_dir),
            required_sequences=required_sequences,
            require_publication_ready=bool(args.require_publication_ready),
        )
    except Exception as exc:
        print(f"FAILED route matrix build: {exc}")
        return 1
    print(f"Built route matrix: {matrix.matrix_id}")
    print(f"Cells: {len(matrix.cells)}")
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
