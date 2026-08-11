"""Build production_road_route_cache.csv from cached OSRM route responses."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from itinerary_system.config import TripConfig, load_trip_config  # noqa: E402
from itinerary_system.routing import OSRM_LOCAL_BASE_URL, build_road_route_cache_from_artifacts  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a validated Phase 0 road-route cache from OSRM cache files.")
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "configs" / "default_trip_config.yaml"),
        help="Trip config providing explicit airport routing anchors.",
    )
    parser.add_argument("--output-dir", default="results/outputs", help="Directory containing production route stops.")
    parser.add_argument(
        "--route-stops",
        default="",
        help="Optional route-stop CSV path. Defaults to production_method_route_stops.csv in output-dir.",
    )
    parser.add_argument(
        "--cache-dir",
        default="",
        help="Optional OSRM JSON cache directory. Defaults to the project cache beside output-dir.",
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Return a nonzero exit code when any requested leg lacks validated cached OSRM evidence.",
    )
    parser.add_argument(
        "--fetch-missing",
        action="store_true",
        help="Fetch missing route responses from OSRM before building the validated cache.",
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
        "--max-snap-distance-m",
        type=float,
        default=100.0,
        help="Maximum acceptable OSRM endpoint snap distance recorded in the audit.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=35,
        help="HTTP timeout for each OSRM request used only with --fetch-missing.",
    )
    return parser.parse_args()


def _configured_routing_anchors(config: TripConfig) -> dict[str, tuple[float, float]]:
    gateways = config.get("transport", "airport_gateways", {})
    if not isinstance(gateways, dict):
        return {}
    anchors: dict[str, tuple[float, float]] = {}
    for city, raw in gateways.items():
        if not isinstance(raw, dict):
            continue
        latitude = raw.get("routing_latitude")
        longitude = raw.get("routing_longitude")
        if latitude is None and longitude is None:
            continue
        if latitude is None or longitude is None:
            raise ValueError(f"routing anchor for {city!r} must provide both latitude and longitude")
        label = str(raw.get("name") or raw.get("code") or city).strip()
        if not label:
            raise ValueError(f"routing anchor for {city!r} has no label")
        anchors[label] = (float(latitude), float(longitude))
    return anchors


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    route_stops_df = None
    if args.route_stops:
        import pandas as pd

        route_stops_df = pd.read_csv(args.route_stops)
    try:
        routing_anchors = _configured_routing_anchors(load_trip_config(args.config))
        result = build_road_route_cache_from_artifacts(
            output_dir=output_dir,
            route_stops_df=route_stops_df,
            cache_dir=args.cache_dir or None,
            fetch_missing=bool(args.fetch_missing),
            osrm_base_url=args.osrm_base_url,
            request_timeout_seconds=int(args.timeout_seconds),
            allow_public_osrm=bool(args.allow_public_osrm),
            max_snap_distance_m=float(args.max_snap_distance_m),
            route_point_overrides=routing_anchors,
        )
    except ValueError as exc:
        print(f"FAILED road-route cache build: {exc}")
        return 2
    if routing_anchors:
        print(f"Configured routing anchors: {', '.join(sorted(routing_anchors))}")
    validated = int(result.cache_df["road_validated"].astype(bool).sum()) if not result.cache_df.empty else 0
    snap_validated = (
        int(result.audit_df["snap_validated"].astype(bool).sum())
        if "snap_validated" in result.audit_df.columns
        else 0
    )
    requested = int(len(result.audit_df))
    print(f"Wrote {result.cache_path}")
    print(f"Wrote {result.audit_path}")
    print(f"Wrote {result.request_path}")
    print(f"Validated road legs: {validated}/{requested}")
    print(f"Validated endpoint-snap legs: {snap_validated}/{requested}")
    if result.complete:
        print("PASSED road-route cache build")
        return 0
    if args.require_complete and not result.complete:
        print("FAILED road-route cache build: missing road or endpoint-snap evidence")
        return 1
    print("COMPLETED road-route cache build with missing evidence; inspect the audit before paper use")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
