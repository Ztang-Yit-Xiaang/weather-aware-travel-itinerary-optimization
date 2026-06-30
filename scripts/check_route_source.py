"""Check whether route-source evidence collection is ready for Phase 0."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from itinerary_system.routing import (
    OSRM_LOCAL_BASE_URL,
    ROAD_ROUTE_REQUESTS_FILENAME,
    fetch_osrm_payload,
    validate_route_fetch_policy,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Phase 0 route-source readiness.")
    parser.add_argument("--output-dir", default="results/outputs", help="Directory containing route request manifest.")
    parser.add_argument(
        "--osrm-base-url",
        default=OSRM_LOCAL_BASE_URL,
        help="OSRM endpoint to check. Defaults to local OSRM.",
    )
    parser.add_argument(
        "--allow-public-osrm",
        action="store_true",
        help="Permit the public OSRM demo endpoint if --osrm-base-url points at router.project-osrm.org.",
    )
    parser.add_argument(
        "--probe",
        action="store_true",
        help="Send a probe request for the first route leg. Defaults to policy/manifest checks only.",
    )
    parser.add_argument("--timeout-seconds", type=int, default=5, help="HTTP timeout for --probe.")
    return parser.parse_args()


def _load_requests(output_dir: Path) -> tuple[pd.DataFrame, Path]:
    request_path = output_dir / ROAD_ROUTE_REQUESTS_FILENAME
    if not request_path.exists():
        return pd.DataFrame(), request_path
    try:
        return pd.read_csv(request_path), request_path
    except Exception:
        return pd.DataFrame(), request_path


def _first_leg_points(requests: pd.DataFrame) -> list[tuple[float, float]]:
    row = requests.iloc[0]
    return [
        (float(row["origin_latitude"]), float(row["origin_longitude"])),
        (float(row["destination_latitude"]), float(row["destination_longitude"])),
    ]


def check_route_source(
    *,
    output_dir: str | Path,
    osrm_base_url: str = OSRM_LOCAL_BASE_URL,
    allow_public_osrm: bool = False,
    probe: bool = False,
    timeout_seconds: int = 5,
) -> tuple[int, list[str]]:
    """Return a process-style status code and human-readable route-source checks."""

    messages: list[str] = []
    try:
        validate_route_fetch_policy(osrm_base_url, allow_public_osrm=allow_public_osrm)
    except ValueError as exc:
        return 2, [f"FAILED route-source policy check: {exc}"]

    requests, request_path = _load_requests(Path(output_dir))
    if requests.empty:
        return 1, [f"FAILED route-source readiness: missing or empty {request_path}"]

    messages.append(f"Route request manifest: {request_path}")
    messages.append(f"Requested legs: {len(requests)}")
    messages.append(f"OSRM endpoint: {osrm_base_url}")

    required_columns = {
        "origin_latitude",
        "origin_longitude",
        "destination_latitude",
        "destination_longitude",
        "cache_key",
        "cache_path",
    }
    missing = sorted(required_columns - set(requests.columns))
    if missing:
        return 1, [*messages, f"FAILED route-source readiness: request manifest missing columns {missing}"]

    if not probe:
        messages.append("Probe: skipped")
        messages.append("PASSED route-source readiness precheck")
        return 0, messages

    try:
        payload = fetch_osrm_payload(
            _first_leg_points(requests),
            osrm_base_url=osrm_base_url,
            timeout_seconds=int(timeout_seconds),
        )
    except Exception as exc:
        return 1, [*messages, f"FAILED route-source probe: {type(exc).__name__}: {exc}"]

    routes = payload.get("routes", []) if isinstance(payload, dict) else []
    if not routes:
        return 1, [*messages, "FAILED route-source probe: OSRM response did not include routes"]
    route = routes[0]
    missing_fields = [field for field in ("distance", "duration", "geometry") if field not in route]
    if missing_fields:
        return 1, [*messages, f"FAILED route-source probe: route missing fields {missing_fields}"]
    messages.append("Probe: passed")
    messages.append("PASSED route-source readiness check")
    return 0, messages


def main() -> int:
    args = parse_args()
    status, messages = check_route_source(
        output_dir=args.output_dir,
        osrm_base_url=args.osrm_base_url,
        allow_public_osrm=bool(args.allow_public_osrm),
        probe=bool(args.probe),
        timeout_seconds=int(args.timeout_seconds),
    )
    for message in messages:
        print(message)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
