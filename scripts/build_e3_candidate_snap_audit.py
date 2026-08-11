"""Audit frozen E3 candidate snap distances against an explicit OSRM endpoint."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from itinerary_system.routing import route_anchor_key, validate_route_fetch_policy  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-catalog", required=True)
    parser.add_argument("--provider-provenance", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--osrm-base-url", default="http://127.0.0.1:5000")
    parser.add_argument("--max-snap-distance-m", type=float, default=100.0)
    parser.add_argument("--timeout-seconds", type=int, default=10)
    parser.add_argument("--require-complete-fetch", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    catalog_path = Path(args.candidate_catalog)
    provenance_path = Path(args.provider_provenance)
    output_dir = Path(args.output_dir)
    for path in (catalog_path, provenance_path):
        if not path.exists():
            raise FileNotFoundError(path)
    if output_dir.exists():
        raise ValueError(f"immutable candidate snap audit already exists: {output_dir}")
    validate_route_fetch_policy(str(args.osrm_base_url), allow_public_osrm=False)
    threshold = float(args.max_snap_distance_m)
    if not math.isfinite(threshold) or threshold <= 0:
        raise ValueError("max snap distance must be positive and finite")

    rows = []
    with catalog_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for record in csv.DictReader(handle):
            if str(record.get("itinerary_eligible") or "").strip().lower() not in {"1", "true", "yes"}:
                continue
            entity_id = route_anchor_key(record.get("name"))
            try:
                latitude = float(record.get("latitude") or "")
                longitude = float(record.get("longitude") or "")
            except (TypeError, ValueError):
                continue
            row = _nearest_row(
                entity_id=entity_id,
                name=str(record.get("name") or entity_id),
                city=str(record.get("city") or ""),
                latitude=latitude,
                longitude=longitude,
                osrm_base_url=str(args.osrm_base_url),
                threshold=threshold,
                timeout_seconds=int(args.timeout_seconds),
            )
            rows.append(row)

    output_dir.mkdir(parents=True)
    audit_path = output_dir / "candidate_snap_audit.csv"
    fieldnames = tuple(rows[0]) if rows else (
        "entity_id",
        "name",
        "city",
        "status",
        "snap_validated",
    )
    with audit_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    valid_count = sum(bool(row.get("snap_validated")) for row in rows)
    error_count = sum(str(row.get("status")) != "ok" for row in rows)
    manifest = {
        "candidate_catalog": str(catalog_path),
        "candidate_catalog_sha256": _sha256(catalog_path),
        "provider_provenance": str(provenance_path),
        "provider_provenance_sha256": _sha256(provenance_path),
        "osrm_base_url": str(args.osrm_base_url),
        "max_snap_distance_m": threshold,
        "candidate_count": len(rows),
        "snap_validated_count": valid_count,
        "fetch_error_count": error_count,
        "audit_sha256": _sha256(audit_path),
        "generated_at": datetime.now(UTC).isoformat(),
        "schema_version": "e3-candidate-snap-audit-v1",
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + chr(10), encoding="utf-8")
    print(f"Candidates audited: {len(rows)}")
    print(f"Snap validated: {valid_count}")
    print(f"Fetch errors: {error_count}")
    print(f"Wrote {manifest_path}")
    if args.require_complete_fetch and error_count:
        return 1
    return 0


def _nearest_row(
    *,
    entity_id: str,
    name: str,
    city: str,
    latitude: float,
    longitude: float,
    osrm_base_url: str,
    threshold: float,
    timeout_seconds: int,
) -> dict[str, Any]:
    coordinates = f"{longitude},{latitude}"
    query = urllib.parse.urlencode({"number": 1})
    url = f"{osrm_base_url.rstrip('/')}/nearest/v1/driving/{coordinates}?{query}"
    retrieved_at = datetime.now(UTC).isoformat()
    try:
        with urllib.request.urlopen(url, timeout=timeout_seconds) as response:  # nosec B310 - explicit local endpoint.
            payload = json.loads(response.read().decode("utf-8"))
        waypoint = payload["waypoints"][0]
        snapped_longitude, snapped_latitude = waypoint["location"]
        snap_distance = float(waypoint["distance"])
        status = "ok" if payload.get("code") == "Ok" else f"osrm_code:{payload.get('code')}"
    except Exception as exc:
        snapped_latitude = None
        snapped_longitude = None
        snap_distance = None
        status = f"fetch_error:{type(exc).__name__}"
    snap_validated = bool(
        status == "ok"
        and snap_distance is not None
        and math.isfinite(snap_distance)
        and snap_distance <= threshold
    )
    return {
        "entity_id": entity_id,
        "name": name,
        "city": city,
        "input_latitude": latitude,
        "input_longitude": longitude,
        "snapped_latitude": snapped_latitude,
        "snapped_longitude": snapped_longitude,
        "snap_distance_m": snap_distance,
        "snap_threshold_m": threshold,
        "snap_validated": snap_validated,
        "status": status,
        "provider": "local_osrm_nearest",
        "retrieved_at": retrieved_at,
        "query_url": url,
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
