"""Freeze the local-road legs that day moves need, as one product route delta.

The registered coherent demo matrix only holds the legs of the accepted plan and
of the registered replacements. Moving a stop to another day needs legs between
that stop and the other day's stops and anchors, so the repair solver finds no
feasible plan. This script captures every directed leg among the demo's places
(accepted stops and registered candidates) and between each place and every day
anchor, plus each day's start-to-end leg (a day left without stops), from the
pinned local OSRM provider. It binds the result to the unchanged base matrix.

Route points are the road-snapped endpoints already stored in the base matrix,
so each entity is routed from the same road access point as its existing legs.
The delta replaces the earlier single-leg Pfeiffer -> Point Lobos delta and
contains that leg again. Usage:

    python scripts/build_product_move_route_delta.py --output-dir results/route_deltas/<name> \
        --container-id <osrm container> --provider-provenance <source-provenance.json>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from build_product_route_delta import BASE_PATH, ENDPOINT, runtime_witness, sha256, write_json  # noqa: E402

from itinerary_system.routing import (  # noqa: E402
    ROAD_ROUTE_CACHE_FILENAME,
    build_road_route_cache_from_artifacts,
    build_validated_route_matrix_from_cache,
    freeze_route_evidence_bundle,
)
from itinerary_system.routing.matrix import load_route_matrix_from_cache, validate_route_matrix  # noqa: E402
from itinerary_system.routing.road_cache_builder import osrm_route_url  # noqa: E402

PLACES = (
    "golden_gate_bridge", "stanford_university_main_quad", "yosemite_national_park",
    "sequoia_national_park", "kings_canyon_national_park", "bixby_creek_bridge_viewpoint",
    "stearns_wharf", "griffith_observatory",
    "point_lobos_state_natural_reserve", "pfeiffer_big_sur_state_park",
)
MANIFEST = ROOT / "runs/california-coast-coherent-demo-v1/manifest.json"


def day_anchors() -> list[tuple[str, str]]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    days = sorted(manifest["scenario"]["days"], key=lambda row: row["day"])
    return [(str(row["start_anchor_id"]), str(row["end_anchor_id"])) for row in days]


def road_points(base: dict) -> dict[str, tuple[float, float]]:
    """Road-snapped endpoint of each entity, taken from its existing base legs."""
    points: dict[str, set[tuple[float, float]]] = {}
    for cell in base["cells"]:
        geometry = cell["geometry"]
        if len(geometry) < 2 or cell.get("road_validated") is not True:
            continue
        points.setdefault(cell["origin_id"], set()).add(tuple(geometry[0]))
        points.setdefault(cell["destination_id"], set()).add(tuple(geometry[-1]))
    resolved = {}
    for entity, candidates in points.items():
        # One road access point per entity; tiny float noise across legs is allowed.
        first = sorted(candidates)[0]
        if any(abs(a - b) > 1e-4 for point in candidates for a, b in zip(point, first, strict=True)):
            raise ValueError(f"delta_entity_access_point_ambiguous:{entity}")
        resolved[entity] = first
    return resolved


def directed_pairs(base: dict) -> list[tuple[str, str]]:
    anchors = day_anchors()
    anchor_ids = sorted({entity for pair in anchors for entity in pair})
    wanted = {(a, b) for a in PLACES for b in PLACES if a != b}
    wanted |= {(a, p) for a in anchor_ids for p in PLACES} | {(p, a) for a in anchor_ids for p in PLACES}
    wanted |= {(start, end) for start, end in anchors if start != end}
    existing = {(cell["origin_id"], cell["destination_id"]) for cell in base["cells"]}
    return sorted(wanted - existing)


def build_delta(output: Path, container_id: str, provenance_path: Path) -> dict:
    if output.exists():
        raise ValueError("immutable_delta_output_already_exists")
    base_hash = sha256(BASE_PATH)
    base = json.loads(BASE_PATH.read_text(encoding="utf-8"))
    points = road_points(base)
    pairs = directed_pairs(base)
    missing = sorted({entity for pair in pairs for entity in pair} - points.keys())
    if missing:
        raise ValueError(f"delta_entity_without_road_point:{missing}")
    provenance_bytes = provenance_path.read_bytes()
    witness = runtime_witness(container_id, json.loads(provenance_bytes))
    rows = []
    for index, (origin, destination) in enumerate(pairs, start=1):
        for order, entity in enumerate((origin, destination), start=1):
            rows.append({"method": "product_move_route_delta", "day": index, "stop_order": order,
                         "attraction_name": entity, "latitude": points[entity][0], "longitude": points[entity][1]})
    frame = pd.DataFrame(rows)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "runtime_witness.json", witness)
    write_json(output / "exact_inputs.json", {
        "route_rows": rows, "route_point_source": "road-snapped endpoints of the base matrix legs",
        "route_points": {entity: list(point) for entity, point in sorted(points.items())},
        "directed_pairs": [list(pair) for pair in pairs], "base_matrix_sha256": base_hash,
    })
    (output / "source-provenance.json").write_bytes(provenance_bytes)
    calls: list[dict] = []
    responses: list[dict] = []

    def fetch(route_points, base_url, timeout):
        if base_url != ENDPOINT or len(calls) >= len(pairs):
            raise ValueError("delta_provider_attempt_not_authorized")
        url = osrm_route_url(route_points, osrm_base_url=base_url)
        receipt = {"url": url, "method": "GET", "started_at": datetime.now(UTC).isoformat()}
        calls.append(receipt)
        with requests.Session() as session:
            session.trust_env = False
            response = session.get(url, timeout=timeout, allow_redirects=False)
        receipt.update({"status_code": response.status_code,
                        "response_sha256": hashlib.sha256(response.content).hexdigest(),
                        "completed_at": datetime.now(UTC).isoformat()})
        if response.status_code != 200:
            raise ValueError("delta_provider_http_failure")
        payload = response.json()
        responses.append(payload)
        if payload.get("code") != "Ok":
            raise ValueError("delta_provider_route_failure")
        return payload

    try:
        built = build_road_route_cache_from_artifacts(output_dir=output, route_stops_df=frame,
            cache_dir=output / "raw_cache", fetch_missing=True, osrm_base_url=ENDPOINT,
            max_snap_distance_m=100.0, osrm_fetcher=fetch, write=True)
    finally:
        write_json(output / "provider_calls.json", calls)
        write_json(output / "raw_response.json", responses)
    if len(calls) != len(pairs) or len(built.request_df) != len(pairs) or not built.complete:
        raise ValueError("delta_route_or_snap_incomplete")
    bundle = freeze_route_evidence_bundle(output, provider_provenance_path=output / "source-provenance.json",
        expected_request_count=len(pairs), max_cache_age_days=1.0)
    if not bundle.publication_ready:
        raise ValueError(f"delta_freeze_failed:{bundle.errors}")
    matrix, report = build_validated_route_matrix_from_cache(output / ROAD_ROUTE_CACHE_FILENAME,
        base["context_snapshot_id"], output / "routing", required_sequences=pairs, require_publication_ready=True,
        source_bundle_id=bundle.bundle_id, expected_source_sha256=bundle.artifact_hashes["validated_cache"])
    if set(matrix.cells) != set(pairs) or sha256(BASE_PATH) != base_hash or not report.publication_ready:
        raise ValueError("delta_base_changed_or_matrix_invalid")
    return finalize_delta(output)


def finalize_delta(output: Path) -> dict:
    """Package the frozen evidence into a descriptor; never fetch again."""
    if (output / "descriptor.json").exists():
        raise ValueError("immutable_delta_descriptor_already_exists")
    base = json.loads(BASE_PATH.read_text(encoding="utf-8"))
    base_hash = sha256(BASE_PATH)
    inputs = json.loads((output / "exact_inputs.json").read_text(encoding="utf-8"))
    saved = json.loads((output / "route_evidence_bundle_manifest.json").read_text(encoding="utf-8"))
    calls = json.loads((output / "provider_calls.json").read_text(encoding="utf-8"))
    responses = json.loads((output / "raw_response.json").read_text(encoding="utf-8"))
    pairs = [tuple(pair) for pair in inputs["directed_pairs"]]
    if inputs["base_matrix_sha256"] != base_hash or len(calls) != len(pairs) or len(responses) != len(pairs):
        raise ValueError("delta_capture_identity_mismatch")
    if any(call["status_code"] != 200 for call in calls):
        raise ValueError("delta_provider_http_failure")
    bundle = freeze_route_evidence_bundle(output, provider_provenance_path=output / "source-provenance.json",
        expected_request_count=len(pairs), max_cache_age_days=1.0,
        as_of=datetime.fromisoformat(saved["generated_at"]), write=False)
    if not bundle.publication_ready or bundle.bundle_id != saved["bundle_id"] or bundle.artifact_hashes != saved["artifact_hashes"]:
        raise ValueError("delta_frozen_evidence_mismatch")
    matrix = load_route_matrix_from_cache(output / ROAD_ROUTE_CACHE_FILENAME, base["context_snapshot_id"],
        source_bundle_id=bundle.bundle_id, expected_source_sha256=bundle.artifact_hashes["validated_cache"])
    report = validate_route_matrix(matrix, required_sequences=pairs, require_publication_ready=True)
    descriptor = {"schema_version": "product-route-delta-v1", "activated_in_product": False,
        "base_matrix_path": BASE_PATH.relative_to(ROOT).as_posix(), "base_matrix_sha256": base_hash,
        "base_matrix_id": base["matrix_id"], "context_snapshot_id": base["context_snapshot_id"],
        "delta_bundle_id": bundle.bundle_id, "delta_matrix_id": matrix.matrix_id,
        "delta_manifest_sha256": sha256(output / "route_evidence_bundle_manifest.json"),
        "delta_matrix_sha256": sha256(output / "routing/production_validated_route_matrix.csv"),
        "provider_calls_sha256": sha256(output / "provider_calls.json"),
        "directed_pairs": [list(pair) for pair in pairs], "provider_attempt_count": len(calls),
        "raw_response_sha256": sha256(output / "raw_response.json"),
        "runtime_witness_sha256": sha256(output / "runtime_witness.json"),
        "exact_inputs_sha256": sha256(output / "exact_inputs.json"),
        "publication_ready": bundle.publication_ready and report.publication_ready}
    write_json(output / "descriptor.json", descriptor)
    return descriptor


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--container-id")
    parser.add_argument("--provider-provenance", type=Path)
    parser.add_argument("--finalize-existing", action="store_true", help="Package captured evidence without provider calls")
    parser.add_argument("--dry-run", action="store_true", help="List the directed pairs and route points only")
    args = parser.parse_args()
    if args.dry_run:
        base = json.loads(BASE_PATH.read_text(encoding="utf-8"))
        points, pairs = road_points(base), directed_pairs(base)
        print(json.dumps({"pair_count": len(pairs), "entities": sorted({e for p in pairs for e in p}),
                          "without_point": sorted({e for p in pairs for e in p} - points.keys())}, indent=2))
        return 0
    if args.finalize_existing:
        result = finalize_delta(args.output_dir)
    else:
        if not args.container_id or not args.provider_provenance:
            parser.error("new captures require --container-id and --provider-provenance")
        result = build_delta(args.output_dir, args.container_id, args.provider_provenance)
    print(json.dumps({k: v for k, v in result.items() if k != "directed_pairs"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
