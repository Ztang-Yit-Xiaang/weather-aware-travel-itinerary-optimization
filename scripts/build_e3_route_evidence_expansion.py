"""Expand a frozen E2 route bundle for the conservative E3 search universe."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from itinerary_system.routing import (  # noqa: E402
    ROAD_ROUTE_CACHE_AUDIT_FILENAME,
    ROAD_ROUTE_CACHE_FILENAME,
    ROAD_ROUTE_REQUESTS_FILENAME,
    build_road_route_cache_from_artifacts,
    build_validated_route_matrix_from_cache,
    freeze_route_evidence_bundle,
    route_anchor_key,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-bundle-dir", required=True)
    parser.add_argument("--route-coverage", required=True)
    parser.add_argument("--scenarios", required=True)
    parser.add_argument("--route-stops", required=True)
    parser.add_argument("--parent-method-id", default="hierarchical_gurobi_pipeline")
    parser.add_argument("--context-snapshot-id", required=True)
    parser.add_argument("--provider-provenance", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cache-dir", default="results/cache")
    parser.add_argument("--osrm-base-url", default="http://127.0.0.1:5000")
    parser.add_argument("--max-snap-distance-m", type=float, default=100.0)
    parser.add_argument("--max-cache-age-days", type=float, default=30.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_dir = Path(args.base_bundle_dir)
    coverage_path = Path(args.route_coverage)
    scenarios_path = Path(args.scenarios)
    route_stops_path = Path(args.route_stops)
    provenance_path = Path(args.provider_provenance)
    output_dir = Path(args.output_dir)
    for path in (base_dir, coverage_path, scenarios_path, route_stops_path, provenance_path):
        if not path.exists():
            raise FileNotFoundError(path)
    if output_dir.exists():
        raise ValueError(f"immutable E3 route expansion already exists: {output_dir}")

    coverage = _read_json(coverage_path)
    missing_pairs = tuple(
        (route_anchor_key(record.get("origin_id")), route_anchor_key(record.get("destination_id")))
        for record in coverage.get("missing_pairs", ())
    )
    missing_pairs = tuple(dict.fromkeys(pair for pair in missing_pairs if pair[0] and pair[1] and pair[0] != pair[1]))
    if not missing_pairs:
        raise ValueError("route coverage report contains no missing non-identity pairs")

    coordinates = _route_coordinates(
        route_stops_path,
        scenarios_path,
        base_request_path=base_dir / ROAD_ROUTE_REQUESTS_FILENAME,
        method_id=str(args.parent_method_id),
    )
    unlocated = tuple(
        sorted({entity_id for pair in missing_pairs for entity_id in pair if entity_id not in coordinates})
    )
    if unlocated:
        raise ValueError(f"missing coordinates for E3 route entities: {unlocated}")

    output_dir.mkdir(parents=True)
    expansion_dir = output_dir / "expansion_only"
    route_frame = _missing_pair_frame(missing_pairs, coordinates)
    expansion = build_road_route_cache_from_artifacts(
        output_dir=expansion_dir,
        route_stops_df=route_frame,
        cache_dir=Path(args.cache_dir),
        fetch_missing=True,
        osrm_base_url=str(args.osrm_base_url),
        max_snap_distance_m=float(args.max_snap_distance_m),
        write=True,
    )
    if not expansion.complete:
        failed = expansion.audit_df.loc[
            ~(expansion.audit_df["road_validated"].astype(bool) & expansion.audit_df["snap_validated"].astype(bool))
        ]
        raise ValueError(f"E3 OSRM expansion incomplete: {failed.to_dict('records')[:5]}")

    merged = {
        ROAD_ROUTE_REQUESTS_FILENAME: _merge_frames(
            pd.read_csv(base_dir / ROAD_ROUTE_REQUESTS_FILENAME),
            expansion.request_df,
            key="cache_key",
        ),
        ROAD_ROUTE_CACHE_AUDIT_FILENAME: _merge_frames(
            pd.read_csv(base_dir / ROAD_ROUTE_CACHE_AUDIT_FILENAME),
            expansion.audit_df,
            key="cache_key",
        ),
        ROAD_ROUTE_CACHE_FILENAME: _merge_frames(
            pd.read_csv(base_dir / ROAD_ROUTE_CACHE_FILENAME),
            expansion.cache_df,
            key="query_hash",
        ),
    }
    for filename, frame in merged.items():
        frame.to_csv(output_dir / filename, index=False)

    copied_provenance = output_dir / "source-provenance.json"
    shutil.copy2(provenance_path, copied_provenance)
    expected_count = len(merged[ROAD_ROUTE_REQUESTS_FILENAME])
    bundle = freeze_route_evidence_bundle(
        output_dir,
        provider_provenance_path=copied_provenance,
        expected_request_count=expected_count,
        max_cache_age_days=float(args.max_cache_age_days),
    )
    if not bundle.publication_ready:
        raise ValueError(f"expanded route bundle is not publication ready: {bundle.errors}")

    matrix, matrix_report = build_validated_route_matrix_from_cache(
        output_dir / ROAD_ROUTE_CACHE_FILENAME,
        str(args.context_snapshot_id),
        output_dir / "routing",
        require_publication_ready=True,
        source_bundle_id=bundle.bundle_id,
        expected_source_sha256=str(bundle.artifact_hashes["validated_cache"]),
    )
    for origin_id, destination_id in missing_pairs:
        matrix.cell(origin_id, destination_id).require_publication_eligible()

    closeout = {
        "base_bundle_dir": str(base_dir),
        "base_bundle_manifest_sha256": _sha256(base_dir / "route_evidence_bundle_manifest.json"),
        "route_coverage_sha256": _sha256(coverage_path),
        "scenario_sha256": _sha256(scenarios_path),
        "expansion_request_count": len(missing_pairs),
        "combined_request_count": expected_count,
        "bundle_id": bundle.bundle_id,
        "request_set_hash": bundle.request_set_hash,
        "validated_cache_sha256": bundle.artifact_hashes["validated_cache"],
        "matrix_id": matrix.matrix_id,
        "matrix_cell_count": len(matrix.cells),
        "matrix_publication_ready": matrix_report.publication_ready,
        "max_snap_distance_m": float(args.max_snap_distance_m),
        "provider_provenance": str(copied_provenance),
        "schema_version": "e3-route-evidence-expansion-closeout-v1",
    }
    closeout_path = output_dir / "closeout.json"
    closeout_path.write_text(
        json.dumps(closeout, indent=2, sort_keys=True) + chr(10),
        encoding="utf-8",
    )
    print(f"Expansion routes: {len(missing_pairs)}")
    print(f"Combined routes: {expected_count}")
    print(f"Route bundle: {bundle.bundle_id}")
    print(f"Route matrix: {matrix.matrix_id} ({len(matrix.cells)} cells)")
    print(f"Publication ready: {bundle.publication_ready and matrix_report.publication_ready}")
    print(f"Wrote {closeout_path}")
    return 0


def _route_coordinates(
    route_stops_path: Path,
    scenarios_path: Path,
    *,
    base_request_path: Path,
    method_id: str,
) -> dict[str, tuple[float, float]]:
    coordinates: dict[str, tuple[float, float]] = {}

    def add(entity: Any, latitude: Any, longitude: Any) -> None:
        entity_id = route_anchor_key(entity)
        if not entity_id:
            return
        try:
            coordinates[entity_id] = (float(latitude), float(longitude))
        except (TypeError, ValueError):
            return

    with route_stops_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("method") or "") != method_id:
                continue
            add(row.get("route_start_name"), row.get("route_start_latitude"), row.get("route_start_longitude"))
            add(row.get("route_end_name"), row.get("route_end_latitude"), row.get("route_end_longitude"))
            add(row.get("attraction_name"), row.get("latitude"), row.get("longitude"))
            add(row.get("hotel_name"), row.get("hotel_latitude"), row.get("hotel_longitude"))
    for row in pd.read_csv(base_request_path).to_dict("records"):
        add(row.get("origin_label"), row.get("origin_latitude"), row.get("origin_longitude"))
        add(
            row.get("destination_label"),
            row.get("destination_latitude"),
            row.get("destination_longitude"),
        )
    for scenario in _read_jsonl(scenarios_path):
        request = dict(scenario.get("request") or {})
        for record in (*request.get("baseline_route", ()), *request.get("candidate_pois", ())):
            add(
                record.get("stop_id") or record.get("poi_id") or record.get("name") or record.get("attraction_name"),
                record.get("latitude"),
                record.get("longitude"),
            )
    return coordinates


def _missing_pair_frame(
    pairs: tuple[tuple[str, str], ...],
    coordinates: dict[str, tuple[float, float]],
) -> pd.DataFrame:
    rows = []
    for index, (origin_id, destination_id) in enumerate(pairs, start=1):
        origin = coordinates[origin_id]
        destination = coordinates[destination_id]
        rows.append(
            {
                "comparison_type": "method",
                "method": "e3_route_evidence_expansion",
                "day": index,
                "stop_order": 0,
                "route_start_name": origin_id,
                "route_start_latitude": origin[0],
                "route_start_longitude": origin[1],
                "route_end_name": destination_id,
                "route_end_latitude": destination[0],
                "route_end_longitude": destination[1],
                "attraction_name": "",
                "latitude": None,
                "longitude": None,
            }
        )
    return pd.DataFrame(rows)


def _merge_frames(base: pd.DataFrame, expansion: pd.DataFrame, *, key: str) -> pd.DataFrame:
    combined = pd.concat((base, expansion), ignore_index=True, sort=False)
    if key not in combined.columns or combined[key].astype(str).str.strip().eq("").any():
        raise ValueError(f"merged route artifact has missing key column values: {key}")
    duplicated = combined[key].astype(str).duplicated(keep=False)
    if duplicated.any():
        raise ValueError(f"merged route artifact has duplicate {key}: {combined.loc[duplicated, key].tolist()[:5]}")
    return combined.sort_values(key).reset_index(drop=True)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    return tuple(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
