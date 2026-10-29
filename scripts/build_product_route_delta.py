"""Freeze one authentic local-road delta without changing product registration.

The registered demo only contains a frozen matrix, not the original cache/audit
CSV inputs required by the general E3 expansion builder. Keep the new directed
leg separate and bind its descriptor to that unchanged matrix instead of
inventing historical cache rows or modifying an existing evidence bundle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from itinerary_system.product_app.poi_catalog import load_product_poi_catalog  # noqa: E402
from itinerary_system.routing import (  # noqa: E402
    ROAD_ROUTE_CACHE_FILENAME,
    build_road_route_cache_from_artifacts,
    build_validated_route_matrix_from_cache,
    freeze_route_evidence_bundle,
)
from itinerary_system.routing.matrix import load_route_matrix_from_cache, validate_route_matrix  # noqa: E402
from itinerary_system.routing.road_cache_builder import osrm_route_url  # noqa: E402

PAIR = ("pfeiffer_big_sur_state_park", "point_lobos_state_natural_reserve")
ENDPOINT = "http://127.0.0.1:5000"
BASE_PATH = ROOT / "runs/california-coast-coherent-demo-v1/routing/route_matrix_b33928f302ba9055.json"
CATALOG_PATH = ROOT / "configs/product_poi_catalogs/california_coast_approved_seven_day/manifest.json"
DATASET = ROOT / "docker/osrm/data"


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def validate_runtime_witness(container: dict, provenance: dict, pbf_sha256: str) -> None:
    image = provenance.get("osrm_image", "")
    if not image.startswith("ghcr.io/project-osrm/osrm-backend@sha256:"):
        raise ValueError("provider_image_not_pinned")
    if provenance.get("endpoint") != ENDPOINT or provenance.get("provider_kind") != "local_osrm":
        raise ValueError("provider_endpoint_not_approved")
    if container.get("running") is not True:
        raise ValueError("provider_container_not_running")
    if container.get("configured_image") != image or container.get("image_id") != image.split("@", 1)[1]:
        raise ValueError("running_provider_image_mismatch")
    if container.get("command") != ["osrm-routed", "--algorithm", "mld", "/data/california.osrm"]:
        raise ValueError("running_provider_dataset_mismatch")
    mounts = [row for row in container.get("mounts", []) if row.get("Destination") == "/data"]
    if len(mounts) != 1 or mounts[0].get("RW") is not False or Path(mounts[0].get("Source", "")).resolve() != DATASET.resolve():
        raise ValueError("running_provider_mount_mismatch")
    if container.get("ports", {}).get("5000/tcp") != [{"HostIp": "127.0.0.1", "HostPort": "5000"}]:
        raise ValueError("running_provider_port_mismatch")
    if provenance.get("osm_pbf_sha256") != pbf_sha256:
        raise ValueError("running_provider_pbf_mismatch")


def runtime_witness(container_id: str, provenance: dict) -> dict:
    # Deliberately select fields: a full docker inspection includes environment.
    template = '{"container_id":{{json .Id}},"running":{{json .State.Running}},"configured_image":{{json .Config.Image}},"image_id":{{json .Image}},"command":{{json .Config.Cmd}},"mounts":{{json .Mounts}},"ports":{{json .NetworkSettings.Ports}}}'
    result = subprocess.run(["docker", "inspect", container_id, "--format", template], check=True,
        capture_output=True, text=True, timeout=30)
    container = json.loads(result.stdout)
    pbf_hash = sha256(DATASET / "california.osm.pbf")
    validate_runtime_witness(container, provenance, pbf_hash)
    return {"observed_at": datetime.now(UTC).isoformat(), "container": container, "pbf_sha256": pbf_hash,
        "scope": "Observed running image, command, read-only mount, loopback binding and input PBF hash; not a new graph-build attestation."}


def route_inputs(base: dict) -> tuple[pd.DataFrame, dict, dict]:
    catalog = load_product_poi_catalog(CATALOG_PATH)
    rows, overrides, records = [], {}, {}
    for index, place_id in enumerate(PAIR):
        place = catalog.get_place(place_id)
        if place is None:
            raise ValueError("delta_place_missing")
        access = [point for point in place.access_points if point.road_validated
            and point.source_ref == base["matrix_id"] and point.access_type == "road_route_endpoint"]
        if len(access) != 1:
            raise ValueError("delta_access_point_ambiguous_or_unverified")
        point = access[0]
        if not point.evidence_refs or not set(point.evidence_refs).issubset({cell["query_hash"] for cell in base["cells"]}):
            raise ValueError("delta_access_evidence_mismatch")
        rows.append({"method": "product_directed_route_delta", "day": 5, "stop_order": index + 1,
            "attraction_name": place_id, "latitude": place.display_coordinate.latitude,
            "longitude": place.display_coordinate.longitude})
        overrides[place_id] = (point.coordinate.latitude, point.coordinate.longitude)
        records[place_id] = {"display_coordinate": asdict(place.display_coordinate), "route_access": asdict(point)}
    return pd.DataFrame(rows), overrides, {"catalog_sha256": catalog.manifest_sha256, "places": records}


def build_delta(output: Path, container_id: str, provenance_path: Path) -> dict:
    if output.exists():
        raise ValueError("immutable_delta_output_already_exists")
    base_hash = sha256(BASE_PATH)
    base = json.loads(BASE_PATH.read_text(encoding="utf-8"))
    if any((cell["origin_id"], cell["destination_id"]) == PAIR for cell in base["cells"]):
        raise ValueError("delta_pair_already_present")
    provenance_bytes = provenance_path.read_bytes()
    provenance = json.loads(provenance_bytes)
    witness = runtime_witness(container_id, provenance)
    rows, overrides, inputs = route_inputs(base)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "runtime_witness.json", witness)
    write_json(output / "exact_inputs.json", {**inputs, "route_rows": rows.to_dict("records"),
        "route_point_overrides": overrides, "base_matrix_sha256": base_hash})
    (output / "source-provenance.json").write_bytes(provenance_bytes)
    calls = []

    def fetch(points, base_url, timeout):
        if calls or base_url != ENDPOINT:
            raise ValueError("delta_provider_attempt_not_authorized")
        url = osrm_route_url(points, osrm_base_url=base_url)
        receipt = {"url": url, "method": "GET", "started_at": datetime.now(UTC).isoformat()}
        calls.append(receipt)
        write_json(output / "provider_calls.json", calls)
        with requests.Session() as session:
            session.trust_env = False
            response = session.get(url, timeout=timeout, allow_redirects=False)
        receipt.update({"status_code": response.status_code, "response_sha256": hashlib.sha256(response.content).hexdigest(),
            "completed_at": datetime.now(UTC).isoformat()})
        (output / "raw_response.json").write_bytes(response.content)
        write_json(output / "provider_calls.json", calls)
        if response.status_code != 200:
            raise ValueError("delta_provider_http_failure")
        payload = response.json()
        if payload.get("code") != "Ok":
            raise ValueError("delta_provider_route_failure")
        return payload

    built = build_road_route_cache_from_artifacts(output_dir=output, route_stops_df=rows,
        cache_dir=output / "raw_cache", fetch_missing=True, osrm_base_url=ENDPOINT,
        max_snap_distance_m=100.0, route_point_overrides=overrides, osrm_fetcher=fetch, write=True)
    if len(calls) != 1 or len(built.request_df) != 1 or not built.complete:
        raise ValueError("delta_route_or_snap_incomplete")
    bundle = freeze_route_evidence_bundle(output, provider_provenance_path=output / "source-provenance.json",
        expected_request_count=1, max_cache_age_days=1.0)
    if not bundle.publication_ready:
        raise ValueError(f"delta_freeze_failed:{bundle.errors}")
    matrix, report = build_validated_route_matrix_from_cache(output / ROAD_ROUTE_CACHE_FILENAME,
        base["context_snapshot_id"], output / "routing", required_sequences=[PAIR], require_publication_ready=True,
        source_bundle_id=bundle.bundle_id, expected_source_sha256=bundle.artifact_hashes["validated_cache"])
    matrix.cell(*PAIR).require_publication_eligible()
    if sha256(BASE_PATH) != base_hash or not report.publication_ready:
        raise ValueError("delta_base_changed_or_matrix_invalid")
    return finalize_delta(output)


def finalize_delta(output: Path) -> dict:
    """Complete packaging from the original frozen evidence; never fetch again."""
    if (output / "descriptor.json").exists():
        raise ValueError("immutable_delta_descriptor_already_exists")
    base = json.loads(BASE_PATH.read_text(encoding="utf-8"))
    base_hash = sha256(BASE_PATH)
    inputs = json.loads((output / "exact_inputs.json").read_text(encoding="utf-8"))
    saved = json.loads((output / "route_evidence_bundle_manifest.json").read_text(encoding="utf-8"))
    calls = json.loads((output / "provider_calls.json").read_text(encoding="utf-8"))
    if inputs["base_matrix_sha256"] != base_hash or len(calls) != 1 or calls[0]["status_code"] != 200:
        raise ValueError("delta_capture_identity_mismatch")
    if sha256(output / "raw_response.json") != calls[0]["response_sha256"]:
        raise ValueError("delta_raw_response_mismatch")
    bundle = freeze_route_evidence_bundle(output, provider_provenance_path=output / "source-provenance.json",
        expected_request_count=1, max_cache_age_days=1.0, as_of=datetime.fromisoformat(saved["generated_at"]), write=False)
    if not bundle.publication_ready or bundle.bundle_id != saved["bundle_id"] or bundle.artifact_hashes != saved["artifact_hashes"]:
        raise ValueError("delta_frozen_evidence_mismatch")
    matrix = load_route_matrix_from_cache(output / ROAD_ROUTE_CACHE_FILENAME, base["context_snapshot_id"],
        source_bundle_id=bundle.bundle_id, expected_source_sha256=bundle.artifact_hashes["validated_cache"])
    report = validate_route_matrix(matrix, required_sequences=[PAIR], require_publication_ready=True)
    descriptor = {"schema_version": "product-route-delta-v1", "activated_in_product": False,
        "base_matrix_path": BASE_PATH.relative_to(ROOT).as_posix(), "base_matrix_sha256": base_hash,
        "base_matrix_id": base["matrix_id"], "context_snapshot_id": base["context_snapshot_id"],
        "delta_bundle_id": bundle.bundle_id, "delta_matrix_id": matrix.matrix_id,
        "delta_manifest_sha256": sha256(output / "route_evidence_bundle_manifest.json"),
        "delta_matrix_sha256": sha256(output / "routing/production_validated_route_matrix.csv"),
        "provider_calls_sha256": sha256(output / "provider_calls.json"),
        "directed_pairs": [list(PAIR)], "provider_attempt_count": len(calls),
        "raw_response_sha256": calls[0]["response_sha256"], "catalog_sha256": inputs["catalog_sha256"],
        "runtime_witness_sha256": sha256(output / "runtime_witness.json"),
        "exact_inputs_sha256": sha256(output / "exact_inputs.json"),
        "publication_ready": bundle.publication_ready and report.publication_ready}
    write_json(output / "descriptor.json", descriptor)
    return descriptor


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--container-id")
    parser.add_argument("--provider-provenance", type=Path)
    parser.add_argument("--finalize-existing", action="store_true", help="Verify/package captured evidence without any provider call")
    args = parser.parse_args()
    if args.finalize_existing:
        result = finalize_delta(args.output_dir)
    else:
        if not args.container_id or not args.provider_provenance:
            parser.error("new captures require --container-id and --provider-provenance")
        result = build_delta(args.output_dir, args.container_id, args.provider_provenance)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
