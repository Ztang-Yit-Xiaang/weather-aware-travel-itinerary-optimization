from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from itinerary_system.routing import (
    ROAD_ROUTE_CACHE_AUDIT_FILENAME,
    ROAD_ROUTE_CACHE_FILENAME,
    ROAD_ROUTE_REQUESTS_FILENAME,
    freeze_route_evidence_bundle,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def write_bundle(root: Path, *, snap_validated: bool = True) -> Path:
    keys = ("key_a", "key_b")
    pd.DataFrame(
        [
            {"cache_key": key, "origin_latitude": 1, "origin_longitude": 2, "destination_latitude": 3, "destination_longitude": 4}
            for key in keys
        ]
    ).to_csv(root / ROAD_ROUTE_REQUESTS_FILENAME, index=False)
    pd.DataFrame(
        [
            {
                "cache_key": key,
                "status": "cached_osrm_validated",
                "road_validated": True,
                "snap_validated": snap_validated,
            }
            for key in keys
        ]
    ).to_csv(root / ROAD_ROUTE_CACHE_AUDIT_FILENAME, index=False)
    pd.DataFrame(
        [
            {
                "query_hash": key,
                "origin_id": "start" if index == 0 else "poi",
                "destination_id": "poi" if index == 0 else "end",
                "distance_m": 1000 + index,
                "duration_s": 100 + index,
                "provider": "cached_osrm",
                "road_validated": True,
                "fallback_used": False,
                "retrieved_at": "2026-07-12T00:00:00+00:00",
            }
            for index, key in enumerate(keys)
        ]
    ).to_csv(root / ROAD_ROUTE_CACHE_FILENAME, index=False)
    provenance_path = root / "source-provenance.json"
    provenance_path.write_text(
        json.dumps(
            {
                "provider_id": "local_osrm_california_test",
                "provider_kind": "local_osrm",
                "endpoint": "http://127.0.0.1:5000",
                "osrm_image": f"ghcr.io/project-osrm/osrm-backend@sha256:{'a' * 64}",
                "osrm_release": "26.5.0",
                "osm_pbf_url": "https://download.geofabrik.de/north-america/us/california-260713.osm.pbf",
                "osm_pbf_sha256": "b" * 64,
                "routing_profile": "driving/car.lua",
                "license_or_terms": "OpenStreetMap ODbL and reviewed extract terms",
                "license_url": "https://opendatacommons.org/licenses/odbl/1-0/",
                "reviewed_by": "unit-test-reviewer",
                "reviewed_at": "2026-07-13T12:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    return provenance_path


def test_complete_route_evidence_bundle_is_hashed_and_publication_ready(tmp_path):
    provenance_path = write_bundle(tmp_path)

    result = freeze_route_evidence_bundle(
        tmp_path,
        provider_provenance_path=provenance_path,
        expected_request_count=2,
        as_of=datetime(2026, 7, 13, tzinfo=UTC),
    )

    assert result.publication_ready is True
    assert result.request_count == 2
    assert result.validated_route_count == 2
    assert result.snap_validated_count == 2
    assert result.request_audit_alignment_valid is True
    assert result.cache_alignment_valid is True
    assert len(result.request_set_hash) == 16
    assert set(result.artifact_hashes) == {
        "requests",
        "cache_audit",
        "validated_cache",
        "provider_provenance",
    }
    manifest = json.loads((tmp_path / "route_evidence_bundle_manifest.json").read_text(encoding="utf-8"))
    assert manifest["bundle_id"] == result.bundle_id
    assert manifest["schema_version"] == "route-evidence-bundle-v1"


def test_bundle_fails_closed_on_missing_snap_and_provider_provenance(tmp_path):
    write_bundle(tmp_path, snap_validated=False)

    result = freeze_route_evidence_bundle(
        tmp_path,
        expected_request_count=2,
        as_of=datetime(2026, 7, 13, tzinfo=UTC),
        write=False,
    )

    assert result.publication_ready is False
    assert result.snap_validation_complete is False
    assert result.provider_provenance_valid is False
    assert "snap_validation_incomplete" in result.errors
    assert "provider_provenance_missing" in result.errors


def test_bundle_rejects_request_cache_key_drift_and_stale_cache(tmp_path):
    provenance_path = write_bundle(tmp_path)
    cache = pd.read_csv(tmp_path / ROAD_ROUTE_CACHE_FILENAME)
    cache.loc[0, "query_hash"] = "different_key"
    cache.loc[:, "retrieved_at"] = "2025-01-01T00:00:00+00:00"
    cache.to_csv(tmp_path / ROAD_ROUTE_CACHE_FILENAME, index=False)

    result = freeze_route_evidence_bundle(
        tmp_path,
        provider_provenance_path=provenance_path,
        expected_request_count=2,
        max_cache_age_days=30,
        as_of=datetime(2026, 7, 13, tzinfo=UTC),
        write=False,
    )

    assert result.publication_ready is False
    assert result.cache_alignment_valid is False
    assert result.freshness_valid is False
    assert "validated_cache_key_mismatch" in result.errors
    assert any(error.startswith("cache_evidence_stale:") for error in result.errors)


def test_freeze_route_evidence_script_returns_nonzero_for_incomplete_bundle(tmp_path):
    write_bundle(tmp_path, snap_validated=False)
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "freeze_route_evidence_bundle.py"),
        "--output-dir",
        str(tmp_path),
        "--expected-request-count",
        "2",
        "--require-publication-ready",
    ]

    completed = subprocess.run(command, capture_output=True, text=True, check=False)

    assert completed.returncode == 1
    assert "Publication ready: False" in completed.stdout
    assert "snap_validation_incomplete" in completed.stdout


def test_placeholder_provenance_template_can_never_certify_bundle(tmp_path):
    write_bundle(tmp_path)
    template = REPO_ROOT / "docker" / "osrm" / "source-provenance.example.json"

    result = freeze_route_evidence_bundle(
        tmp_path,
        provider_provenance_path=template,
        expected_request_count=2,
        as_of=datetime(2026, 7, 13, tzinfo=UTC),
        write=False,
    )

    assert result.provider_provenance_valid is False
    assert result.publication_ready is False
    assert "provider_provenance_contains_placeholder" in result.errors
    assert "provider_image_not_digest_pinned" in result.errors


def test_strict_matrix_build_binds_ready_bundle_and_rejects_cache_tampering(tmp_path):
    provenance_path = write_bundle(tmp_path)
    bundle = freeze_route_evidence_bundle(
        tmp_path,
        provider_provenance_path=provenance_path,
        expected_request_count=2,
        as_of=datetime(2026, 7, 13, tzinfo=UTC),
    )
    cache_path = tmp_path / ROAD_ROUTE_CACHE_FILENAME
    manifest_path = tmp_path / "route_evidence_bundle_manifest.json"
    output_dir = tmp_path / "matrix"
    base_command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "build_validated_route_matrix.py"),
        "--input",
        str(cache_path),
        "--context-snapshot-id",
        "context_evidence",
        "--output-dir",
        str(output_dir),
        "--route-evidence-manifest",
        str(manifest_path),
        "--required-sequence",
        "start,poi,end",
        "--require-publication-ready",
    ]

    passed = subprocess.run(base_command, capture_output=True, text=True, check=False)

    assert passed.returncode == 0, passed.stdout + passed.stderr
    assert f"Source bundle: {bundle.bundle_id}" in passed.stdout
    report = json.loads((output_dir / "production_validated_route_matrix_report.json").read_text(encoding="utf-8"))
    assert report["publication_ready"] is True
    assert report["source_bundle_id"] == bundle.bundle_id
    assert report["source_content_sha256"] == bundle.artifact_hashes["validated_cache"]

    cache = pd.read_csv(cache_path)
    cache.loc[0, "duration_s"] += 1
    cache.to_csv(cache_path, index=False)
    failed = subprocess.run(base_command, capture_output=True, text=True, check=False)

    assert failed.returncode == 1
    assert "SHA-256 does not match route-evidence manifest" in failed.stdout
