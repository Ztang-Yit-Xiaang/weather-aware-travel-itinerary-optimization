"""Hash-bound, additive frozen route evidence; never rewrite registered cells."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from ..routing import ROAD_ROUTE_CACHE_FILENAME, freeze_route_evidence_bundle
from ..routing.matrix import load_route_matrix_from_cache, validate_route_matrix
from .workspace import WorkspaceError


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _contained(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    if Path(relative).is_absolute() or not candidate.is_relative_to(root.resolve()):
        raise ValueError("route_delta_path_not_contained")
    return candidate


def configured_route_matrix(root: Path, run_id: str, base: dict[str, Any]) -> dict[str, Any]:
    registry_path = root / "configs/product_route_delta.json"
    if not registry_path.is_file():
        return base
    try:
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        if (
            set(registry) != {"schema_version", "run_id", "descriptor_path", "descriptor_sha256"}
            or registry["schema_version"] != "product-route-delta-registration-v1"
        ):
            raise ValueError("route_delta_registration_invalid")
        if registry["run_id"] != run_id:
            return base
        descriptor_path = _contained(root, registry["descriptor_path"])
        return load_frozen_route_delta(root, descriptor_path, registry["descriptor_sha256"], base)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise WorkspaceError("route_delta_evidence_invalid", 409) from exc


def load_frozen_route_delta(
    root: Path, descriptor_path: Path, expected_hash: str, base: dict[str, Any]
) -> dict[str, Any]:
    if _sha256(descriptor_path) != expected_hash:
        raise ValueError("route_delta_descriptor_hash_mismatch")
    descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
    if descriptor.get("schema_version") != "product-route-delta-v1" or descriptor.get("publication_ready") is not True:
        raise ValueError("route_delta_descriptor_invalid")
    base_path = _contained(root, descriptor["base_matrix_path"])
    if (
        _sha256(base_path) != descriptor["base_matrix_sha256"]
        or json.loads(base_path.read_text(encoding="utf-8")) != base
    ):
        raise ValueError("route_delta_base_hash_mismatch")
    if (
        descriptor["base_matrix_id"] != base["matrix_id"]
        or descriptor["context_snapshot_id"] != base["context_snapshot_id"]
    ):
        raise ValueError("route_delta_context_mismatch")
    directory = descriptor_path.parent
    for filename, field in (
        ("route_evidence_bundle_manifest.json", "delta_manifest_sha256"),
        ("routing/production_validated_route_matrix.csv", "delta_matrix_sha256"),
        ("provider_calls.json", "provider_calls_sha256"),
        ("raw_response.json", "raw_response_sha256"),
        ("runtime_witness.json", "runtime_witness_sha256"),
        ("exact_inputs.json", "exact_inputs_sha256"),
    ):
        if _sha256(directory / filename) != descriptor[field]:
            raise ValueError("route_delta_artifact_hash_mismatch")
    saved = json.loads((directory / "route_evidence_bundle_manifest.json").read_text(encoding="utf-8"))
    # Recheck the recorded freeze instant, not today's time: historical cells
    # retain their original evidence age and are never relabeled as fresh.
    audit = freeze_route_evidence_bundle(
        directory,
        provider_provenance_path=directory / "source-provenance.json",
        expected_request_count=len(descriptor["directed_pairs"]),
        max_cache_age_days=saved["max_allowed_cache_age_days"],
        as_of=datetime.fromisoformat(saved["generated_at"]),
        write=False,
    )
    if (
        not audit.publication_ready
        or audit.bundle_id != descriptor["delta_bundle_id"]
        or audit.artifact_hashes != saved["artifact_hashes"]
    ):
        raise ValueError("route_delta_freeze_mismatch")
    matrix = load_route_matrix_from_cache(
        directory / ROAD_ROUTE_CACHE_FILENAME,
        base["context_snapshot_id"],
        source_bundle_id=audit.bundle_id,
        expected_source_sha256=audit.artifact_hashes["validated_cache"],
    )
    pairs = {tuple(pair) for pair in descriptor["directed_pairs"]}
    if matrix.matrix_id != descriptor["delta_matrix_id"] or set(matrix.cells) != pairs:
        raise ValueError("route_delta_pair_or_matrix_mismatch")
    if not validate_route_matrix(
        matrix, required_sequences=list(pairs), require_publication_ready=True
    ).publication_ready:
        raise ValueError("route_delta_matrix_ineligible")
    old_pairs = {(cell["origin_id"], cell["destination_id"]) for cell in base["cells"]}
    if old_pairs & pairs:
        raise ValueError("route_delta_must_not_replace_base_cells")
    combined = deepcopy(base)
    combined["cells"].extend(json.loads(json.dumps(asdict(cell))) for cell in matrix.cells.values())
    combined["entity_ids"] = sorted(set(base["entity_ids"]) | set(matrix.entity_ids))
    combined["matrix_id"] = f"route_matrix_delta_{expected_hash[:16]}"
    combined["source_bundle_id"] = f"route_bundle_composite_{expected_hash[:16]}"
    # This is the actual immutable descriptor hash, not an invented cache hash.
    combined["source_content_sha256"] = expected_hash
    return combined
