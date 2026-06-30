"""Artifact freshness metadata for production dashboard outputs."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import TripConfig

ARTIFACT_METADATA_FILE = "production_artifact_metadata.json"
ARTIFACT_CONTRACT_VERSION = "statewide-nature-artifacts-v1"


def config_hash(config: TripConfig) -> str:
    payload = json.dumps(config.to_dict(), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def artifact_metadata_payload(config: TripConfig, *, artifact_files: list[str] | None = None) -> dict[str, Any]:
    config_fingerprint = config_hash(config)
    configured_run_id = str(config.get("run", "run_id", "auto") or "auto")
    run_id = f"auto_{config_fingerprint}" if configured_run_id == "auto" else configured_run_id
    return {
        "artifact_contract_version": ARTIFACT_CONTRACT_VERSION,
        "scenario": str(config.get("trip", "scenario", "california_coast")),
        "interest_profile": str(
            config.get("trip", "interest_profile", config.get("interest", "mode", "balanced_interest"))
        ),
        "interest_mode": str(config.get("interest", "mode", "balanced_interest")),
        "trip_days": int(config.get("trip", "trip_days", 7)),
        "start_city_options": list(config.get("trip", "start_city_options", [])),
        "end_city_options": list(config.get("trip", "end_city_options", [])),
        "run_live_apis": bool(config.get("enrichment", "run_live_apis", False)),
        "catalog_snapshot_id": str(config.get("data", "catalog_snapshot_id", "california_v1")),
        "context_snapshot_id": str(config.get("data", "context_snapshot_id", "context_static_demo_2026_06")),
        "refresh_policy": str(config.get("data", "refresh_policy", "never")),
        "run_id": run_id,
        "run_role": str(config.get("run", "role", "demonstration")),
        "config_hash": config_fingerprint,
        "source_path": config.source_path,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "artifact_files": artifact_files or [],
    }


def write_artifact_metadata(
    output_dir: str | Path,
    config: TripConfig,
    *,
    artifact_files: list[str] | None = None,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / ARTIFACT_METADATA_FILE
    path.write_text(json.dumps(artifact_metadata_payload(config, artifact_files=artifact_files), indent=2), encoding="utf-8")
    return path


def read_artifact_metadata(output_dir: str | Path) -> dict[str, Any]:
    path = Path(output_dir) / ARTIFACT_METADATA_FILE
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def artifact_metadata_matches(output_dir: str | Path, config: TripConfig) -> bool:
    metadata = read_artifact_metadata(output_dir)
    if not metadata:
        return False
    expected = artifact_metadata_payload(config)
    keys = [
        "artifact_contract_version",
        "scenario",
        "interest_profile",
        "interest_mode",
        "trip_days",
        "start_city_options",
        "end_city_options",
        "run_live_apis",
        "catalog_snapshot_id",
        "context_snapshot_id",
        "refresh_policy",
        "run_id",
        "run_role",
        "config_hash",
    ]
    return all(metadata.get(key) == expected.get(key) for key in keys)
