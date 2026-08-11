"""Freeze and audit publication-oriented route evidence without provider calls."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd

from .cache import ROAD_ROUTE_CACHE_FILENAME
from .road_cache_builder import ROAD_ROUTE_CACHE_AUDIT_FILENAME, ROAD_ROUTE_REQUESTS_FILENAME

ROUTE_EVIDENCE_MANIFEST_FILENAME = "route_evidence_bundle_manifest.json"
_IMAGE_DIGEST_PATTERN = re.compile(r"^[^\s@]+@sha256:[0-9a-fA-F]{64}$")
_SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")
_PLACEHOLDER_MARKERS = ("replace", "placeholder", "example.invalid", "todo", "tbd")


@dataclass(frozen=True)
class RouteEvidenceBundleAudit:
    """Machine-readable E2 route-evidence freeze finding."""

    bundle_id: str
    output_dir: str
    generated_at: str
    expected_request_count: int | None
    request_count: int
    unique_request_count: int
    validated_route_count: int
    snap_validated_count: int
    cache_row_count: int
    request_set_hash: str
    artifact_hashes: Mapping[str, str]
    status_counts: Mapping[str, int]
    provider_counts: Mapping[str, int]
    earliest_retrieved_at: str
    latest_retrieved_at: str
    maximum_cache_age_days: float | None
    max_allowed_cache_age_days: float
    provider_provenance_valid: bool
    request_audit_alignment_valid: bool
    cache_alignment_valid: bool
    road_validation_complete: bool
    snap_validation_complete: bool
    fallback_free: bool
    freshness_valid: bool
    publication_ready: bool
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_record(self) -> dict[str, Any]:
        return {
            "bundle_id": self.bundle_id,
            "output_dir": self.output_dir,
            "generated_at": self.generated_at,
            "expected_request_count": self.expected_request_count,
            "request_count": self.request_count,
            "unique_request_count": self.unique_request_count,
            "validated_route_count": self.validated_route_count,
            "snap_validated_count": self.snap_validated_count,
            "cache_row_count": self.cache_row_count,
            "request_set_hash": self.request_set_hash,
            "artifact_hashes": dict(self.artifact_hashes),
            "status_counts": dict(self.status_counts),
            "provider_counts": dict(self.provider_counts),
            "earliest_retrieved_at": self.earliest_retrieved_at,
            "latest_retrieved_at": self.latest_retrieved_at,
            "maximum_cache_age_days": self.maximum_cache_age_days,
            "max_allowed_cache_age_days": self.max_allowed_cache_age_days,
            "provider_provenance_valid": self.provider_provenance_valid,
            "request_audit_alignment_valid": self.request_audit_alignment_valid,
            "cache_alignment_valid": self.cache_alignment_valid,
            "road_validation_complete": self.road_validation_complete,
            "snap_validation_complete": self.snap_validation_complete,
            "fallback_free": self.fallback_free,
            "freshness_valid": self.freshness_valid,
            "publication_ready": self.publication_ready,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "schema_version": "route-evidence-bundle-v1",
        }


def freeze_route_evidence_bundle(
    output_dir: str | Path,
    *,
    provider_provenance_path: str | Path | None = None,
    expected_request_count: int | None = None,
    max_cache_age_days: float = 30.0,
    manifest_path: str | Path | None = None,
    as_of: datetime | None = None,
    write: bool = True,
) -> RouteEvidenceBundleAudit:
    """Audit and hash an existing route-evidence bundle without network access."""

    requested_root = Path(output_dir)
    root = requested_root.resolve()
    generated = _utc(as_of or datetime.now(UTC))
    errors: list[str] = []
    warnings: list[str] = []
    if expected_request_count is not None and expected_request_count <= 0:
        raise ValueError("expected_request_count must be positive when provided")
    if not math.isfinite(float(max_cache_age_days)) or float(max_cache_age_days) <= 0:
        raise ValueError("max_cache_age_days must be a positive finite value")

    request_path = root / ROAD_ROUTE_REQUESTS_FILENAME
    audit_path = root / ROAD_ROUTE_CACHE_AUDIT_FILENAME
    cache_path = root / ROAD_ROUTE_CACHE_FILENAME
    paths = {
        "requests": request_path,
        "cache_audit": audit_path,
        "validated_cache": cache_path,
    }
    requests = _read_csv(request_path, "requests", errors)
    audit = _read_csv(audit_path, "cache_audit", errors)
    cache = _read_csv(cache_path, "validated_cache", errors)

    request_keys = _column_values(requests, "cache_key", "requests", errors)
    audit_keys = _column_values(audit, "cache_key", "cache_audit", errors)
    cache_keys = _column_values(cache, "query_hash", "validated_cache", errors)
    request_key_set = set(request_keys)
    audit_key_set = set(audit_keys)
    cache_key_set = set(cache_keys)
    unique_request_count = len(request_key_set)
    request_count = len(requests)
    if request_count != unique_request_count:
        errors.append("request_cache_keys_not_unique")
    if expected_request_count is not None and request_count != expected_request_count:
        errors.append(f"unexpected_request_count:{request_count}!={expected_request_count}")

    request_audit_alignment_valid = (
        bool(request_keys)
        and len(audit) == request_count
        and len(audit_keys) == len(set(audit_keys))
        and request_key_set == audit_key_set
    )
    if not request_audit_alignment_valid:
        errors.append("request_cache_audit_key_mismatch")

    road_flags = _bool_series(audit, "road_validated", "cache_audit", errors)
    snap_flags = _bool_series(audit, "snap_validated", "cache_audit", errors)
    validated_keys = {
        key for key, valid in zip(audit_keys, road_flags, strict=False) if valid and key
    }
    cache_alignment_valid = (
        len(cache_keys) == len(set(cache_keys))
        and cache_key_set == validated_keys
    )
    if not cache_alignment_valid:
        errors.append("validated_cache_key_mismatch")

    cache_road_flags = _bool_series(cache, "road_validated", "validated_cache", errors)
    cache_fallback_flags = _bool_series(cache, "fallback_used", "validated_cache", errors)
    positive_values = _positive_columns(cache, ("distance_m", "duration_s"), errors)
    road_validation_complete = (
        request_count > 0
        and len(road_flags) == request_count
        and all(road_flags)
        and len(cache) == request_count
        and all(cache_road_flags)
        and positive_values
    )
    if not road_validation_complete:
        errors.append("road_validation_incomplete")
    snap_validation_complete = request_count > 0 and len(snap_flags) == request_count and all(snap_flags)
    if not snap_validation_complete:
        errors.append("snap_validation_incomplete")
    fallback_free = len(cache_fallback_flags) == len(cache) and not any(cache_fallback_flags)
    if not fallback_free:
        errors.append("fallback_route_present")

    earliest, latest, maximum_age, freshness_valid = _freshness(
        cache,
        as_of=generated,
        max_cache_age_days=float(max_cache_age_days),
        errors=errors,
    )
    provenance, provider_provenance_valid = _provider_provenance(provider_provenance_path, errors)
    if provider_provenance_path is not None:
        provenance_path = Path(provider_provenance_path).resolve()
        if provenance_path.exists():
            paths["provider_provenance"] = provenance_path

    artifact_hashes = {
        name: _sha256(path) for name, path in paths.items() if path.exists() and path.is_file()
    }
    request_set_hash = _stable_hash(sorted(request_key_set)) if request_key_set else ""
    status_counts = _counts(audit, "status")
    provider_counts = _counts(cache, "provider")
    publication_ready = all(
        (
            request_audit_alignment_valid,
            cache_alignment_valid,
            road_validation_complete,
            snap_validation_complete,
            fallback_free,
            freshness_valid,
            provider_provenance_valid,
            not errors,
        )
    )
    bundle_seed = {
        "artifact_hashes": artifact_hashes,
        "request_set_hash": request_set_hash,
        "provider_provenance": provenance,
        "expected_request_count": expected_request_count,
        "max_cache_age_days": float(max_cache_age_days),
    }
    result = RouteEvidenceBundleAudit(
        bundle_id=f"route_bundle_{_stable_hash(bundle_seed)}",
        output_dir=requested_root.as_posix(),
        generated_at=generated.isoformat(),
        expected_request_count=expected_request_count,
        request_count=request_count,
        unique_request_count=unique_request_count,
        validated_route_count=sum(road_flags),
        snap_validated_count=sum(snap_flags),
        cache_row_count=len(cache),
        request_set_hash=request_set_hash,
        artifact_hashes=artifact_hashes,
        status_counts=status_counts,
        provider_counts=provider_counts,
        earliest_retrieved_at=earliest,
        latest_retrieved_at=latest,
        maximum_cache_age_days=maximum_age,
        max_allowed_cache_age_days=float(max_cache_age_days),
        provider_provenance_valid=provider_provenance_valid,
        request_audit_alignment_valid=request_audit_alignment_valid,
        cache_alignment_valid=cache_alignment_valid,
        road_validation_complete=road_validation_complete,
        snap_validation_complete=snap_validation_complete,
        fallback_free=fallback_free,
        freshness_valid=freshness_valid,
        publication_ready=publication_ready,
        errors=tuple(dict.fromkeys(errors)),
        warnings=tuple(dict.fromkeys(warnings)),
    )
    if write:
        target = Path(manifest_path) if manifest_path is not None else root / ROUTE_EVIDENCE_MANIFEST_FILENAME
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(result.to_record(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def _read_csv(path: Path, label: str, errors: list[str]) -> pd.DataFrame:
    if not path.exists():
        errors.append(f"missing_artifact:{label}")
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as exc:
        errors.append(f"unreadable_artifact:{label}:{type(exc).__name__}")
        return pd.DataFrame()


def _column_values(frame: pd.DataFrame, column: str, label: str, errors: list[str]) -> tuple[str, ...]:
    if column not in frame.columns:
        errors.append(f"missing_column:{label}:{column}")
        return ()
    return tuple(str(value).strip() for value in frame[column].fillna("") if str(value).strip())


def _bool_series(frame: pd.DataFrame, column: str, label: str, errors: list[str]) -> tuple[bool, ...]:
    if column not in frame.columns:
        errors.append(f"missing_column:{label}:{column}")
        return ()
    return tuple(_truthy(value) for value in frame[column])


def _positive_columns(frame: pd.DataFrame, columns: tuple[str, ...], errors: list[str]) -> bool:
    valid = True
    for column in columns:
        if column not in frame.columns:
            errors.append(f"missing_column:validated_cache:{column}")
            valid = False
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any() or not values.gt(0).all():
            errors.append(f"invalid_positive_values:validated_cache:{column}")
            valid = False
    return valid and not frame.empty


def _freshness(
    cache: pd.DataFrame,
    *,
    as_of: datetime,
    max_cache_age_days: float,
    errors: list[str],
) -> tuple[str, str, float | None, bool]:
    if "retrieved_at" not in cache.columns or cache.empty:
        errors.append("cache_freshness_evidence_missing")
        return "", "", None, False
    timestamps = pd.to_datetime(cache["retrieved_at"], utc=True, errors="coerce")
    if timestamps.isna().any():
        errors.append("cache_retrieved_at_invalid")
        return "", "", None, False
    earliest = timestamps.min().to_pydatetime()
    latest = timestamps.max().to_pydatetime()
    maximum_age = max(0.0, (as_of - earliest).total_seconds() / 86_400.0)
    valid = maximum_age <= max_cache_age_days
    if not valid:
        errors.append(f"cache_evidence_stale:{maximum_age:.3f}>{max_cache_age_days:.3f}")
    return earliest.isoformat(), latest.isoformat(), maximum_age, valid


def _provider_provenance(
    path: str | Path | None,
    errors: list[str],
) -> tuple[dict[str, Any], bool]:
    if path is None:
        errors.append("provider_provenance_missing")
        return {}, False
    provenance_path = Path(path)
    if not provenance_path.exists():
        errors.append("provider_provenance_missing")
        return {}, False
    try:
        raw = json.loads(provenance_path.read_text(encoding="utf-8"))
    except Exception as exc:
        errors.append(f"provider_provenance_unreadable:{type(exc).__name__}")
        return {}, False
    if not isinstance(raw, Mapping):
        errors.append("provider_provenance_not_object")
        return {}, False
    record = {str(key): value for key, value in raw.items()}
    required = (
        "provider_id",
        "provider_kind",
        "endpoint",
        "osrm_image",
        "osrm_release",
        "osm_pbf_url",
        "osm_pbf_sha256",
        "routing_profile",
        "license_or_terms",
        "license_url",
        "reviewed_by",
        "reviewed_at",
    )
    missing = tuple(field for field in required if not str(record.get(field) or "").strip())
    if missing:
        errors.append(f"provider_provenance_fields_missing:{','.join(missing)}")
    values = " ".join(str(record.get(field) or "") for field in required).lower()
    if any(marker in values for marker in _PLACEHOLDER_MARKERS):
        errors.append("provider_provenance_contains_placeholder")
    image = str(record.get("osrm_image") or "")
    if not _IMAGE_DIGEST_PATTERN.fullmatch(image):
        errors.append("provider_image_not_digest_pinned")
    pbf_sha = str(record.get("osm_pbf_sha256") or "")
    if not _SHA256_PATTERN.fullmatch(pbf_sha):
        errors.append("provider_pbf_sha256_invalid")
    pbf_url = urlparse(str(record.get("osm_pbf_url") or ""))
    if pbf_url.scheme != "https" or not pbf_url.hostname or "latest" in pbf_url.path.lower():
        errors.append("provider_pbf_url_not_fixed_https")
    license_url = urlparse(str(record.get("license_url") or ""))
    if license_url.scheme != "https" or not license_url.hostname:
        errors.append("provider_license_url_invalid")
    try:
        reviewed_at = pd.to_datetime(record.get("reviewed_at"), utc=True, errors="raise")
        if pd.isna(reviewed_at):
            raise ValueError("empty review timestamp")
    except Exception:
        errors.append("provider_reviewed_at_invalid")
    endpoint = urlparse(str(record.get("endpoint") or ""))
    if (endpoint.hostname or "").lower() not in {"127.0.0.1", "localhost", "::1"}:
        errors.append("provider_endpoint_not_local")
    if str(record.get("provider_kind") or "").strip().lower() != "local_osrm":
        errors.append("provider_kind_not_local_osrm")
    return record, not any(error.startswith("provider_") for error in errors)


def _counts(frame: pd.DataFrame, column: str) -> dict[str, int]:
    if column not in frame.columns:
        return {}
    values = frame[column].fillna("").astype(str).str.strip()
    return {str(key): int(value) for key, value in values.value_counts().sort_index().items() if str(key)}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "validated", "passed"}


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)
