"""Contracts for the separately versioned product dashboard artifact."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PRODUCT_DASHBOARD_SCHEMA = "product-dashboard-manifest-v1"
PRODUCT_DASHBOARD_DATA_SCHEMA = "product-dashboard-data-v1"
PRODUCT_DASHBOARD_VERSION = "1.0.0"
PRODUCT_COMPATIBILITY_BOUNDARY = (
    "Read-only presentation of canonical run artifacts. It does not replace or "
    "mutate the optimizer, evaluator, benchmark, route model, plan repository, "
    "legacy dashboard, Folium renderer, or interaction authority."
)

TRUTH_STATE_LABELS = {
    "eligible_repair": ("Eligible repair", "success"),
    "ineligible_repair": ("Ineligible repair", "danger"),
    "failed_method": ("Method failed", "danger"),
    "exact_search_incomplete": ("Exact search incomplete", "warning"),
    "complete_infeasibility": ("Complete infeasibility", "danger"),
    "missing_route_evidence": ("Route evidence missing", "warning"),
    "fallback_route": ("Fallback route", "warning"),
    "stale_artifact": ("Stale artifact", "warning"),
    "missing_certificate": ("Certificate missing", "warning"),
    "certificate_mismatch": ("Certificate mismatch", "danger"),
    "no_child_plan": ("No child plan", "neutral"),
    "no_material_change": ("No material change", "neutral"),
    "unchanged_parent": ("Original plan retained", "neutral"),
    "null_metric": ("Metric is null", "neutral"),
    "unavailable_metric": ("Metric unavailable", "neutral"),
    "permission_required": ("Permission required", "warning"),
    "locked_change_blocked": ("Locked change blocked", "danger"),
    "hypothetical_probe": ("Hypothetical probe", "info"),
    "interaction_mode_disabled": ("Interaction mode disabled", "info"),
    "empty_data": ("No data", "neutral"),
    "loading": ("Loading artifact", "info"),
    "malformed_artifact": ("Malformed artifact", "danger"),
    "partial_run": ("Partial run", "warning"),
}


@dataclass(frozen=True)
class ProductDashboardSourceBundle:
    """Validated canonical inputs used to build one read-only product view."""

    run_dir: Path
    run_manifest: dict[str, Any]
    request: dict[str, Any]
    parent_plan: dict[str, Any]
    child_plan: dict[str, Any] | None
    diff: dict[str, Any] | None
    certificate: dict[str, Any] | None
    explanations: tuple[dict[str, Any], ...]
    route_matrix: dict[str, Any] | None
    planner_runs: tuple[dict[str, Any], ...]
    benchmark_planner_runs: tuple[dict[str, Any], ...]
    metrics: dict[str, Any]
    benchmark_rows: tuple[dict[str, Any], ...] = ()
    source_paths: dict[str, tuple[str, ...]] = field(default_factory=dict)
    source_hashes: dict[str, str] = field(default_factory=dict)
    truth_states: tuple[str, ...] = ()


@dataclass(frozen=True)
class ProductDashboardManifest:
    """Versioned manifest for generated product dashboard assets."""

    run_id: str
    source_artifact_refs: dict[str, tuple[str, ...]]
    source_content_hashes: dict[str, str]
    parent_plan_id: str
    parent_content_hash: str
    child_plan_id: str | None
    child_content_hash: str | None
    diff_id: str | None
    certificate_id: str | None
    explanation_ids: tuple[str, ...]
    route_matrix_id: str | None
    requested_method_ids: tuple[str, ...]
    executed_method_ids: tuple[str, ...]
    run_status: str
    truth_states: tuple[str, ...]
    asset_hashes: dict[str, str]
    generated_at: str
    schema_version: str = PRODUCT_DASHBOARD_SCHEMA
    product_version: str = PRODUCT_DASHBOARD_VERSION
    run_manifest_path: str = "../manifest.json"
    compatibility_boundary: str = PRODUCT_COMPATIBILITY_BOUNDARY

    def to_record(self) -> dict[str, Any]:
        """Return a JSON-compatible manifest record."""

        return {
            "schema_version": self.schema_version,
            "product_version": self.product_version,
            "run_id": self.run_id,
            "run_manifest_path": self.run_manifest_path,
            "source_artifact_refs": {
                key: list(paths) for key, paths in sorted(self.source_artifact_refs.items())
            },
            "source_content_hashes": dict(sorted(self.source_content_hashes.items())),
            "parent_plan_id": self.parent_plan_id,
            "parent_content_hash": self.parent_content_hash,
            "child_plan_id": self.child_plan_id,
            "child_content_hash": self.child_content_hash,
            "diff_id": self.diff_id,
            "certificate_id": self.certificate_id,
            "explanation_ids": list(self.explanation_ids),
            "route_matrix_id": self.route_matrix_id,
            "requested_method_ids": list(self.requested_method_ids),
            "executed_method_ids": list(self.executed_method_ids),
            "run_status": self.run_status,
            "truth_states": list(self.truth_states),
            "asset_hashes": dict(sorted(self.asset_hashes.items())),
            "generated_at": self.generated_at,
            "compatibility_boundary": self.compatibility_boundary,
        }


class ProductDashboardValidationError(ValueError):
    """Raised when canonical artifacts cannot support a truthful product view."""

    def __init__(self, issues: list[str] | tuple[str, ...]):
        self.issues = tuple(str(issue) for issue in issues)
        super().__init__("; ".join(self.issues))


def file_sha256(path: Path) -> str:
    """Return the complete SHA-256 for one file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_text(payload: Any, *, indent: int | None = 2) -> str:
    """Serialize finite JSON with stable keys."""

    assert_finite_json(payload)
    return json.dumps(payload, indent=indent, sort_keys=True, ensure_ascii=False, allow_nan=False)


def assert_finite_json(value: Any, *, path: str = "$") -> None:
    """Reject non-finite numbers anywhere in a dashboard payload."""

    if isinstance(value, float) and not math.isfinite(value):
        raise ProductDashboardValidationError([f"nonfinite_value:{path}"])
    if isinstance(value, dict):
        for key, item in value.items():
            assert_finite_json(item, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            assert_finite_json(item, path=f"{path}[{index}]")


def safe_run_relative_path(run_dir: Path, path_text: str) -> Path:
    """Resolve a manifest path while rejecting absolute and escaping paths."""

    candidate = Path(str(path_text))
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ProductDashboardValidationError([f"unsafe_artifact_path:{path_text}"])
    resolved_run = run_dir.resolve()
    resolved = (resolved_run / candidate).resolve()
    try:
        resolved.relative_to(resolved_run)
    except ValueError as exc:
        raise ProductDashboardValidationError([f"artifact_path_escapes_run:{path_text}"]) from exc
    return resolved
