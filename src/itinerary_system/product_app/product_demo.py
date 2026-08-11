"""Validated adapter for the deterministic parent-plus-two-child product demo."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any

from ..product_dashboard_models import ProductDashboardSourceBundle
from ..research_artifacts import stable_content_hash


class ProductDemoError(ValueError):
    """Raised when the W2 package does not satisfy its declared lineage."""


@dataclass(frozen=True)
class ProductDemoPackage:
    primary_bundle: Any
    additional_plans: tuple[tuple[dict[str, Any], str], ...]
    alternatives: tuple[dict[str, Any], ...]
    evidence: dict[str, dict[str, Any]]
    evidence_bundles: dict[str, Any]
    route_legs_by_plan: dict[str, tuple[dict[str, Any], ...]]


def load_product_demo_package(repository_root: Path, package_dir: Path) -> ProductDemoPackage:
    manifest = _json(package_dir / "manifest.json")
    if (
        manifest.get("schema_version") != "product-demo-package-v1"
        or manifest.get("run_id") != "california_coast_product_demo_v2"
        or manifest.get("package_id") != "california_coast_product_demo_v2"
    ):
        raise ProductDemoError("product_demo_schema_invalid")
    _verify_package_inventory(package_dir, manifest.get("artifacts_sha256"))
    _validate_redacted_config_paths(package_dir, manifest.get("artifacts_sha256"))

    source = manifest.get("source_run")
    if not isinstance(source, dict):
        raise ProductDemoError("product_demo_source_invalid")
    source_dir = _contained(repository_root, str(source.get("relative_path") or ""))
    source_manifest = source_dir / "manifest.json"
    expected_source_hash = str(source.get("manifest_sha256") or "")
    if not source_manifest.is_file() or _sha256(source_manifest) != expected_source_hash:
        raise ProductDemoError("product_demo_source_hash_mismatch")

    declared = manifest.get("alternatives")
    if not isinstance(declared, list) or [row.get("role") for row in declared if isinstance(row, dict)] != [
        "recommended",
        "low_driving",
    ]:
        raise ProductDemoError("product_demo_alternatives_invalid")

    bundles: list[Any] = []
    summaries: list[dict[str, Any]] = []
    evidence: dict[str, dict[str, Any]] = {}
    route_legs_by_plan: dict[str, tuple[dict[str, Any], ...]] = {}
    parent = manifest.get("parent") if isinstance(manifest.get("parent"), dict) else {}
    parent_plan_id = str(parent.get("plan_id") or "")
    parent_content_hash = str(parent.get("content_hash") or "")
    route_evidence = manifest.get("route_evidence")
    if not isinstance(route_evidence, dict):
        raise ProductDemoError("product_demo_route_evidence_invalid")
    external_matrix = _contained(repository_root, str(route_evidence.get("relative_path") or ""))
    matrix_file_sha256 = str(route_evidence.get("matrix_file_sha256") or "")
    if len(matrix_file_sha256) != 64 or _sha256(external_matrix) != matrix_file_sha256:
        raise ProductDemoError("product_demo_route_matrix_hash_mismatch")
    for row in declared:
        run_dir = _contained(package_dir, str(row.get("run_relative_path") or ""))
        bundle = _load_demo_alternative(run_dir)
        child = bundle.child_plan
        certificate = bundle.certificate
        diff = bundle.diff
        if not child or not certificate or not diff:
            raise ProductDemoError("product_demo_alternative_artifacts_missing")
        plan_id = str(child.get("plan_id") or "")
        content_hash = str(child.get("content_hash") or "")
        route_validation = certificate.get("route_validation") or {}
        route_matrix = bundle.route_matrix or {}
        routing_paths = bundle.source_paths.get("routing", ())
        if (
            plan_id != row.get("plan_id")
            or content_hash != row.get("plan_content_hash")
            or child.get("parent_plan_id") != parent_plan_id
            or bundle.parent_plan.get("content_hash") != parent_content_hash
            or certificate.get("certificate_id") != row.get("certificate_id")
            or certificate.get("plan_id") != plan_id
            or certificate.get("plan_content_hash") != content_hash
            or certificate.get("comparison_eligibility") != "eligible"
            or certificate.get("evaluation_status") not in {"PASSED", "PASSED_WITH_WARNINGS"}
            or route_validation.get("publication_ready") is not True
            or route_validation.get("fallback_leg_count") != 0
            or route_matrix.get("matrix_id") != route_evidence.get("matrix_id")
            or route_matrix.get("source_bundle_id") != route_evidence.get("source_bundle_id")
            or route_matrix.get("source_content_sha256")
            != route_evidence.get("source_content_sha256")
            or route_validation.get("matrix_id") != route_evidence.get("matrix_id")
            or route_validation.get("source_bundle_id") != route_evidence.get("source_bundle_id")
            or route_validation.get("source_content_sha256")
            != route_evidence.get("source_content_sha256")
            or len(routing_paths) != 1
            or bundle.source_hashes.get(routing_paths[0]) != matrix_file_sha256
        ):
            raise ProductDemoError("product_demo_alternative_lineage_invalid")
        _validate_diff(diff)
        _validate_certificate(certificate)
        route_legs_by_plan[plan_id] = _validate_route_legs(
            row.get("route_legs"),
            route_matrix,
            expected_count=int(route_validation.get("required_leg_count") or -1),
            expected_total_minutes=float(row.get("route_total_minutes") or -1.0),
        )
        label = "Recommended repair" if row["role"] == "recommended" else "Low-driving repair"
        summary = {
            "method_id": plan_id,
            "plan_id": plan_id,
            "plan_content_hash": content_hash,
            "method_label": label,
            "role": row["role"],
            "status": "eligible",
            "display_status": "Eligible",
            "ranking_eligible": True,
            "failure_reason": None,
            "weighted_edit_cost": diff.get("weighted_edit_cost"),
            "route_total_minutes": row.get("route_total_minutes"),
            "route_validated": True,
            "certificate_id": certificate.get("certificate_id"),
            "evaluation_status": certificate.get("evaluation_status"),
        }
        summaries.append(summary)
        evidence[plan_id] = {
            **summary,
            "parent_plan_id": parent_plan_id,
            "diff_id": diff.get("diff_id"),
            "certificate": certificate,
            "diff": diff,
            "methods": {
                "requested": sorted(
                    {
                        str(record.get("method_requested"))
                        for record in bundle.planner_runs
                        if record.get("method_requested")
                    }
                ),
                "executed": sorted(
                    {
                        str(record.get("method_executed"))
                        for record in bundle.planner_runs
                        if record.get("method_executed")
                    }
                ),
            },
            "source_hashes": dict(bundle.source_hashes),
        }
        bundles.append(bundle)

    if len({row["plan_id"] for row in summaries}) != 2 or len(
        {row["plan_content_hash"] for row in summaries}
    ) != 2:
        raise ProductDemoError("product_demo_alternatives_not_distinct")
    if summaries[1]["route_total_minutes"] >= summaries[0]["route_total_minutes"]:
        raise ProductDemoError("product_demo_low_driving_claim_invalid")
    route_legs_by_plan[parent_plan_id] = _validate_route_legs(
        parent.get("route_legs"),
        bundles[0].route_matrix or {},
        expected_scope="road_validated_daily_route_leg",
    )
    return ProductDemoPackage(
        primary_bundle=bundles[0],
        additional_plans=((bundles[1].child_plan, "Low-driving repair"),),
        alternatives=tuple(summaries),
        evidence=evidence,
        evidence_bundles={str(bundle.child_plan["plan_id"]): bundle for bundle in bundles},
        route_legs_by_plan=route_legs_by_plan,
    )


def _contained(root: Path, relative_text: str) -> Path:
    relative = Path(relative_text)
    if not relative_text or relative.is_absolute() or ".." in relative.parts:
        raise ProductDemoError("product_demo_path_unsafe")
    root = root.resolve()
    candidate = (root / relative).resolve()
    if root != candidate and root not in candidate.parents:
        raise ProductDemoError("product_demo_path_unsafe")
    return candidate


def _load_demo_alternative(run_dir: Path) -> ProductDashboardSourceBundle:
    """Load a strict pipeline run while permitting explanation schemas without diff IDs."""

    manifest = _json(run_dir / "manifest.json")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ProductDemoError("product_demo_run_manifest_invalid")

    def records(category: str) -> list[dict[str, Any]]:
        paths = artifacts.get(category) or []
        if not isinstance(paths, list):
            raise ProductDemoError("product_demo_run_manifest_invalid")
        return [_json(_contained(run_dir, str(relative))) for relative in paths]

    requests = records("requests")
    plans = records("plans")
    diffs = records("diffs")
    evaluations = records("evaluations")
    explanations = records("explanations")
    routing = records("routing")
    metrics = records("metrics")
    if not all(len(group) == 1 for group in (requests, diffs, evaluations, routing, metrics)):
        raise ProductDemoError("product_demo_run_artifacts_invalid")
    parent_plan_id = str(manifest.get("parent_plan_id") or "")
    output_ids = {
        str(value)
        for value in (manifest.get("pipeline_run") or {}).get("output_plan_ids", ())
        if str(value)
    }
    parent = next((plan for plan in plans if plan.get("plan_id") == parent_plan_id), None)
    child = next((plan for plan in plans if plan.get("plan_id") in output_ids), None)
    if not parent or not child:
        raise ProductDemoError("product_demo_plan_lineage_invalid")
    for plan in (parent, child):
        payload = dict(plan)
        expected = str(payload.pop("content_hash", ""))
        if not expected or stable_content_hash(payload) != expected:
            raise ProductDemoError("product_demo_plan_hash_invalid")
    diff = diffs[0]
    certificate = evaluations[0]
    if (
        child.get("parent_plan_id") != parent_plan_id
        or diff.get("parent_plan_id") != parent_plan_id
        or diff.get("child_plan_id") != child.get("plan_id")
        or certificate.get("plan_id") != child.get("plan_id")
        or certificate.get("plan_content_hash") != child.get("content_hash")
        or any(
            explanation.get("parent_plan_id") != parent_plan_id
            or explanation.get("child_plan_id") != child.get("plan_id")
            for explanation in explanations
        )
    ):
        raise ProductDemoError("product_demo_run_lineage_invalid")

    source_paths = {
        category: tuple(str(value) for value in artifacts.get(category) or [])
        for category in (
            "requests",
            "plans",
            "diffs",
            "evaluations",
            "explanations",
            "routing",
            "planner_runs",
            "metrics",
        )
    }
    source_hashes = {
        relative: _sha256(_contained(run_dir, relative))
        for paths in source_paths.values()
        for relative in paths
    }
    planner_runs: list[dict[str, Any]] = []
    for relative in source_paths["planner_runs"]:
        for line in _contained(run_dir, relative).read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ProductDemoError("product_demo_planner_record_invalid")
                planner_runs.append(record)
    return ProductDashboardSourceBundle(
        run_dir=run_dir,
        run_manifest=manifest,
        request=requests[0],
        parent_plan=parent,
        child_plan=child,
        diff=diff,
        certificate=certificate,
        explanations=tuple(explanations),
        route_matrix=routing[0],
        planner_runs=tuple(planner_runs),
        benchmark_planner_runs=(),
        metrics=metrics[0],
        benchmark_rows=(),
        source_paths=source_paths,
        source_hashes=source_hashes,
        truth_states=("interaction_mode_disabled", "eligible_repair"),
    )


def _verify_package_inventory(package_dir: Path, declared: Any) -> None:
    if not isinstance(declared, dict) or not declared:
        raise ProductDemoError("product_demo_inventory_missing")
    if not all(
        isinstance(relative, str)
        and isinstance(expected, str)
        and len(expected) == 64
        for relative, expected in declared.items()
    ):
        raise ProductDemoError("product_demo_inventory_invalid")
    actual_paths = {
        path.relative_to(package_dir).as_posix()
        for path in package_dir.rglob("*")
        if path.is_file() and path != package_dir / "manifest.json"
    }
    if set(declared) != actual_paths:
        raise ProductDemoError("product_demo_inventory_mismatch")
    for relative, expected in declared.items():
        if _sha256(_contained(package_dir, relative)) != expected:
            raise ProductDemoError("product_demo_artifact_hash_mismatch")


def _validate_redacted_config_paths(package_dir: Path, declared: Any) -> None:
    for relative in declared:
        if not str(relative).endswith("resolved_config.redacted.json"):
            continue
        config = _json(_contained(package_dir, str(relative)))
        source_path = str(config.get("_source_path") or "")
        path = Path(source_path)
        if (
            not source_path
            or path.is_absolute()
            or PureWindowsPath(source_path).drive
            or ".." in path.parts
        ):
            raise ProductDemoError("product_demo_host_path_disclosure")


def _validate_diff(diff: dict[str, Any]) -> None:
    change_groups = (
        ("added_stops", "added"),
        ("deleted_stops", "deleted"),
        ("day_moves", "day_moves"),
        ("time_shifts", "time_shifts"),
        ("reorder_changes", "reorder"),
        ("lodging_changes", "lodging"),
        ("road_changes", "road"),
    )
    changes: dict[str, list[dict[str, Any]]] = {}
    for record_key, _ in change_groups:
        records = diff.get(record_key)
        if not isinstance(records, list) or any(not isinstance(change, dict) for change in records):
            raise ProductDemoError("product_demo_diff_cost_invalid")
        if any(not _finite_number(change.get("cost")) for change in records):
            raise ProductDemoError("product_demo_diff_cost_invalid")
        changes[record_key] = records
    declared_weighted_cost = diff.get("weighted_edit_cost")
    if not _finite_number(declared_weighted_cost):
        raise ProductDemoError("product_demo_diff_cost_invalid")
    weighted_cost = sum(float(change["cost"]) for records in changes.values() for change in records)
    if abs(weighted_cost - float(declared_weighted_cost)) > 1e-9:
        raise ProductDemoError("product_demo_diff_cost_invalid")
    seed = {
        "parent_plan_id": diff.get("parent_plan_id"),
        "child_plan_id": diff.get("child_plan_id"),
        **{seed_key: diff.get(record_key) or [] for record_key, seed_key in change_groups},
        "unchanged_days": diff.get("unchanged_days") or [],
        "weighted_edit_cost": float(declared_weighted_cost),
    }
    if diff.get("diff_id") != f"diff_{stable_content_hash(seed)}":
        raise ProductDemoError("product_demo_diff_id_invalid")


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _validate_certificate(certificate: dict[str, Any]) -> None:
    payload = dict(certificate)
    content_hash = str(payload.pop("content_hash", ""))
    if not content_hash or stable_content_hash(payload) != content_hash:
        raise ProductDemoError("product_demo_certificate_hash_invalid")
    failures = certificate.get("failures") or []
    warnings = certificate.get("warnings") or []
    seed = {
        "plan_id": certificate.get("plan_id"),
        "plan_content_hash": certificate.get("plan_content_hash"),
        "source_run_id": certificate.get("source_run_id"),
        "failures": [item.get("code") for item in failures if isinstance(item, dict)],
        "warnings": [item.get("code") for item in warnings if isinstance(item, dict)],
    }
    if certificate.get("certificate_id") != f"cert_{stable_content_hash(seed)}":
        raise ProductDemoError("product_demo_certificate_id_invalid")


def _validate_route_legs(
    declared: Any,
    route_matrix: dict[str, Any],
    *,
    expected_count: int | None = None,
    expected_total_minutes: float | None = None,
    expected_scope: str = "certified_daily_route_leg",
) -> tuple[dict[str, Any], ...]:
    if not isinstance(declared, list) or not declared:
        raise ProductDemoError("product_demo_route_legs_invalid")
    if expected_count is not None and len(declared) != expected_count:
        raise ProductDemoError("product_demo_route_leg_count_mismatch")
    cells = {
        (str(cell.get("origin_id") or ""), str(cell.get("destination_id") or "")): cell
        for cell in route_matrix.get("cells") or []
        if isinstance(cell, dict)
    }
    total_seconds = 0.0
    result: list[dict[str, Any]] = []
    previous_destination: str | None = None
    for row in declared:
        if (
            not isinstance(row, dict)
            or set(row) != {"day", "origin_id", "destination_id", "evidence_scope"}
            or not isinstance(row.get("day"), int)
            or row.get("evidence_scope") != expected_scope
        ):
            raise ProductDemoError("product_demo_route_legs_invalid")
        origin = str(row.get("origin_id") or "")
        destination = str(row.get("destination_id") or "")
        if previous_destination is not None and previous_destination != origin:
            raise ProductDemoError("product_demo_route_discontinuous")
        cell = cells.get((origin, destination))
        if (
            not origin
            or not destination
            or not cell
            or cell.get("road_validated") is not True
            or cell.get("fallback_used") is not False
        ):
            raise ProductDemoError("product_demo_route_leg_evidence_invalid")
        total_seconds += float(cell.get("duration_s") or 0.0)
        result.append(dict(row))
        previous_destination = destination
    if expected_total_minutes is not None and abs(total_seconds / 60.0 - expected_total_minutes) > 1e-6:
        raise ProductDemoError("product_demo_route_total_mismatch")
    return tuple(result)


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProductDemoError("product_demo_json_invalid") from exc
    if not isinstance(value, dict):
        raise ProductDemoError("product_demo_json_invalid")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
