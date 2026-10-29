"""Strict loader for the direct coherent California product-demo package."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ..product_dashboard_models import ProductDashboardSourceBundle
from ..research_artifacts import stable_content_hash
from .product_demo_coherence import (
    ProductDemoCoherenceError,
    validate_product_demo_coherence,
)


class CoherentProductDemoError(ValueError):
    """Raised when the coherent package fails exact lineage validation."""


def load_coherent_product_demo_package(package_dir: Path) -> Any:
    """Load a content-addressed direct package into the existing product bundle."""

    from .product_demo import ProductDemoPackage

    manifest = _json(package_dir / "manifest.json")
    if (
        manifest.get("schema_version") != "product-demo-package-v2"
        or manifest.get("run_id") != "california_coast_coherent_demo_v1"
        or manifest.get("package_id") != "california_coast_coherent_demo_v1"
    ):
        _fail("product_demo_schema_invalid")
    _verify_inventory(package_dir, manifest.get("artifacts_sha256"))

    parent_record = manifest.get("parent")
    route_evidence = manifest.get("route_evidence")
    declared = manifest.get("alternatives")
    if (
        not isinstance(parent_record, dict)
        or not isinstance(route_evidence, dict)
        or not isinstance(declared, list)
        or [row.get("role") for row in declared if isinstance(row, dict)] != ["recommended", "low_driving"]
    ):
        _fail("product_demo_schema_invalid")

    parent_path = _contained(package_dir, str(parent_record.get("plan_relative_path") or ""))
    matrix_path = _contained(package_dir, str(route_evidence.get("relative_path") or ""))
    planner_path = _contained(package_dir, str(manifest.get("planner_runs_relative_path") or ""))
    parent = _json(parent_path)
    route_matrix = _json(matrix_path)
    if (
        _sha256(matrix_path) != route_evidence.get("matrix_file_sha256")
        or route_matrix.get("matrix_id") != route_evidence.get("matrix_id")
        or route_matrix.get("source_bundle_id") != route_evidence.get("source_bundle_id")
        or route_matrix.get("source_content_sha256") != route_evidence.get("source_content_sha256")
    ):
        _fail("product_demo_route_matrix_hash_mismatch")
    _validate_plan_hash(parent)
    if parent.get("plan_id") != parent_record.get("plan_id") or parent.get("content_hash") != parent_record.get(
        "content_hash"
    ):
        _fail("product_demo_parent_invalid")

    planner_runs = _json_lines(planner_path)
    planner_by_id = {str(row.get("run_id") or ""): row for row in planner_runs}
    alternatives_for_coherence: dict[str, tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = {}
    bundles: list[ProductDashboardSourceBundle] = []
    summaries: list[dict[str, Any]] = []
    evidence: dict[str, dict[str, Any]] = {}
    route_legs_by_plan: dict[str, tuple[dict[str, Any], ...]] = {}
    for row in declared:
        if not isinstance(row, dict):
            _fail("product_demo_alternatives_invalid")
        role = str(row.get("role") or "")
        child_path = _contained(package_dir, str(row.get("plan_relative_path") or ""))
        diff_path = _contained(package_dir, str(row.get("diff_relative_path") or ""))
        certificate_path = _contained(package_dir, str(row.get("certificate_relative_path") or ""))
        request_path = _contained(package_dir, str(row.get("request_relative_path") or ""))
        explanation_path = _contained(package_dir, str(row.get("explanation_relative_path") or ""))
        metrics_path = _contained(package_dir, str(row.get("metrics_relative_path") or ""))
        child = _json(child_path)
        diff = _json(diff_path)
        certificate = _json(certificate_path)
        request = _json(request_path)
        explanation = _json(explanation_path)
        metrics = _json(metrics_path)
        _validate_plan_hash(child)
        _validate_diff(diff)
        _validate_certificate(certificate)
        route_validation = certificate.get("route_validation") or {}
        planner = planner_by_id.get(str(child.get("source_run_id") or ""))
        if (
            child.get("plan_id") != row.get("plan_id")
            or child.get("content_hash") != row.get("plan_content_hash")
            or child.get("parent_plan_id") != parent.get("plan_id")
            or diff.get("parent_plan_id") != parent.get("plan_id")
            or diff.get("child_plan_id") != child.get("plan_id")
            or certificate.get("certificate_id") != row.get("certificate_id")
            or certificate.get("plan_id") != child.get("plan_id")
            or certificate.get("plan_content_hash") != child.get("content_hash")
            or certificate.get("comparison_eligibility") != "eligible"
            or certificate.get("evaluation_status") not in {"PASSED", "PASSED_WITH_WARNINGS"}
            or route_validation.get("publication_ready") is not True
            or route_validation.get("fallback_leg_count") != 0
            or route_validation.get("matrix_id") != route_matrix.get("matrix_id")
            or route_validation.get("source_bundle_id") != route_matrix.get("source_bundle_id")
            or route_validation.get("source_content_sha256") != route_matrix.get("source_content_sha256")
            or explanation.get("parent_plan_id") != parent.get("plan_id")
            or explanation.get("child_plan_id") != child.get("plan_id")
            or not planner
            or planner.get("execution_status") != "COMPLETED"
            or planner.get("result_plan_id") != child.get("plan_id")
        ):
            _fail("product_demo_alternative_lineage_invalid")
        route_legs_by_plan[str(child["plan_id"])] = _validate_route_legs(
            row.get("route_legs"),
            route_matrix,
            expected_count=int(route_validation.get("required_leg_count") or -1),
            expected_total_minutes=float(row.get("route_total_minutes") or -1.0),
        )
        source_paths = {
            "requests": (request_path.relative_to(package_dir).as_posix(),),
            "plans": (
                parent_path.relative_to(package_dir).as_posix(),
                child_path.relative_to(package_dir).as_posix(),
            ),
            "diffs": (diff_path.relative_to(package_dir).as_posix(),),
            "evaluations": (certificate_path.relative_to(package_dir).as_posix(),),
            "explanations": (explanation_path.relative_to(package_dir).as_posix(),),
            "routing": (matrix_path.relative_to(package_dir).as_posix(),),
            "planner_runs": (planner_path.relative_to(package_dir).as_posix(),),
            "metrics": (metrics_path.relative_to(package_dir).as_posix(),),
        }
        source_hashes = {
            relative: _sha256(_contained(package_dir, relative))
            for paths in source_paths.values()
            for relative in paths
        }
        bundle = ProductDashboardSourceBundle(
            run_dir=package_dir,
            run_manifest=manifest,
            request=request,
            parent_plan=parent,
            child_plan=child,
            diff=diff,
            certificate=certificate,
            explanations=(explanation,),
            route_matrix=route_matrix,
            planner_runs=(planner,),
            benchmark_planner_runs=(),
            metrics=metrics,
            benchmark_rows=(),
            source_paths=source_paths,
            source_hashes=source_hashes,
            truth_states=("eligible_repair",),
        )
        label = "Recommended repair" if role == "recommended" else "Low-driving repair"
        summary = {
            "method_id": child["plan_id"],
            "plan_id": child["plan_id"],
            "plan_content_hash": child["content_hash"],
            "method_label": label,
            "role": role,
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
        evidence[str(child["plan_id"])] = {
            **summary,
            "parent_plan_id": parent["plan_id"],
            "diff_id": diff.get("diff_id"),
            "certificate": certificate,
            "diff": diff,
            "methods": {
                "requested": [str(planner.get("method_requested") or "")],
                "executed": [str(planner.get("method_executed") or "")],
            },
            "source_hashes": source_hashes,
        }
        alternatives_for_coherence[role] = (child, diff, explanation)
        bundles.append(bundle)

    try:
        validate_product_demo_coherence(
            manifest=manifest,
            parent_plan=parent,
            alternatives=alternatives_for_coherence,
            route_matrix=route_matrix,
        )
    except ProductDemoCoherenceError as exc:
        _fail(str(exc))
    route_legs_by_plan[str(parent["plan_id"])] = _validate_route_legs(
        parent_record.get("route_legs"),
        route_matrix,
        expected_scope="road_validated_daily_route_leg",
    )
    if summaries[1]["route_total_minutes"] >= summaries[0]["route_total_minutes"]:
        _fail("product_demo_low_driving_claim_invalid")
    return ProductDemoPackage(
        primary_bundle=bundles[0],
        additional_plans=((bundles[1].child_plan, "Low-driving repair"),),
        alternatives=tuple(summaries),
        evidence=evidence,
        evidence_bundles={str(bundle.child_plan["plan_id"]): bundle for bundle in bundles},
        route_legs_by_plan=route_legs_by_plan,
    )


def _validate_plan_hash(plan: dict[str, Any]) -> None:
    payload = dict(plan)
    expected = str(payload.pop("content_hash", ""))
    if not expected or stable_content_hash(payload) != expected:
        _fail("product_demo_plan_hash_invalid")


def _validate_diff(diff: dict[str, Any]) -> None:
    groups = (
        ("added_stops", "added"),
        ("deleted_stops", "deleted"),
        ("day_moves", "day_moves"),
        ("time_shifts", "time_shifts"),
        ("reorder_changes", "reorder"),
        ("lodging_changes", "lodging"),
        ("road_changes", "road"),
    )
    changes: dict[str, list[dict[str, Any]]] = {}
    for record_key, _ in groups:
        records = diff.get(record_key)
        if not isinstance(records, list) or any(
            not isinstance(change, dict) or not _finite(change.get("cost")) for change in records
        ):
            _fail("product_demo_diff_cost_invalid")
        changes[record_key] = records
    weighted_cost = diff.get("weighted_edit_cost")
    if (
        not _finite(weighted_cost)
        or abs(sum(float(change["cost"]) for records in changes.values() for change in records) - float(weighted_cost))
        > 1e-9
    ):
        _fail("product_demo_diff_cost_invalid")
    seed = {
        "parent_plan_id": diff.get("parent_plan_id"),
        "child_plan_id": diff.get("child_plan_id"),
        **{seed_key: diff.get(record_key) or [] for record_key, seed_key in groups},
        "unchanged_days": diff.get("unchanged_days") or [],
        "weighted_edit_cost": float(weighted_cost),
    }
    if diff.get("diff_id") != f"diff_{stable_content_hash(seed)}":
        _fail("product_demo_diff_id_invalid")


def _validate_certificate(certificate: dict[str, Any]) -> None:
    payload = dict(certificate)
    content_hash = str(payload.pop("content_hash", ""))
    if not content_hash or stable_content_hash(payload) != content_hash:
        _fail("product_demo_certificate_hash_invalid")
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
        _fail("product_demo_certificate_id_invalid")


def _validate_route_legs(
    declared: Any,
    route_matrix: dict[str, Any],
    *,
    expected_count: int | None = None,
    expected_total_minutes: float | None = None,
    expected_scope: str = "certified_daily_route_leg",
) -> tuple[dict[str, Any], ...]:
    if not isinstance(declared, list) or not declared:
        _fail("product_demo_route_legs_invalid")
    if expected_count is not None and len(declared) != expected_count:
        _fail("product_demo_route_leg_count_mismatch")
    cells = {
        (str(cell.get("origin_id") or ""), str(cell.get("destination_id") or "")): cell
        for cell in route_matrix.get("cells") or []
        if isinstance(cell, dict)
    }
    total_seconds = 0.0
    previous_destination: str | None = None
    result: list[dict[str, Any]] = []
    for row in declared:
        if (
            not isinstance(row, dict)
            or set(row) != {"day", "origin_id", "destination_id", "evidence_scope"}
            or not isinstance(row.get("day"), int)
            or row.get("evidence_scope") != expected_scope
        ):
            _fail("product_demo_route_legs_invalid")
        origin = str(row.get("origin_id") or "")
        destination = str(row.get("destination_id") or "")
        if previous_destination is not None and previous_destination != origin:
            _fail("product_demo_route_discontinuous")
        cell = cells.get((origin, destination))
        if (
            not cell
            or cell.get("road_validated") is not True
            or cell.get("fallback_used") is not False
            or not _finite(cell.get("duration_s"))
        ):
            _fail("product_demo_route_leg_evidence_invalid")
        total_seconds += float(cell["duration_s"])
        result.append(dict(row))
        previous_destination = destination
    if expected_total_minutes is not None and abs(total_seconds / 60.0 - expected_total_minutes) > 1e-6:
        _fail("product_demo_route_total_mismatch")
    return tuple(result)


def _verify_inventory(package_dir: Path, declared: Any) -> None:
    if not isinstance(declared, dict) or not declared:
        _fail("product_demo_inventory_missing")
    actual = {
        path.relative_to(package_dir).as_posix()
        for path in package_dir.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    if set(declared) != actual:
        _fail("product_demo_inventory_mismatch")
    for relative, expected in declared.items():
        if not isinstance(expected, str) or len(expected) != 64:
            _fail("product_demo_inventory_invalid")
        if _sha256(_contained(package_dir, relative)) != expected:
            _fail("product_demo_artifact_hash_mismatch")


def _contained(root: Path, relative_text: str) -> Path:
    relative = Path(relative_text)
    if not relative_text or relative.is_absolute() or ".." in relative.parts:
        _fail("product_demo_path_unsafe")
    root = root.resolve()
    candidate = (root / relative).resolve()
    if root != candidate and root not in candidate.parents:
        _fail("product_demo_path_unsafe")
    return candidate


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CoherentProductDemoError("product_demo_json_invalid") from exc
    if not isinstance(value, dict):
        _fail("product_demo_json_invalid")
    return value


def _json_lines(path: Path) -> tuple[dict[str, Any], ...]:
    records: list[dict[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError
            records.append(record)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise CoherentProductDemoError("product_demo_planner_record_invalid") from exc
    if not records:
        _fail("product_demo_planner_record_invalid")
    return tuple(records)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and value == value
        and abs(float(value)) != float("inf")
    )


def _fail(code: str) -> None:
    raise CoherentProductDemoError(code)
