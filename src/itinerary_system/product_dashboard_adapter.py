"""Load and validate canonical run artifacts for the product dashboard."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .product_dashboard_models import (
    ProductDashboardSourceBundle,
    ProductDashboardValidationError,
    assert_finite_json,
    file_sha256,
    safe_run_relative_path,
)
from .research_artifacts import stable_content_hash


def load_product_dashboard_source(run_dir: Path) -> ProductDashboardSourceBundle:
    """Load one immutable run without recomputing planner or evaluator truth."""

    run_dir = Path(run_dir)
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        raise ProductDashboardValidationError(["missing_run_manifest"])
    manifest = _load_json(manifest_path)
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ProductDashboardValidationError(["malformed_run_manifest:artifacts"])

    source_paths = {
        category: tuple(str(path) for path in paths)
        for category, paths in artifacts.items()
        if isinstance(paths, list)
        and category
        in {
            "requests",
            "plans",
            "diffs",
            "evaluations",
            "explanations",
            "routing",
            "planner_runs",
            "metrics",
            "benchmark_metrics",
            "benchmark_planner_runs",
        }
    }
    resolved = _resolve_declared_paths(run_dir, source_paths)

    requests = _load_json_group(resolved.get("requests", ()))
    plans = _load_json_group(resolved.get("plans", ()))
    diffs = _load_json_group(resolved.get("diffs", ()))
    certificates = _load_json_group(resolved.get("evaluations", ()))
    explanations = _load_json_group(resolved.get("explanations", ()))
    route_matrices = _load_json_group(resolved.get("routing", ()))
    metrics_group = _load_json_group(resolved.get("metrics", ()))
    planner_runs = _load_jsonl_group(resolved.get("planner_runs", ()))
    benchmark_rows = _load_jsonl_group(resolved.get("benchmark_metrics", ()))
    benchmark_planner_runs = _load_jsonl_group(
        resolved.get("benchmark_planner_runs", ())
    )

    issues: list[str] = []
    if len(requests) != 1:
        issues.append(f"expected_one_request:found_{len(requests)}")
    parent_plan_id = str(manifest.get("parent_plan_id") or "")
    parent = next((plan for plan in plans if str(plan.get("plan_id")) == parent_plan_id), None)
    if parent is None:
        issues.append("missing_parent_plan")
    child_ids = tuple(
        str(plan_id)
        for plan_id in (manifest.get("pipeline_run") or {}).get("output_plan_ids", ())
        if str(plan_id)
    )
    child = next((plan for plan in plans if str(plan.get("plan_id")) in child_ids), None)
    if child is None:
        child = next((plan for plan in plans if plan.get("parent_plan_id") == parent_plan_id), None)

    if parent is not None:
        _validate_plan_hash(parent, issues, label="parent")
    if child is not None:
        _validate_plan_hash(child, issues, label="child")
        if child.get("parent_plan_id") != parent_plan_id:
            issues.append("child_parent_lineage_mismatch")

    diff = diffs[0] if diffs else None
    certificate = certificates[0] if certificates else None
    route_matrix = route_matrices[0] if route_matrices else None
    if child is not None and diff is None:
        issues.append("missing_plan_diff")
    if diff is not None and child is not None:
        if diff.get("parent_plan_id") != parent_plan_id:
            issues.append("diff_parent_lineage_mismatch")
        if diff.get("child_plan_id") != child.get("plan_id"):
            issues.append("diff_child_lineage_mismatch")
    if explanations and child is not None:
        for explanation in explanations:
            if explanation.get("parent_plan_id") != parent_plan_id:
                issues.append("explanation_parent_lineage_mismatch")
            if explanation.get("child_plan_id") != child.get("plan_id"):
                issues.append("explanation_child_lineage_mismatch")
            if diff is not None and explanation.get("diff_id") != diff.get("diff_id"):
                issues.append("explanation_diff_lineage_mismatch")

    assert_finite_json(
        {
            "request": requests[0] if requests else {},
            "parent": parent or {},
            "child": child,
            "diff": diff,
            "certificate": certificate,
            "explanations": explanations,
            "planner_runs": planner_runs,
            "benchmark_rows": benchmark_rows,
            "benchmark_planner_runs": benchmark_planner_runs,
            "route_matrix": route_matrix,
            "metrics": metrics_group,
        }
    )
    if issues:
        raise ProductDashboardValidationError(issues)

    truth_states = _derive_truth_states(
        manifest=manifest,
        child=child,
        diff=diff,
        certificate=certificate,
        route_matrix=route_matrix,
        planner_runs=planner_runs,
        benchmark_rows=benchmark_rows,
        benchmark_planner_runs=benchmark_planner_runs,
    )
    source_hashes = {
        path_text: file_sha256(path)
        for category in sorted(resolved)
        for path_text, path in zip(source_paths.get(category, ()), resolved[category], strict=True)
    }
    return ProductDashboardSourceBundle(
        run_dir=run_dir,
        run_manifest=manifest,
        request=requests[0],
        parent_plan=parent,
        child_plan=child,
        diff=diff,
        certificate=certificate,
        explanations=tuple(explanations),
        route_matrix=route_matrix,
        planner_runs=tuple(planner_runs),
        benchmark_planner_runs=tuple(benchmark_planner_runs),
        metrics=metrics_group[0] if metrics_group else {},
        benchmark_rows=tuple(benchmark_rows),
        source_paths=source_paths,
        source_hashes=source_hashes,
        truth_states=truth_states,
    )


def _resolve_declared_paths(
    run_dir: Path, source_paths: dict[str, tuple[str, ...]]
) -> dict[str, tuple[Path, ...]]:
    issues: list[str] = []
    resolved: dict[str, tuple[Path, ...]] = {}
    for category, paths in source_paths.items():
        resolved_paths: list[Path] = []
        for path_text in paths:
            try:
                path = safe_run_relative_path(run_dir, path_text)
            except ProductDashboardValidationError as exc:
                issues.extend(exc.issues)
                continue
            if not path.is_file():
                issues.append(f"missing_declared_artifact:{path_text}")
            else:
                resolved_paths.append(path)
        resolved[category] = tuple(resolved_paths)
    if issues:
        raise ProductDashboardValidationError(issues)
    return resolved


def _load_json_group(paths: tuple[Path, ...]) -> list[dict[str, Any]]:
    return [_load_json(path) for path in paths]


def _load_jsonl_group(paths: tuple[Path, ...]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ProductDashboardValidationError(
                    [f"malformed_jsonl:{path.name}:{line_number}:{exc.msg}"]
                ) from exc
            if not isinstance(record, dict):
                raise ProductDashboardValidationError(
                    [f"jsonl_record_not_object:{path.name}:{line_number}"]
                )
            records.append(record)
    return records


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProductDashboardValidationError([f"malformed_json:{path.name}:{exc}"]) from exc
    if not isinstance(payload, dict):
        raise ProductDashboardValidationError([f"json_root_not_object:{path.name}"])
    return payload


def _validate_plan_hash(plan: dict[str, Any], issues: list[str], *, label: str) -> None:
    expected = str(plan.get("content_hash") or "")
    payload = dict(plan)
    payload.pop("content_hash", None)
    actual = stable_content_hash(payload)
    if not expected:
        issues.append(f"{label}_content_hash_missing")
    elif actual != expected:
        issues.append(f"{label}_content_hash_mismatch")


def _derive_truth_states(
    *,
    manifest: dict[str, Any],
    child: dict[str, Any] | None,
    diff: dict[str, Any] | None,
    certificate: dict[str, Any] | None,
    route_matrix: dict[str, Any] | None,
    planner_runs: list[dict[str, Any]],
    benchmark_rows: list[dict[str, Any]],
    benchmark_planner_runs: list[dict[str, Any]],
) -> tuple[str, ...]:
    states: list[str] = ["interaction_mode_disabled"]
    if str(manifest.get("status") or "") != "completed":
        states.append("partial_run")
    if child is None:
        states.extend(("no_child_plan", "unchanged_parent"))
    if certificate is None:
        states.append("missing_certificate")
    elif child is not None and (
        certificate.get("plan_id") != child.get("plan_id")
        or certificate.get("plan_content_hash") != child.get("content_hash")
        or (diff is not None and certificate.get("diff_id") != diff.get("diff_id"))
    ):
        states.append("certificate_mismatch")
    elif bool(certificate.get("eligible")):
        states.append("eligible_repair")
    else:
        states.append("ineligible_repair")

    if diff is not None:
        change_lists = (
            "added_stops",
            "deleted_stops",
            "day_moves",
            "time_shifts",
            "reorder_changes",
            "lodging_changes",
            "road_changes",
        )
        changes = [item for key in change_lists for item in (diff.get(key) or [])]
        if not changes and float(diff.get("weighted_edit_cost") or 0.0) == 0.0:
            states.append("no_material_change")
        strengths = {str(item.get("owner_strength") or "") for item in changes}
        if "booked" in strengths:
            states.append("permission_required")
        if "locked" in strengths:
            states.append("locked_change_blocked")

    route_validation = (certificate or {}).get("route_validation") or {}
    if route_matrix is None or int(route_validation.get("missing_leg_count") or 0) > 0:
        states.append("missing_route_evidence")
    if int(route_validation.get("fallback_leg_count") or 0) > 0:
        states.append("fallback_route")
    if manifest.get("artifact_freshness") == "stale":
        states.append("stale_artifact")
    if any(str(run.get("execution_status")) == "FAILED" for run in planner_runs):
        states.append("failed_method")
    if any(str(row.get("status")) == "failed" for row in benchmark_rows):
        states.append("failed_method")
    if any(
        _is_exact_incomplete(row)
        for row in [*benchmark_rows, *benchmark_planner_runs]
    ):
        states.append("exact_search_incomplete")
    if any(
        (
            str(row.get("status")).lower() == "failed"
            or str(row.get("execution_status")).upper() == "FAILED"
        )
        and "infeasible" in f"{row.get('error_class', '')} {row.get('error_summary', '')}".lower()
        and not _is_exact_incomplete(row)
        for row in [*benchmark_rows, *benchmark_planner_runs]
    ):
        states.append("complete_infeasibility")
    return tuple(dict.fromkeys(states))


def _is_exact_incomplete(row: dict[str, Any]) -> bool:
    text = f"{row.get('error_class', '')} {row.get('error_summary', '')}".lower()
    method = str(
        row.get("method_id")
        or row.get("method_executed")
        or row.get("method_requested")
        or ""
    )
    return method in {"context_blind_solver", "full_reoptimization"} and (
        "candidate_limit" in text
        or "limit_exceeded" in text
        or "incomplete" in text
        or "cap" in text
    )
