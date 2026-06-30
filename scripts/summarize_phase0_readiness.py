"""Summarize Phase 0 evidence readiness for papers and debugging."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Phase 0 artifact readiness.")
    parser.add_argument("--output-dir", default="results/outputs", help="Directory containing Phase 0 artifacts.")
    parser.add_argument(
        "--format",
        choices=("markdown", "json"),
        default="markdown",
        help="Output format printed to stdout.",
    )
    parser.add_argument(
        "--write-dir",
        default="",
        help="Optional directory for phase0_readiness_summary.md and phase0_method_readiness.csv.",
    )
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except Exception:
        return default
    return result if pd.notna(result) else default


def _safe_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except Exception:
        pass
    text = str(value).strip()
    return text if text else default


def _evaluation_lookup(evaluations: pd.DataFrame) -> dict[tuple[str, str], dict[str, Any]]:
    if evaluations.empty:
        return {}
    lookup = {}
    for row in evaluations.to_dict("records"):
        key = (_safe_str(row.get("source_run_id")), _safe_str(row.get("plan_id")))
        lookup[key] = row
    return lookup


def _blocking_reason(summary_row: dict[str, Any], evaluation_row: dict[str, Any] | None) -> str:
    reasons = []
    if _safe_float(summary_row.get("route_road_validation_coverage"), 0.0) < 1.0:
        reasons.append("route_not_road_validated")
    if _safe_str(summary_row.get("comparison_eligibility")) != "eligible":
        reasons.append("comparison_ineligible")
    if evaluation_row:
        if _safe_str(evaluation_row.get("artifact_grounding_status")) != "PASSED":
            reasons.append("artifact_grounding_failed")
        if _safe_str(evaluation_row.get("hard_feasibility_status")) != "PASSED":
            reasons.append("hard_feasibility_failed")
    return "; ".join(dict.fromkeys(reasons)) or "none"


def load_phase0_readiness(output_dir: str | Path) -> dict[str, Any]:
    output_path = Path(output_dir)
    dataset = _read_json(output_path / "production_phase0_dataset_validation.json")
    summary = _read_csv(output_path / "production_phase0_evidence_summary.csv")
    evaluations = _read_csv(output_path / "production_phase0_evaluation_reports.csv")
    cache_audit = _read_csv(output_path / "production_road_route_cache_audit.csv")
    route_cache_coverage = dataset.get("route_cache_coverage", {}) if isinstance(dataset, dict) else {}
    evaluation_lookup = _evaluation_lookup(evaluations)

    method_rows: list[dict[str, Any]] = []
    for row in summary.to_dict("records"):
        key = (_safe_str(row.get("run_id")), _safe_str(row.get("plan_id")))
        evaluation_row = evaluation_lookup.get(key)
        method_rows.append(
            {
                "method": _safe_str(row.get("method")),
                "run_id": _safe_str(row.get("run_id")),
                "plan_id": _safe_str(row.get("plan_id")),
                "comparison_eligibility": _safe_str(row.get("comparison_eligibility"), "ineligible"),
                "route_leg_count": _safe_int(row.get("route_leg_count")),
                "route_validated_leg_count": _safe_int(row.get("route_validated_leg_count")),
                "route_fallback_leg_count": _safe_int(row.get("route_fallback_leg_count")),
                "route_road_validation_coverage": _safe_float(row.get("route_road_validation_coverage")),
                "artifact_grounding_status": _safe_str((evaluation_row or {}).get("artifact_grounding_status")),
                "hard_feasibility_status": _safe_str((evaluation_row or {}).get("hard_feasibility_status")),
                "blocking_reason": _blocking_reason(row, evaluation_row),
            }
        )

    evaluation_count = len(evaluations)
    eligible_count = (
        int(evaluations["comparison_eligibility"].astype(str).eq("eligible").sum())
        if "comparison_eligibility" in evaluations.columns
        else 0
    )
    requested_legs = _safe_int(route_cache_coverage.get("road_route_requested_leg_count"), len(cache_audit))
    validated_legs = _safe_int(route_cache_coverage.get("road_route_validated_leg_count"))
    missing_legs = _safe_int(route_cache_coverage.get("road_route_missing_leg_count"), max(0, requested_legs - validated_legs))
    coverage = _safe_float(route_cache_coverage.get("road_route_validation_coverage"))
    strict_ready = bool(evaluation_count and eligible_count == evaluation_count and requested_legs > 0 and coverage >= 1.0)

    return {
        "output_dir": str(output_path),
        "catalog_snapshot_id": _safe_str(dataset.get("catalog_snapshot_id")),
        "context_snapshot_id": _safe_str(dataset.get("context_snapshot_id")),
        "dataset_can_optimize": bool(dataset.get("can_optimize", False)),
        "dataset_final_comparison_eligible": bool(dataset.get("final_comparison_eligible", False)),
        "road_route_requested_leg_count": requested_legs,
        "road_route_validated_leg_count": validated_legs,
        "road_route_missing_leg_count": missing_legs,
        "road_route_validation_coverage": coverage,
        "road_route_cache_hash": _safe_str(route_cache_coverage.get("road_route_cache_hash")),
        "road_route_cache_audit_hash": _safe_str(route_cache_coverage.get("road_route_cache_audit_hash")),
        "evaluation_count": evaluation_count,
        "eligible_evaluation_count": eligible_count,
        "strict_comparison_ready": strict_ready,
        "method_readiness": method_rows,
    }


def method_readiness_frame(readiness: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(readiness.get("method_readiness", []))


def readiness_markdown(readiness: dict[str, Any]) -> str:
    coverage = _safe_float(readiness.get("road_route_validation_coverage"))
    requested = _safe_int(readiness.get("road_route_requested_leg_count"))
    validated = _safe_int(readiness.get("road_route_validated_leg_count"))
    missing = _safe_int(readiness.get("road_route_missing_leg_count"))
    eligible = _safe_int(readiness.get("eligible_evaluation_count"))
    total = _safe_int(readiness.get("evaluation_count"))
    lines = [
        "# Phase 0 Readiness Summary",
        "",
        f"- Catalog snapshot: `{_safe_str(readiness.get('catalog_snapshot_id'), 'unknown')}`",
        f"- Context snapshot: `{_safe_str(readiness.get('context_snapshot_id'), 'unknown')}`",
        f"- Dataset can optimize: `{bool(readiness.get('dataset_can_optimize'))}`",
        f"- Road-route cache coverage: `{validated}/{requested}` (`{coverage:.3f}`), missing `{missing}`",
        f"- Eligible evaluations: `{eligible}/{total}`",
        f"- Strict comparison ready: `{bool(readiness.get('strict_comparison_ready'))}`",
        "",
        "## Method Readiness",
        "",
        "| Method | Eligibility | Road Coverage | Validated Legs | Fallback Legs | Blocking Reason |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for row in readiness.get("method_readiness", []):
        lines.append(
            "| {method} | {eligibility} | {coverage:.3f} | {validated} | {fallback} | {reason} |".format(
                method=_safe_str(row.get("method"), "unknown"),
                eligibility=_safe_str(row.get("comparison_eligibility"), "unknown"),
                coverage=_safe_float(row.get("route_road_validation_coverage")),
                validated=_safe_int(row.get("route_validated_leg_count")),
                fallback=_safe_int(row.get("route_fallback_leg_count")),
                reason=_safe_str(row.get("blocking_reason"), "unknown"),
            )
        )
    return "\n".join(lines) + "\n"


def write_readiness_outputs(readiness: dict[str, Any], write_dir: str | Path) -> dict[str, Path]:
    output_dir = Path(write_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = output_dir / "phase0_readiness_summary.md"
    method_path = output_dir / "phase0_method_readiness.csv"
    markdown_path.write_text(readiness_markdown(readiness), encoding="utf-8")
    method_readiness_frame(readiness).to_csv(method_path, index=False)
    return {"markdown": markdown_path, "method_csv": method_path}


def main() -> int:
    args = parse_args()
    readiness = load_phase0_readiness(args.output_dir)
    if args.format == "json":
        print(json.dumps(readiness, indent=2, sort_keys=True))
    else:
        print(readiness_markdown(readiness), end="")
    if args.write_dir:
        paths = write_readiness_outputs(readiness, args.write_dir)
        print(f"Wrote {paths['markdown']}")
        print(f"Wrote {paths['method_csv']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
