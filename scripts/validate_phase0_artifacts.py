"""Validate Phase 0 research evidence artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

PHASE0_FILES = [
    "production_phase0_dataset_validation.json",
    "production_phase0_planner_runs.csv",
    "production_phase0_plan_artifacts.jsonl",
    "production_phase0_route_audit.csv",
    "production_phase0_evaluation_reports.csv",
    "production_phase0_evidence_summary.csv",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Phase 0 research evidence artifacts.")
    parser.add_argument("--output-dir", default="results/outputs", help="Directory containing production artifacts.")
    parser.add_argument(
        "--require-final-eligible",
        action="store_true",
        help="Require every Phase 0 plan to be eligible for final publication comparison.",
    )
    return parser.parse_args()


def read_csv(path: Path, errors: list[str]) -> pd.DataFrame:
    if not path.exists():
        errors.append(f"Missing required file: {path.name}")
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as exc:
        errors.append(f"Cannot read {path.name}: {type(exc).__name__}: {exc}")
        return pd.DataFrame()


def read_json(path: Path, errors: list[str]) -> dict:
    if not path.exists():
        errors.append(f"Missing required file: {path.name}")
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        errors.append(f"Cannot read {path.name}: {type(exc).__name__}: {exc}")
        return {}
    if not isinstance(payload, dict):
        errors.append(f"{path.name} must contain a JSON object")
        return {}
    return payload


def bool_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(False, index=frame.index)
    return frame[column].astype(str).str.lower().isin({"true", "1", "yes"})


def require_columns(frame: pd.DataFrame, filename: str, columns: list[str], errors: list[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        errors.append(f"{filename} missing columns: {missing}")


def validate(output_dir: Path, *, require_final_eligible: bool = False) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    for filename in PHASE0_FILES:
        if not (output_dir / filename).exists():
            errors.append(f"Missing required file: {filename}")

    dataset = read_json(output_dir / "production_phase0_dataset_validation.json", errors)
    planner_runs = read_csv(output_dir / "production_phase0_planner_runs.csv", errors)
    route_audit = read_csv(output_dir / "production_phase0_route_audit.csv", errors)
    evaluations = read_csv(output_dir / "production_phase0_evaluation_reports.csv", errors)
    summary = read_csv(output_dir / "production_phase0_evidence_summary.csv", errors)
    plan_path = output_dir / "production_phase0_plan_artifacts.jsonl"
    plan_lines = plan_path.read_text(encoding="utf-8").splitlines() if plan_path.exists() else []

    require_columns(
        planner_runs,
        "production_phase0_planner_runs.csv",
        ["run_id", "planning_request_id", "catalog_snapshot_id", "context_snapshot_id", "execution_status"],
        errors,
    )
    require_columns(
        route_audit,
        "production_phase0_route_audit.csv",
        ["run_id", "plan_id", "route_id", "road_validated", "fallback_used", "route_evaluation_eligible"],
        errors,
    )
    require_columns(
        evaluations,
        "production_phase0_evaluation_reports.csv",
        [
            "plan_id",
            "source_run_id",
            "artifact_grounding_status",
            "hard_feasibility_status",
            "comparison_eligibility",
        ],
        errors,
    )
    require_columns(
        summary,
        "production_phase0_evidence_summary.csv",
        [
            "run_id",
            "plan_id",
            "route_id",
            "route_validated_leg_count",
            "route_fallback_leg_count",
            "route_road_validation_coverage",
            "route_road_validated",
            "route_fallback_used",
            "road_route_requested_leg_count",
            "road_route_validated_leg_count",
            "road_route_validation_coverage",
            "comparison_eligibility",
        ],
        errors,
    )

    if dataset and not dataset.get("can_optimize", False):
        errors.append("Dataset validation reports can_optimize=false")
    if dataset and not dataset.get("final_comparison_eligible", False):
        warnings.append("Dataset validation reports final_comparison_eligible=false")
    route_cache_coverage = dataset.get("route_cache_coverage", {}) if isinstance(dataset, dict) else {}
    if isinstance(route_cache_coverage, dict):
        coverage = pd.to_numeric(pd.Series([route_cache_coverage.get("road_route_validation_coverage", 0.0)]), errors="coerce").fillna(0.0).iloc[0]
        requested = int(pd.to_numeric(pd.Series([route_cache_coverage.get("road_route_requested_leg_count", 0)]), errors="coerce").fillna(0).iloc[0])
        if requested and float(coverage) < 1.0:
            warnings.append(f"Road-route cache coverage incomplete: {coverage:.3f}")
            if require_final_eligible:
                errors.append(f"Road-route cache coverage incomplete for final comparison: {coverage:.3f}")

    if not planner_runs.empty and not evaluations.empty:
        run_ids = set(planner_runs["run_id"].astype(str))
        unknown_runs = sorted(set(evaluations["source_run_id"].astype(str)) - run_ids)
        if unknown_runs:
            errors.append(f"Evaluation reports reference unknown source_run_id values: {unknown_runs}")

    if plan_lines and not evaluations.empty:
        plan_ids = set()
        for line_number, line in enumerate(plan_lines, start=1):
            try:
                record = json.loads(line)
            except Exception as exc:
                errors.append(f"Invalid plan artifact JSON on line {line_number}: {type(exc).__name__}: {exc}")
                continue
            plan_id = str(record.get("plan_id", "")).strip()
            source_run_id = str(record.get("source_run_id", "")).strip()
            if not plan_id:
                errors.append(f"Plan artifact line {line_number} missing plan_id")
            if not source_run_id:
                errors.append(f"Plan artifact line {line_number} missing source_run_id")
            plan_ids.add(plan_id)
        unknown_plans = sorted(set(evaluations["plan_id"].astype(str)) - plan_ids)
        if unknown_plans:
            errors.append(f"Evaluation reports reference unknown plan_id values: {unknown_plans}")

    if not route_audit.empty:
        route_flags = route_audit.groupby("plan_id", dropna=False).agg(
            any_fallback=("fallback_used", lambda values: bool_series(pd.DataFrame({"v": values}), "v").any()),
            all_road_validated=("road_validated", lambda values: bool_series(pd.DataFrame({"v": values}), "v").all()),
            all_route_eligible=(
                "route_evaluation_eligible",
                lambda values: bool_series(pd.DataFrame({"v": values}), "v").all(),
            ),
        )
    else:
        route_flags = pd.DataFrame(columns=["any_fallback", "all_road_validated", "all_route_eligible"])

    if not evaluations.empty:
        eligible_evaluations = evaluations[evaluations["comparison_eligibility"].astype(str).eq("eligible")]
        for row in eligible_evaluations.itertuples(index=False):
            plan_id = str(getattr(row, "plan_id"))
            if plan_id not in route_flags.index:
                errors.append(f"Eligible evaluation has no route audit rows: {plan_id}")
                continue
            flags = route_flags.loc[plan_id]
            if bool(flags["any_fallback"]) or not bool(flags["all_road_validated"]) or not bool(flags["all_route_eligible"]):
                errors.append(f"Eligible evaluation has unvalidated or fallback route audit rows: {plan_id}")
        if require_final_eligible:
            ineligible = evaluations[~evaluations["comparison_eligibility"].astype(str).eq("eligible")]
            if not ineligible.empty:
                errors.append(f"{len(ineligible)} evaluation report(s) are not final-comparison eligible")
            if eligible_evaluations.empty:
                errors.append("No final-comparison eligible Phase 0 evaluation reports found")

    if not summary.empty and "route_road_validation_coverage" in summary.columns:
        summary_coverage = pd.to_numeric(summary["route_road_validation_coverage"], errors="coerce").fillna(0.0)
        if (summary_coverage < 1.0).any() and require_final_eligible:
            errors.append("Evidence summary includes routes with incomplete road-validation coverage")

    if not summary.empty and not evaluations.empty:
        summary_pairs = set(zip(summary["run_id"].astype(str), summary["plan_id"].astype(str), strict=False))
        evaluation_pairs = set(zip(evaluations["source_run_id"].astype(str), evaluations["plan_id"].astype(str), strict=False))
        missing_summary = sorted(evaluation_pairs - summary_pairs)
        if missing_summary:
            errors.append(f"Evidence summary missing evaluation pairs: {missing_summary}")

    return errors, warnings


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    errors, warnings = validate(output_dir, require_final_eligible=bool(args.require_final_eligible))
    for warning in warnings:
        print(f"WARNING: {warning}")
    if errors:
        print("FAILED Phase 0 artifact validation")
        for error in errors:
            print(f"- {error}")
        return 1
    print("PASSED Phase 0 artifact validation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
