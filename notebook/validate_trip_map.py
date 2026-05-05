"""Validate the production itinerary map handoff and rendered HTML.

This script is intentionally lightweight: it only reads exported CSV/HTML
artifacts and writes a concise validation report. It is meant to answer:

1. Did the route-stop CSVs include the expected 7/9/12, method, and traveler
   comparison data?
2. Did the renderer audit every route layer, and did any layer collapse?
3. Does the final HTML contain the expected layer labels and route polylines?
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


EXPECTED_TRIP_DAYS = {7, 9, 12}
EXPECTED_METHODS = {
    "hierarchical_gurobi_pipeline",
    "hierarchical_greedy_baseline",
    "hierarchical_bandit_gurobi_repair",
}
EXPECTED_PROFILES = {"relaxed", "balanced", "explorer"}

REQUIRED_ROUTE_COLUMNS = {
    "comparison_type",
    "comparison_label",
    "method",
    "trip_days",
    "day",
    "stop_order",
    "attraction_name",
    "latitude",
    "longitude",
    "route_start_city",
    "route_start_latitude",
    "route_start_longitude",
    "route_end_city",
    "route_end_latitude",
    "route_end_longitude",
    "pass_through_cities",
    "drive_minutes_to_next_base",
    "available_visit_minutes",
    "drive_time_source",
    "route_type",
}


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _status_row(artifact: str, check: str, status: str, detail: str) -> dict:
    return {
        "artifact": artifact,
        "check": check,
        "status": status,
        "detail": detail,
    }


def _check_expected_values(df: pd.DataFrame, artifact: str, column: str, expected: set) -> list[dict]:
    if df.empty:
        return [_status_row(artifact, f"{column} values", "FAIL", "artifact is missing or empty")]
    if column not in df.columns:
        return [_status_row(artifact, f"{column} values", "FAIL", f"missing column {column}")]
    if column == "trip_days":
        found = set(pd.to_numeric(df[column], errors="coerce").dropna().astype(int).tolist())
    else:
        found = set(df[column].dropna().astype(str).tolist())
    missing = sorted(expected - found)
    status = "PASS" if not missing else "FAIL"
    detail = f"found={sorted(found)}" if not missing else f"missing={missing}; found={sorted(found)}"
    return [_status_row(artifact, f"{column} values", status, detail)]


def _check_route_file(df: pd.DataFrame, artifact: str) -> list[dict]:
    rows = []
    if df.empty:
        return [_status_row(artifact, "route rows", "FAIL", "artifact is missing or empty")]

    missing_columns = sorted(REQUIRED_ROUTE_COLUMNS - set(df.columns))
    rows.append(
        _status_row(
            artifact,
            "required columns",
            "PASS" if not missing_columns else "FAIL",
            "all required route metadata present" if not missing_columns else f"missing={missing_columns}",
        )
    )

    coord_columns = [
        "latitude",
        "longitude",
        "route_start_latitude",
        "route_start_longitude",
        "route_end_latitude",
        "route_end_longitude",
    ]
    bad_coord_columns = [
        column
        for column in coord_columns
        if column in df.columns and pd.to_numeric(df[column], errors="coerce").dropna().empty
    ]
    rows.append(
        _status_row(
            artifact,
            "coordinate columns",
            "PASS" if not bad_coord_columns else "FAIL",
            "coordinates are numeric" if not bad_coord_columns else f"non-numeric or empty={bad_coord_columns}",
        )
    )

    if {"route_start_city", "route_end_city", "drive_minutes_to_next_base"}.issubset(df.columns):
        relocation = df["route_start_city"].astype(str).ne(df["route_end_city"].astype(str))
        drive_minutes = pd.to_numeric(df.loc[relocation, "drive_minutes_to_next_base"], errors="coerce")
        status = "PASS" if not relocation.any() or drive_minutes.gt(0).any() else "FAIL"
        detail = (
            f"relocation_rows={int(relocation.sum())}; positive_drive_rows={int(drive_minutes.gt(0).sum())}"
            if relocation.any()
            else "no base-city relocation rows"
        )
        rows.append(_status_row(artifact, "relocation drive metadata", status, detail))
    return rows


def _html_contains(html: str, label: str) -> bool:
    escaped = label.replace("·", "\\u00b7").replace(">", "\\u003e")
    return label in html or escaped in html


def _visible_summary_detail(map_debug: pd.DataFrame) -> str:
    if map_debug.empty:
        return "no renderer debug rows"
    group_counts = (
        map_debug.groupby("layer_group")["show_by_default"]
        .agg(visible=lambda col: int(col.astype(bool).sum()), total="count")
        .reset_index()
    )
    return "; ".join(
        f"{row.layer_group}: {row.visible}/{row.total}"
        for row in group_counts.itertuples(index=False)
    )


def _check_visible_group(map_debug: pd.DataFrame, group: str, minimum_visible: int) -> dict:
    if map_debug.empty or "layer_group" not in map_debug.columns:
        return _status_row("production_map_route_debug.csv", f"{group} visible by default", "FAIL", "missing route debug rows")
    group_rows = map_debug[map_debug["layer_group"].astype(str).eq(group)].copy()
    visible_count = int(group_rows.get("show_by_default", pd.Series(dtype=bool)).astype(bool).sum())
    status = "PASS" if visible_count >= minimum_visible else "FAIL"
    return _status_row(
        "production_map_route_debug.csv",
        f"{group} visible by default",
        status,
        f"visible={visible_count}; required>={minimum_visible}",
    )


def _check_distance_coverage(map_debug: pd.DataFrame, group: str, minimum_km: float) -> dict:
    if map_debug.empty or "distance_km" not in map_debug.columns:
        return _status_row("production_map_route_debug.csv", f"{group} distance coverage", "FAIL", "missing route debug rows")
    group_rows = map_debug[map_debug["layer_group"].astype(str).eq(group)].copy()
    if group == "Full Scene Overview" and "comparison_type" in group_rows.columns:
        # The aggregate full-scene row concatenates several routes, so its
        # distance is not a route distance. Coverage should be proven by the
        # individual always-visible routes.
        group_rows = group_rows[~group_rows["comparison_type"].astype(str).eq("full_scene")]
    max_distance = float(pd.to_numeric(group_rows.get("distance_km", pd.Series(dtype=float)), errors="coerce").max() or 0.0)
    status = "PASS" if max_distance >= float(minimum_km) else "FAIL"
    return _status_row(
        "production_map_route_debug.csv",
        f"{group} distance coverage",
        status,
        f"max_distance_km={max_distance:.1f}; required>={minimum_km:.1f}",
    )


def _feature_group_for_label(html: str, label: str) -> str | None:
    escaped = re.escape(label.replace("·", "\\u00b7").replace(">", "\\u003e"))
    match = re.search(rf'"{escaped}"\s*:\s*(feature_group_[a-f0-9]+)', html)
    return match.group(1) if match else None


def _feature_group_added_to_map(html: str, feature_group: str | None) -> bool:
    if not feature_group:
        return False
    return f"{feature_group}.addTo(" in html


def validate(output_dir: Path, figure_dir: Path) -> pd.DataFrame:
    rows: list[dict] = []

    trip_routes = _load_csv(output_dir / "production_trip_length_route_stops.csv")
    method_routes = _load_csv(output_dir / "production_method_route_stops.csv")
    profile_routes = _load_csv(output_dir / "production_day_plan_profiles.csv")
    map_debug = _load_csv(output_dir / "production_map_route_debug.csv")

    rows.extend(_check_route_file(trip_routes, "production_trip_length_route_stops.csv"))
    rows.extend(_check_expected_values(trip_routes, "production_trip_length_route_stops.csv", "trip_days", EXPECTED_TRIP_DAYS))

    rows.extend(_check_route_file(method_routes, "production_method_route_stops.csv"))
    rows.extend(_check_expected_values(method_routes, "production_method_route_stops.csv", "method", EXPECTED_METHODS))

    rows.extend(_check_expected_values(profile_routes, "production_day_plan_profiles.csv", "profile", EXPECTED_PROFILES))

    if map_debug.empty:
        rows.append(_status_row("production_map_route_debug.csv", "renderer audit", "FAIL", "missing renderer audit rows"))
    else:
        failed = map_debug[map_debug["draw_status"].astype(str).eq("FAILED")]
        warned = map_debug[map_debug["draw_status"].astype(str).eq("WARN")]
        visible = map_debug[map_debug["show_by_default"].astype(bool)]
        rows.append(
            _status_row(
                "production_map_route_debug.csv",
                "renderer audit statuses",
                "PASS" if failed.empty else "FAIL",
                f"rows={len(map_debug)}; visible_by_default={len(visible)}; warnings={len(warned)}; failed={len(failed)}; {_visible_summary_detail(map_debug)}",
            )
        )
        for expected_group in ["Full Scene Overview", "Trip Length Comparison", "Method Comparison", "Traveler Comparison"]:
            found = map_debug["layer_group"].astype(str).eq(expected_group).any()
            rows.append(
                _status_row(
                    "production_map_route_debug.csv",
                    f"{expected_group} rows",
                    "PASS" if found else "FAIL",
                    "present" if found else "missing",
                )
            )
        rows.append(_check_visible_group(map_debug, "Full Scene Overview", 1))
        rows.append(_check_visible_group(map_debug, "Trip Length Comparison", 3))
        rows.append(_check_visible_group(map_debug, "Method Comparison", 3))
        rows.append(_check_visible_group(map_debug, "Traveler Comparison", 3))
        rows.append(_check_distance_coverage(map_debug, "Full Scene Overview", 700.0))
        rows.append(_check_distance_coverage(map_debug, "Routes", 700.0))
        rows.append(_check_distance_coverage(map_debug, "Selected Result", 400.0))

    html_path = figure_dir / "production_hierarchical_trip_map.html"
    html = html_path.read_text(encoding="utf-8", errors="ignore") if html_path.exists() else ""
    rows.append(
        _status_row(
            "production_hierarchical_trip_map.html",
            "html exists",
            "PASS" if html else "FAIL",
            f"bytes={html_path.stat().st_size}" if html_path.exists() else "missing",
        )
    )
    for label in [
        "Full Scene · Always Visible Route Overview",
        "Trip Length · 7-Day Hierarchical Route",
        "Trip Length · 9-Day Hierarchical Route",
        "Trip Length · 12-Day Hierarchical Route",
        "Method · Hierarchical Gurobi Pipeline",
        "Method · Hierarchical Greedy Baseline",
        "Method · Hierarchical + Bandit + Small Gurobi Repair",
        "Traveler · Relaxed Full Route",
        "Traveler · Balanced Full Route",
        "Traveler · Explorer Full Route",
    ]:
        rows.append(
            _status_row(
                "production_hierarchical_trip_map.html",
                f"html layer label: {label}",
                "PASS" if _html_contains(html, label) else "FAIL",
                "present" if _html_contains(html, label) else "missing",
            )
        )
    for label in [
        "Full Scene · Always Visible Route Overview",
        "Trip Length · 7-Day Hierarchical Route",
        "Trip Length · 9-Day Hierarchical Route",
        "Trip Length · 12-Day Hierarchical Route",
        "Method · Hierarchical Gurobi Pipeline",
        "Method · Hierarchical Greedy Baseline",
        "Method · Hierarchical + Bandit + Small Gurobi Repair",
        "Traveler · Balanced Full Route",
    ]:
        feature_group = _feature_group_for_label(html, label)
        rows.append(
            _status_row(
                "production_hierarchical_trip_map.html",
                f"html visible addTo: {label}",
                "PASS" if _feature_group_added_to_map(html, feature_group) else "FAIL",
                f"feature_group={feature_group or 'missing'}",
            )
        )
    for pane in ["routeContextPane", "routeCorePane", "routeTopPane"]:
        rows.append(
            _status_row(
                "production_hierarchical_trip_map.html",
                f"html route pane: {pane}",
                "PASS" if pane in html else "FAIL",
                "present" if pane in html else "missing",
            )
        )
    polyline_count = html.count("L.polyline(")
    rows.append(
        _status_row(
            "production_hierarchical_trip_map.html",
            "polyline count",
            "PASS" if polyline_count >= 20 else "WARN",
            f"polyline_count={polyline_count}",
        )
    )
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="results/outputs")
    parser.add_argument("--figure-dir", default="results/figures")
    parser.add_argument("--strict", action="store_true", help="return nonzero when any check fails")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    figure_dir = Path(args.figure_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = validate(output_dir, figure_dir)
    report_path = output_dir / "production_map_validation_report.csv"
    summary_path = output_dir / "production_map_validation_summary.json"
    report.to_csv(report_path, index=False)
    summary = report["status"].value_counts().to_dict()
    failed_checks = report[report["status"].astype(str).eq("FAIL")].to_dict("records")
    warning_checks = report[report["status"].astype(str).eq("WARN")].to_dict("records")
    diagnosis = {
        "summary": summary,
        "failed_checks": failed_checks,
        "warning_checks": warning_checks,
        "root_cause_if_failures": (
            "If route CSV checks pass but visible/default or html addTo checks fail, the issue is code/rendering sanity in the map notebook or renderer, not the optimization math."
        ),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "production_map_validation_diagnosis.json").write_text(
        json.dumps(diagnosis, indent=2),
        encoding="utf-8",
    )
    print(report.to_string(index=False))
    print(f"\nWrote {report_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {output_dir / 'production_map_validation_diagnosis.json'}")
    if args.strict and report["status"].astype(str).eq("FAIL").any():
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
