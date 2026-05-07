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

REQUIRED_METHOD_SUMMARY_COLUMNS = {
    "method",
    "allocation_solver",
    "local_route_solver",
    "comparison_score",
    "utility_norm",
    "must_go_norm",
    "diversity_norm",
    "travel_efficiency_norm",
    "cost_efficiency_norm",
    "comparison_score_formula",
}

FORBIDDEN_OLD_NOTEBOOK_MARKERS = [
    "Original Yelp-Only",
    "original_yelp_only",
    "local_yelp_original_pipeline",
    "For a 3-day trip in Santa Barbara",
    "production_local_gurobi_demo",
    "production_gurobi_vs_greedy_baseline",
]

FORBIDDEN_OLD_ARTIFACTS = [
    "gurobi_route_stops.csv",
    "gurobi_route_summary.csv",
    "production_gurobi_vs_greedy_baseline.csv",
    "production_local_gurobi_demo_route_stops.csv",
    "production_local_gurobi_demo_summary.csv",
]


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


def _feature_group_removed_from_map(html: str, feature_group: str | None) -> bool:
    if not feature_group:
        return True
    return f"{feature_group}.remove()" in html


def _check_full_route_offsets(map_debug: pd.DataFrame) -> dict:
    if map_debug.empty or "notes" not in map_debug.columns:
        return _status_row("production_map_route_debug.csv", "full-route selectable offsets", "FAIL", "missing route debug notes")
    full_route_rows = map_debug[map_debug["layer_group"].astype(str).eq("Full Route Overview")].copy()
    offsets: set[float] = set()
    panes: set[str] = set()
    for notes in full_route_rows.get("notes", pd.Series(dtype=str)).fillna("").astype(str):
        offset_match = re.search(r"offset_index=(-?\d+(?:\.\d+)?)", notes)
        pane_match = re.search(r"pane=([^;]+)", notes)
        if offset_match:
            offsets.add(round(float(offset_match.group(1)), 1))
        if pane_match:
            panes.add(pane_match.group(1))
    max_abs_offset = max((abs(offset) for offset in offsets), default=0.0)
    status = "PASS" if len(full_route_rows) >= 9 and len(offsets) >= 5 and max_abs_offset <= 1.3 and panes == {"routeTopPane"} else "FAIL"
    return _status_row(
        "production_map_route_debug.csv",
        "full-route selectable offsets",
        status,
        f"offsets={sorted(offsets)}; max_abs_offset={max_abs_offset:.1f}; panes={sorted(panes)}; rows={len(full_route_rows)}",
    )


def _check_full_route_families(map_debug: pd.DataFrame) -> dict:
    if map_debug.empty or "layer_name" not in map_debug.columns:
        return _status_row("production_map_route_debug.csv", "full-route required families", "FAIL", "missing route debug rows")
    full_route_names = (
        map_debug[map_debug["layer_group"].astype(str).eq("Full Route Overview")]["layer_name"]
        .fillna("")
        .astype(str)
        .tolist()
    )
    required_fragments = [
        "7-Day",
        "9-Day",
        "12-Day",
        "Gurobi",
        "Greedy",
        "Bandit",
        "Traveler: Relaxed",
        "Balanced full route",
        "Traveler: Explorer",
    ]
    missing = [fragment for fragment in required_fragments if not any(fragment in name for name in full_route_names)]
    status = "PASS" if not missing else "FAIL"
    return _status_row(
        "production_map_route_debug.csv",
        "full-route required families",
        status,
        f"routes={len(full_route_names)}; missing={missing}",
    )


def _check_full_route_arrows(map_debug: pd.DataFrame) -> dict:
    if map_debug.empty or "geometry_mode" not in map_debug.columns:
        return _status_row("production_map_route_debug.csv", "full-route arrows/no casing", "FAIL", "missing geometry mode rows")
    full_route_rows = map_debug[map_debug["layer_group"].astype(str).eq("Full Route Overview")].copy()
    geometry_ok = full_route_rows["geometry_mode"].fillna("").astype(str).str.contains("full-route-polyline-with-arrows").all()
    notes = full_route_rows.get("notes", pd.Series("", index=full_route_rows.index)).fillna("").astype(str)
    notes_ok = notes.str.contains("direction_arrows=True").all() and notes.str.contains("no_white_casing=True").all()
    status = "PASS" if not full_route_rows.empty and geometry_ok and notes_ok else "FAIL"
    return _status_row(
        "production_map_route_debug.csv",
        "full-route arrows/no casing",
        status,
        f"rows={len(full_route_rows)}; geometry_ok={bool(geometry_ok)}; notes_ok={bool(notes_ok)}",
    )


def _check_balanced_only_defaults(map_debug: pd.DataFrame) -> list[dict]:
    if map_debug.empty or "layer_group" not in map_debug.columns:
        return [_status_row("production_map_route_debug.csv", "balanced-only default", "FAIL", "missing route debug rows")]
    rows = []
    expected_hidden_groups = [
        "Routes",
        "Selected Result",
        "Trip Length Comparison",
        "Method Comparison",
        "Traveler Comparison",
        "Relaxed Traveler",
        "Balanced Traveler",
        "Explorer Traveler",
        "City Detail",
    ]
    for group in expected_hidden_groups:
        group_rows = map_debug[map_debug["layer_group"].astype(str).eq(group)]
        visible = int(group_rows.get("show_by_default", pd.Series(dtype=bool)).astype(bool).sum())
        rows.append(
            _status_row(
                "production_map_route_debug.csv",
                f"balanced-only hidden default: {group}",
                "PASS" if visible == 0 else "FAIL",
                f"visible={visible}; rows={len(group_rows)}",
            )
        )
    full_route_rows = map_debug[map_debug["layer_group"].astype(str).eq("Full Route Overview")].copy()
    visible_rows = full_route_rows[full_route_rows.get("show_by_default", pd.Series(dtype=bool)).astype(bool)]
    visible_names = visible_rows.get("layer_name", pd.Series(dtype=str)).fillna("").astype(str).tolist()
    rows.append(
        _status_row(
            "production_map_route_debug.csv",
            "balanced-only visible default: Balanced full route",
            "PASS" if len(visible_rows) == 1 and any("Balanced full route" in name for name in visible_names) else "FAIL",
            f"visible={len(visible_rows)}; visible_names={visible_names}",
        )
    )
    return rows


def _extract_route_registry(html: str) -> list[dict]:
    match = re.search(
        r'<script id="blueprint-route-debug-registry" type="application/json">(.*?)</script>',
        html,
        flags=re.DOTALL,
    )
    if not match:
        return []
    try:
        return json.loads(match.group(1).replace("<\\/", "</"))
    except Exception:
        return []


def _check_hidden_by_default_html(html: str, label: str) -> dict:
    feature_group = _feature_group_for_label(html, label)
    added = _feature_group_added_to_map(html, feature_group)
    removed = _feature_group_removed_from_map(html, feature_group)
    hidden = bool(feature_group and (not added or removed))
    return _status_row(
        "production_hierarchical_trip_map.html",
        f"html hidden by default: {label}",
        "PASS" if hidden else "FAIL",
        f"feature_group={feature_group or 'missing'}; addTo={added}; remove_call={removed}",
    )


def _check_method_summary_file(df: pd.DataFrame) -> list[dict]:
    rows = []
    artifact = "production_method_comparison.csv"
    if df.empty:
        return [_status_row(artifact, "method summary", "FAIL", "artifact is missing or empty")]
    missing = sorted(REQUIRED_METHOD_SUMMARY_COLUMNS - set(df.columns))
    rows.append(
        _status_row(
            artifact,
            "method metric columns",
            "PASS" if not missing else "FAIL",
            "solver roles and normalized comparison metrics present" if not missing else f"missing={missing}",
        )
    )
    if "method" in df.columns and {"allocation_solver", "local_route_solver"}.issubset(df.columns):
        expected = {
            "hierarchical_gurobi_pipeline": ("gurobi_master", "legacy_local_gurobi"),
            "hierarchical_greedy_baseline": ("greedy_master", "greedy_local_route"),
            "hierarchical_bandit_gurobi_repair": ("bandit_master", "small_gurobi_repair"),
        }
        mismatches = []
        for method, (allocation_solver, local_solver) in expected.items():
            method_rows = df[df["method"].astype(str).eq(method)]
            if method_rows.empty:
                mismatches.append(f"{method}: missing")
                continue
            first = method_rows.iloc[0]
            if str(first.get("allocation_solver")) != allocation_solver or str(first.get("local_route_solver")) != local_solver:
                mismatches.append(
                    f"{method}: {first.get('allocation_solver')}/{first.get('local_route_solver')}"
                )
        rows.append(
            _status_row(
                artifact,
                "method solver roles",
                "PASS" if not mismatches else "FAIL",
                "all three methods use the intended allocation/local route solvers" if not mismatches else f"mismatches={mismatches}",
            )
        )
    if "comparison_score" in df.columns:
        scores = pd.to_numeric(df["comparison_score"], errors="coerce")
        score_ok = bool(scores.dropna().between(0.0, 1.0).all()) and int(scores.dropna().shape[0]) >= 3
        rows.append(
            _status_row(
                artifact,
                "method comparison score range",
                "PASS" if score_ok else "FAIL",
                f"finite_scores={int(scores.dropna().shape[0])}; min={scores.min()}; max={scores.max()}",
            )
        )
    if "notes" in df.columns:
        greedy_notes = " ".join(
            df.loc[df["method"].astype(str).eq("hierarchical_greedy_baseline"), "notes"]
            .fillna("")
            .astype(str)
            .tolist()
        )
        rows.append(
            _status_row(
                artifact,
                "greedy independent allocation",
                "PASS" if "greedy city/base/day allocation" in greedy_notes.lower() else "FAIL",
                greedy_notes[:240] if greedy_notes else "missing greedy notes",
            )
        )
    return rows


def _check_notebook_old_three_day_removed(project_root: Path) -> list[dict]:
    notebook_path = project_root / "notebook" / "production_system_blueprint.ipynb"
    if not notebook_path.exists():
        return [_status_row("production_system_blueprint.ipynb", "old 3-day notebook path removed", "FAIL", "notebook missing")]
    text = notebook_path.read_text(encoding="utf-8", errors="ignore")
    found = [marker for marker in FORBIDDEN_OLD_NOTEBOOK_MARKERS if marker in text]
    return [
        _status_row(
            "production_system_blueprint.ipynb",
            "old 3-day notebook path removed",
            "PASS" if not found else "FAIL",
            "old Santa Barbara/Yelp-only compatibility cells removed" if not found else f"found={found}",
        )
    ]


def _check_old_three_day_artifacts_removed(output_dir: Path) -> list[dict]:
    rows = []
    stale_existing = [name for name in FORBIDDEN_OLD_ARTIFACTS if (output_dir / name).exists()]
    rows.append(
        _status_row(
            "results/outputs",
            "old 3-day standalone artifacts removed",
            "PASS" if not stale_existing else "FAIL",
            "no old standalone local demo artifacts remain" if not stale_existing else f"existing={stale_existing}",
        )
    )
    contaminated = []
    for name in ["production_experiment_summary.csv", "production_model_comparison.csv", "production_resolved_config.json"]:
        path = output_dir / name
        if path.exists():
            text = path.read_text(encoding="utf-8", errors="ignore")
            if any(marker in text for marker in FORBIDDEN_OLD_NOTEBOOK_MARKERS):
                contaminated.append(name)
    rows.append(
        _status_row(
            "results/outputs",
            "old 3-day markers absent from production artifacts",
            "PASS" if not contaminated else "FAIL",
            "production summaries/config do not contain removed standalone local demo markers" if not contaminated else f"contaminated={contaminated}",
        )
    )
    return rows


def validate(output_dir: Path, figure_dir: Path) -> pd.DataFrame:
    rows: list[dict] = []

    trip_routes = _load_csv(output_dir / "production_trip_length_route_stops.csv")
    method_routes = _load_csv(output_dir / "production_method_route_stops.csv")
    method_summary = _load_csv(output_dir / "production_method_comparison.csv")
    profile_routes = _load_csv(output_dir / "production_day_plan_profiles.csv")
    map_debug = _load_csv(output_dir / "production_map_route_debug.csv")

    rows.extend(_check_route_file(trip_routes, "production_trip_length_route_stops.csv"))
    rows.extend(_check_expected_values(trip_routes, "production_trip_length_route_stops.csv", "trip_days", EXPECTED_TRIP_DAYS))

    rows.extend(_check_route_file(method_routes, "production_method_route_stops.csv"))
    rows.extend(_check_expected_values(method_routes, "production_method_route_stops.csv", "method", EXPECTED_METHODS))
    rows.extend(_check_method_summary_file(method_summary))

    rows.extend(_check_expected_values(profile_routes, "production_day_plan_profiles.csv", "profile", EXPECTED_PROFILES))
    rows.extend(_check_notebook_old_three_day_removed(Path.cwd()))
    rows.extend(_check_old_three_day_artifacts_removed(output_dir))

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
        for expected_group in ["Full Route Overview", "City Detail", "Trip Length Comparison", "Method Comparison", "Traveler Comparison"]:
            found = map_debug["layer_group"].astype(str).eq(expected_group).any()
            rows.append(
                _status_row(
                    "production_map_route_debug.csv",
                    f"{expected_group} rows",
                    "PASS" if found else "FAIL",
                    "present" if found else "missing",
                )
            )
        rows.append(_check_visible_group(map_debug, "Full Route Overview", 1))
        rows.append(_check_distance_coverage(map_debug, "Full Route Overview", 400.0))
        rows.append(_check_distance_coverage(map_debug, "City Detail", 20.0))
        rows.append(_check_distance_coverage(map_debug, "Routes", 700.0))
        rows.append(_check_distance_coverage(map_debug, "Selected Result", 400.0))
        rows.append(_check_full_route_offsets(map_debug))
        rows.append(_check_full_route_families(map_debug))
        rows.append(_check_full_route_arrows(map_debug))
        rows.extend(_check_balanced_only_defaults(map_debug))

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
        "Full Route · Balanced full route",
        "Full Route · Trip Length · 7-Day Hierarchical Route",
        "Full Route · Trip Length · 9-Day Hierarchical Route",
        "Full Route · Trip Length · 12-Day Hierarchical Route",
        "Full Route · Method · Hierarchical Gurobi Pipeline",
        "Full Route · Method · Hierarchical Greedy Baseline",
        "Full Route · Method · Hierarchical + Bandit + Small Gurobi Repair",
        "Full Route · Traveler: Relaxed",
        "Full Route · Traveler: Explorer",
        "City Detail · San Francisco",
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
    balanced_label = "Full Route · Balanced full route"
    feature_group = _feature_group_for_label(html, balanced_label)
    added = _feature_group_added_to_map(html, feature_group)
    removed = _feature_group_removed_from_map(html, feature_group)
    rows.append(
        _status_row(
            "production_hierarchical_trip_map.html",
            f"html visible addTo: {balanced_label}",
            "PASS" if added else "FAIL",
            f"feature_group={feature_group or 'missing'}; removed={removed}",
        )
    )
    rows.append(
        _status_row(
            "production_hierarchical_trip_map.html",
            f"html visible not removed: {balanced_label}",
            "PASS" if added and not removed else "FAIL",
            f"feature_group={feature_group or 'missing'}; addTo={added}; remove_call={removed}",
        )
    )
    for label in [
        "Fastest inter-city route",
        "Scenic CA-1 / PCH route with Stanford detour",
        "Selected Result · Default Hierarchical Route",
        "Full Route · Trip Length · 7-Day Hierarchical Route",
        "Full Route · Trip Length · 9-Day Hierarchical Route",
        "Full Route · Trip Length · 12-Day Hierarchical Route",
        "Full Route · Method · Hierarchical Gurobi Pipeline",
        "Full Route · Method · Hierarchical Greedy Baseline",
        "Full Route · Method · Hierarchical + Bandit + Small Gurobi Repair",
        "Full Route · Traveler: Relaxed",
        "Full Route · Traveler: Explorer",
        "City Detail · San Francisco",
        "Trip Length · 7-Day Hierarchical Route",
        "Trip Length · 9-Day Hierarchical Route",
        "Trip Length · 12-Day Hierarchical Route",
        "Method · Hierarchical Gurobi Pipeline",
        "Method · Hierarchical Greedy Baseline",
        "Method · Hierarchical + Bandit + Small Gurobi Repair",
        "Traveler · Balanced Full Route",
    ]:
        rows.append(_check_hidden_by_default_html(html, label))
    rows.append(
        _status_row(
            "production_hierarchical_trip_map.html",
            "html no always-on full-scene guard",
            "PASS" if "keepFullSceneVisible" not in html else "FAIL",
            "no full-scene guard; checkbox selector owns route state" if "keepFullSceneVisible" not in html else "old keepFullSceneVisible guard still present",
        )
    )
    rows.append(
        _status_row(
            "production_hierarchical_trip_map.html",
            "html route selector controls",
            "PASS" if all(token in html for token in ["blueprint-route-selector", "data-route-checkbox", "syncCheckboxState", "Balanced only"]) else "FAIL",
            "route selector checkbox controls present" if "blueprint-route-selector" in html else "missing route selector",
        )
    )
    rows.append(
        _status_row(
            "production_hierarchical_trip_map.html",
            "html deferred selector bootstrap",
            "PASS" if all(token in html for token in ["mapVarName", "bootstrapRouteSelector", "blueprintRouteRuntimeDiagnostics"]) and "var mapObject = map_" not in html else "FAIL",
            "selector lazily resolves map/layer variables after Folium declarations" if "bootstrapRouteSelector" in html else "missing deferred selector bootstrap",
        )
    )
    rows.append(
        _status_row(
            "production_hierarchical_trip_map.html",
            "html selector focus authority",
            "PASS" if "applyFinalBlueprintFocus" not in html else "FAIL",
            "no later final-focus script overrides route zoom" if "applyFinalBlueprintFocus" not in html else "applyFinalBlueprintFocus still present",
        )
    )
    rows.append(
        _status_row(
            "production_hierarchical_trip_map.html",
            "html collapsible panels",
            "PASS" if all(token in html for token in ["blueprint-panel-header", "blueprint-collapsed", 'data-panel-id="dashboard"', 'data-panel-id="legend"', 'data-panel-id="route-selector"', 'data-panel-id="route-debug"']) else "FAIL",
            "dashboard, legend, selector, and route debug have collapsible panel markup",
        )
    )
    rows.append(
        _status_row(
            "production_hierarchical_trip_map.html",
            "html hidden route debug drawer",
            "PASS" if all(token in html for token in ["blueprint-route-debug-tab", "blueprint-debug-open", "blueprintSetDebugDrawerOpen", "left: -326px"]) else "FAIL",
            "route debug is hidden behind a left-edge reveal tab" if "blueprint-route-debug-tab" in html else "route debug reveal tab missing",
        )
    )
    rows.append(
        _status_row(
            "production_hierarchical_trip_map.html",
            "html draggable panels",
            "PASS" if all(token in html for token in ["blueprint-draggable", "initBlueprintPanels", "pointerdown", 'id="blueprint-route-selector"']) else "FAIL",
            "dashboard and route selector drag script present",
        )
    )
    route_registry = _extract_route_registry(html)
    registry_families = set(str(route.get("family", "")) for route in route_registry)
    default_checked = [str(route.get("control_id", "")) for route in route_registry if bool(route.get("default_checked", route.get("default_visible", False)))]
    rows.append(
        _status_row(
            "production_hierarchical_trip_map.html",
            "html route selector registry",
            "PASS" if len(route_registry) >= 14 and {"trip_length", "method", "traveler_profile", "city_detail", "context"}.issubset(registry_families) else "FAIL",
            f"routes={len(route_registry)}; families={sorted(registry_families)}",
        )
    )
    rows.append(
        _status_row(
            "production_hierarchical_trip_map.html",
            "html balanced-only default checked",
            "PASS" if default_checked == ["traveler_balanced"] else "FAIL",
            f"default_checked={default_checked}",
        )
    )
    rows.append(
        _status_row(
            "production_hierarchical_trip_map.html",
            "html route arrows",
            "PASS" if html.count(".setText(") >= 10 and html.count("\\u003e") >= 10 else "FAIL",
            f"setText={html.count('.setText(')}; escaped_arrow={html.count('\\u003e')}",
        )
    )
    route_badge_count = html.count("blueprint-route-badge")
    rows.append(
        _status_row(
            "production_hierarchical_trip_map.html",
            "html route labels",
            "PASS" if route_badge_count >= 9 else "FAIL",
            f"route_badge_count={route_badge_count}; registry_routes={len(route_registry)}",
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
        "conclusion": (
            "PASS: route data, checkbox registry, balanced-only first-open state, directional arrows, city-detail layers, collapsible/draggable panels, and route panes all validate. "
            "The map is now controlled by selectable route layers rather than one always-on diagnostic bundle."
            if not failed_checks
            else "FAIL: inspect failed_checks; CSV failures suggest upstream data/export issues, while HTML/registry/default-state failures suggest renderer or notebook-state issues."
        ),
        "root_cause_if_failures": (
            "If route CSV checks pass but checkbox/default/arrow/city-detail checks fail, the issue is code/rendering sanity in the map notebook or renderer, not the optimization math."
        ),
        "required_rendering_contract": {
            "default_route": "Only traveler_balanced / Balanced full route is checked and visible on first open.",
            "trip_length_routes": "7, 9, and 12 day full routes must be registered and selectable, but unchecked by default.",
            "method_routes": "hierarchical_gurobi_pipeline, hierarchical_greedy_baseline, and hierarchical_bandit_gurobi_repair must be registered and selectable, but unchecked by default.",
            "traveler_routes": "relaxed, balanced, and explorer full-route overview layers must be registered; balanced is the only default-checked route.",
            "city_detail_routes": "Balanced city/transition detail layers must be registered, directional, and unchecked by default.",
        },
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
