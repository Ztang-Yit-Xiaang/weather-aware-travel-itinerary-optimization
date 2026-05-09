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
EXPECTED_METHOD_ORDER = [
    "hierarchical_gurobi_pipeline",
    "hierarchical_greedy_baseline",
    "hierarchical_bandit_gurobi_repair",
]
EXPECTED_PROFILE_ORDER = ["relaxed", "balanced", "explorer"]
METHOD_ROUTE_KEY_SLUG = {
    "hierarchical_gurobi_pipeline": "gurobi",
    "hierarchical_greedy_baseline": "greedy",
    "hierarchical_bandit_gurobi_repair": "bandit_repair",
}
MATRIX_DEFAULT_ROUTE_KEY = "d7__bandit_repair__balanced"
EXPECTED_ROUTE_MATRIX_KEYS = {
    f"d{days}__{METHOD_ROUTE_KEY_SLUG[method]}__{profile}"
    for days in EXPECTED_TRIP_DAYS
    for method in EXPECTED_METHODS
    for profile in EXPECTED_PROFILES
}

REQUIRED_ROUTE_MATRIX_COLUMNS = {
    "route_key",
    "trip_days",
    "method",
    "method_display_name",
    "method_short_label",
    "profile",
    "profile_label",
    "comparison_label",
    "hotel_showcase_group",
    "selector_parent",
    "route_layer_var",
    "hotel_layer_var",
    "city_detail_layer_vars",
    "transition_layer_vars",
    "comparison_score",
    "must_go_selected_count",
    "must_go_skipped_count",
    "must_go_coverage_ratio",
}

REQUIRED_ROUTE_MATRIX_STOP_COLUMNS = {
    "route_key",
    "profile",
    "profile_label",
    "selector_parent",
    "hotel_showcase_group",
    "comparison_type",
    "comparison_label",
    "method",
    "method_display_name",
    "trip_days",
    "day",
    "stop_order",
    "attraction_name",
    "city",
    "overnight_city",
    "hotel_name",
    "hotel_latitude",
    "hotel_longitude",
    "latitude",
    "longitude",
    "route_type",
    "route_start_city",
    "route_end_city",
}

REQUIRED_ROUTE_MATRIX_HOTEL_COLUMNS = {
    "route_key",
    "trip_days",
    "method",
    "profile",
    "comparison_label",
    "city",
    "hotel_name",
    "candidate_rank",
    "selected",
    "latitude",
    "longitude",
    "hotel_score",
    "selected_hotel_reason",
}

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
    "social_must_go",
    "social_score",
    "must_go_weight",
    "social_reason",
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
    "must_go_total",
    "must_go_selected_count",
    "must_go_skipped_count",
    "must_go_coverage_ratio",
    "must_go_skipped_names",
    "must_go_policy",
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


def _check_trip_length_uses_hybrid_bandit(df: pd.DataFrame) -> list[dict]:
    if df.empty or "method" not in df.columns:
        return [_status_row("production_trip_length_route_stops.csv", "trip-length hybrid method", "FAIL", "missing trip-length route stops")]
    methods = sorted(set(df["method"].dropna().astype(str).tolist()))
    labels = sorted(set(df.get("comparison_label", pd.Series(dtype=str)).dropna().astype(str).tolist()))
    ok_methods = methods == ["hierarchical_bandit_gurobi_repair"]
    ok_labels = all("Hybrid Bandit + Small Gurobi" in label for label in labels) if labels else False
    return [
        _status_row(
            "production_trip_length_route_stops.csv",
            "trip-length hybrid method",
            "PASS" if ok_methods and ok_labels else "FAIL",
            f"methods={methods}; labels={labels[:3]}",
        )
    ]


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
    geometry_ok = full_route_rows["geometry_mode"].fillna("").astype(str).str.contains("full-route-polyline-with-canvas-arrows").all()
    notes = full_route_rows.get("notes", pd.Series("", index=full_route_rows.index)).fillna("").astype(str)
    notes_ok = notes.str.contains("direction_arrows=canvas").all() and notes.str.contains("no_white_casing=True").all()
    status = "PASS" if not full_route_rows.empty and geometry_ok and notes_ok else "FAIL"
    return _status_row(
        "production_map_route_debug.csv",
        "full-route arrows/no casing",
        status,
        f"rows={len(full_route_rows)}; geometry_ok={bool(geometry_ok)}; notes_ok={bool(notes_ok)}",
    )


def _check_city_loop_semantics(map_debug: pd.DataFrame) -> list[dict]:
    rows = []
    city_rows = map_debug[map_debug["comparison_type"].astype(str).eq("city_detail")].copy() if not map_debug.empty else pd.DataFrame()
    if city_rows.empty:
        rows.append(_status_row("production_map_route_debug.csv", "city detail hotel loops", "FAIL", "missing city detail rows"))
    else:
        same_city = bool(city_rows["route_start_city"].astype(str).eq(city_rows["route_end_city"].astype(str)).all())
        notes = city_rows.get("notes", pd.Series("", index=city_rows.index)).fillna("").astype(str)
        route_types = city_rows.get("route_type", pd.Series("", index=city_rows.index)).astype(str)
        notes_ok = bool(notes.str.contains("hotel_loop_start_end_same=true").all())
        route_type_ok = bool(route_types.eq("hotel_loop").all())
        rows.append(
            _status_row(
                "production_map_route_debug.csv",
                "city detail hotel loops",
                "PASS" if same_city and notes_ok and route_type_ok else "FAIL",
                f"rows={len(city_rows)}; same_city={same_city}; notes_ok={notes_ok}; route_type_ok={route_type_ok}",
            )
        )

    transition_rows = map_debug[map_debug["comparison_type"].astype(str).eq("transition")].copy() if not map_debug.empty else pd.DataFrame()
    if transition_rows.empty:
        rows.append(_status_row("production_map_route_debug.csv", "transition routes separated", "FAIL", "missing transition route rows"))
    else:
        separate = bool(transition_rows["route_start_city"].astype(str).ne(transition_rows["route_end_city"].astype(str)).any())
        notes = transition_rows.get("notes", pd.Series("", index=transition_rows.index)).fillna("").astype(str)
        notes_ok = bool(notes.str.contains("transition_route_separate_from_city_loop=true").all())
        rows.append(
            _status_row(
                "production_map_route_debug.csv",
                "transition routes separated",
                "PASS" if separate and notes_ok else "FAIL",
                f"rows={len(transition_rows)}; has_intercity_transition={separate}; notes_ok={notes_ok}",
            )
        )
    return rows


def _check_road_geometry_report(map_debug: pd.DataFrame) -> list[dict]:
    rows = []
    required = {"geometry_source", "straight_line_fallback_count", "road_geometry_percent"}
    missing = sorted(required - set(map_debug.columns))
    rows.append(
        _status_row(
            "production_map_route_debug.csv",
            "road geometry report columns",
            "PASS" if not missing else "FAIL",
            "present" if not missing else f"missing={missing}",
        )
    )
    if missing or map_debug.empty:
        return rows
    sources = set(map_debug["geometry_source"].fillna("").astype(str).tolist())
    road_percent = pd.to_numeric(map_debug["road_geometry_percent"], errors="coerce")
    fallback_count = int(pd.to_numeric(map_debug["straight_line_fallback_count"], errors="coerce").fillna(0).sum())
    rows.append(
        _status_row(
            "production_map_route_debug.csv",
            "road geometry source coverage",
            "PASS" if any(source for source in sources if source and source != "unknown") else "FAIL",
            f"sources={sorted(sources)}; fallback_flags={fallback_count}; avg_road_percent={road_percent.dropna().mean() if not road_percent.dropna().empty else 'n/a'}",
        )
    )
    return rows


def _check_must_go_coverage_file(df: pd.DataFrame) -> list[dict]:
    required = {"method", "must_go_name", "selected", "status", "must_go_policy", "must_go_weight", "social_score"}
    if df.empty:
        return [_status_row("production_must_go_coverage.csv", "must-go coverage artifact", "FAIL", "missing or empty")]
    missing = sorted(required - set(df.columns))
    rows = [
        _status_row(
            "production_must_go_coverage.csv",
            "must-go coverage columns",
            "PASS" if not missing else "FAIL",
            "present" if not missing else f"missing={missing}",
        )
    ]
    methods = set(df.get("method", pd.Series(dtype=str)).dropna().astype(str).tolist())
    rows.append(
        _status_row(
            "production_must_go_coverage.csv",
            "must-go coverage methods",
            "PASS" if EXPECTED_METHODS.issubset(methods) else "FAIL",
            f"methods={sorted(methods)}",
        )
    )
    selected_count = int(df.get("selected", pd.Series(dtype=bool)).astype(bool).sum()) if "selected" in df.columns else 0
    rows.append(
        _status_row(
            "production_must_go_coverage.csv",
            "must-go selected/skipped states",
            "PASS" if selected_count > 0 and "skipped_soft_reward" in set(df.get("status", pd.Series(dtype=str)).astype(str)) else "FAIL",
            f"selected_rows={selected_count}; statuses={sorted(set(df.get('status', pd.Series(dtype=str)).astype(str)))}",
        )
    )
    return rows


def _check_hotel_debug_file(df: pd.DataFrame) -> list[dict]:
    required = {"city", "hotel_name", "candidate_rank", "selected", "hotel_score", "selected_hotel_reason"}
    if df.empty:
        return [_status_row("production_hotel_selection_debug.csv", "hotel debug artifact", "FAIL", "missing or empty")]
    missing = sorted(required - set(df.columns))
    selected_count = int(df.get("selected", pd.Series(dtype=bool)).astype(bool).sum()) if "selected" in df.columns else 0
    return [
        _status_row(
            "production_hotel_selection_debug.csv",
            "hotel debug columns",
            "PASS" if not missing else "FAIL",
            "present" if not missing else f"missing={missing}",
        ),
        _status_row(
            "production_hotel_selection_debug.csv",
            "selected hotel rows",
            "PASS" if selected_count > 0 else "FAIL",
            f"selected_rows={selected_count}; rows={len(df)}",
        ),
    ]


def _extract_route_key(notes: str) -> str:
    match = re.search(r"route_key=([^;]+)", str(notes or ""))
    return match.group(1).strip() if match else ""


def _route_matrix_combo_keys(df: pd.DataFrame) -> set[str]:
    if df.empty or "route_key" not in df.columns:
        return set()
    return set(df["route_key"].dropna().astype(str).tolist())


def _check_route_matrix_artifacts(
    matrix_df: pd.DataFrame,
    matrix_stops: pd.DataFrame,
    matrix_hotels: pd.DataFrame,
) -> list[dict]:
    rows: list[dict] = []
    artifact = "production_route_matrix_comparison.csv"
    if matrix_df.empty:
        rows.append(_status_row(artifact, "route matrix artifact", "FAIL", "missing or empty"))
    else:
        missing = sorted(REQUIRED_ROUTE_MATRIX_COLUMNS - set(matrix_df.columns))
        rows.append(
            _status_row(
                artifact,
                "route matrix columns",
                "PASS" if not missing else "FAIL",
                "present" if not missing else f"missing={missing}",
            )
        )
        if {"trip_days", "method", "profile", "route_key"}.issubset(matrix_df.columns):
            combos = matrix_df[["trip_days", "method", "profile", "route_key"]].drop_duplicates().copy()
            combos["trip_days"] = pd.to_numeric(combos["trip_days"], errors="coerce").fillna(0).astype(int)
            found_days = set(combos["trip_days"].tolist())
            found_methods = set(combos["method"].astype(str).tolist())
            found_profiles = set(combos["profile"].astype(str).tolist())
            found_keys = set(combos["route_key"].astype(str).tolist())
            duplicate_keys = matrix_df["route_key"].astype(str).value_counts()
            duplicate_keys = duplicate_keys[duplicate_keys.gt(1)].index.tolist()
            ok = (
                len(combos) == 27
                and found_days == EXPECTED_TRIP_DAYS
                and found_methods == EXPECTED_METHODS
                and found_profiles == EXPECTED_PROFILES
                and found_keys == EXPECTED_ROUTE_MATRIX_KEYS
                and not duplicate_keys
            )
            rows.append(
                _status_row(
                    artifact,
                    "27-route matrix coverage",
                    "PASS" if ok else "FAIL",
                    f"combos={len(combos)}; days={sorted(found_days)}; methods={sorted(found_methods)}; profiles={sorted(found_profiles)}; missing_keys={sorted(EXPECTED_ROUTE_MATRIX_KEYS - found_keys)[:8]}; duplicates={duplicate_keys[:8]}",
                )
            )
            route_key_format_ok = all(
                str(row.route_key) == f"d{int(row.trip_days)}__{METHOD_ROUTE_KEY_SLUG.get(str(row.method), str(row.method))}__{row.profile}"
                for row in combos.itertuples(index=False)
            )
            rows.append(
                _status_row(
                    artifact,
                    "stable route_key format",
                    "PASS" if route_key_format_ok else "FAIL",
                    "route keys match d{days}__{method_slug}__{profile}" if route_key_format_ok else "one or more route keys do not match their method/day/profile",
                )
            )
        layer_columns = ["route_layer_var", "hotel_layer_var", "city_detail_layer_vars", "transition_layer_vars"]
        if set(layer_columns).issubset(matrix_df.columns):
            blanks = {
                column: int(matrix_df[column].fillna("").astype(str).str.strip().eq("").sum())
                for column in layer_columns
            }
            rows.append(
                _status_row(
                    artifact,
                    "route matrix layer vars",
                    "PASS" if all(count == 0 for count in blanks.values()) else "FAIL",
                    f"blank_counts={blanks}",
                )
            )

    stop_artifact = "production_route_matrix_route_stops.csv"
    if matrix_stops.empty:
        rows.append(_status_row(stop_artifact, "route matrix stops", "FAIL", "missing or empty"))
    else:
        missing = sorted(REQUIRED_ROUTE_MATRIX_STOP_COLUMNS - set(matrix_stops.columns))
        rows.append(
            _status_row(
                stop_artifact,
                "route matrix stop columns",
                "PASS" if not missing else "FAIL",
                "present" if not missing else f"missing={missing}",
            )
        )
        found_stop_keys = _route_matrix_combo_keys(matrix_stops)
        rows.append(
            _status_row(
                stop_artifact,
                "route matrix stops per route",
                "PASS" if found_stop_keys == EXPECTED_ROUTE_MATRIX_KEYS else "FAIL",
                f"route_keys={len(found_stop_keys)}; missing={sorted(EXPECTED_ROUTE_MATRIX_KEYS - found_stop_keys)[:8]}",
            )
        )

    hotel_artifact = "production_route_matrix_hotel_selection_debug.csv"
    if matrix_hotels.empty:
        rows.append(_status_row(hotel_artifact, "route matrix hotels", "FAIL", "missing or empty"))
    else:
        missing = sorted(REQUIRED_ROUTE_MATRIX_HOTEL_COLUMNS - set(matrix_hotels.columns))
        rows.append(
            _status_row(
                hotel_artifact,
                "route matrix hotel columns",
                "PASS" if not missing else "FAIL",
                "present" if not missing else f"missing={missing}",
            )
        )
        selected_hotels = matrix_hotels[matrix_hotels.get("selected", pd.Series(dtype=bool)).astype(bool)].copy()
        selected_keys = _route_matrix_combo_keys(selected_hotels)
        rows.append(
            _status_row(
                hotel_artifact,
                "route-specific selected hotels",
                "PASS" if selected_keys == EXPECTED_ROUTE_MATRIX_KEYS else "FAIL",
                f"selected_route_keys={len(selected_keys)}; missing={sorted(EXPECTED_ROUTE_MATRIX_KEYS - selected_keys)[:8]}",
            )
        )
        if not matrix_stops.empty and {"route_key", "overnight_city"}.issubset(matrix_stops.columns) and {"route_key", "city"}.issubset(selected_hotels.columns):
            overnight_by_key = {
                key: set(group["overnight_city"].dropna().astype(str).tolist())
                for key, group in matrix_stops.groupby("route_key", dropna=False)
            }
            mismatches = []
            for row in selected_hotels.itertuples(index=False):
                key = str(row.route_key)
                city = str(row.city)
                if city not in overnight_by_key.get(key, set()):
                    mismatches.append(f"{key}:{city}")
            rows.append(
                _status_row(
                    hotel_artifact,
                    "selected hotel city binding",
                    "PASS" if not mismatches else "FAIL",
                    "selected hotels belong only to their route overnight cities" if not mismatches else f"mismatches={mismatches[:10]}",
                )
            )
    return rows


def _check_route_matrix_debug(map_debug: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []
    if map_debug.empty:
        return [_status_row("production_map_route_debug.csv", "route matrix renderer rows", "FAIL", "missing route debug rows")]

    group_counts = map_debug["layer_group"].astype(str).value_counts().to_dict() if "layer_group" in map_debug.columns else {}
    for group in ["Route Matrix Full Route", "Route Matrix Hotels", "Route Matrix City Loop", "Route Matrix Transition"]:
        count = int(group_counts.get(group, 0))
        rows.append(
            _status_row(
                "production_map_route_debug.csv",
                f"{group} rows",
                "PASS" if count > 0 else "FAIL",
                f"rows={count}",
            )
        )

    if "notes" not in map_debug.columns:
        rows.append(_status_row("production_map_route_debug.csv", "route matrix route_key notes", "FAIL", "missing notes column"))
        return rows

    debug_with_keys = map_debug.copy()
    debug_with_keys["debug_route_key"] = debug_with_keys["notes"].fillna("").astype(str).map(_extract_route_key)

    full_rows = debug_with_keys[debug_with_keys["layer_group"].astype(str).eq("Route Matrix Full Route")].copy()
    full_keys = set(full_rows["debug_route_key"].dropna().astype(str).tolist())
    visible_full = full_rows[full_rows.get("show_by_default", pd.Series(dtype=bool)).astype(bool)]
    visible_keys = sorted(set(visible_full["debug_route_key"].dropna().astype(str).tolist()))
    rows.append(
        _status_row(
            "production_map_route_debug.csv",
            "route matrix full-route coverage",
            "PASS" if full_keys == EXPECTED_ROUTE_MATRIX_KEYS else "FAIL",
            f"full_route_keys={len(full_keys)}; missing={sorted(EXPECTED_ROUTE_MATRIX_KEYS - full_keys)[:8]}",
        )
    )
    rows.append(
        _status_row(
            "production_map_route_debug.csv",
            "route matrix default visible route",
            "PASS" if visible_keys == [MATRIX_DEFAULT_ROUTE_KEY] else "FAIL",
            f"visible_keys={visible_keys}",
        )
    )

    hotel_rows = debug_with_keys[debug_with_keys["layer_group"].astype(str).eq("Route Matrix Hotels")].copy()
    hotel_keys = set(hotel_rows["debug_route_key"].dropna().astype(str).tolist())
    visible_hotels = int(hotel_rows.get("show_by_default", pd.Series(dtype=bool)).astype(bool).sum())
    hotel_notes_ok = bool(hotel_rows.get("notes", pd.Series("", index=hotel_rows.index)).fillna("").astype(str).str.contains("hotel_showcase_controlled=true").all()) if not hotel_rows.empty else False
    rows.append(
        _status_row(
            "production_map_route_debug.csv",
            "route matrix hotel showcase layers",
            "PASS" if hotel_keys == EXPECTED_ROUTE_MATRIX_KEYS and visible_hotels == 0 and hotel_notes_ok else "FAIL",
            f"hotel_keys={len(hotel_keys)}; visible_by_default={visible_hotels}; notes_ok={hotel_notes_ok}",
        )
    )

    city_rows = debug_with_keys[debug_with_keys["layer_group"].astype(str).eq("Route Matrix City Loop")].copy()
    city_keys = set(city_rows["debug_route_key"].dropna().astype(str).tolist())
    same_city = bool(city_rows["route_start_city"].astype(str).eq(city_rows["route_end_city"].astype(str)).all()) if not city_rows.empty and {"route_start_city", "route_end_city"}.issubset(city_rows.columns) else False
    notes_ok = bool(city_rows.get("notes", pd.Series("", index=city_rows.index)).fillna("").astype(str).str.contains("hotel_loop_start_end_same=true").all()) if not city_rows.empty else False
    route_type_ok = bool(city_rows.get("route_type", pd.Series("", index=city_rows.index)).astype(str).eq("hotel_loop").all()) if not city_rows.empty else False
    rows.append(
        _status_row(
            "production_map_route_debug.csv",
            "route matrix city loops hotel-to-same-hotel",
            "PASS" if city_keys == EXPECTED_ROUTE_MATRIX_KEYS and same_city and notes_ok and route_type_ok else "FAIL",
            f"city_loop_keys={len(city_keys)}; same_city={same_city}; notes_ok={notes_ok}; route_type_ok={route_type_ok}; missing={sorted(EXPECTED_ROUTE_MATRIX_KEYS - city_keys)[:8]}",
        )
    )

    transition_rows = debug_with_keys[debug_with_keys["layer_group"].astype(str).eq("Route Matrix Transition")].copy()
    transition_keys = set(transition_rows["debug_route_key"].dropna().astype(str).tolist())
    transition_notes_ok = bool(transition_rows.get("notes", pd.Series("", index=transition_rows.index)).fillna("").astype(str).str.contains("transition_route_separate_from_city_loop=true").all()) if not transition_rows.empty else False
    has_intercity = bool(transition_rows["route_start_city"].astype(str).ne(transition_rows["route_end_city"].astype(str)).any()) if not transition_rows.empty and {"route_start_city", "route_end_city"}.issubset(transition_rows.columns) else False
    rows.append(
        _status_row(
            "production_map_route_debug.csv",
            "route matrix transitions separated",
            "PASS" if transition_keys == EXPECTED_ROUTE_MATRIX_KEYS and transition_notes_ok and has_intercity else "FAIL",
            f"transition_keys={len(transition_keys)}; has_intercity={has_intercity}; notes_ok={transition_notes_ok}; missing={sorted(EXPECTED_ROUTE_MATRIX_KEYS - transition_keys)[:8]}",
        )
    )

    road_rows = debug_with_keys[
        debug_with_keys["layer_group"].astype(str).isin(
            ["Route Matrix Full Route", "Route Matrix City Loop", "Route Matrix Transition"]
        )
    ].copy()
    fallback_sum = int(pd.to_numeric(road_rows.get("straight_line_fallback_count", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
    road_percent = pd.to_numeric(road_rows.get("road_geometry_percent", pd.Series(dtype=float)), errors="coerce").dropna()
    geometry_sources = set(road_rows.get("geometry_source", pd.Series(dtype=str)).fillna("").astype(str).tolist())
    road_ok = fallback_sum == 0 and not road_percent.empty and float(road_percent.min()) >= 99.0 and any(
        "osrm" in source or "cached" in source for source in geometry_sources
    )
    rows.append(
        _status_row(
            "production_map_route_debug.csv",
            "route matrix road-aligned geometry",
            "PASS" if road_ok else "FAIL",
            f"fallback_sum={fallback_sum}; min_road_percent={float(road_percent.min()) if not road_percent.empty else 'n/a'}; sources={sorted(geometry_sources)}",
        )
    )
    return rows


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
        "Transition Route",
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


def _check_greedy_registry_endpoints(route_registry: list[dict], method_routes: pd.DataFrame) -> dict:
    greedy_routes = [route for route in route_registry if str(route.get("control_id")) == "method_greedy"]
    if not greedy_routes:
        return _status_row("production_hierarchical_trip_map.html", "greedy route endpoint labels", "FAIL", "method_greedy registry row missing")
    greedy_route = greedy_routes[0]
    greedy_rows = method_routes[method_routes["method"].astype(str).eq("hierarchical_greedy_baseline")].copy() if not method_routes.empty and "method" in method_routes.columns else pd.DataFrame()
    if greedy_rows.empty:
        return _status_row("production_hierarchical_trip_map.html", "greedy route endpoint labels", "FAIL", "greedy route-stop rows missing")
    greedy_rows["day"] = pd.to_numeric(greedy_rows.get("day", 1), errors="coerce").fillna(1).astype(int)
    greedy_rows["stop_order"] = pd.to_numeric(greedy_rows.get("stop_order", 1), errors="coerce").fillna(1).astype(int)
    greedy_rows = greedy_rows.sort_values(["day", "stop_order"]).reset_index(drop=True)
    expected_start = str(greedy_rows.iloc[0].get("route_start_city", ""))
    expected_end = str(greedy_rows.iloc[-1].get("route_end_city", ""))
    found_start = str(greedy_route.get("start_city", ""))
    found_end = str(greedy_route.get("end_city", ""))
    ok = expected_start == found_start and expected_end == found_end
    return _status_row(
        "production_hierarchical_trip_map.html",
        "greedy route endpoint labels",
        "PASS" if ok else "FAIL",
        f"expected={expected_start}->{expected_end}; registry={found_start}->{found_end}",
    )


def _registry_layer_var_for_label(route_registry: list[dict], label: str) -> str | None:
    candidates = {label, label.replace("Full Route · ", "")}
    for route in route_registry:
        values = {
            str(route.get("control_label", "")),
            str(route.get("label", "")),
        }
        if candidates & values:
            return str(route.get("layer_var", "")) or None
    return None


def _registry_route_by_control_id(route_registry: list[dict], control_id: str) -> dict | None:
    for route in route_registry:
        if str(route.get("control_id", "")) == str(control_id):
            return route
    return None


def _check_route_matrix_registry(route_registry: list[dict], html: str) -> list[dict]:
    rows: list[dict] = []
    matrix_routes = [route for route in route_registry if str(route.get("selector_group")) == "route_matrix"]
    full_routes = [route for route in matrix_routes if str(route.get("layer_role")) == "full_route"]
    hotel_routes = [route for route in matrix_routes if str(route.get("layer_role")) == "selected_hotels"]
    city_routes = [route for route in matrix_routes if str(route.get("layer_role")) == "city_loop"]
    transition_routes = [route for route in matrix_routes if str(route.get("layer_role")) == "transition"]
    must_go_routes = [route for route in matrix_routes if str(route.get("layer_role")) == "must_go"]
    full_keys = {str(route.get("route_key")) for route in full_routes}
    hotel_keys = {str(route.get("route_key")) for route in hotel_routes}
    city_keys = {str(route.get("route_key")) for route in city_routes}
    transition_keys = {str(route.get("route_key")) for route in transition_routes}
    default_checked = [str(route.get("control_id", "")) for route in route_registry if bool(route.get("default_checked", route.get("default_visible", False)))]
    rows.append(
        _status_row(
            "production_hierarchical_trip_map.html",
            "route matrix selector registry",
            "PASS" if full_keys == EXPECTED_ROUTE_MATRIX_KEYS and len(full_routes) == 27 else "FAIL",
            f"full_routes={len(full_routes)}; matrix_entries={len(matrix_routes)}; missing={sorted(EXPECTED_ROUTE_MATRIX_KEYS - full_keys)[:8]}",
        )
    )
    rows.append(
        _status_row(
            "production_hierarchical_trip_map.html",
            "route-specific hotel registry",
            "PASS" if hotel_keys == EXPECTED_ROUTE_MATRIX_KEYS and all(bool(route.get("hotel_showcase_controlled")) for route in hotel_routes) else "FAIL",
            f"hotel_routes={len(hotel_routes)}; hotel_keys={len(hotel_keys)}; missing={sorted(EXPECTED_ROUTE_MATRIX_KEYS - hotel_keys)[:8]}",
        )
    )
    rows.append(
        _status_row(
            "production_hierarchical_trip_map.html",
            "route-specific detail registry",
            "PASS" if city_keys == EXPECTED_ROUTE_MATRIX_KEYS and transition_keys == EXPECTED_ROUTE_MATRIX_KEYS else "FAIL",
            f"city_keys={len(city_keys)}; transition_keys={len(transition_keys)}; missing_city={sorted(EXPECTED_ROUTE_MATRIX_KEYS - city_keys)[:8]}; missing_transition={sorted(EXPECTED_ROUTE_MATRIX_KEYS - transition_keys)[:8]}",
        )
    )
    rows.append(
        _status_row(
            "production_hierarchical_trip_map.html",
            "matrix default checked route only",
            "PASS" if default_checked == [f"route_matrix_full__{MATRIX_DEFAULT_ROUTE_KEY}"] else "FAIL",
            f"default_checked={default_checked}",
        )
    )
    selected_hotel_checked = [str(route.get("control_id", "")) for route in hotel_routes if bool(route.get("default_checked", route.get("default_visible", False)))]
    rows.append(
        _status_row(
            "production_hierarchical_trip_map.html",
            "hotel showcase off by default",
            "PASS" if not selected_hotel_checked and "Hotel Showcase: Off" in html and "blueprintHotelShowcaseEnabled = hotelShowcaseEnabled" in html else "FAIL",
            f"default_checked_hotel_layers={selected_hotel_checked}; showcase_markup={'Hotel Showcase: Off' in html}",
        )
    )
    rows.append(
        _status_row(
            "production_hierarchical_trip_map.html",
            "matrix must-go route-specific registry",
            "PASS" if {str(route.get("route_key")) for route in must_go_routes} == EXPECTED_ROUTE_MATRIX_KEYS else "FAIL",
            f"must_go_routes={len(must_go_routes)}",
        )
    )
    default_route = _registry_route_by_control_id(route_registry, f"route_matrix_full__{MATRIX_DEFAULT_ROUTE_KEY}")
    default_layer = str(default_route.get("layer_var", "")) if default_route else None
    rows.append(
        _status_row(
            "production_hierarchical_trip_map.html",
            "matrix default route visible addTo",
            "PASS" if _feature_group_added_to_map(html, default_layer) and not _feature_group_removed_from_map(html, default_layer) else "FAIL",
            f"feature_group={default_layer or 'missing'}; addTo={_feature_group_added_to_map(html, default_layer)}; remove_call={_feature_group_removed_from_map(html, default_layer)}",
        )
    )
    non_default_visible = []
    for route in full_routes:
        if str(route.get("route_key")) == MATRIX_DEFAULT_ROUTE_KEY:
            continue
        layer_var = str(route.get("layer_var", ""))
        if _feature_group_added_to_map(html, layer_var) and not _feature_group_removed_from_map(html, layer_var):
            non_default_visible.append(str(route.get("control_id", "")))
    rows.append(
        _status_row(
            "production_hierarchical_trip_map.html",
            "matrix non-default routes hidden",
            "PASS" if not non_default_visible else "FAIL",
            f"visible_non_default={non_default_visible[:8]}",
        )
    )
    return rows


def _html_or_registry_contains_label(html: str, route_registry: list[dict], label: str) -> bool:
    return _html_contains(html, label) or bool(_registry_layer_var_for_label(route_registry, label))


def _check_hidden_by_default_html(html: str, label: str, route_registry: list[dict] | None = None) -> dict:
    feature_group = _feature_group_for_label(html, label)
    if not feature_group and route_registry is not None:
        feature_group = _registry_layer_var_for_label(route_registry, label)
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
    route_matrix = _load_csv(output_dir / "production_route_matrix_comparison.csv")
    route_matrix_stops = _load_csv(output_dir / "production_route_matrix_route_stops.csv")
    route_matrix_hotels = _load_csv(output_dir / "production_route_matrix_hotel_selection_debug.csv")
    profile_routes = _load_csv(output_dir / "production_day_plan_profiles.csv")
    map_debug = _load_csv(output_dir / "production_map_route_debug.csv")
    must_go_coverage = _load_csv(output_dir / "production_must_go_coverage.csv")
    hotel_debug = _load_csv(output_dir / "production_hotel_selection_debug.csv")

    matrix_mode = not route_matrix.empty and not route_matrix_stops.empty
    rows.extend(_check_route_matrix_artifacts(route_matrix, route_matrix_stops, route_matrix_hotels))
    rows.extend(_check_route_file(route_matrix_stops, "production_route_matrix_route_stops.csv"))

    rows.extend(_check_route_file(trip_routes, "production_trip_length_route_stops.csv"))
    rows.extend(_check_expected_values(trip_routes, "production_trip_length_route_stops.csv", "trip_days", EXPECTED_TRIP_DAYS))
    rows.extend(_check_trip_length_uses_hybrid_bandit(trip_routes))

    rows.extend(_check_route_file(method_routes, "production_method_route_stops.csv"))
    rows.extend(_check_expected_values(method_routes, "production_method_route_stops.csv", "method", EXPECTED_METHODS))
    rows.extend(_check_method_summary_file(method_summary))
    rows.extend(_check_must_go_coverage_file(must_go_coverage))
    rows.extend(_check_hotel_debug_file(hotel_debug))

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
        if matrix_mode:
            rows.extend(_check_route_matrix_debug(map_debug))
            rows.append(_check_visible_group(map_debug, "Route Matrix Full Route", 1))
            rows.append(_check_distance_coverage(map_debug, "Route Matrix Full Route", 400.0))
            rows.append(_check_distance_coverage(map_debug, "Route Matrix City Loop", 20.0))
            rows.append(_check_distance_coverage(map_debug, "Route Matrix Transition", 80.0))
            rows.append(_check_distance_coverage(map_debug, "Routes", 700.0))
        else:
            for expected_group in ["Full Route Overview", "City Detail", "Transition Route", "Trip Length Comparison", "Method Comparison", "Traveler Comparison"]:
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
            rows.append(_check_distance_coverage(map_debug, "Transition Route", 80.0))
            rows.append(_check_distance_coverage(map_debug, "Routes", 700.0))
            rows.append(_check_distance_coverage(map_debug, "Selected Result", 400.0))
            rows.append(_check_full_route_offsets(map_debug))
            rows.append(_check_full_route_families(map_debug))
            rows.append(_check_full_route_arrows(map_debug))
            rows.extend(_check_city_loop_semantics(map_debug))
        rows.extend(_check_road_geometry_report(map_debug))
        if not matrix_mode:
            rows.extend(_check_balanced_only_defaults(map_debug))

    html_path = figure_dir / "production_hierarchical_trip_map.html"
    html = html_path.read_text(encoding="utf-8", errors="ignore") if html_path.exists() else ""
    route_registry = _extract_route_registry(html)
    rows.append(
        _status_row(
            "production_hierarchical_trip_map.html",
            "html exists",
            "PASS" if html else "FAIL",
            f"bytes={html_path.stat().st_size}" if html_path.exists() else "missing",
        )
    )
    if matrix_mode:
        expected_html_labels = [
            "7-Day · Bandit + Small Gurobi · Balanced",
            "9-Day · Bandit + Small Gurobi · Balanced",
            "12-Day · Bandit + Small Gurobi · Balanced",
            "7-Day Routes",
            "9-Day Routes",
            "12-Day Routes",
            "Gurobi",
            "Greedy",
            "Bandit + Small Gurobi",
            "City loops",
            "Transitions",
            "Must-go coverage",
            "Hotel candidates",
            "Must-go candidates",
        ]
    else:
        expected_html_labels = [
            "Full Route · Balanced full route",
            "Full Route · Trip Length · 7-Day Hybrid Bandit + Small Gurobi",
            "Full Route · Trip Length · 9-Day Hybrid Bandit + Small Gurobi",
            "Full Route · Trip Length · 12-Day Hybrid Bandit + Small Gurobi",
            "Full Route · Method · Hierarchical Gurobi Pipeline",
            "Full Route · Method · Hierarchical Greedy Baseline",
            "Full Route · Method · Hierarchical + Bandit + Small Gurobi Repair",
            "Full Route · Traveler: Relaxed",
            "Full Route · Traveler: Explorer",
            "City Loop · San Francisco",
            "Transition Route · Day",
            "Trip Length · 7-Day Hybrid Bandit + Small Gurobi",
            "Trip Length · 9-Day Hybrid Bandit + Small Gurobi",
            "Trip Length · 12-Day Hybrid Bandit + Small Gurobi",
            "Method · Hierarchical Gurobi Pipeline",
            "Method · Hierarchical Greedy Baseline",
            "Method · Hierarchical + Bandit + Small Gurobi Repair",
            "Traveler · Relaxed Full Route",
            "Traveler · Balanced Full Route",
            "Traveler · Explorer Full Route",
            "Selected hotel/base pins",
            "Hotel candidates",
            "Must-go candidates",
        ]
    for label in expected_html_labels:
        found = _html_or_registry_contains_label(html, route_registry, label)
        rows.append(
            _status_row(
                "production_hierarchical_trip_map.html",
                f"html layer label: {label}",
                "PASS" if found else "FAIL",
                "present in html or route selector registry" if found else "missing",
            )
        )
    if matrix_mode:
        rows.extend(_check_route_matrix_registry(route_registry, html))
        for label in ["Fastest inter-city route", "Scenic CA-1 / PCH route", "Hotel candidates", "Must-go candidates"]:
            rows.append(_check_hidden_by_default_html(html, label, route_registry))
    else:
        balanced_label = "Full Route · Balanced full route"
        feature_group = _feature_group_for_label(html, balanced_label)
        if not feature_group:
            feature_group = _registry_layer_var_for_label(route_registry, balanced_label)
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
        hotel_label = "Selected hotel/base pins"
        hotel_feature_group = _feature_group_for_label(html, hotel_label)
        if not hotel_feature_group:
            hotel_feature_group = _registry_layer_var_for_label(route_registry, hotel_label)
        hotel_added = _feature_group_added_to_map(html, hotel_feature_group)
        hotel_removed = _feature_group_removed_from_map(html, hotel_feature_group)
        rows.append(
            _status_row(
                "production_hierarchical_trip_map.html",
                f"html selected hotels visible: {hotel_label}",
                "PASS" if hotel_added and not hotel_removed else "FAIL",
                f"feature_group={hotel_feature_group or 'missing'}; addTo={hotel_added}; remove_call={hotel_removed}",
            )
        )
        for label in [
            "Fastest inter-city route",
            "Scenic CA-1 / PCH route",
            "Full Route · Trip Length · 7-Day Hybrid Bandit + Small Gurobi",
            "Full Route · Trip Length · 9-Day Hybrid Bandit + Small Gurobi",
            "Full Route · Trip Length · 12-Day Hybrid Bandit + Small Gurobi",
            "Full Route · Method · Hierarchical Gurobi Pipeline",
            "Full Route · Method · Hierarchical Greedy Baseline",
            "Full Route · Method · Hierarchical + Bandit + Small Gurobi Repair",
            "Full Route · Traveler: Relaxed",
            "Full Route · Traveler: Explorer",
            "City Loop · San Francisco",
            "Hotel candidates",
            "Must-go candidates",
        ]:
            rows.append(_check_hidden_by_default_html(html, label, route_registry))
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
            "PASS" if all(token in html for token in ["blueprint-route-selector", "data-route-checkbox", "syncCheckboxState", "Balanced default", "Show city loops", "Show transitions", "Hotel Showcase: Off"]) else "FAIL",
            "route selector checkbox controls present" if "blueprint-route-selector" in html else "missing route selector",
        )
    )
    rows.append(
        _status_row(
            "production_hierarchical_trip_map.html",
            "html child-layer scoped quick actions",
            "PASS" if all(token in html for token in ["setChildRoleForCheckedRoutes", "parentRouteChecked(route)", 'setChildRoleForCheckedRoutes("city_loop")']) and "String(route.profile) === \"balanced\" || routeHasQuickGroup(route, \"city_detail\")" not in html else "FAIL",
            "city loops/transitions are toggled only for currently checked parent routes",
        )
    )
    rows.append(
        _status_row(
            "production_hierarchical_trip_map.html",
            "html no PolyLineTextPath under canvas",
            "PASS" if "PolyLineTextPath(" not in html and ".setText(" not in html else "FAIL",
            f"PolyLineTextPath={html.count('PolyLineTextPath(')}; setText={html.count('.setText(')}; preferCanvas={'preferCanvas' in html}",
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
            "PASS" if all(token in html for token in ["blueprint-panel-header", "blueprint-collapsed", 'data-panel-id="dashboard"', 'data-panel-id="route-selector"', 'data-panel-id="route-debug"']) and 'data-panel-id="legend"' not in html else "FAIL",
            "dashboard, selector, and route debug have collapsible panel markup; old legend panel absent",
        )
    )
    rows.append(
        _status_row(
            "production_hierarchical_trip_map.html",
            "html single integrated route selector",
            "PASS" if "Blueprint Map Layers" not in html and "L.control.groupedLayers" not in html else "FAIL",
            f"blueprint_legend={'Blueprint Map Layers' in html}; grouped_control={'L.control.groupedLayers' in html}",
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
    registry_families = set(str(route.get("family", "")) for route in route_registry)
    selector_groups = set(str(route.get("selector_group", route.get("family", ""))) for route in route_registry)
    default_checked = [str(route.get("control_id", "")) for route in route_registry if bool(route.get("default_checked", route.get("default_visible", False)))]
    if matrix_mode:
        rows.append(
            _status_row(
                "production_hierarchical_trip_map.html",
                "html route selector registry",
                "PASS" if len(route_registry) >= 110 and {"route_matrix", "city_detail", "transition", "context", "must_go", "hotel"}.issubset(registry_families) and {"route_matrix", "hotel", "must_go", "context"}.issubset(selector_groups) else "FAIL",
                f"routes={len(route_registry)}; families={sorted(registry_families)}; selector_groups={sorted(selector_groups)}",
            )
        )
        rows.append(
            _status_row(
                "production_hierarchical_trip_map.html",
                "html matrix default checked",
                "PASS" if default_checked == [f"route_matrix_full__{MATRIX_DEFAULT_ROUTE_KEY}"] else "FAIL",
                f"default_checked={default_checked}",
            )
        )
    else:
        rows.append(
            _status_row(
                "production_hierarchical_trip_map.html",
                "html route selector registry",
                "PASS" if len(route_registry) >= 16 and {"trip_length", "method", "traveler_profile", "city_detail", "transition", "context", "must_go", "hotel"}.issubset(registry_families) and {"core", "trip_length", "method", "city_detail", "transition", "must_go", "hotel", "context"}.issubset(selector_groups) else "FAIL",
                f"routes={len(route_registry)}; families={sorted(registry_families)}; selector_groups={sorted(selector_groups)}",
            )
        )
        rows.append(_check_greedy_registry_endpoints(route_registry, method_routes))
        rows.append(
            _status_row(
                "production_hierarchical_trip_map.html",
                "html balanced-only default checked",
                "PASS" if default_checked == ["hotel_selected", "traveler_balanced"] or default_checked == ["traveler_balanced", "hotel_selected"] else "FAIL",
                f"default_checked={default_checked}",
            )
        )
    rows.append(
        _status_row(
            "production_hierarchical_trip_map.html",
            "html route arrows",
            "PASS" if html.count("blueprint-route-arrow") >= 10 and (not map_debug.empty and map_debug.get("geometry_mode", pd.Series(dtype=str)).fillna("").astype(str).str.contains("canvas-arrow-markers").any()) else "FAIL",
            f"canvas_arrow_class={html.count('blueprint-route-arrow')}; debug_canvas_arrow_rows={int(map_debug.get('geometry_mode', pd.Series(dtype=str)).fillna('').astype(str).str.contains('canvas-arrow-markers').sum()) if not map_debug.empty else 0}; setText={html.count('.setText(')}",
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
            "PASS: route data, checkbox registry, balanced-only first-open state, canvas-safe direction arrows, road-geometry diagnostics, must-go/hotel artifacts, city-detail layers, collapsible/draggable panels, and route panes all validate. "
            "The map is now controlled by selectable route layers without PolyLineTextPath under canvas rendering."
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
