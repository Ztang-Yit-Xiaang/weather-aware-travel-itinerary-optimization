"""Validate statewide nature itinerary artifacts.

This script is intentionally read-only.  It reports whether the active config,
candidate catalog, selected route stops, and exported dashboard metadata agree.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from itinerary_system.artifact_metadata import artifact_metadata_matches, read_artifact_metadata
from itinerary_system.config import load_trip_config

REQUIRED_ANCHORS = {
    "San Francisco": ["san francisco", "golden gate", "stanford"],
    "Los Angeles": ["los angeles", "hollywood", "griffith", "tcl chinese"],
    "Monterey": ["monterey", "bixby creek", "17-mile"],
    "Big Sur": ["big sur", "bixby creek"],
    "Santa Barbara": ["santa barbara", "stearns wharf"],
    "Yosemite": ["yosemite"],
    "Sequoia": ["sequoia"],
    "Kings Canyon": ["kings canyon"],
    "Joshua Tree": ["joshua tree"],
}
OPTIONAL_ANCHORS = {"Bixby Creek": ["bixby creek"]}
ANCHORS = {**REQUIRED_ANCHORS, **OPTIONAL_ANCHORS}
ALLOWED_ROLES = {
    "selected_stop",
    "gateway",
    "base_city",
    "candidate_only",
    "context_only",
    "missing",
    "infeasible",
}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _contains_anchor(frame: pd.DataFrame, anchor: str, columns: list[str]) -> bool:
    if frame.empty:
        return False
    text = pd.Series("", index=frame.index)
    for column in columns:
        if column in frame.columns:
            text = text + " " + frame[column].fillna("").astype(str)
    lowered = text.str.lower()
    return bool(lowered.str.contains("|".join(ANCHORS[anchor]), regex=True, na=False).any())


def _truthy(value, default: bool = False) -> bool:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _selected_method(route_stops: pd.DataFrame, config) -> str:
    if route_stops.empty or "method" not in route_stops.columns:
        return ""
    configured_days = int(config.get("trip", "trip_days", 7))
    frame = route_stops.copy()
    if "trip_days" in frame.columns:
        filtered = frame[pd.to_numeric(frame["trip_days"], errors="coerce").fillna(-1).astype(int).eq(configured_days)]
        if not filtered.empty:
            frame = filtered
    preferred = frame[frame["method"].astype(str).eq("hierarchical_bandit_gurobi_repair")]
    if not preferred.empty:
        return "hierarchical_bandit_gurobi_repair"
    values = frame["method"].dropna().astype(str)
    return values.iloc[0] if not values.empty else ""


def _required_selected_anchors(config) -> set[str]:
    scenario = str(config.get("trip", "scenario", "")).lower()
    interest = str(config.get("interest", "mode", config.get("trip", "interest_profile", ""))).lower()
    if scenario != "california_statewide_nature" or interest != "nature_heavy":
        return set()
    raw = config.get("nature", "required_selected_anchors_for_nature_heavy", [])
    if isinstance(raw, str):
        raw = [part.strip() for part in raw.split(",") if part.strip()]
    if not isinstance(raw, list):
        return set()
    return {str(anchor).strip() for anchor in raw if str(anchor).strip()}


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate statewide nature route artifacts.")
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "nature_trip_config.yaml"))
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "results" / "outputs"))
    parser.add_argument("--dashboard-assets", default=str(REPO_ROOT / "results" / "figures" / "full_interactive_dashboard" / "assets"))
    parser.add_argument("--strict", action="store_true", help="Exit non-zero when acceptance checks fail.")
    args = parser.parse_args()

    config = load_trip_config(args.config)
    output_dir = Path(args.output_dir)
    assets_dir = Path(args.dashboard_assets)

    enriched = _read_csv(output_dir / "production_enriched_poi_catalog.csv")
    route_stops = _read_csv(output_dir / "production_method_route_stops.csv")
    anchor_audit = _read_csv(output_dir / "production_dataset_completeness_audit.csv")
    if anchor_audit.empty:
        anchor_audit = _read_csv(output_dir / "production_route_anchor_audit.csv")
    metadata = read_artifact_metadata(output_dir)
    metadata_fresh = artifact_metadata_matches(output_dir, config)
    method = _selected_method(route_stops, config)
    selected = route_stops[route_stops.get("method", pd.Series(dtype=str)).astype(str).eq(method)].copy() if method else pd.DataFrame()

    metrics_path = assets_dir / "dashboard_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}

    print(f"active_scenario: {config.get('trip', 'scenario')}")
    print(f"active_interest: {config.get('interest', 'mode')}")
    print(f"trip_days: {config.get('trip', 'trip_days')}")
    print(f"route_method: {method or 'missing'}")
    print(f"artifact_metadata_fresh: {metadata_fresh}")
    print(f"artifact_metadata_scenario: {metadata.get('scenario', 'missing')}")
    print(f"artifact_metadata_interest: {metadata.get('interest_mode', 'missing')}")
    print(f"artifact_metadata_timestamp_utc: {metadata.get('timestamp_utc', 'missing')}")

    if not enriched.empty:
        group_cols = [col for col in ["city", "nature_region", "category", "source_list"] if col in enriched.columns]
        print("candidate_counts_by_city_anchor_category_source:")
        print(enriched.groupby(group_cols, dropna=False).size().reset_index(name="count").to_string(index=False))
    else:
        print("candidate_counts_by_city_anchor_category_source: missing")

    if not selected.empty:
        group_cols = [col for col in ["city", "overnight_city", "category"] if col in selected.columns]
        print("selected_stop_counts_by_city_anchor_category:")
        print(selected.groupby(group_cols, dropna=False).size().reset_index(name="count").to_string(index=False))
    else:
        print("selected_stop_counts_by_city_anchor_category: missing")

    selected_nature = (
        int(selected.get("is_nature", pd.Series(dtype=bool)).fillna(False).astype(bool).sum())
        if not selected.empty
        else 0
    )
    print(f"selected_nature_stop_count: {selected_nature}")

    failures = []
    required_selected = _required_selected_anchors(config)
    for anchor in REQUIRED_ANCHORS:
        candidate = _contains_anchor(enriched, anchor, ["name", "city", "nature_region", "category"])
        chosen = _contains_anchor(selected, anchor, ["attraction_name", "name", "city", "overnight_city", "nature_region"])
        audit_row = (
            anchor_audit[anchor_audit.get("anchor", pd.Series(dtype=str)).astype(str).eq(anchor)]
            if not anchor_audit.empty and "anchor" in anchor_audit.columns
            else pd.DataFrame()
        )
        if not audit_row.empty:
            candidate = _truthy(audit_row.iloc[0].get("candidate"), candidate)
            chosen = _truthy(audit_row.iloc[0].get("selected"), chosen)
        role = str(audit_row.iloc[0].get("role", audit_row.iloc[0].get("status", ""))) if not audit_row.empty else ""
        reason = str(audit_row.iloc[0].get("reason", "")) if not audit_row.empty else ""
        print(f"anchor_status.{anchor}.candidate: {candidate}")
        print(f"anchor_status.{anchor}.selected: {chosen}")
        print(f"anchor_status.{anchor}.role: {role or 'missing_audit_row'}")
        print(f"anchor_status.{anchor}.reason: {reason or 'missing_reason'}")
        if anchor in {"Yosemite", "Sequoia"} and not candidate:
            failures.append(f"{anchor} is not an optimizer candidate")
        if anchor in required_selected and not chosen:
            failures.append(f"{anchor} is configured as a required nature-heavy anchor but is not selected")
        if not audit_row.empty and role not in ALLOWED_ROLES:
            failures.append(f"{anchor} has invalid audit role: {role}")
        if audit_row.empty or not reason:
            failures.append(f"{anchor} is missing an audit role/reason")
        if role == "missing":
            failures.append(f"{anchor} is missing from candidate/route/context artifacts")

    la_selected = _contains_anchor(selected, "Los Angeles", ["attraction_name", "name", "city", "overnight_city"])
    la_gateway = False
    if not selected.empty:
        for column in ["gateway_start", "gateway_end", "overnight_city", "route_start_city", "route_end_city"]:
            if column in selected.columns:
                la_gateway = la_gateway or selected[column].fillna("").astype(str).str.contains("Los Angeles", case=False).any()
    if not anchor_audit.empty and "anchor" in anchor_audit.columns:
        la_rows = anchor_audit[anchor_audit["anchor"].astype(str).eq("Los Angeles")]
        if not la_rows.empty:
            la_selected = _truthy(la_rows.iloc[0].get("selected"), la_selected)
            la_role = str(la_rows.iloc[0].get("role", la_rows.iloc[0].get("status", "")))
            la_gateway = la_gateway or la_role in {"gateway", "base_city"}
            la_gateway = _truthy(la_rows.iloc[0].get("gateway_or_base"), la_gateway)
    print(f"la_selected_or_gateway: {bool(la_selected or la_gateway)}")

    ui_matches = True
    if metrics:
        ui_matches = (
            str(metrics.get("scenario")) == str(config.get("trip", "scenario"))
            and int(metrics.get("trip_days", -1)) == int(config.get("trip", "trip_days", 7))
            and (not method or str(metrics.get("default_route_method", method)) == method)
        )
    print(f"ui_default_route_matches_selected_data: {ui_matches}")

    if not metadata_fresh:
        failures.append("artifact metadata is stale or missing")
    if not bool(la_selected or la_gateway):
        failures.append("Los Angeles is neither selected nor used as gateway/base")
    if not ui_matches:
        failures.append("UI default route metrics do not match selected route data")

    if failures:
        print("validation_failures:")
        for failure in failures:
            print(f"- {failure}")
    else:
        print("validation_failures: none")

    return 1 if failures and args.strict else 0


if __name__ == "__main__":
    raise SystemExit(main())
