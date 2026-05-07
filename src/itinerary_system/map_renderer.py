"""Map rendering compatibility wrapper.

This wrapper protects the production hierarchical California map from stale
local 3-day Gurobi outputs in notebook memory, prepares comparison dashboard
artifacts, calls the legacy blueprint map renderer, then re-saves the HTML with
focused initial bounds.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from ._legacy import import_legacy_module
from .config import TripConfig
from .experiment_runner import prepare_comparison_dashboard_outputs


REQUIRED_METHODS = {
    "hierarchical_gurobi_pipeline",
    "hierarchical_greedy_baseline",
    "hierarchical_bandit_gurobi_repair",
}

REQUIRED_TRIP_LENGTHS = {7, 9, 12}


def _config_get(config: TripConfig, *keys: Any, default=None):
    """Safely read TripConfig values.

    Most project calls use config.get(section, key, default), but this helper
    avoids breaking if the config object behaves slightly differently.
    """
    try:
        return config.get(*keys, default)
    except TypeError:
        try:
            return config.get(*keys)
        except Exception:
            return default
    except Exception:
        return default


def _safe_int(value, default=None):
    try:
        if value is None or pd.isna(value):
            return default
        return int(float(value))
    except Exception:
        return default


def _safe_float(value, default=None):
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _canonical_trip_days(context: dict, config: TripConfig) -> int:
    """Resolve one production trip length for the dashboard."""
    if context.get("CANONICAL_TRIP_DAYS") is not None:
        return int(context["CANONICAL_TRIP_DAYS"])

    trip = context.get("best_hierarchical_trip", {})
    days_by_city = trip.get("days_by_city", {}) if isinstance(trip, dict) else {}
    if isinstance(days_by_city, dict) and days_by_city:
        return int(sum(int(v) for v in days_by_city.values()))

    return int(_config_get(config, "trip", "trip_days", default=7))


def _canonical_total_budget(context: dict, config: TripConfig, output_dir: Path) -> float:
    """Resolve one production budget for all method-comparison rows."""
    if context.get("CANONICAL_TOTAL_BUDGET") is not None:
        return float(context["CANONICAL_TOTAL_BUDGET"])

    budget_path = Path(output_dir) / "production_budget_estimate.csv"
    if budget_path.exists():
        try:
            budget_df = pd.read_csv(budget_path)
            for component in ["buffered_total", "soft_budget", "hard_budget"]:
                rows = budget_df[budget_df["component"].astype(str).eq(component)]
                if not rows.empty:
                    value = _safe_float(rows["expected"].iloc[0])
                    if value is not None:
                        return float(value)
        except Exception:
            pass

    return float(_config_get(config, "trip", "budget", default=2201.64))


def _default_output_dirs(context: dict, output_path: str | Path | None) -> dict:
    """Make sure OUTPUT_DIR and FIGURE_DIR always exist in the context."""
    merged = dict(context)

    if output_path is not None:
        output_path = Path(output_path)
        merged.setdefault("FIGURE_DIR", output_path.parent)
        merged.setdefault("OUTPUT_DIR", output_path.parent.parent / "outputs")

    merged.setdefault("OUTPUT_DIR", Path("results/outputs"))
    merged.setdefault("FIGURE_DIR", Path("results/figures"))

    merged["OUTPUT_DIR"] = Path(merged["OUTPUT_DIR"])
    merged["FIGURE_DIR"] = Path(merged["FIGURE_DIR"])

    return merged


def _sanitize_production_map_context(
    context: dict,
    config: TripConfig,
    output_path: str | Path | None = None,
) -> dict:
    """Remove stale local optimizer state and enforce canonical production settings.

    The notebook may still contain old local Gurobi outputs:
      - optimizer_outputs with K=3
      - route_details
      - routes
      - baseline payloads

    Those should not feed the 7-day hierarchical California dashboard.
    """
    merged_context = _default_output_dirs(context, output_path)
    output_dir = Path(merged_context["OUTPUT_DIR"])

    canonical_days = _canonical_trip_days(merged_context, config)
    canonical_budget = _canonical_total_budget(merged_context, config, output_dir)
    canonical_daily_budget = float(merged_context.get("CANONICAL_DAILY_TIME_BUDGET", 720.0))

    old_optimizer = merged_context.get("optimizer_outputs")
    old_k = _safe_int(old_optimizer.get("K")) if isinstance(old_optimizer, dict) else None

    # Production map should exclude the local demo by default.
    use_local_demo = bool(merged_context.get("USE_LOCAL_GUROBI_DEMO_IN_PRODUCTION_MAP", False))
    stale_k_conflict = old_k is not None and old_k != int(canonical_days)
    exclude_local = (not use_local_demo) or stale_k_conflict

    if exclude_local:
        if old_optimizer is not None:
            merged_context["LOCAL_GUROBI_DEMO_OUTPUTS"] = old_optimizer

        for stale_key in [
            "optimizer_outputs",
            "full_optimizer_outputs",
            "demo_optimizer_outputs",
            "routes",
            "route_details",
            "edges",
            "baseline_payload",
            "baseline_comparison_df",
            "adaptive_outputs",
            "adaptive_summary",
            "K",
            "n",
            "num_hotels",
        ]:
            merged_context.pop(stale_key, None)

    merged_context["USE_LOCAL_GUROBI_DEMO_IN_PRODUCTION_MAP"] = False
    merged_context["CANONICAL_TRIP_DAYS"] = int(canonical_days)
    merged_context["CANONICAL_TOTAL_BUDGET"] = float(canonical_budget)
    merged_context["CANONICAL_DAILY_TIME_BUDGET"] = float(canonical_daily_budget)
    merged_context["RUN_LIVE_APIS"] = bool(
        _config_get(config, "enrichment", "run_live_apis", default=False)
    )
    # The notebook may carry older "debug all routes" flags in memory.
    # Force the clean first-open contract: only the Balanced full route is
    # checked, with every comparison/detail route available in Route Selector.
    merged_context["MAP_ROUTE_ONLY_DEBUG_VIEW"] = False
    merged_context["MAP_BALANCED_ONLY_DEFAULT_VIEW"] = bool(
        _config_get(config, "map", "balanced_only_default_view", default=True)
    )
    merged_context["MAP_REFRESH_ROAD_GEOMETRY"] = bool(
        _config_get(config, "map", "refresh_road_geometry", default=False)
    )
    merged_context["MAP_SHOW_DEBUG_ON_LOAD"] = bool(
        _config_get(config, "map", "show_debug_on_load", default=False)
    )
    merged_context["SHOW_FULL_SCENE_DEFAULT"] = False
    merged_context["SHOW_CONTEXT_ROUTES_BY_DEFAULT"] = False
    merged_context["SHOW_COMPARISON_LAYERS_BY_DEFAULT"] = False
    merged_context["SHOW_TRAVELER_OVERVIEWS_BY_DEFAULT"] = False
    merged_context["SHOW_SELECTED_RESULT_BY_DEFAULT"] = False

    merged_context["MODEL_SETTINGS"] = {
        **merged_context.get("MODEL_SETTINGS", {}),
        "num_days": int(canonical_days),
        "total_budget": float(canonical_budget),
        "daily_time_budget": float(canonical_daily_budget),
    }

    merged_context["SOLVER_SETTINGS"] = {
        **merged_context.get("SOLVER_SETTINGS", {}),
        "total_budget": float(canonical_budget),
        "daily_time_budget": float(canonical_daily_budget),
    }

    return merged_context


def _valid_route_stop_file(path: str | Path, required_methods=None, required_days=None, canonical_days=None,) -> bool:
    """Validate route-stop CSVs before deciding to keep existing artifacts."""
    path = Path(path)
    if not path.exists():
        return False

    try:
        df = pd.read_csv(path)
    except Exception:
        return False

    if df.empty:
        return False

    required_cols = {
        "method",
        "day",
        "stop_order",
        "latitude",
        "longitude",
        "attraction_name",
        "route_start_city",
        "route_end_city",
        "pass_through_cities",
        "drive_minutes_to_next_base",
        "available_visit_minutes",
        "drive_time_source",
    }
    if not required_cols.issubset(df.columns):
        return False

    for coord_col in [
        "latitude",
        "longitude",
        "route_start_latitude",
        "route_start_longitude",
        "route_end_latitude",
        "route_end_longitude",
    ]:
        if coord_col in df.columns and pd.to_numeric(df[coord_col], errors="coerce").dropna().empty:
            return False

    if {"route_start_city", "route_end_city", "drive_minutes_to_next_base"}.issubset(df.columns):
        relocation = df["route_start_city"].astype(str).ne(df["route_end_city"].astype(str))
        if relocation.any():
            drive_minutes = pd.to_numeric(df.loc[relocation, "drive_minutes_to_next_base"], errors="coerce")
            if drive_minutes.dropna().empty or not drive_minutes.gt(0).any():
                return False

    if required_methods is not None:
        found_methods = set(df["method"].astype(str).unique())
        if not set(required_methods).issubset(found_methods):
            return False

    if required_days is not None:
        if "trip_days" not in df.columns:
            return False
        found_days = set(
            pd.to_numeric(df["trip_days"], errors="coerce")
            .dropna()
            .astype(int)
            .unique()
        )
        if not set(required_days).issubset(found_days):
            return False
    if canonical_days is not None and "trip_days" in df.columns:
        found_days = set(
            pd.to_numeric(df["trip_days"], errors="coerce")
            .dropna()
            .astype(int)
            .unique()
        )
        if found_days and found_days != {int(canonical_days)}:
            return False
    return True


def _valid_method_summary_file(
    path: str | Path,
    canonical_days: int | None = None,
    canonical_budget: float | None = None,
) -> bool:
    """Validate method summary consistency.

    This prevents the renderer from keeping stale method-comparison tables with
    old 3-day or $1000 local Gurobi values.
    """
    path = Path(path)
    if not path.exists():
        return False

    try:
        df = pd.read_csv(path)
    except Exception:
        return False

    if df.empty or "method" not in df.columns:
        return False

    found_methods = set(df["method"].astype(str).unique())
    if not REQUIRED_METHODS.issubset(found_methods):
        return False

    if canonical_days is not None and "trip_days" in df.columns:
        found_days = set(
            pd.to_numeric(df["trip_days"], errors="coerce")
            .dropna()
            .astype(int)
            .unique()
        )
        if found_days and found_days != {int(canonical_days)}:
            return False

    if canonical_budget is not None and "total_budget" in df.columns:
        budgets = (
            pd.to_numeric(df["total_budget"], errors="coerce")
            .dropna()
            .round(2)
            .unique()
        )
        if len(budgets) != 1:
            return False
        if abs(float(budgets[0]) - float(canonical_budget)) > 1.0:
            return False

    return True


def _valid_trip_length_summary_file(path: str | Path, required_days=None) -> bool:
    path = Path(path)
    if not path.exists():
        return False

    try:
        df = pd.read_csv(path)
    except Exception:
        return False

    if df.empty or "trip_days" not in df.columns:
        return False

    if required_days is not None:
        found_days = set(
            pd.to_numeric(df["trip_days"], errors="coerce")
            .dropna()
            .astype(int)
            .unique()
        )
        if not set(required_days).issubset(found_days):
            return False

    return True


def _dashboard_artifacts_ready(
    output_dir: str | Path,
    canonical_days: int | None = None,
    canonical_budget: float | None = None,
) -> bool:
    output_dir = Path(output_dir)

    return (
        _valid_route_stop_file(
            output_dir / "production_method_route_stops.csv",
            required_methods=REQUIRED_METHODS,
            canonical_days=canonical_days,
        )
        and _valid_route_stop_file(
            output_dir / "production_trip_length_route_stops.csv",
            required_days=REQUIRED_TRIP_LENGTHS,
        )
        and _valid_method_summary_file(
            output_dir / "production_method_comparison.csv",
            canonical_days=canonical_days,
            canonical_budget=canonical_budget,
        )
        and _valid_trip_length_summary_file(
            output_dir / "production_trip_length_comparison.csv",
            required_days=REQUIRED_TRIP_LENGTHS,
        )
    )


def prepare_map_dashboard_data(
    context: dict,
    config: TripConfig,
    output_path: str | Path | None = None,
):
    """Prepare or reuse production dashboard CSV artifacts."""
    merged_context = _sanitize_production_map_context(context, config, output_path)
    output_dir = Path(merged_context["OUTPUT_DIR"])

    force_rebuild = bool(merged_context.get("FORCE_REBUILD_MAP_DASHBOARD_DATA", False))

    if not force_rebuild and _dashboard_artifacts_ready(
        output_dir,
        canonical_days=merged_context.get("CANONICAL_TRIP_DAYS"),
        canonical_budget=merged_context.get("CANONICAL_TOTAL_BUDGET"),
    ):
        print("Keeping existing production comparison dashboard artifacts.")
        return {
            "status": "kept_existing",
            "output_dir": str(output_dir),
            "canonical_trip_days": merged_context.get("CANONICAL_TRIP_DAYS"),
            "canonical_total_budget": merged_context.get("CANONICAL_TOTAL_BUDGET"),
        }

    print("Refreshing canonical production comparison artifacts through experiment_runner...")
    return prepare_comparison_dashboard_outputs(context=merged_context, config=config)


def _initial_focus_points(day_plan_df: pd.DataFrame, focus_mode: str = "full_trip") -> list[list[float]]:
    """Return only points that should control initial map zoom.

    Do not include scenic route, fastest route, social candidates, or candidate
    hotels here; those layers make the HTML open too zoomed out.
    """
    if day_plan_df is None or day_plan_df.empty:
        return []

    focus_df = day_plan_df.copy()

    if focus_mode == "first_day" and "day" in focus_df.columns:
        day_values = pd.to_numeric(focus_df["day"], errors="coerce").dropna()
        if not day_values.empty:
            first_day = int(day_values.min())
            focus_df = focus_df[pd.to_numeric(focus_df["day"], errors="coerce").eq(first_day)]

    points: list[list[float]] = []
    for lat_col, lon_col in [
        ("route_start_latitude", "route_start_longitude"),
        ("latitude", "longitude"),
        ("route_end_latitude", "route_end_longitude"),
        ("hotel_latitude", "hotel_longitude"),
    ]:
        if lat_col in focus_df.columns and lon_col in focus_df.columns:
            coords = (
                focus_df[[lat_col, lon_col]]
                .dropna()
                .astype(float)
                .values
                .tolist()
            )
            points.extend(coords)

    deduped: list[list[float]] = []
    seen = set()
    for lat, lon in points:
        key = (round(float(lat), 5), round(float(lon), 5))
        if key not in seen:
            seen.add(key)
            deduped.append([float(lat), float(lon)])

    return deduped

def _inject_final_focus_script(html_path: Path, focus_points: list[list[float]]) -> None:
    """Force final browser-side map focus after all Folium/legacy scripts run."""
    if html_path is None or not Path(html_path).exists() or not focus_points:
        return

    import json
    import re

    html_path = Path(html_path)
    html = html_path.read_text(encoding="utf-8")

    match = re.search(r"var\s+(map_[A-Za-z0-9_]+)\s*=\s*L\.map", html)
    if not match:
        return

    map_var = match.group(1)
    focus_json = json.dumps(focus_points)

    final_script = f"""
                    <script>
                    (function() {{
                        var focusPoints = {focus_json};

                        function applyFinalBlueprintFocus() {{
                            if (typeof {map_var} !== "undefined" && focusPoints && focusPoints.length >= 2) {{
                                {map_var}.fitBounds(focusPoints, {{padding: [50, 50]}});
                            }}
                        }}

                        window.setTimeout(applyFinalBlueprintFocus, 300);
                        window.setTimeout(applyFinalBlueprintFocus, 900);
                        window.setTimeout(applyFinalBlueprintFocus, 1600);
                    }})();
                    </script>
                    """

    if "applyFinalBlueprintFocus" not in html:
        html = html.replace("</html>", final_script + "\n</html>")
        html_path.write_text(html, encoding="utf-8")

def build_map(
    context: dict,
    config: TripConfig,
    output_path: str | Path | None = None,
):
    """Build the production HTML map.

    The actual map drawing still lives in legacy blueprint_trip_map.py.
    This wrapper sanitizes the context and adjusts the saved HTML view.
    """
    blueprint_trip_map = import_legacy_module("blueprint_trip_map")

    merged_context = _sanitize_production_map_context(context, config, output_path)

    print(
        "Map dashboard context:",
        f"trip_days={merged_context.get('CANONICAL_TRIP_DAYS')},",
        f"budget={merged_context.get('CANONICAL_TOTAL_BUDGET'):.2f},",
        "local_gurobi_demo_excluded=True",
    )

    prepare_map_dashboard_data(merged_context, config=config, output_path=output_path)

    html_path = Path(output_path) if output_path is not None else None
    trip_map, day_plan_df, returned_html_path = blueprint_trip_map.build_production_trip_map(
        merged_context,
        output_path=output_path,
        run_live_routing=bool(_config_get(config, "enrichment", "run_live_apis", default=False)),
    )

    if returned_html_path is not None:
        html_path = Path(returned_html_path)

    focus_points = _initial_focus_points(
        day_plan_df,
        focus_mode=merged_context.get("MAP_FOCUS_MODE", "full_trip"),
    )

    if focus_points:
        trip_map.fit_bounds(focus_points, padding=(50, 50))

    if html_path is not None:
        trip_map.save(str(html_path))
    elif output_path is not None:
        trip_map.save(str(output_path))
        html_path = Path(output_path)

    selector_owns_focus = bool(merged_context.get("MAP_ROUTE_ONLY_DEBUG_VIEW", False)) or bool(
        merged_context.get("MAP_BALANCED_ONLY_DEFAULT_VIEW", False)
    )
    if html_path is not None and focus_points and not selector_owns_focus:
        _inject_final_focus_script(Path(html_path), focus_points)

    return trip_map, day_plan_df, html_path
