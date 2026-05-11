"""Thin orchestration helpers for the blueprint notebook."""

from __future__ import annotations

import ast
from pathlib import Path
from copy import deepcopy

import numpy as np
import pandas as pd
from geopy.distance import geodesic

from ._legacy import import_legacy_module
from .bandit_candidate_selector import build_bandit_arms, run_bandit_gurobi_stress_benchmark, run_bandit_search
from .budget import estimate_budget_range
from .config import TripConfig, save_resolved_config
from .data_enrichment import build_enriched_catalog
from .diversity import diversity_audit_row
from .hierarchical_gurobi import solve_hierarchical_trip_with_gurobi, solve_hierarchical_trip_with_greedy
from .nature_catalog import NATURE_POI_COLUMNS, write_nature_route_artifacts
from .route_gurobi_oracle import solve_enriched_route_with_gurobi


def _normalized_shannon_diversity(labels) -> float:
    values = pd.Series(labels).dropna().astype(str)
    if values.empty:
        return 0.0
    counts = values.value_counts().to_numpy(dtype=float)
    probabilities = counts / counts.sum()
    entropy = -np.sum(probabilities * np.log(probabilities))
    return float(entropy / np.log(len(counts))) if len(counts) > 1 else 0.0


def _approximate_route_minutes(frame: pd.DataFrame) -> float:
    if frame.empty or len(frame) < 2:
        return 0.0
    points = frame[["latitude", "longitude"]].astype(float).values.tolist()
    return float(
        sum(
            geodesic(tuple(left), tuple(right)).km * 1.25 / 38.0 * 60.0
            for left, right in zip(points[:-1], points[1:])
        )
    )


def _numeric_column(frame: pd.DataFrame, column: str, default: float = 0.0, fallback_column: str | None = None) -> pd.Series:
    """Return a numeric Series even when optional artifact columns are absent."""
    if column in frame.columns:
        raw = frame[column]
    elif fallback_column and fallback_column in frame.columns:
        raw = frame[fallback_column]
    else:
        raw = pd.Series(default, index=frame.index)
    return pd.to_numeric(raw, errors="coerce").fillna(default)


def _enriched_metric_row(strategy: str, frame: pd.DataFrame, budget_df: pd.DataFrame) -> dict:
    attraction_budget = (
        float(budget_df.loc[budget_df["component"].eq("attractions"), "expected"].iloc[0])
        if not budget_df.empty and budget_df["component"].eq("attractions").any()
        else np.nan
    )
    return {
        "strategy": strategy,
        "profile": "balanced",
        "data_source_scope": "open_enriched_catalog",
        "total_utility": float(pd.to_numeric(frame.get("final_poi_value", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()),
        "interest_adjusted_utility": float(pd.to_numeric(frame.get("interest_adjusted_value", frame.get("final_poi_value", pd.Series(dtype=float))), errors="coerce").fillna(0).sum()),
        "nature_score": float(pd.to_numeric(frame.get("nature_score", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()),
        "scenic_score": float(pd.to_numeric(frame.get("scenic_score", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()),
        "national_parks_selected": int(pd.to_numeric(frame.get("is_national_park", pd.Series(dtype=float)), errors="coerce").fillna(0).astype(bool).sum()) if not frame.empty else 0,
        "protected_parks_selected": int((pd.to_numeric(frame.get("is_state_park", pd.Series(dtype=float)), errors="coerce").fillna(0).astype(bool) | pd.to_numeric(frame.get("is_protected_area", pd.Series(dtype=float)), errors="coerce").fillna(0).astype(bool)).sum()) if not frame.empty else 0,
        "outdoor_weather_risk": float((pd.to_numeric(frame.get("weather_sensitivity", pd.Series(dtype=float)), errors="coerce").fillna(0) * pd.to_numeric(frame.get("weather_risk", pd.Series(0.15, index=frame.index)), errors="coerce").fillna(0.15)).sum()) if not frame.empty else 0.0,
        "seasonality_risk": float(pd.to_numeric(frame.get("seasonality_risk", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if not frame.empty else 0.0,
        "number_attractions": int(len(frame)),
        "total_waiting_time": np.nan,
        "within_city_travel_time": _approximate_route_minutes(frame),
        "intercity_driving_time": 0.0,
        "budget_used": attraction_budget,
        "route_feasibility": True,
        "diversity_score": _normalized_shannon_diversity(frame.get("category", pd.Series(dtype=str))),
        "runtime_seconds": np.nan,
        "optimality_gap": np.nan,
        "data_quality_score": float(pd.to_numeric(frame.get("data_confidence", pd.Series(dtype=float)), errors="coerce").fillna(0.15).mean()) if not frame.empty else 0.15,
        "gateway_start": np.nan,
        "gateway_end": np.nan,
        "days_by_city": None,
        "route_search_strategy": None,
        "small_optimizer_calls": np.nan,
    }


def build_enriched_experiment_summary(
    *,
    enriched_df: pd.DataFrame,
    best_trip: dict,
    budget_df: pd.DataFrame,
    city_summary_df: pd.DataFrame,
    bandit_stress_runs_df: pd.DataFrame,
    epsilon_route_result: dict | None = None,
    config: TripConfig,
) -> pd.DataFrame:
    max_stops = int(config.get("trip", "trip_days", 7)) * 3
    ranking_frame = enriched_df.sort_values(["final_poi_value", "data_confidence"], ascending=False).head(max_stops)
    greedy_pool = enriched_df.sort_values("final_poi_value", ascending=False).head(max_stops * 3).copy()
    if greedy_pool.empty:
        greedy_frame = greedy_pool
    else:
        selected = [greedy_pool.index[0]]
        remaining = set(greedy_pool.index.tolist()) - set(selected)
        while remaining and len(selected) < max_stops:
            current = enriched_df.loc[selected[-1], ["latitude", "longitude"]].astype(float).tolist()
            next_idx = max(
                remaining,
                key=lambda idx: float(enriched_df.loc[idx, "final_poi_value"])
                - 0.01 * geodesic(tuple(current), tuple(enriched_df.loc[idx, ["latitude", "longitude"]].astype(float).tolist())).km,
            )
            selected.append(next_idx)
            remaining.remove(next_idx)
        greedy_frame = enriched_df.loc[selected].copy()

    buffered_total = float(budget_df.loc[budget_df["component"].eq("buffered_total"), "expected"].iloc[0])
    rows = []
    for strategy, column in [
        ("mcda_weighted", "utility_mcda_weighted"),
        ("topsis", "utility_topsis"),
        ("bayesian_ucb", "utility_bayesian_ucb"),
    ]:
        if column in enriched_df.columns:
            method_frame = enriched_df.sort_values([column, "data_confidence"], ascending=False).head(max_stops).copy()
            method_row = _enriched_metric_row(strategy, method_frame, budget_df)
            method_row["total_utility"] = float(pd.to_numeric(method_frame[column], errors="coerce").fillna(0.0).sum())
            method_row["utility_score_column"] = column
            rows.append(method_row)
    rows.extend(
        [
            _enriched_metric_row("ranking_baseline", ranking_frame, budget_df),
            _enriched_metric_row("greedy_route_baseline", greedy_frame, budget_df),
        ]
    )
    if epsilon_route_result:
        rows.append(
            {
                "strategy": "epsilon_gurobi",
                "profile": "balanced",
                "data_source_scope": "open_enriched_catalog",
                "total_utility": float(epsilon_route_result.get("objective_value", np.nan)),
                "number_attractions": int(len(epsilon_route_result.get("selected_pois", []))),
                "total_waiting_time": np.nan,
                "within_city_travel_time": float(epsilon_route_result.get("total_travel_minutes", np.nan)),
                "intercity_driving_time": 0.0,
                "budget_used": float(epsilon_route_result.get("total_cost", np.nan)),
                "route_feasibility": bool(epsilon_route_result.get("feasible", False)),
                "diversity_score": float(epsilon_route_result.get("diversity_score", np.nan)),
                "runtime_seconds": float(epsilon_route_result.get("runtime_seconds", np.nan)),
                "optimality_gap": epsilon_route_result.get("optimality_gap", np.nan),
                "data_quality_score": float(pd.to_numeric(enriched_df.get("data_confidence", pd.Series(dtype=float)), errors="coerce").fillna(0.15).mean()),
                "gateway_start": np.nan,
                "gateway_end": np.nan,
                "days_by_city": None,
                "route_search_strategy": "epsilon_constraint_route",
                "small_optimizer_calls": 1,
                "solver_status": epsilon_route_result.get("solver_status"),
            }
        )
    rows.append(
        {
            "strategy": "hierarchical_gurobi",
            "profile": "balanced",
            "data_source_scope": "open_enriched_catalog",
            "total_utility": best_trip["objective"],
            "number_attractions": np.nan,
            "total_waiting_time": np.nan,
            "within_city_travel_time": np.nan,
            "intercity_driving_time": best_trip["intercity_drive_minutes"],
            "budget_used": buffered_total,
            "route_feasibility": sum(best_trip["days_by_city"].values()) == int(config.get("trip", "trip_days", 7)),
            "diversity_score": _normalized_shannon_diversity(best_trip["days_by_city"].keys()),
            "runtime_seconds": np.nan,
            "optimality_gap": np.nan,
            "data_quality_score": float(city_summary_df["data_quality_score"].mean()),
            "gateway_start": best_trip["gateway_start"],
            "gateway_end": best_trip["gateway_end"],
            "days_by_city": str(best_trip["days_by_city"]),
            "route_search_strategy": None,
            "small_optimizer_calls": np.nan,
        },
    )
    if not bandit_stress_runs_df.empty:
        best_hybrid = bandit_stress_runs_df.sort_values("combined_reward", ascending=False).iloc[0]
        rows.append(
            {
                "strategy": "hybrid_bandit_gurobi",
                "profile": "balanced",
                "data_source_scope": "open_enriched_catalog",
                "total_utility": float(best_hybrid.get("combined_reward", np.nan)),
                "number_attractions": int(best_hybrid.get("selected_poi_count", 0)),
                "total_waiting_time": np.nan,
                "within_city_travel_time": float(best_hybrid.get("total_travel_minutes", np.nan)),
                "intercity_driving_time": np.nan,
                "budget_used": float(best_hybrid.get("estimated_total_cost", np.nan)),
                "route_feasibility": bool(best_hybrid.get("route_feasible", True)),
                "diversity_score": np.nan,
                "runtime_seconds": float(best_hybrid.get("runtime_seconds", np.nan)),
                "optimality_gap": best_hybrid.get("optimality_gap", np.nan),
                "data_quality_score": float(city_summary_df["data_quality_score"].mean()),
                "gateway_start": best_hybrid.get("gateway_start"),
                "gateway_end": best_hybrid.get("gateway_end"),
                "days_by_city": None,
                "route_search_strategy": best_hybrid.get("route_search_strategy"),
                "small_optimizer_calls": int(best_hybrid.get("small_optimizer_calls", 0)),
            }
        )
    summary = pd.DataFrame(rows)
    summary["optimized_total_utility"] = summary["total_utility"]
    summary["optimized_total_wait"] = summary["total_waiting_time"]
    summary["optimized_total_cost"] = summary["budget_used"]
    summary["intercity_drive_hours"] = summary["intercity_driving_time"] / 60.0
    return summary


def _merge_config_data(base: dict, override: dict) -> dict:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge_config_data(result[key], value)
        else:
            result[key] = value
    return result


def _first_context_value(context: dict, *keys, default=None):
    for key in keys:
        if key in context:
            return context[key]
    return default


def _load_frame_from_context_or_csv(context: dict, keys: list[str], path: str | Path) -> pd.DataFrame:
    for key in keys:
        frame = context.get(key)
        if isinstance(frame, pd.DataFrame) and not frame.empty:
            return frame.copy()
    csv_path = Path(path)
    if csv_path.exists():
        try:
            return pd.read_csv(csv_path)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def _load_csv(path: str | Path) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(csv_path)
    except Exception:
        return pd.DataFrame()


def _parse_name_list(value) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, tuple):
        return [str(item) for item in value if str(item).strip()]
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple)):
            return [str(item) for item in parsed if str(item).strip()]
    except Exception:
        pass
    for separator in [" | ", " -> ", ";", "|", ","]:
        if separator in text:
            return [part.strip().strip("'\"") for part in text.split(separator) if part.strip().strip("'\"")]
    return [text.strip("'\"")]


def _route_sequence_distance_km(points: list[list[float]]) -> float:
    clean_points = []
    for point in points:
        if point is None or len(point) < 2:
            continue
        try:
            lat = float(point[0])
            lon = float(point[1])
        except Exception:
            continue
        if clean_points and abs(clean_points[-1][0] - lat) < 1e-9 and abs(clean_points[-1][1] - lon) < 1e-9:
            continue
        clean_points.append([lat, lon])
    if len(clean_points) < 2:
        return 0.0
    return float(
        sum(geodesic(tuple(left), tuple(right)).km for left, right in zip(clean_points[:-1], clean_points[1:]))
    )


def _normalized_route_stops(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    output = frame.copy()
    output["day"] = pd.to_numeric(output.get("day", 1), errors="coerce").fillna(1).astype(int)
    output["stop_order"] = pd.to_numeric(output.get("stop_order", 1), errors="coerce").fillna(1).astype(int)
    return output.sort_values(["day", "stop_order", "attraction_name"], ascending=[True, True, True]).reset_index(drop=True)


def _route_metrics_from_stops(frame: pd.DataFrame) -> dict:
    normalized = _normalized_route_stops(frame)
    if normalized.empty:
        return {
            "total_utility": 0.0,
            "total_waiting_time": np.nan,
            "total_travel_time": 0.0,
            "total_travel_distance_km": 0.0,
            "selected_attractions": 0,
            "diversity_score": 0.0,
        }

    distance_km = 0.0
    for _, group in normalized.groupby("day", sort=True):
        route_points = []
        first = group.iloc[0]
        for left_col, right_col in [
            ("route_start_latitude", "route_start_longitude"),
            ("hotel_latitude", "hotel_longitude"),
        ]:
            if left_col in group.columns and right_col in group.columns:
                try:
                    route_points = [[float(first[left_col]), float(first[right_col])]]
                    break
                except Exception:
                    route_points = []
        route_points.extend(group[["latitude", "longitude"]].astype(float).values.tolist())
        for left_col, right_col in [
            ("route_end_latitude", "route_end_longitude"),
            ("hotel_latitude", "hotel_longitude"),
        ]:
            if left_col in group.columns and right_col in group.columns:
                try:
                    route_points.append([float(first[left_col]), float(first[right_col])])
                    break
                except Exception:
                    continue
        distance_km += _route_sequence_distance_km(route_points)

    utility_series = pd.to_numeric(normalized.get("final_poi_value", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    waiting_series = pd.to_numeric(normalized.get("pred_wait_minutes", pd.Series(dtype=float)), errors="coerce")
    if waiting_series.dropna().empty:
        total_waiting_time = np.nan
    else:
        total_waiting_time = float(waiting_series.fillna(0.0).sum())
    return {
        "total_utility": float(utility_series.sum()),
        "total_waiting_time": total_waiting_time,
        "total_travel_distance_km": float(distance_km),
        "total_travel_time": float(distance_km * 1.25 / 38.0 * 60.0),
        "selected_attractions": int(len(normalized)),
        "diversity_score": _normalized_shannon_diversity(normalized.get("category", pd.Series(dtype=str))),
    }


def _must_go_candidates_from_output(output_dir: Path, enriched_df: pd.DataFrame | None = None) -> pd.DataFrame:
    candidates = _load_csv(output_dir / "production_social_must_go_candidates.csv")
    if candidates.empty and enriched_df is not None and not enriched_df.empty:
        candidates = enriched_df.copy()
    if candidates.empty:
        candidates = _load_csv(output_dir / "production_enriched_poi_catalog.csv")
    if candidates.empty:
        return pd.DataFrame()
    candidates = candidates.copy()
    if "social_must_go" not in candidates.columns:
        candidates["social_must_go"] = False
    candidates["social_must_go"] = candidates["social_must_go"].fillna(False).astype(bool)
    candidates["social_score"] = _numeric_column(candidates, "social_score")
    candidates["must_go_weight"] = _numeric_column(candidates, "must_go_weight")
    candidates = candidates[candidates["social_must_go"] | candidates["social_score"].ge(0.70)].copy()
    keep_columns = [
        "name",
        "city",
        "latitude",
        "longitude",
        "category",
        "source_list",
        "social_score",
        "social_must_go",
        "must_go_weight",
        "social_reason",
    ]
    for column in keep_columns:
        if column not in candidates.columns:
            candidates[column] = ""
    return candidates[keep_columns].drop_duplicates("name").reset_index(drop=True)


def _route_must_go_summary(route_stops: pd.DataFrame, must_go_candidates: pd.DataFrame, config: TripConfig) -> dict:
    policy = "mandatory" if bool(config.get("social", "must_go_is_mandatory", False)) else "soft_reward_not_mandatory"
    if must_go_candidates.empty:
        return {
            "must_go_total": 0,
            "must_go_selected_count": 0,
            "must_go_skipped_count": 0,
            "must_go_coverage_ratio": np.nan,
            "must_go_skipped_names": "",
            "must_go_policy": policy,
        }
    selected_names = (
        set(route_stops.get("attraction_name", pd.Series(dtype=str)).dropna().astype(str).tolist())
        if not route_stops.empty
        else set()
    )
    must_go_names = must_go_candidates["name"].dropna().astype(str).tolist()
    selected = sorted(name for name in must_go_names if name in selected_names)
    skipped = sorted(name for name in must_go_names if name not in selected_names)
    total = len(must_go_names)
    return {
        "must_go_total": int(total),
        "must_go_selected_count": int(len(selected)),
        "must_go_skipped_count": int(len(skipped)),
        "must_go_coverage_ratio": float(len(selected) / total) if total else np.nan,
        "must_go_skipped_names": "; ".join(skipped),
        "must_go_policy": policy,
    }


def _minmax_component(series: pd.Series, *, higher_is_better: bool) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    finite = values.dropna()
    if finite.empty:
        return pd.Series(np.nan, index=series.index)
    min_value = float(finite.min())
    max_value = float(finite.max())
    if abs(max_value - min_value) < 1e-9:
        return pd.Series(np.where(values.notna(), 1.0, np.nan), index=series.index)
    normalized = (values - min_value) / (max_value - min_value)
    if not higher_is_better:
        normalized = 1.0 - normalized
    return normalized.clip(lower=0.0, upper=1.0)


def _add_method_comparison_scores(method_df: pd.DataFrame) -> pd.DataFrame:
    """Add mathematically explained normalized comparison metrics.

    Score = 0.40 U' + 0.15 M' + 0.15 H + 0.15 T' + 0.10 C' + 0.05 W'.
    Lower-is-better metrics are inverted during min-max normalization. If all
    waiting values are missing, the waiting weight is omitted and the remaining
    weights are renormalized.
    """
    if method_df.empty:
        return method_df
    output = method_df.copy()
    output["utility_norm"] = _minmax_component(output.get("total_utility", pd.Series(index=output.index)), higher_is_better=True)
    must_go_metric = output.get("must_go_selected_count", output.get("must_go_count", pd.Series(index=output.index)))
    output["must_go_norm"] = _minmax_component(must_go_metric, higher_is_better=True)
    output["diversity_norm"] = pd.to_numeric(output.get("diversity_score", pd.Series(index=output.index)), errors="coerce").clip(lower=0.0, upper=1.0)
    output["travel_efficiency_norm"] = _minmax_component(output.get("total_travel_time", pd.Series(index=output.index)), higher_is_better=False)
    output["cost_efficiency_norm"] = _minmax_component(output.get("total_cost", output.get("budget_used", pd.Series(index=output.index))), higher_is_better=False)
    output["waiting_efficiency_norm"] = _minmax_component(output.get("total_waiting_time", pd.Series(index=output.index)), higher_is_better=False)
    components = {
        "utility_norm": 0.40,
        "must_go_norm": 0.15,
        "diversity_norm": 0.15,
        "travel_efficiency_norm": 0.15,
        "cost_efficiency_norm": 0.10,
        "waiting_efficiency_norm": 0.05,
    }
    scores = []
    for idx, row in output.iterrows():
        used_weight = 0.0
        score_value = 0.0
        for column, weight in components.items():
            value = row.get(column, np.nan)
            if pd.isna(value):
                continue
            used_weight += float(weight)
            score_value += float(weight) * float(value)
        scores.append(score_value / used_weight if used_weight > 0 else np.nan)
    output["comparison_score"] = scores
    output["comparison_score_formula"] = (
        "0.40*utility_norm + 0.15*must_go_norm + 0.15*diversity_norm + "
        "0.15*travel_efficiency_norm + 0.10*cost_efficiency_norm + 0.05*waiting_efficiency_norm; "
        "available components are renormalized if a metric is missing"
    )
    return output


def _buffered_budget_value(budget_df: pd.DataFrame) -> float:
    if budget_df.empty or "component" not in budget_df.columns:
        return np.nan
    matches = budget_df[budget_df["component"].astype(str).eq("buffered_total")]
    if matches.empty:
        return np.nan
    return float(pd.to_numeric(matches["expected"], errors="coerce").iloc[0])


USE_LOCAL_GUROBI_DEMO_IN_PRODUCTION_MAP = False


def _budget_component_value(budget_df: pd.DataFrame, component: str) -> float:
    if budget_df.empty or "component" not in budget_df.columns:
        return np.nan
    matches = budget_df[budget_df["component"].astype(str).eq(str(component))]
    if matches.empty:
        return np.nan
    return float(pd.to_numeric(matches["expected"], errors="coerce").iloc[0])


def get_canonical_trip_days(config: TripConfig, best_hierarchical_trip: dict | None = None) -> int:
    if isinstance(best_hierarchical_trip, dict):
        days_by_city = best_hierarchical_trip.get("days_by_city", {})
        if isinstance(days_by_city, dict) and days_by_city:
            try:
                return int(sum(int(value) for value in days_by_city.values()))
            except Exception:
                pass
    return int(config.get("trip", "trip_days", 7))


def _canonical_budget_value_and_source(config: TripConfig, budget_estimate_df: pd.DataFrame | None = None) -> tuple[float, str]:
    budget_estimate_df = budget_estimate_df if isinstance(budget_estimate_df, pd.DataFrame) else pd.DataFrame()
    for component in ["buffered_total", "hard_budget", "soft_budget"]:
        value = _budget_component_value(budget_estimate_df, component)
        if np.isfinite(value):
            return float(value), component
    for section, key in [
        ("trip", "budget"),
        ("trip", "total_budget"),
        ("optimization", "total_budget"),
        ("budget", "user_budget"),
    ]:
        try:
            value = float(config.get(section, key))
        except Exception:
            value = np.nan
        if np.isfinite(value):
            return float(value), f"config:{section}.{key}"
    return 2200.0, "fallback:2200.0"


def get_canonical_trip_budget(config: TripConfig, budget_estimate_df: pd.DataFrame | None = None) -> float:
    return float(_canonical_budget_value_and_source(config, budget_estimate_df)[0])


def build_canonical_budget_context(
    best_trip: dict,
    budget_estimate_df: pd.DataFrame | None = None,
    selected_hotels_by_city: dict | None = None,
    *,
    config: TripConfig,
) -> dict:
    budget_estimate_df = budget_estimate_df if isinstance(budget_estimate_df, pd.DataFrame) else pd.DataFrame()
    total_budget, budget_source = _canonical_budget_value_and_source(config, budget_estimate_df)
    food_budget = _budget_component_value(budget_estimate_df, "food")
    transport_budget = _budget_component_value(budget_estimate_df, "fuel_parking_local_transport")
    hotel_budget_expected = _budget_component_value(budget_estimate_df, "hotel")
    reserved_food_transport = float(
        sum(value for value in [food_budget, transport_budget] if np.isfinite(value))
    )

    days_by_city = best_trip.get("days_by_city", {}) if isinstance(best_trip, dict) else {}
    city_sequence = [str(item) for item in best_trip.get("city_sequence", [])] if isinstance(best_trip, dict) else []
    base_cities = [city for city in city_sequence if int(days_by_city.get(city, 0)) > 0]
    booked_nights = max(0, get_canonical_trip_days(config, best_trip) - 1)
    expected_per_night = hotel_budget_expected / booked_nights if np.isfinite(hotel_budget_expected) and booked_nights > 0 else np.nan

    committed_hotel_cost = 0.0
    if isinstance(selected_hotels_by_city, dict) and base_cities:
        for city in base_cities:
            hotel_row = selected_hotels_by_city.get(city)
            allocated_days = int(days_by_city.get(city, 0))
            is_final_base = bool(base_cities and city == base_cities[-1])
            booked_city_nights = max(0, allocated_days - 1) if is_final_base else max(0, allocated_days)
            nightly_price = np.nan
            if hotel_row is not None:
                try:
                    nightly_price = float(hotel_row.get("nightly_price", np.nan))
                except Exception:
                    nightly_price = np.nan
            if np.isfinite(nightly_price):
                committed_hotel_cost += float(booked_city_nights) * float(nightly_price)
            elif np.isfinite(expected_per_night):
                committed_hotel_cost += float(booked_city_nights) * float(expected_per_night)
    elif np.isfinite(hotel_budget_expected):
        committed_hotel_cost = float(hotel_budget_expected)

    remaining_route_budget = max(0.0, float(total_budget) - float(reserved_food_transport) - float(committed_hotel_cost))
    return {
        "total_budget": float(total_budget),
        "budget_source": str(budget_source),
        "reserved_food_transport": float(reserved_food_transport),
        "committed_hotel_cost": float(committed_hotel_cost),
        "remaining_route_budget": float(remaining_route_budget),
    }


METHOD_DISPLAY_NAMES = {
    "hierarchical_gurobi_pipeline": "Hierarchical Gurobi Pipeline",
    "hierarchical_greedy_baseline": "Hierarchical Greedy Baseline",
    "hierarchical_bandit_gurobi_repair": "Hierarchical + Bandit + Small Gurobi Repair",
}

METHOD_SOLVER_ROLES = {
    "hierarchical_gurobi_pipeline": ("gurobi_master", "legacy_local_gurobi"),
    "hierarchical_greedy_baseline": ("greedy_master", "greedy_local_route"),
    "hierarchical_bandit_gurobi_repair": ("bandit_master", "small_gurobi_repair"),
}

MATRIX_METHOD_ORDER = [
    "hierarchical_gurobi_pipeline",
    "hierarchical_greedy_baseline",
    "hierarchical_bandit_gurobi_repair",
]

MATRIX_PROFILE_ORDER = ["relaxed", "balanced", "explorer"]

METHOD_ROUTE_KEY_SLUG = {
    "hierarchical_gurobi_pipeline": "gurobi",
    "hierarchical_greedy_baseline": "greedy",
    "hierarchical_bandit_gurobi_repair": "bandit_repair",
}

METHOD_SHORT_LABELS = {
    "hierarchical_gurobi_pipeline": "Gurobi",
    "hierarchical_greedy_baseline": "Greedy",
    "hierarchical_bandit_gurobi_repair": "Bandit + Small Gurobi",
}

PROFILE_STRATEGY_DEFAULTS = {
    "relaxed": "fastest_low_cost",
    "balanced": "balanced",
    "explorer": "scenic_social",
}


def _method_display_name(method: str) -> str:
    return METHOD_DISPLAY_NAMES.get(str(method), str(method))


def _method_solver_roles(method: str) -> tuple[str, str]:
    return METHOD_SOLVER_ROLES.get(str(method), ("unknown_allocation", "unknown_local_route"))


def _method_short_label(method: str) -> str:
    return METHOD_SHORT_LABELS.get(str(method), _method_display_name(method))


def _route_key(trip_days: int, method: str, profile: str) -> str:
    method_slug = METHOD_ROUTE_KEY_SLUG.get(str(method), str(method).lower().replace(" ", "_"))
    profile_slug = str(profile).lower().replace(" ", "_")
    return f"d{int(trip_days)}__{method_slug}__{profile_slug}"


def _profile_label(blueprint_trip_map, profile: str) -> str:
    try:
        return str(blueprint_trip_map._profile_config(profile).get("label", str(profile).title()))
    except Exception:
        return str(profile).title()


def _profile_strategy(profile: str, fallback: str = "balanced") -> str:
    return PROFILE_STRATEGY_DEFAULTS.get(str(profile).lower(), fallback)


def _comparison_route_stop_columns() -> list[str]:
    return [
        "comparison_type",
        "comparison_label",
        "method",
        "method_display_name",
        "trip_days",
        "gateway_start",
        "gateway_end",
        "day",
        "stop_order",
        "city",
        "overnight_city",
        "hotel_name",
        "hotel_latitude",
        "hotel_longitude",
        "attraction_name",
        "latitude",
        "longitude",
        "category",
        "source_list",
        "social_must_go",
        "social_score",
        "must_go_weight",
        "social_reason",
        "final_poi_value",
        "route_start_city",
        "route_start_name",
        "route_start_latitude",
        "route_start_longitude",
        "route_end_city",
        "route_end_name",
        "route_end_latitude",
        "route_end_longitude",
        "pass_through_cities",
        "drive_minutes_to_next_base",
        "available_visit_minutes",
        "drive_time_source",
        "route_type",
        "status",
        "notes",
        *NATURE_POI_COLUMNS,
    ]


def _day_plan_to_comparison_route_stops(
    day_plan_df: pd.DataFrame,
    *,
    comparison_type: str,
    comparison_label: str,
    method: str,
    method_display_name: str | None = None,
    trip_days: int,
    notes: str = "",
) -> pd.DataFrame:
    columns = _comparison_route_stop_columns()
    if day_plan_df.empty:
        return pd.DataFrame(columns=columns)

    output = day_plan_df.copy()
    output["comparison_type"] = comparison_type
    output["comparison_label"] = comparison_label
    output["method"] = method
    output["method_display_name"] = method_display_name or _method_display_name(method)
    output["trip_days"] = int(trip_days)
    output["status"] = output.get("status", "READY")
    output["notes"] = notes
    source_score_default = (
        pd.to_numeric(output["source_score"], errors="coerce").fillna(0.0)
        if "source_score" in output.columns
        else pd.Series(0.0, index=output.index)
    )
    defaults = {
        "overnight_city": output.get("city", ""),
        "hotel_name": "",
        "hotel_latitude": np.nan,
        "hotel_longitude": np.nan,
        "category": "attraction",
        "source_list": output.get("attraction_source", output.get("source", "unknown")),
        "social_must_go": False,
        "social_score": 0.0,
        "must_go_weight": 0.0,
        "social_reason": "",
        "final_poi_value": source_score_default,
        "route_start_city": output.get("route_start_city", output.get("city", "")),
        "route_start_name": output.get("hotel_name", ""),
        "route_start_latitude": output.get("hotel_latitude", np.nan),
        "route_start_longitude": output.get("hotel_longitude", np.nan),
        "route_end_city": output.get("route_end_city", output.get("overnight_city", output.get("city", ""))),
        "route_end_name": output.get("hotel_name", ""),
        "route_end_latitude": output.get("hotel_latitude", np.nan),
        "route_end_longitude": output.get("hotel_longitude", np.nan),
        "pass_through_cities": output.get("pass_through_cities", ""),
        "drive_minutes_to_next_base": output.get("drive_minutes_to_next_base", 0.0),
        "available_visit_minutes": output.get("available_visit_minutes", np.nan),
        "drive_time_source": output.get("drive_time_source", "unknown"),
        "route_type": "comparison_route",
    }
    for column, default in defaults.items():
        if column not in output.columns:
            output[column] = default
    for column in columns:
        if column not in output.columns:
            output[column] = np.nan if column.endswith(("latitude", "longitude")) else ""
    output = _normalized_route_stops(output)
    return output[columns]


def _catalog_rows_for_names(names: list[str], enriched_df: pd.DataFrame) -> pd.DataFrame:
    if not names or enriched_df.empty:
        return pd.DataFrame()
    catalog = enriched_df.copy().drop_duplicates("name")
    catalog["_name_key"] = catalog["name"].astype(str).str.lower()
    rows = []
    for order, name in enumerate(names, start=1):
        matches = catalog[catalog["_name_key"].eq(str(name).lower())]
        if matches.empty:
            continue
        row = matches.iloc[0].copy()
        row["day"] = 1
        row["stop_order"] = order
        row["attraction_name"] = row.get("name", name)
        rows.append(row)
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _catalog_rows_for_positions(positions: list[int], frame: pd.DataFrame) -> pd.DataFrame:
    if not positions or frame.empty:
        return pd.DataFrame()
    normalized = frame.reset_index(drop=True).copy()
    rows = []
    for order, position in enumerate(positions, start=1):
        try:
            row = normalized.iloc[int(position)].copy()
        except Exception:
            continue
        row["day"] = 1
        row["stop_order"] = order
        row["attraction_name"] = row.get("name", f"selected stop {order}")
        rows.append(row)
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _trip_summary_fields(trip: dict) -> dict:
    return {
        "gateway_start": trip.get("gateway_start"),
        "gateway_end": trip.get("gateway_end"),
        "city_sequence": " -> ".join([str(item) for item in trip.get("city_sequence", [])]),
        "days_by_city": str(trip.get("days_by_city", {})),
    }


def _trip_from_bandit_row(row, blueprint_trip_map) -> dict:
    city_sequence = _parse_name_list(row.get("city_sequence"))
    days_by_city = row.get("days_by_city", {})
    if not isinstance(days_by_city, dict):
        try:
            days_by_city = ast.literal_eval(str(days_by_city))
        except Exception:
            days_by_city = {}
    if not city_sequence:
        city_sequence = list(days_by_city.keys())
    overnight_bases = list(days_by_city.keys())
    pass_through_cities = [city for city in city_sequence if city not in set(overnight_bases)]
    intercity_drive_minutes = 0.0
    try:
        base_cities = [city for city in city_sequence if int(days_by_city.get(city, 0)) > 0]
        for base_index, city in enumerate(base_cities):
            if base_index == 0:
                continue
            previous_city = base_cities[base_index - 1]
            segment = blueprint_trip_map._sequence_between(city_sequence, previous_city, city)
            segment_pass = [stop_city for stop_city in segment[1:-1] if stop_city != city]
            drive_minutes, _ = blueprint_trip_map._transition_drive_minutes(
                {"OUTPUT_DIR": row.get("OUTPUT_DIR", "")},
                previous_city,
                city,
                segment_pass,
            )
            intercity_drive_minutes += float(drive_minutes)
    except Exception:
        intercity_drive_minutes = 0.0
    return {
        "gateway_start": row.get("gateway_start"),
        "gateway_end": row.get("gateway_end"),
        "city_sequence": city_sequence,
        "days_by_city": days_by_city,
        "overnight_bases": overnight_bases,
        "pass_through_cities": pass_through_cities,
        "intercity_drive_minutes": intercity_drive_minutes,
        "objective": float(row.get("hierarchical_objective", np.nan)) if pd.notna(row.get("hierarchical_objective", np.nan)) else np.nan,
        "solver_status": row.get("solver_status", "bandit_summary"),
    }


def _estimate_visit_minutes_from_row(row: pd.Series) -> float:
    category = str(row.get("category", "")).lower()
    if "museum" in category or "gallery" in category:
        return 105.0
    if "view" in category or "bridge" in category or "landmark" in category:
        return 50.0
    if "park" in category or "garden" in category:
        return 90.0
    if "campus" in category:
        return 75.0
    return 75.0


def _strategy_weights(strategy_name: str) -> dict:
    base = {
        "value": 1.00,
        "source": 0.16,
        "rating": 0.18,
        "review": 0.12,
        "social": 0.40,
        "must_go": 0.24,
        "corridor": 0.16,
        "view": 0.10,
        "distance": 0.055,
        "detour": 0.006,
        "local_city_bonus": 0.16,
        "pass_through_bonus": 0.10,
    }
    strategy = str(strategy_name or "balanced").lower()
    if strategy == "scenic_social":
        base.update({"social": 0.78, "must_go": 0.34, "corridor": 0.34, "view": 0.30, "distance": 0.040, "detour": 0.004})
    elif strategy == "fastest_low_cost":
        base.update({"social": 0.20, "must_go": 0.10, "corridor": 0.08, "view": 0.06, "distance": 0.095, "detour": 0.018})
    elif strategy == "high_must_go":
        base.update({"social": 0.60, "must_go": 0.55, "corridor": 0.20, "view": 0.18, "distance": 0.050})
    elif strategy == "low_detour":
        base.update({"social": 0.24, "must_go": 0.14, "corridor": 0.12, "distance": 0.070, "detour": 0.020})
    elif strategy == "nature_heavy":
        base.update({"view": 0.32, "corridor": 0.18, "distance": 0.045, "detour": 0.010})
    elif strategy == "scenic_parks":
        base.update({"view": 0.44, "corridor": 0.22, "distance": 0.040, "detour": 0.009})
    elif strategy == "national_park_priority":
        base.update({"view": 0.35, "corridor": 0.14, "distance": 0.038, "detour": 0.008})
    elif strategy == "city_nature_balanced":
        base.update({"view": 0.22, "social": 0.32, "corridor": 0.16, "distance": 0.052})
    elif strategy == "weather_safe_indoor_backup":
        base.update({"view": 0.04, "social": 0.24, "distance": 0.075, "detour": 0.018})
    return base


def _score_daily_candidate_pool(
    pool: pd.DataFrame,
    *,
    hotel_lat: float,
    hotel_lon: float,
    overnight_city: str,
    strategy_name: str,
    transition_day: bool,
) -> pd.DataFrame:
    if pool.empty:
        return pool.copy()
    scored = pool.copy()
    yelp_rating_series = (
        pd.to_numeric(scored["yelp_rating"], errors="coerce").fillna(0.0)
        if "yelp_rating" in scored.columns
        else pd.to_numeric(scored["rating"], errors="coerce").fillna(0.0)
        if "rating" in scored.columns
        else pd.Series(0.0, index=scored.index)
    )
    review_count_series = (
        pd.to_numeric(scored["yelp_review_count"], errors="coerce").fillna(0.0)
        if "yelp_review_count" in scored.columns
        else pd.to_numeric(scored["review_count"], errors="coerce").fillna(0.0)
        if "review_count" in scored.columns
        else pd.Series(0.0, index=scored.index)
    )
    value_source = "interest_adjusted_value" if str(strategy_name).lower() in {
        "nature_heavy",
        "scenic_parks",
        "national_park_priority",
        "city_nature_balanced",
        "weather_safe_indoor_backup",
    } and "interest_adjusted_value" in scored.columns else "final_poi_value"
    scored["final_poi_value"] = _numeric_column(scored, value_source, fallback_column="source_score")
    scored["source_score"] = _numeric_column(scored, "source_score")
    scored["social_score"] = _numeric_column(scored, "social_score")
    scored["must_go_weight"] = _numeric_column(scored, "must_go_weight")
    scored["corridor_fit"] = _numeric_column(scored, "corridor_fit")
    scored["detour_minutes"] = _numeric_column(scored, "detour_minutes")
    for interest_column in ["nature_score", "city_score", "culture_score", "history_score", "scenic_score", "hiking_score", "park_bonus", "weather_sensitivity", "weather_risk", "outdoor_intensity"]:
        scored[interest_column] = _numeric_column(scored, interest_column)
    scored["yelp_rating"] = yelp_rating_series
    scored["yelp_review_count"] = review_count_series
    scored["social_must_go"] = scored.get("social_must_go", False)
    scored["social_must_go"] = scored["social_must_go"].fillna(False).astype(bool)
    scored["distance_to_hotel_km"] = scored.apply(
        lambda row: geodesic((hotel_lat, hotel_lon), (float(row["latitude"]), float(row["longitude"]))).km,
        axis=1,
    )
    weights = _strategy_weights(strategy_name)
    max_value = max(float(scored["final_poi_value"].max()), 1e-6)
    max_source = max(float(scored["source_score"].max()), 1e-6)
    max_rating = max(float(scored["yelp_rating"].max()), 1e-6)
    max_reviews = max(float(scored["yelp_review_count"].max()), 1e-6)
    scored["value_norm"] = scored["final_poi_value"] / max_value
    scored["source_norm"] = scored["source_score"] / max_source
    scored["rating_norm"] = scored["yelp_rating"] / max_rating
    scored["review_norm"] = np.log1p(scored["yelp_review_count"]) / max(np.log1p(max_reviews), 1e-6)
    scored["view_signal"] = scored["category"].astype(str).str.contains(
        "view|bridge|beach|waterfront|landmark|scenic|observatory|campus",
        case=False,
        na=False,
    ).astype(float)
    scored["local_city_match"] = scored["city"].astype(str).str.lower().eq(str(overnight_city).lower()).astype(float)
    scored["transition_bonus"] = scored["view_signal"] if transition_day else 0.0
    scored["route_candidate_score"] = (
        weights["value"] * scored["value_norm"]
        + weights["source"] * scored["source_norm"]
        + weights["rating"] * scored["rating_norm"]
        + weights["review"] * scored["review_norm"]
        + weights["social"] * scored["social_score"]
        + weights["must_go"] * scored["must_go_weight"]
        + 0.18 * scored["social_must_go"].astype(float)
        + 0.12 * scored["must_go_weight"] * scored["social_score"]
        + weights["corridor"] * scored["corridor_fit"]
        + weights["view"] * scored["view_signal"]
        + weights["local_city_bonus"] * scored["local_city_match"]
        + weights["pass_through_bonus"] * scored["transition_bonus"]
        - weights["distance"] * scored["distance_to_hotel_km"]
        - weights["detour"] * scored["detour_minutes"]
    )
    strategy = str(strategy_name or "balanced").lower()
    if strategy == "nature_heavy":
        scored["route_candidate_score"] += 0.80 * scored["nature_score"] + 0.30 * scored["scenic_score"] + 0.25 * scored["hiking_score"]
    elif strategy == "scenic_parks":
        scored["route_candidate_score"] += 0.75 * scored["scenic_score"] + 0.35 * scored["park_bonus"]
    elif strategy == "national_park_priority":
        scored["route_candidate_score"] += 0.90 * scored["park_bonus"] + 0.25 * scored["nature_score"]
    elif strategy == "city_nature_balanced":
        scored["route_candidate_score"] += 0.28 * scored["nature_score"] + 0.22 * scored["city_score"] + 0.14 * scored["culture_score"]
    elif strategy == "weather_safe_indoor_backup":
        weather_exposure = (scored["weather_sensitivity"] * scored["weather_risk"]).clip(0.0, 1.0)
        scored["route_candidate_score"] += 0.38 * (1.0 - weather_exposure) - 0.18 * scored["outdoor_intensity"]
    return scored.sort_values(["route_candidate_score", "final_poi_value", "name"], ascending=[False, False, True]).reset_index(drop=True)


def _candidate_bundle_for_hierarchical_day(
    *,
    context: dict,
    blueprint_trip_map,
    overnight_city: str,
    hotel,
    used_names_by_city: dict,
    pass_through_cities: list[str],
    strategy_name: str,
    max_candidates: int,
    previous_city: str | None = None,
) -> pd.DataFrame:
    city_order = []
    if previous_city == "San Francisco" and "Santa Cruz" in pass_through_cities:
        city_order.append("Palo Alto / Stanford")
    city_order.extend(pass_through_cities)
    city_order.append(overnight_city)
    candidate_frames = []
    for city_name in blueprint_trip_map._unique_in_order(city_order):
        if city_name == "Palo Alto / Stanford":
            stanford = blueprint_trip_map._stanford_transition_stop(context, used_names_by_city)
            if not stanford.empty:
                candidate_frames.append(stanford)
            continue
        catalog = blueprint_trip_map._city_poi_catalog(context, city_name).copy()
        if catalog.empty:
            continue
        used_names = used_names_by_city.setdefault(city_name, set())
        catalog = catalog[~catalog["name"].astype(str).isin(used_names)].copy()
        if catalog.empty:
            continue
        scored = _score_daily_candidate_pool(
            catalog,
            hotel_lat=float(hotel.get("latitude")),
            hotel_lon=float(hotel.get("longitude")),
            overnight_city=overnight_city,
            strategy_name=strategy_name,
            transition_day=bool(pass_through_cities),
        )
        per_city_limit = max_candidates if city_name == overnight_city else max(2, max_candidates // 3)
        candidate_frames.append(scored.head(per_city_limit))
    if not candidate_frames:
        return pd.DataFrame()
    bundle = pd.concat(candidate_frames, ignore_index=True, sort=False).drop_duplicates("name").reset_index(drop=True)
    if "route_candidate_score" not in bundle.columns:
        bundle = _score_daily_candidate_pool(
            bundle,
            hotel_lat=float(hotel.get("latitude")),
            hotel_lon=float(hotel.get("longitude")),
            overnight_city=overnight_city,
            strategy_name=strategy_name,
            transition_day=bool(pass_through_cities),
        )
    return bundle.sort_values(["route_candidate_score", "final_poi_value"], ascending=False).head(max_candidates).reset_index(drop=True)


def _solve_greedy_day_route(
    candidate_pool: pd.DataFrame,
    *,
    route_start: tuple[float, float],
    route_end: tuple[float, float],
    available_visit_minutes: float,
    max_stops: int,
) -> tuple[pd.DataFrame, dict]:
    if candidate_pool.empty:
        return pd.DataFrame(), {"solver_status": "FAILED", "objective_value": np.nan, "optimality_gap": np.nan, "runtime_seconds": 0.0}
    remaining = candidate_pool.copy().reset_index(drop=True)
    selected_rows = []
    current_lat, current_lon = route_start
    spent_minutes = 0.0
    while len(selected_rows) < max_stops and not remaining.empty:
        remaining["travel_from_current"] = remaining.apply(
            lambda row: geodesic((current_lat, current_lon), (float(row["latitude"]), float(row["longitude"]))).km * 1.25 / 38.0 * 60.0,
            axis=1,
        )
        remaining["travel_to_end"] = remaining.apply(
            lambda row: geodesic((float(row["latitude"]), float(row["longitude"])), route_end).km * 1.25 / 38.0 * 60.0,
            axis=1,
        )
        remaining["visit_minutes_est"] = remaining.apply(_estimate_visit_minutes_from_row, axis=1)
        remaining["greedy_step_score"] = (
            _numeric_column(remaining, "route_candidate_score")
            - 0.015 * remaining["travel_from_current"]
            - 0.008 * remaining["travel_to_end"]
        )
        feasible = remaining[
            spent_minutes + remaining["travel_from_current"] + remaining["visit_minutes_est"] + remaining["travel_to_end"]
            <= float(available_visit_minutes)
        ].copy()
        if feasible.empty:
            break
        next_row = feasible.sort_values(["greedy_step_score", "route_candidate_score", "final_poi_value"], ascending=[False, False, False]).iloc[0]
        selected_rows.append(next_row)
        spent_minutes += float(next_row["travel_from_current"]) + float(next_row["visit_minutes_est"])
        current_lat = float(next_row["latitude"])
        current_lon = float(next_row["longitude"])
        remaining = remaining[remaining["name"].astype(str) != str(next_row["name"])].reset_index(drop=True)
    selected_df = pd.DataFrame(selected_rows).reset_index(drop=True) if selected_rows else pd.DataFrame()
    objective_value = float(_numeric_column(selected_df, "route_candidate_score").sum()) if not selected_df.empty else np.nan
    return selected_df, {
        "solver_status": "HEURISTIC",
        "objective_value": objective_value,
        "optimality_gap": np.nan,
        "runtime_seconds": 0.0,
    }


def _estimated_route_stop_cost(frame: pd.DataFrame) -> float:
    if frame.empty:
        return 0.0
    gurobi_itinerary_solver = import_legacy_module("gurobi_itinerary_solver")
    return float(frame.apply(gurobi_itinerary_solver.estimate_local_route_cost, axis=1).sum())


def _solve_legacy_local_gurobi_day_route(
    candidate_pool: pd.DataFrame,
    *,
    config: TripConfig,
    available_visit_minutes: float,
    remaining_budget: float,
    max_stops: int,
    route_start: tuple[float, float],
    route_end: tuple[float, float],
    profile_name: str = "balanced",
) -> tuple[pd.DataFrame, dict]:
    if candidate_pool.empty:
        return pd.DataFrame(), {"solver_status": "FAILED", "objective_value": np.nan, "optimality_gap": np.nan, "runtime_seconds": 0.0}
    gurobi_itinerary_solver = import_legacy_module("gurobi_itinerary_solver")
    route_result = gurobi_itinerary_solver.solve_legacy_local_day_route(
        candidate_pool,
        route_start=route_start,
        route_end=route_end,
        remaining_time=max(60.0, float(available_visit_minutes)),
        remaining_budget=max(0.0, float(remaining_budget)),
        max_pois=int(max_stops),
        profile_name=profile_name,
        solver_overrides={
            "time_limit_seconds": float(config.get("optimization", "gurobi_time_limit_seconds", 120)),
            "mip_gap_target": float(config.get("optimization", "gurobi_mip_gap", 0.05)),
            "max_travel": float(config.get("optimization", "max_travel_minutes", 90)),
            "gamma": float(config.get("optimization", "cost_penalty_weight", config.get("multi_objective", "secondary_travel_penalty", 0.01))),
            "social_score_weight": float(config.get("social", "social_score_weight", 1.10)) * 0.45,
            "must_go_bonus_weight": float(config.get("social", "must_go_bonus_weight", 0.85)),
            "must_go_is_mandatory": bool(config.get("social", "must_go_is_mandatory", False)),
        },
    )
    selected_positions = [int(position) for position in route_result.get("selected_positions", [])]
    selected_df = _catalog_rows_for_positions(selected_positions, candidate_pool)
    if selected_df.empty and not candidate_pool.empty:
        fallback = candidate_pool.sort_values(["route_candidate_score", "final_poi_value"], ascending=False).head(1).copy()
        fallback["day"] = 1
        fallback["stop_order"] = 1
        fallback["attraction_name"] = fallback.get("name", "selected stop")
        selected_df = fallback
    return selected_df.reset_index(drop=True), route_result


def _solve_small_gurobi_day_route(
    candidate_pool: pd.DataFrame,
    *,
    config: TripConfig,
    available_visit_minutes: float,
    remaining_budget: float,
    max_stops: int,
    route_start: tuple[float, float],
) -> tuple[pd.DataFrame, dict]:
    if candidate_pool.empty:
        return pd.DataFrame(), {"solver_status": "FAILED", "objective_value": np.nan, "optimality_gap": np.nan, "runtime_seconds": 0.0}
    day_config = TripConfig(
        data=_merge_config_data(
            config.to_dict(),
            {
                "time": {"daily_time_budget_minutes": max(60.0, float(available_visit_minutes))},
                "budget": {"user_budget": max(1.0, float(remaining_budget))},
                "optimization": {"max_pois_per_day": int(max_stops)},
                "multi_objective": {"daily_hard_budget": max(1.0, float(remaining_budget))},
            },
        ),
        source_path=config.source_path,
    )
    route_result = solve_enriched_route_with_gurobi(
        candidate_pool,
        day_config,
        depot=route_start,
        candidate_size=len(candidate_pool),
    )
    selected_names = [str(name) for name in route_result.get("selected_pois", []) if str(name).strip()]
    selected_df = _catalog_rows_for_names(selected_names, candidate_pool)
    if selected_df.empty and not candidate_pool.empty:
        fallback = candidate_pool.sort_values(["route_candidate_score", "final_poi_value"], ascending=False).head(1).copy()
        fallback["day"] = 1
        fallback["stop_order"] = 1
        fallback["attraction_name"] = fallback.get("name", "selected stop")
        selected_df = fallback
    return selected_df.reset_index(drop=True), route_result


def _build_route_rows_from_selected(
    selected_df: pd.DataFrame,
    *,
    method: str,
    method_display_name: str,
    trip_days: int,
    gateway_start: str,
    gateway_end: str,
    day_number: int,
    overnight_city: str,
    hotel,
    previous_city: str,
    pass_through_cities: list[str] | None,
    route_start_name: str,
    route_start_latitude: float,
    route_start_longitude: float,
    route_end_name: str,
    route_end_latitude: float,
    route_end_longitude: float,
    drive_minutes_to_next_base: float,
    available_visit_minutes: float,
    drive_time_source: str,
    route_type: str,
    status: str,
    notes: str,
) -> pd.DataFrame:
    if selected_df.empty:
        return pd.DataFrame(columns=_comparison_route_stop_columns())
    output = selected_df.copy().reset_index(drop=True)
    output["comparison_type"] = "method"
    output["comparison_label"] = f"Method · {method_display_name}"
    output["method"] = method
    output["method_display_name"] = method_display_name
    output["trip_days"] = int(trip_days)
    output["gateway_start"] = gateway_start
    output["gateway_end"] = gateway_end
    output["day"] = int(day_number)
    output["stop_order"] = range(1, len(output) + 1)
    output["city"] = output.get("city", overnight_city)
    output["overnight_city"] = overnight_city
    output["hotel_name"] = hotel.get("name", f"{overnight_city} hotel")
    output["hotel_latitude"] = float(hotel.get("latitude", route_end_latitude))
    output["hotel_longitude"] = float(hotel.get("longitude", route_end_longitude))
    output["attraction_name"] = output.get("attraction_name", output.get("name", "selected stop"))
    output["category"] = output.get("category", "attraction")
    output["source_list"] = output.get("source_list", output.get("source", "unknown"))
    output["social_must_go"] = output.get("social_must_go", False)
    output["social_must_go"] = output["social_must_go"].fillna(False).astype(bool)
    output["social_score"] = _numeric_column(output, "social_score")
    output["must_go_weight"] = _numeric_column(output, "must_go_weight")
    output["social_reason"] = output["social_reason"].fillna("") if "social_reason" in output.columns else ""
    output["final_poi_value"] = _numeric_column(output, "final_poi_value", fallback_column="source_score")
    output["route_start_city"] = previous_city
    output["route_start_name"] = route_start_name
    output["route_start_latitude"] = route_start_latitude
    output["route_start_longitude"] = route_start_longitude
    output["route_end_city"] = overnight_city
    output["route_end_name"] = route_end_name
    output["route_end_latitude"] = route_end_latitude
    output["route_end_longitude"] = route_end_longitude
    output["pass_through_cities"] = "; ".join([str(city) for city in (pass_through_cities or [])])
    output["drive_minutes_to_next_base"] = float(drive_minutes_to_next_base)
    output["available_visit_minutes"] = float(available_visit_minutes)
    output["drive_time_source"] = drive_time_source
    output["route_type"] = route_type
    output["status"] = status
    output["notes"] = notes
    for column in _comparison_route_stop_columns():
        if column not in output.columns:
            output[column] = np.nan if column.endswith(("latitude", "longitude")) else ""
    return output[_comparison_route_stop_columns()]


def _intercity_route_metrics(trip: dict) -> tuple[float, float]:
    drive_minutes = float(trip.get("intercity_drive_minutes", 0.0) or 0.0)
    drive_distance_km = drive_minutes / 60.0 * 72.0 / 1.25 if drive_minutes > 0 else 0.0
    return drive_minutes, drive_distance_km


def _write_must_go_coverage(
    *,
    output_dir: Path,
    method_df: pd.DataFrame,
    route_stops_df: pd.DataFrame,
    must_go_candidates: pd.DataFrame,
    config: TripConfig,
) -> pd.DataFrame:
    policy = "mandatory" if bool(config.get("social", "must_go_is_mandatory", False)) else "soft_reward_not_mandatory"
    rows = []
    if must_go_candidates.empty or method_df.empty:
        coverage_df = pd.DataFrame(
            columns=[
                "method",
                "method_display_name",
                "must_go_name",
                "city",
                "social_score",
                "must_go_weight",
                "selected",
                "status",
                "selected_day",
                "selected_stop_order",
                "must_go_policy",
                "social_reason",
            ]
        )
        coverage_df.to_csv(output_dir / "production_must_go_coverage.csv", index=False)
        return coverage_df

    for method_row in method_df.itertuples(index=False):
        method = str(getattr(method_row, "method", ""))
        display = str(getattr(method_row, "method_display_name", method))
        method_routes = route_stops_df[route_stops_df.get("method", pd.Series(dtype=str)).astype(str).eq(method)].copy()
        selected_lookup = {}
        if not method_routes.empty:
            for stop in method_routes.itertuples(index=False):
                selected_lookup[str(getattr(stop, "attraction_name", ""))] = stop
        for candidate in must_go_candidates.itertuples(index=False):
            name = str(getattr(candidate, "name", ""))
            selected_stop = selected_lookup.get(name)
            selected = selected_stop is not None
            rows.append(
                {
                    "method": method,
                    "method_display_name": display,
                    "must_go_name": name,
                    "city": str(getattr(candidate, "city", "")),
                    "social_score": float(getattr(candidate, "social_score", 0.0) or 0.0),
                    "must_go_weight": float(getattr(candidate, "must_go_weight", 0.0) or 0.0),
                    "selected": bool(selected),
                    "status": "selected" if selected else "skipped_soft_reward",
                    "selected_day": int(getattr(selected_stop, "day", 0)) if selected else np.nan,
                    "selected_stop_order": int(getattr(selected_stop, "stop_order", 0)) if selected else np.nan,
                    "must_go_policy": policy,
                    "social_reason": str(getattr(candidate, "social_reason", "")),
                }
            )
    coverage_df = pd.DataFrame(rows)
    coverage_df.to_csv(output_dir / "production_must_go_coverage.csv", index=False)
    return coverage_df


def _write_hotel_selection_debug(
    *,
    output_dir: Path,
    route_stops_df: pd.DataFrame,
    must_go_candidates: pd.DataFrame,
) -> pd.DataFrame:
    hotel_catalog = _load_csv(output_dir / "production_city_hotel_catalog.csv")
    if hotel_catalog.empty:
        hotel_catalog = _load_csv(output_dir / "osm_city_hotel_catalog.csv")
    if hotel_catalog.empty:
        debug_df = pd.DataFrame()
        debug_df.to_csv(output_dir / "production_hotel_selection_debug.csv", index=False)
        return debug_df

    selected_pairs = set()
    selected_by_city: dict[str, str] = {}
    if not route_stops_df.empty:
        for row in route_stops_df[["overnight_city", "hotel_name"]].dropna().drop_duplicates().itertuples(index=False):
            city = str(getattr(row, "overnight_city", ""))
            hotel = str(getattr(row, "hotel_name", ""))
            selected_pairs.add((city, hotel))
            selected_by_city.setdefault(city, hotel)

    rows = []
    for city, city_hotels in hotel_catalog.groupby("city", dropna=False):
        city_name = str(city)
        stop_points = route_stops_df[route_stops_df.get("overnight_city", pd.Series(dtype=str)).astype(str).eq(city_name)]
        must_go_points = must_go_candidates[must_go_candidates.get("city", pd.Series(dtype=str)).astype(str).eq(city_name)]
        for rank, hotel in enumerate(city_hotels.head(12).itertuples(index=False), start=1):
            hotel_name = str(getattr(hotel, "name", f"{city_name} hotel"))
            hotel_lat = float(getattr(hotel, "latitude", np.nan))
            hotel_lon = float(getattr(hotel, "longitude", np.nan))
            if not np.isfinite(hotel_lat) or not np.isfinite(hotel_lon):
                continue
            if not stop_points.empty:
                stop_distance = float(
                    stop_points.apply(
                        lambda row: geodesic((hotel_lat, hotel_lon), (float(row["latitude"]), float(row["longitude"]))).km,
                        axis=1,
                    ).mean()
                )
            else:
                stop_distance = np.nan
            if not must_go_points.empty:
                must_go_distance = float(
                    must_go_points.apply(
                        lambda row: geodesic((hotel_lat, hotel_lon), (float(row["latitude"]), float(row["longitude"]))).km,
                        axis=1,
                    ).mean()
                )
            else:
                must_go_distance = np.nan
            rating = float(getattr(hotel, "stars", getattr(hotel, "rating_score", 0.0)) or 0.0)
            selected = (city_name, hotel_name) in selected_pairs
            hotel_score = (
                0.35 * rating
                - 0.10 * (0.0 if pd.isna(stop_distance) else stop_distance)
                - 0.08 * (0.0 if pd.isna(must_go_distance) else must_go_distance)
                + (1.50 if selected else 0.0)
                - 0.03 * rank
            )
            rows.append(
                {
                    "city": city_name,
                    "hotel_name": hotel_name,
                    "candidate_rank": int(rank),
                    "selected": bool(selected),
                    "selected_for_city": selected_by_city.get(city_name, ""),
                    "latitude": hotel_lat,
                    "longitude": hotel_lon,
                    "source": str(getattr(hotel, "source", "unknown")),
                    "rating_component": rating,
                    "mean_distance_to_selected_stops_km": stop_distance,
                    "mean_distance_to_must_go_km": must_go_distance,
                    "hotel_score": float(hotel_score),
                    "selected_hotel_reason": "selected route base" if selected else "candidate alternative",
                }
            )
    debug_df = pd.DataFrame(rows).sort_values(["city", "selected", "hotel_score"], ascending=[True, False, False]).reset_index(drop=True)
    debug_df.to_csv(output_dir / "production_hotel_selection_debug.csv", index=False)
    return debug_df


def _hierarchical_method_route_outputs(
    *,
    context: dict,
    blueprint_trip_map,
    trip: dict,
    config: TripConfig,
    method: str,
    strategy_name: str,
    local_solver: str,
    budget_df: pd.DataFrame | None = None,
    profile_name: str = "balanced",
) -> tuple[pd.DataFrame, dict]:
    method_display_name = _method_display_name(method)
    profile_name = str(profile_name or "balanced").lower()
    profile_config = blueprint_trip_map._profile_config(profile_name)
    stops_per_day = min(int(profile_config["stops_per_day"]), int(config.get("optimization", "max_pois_per_day", 4)))
    candidate_size = max(12, stops_per_day * 4)
    sequence = blueprint_trip_map._trip_sequence_with_pass_through(trip)
    days_by_city = blueprint_trip_map._coerce_days_by_city(trip.get("days_by_city", {}))
    base_cities = [city for city in sequence if int(days_by_city.get(city, 0)) > 0]
    used_names_by_city = {city: set() for city in sequence}
    selected_hotels_by_city = {}
    poi_catalogs_by_city = {}
    route_frames = []
    day_statuses = []
    day_gaps = []
    total_solve_seconds = 0.0
    total_objective = 0.0
    notes = []
    trip_days = get_canonical_trip_days(config, trip)

    for city in base_cities:
        allocated_days = int(days_by_city.get(city, 0))
        if allocated_days <= 0:
            continue
        poi_catalog = blueprint_trip_map._city_poi_catalog(context, city).reset_index(drop=True)
        hotel_catalog = blueprint_trip_map._city_hotel_catalog(context, city).reset_index(drop=True)
        poi_catalogs_by_city[city] = poi_catalog
        used_names = used_names_by_city.setdefault(city, set())
        hotel = blueprint_trip_map._select_hotel_for_city_plan(
            hotel_catalog,
            poi_catalog,
            used_names,
            stops_per_day=stops_per_day,
            profile_config=profile_config,
        )
        selected_hotels_by_city[city] = hotel

    budget_context = build_canonical_budget_context(
        trip,
        budget_df,
        selected_hotels_by_city,
        config=config,
    )
    remaining_route_budget = float(budget_context["remaining_route_budget"])

    for base_index, city in enumerate(base_cities):
        allocated_days = int(days_by_city.get(city, 0))
        if allocated_days <= 0:
            continue
        poi_catalog = poi_catalogs_by_city.get(city, pd.DataFrame())
        hotel = selected_hotels_by_city.get(city)
        for local_day in range(1, allocated_days + 1):
            day_number = len(day_statuses) + 1
            is_transition_day = bool(local_day == 1 and base_index > 0)
            previous_city = base_cities[base_index - 1] if is_transition_day else city
            previous_hotel = selected_hotels_by_city.get(previous_city, hotel)
            route_start_name = previous_hotel.get("name", f"{previous_city} hotel")
            route_start_latitude = float(previous_hotel.get("latitude", blueprint_trip_map.CITY_COORDS[previous_city][0]))
            route_start_longitude = float(previous_hotel.get("longitude", blueprint_trip_map.CITY_COORDS[previous_city][1]))
            route_end_name = hotel.get("name", f"{city} hotel")
            route_end_latitude = float(hotel.get("latitude", blueprint_trip_map.CITY_COORDS[city][0]))
            route_end_longitude = float(hotel.get("longitude", blueprint_trip_map.CITY_COORDS[city][1]))
            route_type = "base_city_local"
            pass_through_cities = []
            drive_minutes = 0.0
            drive_time_source = "local_day_no_intercity_drive"
            if is_transition_day:
                segment = blueprint_trip_map._sequence_between(sequence, previous_city, city)
                pass_through_cities = [stop_city for stop_city in segment[1:-1] if stop_city != city]
                drive_minutes, drive_time_source = blueprint_trip_map._transition_drive_minutes(
                    context,
                    previous_city,
                    city,
                    pass_through_cities,
                )
                route_type = "relocation_pass_through" if pass_through_cities else "relocation_direct"

            available_visit_minutes = max(60.0, 720.0 - float(drive_minutes) - 45.0)
            candidate_pool = _candidate_bundle_for_hierarchical_day(
                context=context,
                blueprint_trip_map=blueprint_trip_map,
                overnight_city=city,
                hotel=hotel,
                used_names_by_city=used_names_by_city,
                pass_through_cities=pass_through_cities,
                strategy_name=strategy_name,
                max_candidates=candidate_size,
                previous_city=previous_city,
            )

            day_route_budget = max(0.0, float(remaining_route_budget))
            if local_solver == "legacy_gurobi":
                selected_df, day_result = _solve_legacy_local_gurobi_day_route(
                    candidate_pool,
                    config=config,
                    available_visit_minutes=available_visit_minutes,
                    remaining_budget=day_route_budget,
                    max_stops=stops_per_day,
                    route_start=(route_start_latitude, route_start_longitude),
                    route_end=(route_end_latitude, route_end_longitude),
                    profile_name=profile_name,
                )
                route_type_label = f"{route_type}_legacy_local_gurobi"
            elif local_solver == "small_gurobi":
                selected_df, day_result = _solve_small_gurobi_day_route(
                    candidate_pool,
                    config=config,
                    available_visit_minutes=available_visit_minutes,
                    remaining_budget=day_route_budget,
                    max_stops=stops_per_day,
                    route_start=(route_start_latitude, route_start_longitude),
                )
                route_type_label = f"{route_type}_small_gurobi_repair"
            else:
                selected_df, day_result = _solve_greedy_day_route(
                    candidate_pool,
                    route_start=(route_start_latitude, route_start_longitude),
                    route_end=(route_end_latitude, route_end_longitude),
                    available_visit_minutes=available_visit_minutes,
                    max_stops=stops_per_day,
                )
                route_type_label = f"{route_type}_hierarchical_greedy"

            if selected_df.empty and not poi_catalog.empty:
                selected_df = poi_catalog.head(1).copy()
                selected_df["attraction_name"] = selected_df.get("name", "selected stop")
                day_result.setdefault("solver_status", "FALLBACK")
                notes.append(
                    f"Day {day_number} fell back to the top-ranked city POI because no feasible route was returned."
                )

            selected_cost = _estimated_route_stop_cost(selected_df)
            remaining_route_budget = max(0.0, float(remaining_route_budget) - float(selected_cost))

            for row in selected_df.itertuples(index=False):
                used_names_by_city.setdefault(str(getattr(row, "city", city)), set()).add(
                    str(getattr(row, "name", getattr(row, "attraction_name", "")))
                )

            solver_status = str(day_result.get("solver_status", "UNKNOWN"))
            route_frames.append(
                _build_route_rows_from_selected(
                    selected_df,
                    method=method,
                    method_display_name=method_display_name,
                    trip_days=trip_days,
                    gateway_start=str(trip.get("gateway_start", "")),
                    gateway_end=str(trip.get("gateway_end", "")),
                    day_number=day_number,
                    overnight_city=city,
                    hotel=hotel,
                    previous_city=previous_city,
                    pass_through_cities=pass_through_cities,
                    route_start_name=route_start_name,
                    route_start_latitude=route_start_latitude,
                    route_start_longitude=route_start_longitude,
                    route_end_name=route_end_name,
                    route_end_latitude=route_end_latitude,
                    route_end_longitude=route_end_longitude,
                    drive_minutes_to_next_base=drive_minutes,
                    available_visit_minutes=available_visit_minutes,
                    drive_time_source=drive_time_source,
                    route_type=route_type_label,
                    status=solver_status,
                    notes=(
                        f"{method_display_name} local route generated with strategy={strategy_name}; "
                        f"budget source={budget_context['budget_source']}; "
                        f"remaining route budget after day={remaining_route_budget:.2f}."
                    ),
                )
            )
            day_statuses.append(solver_status)
            total_solve_seconds += float(day_result.get("runtime_seconds", 0.0) or 0.0)
            objective_value = day_result.get("objective_value", np.nan)
            if pd.notna(objective_value):
                total_objective += float(objective_value)
            gap_value = day_result.get("optimality_gap", np.nan)
            if pd.notna(gap_value):
                day_gaps.append(float(gap_value))

    route_df = pd.concat(route_frames, ignore_index=True, sort=False) if route_frames else pd.DataFrame(columns=_comparison_route_stop_columns())
    status = "FAILED"
    if not route_df.empty:
        lowered_statuses = [str(day_status).lower() for day_status in day_statuses]
        if any("no route" in day_status or day_status == "failed" for day_status in lowered_statuses):
            status = "FAILED"
        elif lowered_statuses and all(day_status == "heuristic" for day_status in lowered_statuses):
            status = "HEURISTIC"
        elif any("heuristic" in day_status or "after_gurobi_error" in day_status for day_status in lowered_statuses):
            status = "HEURISTIC_FALLBACK"
        elif day_statuses:
            status = day_statuses[0] if len(set(day_statuses)) == 1 else "MIXED"
    return route_df, {
        "status": status,
        "solve_seconds": total_solve_seconds,
        "mip_gap": float(np.nanmax(day_gaps)) if day_gaps else np.nan,
        "objective": total_objective if total_objective else np.nan,
        "notes": "; ".join(notes) if notes else "",
        "selected_hotels_by_city": selected_hotels_by_city,
        **budget_context,
    }


def _bandit_route_stop_rows(
    *,
    context: dict,
    best_trip: dict,
    config: TripConfig,
    output_dir: Path,
    enriched_df: pd.DataFrame,
    budget_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict]:
    blueprint_trip_map = import_legacy_module("blueprint_trip_map")
    bandit_runs_df = _load_frame_from_context_or_csv(context, ["bandit_stress_runs_df"], output_dir / "production_bandit_stress_runs.csv")
    bandit_summary_df = _load_frame_from_context_or_csv(
        context,
        ["hybrid_bandit_summary_df"],
        output_dir / "production_hybrid_bandit_optimization_summary.csv",
    )
    method = "hierarchical_bandit_gurobi_repair"
    method_display_name = _method_display_name(method)
    canonical_budget = build_canonical_budget_context(best_trip, budget_df, None, config=config)
    summary_row = {
        "method": method,
        "method_display_name": method_display_name,
        "allocation_solver": "bandit_master",
        "local_route_solver": "small_gurobi_repair",
        "trip_days": get_canonical_trip_days(config, best_trip),
        "gateway_start": best_trip.get("gateway_start"),
        "gateway_end": best_trip.get("gateway_end"),
        "city_sequence": " -> ".join(best_trip.get("city_sequence", [])),
        "days_by_city": str(best_trip.get("days_by_city", {})),
        "total_budget": canonical_budget["total_budget"],
        "budget_used": np.nan,
        "budget_source": canonical_budget["budget_source"],
        "total_utility": np.nan,
        "total_waiting_time": np.nan,
        "total_travel_time": np.nan,
        "total_travel_distance_km": np.nan,
        "total_cost": np.nan,
        "selected_attractions": 0,
        "must_go_count": 0,
        "diversity_score": np.nan,
        "comparison_score": np.nan,
        "objective": np.nan,
        "posterior_reward": np.nan,
        "solve_seconds": np.nan,
        "mip_gap": np.nan,
        "status": "FAILED",
        "notes": "Hierarchical bandit + small Gurobi comparison could not be generated.",
    }
    if bandit_runs_df.empty or bandit_summary_df.empty:
        summary_row["notes"] = "Bandit stress or bandit optimization summary artifacts are missing or empty."
        return pd.DataFrame(columns=_comparison_route_stop_columns()), summary_row

    summary_rank = bandit_summary_df.sort_values(
        ["posterior_mean_reward", "hierarchical_objective", "estimated_total_cost"],
        ascending=[False, False, True],
    )
    best_summary_row = summary_rank.iloc[0]
    best_arm_id = str(best_summary_row.get("arm_id"))
    runs_for_arm = bandit_runs_df[bandit_runs_df["arm_id"].astype(str).eq(best_arm_id)].copy()
    if not runs_for_arm.empty and "route_feasible" in runs_for_arm.columns:
        runs_for_arm = runs_for_arm.sort_values(
            ["route_feasible", "combined_reward", "posterior_mean_reward"],
            ascending=[False, False, False],
        )
    best_run_row = runs_for_arm.iloc[0] if not runs_for_arm.empty else pd.Series(dtype=object)

    bandit_trip_row = dict(best_summary_row)
    bandit_trip_row["OUTPUT_DIR"] = output_dir
    bandit_trip = _trip_from_bandit_row(bandit_trip_row, blueprint_trip_map)
    route_stops, route_info = _hierarchical_method_route_outputs(
        context=context,
        blueprint_trip_map=blueprint_trip_map,
        trip=bandit_trip,
        config=config,
        method=method,
        strategy_name=str(best_summary_row.get("route_search_strategy", "balanced")),
        local_solver="small_gurobi",
        budget_df=budget_df,
    )
    if route_stops.empty:
        summary_row["notes"] = "Best bandit arm could be identified, but no hierarchical route-stop rows were produced."
        summary_row["status"] = str(route_info.get("status", "FAILED"))
        return pd.DataFrame(columns=_comparison_route_stop_columns()), summary_row

    metrics = _route_metrics_from_stops(route_stops)
    intercity_minutes, intercity_distance = _intercity_route_metrics(bandit_trip)
    attraction_cost = _estimated_route_stop_cost(route_stops)
    repaired_total_cost = (
        float(route_info.get("reserved_food_transport", canonical_budget["reserved_food_transport"]))
        + float(route_info.get("committed_hotel_cost", canonical_budget["committed_hotel_cost"]))
        + float(attraction_cost)
    )
    selected_arm_estimated_cost = float(best_summary_row.get("estimated_total_cost", np.nan))
    summary_row.update(
        {
            **_trip_summary_fields(bandit_trip),
            "allocation_solver": "bandit_master",
            "local_route_solver": "small_gurobi_repair",
            "trip_days": get_canonical_trip_days(config, bandit_trip),
            "total_budget": route_info.get("total_budget", canonical_budget["total_budget"]),
            "budget_used": float(repaired_total_cost),
            "budget_source": route_info.get("budget_source", canonical_budget["budget_source"]),
            "total_utility": metrics["total_utility"],
            "total_waiting_time": metrics["total_waiting_time"],
            "total_travel_time": metrics["total_travel_time"] + intercity_minutes,
            "total_travel_distance_km": metrics["total_travel_distance_km"] + intercity_distance,
            "total_cost": float(repaired_total_cost),
            "selected_attractions": metrics["selected_attractions"],
            "must_go_count": int(route_stops["social_must_go"].astype(bool).sum()) if "social_must_go" in route_stops.columns else 0,
            "diversity_score": metrics["diversity_score"],
            "objective": float(best_run_row.get("combined_reward", best_summary_row.get("hierarchical_objective", np.nan)) or best_summary_row.get("hierarchical_objective", np.nan)),
            "posterior_reward": float(best_summary_row.get("posterior_mean_reward", np.nan)),
            "solve_seconds": float(route_info.get("solve_seconds", 0.0) or 0.0) + float(best_run_row.get("runtime_seconds", 0.0) or 0.0),
            "mip_gap": route_info.get("mip_gap", np.nan),
            "status": str(route_info.get("status", best_run_row.get("solver_status", "FAILED"))),
            "notes": (
                f"Best hierarchical bandit arm `{best_arm_id}` with strategy={best_summary_row.get('route_search_strategy', 'balanced')}; "
                f"posterior reward={float(best_summary_row.get('posterior_mean_reward', np.nan)):.3f}; "
                f"selected-arm estimated total cost={selected_arm_estimated_cost:.2f}; "
                f"repaired canonical route cost={repaired_total_cost:.2f}."
            ),
        }
    )
    summary_row["budget_used"] = summary_row["total_cost"]
    return route_stops, summary_row


ROUTE_MATRIX_SUMMARY_COLUMNS = [
    "route_key",
    "comparison_type",
    "comparison_label",
    "trip_days",
    "method",
    "method_display_name",
    "method_short_label",
    "profile",
    "profile_label",
    "allocation_solver",
    "local_route_solver",
    "gateway_start",
    "gateway_end",
    "city_sequence",
    "days_by_city",
    "total_budget",
    "budget_used",
    "budget_source",
    "total_utility",
    "total_waiting_time",
    "total_travel_time",
    "total_travel_distance_km",
    "total_cost",
    "selected_attractions",
    "must_go_count",
    "must_go_total",
    "must_go_selected_count",
    "must_go_skipped_count",
    "must_go_coverage_ratio",
    "must_go_skipped_names",
    "must_go_policy",
    "diversity_score",
    "utility_norm",
    "must_go_norm",
    "diversity_norm",
    "travel_efficiency_norm",
    "cost_efficiency_norm",
    "waiting_efficiency_norm",
    "comparison_score",
    "comparison_score_formula",
    "objective",
    "posterior_reward",
    "solve_seconds",
    "mip_gap",
    "status",
    "selected_strategy",
    "hotel_showcase_group",
    "selector_parent",
    "route_layer_var",
    "hotel_layer_var",
    "city_detail_layer_vars",
    "transition_layer_vars",
    "notes",
]


def _route_matrix_stop_columns() -> list[str]:
    return [
        "route_key",
        "profile",
        "profile_label",
        "selector_parent",
        "hotel_showcase_group",
        *_comparison_route_stop_columns(),
    ]


def _annotate_route_matrix_stops(
    route_stops: pd.DataFrame,
    *,
    route_key: str,
    trip_days: int,
    method: str,
    profile: str,
    profile_label: str,
    comparison_label: str,
    selector_parent: str,
) -> pd.DataFrame:
    if route_stops.empty:
        return pd.DataFrame(columns=_route_matrix_stop_columns())
    output = route_stops.copy()
    output["route_key"] = route_key
    output["profile"] = profile
    output["profile_label"] = profile_label
    output["selector_parent"] = selector_parent
    output["hotel_showcase_group"] = route_key
    output["comparison_type"] = "route_matrix"
    output["comparison_label"] = comparison_label
    output["method"] = method
    output["method_display_name"] = _method_display_name(method)
    output["trip_days"] = int(trip_days)
    for column in _route_matrix_stop_columns():
        if column not in output.columns:
            output[column] = np.nan if column.endswith(("latitude", "longitude")) else ""
    return output[_route_matrix_stop_columns()]


def _route_matrix_summary_row(
    *,
    route_key: str,
    trip_days: int,
    method: str,
    profile: str,
    profile_label: str,
    comparison_label: str,
    trip_for_method: dict,
    route_stops: pd.DataFrame,
    route_info: dict,
    budget_df: pd.DataFrame,
    must_go_candidates: pd.DataFrame,
    config: TripConfig,
    objective_value: float,
    posterior_reward: float,
    selected_strategy: str,
    notes: str,
) -> dict:
    metrics = _route_metrics_from_stops(route_stops)
    must_go_summary = _route_must_go_summary(route_stops, must_go_candidates, config)
    intercity_minutes, intercity_distance = _intercity_route_metrics(trip_for_method)
    attraction_cost = _estimated_route_stop_cost(route_stops)
    total_cost = (
        float(route_info.get("reserved_food_transport", 0.0))
        + float(route_info.get("committed_hotel_cost", 0.0))
        + float(attraction_cost)
    )
    allocation_solver, local_route_solver = _method_solver_roles(method)
    return {
        "route_key": route_key,
        "comparison_type": "route_matrix",
        "comparison_label": comparison_label,
        "trip_days": int(trip_days),
        "method": method,
        "method_display_name": _method_display_name(method),
        "method_short_label": _method_short_label(method),
        "profile": profile,
        "profile_label": profile_label,
        "allocation_solver": allocation_solver,
        "local_route_solver": local_route_solver,
        **_trip_summary_fields(trip_for_method),
        "total_budget": float(route_info.get("total_budget", get_canonical_trip_budget(config, budget_df))),
        "budget_used": float(total_cost),
        "budget_source": str(route_info.get("budget_source", _canonical_budget_value_and_source(config, budget_df)[1])),
        "total_utility": metrics["total_utility"],
        "total_waiting_time": metrics["total_waiting_time"],
        "total_travel_time": metrics["total_travel_time"] + intercity_minutes,
        "total_travel_distance_km": metrics["total_travel_distance_km"] + intercity_distance,
        "total_cost": float(total_cost),
        "selected_attractions": metrics["selected_attractions"],
        "must_go_count": int(route_stops["social_must_go"].astype(bool).sum()) if not route_stops.empty and "social_must_go" in route_stops.columns else 0,
        **must_go_summary,
        "diversity_score": metrics["diversity_score"],
        "objective": float(objective_value) if pd.notna(objective_value) else np.nan,
        "posterior_reward": float(posterior_reward) if pd.notna(posterior_reward) else np.nan,
        "solve_seconds": float(route_info.get("solve_seconds", np.nan) or 0.0),
        "mip_gap": route_info.get("mip_gap", np.nan),
        "status": str(route_info.get("status", "FAILED")),
        "selected_strategy": selected_strategy,
        "hotel_showcase_group": route_key,
        "selector_parent": f"d{int(trip_days)}__{METHOD_ROUTE_KEY_SLUG.get(method, method)}",
        "route_layer_var": "",
        "hotel_layer_var": "",
        "city_detail_layer_vars": "",
        "transition_layer_vars": "",
        "notes": notes,
    }


def _write_route_matrix_hotel_debug(
    *,
    output_dir: Path,
    route_stops_df: pd.DataFrame,
    must_go_candidates: pd.DataFrame,
) -> pd.DataFrame:
    hotel_catalog = _load_csv(output_dir / "production_city_hotel_catalog.csv")
    if hotel_catalog.empty:
        hotel_catalog = _load_csv(output_dir / "osm_city_hotel_catalog.csv")
    if hotel_catalog.empty or route_stops_df.empty:
        debug_df = pd.DataFrame()
        debug_df.to_csv(output_dir / "production_route_matrix_hotel_selection_debug.csv", index=False)
        return debug_df

    rows = []
    for route_key, route_rows in route_stops_df.groupby("route_key", dropna=False):
        route_key = str(route_key)
        selected_pairs = {
            (str(row.overnight_city), str(row.hotel_name))
            for row in route_rows[["overnight_city", "hotel_name"]].dropna().drop_duplicates().itertuples(index=False)
        }
        route_meta = route_rows.iloc[0]
        for city, city_hotels in hotel_catalog.groupby("city", dropna=False):
            city_name = str(city)
            if not route_rows["overnight_city"].astype(str).eq(city_name).any():
                continue
            stop_points = route_rows[route_rows["overnight_city"].astype(str).eq(city_name)]
            must_go_points = must_go_candidates[must_go_candidates.get("city", pd.Series(dtype=str)).astype(str).eq(city_name)]
            for rank, hotel in enumerate(city_hotels.head(10).itertuples(index=False), start=1):
                hotel_name = str(getattr(hotel, "name", f"{city_name} hotel"))
                hotel_lat = float(getattr(hotel, "latitude", np.nan))
                hotel_lon = float(getattr(hotel, "longitude", np.nan))
                if not np.isfinite(hotel_lat) or not np.isfinite(hotel_lon):
                    continue
                if not stop_points.empty:
                    stop_distance = float(
                        stop_points.apply(
                            lambda row: geodesic((hotel_lat, hotel_lon), (float(row["latitude"]), float(row["longitude"]))).km,
                            axis=1,
                        ).mean()
                    )
                else:
                    stop_distance = np.nan
                if not must_go_points.empty:
                    must_go_distance = float(
                        must_go_points.apply(
                            lambda row: geodesic((hotel_lat, hotel_lon), (float(row["latitude"]), float(row["longitude"]))).km,
                            axis=1,
                        ).mean()
                    )
                else:
                    must_go_distance = np.nan
                rating = float(getattr(hotel, "stars", getattr(hotel, "rating_score", 0.0)) or 0.0)
                selected = (city_name, hotel_name) in selected_pairs
                hotel_score = (
                    0.35 * rating
                    - 0.10 * (0.0 if pd.isna(stop_distance) else stop_distance)
                    - 0.08 * (0.0 if pd.isna(must_go_distance) else must_go_distance)
                    + (1.50 if selected else 0.0)
                    - 0.03 * rank
                )
                rows.append(
                    {
                        "route_key": route_key,
                        "trip_days": int(route_meta.get("trip_days", 0) or 0),
                        "method": str(route_meta.get("method", "")),
                        "profile": str(route_meta.get("profile", "")),
                        "comparison_label": str(route_meta.get("comparison_label", "")),
                        "city": city_name,
                        "hotel_name": hotel_name,
                        "candidate_rank": int(rank),
                        "selected": bool(selected),
                        "latitude": hotel_lat,
                        "longitude": hotel_lon,
                        "source": str(getattr(hotel, "source", "unknown")),
                        "rating_component": rating,
                        "mean_distance_to_selected_stops_km": stop_distance,
                        "mean_distance_to_must_go_km": must_go_distance,
                        "hotel_score": float(hotel_score),
                        "selected_hotel_reason": "selected route base" if selected else "candidate alternative",
                    }
                )
    debug_df = pd.DataFrame(rows)
    if not debug_df.empty:
        debug_df = debug_df.sort_values(
            ["route_key", "city", "selected", "hotel_score"],
            ascending=[True, True, False, False],
        ).reset_index(drop=True)
    debug_df.to_csv(output_dir / "production_route_matrix_hotel_selection_debug.csv", index=False)
    return debug_df


def _write_route_matrix_compatibility_artifacts(
    *,
    output_dir: Path,
    matrix_df: pd.DataFrame,
    matrix_route_stops_df: pd.DataFrame,
    config: TripConfig,
    must_go_candidates: pd.DataFrame,
) -> None:
    canonical_days = int(config.get("trip", "trip_days", 7))
    trip_subset = matrix_route_stops_df[
        matrix_route_stops_df["method"].astype(str).eq("hierarchical_bandit_gurobi_repair")
        & matrix_route_stops_df["profile"].astype(str).eq("balanced")
    ].copy()
    if not trip_subset.empty:
        trip_subset["comparison_type"] = "trip_length"
        trip_subset["comparison_label"] = trip_subset["trip_days"].astype(int).map(
            lambda days: f"Trip Length · {days}-Day Hybrid Bandit + Small Gurobi"
        )
        trip_subset = trip_subset[_comparison_route_stop_columns()]
    trip_subset.to_csv(output_dir / "production_trip_length_route_stops.csv", index=False)

    trip_summary = matrix_df[
        matrix_df["method"].astype(str).eq("hierarchical_bandit_gurobi_repair")
        & matrix_df["profile"].astype(str).eq("balanced")
    ].copy()
    if not trip_summary.empty:
        trip_summary["comparison_type"] = "trip_length"
        trip_summary["comparison_label"] = trip_summary["trip_days"].astype(int).map(
            lambda days: f"Trip Length · {days}-Day Hybrid Bandit + Small Gurobi"
        )
        trip_summary["estimated_budget"] = trip_summary["total_budget"]
        trip_summary["intercity_drive_minutes"] = pd.to_numeric(trip_summary["total_travel_time"], errors="coerce").fillna(0.0)
        trip_summary["intercity_drive_hours"] = trip_summary["intercity_drive_minutes"] / 60.0
        trip_summary["solver_status"] = trip_summary["status"]
        trip_summary = trip_summary[
            [
                "comparison_type",
                "comparison_label",
                "trip_days",
                "gateway_start",
                "gateway_end",
                "city_sequence",
                "days_by_city",
                "intercity_drive_minutes",
                "intercity_drive_hours",
                "estimated_budget",
                "budget_source",
                "objective",
                "posterior_reward",
                "selected_strategy",
                "solver_status",
                "status",
                "notes",
            ]
        ]
    trip_summary.to_csv(output_dir / "production_trip_length_comparison.csv", index=False)

    method_subset = matrix_route_stops_df[
        pd.to_numeric(matrix_route_stops_df["trip_days"], errors="coerce").fillna(0).astype(int).eq(canonical_days)
        & matrix_route_stops_df["profile"].astype(str).eq("balanced")
    ].copy()
    if not method_subset.empty:
        method_subset["comparison_type"] = "method"
        method_subset["comparison_label"] = "Method · " + method_subset["method_display_name"].astype(str)
        method_subset = method_subset[_comparison_route_stop_columns()]
    method_subset.to_csv(output_dir / "production_method_route_stops.csv", index=False)

    method_summary = matrix_df[
        pd.to_numeric(matrix_df["trip_days"], errors="coerce").fillna(0).astype(int).eq(canonical_days)
        & matrix_df["profile"].astype(str).eq("balanced")
    ].copy()
    if not method_summary.empty:
        method_summary = method_summary[
            [
                "method",
                "method_display_name",
                "allocation_solver",
                "local_route_solver",
                "trip_days",
                "gateway_start",
                "gateway_end",
                "city_sequence",
                "days_by_city",
                "total_budget",
                "budget_used",
                "budget_source",
                "total_utility",
                "total_waiting_time",
                "total_travel_time",
                "total_travel_distance_km",
                "total_cost",
                "selected_attractions",
                "must_go_count",
                "must_go_total",
                "must_go_selected_count",
                "must_go_skipped_count",
                "must_go_coverage_ratio",
                "must_go_skipped_names",
                "must_go_policy",
                "diversity_score",
                "utility_norm",
                "must_go_norm",
                "diversity_norm",
                "travel_efficiency_norm",
                "cost_efficiency_norm",
                "waiting_efficiency_norm",
                "comparison_score",
                "comparison_score_formula",
                "objective",
                "posterior_reward",
                "solve_seconds",
                "mip_gap",
                "status",
                "notes",
            ]
        ]
    method_summary.to_csv(output_dir / "production_method_comparison.csv", index=False)
    _write_must_go_coverage(
        output_dir=output_dir,
        method_df=method_summary if not method_summary.empty else pd.DataFrame(),
        route_stops_df=method_subset if not method_subset.empty else pd.DataFrame(),
        must_go_candidates=must_go_candidates,
        config=config,
    )
    _write_hotel_selection_debug(
        output_dir=output_dir,
        route_stops_df=method_subset if not method_subset.empty else pd.DataFrame(),
        must_go_candidates=must_go_candidates,
    )


def build_route_matrix_comparison(
    *,
    context: dict,
    config: TripConfig,
    output_dir: str | Path | None = None,
    trip_days_grid: list[int] | None = None,
    primary_city: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    blueprint_trip_map = import_legacy_module("blueprint_trip_map")
    output_dir = Path(output_dir or _first_context_value(context, "OUTPUT_DIR", default="results/outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)
    city_summary_df = _load_frame_from_context_or_csv(
        context,
        ["city_catalog_summary_df"],
        output_dir / "production_city_catalog_summary.csv",
    )
    enriched_df = _load_frame_from_context_or_csv(
        context,
        ["production_enriched_poi_catalog_df"],
        output_dir / "production_enriched_poi_catalog.csv",
    )
    hotels_df = _load_frame_from_context_or_csv(context, ["hotels_df"], output_dir / "hotel_dataset.csv")
    osm_city_hotel_catalog_df = _load_frame_from_context_or_csv(
        context,
        ["osm_city_hotel_catalog_df"],
        output_dir / "production_city_hotel_catalog.csv",
    )
    if city_summary_df.empty:
        empty_summary = pd.DataFrame(columns=ROUTE_MATRIX_SUMMARY_COLUMNS)
        empty_stops = pd.DataFrame(columns=_route_matrix_stop_columns())
        empty_summary.to_csv(output_dir / "production_route_matrix_comparison.csv", index=False)
        empty_stops.to_csv(output_dir / "production_route_matrix_route_stops.csv", index=False)
        return empty_summary, empty_stops

    raw_days = trip_days_grid or config.get("trip_duration_comparison", "days", [7, 9, 12])
    day_options = sorted({int(value) for value in raw_days if int(value) > 0}) or [7, 9, 12]
    base_days = max(1, int(config.get("trip", "trip_days", 7)))
    base_budget = float(config.get("budget", "user_budget", 2000.0))
    scale_budget = bool(config.get("trip_duration_comparison", "scale_budget_with_days", True))
    city_name = str(primary_city or _first_context_value(context, "CITY", default="Santa Barbara"))
    bandit_summary_df = _load_frame_from_context_or_csv(
        context,
        ["hybrid_bandit_summary_df"],
        output_dir / "production_hybrid_bandit_optimization_summary.csv",
    )
    if not bandit_summary_df.empty and "posterior_mean_reward" in bandit_summary_df.columns:
        best_bandit_row = bandit_summary_df.sort_values("posterior_mean_reward", ascending=False).iloc[0]
        default_bandit_strategy = str(best_bandit_row.get("route_search_strategy", "scenic_social"))
        posterior_reward = float(best_bandit_row.get("posterior_mean_reward", np.nan))
    else:
        default_bandit_strategy = "scenic_social"
        posterior_reward = np.nan
    must_go_candidates = _must_go_candidates_from_output(output_dir, enriched_df)

    trip_cache: dict[tuple[int, str], tuple[dict, pd.DataFrame, TripConfig]] = {}
    summary_rows = []
    route_frames = []
    for trip_days in day_options:
        scaled_budget = base_budget * trip_days / base_days if scale_budget else base_budget
        duration_config = TripConfig(
            data=_merge_config_data(
                config.to_dict(),
                {
                    "trip": {"trip_days": trip_days},
                    "budget": {"user_budget": scaled_budget},
                },
            ),
            source_path=config.source_path,
        )
        for method in MATRIX_METHOD_ORDER:
            try:
                if method == "hierarchical_greedy_baseline":
                    trip_for_method, plan_df = solve_hierarchical_trip_with_greedy(duration_config, city_summary_df)
                else:
                    trip_for_method, plan_df = solve_hierarchical_trip_with_gurobi(duration_config, city_summary_df)
            except Exception:
                trip_for_method, plan_df = solve_hierarchical_trip_with_gurobi(duration_config, city_summary_df)
            trip_cache[(trip_days, method)] = (trip_for_method, plan_df, duration_config)
            budget_df = estimate_budget_range(
                trip=trip_for_method,
                hotels_df=hotels_df,
                osm_city_hotel_catalog_df=osm_city_hotel_catalog_df,
                output_dir=output_dir,
                config=duration_config,
                primary_city=city_name,
                write_output=False,
            )
            for profile in MATRIX_PROFILE_ORDER:
                profile_label = _profile_label(blueprint_trip_map, profile)
                route_key = _route_key(trip_days, method, profile)
                selector_parent = f"d{int(trip_days)}__{METHOD_ROUTE_KEY_SLUG.get(method, method)}"
                comparison_label = f"{int(trip_days)}-Day · {_method_short_label(method)} · {profile_label}"
                if method == "hierarchical_bandit_gurobi_repair":
                    local_solver = "small_gurobi"
                    strategy_name = default_bandit_strategy if profile == "balanced" else _profile_strategy(profile, default_bandit_strategy)
                    selected_strategy = f"bandit_repair:{strategy_name}"
                elif method == "hierarchical_greedy_baseline":
                    local_solver = "greedy"
                    strategy_name = _profile_strategy(profile)
                    selected_strategy = f"greedy:{strategy_name}"
                    solver_note = "greedy city/base/day allocation and greedy local route"
                else:
                    local_solver = "legacy_gurobi"
                    strategy_name = _profile_strategy(profile)
                    selected_strategy = f"gurobi:{strategy_name}"
                    solver_note = "gurobi city/base/day allocation and legacy local gurobi route"
                if method == "hierarchical_bandit_gurobi_repair":
                    solver_note = "bandit strategy selection with small gurobi repair"
                route_context = dict(context)
                route_context["OUTPUT_DIR"] = output_dir
                route_context["best_hierarchical_trip"] = trip_for_method
                route_stops, route_info = _hierarchical_method_route_outputs(
                    context=route_context,
                    blueprint_trip_map=blueprint_trip_map,
                    trip=trip_for_method,
                    config=duration_config,
                    method=method,
                    strategy_name=strategy_name,
                    local_solver=local_solver,
                    budget_df=budget_df,
                    profile_name=profile,
                )
                route_stops = _annotate_route_matrix_stops(
                    route_stops,
                    route_key=route_key,
                    trip_days=trip_days,
                    method=method,
                    profile=profile,
                    profile_label=profile_label,
                    comparison_label=comparison_label,
                    selector_parent=selector_parent,
                )
                if not route_stops.empty:
                    route_frames.append(route_stops)
                objective_value = float(trip_for_method.get("objective", route_info.get("objective", np.nan)))
                route_posterior = posterior_reward if method == "hierarchical_bandit_gurobi_repair" else np.nan
                summary_rows.append(
                    _route_matrix_summary_row(
                        route_key=route_key,
                        trip_days=trip_days,
                        method=method,
                        profile=profile,
                        profile_label=profile_label,
                        comparison_label=comparison_label,
                        trip_for_method=trip_for_method,
                        route_stops=route_stops,
                        route_info=route_info,
                        budget_df=budget_df,
                        must_go_candidates=must_go_candidates,
                        config=duration_config,
                        objective_value=objective_value,
                        posterior_reward=route_posterior,
                        selected_strategy=selected_strategy,
                        notes=(
                            f"Canonical route matrix row for {comparison_label}; "
                            f"{solver_note}; profile controls stops/day and hotel scoring; strategy={strategy_name}."
                        ),
                    )
                )

    matrix_df = pd.DataFrame(summary_rows)
    for column in ROUTE_MATRIX_SUMMARY_COLUMNS:
        if column not in matrix_df.columns:
            matrix_df[column] = "" if column in {"route_key", "method", "profile", "status", "notes"} else np.nan
    matrix_df = _add_method_comparison_scores(matrix_df)
    matrix_df["method_sort"] = matrix_df["method"].map({method: i for i, method in enumerate(MATRIX_METHOD_ORDER)}).fillna(99)
    matrix_df["profile_sort"] = matrix_df["profile"].map({profile: i for i, profile in enumerate(MATRIX_PROFILE_ORDER)}).fillna(99)
    matrix_df = matrix_df.sort_values(["trip_days", "method_sort", "profile_sort"]).drop(columns=["method_sort", "profile_sort"]).reset_index(drop=True)
    matrix_df = matrix_df[ROUTE_MATRIX_SUMMARY_COLUMNS]

    matrix_route_stops_df = (
        pd.concat(route_frames, ignore_index=True, sort=False)
        if route_frames
        else pd.DataFrame(columns=_route_matrix_stop_columns())
    )
    if not matrix_route_stops_df.empty:
        matrix_route_stops_df["method_sort"] = matrix_route_stops_df["method"].map({method: i for i, method in enumerate(MATRIX_METHOD_ORDER)}).fillna(99)
        matrix_route_stops_df["profile_sort"] = matrix_route_stops_df["profile"].map({profile: i for i, profile in enumerate(MATRIX_PROFILE_ORDER)}).fillna(99)
        matrix_route_stops_df = (
            matrix_route_stops_df.sort_values(["trip_days", "method_sort", "profile_sort", "day", "stop_order"])
            .drop(columns=["method_sort", "profile_sort"])
            .reset_index(drop=True)
        )
        matrix_route_stops_df = matrix_route_stops_df[_route_matrix_stop_columns()]

    matrix_df.to_csv(output_dir / "production_route_matrix_comparison.csv", index=False)
    matrix_route_stops_df.to_csv(output_dir / "production_route_matrix_route_stops.csv", index=False)
    write_nature_route_artifacts(matrix_route_stops_df, output_dir, config)
    _write_route_matrix_hotel_debug(
        output_dir=output_dir,
        route_stops_df=matrix_route_stops_df,
        must_go_candidates=must_go_candidates,
    )
    _write_route_matrix_compatibility_artifacts(
        output_dir=output_dir,
        matrix_df=matrix_df,
        matrix_route_stops_df=matrix_route_stops_df,
        config=config,
        must_go_candidates=must_go_candidates,
    )
    return matrix_df, matrix_route_stops_df


def compare_trip_lengths(
    *,
    config: TripConfig,
    city_summary_df: pd.DataFrame,
    hotels_df: pd.DataFrame,
    osm_city_hotel_catalog_df: pd.DataFrame,
    output_dir: str | Path,
    primary_city: str = "Santa Barbara",
) -> pd.DataFrame:
    """Compare hierarchical Gurobi plans for configurable trip lengths."""
    if not bool(config.get("trip_duration_comparison", "enabled", True)):
        return pd.DataFrame()
    raw_days = config.get("trip_duration_comparison", "days", [7, 9, 12])
    day_options = [int(value) for value in raw_days if int(value) > 0]
    base_days = max(1, int(config.get("trip", "trip_days", 7)))
    base_budget = float(config.get("budget", "user_budget", 2000.0))
    scale_budget = bool(config.get("trip_duration_comparison", "scale_budget_with_days", True))
    rows = []
    for trip_days in day_options:
        scaled_budget = base_budget * trip_days / base_days if scale_budget else base_budget
        duration_config = TripConfig(
            data=_merge_config_data(
                config.to_dict(),
                {
                    "trip": {"trip_days": trip_days},
                    "budget": {"user_budget": scaled_budget},
                },
            ),
            source_path=config.source_path,
        )
        best_trip, plan_df = solve_hierarchical_trip_with_gurobi(duration_config, city_summary_df)
        budget_df = estimate_budget_range(
            trip=best_trip,
            hotels_df=hotels_df,
            osm_city_hotel_catalog_df=osm_city_hotel_catalog_df,
            output_dir=output_dir,
            config=duration_config,
            primary_city=primary_city,
            write_output=False,
        )
        buffered_total = float(budget_df.loc[budget_df["component"].eq("buffered_total"), "expected"].iloc[0])
        soft_budget = float(budget_df.loc[budget_df["component"].eq("soft_budget"), "expected"].iloc[0])
        hard_budget = float(budget_df.loc[budget_df["component"].eq("hard_budget"), "expected"].iloc[0])
        rows.append(
            {
                "trip_days": trip_days,
                "gateway_start": best_trip["gateway_start"],
                "gateway_end": best_trip["gateway_end"],
                "city_sequence": " -> ".join(best_trip["city_sequence"]),
                "overnight_bases": " -> ".join(best_trip["overnight_bases"]),
                "pass_through_cities": " -> ".join(best_trip["pass_through_cities"]),
                "days_by_city": str(best_trip["days_by_city"]),
                "base_city_count": int(len(best_trip["overnight_bases"])),
                "pass_through_city_count": int(len(best_trip["pass_through_cities"])),
                "intercity_drive_minutes": float(best_trip["intercity_drive_minutes"]),
                "objective": float(best_trip["objective"]),
                "city_value_component": float(best_trip.get("city_value_component", 0.0)),
                "pass_through_value_component": float(best_trip.get("pass_through_value_component", 0.0)),
                "intercity_drive_hours": float(best_trip["intercity_drive_minutes"]) / 60.0,
                "posterior_reward": np.nan,
                "estimated_budget": buffered_total,
                "soft_budget": soft_budget,
                "hard_budget": hard_budget,
                "route_feasible": bool(sum(best_trip["days_by_city"].values()) == trip_days),
                "solver_status": best_trip.get("solver_status"),
                "candidate_plan_count": int(len(plan_df)),
                "selected_strategy": "hierarchical_gurobi",
                "method": "hierarchical_gurobi",
                "data_source_scope": "open_enriched_catalog",
            }
        )
    comparison_df = pd.DataFrame(rows).sort_values("trip_days").reset_index(drop=True)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(output_path / "production_trip_length_comparison.csv", index=False)
    return comparison_df


def build_trip_length_comparison(
    *,
    context: dict,
    config: TripConfig,
    output_dir: str | Path | None = None,
    trip_days_grid: list[int] | None = None,
    primary_city: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    blueprint_trip_map = import_legacy_module("blueprint_trip_map")
    output_dir = Path(output_dir or _first_context_value(context, "OUTPUT_DIR", default="results/outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)
    city_summary_df = _load_frame_from_context_or_csv(
        context,
        ["city_catalog_summary_df"],
        output_dir / "production_city_catalog_summary.csv",
    )
    hotels_df = _load_frame_from_context_or_csv(context, ["hotels_df"], output_dir / "hotel_dataset.csv")
    osm_city_hotel_catalog_df = _load_frame_from_context_or_csv(
        context,
        ["osm_city_hotel_catalog_df"],
        output_dir / "production_city_hotel_catalog.csv",
    )

    summary_columns = [
        "comparison_type",
        "comparison_label",
        "trip_days",
        "gateway_start",
        "gateway_end",
        "city_sequence",
        "days_by_city",
        "intercity_drive_minutes",
        "intercity_drive_hours",
        "estimated_budget",
        "budget_source",
        "objective",
        "posterior_reward",
        "selected_strategy",
        "solver_status",
        "status",
        "notes",
    ]
    if city_summary_df.empty:
        pd.DataFrame(columns=summary_columns).to_csv(output_dir / "production_trip_length_comparison.csv", index=False)
        pd.DataFrame(columns=_comparison_route_stop_columns()).to_csv(
            output_dir / "production_trip_length_route_stops.csv",
            index=False,
        )
        return pd.DataFrame(columns=summary_columns), pd.DataFrame(columns=_comparison_route_stop_columns())

    raw_days = trip_days_grid or config.get("trip_duration_comparison", "days", [7, 9, 12])
    day_options = sorted({int(value) for value in raw_days if int(value) > 0}) or [7, 9, 12]
    base_days = max(1, int(config.get("trip", "trip_days", 7)))
    base_budget = float(config.get("budget", "user_budget", 2000.0))
    scale_budget = bool(config.get("trip_duration_comparison", "scale_budget_with_days", True))
    city_name = str(primary_city or _first_context_value(context, "CITY", default="Santa Barbara"))
    bandit_summary_df = _load_frame_from_context_or_csv(
        context,
        ["hybrid_bandit_summary_df"],
        output_dir / "production_hybrid_bandit_optimization_summary.csv",
    )
    if not bandit_summary_df.empty and "posterior_mean_reward" in bandit_summary_df.columns:
        duration_strategy = str(
            bandit_summary_df.sort_values("posterior_mean_reward", ascending=False).iloc[0].get(
                "route_search_strategy",
                "scenic_social",
            )
        )
    else:
        duration_strategy = "scenic_social"

    comparison_rows = []
    route_frames = []
    for trip_days in day_options:
        scaled_budget = base_budget * trip_days / base_days if scale_budget else base_budget
        duration_config = TripConfig(
            data=_merge_config_data(
                config.to_dict(),
                {
                    "trip": {"trip_days": trip_days},
                    "budget": {"user_budget": scaled_budget},
                },
            ),
            source_path=config.source_path,
        )
        best_trip, _ = solve_hierarchical_trip_with_gurobi(duration_config, city_summary_df)
        budget_df = estimate_budget_range(
            trip=best_trip,
            hotels_df=hotels_df,
            osm_city_hotel_catalog_df=osm_city_hotel_catalog_df,
            output_dir=output_dir,
            config=duration_config,
            primary_city=city_name,
            write_output=False,
        )
        duration_context = dict(context)
        duration_context["OUTPUT_DIR"] = output_dir
        duration_context["best_hierarchical_trip"] = best_trip
        route_stops, route_info = _hierarchical_method_route_outputs(
            context=duration_context,
            blueprint_trip_map=blueprint_trip_map,
            trip=best_trip,
            config=duration_config,
            method="hierarchical_bandit_gurobi_repair",
            strategy_name=duration_strategy,
            local_solver="small_gurobi",
            budget_df=budget_df,
        )
        comparison_label = f"Trip Length · {int(trip_days)}-Day Hybrid Bandit + Small Gurobi"
        if not route_stops.empty:
            route_stops = route_stops.copy()
            route_stops["comparison_type"] = "trip_length"
            route_stops["comparison_label"] = comparison_label
            route_stops["method"] = "hierarchical_bandit_gurobi_repair"
            route_stops["method_display_name"] = _method_display_name("hierarchical_bandit_gurobi_repair")
            route_stops["trip_days"] = int(trip_days)
            route_frames.append(route_stops)

        comparison_rows.append(
            {
                "comparison_type": "trip_length",
                "comparison_label": comparison_label,
                "trip_days": int(trip_days),
                "gateway_start": best_trip.get("gateway_start"),
                "gateway_end": best_trip.get("gateway_end"),
                "city_sequence": " -> ".join([str(item) for item in best_trip.get("city_sequence", [])]),
                "days_by_city": str(best_trip.get("days_by_city", {})),
                "intercity_drive_minutes": float(best_trip.get("intercity_drive_minutes", 0.0) or 0.0),
                "intercity_drive_hours": float(best_trip.get("intercity_drive_minutes", 0.0) or 0.0) / 60.0,
                "estimated_budget": float(route_info.get("total_budget", get_canonical_trip_budget(duration_config, budget_df))),
                "budget_source": str(route_info.get("budget_source", _canonical_budget_value_and_source(duration_config, budget_df)[1])),
                "objective": float(best_trip.get("objective", np.nan)),
                "posterior_reward": np.nan,
                "selected_strategy": f"hierarchical_bandit_gurobi_repair:{duration_strategy}",
                "solver_status": str(best_trip.get("solver_status", "UNKNOWN")),
                "status": str(route_info.get("status", "FAILED")),
                "notes": (
                    "Trip-length comparison uses top-level hierarchical day/city allocation "
                    f"with bandit strategy `{duration_strategy}` and small-Gurobi local repair. "
                    + str(route_info.get("notes", ""))
                ),
            }
        )

    comparison_df = pd.DataFrame(comparison_rows)
    if not comparison_df.empty:
        comparison_df = comparison_df.sort_values(["trip_days", "gateway_start", "gateway_end"]).reset_index(drop=True)
        comparison_df = comparison_df[summary_columns]
    route_stops_df = pd.concat(route_frames, ignore_index=True, sort=False) if route_frames else pd.DataFrame(columns=_comparison_route_stop_columns())
    if not route_stops_df.empty:
        route_stops_df = route_stops_df.sort_values(["trip_days", "day", "stop_order"]).reset_index(drop=True)
        route_stops_df = route_stops_df[_comparison_route_stop_columns()]
    comparison_df.to_csv(output_dir / "production_trip_length_comparison.csv", index=False)
    route_stops_df.to_csv(output_dir / "production_trip_length_route_stops.csv", index=False)
    return comparison_df, route_stops_df


def export_trip_length_comparison_dashboard(
    *,
    context: dict,
    config: TripConfig,
    output_dir: str | Path | None = None,
    trip_days_grid: list[int] | None = None,
    primary_city: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return build_trip_length_comparison(
        context=context,
        config=config,
        output_dir=output_dir,
        trip_days_grid=trip_days_grid,
        primary_city=primary_city,
    )


def build_production_method_comparison(
    *,
    context: dict,
    config: TripConfig,
    output_dir: str | Path | None = None,
    primary_city: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    blueprint_trip_map = import_legacy_module("blueprint_trip_map")
    output_dir = Path(output_dir or _first_context_value(context, "OUTPUT_DIR", default="results/outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)
    city_summary_df = _load_frame_from_context_or_csv(
        context,
        ["city_catalog_summary_df"],
        output_dir / "production_city_catalog_summary.csv",
    )
    enriched_df = _load_frame_from_context_or_csv(
        context,
        ["production_enriched_poi_catalog_df"],
        output_dir / "production_enriched_poi_catalog.csv",
    )
    hotels_df = _load_frame_from_context_or_csv(context, ["hotels_df"], output_dir / "hotel_dataset.csv")
    osm_city_hotel_catalog_df = _load_frame_from_context_or_csv(
        context,
        ["osm_city_hotel_catalog_df"],
        output_dir / "production_city_hotel_catalog.csv",
    )
    budget_df = _load_frame_from_context_or_csv(context, ["budget_estimate_df"], output_dir / "production_budget_estimate.csv")
    city_name = str(primary_city or _first_context_value(context, "CITY", default="Santa Barbara"))

    summary_columns = [
        "method",
        "method_display_name",
        "allocation_solver",
        "local_route_solver",
        "trip_days",
        "gateway_start",
        "gateway_end",
        "city_sequence",
        "days_by_city",
        "total_budget",
        "budget_used",
        "budget_source",
        "total_utility",
        "total_waiting_time",
        "total_travel_time",
        "total_travel_distance_km",
        "total_cost",
        "selected_attractions",
        "must_go_count",
        "must_go_total",
        "must_go_selected_count",
        "must_go_skipped_count",
        "must_go_coverage_ratio",
        "must_go_skipped_names",
        "must_go_policy",
        "diversity_score",
        "utility_norm",
        "must_go_norm",
        "diversity_norm",
        "travel_efficiency_norm",
        "cost_efficiency_norm",
        "waiting_efficiency_norm",
        "comparison_score",
        "comparison_score_formula",
        "objective",
        "posterior_reward",
        "solve_seconds",
        "mip_gap",
        "status",
        "notes",
    ]
    if city_summary_df.empty:
        pd.DataFrame(columns=summary_columns).to_csv(output_dir / "production_method_comparison.csv", index=False)
        pd.DataFrame(columns=_comparison_route_stop_columns()).to_csv(output_dir / "production_method_route_stops.csv", index=False)
        return pd.DataFrame(columns=summary_columns), pd.DataFrame(columns=_comparison_route_stop_columns())

    best_trip = _first_context_value(context, "best_hierarchical_trip")
    if not isinstance(best_trip, dict) or not best_trip:
        best_trip, _ = solve_hierarchical_trip_with_gurobi(config, city_summary_df)

    if budget_df.empty:
        budget_df = estimate_budget_range(
            trip=best_trip,
            hotels_df=hotels_df,
            osm_city_hotel_catalog_df=osm_city_hotel_catalog_df,
            output_dir=output_dir,
            config=config,
            primary_city=city_name,
            write_output=False,
        )

    trip_days = get_canonical_trip_days(config, best_trip)
    hierarchical_context = dict(context)
    hierarchical_context["OUTPUT_DIR"] = output_dir
    hierarchical_context["best_hierarchical_trip"] = best_trip
    must_go_candidates = _must_go_candidates_from_output(output_dir, enriched_df)
    method_rows = []
    route_frames = []

    def _finalize_method_row(
        method: str,
        route_stops: pd.DataFrame,
        route_info: dict,
        *,
        trip_for_method: dict,
        objective_value: float,
        posterior_reward: float,
        notes: str,
    ) -> dict:
        metrics = _route_metrics_from_stops(route_stops)
        must_go_summary = _route_must_go_summary(route_stops, must_go_candidates, config)
        intercity_minutes, intercity_distance = _intercity_route_metrics(trip_for_method)
        attraction_cost = _estimated_route_stop_cost(route_stops)
        total_cost = float(route_info.get("reserved_food_transport", 0.0)) + float(route_info.get("committed_hotel_cost", 0.0)) + float(attraction_cost)
        allocation_solver, local_route_solver = _method_solver_roles(method)
        return {
            "method": method,
            "method_display_name": _method_display_name(method),
            "allocation_solver": allocation_solver,
            "local_route_solver": local_route_solver,
            "trip_days": get_canonical_trip_days(config, trip_for_method),
            **_trip_summary_fields(trip_for_method),
            "total_budget": float(route_info.get("total_budget", get_canonical_trip_budget(config, budget_df))),
            "budget_used": float(total_cost),
            "budget_source": str(route_info.get("budget_source", _canonical_budget_value_and_source(config, budget_df)[1])),
            "total_utility": metrics["total_utility"],
            "total_waiting_time": metrics["total_waiting_time"],
            "total_travel_time": metrics["total_travel_time"] + intercity_minutes,
            "total_travel_distance_km": metrics["total_travel_distance_km"] + intercity_distance,
            "total_cost": float(total_cost),
            "selected_attractions": metrics["selected_attractions"],
            "must_go_count": int(route_stops["social_must_go"].astype(bool).sum()) if not route_stops.empty and "social_must_go" in route_stops.columns else 0,
            **must_go_summary,
            "diversity_score": metrics["diversity_score"],
            "objective": float(objective_value) if pd.notna(objective_value) else np.nan,
            "posterior_reward": float(posterior_reward) if pd.notna(posterior_reward) else np.nan,
            "solve_seconds": float(route_info.get("solve_seconds", np.nan) or 0.0),
            "mip_gap": route_info.get("mip_gap", np.nan),
            "status": str(route_info.get("status", "FAILED")),
            "notes": notes,
        }

    hierarchical_route_stops, hierarchical_route_info = _hierarchical_method_route_outputs(
        context=hierarchical_context,
        blueprint_trip_map=blueprint_trip_map,
        trip=best_trip,
        config=config,
        method="hierarchical_gurobi_pipeline",
        strategy_name="balanced",
        local_solver="legacy_gurobi",
        budget_df=budget_df,
    )
    if not hierarchical_route_stops.empty:
        route_frames.append(hierarchical_route_stops)
    method_rows.append(
        _finalize_method_row(
            "hierarchical_gurobi_pipeline",
            hierarchical_route_stops,
            hierarchical_route_info,
            trip_for_method=best_trip,
            objective_value=float(best_trip.get("objective", np.nan)),
            posterior_reward=np.nan,
            notes=(
                "Best hierarchical California trip using the legacy local Gurobi day-route oracle "
                "under the shared canonical production budget."
            ),
        )
    )

    try:
        greedy_trip, _greedy_plan_df = solve_hierarchical_trip_with_greedy(config, city_summary_df)
        greedy_context = dict(context)
        greedy_context["OUTPUT_DIR"] = output_dir
        greedy_context["best_hierarchical_trip"] = greedy_trip
        greedy_route_stops, greedy_route_info = _hierarchical_method_route_outputs(
            context=greedy_context,
            blueprint_trip_map=blueprint_trip_map,
            trip=greedy_trip,
            config=config,
            method="hierarchical_greedy_baseline",
            strategy_name="balanced",
            local_solver="greedy",
            budget_df=budget_df,
        )
        if not greedy_route_stops.empty:
            route_frames.append(greedy_route_stops)
        method_rows.append(
            _finalize_method_row(
                "hierarchical_greedy_baseline",
                greedy_route_stops,
                greedy_route_info,
                trip_for_method=greedy_trip,
                objective_value=float(greedy_trip.get("objective", greedy_route_info.get("objective", np.nan))),
                posterior_reward=np.nan,
                notes=(
                    "Greedy hierarchical baseline: greedy city/base/day allocation followed by greedy local "
                    "attraction selection and compact route ordering."
                ),
            )
        )
    except Exception as exc:
        allocation_solver, local_route_solver = _method_solver_roles("hierarchical_greedy_baseline")
        method_rows.append(
            {
                "method": "hierarchical_greedy_baseline",
                "method_display_name": _method_display_name("hierarchical_greedy_baseline"),
                "allocation_solver": allocation_solver,
                "local_route_solver": local_route_solver,
                "trip_days": trip_days,
                **_trip_summary_fields(best_trip),
                "total_budget": get_canonical_trip_budget(config, budget_df),
                "budget_used": np.nan,
                "budget_source": _canonical_budget_value_and_source(config, budget_df)[1],
                "total_utility": np.nan,
                "total_waiting_time": np.nan,
                "total_travel_time": np.nan,
                "total_travel_distance_km": np.nan,
                "total_cost": np.nan,
                "selected_attractions": 0,
                "must_go_count": 0,
                "diversity_score": np.nan,
                "objective": np.nan,
                "posterior_reward": np.nan,
                "solve_seconds": np.nan,
                "mip_gap": np.nan,
                "status": "FAILED",
                "notes": f"Hierarchical greedy baseline failed: {type(exc).__name__}",
            }
        )

    bandit_route_stops, bandit_summary_row = _bandit_route_stop_rows(
        context=context,
        best_trip=best_trip,
        config=config,
        output_dir=output_dir,
        enriched_df=enriched_df,
        budget_df=budget_df,
    )
    if not bandit_route_stops.empty:
        route_frames.append(bandit_route_stops)
    bandit_summary_row.update(_route_must_go_summary(bandit_route_stops, must_go_candidates, config))
    method_rows.append(bandit_summary_row)

    method_df = pd.DataFrame(method_rows)
    for column in summary_columns:
        if column not in method_df.columns:
            method_df[column] = (
                ""
                if column
                in {
                    "method",
                    "method_display_name",
                    "allocation_solver",
                    "local_route_solver",
                    "gateway_start",
                    "gateway_end",
                    "city_sequence",
                    "days_by_city",
                    "budget_source",
                    "status",
                    "notes",
                    "comparison_score_formula",
                    "must_go_skipped_names",
                    "must_go_policy",
                }
                else np.nan
            )
    method_df = _add_method_comparison_scores(method_df)
    preferred_order = {
        "hierarchical_gurobi_pipeline": 0,
        "hierarchical_greedy_baseline": 1,
        "hierarchical_bandit_gurobi_repair": 2,
    }
    method_df["sort_order"] = method_df["method"].map(preferred_order).fillna(99)
    method_df = method_df.sort_values(["sort_order", "method"]).drop(columns=["sort_order"]).reset_index(drop=True)
    method_df = method_df[summary_columns]

    route_stops_df = pd.concat(route_frames, ignore_index=True, sort=False) if route_frames else pd.DataFrame(columns=_comparison_route_stop_columns())
    if not route_stops_df.empty:
        route_stops_df = route_stops_df.sort_values(["comparison_type", "method", "trip_days", "day", "stop_order"]).reset_index(drop=True)
        route_stops_df = route_stops_df[_comparison_route_stop_columns()]

    method_df.to_csv(output_dir / "production_method_comparison.csv", index=False)
    route_stops_df.to_csv(output_dir / "production_method_route_stops.csv", index=False)
    write_nature_route_artifacts(route_stops_df, output_dir, config)
    _write_must_go_coverage(
        output_dir=output_dir,
        method_df=method_df,
        route_stops_df=route_stops_df,
        must_go_candidates=must_go_candidates,
        config=config,
    )
    _write_hotel_selection_debug(
        output_dir=output_dir,
        route_stops_df=route_stops_df,
        must_go_candidates=must_go_candidates,
    )
    return method_df, route_stops_df


def export_method_comparison_dashboard(
    *,
    context: dict,
    config: TripConfig,
    output_dir: str | Path | None = None,
    primary_city: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return build_production_method_comparison(
        context=context,
        config=config,
        output_dir=output_dir,
        primary_city=primary_city,
    )


def prepare_comparison_dashboard_outputs(
    *,
    context: dict,
    config: TripConfig,
    output_dir: str | Path | None = None,
    trip_days_grid: list[int] | None = None,
) -> dict:
    output_dir = Path(output_dir or _first_context_value(context, "OUTPUT_DIR", default="results/outputs"))
    primary_city = str(_first_context_value(context, "CITY", default="Santa Barbara"))
    route_matrix_df, route_matrix_route_stops_df = build_route_matrix_comparison(
        context=context,
        config=config,
        output_dir=output_dir,
        trip_days_grid=trip_days_grid,
        primary_city=primary_city,
    )
    trip_length_df = _load_csv(output_dir / "production_trip_length_comparison.csv")
    trip_length_route_stops_df = _load_csv(output_dir / "production_trip_length_route_stops.csv")
    method_df = _load_csv(output_dir / "production_method_comparison.csv")
    method_route_stops_df = _load_csv(output_dir / "production_method_route_stops.csv")
    return {
        "production_route_matrix_comparison_df": route_matrix_df,
        "production_route_matrix_route_stops_df": route_matrix_route_stops_df,
        "production_trip_length_comparison_df": trip_length_df,
        "production_trip_length_route_stops_df": trip_length_route_stops_df,
        "production_method_comparison_df": method_df,
        "production_method_route_stops_df": method_route_stops_df,
    }


def run_configurable_blueprint_pipeline(
    *,
    all_business_df: pd.DataFrame,
    hotels_df: pd.DataFrame,
    city_names: list[str],
    output_dir: str | Path,
    config: TripConfig,
    primary_city: str = "Santa Barbara",
) -> dict:
    output_dir = Path(output_dir)
    save_resolved_config(config, output_dir / "production_resolved_config.json")
    enrichment = build_enriched_catalog(
        all_business_df=all_business_df,
        city_names=city_names,
        output_dir=output_dir,
        config=config,
    )
    best_trip, hierarchical_df = solve_hierarchical_trip_with_gurobi(config, enrichment["city_catalog_summary_df"])
    budget_df = estimate_budget_range(
        trip=best_trip,
        hotels_df=hotels_df,
        osm_city_hotel_catalog_df=enrichment["osm_city_hotel_catalog_df"],
        output_dir=output_dir,
        config=config,
        primary_city=primary_city,
    )
    trip_length_comparison_df = pd.DataFrame()
    arms_df = build_bandit_arms(
        hierarchical_candidates=hierarchical_df.to_dict("records"),
        enriched_df=enrichment["production_enriched_poi_catalog_df"],
        budget_df=budget_df,
        config=config,
    )
    bandit_log_df, bandit_summary_df = run_bandit_search(arms_df, config)
    bandit_stress_runs_df, bandit_stress_summary_df = run_bandit_gurobi_stress_benchmark(
        arms_df=arms_df,
        enriched_df=enrichment["production_enriched_poi_catalog_df"],
        budget_df=budget_df,
        config=config,
    )
    mmr_candidate_log_df = bandit_stress_runs_df.attrs.get("mmr_candidate_log_df", pd.DataFrame())
    if isinstance(mmr_candidate_log_df, pd.DataFrame) and not mmr_candidate_log_df.empty:
        mmr_candidate_log_df.to_csv(output_dir / "production_mmr_candidate_log.csv", index=False)
    else:
        pd.DataFrame().to_csv(output_dir / "production_mmr_candidate_log.csv", index=False)
    epsilon_candidates = enrichment["production_enriched_poi_catalog_df"].sort_values("final_poi_value", ascending=False).head(
        int(config.get("optimization", "max_pois_per_day", 4)) * 4
    )
    epsilon_route_result = solve_enriched_route_with_gurobi(epsilon_candidates, config, candidate_size=len(epsilon_candidates))
    bandit_log_df.to_csv(output_dir / "production_bandit_gurobi_runs.csv", index=False)
    bandit_summary_df.to_csv(output_dir / "production_hybrid_bandit_optimization_summary.csv", index=False)
    bandit_stress_runs_df.to_csv(output_dir / "production_bandit_stress_runs.csv", index=False)
    bandit_stress_summary_df.to_csv(output_dir / "production_bandit_stress_summary.csv", index=False)
    bandit_stress_summary_df.to_csv(output_dir / "production_solver_scalability_summary.csv", index=False)
    route_component_columns = [
        "arm_id",
        "route_search_strategy",
        "selected_pois",
        "gurobi_oracle_reward",
        "combined_reward",
        "route_feasible",
        "submodular_diversity",
        "total_travel_minutes",
        "total_visit_minutes",
        "total_detour_minutes",
        "total_cost",
        "total_weather_risk",
        "interest_adjusted_utility",
        "interest_delta",
        "nature_score",
        "scenic_score",
        "national_parks_selected",
        "protected_parks_selected",
        "outdoor_weather_risk",
        "seasonality_risk",
        "balance_score",
        "epsilon_time_budget",
        "epsilon_cost_budget",
        "epsilon_detour_minutes",
        "epsilon_weather_risk",
        "epsilon_min_diversity",
        "solver_status",
        "optimality_gap",
    ]
    bandit_stress_runs_df[[col for col in route_component_columns if col in bandit_stress_runs_df.columns]].to_csv(
        output_dir / "production_route_objective_components.csv",
        index=False,
    )
    pd.DataFrame(
        [
            diversity_audit_row(enrichment["production_enriched_poi_catalog_df"], "enriched_catalog"),
            diversity_audit_row(epsilon_candidates, "epsilon_gurobi_candidate_pool"),
        ]
    ).to_csv(output_dir / "production_diversity_audit.csv", index=False)
    experiment_summary_df = build_enriched_experiment_summary(
        enriched_df=enrichment["production_enriched_poi_catalog_df"],
        best_trip=best_trip,
        budget_df=budget_df,
        city_summary_df=enrichment["city_catalog_summary_df"],
        bandit_stress_runs_df=bandit_stress_runs_df,
        epsilon_route_result=epsilon_route_result,
        config=config,
    )
    experiment_summary_df.to_csv(output_dir / "production_experiment_summary.csv", index=False)
    experiment_summary_df.to_csv(output_dir / "production_model_comparison.csv", index=False)
    pd.DataFrame([epsilon_route_result]).to_csv(output_dir / "production_pareto_epsilon_runs.csv", index=False)
    bandit_stress_runs_df.assign(
        experiment_id="hybrid_bandit_gurobi_stress",
        strategy="hybrid_bandit_gurobi",
        data_source_scope="open_enriched_catalog",
    ).to_csv(output_dir / "production_experiment_runs.csv", index=False)
    hierarchical_df.to_csv(output_dir / "production_hierarchical_gurobi_plan.csv", index=False)
    comparison_context = {
        **enrichment,
        "OUTPUT_DIR": output_dir,
        "CITY": primary_city,
        "best_hierarchical_trip": best_trip,
        "budget_estimate_df": budget_df,
        "hotels_df": hotels_df,
        "hybrid_bandit_arms_df": arms_df,
        "hybrid_bandit_log_df": bandit_log_df,
        "hybrid_bandit_summary_df": bandit_summary_df,
        "bandit_stress_runs_df": bandit_stress_runs_df,
        "bandit_stress_summary_df": bandit_stress_summary_df,
        "production_experiment_summary_df": experiment_summary_df,
    }
    route_matrix_comparison_df, route_matrix_route_stops_df = build_route_matrix_comparison(
        context=comparison_context,
        config=config,
        output_dir=output_dir,
        primary_city=primary_city,
    )
    trip_length_comparison_df = _load_csv(output_dir / "production_trip_length_comparison.csv")
    trip_length_route_stops_df = _load_csv(output_dir / "production_trip_length_route_stops.csv")
    method_comparison_df = _load_csv(output_dir / "production_method_comparison.csv")
    method_route_stops_df = _load_csv(output_dir / "production_method_route_stops.csv")
    return {
        **enrichment,
        "best_hierarchical_trip": best_trip,
        "hierarchical_trip_df": hierarchical_df,
        "budget_estimate_df": budget_df,
        "route_matrix_comparison_df": route_matrix_comparison_df,
        "route_matrix_route_stops_df": route_matrix_route_stops_df,
        "trip_length_comparison_df": trip_length_comparison_df,
        "trip_length_route_stops_df": trip_length_route_stops_df,
        "hybrid_bandit_arms_df": arms_df,
        "hybrid_bandit_log_df": bandit_log_df,
        "hybrid_bandit_summary_df": bandit_summary_df,
        "bandit_stress_runs_df": bandit_stress_runs_df,
        "bandit_stress_summary_df": bandit_stress_summary_df,
        "mmr_candidate_log_df": mmr_candidate_log_df,
        "epsilon_route_result": epsilon_route_result,
        "production_experiment_summary_df": experiment_summary_df,
        "production_method_comparison_df": method_comparison_df,
        "production_method_route_stops_df": method_route_stops_df,
    }
