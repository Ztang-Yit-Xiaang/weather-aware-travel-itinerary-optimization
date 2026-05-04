"""Thin orchestration helpers for the blueprint notebook."""

from __future__ import annotations

from pathlib import Path
from copy import deepcopy

import numpy as np
import pandas as pd
from geopy.distance import geodesic

from .bandit_candidate_selector import build_bandit_arms, run_bandit_gurobi_stress_benchmark, run_bandit_search
from .budget import estimate_budget_range
from .config import TripConfig, save_resolved_config
from .data_enrichment import build_enriched_catalog
from .diversity import diversity_audit_row
from .hierarchical_gurobi import solve_hierarchical_trip_with_gurobi
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
                "objective": float(best_trip["objective"]),
                "city_value_component": float(best_trip.get("city_value_component", 0.0)),
                "pass_through_value_component": float(best_trip.get("pass_through_value_component", 0.0)),
                "intercity_drive_hours": float(best_trip["intercity_drive_minutes"]) / 60.0,
                "estimated_budget": buffered_total,
                "soft_budget": soft_budget,
                "hard_budget": hard_budget,
                "route_feasible": bool(sum(best_trip["days_by_city"].values()) == trip_days),
                "solver_status": best_trip.get("solver_status"),
                "candidate_plan_count": int(len(plan_df)),
                "data_source_scope": "open_enriched_catalog",
            }
        )
    comparison_df = pd.DataFrame(rows).sort_values("trip_days").reset_index(drop=True)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(output_path / "production_trip_length_comparison.csv", index=False)
    return comparison_df


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
    trip_length_comparison_df = compare_trip_lengths(
        config=config,
        city_summary_df=enrichment["city_catalog_summary_df"],
        hotels_df=hotels_df,
        osm_city_hotel_catalog_df=enrichment["osm_city_hotel_catalog_df"],
        output_dir=output_dir,
        primary_city=primary_city,
    )
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
    experiment_summary_df.to_csv(output_dir / "production_method_comparison.csv", index=False)
    pd.DataFrame([epsilon_route_result]).to_csv(output_dir / "production_pareto_epsilon_runs.csv", index=False)
    bandit_stress_runs_df.assign(
        experiment_id="hybrid_bandit_gurobi_stress",
        strategy="hybrid_bandit_gurobi",
        data_source_scope="open_enriched_catalog",
    ).to_csv(output_dir / "production_experiment_runs.csv", index=False)
    hierarchical_df.to_csv(output_dir / "production_hierarchical_gurobi_plan.csv", index=False)
    return {
        **enrichment,
        "best_hierarchical_trip": best_trip,
        "hierarchical_trip_df": hierarchical_df,
        "budget_estimate_df": budget_df,
        "trip_length_comparison_df": trip_length_comparison_df,
        "hybrid_bandit_arms_df": arms_df,
        "hybrid_bandit_log_df": bandit_log_df,
        "hybrid_bandit_summary_df": bandit_summary_df,
        "bandit_stress_runs_df": bandit_stress_runs_df,
        "bandit_stress_summary_df": bandit_stress_summary_df,
        "mmr_candidate_log_df": mmr_candidate_log_df,
        "epsilon_route_result": epsilon_route_result,
        "production_experiment_summary_df": experiment_summary_df,
    }
