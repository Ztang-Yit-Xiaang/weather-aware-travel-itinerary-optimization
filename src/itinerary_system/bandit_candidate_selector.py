"""Bandit candidate selection before small Gurobi route solves."""

from __future__ import annotations

import ast

import pandas as pd

from ._legacy import import_legacy_module
from .config import TripConfig
from .diversity import mmr_select_candidates, submodular_diversity
from .nature_catalog import interest_value_column
from .route_gurobi_oracle import solve_enriched_route_with_gurobi

NATURE_ROUTE_SEARCH_STRATEGIES = {
    "nature_heavy": {
        "reward_bias": 0.08,
        "runtime_minutes_per_city": 2.4,
        "poi_value_weight": 0.28,
        "social_weight": 0.16,
        "must_go_weight": 0.12,
        "corridor_weight": 0.12,
        "detour_penalty": 0.012,
        "cost_multiplier": 1.02,
        "optimizer_backend": "fast_local_route_oracle_with_nature_scoring",
    },
    "scenic_parks": {
        "reward_bias": 0.07,
        "runtime_minutes_per_city": 2.3,
        "poi_value_weight": 0.24,
        "social_weight": 0.12,
        "must_go_weight": 0.10,
        "corridor_weight": 0.18,
        "detour_penalty": 0.010,
        "cost_multiplier": 1.00,
        "optimizer_backend": "fast_local_route_oracle_with_scenic_park_scoring",
    },
    "national_park_priority": {
        "reward_bias": 0.10,
        "runtime_minutes_per_city": 2.8,
        "poi_value_weight": 0.26,
        "social_weight": 0.10,
        "must_go_weight": 0.12,
        "corridor_weight": 0.10,
        "detour_penalty": 0.009,
        "cost_multiplier": 1.04,
        "optimizer_backend": "fast_local_route_oracle_with_park_priority",
    },
    "city_nature_balanced": {
        "reward_bias": 0.06,
        "runtime_minutes_per_city": 2.1,
        "poi_value_weight": 0.27,
        "social_weight": 0.18,
        "must_go_weight": 0.14,
        "corridor_weight": 0.14,
        "detour_penalty": 0.012,
        "cost_multiplier": 1.00,
        "optimizer_backend": "fast_local_route_oracle_city_nature_balanced",
    },
    "weather_safe_indoor_backup": {
        "reward_bias": 0.04,
        "runtime_minutes_per_city": 1.8,
        "poi_value_weight": 0.22,
        "social_weight": 0.14,
        "must_go_weight": 0.12,
        "corridor_weight": 0.10,
        "detour_penalty": 0.016,
        "cost_multiplier": 0.96,
        "optimizer_backend": "fast_local_route_oracle_weather_safe_backup",
    },
}


def route_search_strategies_from_config(config: TripConfig) -> dict:
    production_enrichment = import_legacy_module("production_enrichment")
    base_strategies = {**production_enrichment.ROUTE_SEARCH_STRATEGIES, **NATURE_ROUTE_SEARCH_STRATEGIES}
    allowed = set(config.get("bandit", "arms", production_enrichment.ROUTE_SEARCH_STRATEGIES.keys()))
    social_weight = float(config.get("social", "social_score_weight", 1.10))
    must_go_weight = float(config.get("social", "must_go_bonus_weight", 0.85))
    corridor_weight = float(config.get("social", "corridor_fit_weight", 0.30))
    detour_penalty = float(config.get("social", "detour_penalty_weight", 0.01))
    strategies = {}
    for name, strategy in base_strategies.items():
        if name not in allowed:
            continue
        configured = dict(strategy)
        configured["social_weight"] = configured.get("social_weight", 0.20) * social_weight / 1.10
        configured["must_go_weight"] = configured.get("must_go_weight", 0.20) * must_go_weight / 0.85
        configured["corridor_weight"] = configured.get("corridor_weight", 0.10) * corridor_weight / 0.30
        configured["detour_penalty"] = detour_penalty
        strategies[name] = configured
    return strategies


def build_bandit_arms(
    *,
    hierarchical_candidates: list[dict],
    enriched_df: pd.DataFrame,
    budget_df: pd.DataFrame,
    config: TripConfig,
) -> pd.DataFrame:
    production_enrichment = import_legacy_module("production_enrichment")
    return production_enrichment.build_hybrid_bandit_arms(
        hierarchical_candidates,
        enriched_df=enriched_df,
        budget_df=budget_df,
        route_strategies=route_search_strategies_from_config(config),
    )


def run_bandit_search(arms_df: pd.DataFrame, config: TripConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    production_enrichment = import_legacy_module("production_enrichment")
    return production_enrichment.run_hybrid_bandit_optimization_search(
        arms_df,
        episodes=int(config.get("bandit", "episodes", 40)),
        seed=42,
    )


def _as_list(value, default):
    if isinstance(value, list):
        return value
    if value is None:
        return default
    return list(value) if isinstance(value, tuple) else default


def _days_by_city(value) -> dict:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def _numeric_column(frame: pd.DataFrame, column: str, default=0.0) -> pd.Series:
    if column in frame.columns:
        return pd.to_numeric(frame[column], errors="coerce").fillna(default)
    return pd.Series(default, index=frame.index)


def _bool_column(frame: pd.DataFrame, column: str, default=False) -> pd.Series:
    if column in frame.columns:
        return frame[column].fillna(default).astype(bool)
    return pd.Series(default, index=frame.index)


def _candidate_bundle_for_arm(
    arm_row: pd.Series, enriched_df: pd.DataFrame, candidate_size: int, config: TripConfig
) -> pd.DataFrame:
    days_by_city = _days_by_city(arm_row.get("days_by_city"))
    cities = set(days_by_city.keys())
    strategy = str(arm_row.get("route_search_strategy", "balanced"))
    pool = enriched_df.copy()
    if cities:
        mask = pool["city"].isin(cities)
        if strategy in {"scenic_social", "high_must_go"}:
            mask = mask | _bool_column(pool, "social_must_go") | _numeric_column(pool, "corridor_fit").ge(0.55)
        pool = pool[mask].copy()
    if pool.empty:
        pool = enriched_df.copy()
    base_score_column = interest_value_column(config)
    if base_score_column not in pool.columns:
        base_score_column = "final_poi_value"
    # Nature-arm scoring math:
    # bundle_score_i(a) = base_score_i + phi_n(a)*nature_score_i
    #   + phi_s(a)*scenic_score_i + phi_h(a)*hiking_score_i
    #   + phi_p(a)*park_bonus_i
    #   + phi_safe(a)*(1 - weather_sensitivity_i*weather_risk_i)
    #   - phi_d(a)*detour_minutes_i
    pool["bundle_score"] = _numeric_column(pool, base_score_column)
    if strategy in {"scenic_social", "high_must_go"}:
        pool["bundle_score"] += 1.25 * _numeric_column(pool, "social_score")
        pool["bundle_score"] += 0.75 * _numeric_column(pool, "must_go_weight")
    if strategy in {"scenic_social", "low_detour"}:
        pool["bundle_score"] += 0.40 * _numeric_column(pool, "corridor_fit")
        pool["bundle_score"] -= 0.01 * _numeric_column(pool, "detour_minutes")
    if strategy == "fastest_low_cost":
        pool["bundle_score"] -= 0.02 * _numeric_column(pool, "detour_minutes")
    if strategy == "nature_heavy":
        pool["bundle_score"] += 0.90 * _numeric_column(pool, "nature_score")
        pool["bundle_score"] += 0.35 * _numeric_column(pool, "scenic_score")
        pool["bundle_score"] += 0.25 * _numeric_column(pool, "hiking_score")
        pool["bundle_score"] += 0.30 * _numeric_column(pool, "park_bonus")
        pool["bundle_score"] -= 0.010 * _numeric_column(pool, "detour_minutes")
    elif strategy == "scenic_parks":
        pool["bundle_score"] += 0.75 * _numeric_column(pool, "scenic_score")
        pool["bundle_score"] += 0.45 * _numeric_column(pool, "park_bonus")
        pool["bundle_score"] -= 0.008 * _numeric_column(pool, "detour_minutes")
    elif strategy == "national_park_priority":
        pool["bundle_score"] += 1.10 * _numeric_column(pool, "is_national_park")
        pool["bundle_score"] += 0.55 * _numeric_column(pool, "park_bonus")
        pool["bundle_score"] += 0.20 * _numeric_column(pool, "nature_score")
    elif strategy == "city_nature_balanced":
        pool["bundle_score"] += 0.35 * _numeric_column(pool, "nature_score")
        pool["bundle_score"] += 0.25 * _numeric_column(pool, "city_score")
        pool["bundle_score"] += 0.20 * _numeric_column(pool, "culture_score")
    elif strategy == "weather_safe_indoor_backup":
        weather_exposure = _numeric_column(pool, "weather_sensitivity") * _numeric_column(pool, "weather_risk", 0.15)
        pool["bundle_score"] += 0.45 * (1.0 - weather_exposure.clip(0.0, 1.0))
        pool["bundle_score"] -= 0.20 * _numeric_column(pool, "outdoor_intensity")
    selected, mmr_log = mmr_select_candidates(
        pool, config, candidate_size=int(candidate_size), score_column="bundle_score"
    )
    output = selected.reset_index(drop=True)
    output.attrs["mmr_log"] = (
        mmr_log.assign(
            arm_id=arm_row.get("arm_id"),
            route_search_strategy=strategy,
            gateway_start=arm_row.get("gateway_start"),
            gateway_end=arm_row.get("gateway_end"),
            candidate_size=int(candidate_size),
        )
        if not mmr_log.empty
        else mmr_log
    )
    return output


def run_bandit_gurobi_stress_benchmark(
    *,
    arms_df: pd.DataFrame,
    enriched_df: pd.DataFrame,
    budget_df: pd.DataFrame,
    config: TripConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stress-test bandit-selected bundles with small Gurobi route repairs."""
    production_enrichment = import_legacy_module("production_enrichment")
    seeds = [int(seed) for seed in _as_list(config.get("bandit", "seeds", [42]), [42])]
    episode_grid = [
        int(value)
        for value in _as_list(config.get("bandit", "episode_grid", [config.get("bandit", "episodes", 40)]), [40])
    ]
    candidate_size_grid = [int(value) for value in _as_list(config.get("bandit", "candidate_size_grid", [12]), [12])]
    repairs = int(config.get("bandit", "top_k_gurobi_repairs", 3))
    budget = production_enrichment.budget_bounds_from_df(budget_df)
    rows = []
    mmr_logs = []

    for seed in seeds:
        for episodes in episode_grid:
            bandit_log_df, bandit_summary_df = production_enrichment.run_hybrid_bandit_optimization_search(
                arms_df,
                episodes=episodes,
                seed=seed,
            )
            selected_arms = bandit_summary_df.head(max(1, repairs)).copy()
            for candidate_size in candidate_size_grid:
                for repair_rank, arm_row in enumerate(selected_arms.itertuples(index=False), start=1):
                    arm_series = pd.Series(arm_row._asdict())
                    candidate_bundle = _candidate_bundle_for_arm(arm_series, enriched_df, candidate_size, config)
                    if (
                        isinstance(candidate_bundle.attrs.get("mmr_log"), pd.DataFrame)
                        and not candidate_bundle.attrs["mmr_log"].empty
                    ):
                        mmr_logs.append(
                            candidate_bundle.attrs["mmr_log"].assign(
                                seed=seed,
                                episodes=episodes,
                                repair_rank=repair_rank,
                            )
                        )
                    route_result = solve_enriched_route_with_gurobi(
                        candidate_bundle, config, candidate_size=candidate_size
                    )
                    budget_penalty = max(
                        0.0, float(arm_series.get("estimated_total_cost", budget.expected_total)) - budget.soft_budget
                    ) / max(1.0, budget.soft_budget)
                    combined_reward = (
                        float(arm_series.get("posterior_mean_reward", 0.0) or 0.0)
                        + 0.25 * float(route_result.get("total_value", 0.0))
                        - 0.10 * float(route_result.get("runtime_seconds", 0.0))
                        - 0.35 * budget_penalty
                    )
                    rows.append(
                        {
                            "seed": seed,
                            "episodes": episodes,
                            "candidate_size": candidate_size,
                            "repair_rank": repair_rank,
                            "arm_id": arm_series.get("arm_id"),
                            "route_search_strategy": arm_series.get("route_search_strategy"),
                            "gateway_start": arm_series.get("gateway_start"),
                            "gateway_end": arm_series.get("gateway_end"),
                            "candidate_bundle_rows": int(len(candidate_bundle)),
                            "selected_pois": " | ".join(route_result.get("selected_pois", [])),
                            "selected_poi_count": int(len(route_result.get("selected_pois", []))),
                            "submodular_diversity": float(
                                route_result.get("diversity_score", submodular_diversity(candidate_bundle))
                            ),
                            "total_detour_minutes": float(route_result.get("total_detour_minutes", 0.0) or 0.0),
                            "total_cost": float(route_result.get("total_cost", 0.0) or 0.0),
                            "total_weather_risk": float(route_result.get("total_weather_risk", 0.0) or 0.0),
                            "interest_adjusted_utility": float(
                                route_result.get("interest_adjusted_utility", route_result.get("total_value", 0.0))
                                or 0.0
                            ),
                            "interest_delta": float(route_result.get("interest_delta", 0.0) or 0.0),
                            "nature_score": float(route_result.get("nature_score", 0.0) or 0.0),
                            "scenic_score": float(route_result.get("scenic_score", 0.0) or 0.0),
                            "national_parks_selected": int(route_result.get("national_parks_selected", 0) or 0),
                            "protected_parks_selected": int(route_result.get("protected_parks_selected", 0) or 0),
                            "outdoor_weather_risk": float(route_result.get("outdoor_weather_risk", 0.0) or 0.0),
                            "seasonality_risk": float(route_result.get("seasonality_risk", 0.0) or 0.0),
                            "balance_score": float(route_result.get("balance_score", 0.0) or 0.0),
                            "epsilon_time_budget": route_result.get("epsilon_time_budget"),
                            "epsilon_cost_budget": route_result.get("epsilon_cost_budget"),
                            "epsilon_detour_minutes": route_result.get("epsilon_detour_minutes"),
                            "epsilon_weather_risk": route_result.get("epsilon_weather_risk"),
                            "epsilon_min_diversity": route_result.get("epsilon_min_diversity"),
                            "posterior_mean_reward": float(arm_series.get("posterior_mean_reward", 0.0) or 0.0),
                            "gurobi_oracle_reward": float(route_result.get("objective_value", 0.0) or 0.0),
                            "combined_reward": float(combined_reward),
                            "estimated_total_cost": float(
                                arm_series.get("estimated_total_cost", budget.expected_total) or budget.expected_total
                            ),
                            "soft_budget": budget.soft_budget,
                            "hard_budget": budget.hard_budget,
                            "hard_budget_feasible": bool(arm_series.get("hard_budget_feasible", True)),
                            "route_feasible": bool(route_result.get("feasible", False)),
                            "small_optimizer_calls": int(arm_series.get("small_optimizer_calls", 0) or 0),
                            "solver_status": route_result.get("solver_status"),
                            "optimality_gap": route_result.get("optimality_gap"),
                            "runtime_seconds": route_result.get("runtime_seconds"),
                            "total_travel_minutes": route_result.get("total_travel_minutes"),
                            "total_visit_minutes": route_result.get("total_visit_minutes"),
                            "data_source_scope": "open_enriched_catalog",
                        }
                    )

    runs_df = pd.DataFrame(rows)
    runs_df.attrs["mmr_candidate_log_df"] = (
        pd.concat(mmr_logs, ignore_index=True, sort=False) if mmr_logs else pd.DataFrame()
    )
    if runs_df.empty:
        return runs_df, pd.DataFrame()
    summary_df = (
        runs_df.groupby(["episodes", "candidate_size", "route_search_strategy"], dropna=False)
        .agg(
            runs=("arm_id", "count"),
            mean_combined_reward=("combined_reward", "mean"),
            best_combined_reward=("combined_reward", "max"),
            mean_runtime_seconds=("runtime_seconds", "mean"),
            feasible_rate=("route_feasible", "mean"),
            mean_selected_poi_count=("selected_poi_count", "mean"),
            mean_optimality_gap=("optimality_gap", "mean"),
        )
        .reset_index()
        .sort_values(["best_combined_reward", "feasible_rate"], ascending=False)
    )
    return runs_df, summary_df
