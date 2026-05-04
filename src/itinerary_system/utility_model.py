"""Utility scoring models for open-enriched POI catalogs.

The project keeps Yelp as one local signal, then combines open-data, social,
route-corridor, and uncertainty evidence before optimization. This module
produces three reportable utility variants:

* MCDA weighted score
* TOPSIS score
* Empirical-Bayes UCB score used by the main optimizer
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import TripConfig


DEFAULT_MCDA_WEIGHTS = {
    "base_score": 0.18,
    "yelp_signal": 0.14,
    "social_signal": 0.18,
    "must_go_signal": 0.14,
    "corridor_fit": 0.11,
    "wikipedia_signal": 0.08,
    "data_confidence": 0.10,
    "weather_safety": 0.04,
    "low_detour": 0.03,
}


def _numeric(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column in frame.columns:
        return pd.to_numeric(frame[column], errors="coerce").fillna(default).astype(float)
    return pd.Series(default, index=frame.index, dtype=float)


def _minmax(series: pd.Series, default: float = 0.5) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").fillna(0.0).astype(float)
    if values.empty:
        return values
    span = float(values.max() - values.min())
    if np.isclose(span, 0.0):
        return pd.Series(default, index=values.index, dtype=float)
    return (values - float(values.min())) / span


def _source_count(series: pd.Series) -> pd.Series:
    return series.astype(str).apply(lambda value: len([part for part in value.replace(",", "|").split("|") if part.strip()]))


def _weights_from_config(config: TripConfig) -> dict[str, float]:
    configured = config.get("utility", "mcda_weights", None)
    if isinstance(configured, dict):
        weights = {key: float(configured.get(key, DEFAULT_MCDA_WEIGHTS.get(key, 0.0))) for key in DEFAULT_MCDA_WEIGHTS}
    else:
        weights = dict(DEFAULT_MCDA_WEIGHTS)
    total = sum(max(0.0, value) for value in weights.values())
    if total <= 0:
        return dict(DEFAULT_MCDA_WEIGHTS)
    return {key: max(0.0, value) / total for key, value in weights.items()}


def build_signal_matrix(enriched_df: pd.DataFrame, config: TripConfig) -> pd.DataFrame:
    """Return normalized POI signals used by all scoring models."""
    output = pd.DataFrame(index=enriched_df.index)
    output["name"] = enriched_df.get("name", pd.Series("", index=enriched_df.index)).astype(str)
    output["city"] = enriched_df.get("city", pd.Series("", index=enriched_df.index)).astype(str)
    output["category"] = enriched_df.get("category", pd.Series("", index=enriched_df.index)).astype(str)
    output["source_list"] = enriched_df.get("source_list", pd.Series("", index=enriched_df.index)).astype(str)

    base_source = _numeric(enriched_df, "base_score_norm", np.nan)
    if base_source.isna().all():
        base_source = _minmax(_numeric(enriched_df, "source_score", 0.0))
    output["base_score_signal"] = _minmax(base_source.fillna(0.0))

    yelp_signal = _numeric(enriched_df, "yelp_signal_norm", np.nan)
    if yelp_signal.isna().all():
        yelp_signal = _numeric(enriched_df, "yelp_rating", 0.0) * np.log1p(_numeric(enriched_df, "yelp_review_count", 0.0))
    output["yelp_signal"] = _minmax(yelp_signal.fillna(0.0))
    output["social_signal"] = _numeric(enriched_df, "social_score", 0.0).clip(0.0, 1.0)
    output["must_go_signal"] = (_numeric(enriched_df, "must_go_weight", 0.0) * output["social_signal"]).clip(0.0, 1.0)
    output["corridor_fit"] = _numeric(enriched_df, "corridor_fit", 0.0).clip(0.0, 1.0)
    output["detour_minutes"] = _numeric(enriched_df, "detour_minutes", 0.0).clip(lower=0.0)
    output["low_detour_signal"] = 1.0 - _minmax(output["detour_minutes"], default=0.0)
    output["wikipedia_signal"] = _numeric(enriched_df, "wikipedia_pageview_score", 0.0).clip(0.0, 1.0)
    output["data_confidence"] = _numeric(enriched_df, "data_confidence", 0.15).clip(0.0, 1.0)
    output["data_uncertainty"] = (1.0 - output["data_confidence"]).clip(0.0, 1.0)
    output["weather_risk"] = _numeric(enriched_df, "weather_risk", 0.15).clip(0.0, 1.0)
    output["weather_safety"] = 1.0 - output["weather_risk"]
    output["source_count"] = _source_count(output["source_list"])
    output["review_strength"] = np.log1p(_numeric(enriched_df, "yelp_review_count", 0.0).clip(lower=0.0))
    return output


def score_mcda(signal_df: pd.DataFrame, config: TripConfig) -> pd.Series:
    weights = _weights_from_config(config)
    score = pd.Series(0.0, index=signal_df.index, dtype=float)
    column_map = {
        "base_score": "base_score_signal",
        "yelp_signal": "yelp_signal",
        "social_signal": "social_signal",
        "must_go_signal": "must_go_signal",
        "corridor_fit": "corridor_fit",
        "wikipedia_signal": "wikipedia_signal",
        "data_confidence": "data_confidence",
        "weather_safety": "weather_safety",
        "low_detour": "low_detour_signal",
    }
    for weight_name, column in column_map.items():
        score += float(weights.get(weight_name, 0.0)) * pd.to_numeric(signal_df[column], errors="coerce").fillna(0.0)
    return score.clip(lower=0.0)


def score_topsis(signal_df: pd.DataFrame, config: TripConfig) -> pd.Series:
    """Compute TOPSIS closeness using the same normalized criteria."""
    criteria = [
        "base_score_signal",
        "yelp_signal",
        "social_signal",
        "must_go_signal",
        "corridor_fit",
        "wikipedia_signal",
        "data_confidence",
        "weather_safety",
        "low_detour_signal",
    ]
    matrix = signal_df[criteria].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    if matrix.size == 0:
        return pd.Series(dtype=float)
    weights = _weights_from_config(config)
    weight_vector = np.array(
        [
            weights["base_score"],
            weights["yelp_signal"],
            weights["social_signal"],
            weights["must_go_signal"],
            weights["corridor_fit"],
            weights["wikipedia_signal"],
            weights["data_confidence"],
            weights["weather_safety"],
            weights["low_detour"],
        ],
        dtype=float,
    )
    denom = np.sqrt((matrix**2).sum(axis=0))
    denom[denom == 0] = 1.0
    weighted = (matrix / denom) * weight_vector
    ideal_best = weighted.max(axis=0)
    ideal_worst = weighted.min(axis=0)
    distance_best = np.sqrt(((weighted - ideal_best) ** 2).sum(axis=1))
    distance_worst = np.sqrt(((weighted - ideal_worst) ** 2).sum(axis=1))
    closeness = distance_worst / np.maximum(distance_best + distance_worst, 1e-12)
    return pd.Series(closeness, index=signal_df.index, dtype=float).fillna(0.0)


def score_bayesian_ucb(signal_df: pd.DataFrame, config: TripConfig) -> pd.DataFrame:
    """Empirical-Bayes posterior mean/variance plus UCB route/social terms."""
    mcda = score_mcda(signal_df, config)
    prior_mean = float(config.get("utility", "bayes_prior_mean", 0.50))
    prior_strength = float(config.get("utility", "bayes_prior_strength", 6.0))
    kappa = float(config.get("utility", "uncertainty_bonus_kappa", 0.25))
    corridor_weight = float(config.get("utility", "corridor_fit_weight", config.get("social", "corridor_fit_weight", 0.30)))
    detour_penalty = float(config.get("utility", "detour_penalty_weight", config.get("social", "detour_penalty_weight", 0.01)))
    must_go_weight = float(config.get("utility", "must_go_bonus_weight", config.get("social", "must_go_bonus_weight", 0.85)))
    weather_penalty = float(config.get("utility", "weather_risk_penalty_weight", 0.08))

    evidence_strength = (
        1.0
        + signal_df["review_strength"]
        + 2.0 * signal_df["source_count"]
        + 5.0 * signal_df["data_confidence"]
    ).clip(lower=1.0)
    posterior_mean = (prior_strength * prior_mean + evidence_strength * mcda) / (prior_strength + evidence_strength)
    posterior_variance = (
        posterior_mean.clip(0.0, 1.0)
        * (1.0 - posterior_mean.clip(0.0, 1.0))
        / (prior_strength + evidence_strength + 1.0)
    ).clip(lower=1e-6)
    posterior_std = np.sqrt(posterior_variance)
    bayes_ucb = (
        posterior_mean
        + kappa * posterior_std
        + corridor_weight * signal_df["corridor_fit"]
        - detour_penalty * signal_df["detour_minutes"]
        + must_go_weight * signal_df["must_go_signal"]
        - weather_penalty * signal_df["weather_risk"]
    ).clip(lower=0.0)
    return pd.DataFrame(
        {
            "utility_posterior_mean": posterior_mean.astype(float),
            "utility_posterior_variance": posterior_variance.astype(float),
            "utility_posterior_std": posterior_std.astype(float),
            "utility_bayesian_ucb": bayes_ucb.astype(float),
            "utility_uncertainty_bonus": (kappa * posterior_std).astype(float),
        },
        index=signal_df.index,
    )


def apply_utility_models(
    enriched_df: pd.DataFrame,
    output_dir: str | Path | None,
    config: TripConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Score an enriched catalog and write report-ready utility artifacts."""
    output = enriched_df.copy()
    if output.empty:
        empty = pd.DataFrame()
        return output, empty, empty

    signal_df = build_signal_matrix(output, config)
    mcda = score_mcda(signal_df, config)
    topsis = score_topsis(signal_df, config)
    bayesian = score_bayesian_ucb(signal_df, config)
    method = str(config.get("utility", "method", "bayesian_ucb"))
    selected_column = {
        "mcda_weighted": "utility_mcda_weighted",
        "topsis": "utility_topsis",
        "bayesian_ucb": "utility_bayesian_ucb",
    }.get(method, "utility_bayesian_ucb")

    output["pre_utility_final_poi_value"] = _numeric(output, "final_poi_value", 0.0)
    output["utility_mcda_weighted"] = mcda
    output["utility_topsis"] = topsis
    for column in bayesian.columns:
        output[column] = bayesian[column]
    output["final_poi_value"] = pd.to_numeric(output[selected_column], errors="coerce").fillna(0.0).clip(lower=0.0)
    output["utility_method"] = method
    output["data_uncertainty"] = (1.0 - _numeric(output, "data_confidence", 0.15)).clip(0.0, 1.0)

    signal_output = signal_df.copy()
    signal_output["utility_mcda_weighted"] = output["utility_mcda_weighted"]
    signal_output["utility_topsis"] = output["utility_topsis"]
    signal_output["utility_bayesian_ucb"] = output["utility_bayesian_ucb"]

    utility_columns = [
        "name",
        "city",
        "category",
        "source_list",
        "utility_method",
        "pre_utility_final_poi_value",
        "utility_mcda_weighted",
        "utility_topsis",
        "utility_posterior_mean",
        "utility_posterior_variance",
        "utility_posterior_std",
        "utility_uncertainty_bonus",
        "utility_bayesian_ucb",
        "final_poi_value",
        "data_confidence",
        "data_uncertainty",
        "corridor_fit",
        "detour_minutes",
        "social_score",
        "must_go_weight",
        "social_must_go",
    ]
    utility_scores = output[[column for column in utility_columns if column in output.columns]].copy()
    utility_scores = utility_scores.sort_values("final_poi_value", ascending=False).reset_index(drop=True)
    audit_df = pd.DataFrame(
        [
            {
                "audit_type": "utility_model",
                "utility_method": method,
                "selected_score_column": selected_column,
                "rows_scored": int(len(output)),
                "mean_posterior_std": float(output["utility_posterior_std"].mean()),
                "sparse_data_policy": "uncertainty_bonus_not_value_penalty",
                "must_go_policy": "soft_reward_not_mandatory",
            }
        ]
    )

    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        signal_output.to_csv(output_path / "production_signal_matrix.csv", index=False)
        utility_scores.to_csv(output_path / "production_utility_scores.csv", index=False)
        audit_df.to_csv(output_path / "production_utility_model_audit.csv", index=False)
    return output.sort_values(["final_poi_value", "social_score"], ascending=[False, False]).reset_index(drop=True), utility_scores, audit_df


def utility_score_columns() -> list[str]:
    return [
        "utility_mcda_weighted",
        "utility_topsis",
        "utility_posterior_mean",
        "utility_posterior_variance",
        "utility_posterior_std",
        "utility_bayesian_ucb",
    ]


def learning_to_rank_audit(enriched_df: pd.DataFrame, config: TripConfig) -> pd.DataFrame:
    """Audit whether an optional BPR/LTR benchmark is defensible yet."""
    min_pairs = int(config.get("learning_to_rank", "min_pairwise_examples", 200))
    enabled = bool(config.get("learning_to_rank", "enabled", False))
    candidate_rows = int(len(enriched_df))
    rough_pair_count = int(candidate_rows * max(0, candidate_rows - 1) / 2)
    status = "disabled_by_config"
    if enabled and rough_pair_count >= min_pairs:
        status = "ready_for_pairwise_bpr_benchmark"
    elif enabled:
        status = "insufficient_pairwise_examples"
    return pd.DataFrame(
        [
            {
                "audit_type": "learning_to_rank",
                "enabled": enabled,
                "method": config.get("learning_to_rank", "method", "bpr"),
                "candidate_rows": candidate_rows,
                "rough_pair_count": rough_pair_count,
                "min_pairwise_examples": min_pairs,
                "status": status,
            }
        ]
    )
