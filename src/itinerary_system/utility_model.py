"""Utility scoring models for open-enriched POI catalogs.

The project keeps Yelp as one local signal, then combines open-data, social,
route-corridor, and uncertainty evidence before optimization. This module
produces three reportable utility variants:

* MCDA weighted score
* TOPSIS score
* Empirical-Bayes UCB score used by the main optimizer
"""

from __future__ import annotations

from dataclasses import dataclass
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

SOURCE_FAMILIES = ("osm", "yelp", "curated", "wikidata", "wikipedia", "weather", "route")
SOURCE_COVERAGE_WEIGHTS = {
    "osm": 0.25,
    "yelp": 0.20,
    "curated": 0.15,
    "wikidata": 0.10,
    "wikipedia": 0.10,
    "weather": 0.10,
    "route": 0.10,
}
UTILITY_TERM_COLUMNS = {
    "base_score": "base_score_signal",
    "yelp_signal": "yelp_signal",
    "social_signal": "social_signal",
    "must_go_signal": "must_go_signal",
    "corridor_fit": "corridor_fit",
    "wikipedia_signal": "wikipedia_signal",
    "weather_safety": "weather_safety",
    "low_detour": "low_detour_signal",
}
UTILITY_TERM_AVAILABILITY = {
    "base_score": "has_base_score",
    "yelp_signal": "has_yelp",
    "social_signal": "has_social",
    "must_go_signal": "has_social",
    "corridor_fit": "has_route",
    "wikipedia_signal": "has_wikipedia",
    "weather_safety": "has_weather",
    "low_detour": "has_route",
}
UTILITY_TERM_SOURCE = {
    "base_score": "curated",
    "yelp_signal": "yelp",
    "corridor_fit": "route",
    "wikipedia_signal": "wikipedia",
    "weather_safety": "weather",
    "low_detour": "route",
}


@dataclass(frozen=True)
class SourceSignalMask:
    """Availability mask for source-family utility terms."""

    poi_id: str
    available_sources: frozenset[str]
    missing_sources: frozenset[str]
    source_weights: dict[str, float]

    def is_available(self, source: str) -> bool:
        return str(source) in self.available_sources

    def active_weight_sum(self) -> float:
        return float(sum(self.source_weights.get(source, 0.0) for source in self.available_sources))

    def to_record(self) -> dict[str, Any]:
        return {
            "poi_id": self.poi_id,
            "available_sources": "|".join(sorted(self.available_sources)),
            "missing_sources": "|".join(sorted(self.missing_sources)),
            "active_source_weight": self.active_weight_sum(),
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
    return series.astype(str).apply(
        lambda value: len([part for part in value.replace(",", "|").split("|") if part.strip()])
    )


def _text_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series("", index=frame.index, dtype=str)
    return frame[column].fillna("").astype(str)


def _nonempty_text(frame: pd.DataFrame, column: str) -> pd.Series:
    text = _text_series(frame, column).str.strip()
    return text.ne("") & ~text.str.lower().isin({"nan", "none", "null"})


def _source_contains(frame: pd.DataFrame, pattern: str) -> pd.Series:
    return _text_series(frame, "source_list").str.contains(pattern, case=False, na=False, regex=True)


def _has_numeric(frame: pd.DataFrame, column: str, *, positive: bool = False) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(False, index=frame.index, dtype=bool)
    values = pd.to_numeric(frame[column], errors="coerce")
    present = values.notna()
    return present & values.gt(0.0) if positive else present


def _source_coverage_from_masks(mask_df: pd.DataFrame) -> pd.Series:
    coverage = pd.Series(0.0, index=mask_df.index, dtype=float)
    total_weight = float(sum(SOURCE_COVERAGE_WEIGHTS.values()))
    if total_weight <= 0:
        raise ValueError("source coverage weights must sum to a positive value")
    for source, weight in SOURCE_COVERAGE_WEIGHTS.items():
        coverage += float(weight) * mask_df[f"has_{source}"].astype(float)
    return (coverage / total_weight).clip(0.0, 1.0)


def build_source_masks(enriched_df: pd.DataFrame) -> pd.DataFrame:
    """Return explicit source availability masks and source coverage fields."""

    mask_df = pd.DataFrame(index=enriched_df.index)
    poi_source = enriched_df.get("poi_id", enriched_df.get("id", enriched_df.get("name", pd.Series("", index=enriched_df.index))))
    mask_df["poi_id"] = poi_source.fillna("").astype(str)
    mask_df["has_osm"] = _source_contains(enriched_df, "osm|overpass|openstreetmap") | _nonempty_text(enriched_df, "osm_tags")
    mask_df["has_yelp"] = (
        _source_contains(enriched_df, "yelp")
        | _has_numeric(enriched_df, "yelp_rating")
        | _has_numeric(enriched_df, "yelp_review_count", positive=True)
    )
    mask_df["has_curated"] = _source_contains(enriched_df, "curated|seed|manual")
    mask_df["has_wikidata"] = _nonempty_text(enriched_df, "wikidata_id")
    mask_df["has_wikipedia"] = (
        _source_contains(enriched_df, "wikipedia")
        | _nonempty_text(enriched_df, "wikipedia_title")
        | _has_numeric(enriched_df, "wikipedia_pageview_score")
    )
    mask_df["has_weather"] = _has_numeric(enriched_df, "weather_risk")
    mask_df["has_route"] = (
        _source_contains(enriched_df, "route|osrm")
        | _has_numeric(enriched_df, "route_fit")
        | _has_numeric(enriched_df, "corridor_fit")
        | _has_numeric(enriched_df, "detour_minutes")
    )
    mask_df["has_social"] = _has_numeric(enriched_df, "social_score") | _has_numeric(enriched_df, "must_go_weight")
    mask_df["has_base_score"] = (
        _has_numeric(enriched_df, "base_score_norm")
        | _has_numeric(enriched_df, "source_score")
        | mask_df["has_curated"]
        | mask_df["has_osm"]
    )
    computed_coverage = _source_coverage_from_masks(mask_df)
    provided_coverage = _numeric(enriched_df, "source_coverage_score", np.nan)
    mask_df["source_coverage_score"] = provided_coverage.fillna(computed_coverage).clip(0.0, 1.0)
    mask_df["data_confidence"] = mask_df["source_coverage_score"]
    mask_df["missing_source_list"] = mask_df.apply(
        lambda row: "|".join(source for source in SOURCE_FAMILIES if not bool(row[f"has_{source}"])),
        axis=1,
    )
    mask_df["available_source_list"] = mask_df.apply(
        lambda row: "|".join(source for source in SOURCE_FAMILIES if bool(row[f"has_{source}"])),
        axis=1,
    )
    return mask_df


def normalize_source_signal(series: pd.Series, mask: pd.Series, default: float = 0.0) -> pd.Series:
    """Normalize only available source values; unavailable rows receive default."""

    values = pd.to_numeric(series, errors="coerce")
    active = mask.fillna(False).astype(bool)
    output = pd.Series(float(default), index=series.index, dtype=float)
    if not active.any():
        return output
    active_values = values[active]
    normalized = _minmax(active_values, default=0.5)
    output.loc[active] = normalized
    return output.clip(0.0, 1.0)


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
    masks = build_source_masks(enriched_df)
    output = pd.DataFrame(index=enriched_df.index)
    output["poi_id"] = masks["poi_id"]
    output["name"] = enriched_df.get("name", pd.Series("", index=enriched_df.index)).astype(str)
    output["city"] = enriched_df.get("city", pd.Series("", index=enriched_df.index)).astype(str)
    output["category"] = enriched_df.get("category", pd.Series("", index=enriched_df.index)).astype(str)
    output["source_list"] = enriched_df.get("source_list", pd.Series("", index=enriched_df.index)).astype(str)
    for column in masks.columns:
        if column != "poi_id":
            output[column] = masks[column]

    base_source = _numeric(enriched_df, "base_score_norm", np.nan)
    if base_source.isna().all():
        base_source = _minmax(_numeric(enriched_df, "source_score", 0.0))
    output["base_score_signal"] = normalize_source_signal(base_source, output["has_base_score"])

    yelp_signal = _numeric(enriched_df, "yelp_signal_norm", np.nan)
    if yelp_signal.isna().all():
        yelp_signal = _numeric(enriched_df, "yelp_rating", 0.0) * np.log1p(
            _numeric(enriched_df, "yelp_review_count", 0.0)
        )
    output["yelp_signal"] = normalize_source_signal(yelp_signal, output["has_yelp"])
    output["social_signal"] = _numeric(enriched_df, "social_score", 0.0).clip(0.0, 1.0)
    output["must_go_signal"] = (_numeric(enriched_df, "must_go_weight", 0.0) * output["social_signal"]).clip(0.0, 1.0)
    output["corridor_fit"] = _numeric(enriched_df, "corridor_fit", 0.0).clip(0.0, 1.0)
    output["detour_minutes"] = _numeric(enriched_df, "detour_minutes", 0.0).clip(lower=0.0)
    output["low_detour_signal"] = normalize_source_signal(
        1.0 - _minmax(output["detour_minutes"], default=0.0),
        output["has_route"],
    )
    output["wikipedia_signal"] = normalize_source_signal(
        _numeric(enriched_df, "wikipedia_pageview_score", 0.0),
        output["has_wikipedia"],
    )
    output["model_uncertainty"] = _numeric(enriched_df, "model_uncertainty", 0.0).clip(0.0, 1.0)
    output["data_uncertainty"] = _numeric(enriched_df, "data_uncertainty", np.nan)
    if output["data_uncertainty"].isna().all():
        output["data_uncertainty"] = output["model_uncertainty"]
    output["data_uncertainty"] = output["data_uncertainty"].fillna(output["model_uncertainty"]).clip(0.0, 1.0)
    output["weather_risk"] = _numeric(enriched_df, "weather_risk", 0.15).clip(0.0, 1.0)
    output["weather_safety"] = 1.0 - output["weather_risk"]
    output["source_count"] = _source_count(output["source_list"])
    output["review_strength"] = np.log1p(_numeric(enriched_df, "yelp_review_count", 0.0).clip(lower=0.0))
    output["active_source_weight"] = active_utility_weight(output, config)
    output["utility_masked_mcda"] = score_masked_weighted_utility(output, config)
    return output


def active_utility_weight(
    signal_df: pd.DataFrame,
    config: TripConfig,
    *,
    disabled_sources: set[str] | frozenset[str] | tuple[str, ...] = (),
) -> pd.Series:
    """Return the row-wise active MCDA denominator after missing-source masks."""

    weights = _weights_from_config(config)
    disabled = {str(source) for source in disabled_sources}
    active_weight = pd.Series(0.0, index=signal_df.index, dtype=float)
    for weight_name in UTILITY_TERM_COLUMNS:
        source = UTILITY_TERM_SOURCE.get(weight_name)
        if source in disabled:
            continue
        availability_column = UTILITY_TERM_AVAILABILITY[weight_name]
        available = signal_df.get(availability_column, pd.Series(True, index=signal_df.index)).astype(bool)
        active_weight += float(weights.get(weight_name, 0.0)) * available.astype(float)
    return active_weight.astype(float)


def score_masked_weighted_utility(
    signal_df: pd.DataFrame,
    config: TripConfig,
    *,
    disabled_sources: set[str] | frozenset[str] | tuple[str, ...] = (),
) -> pd.Series:
    """Score MCDA terms while excluding unavailable source terms from the denominator."""

    weights = _weights_from_config(config)
    disabled = {str(source) for source in disabled_sources}
    numerator = pd.Series(0.0, index=signal_df.index, dtype=float)
    denominator = pd.Series(0.0, index=signal_df.index, dtype=float)
    for weight_name, column in UTILITY_TERM_COLUMNS.items():
        source = UTILITY_TERM_SOURCE.get(weight_name)
        if source in disabled:
            continue
        weight = float(weights.get(weight_name, 0.0))
        if weight <= 0:
            continue
        availability_column = UTILITY_TERM_AVAILABILITY[weight_name]
        available = signal_df.get(availability_column, pd.Series(True, index=signal_df.index)).astype(bool)
        values = pd.to_numeric(signal_df[column], errors="coerce").fillna(0.0).clip(0.0, 1.0)
        numerator += weight * values * available.astype(float)
        denominator += weight * available.astype(float)
    fallback = float(config.get("utility", "missing_source_fallback_utility", 0.15))
    score = numerator / denominator.replace(0.0, np.nan)
    return score.fillna(fallback).clip(lower=0.0)


def score_mcda(signal_df: pd.DataFrame, config: TripConfig) -> pd.Series:
    return score_masked_weighted_utility(signal_df, config)


def score_topsis(signal_df: pd.DataFrame, config: TripConfig) -> pd.Series:
    """Compute TOPSIS closeness using the same normalized criteria."""
    criteria = list(UTILITY_TERM_COLUMNS.values())
    normalized_columns = []
    for weight_name, column in UTILITY_TERM_COLUMNS.items():
        values = pd.to_numeric(signal_df[column], errors="coerce").fillna(0.0).clip(0.0, 1.0)
        availability_column = UTILITY_TERM_AVAILABILITY[weight_name]
        available = signal_df.get(availability_column, pd.Series(True, index=signal_df.index)).astype(bool)
        neutral = float(values[available].mean()) if available.any() else 0.5
        normalized_columns.append(values.where(available, neutral))
    matrix = pd.concat(normalized_columns, axis=1)
    matrix.columns = criteria
    matrix_values = matrix.to_numpy(dtype=float)
    if matrix_values.size == 0:
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
            weights["weather_safety"],
            weights["low_detour"],
        ],
        dtype=float,
    )
    denom = np.sqrt((matrix_values**2).sum(axis=0))
    denom[denom == 0] = 1.0
    weighted = (matrix_values / denom) * weight_vector
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
    corridor_weight = float(
        config.get("utility", "corridor_fit_weight", config.get("social", "corridor_fit_weight", 0.30))
    )
    detour_penalty = float(
        config.get("utility", "detour_penalty_weight", config.get("social", "detour_penalty_weight", 0.01))
    )
    must_go_weight = float(
        config.get("utility", "must_go_bonus_weight", config.get("social", "must_go_bonus_weight", 0.85))
    )
    weather_penalty = float(config.get("utility", "weather_risk_penalty_weight", 0.08))

    evidence_strength = (
        1.0 + signal_df["review_strength"] + 2.0 * signal_df["source_count"] + 5.0 * signal_df["data_confidence"]
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


class MaskedUtilitySignalBuilder:
    """Stateless facade for source masks and masked utility signals."""

    @staticmethod
    def build_masks(frame: pd.DataFrame) -> pd.DataFrame:
        return build_source_masks(frame)

    @staticmethod
    def build_signals(frame: pd.DataFrame, config: TripConfig) -> pd.DataFrame:
        return build_signal_matrix(frame, config)

    @staticmethod
    def score_masked_mcda(signals: pd.DataFrame, config: TripConfig) -> pd.Series:
        return score_masked_weighted_utility(signals, config)


class SourceAblationReport:
    """Create deterministic utility source-family ablation rows."""

    @staticmethod
    def compute(enriched_df: pd.DataFrame, config: TripConfig) -> pd.DataFrame:
        signal_df = build_signal_matrix(enriched_df, config)
        baseline = score_masked_weighted_utility(signal_df, config)
        rows: list[dict[str, Any]] = []
        for source in SOURCE_FAMILIES:
            ablated = score_masked_weighted_utility(signal_df, config, disabled_sources=(source,))
            for idx in signal_df.index:
                rows.append(
                    {
                        "poi_id": signal_df.at[idx, "poi_id"],
                        "name": signal_df.at[idx, "name"],
                        "source_family": source,
                        "source_available": bool(signal_df.at[idx, f"has_{source}"]),
                        "baseline_utility_masked_mcda": float(baseline.loc[idx]),
                        "utility_without_source": float(ablated.loc[idx]),
                        "utility_delta": float(baseline.loc[idx] - ablated.loc[idx]),
                        "source_coverage_score": float(signal_df.at[idx, "source_coverage_score"]),
                    }
                )
        return pd.DataFrame(rows).sort_values(["poi_id", "source_family"]).reset_index(drop=True)

    @staticmethod
    def write(enriched_df: pd.DataFrame, output_dir: str | Path, config: TripConfig) -> Path:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        audit = SourceAblationReport.compute(enriched_df, config)
        path = output_path / "production_utility_source_ablation.csv"
        audit.to_csv(path, index=False)
        return path


def write_source_ablation_audit(enriched_df: pd.DataFrame, output_dir: Path, config: TripConfig) -> pd.DataFrame:
    """Write and return deterministic source-family ablation rows."""

    output_dir.mkdir(parents=True, exist_ok=True)
    audit = SourceAblationReport.compute(enriched_df, config)
    audit.to_csv(output_dir / "production_utility_source_ablation.csv", index=False)
    return audit


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
    output["utility_masked_mcda"] = mcda
    output["utility_topsis"] = topsis
    for column in bayesian.columns:
        output[column] = bayesian[column]
    output["final_poi_value"] = pd.to_numeric(output[selected_column], errors="coerce").fillna(0.0).clip(lower=0.0)
    output["utility_method"] = method
    for column in [
        "source_coverage_score",
        "data_confidence",
        "active_source_weight",
        "missing_source_list",
        "available_source_list",
        "has_osm",
        "has_yelp",
        "has_curated",
        "has_wikidata",
        "has_wikipedia",
        "has_weather",
        "has_route",
    ]:
        if column in signal_df.columns:
            output[column] = signal_df[column]
    output["source_coverage_score"] = _numeric(output, "source_coverage_score", 0.0).clip(0.0, 1.0)
    output["data_confidence"] = output["source_coverage_score"]
    if "model_uncertainty" not in output.columns:
        output["model_uncertainty"] = 0.0
    output["data_uncertainty"] = _numeric(output, "data_uncertainty", np.nan)
    if output["data_uncertainty"].isna().all():
        output["data_uncertainty"] = _numeric(output, "model_uncertainty", 0.0).clip(0.0, 1.0)
    output["data_uncertainty"] = (
        output["data_uncertainty"].fillna(_numeric(output, "model_uncertainty", 0.0)).clip(0.0, 1.0)
    )

    signal_output = signal_df.copy()
    signal_output["utility_mcda_weighted"] = output["utility_mcda_weighted"]
    signal_output["utility_masked_mcda"] = output["utility_masked_mcda"]
    signal_output["utility_topsis"] = output["utility_topsis"]
    signal_output["utility_bayesian_ucb"] = output["utility_bayesian_ucb"]

    utility_columns = [
        "name",
        "city",
        "category",
        "source_list",
        "utility_method",
        "pre_utility_final_poi_value",
        "utility_masked_mcda",
        "utility_mcda_weighted",
        "utility_topsis",
        "utility_posterior_mean",
        "utility_posterior_variance",
        "utility_posterior_std",
        "utility_uncertainty_bonus",
        "utility_bayesian_ucb",
        "final_poi_value",
        "data_confidence",
        "source_coverage_score",
        "active_source_weight",
        "missing_source_list",
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
                "missing_source_policy": "unavailable_source_terms_omitted_from_utility_denominator",
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
        write_source_ablation_audit(output, output_path, config)
    return (
        output.sort_values(["final_poi_value", "social_score"], ascending=[False, False]).reset_index(drop=True),
        utility_scores,
        audit_df,
    )


def utility_score_columns() -> list[str]:
    return [
        "utility_masked_mcda",
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
