"""Diversity and MMR candidate selection utilities."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from geopy.distance import geodesic

from .config import TripConfig


def _numeric(row: pd.Series, column: str, default: float = 0.0) -> float:
    try:
        value = row.get(column, default)
        if pd.isna(value):
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _category_key(value: Any) -> str:
    text = str(value or "unknown").lower()
    for separator in [":", ",", "|", "/"]:
        if separator in text:
            text = text.split(separator, 1)[0]
    text = "".join(ch if ch.isalnum() else "_" for ch in text).strip("_")
    return text or "unknown"


def category_labels(frame: pd.DataFrame) -> pd.Series:
    if "category" not in frame.columns:
        return pd.Series("unknown", index=frame.index)
    return frame["category"].apply(_category_key)


def submodular_diversity_from_labels(labels: pd.Series | list[str]) -> float:
    values = pd.Series(labels).dropna().astype(str)
    if values.empty:
        return 0.0
    counts = values.value_counts()
    return float(sum(math.sqrt(float(count)) for count in counts))


def submodular_diversity(frame: pd.DataFrame) -> float:
    return submodular_diversity_from_labels(category_labels(frame))


def poi_similarity(left: pd.Series, right: pd.Series, config: TripConfig) -> float:
    """Similarity used by MMR to avoid duplicate categories/locations."""
    diversity_cfg = config.section("diversity")
    category_weight = float(diversity_cfg.get("category_similarity_weight", 0.45))
    geo_weight = float(diversity_cfg.get("geo_similarity_weight", 0.25))
    city_weight = float(diversity_cfg.get("city_similarity_weight", 0.20))
    source_weight = float(diversity_cfg.get("source_similarity_weight", 0.10))

    left_category = _category_key(left.get("category", ""))
    right_category = _category_key(right.get("category", ""))
    category_sim = 1.0 if left_category == right_category else 0.0
    city_sim = 1.0 if str(left.get("city", "")).lower() == str(right.get("city", "")).lower() else 0.0

    try:
        left_point = (float(left["latitude"]), float(left["longitude"]))
        right_point = (float(right["latitude"]), float(right["longitude"]))
        distance_km = geodesic(left_point, right_point).km
        geo_sim = math.exp(-distance_km / 8.0)
    except Exception:
        geo_sim = 0.0

    left_sources = {part.strip().lower() for part in str(left.get("source_list", "")).replace(",", "|").split("|") if part.strip()}
    right_sources = {part.strip().lower() for part in str(right.get("source_list", "")).replace(",", "|").split("|") if part.strip()}
    if left_sources or right_sources:
        source_sim = len(left_sources & right_sources) / max(1, len(left_sources | right_sources))
    else:
        source_sim = 0.0

    return float(
        category_weight * category_sim
        + geo_weight * geo_sim
        + city_weight * city_sim
        + source_weight * source_sim
    )


def mmr_select_candidates(
    candidate_df: pd.DataFrame,
    config: TripConfig,
    *,
    candidate_size: int,
    score_column: str = "final_poi_value",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select a diverse high-value bundle using maximal marginal relevance."""
    if candidate_df.empty or candidate_size <= 0:
        return candidate_df.head(0).copy(), pd.DataFrame()

    mmr_lambda = float(config.get("diversity", "mmr_lambda", 0.72))
    pool = candidate_df.copy().reset_index(drop=True)
    if score_column not in pool.columns:
        score_column = "final_poi_value" if "final_poi_value" in pool.columns else pool.select_dtypes("number").columns[0]
    scores = pd.to_numeric(pool[score_column], errors="coerce").fillna(0.0)
    if scores.max() > scores.min():
        norm_scores = (scores - scores.min()) / (scores.max() - scores.min())
    else:
        norm_scores = pd.Series(0.5, index=pool.index)
    selected: list[int] = []
    remaining = set(pool.index.tolist())
    log_rows = []

    while remaining and len(selected) < int(candidate_size):
        best_idx = None
        best_score = -np.inf
        best_redundancy = 0.0
        for idx in sorted(remaining):
            if not selected:
                redundancy = 0.0
            else:
                redundancy = max(poi_similarity(pool.loc[idx], pool.loc[chosen], config) for chosen in selected)
            mmr_score = mmr_lambda * float(norm_scores.loc[idx]) - (1.0 - mmr_lambda) * redundancy
            if mmr_score > best_score:
                best_idx = int(idx)
                best_score = float(mmr_score)
                best_redundancy = float(redundancy)
        if best_idx is None:
            break
        selected.append(best_idx)
        remaining.remove(best_idx)
        log_rows.append(
            {
                "selection_order": len(selected),
                "name": pool.loc[best_idx, "name"] if "name" in pool.columns else best_idx,
                "city": pool.loc[best_idx, "city"] if "city" in pool.columns else "",
                "category": pool.loc[best_idx, "category"] if "category" in pool.columns else "",
                "score_column": score_column,
                "utility_component": float(norm_scores.loc[best_idx]),
                "max_similarity_to_selected": best_redundancy,
                "mmr_score": best_score,
                "submodular_diversity_after_selection": submodular_diversity(pool.loc[selected]),
            }
        )

    selected_df = pool.loc[selected].copy().reset_index(drop=True)
    selected_df["mmr_selection_order"] = range(1, len(selected_df) + 1)
    log_df = pd.DataFrame(log_rows)
    return selected_df, log_df


def diversity_audit_row(frame: pd.DataFrame, label: str) -> dict[str, Any]:
    labels = category_labels(frame)
    return {
        "label": label,
        "row_count": int(len(frame)),
        "unique_categories": int(labels.nunique()) if len(labels) else 0,
        "submodular_diversity": submodular_diversity_from_labels(labels),
        "top_categories": " | ".join(labels.value_counts().head(5).index.astype(str).tolist()) if len(labels) else "",
    }
