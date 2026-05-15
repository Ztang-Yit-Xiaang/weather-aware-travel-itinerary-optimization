"""Approximate trip budget estimation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ._legacy import import_legacy_module
from .config import TripConfig


def estimate_budget_range(
    *,
    trip: dict,
    hotels_df: pd.DataFrame,
    osm_city_hotel_catalog_df: pd.DataFrame,
    output_dir: str | Path,
    config: TripConfig,
    primary_city: str = "Santa Barbara",
    write_output: bool = True,
) -> pd.DataFrame:
    production_enrichment = import_legacy_module("production_enrichment")
    trip_days = int(config.get("trip", "trip_days", 7))
    day_base_sequence = production_enrichment.expand_day_city_sequence(trip["days_by_city"])
    overnight_bases = day_base_sequence[: max(0, trip_days - 1)]
    expected_hotel_cost = sum(
        production_enrichment.hotel_price_proxy_for_city(
            city,
            hotels_df,
            osm_city_hotel_catalog_df,
            primary_city=primary_city,
        )
        for city in overnight_bases
    )
    low_hotel_cost = expected_hotel_cost * 0.78
    high_hotel_cost = expected_hotel_cost * 1.35

    attraction_per_day = float(config.get("budget", "attraction_cost_per_day", 28))
    food_per_day = float(config.get("budget", "food_cost_per_day", 72))
    local_buffer = float(config.get("budget", "local_transport_buffer_per_day", 18))
    expected_attraction_cost = attraction_per_day * trip_days
    low_attraction_cost = expected_attraction_cost * 0.36
    high_attraction_cost = expected_attraction_cost * 1.96
    expected_food_cost = food_per_day * trip_days
    low_food_cost = expected_food_cost * 0.67
    high_food_cost = expected_food_cost * 1.60

    drive_hours = float(trip["intercity_drive_minutes"]) / 60.0
    approximate_drive_miles = drive_hours * 48.0
    expected_drive_cost = approximate_drive_miles * 0.24 + local_buffer * trip_days
    low_drive_cost = approximate_drive_miles * 0.16 + (local_buffer * 0.44) * trip_days
    high_drive_cost = approximate_drive_miles * 0.36 + (local_buffer * 1.78) * trip_days

    low_total = (low_hotel_cost + low_attraction_cost + low_food_cost + low_drive_cost) * 1.08
    expected_total = (expected_hotel_cost + expected_attraction_cost + expected_food_cost + expected_drive_cost) * 1.15
    high_total = (high_hotel_cost + high_attraction_cost + high_food_cost + high_drive_cost) * 1.22

    user_budget = production_enrichment.safe_float(config.get("budget", "user_budget", np.nan), np.nan)
    soft_budget = float(user_budget) if np.isfinite(user_budget) and user_budget > 0 else float(expected_total)
    hard_multiplier = float(config.get("budget", "hard_budget_multiplier", 1.20))
    hard_budget = max(float(high_total), soft_budget * hard_multiplier)
    budget_df = pd.DataFrame(
        [
            {"component": "hotel", "low": low_hotel_cost, "expected": expected_hotel_cost, "high": high_hotel_cost},
            {
                "component": "attractions",
                "low": low_attraction_cost,
                "expected": expected_attraction_cost,
                "high": high_attraction_cost,
            },
            {"component": "food", "low": low_food_cost, "expected": expected_food_cost, "high": high_food_cost},
            {
                "component": "fuel_parking_local_transport",
                "low": low_drive_cost,
                "expected": expected_drive_cost,
                "high": high_drive_cost,
            },
            {"component": "buffered_total", "low": low_total, "expected": expected_total, "high": high_total},
            {"component": "soft_budget", "low": soft_budget, "expected": soft_budget, "high": soft_budget},
            {"component": "hard_budget", "low": hard_budget, "expected": hard_budget, "high": hard_budget},
        ]
    ).round(2)
    if write_output:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        budget_df.to_csv(output_dir / "production_budget_estimate.csv", index=False)
    return budget_df


def budget_bounds_from_df(budget_df: pd.DataFrame):
    production_enrichment = import_legacy_module("production_enrichment")
    return production_enrichment.budget_bounds_from_df(budget_df)
