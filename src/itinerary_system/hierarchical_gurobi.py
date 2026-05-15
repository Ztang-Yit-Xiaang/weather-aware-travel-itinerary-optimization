"""Gurobi-backed hierarchical trip planning."""

from __future__ import annotations

import itertools
import math

import pandas as pd
from geopy.distance import geodesic

from .config import TripConfig
from .region_scenarios import (
    NatureRegionDefinition,
    all_scenario_coordinates,
    get_nature_region_definitions,
    get_route_options,
    get_scenario_definition,
)

CITY_COORDS = {
    "San Diego": (32.7157, -117.1611),
    "Los Angeles": (34.0522, -118.2437),
    "Santa Barbara": (34.4208, -119.6982),
    "San Luis Obispo": (35.2828, -120.6596),
    "Monterey": (36.6002, -121.8947),
    "Santa Cruz": (36.9741, -122.0308),
    "San Francisco": (37.7749, -122.4194),
}
CITY_COORDS.update(all_scenario_coordinates())

DEFAULT_SCENARIO_ROUTES = {
    ("Los Angeles", "Los Angeles"): ["Los Angeles", "Santa Barbara", "San Luis Obispo", "Monterey", "Los Angeles"],
    ("Los Angeles", "San Francisco"): [
        "Los Angeles",
        "Santa Barbara",
        "San Luis Obispo",
        "Monterey",
        "Santa Cruz",
        "San Francisco",
    ],
    ("San Francisco", "Los Angeles"): [
        "San Francisco",
        "Santa Cruz",
        "Monterey",
        "San Luis Obispo",
        "Santa Barbara",
        "Los Angeles",
    ],
    ("San Francisco", "San Francisco"): [
        "San Francisco",
        "Santa Cruz",
        "Monterey",
        "San Luis Obispo",
        "Santa Barbara",
        "San Francisco",
    ],
}


def city_score(city: str, city_catalog_summary_df: pd.DataFrame, default=0.45) -> float:
    row = city_catalog_summary_df[city_catalog_summary_df["city"].astype(str).eq(city)]
    if row.empty:
        return float(default)
    return max(0.05, float(row.iloc[0].get("city_value_score", default)))


def data_uncertainty(city: str, city_catalog_summary_df: pd.DataFrame) -> float:
    row = city_catalog_summary_df[city_catalog_summary_df["city"].astype(str).eq(city)]
    if row.empty:
        return 1.0
    return float(row.iloc[0].get("data_uncertainty", 1.0))


def city_must_go_score(city: str, city_catalog_summary_df: pd.DataFrame) -> float:
    row = city_catalog_summary_df[city_catalog_summary_df["city"].astype(str).eq(city)]
    if row.empty:
        return 0.0
    record = row.iloc[0]
    explicit_score = float(record.get("city_must_go_score", 0.0) or 0.0)
    if explicit_score > 0:
        return explicit_score
    count = float(record.get("city_must_go_count", 0.0) or 0.0)
    social = float(record.get("city_social_score", record.get("social_signal_score", 0.0)) or 0.0)
    return count * max(0.0, social)


def city_social_score(city: str, city_catalog_summary_df: pd.DataFrame) -> float:
    row = city_catalog_summary_df[city_catalog_summary_df["city"].astype(str).eq(city)]
    if row.empty:
        return 0.0
    return float(row.iloc[0].get("city_social_score", row.iloc[0].get("social_signal_score", 0.0)) or 0.0)


def must_go_city_weight(config: TripConfig) -> float:
    return float(config.get("social", "must_go_bonus_weight", 0.85)) * 0.18


def assert_sparse_data_is_not_penalized(city_catalog_summary_df: pd.DataFrame) -> None:
    required = {"city_value_score", "data_uncertainty", "data_confidence"}
    missing = required - set(city_catalog_summary_df.columns)
    if missing:
        raise ValueError(f"City summary missing scoring columns: {sorted(missing)}")
    if "data_penalty" in city_catalog_summary_df.columns:
        raise ValueError("Sparse-data penalty column should not be used in the revised pipeline")


def unique_in_order(values):
    seen = set()
    output = []
    for value in values:
        if value not in seen:
            output.append(value)
            seen.add(value)
    return output


def drive_minutes_between_cities(city_a: str, city_b: str) -> float:
    coord_a = CITY_COORDS[city_a]
    coord_b = CITY_COORDS[city_b]
    return geodesic(coord_a, coord_b).km * 1.25 / 72.0 * 60.0


def scenario_routes(config: TripConfig) -> dict[tuple[str, str], list[str]]:
    scenario = get_scenario_definition(config.get("trip", "scenario", "california_coast"))
    return scenario.gateway_routes or DEFAULT_SCENARIO_ROUTES


def scenario_route_options(config: TripConfig) -> list[dict]:
    starts = set(config.get("trip", "start_city_options", ["San Francisco", "Los Angeles"]))
    ends = set(config.get("trip", "end_city_options", ["Los Angeles", "San Francisco"]))
    options = []
    for option in get_route_options(config.get("trip", "scenario", "california_coast")):
        if option["gateway_start"] in starts and option["gateway_end"] in ends:
            options.append(option)
    if options:
        return options
    routes = scenario_routes(config)
    return [
        {
            "label": f"{start} to {end}",
            "gateway_start": start,
            "gateway_end": end,
            "sequence": sequence,
            "nature_regions": [],
        }
        for start in starts
        for end in ends
        if (start, end) in routes
        for sequence in [routes[(start, end)]]
    ]


def build_gateway_scenarios(config: TripConfig) -> list[tuple[str, str]]:
    return unique_in_order(
        [(option["gateway_start"], option["gateway_end"]) for option in scenario_route_options(config)]
    )


def _base_city_sequence_for_config(
    config: TripConfig, sequence: list[str], start_city: str, end_city: str
) -> list[str]:
    """Restrict day-allocation candidates to overnight bases, not nature regions."""
    scenario = get_scenario_definition(config.get("trip", "scenario", "california_coast"))
    allowed_bases = set(scenario.overnight_base_cities) | {start_city, end_city}
    base_sequence = [city for city in unique_in_order(sequence) if city in allowed_bases]
    for city in [start_city, end_city]:
        if city and city not in base_sequence:
            base_sequence.append(city)
    return base_sequence


def generate_day_allocations(config: TripConfig, cities: list[str]) -> list[dict[str, int]]:
    total_days = int(config.get("trip", "trip_days", 7))
    min_days = int(config.get("optimization", "min_days_per_base_city", 1))
    max_days = int(config.get("optimization", "max_days_per_base_city", 3))
    allocations = []
    for allocation_tuple in itertools.product(range(min_days, max_days + 1), repeat=len(cities)):
        if sum(allocation_tuple) == total_days:
            allocations.append(dict(zip(cities, allocation_tuple, strict=False)))
    return allocations


def generate_base_city_sets(config: TripConfig, sequence: list[str], start_city: str, end_city: str) -> list[list[str]]:
    """Generate possible overnight/base city subsets.

    The full sequence can include pass-through cities. Only the returned base
    cities receive allocated days and hotel/base decisions.
    """
    candidate_cities = unique_in_order(sequence)
    total_days = int(config.get("trip", "trip_days", 7))
    min_days = int(config.get("optimization", "min_days_per_base_city", 1))
    max_days = int(config.get("optimization", "max_days_per_base_city", 3))
    max_config_cities = int(config.get("optimization", "max_cities", 6))
    min_base_count = max(1, math.ceil(total_days / max(1, max_days)))
    max_base_count = min(len(candidate_cities), max_config_cities, max(1, total_days // max(1, min_days)))

    mandatory = [start_city]
    if end_city != start_city:
        mandatory.append(end_city)
    mandatory = [city for city in unique_in_order(mandatory) if city in candidate_cities]
    max_base_count = max(max_base_count, len(mandatory))

    optional = [city for city in candidate_cities if city not in set(mandatory)]
    base_sets = []
    for size in range(max(len(mandatory), min_base_count), max_base_count + 1):
        choose_count = size - len(mandatory)
        for combo in itertools.combinations(optional, choose_count):
            selected = set(mandatory) | set(combo)
            ordered = [city for city in candidate_cities if city in selected]
            if ordered not in base_sets:
                base_sets.append(ordered)
    return base_sets


def _interest_nature_weight(config: TripConfig) -> float:
    weights = config.get("interest", "weights", {})
    if isinstance(weights, dict):
        try:
            return float(weights.get("nature", 0.25) or 0.25)
        except Exception:
            return 0.25
    return 0.25


def _nature_region_bonus(region: NatureRegionDefinition, config: TripConfig) -> float:
    bonus = 0.0
    if region.national_park:
        bonus += float(config.get("nature", "national_park_bonus", 0.80))
    if region.state_park:
        bonus += float(config.get("nature", "state_park_bonus", 0.45))
    if region.protected_area:
        bonus += float(config.get("nature", "protected_area_bonus", 0.35))
    if region.scenic_region:
        bonus += float(config.get("nature", "scenic_region_bonus", 0.30))
    return bonus


def _minutes_from_sequence_to_region(sequence: list[str], region: NatureRegionDefinition) -> float:
    region_point = (region.latitude, region.longitude)
    minutes = []
    for city in sequence:
        if city not in CITY_COORDS:
            continue
        minutes.append(geodesic(CITY_COORDS[city], region_point).km * 1.25 / 72.0 * 60.0)
    return float(min(minutes)) if minutes else 9999.0


def _region_reason(
    *,
    scenario_id: str,
    region: NatureRegionDefinition,
    sequence: list[str],
    route_region_names: set[str],
    total_days: int,
    detour_minutes: float,
    nature_fit: float,
    selected_scores: list[float],
    config: TripConfig,
) -> str:
    max_detour = float(config.get("nature", "max_nature_detour_minutes", 240))
    max_daily_drive = float(config.get("nature", "max_single_day_drive_minutes_for_nature", 360))
    min_fit = float(config.get("nature", "min_nature_score_for_nature_heavy", 0.35))
    route_has_gateway = any(base in sequence for base in region.gateway_bases) or region.name in route_region_names
    if scenario_id == "california_coast" and region.name not in route_region_names and not route_has_gateway:
        return "coast_only_scenario_restriction"
    if detour_minutes > max_detour:
        return "excessive_inland_detour"
    if total_days < int(region.min_days):
        return "insufficient_trip_days"
    if detour_minutes > max_daily_drive:
        return "daily_drive_limit_exceeded"
    if float(region.weather_sensitivity + region.seasonality_risk) / 2.0 > 0.72:
        return "weather_risk_too_high"
    if selected_scores and nature_fit < min(selected_scores):
        return "lower_interest_fit_than_selected_regions"
    if nature_fit < min_fit:
        return "lower_interest_fit_than_selected_regions"
    return "budget_or_time_conflict"


def _annotate_nature_regions(
    plan: dict,
    config: TripConfig,
    *,
    sequence: list[str],
    route_nature_regions: list[str] | None = None,
) -> dict:
    scenario_id = str(config.get("trip", "scenario", "california_coast"))
    if not bool(config.get("nature", "enabled", True)):
        plan.update(
            {
                "scenario": scenario_id,
                "selected_nature_regions": [],
                "nature_region_sequence": [],
                "nature_days_by_region": {},
                "gateway_bases_by_region": {},
                "unselected_high_value_nature_regions": [],
                "why_not_selected": {},
                "nature_value_component": 0.0,
                "scenic_value_component": 0.0,
                "region_detour_penalty_component": 0.0,
                "weather_exposure_component": 0.0,
            }
        )
        return plan

    regions = get_nature_region_definitions(scenario_id)
    route_region_names = set(route_nature_regions or [])
    nature_weight = _interest_nature_weight(config)
    total_days = int(config.get("trip", "trip_days", 7))
    max_detour = float(config.get("nature", "max_nature_detour_minutes", 240))
    max_daily_drive = float(config.get("nature", "max_single_day_drive_minutes_for_nature", 360))
    min_nature_score = float(config.get("nature", "min_nature_score_for_nature_heavy", 0.35))
    nature_heavy = str(config.get("interest", "mode", config.get("trip", "interest_profile", ""))) == "nature_heavy"

    candidates = []
    for region in regions:
        detour_minutes = _minutes_from_sequence_to_region(sequence, region)
        gateway_hit = any(base in sequence or base in plan.get("overnight_bases", []) for base in region.gateway_bases)
        route_hit = region.name in route_region_names or region.region_id in route_region_names
        nature_fit = nature_weight * float(region.nature_score) + 0.20 * float(region.scenic_score)
        bonus = _nature_region_bonus(region, config)
        value = nature_fit + bonus
        penalty = 0.006 * detour_minutes
        weather_penalty = 0.25 * float(region.weather_sensitivity) + 0.18 * float(region.seasonality_risk)
        score = value - penalty - weather_penalty
        eligible = (
            (gateway_hit or route_hit)
            and total_days >= int(region.min_days)
            and detour_minutes <= max_detour
            and detour_minutes <= max_daily_drive
            and (not nature_heavy or nature_fit >= min_nature_score)
        )
        candidates.append(
            {
                "region": region,
                "eligible": eligible,
                "score": float(score),
                "nature_fit": float(nature_fit),
                "detour_minutes": float(detour_minutes),
                "gateway_hit": gateway_hit,
                "route_hit": route_hit,
            }
        )

    selected_candidates = [item for item in candidates if item["eligible"] and item["score"] > 0]
    selected_candidates.sort(key=lambda item: item["score"], reverse=True)
    if nature_heavy and selected_candidates:
        min_days = int(config.get("nature", "min_nature_days_for_nature_heavy", 2))
    else:
        min_days = 0

    selected = selected_candidates[: max(1, min(len(selected_candidates), total_days))]
    selected_scores = [item["nature_fit"] for item in selected]
    selected_regions = [item["region"] for item in selected]
    selected_names = [region.name for region in selected_regions]

    nature_days = {}
    remaining_required = min_days
    for item in selected:
        region = item["region"]
        days = min(int(region.recommended_days), total_days)
        if remaining_required > 0:
            days = max(days, min(int(region.recommended_days), remaining_required))
        nature_days[region.name] = max(1, days)
        remaining_required = max(0, remaining_required - nature_days[region.name])

    unselected = []
    why_not = {}
    for item in sorted(candidates, key=lambda row: row["score"], reverse=True):
        region = item["region"]
        if region.name in selected_names:
            continue
        reason = _region_reason(
            scenario_id=scenario_id,
            region=region,
            sequence=sequence,
            route_region_names=route_region_names,
            total_days=total_days,
            detour_minutes=item["detour_minutes"],
            nature_fit=item["nature_fit"],
            selected_scores=selected_scores,
            config=config,
        )
        if item["score"] > 0 or region.national_park:
            unselected.append(region.name)
            why_not[region.name] = reason

    nature_value = sum(item["nature_fit"] + _nature_region_bonus(item["region"], config) for item in selected)
    scenic_value = sum(0.25 * float(item["region"].scenic_score) for item in selected)
    detour_penalty = sum(0.006 * float(item["detour_minutes"]) for item in selected)
    weather_exposure = sum(
        0.25 * float(item["region"].weather_sensitivity) + 0.18 * float(item["region"].seasonality_risk)
        for item in selected
    )

    plan.update(
        {
            "scenario": scenario_id,
            "selected_nature_regions": selected_names,
            "nature_region_sequence": selected_names,
            "nature_days_by_region": nature_days,
            "gateway_bases_by_region": {region.name: list(region.gateway_bases) for region in selected_regions},
            "unselected_high_value_nature_regions": unselected,
            "why_not_selected": why_not,
            "nature_value_component": float(nature_value),
            "scenic_value_component": float(scenic_value),
            "region_detour_penalty_component": float(detour_penalty),
            "weather_exposure_component": float(weather_exposure),
        }
    )
    plan["objective"] = (
        float(plan.get("objective", 0.0))
        + float(nature_value)
        + float(scenic_value)
        - float(detour_penalty)
        - float(weather_exposure)
    )
    return plan


def candidate_plans(config: TripConfig, city_catalog_summary_df: pd.DataFrame) -> list[dict]:
    plans = []
    value_exponent = float(config.get("optimization", "city_day_value_exponent", 0.82))
    pass_through_weight = float(config.get("optimization", "pass_through_city_bonus_weight", 0.18))
    base_switch_penalty = float(config.get("optimization", "base_switch_penalty", 0.07))
    must_go_weight = must_go_city_weight(config)
    for route_option in scenario_route_options(config):
        start_city = route_option["gateway_start"]
        end_city = route_option["gateway_end"]
        sequence = route_option["sequence"]
        base_sequence = _base_city_sequence_for_config(config, sequence, start_city, end_city)
        sequence_cities = unique_in_order(sequence)
        leg_minutes = [drive_minutes_between_cities(sequence[i], sequence[i + 1]) for i in range(len(sequence) - 1)]
        total_drive_minutes = float(sum(leg_minutes))
        for base_cities in generate_base_city_sets(config, base_sequence, start_city, end_city):
            pass_through_cities = [city for city in sequence_cities if city not in set(base_cities)]
            for allocation in generate_day_allocations(config, base_cities):
                city_value = sum(
                    city_score(city, city_catalog_summary_df) * (float(days) ** value_exponent)
                    for city, days in allocation.items()
                )
                must_go_value = sum(
                    must_go_weight * city_must_go_score(city, city_catalog_summary_df) * (float(days) ** value_exponent)
                    for city, days in allocation.items()
                )
                pass_through_value = pass_through_weight * sum(
                    city_score(city, city_catalog_summary_df)
                    + 0.50 * must_go_weight * city_must_go_score(city, city_catalog_summary_df)
                    for city in pass_through_cities
                )
                switch_penalty = base_switch_penalty * max(0, len(base_cities) - 2)
                uncertainty_bonus = 0.03 * sum(data_uncertainty(city, city_catalog_summary_df) for city in allocation)
                drive_penalty = 0.018 * total_drive_minutes / 60.0
                endpoint_penalty = 0.10 if start_city == end_city else 0.0
                objective = (
                    city_value
                    + must_go_value
                    + pass_through_value
                    + uncertainty_bonus
                    - drive_penalty
                    - endpoint_penalty
                    - switch_penalty
                )
                plan = {
                    "gateway_start": start_city,
                    "gateway_end": end_city,
                    "route_option_label": route_option.get("label", f"{start_city} to {end_city}"),
                    "city_sequence": sequence,
                    "days_by_city": allocation,
                    "intercity_drive_minutes": total_drive_minutes,
                    "objective": float(objective),
                    "city_value_component": float(city_value),
                    "must_go_city_component": float(must_go_value),
                    "pass_through_value_component": float(pass_through_value),
                    "data_uncertainty_exploration_bonus": float(uncertainty_bonus),
                    "base_switch_penalty": float(switch_penalty),
                    "overnight_bases": list(allocation.keys()),
                    "pass_through_cities": pass_through_cities,
                    "pass_through_pois": pass_through_cities,
                    "problem_layer": "hierarchical_multicity_day_allocation",
                    "optimizer_role": "hierarchical_gurobi_master",
                }
                plan = _annotate_nature_regions(
                    plan,
                    config,
                    sequence=sequence,
                    route_nature_regions=route_option.get("nature_regions", []),
                )
                plans.append(plan)
    return plans


def _score_hierarchical_plan(
    config: TripConfig,
    city_catalog_summary_df: pd.DataFrame,
    *,
    start_city: str,
    end_city: str,
    sequence: list[str],
    allocation: dict[str, int],
    route_option_label: str | None = None,
    route_nature_regions: list[str] | None = None,
) -> dict:
    value_exponent = float(config.get("optimization", "city_day_value_exponent", 0.82))
    pass_through_weight = float(config.get("optimization", "pass_through_city_bonus_weight", 0.18))
    base_switch_penalty = float(config.get("optimization", "base_switch_penalty", 0.07))
    must_go_weight = must_go_city_weight(config)
    sequence_cities = unique_in_order(sequence)
    base_cities = [city for city in sequence_cities if int(allocation.get(city, 0)) > 0]
    pass_through_cities = [city for city in sequence_cities if city not in set(base_cities)]
    leg_minutes = [drive_minutes_between_cities(sequence[i], sequence[i + 1]) for i in range(len(sequence) - 1)]
    total_drive_minutes = float(sum(leg_minutes))
    city_value = sum(
        city_score(city, city_catalog_summary_df) * (float(days) ** value_exponent) for city, days in allocation.items()
    )
    must_go_value = sum(
        must_go_weight * city_must_go_score(city, city_catalog_summary_df) * (float(days) ** value_exponent)
        for city, days in allocation.items()
    )
    pass_through_value = pass_through_weight * sum(
        city_score(city, city_catalog_summary_df)
        + 0.50 * must_go_weight * city_must_go_score(city, city_catalog_summary_df)
        for city in pass_through_cities
    )
    switch_penalty = base_switch_penalty * max(0, len(base_cities) - 2)
    uncertainty_bonus = 0.03 * sum(data_uncertainty(city, city_catalog_summary_df) for city in allocation)
    drive_penalty = 0.018 * total_drive_minutes / 60.0
    endpoint_penalty = 0.10 if start_city == end_city else 0.0
    objective = (
        city_value
        + must_go_value
        + pass_through_value
        + uncertainty_bonus
        - drive_penalty
        - endpoint_penalty
        - switch_penalty
    )
    plan = {
        "gateway_start": start_city,
        "gateway_end": end_city,
        "route_option_label": route_option_label or f"{start_city} to {end_city}",
        "city_sequence": sequence,
        "days_by_city": allocation,
        "intercity_drive_minutes": total_drive_minutes,
        "objective": float(objective),
        "city_value_component": float(city_value),
        "must_go_city_component": float(must_go_value),
        "pass_through_value_component": float(pass_through_value),
        "data_uncertainty_exploration_bonus": float(uncertainty_bonus),
        "base_switch_penalty": float(switch_penalty),
        "overnight_bases": list(allocation.keys()),
        "pass_through_cities": pass_through_cities,
        "pass_through_pois": pass_through_cities,
        "problem_layer": "hierarchical_multicity_day_allocation",
    }
    return _annotate_nature_regions(plan, config, sequence=sequence, route_nature_regions=route_nature_regions)


def solve_hierarchical_trip_with_greedy(
    config: TripConfig, city_catalog_summary_df: pd.DataFrame
) -> tuple[dict, pd.DataFrame]:
    """Greedily choose gateways/base cities/day counts for a method baseline.

    This mirrors the hierarchical Gurobi pipeline shape without using the
    Gurobi master decision. It first chooses base cities from each gateway
    corridor by marginal city value, then allocates remaining days one at a
    time to the base city with the highest diminishing-return gain.
    """
    total_days = int(config.get("trip", "trip_days", 7))
    min_days = int(config.get("optimization", "min_days_per_base_city", 1))
    max_days = int(config.get("optimization", "max_days_per_base_city", 3))
    max_config_cities = int(config.get("optimization", "max_cities", 6))
    value_exponent = float(config.get("optimization", "city_day_value_exponent", 0.82))
    base_switch_penalty = float(config.get("optimization", "base_switch_penalty", 0.07))
    must_go_weight = must_go_city_weight(config)

    greedy_plans = []
    for route_option in scenario_route_options(config):
        start_city = route_option["gateway_start"]
        end_city = route_option["gateway_end"]
        sequence = route_option["sequence"]
        candidate_cities = _base_city_sequence_for_config(config, sequence, start_city, end_city)
        mandatory = [start_city]
        if end_city != start_city:
            mandatory.append(end_city)
        mandatory = [city for city in unique_in_order(mandatory) if city in candidate_cities]
        min_base_count = max(len(mandatory), math.ceil(total_days / max(1, max_days)))
        max_base_count = min(len(candidate_cities), max_config_cities, max(1, total_days // max(1, min_days)))
        if min_base_count > max_base_count:
            continue

        selected = list(mandatory)
        optional = [city for city in candidate_cities if city not in set(selected)]

        def city_pick_score(city: str) -> float:
            return (
                city_score(city, city_catalog_summary_df)
                + must_go_weight * city_must_go_score(city, city_catalog_summary_df)
                + 0.04 * city_social_score(city, city_catalog_summary_df)
                + 0.03 * data_uncertainty(city, city_catalog_summary_df)
                - base_switch_penalty
            )

        while len(selected) < min_base_count and optional:
            next_city = max(optional, key=city_pick_score)
            selected.append(next_city)
            optional.remove(next_city)

        while optional and len(selected) < max_base_count:
            next_city = max(optional, key=city_pick_score)
            if city_pick_score(next_city) <= 0.0:
                break
            selected.append(next_city)
            optional.remove(next_city)

        base_cities = [city for city in candidate_cities if city in set(selected)]
        if total_days < len(base_cities) * min_days or total_days > len(base_cities) * max_days:
            continue

        allocation = {city: min_days for city in base_cities}
        remaining_days = total_days - len(base_cities) * min_days
        while remaining_days > 0:
            feasible_cities = [city for city in base_cities if allocation[city] < max_days]
            if not feasible_cities:
                break
            next_city = max(
                feasible_cities,
                key=lambda city: (
                    (
                        city_score(city, city_catalog_summary_df)
                        + must_go_weight * city_must_go_score(city, city_catalog_summary_df)
                    )
                    * ((allocation[city] + 1) ** value_exponent - allocation[city] ** value_exponent)
                    + 0.01 * data_uncertainty(city, city_catalog_summary_df)
                ),
            )
            allocation[next_city] += 1
            remaining_days -= 1

        if sum(allocation.values()) != total_days:
            continue

        plan = _score_hierarchical_plan(
            config,
            city_catalog_summary_df,
            start_city=start_city,
            end_city=end_city,
            sequence=sequence,
            allocation=allocation,
            route_option_label=route_option.get("label", f"{start_city} to {end_city}"),
            route_nature_regions=route_option.get("nature_regions", []),
        )
        plan["optimizer_role"] = "hierarchical_greedy_master"
        plan["solver_status"] = "greedy_heuristic"
        greedy_plans.append(plan)

    if not greedy_plans:
        fallback_plans = candidate_plans(config, city_catalog_summary_df)
        if not fallback_plans:
            raise ValueError("No feasible hierarchical greedy candidate plans generated")
        fallback = max(fallback_plans, key=lambda item: float(item["objective"]))
        result = dict(fallback)
        result["optimizer_role"] = "hierarchical_greedy_master"
        result["solver_status"] = "greedy_fallback_to_enumeration"
        plan_df = pd.DataFrame(fallback_plans).sort_values("objective", ascending=False).reset_index(drop=True)
        plan_df["solver_status"] = result["solver_status"]
        return result, plan_df

    result = max(greedy_plans, key=lambda item: float(item["objective"]))
    plan_df = pd.DataFrame(greedy_plans).sort_values("objective", ascending=False).reset_index(drop=True)
    plan_df["solver_status"] = "greedy_heuristic"
    return dict(result), plan_df


def solve_hierarchical_trip_with_gurobi(
    config: TripConfig, city_catalog_summary_df: pd.DataFrame
) -> tuple[dict, pd.DataFrame]:
    """Use Gurobi to choose the best pre-generated hierarchical plan.

    The candidate generation is deterministic and small; Gurobi chooses the best
    gateway/allocation binary. If Gurobi is unavailable, the method falls back to
    the same objective by enumeration and marks the solver status.
    """
    plans = candidate_plans(config, city_catalog_summary_df)
    if not plans:
        raise ValueError("No feasible hierarchical candidate plans generated")

    try:
        import gurobipy as gp
        from gurobipy import GRB

        model = gp.Model("hierarchical_trip_plan")
        model.Params.OutputFlag = 0
        model.Params.TimeLimit = float(config.get("optimization", "gurobi_time_limit_seconds", 120))
        model.Params.MIPGap = float(config.get("optimization", "gurobi_mip_gap", 0.05))
        z = model.addVars(len(plans), vtype=GRB.BINARY, name="choose_plan")
        model.addConstr(gp.quicksum(z[i] for i in range(len(plans))) == 1, name="choose_exactly_one")
        model.setObjective(gp.quicksum(float(plans[i]["objective"]) * z[i] for i in range(len(plans))), GRB.MAXIMIZE)
        model.optimize()
        chosen_idx = max(range(len(plans)), key=lambda i: z[i].X if model.SolCount else float(plans[i]["objective"]))
        solver_status = f"gurobi_status_{int(model.Status)}"
    except Exception as exc:
        chosen_idx = max(range(len(plans)), key=lambda i: float(plans[i]["objective"]))
        solver_status = f"enumeration_fallback:{type(exc).__name__}"

    result = dict(plans[chosen_idx])
    result["solver_status"] = solver_status
    plan_df = pd.DataFrame(plans).sort_values("objective", ascending=False).reset_index(drop=True)
    plan_df["solver_status"] = solver_status
    return result, plan_df
