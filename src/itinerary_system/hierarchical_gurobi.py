"""Gurobi-backed hierarchical trip planning."""

from __future__ import annotations

import itertools
import math

import pandas as pd
from geopy.distance import geodesic

from .config import TripConfig


CITY_COORDS = {
    "San Diego": (32.7157, -117.1611),
    "Los Angeles": (34.0522, -118.2437),
    "Santa Barbara": (34.4208, -119.6982),
    "San Luis Obispo": (35.2828, -120.6596),
    "Monterey": (36.6002, -121.8947),
    "Santa Cruz": (36.9741, -122.0308),
    "San Francisco": (37.7749, -122.4194),
}

DEFAULT_SCENARIO_ROUTES = {
    ("Los Angeles", "Los Angeles"): ["Los Angeles", "Santa Barbara", "San Luis Obispo", "Monterey", "Los Angeles"],
    ("Los Angeles", "San Francisco"): ["Los Angeles", "Santa Barbara", "San Luis Obispo", "Monterey", "Santa Cruz", "San Francisco"],
    ("San Francisco", "Los Angeles"): ["San Francisco", "Santa Cruz", "Monterey", "San Luis Obispo", "Santa Barbara", "Los Angeles"],
    ("San Francisco", "San Francisco"): ["San Francisco", "Santa Cruz", "Monterey", "San Luis Obispo", "Santa Barbara", "San Francisco"],
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


def build_gateway_scenarios(config: TripConfig) -> list[tuple[str, str]]:
    starts = list(config.get("trip", "start_city_options", ["San Francisco", "Los Angeles"]))
    ends = list(config.get("trip", "end_city_options", ["Los Angeles", "San Francisco"]))
    return [(start, end) for start in starts for end in ends if (start, end) in DEFAULT_SCENARIO_ROUTES]


def generate_day_allocations(config: TripConfig, cities: list[str]) -> list[dict[str, int]]:
    total_days = int(config.get("trip", "trip_days", 7))
    min_days = int(config.get("optimization", "min_days_per_base_city", 1))
    max_days = int(config.get("optimization", "max_days_per_base_city", 3))
    allocations = []
    for allocation_tuple in itertools.product(range(min_days, max_days + 1), repeat=len(cities)):
        if sum(allocation_tuple) == total_days:
            allocations.append(dict(zip(cities, allocation_tuple)))
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


def candidate_plans(config: TripConfig, city_catalog_summary_df: pd.DataFrame) -> list[dict]:
    plans = []
    value_exponent = float(config.get("optimization", "city_day_value_exponent", 0.82))
    pass_through_weight = float(config.get("optimization", "pass_through_city_bonus_weight", 0.18))
    base_switch_penalty = float(config.get("optimization", "base_switch_penalty", 0.07))
    must_go_weight = must_go_city_weight(config)
    for start_city, end_city in build_gateway_scenarios(config):
        sequence = DEFAULT_SCENARIO_ROUTES[(start_city, end_city)]
        sequence_cities = unique_in_order(sequence)
        leg_minutes = [drive_minutes_between_cities(sequence[i], sequence[i + 1]) for i in range(len(sequence) - 1)]
        total_drive_minutes = float(sum(leg_minutes))
        for base_cities in generate_base_city_sets(config, sequence, start_city, end_city):
            pass_through_cities = [city for city in sequence_cities if city not in set(base_cities)]
            for allocation in generate_day_allocations(config, base_cities):
                city_value = sum(
                    city_score(city, city_catalog_summary_df) * (float(days) ** value_exponent)
                    for city, days in allocation.items()
                )
                must_go_value = sum(
                    must_go_weight
                    * city_must_go_score(city, city_catalog_summary_df)
                    * (float(days) ** value_exponent)
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
                objective = city_value + must_go_value + pass_through_value + uncertainty_bonus - drive_penalty - endpoint_penalty - switch_penalty
                plans.append(
                    {
                        "gateway_start": start_city,
                        "gateway_end": end_city,
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
                )
    return plans


def _score_hierarchical_plan(
    config: TripConfig,
    city_catalog_summary_df: pd.DataFrame,
    *,
    start_city: str,
    end_city: str,
    sequence: list[str],
    allocation: dict[str, int],
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
        city_score(city, city_catalog_summary_df) * (float(days) ** value_exponent)
        for city, days in allocation.items()
    )
    must_go_value = sum(
        must_go_weight
        * city_must_go_score(city, city_catalog_summary_df)
        * (float(days) ** value_exponent)
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
    objective = city_value + must_go_value + pass_through_value + uncertainty_bonus - drive_penalty - endpoint_penalty - switch_penalty
    return {
        "gateway_start": start_city,
        "gateway_end": end_city,
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


def solve_hierarchical_trip_with_greedy(config: TripConfig, city_catalog_summary_df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
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
    for start_city, end_city in build_gateway_scenarios(config):
        sequence = DEFAULT_SCENARIO_ROUTES[(start_city, end_city)]
        candidate_cities = unique_in_order(sequence)
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


def solve_hierarchical_trip_with_gurobi(config: TripConfig, city_catalog_summary_df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
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
