"""Epsilon-constrained small route optimizer for bandit-selected bundles."""

from __future__ import annotations

import math
import time
from typing import Any

import pandas as pd
from geopy.distance import geodesic

from .config import TripConfig
from .diversity import category_labels, submodular_diversity
from .nature_catalog import interest_value_column, route_interest_metrics


def poi_value(row: pd.Series, config: TripConfig | None = None) -> float:
    value_column = interest_value_column(config) if config is not None else "final_poi_value"
    base_value = float(
        row.get(value_column, row.get("final_poi_value", row.get("utility_bayesian_ucb", row.get("source_score", 0.0))))
        or 0.0
    )
    if config is None:
        social_weight = 1.10
        must_go_weight = 0.85
    else:
        social_weight = float(config.get("social", "social_score_weight", 1.10))
        must_go_weight = float(config.get("social", "must_go_bonus_weight", 0.85))
    social_score = float(row.get("social_score", 0.0) or 0.0)
    row_must_go_weight = float(row.get("must_go_weight", 0.0) or 0.0)
    is_must_go = 1.0 if bool(row.get("social_must_go", False)) else 0.0
    must_go_signal = max(row_must_go_weight, is_must_go) * max(social_score, is_must_go)
    return float(base_value + 0.08 * social_weight * social_score + 0.22 * must_go_weight * must_go_signal)


def visit_minutes(row: pd.Series) -> float:
    category = str(row.get("category", "")).lower()
    if "museum" in category or "gallery" in category:
        return 105.0
    if "viewpoint" in category or "bridge" in category or "landmark" in category:
        return 50.0
    if "park" in category or "garden" in category:
        return 90.0
    if "theatre" in category:
        return 120.0
    if "campus" in category:
        return 75.0
    return 75.0


def travel_minutes(a: tuple[float, float], b: tuple[float, float]) -> float:
    return float(geodesic(a, b).km * 1.25 / 38.0 * 60.0)


def _poi_cost(row: pd.Series, config: TripConfig) -> float:
    for column in ["cost", "admission_cost", "estimated_cost"]:
        if column in row.index:
            try:
                value = float(row.get(column, 0.0) or 0.0)
                if value >= 0:
                    return value
            except Exception:
                pass
    max_pois = max(1, int(config.get("optimization", "max_pois_per_day", 4)))
    return float(config.get("budget", "attraction_cost_per_day", 28.0)) / max_pois


def _weather_risk(row: pd.Series) -> float:
    try:
        return max(0.0, min(1.0, float(row.get("weather_risk", 0.15) or 0.15)))
    except Exception:
        return 0.15


def route_stats(
    selected: list[int],
    pois: pd.DataFrame,
    start_depot: tuple[float, float],
    config: TripConfig,
    *,
    end_depot: tuple[float, float] | None = None,
) -> dict[str, float]:
    end_depot = end_depot or start_depot
    if not selected:
        direct_travel = travel_minutes(start_depot, end_depot) if start_depot != end_depot else 0.0
        return {
            "total_value": 0.0,
            "total_travel_minutes": float(direct_travel),
            "total_visit_minutes": 0.0,
            "total_time_minutes": float(direct_travel),
            "total_cost": 0.0,
            "total_detour_minutes": 0.0,
            "total_weather_risk": 0.0,
            "diversity_score": 0.0,
        }
    points = (
        [start_depot]
        + [(float(pois.loc[idx, "latitude"]), float(pois.loc[idx, "longitude"])) for idx in selected]
        + [end_depot]
    )
    travel = sum(travel_minutes(left, right) for left, right in zip(points[:-1], points[1:], strict=False))
    visit = sum(visit_minutes(pois.loc[idx]) for idx in selected)
    selected_frame = pois.loc[selected].copy()
    interest_metrics = route_interest_metrics(selected_frame, config)
    return {
        "total_value": float(sum(poi_value(pois.loc[idx], config) for idx in selected)),
        "total_travel_minutes": float(travel),
        "total_visit_minutes": float(visit),
        "total_time_minutes": float(travel + visit),
        "total_cost": float(sum(_poi_cost(pois.loc[idx], config) for idx in selected)),
        "total_detour_minutes": float(sum(float(pois.loc[idx].get("detour_minutes", 0.0) or 0.0) for idx in selected)),
        "total_weather_risk": float(sum(_weather_risk(pois.loc[idx]) for idx in selected)),
        "diversity_score": float(submodular_diversity(selected_frame)),
        **interest_metrics,
    }


def _constraints(config: TripConfig, max_pois: int) -> dict[str, float]:
    trip_days = max(1, int(config.get("trip", "trip_days", 7)))
    hard_multiplier = float(config.get("budget", "hard_budget_multiplier", 1.20))
    user_budget = float(config.get("budget", "user_budget", 2000.0))
    daily_hard_budget = user_budget * hard_multiplier / trip_days
    min_diversity = float(config.get("multi_objective", "min_diversity_score", 2.5))
    return {
        "time_budget": float(config.get("time", "daily_time_budget_minutes", 720)),
        "daily_hard_budget": float(config.get("multi_objective", "daily_hard_budget", daily_hard_budget)),
        "epsilon_detour": float(
            config.get("multi_objective", "epsilon_detour_minutes", config.get("time", "max_detour_minutes", 45))
        ),
        "epsilon_weather": float(config.get("multi_objective", "epsilon_weather_risk", 0.35)) * max(1, max_pois),
        "min_diversity": min(min_diversity, math.sqrt(max(1, max_pois)) * min(max_pois, 3)),
    }


def _greedy_epsilon_repair(
    candidate_df: pd.DataFrame,
    config: TripConfig,
    start_depot: tuple[float, float],
    *,
    end_depot: tuple[float, float] | None = None,
) -> dict[str, Any]:
    max_pois = int(config.get("optimization", "max_pois_per_day", 4))
    constraints = _constraints(config, max_pois)
    end_depot = end_depot or start_depot
    pois = candidate_df.reset_index(drop=True).copy()
    if pois.empty:
        stats = route_stats([], pois, start_depot, config, end_depot=end_depot)
        return {
            "solver_status": "heuristic_epsilon_repair_empty",
            "selected_indices": [],
            "selected_pois": [],
            "objective_value": 0.0,
            "feasible": True,
            "optimality_gap": math.nan,
            **stats,
        }
    pois["repair_score"] = pois.apply(
        lambda row: (
            poi_value(row, config)
            + float(config.get("multi_objective", "diversity_bonus_weight", 0.12))
            - 0.004 * float(row.get("detour_minutes", 0.0) or 0.0)
            - 0.08 * _weather_risk(row)
        ),
        axis=1,
    )
    selected: list[int] = []
    current = start_depot
    remaining = pois.sort_values("repair_score", ascending=False).index.tolist()
    while remaining and len(selected) < max_pois:
        ranked: list[tuple[float, int]] = []
        for idx in remaining:
            point = (float(pois.loc[idx, "latitude"]), float(pois.loc[idx, "longitude"]))
            travel_from_current = travel_minutes(current, point)
            travel_to_end = travel_minutes(point, end_depot)
            ranked.append((float(pois.loc[idx, "repair_score"]) - 0.006 * travel_from_current - 0.003 * travel_to_end, int(idx)))
        accepted_idx: int | None = None
        for _, idx in sorted(ranked, reverse=True):
            trial = selected + [int(idx)]
            stats = route_stats(trial, pois, start_depot, config, end_depot=end_depot)
            if stats["total_time_minutes"] > constraints["time_budget"]:
                continue
            if stats["total_cost"] > constraints["daily_hard_budget"]:
                continue
            if stats["total_detour_minutes"] > constraints["epsilon_detour"]:
                continue
            if stats["total_weather_risk"] > constraints["epsilon_weather"]:
                continue
            accepted_idx = int(idx)
            break
        if accepted_idx is None:
            break
        selected.append(accepted_idx)
        current = (float(pois.loc[accepted_idx, "latitude"]), float(pois.loc[accepted_idx, "longitude"]))
        remaining = [idx for idx in remaining if int(idx) != accepted_idx]
    stats = route_stats(selected, pois, start_depot, config, end_depot=end_depot)
    # The loop maintains feasibility, but keep the explicit objective construction close to the solver contract.
    objective = (
        stats["total_value"]
        + float(config.get("multi_objective", "diversity_bonus_weight", 0.12)) * stats["diversity_score"]
        - float(config.get("multi_objective", "secondary_travel_penalty", 0.005)) * stats["total_travel_minutes"]
    )
    return {
        "solver_status": "heuristic_epsilon_repair",
        "selected_indices": selected,
        "selected_pois": pois.loc[selected, "name"].astype(str).tolist() if selected else [],
        "objective_value": float(objective),
        "feasible": bool(
            stats["total_time_minutes"] <= constraints["time_budget"]
            and stats["total_cost"] <= constraints["daily_hard_budget"]
            and stats["total_detour_minutes"] <= constraints["epsilon_detour"]
            and stats["total_weather_risk"] <= constraints["epsilon_weather"]
            and stats["diversity_score"] + 1e-9 >= constraints["min_diversity"]
        ),
        "optimality_gap": math.nan,
        **stats,
    }


def solve_multi_objective_route(
    candidate_df: pd.DataFrame,
    config: TripConfig,
    *,
    depot: tuple[float, float] | None = None,
    start_depot: tuple[float, float] | None = None,
    end_depot: tuple[float, float] | None = None,
    candidate_size: int | None = None,
    route_id: str = "epsilon_gurobi_route",
) -> dict[str, Any]:
    """Solve a bandit-selected small route with explicit epsilon constraints."""
    start = time.perf_counter()
    if start_depot is None:
        start_depot = depot
    if end_depot is None:
        end_depot = depot or start_depot
    if candidate_df.empty:
        result = _greedy_epsilon_repair(
            candidate_df,
            config,
            start_depot or (0.0, 0.0),
            end_depot=end_depot or start_depot or (0.0, 0.0),
        )
        result["runtime_seconds"] = time.perf_counter() - start
        result["route_id"] = route_id
        return result

    max_candidates = int(candidate_size or int(config.get("optimization", "max_pois_per_day", 4)) * 3)
    value_column = interest_value_column(config)
    if value_column not in candidate_df.columns:
        value_column = "final_poi_value"
    pois = (
        candidate_df.sort_values([value_column, "social_score"], ascending=False)
        .head(max_candidates)
        .reset_index(drop=True)
    )
    if start_depot is None:
        start_depot = (float(pois["latitude"].mean()), float(pois["longitude"].mean()))
    if end_depot is None:
        end_depot = start_depot
    max_pois = min(int(config.get("optimization", "max_pois_per_day", 4)), len(pois))
    constraints = _constraints(config, max_pois)
    values = [poi_value(pois.loc[i], config) for i in range(len(pois))]
    visit = [visit_minutes(pois.loc[i]) for i in range(len(pois))]
    costs = [_poi_cost(pois.loc[i], config) for i in range(len(pois))]
    detours = [float(pois.loc[i].get("detour_minutes", 0.0) or 0.0) for i in range(len(pois))]
    weather = [_weather_risk(pois.loc[i]) for i in range(len(pois))]
    start_node = 0
    end_node = len(pois) + 1
    poi_nodes = list(range(1, len(pois) + 1))
    points = (
        [start_depot]
        + [(float(pois.loc[i, "latitude"]), float(pois.loc[i, "longitude"])) for i in range(len(pois))]
        + [end_depot]
    )
    travel = {
        (i, j): travel_minutes(points[i], points[j])
        for i in range(len(pois) + 2)
        for j in range(len(pois) + 2)
        if i != j and i != end_node and j != start_node
    }

    try:
        import gurobipy as gp
        from gurobipy import GRB

        n = len(pois)
        model = gp.Model("epsilon_constrained_enriched_route")
        model.Params.OutputFlag = 0
        model.Params.TimeLimit = float(config.get("optimization", "gurobi_time_limit_seconds", 120))
        model.Params.MIPGap = float(config.get("optimization", "gurobi_mip_gap", 0.05))
        x = model.addVars(n, vtype=GRB.BINARY, name="visit")
        y = model.addVars(travel.keys(), vtype=GRB.BINARY, name="arc")
        u = model.addVars(range(1, n + 1), lb=0.0, ub=float(n), vtype=GRB.CONTINUOUS, name="order")
        z = model.addVar(vtype=GRB.BINARY, name="route_active")

        model.addConstr(gp.quicksum(x[i] for i in range(n)) <= max_pois, name="max_pois")
        model.addConstr(gp.quicksum(x[i] for i in range(n)) <= max_pois * z, name="active_if_any_visit")
        model.addConstr(gp.quicksum(x[i] for i in range(n)) >= z, name="visit_if_active")
        model.addConstr(gp.quicksum(y[start_node, j] for j in poi_nodes + [end_node]) == z, name="leave_start")
        model.addConstr(gp.quicksum(y[i, end_node] for i in [start_node] + poi_nodes) == z, name="enter_end")
        for i in range(n):
            node = i + 1
            model.addConstr(
                gp.quicksum(y[node, j] for j in poi_nodes + [end_node] if j != node) == x[i],
                name=f"out_{i}",
            )
            model.addConstr(
                gp.quicksum(y[j, node] for j in [start_node] + poi_nodes if j != node) == x[i],
                name=f"in_{i}",
            )
            model.addConstr(u[node] <= n * x[i], name=f"order_active_{i}")
        for i in range(1, n + 1):
            for j in range(1, n + 1):
                if i != j:
                    model.addConstr(u[i] - u[j] + n * y[i, j] <= n - 1, name=f"mtz_{i}_{j}")

        travel_expr = gp.quicksum(float(minutes) * y[i, j] for (i, j), minutes in travel.items())
        visit_expr = gp.quicksum(float(visit[i]) * x[i] for i in range(n))
        cost_expr = gp.quicksum(float(costs[i]) * x[i] for i in range(n))
        detour_expr = gp.quicksum(float(detours[i]) * x[i] for i in range(n))
        weather_expr = gp.quicksum(float(weather[i]) * x[i] for i in range(n))
        model.addConstr(visit_expr + travel_expr <= constraints["time_budget"], name="epsilon_time_budget")
        model.addConstr(cost_expr <= constraints["daily_hard_budget"], name="epsilon_cost_budget")
        if bool(config.get("multi_objective", "use_epsilon_constraints", True)):
            model.addConstr(detour_expr <= constraints["epsilon_detour"], name="epsilon_detour")
            model.addConstr(weather_expr <= constraints["epsilon_weather"], name="epsilon_weather")

        labels = category_labels(pois).tolist()
        category_to_indices: dict[str, list[int]] = {}
        for idx, label in enumerate(labels):
            category_to_indices.setdefault(label, []).append(idx)
        z = {}
        diversity_terms = []
        for category, indices in category_to_indices.items():
            max_count = min(max_pois, len(indices))
            for k in range(1, max_count + 1):
                z[category, k] = model.addVar(vtype=GRB.BINARY, name=f"div_{category}_{k}")
                diversity_terms.append((math.sqrt(k) - math.sqrt(k - 1)) * z[category, k])
            model.addConstr(
                gp.quicksum(z[category, k] for k in range(1, max_count + 1)) == gp.quicksum(x[i] for i in indices),
                name=f"div_count_{category}",
            )
        diversity_expr = gp.quicksum(diversity_terms) if diversity_terms else 0.0
        if bool(config.get("multi_objective", "use_epsilon_constraints", True)):
            model.addConstr(diversity_expr >= constraints["min_diversity"], name="epsilon_diversity")

        lambda_div = float(config.get("multi_objective", "diversity_bonus_weight", 0.12))
        travel_penalty = float(config.get("multi_objective", "secondary_travel_penalty", 0.005))
        # Interest-aware objective:
        # maximize sum_i value_i x_i + lambda_div*Diversity(x)
        #          - lambda_travel*sum_ij travel_ij y_ij
        # where value_i = interest_adjusted_value_i when interest.enabled,
        # otherwise the legacy final_poi_value_i. Route balance
        # Balance(S,p)=1-||mean_i(a_i)-p||_1 is reported in route_stats.
        model.setObjective(
            gp.quicksum(float(values[i]) * x[i] for i in range(n))
            + lambda_div * diversity_expr
            - travel_penalty * travel_expr,
            GRB.MAXIMIZE,
        )
        model.optimize()
        if not model.SolCount:
            raise RuntimeError(f"no_gurobi_solution_status_{int(model.Status)}")

        selected = [i for i in range(n) if x[i].X > 0.5]
        route_arcs = [(i, j) for (i, j) in travel if y[i, j].X > 0.5]
        ordered = []
        current = start_node
        guard = 0
        while route_arcs and guard <= n + 1:
            guard += 1
            next_edges = [(i, j) for i, j in route_arcs if i == current]
            if not next_edges:
                break
            edge = next_edges[0]
            _, nxt = edge
            route_arcs.remove(edge)
            if nxt == end_node:
                break
            ordered.append(nxt - 1)
            current = nxt
        selected = ordered or selected
        stats = route_stats(selected, pois, start_depot, config, end_depot=end_depot)
        result = {
            "solver_status": f"gurobi_status_{int(model.Status)}",
            "selected_indices": selected,
            "selected_pois": pois.loc[selected, "name"].astype(str).tolist() if selected else [],
            "objective_value": float(model.ObjVal),
            "feasible": bool(
                stats["total_time_minutes"] <= constraints["time_budget"]
                and stats["total_cost"] <= constraints["daily_hard_budget"]
                and stats["total_detour_minutes"] <= constraints["epsilon_detour"]
                and stats["total_weather_risk"] <= constraints["epsilon_weather"]
                and stats["diversity_score"] + 1e-9 >= constraints["min_diversity"]
            ),
            "optimality_gap": float(model.MIPGap),
            **stats,
        }
    except Exception as exc:
        result = _greedy_epsilon_repair(pois, config, start_depot, end_depot=end_depot)
        result["solver_status"] = f"{result['solver_status']}_after_gurobi_error:{type(exc).__name__}"

    result["runtime_seconds"] = float(time.perf_counter() - start)
    result["candidate_size"] = int(len(pois))
    result["route_id"] = route_id
    result["epsilon_time_budget"] = constraints["time_budget"]
    result["epsilon_cost_budget"] = constraints["daily_hard_budget"]
    result["epsilon_detour_minutes"] = constraints["epsilon_detour"]
    result["epsilon_weather_risk"] = constraints["epsilon_weather"]
    result["epsilon_min_diversity"] = constraints["min_diversity"]
    return result
