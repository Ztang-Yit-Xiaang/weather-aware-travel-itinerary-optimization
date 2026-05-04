"""Epsilon-constrained small route optimizer for bandit-selected bundles."""

from __future__ import annotations

import math
import time
from typing import Any

import pandas as pd
from geopy.distance import geodesic

from .config import TripConfig
from .diversity import category_labels, submodular_diversity


def poi_value(row: pd.Series) -> float:
    return float(row.get("final_poi_value", row.get("utility_bayesian_ucb", row.get("source_score", 0.0))) or 0.0)


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


def route_stats(selected: list[int], pois: pd.DataFrame, depot: tuple[float, float], config: TripConfig) -> dict[str, float]:
    if not selected:
        return {
            "total_value": 0.0,
            "total_travel_minutes": 0.0,
            "total_visit_minutes": 0.0,
            "total_time_minutes": 0.0,
            "total_cost": 0.0,
            "total_detour_minutes": 0.0,
            "total_weather_risk": 0.0,
            "diversity_score": 0.0,
        }
    points = [depot] + [(float(pois.loc[idx, "latitude"]), float(pois.loc[idx, "longitude"])) for idx in selected] + [depot]
    travel = sum(travel_minutes(left, right) for left, right in zip(points[:-1], points[1:]))
    visit = sum(visit_minutes(pois.loc[idx]) for idx in selected)
    selected_frame = pois.loc[selected].copy()
    return {
        "total_value": float(sum(poi_value(pois.loc[idx]) for idx in selected)),
        "total_travel_minutes": float(travel),
        "total_visit_minutes": float(visit),
        "total_time_minutes": float(travel + visit),
        "total_cost": float(sum(_poi_cost(pois.loc[idx], config) for idx in selected)),
        "total_detour_minutes": float(sum(float(pois.loc[idx].get("detour_minutes", 0.0) or 0.0) for idx in selected)),
        "total_weather_risk": float(sum(_weather_risk(pois.loc[idx]) for idx in selected)),
        "diversity_score": float(submodular_diversity(selected_frame)),
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
        "epsilon_detour": float(config.get("multi_objective", "epsilon_detour_minutes", config.get("time", "max_detour_minutes", 45))),
        "epsilon_weather": float(config.get("multi_objective", "epsilon_weather_risk", 0.35)) * max(1, max_pois),
        "min_diversity": min(min_diversity, math.sqrt(max(1, max_pois)) * min(max_pois, 3)),
    }


def _greedy_epsilon_repair(candidate_df: pd.DataFrame, config: TripConfig, depot: tuple[float, float]) -> dict[str, Any]:
    max_pois = int(config.get("optimization", "max_pois_per_day", 4))
    constraints = _constraints(config, max_pois)
    pois = candidate_df.reset_index(drop=True).copy()
    if pois.empty:
        stats = route_stats([], pois, depot, config)
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
        lambda row: poi_value(row)
        + float(config.get("multi_objective", "diversity_bonus_weight", 0.12))
        - 0.004 * float(row.get("detour_minutes", 0.0) or 0.0)
        - 0.08 * _weather_risk(row),
        axis=1,
    )
    selected: list[int] = []
    for idx in pois.sort_values("repair_score", ascending=False).index.tolist():
        trial = selected + [int(idx)]
        stats = route_stats(trial, pois, depot, config)
        if len(trial) > max_pois:
            continue
        if stats["total_time_minutes"] > constraints["time_budget"]:
            continue
        if stats["total_cost"] > constraints["daily_hard_budget"]:
            continue
        if stats["total_detour_minutes"] > constraints["epsilon_detour"]:
            continue
        if stats["total_weather_risk"] > constraints["epsilon_weather"]:
            continue
        selected = trial
    stats = route_stats(selected, pois, depot, config)
    objective = (
        stats["total_value"]
        + float(config.get("multi_objective", "diversity_bonus_weight", 0.12)) * stats["diversity_score"]
        - float(config.get("multi_objective", "secondary_travel_penalty", 0.005)) * stats["total_travel_minutes"]
    )
    feasible = bool(
        stats["total_time_minutes"] <= constraints["time_budget"]
        and stats["total_cost"] <= constraints["daily_hard_budget"]
        and stats["total_detour_minutes"] <= constraints["epsilon_detour"]
        and stats["total_weather_risk"] <= constraints["epsilon_weather"]
        and stats["diversity_score"] + 1e-9 >= constraints["min_diversity"]
    )
    return {
        "solver_status": "heuristic_epsilon_repair",
        "selected_indices": selected,
        "selected_pois": pois.loc[selected, "name"].astype(str).tolist() if selected else [],
        "objective_value": float(objective),
        "feasible": feasible,
        "optimality_gap": math.nan,
        **stats,
    }


def solve_multi_objective_route(
    candidate_df: pd.DataFrame,
    config: TripConfig,
    *,
    depot: tuple[float, float] | None = None,
    candidate_size: int | None = None,
    route_id: str = "epsilon_gurobi_route",
) -> dict[str, Any]:
    """Solve a bandit-selected small route with explicit epsilon constraints."""
    start = time.perf_counter()
    if candidate_df.empty:
        result = _greedy_epsilon_repair(candidate_df, config, depot or (0.0, 0.0))
        result["runtime_seconds"] = time.perf_counter() - start
        result["route_id"] = route_id
        return result

    max_candidates = int(candidate_size or int(config.get("optimization", "max_pois_per_day", 4)) * 3)
    pois = candidate_df.sort_values(["final_poi_value", "social_score"], ascending=False).head(max_candidates).reset_index(drop=True)
    if depot is None:
        depot = (float(pois["latitude"].mean()), float(pois["longitude"].mean()))
    max_pois = min(int(config.get("optimization", "max_pois_per_day", 4)), len(pois))
    constraints = _constraints(config, max_pois)
    values = [poi_value(pois.loc[i]) for i in range(len(pois))]
    visit = [visit_minutes(pois.loc[i]) for i in range(len(pois))]
    costs = [_poi_cost(pois.loc[i], config) for i in range(len(pois))]
    detours = [float(pois.loc[i].get("detour_minutes", 0.0) or 0.0) for i in range(len(pois))]
    weather = [_weather_risk(pois.loc[i]) for i in range(len(pois))]
    points = [depot] + [(float(pois.loc[i, "latitude"]), float(pois.loc[i, "longitude"])) for i in range(len(pois))]
    travel = {(i, j): travel_minutes(points[i], points[j]) for i in range(len(pois) + 1) for j in range(len(pois) + 1) if i != j}

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

        model.addConstr(gp.quicksum(x[i] for i in range(n)) <= max_pois, name="max_pois")
        model.addConstr(gp.quicksum(y[0, j] for j in range(1, n + 1)) <= 1, name="leave_depot")
        model.addConstr(gp.quicksum(y[i, 0] for i in range(1, n + 1)) <= 1, name="return_depot")
        for i in range(n):
            node = i + 1
            model.addConstr(gp.quicksum(y[node, j] for j in range(n + 1) if j != node) == x[i], name=f"out_{i}")
            model.addConstr(gp.quicksum(y[j, node] for j in range(n + 1) if j != node) == x[i], name=f"in_{i}")
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
                gp.quicksum(z[category, k] for k in range(1, max_count + 1))
                == gp.quicksum(x[i] for i in indices),
                name=f"div_count_{category}",
            )
        diversity_expr = gp.quicksum(diversity_terms) if diversity_terms else 0.0
        if bool(config.get("multi_objective", "use_epsilon_constraints", True)):
            model.addConstr(diversity_expr >= constraints["min_diversity"], name="epsilon_diversity")

        lambda_div = float(config.get("multi_objective", "diversity_bonus_weight", 0.12))
        travel_penalty = float(config.get("multi_objective", "secondary_travel_penalty", 0.005))
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
        current = 0
        guard = 0
        while route_arcs and guard <= n + 1:
            guard += 1
            next_edges = [(i, j) for i, j in route_arcs if i == current]
            if not next_edges:
                break
            edge = next_edges[0]
            _, nxt = edge
            route_arcs.remove(edge)
            if nxt == 0:
                break
            ordered.append(nxt - 1)
            current = nxt
        selected = ordered or selected
        stats = route_stats(selected, pois, depot, config)
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
        result = _greedy_epsilon_repair(pois, config, depot)
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
