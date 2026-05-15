"""Small-step Gurobi route optimization oracle."""

from __future__ import annotations

import math
import time
from typing import Any

import pandas as pd
from geopy.distance import geodesic

from ._legacy import import_legacy_module
from .config import TripConfig
from .multi_objective_route import solve_multi_objective_route


def build_solver_overrides(config: TripConfig) -> dict[str, Any]:
    return {
        "backend": "gurobi",
        "fallback_to_heuristic": True,
        "time_limit_seconds": float(config.get("optimization", "gurobi_time_limit_seconds", 120)),
        "mip_gap_target": float(config.get("optimization", "gurobi_mip_gap", 0.05)),
        "max_candidate_attractions": int(config.get("optimization", "max_pois_per_day", 4)) * 3,
    }


def solve_candidate_route_pool(
    *,
    context: dict,
    profile_name: str,
    current_hotel_idx: int,
    remaining_time: float,
    remaining_budget: float,
    config: TripConfig,
    mode: str = "day",
) -> list[dict]:
    """Call the existing Gurobi candidate-pool generator for a selected bundle."""
    gurobi_itinerary_solver = import_legacy_module("gurobi_itinerary_solver")
    return gurobi_itinerary_solver.generate_receding_horizon_candidate_pool(
        context,
        profile_name=profile_name,
        current_hotel_idx=current_hotel_idx,
        current_attr_idx=None,
        remaining_time=remaining_time,
        remaining_budget=remaining_budget,
        visited_attractions=set(),
        visited_theme_counts={},
        candidate_pool_size=int(config.get("bandit", "top_k_gurobi_repairs", 3)),
        mode=mode,
        solver_overrides=build_solver_overrides(config),
    )


def _poi_value(row: pd.Series) -> float:
    return float(row.get("final_poi_value", row.get("source_score", 0.0)) or 0.0)


def _visit_minutes(row: pd.Series) -> float:
    category = str(row.get("category", "")).lower()
    if "museum" in category or "gallery" in category:
        return 105.0
    if "viewpoint" in category or "bridge" in category:
        return 45.0
    if "park" in category or "garden" in category:
        return 90.0
    if "theatre" in category:
        return 120.0
    return 75.0


def _travel_minutes(a: tuple[float, float], b: tuple[float, float]) -> float:
    return float(geodesic(a, b).km * 1.25 / 38.0 * 60.0)


def _route_stats(selected: list[int], pois: pd.DataFrame, depot: tuple[float, float]) -> tuple[float, float]:
    if not selected:
        return 0.0, 0.0
    points = (
        [depot] + [(float(pois.loc[idx, "latitude"]), float(pois.loc[idx, "longitude"])) for idx in selected] + [depot]
    )
    travel = sum(_travel_minutes(left, right) for left, right in zip(points[:-1], points[1:], strict=False))
    visit = sum(_visit_minutes(pois.loc[idx]) for idx in selected)
    return float(travel), float(visit)


def _greedy_route_repair(candidate_df: pd.DataFrame, config: TripConfig, depot: tuple[float, float]) -> dict[str, Any]:
    time_budget = float(config.get("time", "daily_time_budget_minutes", 720))
    max_pois = int(config.get("optimization", "max_pois_per_day", 4))
    pois = candidate_df.reset_index(drop=True).copy()
    pois["score"] = pois.apply(
        lambda row: _poi_value(row) - 0.006 * float(row.get("detour_minutes", 0.0) or 0.0), axis=1
    )
    selected: list[int] = []
    for idx in pois.sort_values("score", ascending=False).index.tolist():
        trial = selected + [int(idx)]
        travel, visit = _route_stats(trial, pois, depot)
        if len(trial) <= max_pois and travel + visit <= time_budget:
            selected = trial
    travel, visit = _route_stats(selected, pois, depot)
    value = sum(_poi_value(pois.loc[idx]) for idx in selected)
    return {
        "solver_status": "heuristic_repair",
        "selected_indices": selected,
        "selected_pois": pois.loc[selected, "name"].astype(str).tolist() if selected else [],
        "objective_value": float(value - 0.01 * travel),
        "total_value": float(value),
        "total_travel_minutes": float(travel),
        "total_visit_minutes": float(visit),
        "total_time_minutes": float(travel + visit),
        "feasible": bool(travel + visit <= time_budget),
        "optimality_gap": math.nan,
    }


def solve_enriched_route_with_gurobi(
    candidate_df: pd.DataFrame,
    config: TripConfig,
    *,
    depot: tuple[float, float] | None = None,
    candidate_size: int | None = None,
) -> dict[str, Any]:
    """Solve a small enriched-POI route problem with Gurobi, fallback heuristic.

    This is intentionally smaller than the original monolithic route model: the
    bandit chooses a bundle, and this oracle optimizes that selected bundle.
    """
    return solve_multi_objective_route(
        candidate_df,
        config,
        depot=depot,
        candidate_size=candidate_size,
        route_id="bandit_selected_epsilon_route",
    )
    start = time.perf_counter()
    if candidate_df.empty:
        result = _greedy_route_repair(candidate_df, config, depot or (0.0, 0.0))
        result["runtime_seconds"] = time.perf_counter() - start
        return result

    max_candidates = int(candidate_size or int(config.get("optimization", "max_pois_per_day", 4)) * 3)
    pois = (
        candidate_df.sort_values(["final_poi_value", "social_score"], ascending=False)
        .head(max_candidates)
        .reset_index(drop=True)
    )
    if depot is None:
        depot = (float(pois["latitude"].mean()), float(pois["longitude"].mean()))
    time_budget = float(config.get("time", "daily_time_budget_minutes", 720))
    max_pois = int(config.get("optimization", "max_pois_per_day", 4))
    n = len(pois)
    values = [_poi_value(pois.loc[i]) for i in range(n)]
    visit = [_visit_minutes(pois.loc[i]) for i in range(n)]
    points = [depot] + [(float(pois.loc[i, "latitude"]), float(pois.loc[i, "longitude"])) for i in range(n)]
    travel = {(i, j): _travel_minutes(points[i], points[j]) for i in range(n + 1) for j in range(n + 1) if i != j}

    try:
        import gurobipy as gp
        from gurobipy import GRB

        model = gp.Model("bandit_selected_enriched_route")
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
        model.addConstr(
            gp.quicksum(visit[i] * x[i] for i in range(n))
            + gp.quicksum(float(minutes) * y[i, j] for (i, j), minutes in travel.items())
            <= time_budget,
            name="time_budget",
        )
        model.setObjective(
            gp.quicksum(values[i] * x[i] for i in range(n))
            - 0.01 * gp.quicksum(float(minutes) * y[i, j] for (i, j), minutes in travel.items()),
            GRB.MAXIMIZE,
        )
        model.optimize()
        selected = [i for i in range(n) if x[i].X > 0.5] if model.SolCount else []
        route_arcs = [(i, j) for (i, j) in travel if y[i, j].X > 0.5] if model.SolCount else []
        ordered = []
        current = 0
        while route_arcs:
            next_edges = [(i, j) for i, j in route_arcs if i == current]
            if not next_edges:
                break
            _, nxt = next_edges[0]
            if nxt == 0:
                break
            ordered.append(nxt - 1)
            current = nxt
            route_arcs.remove(next_edges[0])
        selected = ordered or selected
        route_travel, route_visit = _route_stats(selected, pois, depot)
        total_value = sum(values[i] for i in selected)
        result = {
            "solver_status": f"gurobi_status_{int(model.Status)}",
            "selected_indices": selected,
            "selected_pois": pois.loc[selected, "name"].astype(str).tolist() if selected else [],
            "objective_value": float(model.ObjVal) if model.SolCount else math.nan,
            "total_value": float(total_value),
            "total_travel_minutes": float(route_travel),
            "total_visit_minutes": float(route_visit),
            "total_time_minutes": float(route_travel + route_visit),
            "feasible": bool(model.SolCount and route_travel + route_visit <= time_budget),
            "optimality_gap": float(model.MIPGap) if model.SolCount else math.nan,
        }
    except Exception as exc:
        result = _greedy_route_repair(pois, config, depot)
        result["solver_status"] = f"{result['solver_status']}_after_gurobi_error:{type(exc).__name__}"

    result["runtime_seconds"] = float(time.perf_counter() - start)
    result["candidate_size"] = int(len(pois))
    return result
