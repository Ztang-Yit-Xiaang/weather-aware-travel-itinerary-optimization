import copy
import os
import time

import numpy as np
import pandas as pd
from geopy.distance import geodesic

DEFAULT_TOURIST_PROFILES = {
    "relaxed": {
        "alpha": 3.0,
        "beta": 0.8,
        "min_attractions": 4,
        "target_attractions": 6,
        "max_attractions": 5,
        "min_daily_attractions_if_used": 1,
        "attraction_bonus": 0.03,
        "max_wait": 60,
    },
    "balanced": {
        "alpha": 2.0,
        "beta": 1.0,
        "min_attractions": 6,
        "target_attractions": 8,
        "max_attractions": 6,
        "min_daily_attractions_if_used": 2,
        "attraction_bonus": 0.12,
        "max_wait": 90,
    },
    "explorer": {
        "alpha": 1.0,
        "beta": 0.9,
        "min_attractions": 8,
        "target_attractions": 10,
        "max_attractions": 10,
        "min_daily_attractions_if_used": 2,
        "attraction_bonus": 0.22,
        "max_wait": 180,
    },
}

DEFAULT_MODEL_SETTINGS = {
    "daily_time_budget": 720,
    "total_budget": 1000,
    "gamma": 0.01,
    "hotel_utility_weight": 0.25,
    "switch_time_weight": 0.10,
    "switch_fixed_penalty": 0.08,
    "content_repeat_penalty_weight": 0.18,
    "content_heavy_repeat_penalty_weight": 0.12,
    "attraction_shortfall_penalty_weight": 2.5,
    "overnight_relocation_limit": 180,
    "num_days": 3,
}

DEFAULT_SOLVER_SETTINGS = {
    "time_limit_seconds": 900.0,
    "mip_gap_target": 0.02,
    "objective_scale": 1000,
    "max_travel": 90,
    "num_threads": None,
    "mip_focus": 1,
    "heuristics": 0.20,
    "seed": 42,
    "presolve": 2,
}


def _safe_optimizer_outputs_for_production(context):
    """
    Prevent old local 3-day optimizer outputs from leaking into
    the hierarchical California production pipeline.
    """
    if not isinstance(context, dict):
        return {}

    if context.get("USE_LOCAL_GUROBI_DEMO_IN_PRODUCTION_MAP") is False:
        return {}

    optimizer_outputs = context.get("optimizer_outputs", {})
    if not isinstance(optimizer_outputs, dict):
        return {}

    canonical_days = context.get("CANONICAL_TRIP_DAYS")
    old_k = optimizer_outputs.get("K")

    try:
        if canonical_days is not None and old_k is not None and int(old_k) != int(canonical_days):
            return {}
    except Exception:
        return {}

    return optimizer_outputs


def infer_content_theme(row):
    category_text = str(row.get("categories", "")).lower()
    type_text = str(row.get("type", "")).lower()
    name_text = str(row.get("name", "")).lower()
    theme_rules = [
        ("beach", ["beach"]),
        ("park", ["park", "preserve", "garden", "botanical"]),
        ("museum", ["museum", "historic", "history", "art gallery"]),
        ("zoo_aquarium", ["zoo", "aquarium"]),
        ("tour", ["tour", "wine", "whale", "cruise", "expedition", "sailing", "tasting"]),
        ("activity", ["adventure", "segway", "bike", "horseback", "surf", "hike"]),
        ("landmark", ["landmark", "monument", "observatory", "courthouse", "wharf", "mission"]),
    ]
    combined_text = " ".join([category_text, type_text, name_text])
    for theme_name, keywords in theme_rules:
        if any(keyword in combined_text for keyword in keywords):
            return theme_name
    return type_text if type_text else "general"


def safe_minmax(values):
    values = np.asarray(values, dtype=float)
    if np.allclose(values.max(), values.min()):
        return np.zeros_like(values)
    return (values - values.min()) / (values.max() - values.min())


def scale_to_int(values, multiplier):
    return np.rint(np.asarray(values, dtype=float) * multiplier).astype(int)


def _status_name(status, grb):
    mapping = {
        grb.OPTIMAL: "OPTIMAL",
        grb.SUBOPTIMAL: "SUBOPTIMAL",
        grb.TIME_LIMIT: "TIME_LIMIT",
        grb.SOLUTION_LIMIT: "SOLUTION_LIMIT",
        grb.INTERRUPTED: "INTERRUPTED",
        grb.INFEASIBLE: "INFEASIBLE",
        grb.INF_OR_UNBD: "INF_OR_UNBD",
        grb.UNBOUNDED: "UNBOUNDED",
        grb.NUMERIC: "NUMERIC",
    }
    return mapping.get(status, f"STATUS_{status}")


def _detect_license_path():
    candidates = [
        os.environ.get("GRB_LICENSE_FILE"),
        os.path.expanduser("~/Library/gurobi/gurobi.lic"),
        os.path.expanduser("~/gurobi/gurobi.lic"),
        os.path.expanduser("~/gurobi.lic"),
        r"C:\gurobi\gurobi.lic",
        r"C:\gurobi1300\gurobi.lic",
        r"C:\gurobi1200\gurobi.lic",
        r"C:\gurobi1100\gurobi.lic",
        "/Library/gurobi/gurobi.lic",
    ]
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    return None


def _has_wls_env_credentials():
    return all(os.environ.get(var_name) for var_name in ["GRB_WLSACCESSID", "GRB_WLSSECRET", "GRB_LICENSEID"])


def _merge_profile_overrides(base_profiles, profile_overrides=None):
    merged_profiles = copy.deepcopy(base_profiles)
    if not profile_overrides:
        return merged_profiles

    for profile_name, overrides in profile_overrides.items():
        if profile_name not in merged_profiles:
            raise KeyError(f"Unknown tourist profile override: {profile_name}")

        normalized_overrides = dict(overrides)
        if "lambda" in normalized_overrides and "attraction_bonus" not in normalized_overrides:
            normalized_overrides["attraction_bonus"] = normalized_overrides.pop("lambda")

        merged_profiles[profile_name].update(normalized_overrides)

    return merged_profiles


def _select_profiles(profile_dict, selected_profiles=None):
    if selected_profiles is None:
        return profile_dict

    if isinstance(selected_profiles, str):
        selected_profiles = [selected_profiles]

    missing_profiles = [profile_name for profile_name in selected_profiles if profile_name not in profile_dict]
    if missing_profiles:
        raise KeyError(f"Unknown selected profile(s): {missing_profiles}")

    return {profile_name: profile_dict[profile_name] for profile_name in selected_profiles}


def run_gurobi_itinerary_optimization(
    context,
    profile_overrides=None,
    global_overrides=None,
    solver_overrides=None,
    selected_profiles=None,
):
    from sklearn.preprocessing import MinMaxScaler

    required_vars = [
        "top100_with_waiting_time",
        "travel_time_matrix",
        "hotels_df",
        "hotel_to_attr_time",
        "hotel_to_hotel_time",
    ]
    required_attraction_columns = ["utility", "waiting_final", "visit_duration_sim", "cost", "name"]
    required_hotel_columns = ["nightly_price", "rating_score", "experience_score", "name"]

    missing_vars = [var for var in required_vars if var not in context]
    if missing_vars:
        print("ERROR: Missing required variables for optimization setup")
        print(f"  Missing: {', '.join(missing_vars)}")
        print()
        print("Please execute these cells in order BEFORE running this cell:")
        print("  1. Section 7.6: XGBoost waiting time aggregation (creates top100_with_waiting_time)")
        print("  2. Section 8.1-8.2: Real-time weather prediction (adds waiting_time_live, waiting_final)")
        print("  3. Section 8.3: Cost estimation (adds cost column)")
        print("  4. Section 4: Travel distance matrices (creates travel_time_matrix, hotel_to_attr_time)")
        print()
        raise RuntimeError(f"Missing prerequisite variables: {missing_vars}")

    top100_with_waiting_time = context["top100_with_waiting_time"].copy()
    hotels_df = context["hotels_df"].copy()
    travel_time_matrix = np.asarray(context["travel_time_matrix"], dtype=float)
    hotel_to_attr_time = np.asarray(context["hotel_to_attr_time"], dtype=float)
    hotel_to_hotel_time = np.asarray(context["hotel_to_hotel_time"], dtype=float)

    missing_attraction_cols = [
        col for col in required_attraction_columns if col not in top100_with_waiting_time.columns
    ]
    missing_hotel_cols = [col for col in required_hotel_columns if col not in hotels_df.columns]
    if missing_attraction_cols or missing_hotel_cols:
        print("ERROR: Missing required columns for optimization setup")
        if missing_attraction_cols:
            print(f"  Missing attraction columns: {missing_attraction_cols}")
        if missing_hotel_cols:
            print(f"  Missing hotel columns: {missing_hotel_cols}")
        raise RuntimeError("Run the enrichment and cost-estimation cells before optimization.")

    license_path = _detect_license_path()
    if license_path:
        os.environ["GRB_LICENSE_FILE"] = license_path

    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError as exc:
        raise ImportError(
            "gurobipy is not available in this notebook kernel. Run `!pip3.12 install gurobipy` "
            "in the setup cell, restart the kernel if needed, and try again."
        ) from exc

    print("All prerequisite data verified")
    if license_path:
        print(f"Using Gurobi license: {license_path}")
    elif _has_wls_env_credentials():
        print("Using Gurobi WLS credentials from GRB_WLSACCESSID / GRB_WLSSECRET / GRB_LICENSEID.")
    else:
        print("Gurobi license path not auto-detected; falling back to Gurobi defaults.")
    print()

    top100_with_waiting_time["content_theme"] = top100_with_waiting_time.apply(infer_content_theme, axis=1)

    utility = top100_with_waiting_time["utility"].values
    waiting_time = top100_with_waiting_time["waiting_final"].values
    visiting_time = top100_with_waiting_time["visit_duration_sim"].values
    cost = top100_with_waiting_time["cost"].values
    hotel_price = hotels_df["nightly_price"].astype(float).values
    hotel_rating = hotels_df["rating_score"].astype(float).values
    hotel_experience = hotels_df["experience_score"].astype(float).values

    scaler = MinMaxScaler()
    utility_norm = scaler.fit_transform(utility.reshape(-1, 1)).flatten()
    waiting_time_norm = scaler.fit_transform(waiting_time.reshape(-1, 1)).flatten()
    cost_norm = scaler.fit_transform(cost.reshape(-1, 1)).flatten()
    hotel_price_norm = safe_minmax(hotel_price)
    hotel_rating_norm = safe_minmax(hotel_rating)
    hotel_experience_norm = safe_minmax(hotel_experience)
    hotel_switch_norm = safe_minmax(hotel_to_hotel_time.flatten()).reshape(hotel_to_hotel_time.shape)
    hotel_value = 0.45 * hotel_rating_norm + 0.35 * hotel_experience_norm - 0.20 * hotel_price_norm

    travel_time_int = np.rint(travel_time_matrix).astype(int)
    hotel_to_attr_time_int = np.rint(hotel_to_attr_time).astype(int)
    hotel_to_hotel_time_int = np.rint(hotel_to_hotel_time).astype(int)
    visiting_time_int = np.rint(np.asarray(visiting_time, dtype=float)).astype(int)
    waiting_time_int = np.rint(np.asarray(waiting_time, dtype=float)).astype(int)
    cost_int = np.rint(np.asarray(cost, dtype=float)).astype(int)
    hotel_price_int = np.rint(np.asarray(hotel_price, dtype=float)).astype(int)

    p = 0.7
    cost_penalty = cost_norm**p

    print("Feature Normalization Summary:")
    print(f"  Utility (mean={utility_norm.mean():.3f}, std={utility_norm.std():.3f})")
    print(f"  Waiting Time (mean={waiting_time_norm.mean():.3f}, std={waiting_time_norm.std():.3f})")
    print(f"  Cost (mean={cost_norm.mean():.3f}, std={cost_norm.std():.3f})")
    print(
        "  Accommodation Cost Proxy "
        f"(mean={hotel_price.mean():.2f}, min={hotel_price.min():.2f}, max={hotel_price.max():.2f})"
    )
    print(f"  Hotel Rating (mean={hotel_rating.mean():.2f})")
    print()

    tourist_profiles = _merge_profile_overrides(DEFAULT_TOURIST_PROFILES, profile_overrides)
    tourist_profiles = _select_profiles(tourist_profiles, selected_profiles)

    model_cfg = DEFAULT_MODEL_SETTINGS.copy()
    if global_overrides:
        model_cfg.update(global_overrides)

    solver_cfg = DEFAULT_SOLVER_SETTINGS.copy()
    if solver_overrides:
        solver_cfg.update(solver_overrides)

    t_daily = int(round(float(model_cfg["daily_time_budget"])))
    total_budget = float(model_cfg["total_budget"])
    gamma = float(model_cfg["gamma"])
    hotel_utility_weight = float(model_cfg["hotel_utility_weight"])
    switch_time_weight = float(model_cfg["switch_time_weight"])
    switch_fixed_penalty = float(model_cfg["switch_fixed_penalty"])
    content_repeat_penalty_weight = float(model_cfg["content_repeat_penalty_weight"])
    content_heavy_repeat_penalty_weight = float(model_cfg["content_heavy_repeat_penalty_weight"])
    attraction_shortfall_penalty_weight = float(model_cfg["attraction_shortfall_penalty_weight"])
    overnight_relocation_limit = float(model_cfg["overnight_relocation_limit"])
    num_days = int(model_cfg["num_days"])

    time_limit_seconds = float(solver_cfg["time_limit_seconds"])
    mip_gap_target = float(solver_cfg["mip_gap_target"])
    objective_scale = int(round(float(solver_cfg["objective_scale"])))
    max_travel = float(solver_cfg["max_travel"])
    num_threads = solver_cfg["num_threads"]
    if num_threads is None:
        num_threads = max(1, min(8, os.cpu_count() or 1))
    else:
        num_threads = int(num_threads)
    mip_focus = int(solver_cfg["mip_focus"])
    heuristics = float(solver_cfg["heuristics"])
    random_seed = int(solver_cfg["seed"])
    presolve = int(solver_cfg["presolve"])

    travel_norm = safe_minmax(travel_time_matrix)
    routes = {}
    edges = {}
    route_details = {}

    wine_indices = [i for i, row in top100_with_waiting_time.iterrows() if "Wine" in row["name"]]
    whale_indices = [i for i, row in top100_with_waiting_time.iterrows() if "Whale" in row["name"]]
    content_theme_groups = {
        theme: theme_df.index.tolist() for theme, theme_df in top100_with_waiting_time.groupby("content_theme")
    }

    n = len(top100_with_waiting_time)
    num_hotels = len(hotels_df)
    base_arcs = [(i, j) for i in range(n) for j in range(n) if i != j and travel_time_int[i, j] <= max_travel]
    out_neighbors = {i: [] for i in range(n)}
    in_neighbors = {i: [] for i in range(n)}
    for i, j in base_arcs:
        out_neighbors[i].append(j)
        in_neighbors[j].append(i)

    print("Optimization Parameters:")
    print(f"  Number of attractions: {n}")
    print(f"  Number of hotels: {num_hotels}")
    print(f"  Number of days: {num_days}")
    print(f"  Time budget per day: {t_daily} minutes")
    print(f"  Total budget: ${total_budget}")
    print(f"  Time limit: {time_limit_seconds:.0f} seconds")
    print(f"  Target MIP gap: {mip_gap_target:.2%}")
    print(f"  Threads: {num_threads}")
    print(
        "  Content decay penalties: "
        f"repeat={content_repeat_penalty_weight}, heavy_repeat={content_heavy_repeat_penalty_weight}"
    )
    print(f"  Overnight relocation limit: {overnight_relocation_limit} minutes")
    print(f"  Candidate attraction arcs per day: {len(base_arcs)}")
    print()

    for profile, profile_cfg in tourist_profiles.items():
        print(f"Solving for '{profile}' profile with Gurobi...")

        model = gp.Model(f"tour_itinerary_{profile}")
        model.Params.OutputFlag = 0
        model.Params.TimeLimit = time_limit_seconds
        model.Params.MIPGap = mip_gap_target
        model.Params.Threads = num_threads
        model.Params.MIPFocus = mip_focus
        model.Params.Heuristics = heuristics
        model.Params.Seed = random_seed
        model.Params.Presolve = presolve

        alpha = profile_cfg["alpha"]
        beta = profile_cfg["beta"]
        attraction_bonus = profile_cfg["attraction_bonus"]
        min_total_attractions = profile_cfg["min_attractions"]
        target_total_attractions = profile_cfg["target_attractions"]
        min_daily_attractions_if_used = profile_cfg["min_daily_attractions_if_used"]
        max_k = profile_cfg["max_attractions"]
        max_trip_visits = num_days * max_k

        day_used = model.addVars(num_days, vtype=GRB.BINARY, name="day_used")
        z = model.addVars(num_hotels, num_days, vtype=GRB.BINARY, name="hotel")
        q_keys = [(h, g, d) for h in range(num_hotels) for g in range(num_hotels) for d in range(num_days - 1)]
        q = model.addVars(q_keys, vtype=GRB.BINARY, name="hotel_transition")
        x = model.addVars(n, num_days, vtype=GRB.BINARY, name="attraction")
        y_keys = [(i, j, d) for i, j in base_arcs for d in range(num_days)]
        y = model.addVars(y_keys, vtype=GRB.BINARY, name="travel")
        s_keys = [(h, i, d) for h in range(num_hotels) for i in range(n) for d in range(num_days)]
        e_keys = [(i, h, d) for i in range(n) for h in range(num_hotels) for d in range(num_days)]
        s = model.addVars(s_keys, vtype=GRB.BINARY, name="start")
        e = model.addVars(e_keys, vtype=GRB.BINARY, name="end")
        u = model.addVars(n, num_days, lb=0.0, ub=float(max_k), vtype=GRB.CONTINUOUS, name="u")
        attraction_shortfall = model.addVar(
            lb=0.0,
            ub=float(target_total_attractions),
            vtype=GRB.INTEGER,
            name=f"attraction_shortfall_{profile}",
        )
        content_repeat = model.addVars(
            list(content_theme_groups.keys()),
            lb=0.0,
            ub=float(max(0, max_trip_visits - 1)),
            vtype=GRB.INTEGER,
            name="content_repeat",
        )
        content_heavy_repeat = model.addVars(
            list(content_theme_groups.keys()),
            lb=0.0,
            ub=float(max(0, max_trip_visits - 2)),
            vtype=GRB.INTEGER,
            name="content_heavy_repeat",
        )

        utility_score = scale_to_int(utility_norm, objective_scale)
        wait_penalty = scale_to_int(alpha * waiting_time_norm, objective_scale)
        travel_penalty = scale_to_int(beta * travel_norm, objective_scale)
        attraction_cost_penalty = scale_to_int(gamma * cost_penalty, objective_scale)
        hotel_utility_score = scale_to_int(hotel_utility_weight * hotel_value, objective_scale)
        switch_time_penalty = scale_to_int(switch_time_weight * hotel_switch_norm, objective_scale)
        switch_fixed_penalty_scaled = int(round(objective_scale * switch_fixed_penalty))
        content_repeat_penalty = int(round(objective_scale * content_repeat_penalty_weight))
        content_heavy_repeat_penalty = int(round(objective_scale * content_heavy_repeat_penalty_weight))
        attraction_shortfall_penalty = int(round(objective_scale * attraction_shortfall_penalty_weight))
        attraction_bonus_score = int(round(objective_scale * attraction_bonus))

        objective = gp.quicksum(
            (utility_score[i] - wait_penalty[i] - attraction_cost_penalty[i] + attraction_bonus_score) * x[i, d]
            for i in range(n)
            for d in range(num_days)
        )
        objective -= gp.quicksum(
            float(travel_penalty[i, j]) * y[i, j, d] for i, j in base_arcs for d in range(num_days)
        )
        objective += gp.quicksum(hotel_utility_score[h] * z[h, d] for h in range(num_hotels) for d in range(num_days))
        objective -= gp.quicksum(
            float(switch_time_penalty[h, g]) * q[h, g, d]
            for h in range(num_hotels)
            for g in range(num_hotels)
            for d in range(num_days - 1)
        )
        objective -= gp.quicksum(
            switch_fixed_penalty_scaled * q[h, g, d]
            for h in range(num_hotels)
            for g in range(num_hotels)
            for d in range(num_days - 1)
            if h != g
        )
        objective -= gp.quicksum(
            content_repeat_penalty * content_repeat[theme_name]
            + content_heavy_repeat_penalty * content_heavy_repeat[theme_name]
            for theme_name in content_theme_groups
        )
        objective -= attraction_shortfall_penalty * attraction_shortfall
        model.setObjective(objective, GRB.MAXIMIZE)

        for d in range(num_days):
            model.addConstr(
                gp.quicksum((visiting_time_int[i] + waiting_time_int[i]) * x[i, d] for i in range(n))
                + gp.quicksum(travel_time_int[i, j] * y[i, j, d] for i, j in base_arcs)
                + gp.quicksum(hotel_to_attr_time_int[h, i] * s[h, i, d] for h in range(num_hotels) for i in range(n))
                + gp.quicksum(hotel_to_attr_time_int[h, i] * e[i, h, d] for i in range(n) for h in range(num_hotels))
                <= t_daily,
                name=f"daily_time_{d}",
            )

        model.addConstr(
            gp.quicksum(cost_int[i] * x[i, d] for i in range(n) for d in range(num_days))
            + gp.quicksum(hotel_price_int[h] * z[h, d] for h in range(num_hotels) for d in range(num_days))
            <= total_budget,
            name="total_budget",
        )

        for d in range(num_days):
            model.addConstr(gp.quicksum(z[h, d] for h in range(num_hotels)) == 1, name=f"one_hotel_{d}")

        for d in range(num_days - 1):
            for h in range(num_hotels):
                model.addConstr(
                    gp.quicksum(q[h, g, d] for g in range(num_hotels)) == z[h, d],
                    name=f"hotel_transition_out_{h}_{d}",
                )
            for g in range(num_hotels):
                model.addConstr(
                    gp.quicksum(q[h, g, d] for h in range(num_hotels)) == z[g, d + 1],
                    name=f"hotel_transition_in_{g}_{d}",
                )

        for d in range(num_days):
            model.addConstr(
                gp.quicksum(s[h, i, d] for h in range(num_hotels) for i in range(n)) == day_used[d],
                name=f"one_start_{d}",
            )
            model.addConstr(
                gp.quicksum(e[i, h, d] for i in range(n) for h in range(num_hotels)) == day_used[d],
                name=f"one_end_{d}",
            )

        for h in range(num_hotels):
            for d in range(num_days):
                model.addConstr(
                    gp.quicksum(s[h, i, d] for i in range(n)) <= z[h, d],
                    name=f"hotel_start_link_{h}_{d}",
                )
                model.addConstr(
                    gp.quicksum(e[i, h, d] for i in range(n)) <= z[h, d],
                    name=f"hotel_end_link_{h}_{d}",
                )

        for i in range(n):
            model.addConstr(gp.quicksum(x[i, d] for d in range(num_days)) <= 1, name=f"visit_once_{i}")

        total_selected_expr = gp.quicksum(x[i, d] for i in range(n) for d in range(num_days))
        for d in range(num_days):
            day_attractions = gp.quicksum(x[i, d] for i in range(n))
            model.addConstr(day_attractions <= max_k * day_used[d], name=f"max_attractions_day_{d}")
            model.addConstr(
                day_attractions >= min_daily_attractions_if_used * day_used[d],
                name=f"min_attractions_day_{d}",
            )
            model.addConstr(
                gp.quicksum(waiting_time_int[i] * x[i, d] for i in range(n))
                <= profile_cfg["max_wait"] * day_attractions,
                name=f"max_wait_day_{d}",
            )

        model.addConstr(gp.quicksum(day_used[d] for d in range(num_days)) >= 1, name="use_at_least_one_day")
        model.addConstr(total_selected_expr >= min_total_attractions, name="min_total_attractions")
        model.addConstr(
            attraction_shortfall >= target_total_attractions - total_selected_expr,
            name="target_shortfall",
        )

        if wine_indices:
            model.addConstr(
                gp.quicksum(x[i, d] for i in wine_indices for d in range(num_days)) <= 2,
                name="wine_limit",
            )
        if whale_indices:
            model.addConstr(
                gp.quicksum(x[i, d] for i in whale_indices for d in range(num_days)) <= 2,
                name="whale_limit",
            )

        for theme_name, theme_indices in content_theme_groups.items():
            theme_visits = gp.quicksum(x[i, d] for i in theme_indices for d in range(num_days))
            model.addConstr(content_repeat[theme_name] >= theme_visits - 1, name=f"repeat_penalty_{theme_name}")
            model.addConstr(
                content_heavy_repeat[theme_name] >= theme_visits - 2,
                name=f"heavy_repeat_penalty_{theme_name}",
            )

        for i in range(n):
            for d in range(num_days):
                outgoing_attr = gp.quicksum(y[i, j, d] for j in out_neighbors[i])
                incoming_attr = gp.quicksum(y[j, i, d] for j in in_neighbors[i])
                model.addConstr(
                    outgoing_attr + gp.quicksum(e[i, h, d] for h in range(num_hotels)) == x[i, d],
                    name=f"flow_out_{i}_{d}",
                )
                model.addConstr(
                    incoming_attr + gp.quicksum(s[h, i, d] for h in range(num_hotels)) == x[i, d],
                    name=f"flow_in_{i}_{d}",
                )

        for d in range(num_days):
            model.addConstr(
                gp.quicksum(y[i, j, d] for i, j in base_arcs) == gp.quicksum(x[i, d] for i in range(n)) - day_used[d],
                name=f"path_edge_count_{d}",
            )

        for d in range(num_days - 1):
            model.addConstr(
                gp.quicksum(
                    hotel_to_hotel_time_int[h, g] * q[h, g, d] for h in range(num_hotels) for g in range(num_hotels)
                )
                <= overnight_relocation_limit,
                name=f"overnight_limit_{d}",
            )

        for i in range(n):
            for d in range(num_days):
                model.addConstr(u[i, d] >= x[i, d], name=f"mtz_lb_{i}_{d}")
                model.addConstr(u[i, d] <= max_k * x[i, d], name=f"mtz_ub_{i}_{d}")
        for i, j in base_arcs:
            for d in range(num_days):
                model.addConstr(
                    u[i, d] - u[j, d] + max_k * y[i, j, d] <= max_k - 1,
                    name=f"mtz_arc_{i}_{j}_{d}",
                )

        model.update()
        print(f"  Model size: {model.NumVars} variables, {model.NumConstrs} constraints")

        solve_start = time.time()
        model.optimize()
        solve_seconds = time.time() - solve_start
        status_label = _status_name(model.Status, GRB)
        has_solution = model.SolCount > 0

        if not has_solution:
            print(
                f"  Solver finished with status [{status_label}] after {solve_seconds:.1f}s and no feasible incumbent"
            )
            print()
            continue

        day_used_by_day = {d: int(round(day_used[d].X)) for d in range(num_days)}
        selected_by_day = {d: [i for i in range(n) if x[i, d].X > 0.5] for d in range(num_days)}
        travel_edges_by_day = {d: [(i, j) for i, j in base_arcs if y[i, j, d].X > 0.5] for d in range(num_days)}
        start_edges_by_day = {
            d: [(h, i) for h in range(num_hotels) for i in range(n) if s[h, i, d].X > 0.5] for d in range(num_days)
        }
        end_edges_by_day = {
            d: [(i, h) for i in range(n) for h in range(num_hotels) if e[i, h, d].X > 0.5] for d in range(num_days)
        }
        hotels_by_day = {d: next((h for h in range(num_hotels) if z[h, d].X > 0.5), None) for d in range(num_days)}
        transitions_by_day = {
            d: [(h, g) for h in range(num_hotels) for g in range(num_hotels) if q[h, g, d].X > 0.5]
            for d in range(num_days - 1)
        }

        selected_indices = sorted({i for day_nodes in selected_by_day.values() for i in day_nodes})
        content_mix = (
            top100_with_waiting_time.iloc[selected_indices]["content_theme"].value_counts().to_dict()
            if selected_indices
            else {}
        )

        objective_value = float(model.ObjVal)
        best_bound = float(model.ObjBound)
        mip_gap = float(model.MIPGap)

        routes[profile] = selected_indices
        edges[profile] = [edge for day_edges in travel_edges_by_day.values() for edge in day_edges]
        route_details[profile] = {
            "day_used_by_day": day_used_by_day,
            "selected_by_day": selected_by_day,
            "travel_edges_by_day": travel_edges_by_day,
            "start_edges_by_day": start_edges_by_day,
            "end_edges_by_day": end_edges_by_day,
            "hotels_by_day": hotels_by_day,
            "transitions_by_day": transitions_by_day,
            "status": status_label,
            "solve_seconds": solve_seconds,
            "objective_value": objective_value,
            "best_bound": best_bound,
            "mip_gap": mip_gap,
            "solution_count": int(model.SolCount),
            "solver_name": "gurobi",
            "content_mix": content_mix,
            "applied_parameters": {
                "alpha": alpha,
                "beta": beta,
                "gamma": gamma,
                "lambda": attraction_bonus,
                "attraction_bonus": attraction_bonus,
                "min_attractions": min_total_attractions,
                "target_attractions": target_total_attractions,
                "max_attractions": max_k,
                "max_wait": float(profile_cfg["max_wait"]),
                "content_repeat_penalty_weight": content_repeat_penalty_weight,
                "content_heavy_repeat_penalty_weight": content_heavy_repeat_penalty_weight,
                "attraction_shortfall_penalty_weight": attraction_shortfall_penalty_weight,
                "switch_time_weight": switch_time_weight,
                "switch_fixed_penalty": switch_fixed_penalty,
            },
        }

        total_visit_wait = sum(
            visiting_time_int[i] + waiting_time_int[i] for day_nodes in selected_by_day.values() for i in day_nodes
        )
        total_travel = sum(travel_time_int[i, j] for day_edges in travel_edges_by_day.values() for i, j in day_edges)
        total_hotel_legs = sum(
            hotel_to_attr_time_int[h, i] for day_edges in start_edges_by_day.values() for h, i in day_edges
        ) + sum(hotel_to_attr_time_int[h, i] for day_edges in end_edges_by_day.values() for i, h in day_edges)
        total_switch_time = sum(
            hotel_to_hotel_time_int[h, g] for day_edges in transitions_by_day.values() for h, g in day_edges if h != g
        )
        total_time = total_visit_wait + total_travel + total_hotel_legs
        total_hotel_cost = sum(hotel_price_int[h] for h in hotels_by_day.values() if h is not None)
        total_cost = sum(cost_int[i] for i in selected_indices) + total_hotel_cost
        solution_type = "optimal" if model.Status == GRB.OPTIMAL else "feasible"

        headline = (
            f"  {solution_type.capitalize()} solution found [{status_label}] in {solve_seconds:.1f}s: "
            f"{len(selected_indices)} attractions, Cost: ${total_cost:.2f}, Time: {total_time:.0f} min"
        )
        if model.Status != GRB.OPTIMAL:
            headline += f", Gap~{mip_gap:.2%}"
        print(headline)

        for d in range(num_days):
            hotel_idx = hotels_by_day[d]
            hotel_name = hotels_df.iloc[hotel_idx]["name"] if hotel_idx is not None else "N/A"
            nightly_rate = hotel_price_int[hotel_idx] if hotel_idx is not None else 0
            if day_used_by_day[d] == 0:
                print(f"    Day {d + 1}: Hotel={hotel_name} (${nightly_rate:.0f}), No sightseeing route selected")
                continue

            day_names = top100_with_waiting_time.iloc[selected_by_day[d]]["name"].tolist()
            preview = ", ".join(day_names[:3])
            if len(day_names) > 3:
                preview += ", ..."
            print(
                f"    Day {d + 1}: Hotel={hotel_name} (${nightly_rate:.0f}), "
                f"Attractions={len(day_names)}, Route={preview}"
            )

            if d < num_days - 1 and transitions_by_day[d]:
                current_h, next_h = transitions_by_day[d][0]
                if current_h == next_h:
                    print("      Overnight decision: stay at the same accommodation")
                else:
                    next_name = hotels_df.iloc[next_h]["name"]
                    switch_minutes = hotel_to_hotel_time_int[current_h, next_h]
                    print(f"      Overnight decision: switch to {next_name} ({switch_minutes:.1f} min relocation)")

        print(f"    Total hotel cost: ${total_hotel_cost:.2f} | Hotel-switch time: {total_switch_time:.1f} min")
        print(f"    Objective value: {objective_value:.1f} | Best bound: {best_bound:.1f}")
        print(f"    Gurobi gap: {mip_gap:.2%} | Solutions kept: {int(model.SolCount)}")
        print(f"    Content mix: {content_mix}")
        print()

    solver_settings = {
        "solver_name": "gurobi",
        "time_limit_seconds": time_limit_seconds,
        "mip_gap_target": mip_gap_target,
        "threads": num_threads,
        "mip_focus": mip_focus,
        "heuristics": heuristics,
        "seed": random_seed,
        "presolve": presolve,
        "objective_scale": objective_scale,
        "max_travel": max_travel,
        "daily_time_budget": t_daily,
        "total_budget": total_budget,
        "overnight_relocation_limit": overnight_relocation_limit,
    }

    model_settings = {
        "daily_time_budget": t_daily,
        "total_budget": total_budget,
        "gamma": gamma,
        "hotel_utility_weight": hotel_utility_weight,
        "switch_time_weight": switch_time_weight,
        "switch_fixed_penalty": switch_fixed_penalty,
        "content_repeat_penalty_weight": content_repeat_penalty_weight,
        "content_heavy_repeat_penalty_weight": content_heavy_repeat_penalty_weight,
        "attraction_shortfall_penalty_weight": attraction_shortfall_penalty_weight,
        "overnight_relocation_limit": overnight_relocation_limit,
        "num_days": num_days,
    }

    return {
        "top100_with_waiting_time": top100_with_waiting_time,
        "routes": routes,
        "edges": edges,
        "route_details": route_details,
        "utility_norm": utility_norm,
        "waiting_time_norm": waiting_time_norm,
        "tourist_profiles": tourist_profiles,
        "solver_settings": solver_settings,
        "model_settings": model_settings,
        "K": num_days,
        "n": n,
        "num_hotels": num_hotels,
    }


def _compute_theme_novelty_bonus(theme_name, visited_theme_counts):
    visited_theme_counts = visited_theme_counts or {}
    return 1.0 / (1.0 + float(visited_theme_counts.get(theme_name, 0)))


def _reconstruct_ordered_path(start_node, edge_list, selected_nodes):
    if start_node is None:
        return list(selected_nodes)

    next_node = {i: j for i, j in edge_list}
    ordered = []
    current = start_node
    visited = set()

    while current is not None and current not in visited:
        ordered.append(current)
        visited.add(current)
        current = next_node.get(current)

    for node in selected_nodes:
        if node not in visited:
            ordered.append(node)

    return ordered


def estimate_local_route_visit_minutes(row):
    value = row.get("visit_duration_sim")
    try:
        value = float(value)
    except Exception:
        value = np.nan
    if np.isfinite(value) and value > 0:
        return float(value)

    category_text = str(row.get("category", row.get("categories", ""))).lower()
    name_text = str(row.get("name", "")).lower()
    combined_text = " ".join([category_text, name_text])
    if any(keyword in combined_text for keyword in ["museum", "gallery"]):
        return 105.0
    if any(keyword in combined_text for keyword in ["theatre", "theater", "observatory"]):
        return 100.0
    if any(keyword in combined_text for keyword in ["park", "garden", "beach", "wharf", "waterfront"]):
        return 75.0
    if any(keyword in combined_text for keyword in ["bridge", "viewpoint", "landmark", "courthouse", "campus"]):
        return 50.0
    if any(keyword in combined_text for keyword in ["tour", "wine", "cruise", "adventure", "surf"]):
        return 120.0
    return 75.0


def estimate_local_route_wait_minutes(row):
    for column in ["pred_wait_minutes", "waiting_final", "waiting_time_live", "waiting_time_avg"]:
        value = row.get(column)
        try:
            value = float(value)
        except Exception:
            value = np.nan
        if np.isfinite(value) and value >= 0:
            return float(value)
    return 0.0


def estimate_local_route_cost(row):
    for column in ["cost", "estimated_cost", "ticket_price", "price_proxy"]:
        value = row.get(column)
        try:
            value = float(value)
        except Exception:
            value = np.nan
        if np.isfinite(value) and value >= 0:
            return float(value)

    category_text = str(row.get("category", row.get("categories", ""))).lower()
    name_text = str(row.get("name", "")).lower()
    combined_text = " ".join([category_text, name_text])
    if any(keyword in combined_text for keyword in ["museum", "gallery", "theatre", "theater", "observatory"]):
        return 24.0
    if any(keyword in combined_text for keyword in ["tour", "wine", "cruise", "adventure", "whale", "surf"]):
        return 38.0
    if any(keyword in combined_text for keyword in ["zoo", "aquarium"]):
        return 32.0
    if any(keyword in combined_text for keyword in ["landmark", "bridge", "viewpoint", "courthouse", "campus"]):
        return 10.0
    if any(keyword in combined_text for keyword in ["park", "garden", "beach", "wharf", "waterfront"]):
        return 6.0
    return 12.0


def solve_legacy_local_day_route(
    candidate_df,
    *,
    route_start,
    route_end=None,
    remaining_time=None,
    remaining_budget=None,
    max_pois=None,
    profile_name="balanced",
    solver_overrides=None,
):
    """Solve one intra-day route using the legacy local Gurobi path formulation.

    This is a reusable day-level oracle for the hierarchical California pipeline.
    It intentionally accepts arbitrary candidate POIs and does not inherit the
    old monolithic defaults such as 3 fixed days or a fixed total trip budget.
    """

    solver_overrides = dict(solver_overrides or {})
    route_end = route_end or route_start
    candidate_df = candidate_df.copy().reset_index(drop=True)
    if candidate_df.empty:
        return {
            "solver_status": "FAILED",
            "selected_positions": [],
            "selected_pois": [],
            "objective_value": np.nan,
            "total_travel_minutes": 0.0,
            "total_visit_minutes": 0.0,
            "total_waiting_minutes": 0.0,
            "total_cost": 0.0,
            "total_time_minutes": 0.0,
            "feasible": False,
            "optimality_gap": np.nan,
            "runtime_seconds": 0.0,
        }

    candidate_df["_legacy_route_position"] = np.arange(len(candidate_df))
    candidate_df["latitude"] = candidate_df["latitude"].astype(float)
    candidate_df["longitude"] = candidate_df["longitude"].astype(float)
    candidate_df = candidate_df.dropna(subset=["latitude", "longitude"]).reset_index(drop=True)
    if candidate_df.empty:
        return {
            "solver_status": "FAILED",
            "selected_positions": [],
            "selected_pois": [],
            "objective_value": np.nan,
            "total_travel_minutes": 0.0,
            "total_visit_minutes": 0.0,
            "total_waiting_minutes": 0.0,
            "total_cost": 0.0,
            "total_time_minutes": 0.0,
            "feasible": False,
            "optimality_gap": np.nan,
            "runtime_seconds": 0.0,
        }

    start = time.perf_counter()
    utility_raw = (
        pd.Series(
            candidate_df.get("utility", candidate_df.get("final_poi_value", candidate_df.get("source_score", 0.0)))
        )
        .astype(float)
        .to_numpy()
    )
    waiting_raw = candidate_df.apply(estimate_local_route_wait_minutes, axis=1).astype(float).to_numpy()
    visit_raw = candidate_df.apply(estimate_local_route_visit_minutes, axis=1).astype(float).to_numpy()
    cost_raw = candidate_df.apply(estimate_local_route_cost, axis=1).astype(float).to_numpy()
    social_bonus = pd.to_numeric(candidate_df.get("social_score", 0.0), errors="coerce").fillna(0.0).to_numpy()
    must_go_flag = pd.to_numeric(candidate_df.get("social_must_go", False), errors="coerce").fillna(0.0).to_numpy()
    must_go_weight_raw = pd.to_numeric(candidate_df.get("must_go_weight", 0.0), errors="coerce").fillna(0.0).to_numpy()
    detour_penalty = pd.to_numeric(candidate_df.get("detour_minutes", 0.0), errors="coerce").fillna(0.0).to_numpy()
    theme_series = candidate_df.apply(infer_content_theme, axis=1)

    solver_overrides = solver_overrides or {}
    social_weight = float(solver_overrides.get("social_score_weight", 0.45))
    must_go_weight = float(solver_overrides.get("must_go_bonus_weight", 0.85))
    must_go_signal = np.maximum(must_go_weight_raw, must_go_flag) * np.maximum(social_bonus, must_go_flag)
    utility_signal = (
        utility_raw + social_weight * social_bonus + must_go_weight * must_go_signal - 0.005 * detour_penalty
    )
    utility_norm = safe_minmax(utility_signal)
    waiting_norm = safe_minmax(waiting_raw)
    cost_penalty = safe_minmax(cost_raw) ** 0.7

    n_local = len(candidate_df)
    points = candidate_df[["latitude", "longitude"]].astype(float).values.tolist()
    local_travel = np.asarray(
        [
            [
                geodesic(tuple(left), tuple(right)).km * 1.25 / 38.0 * 60.0 if i != j else 0.0
                for j, right in enumerate(points)
            ]
            for i, left in enumerate(points)
        ],
        dtype=float,
    )
    local_start = np.asarray(
        [geodesic(tuple(route_start), tuple(point)).km * 1.25 / 38.0 * 60.0 for point in points],
        dtype=float,
    )
    local_return = np.asarray(
        [geodesic(tuple(point), tuple(route_end)).km * 1.25 / 38.0 * 60.0 for point in points],
        dtype=float,
    )

    profile_cfg = dict(DEFAULT_TOURIST_PROFILES.get(profile_name, DEFAULT_TOURIST_PROFILES["balanced"]))
    solver_cfg = dict(DEFAULT_SOLVER_SETTINGS)
    model_cfg = dict(DEFAULT_MODEL_SETTINGS)
    if solver_overrides:
        profile_cfg.update({key: value for key, value in solver_overrides.items() if key in profile_cfg})
        solver_cfg.update({key: value for key, value in solver_overrides.items() if key in solver_cfg})
        model_cfg.update({key: value for key, value in solver_overrides.items() if key in model_cfg})

    if remaining_time is None:
        remaining_time = float(solver_overrides.get("daily_time_budget", model_cfg["daily_time_budget"]))
    if remaining_budget is None:
        remaining_budget = float(solver_overrides.get("remaining_budget", model_cfg["total_budget"]))
    max_pois = int(max_pois or profile_cfg.get("max_attractions", 6))
    max_pois = min(max_pois, int(profile_cfg.get("max_attractions", max_pois)))

    alpha = float(profile_cfg["alpha"])
    beta = float(profile_cfg["beta"])
    gamma = float(solver_overrides.get("gamma", model_cfg["gamma"]))
    attraction_bonus = float(profile_cfg["attraction_bonus"])
    max_wait = float(profile_cfg.get("max_wait", 90))

    def _heuristic_result(status_suffix="heuristic_repair"):
        order = sorted(
            range(n_local),
            key=lambda idx: (
                float(utility_signal[idx]) - 0.012 * float(local_start[idx]) - 0.007 * float(local_return[idx])
            ),
            reverse=True,
        )
        selected = []
        spent_time = 0.0
        spent_cost = 0.0
        current_point = tuple(route_start)
        remaining = set(order)
        while remaining and len(selected) < max_pois:
            feasible = []
            for idx in list(remaining):
                travel_out = geodesic(current_point, tuple(points[idx])).km * 1.25 / 38.0 * 60.0
                travel_back = geodesic(tuple(points[idx]), tuple(route_end)).km * 1.25 / 38.0 * 60.0
                total_time = spent_time + travel_out + visit_raw[idx] + waiting_raw[idx] + travel_back
                total_cost = spent_cost + cost_raw[idx]
                if total_time <= float(remaining_time) and total_cost <= float(remaining_budget):
                    feasible.append((idx, travel_out))
            if not feasible:
                break
            next_idx = max(
                feasible,
                key=lambda item: (
                    float(utility_signal[item[0]]) - 0.015 * float(item[1]) - 0.003 * float(local_return[item[0]])
                ),
            )[0]
            selected.append(next_idx)
            remaining.remove(next_idx)
            spent_time += (
                float(geodesic(current_point, tuple(points[next_idx])).km * 1.25 / 38.0 * 60.0)
                + float(visit_raw[next_idx])
                + float(waiting_raw[next_idx])
            )
            spent_cost += float(cost_raw[next_idx])
            current_point = tuple(points[next_idx])
        total_travel = 0.0
        current_point = tuple(route_start)
        for idx in selected:
            total_travel += float(geodesic(current_point, tuple(points[idx])).km * 1.25 / 38.0 * 60.0)
            current_point = tuple(points[idx])
        if selected:
            total_travel += float(geodesic(current_point, tuple(route_end)).km * 1.25 / 38.0 * 60.0)
        total_visit = float(sum(visit_raw[idx] for idx in selected))
        total_wait = float(sum(waiting_raw[idx] for idx in selected))
        total_cost = float(sum(cost_raw[idx] for idx in selected))
        objective = float(
            sum(utility_signal[idx] for idx in selected) - 0.01 * total_travel - gamma * total_cost * 0.02
        )
        return {
            "solver_status": status_suffix,
            "selected_positions": [int(candidate_df.iloc[idx]["_legacy_route_position"]) for idx in selected],
            "selected_pois": candidate_df.iloc[selected]["name"].astype(str).tolist() if selected else [],
            "objective_value": objective,
            "total_travel_minutes": float(total_travel),
            "total_visit_minutes": total_visit,
            "total_waiting_minutes": total_wait,
            "total_cost": total_cost,
            "total_time_minutes": float(total_travel + total_visit + total_wait),
            "feasible": bool(
                total_travel + total_visit + total_wait <= float(remaining_time)
                and total_cost <= float(remaining_budget)
            ),
            "optimality_gap": np.nan,
            "runtime_seconds": float(time.perf_counter() - start),
        }

    travel_scale = max(
        1.0,
        float(np.max(local_travel)) if local_travel.size else 1.0,
        float(np.max(local_start)) if local_start.size else 1.0,
    )
    local_arcs = [
        (i, j)
        for i in range(n_local)
        for j in range(n_local)
        if i != j and float(local_travel[i, j]) <= float(solver_cfg["max_travel"])
    ]
    out_neighbors = {i: [] for i in range(n_local)}
    in_neighbors = {i: [] for i in range(n_local)}
    for i, j in local_arcs:
        out_neighbors[i].append(j)
        in_neighbors[j].append(i)

    try:
        import gurobipy as gp
        from gurobipy import GRB

        model = gp.Model(f"legacy_local_day_route_{profile_name}")
        model.Params.OutputFlag = 0
        model.Params.TimeLimit = float(solver_cfg["time_limit_seconds"])
        model.Params.MIPGap = float(solver_cfg["mip_gap_target"])
        model.Params.Threads = int(
            solver_cfg["num_threads"] if solver_cfg["num_threads"] is not None else max(1, min(4, os.cpu_count() or 1))
        )
        model.Params.MIPFocus = int(solver_cfg["mip_focus"])
        model.Params.Heuristics = float(solver_cfg["heuristics"])
        model.Params.Seed = int(solver_cfg["seed"])
        model.Params.Presolve = int(solver_cfg["presolve"])

        route_used = model.addVar(vtype=GRB.BINARY, name="route_used")
        x = model.addVars(n_local, vtype=GRB.BINARY, name="visit")
        y = model.addVars(local_arcs, vtype=GRB.BINARY, name="travel")
        s = model.addVars(n_local, vtype=GRB.BINARY, name="start")
        e = model.addVars(n_local, vtype=GRB.BINARY, name="end")
        u = model.addVars(n_local, lb=0.0, ub=float(max_pois), vtype=GRB.CONTINUOUS, name="u")

        utility_score = scale_to_int(utility_norm, int(solver_cfg["objective_scale"]))
        wait_penalty = scale_to_int(alpha * waiting_norm, int(solver_cfg["objective_scale"]))
        cost_penalty_scaled = scale_to_int(gamma * cost_penalty, int(solver_cfg["objective_scale"]))
        travel_penalty = scale_to_int(beta * safe_minmax(local_travel), int(solver_cfg["objective_scale"]))
        start_penalty = scale_to_int(beta * safe_minmax(local_start), int(solver_cfg["objective_scale"]))
        return_penalty = scale_to_int(beta * safe_minmax(local_return), int(solver_cfg["objective_scale"]))
        attraction_bonus_score = int(round(float(solver_cfg["objective_scale"]) * attraction_bonus))

        objective = gp.quicksum(
            (utility_score[i] - wait_penalty[i] - cost_penalty_scaled[i] + attraction_bonus_score) * x[i]
            for i in range(n_local)
        )
        objective -= gp.quicksum(float(travel_penalty[i, j]) * y[i, j] for i, j in local_arcs)
        objective -= gp.quicksum(float(start_penalty[i]) * s[i] for i in range(n_local))
        objective -= gp.quicksum(float(return_penalty[i]) * e[i] for i in range(n_local))
        model.setObjective(objective, GRB.MAXIMIZE)

        total_selected = gp.quicksum(x[i] for i in range(n_local))
        model.addConstr(total_selected <= int(max_pois) * route_used, name="max_attractions")
        model.addConstr(total_selected >= route_used, name="route_active")
        model.addConstr(gp.quicksum(s[i] for i in range(n_local)) == route_used, name="single_start")
        model.addConstr(gp.quicksum(e[i] for i in range(n_local)) == route_used, name="single_end")
        model.addConstr(
            gp.quicksum((visit_raw[i] + waiting_raw[i]) * x[i] for i in range(n_local))
            + gp.quicksum(local_travel[i, j] * y[i, j] for i, j in local_arcs)
            + gp.quicksum(local_start[i] * s[i] for i in range(n_local))
            + gp.quicksum(local_return[i] * e[i] for i in range(n_local))
            <= float(remaining_time),
            name="daily_time_budget",
        )
        model.addConstr(
            gp.quicksum(cost_raw[i] * x[i] for i in range(n_local)) <= float(remaining_budget),
            name="remaining_budget",
        )
        model.addConstr(
            gp.quicksum(waiting_raw[i] * x[i] for i in range(n_local)) <= max_wait * total_selected,
            name="max_wait",
        )
        if bool(solver_overrides.get("must_go_is_mandatory", False)):
            mandatory_indices = [
                i
                for i in range(n_local)
                if bool(must_go_flag[i] > 0.5)
                and float(visit_raw[i] + waiting_raw[i] + local_start[i] + local_return[i]) <= float(remaining_time)
                and float(cost_raw[i]) <= float(remaining_budget)
            ]
            if mandatory_indices:
                model.addConstr(
                    gp.quicksum(x[i] for i in mandatory_indices) >= min(len(mandatory_indices), int(max_pois)),
                    name="mandatory_must_go_coverage",
                )

        for i in range(n_local):
            outgoing = gp.quicksum(y[i, j] for j in out_neighbors[i])
            incoming = gp.quicksum(y[j, i] for j in in_neighbors[i])
            model.addConstr(outgoing + e[i] == x[i], name=f"flow_out_{i}")
            model.addConstr(incoming + s[i] == x[i], name=f"flow_in_{i}")
            model.addConstr(u[i] >= x[i], name=f"mtz_lb_{i}")
            model.addConstr(u[i] <= float(max_pois) * x[i], name=f"mtz_ub_{i}")

        model.addConstr(
            gp.quicksum(y[i, j] for i, j in local_arcs) == total_selected - route_used,
            name="path_edge_count",
        )
        for i, j in local_arcs:
            model.addConstr(u[i] - u[j] + float(max_pois) * y[i, j] <= float(max_pois) - 1, name=f"mtz_arc_{i}_{j}")

        model.optimize()
        if model.SolCount == 0:
            return _heuristic_result("heuristic_repair_after_gurobi_error:NoFeasibleRoute")

        selected_local = [i for i in range(n_local) if x[i].X > 0.5]
        start_local = next((i for i in range(n_local) if s[i].X > 0.5), None)
        edges_local = [(i, j) for i, j in local_arcs if y[i, j].X > 0.5]
        ordered_local = _reconstruct_ordered_path(start_local, edges_local, selected_local)
        selected_positions = [int(candidate_df.iloc[idx]["_legacy_route_position"]) for idx in ordered_local]
        total_travel = 0.0
        current_point = tuple(route_start)
        for idx in ordered_local:
            total_travel += float(geodesic(current_point, tuple(points[idx])).km * 1.25 / 38.0 * 60.0)
            current_point = tuple(points[idx])
        if ordered_local:
            total_travel += float(geodesic(current_point, tuple(route_end)).km * 1.25 / 38.0 * 60.0)
        total_visit = float(sum(visit_raw[idx] for idx in ordered_local))
        total_wait = float(sum(waiting_raw[idx] for idx in ordered_local))
        total_cost = float(sum(cost_raw[idx] for idx in ordered_local))
        return {
            "solver_status": _status_name(model.Status, GRB),
            "selected_positions": selected_positions,
            "selected_pois": candidate_df.iloc[ordered_local]["name"].astype(str).tolist() if ordered_local else [],
            "objective_value": float(model.ObjVal) if model.SolCount else np.nan,
            "total_travel_minutes": float(total_travel),
            "total_visit_minutes": total_visit,
            "total_waiting_minutes": total_wait,
            "total_cost": total_cost,
            "total_time_minutes": float(total_travel + total_visit + total_wait),
            "feasible": bool(
                total_travel + total_visit + total_wait <= float(remaining_time)
                and total_cost <= float(remaining_budget)
            ),
            "optimality_gap": float(model.MIPGap) if model.SolCount else np.nan,
            "runtime_seconds": float(time.perf_counter() - start),
        }
    except Exception as exc:
        result = _heuristic_result(f"heuristic_repair_after_gurobi_error:{type(exc).__name__}")
        result["runtime_seconds"] = float(time.perf_counter() - start)
        return result


def _predict_candidate_metrics(
    sequence,
    top100_with_waiting_time,
    travel_time_matrix,
    hotel_to_attr_time,
    utility_norm,
    waiting_time_norm,
    cost_penalty,
    alpha,
    beta,
    gamma,
    attraction_bonus,
    current_hotel_idx,
    current_attr_idx,
    visited_theme_counts,
):
    if not sequence:
        return {
            "predicted_total_utility": 0.0,
            "predicted_total_wait": 0.0,
            "predicted_total_visit": 0.0,
            "predicted_total_travel_time": 0.0,
            "predicted_total_cost": 0.0,
            "predicted_total_reward": 0.0,
            "predicted_diversity_score": 0.0,
            "predicted_novelty_bonus": 0.0,
        }

    utility = top100_with_waiting_time["utility"].astype(float)
    waiting = top100_with_waiting_time["waiting_final"].astype(float)
    visiting = top100_with_waiting_time["visit_duration_sim"].astype(float)
    cost = top100_with_waiting_time["cost"].astype(float)
    themes = top100_with_waiting_time["content_theme"].astype(str)

    novelty_bonus = 0.0
    travel_time = 0.0
    theme_counter = dict(visited_theme_counts or {})
    previous_attr_idx = current_attr_idx

    for attr_idx in sequence:
        theme_name = themes.iloc[attr_idx]
        novelty_bonus += _compute_theme_novelty_bonus(theme_name, theme_counter)
        theme_counter[theme_name] = theme_counter.get(theme_name, 0) + 1

        if previous_attr_idx is None:
            travel_time += float(hotel_to_attr_time[current_hotel_idx, attr_idx])
        else:
            travel_time += float(travel_time_matrix[previous_attr_idx, attr_idx])
        previous_attr_idx = attr_idx

    travel_time += float(hotel_to_attr_time[current_hotel_idx, sequence[-1]])

    content_mix = top100_with_waiting_time.iloc[list(sequence)]["content_theme"].value_counts()
    if content_mix.empty or len(content_mix) <= 1:
        diversity_score = 0.0
    else:
        probs = content_mix.astype(float).values / float(content_mix.sum())
        entropy = -np.sum(probs * np.log(probs))
        diversity_score = float(entropy / np.log(len(probs)))

    predicted_reward = float(
        utility_norm[list(sequence)].sum()
        - alpha * waiting_time_norm[list(sequence)].sum()
        - gamma * cost_penalty[list(sequence)].sum()
        + attraction_bonus * novelty_bonus
    )

    combined_travel_scale = max(
        1.0,
        float(np.max(travel_time_matrix)) if np.size(travel_time_matrix) else 1.0,
        float(np.max(hotel_to_attr_time)) if np.size(hotel_to_attr_time) else 1.0,
    )
    predicted_reward -= beta * (travel_time / combined_travel_scale)

    return {
        "predicted_total_utility": float(utility.iloc[list(sequence)].sum()),
        "predicted_total_wait": float(waiting.iloc[list(sequence)].sum()),
        "predicted_total_visit": float(visiting.iloc[list(sequence)].sum()),
        "predicted_total_travel_time": float(travel_time),
        "predicted_total_cost": float(cost.iloc[list(sequence)].sum()),
        "predicted_total_reward": predicted_reward,
        "predicted_diversity_score": diversity_score,
        "predicted_novelty_bonus": float(novelty_bonus),
    }


def _generate_heuristic_candidate_pool(
    candidate_indices,
    top100_with_waiting_time,
    travel_time_matrix,
    hotel_to_attr_time,
    utility_norm,
    waiting_time_norm,
    cost_penalty,
    profile_cfg,
    current_hotel_idx,
    current_attr_idx,
    remaining_time,
    remaining_budget,
    visited_theme_counts,
    candidate_pool_size,
):
    if not candidate_indices:
        return []

    alpha = float(profile_cfg["alpha"])
    beta = float(profile_cfg["beta"])
    gamma = float(profile_cfg.get("gamma", DEFAULT_MODEL_SETTINGS["gamma"]))
    attraction_bonus = float(profile_cfg["attraction_bonus"])
    max_wait = float(profile_cfg["max_wait"])
    max_attractions = int(profile_cfg["max_attractions"])

    top100 = top100_with_waiting_time
    waiting = top100["waiting_final"].astype(float)
    visiting = top100["visit_duration_sim"].astype(float)
    cost = top100["cost"].astype(float)
    themes = top100["content_theme"].astype(str)
    travel_scale = max(
        1.0,
        float(np.max(travel_time_matrix)) if np.size(travel_time_matrix) else 1.0,
        float(np.max(hotel_to_attr_time)) if np.size(hotel_to_attr_time) else 1.0,
    )

    def immediate_score(from_attr_idx, attr_idx, local_theme_counts):
        travel_time = (
            float(hotel_to_attr_time[current_hotel_idx, attr_idx])
            if from_attr_idx is None
            else float(travel_time_matrix[from_attr_idx, attr_idx])
        )
        return (
            float(utility_norm[attr_idx])
            - alpha * float(waiting_time_norm[attr_idx])
            - beta * (travel_time / travel_scale)
            - gamma * float(cost_penalty[attr_idx])
            + attraction_bonus * _compute_theme_novelty_bonus(themes.iloc[attr_idx], local_theme_counts)
        )

    ranked_starts = sorted(
        candidate_indices,
        key=lambda idx: immediate_score(current_attr_idx, idx, visited_theme_counts),
        reverse=True,
    )

    candidates = []
    seen_sequences = set()
    for seed_attr_idx in ranked_starts[: max(candidate_pool_size * 2, candidate_pool_size)]:
        route = []
        local_theme_counts = dict(visited_theme_counts or {})
        local_remaining_time = float(remaining_time)
        local_remaining_budget = float(remaining_budget)
        local_wait_total = 0.0
        current_idx = current_attr_idx
        available = set(candidate_indices)

        while available and len(route) < max_attractions:
            feasible_next = []
            for attr_idx in list(available):
                outbound = (
                    float(hotel_to_attr_time[current_hotel_idx, attr_idx])
                    if current_idx is None
                    else float(travel_time_matrix[current_idx, attr_idx])
                )
                return_leg = float(hotel_to_attr_time[current_hotel_idx, attr_idx])
                required_time = outbound + float(visiting.iloc[attr_idx]) + float(waiting.iloc[attr_idx]) + return_leg
                if required_time > local_remaining_time:
                    continue
                if float(cost.iloc[attr_idx]) > local_remaining_budget:
                    continue
                projected_wait = local_wait_total + float(waiting.iloc[attr_idx])
                projected_count = len(route) + 1
                if projected_wait > max_wait * projected_count:
                    continue
                feasible_next.append(attr_idx)

            if not feasible_next:
                break

            if not route and seed_attr_idx in feasible_next:
                chosen_attr_idx = seed_attr_idx
            else:
                chosen_attr_idx = max(
                    feasible_next,
                    key=lambda idx: immediate_score(current_idx, idx, local_theme_counts),
                )

            outbound = (
                float(hotel_to_attr_time[current_hotel_idx, chosen_attr_idx])
                if current_idx is None
                else float(travel_time_matrix[current_idx, chosen_attr_idx])
            )
            local_remaining_time -= (
                outbound + float(visiting.iloc[chosen_attr_idx]) + float(waiting.iloc[chosen_attr_idx])
            )
            local_remaining_budget -= float(cost.iloc[chosen_attr_idx])
            local_wait_total += float(waiting.iloc[chosen_attr_idx])
            current_idx = chosen_attr_idx
            route.append(chosen_attr_idx)
            available.remove(chosen_attr_idx)
            theme_name = themes.iloc[chosen_attr_idx]
            local_theme_counts[theme_name] = local_theme_counts.get(theme_name, 0) + 1

        if not route:
            continue

        sequence_key = tuple(route)
        if sequence_key in seen_sequences:
            continue
        seen_sequences.add(sequence_key)

        metrics = _predict_candidate_metrics(
            route,
            top100,
            travel_time_matrix,
            hotel_to_attr_time,
            utility_norm,
            waiting_time_norm,
            cost_penalty,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            attraction_bonus=attraction_bonus,
            current_hotel_idx=current_hotel_idx,
            current_attr_idx=current_attr_idx,
            visited_theme_counts=visited_theme_counts,
        )
        candidates.append(
            {
                "selected_indices": route,
                "sequence": route,
                "travel_edges": list(zip(route[:-1], route[1:])),
                "first_attraction": route[0],
                "backend": "heuristic",
                **metrics,
            }
        )

    candidates.sort(key=lambda row: row["predicted_total_reward"], reverse=True)
    return candidates[:candidate_pool_size]


def generate_receding_horizon_candidate_pool(
    context,
    profile_name,
    current_hotel_idx,
    current_attr_idx=None,
    remaining_time=None,
    remaining_budget=None,
    visited_attractions=None,
    visited_theme_counts=None,
    candidate_pool_size=5,
    mode="step",
    solver_overrides=None,
):
    visited_attractions = set(visited_attractions or [])
    visited_theme_counts = dict(visited_theme_counts or {})
    solver_overrides = dict(solver_overrides or {})

    backend = str(solver_overrides.get("backend", "gurobi")).lower()
    fallback_to_heuristic = bool(solver_overrides.get("fallback_to_heuristic", True))

    top100_with_waiting_time = context["top100_with_waiting_time"].copy()
    if "content_theme" not in top100_with_waiting_time.columns:
        top100_with_waiting_time["content_theme"] = top100_with_waiting_time.apply(infer_content_theme, axis=1)

    travel_time_matrix = np.asarray(context["travel_time_matrix"], dtype=float)
    hotel_to_attr_time = np.asarray(context["hotel_to_attr_time"], dtype=float)
    hotels_df = context["hotels_df"].copy()

    tourist_profiles = context.get("TOURIST_PROFILES") or context.get("tourist_profiles") or DEFAULT_TOURIST_PROFILES
    if profile_name not in tourist_profiles:
        raise KeyError(f"Unknown profile '{profile_name}' for candidate pool generation")

    profile_cfg = dict(DEFAULT_TOURIST_PROFILES.get(profile_name, {}))
    profile_cfg.update(tourist_profiles[profile_name])

    optimizer_outputs = _safe_optimizer_outputs_for_production(context)
    model_settings = dict(DEFAULT_MODEL_SETTINGS)
    if isinstance(optimizer_outputs, dict):
        model_settings.update(optimizer_outputs.get("model_settings", {}))
    model_settings.update(context.get("MODEL_SETTINGS", {}))
    if "gamma" in solver_overrides:
        model_settings["gamma"] = solver_overrides["gamma"]
    profile_cfg["gamma"] = float(model_settings["gamma"])

    solver_settings = dict(DEFAULT_SOLVER_SETTINGS)
    if isinstance(optimizer_outputs, dict):
        solver_settings.update(optimizer_outputs.get("solver_settings", {}))
    solver_settings.update(context.get("SOLVER_SETTINGS", {}))
    solver_settings.update(
        {
            key: value
            for key, value in solver_overrides.items()
            if key
            in {
                "time_limit_seconds",
                "mip_gap_target",
                "objective_scale",
                "max_travel",
                "num_threads",
                "mip_focus",
                "heuristics",
                "seed",
                "presolve",
            }
        }
    )

    if remaining_time is None:
        remaining_time = float(
            context.get(
                "CANONICAL_DAILY_TIME_BUDGET",
                solver_settings.get("daily_time_budget", DEFAULT_MODEL_SETTINGS["daily_time_budget"]),
            )
        )
    if remaining_budget is None:
        remaining_budget = float(
            context.get(
                "CANONICAL_TOTAL_BUDGET",
                solver_settings.get("total_budget", DEFAULT_MODEL_SETTINGS["total_budget"]),
            )
        )

    utility = top100_with_waiting_time["utility"].astype(float).to_numpy()
    waiting_time = top100_with_waiting_time["waiting_final"].astype(float).to_numpy()
    cost = top100_with_waiting_time["cost"].astype(float).to_numpy()

    utility_norm = safe_minmax(utility)
    waiting_time_norm = safe_minmax(waiting_time)
    cost_norm = safe_minmax(cost)
    cost_penalty = cost_norm**0.7

    immediate_travel = np.asarray(
        [
            hotel_to_attr_time[current_hotel_idx, i]
            if current_attr_idx is None
            else travel_time_matrix[current_attr_idx, i]
            for i in range(len(top100_with_waiting_time))
        ],
        dtype=float,
    )
    return_travel = np.asarray(hotel_to_attr_time[current_hotel_idx], dtype=float)
    top100_themes = top100_with_waiting_time["content_theme"].astype(str)

    candidate_indices = []
    for attr_idx in range(len(top100_with_waiting_time)):
        if attr_idx in visited_attractions:
            continue
        travel_out = float(immediate_travel[attr_idx])
        visit_wait = float(top100_with_waiting_time.iloc[attr_idx]["visit_duration_sim"]) + float(
            top100_with_waiting_time.iloc[attr_idx]["waiting_final"]
        )
        travel_back = float(return_travel[attr_idx])
        if travel_out + visit_wait + travel_back > float(remaining_time):
            continue
        if float(cost[attr_idx]) > float(remaining_budget):
            continue
        candidate_indices.append(attr_idx)

    if not candidate_indices:
        return []

    travel_scale = max(
        1.0,
        float(np.max(travel_time_matrix)) if np.size(travel_time_matrix) else 1.0,
        float(np.max(hotel_to_attr_time)) if np.size(hotel_to_attr_time) else 1.0,
    )
    alpha = float(profile_cfg["alpha"])
    beta = float(profile_cfg["beta"])
    gamma = float(profile_cfg["gamma"])
    attraction_bonus = float(profile_cfg["attraction_bonus"])
    max_candidate_attractions = int(solver_overrides.get("max_candidate_attractions", max(candidate_pool_size * 3, 12)))

    scored_candidates = []
    for attr_idx in candidate_indices:
        novelty_bonus = _compute_theme_novelty_bonus(top100_themes.iloc[attr_idx], visited_theme_counts)
        immediate_score = (
            float(utility_norm[attr_idx])
            - alpha * float(waiting_time_norm[attr_idx])
            - beta * (float(immediate_travel[attr_idx]) / travel_scale)
            - gamma * float(cost_penalty[attr_idx])
            + attraction_bonus * novelty_bonus
        )
        scored_candidates.append((attr_idx, immediate_score))
    scored_candidates.sort(key=lambda item: item[1], reverse=True)
    filtered_candidate_indices = [idx for idx, _ in scored_candidates[:max_candidate_attractions]]

    if backend == "heuristic":
        return _generate_heuristic_candidate_pool(
            filtered_candidate_indices,
            top100_with_waiting_time,
            travel_time_matrix,
            hotel_to_attr_time,
            utility_norm,
            waiting_time_norm,
            cost_penalty,
            profile_cfg,
            current_hotel_idx,
            current_attr_idx,
            remaining_time,
            remaining_budget,
            visited_theme_counts,
            candidate_pool_size,
        )

    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError as exc:
        if not fallback_to_heuristic:
            raise ImportError("gurobipy is required for Gurobi candidate-pool generation") from exc
        return _generate_heuristic_candidate_pool(
            filtered_candidate_indices,
            top100_with_waiting_time,
            travel_time_matrix,
            hotel_to_attr_time,
            utility_norm,
            waiting_time_norm,
            cost_penalty,
            profile_cfg,
            current_hotel_idx,
            current_attr_idx,
            remaining_time,
            remaining_budget,
            visited_theme_counts,
            candidate_pool_size,
        )

    try:
        local_to_global = list(filtered_candidate_indices)
        global_to_local = {global_idx: local_idx for local_idx, global_idx in enumerate(local_to_global)}
        n_local = len(local_to_global)
        if n_local == 0:
            return []

        local_utility_norm = utility_norm[local_to_global]
        local_waiting_norm = waiting_time_norm[local_to_global]
        local_cost_penalty = cost_penalty[local_to_global]
        local_wait = waiting_time[local_to_global]
        local_visit = top100_with_waiting_time["visit_duration_sim"].astype(float).to_numpy()[local_to_global]
        local_cost = cost[local_to_global]
        local_themes = top100_themes.iloc[local_to_global].tolist()
        local_travel = travel_time_matrix[np.ix_(local_to_global, local_to_global)]
        local_start = immediate_travel[local_to_global]
        local_return = return_travel[local_to_global]
        novelty_bonus = np.asarray(
            [_compute_theme_novelty_bonus(theme_name, visited_theme_counts) for theme_name in local_themes],
            dtype=float,
        )

        local_arcs = [
            (i, j)
            for i in range(n_local)
            for j in range(n_local)
            if i != j and float(local_travel[i, j]) <= float(solver_settings["max_travel"])
        ]
        out_neighbors = {i: [] for i in range(n_local)}
        in_neighbors = {i: [] for i in range(n_local)}
        for i, j in local_arcs:
            out_neighbors[i].append(j)
            in_neighbors[j].append(i)

        model = gp.Model(f"adaptive_candidate_pool_{profile_name}_{mode}")
        model.Params.OutputFlag = 0
        model.Params.TimeLimit = float(solver_settings["time_limit_seconds"])
        model.Params.MIPGap = float(solver_settings["mip_gap_target"])
        model.Params.Threads = int(
            solver_settings["num_threads"]
            if solver_settings["num_threads"] is not None
            else max(1, min(4, os.cpu_count() or 1))
        )
        model.Params.MIPFocus = int(solver_settings["mip_focus"])
        model.Params.Heuristics = float(solver_settings["heuristics"])
        model.Params.Seed = int(solver_settings["seed"])
        model.Params.Presolve = int(solver_settings["presolve"])
        model.Params.PoolSearchMode = 2
        model.Params.PoolSolutions = int(max(candidate_pool_size * 2, candidate_pool_size))
        model.Params.PoolGap = 0.25

        route_used = model.addVar(vtype=GRB.BINARY, name="route_used")
        x = model.addVars(n_local, vtype=GRB.BINARY, name="visit")
        y = model.addVars(local_arcs, vtype=GRB.BINARY, name="travel")
        s = model.addVars(n_local, vtype=GRB.BINARY, name="start")
        e = model.addVars(n_local, vtype=GRB.BINARY, name="end")
        u = model.addVars(
            n_local,
            lb=0.0,
            ub=float(profile_cfg["max_attractions"]),
            vtype=GRB.CONTINUOUS,
            name="u",
        )

        utility_score = scale_to_int(local_utility_norm, int(solver_settings["objective_scale"]))
        wait_penalty = scale_to_int(alpha * local_waiting_norm, int(solver_settings["objective_scale"]))
        cost_penalty_scaled = scale_to_int(gamma * local_cost_penalty, int(solver_settings["objective_scale"]))
        novelty_score = scale_to_int(
            attraction_bonus * novelty_bonus,
            int(solver_settings["objective_scale"]),
        )
        local_travel_norm = safe_minmax(local_travel)
        start_travel_norm = safe_minmax(local_start)
        return_travel_norm = safe_minmax(local_return)
        travel_penalty = scale_to_int(beta * local_travel_norm, int(solver_settings["objective_scale"]))
        start_penalty = scale_to_int(beta * start_travel_norm, int(solver_settings["objective_scale"]))
        return_penalty = scale_to_int(beta * return_travel_norm, int(solver_settings["objective_scale"]))

        objective = gp.quicksum(
            (utility_score[i] - wait_penalty[i] - cost_penalty_scaled[i] + novelty_score[i]) * x[i]
            for i in range(n_local)
        )
        objective -= gp.quicksum(float(travel_penalty[i, j]) * y[i, j] for i, j in local_arcs)
        objective -= gp.quicksum(float(start_penalty[i]) * s[i] for i in range(n_local))
        objective -= gp.quicksum(float(return_penalty[i]) * e[i] for i in range(n_local))
        model.setObjective(objective, GRB.MAXIMIZE)

        total_selected = gp.quicksum(x[i] for i in range(n_local))
        model.addConstr(total_selected <= int(profile_cfg["max_attractions"]) * route_used, name="max_attractions")
        model.addConstr(total_selected >= route_used, name="route_active")
        model.addConstr(gp.quicksum(s[i] for i in range(n_local)) == route_used, name="single_start")
        model.addConstr(gp.quicksum(e[i] for i in range(n_local)) == route_used, name="single_end")
        model.addConstr(
            gp.quicksum((local_visit[i] + local_wait[i]) * x[i] for i in range(n_local))
            + gp.quicksum(local_travel[i, j] * y[i, j] for i, j in local_arcs)
            + gp.quicksum(local_start[i] * s[i] for i in range(n_local))
            + gp.quicksum(local_return[i] * e[i] for i in range(n_local))
            <= float(remaining_time),
            name="remaining_time",
        )
        model.addConstr(
            gp.quicksum(local_cost[i] * x[i] for i in range(n_local)) <= float(remaining_budget),
            name="remaining_budget",
        )
        model.addConstr(
            gp.quicksum(local_wait[i] * x[i] for i in range(n_local))
            <= float(profile_cfg["max_wait"]) * total_selected,
            name="max_wait",
        )

        for i in range(n_local):
            outgoing = gp.quicksum(y[i, j] for j in out_neighbors[i])
            incoming = gp.quicksum(y[j, i] for j in in_neighbors[i])
            model.addConstr(outgoing + e[i] == x[i], name=f"flow_out_{i}")
            model.addConstr(incoming + s[i] == x[i], name=f"flow_in_{i}")
            model.addConstr(s[i] <= x[i], name=f"start_link_{i}")
            model.addConstr(e[i] <= x[i], name=f"end_link_{i}")
            model.addConstr(u[i] >= x[i], name=f"mtz_lb_{i}")
            model.addConstr(u[i] <= float(profile_cfg["max_attractions"]) * x[i], name=f"mtz_ub_{i}")

        model.addConstr(
            gp.quicksum(y[i, j] for i, j in local_arcs) == total_selected - route_used,
            name="path_edge_count",
        )
        for i, j in local_arcs:
            model.addConstr(
                u[i] - u[j] + float(profile_cfg["max_attractions"]) * y[i, j]
                <= float(profile_cfg["max_attractions"]) - 1,
                name=f"mtz_arc_{i}_{j}",
            )

        model.optimize()
        if model.SolCount == 0:
            raise RuntimeError("No feasible candidate-pool route found")

        candidates = []
        seen_sequences = set()
        for solution_idx in range(min(int(model.SolCount), int(max(candidate_pool_size * 2, candidate_pool_size)))):
            model.Params.SolutionNumber = solution_idx
            selected_local = [i for i in range(n_local) if x[i].Xn > 0.5]
            if not selected_local:
                continue
            start_local = next((i for i in range(n_local) if s[i].Xn > 0.5), None)
            edges_local = [(i, j) for i, j in local_arcs if y[i, j].Xn > 0.5]
            ordered_local = _reconstruct_ordered_path(start_local, edges_local, selected_local)
            sequence = [local_to_global[local_idx] for local_idx in ordered_local]
            sequence_key = tuple(sequence)
            if sequence_key in seen_sequences:
                continue
            seen_sequences.add(sequence_key)

            metrics = _predict_candidate_metrics(
                sequence,
                top100_with_waiting_time,
                travel_time_matrix,
                hotel_to_attr_time,
                utility_norm,
                waiting_time_norm,
                cost_penalty,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                attraction_bonus=attraction_bonus,
                current_hotel_idx=current_hotel_idx,
                current_attr_idx=current_attr_idx,
                visited_theme_counts=visited_theme_counts,
            )
            candidates.append(
                {
                    "selected_indices": sequence,
                    "sequence": sequence,
                    "travel_edges": list(zip(sequence[:-1], sequence[1:])),
                    "first_attraction": sequence[0],
                    "backend": "gurobi",
                    "pool_objective_value": float(model.PoolObjVal),
                    **metrics,
                }
            )

        if not candidates:
            raise RuntimeError("Candidate-pool solve succeeded but produced no unique routes")

        candidates.sort(key=lambda row: row["predicted_total_reward"], reverse=True)
        return candidates[:candidate_pool_size]
    except Exception:
        if not fallback_to_heuristic:
            raise
        return _generate_heuristic_candidate_pool(
            filtered_candidate_indices,
            top100_with_waiting_time,
            travel_time_matrix,
            hotel_to_attr_time,
            utility_norm,
            waiting_time_norm,
            cost_penalty,
            profile_cfg,
            current_hotel_idx,
            current_attr_idx,
            remaining_time,
            remaining_budget,
            visited_theme_counts,
            candidate_pool_size,
        )
