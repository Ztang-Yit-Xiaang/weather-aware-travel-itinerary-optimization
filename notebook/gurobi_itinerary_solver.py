import os
import time

import numpy as np


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
        os.path.expanduser("~/gurobi.lic"),
        "/Library/gurobi/gurobi.lic",
    ]
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    return None


def run_gurobi_itinerary_optimization(context):
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
    cost_penalty = cost_norm ** p

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

    tourist_profiles = {
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

    t_daily = 720
    total_budget = 1000
    gamma = 0.01
    hotel_utility_weight = 0.25
    switch_time_weight = 0.10
    switch_fixed_penalty = 0.08
    content_repeat_penalty_weight = 0.18
    content_heavy_repeat_penalty_weight = 0.12
    attraction_shortfall_penalty_weight = 2.5
    overnight_relocation_limit = 180
    time_limit_seconds = 900.0
    mip_gap_target = 0.02
    objective_scale = 1000
    max_travel = 90
    num_threads = max(1, min(8, os.cpu_count() or 1))
    mip_focus = 1
    heuristics = 0.20

    travel_norm = safe_minmax(travel_time_matrix)
    routes = {}
    edges = {}
    route_details = {}

    wine_indices = [i for i, row in top100_with_waiting_time.iterrows() if "Wine" in row["name"]]
    whale_indices = [i for i, row in top100_with_waiting_time.iterrows() if "Whale" in row["name"]]
    content_theme_groups = {
        theme: theme_df.index.tolist()
        for theme, theme_df in top100_with_waiting_time.groupby("content_theme")
    }

    n = len(top100_with_waiting_time)
    num_hotels = len(hotels_df)
    num_days = 3
    base_arcs = [
        (i, j)
        for i in range(n)
        for j in range(n)
        if i != j and travel_time_int[i, j] <= max_travel
    ]
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
        model.Params.Seed = 42
        model.Params.Presolve = 2

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
            (
                utility_score[i]
                - wait_penalty[i]
                - attraction_cost_penalty[i]
                + attraction_bonus_score
            )
            * x[i, d]
            for i in range(n)
            for d in range(num_days)
        )
        objective -= gp.quicksum(
            float(travel_penalty[i, j]) * y[i, j, d]
            for i, j in base_arcs
            for d in range(num_days)
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
                gp.quicksum(y[i, j, d] for i, j in base_arcs)
                == gp.quicksum(x[i, d] for i in range(n)) - day_used[d],
                name=f"path_edge_count_{d}",
            )

        for d in range(num_days - 1):
            model.addConstr(
                gp.quicksum(hotel_to_hotel_time_int[h, g] * q[h, g, d] for h in range(num_hotels) for g in range(num_hotels))
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
            print(f"  Solver finished with status [{status_label}] after {solve_seconds:.1f}s and no feasible incumbent")
            print()
            continue

        day_used_by_day = {d: int(round(day_used[d].X)) for d in range(num_days)}
        selected_by_day = {
            d: [i for i in range(n) if x[i, d].X > 0.5]
            for d in range(num_days)
        }
        travel_edges_by_day = {
            d: [(i, j) for i, j in base_arcs if y[i, j, d].X > 0.5]
            for d in range(num_days)
        }
        start_edges_by_day = {
            d: [(h, i) for h in range(num_hotels) for i in range(n) if s[h, i, d].X > 0.5]
            for d in range(num_days)
        }
        end_edges_by_day = {
            d: [(i, h) for i in range(n) for h in range(num_hotels) if e[i, h, d].X > 0.5]
            for d in range(num_days)
        }
        hotels_by_day = {
            d: next((h for h in range(num_hotels) if z[h, d].X > 0.5), None)
            for d in range(num_days)
        }
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
        }

        total_visit_wait = sum(
            visiting_time_int[i] + waiting_time_int[i]
            for day_nodes in selected_by_day.values()
            for i in day_nodes
        )
        total_travel = sum(
            travel_time_int[i, j]
            for day_edges in travel_edges_by_day.values()
            for i, j in day_edges
        )
        total_hotel_legs = sum(
            hotel_to_attr_time_int[h, i]
            for day_edges in start_edges_by_day.values()
            for h, i in day_edges
        ) + sum(
            hotel_to_attr_time_int[h, i]
            for day_edges in end_edges_by_day.values()
            for i, h in day_edges
        )
        total_switch_time = sum(
            hotel_to_hotel_time_int[h, g]
            for day_edges in transitions_by_day.values()
            for h, g in day_edges
            if h != g
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
                print(
                    f"    Day {d + 1}: Hotel={hotel_name} (${nightly_rate:.0f}), "
                    "No sightseeing route selected"
                )
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
                    print(
                        f"      Overnight decision: switch to {next_name} "
                        f"({switch_minutes:.1f} min relocation)"
                    )

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
        "daily_time_budget": t_daily,
        "total_budget": total_budget,
        "overnight_relocation_limit": overnight_relocation_limit,
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
        "K": num_days,
        "n": n,
        "num_hotels": num_hotels,
    }
