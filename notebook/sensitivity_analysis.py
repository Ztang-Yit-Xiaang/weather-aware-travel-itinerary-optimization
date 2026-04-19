import contextlib
import importlib
import io
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import gurobi_itinerary_solver


PARAMETER_ORDER = ["alpha", "beta", "gamma", "lambda"]
PARAMETER_LABELS = {
    "alpha": "Congestion penalty alpha",
    "beta": "Travel penalty beta",
    "gamma": "Cost penalty gamma",
    "lambda": "Exploration bonus lambda",
}
METRIC_LABELS = {
    "total_attraction_utility": "Total attraction utility",
    "total_waiting_time": "Total waiting time (min)",
    "total_travel_distance_km": "Total travel distance (km)",
    "diversity_score": "Diversity score",
}
DEFAULT_SWEEP_SOLVER_OVERRIDES = {
    "time_limit_seconds": 60.0,
    "mip_gap_target": 0.05,
}


def _round_unique(values, decimals=6):
    rounded_values = [round(float(value), decimals) for value in values]
    return sorted(dict.fromkeys(rounded_values))


def _baseline_parameter_values(context, profile_name):
    if profile_name not in gurobi_itinerary_solver.DEFAULT_TOURIST_PROFILES:
        raise KeyError(f"Unknown profile '{profile_name}'")

    notebook_profiles = context.get("TOURIST_PROFILES") or context.get("tourist_profiles") or {}
    profile_cfg = dict(gurobi_itinerary_solver.DEFAULT_TOURIST_PROFILES[profile_name])
    if profile_name in notebook_profiles:
        profile_cfg.update(notebook_profiles[profile_name])

    gamma = gurobi_itinerary_solver.DEFAULT_MODEL_SETTINGS["gamma"]
    optimizer_outputs = context.get("optimizer_outputs")
    if isinstance(optimizer_outputs, dict):
        gamma = optimizer_outputs.get("model_settings", {}).get("gamma", gamma)

    return {
        "alpha": float(profile_cfg["alpha"]),
        "beta": float(profile_cfg["beta"]),
        "gamma": float(gamma),
        "lambda": float(profile_cfg.get("attraction_bonus", profile_cfg.get("lambda", 0.0))),
    }


def build_default_parameter_grid(context, profile_name="balanced"):
    base_values = _baseline_parameter_values(context, profile_name)

    alpha_grid = _round_unique(
        [
            max(0.0, base_values["alpha"] * factor)
            for factor in (0.5, 0.75, 1.0, 1.25, 1.5)
        ],
        decimals=4,
    )
    beta_grid = _round_unique(
        [
            max(0.0, base_values["beta"] * factor)
            for factor in (0.5, 0.75, 1.0, 1.25, 1.5)
        ],
        decimals=4,
    )

    gamma_base = base_values["gamma"]
    if gamma_base > 0:
        gamma_grid = _round_unique(
            [0.0, gamma_base * 0.5, gamma_base, gamma_base * 2.0, gamma_base * 4.0],
            decimals=5,
        )
    else:
        gamma_grid = [0.0, 0.005, 0.01, 0.02, 0.04]

    lambda_base = base_values["lambda"]
    if lambda_base > 0:
        lambda_grid = _round_unique(
            [0.0, lambda_base * 0.5, lambda_base, lambda_base * 1.5, lambda_base * 2.0],
            decimals=4,
        )
    else:
        lambda_grid = [0.0, 0.05, 0.1, 0.15, 0.2]

    return {
        "alpha": alpha_grid,
        "beta": beta_grid,
        "gamma": gamma_grid,
        "lambda": lambda_grid,
    }


def _normalize_grid(parameter_values, base_values):
    if parameter_values is None:
        return build_default_parameter_grid({}, profile_name="balanced")

    normalized = {}
    for parameter_name, values in parameter_values.items():
        if parameter_name not in PARAMETER_ORDER:
            continue
        if np.isscalar(values):
            values = [values]
        merged_values = list(values)
        merged_values.append(base_values[parameter_name])
        normalized[parameter_name] = _round_unique(merged_values)
    return normalized


def _matrix_from_context(context, primary_key, fallback_time_key=None):
    if primary_key in context:
        return np.asarray(context[primary_key], dtype=float)

    if fallback_time_key is None:
        return None

    avg_speed = float(context.get("avg_speed", 20.0))
    if fallback_time_key not in context:
        return None
    return np.asarray(context[fallback_time_key], dtype=float) * avg_speed / 60.0


def _normalized_shannon_diversity(counts):
    if not counts:
        return 0.0

    values = np.asarray(list(counts.values()), dtype=float)
    total = float(values.sum())
    if total <= 0 or len(values) <= 1:
        return 0.0

    probabilities = values / total
    entropy = -np.sum(probabilities * np.log(probabilities))
    return float(entropy / np.log(len(values)))


def _summarize_outputs(outputs, context, profile_name, parameter_name, parameter_value, base_values):
    route_details = outputs.get("route_details", {})
    if profile_name not in route_details:
        return {
            "profile": profile_name,
            "parameter_name": parameter_name,
            "parameter_value": float(parameter_value),
            "is_baseline": bool(np.isclose(parameter_value, base_values[parameter_name])),
            "has_solution": False,
            "status": "NO_SOLUTION",
            "selected_attractions": 0,
            "total_attraction_utility": np.nan,
            "total_waiting_time": np.nan,
            "total_travel_time_min": np.nan,
            "total_travel_distance_km": np.nan,
            "diversity_score": np.nan,
            "unique_themes": 0,
            "total_cost": np.nan,
            "objective_value": np.nan,
            "mip_gap": np.nan,
            "solve_seconds": np.nan,
            "alpha": np.nan,
            "beta": np.nan,
            "gamma": np.nan,
            "lambda": np.nan,
        }

    details = route_details[profile_name]
    top100 = outputs["top100_with_waiting_time"]
    selected_indices = outputs["routes"].get(profile_name, [])

    travel_time_matrix = np.asarray(context["travel_time_matrix"], dtype=float)
    hotel_to_attr_time = np.asarray(context["hotel_to_attr_time"], dtype=float)
    hotel_to_hotel_time = np.asarray(context["hotel_to_hotel_time"], dtype=float)
    distance_matrix = _matrix_from_context(context, "distance_matrix", "travel_time_matrix")
    hotel_to_attr_distance = _matrix_from_context(context, "hotel_to_attr", "hotel_to_attr_time")
    hotel_to_hotel_distance = _matrix_from_context(context, "hotel_to_hotel", "hotel_to_hotel_time")

    utility = top100["utility"].astype(float)
    waiting = top100["waiting_final"].astype(float)
    attraction_cost = top100["cost"].astype(float)

    travel_edges_by_day = details.get("travel_edges_by_day", {})
    start_edges_by_day = details.get("start_edges_by_day", {})
    end_edges_by_day = details.get("end_edges_by_day", {})
    transitions_by_day = details.get("transitions_by_day", {})
    content_mix = details.get("content_mix", {})

    travel_time_between = float(
        sum(travel_time_matrix[i, j] for day_edges in travel_edges_by_day.values() for i, j in day_edges)
    )
    hotel_leg_time = float(
        sum(hotel_to_attr_time[h, i] for day_edges in start_edges_by_day.values() for h, i in day_edges)
        + sum(hotel_to_attr_time[h, i] for day_edges in end_edges_by_day.values() for i, h in day_edges)
    )
    switch_time = float(
        sum(
            hotel_to_hotel_time[h, g]
            for day_edges in transitions_by_day.values()
            for h, g in day_edges
            if h != g
        )
    )
    total_travel_time = travel_time_between + hotel_leg_time + switch_time

    total_travel_distance = np.nan
    if (
        distance_matrix is not None
        and hotel_to_attr_distance is not None
        and hotel_to_hotel_distance is not None
    ):
        travel_distance_between = float(
            sum(distance_matrix[i, j] for day_edges in travel_edges_by_day.values() for i, j in day_edges)
        )
        hotel_leg_distance = float(
            sum(hotel_to_attr_distance[h, i] for day_edges in start_edges_by_day.values() for h, i in day_edges)
            + sum(hotel_to_attr_distance[h, i] for day_edges in end_edges_by_day.values() for i, h in day_edges)
        )
        switch_distance = float(
            sum(
                hotel_to_hotel_distance[h, g]
                for day_edges in transitions_by_day.values()
                for h, g in day_edges
                if h != g
            )
        )
        total_travel_distance = travel_distance_between + hotel_leg_distance + switch_distance

    hotels_df = context.get("hotels_df")
    total_hotel_cost = 0.0
    if hotels_df is not None:
        hotel_prices = hotels_df["nightly_price"].astype(float)
        total_hotel_cost = float(
            sum(
                hotel_prices.iloc[h]
                for h in details.get("hotels_by_day", {}).values()
                if h is not None
            )
        )

    applied_parameters = details.get("applied_parameters", {})
    total_attraction_cost = float(attraction_cost.iloc[selected_indices].sum()) if selected_indices else 0.0

    return {
        "profile": profile_name,
        "parameter_name": parameter_name,
        "parameter_value": float(parameter_value),
        "is_baseline": bool(np.isclose(parameter_value, base_values[parameter_name])),
        "has_solution": True,
        "status": details.get("status", "UNKNOWN"),
        "selected_attractions": int(len(selected_indices)),
        "total_attraction_utility": float(utility.iloc[selected_indices].sum()) if selected_indices else 0.0,
        "total_waiting_time": float(waiting.iloc[selected_indices].sum()) if selected_indices else 0.0,
        "total_travel_time_min": total_travel_time,
        "total_travel_distance_km": total_travel_distance,
        "diversity_score": _normalized_shannon_diversity(content_mix),
        "unique_themes": int(len(content_mix)),
        "total_cost": total_attraction_cost + total_hotel_cost,
        "objective_value": float(details.get("objective_value", np.nan)),
        "mip_gap": float(details.get("mip_gap", np.nan)),
        "solve_seconds": float(details.get("solve_seconds", np.nan)),
        "alpha": float(applied_parameters.get("alpha", np.nan)),
        "beta": float(applied_parameters.get("beta", np.nan)),
        "gamma": float(applied_parameters.get("gamma", np.nan)),
        "lambda": float(applied_parameters.get("lambda", np.nan)),
    }


def run_parameter_sweep(
    context,
    profile_name="balanced",
    parameter_values=None,
    solver_overrides=None,
    save_path=None,
    show_progress=True,
    quiet_solver_output=True,
):
    importlib.reload(gurobi_itinerary_solver)

    base_values = _baseline_parameter_values(context, profile_name)
    parameter_grid = parameter_values or build_default_parameter_grid(context, profile_name=profile_name)

    normalized_grid = {}
    for parameter_name in PARAMETER_ORDER:
        if parameter_name not in parameter_grid:
            continue
        values = parameter_grid[parameter_name]
        if np.isscalar(values):
            values = [values]
        normalized_grid[parameter_name] = _round_unique(list(values) + [base_values[parameter_name]])

    effective_solver_overrides = DEFAULT_SWEEP_SOLVER_OVERRIDES.copy()
    if solver_overrides:
        effective_solver_overrides.update(solver_overrides)

    rows = []
    for parameter_name in PARAMETER_ORDER:
        if parameter_name not in normalized_grid:
            continue

        for parameter_value in normalized_grid[parameter_name]:
            if show_progress:
                print(
                    f"Running sensitivity scenario: profile={profile_name}, "
                    f"{parameter_name}={parameter_value}"
                )

            profile_overrides = {profile_name: {}}
            global_overrides = {}

            if parameter_name == "gamma":
                global_overrides["gamma"] = float(parameter_value)
            elif parameter_name == "lambda":
                profile_overrides[profile_name]["attraction_bonus"] = float(parameter_value)
            else:
                profile_overrides[profile_name][parameter_name] = float(parameter_value)

            solver_call = lambda: gurobi_itinerary_solver.run_gurobi_itinerary_optimization(
                context,
                profile_overrides=profile_overrides,
                global_overrides=global_overrides or None,
                solver_overrides=effective_solver_overrides,
                selected_profiles=[profile_name],
            )

            if quiet_solver_output:
                with contextlib.redirect_stdout(io.StringIO()):
                    outputs = solver_call()
            else:
                outputs = solver_call()

            rows.append(
                _summarize_outputs(
                    outputs,
                    context,
                    profile_name=profile_name,
                    parameter_name=parameter_name,
                    parameter_value=parameter_value,
                    base_values=base_values,
                )
            )

    results_df = pd.DataFrame(rows)
    if not results_df.empty:
        results_df = results_df.sort_values(["parameter_name", "parameter_value"]).reset_index(drop=True)

    if save_path:
        output_dir = os.path.dirname(save_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        results_df.to_csv(save_path, index=False)

    return results_df


def plot_parameter_sensitivity(
    results_df,
    parameter_name,
    metrics=None,
    figsize=(12, 8),
):
    if parameter_name not in PARAMETER_ORDER:
        raise KeyError(f"Unknown parameter '{parameter_name}'")

    parameter_df = results_df[results_df["parameter_name"] == parameter_name].copy()
    if parameter_df.empty:
        raise ValueError(f"No sensitivity results found for parameter '{parameter_name}'")

    parameter_df = parameter_df.sort_values("parameter_value")
    metrics = metrics or list(METRIC_LABELS.keys())
    profile_name = str(parameter_df["profile"].iloc[0])
    parameter_label = PARAMETER_LABELS.get(parameter_name, parameter_name)
    baseline_rows = parameter_df[parameter_df["is_baseline"]]
    baseline_x = float(baseline_rows["parameter_value"].iloc[0]) if not baseline_rows.empty else None

    fig, axes = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)
    axes = np.asarray(axes).reshape(-1)

    for axis, metric_name in zip(axes, metrics):
        axis.plot(
            parameter_df["parameter_value"],
            parameter_df[metric_name],
            marker="o",
            linewidth=2,
            color="#1f77b4",
        )
        if baseline_x is not None:
            axis.axvline(
                baseline_x,
                linestyle="--",
                linewidth=1,
                color="#6b7280",
                label="baseline",
            )
        axis.set_title(METRIC_LABELS.get(metric_name, metric_name))
        axis.set_xlabel(parameter_label)
        axis.set_ylabel(METRIC_LABELS.get(metric_name, metric_name))
        axis.grid(alpha=0.3, linestyle=":")

    for axis in axes[len(metrics):]:
        axis.axis("off")

    fig.suptitle(f"Sensitivity Analysis for {profile_name}: {parameter_label}")
    return fig, axes


def plot_all_parameter_sensitivity(results_df, metrics=None, figsize=(12, 8)):
    figures = {}
    for parameter_name in PARAMETER_ORDER:
        if parameter_name not in results_df.get("parameter_name", pd.Series(dtype=object)).values:
            continue
        fig, _ = plot_parameter_sensitivity(
            results_df,
            parameter_name=parameter_name,
            metrics=metrics,
            figsize=figsize,
        )
        figures[parameter_name] = fig
    return figures
