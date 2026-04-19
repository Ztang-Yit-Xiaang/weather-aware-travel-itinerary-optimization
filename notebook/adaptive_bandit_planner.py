from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import gurobi_itinerary_solver


METHOD_LABELS = {
    "static": "Static Optimization",
    "greedy": "Greedy Sequential",
    "cts": "Contextual Thompson Sampling",
    "ccb": "Contextual Combinatorial Bandit",
}

DEFAULT_SCENARIO_LIBRARY = {
    "nominal": {
        "weather_mean": 1.00,
        "weather_std": 0.06,
        "congestion_mean": 1.00,
        "congestion_std": 0.06,
        "wait_noise_std": 0.08,
        "travel_noise_std": 0.08,
        "utility_noise_std": 0.05,
    },
    "weather_shock": {
        "weather_mean": 1.20,
        "weather_std": 0.12,
        "congestion_mean": 1.05,
        "congestion_std": 0.08,
        "wait_noise_std": 0.12,
        "travel_noise_std": 0.08,
        "utility_noise_std": 0.06,
    },
    "congestion_shock": {
        "weather_mean": 1.02,
        "weather_std": 0.06,
        "congestion_mean": 1.22,
        "congestion_std": 0.14,
        "wait_noise_std": 0.08,
        "travel_noise_std": 0.12,
        "utility_noise_std": 0.05,
    },
}


@dataclass
class ScenarioConfig:
    scenario_mode: str
    weather_multipliers_by_day: list[float]
    congestion_multipliers_by_day: list[float]
    wait_noise_std: float
    travel_noise_std: float
    utility_noise_std: float


@dataclass
class AdaptiveState:
    profile_name: str
    top100_with_waiting_time: pd.DataFrame
    hotels_df: pd.DataFrame
    travel_time_matrix: np.ndarray
    hotel_to_attr_time: np.ndarray
    hotel_to_hotel_time: np.ndarray
    distance_matrix: np.ndarray
    hotel_to_attr_distance: np.ndarray
    hotel_to_hotel_distance: np.ndarray
    utility_raw: np.ndarray
    utility_norm: np.ndarray
    waiting_raw: np.ndarray
    waiting_norm: np.ndarray
    visiting_raw: np.ndarray
    cost_raw: np.ndarray
    cost_norm: np.ndarray
    cost_penalty: np.ndarray
    tourist_profile: dict
    tourist_profiles: dict
    solver_settings: dict
    model_settings: dict
    route_details: dict
    static_day_sequences: dict
    static_hotels_by_day: dict
    num_days: int
    daily_time_budget: float
    total_budget: float
    avg_speed_kmh: float
    travel_scale: float
    seed: int
    scenario_overrides: dict = field(default_factory=dict)


@dataclass
class EpisodeResult:
    method: str
    profile_name: str
    scenario_mode: str
    episode_idx: int
    seed: int
    step_rows: list[dict] = field(default_factory=list)
    total_realized_reward: float = 0.0
    total_realized_utility: float = 0.0
    total_waiting_time: float = 0.0
    total_travel_time: float = 0.0
    total_travel_distance_km: float = 0.0
    total_cost: float = 0.0
    planned_attractions: int = 0
    executed_planned_attractions: int = 0
    truncated_days: int = 0
    visited_sequence: list[int] = field(default_factory=list)
    theme_counts: dict = field(default_factory=dict)


class LinearThompsonSamplingModel:
    def __init__(self, feature_dim, prior_precision=1.0, sample_scale=0.25):
        self.feature_dim = int(feature_dim)
        self.prior_precision = float(prior_precision)
        self.sample_scale = float(sample_scale)
        self.A = np.eye(self.feature_dim, dtype=float) * self.prior_precision
        self.b = np.zeros(self.feature_dim, dtype=float)

    def posterior_mean(self):
        return np.linalg.solve(self.A, self.b)

    def sample_theta(self, rng):
        covariance = np.linalg.inv(self.A)
        mean = covariance @ self.b
        return rng.multivariate_normal(mean, covariance * (self.sample_scale ** 2))

    def sample_scores(self, feature_matrix, rng):
        theta = self.sample_theta(rng)
        return feature_matrix @ theta

    def predict_scores(self, feature_matrix):
        theta = self.posterior_mean()
        return feature_matrix @ theta

    def update(self, feature_vector, reward):
        feature_vector = np.asarray(feature_vector, dtype=float)
        reward = float(reward)
        self.A += np.outer(feature_vector, feature_vector)
        self.b += reward * feature_vector


def _normalized_shannon_diversity(counts):
    if not counts:
        return 0.0
    values = np.asarray(list(counts.values()), dtype=float)
    if len(values) <= 1 or float(values.sum()) <= 0.0:
        return 0.0
    probs = values / float(values.sum())
    entropy = -np.sum(probs * np.log(probs))
    return float(entropy / np.log(len(values)))


def _safe_fraction(numerator, denominator):
    denominator = float(denominator)
    if abs(denominator) <= 1e-12:
        return 0.0
    return float(numerator) / denominator


def _default_static_hotels(hotels_df, num_days):
    if hotels_df.empty:
        return {day_idx: None for day_idx in range(num_days)}
    cheapest_hotel_idx = int(np.argmin(hotels_df["nightly_price"].astype(float).to_numpy()))
    return {day_idx: cheapest_hotel_idx for day_idx in range(num_days)}


def _reconstruct_sequence(start_edges, travel_edges, selected_nodes):
    if not selected_nodes:
        return []
    start_attr = start_edges[0][1] if start_edges else selected_nodes[0]
    next_node = {i: j for i, j in travel_edges}
    ordered = []
    current = start_attr
    visited = set()
    while current is not None and current not in visited:
        ordered.append(current)
        visited.add(current)
        current = next_node.get(current)
    for node in selected_nodes:
        if node not in visited:
            ordered.append(node)
    return ordered


def _build_static_sequences(route_details, num_days):
    sequences = {}
    for day_idx in range(num_days):
        selected_nodes = route_details.get("selected_by_day", {}).get(day_idx, [])
        start_edges = route_details.get("start_edges_by_day", {}).get(day_idx, [])
        travel_edges = route_details.get("travel_edges_by_day", {}).get(day_idx, [])
        sequences[day_idx] = _reconstruct_sequence(start_edges, travel_edges, selected_nodes)
    return sequences


def _resolve_model_settings(notebook_context, optimizer_outputs):
    model_settings = dict(gurobi_itinerary_solver.DEFAULT_MODEL_SETTINGS)
    if isinstance(optimizer_outputs, dict):
        model_settings.update(optimizer_outputs.get("model_settings", {}))
    model_settings.update(notebook_context.get("MODEL_SETTINGS", {}))
    return model_settings


def _resolve_solver_settings(notebook_context, optimizer_outputs):
    solver_settings = dict(gurobi_itinerary_solver.DEFAULT_SOLVER_SETTINGS)
    if isinstance(optimizer_outputs, dict):
        solver_settings.update(optimizer_outputs.get("solver_settings", {}))
    solver_settings.update(notebook_context.get("SOLVER_SETTINGS", {}))
    return solver_settings


def _build_candidate_generation_context(state):
    return {
        "top100_with_waiting_time": state.top100_with_waiting_time,
        "travel_time_matrix": state.travel_time_matrix,
        "hotel_to_attr_time": state.hotel_to_attr_time,
        "hotel_to_hotel_time": state.hotel_to_hotel_time,
        "hotels_df": state.hotels_df,
        "TOURIST_PROFILES": state.tourist_profiles,
        "SOLVER_SETTINGS": state.solver_settings,
        "MODEL_SETTINGS": state.model_settings,
    }


def build_adaptive_bandit_context(notebook_context, profile_name, seed=42, scenario_config=None):
    top100 = notebook_context["top100_with_waiting_time"].copy()
    if "content_theme" not in top100.columns:
        top100["content_theme"] = top100.apply(gurobi_itinerary_solver.infer_content_theme, axis=1)

    hotels_df = notebook_context["hotels_df"].copy()
    travel_time_matrix = np.asarray(notebook_context["travel_time_matrix"], dtype=float)
    hotel_to_attr_time = np.asarray(notebook_context["hotel_to_attr_time"], dtype=float)
    hotel_to_hotel_time = np.asarray(notebook_context["hotel_to_hotel_time"], dtype=float)

    avg_speed_kmh = float(notebook_context.get("avg_speed", 20.0))
    distance_matrix = np.asarray(
        notebook_context.get("distance_matrix", travel_time_matrix * avg_speed_kmh / 60.0),
        dtype=float,
    )
    hotel_to_attr_distance = np.asarray(
        notebook_context.get("hotel_to_attr", hotel_to_attr_time * avg_speed_kmh / 60.0),
        dtype=float,
    )
    hotel_to_hotel_distance = np.asarray(
        notebook_context.get("hotel_to_hotel", hotel_to_hotel_time * avg_speed_kmh / 60.0),
        dtype=float,
    )

    utility_raw = top100["utility"].astype(float).to_numpy()
    waiting_raw = top100["waiting_final"].astype(float).to_numpy()
    visiting_raw = top100["visit_duration_sim"].astype(float).to_numpy()
    cost_raw = top100["cost"].astype(float).to_numpy()
    utility_norm = np.asarray(
        notebook_context.get("utility_norm", gurobi_itinerary_solver.safe_minmax(utility_raw)),
        dtype=float,
    )
    waiting_norm = np.asarray(
        notebook_context.get("waiting_time_norm", gurobi_itinerary_solver.safe_minmax(waiting_raw)),
        dtype=float,
    )
    cost_norm = np.asarray(gurobi_itinerary_solver.safe_minmax(cost_raw), dtype=float)
    cost_penalty = cost_norm ** 0.7

    optimizer_outputs = notebook_context.get("optimizer_outputs", {})
    tourist_profiles = copy.deepcopy(gurobi_itinerary_solver.DEFAULT_TOURIST_PROFILES)
    notebook_profiles = notebook_context.get("TOURIST_PROFILES") or {}
    for local_profile, config in notebook_profiles.items():
        if local_profile not in tourist_profiles:
            tourist_profiles[local_profile] = {}
        tourist_profiles[local_profile].update(config)

    if profile_name not in tourist_profiles:
        raise KeyError(f"Unknown profile '{profile_name}' for adaptive planning context")

    route_details_all = notebook_context.get("route_details") or optimizer_outputs.get("route_details", {}) or {}
    static_route_details = route_details_all.get(profile_name, {})

    model_settings = _resolve_model_settings(notebook_context, optimizer_outputs)
    solver_settings = _resolve_solver_settings(notebook_context, optimizer_outputs)
    num_days = int(
        notebook_context.get("K", optimizer_outputs.get("K", model_settings.get("num_days", 3)))
    )
    static_day_sequences = _build_static_sequences(static_route_details, num_days)
    static_hotels_by_day = dict(static_route_details.get("hotels_by_day", {}))
    if not static_hotels_by_day:
        static_hotels_by_day = _default_static_hotels(hotels_df, num_days)

    return AdaptiveState(
        profile_name=profile_name,
        top100_with_waiting_time=top100,
        hotels_df=hotels_df,
        travel_time_matrix=travel_time_matrix,
        hotel_to_attr_time=hotel_to_attr_time,
        hotel_to_hotel_time=hotel_to_hotel_time,
        distance_matrix=distance_matrix,
        hotel_to_attr_distance=hotel_to_attr_distance,
        hotel_to_hotel_distance=hotel_to_hotel_distance,
        utility_raw=utility_raw,
        utility_norm=utility_norm,
        waiting_raw=waiting_raw,
        waiting_norm=waiting_norm,
        visiting_raw=visiting_raw,
        cost_raw=cost_raw,
        cost_norm=cost_norm,
        cost_penalty=cost_penalty,
        tourist_profile=dict(tourist_profiles[profile_name]),
        tourist_profiles=tourist_profiles,
        solver_settings=solver_settings,
        model_settings=model_settings,
        route_details=static_route_details,
        static_day_sequences=static_day_sequences,
        static_hotels_by_day=static_hotels_by_day,
        num_days=num_days,
        daily_time_budget=float(model_settings["daily_time_budget"]),
        total_budget=float(model_settings["total_budget"]),
        avg_speed_kmh=avg_speed_kmh,
        travel_scale=max(
            1.0,
            float(np.max(travel_time_matrix)) if np.size(travel_time_matrix) else 1.0,
            float(np.max(hotel_to_attr_time)) if np.size(hotel_to_attr_time) else 1.0,
        ),
        seed=int(seed),
        scenario_overrides=dict(scenario_config or {}),
    )


def _build_episode_scenario(state, scenario_mode, episode_seed, overrides=None):
    if scenario_mode not in DEFAULT_SCENARIO_LIBRARY:
        raise KeyError(f"Unknown scenario mode '{scenario_mode}'")
    scenario_values = dict(DEFAULT_SCENARIO_LIBRARY[scenario_mode])
    scenario_values.update(state.scenario_overrides)
    if overrides:
        scenario_values.update(overrides)

    rng = np.random.default_rng(int(episode_seed))
    weather_by_day = np.clip(
        rng.normal(
            loc=float(scenario_values["weather_mean"]),
            scale=float(scenario_values["weather_std"]),
            size=state.num_days,
        ),
        0.6,
        2.0,
    )
    congestion_by_day = np.clip(
        rng.normal(
            loc=float(scenario_values["congestion_mean"]),
            scale=float(scenario_values["congestion_std"]),
            size=state.num_days,
        ),
        0.6,
        2.0,
    )

    return ScenarioConfig(
        scenario_mode=scenario_mode,
        weather_multipliers_by_day=list(weather_by_day.astype(float)),
        congestion_multipliers_by_day=list(congestion_by_day.astype(float)),
        wait_noise_std=float(scenario_values["wait_noise_std"]),
        travel_noise_std=float(scenario_values["travel_noise_std"]),
        utility_noise_std=float(scenario_values["utility_noise_std"]),
    )


def _get_theme_name(state, attr_idx):
    return str(state.top100_with_waiting_time.iloc[attr_idx]["content_theme"])


def _compute_novelty_bonus(state, attr_idx, theme_counts):
    return gurobi_itinerary_solver._compute_theme_novelty_bonus(
        _get_theme_name(state, attr_idx),
        theme_counts,
    )


def _build_step_feature_vector(
    state,
    attr_idx,
    current_hotel_idx,
    current_attr_idx,
    remaining_time,
    remaining_budget,
    visited_attractions,
    theme_counts,
    scenario,
    day_idx,
):
    predicted_travel = (
        float(state.hotel_to_attr_time[current_hotel_idx, attr_idx])
        if current_attr_idx is None
        else float(state.travel_time_matrix[current_attr_idx, attr_idx])
    )
    novelty_bonus = _compute_novelty_bonus(state, attr_idx, theme_counts)
    theme_repeat = float(theme_counts.get(_get_theme_name(state, attr_idx), 0))
    return np.asarray(
        [
            1.0,
            float(state.utility_norm[attr_idx]),
            float(state.waiting_norm[attr_idx]),
            predicted_travel / state.travel_scale,
            float(state.cost_penalty[attr_idx]),
            float(novelty_bonus),
            _safe_fraction(remaining_time, state.daily_time_budget),
            _safe_fraction(remaining_budget, state.total_budget),
            _safe_fraction(len(visited_attractions), len(state.top100_with_waiting_time)),
            theme_repeat,
            float(scenario.weather_multipliers_by_day[day_idx]),
            float(scenario.congestion_multipliers_by_day[day_idx]),
        ],
        dtype=float,
    )


def _build_subset_feature_vector(
    state,
    candidate_row,
    remaining_time,
    remaining_budget,
    visited_attractions,
    theme_counts,
    scenario,
    day_idx,
):
    sequence = list(candidate_row.get("sequence", []))
    if not sequence:
        return np.zeros(12, dtype=float)

    novelty_values = [_compute_novelty_bonus(state, attr_idx, theme_counts) for attr_idx in sequence]
    return np.asarray(
        [
            1.0,
            float(state.utility_norm[sequence].sum()),
            float(state.waiting_norm[sequence].sum()),
            _safe_fraction(candidate_row.get("predicted_total_travel_time", 0.0), state.travel_scale),
            float(state.cost_penalty[sequence].sum()),
            float(candidate_row.get("predicted_diversity_score", 0.0)),
            float(np.mean(novelty_values)),
            _safe_fraction(len(sequence), max(1, state.tourist_profile["max_attractions"])),
            _safe_fraction(remaining_time, state.daily_time_budget),
            _safe_fraction(remaining_budget, state.total_budget),
            float(scenario.weather_multipliers_by_day[day_idx]),
            float(scenario.congestion_multipliers_by_day[day_idx]),
        ],
        dtype=float,
    )


def _predicted_step_reward(state, attr_idx, current_hotel_idx, current_attr_idx, theme_counts):
    predicted_travel = (
        float(state.hotel_to_attr_time[current_hotel_idx, attr_idx])
        if current_attr_idx is None
        else float(state.travel_time_matrix[current_attr_idx, attr_idx])
    )
    profile_cfg = state.tourist_profile
    return float(
        state.utility_norm[attr_idx]
        - float(profile_cfg["alpha"]) * state.waiting_norm[attr_idx]
        - float(profile_cfg["beta"]) * (predicted_travel / state.travel_scale)
        - float(state.model_settings["gamma"]) * state.cost_penalty[attr_idx]
        + float(profile_cfg["attraction_bonus"]) * _compute_novelty_bonus(state, attr_idx, theme_counts)
    )


def _simulate_travel_time(base_time, multiplier, noise_std, rng):
    return max(0.0, float(base_time) * float(multiplier) * max(0.2, 1.0 + float(rng.normal(0.0, noise_std))))


def _simulate_utility_raw(base_utility, noise_std, rng):
    return max(0.0, float(base_utility) * max(0.1, 1.0 + float(rng.normal(0.0, noise_std))))


def _simulate_stop_outcome(
    state,
    scenario,
    day_idx,
    current_hotel_idx,
    current_attr_idx,
    attr_idx,
    theme_counts,
    rng,
):
    if current_attr_idx is None:
        base_travel_time = float(state.hotel_to_attr_time[current_hotel_idx, attr_idx])
        base_travel_distance = float(state.hotel_to_attr_distance[current_hotel_idx, attr_idx])
    else:
        base_travel_time = float(state.travel_time_matrix[current_attr_idx, attr_idx])
        base_travel_distance = float(state.distance_matrix[current_attr_idx, attr_idx])

    travel_multiplier = float(scenario.congestion_multipliers_by_day[day_idx])
    weather_multiplier = float(scenario.weather_multipliers_by_day[day_idx])
    realized_travel_time = _simulate_travel_time(base_travel_time, travel_multiplier, scenario.travel_noise_std, rng)
    realized_travel_distance = max(
        0.0,
        base_travel_distance * travel_multiplier * max(0.2, 1.0 + float(rng.normal(0.0, scenario.travel_noise_std))),
    )
    realized_wait = max(
        0.0,
        float(state.waiting_raw[attr_idx]) * weather_multiplier * max(0.2, 1.0 + float(rng.normal(0.0, scenario.wait_noise_std))),
    )
    realized_visit = float(state.visiting_raw[attr_idx])
    realized_utility_raw = _simulate_utility_raw(state.utility_raw[attr_idx], scenario.utility_noise_std, rng)
    realized_utility_norm = float(
        np.clip(
            state.utility_norm[attr_idx] * max(0.1, 1.0 + float(rng.normal(0.0, scenario.utility_noise_std))),
            0.0,
            None,
        )
    )
    novelty_bonus = _compute_novelty_bonus(state, attr_idx, theme_counts)
    realized_wait_norm = realized_wait / max(1.0, float(np.max(state.waiting_raw)))
    realized_travel_norm = realized_travel_time / state.travel_scale
    realized_reward = float(
        realized_utility_norm
        - float(state.tourist_profile["alpha"]) * realized_wait_norm
        - float(state.tourist_profile["beta"]) * realized_travel_norm
        - float(state.model_settings["gamma"]) * state.cost_penalty[attr_idx]
        + float(state.tourist_profile["attraction_bonus"]) * novelty_bonus
    )

    return {
        "realized_travel_time": realized_travel_time,
        "realized_travel_distance_km": realized_travel_distance,
        "realized_wait_time": realized_wait,
        "realized_visit_time": realized_visit,
        "realized_utility_raw": realized_utility_raw,
        "realized_reward": realized_reward,
        "novelty_bonus": float(novelty_bonus),
    }


def _simulate_return_to_hotel(state, scenario, day_idx, current_attr_idx, current_hotel_idx, rng):
    if current_attr_idx is None or current_hotel_idx is None:
        return {"realized_travel_time": 0.0, "realized_travel_distance_km": 0.0}

    base_travel_time = float(state.hotel_to_attr_time[current_hotel_idx, current_attr_idx])
    base_travel_distance = float(state.hotel_to_attr_distance[current_hotel_idx, current_attr_idx])
    multiplier = float(scenario.congestion_multipliers_by_day[day_idx])
    return {
        "realized_travel_time": _simulate_travel_time(base_travel_time, multiplier, scenario.travel_noise_std, rng),
        "realized_travel_distance_km": max(
            0.0,
            base_travel_distance * multiplier * max(0.2, 1.0 + float(rng.normal(0.0, scenario.travel_noise_std))),
        ),
    }


def _record_step_row(
    result,
    method,
    profile_name,
    scenario_mode,
    episode_idx,
    day_idx,
    action_rank,
    action_type,
    attr_idx,
    candidate_count,
    remaining_time,
    remaining_budget,
    predicted_reward,
    predicted_travel_time,
    predicted_wait_time,
    predicted_cost,
    predicted_diversity_score,
    realized_reward,
    realized_travel_time,
    realized_travel_distance_km,
    realized_wait_time,
    realized_visit_time,
    realized_cost,
    novelty_bonus,
    backend,
    note="",
):
    attr_name = None
    if attr_idx is not None:
        attr_name = str(result["state"].top100_with_waiting_time.iloc[attr_idx]["name"])

    result["episode"].step_rows.append(
        {
            "method": method,
            "method_label": METHOD_LABELS.get(method, method),
            "profile_name": profile_name,
            "scenario_mode": scenario_mode,
            "episode_idx": episode_idx,
            "day_idx": day_idx,
            "action_rank": action_rank,
            "action_type": action_type,
            "attr_idx": attr_idx,
            "attr_name": attr_name,
            "candidate_count": int(candidate_count),
            "remaining_time_before": float(remaining_time),
            "remaining_budget_before": float(remaining_budget),
            "predicted_reward": float(predicted_reward),
            "predicted_travel_time": float(predicted_travel_time),
            "predicted_wait_time": float(predicted_wait_time),
            "predicted_cost": float(predicted_cost),
            "predicted_diversity_score": float(predicted_diversity_score),
            "realized_reward": float(realized_reward),
            "realized_travel_time": float(realized_travel_time),
            "realized_travel_distance_km": float(realized_travel_distance_km),
            "realized_wait_time": float(realized_wait_time),
            "realized_visit_time": float(realized_visit_time),
            "realized_cost": float(realized_cost),
            "novelty_bonus": float(novelty_bonus),
            "backend": backend,
            "note": note,
        }
    )


def _can_visit_attraction(
    state,
    attr_idx,
    current_hotel_idx,
    current_attr_idx,
    remaining_time,
    remaining_budget,
    day_wait_so_far,
    day_visit_count,
):
    outbound = (
        float(state.hotel_to_attr_time[current_hotel_idx, attr_idx])
        if current_attr_idx is None
        else float(state.travel_time_matrix[current_attr_idx, attr_idx])
    )
    return_leg = float(state.hotel_to_attr_time[current_hotel_idx, attr_idx])
    required_time = outbound + float(state.visiting_raw[attr_idx]) + float(state.waiting_raw[attr_idx]) + return_leg
    if required_time > float(remaining_time):
        return False
    if float(state.cost_raw[attr_idx]) > float(remaining_budget):
        return False
    projected_wait = float(day_wait_so_far) + float(state.waiting_raw[attr_idx])
    projected_count = int(day_visit_count) + 1
    if projected_wait > float(state.tourist_profile["max_wait"]) * projected_count:
        return False
    return True


def _initialize_episode_result(method, state, scenario_mode, episode_idx, seed):
    return EpisodeResult(
        method=method,
        profile_name=state.profile_name,
        scenario_mode=scenario_mode,
        episode_idx=int(episode_idx),
        seed=int(seed),
    )


def _finalize_episode_result(result):
    diversity_score = _normalized_shannon_diversity(result.theme_counts)
    planned = max(1, int(result.planned_attractions))
    if result.method in {"cts", "greedy"}:
        feasibility_retention = 1.0 if result.visited_sequence else 0.0
    else:
        feasibility_retention = _safe_fraction(result.executed_planned_attractions, planned)

    return {
        "method": result.method,
        "method_label": METHOD_LABELS.get(result.method, result.method),
        "profile_name": result.profile_name,
        "scenario_mode": result.scenario_mode,
        "episode_idx": int(result.episode_idx),
        "seed": int(result.seed),
        "total_realized_reward": float(result.total_realized_reward),
        "total_realized_utility": float(result.total_realized_utility),
        "total_waiting_time": float(result.total_waiting_time),
        "total_travel_time": float(result.total_travel_time),
        "total_travel_distance_km": float(result.total_travel_distance_km),
        "total_cost": float(result.total_cost),
        "visited_attractions": int(len(result.visited_sequence)),
        "diversity_score": float(diversity_score),
        "planned_attractions": int(result.planned_attractions),
        "executed_planned_attractions": int(result.executed_planned_attractions),
        "feasibility_retention": float(feasibility_retention),
        "truncated_days": int(result.truncated_days),
    }


def _execute_attraction_step(
    state,
    episode_result,
    scenario,
    day_idx,
    current_hotel_idx,
    current_attr_idx,
    attr_idx,
    remaining_time,
    remaining_budget,
    theme_counts,
    visited_attractions,
    action_rank,
    candidate_count,
    predicted_reward,
    backend,
    rng,
):
    predicted_travel = (
        float(state.hotel_to_attr_time[current_hotel_idx, attr_idx])
        if current_attr_idx is None
        else float(state.travel_time_matrix[current_attr_idx, attr_idx])
    )
    predicted_wait = float(state.waiting_raw[attr_idx])
    predicted_cost = float(state.cost_raw[attr_idx])
    outcome = _simulate_stop_outcome(
        state,
        scenario,
        day_idx,
        current_hotel_idx,
        current_attr_idx,
        attr_idx,
        theme_counts,
        rng,
    )

    _record_step_row(
        {
            "episode": episode_result,
            "state": state,
        },
        method=episode_result.method,
        profile_name=state.profile_name,
        scenario_mode=scenario.scenario_mode,
        episode_idx=episode_result.episode_idx,
        day_idx=day_idx,
        action_rank=action_rank,
        action_type="visit",
        attr_idx=attr_idx,
        candidate_count=candidate_count,
        remaining_time=remaining_time,
        remaining_budget=remaining_budget,
        predicted_reward=predicted_reward,
        predicted_travel_time=predicted_travel,
        predicted_wait_time=predicted_wait,
        predicted_cost=predicted_cost,
        predicted_diversity_score=_normalized_shannon_diversity(theme_counts),
        realized_reward=outcome["realized_reward"],
        realized_travel_time=outcome["realized_travel_time"],
        realized_travel_distance_km=outcome["realized_travel_distance_km"],
        realized_wait_time=outcome["realized_wait_time"],
        realized_visit_time=outcome["realized_visit_time"],
        realized_cost=predicted_cost,
        novelty_bonus=outcome["novelty_bonus"],
        backend=backend,
    )

    episode_result.total_realized_reward += float(outcome["realized_reward"])
    episode_result.total_realized_utility += float(outcome["realized_utility_raw"])
    episode_result.total_waiting_time += float(outcome["realized_wait_time"])
    episode_result.total_travel_time += float(outcome["realized_travel_time"])
    episode_result.total_travel_distance_km += float(outcome["realized_travel_distance_km"])
    episode_result.total_cost += float(predicted_cost)
    episode_result.visited_sequence.append(int(attr_idx))
    theme_name = _get_theme_name(state, attr_idx)
    theme_counts[theme_name] = theme_counts.get(theme_name, 0) + 1
    episode_result.theme_counts = dict(theme_counts)
    visited_attractions.add(int(attr_idx))

    return (
        float(remaining_time)
        - float(outcome["realized_travel_time"])
        - float(outcome["realized_wait_time"])
        - float(outcome["realized_visit_time"]),
        float(remaining_budget) - float(predicted_cost),
        float(outcome["realized_wait_time"]),
        float(outcome["realized_reward"]),
    )


def _execute_return_to_hotel(
    state,
    episode_result,
    scenario,
    day_idx,
    current_attr_idx,
    current_hotel_idx,
    remaining_time,
    remaining_budget,
    rng,
):
    return_outcome = _simulate_return_to_hotel(
        state,
        scenario,
        day_idx,
        current_attr_idx,
        current_hotel_idx,
        rng,
    )
    if return_outcome["realized_travel_time"] <= 0.0:
        return remaining_time

    _record_step_row(
        {
            "episode": episode_result,
            "state": state,
        },
        method=episode_result.method,
        profile_name=state.profile_name,
        scenario_mode=scenario.scenario_mode,
        episode_idx=episode_result.episode_idx,
        day_idx=day_idx,
        action_rank=-1,
        action_type="return_to_hotel",
        attr_idx=None,
        candidate_count=0,
        remaining_time=remaining_time,
        remaining_budget=remaining_budget,
        predicted_reward=0.0,
        predicted_travel_time=float(state.hotel_to_attr_time[current_hotel_idx, current_attr_idx]),
        predicted_wait_time=0.0,
        predicted_cost=0.0,
        predicted_diversity_score=0.0,
        realized_reward=0.0,
        realized_travel_time=return_outcome["realized_travel_time"],
        realized_travel_distance_km=return_outcome["realized_travel_distance_km"],
        realized_wait_time=0.0,
        realized_visit_time=0.0,
        realized_cost=0.0,
        novelty_bonus=0.0,
        backend="simulator",
    )
    episode_result.total_travel_time += float(return_outcome["realized_travel_time"])
    episode_result.total_travel_distance_km += float(return_outcome["realized_travel_distance_km"])
    return float(remaining_time) - float(return_outcome["realized_travel_time"])


def _choose_greedy_next_step(state, candidate_pool, current_hotel_idx, current_attr_idx, theme_counts):
    best_choice = None
    best_reward = None
    for candidate in candidate_pool:
        attr_idx = int(candidate["first_attraction"])
        predicted_reward = _predicted_step_reward(state, attr_idx, current_hotel_idx, current_attr_idx, theme_counts)
        if best_choice is None or predicted_reward > best_reward:
            best_choice = candidate
            best_reward = predicted_reward
    return best_choice, float(best_reward or 0.0)


def _run_static_episode(state, scenario, episode_idx, episode_seed):
    rng = np.random.default_rng(int(episode_seed))
    result = _initialize_episode_result("static", state, scenario.scenario_mode, episode_idx, episode_seed)
    visited_attractions = set()
    theme_counts = {}
    remaining_budget = float(state.total_budget)

    for day_idx in range(state.num_days):
        hotel_idx = state.static_hotels_by_day.get(day_idx)
        if hotel_idx is None:
            continue
        remaining_budget -= float(state.hotels_df.iloc[hotel_idx]["nightly_price"])
        remaining_time = float(state.daily_time_budget)
        current_attr_idx = None
        day_wait_so_far = 0.0
        day_visit_count = 0
        planned_sequence = [idx for idx in state.static_day_sequences.get(day_idx, []) if idx not in visited_attractions]
        result.planned_attractions += len(planned_sequence)

        for action_rank, attr_idx in enumerate(planned_sequence):
            if not _can_visit_attraction(
                state,
                attr_idx,
                hotel_idx,
                current_attr_idx,
                remaining_time,
                remaining_budget,
                day_wait_so_far,
                day_visit_count,
            ):
                result.truncated_days += 1
                break

            predicted_reward = _predicted_step_reward(state, attr_idx, hotel_idx, current_attr_idx, theme_counts)
            remaining_time, remaining_budget, realized_wait, _ = _execute_attraction_step(
                state,
                result,
                scenario,
                day_idx,
                hotel_idx,
                current_attr_idx,
                attr_idx,
                remaining_time,
                remaining_budget,
                theme_counts,
                visited_attractions,
                action_rank=action_rank,
                candidate_count=max(1, len(planned_sequence)),
                predicted_reward=predicted_reward,
                backend="static_plan",
                rng=rng,
            )
            current_attr_idx = attr_idx
            day_wait_so_far += realized_wait
            day_visit_count += 1
            result.executed_planned_attractions += 1

        remaining_time = _execute_return_to_hotel(
            state,
            result,
            scenario,
            day_idx,
            current_attr_idx,
            hotel_idx,
            remaining_time,
            remaining_budget,
            rng,
        )
        _ = remaining_time

    return result


def _run_stepwise_policy_episode(
    state,
    scenario,
    episode_idx,
    episode_seed,
    method,
    candidate_pool_size=5,
    solver_overrides=None,
):
    rng = np.random.default_rng(int(episode_seed))
    result = _initialize_episode_result(method, state, scenario.scenario_mode, episode_idx, episode_seed)
    visited_attractions = set()
    theme_counts = {}
    remaining_budget = float(state.total_budget)
    candidate_context = _build_candidate_generation_context(state)
    bandit_model = LinearThompsonSamplingModel(feature_dim=12)

    for day_idx in range(state.num_days):
        hotel_idx = state.static_hotels_by_day.get(day_idx)
        if hotel_idx is None:
            continue
        remaining_budget -= float(state.hotels_df.iloc[hotel_idx]["nightly_price"])
        remaining_time = float(state.daily_time_budget)
        current_attr_idx = None
        day_wait_so_far = 0.0
        day_visit_count = 0
        action_rank = 0

        while True:
            candidate_pool = gurobi_itinerary_solver.generate_receding_horizon_candidate_pool(
                candidate_context,
                profile_name=state.profile_name,
                current_hotel_idx=hotel_idx,
                current_attr_idx=current_attr_idx,
                remaining_time=remaining_time,
                remaining_budget=remaining_budget,
                visited_attractions=visited_attractions,
                visited_theme_counts=theme_counts,
                candidate_pool_size=candidate_pool_size,
                mode="step",
                solver_overrides=solver_overrides,
            )
            if not candidate_pool:
                break

            unique_candidates = {}
            for candidate in candidate_pool:
                attr_idx = int(candidate["first_attraction"])
                if attr_idx not in unique_candidates or candidate["predicted_total_reward"] > unique_candidates[attr_idx]["predicted_total_reward"]:
                    unique_candidates[attr_idx] = candidate
            candidate_rows = list(unique_candidates.values())
            if not candidate_rows:
                break

            if method == "greedy":
                chosen_candidate, predicted_reward = _choose_greedy_next_step(
                    state,
                    candidate_rows,
                    hotel_idx,
                    current_attr_idx,
                    theme_counts,
                )
                feature_vector = _build_step_feature_vector(
                    state,
                    chosen_candidate["first_attraction"],
                    hotel_idx,
                    current_attr_idx,
                    remaining_time,
                    remaining_budget,
                    visited_attractions,
                    theme_counts,
                    scenario,
                    day_idx,
                )
            else:
                feature_matrix = np.vstack(
                    [
                        _build_step_feature_vector(
                            state,
                            candidate["first_attraction"],
                            hotel_idx,
                            current_attr_idx,
                            remaining_time,
                            remaining_budget,
                            visited_attractions,
                            theme_counts,
                            scenario,
                            day_idx,
                        )
                        for candidate in candidate_rows
                    ]
                )
                sampled_scores = bandit_model.sample_scores(feature_matrix, rng)
                chosen_idx = int(np.argmax(sampled_scores))
                chosen_candidate = candidate_rows[chosen_idx]
                predicted_reward = _predicted_step_reward(
                    state,
                    int(chosen_candidate["first_attraction"]),
                    hotel_idx,
                    current_attr_idx,
                    theme_counts,
                )
                feature_vector = feature_matrix[chosen_idx]

            attr_idx = int(chosen_candidate["first_attraction"])
            if not _can_visit_attraction(
                state,
                attr_idx,
                hotel_idx,
                current_attr_idx,
                remaining_time,
                remaining_budget,
                day_wait_so_far,
                day_visit_count,
            ):
                break

            remaining_time, remaining_budget, realized_wait, realized_reward = _execute_attraction_step(
                state,
                result,
                scenario,
                day_idx,
                hotel_idx,
                current_attr_idx,
                attr_idx,
                remaining_time,
                remaining_budget,
                theme_counts,
                visited_attractions,
                action_rank=action_rank,
                candidate_count=len(candidate_rows),
                predicted_reward=predicted_reward,
                backend=chosen_candidate.get("backend", "unknown"),
                rng=rng,
            )
            if method == "cts":
                bandit_model.update(feature_vector, realized_reward)
            current_attr_idx = attr_idx
            day_wait_so_far += realized_wait
            day_visit_count += 1
            action_rank += 1

        remaining_time = _execute_return_to_hotel(
            state,
            result,
            scenario,
            day_idx,
            current_attr_idx,
            hotel_idx,
            remaining_time,
            remaining_budget,
            rng,
        )
        _ = remaining_time

    return result


def _run_ccb_episode(
    state,
    scenario,
    episode_idx,
    episode_seed,
    candidate_pool_size=5,
    solver_overrides=None,
):
    rng = np.random.default_rng(int(episode_seed))
    result = _initialize_episode_result("ccb", state, scenario.scenario_mode, episode_idx, episode_seed)
    visited_attractions = set()
    theme_counts = {}
    remaining_budget = float(state.total_budget)
    candidate_context = _build_candidate_generation_context(state)
    bandit_model = LinearThompsonSamplingModel(feature_dim=12)

    for day_idx in range(state.num_days):
        hotel_idx = state.static_hotels_by_day.get(day_idx)
        if hotel_idx is None:
            continue
        remaining_budget -= float(state.hotels_df.iloc[hotel_idx]["nightly_price"])
        remaining_time = float(state.daily_time_budget)
        current_attr_idx = None
        day_wait_so_far = 0.0
        day_visit_count = 0

        candidate_pool = gurobi_itinerary_solver.generate_receding_horizon_candidate_pool(
            candidate_context,
            profile_name=state.profile_name,
            current_hotel_idx=hotel_idx,
            current_attr_idx=None,
            remaining_time=remaining_time,
            remaining_budget=remaining_budget,
            visited_attractions=visited_attractions,
            visited_theme_counts=theme_counts,
            candidate_pool_size=candidate_pool_size,
            mode="day",
            solver_overrides=solver_overrides,
        )
        if not candidate_pool:
            continue

        feature_matrix = np.vstack(
            [
                _build_subset_feature_vector(
                    state,
                    candidate,
                    remaining_time,
                    remaining_budget,
                    visited_attractions,
                    theme_counts,
                    scenario,
                    day_idx,
                )
                for candidate in candidate_pool
            ]
        )
        sampled_scores = bandit_model.sample_scores(feature_matrix, rng)
        chosen_idx = int(np.argmax(sampled_scores))
        chosen_candidate = candidate_pool[chosen_idx]
        chosen_sequence = [idx for idx in chosen_candidate.get("sequence", []) if idx not in visited_attractions]
        result.planned_attractions += len(chosen_sequence)

        executed_sequence = []
        day_reward = 0.0
        for action_rank, attr_idx in enumerate(chosen_sequence):
            if not _can_visit_attraction(
                state,
                attr_idx,
                hotel_idx,
                current_attr_idx,
                remaining_time,
                remaining_budget,
                day_wait_so_far,
                day_visit_count,
            ):
                result.truncated_days += 1
                break

            predicted_reward = _predicted_step_reward(state, attr_idx, hotel_idx, current_attr_idx, theme_counts)
            remaining_time, remaining_budget, realized_wait, realized_reward = _execute_attraction_step(
                state,
                result,
                scenario,
                day_idx,
                hotel_idx,
                current_attr_idx,
                attr_idx,
                remaining_time,
                remaining_budget,
                theme_counts,
                visited_attractions,
                action_rank=action_rank,
                candidate_count=len(candidate_pool),
                predicted_reward=predicted_reward,
                backend=chosen_candidate.get("backend", "unknown"),
                rng=rng,
            )
            current_attr_idx = attr_idx
            day_wait_so_far += realized_wait
            day_visit_count += 1
            day_reward += realized_reward
            executed_sequence.append(attr_idx)
            result.executed_planned_attractions += 1

        if executed_sequence:
            executed_candidate = dict(chosen_candidate)
            executed_candidate["sequence"] = list(executed_sequence)
            executed_features = _build_subset_feature_vector(
                state,
                executed_candidate,
                remaining_time,
                remaining_budget,
                visited_attractions,
                theme_counts,
                scenario,
                day_idx,
            )
            bandit_model.update(executed_features, day_reward)

        remaining_time = _execute_return_to_hotel(
            state,
            result,
            scenario,
            day_idx,
            current_attr_idx,
            hotel_idx,
            remaining_time,
            remaining_budget,
            rng,
        )
        _ = remaining_time

    return result


def run_adaptive_bandit_benchmark(
    notebook_context,
    profile_name="balanced",
    methods=("static", "greedy", "cts", "ccb"),
    n_episodes=5,
    scenario_modes=("nominal", "weather_shock", "congestion_shock"),
    candidate_pool_size=5,
    seed=42,
    solver_overrides=None,
    scenario_config=None,
):
    state = build_adaptive_bandit_context(
        notebook_context,
        profile_name=profile_name,
        seed=seed,
        scenario_config=scenario_config,
    )

    methods = list(methods)
    step_rows = []
    episode_rows = []
    method_to_runner = {
        "static": _run_static_episode,
        "greedy": _run_stepwise_policy_episode,
        "cts": _run_stepwise_policy_episode,
        "ccb": _run_ccb_episode,
    }

    effective_solver_overrides = dict(solver_overrides or {})
    effective_solver_overrides.setdefault("candidate_pool_size", candidate_pool_size)

    for scenario_idx, scenario_mode in enumerate(scenario_modes):
        for episode_idx in range(int(n_episodes)):
            episode_seed = int(seed) + scenario_idx * 1000 + episode_idx
            scenario = _build_episode_scenario(
                state,
                scenario_mode=scenario_mode,
                episode_seed=episode_seed,
                overrides=scenario_config,
            )

            for method in methods:
                if method not in method_to_runner:
                    raise KeyError(f"Unknown adaptive benchmark method '{method}'")

                if method == "static":
                    episode_result = method_to_runner[method](
                        state,
                        scenario,
                        episode_idx,
                        episode_seed,
                    )
                elif method in {"greedy", "cts"}:
                    episode_result = method_to_runner[method](
                        state,
                        scenario,
                        episode_idx,
                        episode_seed,
                        method,
                        candidate_pool_size=candidate_pool_size,
                        solver_overrides=effective_solver_overrides,
                    )
                else:
                    episode_result = method_to_runner[method](
                        state,
                        scenario,
                        episode_idx,
                        episode_seed,
                        candidate_pool_size=candidate_pool_size,
                        solver_overrides=effective_solver_overrides,
                    )

                step_rows.extend(episode_result.step_rows)
                episode_rows.append(_finalize_episode_result(episode_result))

    step_log = pd.DataFrame(step_rows)
    episode_log = pd.DataFrame(episode_rows)
    if episode_log.empty:
        summary = pd.DataFrame()
        return {"episode_log": episode_log, "step_log": step_log, "summary": summary}

    static_rewards = (
        episode_log[episode_log["method"] == "static"][
            ["profile_name", "scenario_mode", "episode_idx", "total_realized_reward"]
        ]
        .rename(columns={"total_realized_reward": "static_reward_reference"})
        .copy()
    )
    episode_log = episode_log.merge(
        static_rewards,
        on=["profile_name", "scenario_mode", "episode_idx"],
        how="left",
    )
    episode_log["adaptation_gain_vs_static"] = (
        episode_log["total_realized_reward"] - episode_log["static_reward_reference"]
    )
    episode_log.loc[episode_log["method"] == "static", "adaptation_gain_vs_static"] = 0.0

    aggregate_metrics = [
        "total_realized_reward",
        "total_realized_utility",
        "total_waiting_time",
        "total_travel_time",
        "total_travel_distance_km",
        "total_cost",
        "visited_attractions",
        "diversity_score",
        "feasibility_retention",
        "adaptation_gain_vs_static",
    ]
    summary = (
        episode_log.groupby(["profile_name", "scenario_mode", "method", "method_label"])[aggregate_metrics]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        column_name if isinstance(column_name, str) else "_".join([part for part in column_name if part]).strip("_")
        for column_name in summary.columns.to_flat_index()
    ]

    return {"episode_log": episode_log, "step_log": step_log, "summary": summary}


def plot_adaptive_bandit_results(summary_df, episode_log_df):
    import matplotlib.pyplot as plt

    if summary_df.empty:
        return {"figure": None, "comparison_table": pd.DataFrame(), "summary_table": summary_df}

    metrics = [
        "total_realized_reward_mean",
        "total_waiting_time_mean",
        "total_travel_time_mean",
        "total_cost_mean",
        "diversity_score_mean",
        "adaptation_gain_vs_static_mean",
    ]
    metric_titles = {
        "total_realized_reward_mean": "Realized Reward",
        "total_waiting_time_mean": "Waiting Time",
        "total_travel_time_mean": "Travel Time",
        "total_cost_mean": "Total Cost",
        "diversity_score_mean": "Diversity Score",
        "adaptation_gain_vs_static_mean": "Gain vs Static",
    }

    scenarios = list(summary_df["scenario_mode"].dropna().unique())
    methods = [method for method in METHOD_LABELS if method in set(summary_df["method"].unique())]
    x = np.arange(len(methods))
    width = 0.24 if len(scenarios) > 1 else 0.5

    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    axes = np.asarray(axes).reshape(-1)

    for axis, metric_name in zip(axes, metrics):
        for scenario_idx, scenario_mode in enumerate(scenarios):
            scenario_rows = summary_df[summary_df["scenario_mode"] == scenario_mode]
            values = []
            for method in methods:
                matching = scenario_rows[scenario_rows["method"] == method]
                values.append(float(matching.iloc[0][metric_name]) if not matching.empty else 0.0)
            offset = (scenario_idx - (len(scenarios) - 1) / 2.0) * width
            axis.bar(x + offset, values, width=width, label=scenario_mode)
        axis.set_title(metric_titles.get(metric_name, metric_name))
        axis.set_xticks(x)
        axis.set_xticklabels([METHOD_LABELS.get(method, method) for method in methods], rotation=20, ha="right")
        axis.grid(alpha=0.25, linestyle=":")

    if scenarios:
        axes[0].legend(loc="upper left", fontsize=9)
    fig.suptitle("Adaptive Bandit Benchmark Results", fontsize=14)

    comparison_table = (
        episode_log_df.groupby(["scenario_mode", "method_label"], as_index=False)[
            [
                "total_realized_reward",
                "total_waiting_time",
                "total_travel_time",
                "total_cost",
                "diversity_score",
                "adaptation_gain_vs_static",
            ]
        ]
        .mean()
        .sort_values(["scenario_mode", "total_realized_reward"], ascending=[True, False])
        .reset_index(drop=True)
    )

    return {
        "figure": fig,
        "comparison_table": comparison_table,
        "summary_table": summary_df,
    }
