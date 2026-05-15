"""User-editable trip configuration loading and validation."""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_CONFIG = {
    "trip": {
        "state": "California",
        "trip_days": 7,
        "start_city_options": ["San Francisco", "Los Angeles"],
        "end_city_options": ["Los Angeles", "San Francisco"],
        "traveler_profile": "balanced",
        "scenario": "california_coast",
        "interest_profile": "balanced_interest",
    },
    "budget": {
        "user_budget": 2000,
        "budget_mode": "approximate",
        "hard_budget_multiplier": 1.20,
        "food_cost_per_day": 72,
        "attraction_cost_per_day": 28,
        "local_transport_buffer_per_day": 18,
    },
    "time": {
        "daily_time_budget_minutes": 720,
        "max_detour_minutes": 45,
        "max_intercity_drive_minutes_per_day": 360,
    },
    "hotels": {
        "allow_visit_city_without_hotel": True,
        "charge_hotel_only_for_overnight_base": True,
        "hotel_price_fallback": 150,
    },
    "enrichment": {
        "run_live_apis": False,
        "cache_ttl_days": 30,
        "use_yelp_live_api": False,
        "use_local_yelp_dataset": True,
        "use_yelp": True,
        "use_osm": True,
        "use_wikidata": True,
        "use_wikipedia": True,
        "use_open_meteo": True,
        "use_osrm": True,
        "use_social_signals": True,
        "overpass_radius_meters": 18000,
        "yelp_radius_meters": 2500,
    },
    "optimization": {
        "gurobi_time_limit_seconds": 120,
        "gurobi_mip_gap": 0.05,
        "max_cities": 6,
        "min_days_per_base_city": 1,
        "max_days_per_base_city": 3,
        "city_day_value_exponent": 0.82,
        "pass_through_city_bonus_weight": 0.18,
        "base_switch_penalty": 0.07,
        "max_pois_per_day": 4,
    },
    "bandit": {
        "enabled": True,
        "episodes": 40,
        "seeds": [42, 73, 101],
        "episode_grid": [20, 40, 80],
        "candidate_size_grid": [8, 12, 16],
        "arms": ["scenic_social", "fastest_low_cost", "balanced", "high_must_go", "low_detour"],
        "available_nature_arms": [
            "nature_heavy",
            "scenic_parks",
            "national_park_priority",
            "city_nature_balanced",
            "weather_safe_indoor_backup",
        ],
        "exploration_scale": 0.25,
        "top_k_gurobi_repairs": 3,
    },
    "interest": {
        "enabled": False,
        "mode": "custom",
        "axes": ["nature", "city", "culture", "history"],
        "weights": {
            "nature": 0.25,
            "city": 0.25,
            "culture": 0.25,
            "history": 0.25,
        },
        "presets": {
            "balanced_interest": {"nature": 0.25, "city": 0.25, "culture": 0.25, "history": 0.25},
            "nature_heavy": {"nature": 0.55, "city": 0.20, "culture": 0.15, "history": 0.10},
            "city_heavy": {"nature": 0.15, "city": 0.55, "culture": 0.20, "history": 0.10},
            "culture_heavy": {"nature": 0.15, "city": 0.20, "culture": 0.55, "history": 0.10},
            "history_heavy": {"nature": 0.15, "city": 0.20, "culture": 0.15, "history": 0.50},
        },
        "lambda_fit": 0.65,
        "lambda_park": 0.35,
        "lambda_weather": 0.30,
        "lambda_season": 0.20,
        "lambda_detour": 0.006,
        "lambda_balance": 0.20,
        "use_balance_objective": False,
        "preview_top_n": 12,
    },
    "nature": {
        "enabled": True,
        "use_nps_api": False,
        "use_curated_nature_fallback": True,
        "nps_api_key_env": "NPS_API_KEY",
        "max_nature_detour_minutes": 240,
        "max_single_day_drive_minutes_for_nature": 360,
        "min_nature_days_for_nature_heavy": 2,
        "min_nature_score_for_nature_heavy": 0.35,
        "national_park_bonus": 0.80,
        "state_park_bonus": 0.45,
        "protected_area_bonus": 0.35,
        "scenic_region_bonus": 0.30,
    },
    "social": {
        "must_go_is_mandatory": False,
        "must_go_bonus_weight": 0.85,
        "social_score_weight": 1.10,
        "corridor_fit_weight": 0.30,
        "detour_penalty_weight": 0.01,
    },
    "map": {
        "balanced_only_default_view": True,
        "refresh_road_geometry": False,
        "show_debug_on_load": False,
    },
    "map_export": {
        "mode": "both",
        "lightweight_max_routes": 1,
        "externalize_layers": True,
        "lazy_load_optional_layers": True,
        "simplify_geometry_tolerance": 0.0008,
        "include_debug_layers_in_share_map": False,
        "include_candidate_layers_in_share_map": False,
    },
    "utility": {
        "method": "bayesian_ucb",
        "uncertainty_bonus_kappa": 0.25,
        "bayes_prior_mean": 0.50,
        "bayes_prior_strength": 6.0,
        "corridor_fit_weight": 0.30,
        "detour_penalty_weight": 0.01,
        "must_go_bonus_weight": 0.85,
        "weather_risk_penalty_weight": 0.08,
    },
    "multi_objective": {
        "use_epsilon_constraints": True,
        "epsilon_detour_minutes": 45,
        "epsilon_weather_risk": 0.35,
        "min_diversity_score": 2.5,
        "diversity_bonus_weight": 0.12,
        "secondary_travel_penalty": 0.005,
    },
    "diversity": {
        "method": "submodular_sqrt",
        "mmr_lambda": 0.72,
        "category_similarity_weight": 0.45,
        "geo_similarity_weight": 0.25,
        "city_similarity_weight": 0.20,
        "source_similarity_weight": 0.10,
    },
    "learning_to_rank": {
        "enabled": False,
        "method": "bpr",
        "min_pairwise_examples": 200,
    },
    "trip_duration_comparison": {
        "enabled": True,
        "days": [7, 9, 12],
        "scale_budget_with_days": True,
    },
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _has_nested(mapping: dict[str, Any] | None, section: str, key: str) -> bool:
    section_value = mapping.get(section, {}) if isinstance(mapping, dict) else {}
    return isinstance(section_value, dict) and key in section_value


def _normalize_interest_weights(raw_weights: dict[str, Any] | None) -> dict[str, float]:
    axes = ["nature", "city", "culture", "history"]
    raw_weights = raw_weights if isinstance(raw_weights, dict) else {}
    values = {}
    for axis in axes:
        try:
            values[axis] = max(0.0, float(raw_weights.get(axis, 0.0) or 0.0))
        except Exception:
            values[axis] = 0.0
    total = sum(values.values())
    if total <= 0:
        return {axis: 1.0 / len(axes) for axis in axes}
    return {axis: values[axis] / total for axis in axes}


def _resolve_interest_settings(
    merged: dict[str, Any],
    loaded: dict[str, Any] | None = None,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve preset labels and trip-level aliases into the optimizer vector."""
    interest = dict(merged.get("interest", {}))
    trip = dict(merged.get("trip", {}))
    presets = interest.get("presets", {})
    if not isinstance(presets, dict):
        presets = DEFAULT_CONFIG["interest"]["presets"]

    override_enabled = _has_nested(overrides, "interest", "enabled")
    override_mode = _has_nested(overrides, "interest", "mode")
    override_weights = _has_nested(overrides, "interest", "weights")
    override_trip_profile = _has_nested(overrides, "trip", "interest_profile")

    loaded_enabled = _has_nested(loaded, "interest", "enabled")
    loaded_mode = _has_nested(loaded, "interest", "mode")
    loaded_weights = _has_nested(loaded, "interest", "weights")
    loaded_trip_profile = _has_nested(loaded, "trip", "interest_profile")

    trip_profile = str(trip.get("interest_profile") or "balanced_interest")
    mode = str(interest.get("mode") or "custom")

    default_interest = DEFAULT_CONFIG["interest"]
    default_trip_profile = str(DEFAULT_CONFIG["trip"].get("interest_profile", "balanced_interest"))
    loaded_mode_is_custom_default = loaded_mode and mode == str(default_interest.get("mode", "custom"))
    loaded_weights_are_default = loaded_weights and _normalize_interest_weights(
        interest.get("weights")
    ) == _normalize_interest_weights(default_interest.get("weights"))
    loaded_trip_profile_is_default = loaded_trip_profile and trip_profile == default_trip_profile

    explicit_enabled = override_enabled or (loaded_enabled and not override_trip_profile)
    explicit_mode = override_mode or (loaded_mode and not loaded_mode_is_custom_default)
    explicit_weights = override_weights or (loaded_weights and not loaded_weights_are_default)
    explicit_trip_profile = override_trip_profile or (loaded_trip_profile and not loaded_trip_profile_is_default)

    if explicit_trip_profile and not explicit_enabled:
        interest["enabled"] = True

    if not bool(interest.get("enabled", False)):
        interest["weights"] = _normalize_interest_weights(interest.get("weights"))
        merged["interest"] = interest
        merged["trip"] = trip
        return merged

    preset_name: str | None = None
    if explicit_mode and mode != "custom":
        preset_name = mode
    elif not explicit_mode and trip_profile and trip_profile in presets:
        preset_name = trip_profile
    elif mode != "custom" and mode in presets:
        preset_name = mode

    if preset_name:
        interest["mode"] = preset_name
        interest["weights"] = _normalize_interest_weights(presets.get(preset_name, {}))
        trip["interest_profile"] = preset_name
    elif explicit_weights or mode == "custom":
        interest["mode"] = "custom"
        interest["weights"] = _normalize_interest_weights(interest.get("weights"))
    else:
        interest["weights"] = _normalize_interest_weights(interest.get("weights"))

    merged["interest"] = interest
    merged["trip"] = trip
    return merged


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if not value:
        return ""
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"null", "none"}:
        return None
    if value.startswith("[") and value.endswith("]"):
        body = value[1:-1].strip()
        if not body:
            return []
        return [_parse_scalar(part.strip()) for part in body.split(",")]
    try:
        return ast.literal_eval(value)
    except Exception:
        return value


def _minimal_yaml_load(text: str) -> dict[str, Any]:
    """Parse the small YAML subset used by configs/default_trip_config.yaml.

    The fallback intentionally handles only nested mappings and inline scalar
    lists, which is enough for the project config when PyYAML is unavailable.
    """
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        key, separator, value = line.strip().partition(":")
        if not separator:
            raise ValueError(f"Invalid config line: {raw_line}")
        while stack and indent <= stack[-1][0]:
            stack.pop()
        if not stack:
            raise ValueError(f"Invalid config indentation: {raw_line}")
        parent = stack[-1][1]
        if value.strip():
            parent[key.strip()] = _parse_scalar(value)
        else:
            child: dict[str, Any] = {}
            parent[key.strip()] = child
            stack.append((indent, child))
    return root


def _load_yaml(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        loaded = yaml.safe_load(text) or {}
        if not isinstance(loaded, dict):
            raise ValueError("YAML root must be a mapping")
        return loaded
    except ImportError:
        return _minimal_yaml_load(text)


@dataclass(frozen=True)
class TripConfig:
    data: dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_CONFIG))
    source_path: str | None = None

    def section(self, name: str) -> dict[str, Any]:
        return dict(self.data.get(name, {}))

    def get(self, section: str, key: str, default: Any = None) -> Any:
        return self.data.get(section, {}).get(key, default)

    @property
    def trip_days(self) -> int:
        return int(self.get("trip", "trip_days", 7))

    @property
    def user_budget(self) -> float:
        return float(self.get("budget", "user_budget", 2000))

    @property
    def run_live_apis(self) -> bool:
        return bool(self.get("enrichment", "run_live_apis", False))

    @property
    def bandit_arms(self) -> list[str]:
        return list(self.get("bandit", "arms", DEFAULT_CONFIG["bandit"]["arms"]))

    def to_dict(self) -> dict[str, Any]:
        return json.loads(json.dumps(self.data))

    def to_frame(self) -> pd.DataFrame:
        rows = []
        for section, values in self.data.items():
            if isinstance(values, dict):
                for key, value in values.items():
                    rows.append({"section": section, "parameter": key, "value": value})
            else:
                rows.append({"section": section, "parameter": "", "value": values})
        return pd.DataFrame(rows)


def load_trip_config(
    path: str | Path = "configs/default_trip_config.yaml", overrides: dict[str, Any] | None = None
) -> TripConfig:
    path = Path(path)
    loaded = _load_yaml(path) if path.exists() else {}
    merged = _deep_merge(DEFAULT_CONFIG, loaded)
    if overrides:
        merged = _deep_merge(merged, overrides)
    merged = _resolve_interest_settings(merged, loaded=loaded, overrides=overrides)
    return TripConfig(data=merged, source_path=str(path))


def save_resolved_config(config: TripConfig, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = config.to_dict()
    payload["_source_path"] = config.source_path
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path
