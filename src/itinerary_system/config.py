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
        "run_original_yelp_only_baseline": True,
    },
    "bandit": {
        "enabled": True,
        "episodes": 40,
        "seeds": [42, 73, 101],
        "episode_grid": [20, 40, 80],
        "candidate_size_grid": [8, 12, 16],
        "arms": ["scenic_social", "fastest_low_cost", "balanced", "high_must_go", "low_detour"],
        "exploration_scale": 0.25,
        "top_k_gurobi_repairs": 3,
    },
    "social": {
        "must_go_is_mandatory": False,
        "must_go_bonus_weight": 0.85,
        "social_score_weight": 1.10,
        "corridor_fit_weight": 0.30,
        "detour_penalty_weight": 0.01,
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
    """Parse the small YAML subset used by configs/default_trip_config.yaml."""
    root: dict[str, Any] = {}
    current_section: str | None = None
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if not line.startswith(" "):
            key, _, value = line.partition(":")
            current_section = key.strip()
            root[current_section] = _parse_scalar(value) if value.strip() else {}
            continue
        if current_section is None:
            raise ValueError(f"Invalid config line before section: {raw_line}")
        key, _, value = line.strip().partition(":")
        root[current_section][key.strip()] = _parse_scalar(value)
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


def load_trip_config(path: str | Path = "configs/default_trip_config.yaml", overrides: dict[str, Any] | None = None) -> TripConfig:
    path = Path(path)
    loaded = _load_yaml(path) if path.exists() else {}
    merged = _deep_merge(DEFAULT_CONFIG, loaded)
    if overrides:
        merged = _deep_merge(merged, overrides)
    return TripConfig(data=merged, source_path=str(path))


def save_resolved_config(config: TripConfig, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = config.to_dict()
    payload["_source_path"] = config.source_path
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path
