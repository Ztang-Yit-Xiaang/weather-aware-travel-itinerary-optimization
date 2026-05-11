"""Agent/UI-ready request schema for continuous interest-bar planning.

The UI interest bar is represented as p = [p_nature, p_city, p_culture, p_history].
All entries are nonnegative and normalized so sum(p) = 1.
Preset labels only set p; they are not separate optimizer modes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .config import DEFAULT_CONFIG


INTEREST_AXES = ("nature", "city", "culture", "history")


def _default_interest_config() -> dict[str, Any]:
    return dict(DEFAULT_CONFIG.get("interest", {}))


def normalize_interest_weights(raw_weights: dict[str, Any] | None) -> dict[str, float]:
    """Return a nonnegative interest vector over nature/city/culture/history."""
    if not isinstance(raw_weights, dict):
        raw_weights = {}
    values: dict[str, float] = {}
    for axis in INTEREST_AXES:
        try:
            value = float(raw_weights.get(axis, 0.0) or 0.0)
        except Exception:
            value = 0.0
        values[axis] = max(0.0, value)
    total = sum(values.values())
    if total <= 0.0:
        equal = 1.0 / len(INTEREST_AXES)
        return {axis: equal for axis in INTEREST_AXES}
    return {axis: values[axis] / total for axis in INTEREST_AXES}


def preset_to_interest_weights(preset_name: str, presets: dict[str, Any] | None = None) -> dict[str, float]:
    """Resolve a named preset into normalized bar weights."""
    interest_config = _default_interest_config()
    preset_map = presets if isinstance(presets, dict) else interest_config.get("presets", {})
    raw = preset_map.get(str(preset_name), preset_map.get("balanced_interest", {}))
    return normalize_interest_weights(raw if isinstance(raw, dict) else {})


@dataclass(frozen=True)
class TripPlanningRequest:
    start_city: str = "San Francisco"
    trip_days: int = 7
    traveler_profile: str = "balanced"
    scenario: str = "california_coast"
    interest_mode: str = "balanced_interest"
    interest_weights: dict[str, float] | None = None
    must_include: list[str] = field(default_factory=list)
    allow_city_days: bool = True
    allow_nature_regions: bool = True
    max_daily_drive_minutes: int = 360
    budget_level: str = "medium"
    user_budget: float | None = None


def request_to_config_overrides(request: TripPlanningRequest | dict[str, Any]) -> dict[str, Any]:
    """Convert a UI/agent request into TripConfig overrides."""
    if isinstance(request, dict):
        request = TripPlanningRequest(**request)

    interest_config = _default_interest_config()
    if request.interest_weights:
        weights = normalize_interest_weights(request.interest_weights)
        mode = "custom"
    else:
        mode = str(request.interest_mode or "balanced_interest")
        weights = preset_to_interest_weights(mode, interest_config.get("presets", {}))

    budget_by_level = {"low": 1400.0, "medium": 2200.0, "high": 3600.0}
    budget = request.user_budget
    if budget is None:
        budget = budget_by_level.get(str(request.budget_level).lower(), budget_by_level["medium"])

    return {
        "trip": {
            "trip_days": int(request.trip_days),
            "start_city_options": [str(request.start_city)],
            "end_city_options": [str(request.start_city)],
            "traveler_profile": str(request.traveler_profile),
            "scenario": str(request.scenario),
        },
        "budget": {"user_budget": float(budget)},
        "time": {"max_intercity_drive_minutes_per_day": int(request.max_daily_drive_minutes)},
        "interest": {
            "enabled": True,
            "mode": mode,
            "weights": weights,
        },
        "request": {
            "must_include": list(request.must_include),
            "allow_city_days": bool(request.allow_city_days),
            "allow_nature_regions": bool(request.allow_nature_regions),
        },
    }

