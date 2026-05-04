"""Shared dataclass schemas for the production-compatible itinerary system."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TripRequest:
    state: str
    trip_days: int
    start_city_options: list[str]
    end_city_options: list[str]
    traveler_profile: str
    user_budget: float | None = None
    use_social_signals: bool = True
    route_mode: str = "bandit_guided_gurobi"


@dataclass(frozen=True)
class EnrichedPOI:
    name: str
    city: str
    latitude: float
    longitude: float
    final_poi_value: float
    social_score: float = 0.0
    social_must_go: bool = False
    corridor_fit: float = 0.0
    detour_minutes: float = 0.0
    source_list: str = "unknown"


@dataclass(frozen=True)
class CityScore:
    city: str
    city_value_score: float
    data_confidence: float
    data_uncertainty: float
    candidate_attractions: int


@dataclass(frozen=True)
class BudgetEstimate:
    low: float
    expected: float
    high: float
    soft_budget: float
    hard_budget: float


@dataclass(frozen=True)
class HierarchicalGurobiPlan:
    gateway_start: str
    gateway_end: str
    city_sequence: list[str]
    days_by_city: dict[str, int]
    overnight_bases: list[str] = field(default_factory=list)
    pass_through_pois: list[str] = field(default_factory=list)
    objective: float = 0.0
    solver_status: str = "not_solved"


@dataclass(frozen=True)
class BanditArm:
    arm_id: str
    route_search_strategy: str
    candidate_bundle: str
    optimizer_role: str
    estimated_solver_minutes: float


@dataclass(frozen=True)
class GurobiRouteResult:
    route_id: str
    selected_pois: list[str]
    total_value: float
    total_cost: float
    total_time_minutes: float
    feasible: bool
    solver_status: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExperimentResult:
    strategy: str
    total_utility: float
    budget_used: float
    runtime_seconds: float
    feasible: bool
    metadata: dict[str, Any] = field(default_factory=dict)
