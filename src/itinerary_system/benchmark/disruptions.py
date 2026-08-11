"""Deterministic BENCH-001 disruption scenario generation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal

from ..repair_planner import RepairRequest
from ..research_artifacts import PlanArtifactV2, stable_content_hash

EvidenceStatus = Literal["observed", "synthetic"]


class DisruptionFamily(StrEnum):
    """Canonical six disruption families for repair benchmarks."""

    WEATHER_DETERIORATION = "weather_deterioration"
    ROAD_CLOSURE = "road_closure"
    HOTEL_UNAVAILABILITY = "hotel_unavailability"
    ATTRACTION_CLOSURE = "attraction_closure"
    REDUCED_DRIVING_TOLERANCE = "reduced_driving_tolerance"
    NEW_MUST_VISIT = "new_must_visit"


@dataclass(frozen=True)
class DisruptionScenario:
    """One benchmark disruption and the repair request it induces."""

    scenario_id: str
    family: DisruptionFamily
    parent_plan_id: str
    catalog_snapshot_id: str
    context_snapshot_id: str
    seed: int
    evidence_status: EvidenceStatus
    affected_days: tuple[int, ...]
    target_stop_ids: tuple[str, ...]
    request: RepairRequest
    schema_version: str = "benchmark-disruption-scenario-v1"

    def to_record(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "family": self.family.value,
            "parent_plan_id": self.parent_plan_id,
            "catalog_snapshot_id": self.catalog_snapshot_id,
            "context_snapshot_id": self.context_snapshot_id,
            "seed": self.seed,
            "evidence_status": self.evidence_status,
            "affected_days": list(self.affected_days),
            "target_stop_ids": list(self.target_stop_ids),
            "request": _repair_request_record(self.request),
            "schema_version": self.schema_version,
        }


class DisruptionGenerator:
    """Create deterministic repair requests from a frozen parent plan."""

    def __init__(
        self,
        *,
        seed: int = 0,
        evidence_status: EvidenceStatus = "synthetic",
        candidate_pool: tuple[dict[str, Any], ...] = (),
        max_daily_travel_minutes: float = 360.0,
    ) -> None:
        if evidence_status not in {"observed", "synthetic"}:
            raise ValueError("evidence_status must be 'observed' or 'synthetic'")
        self.seed = int(seed)
        self.evidence_status = evidence_status
        self.candidate_pool = tuple(dict(candidate) for candidate in candidate_pool)
        if float(max_daily_travel_minutes) <= 0:
            raise ValueError("max_daily_travel_minutes must be positive")
        self.max_daily_travel_minutes = float(max_daily_travel_minutes)

    def generate(self, parent_plan: PlanArtifactV2, bundle: Any | None = None) -> tuple[DisruptionScenario, ...]:
        """Return one deterministic scenario for each BENCH-001 family."""

        return (
            self.weather(parent_plan, bundle),
            self.road_closure(parent_plan, bundle),
            self.hotel_unavailability(parent_plan, bundle),
            self.attraction_closure(parent_plan, bundle),
            self.reduced_driving_tolerance(parent_plan, bundle),
            self.new_must_visit(parent_plan, bundle),
        )

    def weather(self, parent_plan: PlanArtifactV2, bundle: Any | None = None) -> DisruptionScenario:
        target = _highest_weather_stop(parent_plan)
        day = _stop_day(target, default=_first_day(parent_plan))
        stop_id = _stop_id(target, fallback="weather_target")
        return self._scenario(
            parent_plan,
            bundle,
            family=DisruptionFamily.WEATHER_DETERIORATION,
            affected_days=(day,),
            target_stop_ids=(stop_id,),
            user_intent=f"Repair weather deterioration affecting {stop_id}.",
            tolerance_profile={
                "max_stop_weather_risk": 0.45,
                "max_daily_travel_minutes": self.max_daily_travel_minutes,
            },
            confirmed_constraints={
                "weather_risk_overrides": {stop_id: 0.95},
                "weather_feasible": {stop_id: False},
            },
            candidate_pois=(
                _replacement_candidate(
                    parent_plan,
                    family="weather_backup",
                    day=day,
                    candidate_pool=self.candidate_pool,
                ),
            ),
        )

    def road_closure(self, parent_plan: PlanArtifactV2, bundle: Any | None = None) -> DisruptionScenario:
        day = _busiest_day(parent_plan)
        route_id = str(parent_plan.route_ids_by_day.get(day) or f"route_day_{day}")
        target_ids = _stop_ids_for_day(parent_plan, day)
        return self._scenario(
            parent_plan,
            bundle,
            family=DisruptionFamily.ROAD_CLOSURE,
            affected_days=(day,),
            target_stop_ids=target_ids,
            user_intent=f"Repair road closure on {route_id}.",
            tolerance_profile={"max_daily_travel_minutes": self.max_daily_travel_minutes},
            confirmed_constraints={
                "closed_route_ids": (route_id,),
                "boundary_route_days": (day,),
                "route_revalidation_required": True,
            },
        )

    def hotel_unavailability(self, parent_plan: PlanArtifactV2, bundle: Any | None = None) -> DisruptionScenario:
        day = _busiest_day(parent_plan)
        lodging_id = _lodging_for_day(parent_plan, day)
        return self._scenario(
            parent_plan,
            bundle,
            family=DisruptionFamily.HOTEL_UNAVAILABILITY,
            affected_days=(day,),
            target_stop_ids=_stop_ids_for_day(parent_plan, day),
            user_intent=f"Repair lodging unavailability for {lodging_id}.",
            tolerance_profile={"max_daily_travel_minutes": self.max_daily_travel_minutes},
            confirmed_constraints={
                "unavailable_lodging_ids": (lodging_id,),
                "allow_booked_relaxation": True,
                "replacement_lodging_candidates": (f"{lodging_id}_backup",),
            },
        )

    def attraction_closure(self, parent_plan: PlanArtifactV2, bundle: Any | None = None) -> DisruptionScenario:
        target = _last_stop_on_day(parent_plan, _busiest_day(parent_plan))
        day = _stop_day(target, default=_first_day(parent_plan))
        stop_id = _stop_id(target, fallback="closed_stop")
        return self._scenario(
            parent_plan,
            bundle,
            family=DisruptionFamily.ATTRACTION_CLOSURE,
            affected_days=(day,),
            target_stop_ids=(stop_id,),
            user_intent=f"Repair attraction closure for {stop_id}.",
            tolerance_profile={"max_daily_travel_minutes": self.max_daily_travel_minutes},
            confirmed_constraints={
                "must_delete": (stop_id,),
                "closure_status_by_stop": {stop_id: "closed"},
            },
            candidate_pois=(
                _replacement_candidate(
                    parent_plan,
                    family="closure_backup",
                    day=day,
                    candidate_pool=self.candidate_pool,
                ),
            ),
        )

    def reduced_driving_tolerance(self, parent_plan: PlanArtifactV2, bundle: Any | None = None) -> DisruptionScenario:
        day = _busiest_day(parent_plan)
        return self._scenario(
            parent_plan,
            bundle,
            family=DisruptionFamily.REDUCED_DRIVING_TOLERANCE,
            affected_days=(day,),
            target_stop_ids=_stop_ids_for_day(parent_plan, day),
            user_intent="Repair after the traveler reduces daily driving tolerance.",
            tolerance_profile={"max_daily_travel_minutes": 180.0},
            confirmed_constraints={
                "reduced_driving_tolerance_minutes": 180.0,
                "pace_accessibility": "reduced_driving_tolerance",
            },
        )

    def new_must_visit(self, parent_plan: PlanArtifactV2, bundle: Any | None = None) -> DisruptionScenario:
        day = _busiest_day(parent_plan)
        candidate = _replacement_candidate(
            parent_plan,
            family="must_visit",
            day=day,
            candidate_pool=self.candidate_pool,
        )
        must_visit_id = str(candidate["stop_id"])
        return self._scenario(
            parent_plan,
            bundle,
            family=DisruptionFamily.NEW_MUST_VISIT,
            affected_days=(day,),
            target_stop_ids=(must_visit_id,),
            user_intent=f"Repair route to include new must-visit {must_visit_id}.",
            tolerance_profile={"max_daily_travel_minutes": self.max_daily_travel_minutes},
            confirmed_constraints={
                "must_include": (must_visit_id,),
                "target_days": {must_visit_id: day},
            },
            candidate_pois=(candidate,),
        )

    def _scenario(
        self,
        parent_plan: PlanArtifactV2,
        bundle: Any | None,
        *,
        family: DisruptionFamily,
        affected_days: tuple[int, ...],
        target_stop_ids: tuple[str, ...],
        user_intent: str,
        tolerance_profile: dict[str, float],
        confirmed_constraints: dict[str, Any],
        candidate_pois: tuple[dict[str, Any], ...] = (),
    ) -> DisruptionScenario:
        catalog_snapshot_id, context_snapshot_id = _snapshot_ids(parent_plan, bundle)
        clean_days = tuple(sorted({int(day) for day in affected_days if day is not None}))
        clean_targets = tuple(str(target) for target in target_stop_ids if str(target).strip())
        scenario_id = _scenario_id(
            parent_plan=parent_plan,
            family=family,
            seed=self.seed,
            affected_days=clean_days,
            target_stop_ids=clean_targets,
            catalog_snapshot_id=catalog_snapshot_id,
            context_snapshot_id=context_snapshot_id,
        )
        base_constraints = {
            "benchmark": "bench-001",
            "disruption_id": scenario_id,
            "disruption_family": family.value,
            "parent_plan_id": parent_plan.plan_id,
            "catalog_snapshot_id": catalog_snapshot_id,
            "context_snapshot_id": context_snapshot_id,
            "observation_status": self.evidence_status,
            "observed": self.evidence_status == "observed",
            "synthetic": self.evidence_status == "synthetic",
            "seed": self.seed,
            "affected_days": clean_days,
            "target_stop_ids": clean_targets,
        }
        merged_constraints = {**base_constraints, **confirmed_constraints}
        request = RepairRequest(
            request_id=scenario_id,
            baseline_route=tuple(dict(stop) for stop in parent_plan.selected_stops),
            user_intent=user_intent,
            evidence_records=(
                {
                    "evidence_id": f"evidence_{scenario_id}",
                    "source_type": "benchmark_disruption",
                    "source_status": self.evidence_status,
                    "observed": self.evidence_status == "observed",
                    "synthetic": self.evidence_status == "synthetic",
                    "disruption_family": family.value,
                    "target_stop_ids": list(clean_targets),
                    "affected_days": list(clean_days),
                    "schema_version": "benchmark-disruption-evidence-v1",
                },
            ),
            tolerance_profile=dict(tolerance_profile),
            confirmed_constraints=merged_constraints,
            candidate_pois=tuple(dict(candidate) for candidate in candidate_pois),
        )
        return DisruptionScenario(
            scenario_id=scenario_id,
            family=family,
            parent_plan_id=parent_plan.plan_id,
            catalog_snapshot_id=catalog_snapshot_id,
            context_snapshot_id=context_snapshot_id,
            seed=self.seed,
            evidence_status=self.evidence_status,
            affected_days=clean_days,
            target_stop_ids=clean_targets,
            request=request,
        )


def generate_disruption_scenarios(
    parent_plan: PlanArtifactV2,
    bundle: Any | None = None,
    *,
    seed: int = 0,
    evidence_status: EvidenceStatus = "synthetic",
    candidate_pool: tuple[dict[str, Any], ...] = (),
    max_daily_travel_minutes: float = 360.0,
) -> tuple[DisruptionScenario, ...]:
    """Generate the six deterministic BENCH-001 scenarios."""

    return DisruptionGenerator(
        seed=seed,
        evidence_status=evidence_status,
        candidate_pool=candidate_pool,
        max_daily_travel_minutes=max_daily_travel_minutes,
    ).generate(parent_plan, bundle)


def generate_disruption_requests(
    parent_plan: PlanArtifactV2,
    bundle: Any | None = None,
    *,
    seed: int = 0,
    evidence_status: EvidenceStatus = "synthetic",
    candidate_pool: tuple[dict[str, Any], ...] = (),
    max_daily_travel_minutes: float = 360.0,
) -> tuple[RepairRequest, ...]:
    """Generate repair requests for the six deterministic BENCH-001 scenarios."""

    return tuple(
        scenario.request
        for scenario in generate_disruption_scenarios(
            parent_plan,
            bundle,
            seed=seed,
            evidence_status=evidence_status,
            candidate_pool=candidate_pool,
            max_daily_travel_minutes=max_daily_travel_minutes,
        )
    )


def _repair_request_record(request: RepairRequest) -> dict[str, Any]:
    return {
        "request_id": request.request_id,
        "baseline_route": [dict(stop) for stop in request.baseline_route],
        "user_intent": request.user_intent,
        "evidence_records": [dict(record) for record in request.evidence_records],
        "tolerance_profile": dict(request.tolerance_profile),
        "confirmed_constraints": _jsonable(request.confirmed_constraints),
        "candidate_pois": [dict(candidate) for candidate in request.candidate_pois],
        "travel_graph": dict(request.travel_graph),
    }


def _scenario_id(
    *,
    parent_plan: PlanArtifactV2,
    family: DisruptionFamily,
    seed: int,
    affected_days: tuple[int, ...],
    target_stop_ids: tuple[str, ...],
    catalog_snapshot_id: str,
    context_snapshot_id: str,
) -> str:
    digest = stable_content_hash(
        {
            "parent_plan_id": parent_plan.plan_id,
            "catalog_snapshot_id": catalog_snapshot_id,
            "context_snapshot_id": context_snapshot_id,
            "family": family.value,
            "seed": seed,
            "affected_days": affected_days,
            "target_stop_ids": target_stop_ids,
        }
    )
    return f"bench_{family.value}_{digest[:12]}"


def _snapshot_ids(parent_plan: PlanArtifactV2, bundle: Any | None) -> tuple[str, str]:
    catalog_snapshot_id = str(getattr(bundle, "catalog_snapshot_id", "") or parent_plan.catalog_snapshot_id)
    context_snapshot_id = str(getattr(bundle, "context_snapshot_id", "") or parent_plan.context_snapshot_id)
    return catalog_snapshot_id, context_snapshot_id


def _highest_weather_stop(parent_plan: PlanArtifactV2) -> dict[str, Any]:
    return max(
        _stops(parent_plan),
        key=lambda stop: (
            _float(stop.get("weather_sensitivity")) + _float(stop.get("weather_risk")),
            _stop_day(stop, default=0),
            str(stop.get("stop_id") or stop.get("name") or ""),
        ),
    )


def _last_stop_on_day(parent_plan: PlanArtifactV2, day: int) -> dict[str, Any]:
    stops = [stop for stop in _stops(parent_plan) if _stop_day(stop, default=day) == day]
    return max(stops or _stops(parent_plan), key=lambda stop: (_int(stop.get("stop_order")), _stop_id(stop, fallback="")))


def _busiest_day(parent_plan: PlanArtifactV2) -> int:
    counts: dict[int, int] = {}
    for stop in _stops(parent_plan):
        day = _stop_day(stop, default=_first_day(parent_plan))
        counts[day] = counts.get(day, 0) + 1
    if not counts:
        return 1
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def _first_day(parent_plan: PlanArtifactV2) -> int:
    days = [_stop_day(stop, default=1) for stop in _stops(parent_plan)]
    return min(days) if days else 1


def _stop_ids_for_day(parent_plan: PlanArtifactV2, day: int) -> tuple[str, ...]:
    return tuple(
        _stop_id(stop, fallback=f"day_{day}_stop_{index}")
        for index, stop in enumerate(_stops(parent_plan), start=1)
        if _stop_day(stop, default=day) == day
    )


def _lodging_for_day(parent_plan: PlanArtifactV2, day: int) -> str:
    lodging = parent_plan.lodging_assignments.get(str(day)) or parent_plan.lodging_assignments.get(day)  # type: ignore[arg-type]
    if lodging:
        return str(lodging)
    for stop in _stops(parent_plan):
        if _stop_day(stop, default=day) == day and stop.get("lodging_id"):
            return str(stop["lodging_id"])
    return f"lodging_day_{day}"


def _replacement_candidate(
    parent_plan: PlanArtifactV2,
    *,
    family: str,
    day: int,
    candidate_pool: tuple[dict[str, Any], ...] = (),
) -> dict[str, Any]:
    selected_ids = {_stop_id(stop, fallback="") for stop in _stops(parent_plan)}
    target_cities = {
        str(stop.get("city") or "").strip().lower()
        for stop in _stops(parent_plan)
        if _stop_day(stop, default=day) == day and str(stop.get("city") or "").strip()
    }
    all_located = [
        dict(candidate)
        for candidate in candidate_pool
        if _stop_id(candidate, fallback="") not in selected_ids and _has_coordinates(candidate)
    ]
    same_city = [
        candidate
        for candidate in all_located
        if str(candidate.get("city") or "").strip().lower() in target_cities
    ]
    located = same_city or all_located
    target_points = [
        (float(stop["latitude"]), float(stop["longitude"]))
        for stop in _stops(parent_plan)
        if _stop_day(stop, default=day) == day and _has_coordinates(stop)
    ]
    target_center = (
        (
            sum(latitude for latitude, _ in target_points) / len(target_points),
            sum(longitude for _, longitude in target_points) / len(target_points),
        )
        if target_points
        else None
    )
    located.sort(
        key=lambda candidate: (
            0.0 if same_city else _coordinate_distance_squared(candidate, target_center),
            -_float(candidate.get("final_poi_value")),
            _stop_id(candidate, fallback=""),
        )
    )
    rank = 1 if family == "must_visit" else 0
    if located:
        candidate = dict(located[min(rank, len(located) - 1)])
        candidate["stop_id"] = _stop_id(candidate, fallback=f"bench_{family}_day_{day}")
        candidate.setdefault("name", candidate["stop_id"])
        candidate["day"] = day
        candidate["stop_order"] = len(_stop_ids_for_day(parent_plan, day)) + 1
        candidate.setdefault("visit_duration_minutes", 45)
        candidate.setdefault("weather_risk", 0.1)
        candidate["benchmark_synthetic"] = False
        candidate["benchmark_candidate_source"] = "frozen_catalog"
        return candidate
    return {
        "stop_id": f"bench_{family}_day_{day}",
        "name": f"Benchmark {family.replace('_', ' ')} day {day}",
        "day": day,
        "stop_order": len(_stop_ids_for_day(parent_plan, day)) + 1,
        "visit_duration_minutes": 45,
        "weather_risk": 0.1,
        "benchmark_synthetic": True,
        "benchmark_candidate_source": "synthetic_fallback",
    }


def _coordinate_distance_squared(
    record: dict[str, Any],
    target: tuple[float, float] | None,
) -> float:
    if target is None:
        return 0.0
    return (float(record["latitude"]) - target[0]) ** 2 + (float(record["longitude"]) - target[1]) ** 2


def _has_coordinates(record: dict[str, Any]) -> bool:
    try:
        return math.isfinite(float(record.get("latitude"))) and math.isfinite(float(record.get("longitude")))
    except (TypeError, ValueError):
        return False


def _stops(parent_plan: PlanArtifactV2) -> tuple[dict[str, Any], ...]:
    return tuple(dict(stop) for stop in parent_plan.selected_stops)


def _stop_id(stop: dict[str, Any], *, fallback: str) -> str:
    for key in ("stop_id", "poi_id", "attraction_id", "name"):
        value = stop.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return fallback


def _stop_day(stop: dict[str, Any], *, default: int) -> int:
    return _int(stop.get("day"), default=default)


def _float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value
