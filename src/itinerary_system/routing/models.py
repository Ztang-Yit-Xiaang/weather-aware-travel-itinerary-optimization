"""Routing result contracts with explicit fallback and road-validation fields."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any


def _stable_hash(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class RouteLegResult:
    """One movement between two anchors with provenance for every routing number."""

    origin_id: str
    destination_id: str
    geometry: tuple[tuple[float, float], ...] = ()
    distance_m: float | None = None
    duration_s: float | None = None
    routing_status: str = "unknown"
    provider: str = "unknown"
    routing_profile: str = "driving"
    geometry_source: str = "unknown"
    distance_source: str = "unknown"
    duration_source: str = "unknown"
    road_validated: bool = False
    fallback_used: bool = False
    fallback_reason: str | None = None
    query_hash: str = ""
    retrieved_at: datetime | None = None
    cache_created_at: datetime | None = None
    snap_distance_origin_m: float | None = None
    snap_distance_destination_m: float | None = None

    def __post_init__(self) -> None:
        if not self.query_hash:
            object.__setattr__(
                self,
                "query_hash",
                _stable_hash(
                    {
                        "origin_id": self.origin_id,
                        "destination_id": self.destination_id,
                        "geometry": self.geometry,
                        "routing_status": self.routing_status,
                        "provider": self.provider,
                    }
                ),
            )

    @classmethod
    def approximate_geodesic(
        cls,
        *,
        origin_id: str,
        destination_id: str,
        geometry: tuple[tuple[float, float], ...],
        distance_m: float | None = None,
        duration_s: float | None = None,
        fallback_reason: str = "road_routing_unavailable",
    ) -> "RouteLegResult":
        """Build an explicitly non-road-validated fallback leg."""

        return cls(
            origin_id=origin_id,
            destination_id=destination_id,
            geometry=geometry,
            distance_m=distance_m,
            duration_s=duration_s,
            routing_status="fallback_geodesic_proxy",
            provider="geodesic_proxy",
            geometry_source="straight_waypoints",
            distance_source="geodesic_proxy",
            duration_source="geodesic_speed_proxy",
            road_validated=False,
            fallback_used=True,
            fallback_reason=fallback_reason,
        )


@dataclass(frozen=True)
class RouteResult:
    """A route-level aggregate with road validation and evaluation eligibility."""

    route_id: str
    legs: tuple[RouteLegResult, ...]
    solver_feasible: bool = False
    schedule_feasible: bool = False
    dataset_snapshot_valid: bool = False

    @property
    def total_distance_m(self) -> float | None:
        values = [leg.distance_m for leg in self.legs]
        if not values or any(value is None for value in values):
            return None
        return float(sum(float(value) for value in values if value is not None))

    @property
    def total_duration_s(self) -> float | None:
        values = [leg.duration_s for leg in self.legs]
        if not values or any(value is None for value in values):
            return None
        return float(sum(float(value) for value in values if value is not None))

    @property
    def road_validated(self) -> bool:
        return bool(self.legs) and all(leg.road_validated for leg in self.legs)

    @property
    def fallback_used(self) -> bool:
        return any(leg.fallback_used for leg in self.legs)

    @property
    def evaluation_eligible(self) -> bool:
        return bool(self.solver_feasible and self.schedule_feasible and self.dataset_snapshot_valid and self.road_validated)

    def to_audit_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for index, leg in enumerate(self.legs, start=1):
            rows.append(
                {
                    "route_id": self.route_id,
                    "leg_index": index,
                    "origin_id": leg.origin_id,
                    "destination_id": leg.destination_id,
                    "routing_status": leg.routing_status,
                    "provider": leg.provider,
                    "routing_profile": leg.routing_profile,
                    "geometry_source": leg.geometry_source,
                    "distance_source": leg.distance_source,
                    "duration_source": leg.duration_source,
                    "distance_m": leg.distance_m,
                    "duration_s": leg.duration_s,
                    "road_validated": leg.road_validated,
                    "fallback_used": leg.fallback_used,
                    "fallback_reason": leg.fallback_reason,
                    "query_hash": leg.query_hash,
                }
            )
        return rows

