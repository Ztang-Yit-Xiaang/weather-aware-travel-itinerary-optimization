"""Routing provider protocol types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from .matrix import RouteMatrix
from .models import RouteLegResult


@dataclass(frozen=True)
class RoutingActivation:
    """Provider readiness result."""

    provider_id: str
    available: bool
    status: str
    detail: str = ""


@dataclass(frozen=True)
class RouteMatrixRequest:
    """Request for a provider-backed route matrix."""

    context_snapshot_id: str
    entity_points: dict[str, tuple[float, float]]
    routing_profile: str = "driving"
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class RouteLegRequest:
    """Request for one provider-backed route leg."""

    origin_id: str
    destination_id: str
    origin_point: tuple[float, float]
    destination_point: tuple[float, float]
    routing_profile: str = "driving"


@dataclass(frozen=True)
class SnapResult:
    """Nearest-road snap diagnostics."""

    entity_id: str
    snapped_point: tuple[float, float] | None
    snap_distance_m: float | None
    provider: str
    status: str


class RoutingProvider(Protocol):
    """Protocol implemented by OSRM, cache, or demo routing providers."""

    provider_id: str

    def probe(self) -> RoutingActivation:
        """Return provider readiness without issuing normal route requests."""

    def table(self, request: RouteMatrixRequest) -> RouteMatrix:
        """Return a route matrix for the requested entities."""

    def route(self, request: RouteLegRequest) -> RouteLegResult:
        """Return one route leg with provenance."""

    def nearest(self, entity_id: str, point: tuple[float, float]) -> SnapResult:
        """Return nearest-road snap diagnostics for one point."""
