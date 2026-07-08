"""Canonical plan ownership, repository, and diff APIs."""

from .diff import PlanDiffBuilder, compute_plan_diff
from .models import (
    ConstraintOrigin,
    ConstraintScope,
    ConstraintStrength,
    DayMove,
    LodgingChange,
    OrderChange,
    OwnedConstraint,
    OwnershipPolicy,
    PlanDiff,
    RelaxationPolicy,
    RoadChange,
    StopChange,
    TimeShift,
    active_owned_constraints,
    plan_from_phase0_route_stops,
    validate_owned_constraints,
)
from .repository import PlanNotFound, PlanRepository, PlanRepositoryConflict, load_plan, save_plan_append_only

__all__ = [
    "ConstraintOrigin",
    "ConstraintScope",
    "ConstraintStrength",
    "DayMove",
    "LodgingChange",
    "OrderChange",
    "OwnedConstraint",
    "OwnershipPolicy",
    "PlanDiff",
    "PlanDiffBuilder",
    "PlanNotFound",
    "PlanRepository",
    "PlanRepositoryConflict",
    "RelaxationPolicy",
    "RoadChange",
    "StopChange",
    "TimeShift",
    "active_owned_constraints",
    "compute_plan_diff",
    "load_plan",
    "plan_from_phase0_route_stops",
    "save_plan_append_only",
    "validate_owned_constraints",
]
