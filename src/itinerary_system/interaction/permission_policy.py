"""Fail-closed permission and probe policy over existing OwnedConstraint records."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from ..plans import (
    ConstraintOrigin,
    ConstraintScope,
    ConstraintStrength,
    OwnedConstraint,
    RelaxationPolicy,
    active_owned_constraints,
)
from ..research_artifacts import PlanArtifactV2
from .models import ModelPatch, PermissionDecisionAction, UserPermissionDecision


class ConstraintPermissionClass(StrEnum):
    NEVER = "never"
    LOCKED = "locked"
    PERMISSION_GATED = "permission_gated"
    FLEXIBLE = "flexible"
    INACTIVE = "inactive"


@dataclass(frozen=True)
class PatchPermissionAssessment:
    allowed_for_probe: bool
    allowed_for_authorized_repair: bool
    requires_user_permission: bool
    permission_constraint_ids: tuple[str, ...]
    blocked_constraint_ids: tuple[str, ...]
    reason_codes: tuple[str, ...]


_NEVER_RELATION_MARKERS = (
    "physical",
    "chronology",
    "continuity",
    "route_valid",
    "road_closure",
    "safety",
    "accessibility_hard",
    "nonexistent",
    "opening_window_arithmetic",
    "visit_duration_arithmetic",
    "artifact_lineage",
    "content_hash",
)


class PermissionPolicy:
    def __init__(self, *, allow_locked_explanation_probe: bool = False) -> None:
        self.allow_locked_explanation_probe = allow_locked_explanation_probe

    def classify(self, constraint: OwnedConstraint) -> ConstraintPermissionClass:
        if not constraint.is_active:
            return ConstraintPermissionClass.INACTIVE
        relation = constraint.relation.lower().replace("-", "_").replace(" ", "_")
        if (
            constraint.scope in {ConstraintScope.ROUTE, ConstraintScope.ROAD}
            or any(marker in relation for marker in _NEVER_RELATION_MARKERS)
            or constraint.origin in {ConstraintOrigin.SYSTEM, ConstraintOrigin.EXTERNAL_DATA}
            and constraint.strength == ConstraintStrength.HARD
        ):
            return ConstraintPermissionClass.NEVER
        if constraint.strength == ConstraintStrength.LOCKED:
            return ConstraintPermissionClass.LOCKED
        if (
            constraint.strength == ConstraintStrength.BOOKED
            or constraint.relaxation_policy == RelaxationPolicy.EXPLICIT_ONLY
            or constraint.origin in {ConstraintOrigin.BOOKING, ConstraintOrigin.USER_BOOKING}
        ):
            return ConstraintPermissionClass.PERMISSION_GATED
        if constraint.relaxation_policy == RelaxationPolicy.NEVER or constraint.strength == ConstraintStrength.HARD:
            return ConstraintPermissionClass.NEVER
        return ConstraintPermissionClass.FLEXIBLE

    def assess_patch(
        self,
        parent: PlanArtifactV2,
        patch: ModelPatch,
        *,
        permission_decisions: tuple[UserPermissionDecision, ...] = (),
        repair_session_id: str,
    ) -> PatchPermissionAssessment:
        constraints = {
            item.constraint_id: item
            for item in active_owned_constraints(tuple(dict(record) for record in parent.owned_constraints))
        }
        permission_ids: list[str] = []
        blocked_ids: list[str] = []
        locked_ids: list[str] = []
        for constraint_id in patch.affected_constraint_ids:
            constraint = constraints.get(constraint_id)
            if constraint is None:
                blocked_ids.append(constraint_id)
                continue
            classification = self.classify(constraint)
            if classification == ConstraintPermissionClass.NEVER:
                blocked_ids.append(constraint_id)
            elif classification == ConstraintPermissionClass.LOCKED:
                locked_ids.append(constraint_id)
            elif classification == ConstraintPermissionClass.PERMISSION_GATED:
                permission_ids.append(constraint_id)

        granted, denied = granted_and_denied_constraint_ids(
            permission_decisions,
            repair_session_id=repair_session_id,
            interpretation_id=patch.interpretation_id,
        )
        if denied.intersection(permission_ids):
            blocked_ids.extend(sorted(denied.intersection(permission_ids)))
        probe_blocked = bool(blocked_ids) or bool(locked_ids and not self.allow_locked_explanation_probe)
        unresolved_permission = set(permission_ids) - granted
        authorized_blocked = probe_blocked or bool(unresolved_permission)
        reasons: list[str] = []
        if blocked_ids:
            reasons.append("non_relaxable_constraint_affected")
        if locked_ids:
            reasons.append("locked_constraint_affected")
        if unresolved_permission:
            reasons.append("user_permission_required")
        if denied.intersection(permission_ids):
            reasons.append("user_permission_denied")
        return PatchPermissionAssessment(
            allowed_for_probe=not probe_blocked,
            allowed_for_authorized_repair=not authorized_blocked,
            requires_user_permission=bool(unresolved_permission),
            permission_constraint_ids=tuple(permission_ids),
            blocked_constraint_ids=tuple(dict.fromkeys((*blocked_ids, *locked_ids))),
            reason_codes=tuple(dict.fromkeys(reasons)),
        )


def granted_and_denied_constraint_ids(
    decisions: tuple[UserPermissionDecision, ...],
    *,
    repair_session_id: str,
    interpretation_id: str,
) -> tuple[set[str], set[str]]:
    granted: set[str] = set()
    denied: set[str] = set()
    for decision in decisions:
        if decision.repair_session_id != repair_session_id:
            continue
        if decision.selected_interpretation_id not in {None, interpretation_id}:
            continue
        ids = set(decision.constraint_ids)
        if decision.action in {PermissionDecisionAction.GRANT, PermissionDecisionAction.GRANT_ONCE}:
            granted.update(ids)
            denied.difference_update(ids)
        elif decision.action == PermissionDecisionAction.DENY:
            denied.update(ids)
            granted.difference_update(ids)
    return granted, denied
