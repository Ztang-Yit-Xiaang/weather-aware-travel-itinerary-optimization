from __future__ import annotations

import pytest

from itinerary_system.interaction.models import (
    CounterfactualProbeResult,
    ModelPatch,
    PermissionDecisionAction,
    ProbeStatus,
    UserPermissionDecision,
)
from itinerary_system.interaction.permission_policy import ConstraintPermissionClass, PermissionPolicy
from itinerary_system.plans import (
    ConstraintOrigin,
    ConstraintScope,
    ConstraintStrength,
    OwnedConstraint,
    RelaxationPolicy,
)
from itinerary_system.research_artifacts import PlanArtifactV2


def constraint(
    constraint_id: str,
    *,
    strength: ConstraintStrength,
    scope: ConstraintScope = ConstraintScope.STOP,
    relation: str = "preserve",
    origin: ConstraintOrigin = ConstraintOrigin.USER,
    relaxation: RelaxationPolicy = RelaxationPolicy.AUTO_WITH_PENALTY,
    confirmed: bool = True,
) -> OwnedConstraint:
    return OwnedConstraint(
        constraint_id=constraint_id,
        origin=origin,
        strength=strength,
        scope=scope,
        target_id=f"target_{constraint_id}",
        relation=relation,
        value=True,
        confirmed=confirmed,
        relaxation_policy=relaxation,
        evidence_refs=(f"evidence:{constraint_id}",),
    )


def parent() -> PlanArtifactV2:
    items = (
        constraint("physical", strength=ConstraintStrength.HARD, relation="physical_impossibility"),
        constraint("road", strength=ConstraintStrength.HARD, scope=ConstraintScope.ROAD),
        constraint("locked", strength=ConstraintStrength.LOCKED, relaxation=RelaxationPolicy.NEVER),
        constraint(
            "booked",
            strength=ConstraintStrength.BOOKED,
            scope=ConstraintScope.LODGING,
            origin=ConstraintOrigin.BOOKING,
            relaxation=RelaxationPolicy.EXPLICIT_ONLY,
        ),
        constraint("flexible", strength=ConstraintStrength.SOFT),
        constraint(
            "llm_unconfirmed",
            strength=ConstraintStrength.SOFT,
            origin=ConstraintOrigin.LLM_INTERPRETATION,
            confirmed=False,
        ),
    )
    return PlanArtifactV2(
        plan_id="parent_policy",
        source_run_id="run_parent",
        planning_request_id="request_parent",
        catalog_snapshot_id="catalog",
        context_snapshot_id="context",
        selected_stops=tuple({"stop_id": item.target_id, "day": 1, "name": item.target_id} for item in items),
        sequence=tuple(item.target_id for item in items),
        owned_constraints=tuple(item.to_record() for item in items),
    )


def patch(constraint_id: str) -> ModelPatch:
    return ModelPatch(
        patch_id=f"patch_{constraint_id}",
        interpretation_id=f"interpretation_{constraint_id}",
        patch_type="same_day_replacement",
        target_ids=(f"target_{constraint_id}",),
        parameters={},
        affected_constraint_ids=(constraint_id,),
        validation_status="valid",
        evidence_refs=(f"evidence:{constraint_id}",),
    )


def test_never_locked_booked_and_flexible_policy_boundaries() -> None:
    policy = PermissionPolicy()
    constraints = {item["constraint_id"]: OwnedConstraint.from_record(item) for item in parent().owned_constraints}
    assert policy.classify(constraints["physical"]) == ConstraintPermissionClass.NEVER
    assert policy.classify(constraints["road"]) == ConstraintPermissionClass.NEVER
    assert policy.classify(constraints["locked"]) == ConstraintPermissionClass.LOCKED
    assert policy.classify(constraints["booked"]) == ConstraintPermissionClass.PERMISSION_GATED
    assert policy.classify(constraints["flexible"]) == ConstraintPermissionClass.FLEXIBLE
    assert policy.classify(constraints["llm_unconfirmed"]) == ConstraintPermissionClass.INACTIVE

    physical = policy.assess_patch(parent(), patch("physical"), repair_session_id="session")
    locked = policy.assess_patch(parent(), patch("locked"), repair_session_id="session")
    booked = policy.assess_patch(parent(), patch("booked"), repair_session_id="session")
    flexible = policy.assess_patch(parent(), patch("flexible"), repair_session_id="session")
    assert physical.allowed_for_probe is False
    assert locked.allowed_for_probe is False
    assert booked.allowed_for_probe is True
    assert booked.allowed_for_authorized_repair is False
    assert booked.requires_user_permission is True
    assert flexible.allowed_for_authorized_repair is True


def test_booked_change_requires_session_scoped_permission() -> None:
    decision = UserPermissionDecision(
        permission_decision_id="permission_once",
        repair_session_id="session",
        constraint_ids=("booked",),
        action=PermissionDecisionAction.GRANT_ONCE,
        selected_interpretation_id="interpretation_booked",
        created_at="2026-07-22T00:00:00+00:00",
        evidence_refs=("user_decision:permission_once",),
    )
    policy = PermissionPolicy()
    allowed = policy.assess_patch(
        parent(),
        patch("booked"),
        repair_session_id="session",
        permission_decisions=(decision,),
    )
    wrong_session = policy.assess_patch(
        parent(),
        patch("booked"),
        repair_session_id="other_session",
        permission_decisions=(decision,),
    )
    assert allowed.allowed_for_authorized_repair is True
    assert wrong_session.allowed_for_authorized_repair is False


def test_probe_result_cannot_be_execution_eligible() -> None:
    with pytest.raises(ValueError, match="never be execution-eligible"):
        CounterfactualProbeResult(
            probe_result_id="result",
            probe_request_id="request",
            parent_plan_id="parent",
            hypothetical_plan_id="hypothetical",
            status=ProbeStatus.FEASIBLE_BOUNDED,
            diff_id="diff",
            solver_run_ids=("solver",),
            requires_user_permission=False,
            permission_constraint_ids=(),
            eligible_for_execution=True,
            evidence_refs=("solver:solver",),
        )
