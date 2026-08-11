from __future__ import annotations

from dataclasses import replace

from itinerary_system.interaction.counterfactual_probe import CounterfactualProbeExecutor
from itinerary_system.interaction.models import (
    CounterfactualProbeRequest,
    PermissionDecisionAction,
    UserPermissionDecision,
)
from itinerary_system.interaction.patch_compiler import AllowListedPatchCompiler
from itinerary_system.interaction.permission_policy import PermissionPolicy
from itinerary_system.interaction.semantic_candidates import RuleBasedSemanticCandidateProvider
from itinerary_system.plans import (
    ConstraintOrigin,
    ConstraintScope,
    ConstraintStrength,
    OwnedConstraint,
    RelaxationPolicy,
)
from itinerary_system.repair.neighborhood import RepairRadius
from itinerary_system.repair.progressive import RepairOutcome
from itinerary_system.research_artifacts import PlanArtifactV2


def parent_plan() -> PlanArtifactV2:
    booked = OwnedConstraint(
        constraint_id="booked_hotel",
        origin=ConstraintOrigin.BOOKING,
        strength=ConstraintStrength.BOOKED,
        scope=ConstraintScope.LODGING,
        target_id="hotel_a",
        relation="preserve_booking",
        value=True,
        confirmed=True,
        relaxation_policy=RelaxationPolicy.EXPLICIT_ONLY,
        evidence_refs=("booking:hotel_a",),
    )
    return PlanArtifactV2(
        plan_id="parent_probe",
        source_run_id="parent_run",
        planning_request_id="parent_request",
        catalog_snapshot_id="catalog",
        context_snapshot_id="context",
        selected_stops=(
            {"stop_id": "view", "name": "Viewpoint", "day": 1},
            {"stop_id": "hotel_a", "name": "Booked Hotel", "day": 1},
        ),
        sequence=("view", "hotel_a"),
        lodging_assignments={"1": "hotel_a"},
        owned_constraints=(booked.to_record(),),
    )


def test_probe_uses_test_only_repair_request_and_strips_execution_certificate() -> None:
    parent = parent_plan()
    candidate = RuleBasedSemanticCandidateProvider().candidates(
        parent=parent,
        user_edit="less driving",
        repair_session_id="session",
        evidence_refs=("user_edit:hash",),
    )[0]
    compiler = AllowListedPatchCompiler()
    patch = compiler.compile(parent, candidate)
    assessment = PermissionPolicy().assess_patch(parent, patch, repair_session_id="session")
    request = CounterfactualProbeRequest(
        probe_request_id="probe_request",
        repair_session_id="session",
        parent_plan_id=parent.plan_id,
        interpretation_id=candidate.interpretation_id,
        model_patch_id=patch.patch_id,
        allowed_probe_constraint_ids=(),
        repair_radius=RepairRadius.HOTEL_PRESERVING_REROUTE,
        time_limit_seconds=0.25,
    )
    captured = {}

    def run_repair(repair_request):
        captured["request"] = repair_request
        child = replace(
            parent,
            plan_id="probe_child",
            parent_plan_id=parent.plan_id,
            source_run_id="probe_solver",
            planning_request_id=repair_request.request_id,
            certificate_id="certificate_must_not_escape",
        )
        return RepairOutcome(
            repair_outcome_id="outcome",
            parent_plan_id=parent.plan_id,
            repair_request_id=repair_request.request_id,
            status="accepted",
            accepted_radius=RepairRadius.HOTEL_PRESERVING_REROUTE,
            attempts=(),
            planner_runs=(),
            child_plan=child,
            diff_record={"diff_id": "probe_diff", "weighted_edit_cost": 1.0},
            evaluation_record={"certificate_id": "certificate_must_not_escape", "eligible": True},
        )

    result = CounterfactualProbeExecutor(run_repair, compiler=compiler).execute(
        parent=parent,
        candidate=candidate,
        patch=patch,
        request=request,
        assessment=assessment,
    )
    repair_request = captured["request"]
    assert repair_request.confirmed_constraints["test_only"] is True
    assert repair_request.confirmed_constraints["allowed_radii"] == ("hotel_preserving_reroute",)
    assert repair_request.confirmed_constraints["solver_time_limit_seconds"] == 0.25
    assert result.eligible_for_execution is False
    assert result.hypothetical_plan_record["artifact_role"] == "hypothetical"
    assert result.hypothetical_plan_record["certificate_id"] is None
    assert "certificate_id" not in result.diagnostic_evaluation
    assert parent.certificate_id is None


def test_permission_denial_remains_scoped_and_blocks_authorized_change() -> None:
    parent = parent_plan()
    candidate = RuleBasedSemanticCandidateProvider().candidates(
        parent=parent,
        user_edit="keep Booked Hotel",
        repair_session_id="session",
        evidence_refs=("user_edit:hash",),
    )[0]
    compiler = AllowListedPatchCompiler()
    patch = compiler.compile(parent, candidate)
    denial = UserPermissionDecision(
        permission_decision_id="denial",
        repair_session_id="session",
        constraint_ids=("booked_hotel",),
        action=PermissionDecisionAction.DENY,
        selected_interpretation_id=candidate.interpretation_id,
        created_at="2026-07-22T12:00:00+00:00",
        evidence_refs=("user_decision:denial",),
    )
    assessment = PermissionPolicy().assess_patch(
        parent,
        patch,
        repair_session_id="session",
        permission_decisions=(denial,),
    )
    assert assessment.allowed_for_authorized_repair is False
    assert "user_permission_denied" in assessment.reason_codes
