from __future__ import annotations

import pytest

from itinerary_system.interaction.clarification_policy import decide_clarification
from itinerary_system.interaction.consequence import (
    build_consequence_vector,
    consequences_materially_different,
    equivalent_typed_repairs,
    is_low_consequence,
)
from itinerary_system.interaction.models import (
    ClarificationAction,
    ConsequenceThresholds,
    ConsequenceVector,
    CounterfactualProbeResult,
    CriticalTradeoff,
    InteractionOptions,
    ProbeStatus,
    SemanticInterpretationCandidate,
)
from itinerary_system.research_artifacts import PlanArtifactV2


def candidate(identifier: str) -> SemanticInterpretationCandidate:
    return SemanticInterpretationCandidate(
        interpretation_id=identifier,
        repair_session_id="session",
        user_text_hash="hash",
        target_ids=("stop",),
        interpretation_type="same_day_replacement",
        normalized_parameters={},
        support_score=None,
        evidence_refs=(f"candidate:{identifier}",),
        confirmed=True,
    )


def result(identifier: str, *, diff: dict | None, permission: bool = False, feasible: bool = True):
    return CounterfactualProbeResult(
        probe_result_id=f"result_{identifier}",
        probe_request_id=f"request_{identifier}",
        parent_plan_id="parent",
        hypothetical_plan_id=f"hypothetical_{identifier}" if feasible else None,
        status=ProbeStatus.FEASIBLE_BOUNDED if feasible else ProbeStatus.INFEASIBLE,
        diff_id=diff.get("diff_id") if diff else None,
        solver_run_ids=(f"solver_{identifier}",),
        requires_user_permission=permission,
        permission_constraint_ids=("booked",) if permission else (),
        eligible_for_execution=False,
        evidence_refs=(f"probe:{identifier}",),
        interpretation_id=identifier,
        affected_constraint_ids=("booked",) if permission else (),
        diff_record=diff,
    )


def vector(
    identifier: str,
    *,
    feasible: bool = True,
    permission: int = 0,
    booked: int = 0,
    edit_cost: float | None = 2.0,
    affected_days: int = 1,
    radius: str = "same_day_replacement",
) -> ConsequenceVector:
    return ConsequenceVector(
        consequence_id=f"consequence_{identifier}",
        interpretation_id=identifier,
        probe_result_id=f"result_{identifier}",
        hard_feasible=feasible,
        permission_change_count=permission,
        locked_change_count=0,
        booked_change_count=booked,
        strong_change_count=0,
        flexible_change_count=1,
        core_commitment_changes=(),
        weighted_edit_cost=edit_cost,
        affected_day_count=affected_days,
        lodging_change_count=booked,
        road_change_count=0,
        travel_minutes_delta=None,
        monetary_cost_delta=None,
        walking_burden_delta=None,
        contextual_risk_delta=None,
        utility_delta=100.0,
        accepted_repair_radius=radius,
        evidence_refs=(f"probe:{identifier}",),
    )


def tradeoff() -> CriticalTradeoff:
    return CriticalTradeoff(
        tradeoff_id="tradeoff",
        repair_session_id="session",
        left_interpretation_id="left",
        right_interpretation_id="right",
        primary_dimension="permission",
        left_summary="keeps the booked lodging and adds 180 minutes of travel",
        right_summary="changes the lodging and adds 80.00 in cost",
        numerical_deltas={"travel_minutes_delta": -180.0, "monetary_cost_delta": 80.0},
        permission_required=True,
        evidence_refs=("probe:left", "probe:right", "route:matrix", "plan_diff:right"),
    )


def test_equivalent_language_interpretations_commit_without_question() -> None:
    diff = {
        "diff_id": "diff_same",
        "deleted_stops": ({"stop_id": "cafe", "day": 1, "cost": 2.0},),
        "weighted_edit_cost": 2.0,
    }
    results = (result("left", diff=diff), result("right", diff={**diff, "diff_id": "diff_other_id"}))
    consequences = (vector("left"), vector("right"))
    assert equivalent_typed_repairs(results)
    decision = decide_clarification(
        candidates=(candidate("left"), candidate("right")),
        probe_results=results,
        consequences=consequences,
        thresholds=ConsequenceThresholds(),
        question_count=0,
        max_questions=2,
    )
    assert decision.action == ClarificationAction.COMMIT
    assert decision.question_text is None


def test_semantic_and_permission_differences_ask_one_grounded_question() -> None:
    left = vector("left", affected_days=1)
    right = vector("right", affected_days=2, radius="adjacent_day_move")
    semantic = decide_clarification(
        candidates=(candidate("left"), candidate("right")),
        probe_results=(
            result("left", diff={"diff_id": "left", "deleted_stops": ()}),
            result("right", diff={"diff_id": "right", "day_moves": ({"stop_id": "stop", "from_day": 1, "to_day": 2},)}),
        ),
        consequences=(left, right),
        thresholds=ConsequenceThresholds(),
        question_count=0,
        max_questions=2,
        tradeoff=tradeoff(),
    )
    assert semantic.action == ClarificationAction.ASK_SEMANTIC
    assert semantic.question_text and "Option one" in semantic.question_text

    permission = decide_clarification(
        candidates=(candidate("left"), candidate("right")),
        probe_results=(
            result("left", diff={"diff_id": "left"}),
            result("right", diff={"diff_id": "right", "lodging_changes": ({"day": 1},)}, permission=True),
        ),
        consequences=(left, vector("right", permission=1, booked=1)),
        thresholds=ConsequenceThresholds(),
        question_count=0,
        max_questions=2,
        tradeoff=tradeoff(),
    )
    assert permission.action == ClarificationAction.ASK_PERMISSION
    assert permission.selected_interpretation_id == "right"
    assert permission.question_text and "permission-gated" in permission.question_text
    assert "probe:right" in permission.evidence_refs


def test_hierarchy_beats_utility_and_unknown_metrics_remain_unknown() -> None:
    thresholds = ConsequenceThresholds()
    flexible = vector("flexible", edit_cost=6.0)
    booked = vector("booked", permission=1, booked=1, edit_cost=1.0)
    assert consequences_materially_different((flexible, booked), thresholds)
    assert is_low_consequence(booked, thresholds) is False

    parent = PlanArtifactV2(
        plan_id="parent",
        source_run_id="run",
        planning_request_id="request",
        catalog_snapshot_id="catalog",
        context_snapshot_id="context",
        selected_stops=({"stop_id": "stop", "day": 1},),
        sequence=("stop",),
    )
    consequence = build_consequence_vector(
        parent=parent,
        interpretation_id="unknown_metrics",
        probe_result=result(
            "unknown_metrics",
            diff={"diff_id": "diff", "weighted_edit_cost": 1.0, "deleted_stops": ({"stop_id": "stop", "day": 1},)},
        ),
    )
    assert consequence.travel_minutes_delta is None
    assert consequence.monetary_cost_delta is None
    assert consequence.walking_burden_delta is None


def test_no_safe_option_and_question_budget_defer() -> None:
    no_candidate = decide_clarification(
        candidates=(),
        probe_results=(),
        consequences=(),
        thresholds=ConsequenceThresholds(),
        question_count=0,
        max_questions=2,
    )
    assert no_candidate.action == ClarificationAction.DEFER

    exhausted = decide_clarification(
        candidates=(candidate("left"), candidate("right")),
        probe_results=(
            result("left", diff={"diff_id": "left"}, feasible=True),
            result("right", diff={"diff_id": "right"}, feasible=False),
        ),
        consequences=(vector("left", feasible=True), vector("right", feasible=False)),
        thresholds=ConsequenceThresholds(),
        question_count=2,
        max_questions=2,
    )
    assert exhausted.action == ClarificationAction.DEFER
    assert "question_budget_exhausted" in exhausted.reason_codes


def test_missing_diff_is_not_equivalent_and_nonfinite_cost_is_not_low_consequence() -> None:
    results = (
        result("left", diff=None),
        result("right", diff={"diff_id": "right"}),
    )

    assert equivalent_typed_repairs(results) is False
    assert is_low_consequence(vector("nan", edit_cost=float("nan")), ConsequenceThresholds()) is False


def test_interaction_thresholds_and_options_reject_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="finite and nonnegative"):
        ConsequenceThresholds(max_low_consequence_edit_cost=float("nan"))
    with pytest.raises(ValueError, match="finite and positive"):
        InteractionOptions(probe_time_limit_seconds=float("inf"))