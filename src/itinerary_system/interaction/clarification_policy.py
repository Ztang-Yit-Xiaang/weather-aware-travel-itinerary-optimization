"""Deterministic clarification policy; future learned policies may replace only this seam."""

from __future__ import annotations

from ..research_artifacts import stable_content_hash
from .consequence import (
    consequences_materially_different,
    equivalent_typed_repairs,
    is_low_consequence,
)
from .models import (
    ClarificationAction,
    ClarificationDecision,
    ConsequenceThresholds,
    ConsequenceVector,
    CounterfactualProbeResult,
    CriticalTradeoff,
    ProbeStatus,
    SemanticInterpretationCandidate,
)
from .tradeoff_selector import CriticalTradeoffSelector


def decide_clarification(
    *,
    candidates: tuple[SemanticInterpretationCandidate, ...],
    probe_results: tuple[CounterfactualProbeResult, ...],
    consequences: tuple[ConsequenceVector, ...],
    thresholds: ConsequenceThresholds,
    question_count: int,
    max_questions: int,
    tradeoff: CriticalTradeoff | None = None,
) -> ClarificationDecision:
    session_id = candidates[0].repair_session_id if candidates else "unknown_session"
    valid_ids = {item.interpretation_id for item in consequences}
    valid_candidates = tuple(item for item in candidates if item.interpretation_id in valid_ids)
    if not valid_candidates:
        return _decision(session_id, ClarificationAction.DEFER, None, tradeoff, None, ("no_valid_interpretation",))

    permission_required = any(result.requires_user_permission for result in probe_results)
    feasibility_differs = len({vector.hard_feasible for vector in consequences}) > 1
    materially_different = consequences_materially_different(consequences, thresholds)
    equivalent_low = equivalent_typed_repairs(probe_results) and all(
        is_low_consequence(vector, thresholds) for vector in consequences
    )
    unresolved_probe = any(
        result.status in {ProbeStatus.TIME_LIMIT_NO_INCUMBENT, ProbeStatus.FAILED} for result in probe_results
    )

    selected = sorted(valid_candidates, key=lambda item: item.interpretation_id)[0]
    if equivalent_low:
        return _decision(
            session_id,
            ClarificationAction.COMMIT,
            selected.interpretation_id,
            tradeoff,
            None,
            ("interpretation_consequences_equivalent", "low_consequence_commit"),
        )
    if question_count >= max_questions and (permission_required or feasibility_differs or materially_different):
        return _decision(
            session_id,
            ClarificationAction.DEFER,
            None,
            tradeoff,
            None,
            ("question_budget_exhausted",),
        )
    selector = CriticalTradeoffSelector()
    if permission_required:
        question = (
            selector.question(tradeoff, permission=True)
            if tradeoff
            else "May this repair relax the identified permission-gated commitment?"
        )
        return _decision(
            session_id,
            ClarificationAction.ASK_PERMISSION,
            _permission_interpretation(probe_results),
            tradeoff,
            question,
            ("booked_lodging_permission_required",),
        )
    if feasibility_differs:
        question = (
            selector.question(tradeoff, permission=False)
            if tradeoff
            else "Which feasible interpretation should this repair use?"
        )
        return _decision(
            session_id,
            ClarificationAction.ASK_SEMANTIC,
            None,
            tradeoff,
            question,
            ("feasibility_outcomes_differ",),
        )
    if materially_different:
        question = (
            selector.question(tradeoff, permission=False)
            if tradeoff
            else "Which repair consequence should the system preserve?"
        )
        return _decision(
            session_id,
            ClarificationAction.ASK_SEMANTIC,
            None,
            tradeoff,
            question,
            ("repair_scope_differs",),
        )
    if unresolved_probe:
        return _decision(
            session_id,
            ClarificationAction.PROBE_MORE,
            None,
            tradeoff,
            None,
            ("bounded_probe_inconclusive",),
        )
    selected_vector = next(item for item in consequences if item.interpretation_id == selected.interpretation_id)
    if is_low_consequence(selected_vector, thresholds) and (selected.confirmed or selected.evidence_refs):
        return _decision(
            session_id,
            ClarificationAction.COMMIT,
            selected.interpretation_id,
            tradeoff,
            None,
            ("low_consequence_commit",),
        )
    return _decision(session_id, ClarificationAction.DEFER, None, tradeoff, None, ("no_safe_auto_commit",))


def _decision(
    session_id: str,
    action: ClarificationAction,
    selected_interpretation_id: str | None,
    tradeoff: CriticalTradeoff | None,
    question: str | None,
    reason_codes: tuple[str, ...],
) -> ClarificationDecision:
    evidence_refs = tradeoff.evidence_refs if tradeoff else ()
    seed = {
        "session": session_id,
        "action": action.value,
        "selected": selected_interpretation_id,
        "tradeoff": tradeoff.tradeoff_id if tradeoff else None,
        "reason_codes": reason_codes,
    }
    return ClarificationDecision(
        decision_id=f"clarification_{stable_content_hash(seed)}",
        repair_session_id=session_id,
        action=action,
        selected_interpretation_id=selected_interpretation_id,
        selected_tradeoff_id=tradeoff.tradeoff_id if tradeoff else None,
        question_text=question,
        reason_codes=reason_codes,
        evidence_refs=evidence_refs,
    )


def _permission_interpretation(results: tuple[CounterfactualProbeResult, ...]) -> str | None:
    for result in results:
        if result.requires_user_permission:
            return result.interpretation_id
    return None
