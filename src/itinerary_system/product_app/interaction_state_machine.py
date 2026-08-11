"""Truthful state transitions for direct itinerary interactions.

This module does not persist sessions or enable W5 decisions.  It centralizes
which UI/application transitions are legal so visual, route-checked, and
evaluated states cannot be confused by callers.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .interaction_contracts import FeedbackTierV1, InteractionStateV1


class InteractionTransitionError(ValueError):
    """Rejected state transition carrying only a stable code."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class InteractionActionV1(StrEnum):
    SELECT = "select"
    BEGIN_EDIT = "begin_edit"
    SHOW_GHOST = "show_ghost"
    SHOW_EXPLORATORY = "show_exploratory"
    REQUEST_SNAP = "request_snap"
    REQUIRE_CONFIRMATION = "require_confirmation"
    APPEND_DRAFT = "append_draft"
    COMPLETE_ROUTE_PREVIEW = "complete_route_preview"
    RUN_REPAIR = "run_repair"
    COMPLETE_EVALUATION = "complete_evaluation"
    ACCEPT = "accept"
    REJECT = "reject"
    REVISE = "revise"
    CANCEL = "cancel"
    FAIL = "fail"
    STALE_REVISION = "stale_revision"


@dataclass(frozen=True)
class InteractionTransitionV1:
    previous_state: InteractionStateV1
    action: InteractionActionV1
    state: InteractionStateV1
    feedback_tier: FeedbackTierV1
    certificate_visible: bool
    acceptance_allowed: bool
    code: str
    schema_version: str = "interaction-transition-v1"


_TRANSITIONS: dict[
    tuple[InteractionStateV1, InteractionActionV1], InteractionStateV1
] = {
    (InteractionStateV1.IDLE, InteractionActionV1.SELECT): InteractionStateV1.SELECTED,
    (InteractionStateV1.SELECTED, InteractionActionV1.BEGIN_EDIT): InteractionStateV1.EDITING,
    (InteractionStateV1.EDITING, InteractionActionV1.SHOW_GHOST): InteractionStateV1.GHOST_PREVIEW,
    (InteractionStateV1.EDITING, InteractionActionV1.SHOW_EXPLORATORY): InteractionStateV1.EXPLORATORY_PREVIEW,
    (InteractionStateV1.GHOST_PREVIEW, InteractionActionV1.REQUEST_SNAP): InteractionStateV1.SNAP_PREVIEW,
    (InteractionStateV1.EXPLORATORY_PREVIEW, InteractionActionV1.REQUEST_SNAP): InteractionStateV1.SNAP_PREVIEW,
    (InteractionStateV1.SNAP_PREVIEW, InteractionActionV1.REQUIRE_CONFIRMATION): InteractionStateV1.PENDING_CONFIRMATION,
    (InteractionStateV1.SNAP_PREVIEW, InteractionActionV1.APPEND_DRAFT): InteractionStateV1.DRAFT_APPENDED,
    (InteractionStateV1.PENDING_CONFIRMATION, InteractionActionV1.APPEND_DRAFT): InteractionStateV1.DRAFT_APPENDED,
    (InteractionStateV1.DRAFT_APPENDED, InteractionActionV1.COMPLETE_ROUTE_PREVIEW): InteractionStateV1.ROUTE_CHECKED,
    (InteractionStateV1.ROUTE_CHECKED, InteractionActionV1.RUN_REPAIR): InteractionStateV1.REPAIR_RUNNING,
    (InteractionStateV1.REPAIR_RUNNING, InteractionActionV1.COMPLETE_EVALUATION): InteractionStateV1.EVALUATED,
    (InteractionStateV1.EVALUATED, InteractionActionV1.ACCEPT): InteractionStateV1.ACCEPTED,
    (InteractionStateV1.EVALUATED, InteractionActionV1.REJECT): InteractionStateV1.REJECTED,
    (InteractionStateV1.EVALUATED, InteractionActionV1.REVISE): InteractionStateV1.EDITING,
    (InteractionStateV1.REJECTED, InteractionActionV1.REVISE): InteractionStateV1.EDITING,
}

_CANCELLABLE = frozenset(
    {
        InteractionStateV1.SELECTED,
        InteractionStateV1.EDITING,
        InteractionStateV1.GHOST_PREVIEW,
        InteractionStateV1.EXPLORATORY_PREVIEW,
        InteractionStateV1.SNAP_PREVIEW,
        InteractionStateV1.PENDING_CONFIRMATION,
        InteractionStateV1.DRAFT_APPENDED,
        InteractionStateV1.ROUTE_CHECKED,
        InteractionStateV1.REJECTED,
        InteractionStateV1.FAILED,
    }
)

_FAILABLE = frozenset(state for state in InteractionStateV1 if state not in {InteractionStateV1.ACCEPTED})


def transition_interaction(
    state: InteractionStateV1,
    action: InteractionActionV1,
    *,
    evaluation_eligible: bool = False,
    certificate_fresh: bool = False,
) -> InteractionTransitionV1:
    """Apply one legal transition without mutating external session state."""

    if action is InteractionActionV1.STALE_REVISION:
        if state is InteractionStateV1.ACCEPTED:
            raise InteractionTransitionError("terminal_interaction_state")
        next_state = InteractionStateV1.FAILED
        code = "stale_session_revision"
    elif action is InteractionActionV1.FAIL:
        if state not in _FAILABLE:
            raise InteractionTransitionError("terminal_interaction_state")
        next_state = InteractionStateV1.FAILED
        code = "interaction_failed"
    elif action is InteractionActionV1.CANCEL:
        if state not in _CANCELLABLE:
            raise InteractionTransitionError("interaction_not_cancellable")
        next_state = InteractionStateV1.IDLE
        code = "interaction_cancelled"
    else:
        try:
            next_state = _TRANSITIONS[(state, action)]
        except KeyError as exc:
            raise InteractionTransitionError("invalid_interaction_transition") from exc
        if action is InteractionActionV1.ACCEPT and not (
            evaluation_eligible and certificate_fresh
        ):
            raise InteractionTransitionError("acceptance_evidence_required")
        code = f"interaction_{next_state.value}"

    tier = _feedback_tier(next_state)
    certificate_visible = next_state in {
        InteractionStateV1.EVALUATED,
        InteractionStateV1.ACCEPTED,
        InteractionStateV1.REJECTED,
    } and certificate_fresh
    acceptance_allowed = (
        next_state is InteractionStateV1.EVALUATED
        and evaluation_eligible
        and certificate_fresh
    )
    return InteractionTransitionV1(
        previous_state=state,
        action=action,
        state=next_state,
        feedback_tier=tier,
        certificate_visible=certificate_visible,
        acceptance_allowed=acceptance_allowed,
        code=code,
    )


def _feedback_tier(state: InteractionStateV1) -> FeedbackTierV1:
    if state in {
        InteractionStateV1.ROUTE_CHECKED,
        InteractionStateV1.REPAIR_RUNNING,
    }:
        return FeedbackTierV1.ROUTE_CHECKED
    if state in {
        InteractionStateV1.EVALUATED,
        InteractionStateV1.ACCEPTED,
        InteractionStateV1.REJECTED,
    }:
        return FeedbackTierV1.EVALUATED
    return FeedbackTierV1.VISUAL_ONLY
