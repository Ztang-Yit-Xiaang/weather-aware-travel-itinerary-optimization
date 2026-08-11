from __future__ import annotations

import pytest

from itinerary_system.product_app.interaction_contracts import (
    FeedbackTierV1,
    InteractionStateV1,
)
from itinerary_system.product_app.interaction_state_machine import (
    InteractionActionV1,
    InteractionTransitionError,
    transition_interaction,
)


def test_direct_edit_flow_keeps_feedback_tiers_distinct() -> None:
    state = InteractionStateV1.IDLE
    steps = (
        (InteractionActionV1.SELECT, InteractionStateV1.SELECTED),
        (InteractionActionV1.BEGIN_EDIT, InteractionStateV1.EDITING),
        (InteractionActionV1.SHOW_GHOST, InteractionStateV1.GHOST_PREVIEW),
        (InteractionActionV1.REQUEST_SNAP, InteractionStateV1.SNAP_PREVIEW),
        (InteractionActionV1.REQUIRE_CONFIRMATION, InteractionStateV1.PENDING_CONFIRMATION),
        (InteractionActionV1.APPEND_DRAFT, InteractionStateV1.DRAFT_APPENDED),
        (InteractionActionV1.COMPLETE_ROUTE_PREVIEW, InteractionStateV1.ROUTE_CHECKED),
        (InteractionActionV1.RUN_REPAIR, InteractionStateV1.REPAIR_RUNNING),
        (InteractionActionV1.COMPLETE_EVALUATION, InteractionStateV1.EVALUATED),
    )
    for action, expected in steps:
        result = transition_interaction(
            state,
            action,
            evaluation_eligible=expected is InteractionStateV1.EVALUATED,
            certificate_fresh=expected is InteractionStateV1.EVALUATED,
        )
        state = result.state
        assert state is expected

    assert result.feedback_tier is FeedbackTierV1.EVALUATED
    assert result.certificate_visible is True
    assert result.acceptance_allowed is True


def test_visual_or_route_checked_state_cannot_be_accepted() -> None:
    for state in (InteractionStateV1.GHOST_PREVIEW, InteractionStateV1.ROUTE_CHECKED):
        with pytest.raises(InteractionTransitionError, match="invalid_interaction_transition"):
            transition_interaction(
                state,
                InteractionActionV1.ACCEPT,
                evaluation_eligible=True,
                certificate_fresh=True,
            )


def test_evaluated_state_requires_fresh_eligible_evidence_before_acceptance() -> None:
    with pytest.raises(InteractionTransitionError, match="acceptance_evidence_required"):
        transition_interaction(
            InteractionStateV1.EVALUATED,
            InteractionActionV1.ACCEPT,
            evaluation_eligible=True,
            certificate_fresh=False,
        )

    accepted = transition_interaction(
        InteractionStateV1.EVALUATED,
        InteractionActionV1.ACCEPT,
        evaluation_eligible=True,
        certificate_fresh=True,
    )
    assert accepted.state is InteractionStateV1.ACCEPTED
    assert accepted.certificate_visible is True
    assert accepted.acceptance_allowed is False


def test_stale_revision_fails_without_claiming_route_or_evaluation() -> None:
    result = transition_interaction(
        InteractionStateV1.PENDING_CONFIRMATION,
        InteractionActionV1.STALE_REVISION,
    )
    assert result.state is InteractionStateV1.FAILED
    assert result.feedback_tier is FeedbackTierV1.VISUAL_ONLY
    assert result.certificate_visible is False
    assert result.acceptance_allowed is False
    assert result.code == "stale_session_revision"


def test_cancel_discards_uncommitted_interaction_and_terminal_state_is_closed() -> None:
    cancelled = transition_interaction(
        InteractionStateV1.EXPLORATORY_PREVIEW,
        InteractionActionV1.CANCEL,
    )
    assert cancelled.state is InteractionStateV1.IDLE
    assert cancelled.code == "interaction_cancelled"

    with pytest.raises(InteractionTransitionError, match="terminal_interaction_state"):
        transition_interaction(
            InteractionStateV1.ACCEPTED,
            InteractionActionV1.STALE_REVISION,
        )


def test_revise_evaluated_candidate_invalidates_certificate_visibility() -> None:
    revised = transition_interaction(
        InteractionStateV1.EVALUATED,
        InteractionActionV1.REVISE,
        evaluation_eligible=True,
        certificate_fresh=True,
    )
    assert revised.state is InteractionStateV1.EDITING
    assert revised.feedback_tier is FeedbackTierV1.VISUAL_ONLY
    assert revised.certificate_visible is False
    assert revised.acceptance_allowed is False
