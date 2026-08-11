from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest

from itinerary_system.product_app.copilot import (
    CopilotAdapter,
    CopilotContractError,
    DeterministicCopilotAdapter,
    validate_copilot_interpretation,
)
from itinerary_system.product_app.models import (
    CopilotContextV1,
    CopilotHighlightsV1,
    CopilotIntentV1,
    CopilotInterpretationV1,
)


def context() -> CopilotContextV1:
    return CopilotContextV1(
        run_id="run_demo",
        trip_id="trip_demo",
        session_id="session_demo",
        session_revision=3,
        accepted_plan_id="plan_original",
        selected_day=2,
        selected_stop_id="stop_a",
        selected_segment_id="segment_a_b",
        selected_candidate_id="candidate_b",
        selected_alternative_id=None,
        draft_operations=(),
        evaluated_proposal=None,
        allowed_stop_ids=("stop_a", "stop_b"),
        allowed_candidate_ids=("candidate_b",),
        allowed_days=(1, 2, 3),
        allowed_segment_ids=("segment_a_b",),
    )


def test_deterministic_adapter_satisfies_async_typed_protocol() -> None:
    adapter: CopilotAdapter = DeterministicCopilotAdapter()

    interpretation = asyncio.run(
        adapter.interpret(
            context=context(),
            history=(),
            message="Review a safer weather repair",
        )
    )

    assert adapter.provider_name == "deterministic"
    assert interpretation.state == "proposal_ready"
    assert interpretation.intents == (
        CopilotIntentV1(
            type="review_registered_repair",
            preference="reduce_contextual_risk",
        ),
    )
    assert interpretation.highlights == CopilotHighlightsV1(
        day_ids=(2,),
        stop_ids=("stop_a",),
        segment_ids=("segment_a_b",),
        candidate_ids=("candidate_b",),
    )
    assert "deterministic adapter" in interpretation.assistant_message
    assert "nothing was activated" not in interpretation.assistant_message


def test_deterministic_states_are_safe_and_explicit() -> None:
    adapter = DeterministicCopilotAdapter()

    empty = asyncio.run(adapter.interpret(context=context(), history=(), message="   "))
    booking = asyncio.run(
        adapter.interpret(
            context=context(),
            history=(),
            message="Cancel reservation",
        )
    )
    too_large = asyncio.run(
        adapter.interpret(
            context=context(),
            history=(),
            message="x" * 2_001,
        )
    )

    assert empty.state == "clarification_required"
    assert empty.intents[0].type == "request_clarification"
    assert booking.state == "permission_required"
    assert booking.permission == "external_booking_change"
    assert booking.intents == (CopilotIntentV1(type="external_action_not_supported"),)
    assert "will not be executed" in booking.assistant_message
    assert too_large.state == "failed"
    assert too_large.intents == ()


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("Keep this stop", CopilotIntentV1(type="keep_stop", target_stop_id="stop_a")),
        ("Lock it", CopilotIntentV1(type="lock_stop", target_stop_id="stop_a")),
        (
            "Mark it flexible",
            CopilotIntentV1(type="mark_flexible", target_stop_id="stop_a"),
        ),
        (
            "Move it to day 3",
            CopilotIntentV1(type="move_day", target_stop_id="stop_a", destination_day=3),
        ),
        (
            "Replace it with the selected candidate",
            CopilotIntentV1(
                type="replace_nearby",
                target_stop_id="stop_a",
                candidate_id="candidate_b",
            ),
        ),
        (
            "Add the selected candidate",
            CopilotIntentV1(
                type="add_candidate",
                destination_day=2,
                candidate_id="candidate_b",
            ),
        ),
        (
            "Reduce route driving",
            CopilotIntentV1(type="route_feedback", preference="reduce_driving"),
        ),
    ],
)
def test_deterministic_operations_are_typed_without_activation(
    message: str,
    expected: CopilotIntentV1,
) -> None:
    result = asyncio.run(
        DeterministicCopilotAdapter().interpret(
            context=context(),
            history=(),
            message=message,
        )
    )

    assert result.state == "proposal_ready"
    assert result.intents == (expected,)
    assert "nothing was activated automatically" in result.assistant_message


@pytest.mark.parametrize(
    "interpretation",
    [
        CopilotInterpretationV1(state="invented", assistant_message="No."),
        CopilotInterpretationV1(state="failed", assistant_message=""),
        CopilotInterpretationV1(
            state="proposal_ready",
            assistant_message="No typed operation.",
        ),
        CopilotInterpretationV1(
            state="proposal_ready",
            assistant_message="Invented entity.",
            intents=(CopilotIntentV1(type="keep_stop", target_stop_id="invented"),),
        ),
        CopilotInterpretationV1(
            state="proposal_ready",
            assistant_message="Invalid operation shape.",
            intents=(CopilotIntentV1(type="move_day", target_stop_id="stop_a"),),
        ),
        CopilotInterpretationV1(
            state="proposal_ready",
            assistant_message="Invented preference.",
            intents=(
                CopilotIntentV1(
                    type="route_feedback",
                    preference="scenic_teleportation",
                ),
            ),
        ),
        CopilotInterpretationV1(
            state="proposal_ready",
            assistant_message="Invented geometry.",
            intents=(CopilotIntentV1(type="keep_original"),),
            highlights=CopilotHighlightsV1(segment_ids=("invented_segment",)),
        ),
        CopilotInterpretationV1(
            state="permission_required",
            assistant_message="Unsafe permission shape.",
            intents=(CopilotIntentV1(type="keep_original"),),
            permission="external_booking_change",
        ),
    ],
)
def test_validation_rejects_unknown_entities_enums_and_combinations(
    interpretation: CopilotInterpretationV1,
) -> None:
    with pytest.raises(CopilotContractError, match="copilot_interpretation_invalid"):
        validate_copilot_interpretation(
            context=context(),
            interpretation=interpretation,
        )


def test_validation_accepts_all_allow_list_operation_shapes() -> None:
    intents = (
        CopilotIntentV1(type="keep_stop", target_stop_id="stop_a"),
        CopilotIntentV1(type="move_day", target_stop_id="stop_b", destination_day=3),
        CopilotIntentV1(
            type="replace_nearby",
            target_stop_id="stop_a",
            candidate_id="candidate_b",
        ),
        CopilotIntentV1(
            type="add_candidate",
            destination_day=1,
            candidate_id="candidate_b",
        ),
    )
    interpretation = CopilotInterpretationV1(
        state="proposal_ready",
        assistant_message="Review these typed proposals.",
        intents=intents,
    )

    assert (
        validate_copilot_interpretation(
            context=context(),
            interpretation=interpretation,
        )
        is interpretation
    )


def test_validation_rejects_stale_or_malformed_context() -> None:
    interpretation = CopilotInterpretationV1(
        state="proposal_ready",
        assistant_message="Review this typed proposal.",
        intents=(CopilotIntentV1(type="keep_original"),),
    )

    with pytest.raises(CopilotContractError):
        validate_copilot_interpretation(
            context=replace(context(), selected_stop_id="not_allowed"),
            interpretation=interpretation,
        )

    for corrupted in (
        replace(context(), selected_segment_id="segment_not_in_artifacts"),
        replace(context(), selected_alternative_id="plan_not_in_artifacts"),
    ):
        with pytest.raises(CopilotContractError):
            validate_copilot_interpretation(
                context=corrupted,
                interpretation=interpretation,
            )


@pytest.mark.parametrize(
    "message",
    (
        "I booked and paid successfully.",
        "Your reservation has been confirmed.",
        "We charged your card for the ticket.",
    ),
)
def test_validation_rejects_false_external_execution_claims(message: str) -> None:
    interpretation = CopilotInterpretationV1(
        state="unsupported",
        assistant_message=message,
        intents=(CopilotIntentV1(type="external_action_not_supported"),),
    )

    with pytest.raises(CopilotContractError, match="copilot_interpretation_invalid"):
        validate_copilot_interpretation(context=context(), interpretation=interpretation)


def test_deterministic_adapter_does_not_silently_use_history() -> None:
    adapter = DeterministicCopilotAdapter()

    result = asyncio.run(
        adapter.interpret(
            context=context(),
            history=(),
            message="Do something magical",
        )
    )

    assert result.state == "clarification_required"
    assert result.permission is None
    assert result.schema_version == "copilot-interpretation-v1"
