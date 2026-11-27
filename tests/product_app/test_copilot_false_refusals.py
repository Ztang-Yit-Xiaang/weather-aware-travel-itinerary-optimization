"""Messages the local guards refused although they carried one clear request.

Found by the 2026-11-26 message probe (.duck/reports/2026-11-26-probe.md).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from itinerary_system.product_app.copilot import (
    CopilotIntentV1,
    CopilotInterpretationV1,
    DeterministicCopilotAdapter,
    is_stop_removal_request,
    known_place_clarification,
    request_semantic_guard,
)
from itinerary_system.product_app.openai_copilot import _canonicalize_provider_copy
from itinerary_system.product_app.registry import ProductRunRegistry
from itinerary_system.product_app.service import ProductService

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def context(tmp_path_factory):
    service = ProductService(
        ProductRunRegistry(ROOT, ROOT / "configs/product_app_registry.json"),
        tmp_path_factory.mktemp("copilot_false_refusals"),
        copilot=DeterministicCopilotAdapter(),
    )
    session, _, _ = service.create_session(service.registry.default.run_id)
    return service.copilot_context(session)


@pytest.mark.parametrize(
    "message",
    (
        "I don't want to remove Kings Canyon, just make it shorter.",
        "I do not want to remove Kings Canyon National Park, but shorten it to 90 minutes",
        "Don't skip Kings Canyon, only make it shorter",
    ),
)
def test_negated_removal_does_not_block_the_real_edit(context, message):
    guarded = request_semantic_guard(context, message)
    assert guarded is None or guarded.state == "proposal_ready", guarded
    assert not is_stop_removal_request(message)


def test_negated_removal_alone_still_proposes_keep_stop(context):
    guarded = request_semantic_guard(context, "I don't want to remove Kings Canyon National Park")
    assert guarded is not None and guarded.state == "proposal_ready"
    assert [intent.type for intent in guarded.intents] == ["keep_stop"]


def test_real_removal_next_to_another_edit_is_still_refused(context):
    guarded = request_semantic_guard(
        context,
        "Remove Stanford University Main Quad and make Kings Canyon shorter",
    )
    assert guarded is not None and guarded.state == "unsupported"
    assert "Removing" in guarded.assistant_message


@pytest.mark.parametrize(
    "message",
    (
        "Could we visit Monterey instead of Bixby?",
        "Visit Monterey rather than Bixby Creek Bridge Viewpoint",
    ),
)
def test_instead_of_is_one_swap_not_two_places(context, message):
    guarded = request_semantic_guard(context, message)
    assert guarded is None or "multiple places" not in guarded.assistant_message


def test_two_places_without_a_swap_still_ask_which_one(context):
    for message in ("Visit Monterey and Bixby Creek Bridge Viewpoint", "Visit Monterey, Bixby Creek Bridge Viewpoint"):
        guarded = request_semantic_guard(context, message)
        assert guarded is not None and guarded.state == "clarification_required", message


def test_unvalidated_swap_explains_the_route_gap(context):
    message = "Could we visit Monterey instead of Bixby Creek Bridge Viewpoint?"
    copy = known_place_clarification(context, message)
    assert copy is not None
    assistant_message, question = copy
    assert "Monterey" in question and "Bixby Creek Bridge Viewpoint" in question
    assert "not validated" in question
    refused = CopilotInterpretationV1(
        state="unsupported",
        assistant_message="x",
        intents=(CopilotIntentV1(type="unsupported_request"),),
    )
    result = _canonicalize_provider_copy(refused, context=context, message=message)
    assert result.state == "clarification_required" and "Monterey" in result.assistant_message


def test_provider_question_about_minutes_is_kept(context):
    question = "How many minutes should Kings Canyon National Park be set to?"
    asked = CopilotInterpretationV1(
        state="clarification_required",
        assistant_message="x",
        intents=(CopilotIntentV1(type="request_clarification", clarification_question=question),),
    )
    message = "I don't want to remove Kings Canyon, just make it shorter."
    result = _canonicalize_provider_copy(asked, context=context, message=message)
    assert result.assistant_message == question + " Nothing was changed."


def test_place_named_as_a_day_label_is_not_a_place_request(context):
    assert known_place_clarification(context, "Make the Big Sur day shorter") is None
    assert known_place_clarification(context, "Add Big Sur on day 5") is not None


def test_not_driving_so_much_asks_for_the_daily_cap(context):
    guarded = request_semantic_guard(context, "I'd rather not drive so much on the Big Sur day.")
    assert guarded is not None and guarded.state == "clarification_required"
    assert "maximum daily driving time" in guarded.assistant_message
    assert "discovery catalog" not in guarded.assistant_message
