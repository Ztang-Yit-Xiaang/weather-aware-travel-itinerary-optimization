"""Read-only answers to trip questions, and reason sentences before an edit."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from itinerary_system.product_app.conversations import ConversationRepository
from itinerary_system.product_app.copilot import (
    CopilotContractError,
    DeterministicCopilotAdapter,
    request_semantic_guard,
    validate_copilot_interpretation,
)
from itinerary_system.product_app.copilot_answers import (
    TripAnswerSchemaV1,
    answer_plan_question,
    answered,
    build_trip_facts,
    local_answer,
    looks_like_plan_question,
)
from itinerary_system.product_app.models import CopilotIntentV1, CopilotInterpretationV1
from itinerary_system.product_app.openai_copilot import OpenAICopilotAdapter
from itinerary_system.product_app.registry import ProductRunRegistry
from itinerary_system.product_app.service import ProductService

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    service = ProductService(
        ProductRunRegistry(ROOT, ROOT / "configs/product_app_registry.json"),
        tmp_path_factory.mktemp("copilot_answers"),
        copilot=DeterministicCopilotAdapter(),
    )
    session, _, _ = service.create_session(service.registry.default.run_id)
    view = service.workspace_view_for_session(session)
    context = service.copilot_context(session, view=view)
    return context, build_trip_facts(view, context, repository_root=ROOT)


@pytest.mark.parametrize(
    "message",
    (
        "Which day has the most driving?",
        "What is Bixby Creek Bridge and what should I do there?",
        "How much will this trip cost roughly?",
        "Tell me about day 4",
        "Where do we sleep on day 3?",
        "Any tips for Yosemite?",
        "What alternatives are registered if we skip Bixby Creek Bridge Viewpoint on day 5?",
        "请问第3天去Yosemite National Park那天晚上住哪个酒店？开车要多久？",
        "第6天去Stearns Wharf要开多少分钟的车？",
    ),
)
def test_questions_are_routed_to_answers(message):
    assert looks_like_plan_question(message)


@pytest.mark.parametrize(
    "message",
    (
        "Can you replace Bixby Creek Bridge Viewpoint with Point Lobos?",
        "Replace Bixby Creek Bridge Viewpoint with Pfeiffer Big Sur State Park.",
        "Could you move Stearns Wharf to day 7?",
        "I'd like to spend 2 hours at Bixby Creek Bridge Viewpoint",
        "Plan a 5-day relaxed trip with at most 180 minutes of driving per day",
        "Please shorten the Yosemite visit?",
        "第5天去Bixby Creek Bridge Viewpoint的行程能不能稍微调整一下？",
        "请把第5天的Bixby Creek Bridge Viewpoint替换成Point Lobos State Natural Reserve。",
        "Which alternative do you recommend for Bixby?",
        "Can we spend 90 minutes at Bixby Creek Bridge Viewpoint?",
        "Could we swap Bixby Creek Bridge Viewpoint with Point Lobos State Natural Reserve?",
        "What if I visit Hearst Castle on Day 4?",
    ),
)
def test_edit_requests_stay_on_the_edit_pipeline(message):
    assert not looks_like_plan_question(message)


def test_facts_cover_every_day_with_drive_and_budget(workspace):
    context, facts = workspace
    assert [row["day"] for row in facts["days"]] == list(context.allowed_days)
    # Per-day values are rounded separately, so allow one minute per day.
    assert abs(facts["trip"]["total_drive_minutes"] - sum(row["drive_minutes"] for row in facts["days"])) <= len(
        facts["days"]
    )
    assert all(row["drive_minutes"] > 0 for row in facts["days"])
    budget = facts["budget_estimate_usd"]
    assert budget["total"] == sum(budget[key] for key in ("lodging", "food", "attractions", "fuel", "local_transport"))
    assert "lodging" in budget["assumptions"]


def test_local_answers_are_grounded_in_the_fact_sheet(workspace):
    context, facts = workspace
    top = max(facts["days"], key=lambda row: row["drive_minutes"])
    result = local_answer(context, facts, "Which day has the most driving?")
    assert result.state == "answered" and f"Day {top['day']}" in result.assistant_message
    assert result.highlights.day_ids == (top["day"],)
    budget = local_answer(context, facts, "How much will this cost?")
    assert f"${facts['budget_estimate_usd']['total']:,}" in budget.assistant_message
    validate_copilot_interpretation(context=context, interpretation=result)


def test_answered_state_is_read_only_and_persistable(workspace):
    context, _ = workspace
    answer = answered(context, "Day 4 has the most driving.", stop_ids=["not_a_stop"], day_ids=[4, 99])
    assert answer.highlights.stop_ids == () and answer.highlights.day_ids == (4,)
    validate_copilot_interpretation(context=context, interpretation=answer)
    restored = ConversationRepository._validate_interpretation(answer.as_dict())
    assert restored.state == "answered"
    smuggled = CopilotInterpretationV1(
        state="answered",
        assistant_message="x",
        intents=(CopilotIntentV1(type="answer_question"), CopilotIntentV1(type="keep_original")),
    )
    with pytest.raises(CopilotContractError):
        validate_copilot_interpretation(context=context, interpretation=smuggled)


def test_answer_text_cannot_claim_an_external_action(workspace):
    context, _ = workspace
    result = answered(context, "I have booked the Comfort Inn for you.")
    assert "booked" not in result.assistant_message.lower()


def _provider(parsed):
    async def parse(**kwargs):
        return SimpleNamespace(output_parsed=parsed)

    return OpenAICopilotAdapter(
        model="test",
        api_key="synthetic-unusable-test-key",
        client=SimpleNamespace(responses=SimpleNamespace(parse=parse)),
    )


def test_provider_answer_and_edit_fallthrough(workspace):
    context, facts = workspace
    reply = TripAnswerSchemaV1(kind="answer", answer="Day 4 is the longest drive.", stop_ids=[], day_ids=[4])
    result = asyncio.run(
        answer_plan_question(
            _provider(reply), context=context, history=(), message="Which day is longest?", facts=facts
        )
    )
    assert result.state == "answered" and result.highlights.day_ids == (4,)
    edit = TripAnswerSchemaV1(kind="edit_request", answer="", stop_ids=[], day_ids=[])
    assert (
        asyncio.run(
            answer_plan_question(
                _provider(edit), context=context, history=(), message="What if we swap Bixby?", facts=facts
            )
        )
        is None
    )
    off = TripAnswerSchemaV1(kind="off_topic", answer="I can help with this trip.", stop_ids=[], day_ids=[])
    result = asyncio.run(
        answer_plan_question(
            _provider(off), context=context, history=(), message="What's the capital of France?", facts=facts
        )
    )
    assert result.state == "unsupported"


@pytest.mark.parametrize(
    "message",
    (
        "Bixby is too risky in this weather. Replace Bixby Creek Bridge Viewpoint with Pfeiffer Big Sur State Park.",
        "Day 5 has too much driving. Can you replace Bixby Creek Bridge Viewpoint with Point Lobos State Natural Reserve?",
    ),
)
def test_a_reason_sentence_is_not_a_second_edit(workspace, message):
    context, _ = workspace
    guarded = request_semantic_guard(context, message)
    assert guarded is None or "additional details" not in guarded.assistant_message


def test_a_reason_sentence_does_not_hide_a_negation(workspace):
    context, _ = workspace
    guarded = request_semantic_guard(context, "Bixby is too risky. Don't replace Bixby Creek Bridge Viewpoint.")
    assert guarded is not None and guarded.state != "proposal_ready"


def test_spend_time_question_is_not_a_budget_answer(workspace):
    context, facts = workspace
    result = local_answer(context, facts, "How long do we spend at Bixby Creek Bridge Viewpoint?")
    assert result is not None and "Estimated total" not in result.assistant_message
    assert "Bixby" in result.assistant_message
    money = local_answer(context, facts, "How much will we spend in total?")
    assert money is not None and "Estimated total" in money.assistant_message


def test_generic_first_word_does_not_pick_a_stop(workspace):
    context, facts = workspace
    result = local_answer(context, facts, "What is the point of this trip?")
    assert result is None or "Point Lobos" not in result.assistant_message
