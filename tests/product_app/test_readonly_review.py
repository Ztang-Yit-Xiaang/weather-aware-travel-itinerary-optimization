"""Explicit review boundaries using the registered ProductService context, offline."""

from __future__ import annotations

import asyncio
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from itinerary_system.product_app.copilot import DeterministicCopilotAdapter
from itinerary_system.product_app.models import CopilotContextV2
from itinerary_system.product_app.openai_copilot import OpenAICopilotAdapter
from itinerary_system.product_app.registry import ProductRunRegistry
from itinerary_system.product_app.service import ProductService
from itinerary_system.product_app.trip_planning import trip_request_interpretation

ROOT = Path(__file__).resolve().parents[2]
READONLY_INTENTS = {"request_clarification", "review_registered_repair", "unsupported_request"}


@pytest.fixture(scope="module")
def registered_context(tmp_path_factory: pytest.TempPathFactory) -> CopilotContextV2:
    # Match the real registry-backed fixture in test_trip_planning.py. All new
    # acceptance/workspace/conversation files belong to this test's temp root.
    service = ProductService(
        ProductRunRegistry(ROOT, ROOT / "configs/product_app_registry.json"),
        tmp_path_factory.mktemp("readonly_review"),
        copilot=DeterministicCopilotAdapter(),
    )
    session, _, _ = service.create_session(service.registry.default.run_id)
    context = service.copilot_context(session)
    assert isinstance(context, CopilotContextV2)
    assert context.capabilities["operations"]["replan_trip"]["enabled"] is True
    assert len(context.allowed_days) > 1
    return context


@pytest.fixture
def context(registered_context: CopilotContextV2):
    before = deepcopy(registered_context.as_dict())
    yield registered_context
    # Compare the complete nested context, including capabilities, selection,
    # accepted-plan identity/hash, revision, draft, preview and memory.
    assert registered_context.as_dict() == before


@pytest.fixture(params=("deterministic", "openai"))
def adapter(request: pytest.FixtureRequest):
    calls = []

    async def forbidden(**kwargs):
        calls.append(kwargs)
        pytest.fail("Bounded review/duration requests must return before any SDK call")

    selected = (
        DeterministicCopilotAdapter()
        if request.param == "deterministic"
        else OpenAICopilotAdapter(
            model="never-network",
            api_key="synthetic-unusable-test-key",
            client=SimpleNamespace(responses=SimpleNamespace(parse=forbidden)),
        )
    )
    yield selected
    assert calls == [], "No injected SDK call is allowed, even if the adapter catches its failure"


@pytest.mark.parametrize(
    "message",
    (
        "請用這份行程指出最值得我重新考慮的一天，先不要幫我改動。",
        "请用这份行程指出最值得我重新考虑的一天，先不要帮我改动。",
        "Which day should I review? Do not change the itinerary.",
        "Review my trip without making changes.",
        "請指出這份行程哪一天最值得重新考慮，先不要幫我改動。",
        "Review Day 1 of my trip without making changes.",
    ),
)
def test_explicit_readonly_review_never_proposes_a_mutation(context, adapter, message):
    result = asyncio.run(adapter.interpret(context=context, history=(), message=message))

    assert result.intents, result.as_dict()
    assert {intent.type for intent in result.intents} <= READONLY_INTENTS, result.as_dict()


@pytest.mark.parametrize(
    "message",
    (
        "請用這份行程指出最值得我重新考慮的一天。",
        "请用这份行程指出最值得我重新考虑的一天。",
        "這份行程哪一天最值得重新考慮？",
        "Which day should I review?",
        "Review Day 1 of my trip.",
    ),
)
def test_day_references_do_not_imply_total_trip_duration(context, message):
    # Exercise only duration recognition; this does not specify general
    # informational-question routing or require a new review capability.
    assert trip_request_interpretation(context, message) is None


@pytest.mark.parametrize(
    "message,days",
    (
        ("請幫我規劃一天的行程。", 1),
        ("Plan a 1-day trip.", 1),
        ("Plan a 10-day trip.", 10),
        ("我想製作十天的行程。", 10),
        ("Plan a two week trip.", 14),
    ),
)
def test_genuine_trip_durations_still_offer_the_complete_replan(context, adapter, message, days):
    result = asyncio.run(adapter.interpret(context=context, history=(), message=message))

    assert result.state == "proposal_ready", result.as_dict()
    assert len(result.intents) == 1
    assert result.intents[0].type == "replan_trip"
    assert result.intents[0].parameters["total_days"] == days


@pytest.mark.parametrize(
    "message",
    (
        "先不要幫我改動行程，再把整趟行程改成10天。",
        "先不要帮我改动行程，再把整趟行程改成10天。",
        "Review my trip without making changes, then change it to a 10-day trip.",
        "Do not change the itinerary. Then plan a 10-day trip.",
    ),
)
def test_no_change_clause_is_not_dropped_before_a_later_duration_change(context, adapter, message):
    result = asyncio.run(adapter.interpret(context=context, history=(), message=message))

    assert result.state == "clarification_required", result.as_dict()
    assert [intent.type for intent in result.intents] == ["request_clarification"]
    assert result.intents[0].clarification_question
