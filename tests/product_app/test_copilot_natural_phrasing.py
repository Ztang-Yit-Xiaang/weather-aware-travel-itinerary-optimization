"""Natural phrasings that local guards previously blocked before any interpretation."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from itinerary_system.product_app.copilot import DeterministicCopilotAdapter, request_semantic_guard
from itinerary_system.product_app.copilot_memory import interpret_context_request
from itinerary_system.product_app.openai_copilot import OpenAICopilotAdapter
from itinerary_system.product_app.registry import ProductRunRegistry
from itinerary_system.product_app.service import ProductService

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def context(tmp_path_factory):
    service = ProductService(
        ProductRunRegistry(ROOT, ROOT / "configs/product_app_registry.json"),
        tmp_path_factory.mktemp("copilot_natural_phrasing"),
        copilot=DeterministicCopilotAdapter(),
    )
    session, _, _ = service.create_session(service.registry.default.run_id)
    return service.copilot_context(session)


def _offline_openai(calls: list) -> OpenAICopilotAdapter:
    async def forbidden(**kwargs):
        calls.append(kwargs)
        raise AssertionError("This request must be resolved locally")

    return OpenAICopilotAdapter(
        model="never-network",
        api_key="synthetic-unusable-test-key",
        client=SimpleNamespace(responses=SimpleNamespace(parse=forbidden)),
    )


@pytest.mark.parametrize(
    "message",
    (
        "I'd like to spend 2 hours at Bixby Creek Bridge Viewpoint",
        "I would like to spend 2 hours at Bixby Creek Bridge Viewpoint",
        "I’d like to spend 2 hours at Bixby Creek Bridge Viewpoint",
    ),
)
def test_courtesy_prefix_is_not_counted_as_a_second_edit(context, message):
    guarded = request_semantic_guard(context, message)
    assert guarded is None or "additional details" not in guarded.assistant_message


def test_courtesy_prefix_still_reports_a_genuine_second_edit(context):
    guarded = request_semantic_guard(
        context, "I'd like to spend 2 hours at Bixby Creek Bridge Viewpoint and move it to day 6",
    )
    assert guarded is not None and guarded.state != "proposal_ready"


@pytest.mark.parametrize(
    "message,minutes",
    (
        ("Plan a 5-day relaxed trip with at most 180 minutes of driving per day", 180),
        ("Plan a 5-day trip with 3 hours of driving a day", 180),
        ("Plan a 6-day relaxed trip, max 4 hours driving per day", 240),
    ),
)
def test_daily_drive_limit_before_the_word_driving_is_parsed(context, message, minutes):
    calls: list = []
    result = asyncio.run(_offline_openai(calls).interpret(context=context, history=(), message=message))
    assert not calls
    assert result.state == "proposal_ready"
    assert result.intents[0].type == "replan_trip"
    assert result.intents[0].parameters["max_daily_drive_minutes"] == minutes


def test_replacement_clarification_names_the_registered_choices(context):
    target = "bixby_creek_bridge_viewpoint"
    names = {str(row["place_id"]): str(row["name"]) for row in context.known_places}
    choices = [names[row["candidate_id"]] for row in context.capabilities["candidate_replacements"]
               if row["replaces_stop_id"] == target and row["candidate_id"] in names]
    assert choices, "fixture must register at least one replacement for Bixby"
    result = interpret_context_request(context, "Replace Bixby Creek Bridge Viewpoint with a nearby alternative")
    assert result is not None and result.state == "clarification_required"
    question = result.intents[0].clarification_question
    assert all(choice in question for choice in choices[:5])


@pytest.mark.parametrize(
    "message",
    (
        # A reason sentence with a modal or feeling verb is not a second edit.
        "Highway 1 near the bridge might be super foggy that morning. Replace Bixby Creek Bridge Viewpoint on day 5 with Pfeiffer Big Sur State Park.",
        "I hate waking up early on the first day of vacation. Change our Golden Gate Bridge visit time on day 1 to 2:00 PM.",
        # "visit time" is a noun, not a second "visit" action.
        "Change the visit time for Griffith Observatory on day 7 to 17:30.",
        # A place named only in the reason is not a second target.
        "We want to catch the sunset over Los Angeles on our final night. Change the visit time for Griffith Observatory on day 7 to 5:30 PM.",
        # Chinese reason clause plus a day-qualified replacement.
        "我们更喜欢海边的自然保护区。请把第5天的Bixby Creek Bridge Viewpoint替换成Point Lobos State Natural Reserve。",
    ),
)
def test_reasoned_or_noun_phrased_edits_are_not_refused_as_compound(context, message):
    guarded = request_semantic_guard(context, message)
    assert guarded is None or guarded.state == "proposal_ready", guarded and guarded.assistant_message


def test_moving_a_visit_time_is_not_a_day_move(context):
    result = interpret_context_request(context, "Move our Stearns Wharf visit time on day 6 to 9:30 AM.")
    assert result is None or all(intent.type != "move_day" for intent in result.intents)
    assert result is None or "not enabled" not in result.assistant_message


def test_a_reason_with_a_condition_is_still_a_condition(context):
    guarded = request_semantic_guard(
        context, "It might rain if the storm arrives. Replace Bixby Creek Bridge Viewpoint with Pfeiffer Big Sur State Park.",
    )
    assert guarded is not None and guarded.state == "clarification_required"
