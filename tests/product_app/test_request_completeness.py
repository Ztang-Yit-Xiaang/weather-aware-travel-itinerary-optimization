"""A complete request is resolved or declined as a whole, using registered data."""

from __future__ import annotations

import asyncio
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from itinerary_system.product_app.copilot import DeterministicCopilotAdapter
from itinerary_system.product_app.openai_copilot import OpenAICopilotAdapter
from itinerary_system.product_app.registry import ProductRunRegistry
from itinerary_system.product_app.service import ProductService

ROOT = Path(__file__).resolve().parents[2]
LONG_REPLACEMENT = (
    "Replace Bixby Creek Bridge with Pfeiffer Big Sur State Park on Day 5, preserving the rest of the seven-day trip."
)


@pytest.fixture(scope="module")
def context(tmp_path_factory):
    service = ProductService(
        ProductRunRegistry(ROOT, ROOT / "configs/product_app_registry.json"),
        tmp_path_factory.mktemp("request_completeness"),
        copilot=DeterministicCopilotAdapter(),
    )
    session, _, _ = service.create_session(service.registry.default.run_id)
    return service.copilot_context(session)


def test_preservation_annotation_does_not_become_a_whole_trip_replan(context):
    before = deepcopy(context.as_dict())
    result = asyncio.run(DeterministicCopilotAdapter().interpret(context=context, history=(), message=LONG_REPLACEMENT))
    assert result.state == "proposal_ready"
    assert len(result.intents) == 1
    intent = result.intents[0]
    assert (intent.type, intent.target_stop_id, intent.candidate_id) == (
        "replace_nearby",
        "bixby_creek_bridge_viewpoint",
        "pfeiffer_big_sur_state_park",
    )
    assert context.as_dict() == before


@pytest.fixture(params=("deterministic", "openai"))
def local_adapter(request):
    calls = []

    async def forbidden(**kwargs):
        calls.append(kwargs)
        raise AssertionError("Incomplete requests must not reach the provider")

    adapter = (
        DeterministicCopilotAdapter()
        if request.param == "deterministic"
        else OpenAICopilotAdapter(
            model="never-network",
            api_key="synthetic-unused",
            client=SimpleNamespace(responses=SimpleNamespace(parse=forbidden)),
        )
    )
    yield adapter
    assert calls == []


@pytest.mark.parametrize(
    "message",
    (
        LONG_REPLACEMENT.replace("Day 5", "Day 4"),
        "Add Pfeiffer Big Sur State Park to day 5.",
        "Add Hearst Castle on Day 5 to this ten-day trip.",
        "請把 Hearst Castle 安排在第5天，旅行總共10天。",
        "Replace Bixby Creek Bridge with Pfeiffer Big Sur State Park and plan a ten-day trip.",
    ),
)
def test_local_day_or_edit_cannot_disappear_inside_a_replan(context, local_adapter, message):
    before = deepcopy(context.as_dict())
    result = asyncio.run(local_adapter.interpret(context=context, history=(), message=message))
    assert result.state in {"clarification_required", "unsupported"}, result.as_dict()
    assert {intent.type for intent in result.intents} <= {"request_clarification", "unsupported_request"}
    assert context.as_dict() == before


@pytest.mark.parametrize(
    "message",
    (
        "I prefer nature and less driving.",
        "I prefer less driving and museums.",
        "Plan a ten-day trip with nature and less driving.",
        "我想要更多自然景點，而且少開車。",
        "我想玩十天，喜欢博物馆，也想少开车。",
        "Plan a ten-day trip with nature and less driving than the original, driving at most 3 hours per day.",
        "I prefer nature; driving at most 3 hours per day; must reduce total driving compared with the original.",
    ),
)
def test_comparative_driving_requests_ask_for_an_enforceable_complete_request(context, local_adapter, message):
    result = asyncio.run(local_adapter.interpret(context=context, history=(), message=message))
    assert result.state == "clarification_required", result.as_dict()
    assert [intent.type for intent in result.intents] == ["request_clarification"]
    assert "daily driving" in result.assistant_message or "每天" in result.assistant_message
    assert "complete request" in result.assistant_message or "完整需求" in result.assistant_message


@pytest.mark.parametrize(
    "message,days,cap,interests",
    (
        ("I prefer nature.", 7, 720, ["nature"]),
        ("Plan a ten-day trip with nature, driving at most 3 hours per day.", 10, 180, ["nature"]),
        ("我想花兩週去旅行，喜歡博物館，每天開車不超過4小時。", 14, 240, ["culture"]),
        ("I prefer nature and less driving, driving at most 3 hours per day.", 7, 180, ["nature"]),
    ),
)
def test_supported_interests_duration_and_numeric_caps_remain_complete(
    context, local_adapter, message, days, cap, interests
):
    result = asyncio.run(local_adapter.interpret(context=context, history=(), message=message))
    assert result.state == "proposal_ready", result.as_dict()
    assert len(result.intents) == 1 and result.intents[0].type == "replan_trip"
    params = result.intents[0].parameters
    assert (params["total_days"], params["max_daily_drive_minutes"], params["interests"]) == (days, cap, interests)
    if "less driving" in message:
        assert "remains unverified" in result.assistant_message


@pytest.mark.parametrize("qualifier", ("with wheelchair access", "with a budget of $500", "and spend two hours there"))
def test_ready_insertion_does_not_hide_unrepresented_qualifiers(context, local_adapter, qualifier):
    caps = deepcopy(context.capabilities)
    caps["operations"]["add_candidate"] = {"enabled": True, "preview_executable": True, "blocking_code": None}
    caps["candidate_day_readiness"] = tuple(
        {**row, "preview_executable": True, "blocking_code": None} for row in caps["candidate_day_readiness"]
    )
    ready = replace(context, capabilities=caps)
    result = asyncio.run(
        local_adapter.interpret(
            context=ready,
            history=(),
            message="Add Pfeiffer Big Sur State Park to day 5 " + qualifier + ".",
        )
    )
    assert result.state == "clarification_required", result.as_dict()
    assert [intent.type for intent in result.intents] == ["request_clarification"]
