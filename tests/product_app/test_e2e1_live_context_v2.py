"""Current service context versus explicitly historical provider evidence; no calls."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import re
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from itinerary_system.product_app.copilot import DeterministicCopilotAdapter
from itinerary_system.product_app.copilot_prompt import build
from itinerary_system.product_app.openai_copilot import OpenAICopilotAdapter, OpenAIInterpretationSchemaV2
from itinerary_system.product_app.registry import ProductRunRegistry
from itinerary_system.product_app.service import ProductService

pytest_plugins = ["recorded_runtime_fixture"]
ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("live_v2_verifier", ROOT / "scripts/verify_e2e1_live_copilot_chat.py")
verifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verifier)
OLD = "results/e2e/runtime_optimizer_repair_replay_20260905_v5/live_copilot_chat/provider_exact_llm_io.json"
FAKE = "results/repairs/r07_typed_chat_20260908/exact_io_v2/current_live_context.json"


def request_for(context):
    return {
        "input": "\n".join(
            (
                "<workspace_context>",
                verifier._canonical_prompt_json(context),
                "</workspace_context>",
                "<recent_conversation>",
                "[]",
                "</recent_conversation>",
                "<user_request>",
                verifier._canonical_prompt_json(verifier.EXPECTED_PROMPT),
                "</user_request>",
            )
        ),
        "instructions": verifier.DEVELOPER_INSTRUCTIONS,
        "max_output_tokens": 1200,
        "model": "gpt-5.4-mini",
        "reasoning": {"effort": "low", "context": "current_turn"},
        "store": False,
        "tools": [],
        "text": {
            "verbosity": "low",
            "format": {
                "type": "json_schema",
                "strict": True,
                "name": "OpenAIInterpretationSchemaV2",
                "schema": verifier.to_strict_json_schema(verifier.OpenAIInterpretationSchemaV2),
            },
        },
    }


@pytest.fixture
def current_context(tmp_path):
    registry = ProductRunRegistry(ROOT, ROOT / "configs/product_app_registry.json")
    service = ProductService(registry, tmp_path / "state", copilot=DeterministicCopilotAdapter())
    session, _, _ = service.create_session(registry.default.run_id)
    service.workspace.select(session, {"expected_revision": 0, "selected_day": 5, "selected_stop_id": verifier.TARGET})
    return json.loads(json.dumps(service.copilot_context(session).as_dict()))


def verify_request(request, historical=False):
    return verifier._provider_request_context(
        request, request_text=json.dumps(request), allow_historical_schema=historical
    )


def test_current_context_binds_current_hash_and_new_fields(current_context):
    context, authorization = verify_request(request_for(current_context))
    assert context["schema_version"] == "copilot-context-v2"
    assert set(context) == verifier.CONTEXT_KEYS
    assert authorization["workspace_context"] == current_context
    assert authorization["output_schema_current"] is True


def test_retained_fake_context_is_not_current_authorization(recorded_runtime_inputs):
    evidence = recorded_runtime_inputs / FAKE
    original = evidence.read_bytes()
    with pytest.raises(verifier.LiveE2E1VerificationError, match="e2e1_live_provider_context_v2_invalid"):
        verify_request(request_for(json.loads(original)["context"]))
    assert evidence.read_bytes() == original


def test_current_real_service_context_matches_frozen_authorization(tmp_path):
    registry = ProductRunRegistry(ROOT, ROOT / "configs/product_app_registry.json")
    service = ProductService(registry, tmp_path / "state", copilot=DeterministicCopilotAdapter())
    session, _, _ = service.create_session(registry.default.run_id)
    service.workspace.select(session, {"expected_revision": 0, "selected_day": 5, "selected_stop_id": verifier.TARGET})
    context = service.copilot_context(session)
    prompt = build(context, [], verifier.EXPECTED_PROMPT)
    request = request_for(json.loads(json.dumps(context.as_dict())))
    assert request["input"] == prompt.input
    verified, _ = verify_request(request)
    assert len(verified["allowed_stop_ids"]) == 8
    assert verifier.CANDIDATE not in verified["allowed_stop_ids"]


def test_complete_approved_prompt_dispatches_once_but_local_blocks_do_not(tmp_path):
    registry = ProductRunRegistry(ROOT, ROOT / "configs/product_app_registry.json")
    service = ProductService(registry, tmp_path / "state", copilot=DeterministicCopilotAdapter())
    session, _, _ = service.create_session(registry.default.run_id)
    service.workspace.select(session, {"expected_revision": 0, "selected_day": 5, "selected_stop_id": verifier.TARGET})
    context = service.copilot_context(session)
    calls = []
    parsed = OpenAIInterpretationSchemaV2(
        schema_version="copilot-interpretation-v2",
        state="proposal_ready",
        assistant_message="Provider proposal.",
        intents=[{"type": "replace_nearby", "target_stop_id": verifier.TARGET, "candidate_id": verifier.CANDIDATE}],
    )

    async def parse(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(output=[], output_parsed=parsed)

    adapter = OpenAICopilotAdapter(
        model="gpt-5.4-mini", api_key="not-a-real-key", client=SimpleNamespace(responses=SimpleNamespace(parse=parse))
    )
    for message in (
        "Replace Bixby Creek Bridge.",
        "Move Golden Gate Bridge.",
        "Replace Bixby Creek Bridge and shorten the visit.",
    ):
        result = asyncio.run(adapter.interpret(context=context, history=(), message=message))
        assert result.state in {"clarification_required", "unsupported"}
    assert calls == []
    result = asyncio.run(adapter.interpret(context=context, history=(), message=verifier.EXPECTED_PROMPT))
    assert result.state == "proposal_ready" and result.intents[0].candidate_id == verifier.CANDIDATE
    assert len(calls) == 1 and calls[0]["store"] is False
    projected = json.loads(json.dumps(result.as_dict()))
    assert "parameters" in projected["intents"][0] and projected["intents"][0]["parameters"] is None
    verifier._verify_approved_interpretation(parsed.model_dump(mode="json"), projected)
    assert (
        json.loads(re.search(r"<workspace_context>\n(.*?)\n</workspace_context>", calls[0]["input"], re.S).group(1))[
            "schema_version"
        ]
        == "copilot-context-v2"
    )


def test_historical_v1_requires_explicit_policy_and_keeps_original_bytes(recorded_runtime_inputs):
    evidence = recorded_runtime_inputs / OLD
    original = evidence.read_bytes()
    provider = json.loads(original)
    request_text = provider["records"][0]["request_body_utf8"]
    request = json.loads(request_text)
    with pytest.raises(verifier.LiveE2E1VerificationError):
        verify_request(request)
    context, authorization = verifier._provider_request_context(
        request, request_text=request_text, allow_historical_schema=True
    )
    assert context["schema_version"] == "copilot-context-v1"
    assert authorization["instructions"] == request["instructions"] != verifier.DEVELOPER_INSTRUCTIONS
    assert authorization["request_body_sha256"] == provider["records"][0]["request_body_sha256"]
    assert evidence.read_bytes() == original


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_parameters",
        "extra_parameters",
        "wrong_candidate",
        "wrong_schema",
        "mixed_projection",
        "wrong_highlights",
    ],
)
def test_current_v2_post_response_requires_exact_null_parameters(mutation):
    projection = OpenAIInterpretationSchemaV2(
        schema_version="copilot-interpretation-v2",
        state="proposal_ready",
        assistant_message="Fake approved output.",
        intents=[{"type": "replace_nearby", "target_stop_id": verifier.TARGET, "candidate_id": verifier.CANDIDATE}],
    ).model_dump(mode="json")
    result = deepcopy(projection)
    result["highlights"] = {
        "day_ids": [5],
        "stop_ids": [verifier.TARGET],
        "segment_ids": [],
        "candidate_ids": [verifier.CANDIDATE],
    }
    verifier._verify_approved_interpretation(projection, result)
    if mutation == "missing_parameters":
        result["intents"][0].pop("parameters")
    elif mutation == "extra_parameters":
        result["intents"][0]["parameters"] = {"role": "meal"}
    elif mutation == "wrong_candidate":
        result["intents"][0]["candidate_id"] = "another_candidate"
    elif mutation == "wrong_schema":
        result["schema_version"] = "copilot-interpretation-v3"
    elif mutation == "wrong_highlights":
        result["highlights"]["day_ids"] = [7]
    else:
        result["permission"] = "external_booking_change"
    if mutation != "mixed_projection":
        projection = deepcopy(result)
    with pytest.raises(verifier.LiveE2E1VerificationError):
        verifier._verify_approved_interpretation(projection, result)


@pytest.mark.parametrize(
    "mutation",
    [
        "hash",
        "memory_plan",
        "memory_preference",
        "memory_extra",
        "bool",
        "day_bool",
        "extra",
        "candidate",
        "duplicate",
        "universe",
        "revision_bool",
        "missing_alias",
        "extra_alias",
    ],
)
def test_current_v2_forgery_fails_closed(current_context, mutation):
    context = deepcopy(current_context)
    if mutation == "hash":
        context["accepted_plan_content_hash"] = "0" * 16
    elif mutation == "memory_plan":
        context["conversation_memory"]["plan_id"] = "plan_" + "0" * 16
    elif mutation == "memory_preference":
        context["conversation_memory"]["proposed_preferences"] = [
            {"preference": "reduce_driving", "disposition": "accepted"}
        ]
    elif mutation == "memory_extra":
        context["conversation_memory"]["secret"] = "must-not-be-authorized"
    elif mutation == "bool":
        context["capabilities"]["operations"]["add_candidate"]["enabled"] = 1
    elif mutation == "day_bool":
        context["capabilities"]["candidate_day_readiness"][0]["day"] = True
    elif mutation == "extra":
        context["capabilities"]["extra"] = True
    elif mutation == "candidate":
        context["capabilities"]["candidate_replacements"][0]["replaces_stop_id"] = "unknown_stop"
    elif mutation == "duplicate":
        context["capabilities"]["candidate_day_readiness"].append(context["capabilities"]["candidate_day_readiness"][0])
    elif mutation == "revision_bool":
        context["session_revision"] = True
    elif mutation in {"missing_alias", "extra_alias"}:
        target = next(place for place in context["known_places"] if place["place_id"] == verifier.TARGET)
        if mutation == "missing_alias":
            target["aliases"].remove("Bixby")
        else:
            target["aliases"].append("unauthorized alias")
    else:
        context["allowed_stop_ids"].append(verifier.CANDIDATE)
    with pytest.raises(verifier.LiveE2E1VerificationError):
        verify_request(request_for(context))


def test_historical_flag_cannot_authorize_rewritten_v1_context_or_v2_old_instructions(
    current_context, recorded_runtime_inputs
):
    old = json.loads(
        json.loads((recorded_runtime_inputs / OLD).read_text(encoding="utf-8"))["records"][0]["request_body_utf8"]
    )
    old_context = json.loads(
        re.search(r"<workspace_context>\n(.*?)\n</workspace_context>", old["input"], re.S).group(1)
    )
    old_context["accepted_plan_id"] = "plan_" + "0" * 16
    forged = request_for(old_context)
    forged["instructions"] = old["instructions"]
    with pytest.raises(verifier.LiveE2E1VerificationError):
        verify_request(forged, historical=True)
    forged = request_for(current_context)
    forged["instructions"] = old["instructions"]
    with pytest.raises(verifier.LiveE2E1VerificationError):
        verify_request(forged, historical=True)
