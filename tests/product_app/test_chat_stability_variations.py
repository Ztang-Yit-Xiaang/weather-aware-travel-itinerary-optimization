"""Semantic regression corpus: actual adapters, local artifacts, no paid calls."""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from itinerary_system.product_app.copilot import DeterministicCopilotAdapter
from itinerary_system.product_app.openai_copilot import OpenAICopilotAdapter, OpenAIInterpretationSchemaV2
from itinerary_system.product_app.registry import ProductRunRegistry
from itinerary_system.product_app.service import ProductService

ROOT = Path(__file__).resolve().parents[2]
CASES = json.loads((Path(__file__).parent / "fixtures/chat_stability_variations.json").read_text(encoding="utf-8"))
CASES += json.loads((Path(__file__).parent / "fixtures/chat_stability_boundary.json").read_text(encoding="utf-8"))
RECORDED = json.loads((Path(__file__).parent / "fixtures/chat_stability_recorded.json").read_text(encoding="utf-8"))
PROVIDER_MUTATIONS = json.loads(
    (Path(__file__).parent / "fixtures/chat_stability_provider_mutations.json").read_text(encoding="utf-8")
)


@pytest.fixture(scope="module")
def accepted_context(tmp_path_factory):
    registry = ProductRunRegistry(ROOT, ROOT / "configs/product_app_registry.json")
    service = ProductService(
        registry, tmp_path_factory.mktemp("semantic-corpus"), copilot=DeterministicCopilotAdapter()
    )
    session, _, _ = service.create_session(registry.default.run_id)
    return service.copilot_context(session)


def semantic_intents(result):
    return [{key: value for key, value in intent.as_dict().items() if value is not None} for intent in result.intents]


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
def test_variations_preserve_exact_meaning_in_both_adapters(accepted_context, case):
    deterministic = asyncio.run(
        DeterministicCopilotAdapter().interpret(
            context=accepted_context,
            history=(),
            message=case["message"],
        )
    )
    assert deterministic.state in case["expected_states"], deterministic.assistant_message
    expected = case["expected_intents"]
    if expected is not None:
        assert semantic_intents(deterministic) == expected
    else:
        assert all(intent.type in {"request_clarification", "unsupported_request"} for intent in deterministic.intents)

    calls = []

    async def parse(**kwargs):
        calls.append(kwargs)
        assert expected is not None, "Locally unsafe requests must not reach the provider"
        return SimpleNamespace(
            output=[],
            output_parsed=OpenAIInterpretationSchemaV2(
                schema_version="copilot-interpretation-v2",
                state="proposal_ready",
                assistant_message="Injected transport; this is not a live model response.",
                intents=expected,
            ),
        )

    adapter = OpenAICopilotAdapter(
        model="fake-model", api_key="unused", client=SimpleNamespace(responses=SimpleNamespace(parse=parse))
    )
    output = asyncio.run(adapter.interpret(context=accepted_context, history=(), message=case["message"]))
    assert output.state in case["expected_states"], output.assistant_message
    assert len(calls) == case.get("expected_fake_calls", 1 if expected is not None else 0)
    if expected is not None:
        assert semantic_intents(output) == expected
    else:
        assert all(intent.type in {"request_clarification", "unsupported_request"} for intent in output.intents)
    directory = os.environ.get("CHAT_STABILITY_EVIDENCE_DIR")
    if directory:
        destination = Path(directory)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / (case["id"] + ".json")).write_text(
            json.dumps(
                {
                    **case,
                    "mode": "deterministic_and_injected_sdk",
                    "paid_provider_calls": 0,
                    "fake_transport_calls": len(calls),
                    "deterministic_result": deterministic.as_dict(),
                    "openai_adapter_result": output.as_dict(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )


@pytest.mark.parametrize("mutation", ["retarget", "value", "extra"])
def test_single_edit_provider_mismatch_cannot_become_a_suggestion(accepted_context, mutation):
    intent = {
        "type": "set_stop_duration",
        "target_stop_id": "griffith_observatory",
        "parameters": {
            "duration": {"mode": "exact", "preferred_minutes": 75, "minimum_minutes": 75, "maximum_minutes": 75}
        },
    }
    if mutation == "retarget":
        intent["target_stop_id"] = "stearns_wharf"
    elif mutation == "value":
        intent["parameters"]["duration"] = {
            "mode": "exact",
            "preferred_minutes": 90,
            "minimum_minutes": 90,
            "maximum_minutes": 90,
        }
    intents = [intent]
    if mutation == "extra":
        intents.append({"type": "keep_stop", "target_stop_id": "golden_gate_bridge"})

    async def parse(**kwargs):
        return SimpleNamespace(
            output=[],
            output_parsed=OpenAIInterpretationSchemaV2(
                schema_version="copilot-interpretation-v2",
                state="proposal_ready",
                assistant_message="Injected",
                intents=intents,
            ),
        )

    adapter = OpenAICopilotAdapter(
        model="fake-model", api_key="unused", client=SimpleNamespace(responses=SimpleNamespace(parse=parse))
    )
    result = asyncio.run(
        adapter.interpret(
            context=accepted_context, history=(), message="I'd like 75 minutes at Griffith Observatory instead of 45."
        )
    )
    assert result.state == "clarification_required"
    assert all(intent.type == "request_clarification" for intent in result.intents)


@pytest.mark.parametrize("case", RECORDED, ids=lambda case: case["id"])
def test_recorded_live_requests_at_the_current_offline_boundary(accepted_context, case):
    """Replay expected typed replies; this does not measure a live model's accuracy."""
    calls = []

    async def parse(**kwargs):
        calls.append(kwargs)
        if case["id"] == "C10":
            payload = {
                "state": "permission_required",
                "permission": "external_booking_change",
                "intents": [{"type": "external_action_not_supported"}],
            }
        else:
            assert case["expected_intents"] is not None
            payload = {"state": case["expected_state"], "intents": case["expected_intents"]}
        return SimpleNamespace(
            output=[],
            output_parsed=OpenAIInterpretationSchemaV2(
                schema_version="copilot-interpretation-v2", assistant_message="Injected expected reply.", **payload
            ),
        )

    adapter = OpenAICopilotAdapter(
        model="fake-model", api_key="unused", client=SimpleNamespace(responses=SimpleNamespace(parse=parse))
    )
    result = asyncio.run(adapter.interpret(context=accepted_context, history=(), message=case["prompt"]))
    assert result.state == case["expected_state"], result.assistant_message
    if case["expected_intents"] is not None:
        assert semantic_intents(result) == case["expected_intents"]
    if case["id"] == "C13":
        assert "nature" in result.assistant_message and "not supported" in result.assistant_message
    if not case["requires_provider_interpretation"] and case["expected_state"] != "proposal_ready":
        assert calls == []
    assert len(calls) <= 1


@pytest.mark.parametrize("case", PROVIDER_MUTATIONS, ids=lambda case: case["id"])
def test_provider_reply_preserves_complete_cross_language_request(accepted_context, case):
    """A valid schema cannot authorize dropped, changed or invented edits."""
    selection = case.get("context_selection")
    if selection:
        accepted_context = replace(accepted_context, **selection)
    local = asyncio.run(
        DeterministicCopilotAdapter().interpret(
            context=accepted_context,
            history=(),
            message=case["message"],
        )
    )
    if case["expected_local_intents"] is not None:
        assert local.state == "proposal_ready", local.assistant_message
        assert semantic_intents(local) == case["expected_local_intents"]
    else:
        assert local.state in {"clarification_required", "unsupported"}
        assert all(intent.type in {"request_clarification", "unsupported_request"} for intent in local.intents)
    calls = []

    async def parse(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            output=[],
            output_parsed=OpenAIInterpretationSchemaV2(
                schema_version="copilot-interpretation-v2",
                state="proposal_ready",
                assistant_message="Injected reply mutation; no live model call.",
                intents=case["provider_intents"],
            ),
        )

    adapter = OpenAICopilotAdapter(
        model="fake-model", api_key="unused", client=SimpleNamespace(responses=SimpleNamespace(parse=parse))
    )
    output = asyncio.run(adapter.interpret(context=accepted_context, history=(), message=case["message"]))
    assert output.state in case["expected_states"], output.assistant_message
    assert len(calls) == case["expected_fake_calls"]
    if case["expected_intents"] is not None:
        assert semantic_intents(output) == case["expected_intents"]
    else:
        assert all(intent.type in {"request_clarification", "unsupported_request"} for intent in output.intents)
    directory = os.environ.get("CHAT_STABILITY_EVIDENCE_DIR")
    if directory:
        destination = Path(directory) / "provider_mutations"
        destination.mkdir(parents=True, exist_ok=True)
        (destination / (case["id"] + ".json")).write_text(
            json.dumps(
                {
                    **case,
                    "mode": "deterministic_and_injected_sdk",
                    "paid_provider_calls": 0,
                    "fake_transport_calls": len(calls),
                    "deterministic_result": local.as_dict(),
                    "openai_adapter_result": output.as_dict(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
