"""Prerequisite semantics, including suffixes and selected-stop wording."""

import asyncio
import json
import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_chat_stability_variations import accepted_context as accepted_context
from test_chat_stability_variations import semantic_intents

from itinerary_system.product_app.copilot import DeterministicCopilotAdapter
from itinerary_system.product_app.openai_copilot import OpenAICopilotAdapter, OpenAIInterpretationSchemaV2

CASES = json.loads((Path(__file__).parent / "fixtures/chat_condition_prerequisites.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("provider", ["deterministic", "openai"])
@pytest.mark.parametrize("case", CASES, ids=lambda row: row["id"])
def test_prerequisites_are_not_discarded(accepted_context, case, provider):
    context = replace(accepted_context, selected_stop_id=case["selected_stop_id"])
    calls = []

    async def parse(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            output=[],
            output_parsed=OpenAIInterpretationSchemaV2(
                schema_version="copilot-interpretation-v2",
                state="proposal_ready",
                assistant_message="Scripted unconditional response; not a live model result.",
                intents=case["provider_intents"],
            ),
        )

    adapter = (
        DeterministicCopilotAdapter()
        if provider == "deterministic"
        else OpenAICopilotAdapter(
            model="fake-model", api_key="unused", client=SimpleNamespace(responses=SimpleNamespace(parse=parse))
        )
    )
    result = asyncio.run(adapter.interpret(context=context, history=(), message=case["message"]))
    directory = os.environ.get("CONDITION_VARIANT_EVIDENCE")
    if directory:
        destination = Path(directory)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / (case["id"] + "-" + provider + ".json")).write_text(
            json.dumps(
                {
                    **case,
                    "provider": provider,
                    "actual": result.as_dict(),
                    "fake_calls": len(calls),
                    "paid_calls": 0,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    assert result.state == case["expected_state"], result.as_dict()
    if case["conditional"]:
        assert all(intent.type == "request_clarification" for intent in result.intents)
    else:
        assert semantic_intents(result) == case["provider_intents"]
    if provider == "openai":
        assert len(calls) == case["expected_fake_calls"]
