from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from openai import APIConnectionError
from pydantic import ValidationError

from itinerary_system.product_app import openai_copilot
from itinerary_system.product_app.copilot import CopilotContractError
from itinerary_system.product_app.models import CopilotContextV1, CopilotIntentV1, CopilotTurnV1
from itinerary_system.product_app.openai_copilot import (
    OpenAICopilotAdapter,
    OpenAICopilotError,
    OpenAIHighlightsSchemaV1,
    OpenAIIntentSchemaV1,
    OpenAIInterpretationSchemaV1,
)


def _context() -> CopilotContextV1:
    return CopilotContextV1(
        run_id="run_demo",
        trip_id="trip_demo",
        session_id="session_demo",
        session_revision=4,
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


def _parsed_response(**overrides: Any) -> SimpleNamespace:
    values: dict[str, Any] = {
        "schema_version": "copilot-interpretation-v1",
        "state": "proposal_ready",
        "assistant_message": "Review the typed move before adding it to the draft.",
        "intents": [
            OpenAIIntentSchemaV1(
                type="move_day",
                target_stop_id="stop_a",
                destination_day=3,
            )
        ],
        "highlights": OpenAIHighlightsSchemaV1(day_ids=[2], stop_ids=["stop_a"]),
    }
    values.update(overrides)
    return SimpleNamespace(
        output_parsed=OpenAIInterpretationSchemaV1(**values),
        output=[],
    )


class _FakeResponses:
    def __init__(self, outcomes: list[Any]) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[dict[str, Any]] = []

    async def parse(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        if callable(outcome):
            return await outcome()
        return outcome


class _FakeClient:
    def __init__(self, *outcomes: Any) -> None:
        self.responses = _FakeResponses(list(outcomes))


class _StatusError(Exception):
    def __init__(
        self,
        status_code: int,
        message: str = "provider detail must stay private",
        *,
        code: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.body = {"error": {"code": code}} if code is not None else None
        self.response = SimpleNamespace(headers={})


def _run(adapter: OpenAICopilotAdapter, message: str = "Move this stop to day 3") -> Any:
    return asyncio.run(adapter.interpret(context=_context(), history=(), message=message))


def _turn(index: int, characters: int = 700) -> CopilotTurnV1:
    return CopilotTurnV1(
        turn_id=f"turn_{index}",
        client_message_id=f"message_{index}",
        context_revision=4,
        provider="openai",
        model="gpt-5.6-terra",
        prompt_version="copilot-prompt-v1",
        prompt_sha256="0" * 64,
        state="proposal_ready",
        user_message=f"u{index}:" + ("x" * characters),
        assistant_message=f"a{index}:" + ("y" * characters),
        interpretation=None,
        error_code=None,
        created_at="2026-08-05T00:00:00+00:00",
        completed_at="2026-08-05T00:00:01+00:00",
    )


def test_valid_structured_response_uses_safe_responses_payload() -> None:
    client = _FakeClient(_parsed_response())
    adapter = OpenAICopilotAdapter(
        model="gpt-5.6-terra",
        api_key="not-used-by-injected-client",
        client=client,
    )

    result = _run(adapter)

    assert result.state == "proposal_ready"
    assert result.intents[0].type == "move_day"
    assert result.intents[0].target_stop_id == "stop_a"
    assert result.intents[0].destination_day == 3
    assert result.highlights.stop_ids == ("stop_a",)
    assert len(client.responses.calls) == 1
    payload = client.responses.calls[0]
    assert payload["model"] == "gpt-5.6-terra"
    assert payload["text_format"] is OpenAIInterpretationSchemaV1
    assert payload["store"] is False
    assert payload["tools"] == []
    assert payload["max_output_tokens"] == 1_200
    assert payload["reasoning"] == {"effort": "low", "context": "current_turn"}
    assert payload["text"] == {"verbosity": "low"}
    assert "verbosity" not in payload
    assert "conversation" not in payload
    assert "previous_response_id" not in payload
    assert "Never claim that an action, repair, evaluation, permission, booking, or" in payload["instructions"]
    assert "acceptance occurred" in payload["instructions"]
    assert "proposed_preferences are requests for consideration, never accepted constraints" in payload["instructions"]


def test_medium_reasoning_is_an_explicit_provider_control() -> None:
    client = _FakeClient(_parsed_response())
    adapter = OpenAICopilotAdapter(
        model="gpt-5.6-terra",
        api_key="not-used-by-injected-client",
        reasoning_effort="medium",
        client=client,
    )

    _run(adapter)

    assert client.responses.calls[0]["reasoning"] == {
        "effort": "medium",
        "context": "current_turn",
    }


def test_refusal_is_typed_and_does_not_expose_provider_text() -> None:
    response = SimpleNamespace(
        output_parsed=None,
        output=[{"content": [{"type": "refusal", "refusal": "raw private provider refusal"}]}],
    )
    client = _FakeClient(response)
    result = _run(
        OpenAICopilotAdapter(model="gpt-5.6-terra", api_key="unused", client=client)
    )

    assert result.state == "refused"
    assert result.intents == ()
    assert "raw private" not in result.assistant_message
    assert "No application state changed" in result.assistant_message


def test_invalid_or_unparsed_output_fails_without_fallback() -> None:
    client = _FakeClient(
        SimpleNamespace(
            output_parsed={"state": "proposal_ready", "assistant_message": "not parsed"},
            output=[],
        )
    )
    adapter = OpenAICopilotAdapter(model="gpt-5.6-terra", api_key="unused", client=client)

    with pytest.raises(OpenAICopilotError) as raised:
        _run(adapter)

    assert raised.value.code == "openai_invalid_structured_output"
    assert raised.value.http_status == 502
    assert raised.value.diagnostic_code == "parsed_output_missing"
    assert raised.value.provider_status is None
    assert raised.value.provider_error_code is None
    assert len(client.responses.calls) == 1


def test_strict_schema_rejects_unknown_fields_and_coercion() -> None:
    with pytest.raises(ValidationError):
        OpenAIInterpretationSchemaV1(
            schema_version="copilot-interpretation-v1",
            state="proposal_ready",
            assistant_message="Invalid extra field.",
            invented_action=True,
        )
    with pytest.raises(ValidationError):
        OpenAIInterpretationSchemaV1(
            schema_version="copilot-interpretation-v1",
            state="proposal_ready",
            assistant_message="Invalid day coercion.",
            intents=[{"type": "move_day", "target_stop_id": "stop_a", "destination_day": "3"}],
        )


def test_valid_schema_with_invented_entity_fails_local_contract() -> None:
    client = _FakeClient(
        _parsed_response(
            intents=[OpenAIIntentSchemaV1(type="keep_stop", target_stop_id="invented_stop")]
        )
    )

    with pytest.raises(CopilotContractError, match="copilot_interpretation_invalid"):
        _run(OpenAICopilotAdapter(model="gpt-5.6-terra", api_key="unused", client=client))
    assert len(client.responses.calls) == 1


@pytest.mark.parametrize(
    "provider_claim",
    (
        "I booked and paid successfully.",
        "Your tickets are secured.",
        "The booking is now complete.",
        "Payment went through and the reservation is ready.",
    ),
)
def test_provider_prose_is_replaced_by_canonical_local_copy(provider_claim: str) -> None:
    client = _FakeClient(_parsed_response(assistant_message=provider_claim))

    result = _run(
        OpenAICopilotAdapter(model="gpt-5.6-terra", api_key="unused", client=client)
    )

    assert result.assistant_message == (
        "I prepared a typed itinerary suggestion for your review. "
        "Nothing was changed or executed automatically."
    )
    assert provider_claim not in result.assistant_message


def test_removal_request_cannot_be_rewritten_as_mark_flexible() -> None:
    client = _FakeClient(
        _parsed_response(
            intents=[
                OpenAIIntentSchemaV1(
                    type="mark_flexible",
                    target_stop_id="stop_a",
                )
            ],
            highlights=OpenAIHighlightsSchemaV1(stop_ids=["stop_a"]),
        )
    )

    result = _run(
        OpenAICopilotAdapter(model="gpt-5.6-terra", api_key="unused", client=client),
        message="I do not want to go to stop A",
    )

    assert result.state == "unsupported"
    assert [intent.type for intent in result.intents] == ["unsupported_request"]
    assert result.highlights.stop_ids == ()
    assert "Removing or excluding" in result.assistant_message
    assert client.responses.calls == []


@pytest.mark.parametrize(
    "message",
    [
        "I do not want to go to the Stearns Wharf",
        "Skip stop A",
        "Exclude stop A from this trip",
    ],
)
def test_removal_language_is_rejected_before_provider_call(message: str) -> None:
    client = _FakeClient(_parsed_response())

    result = _run(
        OpenAICopilotAdapter(model="gpt-5.6-terra", api_key="unused", client=client),
        message=message,
    )

    assert result.state == "unsupported"
    assert [intent.type for intent in result.intents] == ["unsupported_request"]
    assert client.responses.calls == []


def test_staged_point_removal_returns_specific_undo_guidance_without_provider_call() -> None:
    context = replace(
        _context(),
        draft_operations=(
            {
                "type": "add_custom_waypoint",
                "target": "waypoint_user_1",
                "parameters": {"name": "Added point"},
            },
        ),
    )
    client = _FakeClient(_parsed_response())

    result = asyncio.run(
        OpenAICopilotAdapter(
            model="gpt-5.4-mini", api_key="unused", client=client
        ).interpret(
            context=context,
            history=(),
            message="remove the added point",
        )
    )

    assert result.state == "unsupported"
    assert [intent.type for intent in result.intents] == ["unsupported_request"]
    assert "staged draft" in result.assistant_message
    assert "Remove this draft waypoint" in result.assistant_message
    assert "Undo latest draft" in result.assistant_message
    assert client.responses.calls == []


def test_human_bixby_inclusion_resolves_to_exact_accepted_stop_without_provider_call() -> None:
    stop_id = "bixby_creek_bridge_viewpoint"
    context = replace(
        _context(),
        selected_stop_id=None,
        allowed_stop_ids=(stop_id,),
        known_places=(
            {
                "place_id": stop_id,
                "name": "Bixby Creek Bridge Viewpoint",
                "aliases": ("Bixby Creek Bridge",),
                "categories": ("itinerary_stop",),
                "region_id": "big_sur",
                "route_status": "accepted_stop",
            },
        ),
    )
    client = _FakeClient(_parsed_response())

    result = asyncio.run(
        OpenAICopilotAdapter(
            model="gpt-5.4-mini", api_key="unused", client=client
        ).interpret(
            context=context,
            history=(),
            message="I want to go to Bixby Creek Bridge",
        )
    )

    assert result.state == "proposal_ready"
    assert result.intents == (CopilotIntentV1(type="keep_stop", target_stop_id=stop_id),)
    assert result.highlights.stop_ids == (stop_id,)
    assert "already in the accepted itinerary" in result.assistant_message
    assert client.responses.calls == []


def test_explicit_replacement_language_still_reaches_typed_provider_boundary() -> None:
    client = _FakeClient(
        _parsed_response(
            intents=[
                OpenAIIntentSchemaV1(
                    type="replace_nearby",
                    target_stop_id="stop_a",
                    candidate_id="candidate_b",
                )
            ]
        )
    )

    result = _run(
        OpenAICopilotAdapter(model="gpt-5.6-terra", api_key="unused", client=client),
        message="Remove stop A and replace it with candidate B",
    )

    assert result.state == "proposal_ready"
    assert [intent.type for intent in result.intents] == ["replace_nearby"]
    assert len(client.responses.calls) == 1


def test_provider_clarification_prose_is_replaced_before_persistence_boundary() -> None:
    provider_claim = "Great news - your tickets are secured."
    client = _FakeClient(
        _parsed_response(
            state="clarification_required",
            assistant_message=provider_claim,
            intents=[
                OpenAIIntentSchemaV1(
                    type="request_clarification",
                    clarification_question=provider_claim,
                )
            ],
            highlights=OpenAIHighlightsSchemaV1(),
        )
    )

    result = _run(
        OpenAICopilotAdapter(model="gpt-5.6-terra", api_key="unused", client=client)
    )

    assert result.assistant_message.startswith("I need more itinerary detail")
    assert result.intents[0].clarification_question == (
        "Which itinerary day, stop, or trade-off should I review?"
    )
    assert provider_claim not in str(result.as_dict())


@pytest.mark.parametrize(
    ("status", "final_code"),
    [(429, "openai_rate_limited"), (500, "openai_provider_unavailable")],
)
def test_transient_status_does_not_silently_retry(status: int, final_code: str) -> None:
    success_client = _FakeClient(_StatusError(status), _parsed_response())
    with pytest.raises(OpenAICopilotError) as raised:
        _run(OpenAICopilotAdapter(model="gpt-5.6-terra", api_key="unused", client=success_client))
    assert raised.value.code == final_code
    assert len(success_client.responses.calls) == 1

    failed_client = _FakeClient(_StatusError(status), _StatusError(status))
    with pytest.raises(OpenAICopilotError) as raised:
        _run(OpenAICopilotAdapter(model="gpt-5.6-terra", api_key="unused", client=failed_client))
    assert raised.value.code == final_code
    assert len(failed_client.responses.calls) == 1


def test_transient_network_failure_does_not_silently_retry() -> None:
    request = httpx.Request("POST", "https://api.openai.com/v1/responses")
    client = _FakeClient(APIConnectionError(request=request), _parsed_response())

    with pytest.raises(OpenAICopilotError, match="openai_provider_unavailable"):
        _run(OpenAICopilotAdapter(model="gpt-5.6-terra", api_key="unused", client=client))
    assert len(client.responses.calls) == 1


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_non_transient_status_is_not_retried_or_exposed(status: int) -> None:
    client = _FakeClient(_StatusError(status, "secret raw provider body"), _parsed_response())
    with pytest.raises(OpenAICopilotError) as raised:
        _run(OpenAICopilotAdapter(model="gpt-5.6-terra", api_key="unused", client=client))

    expected = "openai_authentication_failed" if status in {401, 403} else "openai_invalid_structured_output"
    assert raised.value.code == expected
    assert str(raised.value) == expected
    assert "secret raw provider body" not in raised.value.user_message
    assert len(client.responses.calls) == 1


def test_safe_provider_diagnostics_distinguish_model_access_without_raw_body() -> None:
    client = _FakeClient(
        _StatusError(
            404,
            "raw provider message with private request details",
            code="model_not_found",
        )
    )

    with pytest.raises(OpenAICopilotError) as raised:
        _run(OpenAICopilotAdapter(model="gpt-5.6-terra", api_key="unused", client=client))

    error = raised.value
    assert error.code == "openai_invalid_structured_output"
    assert error.diagnostic_code == "provider_model_or_endpoint_not_found"
    assert error.provider_status == 404
    assert error.provider_error_code == "model_not_found"
    assert str(error) == "openai_invalid_structured_output"
    assert "private request" not in error.user_message
    assert "private request" not in repr(error)


def test_quota_exhaustion_is_not_retried_or_mislabeled_as_rate_limit() -> None:
    client = _FakeClient(_StatusError(429, code="insufficient_quota"), _parsed_response())

    with pytest.raises(OpenAICopilotError) as raised:
        _run(OpenAICopilotAdapter(model="gpt-5.6-terra", api_key="unused", client=client))

    error = raised.value
    assert error.code == "openai_quota_exhausted"
    assert error.diagnostic_code == "provider_quota_exhausted"
    assert error.provider_status == 429
    assert error.provider_error_code == "insufficient_quota"
    assert len(client.responses.calls) == 1


def test_untrusted_provider_error_code_is_discarded() -> None:
    client = _FakeClient(_StatusError(400, code="invalid code containing spaces"))

    with pytest.raises(OpenAICopilotError) as raised:
        _run(OpenAICopilotAdapter(model="gpt-5.6-terra", api_key="unused", client=client))

    assert raised.value.diagnostic_code == "provider_request_invalid"
    assert raised.value.provider_status == 400
    assert raised.value.provider_error_code is None


def test_total_deadline_maps_to_timeout_without_extra_attempt() -> None:
    async def slow_response() -> Any:
        await asyncio.sleep(0.05)
        return _parsed_response()

    client = _FakeClient(slow_response)
    adapter = OpenAICopilotAdapter(
        model="gpt-5.6-terra",
        api_key="unused",
        timeout_seconds=0.005,
        client=client,
    )

    with pytest.raises(OpenAICopilotError) as raised:
        _run(adapter)
    assert raised.value.code == "openai_timeout"
    assert len(client.responses.calls) == 1


def test_cancellation_is_propagated_without_retry_or_fabricated_result() -> None:
    client = _FakeClient(asyncio.CancelledError(), _parsed_response())
    adapter = OpenAICopilotAdapter(model="gpt-5.6-terra", api_key="unused", client=client)

    with pytest.raises(asyncio.CancelledError):
        _run(adapter)
    assert len(client.responses.calls) == 1


def test_missing_key_fails_before_transport_without_secret_in_repr() -> None:
    adapter = OpenAICopilotAdapter(model="gpt-5.6-terra", api_key="")
    with pytest.raises(OpenAICopilotError) as raised:
        _run(adapter)
    assert raised.value.code == "openai_not_configured"

    secret_adapter = OpenAICopilotAdapter(model="gpt-5.6-terra", api_key="super-secret")
    assert "super-secret" not in repr(secret_adapter)


def test_sdk_client_disables_automatic_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}
    client = _FakeClient(_parsed_response())

    def factory(**kwargs: Any) -> _FakeClient:
        captured.update(kwargs)
        return client

    monkeypatch.setattr(openai_copilot, "AsyncOpenAI", factory)
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:9999/v1/")
    result = _run(OpenAICopilotAdapter(model="gpt-5.6-terra", api_key="configured-key"))

    assert result.state == "proposal_ready"
    assert captured["max_retries"] == 0
    assert captured["timeout"] == 30
    assert captured["base_url"] == "https://api.openai.com/v1"


@pytest.mark.parametrize(("messages", "characters"), [(0, 12_000), (12, 0), (0, 0)])
def test_zero_history_limits_are_valid_and_disable_history(messages: int, characters: int) -> None:
    client = _FakeClient(_parsed_response())
    adapter = OpenAICopilotAdapter(
        model="gpt-5.6-terra",
        api_key="unused",
        history_messages=messages,
        history_characters=characters,
        client=client,
    )

    _run(adapter)

    assert "<recent_conversation>\n[]\n</recent_conversation>" in client.responses.calls[0]["input"]


def test_runtime_history_bounds_accept_config_range_while_prompt_remains_capped() -> None:
    client = _FakeClient(_parsed_response())
    adapter = OpenAICopilotAdapter(
        model="gpt-5.6-terra",
        api_key="unused",
        history_messages=20,
        history_characters=20_000,
        client=client,
    )

    asyncio.run(
        adapter.interpret(
            context=_context(),
            history=tuple(_turn(index) for index in range(10)),
            message="Move this stop to day 3",
        )
    )

    assert len(client.responses.calls) == 1
    prompt_input = client.responses.calls[0]["input"]
    encoded_history = prompt_input.split("<recent_conversation>\n", 1)[1].split(
        "\n</recent_conversation>", 1
    )[0]
    outbound_history = json.loads(encoded_history)
    assert len(outbound_history) == 12
    assert sum(len(item["content"]) for item in outbound_history) <= 12_000


@pytest.mark.parametrize(("messages", "characters"), [(-1, 1), (21, 1), (1, -1), (1, 20_001)])
def test_runtime_history_bounds_reject_out_of_range_values(messages: int, characters: int) -> None:
    with pytest.raises(ValueError):
        OpenAICopilotAdapter(
            model="gpt-5.6-terra",
            api_key="unused",
            history_messages=messages,
            history_characters=characters,
        )


def test_invalid_context_is_rejected_before_provider_call() -> None:
    client = _FakeClient(_parsed_response())
    adapter = OpenAICopilotAdapter(model="gpt-5.6-terra", api_key="unused", client=client)
    invalid_context = replace(_context(), selected_stop_id="outside_universe")

    with pytest.raises(CopilotContractError):
        asyncio.run(adapter.interpret(context=invalid_context, history=(), message="Move it"))
    assert client.responses.calls == []

def test_known_statewide_place_gets_specific_local_clarification_without_action() -> None:
    context = replace(
        _context(),
        selected_stop_id=None,
        selected_candidate_id=None,
        known_places=(
            {
                "place_id": "ca_region_redwood",
                "name": "Redwood National and State Parks",
                "aliases": ("Redwood", "Redwoods", "Red Wood"),
                "categories": ("statewide_discovery",),
                "region_id": "redwood",
                "route_status": "discovery_only_route_validation_required",
            },
        ),
    )
    response = _parsed_response(
        state="clarification_required",
        intents=[
            OpenAIIntentSchemaV1(
                type="request_clarification",
                clarification_question="Which place?",
            )
        ],
        highlights=OpenAIHighlightsSchemaV1(),
    )
    adapter = OpenAICopilotAdapter(
        model="gpt-5.6-terra", api_key="unused", client=_FakeClient(response)
    )

    result = asyncio.run(
        adapter.interpret(
            context=context,
            history=(),
            message="what if I want to visit red wood",
        )
    )

    assert result.state == "clarification_required"
    assert result.highlights.day_ids == ()
    assert result.highlights.stop_ids == ()
    assert result.highlights.candidate_ids == ()
    assert "Redwood National and State Parks" in result.assistant_message
    assert "not validated for this trip" in result.assistant_message
    assert result.intents[0].type == "request_clarification"
    assert "which day should I check" in result.intents[0].clarification_question
    assert "typed suggestion" not in result.assistant_message
