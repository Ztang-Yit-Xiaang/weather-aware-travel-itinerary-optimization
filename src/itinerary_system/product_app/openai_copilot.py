"""OpenAI Responses API boundary for typed Copilot interpretation."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Sequence
from dataclasses import replace
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

try:
    from openai import (
        APIConnectionError,
        APIResponseValidationError,
        APITimeoutError,
        AsyncOpenAI,
        AuthenticationError,
        BadRequestError,
        ContentFilterFinishReasonError,
        LengthFinishReasonError,
        PermissionDeniedError,
        RateLimitError,
    )
except ImportError:  # pragma: no cover - exercised by integration packaging checks
    class _UnavailableSDKError(Exception):
        pass

    APIConnectionError = APIResponseValidationError = APITimeoutError = _UnavailableSDKError  # type: ignore[misc]
    AuthenticationError = BadRequestError = PermissionDeniedError = _UnavailableSDKError  # type: ignore[misc]
    ContentFilterFinishReasonError = LengthFinishReasonError = RateLimitError = _UnavailableSDKError  # type: ignore[misc]
    AsyncOpenAI = None  # type: ignore[assignment,misc]

from .copilot import CopilotContractError, validate_copilot_interpretation
from .copilot_prompt import PromptBudgetError, build
from .models import (
    CopilotContextV1,
    CopilotHighlightsV1,
    CopilotIntentV1,
    CopilotInterpretationV1,
    CopilotTurnV1,
)

_INTENT_TYPES = Literal[
    "review_registered_repair",
    "keep_original",
    "keep_stop",
    "lock_stop",
    "mark_flexible",
    "move_day",
    "route_feedback",
    "replace_nearby",
    "add_candidate",
    "request_clarification",
    "unsupported_request",
    "external_action_not_supported",
]
_PREFERENCES = Literal["reduce_contextual_risk", "reduce_driving", "preserve_experience"]
_STATES = Literal[
    "clarification_required",
    "permission_required",
    "proposal_ready",
    "unsupported",
    "refused",
    "failed",
]

_SAFE_MESSAGES = {
    "openai_not_configured": "OpenAI is not configured for this local application.",
    "openai_transport_unavailable": "The OpenAI transport is unavailable.",
    "openai_authentication_failed": "OpenAI authentication or permission failed.",
    "openai_timeout": "OpenAI did not respond within the request deadline.",
    "openai_rate_limited": "OpenAI is temporarily rate limited. Try again later.",
    "openai_quota_exhausted": "OpenAI API quota or credits are unavailable.",
    "openai_provider_unavailable": "OpenAI is temporarily unavailable. Try again later.",
    "openai_invalid_structured_output": "OpenAI returned an unusable structured response.",
}

_SAFE_DIAGNOSTIC_CODES = frozenset(
    {
        "local_not_configured",
        "local_timeout",
        "parsed_output_missing",
        "provider_authentication_rejected",
        "provider_conflict",
        "provider_content_filtered",
        "provider_model_or_endpoint_not_found",
        "provider_output_incomplete",
        "provider_quota_exhausted",
        "provider_rate_limited",
        "provider_request_invalid",
        "provider_response_invalid",
        "provider_unavailable",
        "provider_unprocessable",
        "transport_unavailable",
    }
)
_SAFE_PROVIDER_ERROR_CODE = re.compile(r"[A-Za-z0-9_.:-]{1,64}")
_QUOTA_ERROR_CODES = frozenset(
    {"billing_hard_limit_reached", "insufficient_quota", "usage_limit_reached"}
)


class OpenAICopilotError(RuntimeError):
    """Sanitized provider failure suitable for API-layer status mapping."""

    def __init__(
        self,
        code: str,
        http_status: int,
        *,
        diagnostic_code: str | None = None,
        provider_status: int | None = None,
        provider_error_code: str | None = None,
    ) -> None:
        if code not in _SAFE_MESSAGES:
            raise ValueError("unknown_openai_error_code")
        if diagnostic_code is not None and diagnostic_code not in _SAFE_DIAGNOSTIC_CODES:
            raise ValueError("unknown_openai_diagnostic_code")
        if provider_status is not None and (
            not isinstance(provider_status, int) or not 400 <= provider_status <= 599
        ):
            raise ValueError("invalid_openai_provider_status")
        if provider_error_code is not None and _SAFE_PROVIDER_ERROR_CODE.fullmatch(
            provider_error_code
        ) is None:
            raise ValueError("invalid_openai_provider_error_code")
        self.code = code
        self.http_status = http_status
        self.status_code = http_status
        self.user_message = _SAFE_MESSAGES[code]
        self.diagnostic_code = diagnostic_code
        self.provider_status = provider_status
        self.provider_error_code = provider_error_code
        super().__init__(code)


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class OpenAIIntentSchemaV1(_StrictModel):
    type: _INTENT_TYPES
    target_stop_id: str | None = None
    destination_day: int | None = None
    candidate_id: str | None = None
    preference: _PREFERENCES | None = None
    clarification_question: str | None = Field(default=None, max_length=4_000)


class OpenAIHighlightsSchemaV1(_StrictModel):
    day_ids: list[int] = Field(default_factory=list)
    stop_ids: list[str] = Field(default_factory=list)
    segment_ids: list[str] = Field(default_factory=list)
    candidate_ids: list[str] = Field(default_factory=list)


class OpenAIInterpretationSchemaV1(_StrictModel):
    schema_version: Literal["copilot-interpretation-v1"]
    state: _STATES
    assistant_message: str = Field(min_length=1, max_length=4_000)
    intents: list[OpenAIIntentSchemaV1] = Field(default_factory=list, max_length=4)
    permission: Literal["external_booking_change"] | None = None
    highlights: OpenAIHighlightsSchemaV1 = Field(default_factory=OpenAIHighlightsSchemaV1)


class OpenAICopilotAdapter:
    """Interpret requests with OpenAI while preserving the local decision boundary."""

    provider_name = "openai"
    provider_label = "OpenAI Copilot"

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        timeout_seconds: float = 30,
        history_messages: int = 12,
        history_characters: int = 12_000,
        client: Any | None = None,
    ) -> None:
        if not isinstance(model, str) or not model.strip():
            raise ValueError("openai_model_invalid")
        if not isinstance(api_key, str):
            raise TypeError("api_key must be a string")
        if not isinstance(timeout_seconds, (int, float)) or timeout_seconds <= 0:
            raise ValueError("openai_timeout_invalid")
        if not isinstance(history_messages, int) or not 0 <= history_messages <= 20:
            raise ValueError("openai_history_messages_invalid")
        if not isinstance(history_characters, int) or not 0 <= history_characters <= 20_000:
            raise ValueError("openai_history_characters_invalid")
        self.model = model.strip()
        self._api_key = api_key
        self.timeout_seconds = float(timeout_seconds)
        self.history_messages = history_messages
        self.history_characters = history_characters
        self._client = client

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(model={self.model!r}, "
            f"timeout_seconds={self.timeout_seconds!r})"
        )

    async def interpret(
        self,
        *,
        context: CopilotContextV1,
        history: tuple[CopilotTurnV1, ...],
        message: str,
    ) -> CopilotInterpretationV1:
        validate_copilot_interpretation(
            context=context,
            interpretation=CopilotInterpretationV1(
                state="failed",
                assistant_message="Context validation.",
            ),
        )
        bounded_history = self._bounded_history(history)
        try:
            bundle = build(context, bounded_history, message)
        except (PromptBudgetError, TypeError) as exc:
            raise CopilotContractError from exc

        client = self._require_client()
        try:
            async with asyncio.timeout(self.timeout_seconds):
                response = await self._request_with_retry(client, bundle)
        except asyncio.CancelledError:
            raise
        except TimeoutError as exc:
            raise OpenAICopilotError(
                "openai_timeout", 504, diagnostic_code="local_timeout"
            ) from exc

        if _contains_refusal(response):
            return CopilotInterpretationV1(
                state="refused",
                assistant_message="The provider declined this request. No application state changed.",
            )

        parsed = getattr(response, "output_parsed", None)
        if not isinstance(parsed, OpenAIInterpretationSchemaV1):
            raise OpenAICopilotError(
                "openai_invalid_structured_output",
                502,
                diagnostic_code="parsed_output_missing",
            )
        interpretation = _to_dataclass(parsed)
        # Provider prose is untrusted and never reaches the UI or transcript.
        # The model supplies typed state/intents; user-facing copy is local.
        interpretation = _canonicalize_provider_copy(interpretation)
        return validate_copilot_interpretation(
            context=context,
            interpretation=interpretation,
        )

    def _require_client(self) -> Any:
        if self._client is not None:
            return self._client
        if not self._api_key.strip():
            raise OpenAICopilotError(
                "openai_not_configured", 503, diagnostic_code="local_not_configured"
            )
        if AsyncOpenAI is None:
            raise OpenAICopilotError(
                "openai_transport_unavailable",
                503,
                diagnostic_code="transport_unavailable",
            )
        try:
            self._client = AsyncOpenAI(
                api_key=self._api_key,
                base_url="https://api.openai.com/v1",
                max_retries=0,
                timeout=self.timeout_seconds,
            )
        except Exception as exc:  # Constructor failures are local transport failures.
            raise OpenAICopilotError(
                "openai_transport_unavailable",
                503,
                diagnostic_code="transport_unavailable",
            ) from exc
        return self._client

    async def _request_with_retry(self, client: Any, bundle: Any) -> Any:
        for attempt in range(2):
            try:
                return await client.responses.parse(
                    model=self.model,
                    instructions=bundle.instructions,
                    input=bundle.input,
                    text_format=OpenAIInterpretationSchemaV1,
                    reasoning={
                        "effort": bundle.reasoning_effort,
                        "context": bundle.reasoning_context,
                    },
                    text={"verbosity": bundle.text_verbosity},
                    max_output_tokens=bundle.max_output_tokens,
                    store=False,
                    tools=[],
                    timeout=self.timeout_seconds,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                category = _classify_error(exc)
                if category[0] == "transient" and attempt == 0:
                    delay = _retry_after_seconds(exc)
                    if delay:
                        await asyncio.sleep(delay)
                    continue
                raise _normalized_error(category, exc) from exc
        raise AssertionError("unreachable")

    def _bounded_history(self, history: tuple[CopilotTurnV1, ...]) -> Sequence[dict[str, str]]:
        if not isinstance(history, tuple) or any(not isinstance(turn, CopilotTurnV1) for turn in history):
            raise CopilotContractError
        if self.history_messages == 0 or self.history_characters == 0:
            return ()
        messages: list[dict[str, str]] = []
        for turn in history:
            messages.extend(
                (
                    {"role": "user", "content": turn.user_message},
                    {"role": "assistant", "content": turn.assistant_message},
                )
            )
        selected: list[dict[str, str]] = []
        characters = 0
        for item in reversed(messages[-self.history_messages :]):
            size = len(item["content"])
            if size > self.history_characters:
                continue
            if characters + size > self.history_characters:
                break
            selected.append(item)
            characters += size
        selected.reverse()
        return selected


def _to_dataclass(parsed: OpenAIInterpretationSchemaV1) -> CopilotInterpretationV1:
    return CopilotInterpretationV1(
        schema_version=parsed.schema_version,
        state=parsed.state,
        assistant_message=parsed.assistant_message,
        intents=tuple(
            CopilotIntentV1(
                type=intent.type,
                target_stop_id=intent.target_stop_id,
                destination_day=intent.destination_day,
                candidate_id=intent.candidate_id,
                preference=intent.preference,
                clarification_question=intent.clarification_question,
            )
            for intent in parsed.intents
        ),
        permission=parsed.permission,
        highlights=CopilotHighlightsV1(
            day_ids=tuple(parsed.highlights.day_ids),
            stop_ids=tuple(parsed.highlights.stop_ids),
            segment_ids=tuple(parsed.highlights.segment_ids),
            candidate_ids=tuple(parsed.highlights.candidate_ids),
        ),
    )


def _canonical_assistant_copy(interpretation: CopilotInterpretationV1) -> str:
    if interpretation.state == "proposal_ready":
        return (
            "I prepared a typed itinerary suggestion for your review. "
            "Nothing was changed or executed automatically."
        )
    if interpretation.state == "clarification_required":
        return (
            "I need more itinerary detail before I can prepare a typed suggestion. "
            "Nothing was changed."
        )
    if interpretation.state == "permission_required":
        return (
            "This request involves an external booking or payment action that this "
            "Copilot cannot execute. No external action occurred."
        )
    if interpretation.state == "unsupported":
        return "This request is not supported by the local itinerary tools. Nothing was changed."
    if interpretation.state == "refused":
        return "The provider declined this request. No application state changed."
    return "Copilot could not prepare a safe typed suggestion. No application state changed."


def _canonicalize_provider_copy(
    interpretation: CopilotInterpretationV1,
) -> CopilotInterpretationV1:
    intents = tuple(
        replace(
            intent,
            clarification_question=(
                "Which itinerary day, stop, or trade-off should I review?"
                if intent.type == "request_clarification"
                else intent.clarification_question
            ),
        )
        for intent in interpretation.intents
    )
    return replace(
        interpretation,
        assistant_message=_canonical_assistant_copy(interpretation),
        intents=intents,
    )


def _contains_refusal(response: Any) -> bool:
    for item in getattr(response, "output", ()) or ():
        contents = item.get("content", ()) if isinstance(item, dict) else getattr(item, "content", ())
        for content in contents or ():
            content_type = content.get("type") if isinstance(content, dict) else getattr(content, "type", None)
            if content_type == "refusal":
                return True
    return False


def _classify_error(exc: Exception) -> tuple[str, int | None]:
    status = getattr(exc, "status_code", None)
    provider_error_code = _safe_provider_error_code(exc)
    if status == 429 and provider_error_code in _QUOTA_ERROR_CODES:
        return ("quota", status)
    if isinstance(exc, (AuthenticationError, PermissionDeniedError)) or status in {401, 403}:
        return ("authentication", status)
    if isinstance(exc, RateLimitError) or status == 429:
        return ("transient", 429)
    if isinstance(exc, (APIConnectionError, APITimeoutError)):
        return ("transient", status)
    if isinstance(status, int) and 500 <= status <= 599:
        return ("transient", status)
    if isinstance(
        exc,
        (
            APIResponseValidationError,
            BadRequestError,
            ContentFilterFinishReasonError,
            LengthFinishReasonError,
            ValidationError,
        ),
    ) or status in {400, 404, 409, 422}:
        return ("invalid", status)
    return ("transport", status)


def _normalized_error(
    category: tuple[str, int | None], exc: Exception
) -> OpenAICopilotError:
    kind, status = category
    provider_error_code = _safe_provider_error_code(exc)
    if kind == "authentication":
        return OpenAICopilotError(
            "openai_authentication_failed",
            503,
            diagnostic_code="provider_authentication_rejected",
            provider_status=status,
            provider_error_code=provider_error_code,
        )
    if kind == "quota":
        return OpenAICopilotError(
            "openai_quota_exhausted",
            429,
            diagnostic_code="provider_quota_exhausted",
            provider_status=status,
            provider_error_code=provider_error_code,
        )
    if kind == "invalid":
        return OpenAICopilotError(
            "openai_invalid_structured_output",
            502,
            diagnostic_code=_invalid_diagnostic_code(exc, status),
            provider_status=status,
            provider_error_code=provider_error_code,
        )
    if kind == "transient" and status == 429:
        return OpenAICopilotError(
            "openai_rate_limited",
            429,
            diagnostic_code="provider_rate_limited",
            provider_status=status,
            provider_error_code=provider_error_code,
        )
    if kind in {"transient", "transport"}:
        return OpenAICopilotError(
            "openai_provider_unavailable",
            503,
            diagnostic_code=(
                "provider_unavailable" if status is not None else "transport_unavailable"
            ),
            provider_status=status,
            provider_error_code=provider_error_code,
        )
    raise AssertionError("unknown error category")


def _invalid_diagnostic_code(exc: Exception, status: int | None) -> str:
    if isinstance(exc, ContentFilterFinishReasonError):
        return "provider_content_filtered"
    if isinstance(exc, LengthFinishReasonError):
        return "provider_output_incomplete"
    if isinstance(exc, (APIResponseValidationError, ValidationError)):
        return "provider_response_invalid"
    return {
        400: "provider_request_invalid",
        404: "provider_model_or_endpoint_not_found",
        409: "provider_conflict",
        422: "provider_unprocessable",
    }.get(status, "provider_response_invalid")


def _safe_provider_error_code(exc: Exception) -> str | None:
    candidates: list[Any] = [getattr(exc, "code", None)]
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        candidates.append(body.get("code"))
        error = body.get("error")
        if isinstance(error, dict):
            candidates.append(error.get("code"))
    for candidate in candidates:
        if isinstance(candidate, str) and _SAFE_PROVIDER_ERROR_CODE.fullmatch(candidate):
            return candidate
    return None


def _retry_after_seconds(exc: Exception) -> float:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None:
        return 0.0
    raw = headers.get("retry-after") or headers.get("Retry-After")
    try:
        return min(max(float(raw), 0.0), 2.0)
    except (TypeError, ValueError):
        return 0.0
