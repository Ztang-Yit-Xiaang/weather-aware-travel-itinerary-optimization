"""OpenAI Responses API boundary for typed Copilot interpretation."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Sequence
from dataclasses import replace
from typing import Any, Literal, get_args

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .trip_planning import IndividualizedTripParameters, TripPlanParameters, YosemiteExtensionParameters

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

from .copilot import (
    CopilotContractError,
    is_stop_removal_request,
    is_stop_replacement_request,
    known_place_clarification,
    matched_known_place,
    request_semantic_guard,
    unsupported_stop_removal,
    validate_copilot_interpretation,
)
from .copilot_prompt import PromptBudgetError, build
from .models import (
    CopilotContextV1,
    CopilotContextV2,
    CopilotHighlightsV1,
    CopilotIntentV1,
    CopilotIntentV2,
    CopilotInterpretationV1,
    CopilotInterpretationV2,
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
_DRAFT_POINT_REMOVAL_REQUEST = re.compile(
    r"\b(?:remove|delete|undo)\b.*\b(?:added|draft|staged|custom|route)\s+"
    r"(?:point|stop|waypoint)\b|\bundo\b.*\b(?:point|stop|waypoint|draft)\b",
    re.IGNORECASE,
)
_ACCEPTED_STOP_INCLUSION_REQUEST = re.compile(
    r"\b(?:want\s+to\s+(?:go\s+to|visit)|include|keep|retain)\b",
    re.IGNORECASE,
)
_MAP_DRAFT_OPERATION_TYPES = frozenset(
    {"add_custom_waypoint", "add_route_waypoint", "relocate_custom_waypoint"}
)
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


class OpenAIDurationConstraintV1(_StrictModel):
    mode: Literal["exact", "preferred", "minimum", "maximum", "range"]
    preferred_minutes: int | None = Field(default=None, ge=15, le=480)
    minimum_minutes: int | None = Field(default=None, ge=15, le=480)
    maximum_minutes: int | None = Field(default=None, ge=15, le=480)


class OpenAIDurationEditV1(_StrictModel):
    duration: OpenAIDurationConstraintV1


class OpenAIOrderEditV1(_StrictModel):
    day: int = Field(ge=1, le=21)
    sequence_index: int = Field(ge=0, le=63)


class OpenAITimeWindowEditV1(_StrictModel):
    earliest_arrival: str | None = Field(default=None, pattern=r"^(?:[01][0-9]|2[0-3]):[0-5][0-9]$")
    latest_departure: str | None = Field(default=None, pattern=r"^(?:[01][0-9]|2[0-3]):[0-5][0-9]$")


class OpenAIRoleEditV1(_StrictModel):
    role: Literal["attraction", "activity", "meal", "rest_stop", "scenic_stop"]




class OpenAIIntentSchemaV2(OpenAIIntentSchemaV1):
    type: Literal[*get_args(_INTENT_TYPES), "set_stop_duration", "set_stop_order", "set_stop_time_window", "set_stop_role", "replan_trip"]
    parameters: OpenAIDurationEditV1 | OpenAIOrderEditV1 | OpenAITimeWindowEditV1 | OpenAIRoleEditV1 | YosemiteExtensionParameters | IndividualizedTripParameters | TripPlanParameters | None = None


class OpenAIInterpretationSchemaV2(OpenAIInterpretationSchemaV1):
    schema_version: Literal["copilot-interpretation-v2"]
    intents: list[OpenAIIntentSchemaV2] = Field(default_factory=list, max_length=4)


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
        reasoning_effort: Literal["low", "medium"] = "low",
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
        if reasoning_effort not in {"low", "medium"}:
            raise ValueError("openai_reasoning_effort_invalid")
        self.model = model.strip()
        self._api_key = api_key
        self.timeout_seconds = float(timeout_seconds)
        self.history_messages = history_messages
        self.history_characters = history_characters
        self.reasoning_effort = reasoning_effort
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
        guarded = request_semantic_guard(context, message)
        combined = guarded if guarded is not None and guarded.state == "proposal_ready" and len(guarded.intents) > 1 else None
        if guarded is not None and combined is None:
            return validate_copilot_interpretation(context=context, interpretation=guarded)
        from .copilot_memory import interpret_context_request

        contextual = interpret_context_request(context, message) if combined is None else None
        if contextual is not None and contextual.state != "proposal_ready":
            return validate_copilot_interpretation(context=context, interpretation=contextual)
        draft_guidance = _draft_point_removal_guidance(context, message)
        if draft_guidance is not None:
            return validate_copilot_interpretation(
                context=context,
                interpretation=draft_guidance,
            )
        if (
            is_stop_removal_request(message)
            and not is_stop_replacement_request(message)
        ):
            return validate_copilot_interpretation(
                context=context,
                interpretation=unsupported_stop_removal(),
            )
        inclusion = _accepted_stop_inclusion(context, message) if combined is None and contextual is None else None
        if inclusion is not None:
            return validate_copilot_interpretation(
                context=context,
                interpretation=inclusion,
            )
        bounded_history = self._bounded_history(history)
        try:
            bundle = build(
                context,
                bounded_history,
                message,
                reasoning_effort=self.reasoning_effort,
            )
        except (PromptBudgetError, TypeError) as exc:
            raise CopilotContractError from exc

        client = self._require_client()
        response_schema = OpenAIInterpretationSchemaV2 if isinstance(context, CopilotContextV2) else OpenAIInterpretationSchemaV1
        try:
            async with asyncio.timeout(self.timeout_seconds):
                response = await self._request_once(client, bundle, response_schema=response_schema)
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
        if type(parsed) is not response_schema:
            raise OpenAICopilotError(
                "openai_invalid_structured_output",
                502,
                diagnostic_code="parsed_output_missing",
            )
        interpretation = _to_dataclass(parsed)
        # A provider may resolve an edit from pronouns or wording the bounded
        # local parser does not recognize. Recheck the complete request before
        # exposing that edit; review-only replies do not apply plan changes.
        if interpretation.state == "proposal_ready" and any(
            intent.type != "review_registered_repair" for intent in interpretation.intents
        ):
            guarded = request_semantic_guard(context, message, proposed_edit=True)
            if guarded is not None and guarded.state != "proposal_ready":
                return validate_copilot_interpretation(context=context, interpretation=guarded)
        resolved_request = combined or contextual
        if resolved_request is not None and interpretation.state == "proposal_ready":
            expected = [{**intent.as_dict(), "parameters": getattr(intent, "parameters", None)}
                        for intent in resolved_request.intents]
            actual = [intent.as_dict() for intent in interpretation.intents]
            if actual != expected:
                question = "The reply did not preserve every requested change exactly. Please restate the complete edits with each stop name; no part was staged or changed."
                return CopilotInterpretationV1(state="clarification_required", assistant_message=question,
                    intents=(CopilotIntentV1(type="request_clarification", clarification_question=question),))
            interpretation = replace(interpretation, highlights=resolved_request.highlights)
        # Provider prose is untrusted and never reaches the UI or transcript.
        # The model supplies typed state/intents; user-facing copy is local.
        interpretation = _canonicalize_provider_copy(
            interpretation, context=context, message=message
        )
        from .copilot_memory import guard_contextual_output

        interpretation = guard_contextual_output(context, interpretation)
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

    async def _request_once(self, client: Any, bundle: Any, *, response_schema: Any = OpenAIInterpretationSchemaV1) -> Any:
        # The durable service reservation authorizes exactly one attempt. Even a
        # transient transport error may have reached the provider and incurred usage.
        try:
            return await client.responses.parse(
                model=self.model,
                instructions=bundle.instructions,
                input=bundle.input,
                text_format=response_schema,
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
            raise _normalized_error(_classify_error(exc), exc) from exc

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
    current = isinstance(parsed, OpenAIInterpretationSchemaV2)
    return (CopilotInterpretationV2 if current else CopilotInterpretationV1)(
        schema_version=parsed.schema_version,
        state=parsed.state,
        assistant_message=parsed.assistant_message,
        intents=tuple(
            (CopilotIntentV2 if current else CopilotIntentV1)(**intent.model_dump())
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


def _draft_point_removal_guidance(
    context: CopilotContextV1, message: str
) -> CopilotInterpretationV1 | None:
    if not _DRAFT_POINT_REMOVAL_REQUEST.search(message):
        return None
    if not any(
        isinstance(operation, dict)
        and operation.get("type") in _MAP_DRAFT_OPERATION_TYPES
        for operation in context.draft_operations
    ):
        return None
    return CopilotInterpretationV1(
        state="unsupported",
        assistant_message=(
            "The added point is a staged draft, not an accepted itinerary stop. "
            "Select it and use Remove this draft waypoint, or use Undo latest draft "
            "in the map bar. Nothing was changed automatically."
        ),
        intents=(CopilotIntentV1(type="unsupported_request"),),
    )


def _accepted_stop_inclusion(
    context: CopilotContextV1, message: str
) -> CopilotInterpretationV1 | None:
    if (
        is_stop_removal_request(message)
        or is_stop_replacement_request(message)
        or not _ACCEPTED_STOP_INCLUSION_REQUEST.search(message)
    ):
        return None
    place = matched_known_place(context, message)
    if place is None or place.get("route_status") != "accepted_stop":
        return None
    stop_id = str(place["place_id"])
    if stop_id not in context.allowed_stop_ids:
        return None
    return CopilotInterpretationV1(
        state="proposal_ready",
        assistant_message=(
            f"{place['name']} is already in the accepted itinerary. I prepared a "
            "typed keep-stop suggestion for your review; nothing was changed automatically."
        ),
        intents=(CopilotIntentV1(type="keep_stop", target_stop_id=stop_id),),
        highlights=CopilotHighlightsV1(stop_ids=(stop_id,)),
    )


def _canonicalize_provider_copy(
    interpretation: CopilotInterpretationV1,
    *,
    context: CopilotContextV1,
    message: str,
) -> CopilotInterpretationV1:
    intent_types = {intent.type for intent in interpretation.intents}
    if interpretation.state == "proposal_ready" and len(interpretation.intents) == 1:
        intent = interpretation.intents[0]
        parameters = getattr(intent, "parameters", None)
        if intent.type == "replan_trip" and parameters and "total_days" in parameters:
            p = IndividualizedTripParameters.model_validate(parameters)
            details = f"{p.total_days} 天；節奏 {p.pace}；每日開車上限 {p.max_daily_drive_minutes} 分鐘；興趣 {', '.join(p.interests) or '沿用既有偏好'}；必去 {', '.join(p.required_place_ids) or '保留既有景點'}；避開 {', '.join(p.excluded_place_ids) or '無新增排除'}"
            copy = (f"已準備 {details} 的行程提案。未另指定的偏好會沿用已接受方案。Preview 會重新求解並檢查，尚未更改行程。" if re.search(r"[\u3400-\u9fff]", message) else
                f"Prepared a {p.total_days}-day trip proposal, {p.pace} pace, at most {p.max_daily_drive_minutes} driving minutes per day. Unchanged preferences inherit the accepted profile. Preview will solve and check the route; nothing was applied.")
            return replace(interpretation, assistant_message=copy)
    if (
        is_stop_removal_request(message)
        and "mark_flexible" in intent_types
        and "replace_nearby" not in intent_types
    ):
        return unsupported_stop_removal()
    place_copy = known_place_clarification(context, message)
    if interpretation.state == "clarification_required" and interpretation.intents:
        questions = tuple(intent.clarification_question for intent in interpretation.intents)
        if all(_safe_specific_question(question) for question in questions):
            return replace(
                interpretation,
                assistant_message=" ".join(questions) + " Nothing was changed.",
            )
    if place_copy is not None and interpretation.state in {
        "clarification_required",
        "unsupported",
    }:
        assistant_message, question = place_copy
        return CopilotInterpretationV1(
            state="clarification_required",
            assistant_message=assistant_message,
            intents=(
                CopilotIntentV1(
                    type="request_clarification",
                    clarification_question=question,
                ),
            ),
        )
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


def _safe_specific_question(question: str | None) -> bool:
    """Retain bounded follow-up text, not arbitrary provider execution claims."""
    return (
        isinstance(question, str)
        and 20 <= len(question) <= 512
        and question == question.strip()
        and question.endswith("?")
        and question.count("?") == 1
        and not any(character in question for character in "\r\n<>`")
        and re.match(r"^(?:Which|What|Where|When|How|Would|Do|Should|Can)\b", question, re.IGNORECASE) is not None
        and re.search(r"\b(?:day|stop|place|route|move|replace|add|candidate|duration|trade-off)\b", question, re.IGNORECASE) is not None
        and re.search(
            r"https?://|\b(?:ignore|instruction|password|token|secret|api.key|booked|paid|"
            r"secured|purchased|charged|reserved|cancelled|canceled|accepted|applied|activated)\b",
            question, re.IGNORECASE,
        ) is None
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
