"""Provider-neutral Copilot contract and deterministic adapter."""

from __future__ import annotations

import re
from typing import Protocol

from .models import (
    CopilotContextV1,
    CopilotHighlightsV1,
    CopilotIntentV1,
    CopilotInterpretationV1,
    CopilotTurnV1,
)

MAX_MESSAGE_CHARACTERS = 2_000
MAX_ASSISTANT_MESSAGE_CHARACTERS = 4_000
MAX_INTENTS = 4

ALLOWED_INTENT_TYPES = frozenset(
    {
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
    }
)
ALLOWED_PREFERENCES = frozenset({"reduce_contextual_risk", "reduce_driving", "preserve_experience"})
ALLOWED_STATES = frozenset(
    {
        "clarification_required",
        "permission_required",
        "proposal_ready",
        "unsupported",
        "refused",
        "failed",
    }
)

_PROPOSAL_INTENTS = frozenset(
    {
        "review_registered_repair",
        "keep_original",
        "keep_stop",
        "lock_stop",
        "mark_flexible",
        "move_day",
        "route_feedback",
        "replace_nearby",
        "add_candidate",
    }
)

_FALSE_EXECUTION_CLAIMS = (
    re.compile(
        r"\b(?:i|we|the\s+copilot|itinerary\s+copilot)\s+(?:have\s+)?"
        r"(?:booked|paid|purchased|charged|reserved|cancelled|canceled|contacted|"
        r"emailed|messaged|ordered|ticketed|confirmed)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:your|the)\s+(?:booking|reservation|ticket|payment|card)\s+"
        r"(?:has\s+been|was|is)\s+(?:confirmed|charged|booked|paid|purchased|"
        r"cancelled|canceled)\b",
        re.IGNORECASE,
    ),
)


class CopilotContractError(ValueError):
    """A stable, non-provider-specific Copilot contract failure."""

    code = "copilot_interpretation_invalid"

    def __init__(self) -> None:
        super().__init__(self.code)


class CopilotAdapter(Protocol):
    provider_name: str

    async def interpret(
        self,
        *,
        context: CopilotContextV1,
        history: tuple[CopilotTurnV1, ...],
        message: str,
    ) -> CopilotInterpretationV1: ...


def validate_copilot_interpretation(
    *,
    context: CopilotContextV1,
    interpretation: CopilotInterpretationV1,
) -> CopilotInterpretationV1:
    """Fail closed unless an interpretation belongs to the current workspace universe."""

    _validate_context(context)
    if type(interpretation) is not CopilotInterpretationV1:
        raise CopilotContractError
    if interpretation.schema_version != "copilot-interpretation-v1":
        raise CopilotContractError
    if interpretation.state not in ALLOWED_STATES:
        raise CopilotContractError
    if (
        type(interpretation.assistant_message) is not str
        or not interpretation.assistant_message.strip()
        or len(interpretation.assistant_message) > MAX_ASSISTANT_MESSAGE_CHARACTERS
        or any(pattern.search(interpretation.assistant_message) for pattern in _FALSE_EXECUTION_CLAIMS)
    ):
        raise CopilotContractError
    if type(interpretation.intents) is not tuple or len(interpretation.intents) > MAX_INTENTS:
        raise CopilotContractError
    if interpretation.permission not in {None, "external_booking_change"}:
        raise CopilotContractError
    if type(interpretation.highlights) is not CopilotHighlightsV1:
        raise CopilotContractError

    for intent in interpretation.intents:
        _validate_intent(context, intent)
    _validate_highlights(context, interpretation.highlights)
    _validate_state_combination(interpretation)
    return interpretation


class DeterministicCopilotAdapter:
    """Rule-backed, network-free adapter for explicitly selected demo mode."""

    provider_name: str = "deterministic"
    provider_label: str = "Experimental demo — deterministic adapter"

    async def interpret(
        self,
        *,
        context: CopilotContextV1,
        history: tuple[CopilotTurnV1, ...],
        message: str,
    ) -> CopilotInterpretationV1:
        _validate_history(history)
        _validate_context(context)
        if type(message) is not str:
            raise CopilotContractError

        normalized = " ".join(message.lower().split())
        if len(message) > MAX_MESSAGE_CHARACTERS:
            return self._validated(
                context,
                CopilotInterpretationV1(
                    state="failed",
                    assistant_message=(f"{self.provider_label}: the message exceeds the 2,000-character limit."),
                ),
            )
        if not normalized:
            return self._clarification(
                context,
                "Tell me which day, stop, or trade-off to review.",
            )
        if any(term in normalized for term in ("book", "purchase", "pay", "cancel reservation")):
            return self._validated(
                context,
                CopilotInterpretationV1(
                    state="permission_required",
                    assistant_message=(
                        f"{self.provider_label}: booking changes are outside this Copilot and will not be executed."
                    ),
                    intents=(CopilotIntentV1(type="external_action_not_supported"),),
                    permission="external_booking_change",
                ),
            )

        operation = self._operation_intent(normalized, context)
        if operation is not None:
            return self._validated(
                context,
                CopilotInterpretationV1(
                    state="proposal_ready",
                    assistant_message=(
                        f"{self.provider_label}: I interpreted this as {operation.type}. "
                        "Review the typed proposal before using the existing draft workflow; "
                        "nothing was activated automatically."
                    ),
                    intents=(operation,),
                    highlights=self._selected_highlights(context),
                ),
            )

        if any(term in normalized for term in ("weather", "risk", "safer", "repair", "reduce driving", "easier")):
            preference = "reduce_contextual_risk"
            if "driv" in normalized or "easier" in normalized:
                preference = "reduce_driving"
            return self._validated(
                context,
                CopilotInterpretationV1(
                    state="proposal_ready",
                    assistant_message=(
                        f"{self.provider_label}: I found one registered, independently "
                        "evaluated repair. You can inspect its evidence, but acceptance "
                        "remains disabled until W5."
                    ),
                    intents=(
                        CopilotIntentV1(
                            type="review_registered_repair",
                            preference=preference,
                        ),
                    ),
                    highlights=self._selected_highlights(context),
                ),
            )
        if "keep" in normalized and ("original" in normalized or "same" in normalized):
            return self._validated(
                context,
                CopilotInterpretationV1(
                    state="proposal_ready",
                    assistant_message=(
                        f"{self.provider_label}: Keep original is a separate, non-mutating "
                        "decision in W5; no decision was recorded."
                    ),
                    intents=(CopilotIntentV1(type="keep_original"),),
                ),
            )
        return self._clarification(
            context,
            "Ask about a registered repair, a route trade-off, or keeping the original plan. "
            "Choose a day or stop to provide more context.",
        )

    def _operation_intent(
        self,
        normalized: str,
        context: CopilotContextV1,
    ) -> CopilotIntentV1 | None:
        stop_id = context.selected_stop_id
        candidate_id = context.selected_candidate_id
        if "keep" in normalized and "stop" in normalized and stop_id is not None:
            return CopilotIntentV1(type="keep_stop", target_stop_id=stop_id)
        if "lock" in normalized and stop_id is not None:
            return CopilotIntentV1(type="lock_stop", target_stop_id=stop_id)
        if "flexible" in normalized and stop_id is not None:
            return CopilotIntentV1(type="mark_flexible", target_stop_id=stop_id)
        if "move" in normalized and stop_id is not None:
            day_match = re.search(r"\bday\s+(\d+)\b", normalized)
            if day_match is not None:
                day = int(day_match.group(1))
                if day in context.allowed_days:
                    return CopilotIntentV1(
                        type="move_day",
                        target_stop_id=stop_id,
                        destination_day=day,
                    )
        if "replace" in normalized and stop_id is not None and candidate_id is not None:
            return CopilotIntentV1(
                type="replace_nearby",
                target_stop_id=stop_id,
                candidate_id=candidate_id,
            )
        if "add" in normalized and candidate_id is not None and context.selected_day is not None:
            return CopilotIntentV1(
                type="add_candidate",
                destination_day=context.selected_day,
                candidate_id=candidate_id,
            )
        if "driv" in normalized or "route" in normalized:
            return CopilotIntentV1(type="route_feedback", preference="reduce_driving")
        return None

    def _clarification(
        self,
        context: CopilotContextV1,
        message: str,
    ) -> CopilotInterpretationV1:
        interpretation = CopilotInterpretationV1(
            state="clarification_required",
            assistant_message=f"{self.provider_label}: {message}",
            intents=(
                CopilotIntentV1(
                    type="request_clarification",
                    clarification_question=message,
                ),
            ),
            highlights=self._selected_highlights(context),
        )
        return self._validated(context, interpretation)

    @staticmethod
    def _selected_highlights(context: CopilotContextV1) -> CopilotHighlightsV1:
        return CopilotHighlightsV1(
            day_ids=(context.selected_day,) if context.selected_day is not None else (),
            stop_ids=(context.selected_stop_id,) if context.selected_stop_id is not None else (),
            segment_ids=((context.selected_segment_id,) if context.selected_segment_id is not None else ()),
            candidate_ids=((context.selected_candidate_id,) if context.selected_candidate_id is not None else ()),
        )

    @staticmethod
    def _validated(
        context: CopilotContextV1,
        interpretation: CopilotInterpretationV1,
    ) -> CopilotInterpretationV1:
        return validate_copilot_interpretation(
            context=context,
            interpretation=interpretation,
        )


def _validate_context(context: CopilotContextV1) -> None:
    if type(context) is not CopilotContextV1 or context.schema_version != "copilot-context-v1":
        raise CopilotContractError
    if any(
        type(value) is not str or not value
        for value in (context.run_id, context.trip_id, context.session_id, context.accepted_plan_id)
    ):
        raise CopilotContractError
    if type(context.session_revision) is not int or context.session_revision < 0:
        raise CopilotContractError
    _validate_string_universe(context.allowed_stop_ids)
    _validate_string_universe(context.allowed_candidate_ids)
    _validate_string_universe(context.allowed_segment_ids)
    _validate_string_universe(context.allowed_alternative_ids)
    _validate_day_universe(context.allowed_days)
    if context.selected_stop_id is not None and context.selected_stop_id not in context.allowed_stop_ids:
        raise CopilotContractError
    if context.selected_candidate_id is not None and context.selected_candidate_id not in context.allowed_candidate_ids:
        raise CopilotContractError
    if context.selected_day is not None:
        if type(context.selected_day) is not int or context.selected_day not in context.allowed_days:
            raise CopilotContractError
    for selected in (context.selected_stop_id, context.selected_candidate_id):
        if selected is not None and type(selected) is not str:
            raise CopilotContractError
    if (
        context.selected_segment_id is not None
        and context.selected_segment_id not in context.allowed_segment_ids
    ):
        raise CopilotContractError
    if (
        context.selected_alternative_id is not None
        and context.selected_alternative_id not in context.allowed_alternative_ids
    ):
        raise CopilotContractError
    if type(context.draft_operations) is not tuple:
        raise CopilotContractError
    if any(type(operation) is not dict for operation in context.draft_operations):
        raise CopilotContractError
    if context.evaluated_proposal is not None and type(context.evaluated_proposal) is not dict:
        raise CopilotContractError


def _validate_intent(context: CopilotContextV1, intent: CopilotIntentV1) -> None:
    if type(intent) is not CopilotIntentV1 or intent.type not in ALLOWED_INTENT_TYPES:
        raise CopilotContractError
    if intent.preference is not None and intent.preference not in ALLOWED_PREFERENCES:
        raise CopilotContractError
    if intent.target_stop_id is not None:
        if type(intent.target_stop_id) is not str or intent.target_stop_id not in context.allowed_stop_ids:
            raise CopilotContractError
    if intent.candidate_id is not None:
        if type(intent.candidate_id) is not str or intent.candidate_id not in context.allowed_candidate_ids:
            raise CopilotContractError
    if intent.destination_day is not None:
        if type(intent.destination_day) is not int or intent.destination_day not in context.allowed_days:
            raise CopilotContractError
    if intent.clarification_question is not None and (
        type(intent.clarification_question) is not str
        or not intent.clarification_question.strip()
        or len(intent.clarification_question) > MAX_ASSISTANT_MESSAGE_CHARACTERS
    ):
        raise CopilotContractError

    present = {
        name
        for name in (
            "target_stop_id",
            "destination_day",
            "candidate_id",
            "preference",
            "clarification_question",
        )
        if getattr(intent, name) is not None
    }
    allowed_required: dict[str, tuple[set[str], set[str]]] = {
        "review_registered_repair": ({"preference"}, set()),
        "keep_original": (set(), set()),
        "keep_stop": ({"target_stop_id"}, {"target_stop_id"}),
        "lock_stop": ({"target_stop_id"}, {"target_stop_id"}),
        "mark_flexible": ({"target_stop_id"}, {"target_stop_id"}),
        "move_day": (
            {"target_stop_id", "destination_day"},
            {"target_stop_id", "destination_day"},
        ),
        "route_feedback": ({"preference"}, {"preference"}),
        "replace_nearby": (
            {"target_stop_id", "candidate_id"},
            {"target_stop_id", "candidate_id"},
        ),
        "add_candidate": (
            {"destination_day", "candidate_id"},
            {"destination_day", "candidate_id"},
        ),
        "request_clarification": (
            {"clarification_question"},
            {"clarification_question"},
        ),
        "unsupported_request": (set(), set()),
        "external_action_not_supported": (set(), set()),
    }
    allowed, required = allowed_required[intent.type]
    if not required <= present or not present <= allowed:
        raise CopilotContractError


def _validate_highlights(context: CopilotContextV1, highlights: CopilotHighlightsV1) -> None:
    _validate_day_universe(highlights.day_ids)
    _validate_string_universe(highlights.stop_ids)
    _validate_string_universe(highlights.segment_ids)
    _validate_string_universe(highlights.candidate_ids)
    if not set(highlights.day_ids) <= set(context.allowed_days):
        raise CopilotContractError
    if not set(highlights.stop_ids) <= set(context.allowed_stop_ids):
        raise CopilotContractError
    if not set(highlights.candidate_ids) <= set(context.allowed_candidate_ids):
        raise CopilotContractError
    if not set(highlights.segment_ids) <= set(context.allowed_segment_ids):
        raise CopilotContractError


def _validate_state_combination(interpretation: CopilotInterpretationV1) -> None:
    intent_types = {intent.type for intent in interpretation.intents}
    if interpretation.state == "permission_required":
        if (
            interpretation.permission != "external_booking_change"
            or len(interpretation.intents) != 1
            or intent_types != {"external_action_not_supported"}
        ):
            raise CopilotContractError
        return
    if interpretation.permission is not None:
        raise CopilotContractError
    if interpretation.state == "proposal_ready":
        if not interpretation.intents or not intent_types <= _PROPOSAL_INTENTS:
            raise CopilotContractError
    elif interpretation.state == "clarification_required":
        if intent_types - {"request_clarification"}:
            raise CopilotContractError
    elif interpretation.state == "unsupported":
        if intent_types - {"unsupported_request", "external_action_not_supported"}:
            raise CopilotContractError
    elif interpretation.intents:
        raise CopilotContractError


def _validate_string_universe(values: object) -> None:
    if type(values) is not tuple or any(type(value) is not str or not value for value in values):
        raise CopilotContractError
    if len(values) != len(set(values)):
        raise CopilotContractError


def _validate_day_universe(values: object) -> None:
    if type(values) is not tuple or any(type(value) is not int or value < 1 for value in values):
        raise CopilotContractError
    if len(values) != len(set(values)):
        raise CopilotContractError


def _validate_history(history: tuple[CopilotTurnV1, ...]) -> None:
    if type(history) is not tuple:
        raise CopilotContractError
    if any(type(turn) is not CopilotTurnV1 or turn.schema_version != "copilot-turn-v1" for turn in history):
        raise CopilotContractError
