"""Provider-neutral Copilot contract and deterministic adapter."""

from __future__ import annotations

import re
from dataclasses import replace
from typing import Protocol

from .copilot_typed_edits import TYPED_CHAT_OPERATIONS, validate_parameters
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
KNOWN_PLACE_ROUTE_STATUSES = frozenset(
    {
        "accepted_stop",
        "registered_route_candidate",
        "discovery_only_route_validation_required",
    }
)

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
ALLOWED_INTENT_TYPES |= TYPED_CHAT_OPERATIONS | {"replan_trip"}
_PROPOSAL_INTENTS |= TYPED_CHAT_OPERATIONS | {"replan_trip"}

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
_STOP_REMOVAL_REQUEST = re.compile(
    r"\b(?:remove|delete|exclude|skip)\b|\b(?:do not|don't|dont)\s+want\s+to\s+(?:go\s+to|visit|include)\b",
    re.IGNORECASE,
)
_STOP_REPLACEMENT_REQUEST = re.compile(
    r"\b(?:replace|swap|substitute)\b",
    re.IGNORECASE,
)
_PRESENTATION_REMOVAL = re.compile(
    r"\b(?:remove|delete|exclude|skip)\s+(?:the\s+)?"
    r"(?:explanation|details|summary|introduction|preamble|description)\b",
    re.IGNORECASE,
)
_NEGATED_EDIT = re.compile(
    r"\b(?:do\s+not|don't|dont|never|not)\s+(?:want\s+to\s+)?"
    r"(?P<action>remove|delete|exclude|skip|add|include|replace|swap|move|lock|keep|"
    r"mark|make|retain|reschedule|shorten|extend|change|set|spend|stay|visit|give|treat|classify|reorder|arrive|leave)\b",
    re.IGNORECASE,
)
_EDIT_ACTION = re.compile(
    r"\b(?:keep|retain|lock|flexible|move|replace|swap|substitute|add|include|"
    r"remove|delete|exclude|skip|shorten|extend|reschedule|set|change|spend|stay|visit|give|treat|classify|reorder|arrive|leave|I(?:'d| would) like)\b",
    re.IGNORECASE,
)


def request_edit_text(message: str) -> str:
    """Remove explicit non-edit annotations, without dropping unresolved edits."""
    text = re.sub(
        r"\b(?:keep|preserve|preserving|leave)\s+(?:the\s+)?"
        r"(?:other days|(?:rest|remaining|other)(?:\s+stops)?(?:\s+of\s+the\s+(?:seven-day\s+)?(?:trip|itinerary))?)"
        r"(?:\s+(?:unchanged|alone|as is))?\b",
        "",
        message,
        flags=re.I,
    )
    return re.sub(
        r"(?:^|(?<=[.!?]))\s*(?:the\s+)?(?:coast|weather|forecast)\s+"
        r"(?:looks|is|seems)\s+(?:stormy|rainy|wet|windy)[.!?]",
        "",
        text,
        flags=re.I,
    )


def is_stop_removal_request(message: str) -> bool:
    text = _PRESENTATION_REMOVAL.sub("", message)
    text = _NEGATED_EDIT.sub("", text)
    return bool(_STOP_REMOVAL_REQUEST.search(text))


def is_stop_replacement_request(message: str) -> bool:
    return bool(_STOP_REPLACEMENT_REQUEST.search(message))


def unsupported_stop_removal() -> CopilotInterpretationV1:
    return CopilotInterpretationV1(
        state="unsupported",
        assistant_message=(
            "Removing or excluding an accepted stop is not enabled in this repair workflow. Nothing was changed."
        ),
        intents=(CopilotIntentV1(type="unsupported_request"),),
    )


def _bounded_chinese_edit_request(context: CopilotContextV1, message: str) -> CopilotInterpretationV1 | None:
    """Resolve complete Chinese replacement/duration clauses through existing checks."""
    if not isinstance(context, CopilotContextV2) or not re.search(r"換成|换成|替換|替换|停留|分鐘|分钟", message):
        return None
    # Internal masking tokens must never be accepted as user-supplied place names.
    text = "" if re.search(r"COPILOTZHPLACE\d+", message, re.I) else message
    names = {}
    aliases = []
    for index, place in enumerate(context.known_places):
        token = f"COPILOTZHPLACE{index}"
        names[token] = str(place["name"])
        aliases.extend((str(alias), token) for alias in (place["name"], *place["aliases"]))
    for alias, token in sorted(aliases, key=lambda pair: len(pair[0]), reverse=True):
        text = re.sub(r"(?<![A-Za-z0-9_])" + re.escape(alias) + r"(?![A-Za-z0-9_])", token, text, flags=re.I)
    clauses = [part.strip() for part in re.split(r"[，,；;。.!?]+|並且|并且|而且|以及|並|并|且", text) if part.strip()]
    normalized = []
    for clause in clauses:
        clause = " ".join(clause.split())
        replacement = re.fullmatch(
            r"(?:請|请)?\s*(?:把|將|将)?\s*(COPILOTZHPLACE\d+)\s*"
            r"(?:換成|换成|替換(?:為|成)|替换(?:为|成))\s*(COPILOTZHPLACE\d+)",
            clause,
        )
        duration = re.fullmatch(
            r"(?:請|请)?\s*(?:把|將|将)?\s*(COPILOTZHPLACE\d+)\s*(?:的)?\s*"
            r"(?:停留(?:時間|时间|時長|时长)?|遊覽時間|游览时间)\s*"
            r"(?:設(?:為|置為)|设(?:为|置为)|改(?:為|为))\s*(\d+)\s*(分鐘|分钟|小時|小时)",
            clause,
        ) or re.fullmatch(
            r"(?:請|请)?\s*(?:在)?\s*(COPILOTZHPLACE\d+)\s*(?:停留|遊覽|游览)\s*"
            r"(\d+)\s*(分鐘|分钟|小時|小时)",
            clause,
        )
        if replacement:
            normalized.append(f"Replace {names[replacement[1]]} with {names[replacement[2]]}")
        elif duration:
            unit = "hours" if duration[3] in {"小時", "小时"} else "minutes"
            normalized.append(f"Give {names[duration[1]]} {duration[2]} {unit}")
        elif not re.search(r"[\u3400-\u9fff]", clause):
            clause = re.sub(r"\bCOPILOTZHPLACE\d+\b", lambda match: names[match[0]], clause)
            normalized.append(clause)
        else:
            break
    else:
        if 1 <= len(normalized) <= MAX_INTENTS:
            resolved = _compound_proposal(context, tuple(normalized))
            if resolved is not None:
                return resolved
    question = "Please restate every requested edit with its stop name and exact replacement or duration; I could not safely resolve the complete request."
    return CopilotInterpretationV1(
        state="clarification_required",
        assistant_message=question + " Nothing was staged or changed.",
        intents=(CopilotIntentV1(type="request_clarification", clarification_question=question),),
    )


def request_semantic_guard(
    context: CopilotContextV1,
    message: str,
    *,
    proposed_edit: bool = False,
) -> CopilotInterpretationV1 | None:
    """Fail closed on negation and multi-clause edits our shortcuts cannot preserve.

    This is a bounded guard, not a general natural-language parser. Only complete
    clauses in an executable combination may compose; no convenient subset is used.
    """
    from .trip_planning import trip_request_interpretation

    # An explicit no-change instruction applies to the entire request, including
    # duration words mentioned while asking which day deserves a review.
    if re.search(
        r"\b(?:do not|don['’]?t|never)\s+(?:change|modify|edit|replan|"
        r"(?:make|apply|stage|prepare|propose)\s+(?:any\s+)?(?:changes?|edits?))\b|"
        r"\bwithout\s+(?:making\s+(?:any\s+)?(?:changes?|edits?)|changing|editing|modifying)\b|"
        r"(?:不要|別|别|請勿|请勿)\s*(?:先|再)?\s*(?:(?:幫|帮|替|為|为)\s*(?:我|我們|我们)\s*)?"
        r"(?:直接|自動|自动)?\s*(?:修改|更改|改動|改动|變更|变更|重新(?:安排|規劃|规划))",
        message,
        re.IGNORECASE,
    ):
        question = (
            "我會保留目前行程，不提出修改。你想先檢視駕車時間、景點之間的安排，還是某一天的細節？尚未加入草稿或更改行程。"
            if re.search(r"[\u3400-\u9fff]", message)
            else "I will keep the itinerary unchanged. Would you like to review driving time, stop spacing, or a particular day's details? Nothing was staged or changed."
        )
        return CopilotInterpretationV1(
            state="clarification_required",
            assistant_message=question,
            intents=(CopilotIntentV1(type="request_clarification", clarification_question=question),),
        )
    trip = trip_request_interpretation(context, message)
    if trip is not None:
        return trip
    chinese = _bounded_chinese_edit_request(context, message)
    if chinese is not None:
        return chinese
    if re.fullmatch(
        r"\s*(?:please\s+)?(?:keep|retain)\s+(?:the\s+)?"
        r"(?:original|same|current|accepted)\s+(?:plan|itinerary|trip)"
        r"(?:\s+(?:unchanged|as\s+is))?[.!]?\s*",
        message,
        re.IGNORECASE,
    ):
        return CopilotInterpretationV1(
            state="proposal_ready",
            assistant_message=(
                "Keep original leaves the current accepted itinerary unchanged. "
                "It requires a separate confirmation; no decision was recorded and no draft was changed."
            ),
            intents=(CopilotIntentV1(type="keep_original"),),
        )
    from .copilot_typed_edits import arrival_window_clarification

    clarification = arrival_window_clarification(message)
    if clarification is not None:
        return clarification
    text = request_edit_text(_PRESENTATION_REMOVAL.sub("", message))
    places = _matched_known_places(context, text)
    # Place names can themselves contain conjunctions (Redwood National and
    # State Parks); do not treat those words as a second instruction.
    instruction_text = text
    place_names = {}
    for index, place in enumerate(places):
        token = f"COPILOTPLACE{index}"
        place_names[token.lower()] = str(place["name"])
        for name in sorted((str(place["name"]), *place["aliases"]), key=len, reverse=True):
            instruction_text = re.sub(
                r"(?<!\w)" + re.escape(name) + r"(?!\w)", token, instruction_text, flags=re.IGNORECASE
            )
    # A conjunction joining clock bounds is part of one time-window value.
    instruction_text = re.sub(r"(\d{2}:\d{2})\s+and\s+(?=\d{2}:\d{2})", r"\1 to ", instruction_text, flags=re.I)
    instruction_text = re.sub(
        r"(\d{2}:\d{2})\s+and\s+(?=latest departure\b)", r"\1 with ", instruction_text, flags=re.I
    )
    # These exact clock phrases are values of one time-window edit. Keep other
    # verbs, clauses, place names and the original negation checks visible.
    instruction_text = re.sub(
        r"\b(?:arrive\s+(?:(?:at|not before)\s+)?|leave\s+by\s+)\d{2}:\d{2}\b",
        "COPILOTCLOCK",
        instruction_text,
        flags=re.I,
    )
    instruction_text = re.sub(
        r"(COPILOTCLOCK)\s*(?:,|and)\s*(?=COPILOTCLOCK\b)",
        r"\1 with ",
        instruction_text,
        flags=re.I,
    )
    # A leading day label within each clause qualifies its edit; its comma
    # is not a second request. Retain the day value for the typed order parser.
    instruction_text = re.sub(
        r"(^\s*|(?:\b(?:and|but|then|also)\b|[;,.!?])\s*)((?:(?:if possible|if you can)\s*,?\s*)?on\s+day\s+\d+)\s*,\s*",
        r"\1\2 ",
        instruction_text,
        flags=re.I,
    )
    actions = [match.group().lower() for match in _EDIT_ACTION.finditer(instruction_text)]
    # A typed edit has no condition field. Preserve weather/event prerequisites
    # by asking, rather than silently proposing the unconditional subset.
    # Capability/feasibility courtesies still use the normal evaluated preview.
    condition_text = re.sub(
        r"\bif (?:possible|you can)(?=\s*[.!?]?$|\s*[,;]?\s+(?:on\s+day\s+\d+\s*,?\s*)?(?:please\s+)?"
        + _EDIT_ACTION.pattern
        + r")",
        "",
        instruction_text,
        flags=re.I,
    )
    # A hypothetical visit request has no event prerequisite of its own.
    # Leave any later condition or additional clause visible to the guards.
    condition_text = re.sub(
        r"^\s*what\s+if\s+I\s+(?:want\s+to\s+)?(?=(?:visit|add|include)\b)", "", condition_text, flags=re.I
    )
    # Unknown edit wording can still name a known stop. Its condition must
    # not disappear just because the action is outside the local verb list.
    if places or proposed_edit:
        condition_text = re.sub(r"^\s*(?:if possible\b\s*,?\s*|if you can\s*,\s*)", "", condition_text, flags=re.I)
    # Urgency/frequency courtesies are not event prerequisites.
    condition_text = re.sub(r"\b(?:at|for)\s+once\b", "", condition_text, flags=re.I)
    if (actions or places or proposed_edit) and re.search(
        r"\b(?:if|unless|when|provided|as long as|in case|on condition|assuming|"
        r"subject\s+to|as\s+soon\s+as|contingent\s+(?:on|upon)|"
        r"in\s+the\s+event(?:\s+that)?|so\s+long\s+as|only\s+with|"
        r"presuming|supposing|depending\s+on|conditional\s+(?:on|upon))\b|"
        r"\b(?:weather|traffic|parking|conditions)\s+permitting\b|"
        r"\bonce\s+(?=\w)|"
        r"\b(?:before|after)\b(?!\s+(?:copilotplace\d+|\d{2}:\d{2})\b)",
        condition_text,
        re.I,
    ):
        question = (
            "Your edit includes a condition or event trigger that this workflow cannot evaluate. "
            "Should I propose the edit unconditionally, or wait until you confirm the condition is met?"
        )
        return CopilotInterpretationV1(
            state="clarification_required",
            assistant_message=question + " Nothing was staged or changed.",
            intents=(CopilotIntentV1(type="request_clarification", clarification_question=question),),
        )
    negation = _NEGATED_EDIT.search(text)
    question = None
    clauses = [
        part.strip().lower()
        for part in re.split(
            r"\b(?:and|but|then|also)\b|[;,.!?]",
            instruction_text,
            flags=re.IGNORECASE,
        )
    ]
    clauses = [
        part
        for part in clauses
        if part
        and part
        not in {
            "please",
            "hi",
            "hello",
            "thanks",
            "thank you",
            "okay",
            "ok",
            "sure",
            "if possible",
            "if you can",
        }
    ]
    clauses = [part for part in clauses if not re.fullmatch(r"for copilotplace\d+", part)]
    # 'Remove A and replace it with B' is one replacement, not two edits.
    same_stop_replacement = (
        len(actions) >= 2
        and negation is None
        and actions[0] in {"remove", "delete"}
        and actions[1] in {"replace", "swap", "substitute"}
        and re.search(r"\b(?:replace|swap|substitute)\s+it\b", text, re.IGNORECASE)
    )
    replacement_pair = same_stop_replacement and len(actions) == 2 and len(clauses) == 2
    order = (
        re.fullmatch(
            r"(?:on\s+day\s+(?P<day>\d+)\s+)?(?:please\s+)?visit\s+copilotplace(?P<first>\d+)\s+"
            r"(?P<relation>before|after)\s+copilotplace(?P<second>\d+)",
            clauses[0],
            re.I,
        )
        if len(clauses) == 1
        else None
    )
    unhandled_clause = len(clauses) > 1 and bool(actions)
    if (len(actions) > 1 or unhandled_clause) and not replacement_pair:
        labels = ", ".join(dict.fromkeys(actions)) or "multiple clauses"
        if any(
            re.search(r"\b(?:remove|delete|exclude|skip)\s+copilotplace\d+\b", clause)
            and _NEGATED_EDIT.search(clause) is None
            and not (index == 0 and same_stop_replacement)
            for index, clause in enumerate(clauses)
        ):
            return CopilotInterpretationV1(
                state="unsupported",
                assistant_message=(
                    f"Your request contains {labels}. Removing or excluding an accepted stop is not enabled "
                    "in this repair workflow. No part of this combined request was staged or changed. "
                    "You can submit a supported edit separately; the removal will not be substituted with another operation."
                ),
                intents=(CopilotIntentV1(type="unsupported_request"),),
            )
        if negation is None and isinstance(context, CopilotContextV2) and 2 <= len(clauses) <= MAX_INTENTS:
            restored = []
            for clause in clauses:
                for token, name in place_names.items():
                    clause = clause.replace(token, name)
                restored.append(clause)
            combined = _compound_proposal(context, tuple(restored))
            if combined is not None:
                return combined
        question = (
            f"Your request contains {labels} and additional details. I have not staged any part "
            "because I cannot safely preserve every clause as one edit. Which single edit should I handle first?"
        )
    elif order is not None and negation is None and isinstance(context, CopilotContextV2):
        # Resolve the complete accepted day's order before a provider can drop,
        # reverse or substitute an edit. Reuse the existing typed batch checks.
        indexes = (int(order["first"]), int(order["second"]))
        if len(set(indexes)) == 2 and all(index < len(places) for index in indexes):
            first, second = (str(places[index]["place_id"]) for index in indexes)
            positions = context.capabilities.get("stop_positions", ())
            first_row = next((row for row in positions if row["stop_id"] == first), None)
            second_row = next((row for row in positions if row["stop_id"] == second), None)
            day = int(order["day"]) if order["day"] else first_row["day"] if first_row else None
            if first_row and second_row and first_row["day"] == second_row["day"] == day:
                day_rows = sorted(
                    (row for row in positions if row["day"] == day), key=lambda row: row["sequence_index"]
                )
                names = {str(place["place_id"]): str(place["name"]) for place in context.known_places}
                ordered = [row["stop_id"] for row in day_rows]
                ordered.remove(first)
                ordered.insert(ordered.index(second) + (order["relation"].lower() == "after"), first)
                if len(ordered) <= MAX_INTENTS and all(stop in names for stop in ordered):
                    resolved = _compound_proposal(
                        context,
                        tuple(
                            f"Set {names[stop]} to position {index + 1} on Day {day}"
                            for index, stop in enumerate(ordered)
                        ),
                    )
                    if resolved is not None:
                        return resolved
        question = (
            "Relative ordering needs two different accepted stops on the same accepted day. "
            "Please confirm those stops and their accepted day; no part was staged or changed."
        )
    elif len(places) > (2 if is_stop_replacement_request(text) else 1) and actions:
        question = "Your request names multiple places. Which single stop and edit should I handle first?"
    elif len(set(re.findall(r"\bday\s+(\d+)\b", text, re.IGNORECASE))) > 1 and actions:
        question = "Your request names multiple days. Which single destination day should I use?"
    elif negation is not None:
        if negation.group("action").lower() in {"remove", "delete", "exclude", "skip"} and len(places) == 1:
            place = places[0]
            stop_id = str(place["place_id"])
            if place["route_status"] == "accepted_stop" and stop_id in context.allowed_stop_ids:
                return CopilotInterpretationV1(
                    state="proposal_ready",
                    assistant_message=(
                        f"You asked not to remove {place['name']}. I prepared a keep-stop suggestion "
                        "for your review; nothing was changed automatically."
                    ),
                    intents=(CopilotIntentV1(type="keep_stop", target_stop_id=stop_id),),
                    highlights=CopilotHighlightsV1(stop_ids=(stop_id,)),
                )
        question = "I will not perform the edit you negated. What change, if any, should I propose instead?"
    if question is None:
        return None
    return CopilotInterpretationV1(
        state="clarification_required",
        assistant_message=question,
        intents=(CopilotIntentV1(type="request_clarification", clarification_question=question),),
    )


def _compound_proposal(context: CopilotContextV2, clauses: tuple[str, ...]) -> CopilotInterpretationV1 | None:
    """Compose only complete, individually resolved clauses; never guess a subset."""
    from .copilot_memory import guard_contextual_output, interpret_context_request

    parts = []
    for index, clause in enumerate(clauses, start=1):
        part = request_semantic_guard(context, clause) or interpret_context_request(context, clause)
        if part is None:
            return None
        if part.state != "proposal_ready":
            return replace(
                part,
                assistant_message=f"Change {index}: {part.assistant_message} No part of the combined request was staged or changed.",
            )
        if not part.intents or sum(len(previous.intents) for previous in parts) + len(part.intents) > MAX_INTENTS:
            return None
        days = {int(day) for day in re.findall(r"\bday\s+(\d+)\b", clause, re.IGNORECASE)}
        for intent in part.intents:
            if days and intent.target_stop_id is not None:
                accepted_day = next(
                    (
                        row["day"]
                        for row in context.capabilities["stop_positions"]
                        if row["stop_id"] == intent.target_stop_id
                    ),
                    None,
                )
                if days != {accepted_day}:
                    question = (
                        f"Change {index} names a different day, but this stop is accepted on Day {accepted_day}. "
                        "Moving stops between days is not enabled. Should I keep its accepted day instead? "
                        "No part of the combined request was staged or changed."
                    )
                    return CopilotInterpretationV1(
                        state="clarification_required",
                        assistant_message=question,
                        intents=(CopilotIntentV1(type="request_clarification", clarification_question=question),),
                    )
        parts.append(part)
    intents = tuple(
        CopilotIntentV2(**{**intent.as_dict(), "parameters": getattr(intent, "parameters", None)})
        for part in parts
        for intent in part.intents
    )
    resolved_highlights = [DeterministicCopilotAdapter._operation_highlights(context, intent) for intent in intents]
    highlights = CopilotHighlightsV1(
        **{
            name: tuple(dict.fromkeys(value for scoped in resolved_highlights for value in getattr(scoped, name)))
            for name in ("day_ids", "stop_ids", "segment_ids", "candidate_ids")
        }
    )
    result = CopilotInterpretationV2(
        state="proposal_ready",
        intents=intents,
        highlights=highlights,
        assistant_message=f"I prepared all {len(intents)} requested changes as one combined suggestion. Nothing was staged or changed.",
    )
    return guard_contextual_output(context, result)


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
    versions = {
        CopilotInterpretationV1: "copilot-interpretation-v1",
        CopilotInterpretationV2: "copilot-interpretation-v2",
    }
    if type(interpretation) not in versions or interpretation.schema_version != versions[type(interpretation)]:
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
        if type(intent) is not (
            CopilotIntentV2 if type(interpretation) is CopilotInterpretationV2 else CopilotIntentV1
        ):
            raise CopilotContractError
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
        guarded = request_semantic_guard(context, message)
        if guarded is not None:
            return self._validated(context, guarded)
        from .copilot_memory import interpret_context_request

        contextual = interpret_context_request(context, message)
        if contextual is not None:
            return self._validated(context, contextual)
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
        if is_stop_removal_request(message) and not is_stop_replacement_request(message):
            return self._validated(context, unsupported_stop_removal())

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
                    highlights=self._operation_highlights(context, operation),
                ),
            )

        if any(term in normalized for term in ("weather", "risk", "safer", "repair", "reduce driving", "easier")):
            guarded = request_semantic_guard(context, message, proposed_edit=True)
            if guarded is not None:
                return self._validated(context, guarded)
            preference = "reduce_contextual_risk"
            if "driv" in normalized or "easier" in normalized:
                preference = "reduce_driving"
            return self._validated(
                context,
                CopilotInterpretationV1(
                    state="proposal_ready",
                    assistant_message=(
                        f"{self.provider_label}: I found one registered, independently "
                        "evaluated repair. Use Compare or Evidence to inspect it, or ask "
                        "for a specific supported edit; nothing was changed automatically."
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
        place_clarification = known_place_clarification(context, message)
        if place_clarification is not None:
            assistant_message, question = place_clarification
            return self._validated(
                context,
                CopilotInterpretationV1(
                    state="clarification_required",
                    assistant_message=assistant_message,
                    intents=(
                        CopilotIntentV1(
                            type="request_clarification",
                            clarification_question=question,
                        ),
                    ),
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
        named_place = matched_known_place(context, normalized)
        if named_place is not None and named_place["route_status"] == "discovery_only_route_validation_required":
            return None
        mentioned_stop = matched_known_place(
            context,
            normalized,
            route_status="accepted_stop",
        )
        mentioned_candidate = matched_known_place(
            context,
            normalized,
            route_status="registered_route_candidate",
        )
        stop_id = (
            str(mentioned_stop["place_id"])
            if mentioned_stop is not None and mentioned_stop["place_id"] in context.allowed_stop_ids
            else context.selected_stop_id
        )
        candidate_id = (
            str(mentioned_candidate["place_id"])
            if mentioned_candidate is not None and mentioned_candidate["place_id"] in context.allowed_candidate_ids
            else context.selected_candidate_id
        )
        if "keep" in normalized and ("stop" in normalized or mentioned_stop is not None) and stop_id is not None:
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
        day_match = re.search(r"\bday\s+(\d+)\b", normalized)
        destination_day = int(day_match.group(1)) if day_match else context.selected_day
        if "add" in normalized and candidate_id is not None and destination_day in context.allowed_days:
            return CopilotIntentV1(
                type="add_candidate",
                destination_day=destination_day,
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
    def _operation_highlights(
        context: CopilotContextV1,
        operation: CopilotIntentV1,
    ) -> CopilotHighlightsV1:
        """Highlight the resolved proposal, not an unrelated prior selection."""

        if operation.type == "route_feedback":
            return DeterministicCopilotAdapter._selected_highlights(context)
        target_day = operation.destination_day
        if target_day is None and operation.target_stop_id is not None:
            target_day = next(
                (
                    row["day"]
                    for row in getattr(context, "capabilities", {}).get("stop_positions", ())
                    if row.get("stop_id") == operation.target_stop_id
                ),
                None,
            )
        return CopilotHighlightsV1(
            day_ids=((target_day,) if target_day is not None else ()),
            stop_ids=((operation.target_stop_id,) if operation.target_stop_id is not None else ()),
            candidate_ids=((operation.candidate_id,) if operation.candidate_id is not None else ()),
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
    expected_schema = {CopilotContextV1: "copilot-context-v1", CopilotContextV2: "copilot-context-v2"}
    if type(context) not in expected_schema or context.schema_version != expected_schema[type(context)]:
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
    _validate_known_places(context.known_places)
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
    if context.selected_segment_id is not None and context.selected_segment_id not in context.allowed_segment_ids:
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
    if type(context) is CopilotContextV2:
        from .copilot_memory import validate_context_extensions

        try:
            validate_context_extensions(context)
        except (ValueError, TypeError, KeyError):
            raise CopilotContractError from None


def _validate_intent(context: CopilotContextV1, intent: CopilotIntentV1) -> None:
    if type(intent) not in {CopilotIntentV1, CopilotIntentV2} or intent.type not in ALLOWED_INTENT_TYPES:
        raise CopilotContractError
    if intent.type == "replan_trip":
        from .trip_planning import validate_trip_parameters

        if type(intent) is not CopilotIntentV2 or any(
            getattr(intent, key) is not None
            for key in ("target_stop_id", "destination_day", "candidate_id", "preference", "clarification_question")
        ):
            raise CopilotContractError
        try:
            validate_trip_parameters(intent.parameters)
        except (ValueError, TypeError, KeyError):
            raise CopilotContractError from None
        return
    if intent.type in TYPED_CHAT_OPERATIONS:
        if (
            type(intent) is not CopilotIntentV2
            or intent.target_stop_id not in context.allowed_stop_ids
            or any(
                getattr(intent, key) is not None
                for key in ("destination_day", "candidate_id", "preference", "clarification_question")
            )
        ):
            raise CopilotContractError
        try:
            validate_parameters(intent.type, intent.parameters)
        except (ValueError, TypeError, KeyError):
            raise CopilotContractError from None
        return
    if type(intent) is CopilotIntentV2 and intent.parameters is not None:
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


def _validate_known_places(values: object) -> None:
    if type(values) is not tuple or len(values) > 64:
        raise CopilotContractError
    seen: set[str] = set()
    fields = {"place_id", "name", "aliases", "categories", "region_id", "route_status"}
    for value in values:
        if type(value) is not dict or set(value) != fields:
            raise CopilotContractError
        place_id = value["place_id"]
        name = value["name"]
        aliases = value["aliases"]
        categories = value["categories"]
        region_id = value["region_id"]
        route_status = value["route_status"]
        if (
            type(place_id) is not str
            or not place_id
            or place_id in seen
            or type(name) is not str
            or not name.strip()
            or name != name.strip()
            or type(aliases) is not tuple
            or type(categories) is not tuple
            or any(type(item) is not str or not item.strip() for item in (*aliases, *categories))
            or len(aliases) != len({item.casefold() for item in aliases})
            or len(categories) != len({item.casefold() for item in categories})
            or region_id is not None
            and (type(region_id) is not str or not region_id)
            or route_status not in KNOWN_PLACE_ROUTE_STATUSES
        ):
            raise CopilotContractError
        seen.add(place_id)


def _normalized_place_text(value: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", value.casefold()).split())


def matched_known_place(
    context: CopilotContextV1,
    message: str,
    *,
    route_status: str | None = None,
) -> dict[str, object] | None:
    """Resolve one exact server-owned place name or alias from a user message."""

    matches = _matched_known_places(context, message, route_status=route_status)
    return matches[0] if len(matches) == 1 else None


def _matched_known_places(
    context: CopilotContextV1,
    message: str,
    *,
    route_status: str | None = None,
) -> list[dict[str, object]]:
    if route_status is not None and route_status not in KNOWN_PLACE_ROUTE_STATUSES:
        raise CopilotContractError
    normalized = _normalized_place_text(message)
    if not normalized:
        return []
    padded = f" {normalized} "
    matches: list[tuple[int, int, dict[str, object]]] = []
    for place in context.known_places:
        if route_status is not None and place["route_status"] != route_status:
            continue
        terms = (str(place["name"]), *(str(alias) for alias in place["aliases"]))
        for term in terms:
            term_normalized = _normalized_place_text(term)
            if term_normalized:
                for match in re.finditer(r"(?<!\w)" + re.escape(term_normalized) + r"(?!\w)", padded):
                    matches.append((match.start(), match.end(), place))
    # A longer name may contain a shorter alias, but a separate named place
    # elsewhere in the sentence must never disappear behind a longest match.
    unique = {
        str(place["place_id"]): place
        for start, end, place in matches
        if not any(
            other_start <= start and end <= other_end and other_end - other_start > end - start
            for other_start, other_end, _ in matches
        )
    }
    return list(unique.values())


def known_place_clarification(context: CopilotContextV1, message: str) -> tuple[str, str] | None:
    """Return safe local copy for one recognized place without claiming feasibility."""

    place = matched_known_place(context, message)
    if place is None:
        return None
    name = str(place["name"])
    status = place["route_status"]
    if status == "accepted_stop":
        question = f"{name} is already in this itinerary. Do you want to keep it, move it, or review its assigned day?"
    elif status == "registered_route_candidate":
        question = f"Do you want to add {name} or replace an existing stop with it, and on which day?"
    else:
        question = (
            f"I found {name} in the California statewide discovery catalog, but its road access "
            "and route are not validated for this trip. Do you want to evaluate adding it or "
            "replacing a stop, and which day should I check?"
        )
    return question + " Nothing was changed.", question
