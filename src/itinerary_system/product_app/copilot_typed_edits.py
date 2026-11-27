"""Bounded chat projection of existing evaluated stop-edit contracts."""

from __future__ import annotations

import re
from typing import Any

from .interaction_contracts import typed_edit_capabilities
from .models import CopilotContextV2, CopilotIntentV1, CopilotIntentV2, CopilotInterpretationV1, CopilotInterpretationV2

TYPED_CHAT_OPERATIONS = frozenset({"set_stop_duration", "set_stop_order", "set_stop_time_window", "set_stop_role"})


def validate_parameters(kind: str, parameters: Any) -> None:
    """Reuse canonical workspace validation, including conditional execution limits."""
    from .workspace import WorkspaceError, WorkspaceStore

    if type(parameters) is not dict or kind not in TYPED_CHAT_OPERATIONS:
        raise ValueError("copilot_typed_parameters_invalid")
    capabilities = typed_edit_capabilities()["operations"]
    try:
        if kind == "set_stop_duration":
            if set(parameters) != {"duration"}:
                raise ValueError("copilot_typed_parameters_invalid")
            WorkspaceStore._validate_typed_duration(parameters["duration"])
            if parameters["duration"]["mode"] not in capabilities[kind]["supported_evaluated_modes"]:
                raise ValueError("duration_mode_evaluation_not_supported")
        elif kind == "set_stop_order":
            WorkspaceStore._validate_attribute_value("sequence_order", parameters, 7)
        elif kind == "set_stop_time_window":
            WorkspaceStore._validate_time_window(parameters)
        elif set(parameters) != {"role"} or parameters["role"] not in capabilities[kind]["supported_evaluated_roles"]:
            raise ValueError("stop_role_evaluation_not_supported")
    except WorkspaceError as error:
        raise ValueError(error.code) from None


def typed_edit_kind(message: str) -> str | None:
    if re.search(r"\b(?:time window|earliest|latest|arrival|departure|arrive|leave by)\b", message, re.I):
        return "set_stop_time_window"
    if re.search(r"\b(?:role|classify)\b|\btreat\b.*\bas\b", message, re.I):
        return "set_stop_role"
    if re.search(r"\b(?:position|sequence|reorder)\b|\b(?:first|second|third|last) stop\b", message, re.I):
        return "set_stop_order"
    if re.search(
        r"\bduration\b|\b(?:spend|stay|visit|give)\b.*\b(?:minutes?|hours?)\b|\b(?:I'd like|I would like)\s+\d+\s+(?:minutes?|hours?)\b",
        message,
        re.I,
    ):
        return "set_stop_duration"
    return None


def arrival_window_clarification(message: str) -> CopilotInterpretationV1 | None:
    """An arrival deadline is not the contract's latest departure deadline."""
    if re.search(
        r"\b(?:(?:latest|maximum) arrival|earliest (?:departure|depart|leave)|"
        r"(?:departure|depart|leave)\s+(?:not\s+before|no\s+earlier\s+than|earliest)|"
        r"(?:not\s+before|no\s+earlier\s+than|earliest)\s+\d{2}:\d{2}\s+for\s+(?:departure|depart|leaving)|"
        r"arrival (?:deadline|by)|arrive by)\b",
        message,
        re.I,
    ) or (
        re.search(r"\b(?:arrival|arrive)\b", message, re.I)
        and (
            len(re.findall(r"\b\d{2}:\d{2}\b", message)) == 2
            or re.search(r"\b(?:latest|by|no later than)\b", message, re.I)
        )
        and not re.search(r"\b(?:departure|depart|leave)\b", message, re.I)
    ):
        question = (
            "Do you mean an earliest arrival and a latest departure after the visit? "
            "An arrival upper bound or a departure lower bound is not supported by this time-window edit."
        )
        return CopilotInterpretationV1(
            state="clarification_required",
            assistant_message=question + " Nothing was staged or changed.",
            intents=(CopilotIntentV1(type="request_clarification", clarification_question=question),),
        )
    return None


def typed_blocking_reason(context: CopilotContextV2, intent: CopilotIntentV2) -> str | None:
    operation = context.capabilities["operations"].get(intent.type, {})
    if operation.get("enabled") is not True or operation.get("preview_executable") is not True:
        return "This stop edit is not enabled for evaluated preview on the current accepted plan."
    try:
        validate_parameters(intent.type, intent.parameters)
    except (ValueError, TypeError, KeyError):
        return "These edit parameters are invalid or not supported by independent evaluation."
    if intent.type == "set_stop_order":
        positions = context.capabilities.get("stop_positions", ())
        selected = next((row for row in positions if row["stop_id"] == intent.target_stop_id), None)
        parameters = intent.parameters or {}
        day_rows = [row for row in positions if row["day"] == parameters["day"]]
        if selected is None or selected["day"] != parameters["day"] or parameters["sequence_index"] >= len(day_rows):
            return "Reordering is supported only within this stop's accepted day and existing stop positions."
    return None


def interpret_typed_edit(context: Any, message: str) -> CopilotInterpretationV1 | None:
    if not isinstance(context, CopilotContextV2):
        return None
    kind = typed_edit_kind(message)
    if kind is None:
        return None
    clarification = arrival_window_clarification(message)
    if clarification is not None:
        return clarification
    from .copilot import matched_known_place

    place = matched_known_place(context, message, route_status="accepted_stop")
    target = (
        str(place["place_id"])
        if place
        else context.selected_stop_id
        if re.search(r"\b(?:this|selected) stop\b", message, re.I)
        else None
    )
    operation = context.capabilities["operations"].get(kind, {})
    if operation.get("enabled") is not True or operation.get("preview_executable") is not True:
        return _unsupported("This stop edit is not enabled for evaluated preview on the current accepted plan.")
    parameters = None
    if target and kind == "set_stop_duration":
        if re.search(
            r"\b(?:at least|at most|minimum|maximum|between|prefer)\b|"
            r"\b(?:up to|(?:no )?(?:more|less|fewer|greater) than|over|under|about|approximately|"
            r"around|roughly|nearly|almost|not|except)\s+\d|"
            r"[~≈<>≤≥]\s*=?\s*\d|\b\d+\s*(?:minutes?|hours?)\s*(?:tops\b|\+)",
            message,
            re.I,
        ):
            return _unsupported("Only an exact visit duration is supported for evaluated chat previews.")
        # Do not turn one fragment of a compound, signed, decimal, or alternative
        # quantity into an exact edit. Both adapters run this check before any
        # provider dispatch; unsupported expressions fall through to clarification.
        stop_day = next(
            (row["day"] for row in context.capabilities.get("stop_positions", ()) if row["stop_id"] == target), None
        )
        numeric_text = re.sub(
            r"\bday\s+([0-9]+)\b",
            lambda match: "" if int(match[1]) == stop_day else match[0],
            message,
            flags=re.I,
        )
        # An explicitly labelled old value is context, not a second new value.
        numeric_text = re.sub(r"\binstead of\s+\d+\s*(?:minutes?|hours?)?\b", "", numeric_text, flags=re.I)
        duration = re.search(r"\b(\d+)\s*(minutes?|hours?)\b", numeric_text, re.I)
        numbers = [
            value.strip()
            for value in re.findall(
                r"[+\-\u2212]?\s*(?:[0-9]+(?:\.[0-9]+)?|\.[0-9]+)",
                numeric_text,
            )
        ]
        if duration and numbers == [duration[1]] and not re.search(r"\bor\b", numeric_text, re.I):
            minutes = int(duration[1]) * (60 if duration[2].lower().startswith("hour") else 1)
            parameters = {
                "duration": {
                    "mode": "exact",
                    "preferred_minutes": minutes,
                    "minimum_minutes": minutes,
                    "maximum_minutes": minutes,
                }
            }
    elif target and kind == "set_stop_order":
        positions = context.capabilities.get("stop_positions", ())
        row = next((item for item in positions if item["stop_id"] == target), None)
        day_match = re.search(r"\bday\s+(\d+)\b", message, re.I)
        day = int(day_match[1]) if day_match else row["day"] if row else None
        index_match = re.search(r"\bposition\s+(\d+)\b", message, re.I)
        ordinal = re.search(r"\b(first|second|third|last)\b", message, re.I)
        index = int(index_match[1]) - 1 if index_match else None
        if ordinal and day is not None:
            index = {
                "first": 0,
                "second": 1,
                "third": 2,
                "last": len([item for item in positions if item["day"] == day]) - 1,
            }[ordinal[1].lower()]
        numeric_text = re.sub(r"\bday\s+\d+\b", "", message, flags=re.I)
        numbers = [
            value.strip()
            for value in re.findall(
                r"[+\-\u2212]?\s*(?:[0-9]+(?:\.[0-9]+)?|\.[0-9]+)",
                numeric_text,
            )
        ]
        ordinals = re.findall(r"\b(?:first|second|third|last)\b", message, re.I)
        exact_position = (index_match is not None and not ordinals and numbers == [index_match[1]]) or (
            index_match is None and len(ordinals) == 1 and not numbers
        )
        if day is not None and index is not None and exact_position and not re.search(r"\bor\b", message, re.I):
            parameters = {"day": day, "sequence_index": index}
    elif target and kind == "set_stop_role":
        role_text = re.sub(r"\s+(?:now|please|at once)[.!?]?$", "", message.strip(), flags=re.I)
        role_text = re.sub(r"\s+instead of\s+(?:an?\s+)?[a-z_ ]+[.!?]?$", "", role_text, flags=re.I)
        role = re.search(
            r"\b(?:role\s+(?:to|as)|(?:classify|treat)\b.*\bas)\s+(?:an?\s+)?([a-z_ ]+?)[.!?]?$", role_text, re.I
        )
        if role:
            parameters = {"role": role[1].lower().strip().replace(" ", "_")}
    elif target and kind == "set_stop_time_window":
        times = re.findall(r"\b\d{2}:\d{2}\b", message)
        numeric_text = re.sub(r"\b\d{2}:\d{2}\b", "", message)
        stop_day = next(
            (row["day"] for row in context.capabilities.get("stop_positions", ()) if row["stop_id"] == target), None
        )
        numeric_text = re.sub(
            r"\bday\s+([0-9]+)\b", lambda match: "" if int(match[1]) == stop_day else match[0], numeric_text, flags=re.I
        )
        bounds = re.findall(
            r"\b(earliest arrival|latest departure|arrival|departure|arrive|leave by|not before|earliest|latest)"
            r"\s*(?:(?:at|to|by|not before)\s+)?(\d{2}:\d{2})\b",
            message,
            re.I,
        )
        labelled = {
            "earliest_arrival"
            if label.lower() in {"earliest arrival", "arrival", "arrive", "not before", "earliest"}
            else "latest_departure": time
            for label, time in bounds
        }
        if re.search(r"\d|\bor\b|\b(?:minutes?|hours?)\b", numeric_text, re.I):
            parameters = None
        elif len(bounds) == len(times) and len(labelled) == len(times) and times:
            parameters = {
                "earliest_arrival": labelled.get("earliest_arrival"),
                "latest_departure": labelled.get("latest_departure"),
            }
        elif len(times) == 2 and not re.search(
            r"\b(?:arrival|arrive|departure|leave|earliest|latest)\b", message, re.I
        ):
            parameters = {"earliest_arrival": times[0], "latest_departure": times[1]}
    if target is None or parameters is None:
        question = (
            "Please restate this one edit with the accepted stop name and "
            + {
                "set_stop_duration": "an exact duration in minutes.",
                "set_stop_order": "a stop position such as position 1.",
                "set_stop_role": "a role such as meal or scenic stop.",
                "set_stop_time_window": "a time window such as 10:00-12:00.",
            }[kind]
        )
        return CopilotInterpretationV1(
            state="clarification_required",
            assistant_message=question + " Nothing was staged or changed.",
            intents=(CopilotIntentV1(type="request_clarification", clarification_question=question),),
        )
    intent = CopilotIntentV2(type=kind, target_stop_id=target, parameters=parameters)
    reason = typed_blocking_reason(context, intent)
    if reason:
        return _unsupported(reason)
    return CopilotInterpretationV2(
        state="proposal_ready",
        assistant_message="I prepared this typed stop-edit suggestion for review. Nothing was staged or changed.",
        intents=(intent,),
    )


def _unsupported(reason: str) -> CopilotInterpretationV1:
    return CopilotInterpretationV1(
        state="unsupported",
        assistant_message=reason + " Nothing was staged or changed.",
        intents=(CopilotIntentV1(type="unsupported_request"),),
    )
