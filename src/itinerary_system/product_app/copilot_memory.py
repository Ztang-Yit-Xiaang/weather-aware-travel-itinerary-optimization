"""Small deterministic, plan-bound conversation state; never accepted constraints."""

from __future__ import annotations

import json
import re
from copy import deepcopy
from dataclasses import replace
from typing import Any

from .copilot_typed_edits import TYPED_CHAT_OPERATIONS, interpret_typed_edit, typed_blocking_reason, typed_edit_kind
from .models import CopilotContextV2, CopilotIntentV1, CopilotIntentV2, CopilotInterpretationV1

_REQUIRED = {
    "keep_stop": ("target_stop_id",),
    "lock_stop": ("target_stop_id",),
    "mark_flexible": ("target_stop_id",),
    "move_day": ("target_stop_id", "destination_day"),
    "replace_nearby": ("target_stop_id", "candidate_id"),
    "add_candidate": ("candidate_id", "destination_day"),
}
_CAPABILITY_TYPES = frozenset((*_REQUIRED, "route_feedback", "replan_trip")) | TYPED_CHAT_OPERATIONS
_PREFERENCE_PATTERNS = {
    "reduce_driving": r"(?:less|reduce|reduced|minimal|shorter)\s+driving",
    "reduce_contextual_risk": r"(?:less|reduce|reduced|lower)\s+(?:weather\s+)?risk|safer\s+(?:travel|route)",
    "preserve_experience": r"preserv(?:e|ing)\s+(?:the\s+)?experience",
}


def project_capabilities(
    capabilities: dict[str, Any],
    *,
    registered_repair_current: bool,
    typed_capabilities: dict[str, Any] | None = None,
    stop_positions: tuple = (),
) -> dict[str, Any]:
    combined = dict(capabilities.get("operations", {}))
    if typed_capabilities is not None:
        combined.update(
            {
                key: value
                for key, value in typed_capabilities.get("operations", {}).items()
                if key in TYPED_CHAT_OPERATIONS
            }
        )
    operations = {
        name: {
            "enabled": row.get("enabled") is True,
            "preview_executable": row.get("preview_executable") is True,
            "blocking_code": row.get("blocking_code")
            or (None if row.get("preview_executable") is True else row.get("status", "not_enabled")),
        }
        for name, row in sorted(combined.items())
        if name in _CAPABILITY_TYPES
    }
    return {
        "schema_version": "copilot-capabilities-v2" if typed_capabilities is not None else "copilot-capabilities-v1",
        "operations": operations,
        **(
            {
                "stop_positions": stop_positions,
                "executable_combinations": tuple(
                    {
                        "operations": tuple(
                            name
                            for name in row["operations"]
                            if name in operations and operations[name]["preview_executable"]
                        ),
                        "cardinality": row["cardinality"],
                    }
                    for row in capabilities.get("executable_combinations", ())
                ),
            }
            if typed_capabilities is not None
            else {}
        ),
        "registered_repair_current": registered_repair_current,
        "candidate_replacements": tuple(
            {
                "candidate_id": row["candidate_id"],
                "replaces_stop_id": row["replaces_stop_id"],
            }
            for row in capabilities.get("candidate_choices", ())
        ),
        "candidate_day_readiness": tuple(
            {
                "candidate_id": row["candidate_id"],
                "day": row["day"],
                "preview_executable": row.get("preview_executable") is True,
                "blocking_code": None if row.get("preview_executable") is True else row["status"],
            }
            for row in capabilities.get("operations", {}).get("add_candidate", {}).get("candidate_day_readiness", ())
        ),
    }


def empty_memory(plan_id: str, plan_hash: str) -> dict[str, Any]:
    return {
        "schema_version": "copilot-memory-v1",
        "plan_id": plan_id,
        "plan_content_hash": plan_hash,
        "pending_clarification": None,
        "resolved_ids": {"stop_ids": (), "candidate_ids": (), "day_ids": (), "segment_ids": ()},
        "proposed_preferences": (),
        "last_proposal": None,
    }


def context_binding(context: CopilotContextV2) -> dict[str, Any]:
    return {
        "plan_id": context.accepted_plan_id,
        "plan_content_hash": context.accepted_plan_content_hash,
        **{
            name: getattr(context, name)
            for name in (
                "selected_day",
                "selected_stop_id",
                "selected_candidate_id",
                "selected_segment_id",
                "selected_alternative_id",
            )
        },
    }


def bound_records(context: CopilotContextV2, records: tuple[dict[str, Any], ...]) -> tuple[dict[str, Any], ...]:
    return tuple(
        row
        for row in records
        if (
            isinstance(row.get("context_binding"), dict)
            and row["context_binding"]["plan_id"] == context.accepted_plan_id
            and row["context_binding"]["plan_content_hash"] == context.accepted_plan_content_hash
            and row["context_revision"] <= context.session_revision
            and row["outcome"] == "completed"
            and row["turn"] is not None
            and row["turn"]["state"] == "completed"
        )
    )


def _preferences(message: str) -> tuple[str, ...]:
    if not re.search(r"\b(?:I prefer|my preference|remember I prefer)\b", message, re.IGNORECASE):
        return ()
    if re.search(r"\b(?:not|don't|dont|never)\b", message, re.IGNORECASE):
        return ()
    return tuple(name for name, pattern in _PREFERENCE_PATTERNS.items() if re.search(pattern, message, re.IGNORECASE))


def _request_spec(context: CopilotContextV2, message: str) -> dict[str, Any] | None:
    from .copilot import matched_known_place, request_semantic_guard

    if request_semantic_guard(context, message) is not None or typed_edit_kind(message) is not None:
        return None
    actions = {
        "replace_nearby": r"\b(?:replace|swap|substitute)\b",
        "add_candidate": r"\badd\b",
        "move_day": r"\bmove\b",
        "lock_stop": r"\block\b",
        "mark_flexible": r"\bflexible\b",
        "keep_stop": r"\b(?:keep|retain)\b",
    }
    kinds = [name for name, pattern in actions.items() if re.search(pattern, message, re.IGNORECASE)]
    stop = matched_known_place(context, message, route_status="accepted_stop")
    candidate = matched_known_place(context, message, route_status="registered_route_candidate")
    day_match = re.search(r"\bday\s+(\d+)\b", message, re.IGNORECASE) or re.fullmatch(
        r"\s*(?:actually,?\s*)?(\d+)[.!]?\s*",
        message,
        re.IGNORECASE,
    )
    pending = context.conversation_memory["pending_clarification"]
    if len(kinds) == 1:
        spec = {"intent_type": kinds[0], "target_stop_id": None, "candidate_id": None, "destination_day": None}
    elif not kinds and pending and (stop or candidate or day_match):
        spec = {key: pending[key] for key in ("intent_type", "target_stop_id", "candidate_id", "destination_day")}
    elif not kinds and re.search(r"\bsame\b.*\bother\s+stop\b", message, re.IGNORECASE):
        previous = (context.conversation_memory["last_proposal"] or {}).get("intents", ())
        if (
            len(previous) != 1
            or previous[0]["type"] not in _REQUIRED
            or "target_stop_id" not in _REQUIRED[previous[0]["type"]]
        ):
            return None
        spec = {
            "intent_type": previous[0]["type"],
            "target_stop_id": None,
            "candidate_id": previous[0].get("candidate_id"),
            "destination_day": previous[0].get("destination_day"),
        }
    else:
        return None
    if stop:
        spec["target_stop_id"] = str(stop["place_id"])
    elif re.search(r"\b(?:this|selected)\s+stop\b", message, re.IGNORECASE):
        spec["target_stop_id"] = context.selected_stop_id
    if candidate:
        spec["candidate_id"] = str(candidate["place_id"])
    elif re.search(r"\bselected\s+candidate\b", message, re.IGNORECASE):
        spec["candidate_id"] = context.selected_candidate_id
    if day_match:
        spec["destination_day"] = int(day_match.group(1))
    elif re.search(r"\b(?:this|selected)\s+day\b", message, re.IGNORECASE):
        spec["destination_day"] = context.selected_day
    spec["missing_slots"] = tuple(name for name in _REQUIRED[spec["intent_type"]] if spec[name] is None)
    return spec


def _blocked(context: CopilotContextV2, spec: dict[str, Any]) -> str | None:
    kind = spec["intent_type"]
    capability = context.capabilities["operations"].get(kind, {})
    if capability.get("enabled") is not True or capability.get("preview_executable") is not True:
        if capability.get("blocking_code") == "accepted_plan_requires_new_repair_baseline":
            return "The accepted plan needs a new repair baseline before another edit can run."
        if kind == "add_candidate":
            return "Adding this candidate is blocked because its insertion road-route evidence is missing or invalid."
        return "This edit is not enabled in the current optimizer workflow."
    if spec["destination_day"] is not None and spec["destination_day"] not in context.allowed_days:
        return "That destination day is not in the accepted itinerary."
    if spec["intent_type"] != "move_day" and spec["target_stop_id"] and spec["destination_day"] is not None:
        accepted_day = next(
            (
                row["day"]
                for row in context.capabilities.get("stop_positions", ())
                if row["stop_id"] == spec["target_stop_id"]
            ),
            None,
        )
        if accepted_day is not None and spec["destination_day"] != accepted_day:
            return f"This stop is accepted on Day {accepted_day}. Moving stops between days is not enabled."
    choices = context.capabilities["candidate_replacements"]
    if kind == "replace_nearby" and spec["target_stop_id"] is not None:
        choices = [row for row in choices if row["replaces_stop_id"] == spec["target_stop_id"]]
        if not choices:
            return "No registered replacement is available for that accepted stop."
        if spec["candidate_id"] is not None and not any(row["candidate_id"] == spec["candidate_id"] for row in choices):
            return "That candidate is not a registered replacement for the requested stop."
    if kind == "add_candidate" and spec["candidate_id"] is not None:
        readiness = context.capabilities["candidate_day_readiness"]
        if readiness and not any(
            row["candidate_id"] == spec["candidate_id"]
            and row["preview_executable"] is True
            and (spec["destination_day"] is None or row["day"] == spec["destination_day"])
            for row in readiness
        ):
            return "The exact candidate/day insertion road-route evidence is missing or invalid."
    return None


def _question(context: CopilotContextV2, spec: dict[str, Any]) -> str:
    slot = spec["missing_slots"][0]
    if slot == "target_stop_id":
        return "Which accepted stop should I use for this edit?"
    name = next(
        (str(row["name"]) for row in context.known_places if row["place_id"] == spec["target_stop_id"]), "that stop"
    )
    if slot == "candidate_id":
        return (
            f"Which registered candidate should replace {name}?"
            if spec["intent_type"] == "replace_nearby"
            else "Which registered candidate should I add?"
        )
    return f"Which destination day should I use for {name}?"


def interpret_context_request(context: Any, message: str) -> CopilotInterpretationV1 | None:
    if not isinstance(context, CopilotContextV2):
        return None
    from .copilot import request_edit_text

    message = request_edit_text(message)
    typed = interpret_typed_edit(context, message)
    if typed is not None:
        return typed
    preferences = _preferences(message)
    preference_request = re.search(r"\b(?:I prefer|my preference|remember I prefer)\b\s*(.*)", message, re.I)
    if preference_request:
        unsupported = preference_request[1]
        for name in preferences:
            unsupported = re.sub(_PREFERENCE_PATTERNS[name], "", unsupported, flags=re.I)
        if preferences:
            unsupported = re.sub(r"\binstead\s*[.!?]?\s*$", "", unsupported, flags=re.I)
        unsupported = re.sub(r"\b(?:and|is)\b", "", unsupported, flags=re.I).strip(" ,.!?")
        question = "What itinerary change should I review with that proposed preference?"
        summary = (
            (
                "I noted your proposed preference: "
                + ", ".join(name.replace("_", " ") for name in preferences)
                + ". It is not an accepted itinerary constraint. "
            )
            if preferences
            else ""
        )
        if unsupported:
            summary += f"The preference '{unsupported}' is not supported by the current preference controls. "
        return CopilotInterpretationV1(
            state="clarification_required",
            assistant_message=summary + question,
            intents=(CopilotIntentV1(type="request_clarification", clarification_question=question),),
        )
    spec = _request_spec(context, message)
    if spec is None:
        if re.search(r"\b(?:what|which)\b.*\bpreferences?\b", message, re.IGNORECASE):
            preferences = context.conversation_memory["proposed_preferences"]
            summary = (
                ", ".join(row["preference"].replace("_", " ") for row in preferences) or "none for this accepted plan"
            )
            question = "What itinerary change should I review?"
            return CopilotInterpretationV1(
                state="clarification_required",
                assistant_message=f"Your proposed preferences are: {summary}. "
                "They are not accepted itinerary constraints. " + question,
                intents=(CopilotIntentV1(type="request_clarification", clarification_question=question),),
            )
        return None
    blocked = _blocked(context, spec)
    if blocked:
        return CopilotInterpretationV1(
            state="unsupported",
            assistant_message=blocked + " Nothing was staged or changed.",
            intents=(CopilotIntentV1(type="unsupported_request"),),
        )
    if spec["missing_slots"]:
        question = _question(context, spec)
        return CopilotInterpretationV1(
            state="clarification_required",
            assistant_message=question + " Nothing was staged or changed.",
            intents=(CopilotIntentV1(type="request_clarification", clarification_question=question),),
        )
    intent = CopilotIntentV1(type=spec["intent_type"], **{key: spec[key] for key in _REQUIRED[spec["intent_type"]]})
    from .copilot import DeterministicCopilotAdapter

    return CopilotInterpretationV1(
        state="proposal_ready",
        assistant_message="I prepared this typed suggestion for review. Nothing was staged or changed.",
        intents=(intent,),
        highlights=DeterministicCopilotAdapter._operation_highlights(context, intent),
    )


def guard_contextual_output(context: Any, interpretation: CopilotInterpretationV1) -> CopilotInterpretationV1:
    if not isinstance(context, CopilotContextV2) or interpretation.state != "proposal_ready":
        return interpretation
    reasons = []
    if len(interpretation.intents) > 1 and not any(
        len(interpretation.intents) <= row["cardinality"]
        and all(intent.type in row["operations"] for intent in interpretation.intents)
        for row in context.capabilities.get("executable_combinations", ())
    ):
        reasons.append("This complete combination is not enabled for evaluated preview on the current accepted plan.")
    for intent in interpretation.intents:
        if intent.type in TYPED_CHAT_OPERATIONS:
            reason = typed_blocking_reason(context, intent)
            if reason:
                reasons.append(reason)
        elif intent.type in _CAPABILITY_TYPES:
            reason = _blocked(context, {"intent_type": intent.type, **intent.as_dict()})
            if reason:
                reasons.append(reason)
        elif intent.type == "review_registered_repair" and not context.capabilities["registered_repair_current"]:
            reasons.append("The registered repair belongs to the original plan, not the accepted child.")
    if not reasons:
        if len(interpretation.intents) > 1:
            from .models import DraftOperationV1
            from .workspace import WorkspaceError, WorkspaceStore

            operations = []
            for index, intent in enumerate(interpretation.intents):
                parameters = getattr(intent, "parameters", None) or {
                    "replace_nearby": {"candidate_id": intent.candidate_id},
                    "add_candidate": {"day": intent.destination_day},
                    "move_day": {"day": intent.destination_day},
                    "route_feedback": {"preference": intent.preference},
                }.get(intent.type, {})
                target = (
                    intent.candidate_id
                    if intent.type == "add_candidate"
                    else "selected_route"
                    if intent.type == "route_feedback"
                    else intent.target_stop_id
                )
                operations.append(
                    DraftOperationV1(
                        operation_id=f"copilot_clause_{index}",
                        type=intent.type,
                        target=target,
                        parameters=parameters,
                        source="copilot",
                    )
                )
            try:
                WorkspaceStore._validate_draft_conflicts(tuple(operations))
            except WorkspaceError as error:
                question = {
                    "draft_duplicate_operation": "The same edit was requested more than once. Should I keep just one copy?",
                    "draft_conflicting_attribute_edits": "Different values were requested for the same stop attribute. Which value should I use?",
                    "draft_conflicts_with_replacement": "A stop cannot be preserved or edited and replaced in the same request. Which change should I keep?",
                    "draft_conflicting_candidates": "These destination candidates cannot be evaluated together. Which candidate should I use?",
                }.get(error.code, "These edits conflict with each other. Which change should I handle first?")
                return CopilotInterpretationV1(
                    state="clarification_required",
                    assistant_message=question + " No part was staged or changed.",
                    intents=(CopilotIntentV1(type="request_clarification", clarification_question=question),),
                )
        return interpretation
    return CopilotInterpretationV1(
        state="unsupported",
        assistant_message=" ".join(dict.fromkeys(reasons)) + " No part was staged or changed.",
        intents=(CopilotIntentV1(type="unsupported_request"),),
    )


def derive_memory(context: CopilotContextV2, records: tuple[dict[str, Any], ...]) -> dict[str, Any]:
    memory = empty_memory(context.accepted_plan_id, context.accepted_plan_content_hash)
    for row in bound_records(context, records):
        turn = row["turn"]
        binding = row["context_binding"]
        local = replace(
            context,
            conversation_memory=memory,
            **{name: binding[name] for name in binding if name.startswith("selected_")},
        )
        preferences = _preferences(turn["user_message"])
        if preferences:
            prior = (
                []
                if re.search(r"\b(?:actually|instead|rather)\b", turn["user_message"], re.IGNORECASE)
                else list(memory["proposed_preferences"])
            )
            for preference in preferences:
                prior = [item for item in prior if item["preference"] != preference]
                prior.append(
                    {"preference": preference, "source_turn_id": turn["turn_id"], "disposition": "proposed_only"}
                )
            memory["proposed_preferences"] = tuple(prior)
        interpretation = turn["interpretation"] or {}
        state = interpretation.get("state")
        spec = _request_spec(local, turn["user_message"])
        if typed_edit_kind(turn["user_message"]) is not None:
            memory["pending_clarification"] = None
        if state == "clarification_required" and spec and spec["missing_slots"] and _blocked(local, spec) is None:
            memory["pending_clarification"] = {
                "source_turn_id": turn["turn_id"],
                **spec,
                "question": _question(local, spec),
            }
            memory["resolved_ids"] = {
                "stop_ids": (spec["target_stop_id"],) if spec["target_stop_id"] else (),
                "candidate_ids": (spec["candidate_id"],) if spec["candidate_id"] else (),
                "day_ids": (spec["destination_day"],) if spec["destination_day"] else (),
                "segment_ids": (),
            }
        elif state in {"proposal_ready", "unsupported", "refused"}:
            memory["pending_clarification"] = None
            if state != "proposal_ready":
                memory["resolved_ids"] = {"stop_ids": (), "candidate_ids": (), "day_ids": (), "segment_ids": ()}
        if state == "proposal_ready":
            intents = tuple(deepcopy(interpretation["intents"]))
            memory["last_proposal"] = {
                "source_turn_id": turn["turn_id"],
                "disposition": "suggested",
                "intents": intents,
            }
            memory["resolved_ids"] = {
                "stop_ids": tuple(
                    dict.fromkeys(item["target_stop_id"] for item in intents if item.get("target_stop_id"))
                ),
                "candidate_ids": tuple(
                    dict.fromkeys(item["candidate_id"] for item in intents if item.get("candidate_id"))
                ),
                "day_ids": tuple(
                    dict.fromkeys(item["destination_day"] for item in intents if item.get("destination_day"))
                ),
                "segment_ids": tuple(interpretation.get("highlights", {}).get("segment_ids", ())),
            }
    proposal = memory["last_proposal"]
    if (
        proposal
        and proposal["intents"]
        and all(_intent_is_staged(intent, context.draft_operations) for intent in proposal["intents"])
    ):
        proposal["disposition"] = "staged"
        evaluated = context.evaluated_proposal or {}
        if (
            evaluated.get("state") == "eligible"
            and evaluated.get("session_revision") == context.session_revision
            and evaluated.get("child_plan_id")
            and evaluated.get("child_plan_content_hash")
        ):
            proposal["disposition"] = "previewed"
    return memory


def _intent_is_staged(intent: dict[str, Any], draft: tuple[dict[str, Any], ...]) -> bool:
    for row in draft:
        if row["type"] != intent["type"]:
            continue
        parameters = row.get("parameters", {})
        if intent["type"] == "add_candidate":
            if row["target"] == intent["candidate_id"] and parameters.get("day") == intent["destination_day"]:
                return True
        elif row["target"] == intent.get("target_stop_id"):
            if intent["type"] in TYPED_CHAT_OPERATIONS:
                return parameters == intent.get("parameters")
            if intent["type"] == "replace_nearby":
                return parameters.get("candidate_id") == intent["candidate_id"]
            if intent["type"] == "move_day":
                return parameters.get("day") == intent["destination_day"]
            return intent["type"] in {"keep_stop", "lock_stop", "mark_flexible"}
    return False


def validate_context_extensions(context: CopilotContextV2) -> None:
    """Strictly bound every v2 field before it is used or sent to a provider."""
    from .copilot import _PROPOSAL_INTENTS, ALLOWED_PREFERENCES, _validate_intent

    def require(condition: bool) -> None:
        if not condition:
            raise ValueError("copilot_context_v2_invalid")

    def text(value: Any, maximum: int) -> bool:
        return type(value) is str and 0 < len(value) <= maximum and not any(ord(char) < 32 for char in value)

    def turn_id(value: Any) -> bool:
        return type(value) is str and re.fullmatch(r"copilot_turn_[a-f0-9]{32}", value) is not None

    require(
        type(context.accepted_plan_content_hash) is str
        and re.fullmatch(r"[a-f0-9]{16,64}", context.accepted_plan_content_hash) is not None
    )
    caps = context.capabilities
    require(type(caps) is dict)
    typed_caps = caps.get("schema_version") == "copilot-capabilities-v2"
    require(
        set(caps)
        == {
            "schema_version",
            "operations",
            "registered_repair_current",
            "candidate_replacements",
            "candidate_day_readiness",
        }
        | ({"stop_positions", "executable_combinations"} if typed_caps else set())
    )
    require(
        caps["schema_version"] in {"copilot-capabilities-v1", "copilot-capabilities-v2"}
        and type(caps["registered_repair_current"]) is bool
    )
    if typed_caps:
        combinations = caps["executable_combinations"]
        require(type(combinations) is tuple and len(combinations) <= 16)
        for combination in combinations:
            require(type(combination) is dict and set(combination) == {"operations", "cardinality"})
            names = combination["operations"]
            require(type(names) is tuple and len(names) == len(set(names)) and set(names) <= set(caps["operations"]))
            require(all(caps["operations"][name]["preview_executable"] for name in names))
            require(type(combination["cardinality"]) is int and 1 <= combination["cardinality"] <= 32)
        positions = caps["stop_positions"]
        require(type(positions) is tuple and len(positions) == len(context.allowed_stop_ids) <= 64)
        require(tuple(row.get("stop_id") for row in positions if type(row) is dict) == context.allowed_stop_ids)
        per_day = {}
        for row in positions:
            require(type(row) is dict and set(row) == {"stop_id", "day", "sequence_index"})
            require(
                type(row["day"]) is int and row["day"] in context.allowed_days and type(row["sequence_index"]) is int
            )
            require(row["sequence_index"] == per_day.get(row["day"], 0))
            per_day[row["day"]] = row["sequence_index"] + 1
    require(type(caps["operations"]) is dict and set(caps["operations"]) <= _CAPABILITY_TYPES)
    for capability in caps["operations"].values():
        require(type(capability) is dict and set(capability) == {"enabled", "preview_executable", "blocking_code"})
        require(type(capability["enabled"]) is bool and type(capability["preview_executable"]) is bool)
        require(not capability["preview_executable"] or capability["enabled"])
        require(capability["blocking_code"] is None or text(capability["blocking_code"], 128))
    require(type(caps["candidate_replacements"]) is tuple and len(caps["candidate_replacements"]) <= 64)
    seen = set()
    for row in caps["candidate_replacements"]:
        require(type(row) is dict and set(row) == {"candidate_id", "replaces_stop_id"})
        require(
            row["candidate_id"] in context.allowed_candidate_ids and row["replaces_stop_id"] in context.allowed_stop_ids
        )
        identity = (row["candidate_id"], row["replaces_stop_id"])
        require(identity not in seen)
        seen.add(identity)
    require(type(caps["candidate_day_readiness"]) is tuple and len(caps["candidate_day_readiness"]) <= 128)
    seen = set()
    for row in caps["candidate_day_readiness"]:
        require(type(row) is dict and set(row) == {"candidate_id", "day", "preview_executable", "blocking_code"})
        require(
            row["candidate_id"] in context.allowed_candidate_ids
            and type(row["day"]) is int
            and row["day"] in context.allowed_days
        )
        require(
            type(row["preview_executable"]) is bool
            and (row["blocking_code"] is None or text(row["blocking_code"], 128))
        )
        identity = (row["candidate_id"], row["day"])
        require(identity not in seen)
        seen.add(identity)
    memory = context.conversation_memory
    require(
        type(memory) is dict
        and set(memory)
        == {
            "schema_version",
            "plan_id",
            "plan_content_hash",
            "pending_clarification",
            "resolved_ids",
            "proposed_preferences",
            "last_proposal",
        }
    )
    require(
        memory["schema_version"] == "copilot-memory-v1"
        and memory["plan_id"] == context.accepted_plan_id
        and memory["plan_content_hash"] == context.accepted_plan_content_hash
    )
    universes = {
        "stop_ids": context.allowed_stop_ids,
        "candidate_ids": context.allowed_candidate_ids,
        "day_ids": context.allowed_days,
        "segment_ids": context.allowed_segment_ids,
    }
    require(type(memory["resolved_ids"]) is dict and set(memory["resolved_ids"]) == set(universes))
    for name, universe in universes.items():
        values = memory["resolved_ids"][name]
        require(type(values) is tuple and len(values) <= 4 and len(values) == len(set(values)))
        require(all(type(value) is (int if name == "day_ids" else str) and value in universe for value in values))
    pending = memory["pending_clarification"]
    if pending is not None:
        require(
            type(pending) is dict
            and set(pending)
            == {
                "source_turn_id",
                "intent_type",
                "target_stop_id",
                "candidate_id",
                "destination_day",
                "missing_slots",
                "question",
            }
        )
        require(
            turn_id(pending["source_turn_id"])
            and pending["intent_type"] in _REQUIRED
            and text(pending["question"], 512)
        )
        for field, universe in (
            ("target_stop_id", context.allowed_stop_ids),
            ("candidate_id", context.allowed_candidate_ids),
            ("destination_day", context.allowed_days),
        ):
            value = pending[field]
            require(value is None or type(value) is (int if field == "destination_day" else str) and value in universe)
        require(
            pending["missing_slots"]
            == tuple(field for field in _REQUIRED[pending["intent_type"]] if pending[field] is None)
        )
        require(bool(pending["missing_slots"]) and _blocked(context, pending) is None)
        require(pending["question"] == _question(context, pending))
    preferences = memory["proposed_preferences"]
    require(type(preferences) is tuple and len(preferences) <= len(ALLOWED_PREFERENCES))
    seen = set()
    for preference in preferences:
        require(type(preference) is dict and set(preference) == {"preference", "source_turn_id", "disposition"})
        require(preference["preference"] in ALLOWED_PREFERENCES and preference["preference"] not in seen)
        require(turn_id(preference["source_turn_id"]) and preference["disposition"] == "proposed_only")
        seen.add(preference["preference"])
    proposal = memory["last_proposal"]
    if proposal is not None:
        require(type(proposal) is dict and set(proposal) == {"source_turn_id", "disposition", "intents"})
        require(turn_id(proposal["source_turn_id"]) and proposal["disposition"] in {"suggested", "staged", "previewed"})
        require(type(proposal["intents"]) is tuple and 0 < len(proposal["intents"]) <= 4)
        for row in proposal["intents"]:
            require(type(row) is dict and row.get("type") in _PROPOSAL_INTENTS)
            _validate_intent(context, (CopilotIntentV2 if "parameters" in row else CopilotIntentV1)(**row))
        if proposal["disposition"] in {"staged", "previewed"}:
            require(all(_intent_is_staged(row, context.draft_operations) for row in proposal["intents"]))
        if proposal["disposition"] == "previewed":
            evaluated = context.evaluated_proposal or {}
            require(
                evaluated.get("state") == "eligible"
                and evaluated.get("session_revision") == context.session_revision
                and bool(evaluated.get("child_plan_id"))
                and bool(evaluated.get("child_plan_content_hash"))
            )
    require(len(json.dumps(caps, allow_nan=False)) <= 14000 and len(json.dumps(memory, allow_nan=False)) <= 8000)
