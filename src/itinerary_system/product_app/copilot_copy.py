"""Plain-language copy for a typed proposal, built from the intents alone.

The provider's own prose is never shown, because it could claim an action that
did not happen. This copy names exactly what the typed intents will change, so
the traveler can see that the request was read as meant.
"""

from __future__ import annotations

from typing import Any

PROPOSAL_TAIL = "I will check the new route and schedule on the map; nothing is saved until you accept it."


def _name(context: Any, place_id: str | None) -> str:
    for row in getattr(context, "known_places", ()) or ():
        if row.get("place_id") == place_id:
            return str(row.get("name") or place_id)
    return str(place_id or "that stop")


def _accepted_day(context: Any, stop_id: str | None) -> int | None:
    positions = (getattr(context, "capabilities", None) or {}).get("stop_positions", ())
    return next((row["day"] for row in positions if row.get("stop_id") == stop_id), None)


def _minutes(value: Any) -> str:
    if not isinstance(value, int):
        return "the requested time"
    hours, minutes = divmod(value, 60)
    if hours and minutes:
        return f"{hours} h {minutes} min"
    return f"{hours} h" if hours else f"{minutes} minutes"


def describe_intent(context: Any, intent: Any) -> str | None:
    kind = intent.type
    stop = _name(context, intent.target_stop_id)
    day = _accepted_day(context, intent.target_stop_id)
    on_day = f" on Day {day}" if day is not None else ""
    parameters = getattr(intent, "parameters", None) or {}
    if kind == "replace_nearby":
        return f"replace {stop}{on_day} with {_name(context, intent.candidate_id)}"
    if kind == "move_day":
        origin = f" from Day {day}" if day is not None else ""
        return f"move {stop}{origin} to Day {intent.destination_day}"
    if kind == "remove_stop":
        origin = f" from Day {day}" if day is not None else ""
        return f"remove {stop}{origin} with no replacement"
    if kind == "add_candidate":
        return f"add {_name(context, intent.candidate_id)} on Day {intent.destination_day}"
    if kind == "keep_stop":
        return f"keep {stop}{on_day} in the trip"
    if kind == "lock_stop":
        return f"lock {stop}{on_day} so later repairs leave it in place"
    if kind == "mark_flexible":
        return f"mark {stop} as flexible"
    if kind == "set_stop_duration":
        duration = parameters.get("duration") or {}
        low, high = duration.get("minimum_minutes"), duration.get("maximum_minutes")
        if duration.get("mode") == "exact" or low == high:
            return f"set the time at {stop} to {_minutes(duration.get('preferred_minutes', low))}"
        return f"set the time at {stop} to between {_minutes(low)} and {_minutes(high)}"
    if kind == "set_stop_order":
        return f"change where {stop} sits in its day"
    if kind == "set_stop_time_window":
        return f"change the visiting hours for {stop}"
    if kind == "set_stop_role":
        role = parameters.get("role")
        return f"treat {stop} as {str(role).replace('_', ' ')}" if role else f"change the role of {stop}"
    if kind == "route_feedback":
        preference = str(intent.preference or "your route preference").replace("_", " ")
        return f"review the route with your preference ({preference})"
    if kind == "review_registered_repair":
        return "review the registered weather repair"
    if kind == "keep_original":
        return "keep the original itinerary"
    return None


def clarification_examples(context: Any) -> str | None:
    """A next step the traveler can type, built from the real trip, when the request stayed unclear."""
    capabilities = getattr(context, "capabilities", None) or {}
    positions = capabilities.get("stop_positions", ())
    if not positions:
        return None
    examples = []
    replacements = capabilities.get("candidate_replacements", ())
    if replacements:
        row = replacements[0]
        examples.append(f"Replace {_name(context, row['replaces_stop_id'])} with {_name(context, row['candidate_id'])}")
    days = sorted({row["day"] for row in positions})
    mover = positions[-1]
    other = next((day for day in reversed(days) if day != mover["day"]), None)
    if other is not None:
        examples.append(f"Move {_name(context, mover['stop_id'])} to day {other}")
    examples.append(f"Remove {_name(context, positions[0]['stop_id'])}")
    quoted = ", ".join(f'"{example}"' for example in examples)
    return (
        "I could not turn that into one specific change yet. Name the stop and what to do with it, "
        f"for example {quoted}. Nothing was changed."
    )


def describe_proposal(context: Any, intents: tuple) -> str | None:
    """One sentence that names every change, or None when a change has no plain description."""
    parts = [describe_intent(context, intent) for intent in intents]
    if not parts or any(part is None for part in parts):
        return None
    if len(parts) == 1:
        return f"I prepared a change that will {parts[0]}. {PROPOSAL_TAIL}"
    listed = "; ".join(f"({index}) {part}" for index, part in enumerate(parts, start=1))
    return f"I prepared {len(parts)} changes together: {listed}. {PROPOSAL_TAIL}"
