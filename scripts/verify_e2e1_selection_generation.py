"""Bind the real delayed selection/delete/chat journey to exact HTTP and DOM evidence."""

from __future__ import annotations

import json
from collections import Counter
from urllib.parse import urlsplit

import verify_e2e1_vertical_slice as core


def verify_selection_generation(case, trace, records, turns):
    from verify_e2e1_self_managed import _same_plan, require, stamp

    checkpoints = core._load(case / "checkpoints.json")
    runtime = core._load(case / "runtime_trace.json")
    final = checkpoints["late_selection_applied"]
    calls = final["timing"]["calls"]
    selections = [row for row in calls if row["method"] == "POST" and row["url"].endswith("/selection")]
    deletes = [row for row in calls if row["method"] == "DELETE" and row["url"].endswith("/conversation")]
    require(len(selections) == len(deletes) == 1, "selection_generation_operation_count")
    selection, deleted = selections[0], deletes[0]
    original = json.loads(selection["response_body_utf8"])["session"]
    sid = original["session_id"]
    base = "http://127.0.0.1:8127/api/sessions/" + sid
    first = next(row for row in calls if row["method"] == "GET" and row["url"] == base + "/conversation")
    before, after, renewed = (
        core._load(case / name)
        for name in ("persisted_before_release.json", "persisted_after_release.json", "persisted_after_delete.json")
    )
    current = after["body"]["session"]
    require(
        before == after
        and all(row["status"] == 200 for row in (before, after, renewed))
        and runtime["before_release"] == before
        and runtime["after_release"] == after
        and runtime["visible_after_release"] == final,
        "selection_generation_readback_unbound",
    )
    require(
        current == renewed["body"]["session"]
        and current["conversation_id"] != original["conversation_id"]
        and current["revision"] == original["revision"] == runtime["session_revision"]
        and current["conversation_id"] == runtime["new_conversation_id"]
        and original["conversation_id"] == runtime["original_conversation_id"]
        and type(current["revision"]) is int
        and current["revision"] > 0
        and all(current[key] == original[key] for key in ("session_id", "trip_id", "run_id", "accepted_plan_id"))
        and current["draft"] == []
        and current["proposal"] is None,
        "selection_generation_identity_invalid",
    )
    request = json.loads(selection["request_body_utf8"])
    require(
        request["expected_revision"] + 1 == current["revision"]
        and request["selected_day"] == 2
        and current["selected_day"] == original["selected_day"] == 2
        and current["selected_stop_id"] == request["selected_stop_id"] == "stanford_university_main_quad"
        and all(
            current[key] == request[key]
            for key in ("selected_segment_id", "selected_candidate_id", "selected_alternative_id")
        ),
        "selection_generation_committed_selection_lost",
    )
    deletion = json.loads(deleted["response_body_utf8"])
    require(
        json.loads(deleted["request_body_utf8"]) == {"expected_revision": current["revision"]}
        and deletion["deleted"] is True
        and deletion["session"]["conversation_id"] is None
        and {key: value for key, value in deletion["session"].items() if key != "conversation_id"}
        == {key: value for key, value in original.items() if key != "conversation_id"}
        and json.loads(first["response_body_utf8"])["session"] == original,
        "selection_generation_delete_unbound",
    )

    # Every passive timing observation is bound to a complete exact HTTP capture,
    # including multiplicity; a fabricated timing-only GET cannot satisfy this.
    def key(row, *, observed=False):
        return (
            row["method"],
            row["url"],
            row["request_body_utf8"],
            row["response_body_utf8"],
            row["response_status"] if observed else row["response_status_code"],
        )

    relevant = [row for row in records if row["url"] in {base, base + "/conversation", base + "/selection"}]
    require(
        Counter(key(row, observed=True) for row in calls) == Counter(key(row) for row in relevant)
        and all(row["response_status"] == 200 for row in calls),
        "selection_generation_http_unbound",
    )
    for row in calls:
        require(
            stamp(row["requested_at"])
            <= stamp(row["dispatch_at"])
            <= stamp(row["body_completed_at"])
            <= stamp(row["application_json_read_at"])
            <= stamp(final["observed_at"]),
            "selection_generation_call_timing_invalid",
        )
    held = next(row for row in checkpoints["selection_held"]["timing"]["calls"] if row["method"] == "POST")
    require(
        held["response_body_utf8"] == selection["response_body_utf8"]
        and held["release_count"] == 0
        and held["released_at"] is None
        and held["application_json_read_at"] is None
        and selection["release_count"] == first["release_count"] == 1
        and stamp(first["held_at"])
        <= stamp(selection["body_completed_at"])
        <= stamp(first["released_at"])
        <= stamp(first["dispatch_at"])
        <= stamp(first["application_json_read_at"])
        <= stamp(deleted["dispatch_at"])
        <= stamp(deleted["body_completed_at"])
        <= stamp(checkpoints["new_conversation_restored"]["observed_at"])
        <= stamp(checkpoints["new_conversation_message_visible"]["observed_at"])
        <= stamp(selection["released_at"])
        <= stamp(selection["application_json_read_at"])
        <= stamp(selection["delivery_settled_at"])
        <= stamp(final["observed_at"]),
        "selection_generation_race_order_invalid",
    )
    reconciliation = [
        row
        for row in calls
        if row["url"] == base
        and row["method"] == "GET"
        and stamp(row["dispatch_at"]) >= stamp(selection["released_at"])
    ]
    require(
        len(reconciliation) == 1
        and json.loads(reconciliation[0]["response_body_utf8"])["session"] == current
        and stamp(selection["application_json_read_at"]) <= stamp(reconciliation[0]["dispatch_at"])
        and {"workspace", "registry"} <= json.loads(reconciliation[0]["response_body_utf8"]).keys(),
        "selection_generation_reconciliation_missing",
    )
    for readback in (renewed, before, after):
        require(
            any(
                row["method"] == "GET"
                and row["url"] == base + "/conversation"
                and json.loads(row["response_body_utf8"]) == readback["body"]
                for row in calls
            ),
            "selection_generation_persisted_readback_missing",
        )

    require(len(turns) == 1, "selection_generation_chat_count_invalid")
    message = "What preferences do you remember?"
    turn = turns[0]
    output = turn["output"]
    conversation = after["body"]["conversation"]
    require(
        turn["input"]["message"] == runtime["message"] == message
        and turn["input"]["expected_revision"] == current["revision"]
        and output["type"] == "turn.completed"
        and output["payload"]["session"] == current
        and conversation["conversation_id"] == current["conversation_id"]
        and all(conversation[key] == current[key] for key in ("session_id", "trip_id", "run_id"))
        and len(conversation["turns"]) == 1
        and renewed["body"]["conversation"]["turns"] == [],
        "selection_generation_turn_owner_invalid",
    )
    persisted = conversation["turns"][0]
    require(
        all(
            persisted[key] == output["payload"]["turn"][key]
            for key in (
                "turn_id",
                "client_message_id",
                "user_message",
                "assistant_message",
                "interpretation",
                "context_revision",
                "state",
            )
        )
        and persisted["client_message_id"] == turn["input"]["client_message_id"]
        and persisted["user_message"] == message
        and persisted["context_revision"] == current["revision"]
        and persisted["provider"] == "deterministic"
        and persisted["model"] is None,
        "selection_generation_turn_readback_invalid",
    )
    for point in (checkpoints["new_conversation_message_visible"], final, checkpoints["final_observation"]):
        require(
            point["user_messages"] == [message]
            and point["selected_day"] == "2"
            and "Previous local conversation cleared" not in point["lifecycle"],
            "selection_generation_visible_chat_lost",
        )
    clicks = final["timing"]["clicks"]
    require(
        clicks
        and all(row["trusted"] is True for row in clicks)
        and any(row["day"] == "2" and stamp(row["at"]) <= stamp(selection["dispatch_at"]) for row in clicks)
        and all(
            any(row["id"] == identity for row in clicks)
            for identity in (
                "open-copilot",
                "transcript-settings-button",
                "delete-current-conversation",
                "copilot-submit",
            )
        ),
        "selection_generation_native_controls_missing",
    )
    require(
        runtime["map_initial"] == trace["initial"] == core._load(case / "map_initial.json")
        and runtime["map_final"] == trace["final"] == core._load(case / "map_final.json")
        and _same_plan(trace["initial"], trace["final"])
        and current["accepted_plan_id"] == runtime["accepted_plan_id"] == trace["final"]["active_plan_id"]
        and not any(
            row["method"] not in {"GET", "HEAD"}
            and urlsplit(row["url"]).path
            not in {
                "/api/sessions",
                urlsplit(base + "/selection").path,
                urlsplit(base + "/conversation").path,
                urlsplit(base + "/copilot/messages").path,
                urlsplit(base + "/draft/impact-preview").path,
            }
            for row in records
        ),
        "selection_generation_plan_mutation",
    )
    impacts = [row for row in records if row["url"] == base + "/draft/impact-preview"]
    require(len(impacts) == 2, "selection_generation_impact_inventory_invalid")
    for row, revision in zip(impacts, (current["revision"] - 1, current["revision"]), strict=True):
        impact = json.loads(row["response_body_utf8"])
        require(
            row["method"] == "POST"
            and json.loads(row["request_body_utf8"]) == {"expected_revision": revision}
            and impact["schema_version"] == "product-draft-impact-preview-v1"
            and impact["session_id"] == sid
            and impact["session_revision"] == revision
            and impact["parent_plan_id"] == current["accepted_plan_id"]
            and impact["parent_plan_content_hash"] == trace["initial"]["active_plan_content_hash"]
            and impact["certification_status"] == "not_certified"
            and impact["is_certified"] is False
            and impact["operations"] == [],
            "selection_generation_impact_not_read_only",
        )
    sources = core._load(case / "source_identity_before.json")
    require(
        sources == core._load(case / "source_identity_after.json")
        and sources
        and all(core._sha256(core._safe_path(core.ROOT, path)) == value for path, value in sources.items()),
        "selection_generation_source_changed",
    )
    return ("before", "before_late_selection", "after_late_selection")
