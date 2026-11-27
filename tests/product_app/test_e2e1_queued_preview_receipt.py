"""Portable synthetic receipt counterexamples, not browser acceptance evidence."""

from __future__ import annotations

import hashlib
import json
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import verify_e2e1_self_managed as verifier  # noqa: E402


def _seal(row):
    for side in ("request", "response"):
        digest = hashlib.sha256(row[side + "_body_utf8"].encode()).hexdigest()
        row[side + "_body_sha256"] = row[side + "_original_body_sha256"] = digest
    return row


def _fixture(context, combined):
    continued = context == "accepted_child"
    name = (
        "combined_chat_async_delete_"
        + ("combined" if combined else "single")
        + ("_accepted_child" if continued else "")
    )
    message = (
        "Set Pfeiffer Big Sur State Park duration to 90 minutes"
        if continued
        else "Replace Bixby Creek Bridge with Pfeiffer Big Sur State Park"
    )
    duration = {"duration": {"mode": "exact", "preferred_minutes": 90, "minimum_minutes": 90, "maximum_minutes": 90}}
    operations = (
        [{"type": "set_stop_duration", "target": "pfeiffer_big_sur_state_park", "parameters": duration}]
        if continued
        else [
            {
                "type": "replace_nearby",
                "target": "bixby_creek_bridge_viewpoint",
                "parameters": {"candidate_id": "pfeiffer_big_sur_state_park"},
            }
        ]
    )
    if combined:
        message += (
            "; set Stearns Wharf earliest arrival to 10:00."
            if continued
            else "; set Griffith Observatory duration to 90 minutes."
        )
        operations.append(
            {
                "type": "set_stop_time_window",
                "target": "stearns_wharf",
                "parameters": {"earliest_arrival": "10:00", "latest_departure": None},
            }
            if continued
            else {"type": "set_stop_duration", "target": "griffith_observatory", "parameters": duration}
        )
    operations = [{**row, "source": "copilot"} for row in operations]
    session = {
        "session_id": "unit-owner",
        "trip_id": "unit-trip",
        "run_id": "unit-run",
        "accepted_plan_id": "unit-parent",
        "revision": 0,
        "draft": [],
        "proposal": None,
        "conversation_id": "unit-old-chat",
    }
    initial = {
        "active_plan_id": "unit-parent",
        "active_plan_content_hash": "unit-hash",
        "rendered_map_plan_id": "unit-parent",
        "rendered_geometry_hash": "unit-geometry",
        "rendered_marker_ids": ["unit-marker"],
        "preview_state": "PARENT_RENDERED",
    }
    intent = [
        {
            "type": row["type"],
            "target_stop_id": row["target"],
            "candidate_id": row["parameters"].get("candidate_id"),
            "parameters": None if row["type"] == "replace_nearby" else row["parameters"],
        }
        for row in operations
    ]
    request = {"message": message, "expected_revision": 0, "client_message_id": "client_message_" + "a" * 32}
    turn = {
        "user_message": message,
        "context_revision": 0,
        "client_message_id": request["client_message_id"],
        "interpretation": {"state": "proposal_ready", "intents": intent},
    }
    events = [
        {
            "schema_version": "copilot-stream-v1",
            "type": kind,
            "sequence": index,
            "client_message_id": request["client_message_id"],
            "request_id": "copilot_request_" + "b" * 32,
        }
        for index, kind in enumerate(("message.accepted", "assistant.status", "turn.completed"), 1)
    ]
    events[1].update(state="interpreting", message="Reading the current itinerary and validating a typed suggestion.")
    events[2]["payload"] = {"session": session, "turn": turn}
    staged = {
        **session,
        "revision": 1,
        "draft": [
            {**row, "operation_id": "unit-" + str(index), "created_at": "unit"} for index, row in enumerate(operations)
        ],
    }
    deleted = {**staged, "conversation_id": None}
    restored = {**staged, "conversation_id": "unit-new-chat"}
    chat = {"session": restored, "conversation": {"conversation_id": "unit-new-chat", "turns": []}}
    stage_request = (
        {"operations": operations, "expected_revision": 0} if combined else {**operations[0], "expected_revision": 0}
    )
    impact = {
        "session_id": session["session_id"],
        "session_revision": 1,
        "parent_plan_id": "unit-parent",
        "parent_plan_content_hash": "unit-hash",
        "summary": {"operation_count": len(operations), "can_run_evaluated_preview": True},
    }
    records = []
    for method, suffix, body, output in (
        ("POST", "copilot/messages", request, None),
        ("POST", "draft/operations", stage_request, {"session": staged}),
        ("POST", "draft/impact-preview", {"expected_revision": 1}, impact),
        ("DELETE", "conversation", {"expected_revision": 1}, {"session": deleted}),
        ("GET", "conversation", None, chat),
        ("GET", "", None, {"session": restored}),
        ("GET", "conversation", None, chat),
    ):
        row = {
            "schema_version": "e2e1-product-http-io-v1",
            "body_capture": "exact_utf8",
            "request_headers_persisted": False,
            "response_headers_persisted": False,
            "authorization_persisted": False,
            "session_token_persisted": False,
            "credential_redactions": {"request": [], "response": []},
            "url": "http://127.0.0.1:8127/api/sessions/unit-owner" + ("/" + suffix if suffix else ""),
            "method": method,
            "response_status_code": 200,
            "response_capture_complete": True,
            "request_body_utf8": json.dumps(body) if body else "",
            "response_body_utf8": json.dumps(output)
            if output
            else "".join("data: " + json.dumps(event) + "\n\n" for event in events),
        }
        records.append(_seal(row))

    def ticks(second):
        return "2026-10-07T00:00:" + f"{second:02d}" + "+00:00"

    calls = [
        {
            "method": row["method"],
            "url": row["url"],
            "request_body_utf8": row["request_body_utf8"],
            "dispatch_at": ticks(second),
        }
        for row, second in zip(records[1:4], (2, 3, 5), strict=True)
    ]
    held = {
        **calls[1],
        "response_body_utf8": records[2]["response_body_utf8"],
        "response_status": 200,
        "held_at": ticks(4),
    }
    observed = {
        "session_id": "unit-owner",
        "trip_id": "unit-trip",
        "run_id": "unit-run",
        "session_revision": 1,
        "accepted_plan_id": "unit-parent",
        "accepted_plan_content_hash": "unit-hash",
        "evaluated_proposal": None,
        "draft_operations": [{key: row[key] for key in ("type", "target", "parameters")} for row in operations],
    }
    trace = {
        "scenario": name,
        "context": context,
        "combined": combined,
        "initial": initial,
        "final": initial,
        "assertions_passed": True,
        "accepted": False,
        "preview_request_count": 0,
    }
    race = {
        "combined": combined,
        "input": message,
        "turn": turn,
        "initial_map": initial,
        "final_map": initial,
        "before_delete": observed,
        "after_delete": observed,
        "after_release": observed,
        "preview_requests": [],
        "llm_input": None,
        "llm_output": None,
        "provider_attempts": 0,
        "timing": {"calls": calls, "held": held, "released": True, "json_read": True, "released_at": ticks(6)},
    }
    artifacts = {
        "trace.json": trace,
        "race_io.json": race,
        "exact_chat_io.json": [{"input": request, "events": events, "output": events[-1]}],
        "context_queued_before_stage.json": {
            **observed,
            "session_revision": 0,
            "draft_operations": [],
            "allowed_stop_ids": ["pfeiffer_big_sur_state_park" if continued else "bixby_creek_bridge_viewpoint"],
        },
        **{
            "context_" + label + ".json": observed
            for label in ("before_delete", "after_delete", "after_release", "after_reload")
        },
        "reset_restore_http.json": {key: val for key, val in records[4].items() if key != "response_capture_complete"},
        "reload_conversation_http.json": {
            key: val for key, val in records[6].items() if key != "response_capture_complete"
        },
        "controls_before_reload.json": [
            {
                "trusted": True,
                "label": "Preview combined repair" if combined else "Preview this repair",
                "id": "",
                "at": datetime.fromisoformat(ticks(1)).timestamp() * 1000,
            },
            {
                "trusted": True,
                "label": "Delete this conversation",
                "id": "delete-current-conversation",
                "at": datetime.fromisoformat(ticks(5)).timestamp() * 1000,
            },
        ],
        "http_index.json": [{"file": "http_" + str(index) + ".json"} for index in range(len(records))],
        **{"http_" + str(index) + ".json": row for index, row in enumerate(records)},
    }
    if continued:
        artifacts["accepted_baseline_artifacts.json"] = {
            "context": context,
            "accepted_map": initial,
            "http_boundary_files": [],
        }
    return name, deepcopy(artifacts)


@pytest.mark.parametrize("context", ["original", "accepted_child"])
@pytest.mark.parametrize("combined", [False, True])
def test_queued_receipt_binds_all_four_control_journeys(context, combined):
    name, artifacts = _fixture(context, combined)
    result = verifier.queued_preview_control_receipt(artifacts.__getitem__, name=name)
    assert result["context"] == context and result["combined"] is combined
    assert result["draft_operation_count"] == (2 if combined else 1) and result["preview_request_count"] == 0


@pytest.mark.parametrize("context", ["original", "accepted_child"])
@pytest.mark.parametrize(
    "fault",
    [
        "untrusted_preview",
        "untrusted_delete",
        "release_before_delete",
        "held_wrong_body",
        "held_wrong_revision",
        "response_not_consumed",
        "changed_map",
        "changed_hash",
        "changed_draft",
        "wrong_parent",
        "provider_attempt",
        "unexpected_preview",
        "foreign_delete",
        "wrong_delete_revision",
        "reload_draft_lost",
        "old_reload_transcript",
        "incomplete_http",
        "different_input",
        "partial_stage",
        "lost_clause",
    ],
)
def test_queued_receipt_rejects_disconnected_or_misleading_success(context, fault):
    name, data = _fixture(context, True)
    race = data["race_io.json"]

    def change_http(index, side, key, value):
        row = data["http_" + str(index) + ".json"]
        body = json.loads(row[side + "_body_utf8"])
        body[key] = value
        row[side + "_body_utf8"] = json.dumps(body)
        _seal(row)

    if fault.startswith("untrusted_"):
        data["controls_before_reload.json"][0 if fault.endswith("preview") else 1]["trusted"] = False
    elif fault == "release_before_delete":
        race["timing"]["released_at"] = race["timing"]["held"]["held_at"]
    elif fault == "held_wrong_body":
        race["timing"]["held"]["response_body_utf8"] = "{}"
    elif fault == "held_wrong_revision":
        race["timing"]["held"]["request_body_utf8"] = '{"expected_revision":0}'
    elif fault == "response_not_consumed":
        race["timing"]["json_read"] = False
    elif fault in {"changed_map", "changed_hash"}:
        race["final_map"] = {
            **race["final_map"],
            "rendered_map_plan_id" if fault == "changed_map" else "rendered_geometry_hash": "foreign",
        }
    elif fault == "changed_draft":
        data["context_after_reload.json"] = {**data["context_after_reload.json"], "draft_operations": []}
    elif fault == "wrong_parent":
        data["context_after_reload.json"] = {**data["context_after_reload.json"], "accepted_plan_id": "foreign"}
    elif fault == "provider_attempt":
        race["provider_attempts"] = 1
    elif fault == "unexpected_preview":
        data["http_2.json"]["url"] = data["http_2.json"]["url"].replace("/draft/impact-preview", "/preview")
    elif fault == "foreign_delete":
        data["http_3.json"]["url"] = data["http_3.json"]["url"].replace("/unit-owner/", "/foreign/")
    elif fault == "wrong_delete_revision":
        change_http(3, "request", "expected_revision", 0)
    elif fault == "reload_draft_lost":
        body = json.loads(data["http_5.json"]["response_body_utf8"])
        change_http(5, "response", "session", {**body["session"], "draft": []})
    elif fault == "old_reload_transcript":
        data["reload_conversation_http.json"]["response_body_utf8"] = '{"conversation":{"turns":["old"]}}'
        _seal(data["reload_conversation_http.json"])
    elif fault == "incomplete_http":
        data["http_2.json"]["response_capture_complete"] = False
    elif fault == "different_input":
        race["input"] = "Different instruction"
    elif fault == "partial_stage":
        body = json.loads(data["http_1.json"]["response_body_utf8"])
        change_http(1, "response", "session", {**body["session"], "draft": body["session"]["draft"][:1]})
    elif fault == "lost_clause":
        race["turn"]["interpretation"]["intents"].pop()
    with pytest.raises(verifier.core.E2E1VerificationError):
        verifier.queued_preview_control_receipt(data.__getitem__, name=name)
