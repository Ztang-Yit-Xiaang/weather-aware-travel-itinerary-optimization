"""Offline counterexamples for the genuine-response Undo browser-race oracle."""

from __future__ import annotations

import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import verify_e2e1_chat_closure as verifier  # noqa: E402
import verify_e2e1_vertical_slice as core  # noqa: E402

BASE = "http://127.0.0.1:8127/api/sessions/session"
PHASE = "chat_window_preview"


def _stamp(second):
    return f"2026-09-08T12:00:{second:02d}.000Z"


def _http(method, suffix, request, response):
    request_raw = "" if request is None else json.dumps(request)
    response_raw = json.dumps(response)
    return {
        "phase": PHASE,
        "method": method,
        "url": BASE + suffix,
        "response_status_code": 200,
        "request_body_utf8": request_raw,
        "request_body_sha256": hashlib.sha256(request_raw.encode()).hexdigest(),
        "response_body_utf8": response_raw,
        "response_body_sha256": hashlib.sha256(response_raw.encode()).hexdigest(),
    }


def _case(context="original"):
    operation = {
        "operation_id": "window-operation",
        "type": "set_stop_time_window",
        "target": "wharf",
        "parameters": {"earliest_arrival": "10:00", "latest_departure": None},
        "source": "copilot",
    }
    geography = {
        "plan_id": "parent",
        "content_hash": "parent-hash",
        "validated_legs": {"features": []},
        "stops": {
            "features": [{"properties": {"stop_id": "wharf", "day": 6, "stop_order": 1, "name": "Stearns Wharf"}}]
        },
    }
    parent = {
        "active_plan_id": "parent",
        "active_plan_content_hash": "parent-hash",
        "rendered_map_plan_id": "parent",
        "rendered_geometry_hash": core._completed_race_geometry_hash(geography),
        "rendered_marker_ids": ["wharf"],
        "preview_state": "PARENT_RENDERED",
    }
    rendered = {**parent, "preview_state": "PREVIEW_INELIGIBLE"}
    if context == "accepted_child":
        rendered.update(
            active_plan_id="window-child",
            rendered_map_plan_id="window-child",
            active_plan_content_hash="child-hash",
            rendered_geometry_hash="child-geometry",
            preview_state="PREVIEW_RENDERED",
        )
    proposal = {
        "child_plan_id": "window-child",
        "diff_id": "window-diff",
        "state": "eligible" if context == "accepted_child" else "ineligible",
    }
    old = {
        "session_id": "session",
        "revision": 24,
        "accepted_plan_id": "parent",
        "draft": [operation],
        "proposal": proposal,
    }
    current = {**old, "revision": 25, "draft": [], "proposal": None}
    payload = {"session": old, "proposal": proposal}
    row = {
        "case": PHASE,
        "context": context,
        "parent": parent,
        "rendered": rendered,
        "operations": [operation],
        "child_plan_id": "window-child",
        "evaluation_id": "window-evaluation",
    }
    conversation = {"conversation": {"turns": []}, "session": old}
    # Identical bytes from preview's restore must not be mistaken for the
    # specific later restore request made by the already required page reload.
    records = [
        _http("POST", "/preview", {"expected_revision": 23}, payload),
        _http("GET", "/conversation", None, conversation),
        _http("GET", "", None, payload),
        _http("GET", "/conversation", None, conversation),
        _http("POST", "/draft/undo", {"expected_revision": 24}, {"undone": operation, "session": current}),
        _http("POST", "/draft/impact-preview", {"expected_revision": 25}, {"session_revision": 25}),
    ]
    held = {
        "url": BASE + "/conversation",
        "method": "GET",
        "request_body_utf8": "",
        "status": 200,
        "body_utf8": records[3]["response_body_utf8"],
        "body_sha256": records[3]["response_body_sha256"],
        "requested_at": _stamp(1),
        "body_completed_at": _stamp(2),
        "held_at": _stamp(3),
        "released_at": _stamp(7),
        "application_json_read_at": _stamp(8),
        "delivery_settled_at": _stamp(9),
        "release_count": 1,
        "error": None,
    }
    cards = [
        {
            "day": day,
            "stop_id": "wharf" if day == 6 else "",
            "name": "Stearns Wharf" if day == 6 else "No recorded stop",
        }
        for day in range(1, 8)
    ]
    ui = {
        "map_ack": parent,
        "draft_label": "0 draft changes",
        "preview_disabled": True,
        "accept_disabled": True,
        "evaluated_draft_evidence_count": 0,
        "timeline_cards": cards,
        "surfaces": {"metrics": {"planId": "parent", "planContentHash": "parent-hash"}},
        "text_route_scope": {"planId": "parent", "planContentHash": "parent-hash"},
        "repair_panel": "No evaluated draft is displayed.",
    }
    requests = [{key: http[key] for key in ("method", "url")} for http in records]
    race = {
        "schema_version": "real-conversation-undo-race-v1",
        "context": context,
        "case": PHASE,
        "scope": "actual_response_delivery_delay_only",
        "assertions_passed": True,
        "api_response_mocked": False,
        "response_body_modified": False,
        "original_response_object_returned": True,
        "response_json_observation_only": True,
        "additional_session_reload_count": 0,
        "parent": deepcopy(parent),
        "operation": operation,
        "held_response": held,
        "before_undo": {"map_ack": rendered, "persisted_session": old},
        "undo_http": records[4],
        "undo_clicked_at": _stamp(4),
        "undo_response_received_at": _stamp(5),
        "persisted_before_release": current,
        "persisted_after_rerender": deepcopy(current),
        "rerender_control": {"selector": "#map-select-button", "trusted": True, "clicked_at": _stamp(11)},
        "raw_after_rerender": {
            "draft_label": "0 draft changes",
            "map": {"activePlanId": "parent", "activePlanContentHash": "parent-hash"},
            "metrics": {"planId": "parent", "planContentHash": "parent-hash"},
        },
        "api_requests_before_release": requests,
        "api_requests_after_rerender": deepcopy(requests),
    }
    for label, second in (
        ("after_undo_before_release", 6),
        ("after_release_before_rerender", 10),
        ("after_rerender", 12),
    ):
        race[label] = {**deepcopy(ui), "observed_at": _stamp(second)}
    ready = {
        "screenshot_started_at": _stamp(13),
        "rendered_plan_id": "parent",
        "rendered_geometry_hash": parent["rendered_geometry_hash"],
    }
    return [row, payload, race, records, geography, ready]


@pytest.mark.parametrize("context", ["original", "accepted_child"])
def test_late_real_response_cannot_restore_an_undone_draft_in_either_context(tmp_path, context):
    case = _case(context)
    verifier.late_conversation_undo_race(*case)
    (tmp_path / "exact_verifier_io.json").write_text(
        json.dumps(
            {
                "scope": "offline_synthetic_verifier_fixture",
                "browser_execution_claimed": False,
                "provider_calls": 0,
                "input": case,
                "output": {"verified": True},
            },
            indent=2,
        ),
        encoding="utf-8",
    )


@pytest.mark.parametrize(
    "fault,code",
    [
        ("mocked_body", "late_read_scope_invalid"),
        ("extra_reload", "late_read_scope_invalid"),
        ("missing_reload", "late_read_reload_count_invalid"),
        ("duplicate_held", "late_read_exact_http_unbound"),
        ("changed_bytes", "late_read_exact_http_unbound"),
        ("changed_hash", "late_read_exact_http_unbound"),
        ("response_never_released", "late_read_exact_http_unbound"),
        ("read_before_undo", "late_read_delivery_or_capture_order_invalid"),
        ("undo_before_body", "late_read_delivery_or_capture_order_invalid"),
        ("no_delivery_settled", "e2e1_isolation_timestamp_invalid"),
        ("synthetic_click", "late_read_rerender_not_real_control"),
        ("missing_native_rerender", "late_read_rerender_not_real_control"),
        ("wrong_session", "late_read_revision_or_persistence_mismatch"),
        ("same_revision", "late_read_revision_or_persistence_mismatch"),
        ("wrong_undo_operation", "late_read_revision_or_persistence_mismatch"),
        ("persisted_rollback", "late_read_revision_or_persistence_mismatch"),
        ("draft_resurrected", "late_read_visible_state_rolled_back"),
        ("preview_resurrected", "late_read_visible_state_rolled_back"),
        ("map_resurrected", "late_read_visible_state_rolled_back"),
        ("metrics_resurrected", "late_read_visible_state_rolled_back"),
        ("text_route_resurrected", "late_read_visible_state_rolled_back"),
        ("timeline_resurrected", "late_read_visible_state_rolled_back"),
        ("evidence_resurrected", "late_read_visible_state_rolled_back"),
        ("evidence_old_id", "late_read_visible_state_rolled_back"),
        ("raw_state_disagrees", "late_read_raw_rerender_rolled_back"),
        ("new_api_call", "late_read_extra_api_work"),
        ("missing_request_inventory", "late_read_extra_api_work"),
        ("screenshot_before_release", "late_read_delivery_or_capture_order_invalid"),
        ("wrong_screenshot_plan", "late_read_delivery_or_capture_order_invalid"),
        ("wrong_parent_geometry", "late_read_parent_geography_unbound"),
    ],
)
def test_race_counterexamples_cannot_pass_with_only_an_assertions_boolean(tmp_path, fault, code):
    case = _case()
    verifier.late_conversation_undo_race(*case)
    _, _, race, records, geography, ready = case
    held = race["held_response"]
    if fault == "mocked_body":
        race["response_body_modified"] = True
    elif fault == "extra_reload":
        race["additional_session_reload_count"] = 1
    elif fault == "missing_reload":
        records.pop(2)
    elif fault == "duplicate_held":
        records.insert(4, deepcopy(records[3]))
    elif fault == "changed_bytes":
        held["body_utf8"] += " "
    elif fault == "changed_hash":
        held["body_sha256"] = "unbound"
    elif fault == "response_never_released":
        held["release_count"] = 0
    elif fault == "read_before_undo":
        held["released_at"] = _stamp(4)
    elif fault == "undo_before_body":
        held["body_completed_at"] = _stamp(6)
    elif fault == "no_delivery_settled":
        held["delivery_settled_at"] = None
    elif fault == "synthetic_click":
        race["rerender_control"]["trusted"] = False
    elif fault == "missing_native_rerender":
        race["rerender_control"]["selector"] = "window.renderAll"
    elif fault in {"wrong_session", "same_revision", "wrong_undo_operation"}:
        undo = json.loads(race["undo_http"]["response_body_utf8"])
        if fault == "wrong_session":
            undo["session"]["session_id"] = "other"
        elif fault == "same_revision":
            undo["session"]["revision"] = 24
        else:
            undo["undone"]["operation_id"] = "other"
        raw = json.dumps(undo)
        race["undo_http"]["response_body_utf8"] = raw
        race["undo_http"]["response_body_sha256"] = hashlib.sha256(raw.encode()).hexdigest()
    elif fault == "persisted_rollback":
        race["persisted_after_rerender"]["draft"] = [race["operation"]]
    elif fault == "draft_resurrected":
        race["after_rerender"]["draft_label"] = "1 draft change"
    elif fault == "preview_resurrected":
        race["after_rerender"]["preview_disabled"] = False
    elif fault == "map_resurrected":
        race["after_rerender"]["map_ack"]["active_plan_id"] = "window-child"
    elif fault == "metrics_resurrected":
        race["after_rerender"]["surfaces"]["metrics"]["planId"] = "window-child"
    elif fault == "text_route_resurrected":
        race["after_rerender"]["text_route_scope"]["planContentHash"] = "wrong"
    elif fault == "timeline_resurrected":
        race["after_rerender"]["timeline_cards"][5]["name"] = "Other wharf"
    elif fault == "evidence_resurrected":
        race["after_rerender"]["evaluated_draft_evidence_count"] = 1
    elif fault == "evidence_old_id":
        race["after_rerender"]["repair_panel"] = "Evaluated window-child"
    elif fault == "raw_state_disagrees":
        race["raw_after_rerender"]["draft_label"] = "1 draft change"
    elif fault == "new_api_call":
        race["api_requests_after_rerender"].append({"method": "POST", "url": BASE + "/preview"})
    elif fault == "missing_request_inventory":
        race["api_requests_before_release"] = race["api_requests_after_rerender"] = []
    elif fault == "screenshot_before_release":
        ready["screenshot_started_at"] = _stamp(6)
    elif fault == "wrong_screenshot_plan":
        ready["rendered_plan_id"] = "window-child"
    elif fault == "wrong_parent_geometry":
        geography["content_hash"] = "unrelated"
    with pytest.raises(core.E2E1VerificationError, match=code) as caught:
        verifier.late_conversation_undo_race(*case)
    (tmp_path / "exact_verifier_io.json").write_text(
        json.dumps(
            {
                "scope": "offline_synthetic_verifier_counterexample",
                "browser_execution_claimed": False,
                "provider_calls": 0,
                "input": {"fault": fault, "case": case},
                "output": {"rejected": str(caught.value)},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
