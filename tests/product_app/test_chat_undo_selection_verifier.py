"""Offline adversarial checks for the real-browser accepted-selection receipt."""

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


def _http(method, suffix, request, response):
    request_body = "" if request is None else json.dumps(request)
    response_body = json.dumps(response)
    return {"phase": "case", "method": method, "url": "http://127.0.0.1:8127/api/sessions/session" + suffix,
        "response_status_code": 200, "request_body_utf8": request_body,
        "request_body_sha256": hashlib.sha256(request_body.encode()).hexdigest(),
        "response_body_utf8": response_body, "response_body_sha256": hashlib.sha256(response_body.encode()).hexdigest()}


def _case(orphan=False, count=1):
    geography = {"plan_id": "parent", "content_hash": "parent-hash", "validated_legs": {"features": []},
        "stops": {"features": [{"properties": {"stop_id": "wharf", "day": 6, "name": "Stearns Wharf", "stop_order": 1}}]}}
    parent = {"active_plan_id": "parent", "active_plan_content_hash": "parent-hash", "rendered_map_plan_id": "parent",
        "rendered_marker_ids": ["wharf"], "rendered_geometry_hash": core._completed_race_geometry_hash(geography),
        "preview_state": "PARENT_RENDERED"}
    before = {"session_id": "session", "accepted_plan_id": "parent", "revision": 8,
        "selected_stop_id": "child-only" if orphan else "wharf", "selected_segment_id": "child-leg", "selected_day": 5,
        "draft": [{"operation_id": str(index), "type": "keep_stop", "target": "wharf"} for index in range(count)],
        "proposal": {"state": "eligible", "child_plan_id": "child", "geography_plan": {"plan_id": "child"}}}
    records = [_http("GET", "", None, {"session": before})]
    current = before
    undos = []
    for _ in range(count):
        after = {**current, "revision": current["revision"] + 1, "draft": current["draft"][:-1], "proposal": None,
            "selected_stop_id": None if orphan else "wharf", "selected_segment_id": None, "selected_day": 5 if orphan else 6}
        wire = _http("POST", "/draft/undo", {"expected_revision": current["revision"]},
            {"undone": current["draft"][-1], "session": after})
        undos.append(wire)
        records.append(wire)
        current = after
    api = [{"method": row["method"], "url": row["url"]} for row in records]
    editor = None if orphan else {"click": {"selector": "#inspect-edit-stop", "trusted": True,
            "clicked_at": "2026-09-08T12:00:00Z"}, "dialog_visible": True, "keep_disabled": False,
        "target_text": "Editing accepted-plan stop: Stearns Wharf. Changes remain drafts until Preview repair finishes.",
        "api_requests_before": api, "api_requests_after": deepcopy(api)}
    receipt = {"schema_version": "chat-undo-selection-v1", "case": "case", "before_session": deepcopy(before),
        "undo_http": deepcopy(undos), "after_session": deepcopy(current), "map_ack": deepcopy(parent),
        "controls": {"stop_ids": [] if orphan else ["wharf"], "segment_ids": [], "days": [5 if orphan else 6]},
        "editor": editor, "assertions_passed": True}
    return [{"case": "case", "parent": parent}, receipt, records, geography]


@pytest.mark.parametrize("orphan", [False, True])
@pytest.mark.parametrize("count", [1, 2])
def test_selection_receipt_preserves_valid_context_and_clears_orphans(tmp_path, orphan, count):
    case = _case(orphan, count)
    verifier.undo_selection_receipt(*case)
    (tmp_path / "exact_verifier_io.json").write_text(json.dumps({"scope": "offline_synthetic_verifier_only",
        "provider_calls": 0, "browser_execution_claimed": False, "input": case, "output": {"verified": True}}), encoding="utf-8")


@pytest.mark.parametrize("mutation", [
    "before_session", "wire_list", "wire_hash", "request_revision", "orphan_result", "clear_valid_stop",
    "wrong_day", "stale_segment", "foreign_accepted", "persisted", "map_id", "map_hash", "geography_id",
    "pressed_stop", "pressed_segment", "pressed_day", "untrusted_click", "wrong_control", "hidden_editor",
    "disabled_keep", "stale_editor_text", "extra_api", "invented_api_inventory", "false_assertions",
])
def test_selection_receipt_rejects_inconsistent_or_fabricated_evidence(tmp_path, mutation):
    case = _case()
    _, receipt, records, geography = case
    expected_code = None
    if mutation == "before_session":
        receipt["before_session"]["selected_stop_id"] = "foreign"
    elif mutation == "wire_list":
        receipt["undo_http"] = []
    elif mutation in {"wire_hash", "request_revision", "orphan_result", "clear_valid_stop", "wrong_day", "stale_segment", "foreign_accepted"}:
        wire = records[-1]
        if mutation == "wire_hash":
            wire["response_body_sha256"] = "f" * 64
        elif mutation == "request_revision":
            wire["request_body_utf8"] = json.dumps({"expected_revision": 0})
            wire["request_body_sha256"] = hashlib.sha256(wire["request_body_utf8"].encode()).hexdigest()
        else:
            payload = json.loads(wire["response_body_utf8"])
            field, value = {"orphan_result": ("selected_stop_id", "child-only"), "clear_valid_stop": ("selected_stop_id", None),
                "wrong_day": ("selected_day", 5), "stale_segment": ("selected_segment_id", "child-leg"),
                "foreign_accepted": ("accepted_plan_id", "foreign")}[mutation]
            payload["session"][field] = value
            wire["response_body_utf8"] = json.dumps(payload)
            wire["response_body_sha256"] = hashlib.sha256(wire["response_body_utf8"].encode()).hexdigest()
            receipt["after_session"] = deepcopy(payload["session"])
            expected_code = "undo_selection_not_projected_from_accepted_plan"
        receipt["undo_http"] = [deepcopy(wire)]
    elif mutation == "persisted":
        receipt["after_session"]["selected_stop_id"] = "foreign"
    elif mutation == "map_id":
        receipt["map_ack"]["rendered_map_plan_id"] = "foreign"
    elif mutation == "map_hash":
        receipt["map_ack"]["rendered_geometry_hash"] = "f" * 64
    elif mutation == "geography_id":
        geography["plan_id"] = "foreign"
    elif mutation.startswith("pressed_"):
        key = {"pressed_stop": "stop_ids", "pressed_segment": "segment_ids", "pressed_day": "days"}[mutation]
        receipt["controls"][key] = ["foreign"]
    elif mutation == "untrusted_click":
        receipt["editor"]["click"]["trusted"] = False
    elif mutation == "wrong_control":
        receipt["editor"]["click"]["selector"] = "#other"
    elif mutation == "hidden_editor":
        receipt["editor"]["dialog_visible"] = False
    elif mutation == "disabled_keep":
        receipt["editor"]["keep_disabled"] = True
    elif mutation == "stale_editor_text":
        receipt["editor"]["target_text"] = "The selected marker belongs to a repair preview"
    elif mutation == "extra_api":
        receipt["editor"]["api_requests_after"].append({"method": "POST", "url": "http://127.0.0.1:8127/api/accept"})
    elif mutation == "invented_api_inventory":
        receipt["editor"]["api_requests_before"] = []
        receipt["editor"]["api_requests_after"] = []
    else:
        receipt["assertions_passed"] = False
    with pytest.raises(core.E2E1VerificationError, match=expected_code or "e2e1_chat_closure_undo_selection_") as caught:
        verifier.undo_selection_receipt(*case)
    (tmp_path / "exact_verifier_io.json").write_text(json.dumps({"scope": "offline_synthetic_verifier_only",
        "provider_calls": 0, "browser_execution_claimed": False, "input": case,
        "output": {"verified": False, "error": str(caught.value)}}), encoding="utf-8")
