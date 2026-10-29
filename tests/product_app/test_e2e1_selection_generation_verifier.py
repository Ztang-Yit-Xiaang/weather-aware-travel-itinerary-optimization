"""Zero-provider counterexamples for the independently checked delivery-order gate."""

from __future__ import annotations

import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import verify_e2e1_selection_generation as verifier  # noqa: E402
import verify_e2e1_vertical_slice as core  # noqa: E402


def _fixture(tmp_path, monkeypatch):
    # Synthetic unit evidence only; actual browser proof is a separate run.
    monkeypatch.setattr(core, "ROOT", tmp_path)
    (tmp_path / "fixture.js").write_text("synthetic unit source", encoding="utf-8")
    sources = {"fixture.js": hashlib.sha256(b"synthetic unit source").hexdigest()}
    session = dict(
        session_id="session_unit",
        trip_id="trip_unit",
        run_id="run_unit",
        revision=1,
        conversation_id="conversation_old",
        accepted_plan_id="plan_parent",
        draft=[],
        proposal=None,
        selected_day=2,
        selected_stop_id="stanford_university_main_quad",
        selected_segment_id=None,
        selected_candidate_id=None,
        selected_alternative_id=None,
    )
    current = {**session, "conversation_id": "conversation_new"}
    base = "http://127.0.0.1:8127/api/sessions/session_unit"
    turn = dict(
        turn_id="turn_unit",
        client_message_id="message_unit",
        user_message="What preferences do you remember?",
        assistant_message="none",
        interpretation={"state": "clarification_required"},
        context_revision=1,
        state="completed",
        provider="deterministic",
        model=None,
    )
    conversation = {key: current[key] for key in ("conversation_id", "session_id", "trip_id", "run_id")}
    before = {"status": 200, "body": {"session": current, "conversation": {**conversation, "turns": [turn]}}}
    renewed = {"status": 200, "body": {"session": current, "conversation": {**conversation, "turns": []}}}
    records = []

    def stamp(second):
        return f"2026-09-08T12:00:{second:02d}+00:00"

    def call(method, suffix, response, at, request=None):
        row = dict(
            method=method,
            url=base + suffix,
            request_body_utf8=json.dumps(request) if request else "",
            response_body_utf8=json.dumps(response),
            response_status=200,
            requested_at=stamp(at),
            dispatch_at=stamp(at),
            body_completed_at=stamp(at),
            application_json_read_at=stamp(at),
            held_at=None,
            released_at=None,
            release_count=0,
            delivery_settled_at=stamp(at),
        )
        records.append(
            {
                key: value
                for key, value in row.items()
                if key in ("method", "url", "request_body_utf8", "response_body_utf8")
            }
            | {"response_status_code": 200}
        )
        return row

    first = call("GET", "/conversation", {"session": session}, 4)
    first.update(requested_at=stamp(0), held_at=stamp(0), released_at=stamp(3), release_count=1, dispatch_at=stamp(3))
    request = {
        key: session[key]
        for key in (
            "selected_day",
            "selected_stop_id",
            "selected_segment_id",
            "selected_candidate_id",
            "selected_alternative_id",
        )
    }
    request["expected_revision"] = 0
    selection = call("POST", "/selection", {"session": session}, 1, request)
    selection.update(
        body_completed_at=stamp(2),
        held_at=stamp(2),
        released_at=stamp(8),
        release_count=1,
        application_json_read_at=stamp(8),
        delivery_settled_at=stamp(8),
    )
    held = {**selection, "released_at": None, "release_count": 0, "application_json_read_at": None}
    deleted = call(
        "DELETE",
        "/conversation",
        {"deleted": True, "session": {**session, "conversation_id": None}},
        5,
        {"expected_revision": 1},
    )
    calls = [
        first,
        selection,
        deleted,
        call("GET", "/conversation", renewed["body"], 6),
        call("GET", "/conversation", before["body"], 7),
        call("GET", "", {"session": current, "workspace": {}, "registry": {}}, 9),
        call("GET", "/conversation", before["body"], 10),
    ]
    clicks = [
        dict(trusted=True, day="2" if identity == "day" else None, id=identity, at=stamp(at))
        for identity, at in (
            ("day", 1),
            ("open-copilot", 4),
            ("transcript-settings-button", 4),
            ("delete-current-conversation", 5),
            ("copilot-submit", 7),
        )
    ]
    final = dict(
        timing={"calls": calls, "clicks": clicks},
        user_messages=[turn["user_message"]],
        selected_day="2",
        lifecycle="Conversation restored",
        observed_at=stamp(11),
    )
    ack = dict(
        active_plan_id="plan_parent",
        active_plan_content_hash="plan_hash",
        rendered_map_plan_id="plan_parent",
        rendered_geometry_hash="geometry_hash",
        rendered_marker_ids=["marker"],
        preview_state="PARENT_RENDERED",
    )
    for revision in (0, 1):
        call(
            "POST",
            "/draft/impact-preview",
            dict(
                schema_version="product-draft-impact-preview-v1",
                session_id=session["session_id"],
                session_revision=revision,
                parent_plan_id="plan_parent",
                parent_plan_content_hash="plan_hash",
                certification_status="not_certified",
                is_certified=False,
                operations=[],
            ),
            11,
            {"expected_revision": revision},
        )
    trace = dict(initial=ack, final=ack)
    files = {
        "checkpoints.json": dict(
            selection_held={"timing": {"calls": [held]}},
            new_conversation_restored={"observed_at": stamp(6)},
            new_conversation_message_visible={**final, "observed_at": stamp(7)},
            late_selection_applied=final,
            final_observation=final,
        ),
        "runtime_trace.json": dict(
            before_release=before,
            after_release=before,
            visible_after_release=final,
            original_conversation_id=session["conversation_id"],
            new_conversation_id=current["conversation_id"],
            session_revision=1,
            message=turn["user_message"],
            map_initial=ack,
            map_final=ack,
            accepted_plan_id="plan_parent",
        ),
        "persisted_before_release.json": before,
        "persisted_after_release.json": before,
        "persisted_after_delete.json": renewed,
        "map_initial.json": ack,
        "map_final.json": ack,
        "source_identity_before.json": sources,
        "source_identity_after.json": sources,
    }
    turns = [
        {
            "input": {"message": turn["user_message"], "expected_revision": 1, "client_message_id": "message_unit"},
            "output": {"type": "turn.completed", "payload": {"session": current, "turn": turn}},
        }
    ]
    return files, trace, records, turns


@pytest.mark.parametrize(
    "fault",
    [
        None,
        "visible_lost",
        "wrong_day",
        "same_conversation",
        "wrong_revision",
        "old_reconciliation",
        "missing_reconciliation",
        "unbound_timing",
        "late_response_not_held",
        "delete_before_reply",
        "duplicate_release",
        "untrusted_click",
        "wrong_turn_owner",
        "wrong_map",
        "changed_source",
        "missing_source",
        "unexpected_accept",
        "missing_impact",
        "impact_wrong_revision",
        "impact_certified",
        "impact_has_draft",
        "impact_wrong_parent",
    ],
)
def test_independent_generation_gate_rejects_false_green(tmp_path, monkeypatch, fault):
    files, trace, records, turns = _fixture(tmp_path, monkeypatch)
    final = files["checkpoints.json"]["late_selection_applied"]
    selection = final["timing"]["calls"][1]
    if fault == "visible_lost":
        final["user_messages"] = []
    elif fault == "wrong_day":
        final["selected_day"] = "1"
    elif fault == "same_conversation":
        files["persisted_after_release.json"]["body"]["session"]["conversation_id"] = "conversation_old"
    elif fault == "wrong_revision":
        files["runtime_trace.json"]["session_revision"] = 2
    elif fault == "old_reconciliation":
        row = final["timing"]["calls"][5]
        value = json.loads(row["response_body_utf8"])
        value["session"]["conversation_id"] = "conversation_old"
        row["response_body_utf8"] = records[5]["response_body_utf8"] = json.dumps(value)
    elif fault == "missing_reconciliation":
        final["timing"]["calls"].pop(5)
        records.pop(5)
    elif fault == "unbound_timing":
        records.pop(5)
    elif fault == "late_response_not_held":
        files["checkpoints.json"]["selection_held"]["timing"]["calls"][0]["application_json_read_at"] = selection[
            "body_completed_at"
        ]
    elif fault == "delete_before_reply":
        final["timing"]["calls"][2]["dispatch_at"] = "2026-09-08T12:00:01+00:00"
    elif fault == "duplicate_release":
        selection["release_count"] = 2
    elif fault == "untrusted_click":
        final["timing"]["clicks"][0]["trusted"] = False
    elif fault == "wrong_turn_owner":
        turns[0] = deepcopy(turns[0])
        turns[0]["output"]["payload"]["session"]["conversation_id"] = "conversation_old"
    elif fault == "wrong_map":
        trace["final"] = {**trace["final"], "active_plan_id": "wrong_plan"}
    elif fault == "changed_source":
        files["source_identity_after.json"] = {"fixture.js": "0" * 64}
    elif fault == "missing_source":
        (tmp_path / "fixture.js").unlink()
    elif fault == "unexpected_accept":
        records.append({"method": "POST", "url": "http://127.0.0.1:8127/api/sessions/session_unit/accept"})
    elif fault == "missing_impact":
        records.pop()
    elif fault and fault.startswith("impact_"):
        row = records[-1]
        value = json.loads(row["response_body_utf8"])
        field, replacement = {
            "impact_wrong_revision": ("session_revision", 0),
            "impact_certified": ("is_certified", True),
            "impact_has_draft": ("operations", [{"type": "replace_nearby"}]),
            "impact_wrong_parent": ("parent_plan_id", "child_other"),
        }[fault]
        value[field] = replacement
        row["response_body_utf8"] = json.dumps(value)
    for name, value in files.items():
        (tmp_path / name).write_text(json.dumps(value), encoding="utf-8")
    if fault:
        with pytest.raises(core.E2E1VerificationError):
            verifier.verify_selection_generation(tmp_path, trace, records, turns)
    else:
        assert verifier.verify_selection_generation(tmp_path, trace, records, turns) == (
            "before",
            "before_late_selection",
            "after_late_selection",
        )
