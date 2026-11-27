"""Focused tamper checks for the unregistered deterministic chat-family gate."""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import verify_e2e1_chat_closure as verifier  # noqa: E402
import verify_e2e1_vertical_slice as core  # noqa: E402


def _turn_fixture():
    request = {"message": "Keep Pfeiffer Big Sur State Park.", "client_message_id": "message", "expected_revision": 7}
    context = {
        "accepted_plan_id": "child",
        "accepted_plan_content_hash": "hash",
        "allowed_stop_ids": ["pfeiffer"],
        "session_revision": 7,
        "conversation_memory": {"plan_id": "child", "plan_content_hash": "hash"},
        "selected_day": 5,
        "selected_stop_id": "pfeiffer",
        "selected_candidate_id": None,
        "selected_segment_id": None,
        "selected_alternative_id": None,
    }
    turn = {"provider": "deterministic", "user_message": request["message"], "context_revision": 7}
    output = {"turn": turn, "session": {"accepted_plan_id": "child", "draft": []}}
    binding = {
        "plan_id": "child",
        "plan_content_hash": "hash",
        **{
            name: context[name]
            for name in (
                "selected_day",
                "selected_stop_id",
                "selected_candidate_id",
                "selected_segment_id",
                "selected_alternative_id",
            )
        },
    }
    digest = verifier.context_digest(context, [])
    events = [{"payload": output}]
    row = {
        "input": request,
        "events": events,
        "output": output,
        "context": context,
        "bound_history": [],
        "context_sha256": digest,
        "dispatch_record": {
            "context_binding": binding,
            "context_sha256": digest,
            "client_message_id": "message",
            "attempt_budget": 1,
            "attempt_count": 1,
            "outcome": "completed",
            "turn": turn,
        },
    }
    identities = {("child", "hash"): {"rendered_marker_ids": ["pfeiffer"]}}
    return row, request, events, identities


def test_exact_context_and_durable_dispatch_bound_to_website_turn(monkeypatch):
    row, request, events, identities = _turn_fixture()
    monkeypatch.setattr(verifier, "_stream", lambda http: (request, events))
    verifier.chat_turn(row, {}, identities)


@pytest.mark.parametrize(
    "fault",
    [
        "stale_stop",
        "wrong_plan",
        "wrong_memory_plan",
        "wrong_history",
        "wrong_binding",
        "wrong_message_id",
        "second_attempt",
        "unknown_as_completed",
        "changed_turn",
        "send_staged_draft",
    ],
)
def test_conversation_counterexamples_reject_context_or_dispatch_disconnection(monkeypatch, fault):
    row, request, events, identities = _turn_fixture()
    monkeypatch.setattr(verifier, "_stream", lambda http: (request, events))
    if fault == "stale_stop":
        row["context"]["allowed_stop_ids"] = ["removed_bixby"]
    elif fault == "wrong_plan":
        row["context"]["accepted_plan_id"] = "old_parent"
    elif fault == "wrong_memory_plan":
        row["context"]["conversation_memory"]["plan_id"] = "old_parent"
    elif fault == "wrong_history":
        row["bound_history"] = [{"wrong": "older context"}]
    elif fault == "wrong_binding":
        row["dispatch_record"]["context_binding"]["selected_day"] = 6
    elif fault == "wrong_message_id":
        row["dispatch_record"]["client_message_id"] = "other"
    elif fault == "second_attempt":
        row["dispatch_record"]["attempt_count"] = row["dispatch_record"]["attempt_budget"] = 2
    elif fault == "unknown_as_completed":
        row["dispatch_record"]["outcome"] = "unknown"
    elif fault == "changed_turn":
        row["dispatch_record"]["turn"] = deepcopy(row["output"]["turn"])
        row["dispatch_record"]["turn"]["user_message"] = "changed"
    else:
        row["output"]["session"]["draft"] = [{"type": "keep_stop"}]
    with pytest.raises(core.E2E1VerificationError):
        verifier.chat_turn(row, {}, identities)


@pytest.mark.parametrize(
    "intent,expected",
    [
        ({"type": "keep_stop", "target_stop_id": "golden_gate"}, ("keep_stop", "golden_gate", {})),
        (
            {"type": "replace_nearby", "target_stop_id": "bixby", "candidate_id": "pfeiffer"},
            ("replace_nearby", "bixby", {"candidate_id": "pfeiffer"}),
        ),
        (
            {"type": "add_candidate", "candidate_id": "lobos", "destination_day": 5},
            ("add_candidate", "lobos", {"day": 5}),
        ),
        (
            {"type": "set_stop_role", "target_stop_id": "wharf", "parameters": {"role": "meal"}},
            ("set_stop_role", "wharf", {"role": "meal"}),
        ),
    ],
)
def test_chat_intent_maps_to_same_operation_without_semantic_substitution(intent, expected):
    assert verifier.interpretation_operation(intent) == expected


def test_missing_chat_closure_evidence_is_not_a_pass_or_raw_key_error(tmp_path):
    with pytest.raises(core.E2E1VerificationError, match="e2e1_chat_closure_evidence_malformed"):
        verifier.verify(tmp_path, "http://127.0.0.1:8127")


@pytest.mark.parametrize("label", verifier.NUMERIC_CLARIFICATIONS)
@pytest.mark.parametrize(
    "fault",
    [
        None,
        "prompt",
        "proposal",
        "empty_question",
        "draft",
        "preview",
        "accepted_plan",
        "before_hash",
        "after_geometry",
        "after_markers",
    ],
)
def test_numeric_chat_negative_requires_clarification_and_unchanged_canonical_parent(label, fault):
    parent = {
        "active_plan_id": "accepted",
        "active_plan_content_hash": "parent_hash",
        "rendered_map_plan_id": "accepted",
        "rendered_geometry_hash": "geometry",
        "rendered_marker_ids": ["wharf"],
    }
    row = {
        "case": label,
        "input": verifier.NUMERIC_CLARIFICATIONS[label],
        "map_before": deepcopy(parent),
        "map": deepcopy(parent),
        "output": {
            "turn": {
                "interpretation": {
                    "state": "clarification_required",
                    "intents": [
                        {"type": "request_clarification", "clarification_question": "Which single exact duration?"}
                    ],
                }
            },
            "session": {"draft": [], "proposal": None, "accepted_plan_id": "accepted"},
        },
    }
    if fault == "prompt":
        row["input"] = "Spend 90 minutes at Stearns Wharf."
    elif fault == "proposal":
        row["output"]["turn"]["interpretation"]["state"] = "proposal_ready"
    elif fault == "empty_question":
        row["output"]["turn"]["interpretation"]["intents"][0]["clarification_question"] = ""
    elif fault == "draft":
        row["output"]["session"]["draft"] = [{"type": "set_stop_duration"}]
    elif fault == "preview":
        row["output"]["session"]["proposal"] = {"state": "eligible"}
    elif fault == "accepted_plan":
        row["output"]["session"]["accepted_plan_id"] = "other"
    elif fault == "before_hash":
        row["map_before"]["active_plan_content_hash"] = "other"
    elif fault == "after_geometry":
        row["map"]["rendered_geometry_hash"] = "other"
    elif fault == "after_markers":
        row["map"]["rendered_marker_ids"] = ["other"]
    if fault is None:
        verifier.numeric_clarification(row, parent)
    else:
        with pytest.raises(core.E2E1VerificationError):
            verifier.numeric_clarification(row, parent)


def _lifecycle():
    operation = {"operation_id": "kept", "type": "keep_stop"}
    session = {
        "session_id": "session",
        "accepted_plan_id": "parent",
        "draft": [operation],
        "proposal": {"child": "candidate"},
    }
    row = {
        "case": "keep_only_original",
        "operations": [operation],
        "scope": "evaluated_preview_only",
        "assertions_passed": True,
    }
    records = [
        {
            "phase": row["case"],
            "method": "GET",
            "url": "/api/sessions/session",
            "response_body_utf8": json.dumps({"session": session}),
        },
        {
            "phase": row["case"],
            "method": "POST",
            "url": "/draft/undo",
            "response_body_utf8": json.dumps(
                {"undone": operation, "session": {**session, "draft": [], "proposal": None}}
            ),
        },
    ]
    return row, {"session": session, "proposal": session["proposal"]}, records


def test_reload_and_undo_bind_actual_http():
    verifier.journey_lifecycle_http(*_lifecycle())


@pytest.mark.parametrize(
    "fault",
    [
        "missing_reload",
        "missing_undo",
        "wrong_reloaded_plan",
        "wrong_undo",
        "unexpected_accept",
        "false_acceptance_scope",
    ],
)
def test_lifecycle_cannot_be_claimed_without_exact_reload_and_undo(fault):
    row, payload, records = _lifecycle()
    if fault == "missing_reload":
        records.pop(0)
    elif fault == "missing_undo":
        records.pop()
    elif fault == "wrong_reloaded_plan":
        body = json.loads(records[0]["response_body_utf8"])
        body["session"]["accepted_plan_id"] = "wrong"
        records[0]["response_body_utf8"] = json.dumps(body)
    elif fault == "wrong_undo":
        body = json.loads(records[-1]["response_body_utf8"])
        body["undone"]["operation_id"] = "other"
        records[-1]["response_body_utf8"] = json.dumps(body)
    elif fault == "unexpected_accept":
        records.append({"phase": row["case"], "method": "POST", "url": "/accept"})
    else:
        row["scope"] = "preview_then_explicit_acceptance_and_reload"
    with pytest.raises(core.E2E1VerificationError):
        verifier.journey_lifecycle_http(row, payload, records)
