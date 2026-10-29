from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "run_live_copilot_smoke.py"


def _load_smoke():
    spec = importlib.util.spec_from_file_location("run_live_copilot_smoke", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _session() -> dict:
    return {
        "session_id": "session_private",
        "trip_id": "trip_demo",
        "run_id": "run_demo",
        "revision": 0,
        "accepted_plan_id": "plan_parent",
        "selected_day": None,
        "selected_stop_id": None,
        "selected_segment_id": None,
        "selected_candidate_id": None,
        "selected_alternative_id": None,
        "draft": [],
        "proposal": None,
        "permission_decisions": [],
        "conversation_id": None,
    }


def _turn() -> dict:
    return {
        "state": "completed",
        "provider": "openai",
        "model": "gpt-5.6-terra",
        "prompt_version": "copilot-prompt-v1",
        "prompt_sha256": "1" * 64,
        "user_message": "must not persist",
        "assistant_message": "must not persist",
        "error_code": None,
        "interpretation": {
            "schema_version": "copilot-interpretation-v1",
            "state": "proposal_ready",
            "assistant_message": "must not persist",
            "intents": [
                {
                    "type": "review_registered_repair",
                    "target_stop_id": None,
                    "destination_day": None,
                    "candidate_id": None,
                    "preference": None,
                    "clarification_question": None,
                }
            ],
            "permission": None,
            "highlights": {
                "day_ids": [2],
                "stop_ids": [],
                "segment_ids": [],
                "candidate_ids": [],
            },
        },
    }


def test_smoke_evidence_has_explicit_protected_scope_and_no_raw_copy() -> None:
    smoke = _load_smoke()
    session = _session()
    evidence = smoke.build_smoke_evidence(
        started_at="2026-08-12T00:00:00+00:00",
        completed_at="2026-08-12T00:00:01+00:00",
        health={"status": "ready", "core_ready": True},
        session_id=session["session_id"],
        client_message_id="client_message_" + "a" * 32,
        before_session=session,
        after_session=dict(session),
        turn=_turn(),
    )

    assert evidence["protected_state_scope"] == list(smoke.PROTECTED_STATE_FIELDS)
    assert evidence["protected_state_unchanged"] is True
    assert evidence["protected_state_before_sha256"] == evidence["protected_state_after_sha256"]
    assert evidence["raw_request_content_persisted"] is False
    assert evidence["raw_provider_content_persisted"] is False
    assert "assistant_message" not in evidence["turn"]
    assert "user_message" not in evidence["turn"]
    unsigned = dict(evidence)
    content_hash = unsigned.pop("content_hash")
    assert content_hash == smoke._canonical_sha256(unsigned)


def test_smoke_evidence_detects_protected_permission_mutation() -> None:
    smoke = _load_smoke()
    before = _session()
    after = _session()
    after["permission_decisions"] = [{"permission": "external_booking_change"}]

    evidence = smoke.build_smoke_evidence(
        started_at="2026-08-12T00:00:00+00:00",
        completed_at="2026-08-12T00:00:01+00:00",
        health={"status": "ready", "core_ready": True},
        session_id=before["session_id"],
        client_message_id="client_message_" + "b" * 32,
        before_session=before,
        after_session=after,
        turn=_turn(),
    )

    assert evidence["protected_state_unchanged"] is False
    assert evidence["protected_state_before_sha256"] != evidence["protected_state_after_sha256"]
