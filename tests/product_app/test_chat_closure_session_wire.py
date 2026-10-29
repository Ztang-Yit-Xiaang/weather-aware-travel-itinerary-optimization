"""Full-field browser-observer encoding checks; no product/browser E2E claim."""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from itinerary_system.product_app.models import DraftOperationV1, ProductSession

sys.path.insert(0, str(Path(__file__).parent / "browser"))
from test_chat_runtime_closure_live import _session_wire, _write_json  # noqa: E402


def _session(refs):
    return ProductSession(
        session_id="session-wire-check", mutation_token_salt="not-public-salt",
        mutation_token_verifier="not-public-verifier", trip_id="trip", run_id="run", revision=24,
        accepted_plan_id="parent", selected_day=6, selected_stop_id="stearns_wharf",
        selected_segment_id="directed-leg", selected_candidate_id="candidate",
        selected_alternative_id="alternative", conversation_id="conversation",
        draft=[DraftOperationV1(operation_id="window-operation", type="set_stop_time_window",
            target="stearns_wharf", parameters={"earliest_arrival": "10:00", "latest_departure": None},
            source="copilot", evidence_refs=refs, created_at="2026-09-08T12:00:00+00:00")],
        proposal={"child_plan_id": "child", "child_plan_content_hash": "child-hash", "certificate_id": "evaluation"},
        permission_decisions=[{"permission": "keep_booking", "decision": "allowed"}],
    )


@pytest.mark.parametrize("refs", [(), ("route:directed", "evaluation:independent")])
def test_full_session_encoding_matches_actual_fastapi_response(tmp_path, refs):
    session = _session(refs)
    original = deepcopy(session.public_dict())
    assert type(original["draft"][0]["evidence_refs"]) is tuple
    app = FastAPI()

    @app.get("/session-wire")
    def session_wire():
        # Same ordinary dictionary-return encoding used by product /conversation.
        return {"session": session.public_dict()}

    with TestClient(app) as client:
        response = client.get("/session-wire")
    actual = response.json()["session"]
    encoded = _session_wire(session)
    assert original != actual  # Retained v3 failure is reproducible without IO races.
    assert encoded == actual and set(encoded) == set(original)
    assert encoded["draft"][0]["evidence_refs"] == list(refs)
    assert session.public_dict() == original  # Observer must not mutate disk/model data.
    assert json.dumps({"session": encoded}, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode() == response.content
    assert "mutation_token" not in response.text
    _write_json(tmp_path / "exact_session_wire_io.json", {
        "scope": "unit_real_fastapi_serializer_only", "browser_execution_claimed": False, "provider_calls": 0,
        "model_public_input": original, "model_evidence_refs_python_type": "tuple",
        "actual_response_status": response.status_code, "actual_response_body_utf8": response.text,
        "observer_encoded_output": encoded, "assertions_passed": True,
    })


@pytest.mark.parametrize("field", [
    "session_id", "revision", "accepted_plan_id", "selected_day", "conversation_id", "proposal",
    "operation_id", "target", "parameters", "source", "evidence_refs", "created_at",
])
def test_json_normalization_does_not_hide_public_field_mismatch(tmp_path, field):
    session = _session(("route:directed",))
    actual = _session_wire(session)
    altered = deepcopy(actual)
    target = altered if field in actual else altered["draft"][0]
    if field in {"revision", "selected_day"}:
        target[field] += 1
    elif field == "evidence_refs":
        target[field] = ["route:other"]
    elif field == "parameters":
        target[field]["earliest_arrival"] = "11:00"
    elif field == "proposal":
        target[field]["child_plan_content_hash"] = "other-hash"
    else:
        target[field] = "other"
    assert altered != _session_wire(session)
    _write_json(tmp_path / "exact_session_wire_io.json", {
        "scope": "offline_full_field_mismatch", "provider_calls": 0,
        "input": {"field": field, "expected": actual, "altered": altered},
        "output": {"strict_full_session_equality": False},
    })


def test_raw_held_evidence_is_written_before_decode_and_assertions():
    source = (Path(__file__).parent / "browser/test_chat_runtime_closure_live.py").read_text(encoding="utf-8")
    start = source.index('held = page.evaluate("window.__closureLateConversation")')
    end = source.index('_write_json(case / "evaluation_summary.json"', start)
    block = source[start:end]
    assert block.index('_write_json(case / "held_conversation_response.json", held)') < block.index("json.loads")
    assert block.index('_write_json(case / "late_conversation_undo_race.json", late_conversation_race)') < block.index("assert held")
    assert 'held_payload["session"] == late_conversation_race["before_undo"]["persisted_session"]' in block
    assert "read_session().public_dict()" not in source

