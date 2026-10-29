"""Focused runtime-review counterexamples; no live provider or browser work."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
from openai.lib._pydantic import to_strict_json_schema

from itinerary_system.product_app.conversations import ConversationError, ConversationRepository
from itinerary_system.product_app.copilot import DeterministicCopilotAdapter
from itinerary_system.product_app.copilot_memory import empty_memory, project_capabilities
from itinerary_system.product_app.interaction_contracts import typed_edit_capabilities
from itinerary_system.product_app.models import CopilotContextV2, CopilotIntentV1, CopilotTurnV1, ProductSession
from itinerary_system.product_app.openai_copilot import OpenAICopilotAdapter


def _record(name: str, **record) -> None:
    directory = os.environ.get("COPILOT_BOUNDARY_EVIDENCE_DIR")
    if directory:
        output = Path(directory)
        output.mkdir(parents=True, exist_ok=True)
        (output / f"{name}.json").write_text(
            json.dumps(
                {
                    "provider_calls": 0,
                    "mode": "synthetic_unit_actual_production_methods",
                    **record,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )


def _context() -> CopilotContextV2:
    return CopilotContextV2(
        run_id="review_run",
        trip_id="review_trip",
        session_id="review_session",
        session_revision=0,
        accepted_plan_id="review_parent",
        accepted_plan_content_hash="a" * 64,
        selected_day=6,
        selected_stop_id="stearns_wharf",
        selected_segment_id=None,
        selected_candidate_id=None,
        selected_alternative_id=None,
        draft_operations=(),
        evaluated_proposal=None,
        allowed_stop_ids=("stearns_wharf",),
        allowed_candidate_ids=(),
        allowed_days=(1, 2, 3, 4, 5, 6, 7),
        known_places=(
            {
                "place_id": "stearns_wharf",
                "name": "Stearns Wharf",
                "aliases": (),
                "categories": ("itinerary_stop",),
                "region_id": None,
                "route_status": "accepted_stop",
            },
        ),
        capabilities=project_capabilities(
            {"operations": {}, "candidate_choices": ()},
            registered_repair_current=False,
            typed_capabilities=typed_edit_capabilities(),
            stop_positions=({"stop_id": "stearns_wharf", "day": 6, "sequence_index": 0},),
        ),
        conversation_memory=empty_memory("review_parent", "a" * 64),
    )


def test_resolved_stop_highlight_includes_its_canonical_accepted_day() -> None:
    output = DeterministicCopilotAdapter._operation_highlights(
        _context(),
        CopilotIntentV1(type="keep_stop", target_stop_id="stearns_wharf"),
    )

    assert output.stop_ids == ("stearns_wharf",)
    assert output.day_ids == (6,)


class _FakeClient:
    """Instrument the supplied SDK client boundary, never create an HTTP client."""

    def __init__(self, minutes: int):
        self.responses = self
        self.calls = []
        self.output = {
            "schema_version": "copilot-interpretation-v2",
            "state": "proposal_ready",
            "assistant_message": "Synthetic provider proposes this exact quantity.",
            "intents": [
                {
                    "type": "set_stop_duration",
                    "target_stop_id": "stearns_wharf",
                    "parameters": {
                        "duration": {
                            "mode": "exact",
                            "preferred_minutes": minutes,
                            "minimum_minutes": minutes,
                            "maximum_minutes": minutes,
                        },
                    },
                }
            ],
        }

    async def parse(self, **kwargs):
        schema = kwargs["text_format"]
        self.calls.append(
            {
                **kwargs,
                "text_format": {
                    "python_type": schema.__name__,
                    "strict_json_schema": to_strict_json_schema(schema),
                },
            }
        )
        return SimpleNamespace(output=[], output_parsed=schema(**self.output))


@pytest.mark.parametrize("provider", ["deterministic", "openai_fake_client"])
@pytest.mark.parametrize(
    ("label", "message", "unsafe_minutes", "exact"),
    [
        ("single", "Spend 90 minutes at Stearns Wharf.", 90, True),
        ("single_with_day", "Spend 90 minutes at Stearns Wharf on Day 6.", 90, True),
        ("wrong_stop_day", "Spend 90 minutes at Stearns Wharf on Day 5.", 90, False),
        ("compound", "Spend 1 hour 30 minutes at Stearns Wharf.", 60, False),
        ("alternative", "Set Stearns Wharf duration to 60 or 90 minutes.", 90, False),
        ("decimal", "Spend 1.5 hours at Stearns Wharf.", 300, False),
        ("negative", "Spend -30 minutes at Stearns Wharf.", 30, False),
        ("range", "Set Stearns Wharf duration to 60-90 minutes.", 90, False),
        ("compound_conjunction", "Spend 1 hour and 30 minutes at Stearns Wharf.", 60, False),
    ],
)
def test_duration_requires_one_complete_exact_quantity(provider, label, message, unsafe_minutes, exact):
    context = _context()
    fake = _FakeClient(unsafe_minutes)
    adapter = (
        DeterministicCopilotAdapter()
        if provider == "deterministic"
        else OpenAICopilotAdapter(
            model="fake-model",
            api_key="unused",
            client=fake,
        )
    )
    output = asyncio.run(adapter.interpret(context=context, history=(), message=message))
    _record(
        f"duration_{provider}_{label}",
        input={"message": message, "context": context.as_dict(), "history": []},
        output=output.as_dict(),
        fake_sdk_requests=fake.calls,
        fake_sdk_response=fake.output if fake.calls else None,
        fake_sdk_attempts=len(fake.calls),
    )
    if exact:
        assert output.state == "proposal_ready"
        assert len(output.intents) == 1
        intent = output.intents[0]
        assert intent.type == "set_stop_duration" and intent.target_stop_id == "stearns_wharf"
        assert intent.parameters == {
            "duration": {
                "mode": "exact",
                "preferred_minutes": 90,
                "minimum_minutes": 90,
                "maximum_minutes": 90,
            }
        }
        assert len(fake.calls) == (1 if provider == "openai_fake_client" else 0)
    else:
        assert output.state == "clarification_required", output.as_dict()
        assert [intent.type for intent in output.intents] == ["request_clarification"]
        assert output.intents[0].clarification_question
        assert fake.calls == [], "An ambiguous numeric expression must be blocked before any provider attempt."
    assert context.draft_operations == () and context.evaluated_proposal is None


def _file_record(path: Path) -> dict:
    raw = path.read_bytes()
    return {"name": path.name, "body_utf8": raw.decode("utf-8"), "sha256": hashlib.sha256(raw).hexdigest()}


def test_expired_append_removes_paired_journal_before_empty_transcript_recreation(tmp_path):
    now = [datetime(2026, 8, 1, tzinfo=UTC)]
    monotonic = [0.0]
    repository = ConversationRepository(tmp_path / "state", clock=lambda: now[0], monotonic=lambda: monotonic[0])
    owner = ProductSession(
        session_id="session_" + "1" * 32,
        mutation_token_salt="0" * 32,
        mutation_token_verifier="0" * 64,
        trip_id="review_trip",
        run_id="review_run",
        revision=0,
        accepted_plan_id="review_parent",
    )
    conversation = repository.get_or_create(owner)
    now[0] += timedelta(days=30, milliseconds=-1)
    monotonic[0] += 30 * 24 * 60 * 60 - 0.001
    # A real pre-expiry read runs periodic cleanup. Expiry then occurs inside
    # its throttle interval, between durable result publication and append.
    repository.get(conversation.conversation_id, owner.session_id)
    request = {
        "client_message_id": "client_message_" + "2" * 32,
        "message": "Synthetic retention boundary review",
        "context_sha256": "b" * 64,
        "context_revision": 0,
        "provider": "deterministic",
        "model": None,
        "prompt_sha256": "a" * 64,
        "prompt_version": "synthetic-review",
    }
    repository.reserve_dispatch(conversation.conversation_id, owner.session_id, **request)
    turn = CopilotTurnV1(
        turn_id="copilot_turn_" + "3" * 32,
        client_message_id=request["client_message_id"],
        context_revision=0,
        provider="deterministic",
        model=None,
        prompt_version=request["prompt_version"],
        prompt_sha256=request["prompt_sha256"],
        state="completed",
        user_message=request["message"],
        assistant_message="Synthetic completed response",
        interpretation=None,
        error_code=None,
        created_at=now[0].isoformat(),
        completed_at=now[0].isoformat(),
    )
    repository.finish_dispatch(
        conversation.conversation_id,
        owner.session_id,
        request["client_message_id"],
        request["message"],
        turn=turn,
        outcome="completed",
    )
    transcript_path = repository._path(conversation.conversation_id)
    journal_path = repository._dispatch_path(conversation.conversation_id)
    before = [_file_record(transcript_path), _file_record(journal_path)]
    now[0] += timedelta(milliseconds=2)
    monotonic[0] += 0.002
    with pytest.raises(ConversationError, match="^unknown_conversation$"):
        repository.append_turn(conversation.conversation_id, owner.session_id, turn)
    after_expiry = {"transcript_exists": transcript_path.exists(), "journal_exists": journal_path.exists()}
    fresh = repository.get_or_create(owner)
    records = repository.dispatch_history(fresh.conversation_id, owner.session_id)
    recovered = repository.dispatch_record(
        fresh.conversation_id, owner.session_id, request["client_message_id"], request["message"]
    )
    _record(
        "expired_append",
        input={"reservation": request, "completed_turn": turn.as_dict(), "files": before},
        after_expiry=after_expiry,
        output={"fresh_transcript": fresh.as_dict(), "dispatch_history": records, "recovered": recovered},
        output_files=[_file_record(path) for path in (transcript_path, journal_path) if path.exists()],
    )
    assert after_expiry == {"transcript_exists": False, "journal_exists": False}
    assert fresh.turns == () and records == () and recovered is None
    assert repository.delete_all() == 1
    assert not journal_path.exists()
