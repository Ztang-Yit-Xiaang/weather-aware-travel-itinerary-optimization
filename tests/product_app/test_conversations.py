from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pytest

import itinerary_system.product_app.conversations as conversation_module
from itinerary_system.product_app.conversations import ConversationError, ConversationRepository
from itinerary_system.product_app.models import CopilotTurnV1, ProductSession


class Clock:
    def __init__(self, value: datetime) -> None:
        self.value = value

    def __call__(self) -> datetime:
        return self.value


def session(*, suffix: str | None = None) -> ProductSession:
    return ProductSession(
        session_id=f"session_{suffix or uuid4().hex}",
        mutation_token_salt="0" * 32,
        mutation_token_verifier="0" * 64,
        trip_id="california-coast",
        run_id="demo-run",
        revision=0,
        accepted_plan_id="plan-original",
    )


def turn(message: str = "Keep café time ☕", *, client_suffix: str | None = None) -> CopilotTurnV1:
    timestamp = datetime(2026, 8, 5, 12, tzinfo=UTC).isoformat()
    return CopilotTurnV1(
        turn_id=f"copilot_turn_{uuid4().hex}",
        client_message_id=f"client_message_{client_suffix or uuid4().hex}",
        context_revision=0,
        provider="deterministic",
        model=None,
        prompt_version="copilot-prompt-v1",
        prompt_sha256="a" * 64,
        state="completed",
        user_message=message,
        assistant_message="I’ll preserve that stop.",
        interpretation=None,
        error_code=None,
        created_at=timestamp,
        completed_at=timestamp,
    )


def assert_code(code: str):
    return pytest.raises(ConversationError, match=f"^{code}$")


def test_round_trip_is_utf8_stable_atomic_and_uses_shared_layout_lock(tmp_path: Path) -> None:
    now = datetime(2026, 8, 5, 9, tzinfo=UTC)
    repository = ConversationRepository(tmp_path / "state", clock=Clock(now))
    owner = session(suffix="1" * 32)

    created = repository.get_or_create(owner)
    saved = repository.append_turn(created.conversation_id, owner.session_id, turn())
    restored = repository.get(created.conversation_id, owner.session_id)

    assert restored == saved
    assert restored.revision == 1
    assert restored.turns[0].user_message == "Keep café time ☕"
    assert restored.expires_at == (now + timedelta(days=30)).isoformat()
    assert repository.lock_path == tmp_path / "state" / "locks" / "layout.lock"
    raw = (repository.root / f"{created.conversation_id}.json").read_bytes()
    assert "café".encode() in raw
    assert raw.endswith(b"\n")
    assert not any(path.name.startswith(".tmp-") for path in repository.root.iterdir())
    assert json.loads(raw)["schema_version"] == "product-conversation-v1"


def test_client_message_id_is_idempotent_only_for_the_same_message(tmp_path: Path) -> None:
    repository = ConversationRepository(tmp_path / "state")
    owner = session()
    conversation = repository.get_or_create(owner)
    client_suffix = uuid4().hex
    first = turn("same", client_suffix=client_suffix)
    after_first = repository.append_turn(conversation.conversation_id, owner.session_id, first)
    duplicate = turn("same", client_suffix=client_suffix)

    after_duplicate = repository.append_turn(conversation.conversation_id, owner.session_id, duplicate)

    assert after_duplicate == after_first
    with assert_code("message_id_conflict"):
        repository.append_turn(
            conversation.conversation_id,
            owner.session_id,
            turn("different", client_suffix=client_suffix),
        )


def test_session_ownership_path_and_schema_fail_closed(tmp_path: Path) -> None:
    state_root = tmp_path / "state"
    repository = ConversationRepository(state_root)
    owner = session()
    other = session()
    conversation = repository.get_or_create(owner)

    with assert_code("conversation_session_mismatch"):
        repository.get(conversation.conversation_id, other.session_id)
    with assert_code("invalid_conversation_id"):
        repository.get("../../decisions", owner.session_id)

    path = repository.root / f"{conversation.conversation_id}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["unexpected"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")
    with assert_code("conversation_invalid"):
        repository.get(conversation.conversation_id, owner.session_id)


def test_initialization_rejects_malformed_and_oversize_recovery_files(tmp_path: Path) -> None:
    state_root = tmp_path / "state"
    conversations = state_root / "conversations"
    conversations.mkdir(parents=True)
    malformed = conversations / f"conversation_{uuid4().hex}.json"
    malformed.write_text("{not-json", encoding="utf-8")

    with assert_code("conversation_store_invalid"):
        ConversationRepository(state_root)

    malformed.write_bytes(b"{" + b"x" * (1024 * 1024) + b"}")
    with assert_code("conversation_store_invalid"):
        ConversationRepository(state_root)


def test_retention_delete_one_and_delete_all(tmp_path: Path) -> None:
    clock = Clock(datetime(2026, 8, 5, tzinfo=UTC))
    repository = ConversationRepository(tmp_path / "state", clock=clock)
    first_owner = session()
    second_owner = session()
    first = repository.get_or_create(first_owner)
    second = repository.get_or_create(second_owner)

    assert repository.delete(first.conversation_id, first_owner.session_id)
    assert not repository.delete(first.conversation_id, first_owner.session_id)
    assert repository.delete_all() == 1
    assert not (repository.root / f"{second.conversation_id}.json").exists()

    expiring = repository.get_or_create(session())
    clock.value += timedelta(days=30, seconds=1)
    assert repository.purge_expired() == 1
    with assert_code("unknown_conversation"):
        repository.get(expiring.conversation_id, expiring.session_id)


def test_capacity_and_turn_limits_never_truncate_active_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(conversation_module, "MAX_FILES", 2)
    monkeypatch.setattr(conversation_module, "MAX_TURNS", 1)
    repository = ConversationRepository(tmp_path / "state")
    first_owner = session()
    first = repository.get_or_create(first_owner)
    repository.get_or_create(session())

    with assert_code("conversation_capacity_reached"):
        repository.get_or_create(session())

    saved = repository.append_turn(first.conversation_id, first_owner.session_id, turn("first"))
    with assert_code("conversation_turn_limit_reached"):
        repository.append_turn(first.conversation_id, first_owner.session_id, turn("second"))
    assert repository.get(first.conversation_id, first_owner.session_id) == saved
