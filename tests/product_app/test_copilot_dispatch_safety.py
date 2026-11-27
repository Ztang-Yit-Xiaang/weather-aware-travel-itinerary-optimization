from __future__ import annotations

import asyncio
import copy
import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from uuid import uuid4

import pytest

import itinerary_system.product_app.conversations as conversation_module
from itinerary_system.product_app.conversations import ConversationError
from itinerary_system.product_app.copilot import DeterministicCopilotAdapter
from itinerary_system.product_app.registry import ProductRunRegistry
from itinerary_system.product_app.service import CopilotServiceError, ProductService
from itinerary_system.product_app.workspace import WorkspaceError

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class _Runtime:
    service: ProductService
    session: object
    token: str = field(repr=False)

    def __iter__(self):
        return iter((self.service, self.session, self.token))


@pytest.fixture
def runtime(tmp_path: Path):
    registry = ProductRunRegistry(ROOT, ROOT / "configs/product_app_registry.json")
    service = ProductService(registry, tmp_path / "state")
    session, token, _ = service.create_session(registry.default.run_id)
    return _Runtime(service, session, token)


class CountingAdapter:
    provider_name = "openai"
    model = "fake-no-network"

    def __init__(self, *, cancel: bool = False):
        self.calls = 0
        self.cancel = cancel

    async def interpret(self, **kwargs):
        self.calls += 1
        if self.cancel:
            raise asyncio.CancelledError
        return await DeterministicCopilotAdapter().interpret(**kwargs)


def send(
    service, session, token, client_message_id=None, *, message="Review the current itinerary without changing it."
):
    return service.interpret_copilot_message(
        session_id=session.session_id,
        mutation_token=token,
        expected_revision=session.revision,
        client_message_id=client_message_id or f"client_message_{uuid4().hex}",
        message=message,
    )


def test_full_conversation_never_dispatches_provider(runtime) -> None:
    service, session, token = runtime
    first = asyncio.run(send(service, session, token))["turn"]
    conversation = service.read_conversation(session)
    template = service.conversations._validate_turn(first)
    turns = tuple(
        replace(
            template,
            turn_id=f"copilot_turn_{uuid4().hex}",
            client_message_id=f"client_message_{uuid4().hex}",
        )
        for _ in range(conversation_module.MAX_TURNS)
    )
    service.conversations._save(replace(conversation, turns=turns, revision=len(turns)))
    adapter = CountingAdapter()
    service.copilot = adapter
    with pytest.raises(ConversationError, match="conversation_turn_limit_reached"):
        asyncio.run(send(service, session, token))
    assert adapter.calls == 0


def test_cancelled_message_cannot_dispatch_again_after_restart(runtime) -> None:
    service, session, token = runtime
    client_id = f"client_message_{uuid4().hex}"
    cancelled = CountingAdapter(cancel=True)
    service.copilot = cancelled
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(send(service, session, token, client_id))
    uncertain = service.read_conversation(session).turns
    assert len(uncertain) == 1
    assert uncertain[0].error_code == "copilot_outcome_unknown"
    replacement = ProductService(service.registry, service.conversations.root.parent)
    replacement.copilot = CountingAdapter()
    with pytest.raises(CopilotServiceError, match="copilot_outcome_unknown"):
        asyncio.run(send(replacement, session, token, client_id))
    assert cancelled.calls == 1
    assert replacement.copilot.calls == 0


def test_near_one_mib_conversation_is_rejected_before_dispatch(runtime) -> None:
    service, session, token = runtime
    first = asyncio.run(send(service, session, token))["turn"]
    conversation = service.read_conversation(session)
    template = replace(
        service.conversations._validate_turn(first),
        user_message="u" * 2000,
        assistant_message="a" * 4000,
        interpretation=None,
    )
    turns = []
    while (
        service.conversations._encoded_size(
            replace(
                conversation,
                turns=tuple(turns),
                revision=len(turns),
            ).as_dict()
        )
        < conversation_module.MAX_FILE_BYTES - 64 * 1024
    ):
        turns.append(
            replace(
                template,
                turn_id=f"copilot_turn_{uuid4().hex}",
                client_message_id=f"client_message_{uuid4().hex}",
            )
        )
    assert len(turns) < conversation_module.MAX_TURNS
    full = replace(conversation, turns=tuple(turns), revision=len(turns))
    assert service.conversations._encoded_size(full.as_dict()) < conversation_module.MAX_FILE_BYTES
    service.conversations._save(full)
    service.copilot = CountingAdapter()
    with pytest.raises(ConversationError, match="conversation_size_limit_reached"):
        asyncio.run(send(service, session, token))
    assert service.copilot.calls == 0


def test_completed_result_survives_transcript_write_failure_and_restart(runtime, monkeypatch) -> None:
    service, session, token = runtime
    service.copilot = CountingAdapter()
    client_id = f"client_message_{uuid4().hex}"

    def fail_append(*args, **kwargs):
        raise ConversationError("conversation_write_failed")

    monkeypatch.setattr(service.conversations, "append_turn", fail_append)
    with pytest.raises(ConversationError, match="conversation_write_failed"):
        asyncio.run(send(service, session, token, client_id))
    assert service.copilot.calls == 1
    conversation = service.read_conversation(session)
    assert conversation.turns == ()
    record = service.conversations.dispatch_record(
        conversation.conversation_id,
        session.session_id,
        client_id,
        "Review the current itinerary without changing it.",
    )
    assert record["outcome"] == "completed" and record["turn"] is not None
    restored = ProductService(service.registry, service.conversations.root.parent)
    restored.copilot = CountingAdapter()
    recovered = asyncio.run(send(restored, session, token, client_id))
    assert recovered["turn"]["turn_id"] == record["turn"]["turn_id"]
    assert recovered["conversation_revision"] == 1
    assert restored.copilot.calls == 0


def test_completed_result_is_preserved_but_not_reused_after_context_revision_changes(runtime) -> None:
    service, session, token = runtime
    client_id = f"client_message_{uuid4().hex}"

    class RevisionAdapter(CountingAdapter):
        async def interpret(self, **kwargs):
            result = await super().interpret(**kwargs)
            current = service.workspace.get(session.session_id)
            service.workspace.select(current, {"expected_revision": current.revision, "selected_day": 2})
            return result

    service.copilot = RevisionAdapter()
    with pytest.raises(WorkspaceError, match="stale_session_revision"):
        asyncio.run(send(service, session, token, client_id))
    current = service.workspace.get(session.session_id)
    conversation = service.read_conversation(current)
    assert len(conversation.turns) == 1
    assert conversation.turns[0].context_revision == 0
    with pytest.raises(CopilotServiceError, match="copilot_result_context_stale"):
        asyncio.run(send(service, current, token, client_id))
    assert service.copilot.calls == 1


def test_lost_result_write_retains_pending_budget_and_blocks_restart_retry(runtime, monkeypatch) -> None:
    service, session, token = runtime
    service.copilot = CountingAdapter()
    client_id = f"client_message_{uuid4().hex}"

    def fail_finish(*args, **kwargs):
        raise ConversationError("copilot_dispatch_result_write_failed")

    monkeypatch.setattr(service.conversations, "finish_dispatch", fail_finish)
    with pytest.raises(ConversationError, match="copilot_dispatch_result_write_failed"):
        asyncio.run(send(service, session, token, client_id))
    restored = ProductService(service.registry, service.conversations.root.parent)
    restored.copilot = CountingAdapter()
    with pytest.raises(CopilotServiceError, match="copilot_outcome_unknown"):
        asyncio.run(send(restored, session, token, client_id))
    assert service.copilot.calls == 1 and restored.copilot.calls == 0


def test_reservation_write_failure_never_dispatches(runtime, monkeypatch) -> None:
    service, session, token = runtime
    service.copilot = CountingAdapter()

    def fail_save(*args, **kwargs):
        raise OSError("injected-disk-failure")

    monkeypatch.setattr(service.conversations, "_save_dispatch_ledger", fail_save)
    with pytest.raises(ConversationError, match="copilot_dispatch_reservation_failed"):
        asyncio.run(send(service, session, token))
    assert service.copilot.calls == 0


def test_two_service_instances_cannot_dispatch_same_pending_identity(runtime) -> None:
    service, session, token = runtime
    other = ProductService(service.registry, service.conversations.root.parent)
    other.copilot = CountingAdapter()
    client_id = f"client_message_{uuid4().hex}"

    async def scenario():
        entered, release = asyncio.Event(), asyncio.Event()

        class WaitingAdapter(CountingAdapter):
            async def interpret(self, **kwargs):
                entered.set()
                await release.wait()
                return await super().interpret(**kwargs)

        service.copilot = WaitingAdapter()
        first = asyncio.create_task(send(service, session, token, client_id))
        await entered.wait()
        with pytest.raises(CopilotServiceError, match="copilot_outcome_unknown"):
            await send(other, session, token, client_id)
        release.set()
        result = await first
        assert result["turn"]["state"] == "completed"

    asyncio.run(scenario())
    assert service.copilot.calls == 1 and other.copilot.calls == 0


def test_actual_adapter_and_service_share_one_attempt_across_uncertain_retry(runtime) -> None:
    import httpx
    from openai import APIConnectionError

    from itinerary_system.product_app.openai_copilot import OpenAICopilotAdapter

    service, session, token = runtime

    class FakeResponses:
        def __init__(self):
            self.calls = []

        async def parse(self, **kwargs):
            self.calls.append(kwargs)
            raise APIConnectionError(request=httpx.Request("POST", "https://api.openai.com/v1/responses"))

    class FakeClient:
        responses = FakeResponses()

    client = FakeClient()
    service.copilot = OpenAICopilotAdapter(model="fake-model", api_key="unused", client=client)
    client_id = f"client_message_{uuid4().hex}"
    message = "Which parts of the current itinerary would benefit most from a closer review?"
    with pytest.raises(CopilotServiceError, match="openai_provider_unavailable"):
        asyncio.run(send(service, session, token, client_id, message=message))
    replay = asyncio.run(send(service, session, token, client_id, message=message))
    assert replay["turn"]["state"] == "failed"
    assert replay["turn"]["error_code"] == "openai_provider_unavailable"
    assert len(client.responses.calls) == 1
    conversation = service.read_conversation(session)
    journal = service.conversations.dispatch_record(
        conversation.conversation_id,
        session.session_id,
        client_id,
        message,
    )
    assert journal["outcome"] == "unknown"
    assert journal["attempt_budget"] == journal["attempt_count"] == 1
    assert journal["turn"]["turn_id"] == replay["turn"]["turn_id"]


def test_dispatch_ledger_rejects_tampered_budget_input_and_outcome(runtime) -> None:
    service, session, token = runtime
    asyncio.run(send(service, session, token))
    conversation = service.read_conversation(session)
    path = service.conversations._dispatch_path(conversation.conversation_id)
    original = json.loads(path.read_text(encoding="utf-8"))
    for field_name, value in (
        ("attempt_budget", 2),
        ("attempt_budget", True),
        ("attempt_count", 0),
        ("attempt_count", "1"),
        ("message_sha256", "0" * 64),
        ("user_message", "different input"),
        ("context_sha256", "invalid"),
        ("context_revision", True),
        ("outcome", "pending"),
        ("provider", "other"),
        ("prompt_version", "different-prompt"),
    ):
        forged = copy.deepcopy(original)
        forged["records"][0][field_name] = value
        with pytest.raises(ConversationError, match="copilot_dispatch_store_invalid"):
            service.conversations._validate_dispatch_ledger(forged)


def test_deleting_conversation_removes_its_dispatch_inputs_and_outcomes(runtime) -> None:
    service, session, token = runtime
    asyncio.run(send(service, session, token))
    conversation = service.read_conversation(session)
    journal_path = service.conversations._dispatch_path(conversation.conversation_id)
    assert journal_path.is_file()
    service.delete_conversation(session, session.revision)
    assert not journal_path.exists()
