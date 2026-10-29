from __future__ import annotations

import asyncio
import json
import os
from copy import deepcopy
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

from itinerary_system.product_app.copilot import (
    CopilotContractError,
    DeterministicCopilotAdapter,
    validate_copilot_interpretation,
)
from itinerary_system.product_app.copilot_memory import bound_records
from itinerary_system.product_app.models import CopilotInterpretationV1, ProductSession
from itinerary_system.product_app.openai_copilot import OpenAICopilotAdapter
from itinerary_system.product_app.persistence import LocalStateLayout
from itinerary_system.product_app.registry import ProductRunRegistry
from itinerary_system.product_app.service import ProductService

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Runtime:
    service: ProductService
    session: ProductSession
    token: str = field(repr=False)
    transcript: list = field(default_factory=list)

    def send(self, message, message_id=None):
        context = self.service.copilot_context(self.session).as_dict()
        result = asyncio.run(self.service.interpret_copilot_message(
            session_id=self.session.session_id, mutation_token=self.token, expected_revision=self.session.revision,
            client_message_id=message_id or "client_message_" + uuid4().hex, message=message,
        ))["turn"]
        self.transcript.append({"input": message, "context": context, "output": result})
        return result


class FakeClient:
    def __init__(self, interpretation=None):
        self.responses = self
        self.calls = []
        self.output = interpretation or {
            "schema_version": "copilot-interpretation-v1", "state": "clarification_required",
            "assistant_message": "Untrusted provider wording.",
            "intents": [{"type": "request_clarification", "clarification_question": "Which registered stop should I review?"}],
        }

    async def parse(self, **kwargs):
        self.calls.append(kwargs)
        schema = kwargs["text_format"]
        output = {**self.output, "schema_version": "copilot-interpretation-v2" if schema.__name__ == "OpenAIInterpretationSchemaV2" else "copilot-interpretation-v1"}
        return SimpleNamespace(output=[], output_parsed=schema(**output))


def evidence(name, runtime, **extra):
    directory = os.environ.get("COPILOT_MEMORY_EVIDENCE_DIR")
    if directory:
        destination = Path(directory)
        destination.mkdir(parents=True, exist_ok=True)
        destination.joinpath(name + ".json").write_text(json.dumps({
            "provider_calls": 0, "mode": "deterministic_or_fake_transport", "turns": runtime.transcript,
            "final_context": runtime.service.copilot_context(runtime.session).as_dict(), **extra,
        }, indent=2), encoding="utf-8")


@pytest.fixture
def runtime(tmp_path):
    registry = ProductRunRegistry(ROOT, ROOT / "configs/product_app_registry.json")
    service = ProductService(registry, tmp_path / "state", copilot=DeterministicCopilotAdapter())
    session, token, _ = service.create_session(registry.default.run_id)

    return Runtime(service, session, token)


def test_context_v2_contains_current_capabilities_and_empty_plan_bound_memory(runtime):
    service, session = runtime.service, runtime.session
    context = service.copilot_context(session).as_dict()
    assert context["schema_version"] == "copilot-context-v2"
    assert context["accepted_plan_content_hash"]
    assert context["capabilities"]["operations"]["add_candidate"]["preview_executable"] is False
    assert context["conversation_memory"]["plan_id"] == session.accepted_plan_id
    assert context["conversation_memory"]["pending_clarification"] is None


@pytest.mark.parametrize("message", [
    "I prefer nature and less driving.",
    "I prefer less driving and museums.",
])
def test_mixed_driving_preferences_require_a_complete_enforceable_request(runtime, message):
    response = runtime.send(message)
    assert response["interpretation"]["state"] == "clarification_required"
    copy = response["assistant_message"].lower()
    assert "maximum daily driving time" in copy and "complete request" in copy
    assert runtime.session.draft == [] and runtime.session.proposal is None
    memory = runtime.service.copilot_context(runtime.session).conversation_memory
    assert [row["preference"] for row in memory["proposed_preferences"]] == (["reduce_driving"] if "driving" in message else [])


@pytest.mark.parametrize("provider", ["deterministic", "openai"])
@pytest.mark.parametrize("message", ["Keep the original plan", "Please retain the current itinerary unchanged."])
def test_whole_plan_preservation_is_not_a_stop_edit_or_a_provider_call(runtime, provider, message):
    runtime.send("Replace Bixby Creek Bridge.")
    before = deepcopy(runtime.session.public_dict())
    client = FakeClient()
    if provider == "openai":
        runtime.service.copilot = OpenAICopilotAdapter(model="fake-model", api_key="unused", client=client)
    turn = runtime.send(message)
    evidence("whole_plan_" + provider + "_" + str(len(message)), runtime,
             provider_calls=client.calls, before_session=before, after_session=runtime.session.public_dict())
    assert turn["interpretation"]["state"] == "proposal_ready"
    assert [row["type"] for row in turn["interpretation"]["intents"]] == ["keep_original"]
    assert runtime.session.public_dict() == before
    assert runtime.service.copilot_context(runtime.session).conversation_memory["pending_clarification"] is None
    assert client.calls == []


@pytest.mark.parametrize("provider", ["deterministic", "openai"])
@pytest.mark.parametrize("message", [
    "Do not keep the original plan.",
    "Keep the original plan and replace Bixby Creek Bridge with Pfeiffer Big Sur State Park.",
    "Keep the original plan; email it to me.",
])
def test_whole_plan_shortcut_does_not_discard_negation_or_other_clauses(runtime, provider, message):
    client = FakeClient()
    if provider == "openai":
        runtime.service.copilot = OpenAICopilotAdapter(model="fake-model", api_key="unused", client=client)
    turn = runtime.send(message)
    assert turn["interpretation"]["state"] in {"clarification_required", "unsupported"}
    assert not any(row["type"] == "keep_original" for row in turn["interpretation"]["intents"])
    assert runtime.session.draft == []
    assert client.calls == []


def test_supported_replacement_clarifies_one_slot_and_binds_short_answer(runtime):
    service, session, send = runtime.service, runtime.session, runtime.send
    first = send("Replace Bixby Creek Bridge.")
    assert first["interpretation"]["state"] == "clarification_required"
    pending = service.copilot_context(session).conversation_memory["pending_clarification"]
    assert pending["intent_type"] == "replace_nearby"
    assert pending["target_stop_id"] == "bixby_creek_bridge_viewpoint"
    assert pending["missing_slots"] == ("candidate_id",)
    second = send("Pfeiffer Big Sur State Park.")
    assert second["interpretation"]["state"] == "proposal_ready"
    intent = second["interpretation"]["intents"][0]
    assert intent["type"] == "replace_nearby"
    assert intent["target_stop_id"] == "bixby_creek_bridge_viewpoint"
    assert intent["candidate_id"] == "pfeiffer_big_sur_state_park"
    assert service.copilot_context(session).conversation_memory["pending_clarification"] is None
    assert session.draft == []
    evidence("replacement_slot_answer", runtime)


def test_disabled_move_explains_capability_before_asking_for_a_day(runtime):
    service, session, send = runtime.service, runtime.session, runtime.send
    turn = send("Move Golden Gate Bridge.")
    assert turn["interpretation"]["state"] == "unsupported"
    assert "optimizer" in turn["assistant_message"]
    assert "Which day" not in turn["assistant_message"]
    assert service.copilot_context(session).conversation_memory["pending_clarification"] is None


def test_explicit_preference_survives_beyond_six_turns_as_proposed_only(runtime):
    service, session, send = runtime.service, runtime.session, runtime.send
    first = send("I prefer less driving.")
    for _ in range(7):
        send("Review the safer repair.")
    memory = service.copilot_context(session).conversation_memory
    assert memory["proposed_preferences"] == ({
        "preference": "reduce_driving", "source_turn_id": first["turn_id"], "disposition": "proposed_only",
    },)
    assert session.draft == []
    response = send("What preferences do you remember?")
    assert "reduce driving" in response["assistant_message"]
    assert "not accepted" in response["assistant_message"]
    evidence("long_preference", runtime)


def test_pending_target_correction_and_reset_use_existing_conversation(runtime):
    service, session, send = runtime.service, runtime.session, runtime.send
    send("Replace Bixby Creek Bridge.")
    corrected = send("Actually, Golden Gate Bridge.")
    assert corrected["interpretation"]["state"] == "unsupported"
    assert "No registered replacement" in corrected["assistant_message"]
    pending = service.copilot_context(session).conversation_memory["pending_clarification"]
    assert pending is None
    service.delete_conversation(session, session.revision)
    memory = service.copilot_context(session).conversation_memory
    assert memory["pending_clarification"] is None
    assert memory["proposed_preferences"] == ()
    evidence("correction_reset", runtime)


def test_preference_correction_replaces_proposed_memory_not_constraints(runtime):
    runtime.send("I prefer less driving.")
    corrected = runtime.send("Actually, I prefer preserving the experience instead.")
    assert "not supported" not in corrected["assistant_message"]
    assert "preserve experience" in corrected["assistant_message"]
    memory = runtime.service.copilot_context(runtime.session).conversation_memory
    assert memory["proposed_preferences"] == ({
        "preference": "preserve_experience", "source_turn_id": corrected["turn_id"], "disposition": "proposed_only",
    },)
    assert runtime.session.draft == []


def test_same_for_other_stop_requests_the_missing_target(runtime):
    runtime.send("Replace Bixby Creek Bridge.")
    runtime.send("Pfeiffer Big Sur State Park.")
    response = runtime.send("Do the same for the other stop.")
    assert response["interpretation"]["state"] == "clarification_required"
    assert "Which accepted stop" in response["assistant_message"]
    pending = runtime.service.copilot_context(runtime.session).conversation_memory["pending_clarification"]
    assert pending["target_stop_id"] is None
    assert pending["candidate_id"] == "pfeiffer_big_sur_state_park"


def test_pending_clarification_survives_service_restart(runtime, tmp_path):
    runtime.send("Replace Bixby Creek Bridge.")
    runtime.service = ProductService(runtime.service.registry, tmp_path / "state", copilot=DeterministicCopilotAdapter())
    runtime.session = runtime.service.workspace.get(runtime.session.session_id)
    response = runtime.send("Pfeiffer Big Sur State Park.")
    assert response["interpretation"]["intents"][0]["target_stop_id"] == "bixby_creek_bridge_viewpoint"
    evidence("pending_restart", runtime)


def test_old_unbound_journal_is_recoverable_but_never_seeds_memory(runtime, monkeypatch):
    original = runtime.send("I prefer less driving.")
    repository = runtime.service.conversations
    conversation = runtime.service.read_conversation(runtime.session)
    path = repository._dispatch_path(conversation.conversation_id)
    ledger = repository._files.read(path)
    ledger["schema_version"] = "copilot-dispatch-ledger-v1"
    ledger.pop("context_bindings")
    repository._save_dispatch_ledger(ledger)
    assert runtime.service.copilot_context(runtime.session).conversation_memory["proposed_preferences"] == ()

    async def forbidden(**kwargs):
        raise AssertionError("recovery must not dispatch")

    with monkeypatch.context() as patch:
        patch.setattr(runtime.service.copilot, "interpret", forbidden)
        replay = runtime.send("I prefer less driving.", original["client_message_id"])
    assert replay == original
    runtime.send("Review the safer repair.")
    upgraded = repository._files.read(path)
    assert upgraded["schema_version"] == "copilot-dispatch-ledger-v2"
    assert original["client_message_id"] not in upgraded["context_bindings"]


def test_fake_provider_receives_v2_memory_and_exact_capability_context(runtime):
    runtime.send("I prefer less driving.")
    client = FakeClient()
    runtime.service.copilot = OpenAICopilotAdapter(model="fake-model", api_key="unused", client=client)
    response = runtime.send("Can you explain what you can help me with?")
    assert len(client.calls) == 1
    request = client.calls[0]
    context = json.loads(request["input"].split("<workspace_context>\n")[1].split("\n</workspace_context>")[0])
    assert context["schema_version"] == "copilot-context-v2"
    assert context["conversation_memory"]["proposed_preferences"][0]["preference"] == "reduce_driving"
    assert context["capabilities"]["operations"]["add_candidate"]["preview_executable"] is False
    assert response["interpretation"]["state"] == "clarification_required"
    from openai.lib._pydantic import to_strict_json_schema
    evidence("fake_provider_v2", runtime, fake_request={**request, "text_format": {
        "python_type": request["text_format"].__name__, "strict_json_schema": to_strict_json_schema(request["text_format"]),
    }}, fake_provider_output=client.output)


def test_fake_provider_cannot_override_disabled_capability(runtime):
    client = FakeClient({
        "schema_version": "copilot-interpretation-v1", "state": "proposal_ready", "assistant_message": "Do this edit.",
        "intents": [{"type": "move_day", "target_stop_id": "golden_gate_bridge", "destination_day": 3}],
    })
    runtime.service.copilot = OpenAICopilotAdapter(model="fake-model", api_key="unused", client=client)
    response = runtime.send("Help me decide.")
    assert len(client.calls) == 1
    assert response["interpretation"]["state"] == "unsupported"
    assert runtime.session.draft == []


def test_local_journal_file_lock_retry_never_repeats_interpretation(runtime, monkeypatch):
    from itinerary_system.product_app import persistence
    original_replace = persistence.os.replace
    writes = []
    calls = []
    original_interpret = runtime.service.copilot.interpret

    async def counted(**kwargs):
        calls.append(kwargs["message"])
        return await original_interpret(**kwargs)

    def transient(source, destination):
        if Path(destination).name.startswith("dispatch_"):
            writes.append(str(destination))
            if len(writes) == 2:
                error = PermissionError("injected sharing violation")
                error.winerror = 32
                raise error
        return original_replace(source, destination)

    monkeypatch.setattr(runtime.service.copilot, "interpret", counted)
    monkeypatch.setattr(persistence.os, "replace", transient)
    monkeypatch.setattr(LocalStateLayout, "_fsync_directory", staticmethod(lambda path: None))
    response = runtime.send("Review the safer repair.")
    assert response["state"] == "completed"
    assert len(writes) == 3 and calls == ["Review the safer repair."]


@pytest.mark.parametrize("fault", ["plan_hash", "extra_key", "capability_bool", "duplicate_day", "accepted_memory", "unknown_id", "oversized_memory"])
def test_v2_context_extensions_fail_closed(runtime, fault):
    context = runtime.service.copilot_context(runtime.session)
    memory = deepcopy(context.conversation_memory)
    caps = deepcopy(context.capabilities)
    if fault == "plan_hash":
        memory["plan_content_hash"] = "0" * 16
    elif fault == "extra_key":
        memory["private_path"] = "forbidden"
    elif fault == "capability_bool":
        caps["operations"]["add_candidate"]["enabled"] = 1
    elif fault == "duplicate_day":
        caps["candidate_day_readiness"] += (caps["candidate_day_readiness"][0],)
    elif fault == "accepted_memory":
        memory["proposed_preferences"] = ({"preference": "reduce_driving", "source_turn_id": "copilot_turn_" + "0" * 32, "disposition": "accepted"},)
    elif fault == "unknown_id":
        memory["resolved_ids"]["stop_ids"] = ("unknown_stop",)
    else:
        caps["operations"]["add_candidate"]["blocking_code"] = "x" * 14001
    with pytest.raises(CopilotContractError):
        validate_copilot_interpretation(context=replace(context, capabilities=caps, conversation_memory=memory),
                                        interpretation=CopilotInterpretationV1(state="failed", assistant_message="Context check."))


def test_pending_question_and_preview_disposition_cannot_be_forged(runtime):
    runtime.send("Replace Bixby Creek Bridge.")
    context = runtime.service.copilot_context(runtime.session)
    memory = deepcopy(context.conversation_memory)
    memory["pending_clarification"]["question"] = "Which destination day should I use?"
    with pytest.raises(CopilotContractError):
        validate_copilot_interpretation(context=replace(context, conversation_memory=memory),
                                        interpretation=CopilotInterpretationV1(state="failed", assistant_message="Context check."))
    runtime.send("Pfeiffer Big Sur State Park.")
    context = runtime.service.copilot_context(runtime.session)
    memory = deepcopy(context.conversation_memory)
    memory["last_proposal"]["disposition"] = "previewed"
    with pytest.raises(CopilotContractError):
        validate_copilot_interpretation(context=replace(context, conversation_memory=memory),
                                        interpretation=CopilotInterpretationV1(state="failed", assistant_message="Context check."))


def test_staging_preview_and_acceptance_have_distinct_memory_dispositions(runtime):
    service, session = runtime.service, runtime.session
    runtime.send("I prefer less driving.")
    runtime.send("Replace Bixby Creek Bridge.")
    runtime.send("Pfeiffer Big Sur State Park.")
    before = service.copilot_context(session)
    assert before.conversation_memory["last_proposal"]["disposition"] == "suggested"
    service.workspace.add_operation(session, {"expected_revision": session.revision, "type": "replace_nearby",
        "target": "bixby_creek_bridge_viewpoint", "parameters": {"candidate_id": "pfeiffer_big_sur_state_park"}, "source": "copilot"},
        valid_stop_ids=service.valid_stops(session.run_id), day_count=7)
    assert service.copilot_context(session).conversation_memory["last_proposal"]["disposition"] == "staged"
    proposal = service.preview(session.session_id, session.revision)
    assert proposal["state"] == "eligible"
    assert service.copilot_context(session).conversation_memory["last_proposal"]["disposition"] == "previewed"
    service.accept_evaluated(session, expected_revision=session.revision, confirmation="accept_evaluated_repair",
                            idempotency_key="r14-memory-transition", mutation_token=runtime.token)
    current = service.copilot_context(session)
    assert current.accepted_plan_id != before.accepted_plan_id
    assert current.conversation_memory["last_proposal"] is None
    assert current.conversation_memory["pending_clarification"] is None
    assert current.conversation_memory["proposed_preferences"] == ()
    records = service.conversations.dispatch_history(session.conversation_id, session.session_id)
    assert bound_records(current, records) == ()
    client = FakeClient()
    service.copilot = OpenAICopilotAdapter(model="fake-model", api_key="unused", client=client)
    runtime.send("Can you explain what you can help me with?")
    assert len(client.calls) == 1
    history = json.loads(client.calls[0]["input"].split("<recent_conversation>\n")[1].split("\n</recent_conversation>")[0])
    assert history == []
    evidence("accepted_plan_transition", runtime, preview=proposal)
