"""Disposable real app lifecycle and data-only conversation preconditions.

The SDK fixture replaces only external HTTP transport. It is not a live OpenAI test.
No internal application endpoint, optimizer, evaluator, or browser response is mocked.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import socket
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[3]
for import_root in (ROOT / "src", ROOT / "scripts"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


class DelayedProviderHTTP:
    """Real AsyncOpenAI wire/parse over a non-network, never-fallthrough transport."""

    def __init__(self, evidence):
        import httpx
        from openai import AsyncOpenAI

        self.evidence = evidence
        self.entered = threading.Event()
        self.release = threading.Event()
        self.finished = threading.Event()
        self.calls = []
        self.transport = httpx.MockTransport(self.handle)
        # The loopback alias is deliberately not api.openai.com. Even the
        # isolation guard remains active; MockTransport never opens this socket.
        self.http_client = httpx.AsyncClient(transport=self.transport, trust_env=False)
        self.client = AsyncOpenAI(api_key="synthetic-test-key-not-usable", max_retries=0,
            base_url="http://127.0.0.1:9/v1", http_client=self.http_client)

    async def handle(self, request):
        import httpx
        from openai.lib._pydantic import to_strict_json_schema

        from itinerary_system.product_app.openai_copilot import OpenAIInterpretationSchemaV2

        raw = (await request.aread()).decode("utf-8")
        row = {"schema_version":"fake-provider-exact-http-v1", "method":request.method,"url":str(request.url),
            "request_body_utf8":raw,"request_body_sha256":hashlib.sha256(raw.encode()).hexdigest(),
            "response_body_utf8":None,"response_body_sha256":None,"response_status":None,
            "request_headers_persisted":False,"response_headers_persisted":False,"authorization_persisted":False,
            "live_provider_call_count":0,"sdk_max_retries":self.client.max_retries,
            "transport":"httpx.MockTransport","never_fallthrough":True,
            "outcome":"pending","started_at":datetime.now(UTC).isoformat()}
        self.calls.append(row)
        write_json(self.evidence / "fake_sdk_io.json", self.calls)
        try:
            assert request.method == "POST" and str(request.url) == "http://127.0.0.1:9/v1/responses", "Unexpected fake-provider route"
            wire = json.loads(raw)
            expected_schema = to_strict_json_schema(OpenAIInterpretationSchemaV2)
            assert wire["text"]["format"] == {"type":"json_schema", "name":"OpenAIInterpretationSchemaV2",
                "strict":True, "schema":expected_schema}
            assert wire["store"] is False and wire["tools"] == []
            assert self.client.max_retries == 0
            row["strict_schema_wire_verified"] = True
            self.entered.set()
            while not self.release.is_set():
                await asyncio.sleep(0.02)
            structured = OpenAIInterpretationSchemaV2(schema_version="copilot-interpretation-v2", state="proposal_ready",
                assistant_message="Untrusted fake provider wording; the real adapter must canonicalize this.",
                intents=[{"type": "replace_nearby", "target_stop_id": "bixby_creek_bridge_viewpoint",
                    "candidate_id": "pfeiffer_big_sur_state_park", "parameters": None}]).model_dump(mode="json")
            body = json.dumps({"id":"resp_fake_conversation_transport","object":"response",
                "created_at":1788836400,"status":"completed","model":wire["model"],
                "output":[{"type":"message","id":"msg_fake_conversation_transport","status":"completed",
                    "role":"assistant","content":[{"type":"output_text","annotations":[],
                        "text":json.dumps(structured,ensure_ascii=False,separators=(",",":"))}]}]},
                ensure_ascii=False,separators=(",",":"))
            row.update(response_body_utf8=body,response_body_sha256=hashlib.sha256(body.encode()).hexdigest(),
                response_status=200,outcome="completed")
            return httpx.Response(200,content=body.encode(),headers={"Content-Type":"application/json"},request=request)
        except asyncio.CancelledError:
            row["outcome"] = "cancelled"
            raise
        except Exception as error:
            row.update(outcome="fixture_rejected",error_type=type(error).__name__)
            raise
        finally:
            row["finished_at"] = datetime.now(UTC).isoformat()
            write_json(self.evidence / "fake_sdk_io.json", self.calls)
            self.finished.set()


@contextmanager
def conversation_server(evidence, *, base_url, run_id, fake_sdk=False, state_root=None, registry_path=None):
    """Own the port for one case; never attach to or terminate an existing app."""
    from urllib.parse import urlsplit

    import httpx
    import uvicorn
    from e2e1_provider_isolation import ProviderIsolation, deterministic_environment

    from itinerary_system.product_app.api import create_product_app
    from itinerary_system.product_app.openai_copilot import OpenAICopilotAdapter

    parsed = urlsplit(base_url)
    assert parsed.scheme == "http" and parsed.hostname == "127.0.0.1" and parsed.port == 8127
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", parsed.port))  # In-use is a hard failure, not permission to stop it.
    state = Path(state_root) if state_root is not None else evidence / "disposable_state"
    state.mkdir(parents=True, exist_ok=False)
    environment = deterministic_environment(os.environ)
    environment.update(PRODUCT_APP_ORIGIN=base_url, PRODUCT_E2E_BASE_URL=base_url,
        PRODUCT_E2E_RUN_ID=run_id, PYTHONDONTWRITEBYTECODE="1")
    sdk = DelayedProviderHTTP(evidence) if fake_sdk else None
    adapter = None
    if sdk is not None:
        environment.update(PRODUCT_COPILOT_ADAPTER="openai", OPENAI_COPILOT_MODEL="fake-no-network",
            OPENAI_API_KEY="synthetic-test-key-not-usable")
        adapter = OpenAICopilotAdapter(model="fake-no-network", api_key=environment["OPENAI_API_KEY"],
            client=sdk.client, timeout_seconds=60)
    with patch.dict(os.environ, environment, clear=True), ProviderIsolation(evidence / "provider_isolation.json", run_id) as guard:
        app = create_product_app(state_root=state, application_port=parsed.port, copilot=adapter, registry_path=registry_path)
        runtime = app.state.product_runtime
        # Preserve initialization truth even when the guard correctly blocks a
        # missing service before a listening server or browser can be created.
        write_json(evidence / "initialization_health.json", runtime.health().as_dict())
        if sdk is None:
            guard.check_adapter(runtime)
        else:
            assert type(runtime.service.copilot) is OpenAICopilotAdapter and runtime.service.copilot._client is sdk.client
            guard.record.update(schema_version="conversation-browser-fake-sdk-isolation-v1",
                effective_adapter="openai", actual_adapter="OpenAICopilotAdapter", api_key_disabled=True,
                app_initialized=True, credential_mode="synthetic_unusable_fixture_value",
                external_transport="httpx.MockTransport", actual_sdk="openai.AsyncOpenAI", sdk_max_retries=0,
                fake_provider_endpoint="http://127.0.0.1:9/v1/responses", live_openai_test=False)
            guard._write()
        config = uvicorn.Config(app, host="127.0.0.1", port=parsed.port, log_level="warning")
        server = uvicorn.Server(config)
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        try:
            deadline = time.monotonic() + 25
            while not server.started and thread.is_alive() and time.monotonic() < deadline:
                time.sleep(0.02)
            assert server.started and thread.is_alive(), "Disposable app failed to start"
            response = httpx.get(f"{base_url}/api/health", timeout=15)
            write_json(evidence / "startup_health.json", {"status":response.status_code, "body":response.json()})
            assert response.status_code == 200
            yield SimpleNamespace(service=runtime.service, state=state, sdk=sdk, guard=guard)
        finally:
            if sdk is not None:
                sdk.release.set()
            guard.record["server_started"] = server.started
            server.should_exit = True
            thread.join(timeout=20)
            if sdk is not None:
                asyncio.run(sdk.client.close())
            guard.record["server_stopped"] = not thread.is_alive()
            write_json(evidence / "server_lifecycle.json", {"run_id":run_id, "pid":os.getpid(),
                "state_directory":str(state.resolve()),
                "thread_alive_after_shutdown":thread.is_alive(), "app_build_id":runtime.build_id,
                "live_provider_call_count":0, "fake_sdk_attempt_count":len(sdk.calls) if sdk else 0,
                "external_transport_mocked_only":sdk is not None, "api_response_mocking":False})
            assert not thread.is_alive(), "Owned disposable server did not stop cleanly"


def observe_context(runtime, session_id, evidence, label):
    """Read canonical server state for comparison with visible, real HTTP outcomes."""
    from copy import deepcopy

    # The production context default creates a conversation when none is bound.
    # An evidence observer must only read existing durable history.
    with runtime.service.workspace._lock:
        session = deepcopy(runtime.service.workspace.get(session_id))
        records = runtime.service.conversations.dispatch_history(
            session.conversation_id, session_id,
        ) if session.conversation_id is not None else ()
    # Return the same complete JSON shape persisted below (tuples become arrays).
    context = json.loads(json.dumps(runtime.service.copilot_context(
        session, dispatch_records=records,
    ).as_dict()))
    write_json(evidence / f"context_{label}.json", context)
    return context


def seed_capacity(runtime, session_id, kind, evidence):
    """Use the real serializer under its lock, solely on this test's private state."""
    from itinerary_system.product_app.conversations import DISPATCH_HEADROOM_BYTES, MAX_FILE_BYTES, MAX_TURNS

    session = runtime.service.workspace.get(session_id)
    repository = runtime.service.conversations
    conversation = runtime.service.read_conversation(session)
    assert len(conversation.turns) == 1, "Seed requires exactly one real website-created turn"
    template = replace(conversation.turns[0], user_message="u" * 2000 if kind == "bytes" else "Fixture turn",
        assistant_message="a" * 4000 if kind == "bytes" else "Fixture reply", interpretation=None)
    turns = []
    while len(turns) < MAX_TURNS:
        turns.append(replace(template, turn_id="copilot_turn_" + uuid4().hex,
            client_message_id="client_message_" + uuid4().hex))
        full = replace(conversation, turns=tuple(turns), revision=len(turns))
        size = repository._encoded_size(full.as_dict())
        if kind == "bytes" and size + DISPATCH_HEADROOM_BYTES > MAX_FILE_BYTES:
            break
    assert size < MAX_FILE_BYTES
    assert len(turns) == MAX_TURNS if kind == "turns" else len(turns) < MAX_TURNS
    path = repository._path(conversation.conversation_id)
    before = path.read_bytes()
    with repository._files.locked():
        repository._save(full)
    after = path.read_bytes()
    # The fixture I/O is synthetic and secret-free; preserve exact inputs and bytes.
    write_json(evidence / "capacity_fixture_io.json", {"kind":kind, "path":path.relative_to(runtime.state).as_posix(),
        "before_utf8":before.decode(), "after_utf8":after.decode(),
        "before_sha256":hashlib.sha256(before).hexdigest(), "after_sha256":hashlib.sha256(after).hexdigest(),
        "serialized_bytes":len(after), "turn_count":len(turns), "max_bytes":MAX_FILE_BYTES,
        "max_turns":MAX_TURNS, "reserved_headroom_bytes":DISPATCH_HEADROOM_BYTES,
        "mutation_scope":"Disposable conversation precondition only; real ConversationRepository._save validation"})
    return full


def test_real_sdk_wire_and_parse_fixture_without_server(tmp_path):
    """Validate real SDK strict-schema wire conversion and parsed result, zero sockets."""
    from itinerary_system.product_app.openai_copilot import OpenAICopilotAdapter
    from itinerary_system.product_app.registry import ProductRunRegistry
    from itinerary_system.product_app.service import ProductService

    sdk = DelayedProviderHTTP(tmp_path)
    sdk.release.set()
    adapter = OpenAICopilotAdapter(model="fake-no-network",api_key="synthetic-test-key-not-usable",client=sdk.client)
    registry = ProductRunRegistry(ROOT,ROOT / "configs/product_app_registry.json")
    service = ProductService(registry,tmp_path / "state",copilot=adapter)
    session,token,_ = service.create_session(registry.default.run_id)
    message = "Replace Bixby Creek Bridge with Pfeiffer Big Sur State Park."
    message_id = "client_message_" + uuid4().hex

    async def exercise():
        try:
            return await service.interpret_copilot_message(session_id=session.session_id,mutation_token=token,
                expected_revision=session.revision,client_message_id=message_id,message=message)
        finally:
            await sdk.client.close()

    result = asyncio.run(exercise())
    assert len(sdk.calls) == 1 and sdk.calls[0]["strict_schema_wire_verified"]
    assert sdk.calls[0]["outcome"] == "completed" and sdk.calls[0]["response_status"] == 200
    turn = result["turn"]
    assert turn["interpretation"]["schema_version"] == "copilot-interpretation-v2"
    assert turn["interpretation"]["state"] == "proposal_ready"
    assert [(row["type"],row["target_stop_id"],row["candidate_id"]) for row in turn["interpretation"]["intents"]] == [
        ("replace_nearby","bixby_creek_bridge_viewpoint","pfeiffer_big_sur_state_park")]
    assert "Untrusted fake provider wording" not in turn["assistant_message"]
    for direction in ("request","response"):
        assert hashlib.sha256(sdk.calls[0][f"{direction}_body_utf8"].encode()).hexdigest() == sdk.calls[0][f"{direction}_body_sha256"]
    assert "synthetic-test-key" not in json.dumps(sdk.calls)
    write_json(tmp_path / "service_io.json",{"input":{"message":message,"client_message_id":message_id,
        "expected_revision":0},"output":result,"fake_http_attempt_count":1,"live_provider_call_count":0,
        "actual_sdk_parse_verified":True,"browser_executed":False})


def test_real_sdk_transport_cancel_and_reconcile_fixture_without_server(tmp_path):
    import pytest

    from itinerary_system.product_app.openai_copilot import OpenAICopilotAdapter
    from itinerary_system.product_app.registry import ProductRunRegistry
    from itinerary_system.product_app.service import CopilotServiceError, ProductService

    sdk = DelayedProviderHTTP(tmp_path)
    adapter = OpenAICopilotAdapter(model="fake-no-network",api_key="synthetic-test-key-not-usable",client=sdk.client)
    registry = ProductRunRegistry(ROOT,ROOT / "configs/product_app_registry.json")
    service = ProductService(registry,tmp_path / "state",copilot=adapter)
    session,token,_ = service.create_session(registry.default.run_id)
    envelope = {"session_id":session.session_id,"mutation_token":token,"expected_revision":session.revision,
        "client_message_id":"client_message_"+uuid4().hex,
        "message":"Replace Bixby Creek Bridge with Pfeiffer Big Sur State Park."}

    async def exercise():
        try:
            task = asyncio.create_task(service.interpret_copilot_message(**envelope))
            assert await asyncio.to_thread(sdk.entered.wait,10), "Real SDK request did not reach the mock HTTP boundary"
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            with pytest.raises(CopilotServiceError,match="copilot_outcome_unknown"):
                await service.interpret_copilot_message(**envelope)
        finally:
            await sdk.client.close()

    asyncio.run(exercise())
    assert len(sdk.calls) == 1 and sdk.calls[0]["outcome"] == "cancelled"
    assert sdk.calls[0]["response_body_utf8"] is None and sdk.calls[0]["response_body_sha256"] is None
    conversation = service.read_conversation(session)
    journal = service.conversations.dispatch_record(conversation.conversation_id,session.session_id,
        envelope["client_message_id"],envelope["message"])
    assert journal["attempt_budget"] == journal["attempt_count"] == 1 and journal["outcome"] == "unknown"
    write_json(tmp_path / "service_cancel_io.json", {"input":{key:value for key,value in envelope.items() if key != "mutation_token"},
        "mutation_token_omitted":True,"dispatch":journal,"fake_http_attempt_count":1,"live_provider_call_count":0,
        "browser_executed":False})


def test_mock_transport_rejects_other_urls_without_fallthrough(tmp_path):
    import pytest

    sdk = DelayedProviderHTTP(tmp_path)
    async def exercise():
        try:
            with pytest.raises(AssertionError,match="Unexpected fake-provider route"):
                await sdk.http_client.get("https://example.invalid/must-not-use-network")
        finally:
            await sdk.client.close()
    asyncio.run(exercise())
    assert len(sdk.calls) == 1 and sdk.calls[0]["outcome"] == "fixture_rejected"
    assert sdk.calls[0]["response_body_utf8"] is None
