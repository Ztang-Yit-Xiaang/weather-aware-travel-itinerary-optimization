from __future__ import annotations

import json
import urllib.request
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import uuid4
from zipfile import ZipFile

import pytest
from fastapi.testclient import TestClient

from itinerary_system.product_app import openai_copilot
from itinerary_system.product_app.api import create_product_app
from itinerary_system.product_app.models import (
    ComponentHealthV1,
    CopilotContextV1,
    CopilotHighlightsV1,
    CopilotIntentV1,
    CopilotInterpretationV1,
    CopilotTurnV1,
)
from itinerary_system.product_app.openai_copilot import (
    OpenAICopilotAdapter,
    OpenAIHighlightsSchemaV1,
    OpenAIIntentSchemaV1,
    OpenAIInterpretationSchemaV1,
)
from itinerary_system.product_app.runtime import ProductRuntime

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "configs" / "product_app_registry.json"
ALLOWED_ORIGIN = "http://127.0.0.1:8127"

CONTEXT_FIELDS = {
    "schema_version",
    "run_id",
    "trip_id",
    "session_id",
    "session_revision",
    "accepted_plan_id",
    "selected_day",
    "selected_stop_id",
    "selected_segment_id",
    "selected_candidate_id",
    "selected_alternative_id",
    "draft_operations",
    "evaluated_proposal",
    "allowed_stop_ids",
    "allowed_candidate_ids",
    "allowed_days",
    "allowed_segment_ids",
    "allowed_alternative_ids",
}


@pytest.fixture
def api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("PRODUCT_COPILOT_ADAPTER", "deterministic")
    monkeypatch.setenv("OPENAI_API_KEY", "API_KEY_SENTINEL_MUST_NOT_ESCAPE")
    monkeypatch.setattr(
        ProductRuntime,
        "_probe_map",
        lambda self: ComponentHealthV1("map", "ready", False, "maplibre_ready"),
    )
    state_root = tmp_path / "PRIVATE_STATE_PATH_SENTINEL"
    app = create_product_app(
        repository_root=ROOT,
        registry_path=REGISTRY,
        state_root=state_root,
        additional_allowed_authorities=("testserver",),
    )
    with TestClient(app) as client:
        yield client, app.state.product_service, state_root


def _session(client: TestClient) -> tuple[dict[str, Any], dict[str, str]]:
    response = client.post("/api/sessions", json={})
    assert response.status_code == 200
    payload = response.json()
    return payload["session"], {"X-Session-Token": payload["mutation_token"]}


def _send(
    client: TestClient,
    session: dict[str, Any],
    headers: dict[str, str],
    message: str,
) -> Any:
    return client.post(
        f"/api/sessions/{session['session_id']}/copilot/messages",
        headers=headers,
        json={
            "expected_revision": session["revision"],
            "client_message_id": f"client_message_{uuid4().hex}",
            "message": message,
        },
    )


def _section(prompt_input: str, name: str) -> Any:
    opening = f"<{name}>\n"
    closing = f"\n</{name}>"
    return json.loads(prompt_input.split(opening, 1)[1].split(closing, 1)[0])


class _FakeResponses:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def parse(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return SimpleNamespace(
            output_parsed=OpenAIInterpretationSchemaV1(
                schema_version="copilot-interpretation-v1",
                state="proposal_ready",
                assistant_message="Review this typed request before adding it to the draft.",
                intents=[OpenAIIntentSchemaV1(type="review_registered_repair")],
                highlights=OpenAIHighlightsSchemaV1(),
            ),
            output=[],
        )


class _FakeOpenAIClient:
    def __init__(self) -> None:
        self.responses = _FakeResponses()


class _ProviderFailure(RuntimeError):
    code = "openai_timeout"
    status_code = 504


class _FailingAdapter:
    provider_name = "openai"
    model = "gpt-test"

    async def interpret(self, **kwargs: Any) -> CopilotInterpretationV1:
        del kwargs
        raise _ProviderFailure(
            "RAW_PROVIDER_DETAIL_SENTINEL C:\\private\\provider-payload.json"
        )


class _RevisionChangingAdapter:
    provider_name = "openai"
    model = "gpt-test"

    def __init__(self, service: Any) -> None:
        self.service = service

    async def interpret(
        self,
        *,
        context: CopilotContextV1,
        history: tuple[CopilotTurnV1, ...],
        message: str,
    ) -> CopilotInterpretationV1:
        del history, message
        current = self.service.workspace.get(context.session_id)
        self.service.workspace.select(
            current,
            {"expected_revision": current.revision, "selected_day": 2},
        )
        return CopilotInterpretationV1(
            state="proposal_ready",
            assistant_message="Review this typed request.",
            intents=(CopilotIntentV1(type="review_registered_repair"),),
            highlights=CopilotHighlightsV1(day_ids=(2,)),
        )


def test_deterministic_adapter_never_constructs_openai_or_uses_network(
    api, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, _, _ = api
    session, headers = _session(client)

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise AssertionError("deterministic Copilot attempted external network access")

    monkeypatch.setattr(openai_copilot, "AsyncOpenAI", forbidden)
    monkeypatch.setattr(urllib.request, "urlopen", forbidden)

    response = _send(client, session, headers, "Review the registered repair")

    assert response.status_code == 200
    assert response.json()["turn"]["provider"] == "deterministic"


def test_openai_outbound_payload_uses_only_public_context_and_bounded_history(api) -> None:
    client, service, state_root = api
    session, headers = _session(client)
    for index in range(7):
        response = _send(client, session, headers, f"Review repair option {index}")
        assert response.status_code == 200

    fake_client = _FakeOpenAIClient()
    service.copilot = OpenAICopilotAdapter(
        model="gpt-5.6-terra",
        api_key="OUTBOUND_API_KEY_SENTINEL",
        client=fake_client,
    )
    response = _send(client, session, headers, "Review the registered repair")

    assert response.status_code == 200
    assert len(fake_client.responses.calls) == 1
    outbound = fake_client.responses.calls[0]
    context = _section(outbound["input"], "workspace_context")
    history = _section(outbound["input"], "recent_conversation")
    assert set(context) == CONTEXT_FIELDS
    assert len(history) == 12
    assert sum(len(item["content"]) for item in history) <= 12_000
    assert all(set(item) == {"content", "role"} for item in history)
    assert outbound["store"] is False
    assert outbound["tools"] == []
    serialized = json.dumps(outbound, default=str)
    for prohibited in (
        headers["X-Session-Token"],
        "OUTBOUND_API_KEY_SENTINEL",
        str(state_root),
        "mutation_token",
        "mutation_token_salt",
        "mutation_token_verifier",
    ):
        assert prohibited not in serialized


def test_health_errors_logs_and_evidence_exclude_secrets_paths_and_transcripts(
    api, caplog: pytest.LogCaptureFixture
) -> None:
    client, service, state_root = api
    session, headers = _session(client)
    transcript = "PRIVATE_TRANSCRIPT_SENTINEL"
    assert _send(client, session, headers, transcript).status_code == 200

    service.copilot = _FailingAdapter()
    failed = _send(client, session, headers, "Trigger a safe provider failure")
    health = client.get("/api/health")
    alternatives = client.get(
        f"/api/sessions/{session['session_id']}/alternatives",
        headers=headers,
    ).json()["alternatives"]
    evidence = client.get(
        f"/api/runs/{session['run_id']}/evidence-bundle",
        params={"plan_id": alternatives[0]["plan_id"]},
    )

    assert failed.status_code == 504
    assert failed.json() == {"detail": "openai_timeout"}
    assert health.status_code == evidence.status_code == 200
    with ZipFile(BytesIO(evidence.content)) as archive:
        evidence_bytes = b"".join(archive.read(name) for name in archive.namelist())
    observed = "\n".join(
        (
            failed.text,
            health.text,
            evidence_bytes.decode("utf-8", errors="ignore"),
            caplog.text,
        )
    )
    for prohibited in (
        "API_KEY_SENTINEL_MUST_NOT_ESCAPE",
        headers["X-Session-Token"],
        str(state_root),
        transcript,
        "RAW_PROVIDER_DETAIL_SENTINEL",
        "provider-payload.json",
    ):
        assert prohibited not in observed


def test_failed_provider_turn_persists_only_stable_safe_failure(api) -> None:
    client, service, _ = api
    session, headers = _session(client)
    service.copilot = _FailingAdapter()

    failed = _send(client, session, headers, "Explain the repair")
    conversation = client.get(
        f"/api/sessions/{session['session_id']}/conversation",
        headers=headers,
    ).json()["conversation"]

    assert failed.status_code == 504
    turn = conversation["turns"][0]
    assert turn["state"] == "failed"
    assert turn["error_code"] == "openai_timeout"
    assert turn["interpretation"] is None
    assert turn["assistant_message"] == (
        "OpenAI Copilot did not respond before the local timeout."
    )
    assert "RAW_PROVIDER_DETAIL_SENTINEL" not in json.dumps(turn)
    assert "provider-payload.json" not in json.dumps(turn)


def test_stale_post_provider_revision_appends_no_turn(api) -> None:
    client, service, _ = api
    session, headers = _session(client)
    service.copilot = _RevisionChangingAdapter(service)

    response = _send(client, session, headers, "Review the current repair")
    conversation = client.get(
        f"/api/sessions/{session['session_id']}/conversation",
        headers=headers,
    )

    assert response.status_code == 409
    assert response.json() == {"detail": "stale_session_revision"}
    assert conversation.status_code == 200
    assert conversation.json()["conversation"]["turns"] == []


def test_w5_decisions_remain_fail_closed_and_do_not_change_workspace(api) -> None:
    client, _, _ = api
    session, headers = _session(client)
    before = client.get(f"/api/sessions/{session['session_id']}", headers=headers).json()[
        "session"
    ]

    for action in ("accept", "keep-original"):
        response = client.post(
            f"/api/sessions/{session['session_id']}/{action}",
            headers=headers,
            json={"expected_revision": session["revision"]},
        )
        assert response.status_code == 409
        assert response.json() == {"detail": "acceptance_not_enabled_until_w5"}
        assert response.headers["cache-control"] == "no-store"

    after = client.get(f"/api/sessions/{session['session_id']}", headers=headers).json()[
        "session"
    ]
    for field in ("revision", "accepted_plan_id", "draft", "proposal", "permission_decisions"):
        assert after[field] == before[field]


def test_transcript_routes_are_no_store_and_keep_host_origin_guards(api) -> None:
    client, _, _ = api
    session, headers = _session(client)
    conversation_path = f"/api/sessions/{session['session_id']}/conversation"

    responses = [
        client.get(conversation_path, headers=headers),
        _send(client, session, headers, "Review repair"),
        client.request(
            "DELETE",
            conversation_path,
            headers=headers,
            json={"expected_revision": session["revision"]},
        ),
    ]
    assert all(response.headers["cache-control"] == "no-store" for response in responses)

    blocked_host = client.get(
        conversation_path,
        headers={**headers, "Host": "localhost.evil:8127"},
    )
    blocked_origin = client.post(
        f"/api/sessions/{session['session_id']}/copilot/messages",
        headers={**headers, "Origin": "https://example.com"},
        json={
            "expected_revision": session["revision"],
            "client_message_id": f"client_message_{uuid4().hex}",
            "message": "Review repair",
        },
    )
    assert blocked_host.status_code == 400
    assert blocked_host.json() == {"detail": "host_not_allowed"}
    assert blocked_origin.status_code == 403
    assert blocked_origin.json() == {"detail": "origin_not_allowed"}
    assert "access-control-allow-origin" not in blocked_origin.headers


def test_delete_all_requires_exact_origin_confirmation_and_session_header(api) -> None:
    client, _, _ = api
    session, headers = _session(client)
    assert _send(client, session, headers, "Review repair").status_code == 200
    path = "/api/conversations"
    body = {
        "expected_revision": session["revision"],
        "confirmation": "delete_all_conversations",
    }

    cases = (
        ({**headers, "X-Session-Id": session["session_id"]}, body, 403, "origin_not_allowed"),
        (
            {**headers, "X-Session-Id": session["session_id"], "Origin": "https://127.0.0.1:8127"},
            body,
            403,
            "origin_not_allowed",
        ),
        ({**headers, "Origin": ALLOWED_ORIGIN}, body, 403, "invalid_session_token"),
        (
            {**headers, "X-Session-Id": session["session_id"], "Origin": ALLOWED_ORIGIN},
            {**body, "confirmation": "DELETE_ALL_CONVERSATIONS"},
            422,
            "request_validation_failed",
        ),
    )
    for request_headers, request_body, status, detail in cases:
        response = client.request("DELETE", path, headers=request_headers, json=request_body)
        assert response.status_code == status
        assert response.json() == {"detail": detail}
        assert response.headers["cache-control"] == "no-store"

    deleted = client.request(
        "DELETE",
        path,
        headers={
            **headers,
            "X-Session-Id": session["session_id"],
            "Origin": ALLOWED_ORIGIN,
        },
        json=body,
    )
    assert deleted.status_code == 200
    assert deleted.headers["cache-control"] == "no-store"
    assert deleted.json()["deleted_count"] == 1
