from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from itinerary_system.product_app.api import create_product_app
from itinerary_system.product_app.conversations import ConversationError
from itinerary_system.product_app.copilot import DeterministicCopilotAdapter
from itinerary_system.product_app.models import (
    ComponentHealthV1,
    CopilotContextV1,
    CopilotHighlightsV1,
    CopilotIntentV1,
    CopilotInterpretationV1,
    CopilotTurnV1,
)
from itinerary_system.product_app.runtime import ProductRuntime

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "configs" / "product_app_registry.json"


@pytest.fixture
def api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("PRODUCT_COPILOT_ADAPTER", raising=False)
    monkeypatch.setattr(
        ProductRuntime,
        "_probe_map",
        lambda self: ComponentHealthV1("map", "ready", False, "maplibre_ready"),
    )
    app = create_product_app(
        repository_root=ROOT,
        registry_path=REGISTRY,
        state_root=tmp_path / "state",
        additional_allowed_authorities=("testserver",),
    )
    with TestClient(app) as client:
        yield client, app.state.product_service


def _session(client: TestClient) -> tuple[dict, dict[str, str]]:
    response = client.post("/api/sessions", json={})
    assert response.status_code == 200
    body = response.json()
    return body["session"], {"X-Session-Token": body["mutation_token"]}


def _message(
    client: TestClient,
    session: dict,
    headers: dict[str, str],
    *,
    text: str = "Review the safer repair",
    client_message_id: str | None = None,
):
    return client.post(
        f"/api/sessions/{session['session_id']}/copilot/messages",
        headers=headers,
        json={
            "expected_revision": session["revision"],
            "client_message_id": client_message_id or f"client_message_{uuid4().hex}",
            "message": text,
        },
    )


def test_conversation_auth_empty_record_and_deterministic_send_are_separate_from_plan_state(api) -> None:
    client, service = api
    session, headers = _session(client)
    path = f"/api/sessions/{session['session_id']}/conversation"

    interaction = service.workspace_view(session["run_id"])["interaction"]
    assert interaction == {
        "enabled": True,
        "provider": "deterministic",
        "state": "deterministic_demo",
        "message": "Deterministic demo. Requests stay local and no external provider is called.",
    }

    assert client.get(path).status_code == 403
    empty = client.get(path, headers=headers)
    assert empty.status_code == 200
    assert empty.json()["conversation"]["turns"] == []

    sent = _message(client, session, headers)
    assert sent.status_code == 200
    payload = sent.json()
    assert payload["turn"]["provider"] == "deterministic"
    assert payload["turn"]["interpretation"]["state"] == "proposal_ready"
    assert payload["conversation_revision"] == 1
    assert payload["session"]["revision"] == session["revision"]
    assert payload["session"]["draft"] == session["draft"]
    assert payload["session"]["proposal"] == session["proposal"]
    assert payload["session"]["accepted_plan_id"] == session["accepted_plan_id"]
    assert payload["session"]["permission_decisions"] == []
    assert payload["advisory"]["automatic_activation"] is False


def test_message_requires_canonical_client_id_and_is_idempotent(api) -> None:
    client, _ = api
    session, headers = _session(client)
    invalid = _message(client, session, headers, client_message_id="message-not-canonical")
    assert invalid.status_code == 422

    client_id = f"client_message_{uuid4().hex}"
    first = _message(client, session, headers, client_message_id=client_id)
    repeated = _message(client, session, headers, client_message_id=client_id)
    conflict = _message(
        client,
        session,
        headers,
        text="Keep the original",
        client_message_id=client_id,
    )

    assert first.status_code == repeated.status_code == 200
    assert first.json()["turn"]["turn_id"] == repeated.json()["turn"]["turn_id"]
    assert repeated.json()["conversation_revision"] == 1
    assert conflict.status_code == 409
    assert conflict.json() == {"detail": "message_id_conflict"}


class RevisionChangingAdapter:
    provider_name = "deterministic"
    model = None

    def __init__(self, service) -> None:
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
            assistant_message="Review this advisory proposal.",
            intents=(CopilotIntentV1(type="keep_original"),),
            highlights=CopilotHighlightsV1(day_ids=(2,)),
        )


class ProviderFailure(Exception):
    code = "openai_timeout"


class FailingAdapter:
    provider_name = "openai"
    model = "gpt-test"

    async def interpret(self, **kwargs):
        del kwargs
        raise ProviderFailure("raw provider detail must not escape")


class QuotaFailure(Exception):
    code = "openai_quota_exhausted"


class QuotaFailingAdapter:
    provider_name = "openai"
    model = "gpt-test"

    async def interpret(self, **kwargs):
        del kwargs
        raise QuotaFailure("raw billing detail must not escape")


def test_stale_revision_before_and_after_provider_appends_nothing(api) -> None:
    client, service = api
    session, headers = _session(client)
    stale_pre = client.post(
        f"/api/sessions/{session['session_id']}/copilot/messages",
        headers=headers,
        json={
            "expected_revision": 99,
            "client_message_id": f"client_message_{uuid4().hex}",
            "message": "Review repair",
        },
    )
    assert stale_pre.status_code == 409

    service.copilot = RevisionChangingAdapter(service)
    stale_post = _message(client, session, headers)
    assert stale_post.status_code == 409
    conversation = client.get(
        f"/api/sessions/{session['session_id']}/conversation", headers=headers
    ).json()["conversation"]
    assert conversation["turns"] == []


def test_provider_failure_is_sanitized_and_persisted_without_itinerary_mutation(api) -> None:
    client, service = api
    session, headers = _session(client)
    service.copilot = FailingAdapter()

    failed = _message(client, session, headers)

    assert failed.status_code == 504
    assert failed.json() == {"detail": "openai_timeout"}
    conversation = client.get(
        f"/api/sessions/{session['session_id']}/conversation", headers=headers
    ).json()["conversation"]
    assert conversation["revision"] == 1
    assert conversation["turns"][0]["state"] == "failed"
    assert conversation["turns"][0]["error_code"] == "openai_timeout"
    assert "raw provider detail" not in str(conversation)
    restored = client.get(f"/api/sessions/{session['session_id']}", headers=headers).json()[
        "session"
    ]
    assert restored["revision"] == session["revision"]
    assert restored["draft"] == []
    assert restored["proposal"] is None


def test_quota_failure_is_distinct_from_rate_limit_and_remains_sanitized(api) -> None:
    client, service = api
    session, headers = _session(client)
    service.copilot = QuotaFailingAdapter()

    failed = _message(client, session, headers)

    assert failed.status_code == 429
    assert failed.json() == {"detail": "openai_quota_exhausted"}
    conversation = client.get(
        f"/api/sessions/{session['session_id']}/conversation", headers=headers
    ).json()["conversation"]
    assert conversation["turns"][0]["error_code"] == "openai_quota_exhausted"
    assert "raw billing detail" not in str(conversation)


def test_delete_current_and_delete_all_require_owner_revision_and_confirmation(api) -> None:
    client, _ = api
    first, first_headers = _session(client)
    second, second_headers = _session(client)
    assert _message(client, first, first_headers).status_code == 200
    assert _message(client, second, second_headers).status_code == 200

    current_path = f"/api/sessions/{first['session_id']}/conversation"
    assert client.request(
        "DELETE", current_path, json={"expected_revision": 0}
    ).status_code == 403
    stale = client.request(
        "DELETE",
        current_path,
        headers=first_headers,
        json={"expected_revision": 99},
    )
    assert stale.status_code == 409
    assert len(client.get(current_path, headers=first_headers).json()["conversation"]["turns"]) == 1
    deleted = client.request(
        "DELETE",
        current_path,
        headers=first_headers,
        json={"expected_revision": 0},
    )
    assert deleted.status_code == 200
    assert deleted.json()["deleted"] is True
    assert client.get(current_path, headers=first_headers).json()["conversation"]["turns"] == []

    all_path = "/api/conversations"
    missing_origin = client.request(
        "DELETE",
        all_path,
        headers={**second_headers, "X-Session-Id": second["session_id"]},
        json={"expected_revision": 0, "confirmation": "delete_all_conversations"},
    )
    assert missing_origin.status_code == 403
    wrong = client.request(
        "DELETE",
        all_path,
        headers={
            **second_headers,
            "X-Session-Id": second["session_id"],
            "Origin": "http://127.0.0.1:8127",
        },
        json={"expected_revision": 0, "confirmation": "delete"},
    )
    assert wrong.status_code == 422
    removed = client.request(
        "DELETE",
        all_path,
        headers={
            **second_headers,
            "X-Session-Id": second["session_id"],
            "Origin": "http://127.0.0.1:8127",
        },
        json={"expected_revision": 0, "confirmation": "delete_all_conversations"},
    )
    assert removed.status_code == 200
    assert removed.json()["deleted_count"] >= 1


def test_delete_current_restores_session_binding_when_file_deletion_fails(
    api, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, service = api
    session, headers = _session(client)
    assert _message(client, session, headers).status_code == 200
    current_path = f"/api/sessions/{session['session_id']}/conversation"
    before = client.get(current_path, headers=headers).json()["conversation"]

    def fail_delete(conversation_id: str, session_id: str) -> bool:
        del conversation_id, session_id
        raise ConversationError("conversation_delete_failed")

    monkeypatch.setattr(service.conversations, "delete", fail_delete)
    failed = client.request(
        "DELETE",
        current_path,
        headers=headers,
        json={"expected_revision": session["revision"]},
    )

    assert failed.status_code == 503
    assert failed.json() == {"detail": "conversation_delete_failed"}
    restored = client.get(current_path, headers=headers).json()["conversation"]
    assert restored["conversation_id"] == before["conversation_id"]
    assert restored["turns"] == before["turns"]


def test_deterministic_mode_is_network_free_and_w5_decisions_remain_blocked(
    api, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, service = api
    session, headers = _session(client)
    service.copilot = DeterministicCopilotAdapter()

    def forbidden_network(*args, **kwargs):
        raise AssertionError("deterministic Copilot attempted network access")

    monkeypatch.setattr("urllib.request.urlopen", forbidden_network)
    assert _message(client, session, headers).status_code == 200
    for suffix in ("accept", "keep-original"):
        response = client.post(
            f"/api/sessions/{session['session_id']}/{suffix}",
            headers=headers,
            json={"expected_revision": 0},
        )
        assert response.status_code == 409
        assert response.json() == {"detail": "acceptance_not_enabled_until_w5"}
